"""The self-iteration cycle.

One cycle::

    gather      fetch papers new since last cycle (arXiv, or a replayed stream)
    triage      score them with the current champion config; write a digest
    learn       fold them into term statistics; mine candidate vocabulary
    propose     candidate changes from mutation / literature / LLM, budget
                allocated by a bandit over past acceptance
    evaluate    each candidate against the champion on the tune split
    gate        best passing candidate checked once against holdout
    promote     if it survives, it becomes the champion
    record      history, cycle summary, report, ideas backlog

The cycle is idempotent with respect to its inputs: given the same state
directory, stream and seed, it makes the same decisions. That is what makes
the agent's history auditable rather than anecdotal.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sciscout.discovery import discover

from . import report
from .bench import (DEFAULT_BENCH, MIN_HUMAN_LABELS, SPLITS, Benchmark, BenchResult,
                    default_benchmarks)
from .config import AGENDA_PATH, AgentConfig
from .gate import GatePolicy, holdout_verdict, tune_verdict
from .mining import MinedTerm, mine, update_statistics
from .proposers import (Bandit, LiteratureProposer, LLMProposer, MutationProposer,
                        Proposal, ProposalContext, Proposer)
from .scout import DEFAULT_STREAM, Gathered, gather_live, gather_offline
from .state import Knowledge, StateDir
from .triage import Verdict, triage


@dataclass
class CycleOptions:
    live: bool = False
    candidates: int = 12
    stream_path: Path = DEFAULT_STREAM
    bench_path: Path = DEFAULT_BENCH
    agenda_path: Path = AGENDA_PATH
    seed: int = 0
    policy: GatePolicy = field(default_factory=GatePolicy)
    proposers: list[Proposer] | None = None  # override, mainly for tests
    # In live mode the synthetic benchmark says nothing about real papers, so
    # letting it steer the agent would tune it to the wrong literature. Live
    # cycles therefore only gather, learn and digest until enough human labels
    # exist to form a real-paper benchmark -- unless this is set.
    allow_synthetic_gate: bool = False


def init_state(state: StateDir, agenda_path: Path = AGENDA_PATH, force: bool = False,
               bench_path: Path = DEFAULT_BENCH) -> AgentConfig:
    if state.exists() and not force:
        raise FileExistsError(f"{state.path} already holds an agent; pass force to reset")
    config = AgentConfig.from_agenda(agenda_path)
    benches = default_benchmarks(bench_path, state.labels_path)
    state.save_champion(config, score_all(config, benches), cycle=0, reason="seed agenda")
    state.save_knowledge(Knowledge())
    return config


def score_all(config: AgentConfig, benches: list[Benchmark]) -> dict:
    out: dict = {}
    for bench in benches:
        for split in SPLITS:
            result = bench.evaluate(config, split)
            out.setdefault(bench.name, {})[split] = {
                "mean": round(result.mean, 5), "n": len(result.scores),
                **{k: round(v, 4) for k, v in result.metrics.items()},
            }
    return out


def _evaluate(config: AgentConfig, benches: list[Benchmark], split: str) -> dict[str, BenchResult]:
    return {b.name: b.evaluate(config, split) for b in benches}


def _compact(verdict: Verdict) -> dict:
    w = verdict.work
    return {
        "work": {"id": w.id, "title": w.title, "abstract": w.abstract,
                 "published": w.published.isoformat() if w.published else None,
                 "url": w.url, "source": w.source, "disciplines": w.disciplines},
        "topic": verdict.topic, "score": round(verdict.score, 4), "flagged": verdict.flagged,
    }


def _digest(cycle: int, gathered: Gathered, verdicts: list[Verdict], mined: list[MinedTerm],
            config: AgentConfig) -> str:
    """Markdown digest of what is new: the agent's reading list for the cycle."""
    flagged = [v for v in verdicts if v.flagged]
    lines = [f"# Digest, cycle {cycle}", "",
             f"Source: {gathered.source}. {len(verdicts)} new papers, {len(flagged)} flagged "
             f"for reading.", ""]
    if gathered.errors:
        lines += ["Harvest errors:", *[f"- {e}" for e in gathered.errors[:5]], ""]
    for topic in config.topics():
        items = [v for v in flagged if v.topic == topic][:5]
        if not items:
            continue
        lines.append(f"## {topic}")
        for v in items:
            link = f" <{v.work.url}>" if v.work.url else ""
            lines.append(f"- **{v.work.title}** (score {v.score:.2f}){link}")
        lines.append("")
    if len(flagged) >= 12:
        clusters = discover(
            [v.work for v in flagged],
            threshold=config.get("discovery.threshold"),
            min_cluster_size=int(config.get("discovery.min_cluster_size")),
            min_df=int(config.get("discovery.min_df")),
            max_df_ratio=config.get("discovery.max_df_ratio"),
        )
        if clusters.tracks:
            lines.append("## Directions in this batch")
            for track in clusters.tracks[:8]:
                lines.append(f"- {track.label}: {track.size} papers, cohesion "
                             f"{track.cohesion:.2f}; terms {', '.join(track.terms[:6])}")
            lines.append("")
    if mined:
        lines.append("## Emerging vocabulary (candidates, not yet adopted)")
        for m in mined[:20]:
            lines.append(f"- {m.topic}: `{m.term}` (z={m.z:.1f}, {m.topic_docs} papers)")
        lines.append("")
    return "\n".join(lines)


def _oracle(verdicts: list[Verdict]) -> dict | None:
    """Online triage accuracy, only where the stream carries synthetic labels.

    Never used in any decision -- the agent cannot know this in live operation.
    It is reported so the synthetic experiment can show whether gains on the
    benchmark carry over to the stream the agent actually reads.
    """
    labelled = [v for v in verdicts if "_oracle_topic" in v.work.extra]
    if not labelled:
        return None
    relevant = [v for v in labelled if v.work.extra["_oracle_topic"] != "irrelevant"]
    flagged = [v for v in labelled if v.flagged]
    hits = [v for v in flagged if v.work.extra["_oracle_topic"] != "irrelevant"]
    return {
        "recall": round(len(hits) / len(relevant), 4) if relevant else 0.0,
        "precision": round(len(hits) / len(flagged), 4) if flagged else 0.0,
    }


def run_cycle(state: StateDir, opts: CycleOptions | None = None) -> dict:
    opts = opts or CycleOptions()
    started = time.monotonic()
    if not state.exists():
        init_state(state, opts.agenda_path, bench_path=opts.bench_path)
    champion, _ = state.load_champion()
    knowledge = state.load_knowledge()
    cycle = knowledge.cycle + 1
    rng = np.random.default_rng([opts.seed, cycle])

    # -- gather --------------------------------------------------------------
    if opts.live:
        agenda = yaml.safe_load(Path(opts.agenda_path).read_text())
        gathered = gather_live(champion, knowledge, agenda["queries"]["categories"])
    else:
        gathered = gather_offline(knowledge, opts.stream_path)

    # -- triage and learn ------------------------------------------------------
    verdicts = triage(gathered.works, champion)
    confident = update_statistics(knowledge, verdicts)
    knowledge.seen.update(w.id for w in gathered.works)
    knowledge.recent_works.extend(_compact(v) for v in verdicts)
    LiteratureProposer.expire_rejections(knowledge, cycle)
    mined = mine(knowledge, champion)
    digest = _digest(cycle, gathered, verdicts, mined, champion)
    state.write_text(f"digests/cycle-{cycle:04d}.md", digest + "\n")

    # -- propose -------------------------------------------------------------
    benches = default_benchmarks(opts.bench_path, state.labels_path)
    history = state.read_jsonl("history.jsonl")
    ctx = ProposalContext(champion, knowledge, mined, digest, history, rng)
    proposers = opts.proposers or [MutationProposer(), LiteratureProposer(), LLMProposer()]
    active = [p for p in proposers if p.available(ctx)]
    paused = (opts.live and not opts.allow_synthetic_gate
              and not any(b.name == "triage-real" for b in benches))
    if paused:
        active = []
    bandit = Bandit(knowledge.proposer_stats, rng)
    allocation = bandit.allocate(active, opts.candidates)
    proposals: list[Proposal] = []
    llm_usage: dict[str, int] = {}
    for proposer in active:
        if allocation[proposer.name]:
            proposals.extend(proposer.propose(ctx, allocation[proposer.name]))
        if isinstance(proposer, LLMProposer):
            llm_usage = dict(proposer.usage)
            if proposer.last_error:
                gathered.errors.append(f"llm proposer: {proposer.last_error}")
            for idea in proposer.code_ideas:
                state.append_idea(cycle, proposer.name, idea)

    # -- evaluate on tune ------------------------------------------------------
    champion_tune = _evaluate(champion, benches, "tune")
    seen_fingerprints = {champion.fingerprint()}
    passing: list[tuple[float, Proposal, AgentConfig, list]] = []
    tried = {p.name: 0 for p in active}
    passed = {p.name: 0 for p in active}
    for index, proposal in enumerate(proposals):
        candidate, rejected_parts = proposal.change.apply(champion)
        record = {"cycle": cycle, "index": index, "proposer": proposal.proposer,
                  "rationale": proposal.rationale, "change": proposal.change.to_dict(),
                  "rejected_parts": rejected_parts, "diff": champion.diff(candidate)}
        fingerprint = candidate.fingerprint()
        if fingerprint in seen_fingerprints:
            state.append("history.jsonl", {**record, "accepted": False, "evaluated": False,
                                           "reason": "duplicate or no-op"})
            continue
        seen_fingerprints.add(fingerprint)
        t0 = time.monotonic()
        ok, reason, comps, gain = tune_verdict(
            champion_tune, _evaluate(candidate, benches, "tune"), opts.policy)
        tried[proposal.proposer] += 1
        passed[proposal.proposer] += int(ok)
        bandit.record(proposal.proposer, ok)
        if ok:
            passing.append((gain, proposal, candidate, comps))
        elif proposal.proposer == LiteratureProposer.name:
            LiteratureProposer.record_rejection(knowledge, proposal, cycle)
        state.append("history.jsonl", {
            **record, "evaluated": True, "passed_tune": ok, "accepted": False, "reason": reason,
            "tune": [c.to_dict() for c in comps], "seconds": round(time.monotonic() - t0, 3),
        })

    # -- gate on holdout and promote ------------------------------------------
    promoted = None
    decision = ("live mode: self-modification paused until "
                f"{MIN_HUMAN_LABELS} human labels exist (see `autoscout queue`)"
                if paused else "no candidate passed tune")
    if passing:
        passing.sort(key=lambda item: item[0], reverse=True)
        gain, proposal, candidate, _ = passing[0]
        ok, reason, comps = holdout_verdict(
            _evaluate(champion, benches, "holdout"), _evaluate(candidate, benches, "holdout"),
            opts.policy)
        decision = f"{proposal.proposer}: {reason}"
        state.append("history.jsonl", {
            "cycle": cycle, "proposer": proposal.proposer, "rationale": proposal.rationale,
            "stage": "holdout", "accepted": ok, "reason": reason,
            "holdout": [c.to_dict() for c in comps], "diff": champion.diff(candidate),
        })
        if ok:
            promoted = {"proposer": proposal.proposer, "rationale": proposal.rationale,
                        "diff": champion.diff(candidate)}
            champion = candidate
        elif proposal.proposer == LiteratureProposer.name:
            LiteratureProposer.record_rejection(knowledge, proposal, cycle)

    if MutationProposer.name in tried:
        MutationProposer.adapt(knowledge, tried[MutationProposer.name], passed[MutationProposer.name])

    scores = score_all(champion, benches)
    if promoted:
        state.save_champion(champion, scores, cycle, promoted["rationale"])
    knowledge.cycle = cycle
    state.save_knowledge(knowledge)

    summary = {
        "cycle": cycle,
        "date": dt.date.today().isoformat(),
        "source": gathered.source,
        "new_papers": len(gathered.works),
        "flagged": sum(v.flagged for v in verdicts),
        "confident_examples": confident,
        "mined_terms": len(mined),
        "harvest_errors": len(gathered.errors),
        "allocation": allocation,
        "evaluated": sum(tried.values()),
        "passed_tune": sum(passed.values()),
        "decision": decision,
        "promoted": promoted,
        "champion": champion.fingerprint(),
        "scores": scores,
        "online_oracle": _oracle(verdicts),
        "llm_usage": llm_usage,
        "mutation_step": round(knowledge.mutation_step, 4),
        "seconds": round(time.monotonic() - started, 2),
        "exhausted": gathered.exhausted,
    }
    state.append("cycles.jsonl", summary)
    state.write_text("REPORT.md", report.render(state, AgentConfig.from_agenda(opts.agenda_path)))
    return summary
