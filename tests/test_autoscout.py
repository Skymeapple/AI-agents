"""Tests for autoscout, the self-iterating research agent.

As with the rest of the suite, these assert the claims the design rests on,
not merely that the code runs: the gate rejects no-op and noise changes and
accepts a planted real improvement; mining recovers planted emerging
vocabulary; the sealed split is never read by a decision; a cycle is
reproducible from its seed; nothing a language model returns can escape the
declared search space.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from autoscout import loop as loop_module  # noqa: E402
from autoscout.bench import (MIN_HUMAN_LABELS, BenchResult, DiscoveryBenchmark,  # noqa: E402
                             TriageBenchmark, default_benchmarks, load_labelled, split_of)
from autoscout.cli import main as cli_main  # noqa: E402
from autoscout.config import SPACE, AgentConfig, Change  # noqa: E402
from autoscout.gate import GatePolicy, paired_bootstrap, tune_verdict  # noqa: E402
from autoscout.loop import CycleOptions, run_cycle  # noqa: E402
from autoscout.mining import mine, update_statistics  # noqa: E402
from autoscout.proposers import (Bandit, LLMProposer, MutationProposer,  # noqa: E402
                                 ProposalContext)
from autoscout.scout import build_queries, gather_offline  # noqa: E402
from autoscout.state import Knowledge, StateDir  # noqa: E402
from autoscout.triage import triage  # noqa: E402
from make_agent_corpus import RELEVANT  # noqa: E402


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    # Never spend API money from the test suite, even where a key is present.
    monkeypatch.setenv("AUTOSCOUT_DISABLE_LLM", "1")


@pytest.fixture(scope="module")
def benches():
    return default_benchmarks()


@pytest.fixture(scope="module")
def seed():
    return AgentConfig.from_agenda()


def oracle_config() -> AgentConfig:
    """Seed agenda plus every planted emerging term: a known-better agent."""
    config = AgentConfig.from_agenda()
    for topic, vocab in RELEVANT.items():
        for term in vocab["emerging"]:
            config.profiles[topic][term] = 1.0
    return config


def evaluate(config, benches, split="tune"):
    return {b.name: b.evaluate(config, split) for b in benches}


# -- configuration ------------------------------------------------------------

def test_change_is_confined_to_the_declared_space(seed):
    change = Change(
        set_params={"triage.threshold": 1e9, "not.a.param": 1.0, "triage.min_term_hits": 2.7},
        add_terms={"planning": {"rollouts": 1.0, "the": 1.0, "Bad Term!": 1.0},
                   "astrology": {"horoscope": 1.0}},
    )
    config, rejected = change.apply(seed)
    assert config.get("triage.threshold") == SPACE["triage.threshold"].high
    assert config.get("triage.min_term_hits") == 3.0
    assert "rollouts" in config.profiles["planning"]
    assert "the" not in config.profiles["planning"]
    assert any("not.a.param" in r for r in rejected)
    assert any("astrology" in r for r in rejected)
    assert any("bad term!" in r for r in rejected)
    # The base config is never mutated in place.
    assert "rollouts" not in seed.profiles["planning"]


def test_config_round_trips_and_fingerprints(seed):
    again = AgentConfig.from_dict(json.loads(json.dumps(seed.to_dict())))
    assert again.fingerprint() == seed.fingerprint()
    changed, _ = Change(set_params={"triage.threshold": 0.3}).apply(seed)
    assert changed.fingerprint() != seed.fingerprint()
    assert seed.diff(changed) == ["triage.threshold: 0.6 -> 0.3"]


def test_splits_are_stable_and_roughly_proportioned():
    works = load_labelled(Path(__file__).resolve().parent.parent / "data" / "agent_bench.json")
    counts = {s: sum(split_of(w.id) == s for w in works) for s in ("tune", "holdout", "sealed")}
    assert split_of("synthetic:bench-00001") == split_of("synthetic:bench-00001")
    assert 0.5 < counts["tune"] / len(works) < 0.7
    assert counts["holdout"] > 100 and counts["sealed"] > 100


# -- the gate -----------------------------------------------------------------

def test_gate_rejects_a_no_op(seed, benches):
    champion = evaluate(seed, benches)
    ok, reason, _, _ = tune_verdict(champion, evaluate(seed.copy(), benches), GatePolicy())
    assert not ok and "no significant gain" in reason


def test_gate_accepts_a_planted_real_improvement(seed, benches):
    ok, reason, comps, _ = tune_verdict(evaluate(seed, benches), evaluate(oracle_config(), benches),
                                        GatePolicy())
    assert ok, reason
    triage_comp = next(c for c in comps if c.benchmark == "triage")
    assert triage_comp.lower > 0


def test_gate_rejects_pure_noise():
    """A candidate whose per-item gains are symmetric noise must not pass."""
    rng = np.random.default_rng(3)
    ids = [f"w{i}" for i in range(400)]
    base = rng.normal(0.5, 0.3, 400)
    noise = rng.normal(0.0, 0.3, 400)
    noise -= noise.mean() - 0.003  # tiny positive mean, well inside the noise
    champion = {"b": BenchResult("b", "tune", ids, base)}
    candidate = {"b": BenchResult("b", "tune", ids, base + noise)}
    ok, _, comps, _ = tune_verdict(champion, candidate, GatePolicy())
    assert not ok
    assert comps[0].lower < 0 < comps[0].upper


def test_gate_blocks_a_regression_elsewhere():
    ids = [f"w{i}" for i in range(300)]
    ones, zeros = np.ones(300), np.zeros(300)
    champion = {"a": BenchResult("a", "tune", ids, zeros), "b": BenchResult("b", "tune", ids, ones)}
    better_a_worse_b = {"a": BenchResult("a", "tune", ids, ones),
                        "b": BenchResult("b", "tune", ids, ones - 0.05)}
    ok, reason, _, _ = tune_verdict(champion, better_a_worse_b, GatePolicy())
    assert not ok and "regresses b" in reason


def test_paired_bootstrap_refuses_misaligned_items():
    a = BenchResult("x", "tune", ["1", "2"], np.zeros(2))
    b = BenchResult("x", "tune", ["2", "1"], np.zeros(2))
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, GatePolicy())


# -- benchmarks ----------------------------------------------------------------

def test_triage_benchmark_rewards_the_oracle_and_the_seed_misses_recent_work(seed, benches):
    triage_bench = next(b for b in benches if isinstance(b, TriageBenchmark))
    seed_result = triage_bench.evaluate(seed, "sealed")
    oracle_result = triage_bench.evaluate(oracle_config(), "sealed")
    assert oracle_result.metrics["recall"] > seed_result.metrics["recall"] + 0.1
    assert oracle_result.metrics["utility_share"] > 0.95


def test_discovery_benchmark_penalises_fragmentation(seed, benches):
    bench = next(b for b in benches if isinstance(b, DiscoveryBenchmark))
    good = bench.evaluate(seed, "tune").mean
    shattered, _ = Change(set_params={"discovery.threshold": 0.45,
                                      "discovery.min_cluster_size": 2}).apply(seed)
    assert bench.evaluate(shattered, "tune").mean < good - 0.2


# -- learning from the literature ---------------------------------------------

def test_mining_recovers_planted_emerging_vocabulary(seed):
    knowledge = Knowledge()
    for _ in range(8):
        gathered = gather_offline(knowledge)
        update_statistics(knowledge, triage(gathered.works, seed))
    mined = mine(knowledge, seed, limit_per_topic=10)
    emerging = {(t, term) for t, v in RELEVANT.items() for term in v["emerging"]}
    found = {(m.topic, m.term) for m in mined}
    assert len(found & emerging) >= 12
    # Precision matters as much as recall: nominations should be mostly real.
    assert len(found & emerging) / len(found) > 0.6
    # Mining never nominates what the agent already knows.
    assert not any(m.term in seed.profiles[m.topic] for m in mined)


def test_offline_stream_releases_each_week_once():
    knowledge = Knowledge()
    first = gather_offline(knowledge)
    knowledge.seen.update(w.id for w in first.works)
    second = gather_offline(knowledge)
    assert first.works and second.works
    assert not {w.id for w in first.works} & {w.id for w in second.works}
    knowledge.stream_cursor = 10_000
    assert gather_offline(knowledge).exhausted


def test_live_queries_follow_the_learned_profiles(seed):
    learned, _ = Change(add_terms={"memory": {"context-compaction": 3.0}}).apply(seed)
    queries = build_queries(learned, ["cs.AI"])
    memory_query = next(q for q in queries if '"memory"' in q)
    assert '"context compaction"' in memory_query
    assert "cat:cs.AI" in memory_query


# -- proposers ----------------------------------------------------------------

def _ctx(seed, mined=None):
    return ProposalContext(seed, Knowledge(), mined or [], "", [], np.random.default_rng(0))


def test_mutation_stays_in_bounds_and_leaves_live_params_alone(seed):
    for proposal in MutationProposer().propose(_ctx(seed), 50):
        for name, value in proposal.change.set_params.items():
            assert not name.startswith("scout.")
            assert SPACE[name].low <= value <= SPACE[name].high


def test_one_fifth_rule_adapts_step():
    knowledge = Knowledge()
    MutationProposer.adapt(knowledge, tried=10, accepted=5)
    assert knowledge.mutation_step > 0.25
    MutationProposer.adapt(knowledge, tried=10, accepted=0)
    MutationProposer.adapt(knowledge, tried=10, accepted=0)
    assert knowledge.mutation_step < 0.25


def test_llm_proposer_is_off_without_credentials(seed, monkeypatch):
    monkeypatch.delenv("AUTOSCOUT_DISABLE_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not LLMProposer().available(_ctx(seed))


def test_llm_output_cannot_escape_the_search_space(seed):
    proposer = LLMProposer()
    proposals = proposer.parse({
        "proposals": [{
            "rationale": "lower the bar and learn new words",
            "set_params": [{"name": "triage.threshold", "value": -5},
                           {"name": "os.system", "value": 1}],
            "add_terms": [{"topic": "planning", "term": "rollouts", "weight": 99},
                          {"topic": "planning", "term": "rm -rf", "weight": 1}],
            "remove_terms": [],
        }],
        "code_ideas": ["embed abstracts instead of keyword matching"],
    }, n=3)
    assert proposer.code_ideas == ["embed abstracts instead of keyword matching"]
    config, rejected = proposals[0].change.apply(seed)
    assert config.get("triage.threshold") == SPACE["triage.threshold"].low
    assert config.profiles["planning"]["rollouts"] == 3.0  # weight clipped
    assert "rm -rf" not in config.profiles["planning"]
    assert any("os.system" in r for r in rejected)


def test_bandit_favours_the_proposer_that_wins():
    stats = {"good": {"accepted": 30, "tried": 40}, "bad": {"accepted": 1, "tried": 40}}
    bandit = Bandit(stats, np.random.default_rng(0))

    class P:
        def __init__(self, name):
            self.name = name

    allocation = bandit.allocate([P("good"), P("bad")], 100)
    assert allocation["good"] > 90


# -- the loop -----------------------------------------------------------------

def test_cycles_improve_the_agent_and_are_reproducible(tmp_path):
    fingerprints = []
    for run in ("a", "b"):
        state = StateDir(tmp_path / run)
        summaries = [run_cycle(state, CycleOptions(seed=1, candidates=8)) for _ in range(4)]
        fingerprints.append(state.load_champion()[0].fingerprint())
    assert fingerprints[0] == fingerprints[1]

    first, last = summaries[0]["scores"], summaries[-1]["scores"]
    assert last["triage"]["sealed"]["mean"] > first["triage"]["sealed"]["mean"] - 1e-9
    for name in ("champion.json", "knowledge.json", "history.jsonl", "cycles.jsonl",
                 "REPORT.md", "digests/cycle-0001.md"):
        assert (state.path / name).exists(), name
    # Every promotion in the summaries is backed by an accepted holdout record.
    history = state.read_jsonl("history.jsonl")
    accepted = [h for h in history if h.get("stage") == "holdout" and h["accepted"]]
    assert len(accepted) == sum(1 for s in summaries if s["promoted"])


def test_no_decision_ever_reads_the_sealed_split(tmp_path, monkeypatch):
    splits: list[str] = []
    original = loop_module._evaluate

    def spy(config, benches, split):
        splits.append(split)
        return original(config, benches, split)

    monkeypatch.setattr(loop_module, "_evaluate", spy)
    state = StateDir(tmp_path / "s")
    for _ in range(3):
        run_cycle(state, CycleOptions(candidates=6))
    assert "tune" in splits
    assert "sealed" not in splits


def test_holdout_is_consulted_at_most_twice_per_cycle(tmp_path, monkeypatch):
    """Once for champion, once for the finalist -- never per candidate."""
    calls: list[str] = []
    original = loop_module._evaluate

    def spy(config, benches, split):
        calls.append(split)
        return original(config, benches, split)

    monkeypatch.setattr(loop_module, "_evaluate", spy)
    run_cycle(StateDir(tmp_path / "s"), CycleOptions(candidates=12))
    assert calls.count("holdout") <= 2


def test_human_labels_become_a_benchmark_only_once_there_are_enough(tmp_path):
    state = StateDir(tmp_path / "s")
    run_cycle(state, CycleOptions(candidates=2))
    knowledge = state.load_knowledge()
    real = dict(knowledge.recent_works[0])
    queue = tmp_path / "queue.jsonl"
    rows = []
    for i in range(MIN_HUMAN_LABELS):
        work = dict(real["work"], id=f"2609.{i:05d}")
        rows.append(json.dumps({"work": work, "label": "planning" if i % 2 else "irrelevant"}))
    queue.write_text("\n".join(rows[:5]) + "\n")
    assert cli_main(["--state", str(state.path), "label", "--import", str(queue)]) == 0
    assert len(default_benchmarks(labels_path=state.labels_path)) == 2
    queue.write_text("\n".join(rows[5:]) + "\n")
    assert cli_main(["--state", str(state.path), "label", "--import", str(queue)]) == 0
    names = [b.name for b in default_benchmarks(labels_path=state.labels_path)]
    assert "triage-real" in names


def test_label_import_rejects_unknown_topics(tmp_path):
    state = StateDir(tmp_path / "s")
    run_cycle(state, CycleOptions(candidates=1))
    work = state.load_knowledge().recent_works[0]["work"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"work": work, "label": "astrology"}) + "\n")
    assert cli_main(["--state", str(state.path), "label", "--import", str(bad)]) == 2
    assert not state.labels_path.exists() or not state.labels_path.read_text().strip()


def test_live_mode_does_not_let_synthetic_data_steer_the_agent(tmp_path, monkeypatch):
    from autoscout.scout import Gathered

    monkeypatch.setattr(loop_module, "gather_live",
                        lambda config, knowledge, cats: Gathered([], "arxiv", errors=["offline"]))
    state = StateDir(tmp_path / "s")
    summary = run_cycle(state, CycleOptions(live=True, candidates=6))
    assert summary["evaluated"] == 0 and summary["promoted"] is None
    assert "paused" in summary["decision"]
