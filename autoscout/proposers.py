"""Proposers: where candidate improvements come from.

Three strategies, which differ in what they know:

``mutation``
    Blind local search over the numeric parameters. Knows nothing about the
    literature; cheap and surprisingly effective at the start. Its step size
    adapts by the classic one-fifth rule -- widen after successes, narrow after
    failures -- so it explores early and refines late.
``literature``
    Turns terms mined from the newest papers (:mod:`autoscout.mining`) into
    profile additions. This is the channel through which "gathering the latest
    advancements" becomes an actual change in the agent.
``llm``
    Asks Claude to read the digest of new papers, the experiment history and
    the search space, and propose changes with a rationale. Optional: it runs
    only when an API key is configured, and the loop works without it. Ideas it
    has that configuration cannot express go to the ideas backlog.

Which strategy gets the evaluation budget is itself learned. A Thompson-sampling
bandit tracks each proposer's acceptance rate and allocates candidate slots
accordingly -- the agent iterating not just on its configuration but on how it
looks for improvements to its configuration.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import SPACE, AgentConfig, Change
from .mining import MinedTerm
from .state import Knowledge

# Parameters that only matter in live harvesting cannot be scored offline, so
# blind search leaves them alone rather than burning evaluations on them.
OFFLINE_PARAMS = sorted(name for name in SPACE if not name.startswith("scout."))
REJECTION_MEMORY_CYCLES = 3


@dataclass
class Proposal:
    proposer: str
    change: Change
    rationale: str
    meta: dict = field(default_factory=dict)


@dataclass
class ProposalContext:
    champion: AgentConfig
    knowledge: Knowledge
    mined: list[MinedTerm]
    digest: str
    history: list[dict]
    rng: np.random.Generator


class Proposer:
    name = "proposer"

    def available(self, ctx: ProposalContext) -> bool:
        return True

    def propose(self, ctx: ProposalContext, n: int) -> list[Proposal]:
        raise NotImplementedError


class MutationProposer(Proposer):
    name = "mutation"

    def propose(self, ctx: ProposalContext, n: int) -> list[Proposal]:
        step = ctx.knowledge.mutation_step
        proposals = []
        for _ in range(n):
            k = 1 if ctx.rng.random() < 0.6 else 2
            names = ctx.rng.choice(OFFLINE_PARAMS, size=k, replace=False)
            changes: dict[str, float] = {}
            for name in names:
                param = SPACE[name]
                current = ctx.champion.get(name)
                if param.log:
                    span = math.log(param.high / param.low)
                    value = math.exp(math.log(current) + ctx.rng.normal(0, step * span))
                else:
                    value = current + ctx.rng.normal(0, step * (param.high - param.low))
                value = param.clip(value)
                if param.integer and value == current:
                    value = param.clip(current + (1 if ctx.rng.random() < 0.5 else -1))
                changes[str(name)] = value
            desc = ", ".join(f"{k} {ctx.champion.get(k):.3g}->{v:.3g}" for k, v in changes.items())
            proposals.append(Proposal(self.name, Change(set_params=changes),
                                      f"local search (step {step:.3f}): {desc}"))
        return proposals

    @staticmethod
    def adapt(knowledge: Knowledge, tried: int, accepted: int) -> None:
        """One-fifth success rule on the mutation step size."""
        if tried == 0:
            return
        rate = accepted / tried
        factor = 1.5 if rate > 0.2 else 0.8
        knowledge.mutation_step = float(min(0.5, max(0.02, knowledge.mutation_step * factor)))


class LiteratureProposer(Proposer):
    name = "literature"

    def available(self, ctx: ProposalContext) -> bool:
        return bool(ctx.mined)

    def propose(self, ctx: ProposalContext, n: int) -> list[Proposal]:
        by_topic: dict[str, list[MinedTerm]] = {}
        for term in ctx.mined:
            by_topic.setdefault(term.topic, []).append(term)
        # Strongest topics first; each proposal adds a topic's top few terms.
        order = sorted(by_topic, key=lambda t: -max(m.z for m in by_topic[t]))
        proposals = []
        for topic in order:
            for size in (3, 6):
                terms = by_topic[topic][:size]
                if size > 3 and len(by_topic[topic]) <= 3:
                    continue
                change = Change(add_terms={topic: {m.term: 1.0 for m in terms}})
                evidence = "; ".join(f"{m.term} (z={m.z:.1f}, {m.topic_docs} papers)" for m in terms)
                proposals.append(Proposal(self.name, change, f"mined from recent {topic} papers: {evidence}",
                                          meta={"terms": {topic: [m.term for m in terms]}}))
        if len(order) > 1:
            combined = Change(add_terms={t: {m.term: 1.0 for m in by_topic[t][:3]} for t in order})
            proposals.insert(1, Proposal(self.name, combined, "top mined terms across all topics",
                                         meta={"terms": {t: [m.term for m in by_topic[t][:3]] for t in order}}))
        return proposals[:n]

    @staticmethod
    def record_rejection(knowledge: Knowledge, proposal: Proposal, cycle: int) -> None:
        for topic, terms in proposal.meta.get("terms", {}).items():
            bucket = knowledge.rejected_terms.setdefault(topic, {})
            for term in terms:
                bucket[term] = cycle

    @staticmethod
    def expire_rejections(knowledge: Knowledge, cycle: int) -> None:
        """Forget old rejections: more evidence may have arrived since."""
        for topic in list(knowledge.rejected_terms):
            knowledge.rejected_terms[topic] = {
                t: c for t, c in knowledge.rejected_terms[topic].items()
                if cycle - c < REJECTION_MEMORY_CYCLES
            }


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string"},
                    "set_params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "add_terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "term": {"type": "string"},
                                "weight": {"type": "number"},
                            },
                            "required": ["topic", "term", "weight"],
                            "additionalProperties": False,
                        },
                    },
                    "remove_terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"topic": {"type": "string"}, "term": {"type": "string"}},
                            "required": ["topic", "term"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rationale", "set_params", "add_terms", "remove_terms"],
                "additionalProperties": False,
            },
        },
        "code_ideas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["proposals", "code_ideas"],
    "additionalProperties": False,
}

LLM_SYSTEM = """You are the research lead of an automated research agent that tracks the \
newest papers on AI agents. The agent triages new papers with keyword profiles per topic \
and clusters them into research directions. You improve it by proposing changes to its \
configuration. Every proposal is evaluated on a labelled benchmark with a paired bootstrap \
and is kept only if it measurably helps without regressing anything, so propose concrete, \
testable changes, each different from the others, and do not re-propose changes the \
history shows were rejected. Terms must be single lowercase tokens of letters and hyphens \
(at least three characters), as they appear in paper text. If you see improvements that \
need code rather than configuration, list them in code_ideas; those go to a human."""


class LLMProposer(Proposer):
    """Claude-backed proposer. Inactive unless credentials are configured."""

    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AUTOSCOUT_MODEL", "claude-opus-5")
        self.last_error: str | None = None
        self.code_ideas: list[str] = []
        self.usage: dict[str, int] = {}

    def available(self, ctx: ProposalContext) -> bool:
        if os.environ.get("AUTOSCOUT_DISABLE_LLM"):
            return False
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False
        import importlib.util
        return importlib.util.find_spec("anthropic") is not None

    def _prompt(self, ctx: ProposalContext, n: int) -> str:
        space = {name: {"low": p.low, "high": p.high, "integer": p.integer, "doc": p.doc,
                        "current": ctx.champion.get(name)} for name, p in SPACE.items()}
        profiles = {t: sorted(p, key=p.get, reverse=True) for t, p in ctx.champion.profiles.items()}
        history = [{k: h.get(k) for k in ("cycle", "proposer", "rationale", "accepted", "reason")}
                   for h in ctx.history[-25:]]
        mined = [{"topic": m.topic, "term": m.term, "z": round(m.z, 2), "papers": m.topic_docs}
                 for m in ctx.mined[:40]]
        return (
            f"Propose up to {n} changes.\n\n"
            f"<search_space>\n{json.dumps(space, indent=1)}\n</search_space>\n\n"
            f"<topic_profiles>\n{json.dumps(profiles)}\n</topic_profiles>\n\n"
            f"<mined_candidate_terms>\n{json.dumps(mined)}\n</mined_candidate_terms>\n\n"
            f"<recent_experiments>\n{json.dumps(history, indent=1)}\n</recent_experiments>\n\n"
            f"<latest_papers_digest>\n{ctx.digest[:12000]}\n</latest_papers_digest>"
        )

    def propose(self, ctx: ProposalContext, n: int) -> list[Proposal]:
        import anthropic

        self.last_error = None
        client = anthropic.Anthropic()
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=LLM_SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": "high",
                               "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
                # Re-run a declined request on Anthropic's recommended fallback
                # model rather than losing the cycle's LLM proposals.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": self._prompt(ctx, n)}],
            )
        except anthropic.APIStatusError as exc:
            self.last_error = f"API error {exc.status_code}: {exc.message}"
            return []
        except anthropic.APIConnectionError as exc:
            self.last_error = f"connection error: {exc}"
            return []

        self.usage = {"input_tokens": response.usage.input_tokens,
                      "output_tokens": response.usage.output_tokens}
        if response.stop_reason == "refusal":
            self.last_error = "request declined"
            return []
        if response.stop_reason == "max_tokens":
            self.last_error = "response truncated at max_tokens"
            return []
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self.last_error = f"unparseable response: {exc}"
            return []
        return self.parse(data, n)

    def parse(self, data: dict, n: int) -> list[Proposal]:
        self.code_ideas = [i for i in data.get("code_ideas", []) if isinstance(i, str)][:10]
        proposals = []
        for raw in data.get("proposals", [])[:n]:
            add: dict[str, dict[str, float]] = {}
            for item in raw.get("add_terms", []):
                add.setdefault(item["topic"], {})[item["term"]] = item["weight"]
            remove: dict[str, list[str]] = {}
            for item in raw.get("remove_terms", []):
                remove.setdefault(item["topic"], []).append(item["term"])
            change = Change(
                set_params={p["name"]: p["value"] for p in raw.get("set_params", [])},
                add_terms=add, remove_terms=remove,
            )
            if not change.is_empty():
                proposals.append(Proposal(self.name, change, raw.get("rationale", "")[:500]))
        return proposals


class Bandit:
    """Thompson sampling over proposers, by gate acceptance rate."""

    def __init__(self, stats: dict[str, dict[str, int]], rng: np.random.Generator):
        self.stats = stats
        self.rng = rng

    def allocate(self, proposers: list[Proposer], slots: int) -> dict[str, int]:
        counts = {p.name: 0 for p in proposers}
        if not proposers:
            return counts
        for _ in range(slots):
            draws = {}
            for p in proposers:
                s = self.stats.get(p.name, {})
                accepted, tried = s.get("accepted", 0), s.get("tried", 0)
                draws[p.name] = self.rng.beta(1 + accepted, 1 + tried - accepted)
            counts[max(draws, key=draws.get)] += 1
        return counts

    def record(self, name: str, accepted: bool) -> None:
        s = self.stats.setdefault(name, {"accepted": 0, "tried": 0})
        s["tried"] += 1
        s["accepted"] += int(accepted)
