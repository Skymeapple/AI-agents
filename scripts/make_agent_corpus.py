"""Generate a SYNTHETIC agent-research literature for autoscout, clearly labelled.

Two files come out of this, and keeping them separate is the point:

``data/agent_stream.json``
    What the agent "harvests" in offline mode: twelve weekly batches of new
    papers, released one batch per cycle. The agent never sees these labels;
    they are stored under ``extra["_oracle_topic"]`` purely so a report can say
    how well online triage did, and nothing in the decision path reads them.
``data/agent_bench.json``
    The labelled benchmark the agent is scored against when it proposes a change
    to itself. Generated from a different random stream than the stream file, so
    terms mined from the stream are not mined from the items used to judge them.

The literature has one deliberate property: **its vocabulary drifts.** Each
topic has "classic" terms, which the seed agenda knows about, and "emerging"
terms, which it does not. The share of emerging vocabulary rises week by week,
so an agent that never updates what it looks for gets steadily worse at
spotting new work. That is the situation a self-iterating research agent exists
to handle -- and a benchmark on which doing nothing is already optimal would
not test anything.

It is also why the numbers autoscout reports on this corpus are weak evidence.
The generator and the agenda were written together, the drift was planted, and
the "right" terms are sitting in this file. It shows the loop's mechanics are
sound and would catch a regression. It does not tell you how the loop performs
on real papers -- that requires labelled real papers; see
``autoscout.cli queue`` / ``label``.

Every identifier is prefixed ``synthetic:``. None of these are real papers.

Run:
    python scripts/make_agent_corpus.py
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STREAM_OUT = ROOT / "data" / "agent_stream.json"
BENCH_OUT = ROOT / "data" / "agent_bench.json"
STREAM_SEED = 20260924
BENCH_SEED = 7310
WEEKS = 12
PER_WEEK = 70
BENCH_SIZE = 900
START = dt.date(2026, 6, 1)

# Topics on the research agenda. "classic" terms are what the seed agenda in
# autoscout/agenda.yaml knows about; "emerging" terms are not in it anywhere.
RELEVANT: dict[str, dict[str, list[str]]] = {
    "tool_use": {
        "classic": ["tool", "tools", "api", "function-calling", "plugins",
                    "invocation", "external", "calls"],
        "emerging": ["mcp", "connector", "tool-registry", "schema-constrained",
                     "sandboxed-execution", "toolchain"],
    },
    "planning": {
        "classic": ["planning", "plan", "decomposition", "subgoals", "lookahead",
                    "tree-search", "planner"],
        "emerging": ["test-time-compute", "rollouts", "verifier-guided",
                     "backtracking", "deliberation", "search-budget"],
    },
    "memory": {
        "classic": ["memory", "episodic", "long-term", "recall", "retrieval-augmented",
                    "storage-of-experience"],
        "emerging": ["context-compaction", "memory-consolidation", "scratchpad-files",
                     "kv-offloading", "persistent-notes"],
    },
    "multi_agent": {
        "classic": ["multi-agent", "collaboration", "debate", "roles", "coordination",
                    "agent-society"],
        "emerging": ["orchestrator-worker", "subagents", "handoff", "agent-swarm",
                     "delegation"],
    },
    "self_improvement": {
        "classic": ["self-improvement", "self-refine", "reflection", "self-feedback",
                    "iterative-refinement", "bootstrapping"],
        "emerging": ["self-play", "auto-curriculum", "prompt-evolution",
                     "recursive-self-improvement", "skill-library", "self-rewarding"],
    },
    "evaluation": {
        "classic": ["benchmark", "evaluation", "leaderboard", "success-rate",
                    "test-suite"],
        "emerging": ["contamination", "llm-as-judge", "long-horizon-tasks",
                     "swe-bench", "agentic-evals"],
    },
    "efficiency": {
        "classic": ["efficiency", "cost", "inference-cost", "throughput-of-agents",
                    "cheaper", "compute-efficient"],
        "emerging": ["speculative-decoding", "prompt-caching", "token-budget",
                     "early-exit", "model-routing", "distillation-for-agents"],
    },
}

# Off-agenda fields. Several deliberately share words with agenda topics --
# robotics plans, economists simulate "agents", databases cache and retrieve --
# because a triage step that only ever sees easy negatives is not being tested.
DISTRACTORS: dict[str, list[str]] = {
    "computer_vision": ["image", "segmentation", "convolutional", "pixels",
                        "detection", "video", "tracking"],
    "robot_control": ["robot", "manipulation", "actuator", "locomotion", "torque",
                      "planning", "trajectory"],
    "agent_based_economics": ["agent-based", "market", "households", "equilibrium",
                              "agents", "policy", "simulation"],
    "databases": ["query", "index", "transactions", "sql", "storage", "caching",
                  "retrieval"],
    "networking": ["packet", "routing", "congestion", "throughput", "latency",
                   "bandwidth", "protocol"],
    "recommender_systems": ["recommendation", "users", "ranking", "clicks",
                            "retrieval", "embedding"],
}

# Vocabulary every AI paper uses. It cannot tell topics apart and a triage model
# that leans on it will flag everything.
GENERIC_AI = ["language", "model", "models", "large", "llm", "agent", "agents",
              "reasoning", "performance", "learning", "neural", "training",
              "framework", "system", "capabilities", "experiments"]
GENERIC_SCI = ["accuracy", "baseline", "improvement", "evaluate", "setting",
               "problem", "significant", "compared", "challenge", "empirical"]
CONNECTIVES = ["we", "that", "this", "with", "for", "and", "the", "which", "our"]


def _emerging_share(week: float) -> float:
    """Share of topic vocabulary drawn from emerging terms, by week."""
    return 0.08 + 0.80 * week / (WEEKS - 1)


def _paper(rng: random.Random, topic: str, relevant: bool, week: float) -> tuple[str, str]:
    if relevant:
        vocab = RELEVANT[topic]
        share = _emerging_share(week)

        def topic_term() -> str:
            pool = vocab["emerging"] if rng.random() < share else vocab["classic"]
            return rng.choice(pool)
    else:
        def topic_term() -> str:
            return rng.choice(DISTRACTORS[topic])

    cross_topics = [t for t in RELEVANT if t != topic]

    def cross_term() -> str:
        # Adjacent-topic contamination: real papers mention neighbouring ideas.
        other = rng.choice(cross_topics)
        return rng.choice(RELEVANT[other]["classic"] + RELEVANT[other]["emerging"])

    title_words = [topic_term() for _ in range(rng.randint(2, 3))]
    title_words += rng.sample(GENERIC_AI, rng.randint(1, 2))
    rng.shuffle(title_words)

    n = rng.randint(45, 80)
    words: list[str] = []
    # Topic vocabulary is a minority of each abstract, as it is in real ones,
    # and off-agenda papers borrow agenda words often enough to be confusable.
    generic_ai_rate = 0.30 if relevant else 0.18
    cross_rate = 0.07 if relevant else 0.04
    topic_rate = 0.11 if relevant else 0.16
    for _ in range(n):
        r = rng.random()
        if r < topic_rate:
            words.append(topic_term())
        elif r < topic_rate + cross_rate:
            words.append(cross_term())
        elif r < topic_rate + cross_rate + generic_ai_rate:
            words.append(rng.choice(GENERIC_AI))
        elif r < 0.85:
            words.append(rng.choice(GENERIC_SCI))
        else:
            words.append(rng.choice(CONNECTIVES))
    return " ".join(title_words).capitalize(), " ".join(words).capitalize() + "."


def _record(rng, prefix: str, index: int, topic: str, relevant: bool,
            week: float, labelled_key: str) -> dict:
    title, abstract = _paper(rng, topic, relevant, week)
    published = START + dt.timedelta(days=int(week * 7) + rng.randint(0, 6))
    return {
        "id": f"synthetic:{prefix}-{index:05d}",
        "title": title,
        "abstract": abstract,
        "authors": [{"name": f"Author {rng.randint(1, 400)}",
                     "affiliation": f"Institution {chr(65 + rng.randint(0, 25))}"}],
        "published": published.isoformat(),
        "venue": "arXiv",
        "maturity": "preprint",
        "disciplines": ["computer science"],
        "citations": None,
        "source": "synthetic",
        "extra": {
            labelled_key: topic if relevant else "irrelevant",
            "synthetic_field": topic,
            "synthetic_week": round(week, 2),
        },
    }


def _draw_topic(rng: random.Random, relevant_share: float) -> tuple[str, bool]:
    if rng.random() < relevant_share:
        return rng.choice(sorted(RELEVANT)), True
    return rng.choice(sorted(DISTRACTORS)), False


def make_stream() -> list[dict]:
    rng = random.Random(STREAM_SEED)
    records: list[dict] = []
    index = 0
    for week in range(WEEKS):
        for _ in range(PER_WEEK):
            topic, relevant = _draw_topic(rng, 0.5)
            records.append(_record(rng, "stream", index, topic, relevant, week,
                                   "_oracle_topic"))
            index += 1
    return records


def make_bench() -> list[dict]:
    rng = random.Random(BENCH_SEED)
    records = []
    for index in range(BENCH_SIZE):
        topic, relevant = _draw_topic(rng, 0.5)
        # Benchmark items span the whole period, weighted slightly towards the
        # recent end: an agent is judged on the literature as it now is.
        week = min(WEEKS - 1, (WEEKS - 1) * rng.random() ** 0.8)
        records.append(_record(rng, "bench", index, topic, relevant, week, "topic"))
    return records


def main() -> None:
    STREAM_OUT.parent.mkdir(parents=True, exist_ok=True)
    stream, bench = make_stream(), make_bench()
    STREAM_OUT.write_text(json.dumps(stream, separators=(",", ":")))
    BENCH_OUT.write_text(json.dumps(bench, separators=(",", ":")))
    print(f"wrote {len(stream)} stream works ({WEEKS} weeks) to {STREAM_OUT}")
    print(f"wrote {len(bench)} benchmark works to {BENCH_OUT}")


if __name__ == "__main__":
    main()
