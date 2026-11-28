"""Benchmarks: how the agent knows whether a change to itself helped.

Every benchmark returns a **per-item** score rather than a single aggregate.
That is what lets :mod:`autoscout.gate` run a paired bootstrap -- the same items
scored under the champion and the candidate -- and ask whether an improvement
is bigger than the noise from which items happen to be in the set. An agent
that accepts any change whose aggregate ticks up will, over enough cycles,
accept a great many changes that are nothing but luck.

Items are split three ways by a stable hash of their id:

``tune`` (60%)
    What proposals are selected on. Freely overfit-able, and the gate assumes so.
``holdout`` (20%)
    Consulted once per cycle, for the single best candidate only, as a
    non-inferiority check. Consulting it for every candidate would turn it into
    a second tuning set within a few cycles.
``sealed`` (20%)
    Never read by any decision. Reported alongside the others so that drift
    between what the agent optimises and what it actually achieves is visible.
    If tune keeps rising and sealed does not, the agent is overfitting its
    benchmark, and the trajectory in the report will show it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sciscout.discovery import discover
from sciscout.models import Work
from sciscout.sources.local import LocalCorpus, _parse_work

from .config import AgentConfig
from .triage import triage

SPLITS = ("tune", "holdout", "sealed")
DEFAULT_BENCH = Path(__file__).resolve().parent.parent / "data" / "agent_bench.json"

# Utility of each triage outcome. A false positive is charged because reading
# costs something; it is charged less than a hit is worth because missing an
# advance is usually worse than skimming one irrelevant abstract.
HIT = 1.0
HIT_WRONG_TOPIC = 0.5
FALSE_POSITIVE = -0.5


def split_of(work_id: str) -> str:
    bucket = int(hashlib.md5(work_id.encode()).hexdigest()[:8], 16) % 10
    return "tune" if bucket < 6 else "holdout" if bucket < 8 else "sealed"


@dataclass
class BenchResult:
    """Per-item scores for one benchmark on one split."""

    name: str
    split: str
    item_ids: list[str]
    scores: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.scores.mean()) if len(self.scores) else 0.0


class Benchmark:
    name = "benchmark"

    def evaluate(self, config: AgentConfig, split: str) -> BenchResult:
        raise NotImplementedError


class TriageBenchmark(Benchmark):
    """Does the agent flag the papers it should, under the right topic?

    Labels live in ``extra[label_key]``: a topic name, or ``"irrelevant"``.
    """

    def __init__(self, works: list[Work], name: str = "triage", label_key: str = "topic"):
        self.name = name
        self.label_key = label_key
        self.by_split = {s: [w for w in works if split_of(w.id) == s] for s in SPLITS}

    def evaluate(self, config: AgentConfig, split: str) -> BenchResult:
        works = self.by_split[split]
        verdicts = {v.work.id: v for v in triage(works, config)}
        ids, scores = [], []
        tp = fp = fn = wrong_topic = 0
        for work in works:
            truth = work.extra.get(self.label_key, "irrelevant")
            relevant = truth != "irrelevant"
            verdict = verdicts[work.id]
            if verdict.flagged and relevant:
                correct = verdict.topic == truth
                scores.append(HIT if correct else HIT_WRONG_TOPIC)
                tp += 1
                wrong_topic += 0 if correct else 1
            elif verdict.flagged:
                scores.append(FALSE_POSITIVE)
                fp += 1
            else:
                scores.append(0.0)
                fn += relevant
            ids.append(work.id)
        n_relevant = tp + fn
        metrics = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / n_relevant if n_relevant else 0.0,
            "topic_accuracy": (tp - wrong_topic) / tp if tp else 0.0,
            # Papers someone has to read per relevant paper that exists. The
            # efficiency side of triage: 1.0 is perfect, higher is waste.
            "reading_load": (tp + fp) / n_relevant if n_relevant else 0.0,
            # Utility as a share of the best achievable (flag exactly the
            # relevant ones, all under the right topic).
            "utility_share": (sum(scores) / (HIT * n_relevant)) if n_relevant else 0.0,
        }
        return BenchResult(self.name, split, ids, np.array(scores, dtype=float), metrics)


class DiscoveryBenchmark(Benchmark):
    """Does clustering the relevant papers recover the agenda's directions?

    Per-item score: 1 if a work sits in the largest cluster its topic formed
    and that cluster is dominated by its topic, else 0. Taking only the largest
    cluster penalises fragmentation -- a topic shattered into five pure shards
    has not been recovered -- which a plain purity score would miss.

    The bootstrap over these scores treats the clustering as fixed and
    resamples items, so it captures which-items noise but not clustering
    instability. That understates uncertainty; the gate's minimum effect size
    is set with that in mind.
    """

    name = "discovery"

    def __init__(self, works: list[Work], label_key: str = "topic"):
        self.label_key = label_key
        relevant = [w for w in works if w.extra.get(label_key, "irrelevant") != "irrelevant"]
        self.by_split = {s: [w for w in relevant if split_of(w.id) == s] for s in SPLITS}

    def evaluate(self, config: AgentConfig, split: str) -> BenchResult:
        works = self.by_split[split]
        result = discover(
            works,
            threshold=config.get("discovery.threshold"),
            min_cluster_size=int(config.get("discovery.min_cluster_size")),
            min_df=int(config.get("discovery.min_df")),
            max_df_ratio=config.get("discovery.max_df_ratio"),
        )
        truth = {w.id: w.extra[self.label_key] for w in works}
        home: dict[str, tuple[int, int]] = {}  # label -> (cluster index, size)
        dominant: dict[int, str] = {}
        for index, track in enumerate(result.tracks):
            counts = Counter(truth[w.id] for w in track.works)
            dominant[index] = counts.most_common(1)[0][0]
            for label, count in counts.items():
                if count > home.get(label, (-1, 0))[1]:
                    home[label] = (index, count)
        placement = {w.id: i for i, t in enumerate(result.tracks) for w in t.works}
        ids, scores = [], []
        for work in works:
            label = truth[work.id]
            index = placement.get(work.id)
            ok = (index is not None and home.get(label, (-1, 0))[0] == index
                  and dominant[index] == label)
            ids.append(work.id)
            scores.append(1.0 if ok else 0.0)
        labels = set(truth.values())
        recovered = {label for label, (i, _) in home.items() if dominant.get(i) == label}
        metrics = {
            "clusters": float(len(result.tracks)),
            "recovery_rate": len(recovered) / len(labels) if labels else 0.0,
            "coverage": result.coverage(),
        }
        return BenchResult(self.name, split, ids, np.array(scores, dtype=float), metrics)


def load_labelled(path: Path) -> list[Work]:
    result = LocalCorpus(path).harvest()
    if result.errors:
        raise ValueError(f"benchmark {path}: {result.errors[0]}")
    return result.works


def load_human_labels(path: Path) -> list[Work]:
    """Real papers labelled by a person, from ``labels.jsonl`` in the state dir.

    This is the benchmark that matters in live operation: the synthetic one
    cannot judge whether a term mined from real arXiv papers helps on real
    arXiv papers. It is kept separate so its (small, growing) size is visible.
    """
    if not path.exists():
        return []
    works = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("label") is None:
            continue
        work = _parse_work(raw["work"], "human-labelled")
        work.extra["human_label"] = raw["label"]
        works.append(work)
    return works


# Below this many labelled real papers, a real-paper benchmark is too small to
# distinguish a change from noise and is left out of gating (but still counted).
MIN_HUMAN_LABELS = 40


def default_benchmarks(bench_path: Path = DEFAULT_BENCH,
                       labels_path: Path | None = None) -> list[Benchmark]:
    works = load_labelled(bench_path)
    benches: list[Benchmark] = [TriageBenchmark(works), DiscoveryBenchmark(works)]
    if labels_path is not None:
        human = load_human_labels(labels_path)
        if len(human) >= MIN_HUMAN_LABELS:
            benches.append(TriageBenchmark(human, name="triage-real", label_key="human_label"))
    return benches
