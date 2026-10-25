"""Run discover() once from a given checkout and print one JSON line of measurements.

Each run is its own process so peak memory is measured cleanly:

    python bench/run_one.py --impl /path/to/checkout --corpus data/stress_12000.json

Reports wall and CPU time of discover() alone (corpus loading excluded), peak
resident memory of the whole process and of loading alone, quality against the
planted labels, and a fingerprint of the exact partition so two implementations
can be checked for identical output rather than merely similar scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def adjusted_rand(labels_true: list[str], labels_pred: list[str]) -> float:
    def comb2(x: int) -> float:
        return x * (x - 1) / 2

    pairs = Counter(zip(labels_true, labels_pred))
    a = Counter(labels_true)
    b = Counter(labels_pred)
    n = len(labels_true)
    index = sum(comb2(v) for v in pairs.values())
    sum_a = sum(comb2(v) for v in a.values())
    sum_b = sum(comb2(v) for v in b.values())
    expected = sum_a * sum_b / comb2(n)
    maximum = (sum_a + sum_b) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.16)
    parser.add_argument("--by-discipline", action="store_true")
    parser.add_argument("--max-exact", type=int, default=None,
                        help="override discover(max_exact=...) on checkouts that have it")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the partition as JSON for compare.py")
    args = parser.parse_args()

    sys.path.insert(0, str(args.impl.resolve()))
    from sciscout.discovery import discover, discover_by_discipline, evaluate_against_labels
    from sciscout.sources.local import LocalCorpus

    works = LocalCorpus(args.corpus).harvest(limit=10**9).works
    load_rss = peak_rss_mb()
    finder = discover_by_discipline if args.by_discipline else discover

    wall, cpu = time.perf_counter(), time.process_time()
    try:
        kwargs = {"threshold": args.threshold}
        if args.max_exact is not None:
            kwargs["max_exact"] = args.max_exact
        result = finder(works, **kwargs)
    except (ValueError, MemoryError) as exc:
        print(json.dumps({"impl": str(args.impl), "corpus": args.corpus.name,
                          "n": len(works), "error": f"{type(exc).__name__}: {str(exc)[:120]}"}))
        return 0
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu

    partition = sorted(sorted(w.id for w in t.works) for t in result.tracks)
    unclustered = sorted(w.id for w in result.unclustered)
    if args.save:
        args.save.write_text(json.dumps({"tracks": partition, "unclustered": unclustered}))
    fingerprint = hashlib.sha256(json.dumps([partition, unclustered]).encode()).hexdigest()[:16]

    assigned = {}
    for k, t in enumerate(result.tracks):
        for w in t.works:
            assigned[w.id] = f"c{k}"
    truth = [str(w.extra.get("track")) for w in works]
    pred = [assigned.get(w.id, f"u:{w.id}") for w in works]

    scores = evaluate_against_labels(result, works)
    print(json.dumps({
        "impl": args.impl.name, "corpus": args.corpus.name, "n": len(works),
        "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1), "load_rss_mb": round(load_rss, 1),
        "clusters": scores["clusters"],
        "recovered": f"{scores['directions_recovered']}/{scores['true_directions']}",
        "purity": round(scores["mean_purity"], 4),
        "coverage": round(scores["coverage"], 4),
        "ari": round(adjusted_rand(truth, pred), 4),
        "fingerprint": fingerprint,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
