"""Exact average-linkage partition for corpora above discover()'s exact limit.

Ground truth for checking the sampled path. Builds the similarity matrix in row
blocks (one large BLAS call segfaults in some containers) and runs the same
_agglomerate the exact path uses, so the only difference from discover() is
that every work is clustered exactly. Needs ~8 bytes * n^2 of memory.

    python bench/exact_reference.py --corpus data/stress_30000.json --save exact.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sciscout.discovery import _agglomerate, _build_tfidf  # noqa: E402
from sciscout.sources.local import LocalCorpus  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=Path, required=True)
parser.add_argument("--save", type=Path, required=True)
parser.add_argument("--threshold", type=float, default=0.16)
args = parser.parse_args()

works = LocalCorpus(args.corpus).harvest(limit=10**9).works
start = time.perf_counter()
matrix, _ = _build_tfidf(works, 2, 0.5)
n = len(works)
similarity = np.empty((n, n))
for row in range(0, n, 4000):
    similarity[row : row + 4000] = matrix[row : row + 4000] @ matrix.T
clusters = _agglomerate(similarity, args.threshold)
tracks = sorted(sorted(works[i].id for i in c) for c in clusters if len(c) >= 3)
unclustered = sorted(works[i].id for c in clusters if len(c) < 3 for i in c)
args.save.write_text(json.dumps({"tracks": tracks, "unclustered": unclustered}))
print(json.dumps({"n": n, "wall_s": round(time.perf_counter() - start, 1),
                  "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                  "clusters": len(tracks)}))
