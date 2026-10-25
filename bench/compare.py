"""Agreement between two saved partitions of the same corpus.

    python bench/compare.py exact.json scaled.json

Prints the adjusted Rand index between them (1.0 means identical up to
relabelling) and the share of works whose cluster-mates are mostly the same.
Unclustered works count as singletons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one import adjusted_rand  # noqa: E402


def labels(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    out = {i: f"c{k}" for k, members in enumerate(data["tracks"]) for i in members}
    out.update({i: f"u:{i}" for i in data["unclustered"]})
    return out


a, b = labels(sys.argv[1]), labels(sys.argv[2])
ids = sorted(a)
assert ids == sorted(b), "partitions cover different works"
print(json.dumps({"ari_between": round(adjusted_rand([a[i] for i in ids], [b[i] for i in ids]), 4),
                  "works": len(ids)}))
