"""Score a saved partition against the planted labels in its corpus.

    python bench/score_saved.py data/stress_30000.json exact.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one import adjusted_rand  # noqa: E402

records = json.loads(Path(sys.argv[1]).read_text())
saved = json.loads(Path(sys.argv[2]).read_text())
truth = {r["id"]: r["extra"]["track"] for r in records}
pred = {i: f"c{k}" for k, members in enumerate(saved["tracks"]) for i in members}
pred.update({i: f"u:{i}" for i in saved["unclustered"]})
purities, recovered = [], set()
for members in saved["tracks"]:
    label, count = Counter(truth[i] for i in members).most_common(1)[0]
    purities.append(count / len(members))
    recovered.add(label)
ids = sorted(truth)
print(json.dumps({"clusters": len(saved["tracks"]),
                  "recovered": f"{len(recovered)}/{len(set(truth.values()))}",
                  "purity": round(sum(purities) / len(purities), 4),
                  "ari": round(adjusted_rand([truth[i] for i in ids], [pred[i] for i in ids]), 4)}))
