#!/usr/bin/env bash
# EmbSpatial-Bench: write the text fields and object boxes (no images) for CPU analysis.
set -euo pipefail
pip install -q pandas pyarrow requests pillow
python - <<'PY'
import requests, pandas as pd, io, json, os
from PIL import Image
p = requests.get("https://huggingface.co/api/datasets/FlagEval/EmbSpatial-Bench/parquet", timeout=60).json()
df = pd.read_parquet(io.BytesIO(requests.get(p["default"]["test"][0], timeout=900).content))
def plain(x):
    if hasattr(x, "tolist"): x = x.tolist()
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items() if not isinstance(v, bytes)}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x
with open(os.path.join(os.environ["OUT"], "embspatial.jsonl"), "w") as f:
    for r in df.itertuples():
        try:
            w, h = Image.open(io.BytesIO(r.image["bytes"])).size
        except Exception:
            w = h = None
        f.write(json.dumps({"id": r.question_id, "source": r.data_source, "scene": r.scene_id, "relation": r.relation, "question": r.question,
                            "options": plain(r.answer_options), "answer": int(r.answer), "objects": plain(r.objects), "width": w, "height": h}, ensure_ascii=False) + "\n")
print(len(df), "records written")
PY
head -c 1500 "$OUT/embspatial.jsonl"
