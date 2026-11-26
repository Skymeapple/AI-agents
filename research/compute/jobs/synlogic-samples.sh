#!/usr/bin/env bash
# SynLogic: write the structured puzzle records of the easy validation set, for building classical solvers.
set -euo pipefail
pip install -q pandas pyarrow requests
python - <<'PY'
import requests, pandas as pd, io, json, os
p = requests.get("https://huggingface.co/api/datasets/MiniMaxAI/SynLogic/parquet", timeout=60).json()
df = pd.read_parquet(io.BytesIO(requests.get(p["easy"]["validation"][0], timeout=300).content))
with open(os.path.join(os.environ["OUT"], "easy-validation.jsonl"), "w") as f:
    for i, r in enumerate(df.itertuples()):
        e = r.extra_info if isinstance(r.extra_info, dict) else json.loads(r.extra_info)
        g = e["game_data_str"]
        f.write(json.dumps({"i": i, "task": r.data_source.split("/")[-1], "game": json.loads(g) if isinstance(g, str) else g}, ensure_ascii=False, default=str) + "\n")
print(len(df), "records written")
PY
