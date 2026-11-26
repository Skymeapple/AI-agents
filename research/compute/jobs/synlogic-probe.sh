#!/usr/bin/env bash
# SynLogic: look at the public dataset's layout (configs, columns, task names, sample rows).
set -euo pipefail
pip install -q pandas pyarrow requests
python - <<'PY'
import requests, pandas as pd, io, collections
for ds in ("MiniMaxAI/SynLogic",):
    p = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60).json()
    print({k: {s: len(u) for s, u in v.items()} for k, v in p.items()} if isinstance(p, dict) else p)
    for cfg, splits in p.items():
        for split, urls in splits.items():
            if split not in ("validation", "test", "val"):
                continue
            df = pd.read_parquet(io.BytesIO(requests.get(urls[0], timeout=300).content))
            print("\n==", cfg, split, df.shape, list(df.columns))
            for c in df.columns:
                if df[c].map(lambda x: isinstance(x, str)).all() and df[c].nunique() < 80:
                    print(c, dict(collections.Counter(df[c]).most_common(80)))
            for _, r in df.groupby(df.columns[0] if "data_source" not in df.columns else "data_source").head(1).head(40).iterrows():
                print("---"); print({k: str(v)[:700] for k, v in r.items()})
PY
