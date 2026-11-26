#!/usr/bin/env bash
# EmbSpatial-Bench and MapQA: find the public copies and print their layout (no images printed).
set -euo pipefail
pip install -q pandas pyarrow requests
python - <<'PY'
import requests, pandas as pd, io, collections
for q in ("EmbSpatial", "MapQA"):
    r = requests.get("https://huggingface.co/api/datasets", params={"search": q, "limit": 20}, timeout=60).json()
    print(q, [(d["id"], d.get("downloads")) for d in r])
    for d in r[:4]:
        try:
            p = requests.get(f"https://huggingface.co/api/datasets/{d['id']}/parquet", timeout=60).json()
            print(" ", d["id"], {k: {s: len(u) for s, u in v.items()} for k, v in p.items()} if isinstance(p, dict) else str(p)[:200])
            if q == "EmbSpatial" and isinstance(p, dict):
                for cfg, splits in p.items():
                    for split, urls in splits.items():
                        h = requests.head(urls[0], allow_redirects=True, timeout=60)
                        print("   ", cfg, split, len(urls), "files; first file bytes", h.headers.get("content-length"))
                        if int(h.headers.get("content-length") or 1e12) < 1.5e9:
                            df = pd.read_parquet(io.BytesIO(requests.get(urls[0], timeout=900).content))
                            print("   ", df.shape, list(df.columns))
                            for c in df.columns:
                                v = df[c].iloc[0]
                                print("     ", c, type(v).__name__, str(v)[:300] if not isinstance(v, (bytes, dict)) else "(binary)")
                                if df[c].map(lambda x: isinstance(x, (str, int))).all() and df[c].nunique() < 40:
                                    print("        values", dict(collections.Counter(df[c]).most_common(40)))
                        break
                    break
        except Exception as e:
            print("  failed", d["id"], repr(e)[:200])
PY
