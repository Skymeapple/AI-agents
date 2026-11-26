#!/usr/bin/env bash
# MapQA (second try): look at the public copies and write question/answer text (no images) for a CPU analysis.
set -euo pipefail
pip install -q pandas pyarrow requests
python - <<'PY'
import requests, pandas as pd, io, json, os, pyarrow.parquet as pq
out = open(os.path.join(os.environ["OUT"], "mapqa.jsonl"), "w")
for ds in ("elliot-mllm/mapqa_cleaned", "nimapourjafar/mm_mapqa"):
    p = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60).json()
    for cfg, splits in p.items():
        for split, urls in splits.items():
            print(ds, cfg, split, len(urls), "files")
            for k, u in enumerate(urls[:3] if ds.startswith("elliot") else urls[:1]):
                resp = requests.get(u, timeout=1200)
                raw = resp.content
                try:
                    t = pq.read_table(io.BytesIO(raw))
                except Exception as e:
                    print("  not readable:", u, resp.status_code, len(raw), raw[:200], repr(e)[:100])
                    break
                cols = [c for c in t.column_names if "image" not in c.lower()]
                df = t.select(cols).to_pandas()
                if k == 0:
                    print("  bytes", len(raw), "rows", len(df), "columns", t.column_names)
                    for c in cols:
                        print("   ", c, str(df[c].iloc[0])[:600])
                for i, r in df.iterrows():
                    out.write(json.dumps({"ds": ds, "file": k, "row": int(i), **{c: (r[c].tolist() if hasattr(r[c], "tolist") else r[c]) for c in cols}}, ensure_ascii=False, default=str) + "\n")
out.close()
print("bytes written", os.path.getsize(os.path.join(os.environ["OUT"], "mapqa.jsonl")))
PY
if [ $(stat -c %s "$OUT/mapqa.jsonl") -gt 40000000 ]; then head -c 40000000 "$OUT/mapqa.jsonl" | head -n -1 > "$OUT/t" && mv "$OUT/t" "$OUT/mapqa.jsonl"; fi
