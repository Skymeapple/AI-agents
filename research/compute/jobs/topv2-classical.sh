#!/usr/bin/env bash
# TOPv2: structure coverage and a classical tagger-parser on CPU.
set -euo pipefail
pip install -q pandas pyarrow sklearn-crfsuite requests numpy
mkdir -p "$DATA/topv2" && cd "$DATA/topv2"
python - <<'PY'
import requests
for ds in ("WillHeld/top_v2", "tasksource/top_v2", "facebook/top_v2"):
    r = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60)
    print(ds, r.status_code, r.text[:300])
    if r.status_code == 200 and r.json():
        p = r.json(); cfg = list(p)[0]; print({k: len(v) for k, v in p[cfg].items()})
        for split, urls in p[cfg].items():
            import pandas as pd, io
            d = pd.concat([pd.read_parquet(io.BytesIO(requests.get(u, timeout=600).content)) for u in urls])
            d.to_parquet(f"{split}.parquet"); print(split, d.shape)
        break
PY
ls -la
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/post-training/topv2-parsing/topv2_classical.py "$DATA/topv2/train.parquet" "$DATA/topv2/test.parquet" 12288 50000
