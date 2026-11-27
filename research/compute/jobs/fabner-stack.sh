#!/usr/bin/env bash
# FabNER: a CRF that is shown the dictionary tags.
set -euo pipefail
pip install -q pandas pyarrow sklearn-crfsuite requests
mkdir -p "$DATA/fabner" && cd "$DATA/fabner"
python - <<'PY'
import json, requests
ds = "DFKI-SLT/fabner"
p = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=60).json()
print({k: list(v) for k, v in p.items()})
cfg = "fabner_bio" if "fabner_bio" in p else list(p)[0]
print("config:", cfg)
for split, urls in p[cfg].items():
    open(f"{split}.parquet", "wb").write(requests.get(urls[0], timeout=300).content)
info = requests.get(f"https://datasets-server.huggingface.co/info?dataset={ds}&config={cfg}", timeout=60).json()
feat = info["dataset_info"]["features"]["ner_tags"]
names = (feat.get("feature") or feat[0])["names"]
print(names)
json.dump(names, open("labels.json", "w"))
PY
ls -la
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/post-training/fabner-extraction/fabner_stack.py "$DATA/fabner/train.parquet" "$DATA/fabner/validation.parquet" "$DATA/fabner/test.parquet" "$DATA/fabner/labels.json"
