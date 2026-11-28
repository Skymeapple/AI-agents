#!/usr/bin/env bash
# FabNER: dictionary plus a small open model for unseen entities, on CPU.
set -euo pipefail
pip install -q pandas pyarrow sklearn-crfsuite requests numpy scikit-learn huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir=__import__('os').environ['DATA']))"
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
timeout 300m python -u research/hfm-projects/post-training/fabner-extraction/fabner_hybrid.py "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" "$DATA/fabner/train.parquet" "$DATA/fabner/test.parquet" "$DATA/fabner/labels.json" "$OUT/hybrid.jsonl" || echo "stopped early"
