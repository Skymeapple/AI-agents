#!/usr/bin/env bash
# TOPv2 with retrieved examples and a small open model on CPU, "plain" decoding.
set -euo pipefail
pip install -q pandas pyarrow numpy scikit-learn requests huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='.'))"
for s in train test; do curl -sSL -o $s.parquet https://huggingface.co/api/datasets/WillHeld/top_v2/parquet/default/$s/0.parquet; done
ls -la
cd "$GITHUB_WORKSPACE"
timeout 300m python -u research/hfm-projects/post-training/topv2-parsing/topv2_retrieval_fewshot.py "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" "$DATA/train.parquet" "$DATA/test.parquet" plain "$OUT/plain.jsonl" || echo "stopped early; partial results kept"
wc -l "$OUT/plain.jsonl"
