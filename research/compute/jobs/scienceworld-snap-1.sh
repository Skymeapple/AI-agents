#!/usr/bin/env bash
# ScienceWorld with no training: small open model, snap mode, part 1 of 2.
set -euo pipefail
java -version 2>&1 | head -1
pip install -q scienceworld huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='.'))"
cd "$GITHUB_WORKSPACE"
P=research/hfm-projects/post-training/scienceworld-training/scienceworld_small_model.py
timeout 300m python -u $P "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" snap 30 "$OUT/snap-1.jsonl" 1 2 || echo 'stopped early; partial results kept'
