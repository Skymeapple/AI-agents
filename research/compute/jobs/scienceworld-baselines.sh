#!/usr/bin/env bash
# ScienceWorld with no training: gold replay check and random agent.
set -euo pipefail
java -version 2>&1 | head -1
pip install -q scienceworld huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
true
cd "$GITHUB_WORKSPACE"
P=research/hfm-projects/post-training/scienceworld-training/scienceworld_small_model.py
python -u $P - gold 200 "$OUT/gold.jsonl"
python -u $P - random 30 "$OUT/random.jsonl"
