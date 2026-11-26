#!/usr/bin/env bash
# SynLogic easy validation with a small open model on CPU, "code" route, part 0 of 3.
set -euo pipefail
pip install -q pandas pyarrow numpy nltk sympy math-verify requests huggingface_hub tqdm pydantic markdown inflect english-words
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='.'))"
python -c "
import requests
p = requests.get('https://huggingface.co/api/datasets/MiniMaxAI/SynLogic/parquet', timeout=60).json()
open('easy-validation.parquet', 'wb').write(requests.get(p['easy']['validation'][0], timeout=300).content)"
git clone -q --depth 1 https://github.com/MiniMax-AI/SynLogic synlogic-repo
cd "$GITHUB_WORKSPACE"
timeout 300m python -u research/hfm-projects/post-training/synlogic-training/synlogic_small_model.py run "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" "$DATA/easy-validation.parquet" "$DATA/synlogic-repo" code 0 3 "$OUT/code-0.jsonl" || echo "stopped early; partial results kept"
wc -l "$OUT/code-0.jsonl"
