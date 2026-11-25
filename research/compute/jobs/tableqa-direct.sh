#!/usr/bin/env bash
# Table QA with a small open model on CPU, "direct" harness.
set -euo pipefail
pip install -q pandas numpy huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='.'))"
curl -sSL -o wtq.zip https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip
unzip -q wtq.zip && ls && ls WikiTableQuestions | head && wc -l WikiTableQuestions/data/pristine-unseen-tables.tsv
cd "$GITHUB_WORKSPACE"
timeout 300m python -u research/hfm-projects/agent-harnesses/table-qa-agent/tableqa_small_model.py "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" "$DATA/WikiTableQuestions" direct 250 "$OUT/direct.jsonl" || echo "stopped early; partial results kept"
wc -l "$OUT/direct.jsonl"
