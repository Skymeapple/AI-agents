#!/usr/bin/env bash
# Terminal-harness stand-in: frozen Qwen2.5-3B on CPU, harness h2, 48 checkable shell tasks (24 dev, 24 held out).
set -euo pipefail
pip install -q huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='.'))"
cd "$GITHUB_WORKSPACE"
timeout 320m python -u research/hfm-projects/agent-harnesses/terminal-agent-harness/shell_harness.py "$DATA/qwen2.5-3b-instruct-q4_k_m.gguf" h2 "$OUT/h2.jsonl" 8 || echo "stopped early; partial results kept"
