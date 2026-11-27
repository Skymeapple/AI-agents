# Dafny round 3: up to four turns (three repair turns with Dafny's errors) on programs still failing: shard 5 of 10.
set -e
pip install -q huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
curl -sSL -f -m 600 -o dafny.zip https://github.com/dafny-lang/dafny/releases/download/v4.11.0/dafny-4.11.0-x64-ubuntu-22.04.zip
unzip -q dafny.zip
git clone -q --depth 1 https://github.com/sun-wendy/DafnyBench.git
python3 -c "from huggingface_hub import hf_hub_download as d; print(d('bartowski/Qwen2.5-Coder-7B-Instruct-GGUF', 'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf', local_dir='.'))"
cd "$GITHUB_WORKSPACE"
DAFNY_TURNS=4 timeout 330m python3 -u research/hfm-projects/agent-harnesses/dafny-agent/dafny_llm.py "$DATA/dafny/dafny" "$DATA/DafnyBench/DafnyBench/dataset/hints_removed" \
  research/compute/results/dafny-floor/floor.jsonl "$DATA/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" "$OUT/llm.jsonl" 5/10 research/compute/results/dafny-llm2-*/llm.jsonl || echo "stopped early; partial results kept"
