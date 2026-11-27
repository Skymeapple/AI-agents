# Prover round 3: model sketch repaired with the tactic portfolio, on theorems left by tactics and rounds 1-2: shard 0 of 10.
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
pip install -q huggingface_hub
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cd "$DATA"
python3 -c "from huggingface_hub import hf_hub_download as d; print(d('mradermacher/DeepSeek-Prover-V1.5-RL-GGUF', 'DeepSeek-Prover-V1.5-RL.Q4_K_M.gguf', local_dir='.'))"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
cd "$GITHUB_WORKSPACE"
R=research/compute/results
timeout 330m python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/prover_repair.py "$DATA/miniF2F-lean4" "$DATA/DeepSeek-Prover-V1.5-RL.Q4_K_M.gguf" "$OUT/repair.json" 0/10 4 \
  $R/lean-floor2-test-a/floor.json $R/lean-floor2-test-b/floor.json $R/lean-floor3-test-a/floor.json $R/lean-floor3-test-b/floor.json $R/lean-prover-llm-*/llm.json $R/lean-prover-llm2-*/llm.json || echo "stopped early; partial results kept"
