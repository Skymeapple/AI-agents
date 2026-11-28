# Strict audit of every counted Lean proof (no error inside the theorem, only standard axioms): shard 3 of 10.
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
cd "$GITHUB_WORKSPACE"
R=research/compute/results
timeout 340m python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/audit.py "$DATA/miniF2F-lean4" "$OUT/audit.json" 3/10 $R/lean-floor2-*/floor.json $R/lean-floor3-*/floor.json $R/lean-prover-llm-*/llm.json $R/lean-prover-llm2-*/llm.json $R/lean-prover-repair-*/repair.json $R/lean-prover-repair2-*/repair.json $R/lean-prover-repair3-*/repair.json || echo "stopped early; partial results kept"
