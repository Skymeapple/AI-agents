# miniF2F with no language model: which theorems fall to a single built-in tactic.
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
df -h . | tail -1
cd "$GITHUB_WORKSPACE"
timeout 330m python -u research/hfm-projects/agent-harnesses/lean-prover-agent/lean_floor.py "$DATA/miniF2F-lean4" "$OUT/floor.json" || echo "stopped early"
