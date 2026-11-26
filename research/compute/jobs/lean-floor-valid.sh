# miniF2F (valid set) with no language model: which theorems fall to a single built-in tactic. (fourth run: memory cap, chunks, no bare decide)
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
df -h . | tail -1; free -g | head -2
cd "$GITHUB_WORKSPACE"
timeout 330m python -u research/hfm-projects/agent-harnesses/lean-prover-agent/lean_floor.py "$DATA/miniF2F-lean4" "$OUT/floor.json" valid || echo "stopped early"
