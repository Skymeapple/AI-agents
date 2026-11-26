# miniF2F validation set, tactics 13 to 18 (split in three so partial results survive)
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
cd "$GITHUB_WORKSPACE"
# hard memory cap enforced by the kernel, so a runaway process is killed instead of the runner
sudo systemd-run --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 --uid=$(id -u) --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  timeout 300m python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/lean_floor.py "$DATA/miniF2F-lean4" "$OUT/floor.json" valid 13:19 || echo "stopped early"
