# miniF2F test set, tactics 0 to 16: first-round tactics rerun with per-theorem recovery, plus second-round compositions (rerun: job was skipped)
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
  timeout 320m python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/lean_floor.py "$DATA/miniF2F-lean4" "$OUT/floor.json" test 0:17 || echo "stopped early"
