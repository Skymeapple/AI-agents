# miniF2F valid set, third round (tactics 34 to 42): one structural step then a fixed portfolio, on theorems rounds one and two left unproved
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
cd "$GITHUB_WORKSPACE"
R=research/compute/results
sudo systemd-run --scope --quiet -p MemoryMax=13G -p MemorySwapMax=0 --uid=$(id -u) --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  timeout 320m python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/lean_floor.py "$DATA/miniF2F-lean4" "$OUT/floor.json" valid 34:43 \
  $R/lean-floor2-valid-a/floor.json $R/lean-floor2-valid-b/floor.json || echo "stopped early"
