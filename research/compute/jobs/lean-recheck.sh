# Re-check the round-3 to 5 Lean proofs in a fresh file and test the IMO statements' hypotheses for contradictions.
set -e
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd "$DATA"
git clone -q --depth 1 https://github.com/yangky11/miniF2F-lean4.git
cd miniF2F-lean4
lake exe cache get 2>&1 | tail -3
cd "$GITHUB_WORKSPACE"
R=research/compute/results
python3 -u research/hfm-projects/agent-harnesses/lean-prover-agent/recheck.py "$DATA/miniF2F-lean4" $R/lean-prover-repair-*/repair.json $R/lean-prover-repair2-*/repair.json $R/lean-prover-repair3-*/repair.json
