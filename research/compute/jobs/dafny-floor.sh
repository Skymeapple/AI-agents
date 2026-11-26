# (second run: postcondition-derived candidates added) DafnyBench with no language model: how many hint-stripped programs verify as they
# are, and how many more a classical guess-and-prune of loop bounds recovers.
set -e
cd "$DATA"
curl -sSL -f -m 600 -o dafny.zip https://github.com/dafny-lang/dafny/releases/download/v4.11.0/dafny-4.11.0-x64-ubuntu-22.04.zip
unzip -q dafny.zip
git clone -q --depth 1 https://github.com/sun-wendy/DafnyBench.git
dotnet --version || true
./dafny/dafny --version
cd "$GITHUB_WORKSPACE"
timeout 320m python -u research/hfm-projects/agent-harnesses/dafny-agent/dafny_floor.py "$DATA/dafny/dafny" "$DATA/DafnyBench/DafnyBench/dataset/hints_removed" "$OUT/floor.jsonl" || echo "stopped early"
