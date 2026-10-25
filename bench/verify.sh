#!/usr/bin/env bash
# Held-out verification round. Usage: bench/verify.sh <baseline-checkout> <data-dir>
set -euo pipefail
B0=$1; D=$2; NEW=$(cd "$(dirname "$0")/.." && pwd); SEED=4242
cd "$NEW"
gen() { python bench/make_stress_corpus.py --seed $SEED --works $1 --rare $2 --out $D/h_$1.json >/dev/null; }
gen 6000 0; gen 12000 0; gen 30000 6; gen 120000 24; gen 240000 48
run() { python bench/run_one.py "$@"; }
for c in data/demo_corpus.json $D/h_6000.json $D/h_12000.json; do
  run --impl $B0 --corpus $c; run --impl . --corpus $c
done
run --impl . --corpus $D/h_30000.json --save $D/h30_new.json
run --impl $B0 --corpus $D/h_30000.json
run --impl $B0 --corpus $D/h_30000.json --by-discipline
python bench/exact_reference.py --corpus $D/h_30000.json --save $D/h30_ref.json
python bench/score_saved.py $D/h_30000.json $D/h30_ref.json
python bench/compare.py $D/h30_ref.json $D/h30_new.json
run --impl . --corpus $D/h_120000.json; run --impl . --corpus $D/h_120000.json
run --impl $B0 --corpus $D/h_120000.json --by-discipline
run --impl . --corpus $D/h_240000.json
