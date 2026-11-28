#!/usr/bin/env bash
# Reproduces every number and figure in paper.md (CPU only, ~3 min total).
set -e
cd "$(dirname "$0")"
for e in e1_contraction e2_redundancy e3_policies e4_eviction e5_estimator; do
  echo "== $e"; python3 $e.py | tee ../results/${e}_log.txt
done
