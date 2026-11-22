#!/usr/bin/env bash
# Reproduce all results (about 2-3 minutes on one CPU core). Python 3 + numpy, scipy, matplotlib.
set -euo pipefail
cd "$(dirname "$0")"
for s in exp1_bifurcation exp2_verify_props exp3_planner exp4_abm exp5_calibration exp6_sensitivity exp7_beta_critical; do
  echo "== $s"; python3 $s.py > ../results/$s.log
done
echo "done; outputs in ../results"
