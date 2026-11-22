#!/bin/bash
# Revision-1 full rerun with the scaled optimiser (common.SCALE=1e6, OPT gtol=1e-7)
cd "$(dirname "$0")/../../code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
chainA() { for s in exp2_break_test_real exp3_simulation_power exp4_data_wall exp5_wall_sensitivity exp6_confound_and_wall_detection exp7_signal_shrinkage; do
  t0=$(date +%s); python3 $s.py > ../results/run_logs/$s.out 2>&1; echo "$s exit=$? $(( $(date +%s)-t0 ))s" >> ../results/run_logs/timing.txt; done; }
chainB() { for s in exp9_real_inference exp9b_two_break fig7_real_profile exp8_detectability_realistic exp10_wall_generalised exp11_competing_mechanisms; do
  t0=$(date +%s); python3 $s.py > ../results/run_logs/$s.out 2>&1; echo "$s exit=$? $(( $(date +%s)-t0 ))s" >> ../results/run_logs/timing.txt; done; }
chainA & chainB & wait
echo ALLDONE
