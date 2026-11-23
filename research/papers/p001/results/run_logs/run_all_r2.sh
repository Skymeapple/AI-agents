#!/bin/bash
# Revision-2 experiments (after run_all_r1.sh and exp1). Logs go to ../results/expNN_log.txt
cd "$(dirname "$0")/../../code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in exp13_chinchilla_horizon exp14_design_law_and_validation exp15_gadre_forecast_test exp16_audit_ceiling exp18_ladder_test exp19_gadre_specification exp20_translation_numbers exp17_insweep_design; do
  n=$(echo $s | cut -d_ -f1); t0=$(date +%s); python3 $s.py > ../results/${n}_log.txt 2>&1; echo "$s exit=$? $(( $(date +%s)-t0 ))s" >> ../results/run_logs/timing_r2.txt; done
echo ALLDONE
