#!/usr/bin/env bash
# Reproduces every number and figure in paper.md (~1.5 min on one CPU core).
set -e
cd "$(dirname "$0")"
python3 exp1_theory_checks.py   > ../results/exp1_stdout.txt
python3 exp2_calibration.py     > ../results/exp2_stdout.txt
python3 exp3_policies.py        > ../results/exp3_policies_stdout.txt
python3 exp4_feedback.py        > ../results/exp4_feedback_stdout.txt
python3 exp5_abm.py 2>/dev/null > ../results/exp5_abm_stdout.txt
echo "done; outputs in ../results"
