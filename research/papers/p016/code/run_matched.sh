#!/bin/bash
# R1 revision: progress-matched optimiser comparison, multi-step short-time test, NTK linearisation
# (8 seeds x 2 scenarios, 4 jobs in parallel; each job takes roughly 8-12 min on one core)
cd "$(dirname "$0")"
for s in 0 1 2 3 4 5 6 7; do for sc in perm split; do echo "$sc $s"; done; done | xargs -P 4 -n 2 python3 mlp_matched.py > ../results/mlpm_log.txt 2>&1
python3 analyze_matched.py > ../results/mlpm_summary.txt
