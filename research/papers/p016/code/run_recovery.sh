#!/bin/bash
# R1 revision: GD recovery time vs rho in an over-parameterised MLP (2 seeds x 2 scenarios, 4 jobs in parallel)
cd "$(dirname "$0")"
for s in 0 1; do for sc in perm split; do echo "$sc $s"; done; done | xargs -P 4 -n 2 sh -c 'METHODS=GD RHOS=0.3,0.1,0.03,0.01 python3 mlp_speedlimit.py $0 $1 > ../results/mlps_log_$0_$1.txt 2>&1'
python3 analyze_mlp_speedlimit.py > ../results/mlps_summary.txt
