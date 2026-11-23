#!/bin/bash
# Reproduces every number and figure in paper.md (about 35 min on 4 CPU cores).
set -e
cd "$(dirname "$0")"
python3 linear_theory.py          # E1, E2, preconditioning (linear)  -> results/linear_results.json, fig1, fig2
python3 linear_extras.py          # Thm 4 / Thm 6b bound checks, noisy case -> results/linear_extras.json
./run_mlp.sh > ../results/mlp_log.txt 2>&1   # MLP main sweep -> results/mlp_<scen>_<seed>.json
for s in 0 1; do for sc in perm split; do echo "$sc $s"; done; done | xargs -P 4 -n 2 python3 mlp_damping.py > ../results/mlpdamp_log.txt
python3 analyze_mlp.py > ../results/mlp_summary.txt   # -> results/mlp_summary.json, fig3, fig4
python3 analyze_damping.py        # -> results/mlpdamp_summary.json
