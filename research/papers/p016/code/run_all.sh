#!/bin/bash
# Reproduces every number and figure in paper.md (roughly 1.5 h on 4 idle CPU cores).
set -e
cd "$(dirname "$0")"
python3 linear_theory.py          # E1 (grid), E2, E3 preconditioning  -> results/linear_results.json, fig1, fig2
python3 linear_extras.py          # Thm 4 / Thm 6b bound checks, noisy case (E4) -> results/linear_extras.json
python3 linear_projection.py > ../results/linear_projection_log.txt   # E3 projection baselines
python3 linear_revision.py A,B,D,E > ../results/linear_revision_log.txt   # E1 (ODE), E5, E7, prose-number audit
python3 linear_revision.py C >> ../results/linear_revision_log.txt        # E6 SGD
python3 linear_revision.py FIG >> ../results/linear_revision_log.txt      # -> results/linear_revision.json, fig5
./run_mlp.sh > ../results/mlp_log.txt 2>&1   # MLP main sweep -> results/mlp_<scen>_<seed>.json
for s in 0 1; do for sc in perm split; do echo "$sc $s"; done; done | xargs -P 4 -n 2 python3 mlp_damping.py > ../results/mlpdamp_log.txt
python3 analyze_mlp.py > ../results/mlp_summary.txt   # -> results/mlp_summary.json, fig3, fig4
python3 analyze_damping.py        # -> results/mlpdamp_summary.json
python3 audit_mlp_r1.py > ../results/mlp_audit_r1.txt   # -> results/mlp_audit_r1.json
./run_matched.sh                  # matched comparison, short-time, linearisation -> results/mlpm_*, fig6, fig7
./run_recovery.sh                 # recovery time in an over-parameterised MLP -> results/mlps_*, fig9
