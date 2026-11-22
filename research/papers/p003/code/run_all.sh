#!/usr/bin/env bash
# Reproduces every number and figure in paper.md (about 20 min on 4 CPU cores).
set -e
cd "$(dirname "$0")"
python3 exp1_theory_checks.py            # results/exp1.json, fig1_three_mode.png   (~10 s)
python3 exp1b_contrast.py              # results/exp1b.json
python3 exp2_tabular_grpo.py             # results/exp2.json, fig2_tabular_grpo.png (~1 min)
python3 exp2c_race_threshold.py          # results/exp2c.json                       (~4 min)
python3 exp4_sequence.py                 # results/exp4.json, fig4_sequence.png     (~30 s)
python3 exp5_race_grid.py                # results/exp5.json, fig5_race_grid.png    (~3 min)
python3 exp6_kl_paradox.py               # results/exp6.json, fig6_kl_paradox.png   (~3 min)
python3 exp7_boundary_earlywarning.py    # results/exp7.json, fig7_*.png            (~10 min)
