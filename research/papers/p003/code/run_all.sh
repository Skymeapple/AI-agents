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
# ---- revision round 1 ----
python3 exp2b_partA_bias.py              # results/exp2b.json  (finite-group (G-1)/G correction; <1 s)
python3 exp8_offline_audits.py           # results/exp8.json   (audits at matched budgets; ~3 min)
python3 exp9_meanfield_boundary.py       # results/exp9.json, fig9 (mean-field boundary; ~8 min)
NPROC=3 python3 exp10_robustness.py      # results/exp10.json  (encodings, optimisers, lr/G, Adam; ~6 min on 3 cores)
NPROC=3 python3 exp12_metastability.py   # results/exp12.json  (KL metastability, 32k steps; ~6 min)
# E11 needs the public JudgeBench judgment files (not redistributed here):
#   git clone https://github.com/ScalerLab/JudgeBench /tmp/JudgeBench && (cd /tmp/JudgeBench && git checkout e2c52c28)
python3 exp11_real_judges.py /tmp/JudgeBench   # results/exp11.json
python3 figs_revision.py                 # fig8_audits.png, fig11_real_judges.png
