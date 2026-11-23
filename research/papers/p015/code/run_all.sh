#!/usr/bin/env bash
# Reproduce every number and figure in paper.md (~60 min on 4 CPU cores; 08 needs PyTorch).
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../data/runs_th11.jsonl ] || bash fetch_data.sh
python3 01_marginal_fits.py            # pooled survival families, CV, extrapolation
python3 02_heterogeneity_index.py      # nonparametric heterogeneity index H, retry gains
python3 03_frailty_glmm.py             # frailty GLMMs: conditional vs pooled Weibull shape; projections
python3 04_robustness.py               # frailty-distribution, data-subset and TH1.0 robustness
python3 05_simulation_identification.py # identification simulation on METR's design
python3 06_figures_and_sensitivity.py  # Wald CIs, capability trend, projection sensitivity, figures
python3 07_passk_heldout.py            # held-out-task pass@k test
python3 08_crossed_model.py            # task-validity audit + crossed agent x task frailty model
python3 09_misspecification_sim.py     # misspecification simulation, exact pooled matching, mechanical-corr check
python3 10_passk_baselines.py          # held-out pass@m vs beta-binomial/IRT-style baselines; H-brackets
python3 11_profile_ci.py               # profile-likelihood intervals for per-agent k
