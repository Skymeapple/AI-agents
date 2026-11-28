#!/usr/bin/env bash
# Reproduces every number and figure in paper.md (~1 minute on 4 CPU cores).
set -e
cd "$(dirname "$0")"
python3 accounting.py       # Sec 5.1  -> results/accounting.json, fig1
python3 run_mc.py           # Sec 5.2  -> results/mc_summary.json, fig2-4
python3 run_mc.py --clat-low  # Sec 5.5 -> results/mc_summary_clatlow.json
python3 run_mc.py --no-lat    # Sec 5.5 -> results/mc_summary_nolat.json
python3 sobol.py            # Sec 5.4  -> results/sobol.json, fig5
python3 revenue_stress.py   # Sec 5.6  -> results/revenue_stress.json, fig6
python3 theory_check.py     # Sec 4    -> results/theory_check.json, fig7
python3 predictions.py      # Sec 6    -> results/predictions.json
