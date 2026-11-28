#!/usr/bin/env bash
# Reproduces every number and figure in paper.md (about 3 CPU-minutes, 2 processes).
set -e
cd "$(dirname "$0")"
python3 exp1_validate.py        > ../results/exp1_log.txt
python3 exp2_dags.py            > ../results/exp2_log.txt
python3 exp2_summarize.py       > ../results/exp2_summary_log.txt
python3 exp2_figure.py
python3 exp3_hetero_diversity.py > ../results/exp3_log.txt
python3 exp4_calibration.py     > ../results/exp4_log.txt
