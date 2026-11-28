#!/usr/bin/env bash
# Reproduce all results (CPU, < 1 minute). Requires python3 with numpy, scipy, matplotlib.
set -e
cd "$(dirname "$0")"
python3 parse_epoch.py   # data/notable_raw.md, data/frontier_raw.md -> data/notable_parsed.csv, results/parse_validation.json
python3 analysis.py      # -> results/empirical.json, fig1-fig4
python3 model.py         # -> results/model.json, fig5
