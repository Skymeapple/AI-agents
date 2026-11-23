#!/bin/bash
# Reproduce all results (approx. 1-2 min on 1 CPU core). Outputs go to ../results/
set -e
cd "$(dirname "$0")"
python3 exp1_validate.py
python3 exp3_aggregation.py
python3 exp5_dynamics.py
