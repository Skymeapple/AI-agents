#!/usr/bin/env bash
# Dissolved-gas model and dam-by-dam vs joint spill on Columbia Basin DART data.
set -euo pipefail
pip install -q numpy pandas scipy
mkdir -p "$DATA/dart"
python -u -W ignore research/hfm-projects/optimization/dam-spill/dam_spill.py "$DATA/dart"
