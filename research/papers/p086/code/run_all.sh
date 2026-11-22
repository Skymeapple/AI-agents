#!/usr/bin/env bash
# Reproduce every number and figure in paper.md (about 30 s on one CPU core).
set -euo pipefail
cd "$(dirname "$0")"
python3 theory_checks.py
python3 fleet.py
python3 montecarlo.py
python3 extras.py
python3 figures.py
