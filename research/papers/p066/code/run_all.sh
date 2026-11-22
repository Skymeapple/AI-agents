#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 theory.py > ../results/theory_log.txt
python3 calibrate.py > ../results/calibrate_log.txt
python3 project.py > ../results/project_log.txt
python3 plot_calib.py
