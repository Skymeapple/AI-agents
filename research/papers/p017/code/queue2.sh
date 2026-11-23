#!/bin/bash
cd "$(dirname "$0")"
( for s in 0 1 2; do python3 reservoir_sweep.py --mode balanced2 --seed $s; done > ../results/logs/sweep_balanced2.txt 2>&1 ) &
( for s in 0 1 2; do python3 reservoir_sweep.py --mode landmark --seed $s; done > ../results/logs/sweep_landmark.txt 2>&1 ) &
( python3 tf_window.py --mode balanced2 --seeds 0 1 2 > ../results/logs/tf_balanced2.txt 2>&1; python3 calibrate_window_decoder.py > ../results/logs/calib.txt 2>&1 ) &
wait
