#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f gru_seeds.py > /dev/null; do sleep 20; done
( for s in 0 1 2; do python3 reservoir_sweep.py --mode balanced2 --seed $s; done > ../results/logs/sweep_balanced2.txt 2>&1 ) &
( for s in 0 1 2; do python3 reservoir_sweep.py --mode landmark --seed $s; done > ../results/logs/sweep_landmark.txt 2>&1 ) &
wait
python3 calibrate_window_decoder.py > ../results/logs/calib.txt 2>&1 &
(cd othello && NTHREADS=1 python3 window_decoders.py > ../../results/logs/window_mlp.txt 2>&1) &
wait
python3 tf_window.py --mode balanced2 --seeds 0 1 2 > ../results/logs/tf_balanced2.txt 2>&1 &
python3 tf_window.py --mode landmark --seeds 0 1 2 > ../results/logs/tf_landmark.txt 2>&1 &
wait
