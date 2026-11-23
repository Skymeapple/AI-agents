#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f analyze.py > /dev/null; do sleep 15; done
(cd othello && mv ../../results/othello/placement.json ../../results/othello/placement_k0.json 2>/dev/null; NTHREADS=1 python3 placement_decoders.py 8 > ../../results/logs/placement_k8.txt 2>&1) &
python3 tf_window.py --mode landmark --seeds 0 1 2 > ../results/logs/tf_landmark.txt 2>&1 &
wait
