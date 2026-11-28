#!/bin/bash
# Confound check: at d=128 the value half of the token embedding is 64-dim, so d_v=128 has effective rank <=64.
# Re-run d_v=128 (and d_v=64 control) with embedding width d=512 (value half 256).
cd "$(dirname "$0")"
printf "linear 16 128\ndelta 16 128\nlinear 16 64\n" | xargs -P 3 -L 1 bash -c 'python3 train_mqar.py --arch $0 --dk $1 --dv $2 --d 512 --steps 1200 --threads 1 > ../results/train/log_$0_$1_$2_d512.txt 2>&1'
