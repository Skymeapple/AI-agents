#!/bin/bash
# Reduced remainder of the Experiment B sweep (machine was shared; d_k=64 dropped for gla/delta/gdelta)
cd "$(dirname "$0")"
while pgrep -f "train_mqar.py --arch (linear --dk 64|gla --dk 8)" >/dev/null; do sleep 10; done
{
for ar in gla delta gdelta; do for dk in 16 32 8; do [ "$ar $dk" == "gla 8" ] || echo "$ar $dk 64"; done; done
for ar in linear delta; do for dv in 32 128; do echo "$ar 16 $dv"; done; done
} | xargs -P 2 -L 1 bash -c 'python3 train_mqar.py --arch $0 --dk $1 --dv $2 --steps 1200 --threads 2 > ../results/train/log_$0_$1_$2.txt 2>&1'
