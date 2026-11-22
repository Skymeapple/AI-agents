#!/bin/bash
# Experiment B sweep: 24 runs, 2 in parallel (~20 min on 4 CPU cores)
cd "$(dirname "$0")"
{
for dk in 8 16; do echo "attn $dk 64"; done
for ar in linear gla delta gdelta; do for dk in 8 16 32 64; do echo "$ar $dk 64"; done; done
for ar in linear delta; do for dv in 16 32 128; do echo "$ar 16 $dv"; done; done
} | xargs -P 2 -L 1 bash -c 'python3 train_mqar.py --arch $0 --dk $1 --dv $2 --steps 1200 --threads 2 > ../results/train/log_$0_$1_$2.txt 2>&1'
