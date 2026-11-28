#!/bin/bash
# Final completion of the Experiment B sweep (run in the revision session):
# Gated DeltaNet at d_k in {8,16,32} and the value-dimension sweep (linear, delta at d_k=16, d_v in {32,128}).
cd "$(dirname "$0")"
{
for dk in 16 8 32; do echo "gdelta $dk 64"; done
for ar in linear delta; do for dv in 32 128; do echo "$ar 16 $dv"; done; done
} | xargs -P 2 -L 1 bash -c 'python3 train_mqar.py --arch $0 --dk $1 --dv $2 --steps 1200 --threads 1 > ../results/train/log_$0_$1_$2.txt 2>&1'
