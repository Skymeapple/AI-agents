#!/bin/bash
# Revision (round 1): 3 seeds x 4 fixed-state families x d_k in {8,16,32}, d_v=64, dense evaluation grid,
# then the overwrite task (W5): each of N live keys written 3 times, target = latest value, d_k=16, 2 seeds.
cd "$(dirname "$0")"
mkdir -p ../results/train_seeds ../results/train_overwrite
{
for s in 0 1 2; do for dk in 8 16 32; do for ar in delta gdelta linear gla; do echo "$ar $dk $s distinct 256 train_seeds"; done; done; done
for s in 0 1; do for ar in delta gdelta linear gla; do echo "$ar 16 $s overwrite 128 train_overwrite"; done; done
} | xargs -P 3 -L 1 bash -c 'python3 train_mqar.py --arch $0 --dk $1 --dv 64 --seed $2 --task $3 --Nmax $4 --outdir $5 --dense_eval 1 --steps 1200 --threads 1 > ../results/$5/log_$0_$1_$2_$3.txt 2>&1'
