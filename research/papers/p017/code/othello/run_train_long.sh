#!/bin/bash
# continuation phase: from the 6000-step checkpoint, 4000 further steps (schedule shortened from 8000 at step 800) (warm-up 5%, cosine-like one-cycle decay, peak lr 6e-4)
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  python3 train_gpt.py --minutes 28 --total_steps 4000 --lr 6e-4 --init gpt_6k.pt --ckpt gpt_long.pt --threads 2 | tail -1
  python3 -c "import torch,sys;sys.exit(0 if torch.load('../../results/othello/gpt_long.pt')['step']>=4000 else 1)" && break
done
