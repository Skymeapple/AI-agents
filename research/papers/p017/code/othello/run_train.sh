#!/bin/bash
# chained 28-minute segments of train_gpt.py until 6000 steps
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do
  python3 train_gpt.py --minutes 28 --total_steps 6000 --threads ${1:-2} | tail -2
  python3 -c "import torch;import sys;sys.exit(0 if torch.load('../../results/othello/gpt.pt')['step']>=6000 else 1)" && break
done
