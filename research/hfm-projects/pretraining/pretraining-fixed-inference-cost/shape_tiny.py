"""Fixed inference cost at toy scale: which shape of transformer is best, and what does longer training buy?

Scaled-down experiment for HFM's "fixed inference cost" pretraining track. The rule
there caps the compute a model may spend per token at inference. Two things follow
that can be checked on a CPU with byte-level models on TinyStories:

  1. Shape. Per-token compute of a transformer is about 12 x layers x width^2, so the
     same budget buys 16 narrow layers, 8 medium ones, 4 wide ones or 2 very wide ones.
     All four shapes below have the same budget to within 0.4%.
  2. Training. The cap is on inference, not training. Doubling the training steps
     leaves inference cost unchanged.

Models reuse the code of the looped study next door (same data order, optimiser and
schedule). Score: bits per byte on held-out text (lower is better).

Usage: python shape_tiny.py <train.txt> <valid.txt> <layers> <width> <steps> <seed> [<seed> ...]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "looped-pretraining-fixed-compute"))
import looped_tiny as lt


def main():
    train = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint8)
    valid = np.frombuffer(open(sys.argv[2], "rb").read()[:4_000_000], dtype=np.uint8)
    layers, width, steps = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    lt.D = width
    torch.set_num_threads(4)
    for seed in map(int, sys.argv[6:]):
        torch.manual_seed(seed)
        model = lt.LM(f"dense-{layers}")
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for step, (x, y) in enumerate(lt.batches(train, steps, 1000 + seed), 1):
            loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % (steps // 4) == 0 and step < steps:
                print(f"  {layers}x{width} seed {seed} step {step}: held-out {lt.evaluate(model, valid, 20):.4f} bits/byte, {time.time() - t0:.0f}s", flush=True)
        print(f"RESULT layers {layers} width {width} seed {seed}: parameters {params / 1e6:.2f}M, per-token compute 12*L*D^2 = {12 * layers * width * width / 1e6:.2f}M, "
              f"steps {steps}, held-out {lt.evaluate(model, valid):.4f} bits/byte, train time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
