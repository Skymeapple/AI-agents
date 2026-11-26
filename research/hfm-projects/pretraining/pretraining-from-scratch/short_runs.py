"""Does a short run rank training recipes the way a long run does? Toy scale, on CPU.

Scaled-down experiment for HFM's "Pretraining architecture development from scratch"
track, which needs GPUs. An agent there has 168 hours and must choose which ideas to
keep from runs much shorter than the final one. This trains the eight recipes of the
factorial study next door (../agentic-pretraining/factorial_tiny.py) for a fraction of
the full step count, each with its own complete learning-rate schedule, so that the
ranking can be compared with the ranking after the full 6,000 steps.

Usage: python short_runs.py <train.txt> <valid.txt> <steps> <seed> [<seed> ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agentic-pretraining"))
import factorial_tiny as ft

ARMS = ("base", "rope", "wsd", "long", "rope+wsd", "rope+long", "wsd+long", "rope+wsd+long")


def main():
    train = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint8)
    valid = np.frombuffer(open(sys.argv[2], "rb").read()[:4_000_000], dtype=np.uint8)
    steps = int(sys.argv[3])
    torch.set_num_threads(4)
    for seed in map(int, sys.argv[4:]):
        for arm in ARMS:
            bpb, secs = ft.train(train, valid, arm, steps, seed, lambda s: None)
            print(f"RESULT {arm} seed {seed}: steps {steps}, held-out {bpb:.4f} bits/byte, train time {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
