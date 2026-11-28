"""How much does the training-data mix move each domain's score? Toy scale, on CPU.

Scaled-down experiment for two HFM tracks that need GPUs. Both tracks' best proposals
are about document sampling: one compares a 0.60/0.20/0.20 against a 0.50/0.25/0.25
mix of web, code and mathematics on a four-domain score; the other shifts sampling
towards FineWeb-Edu to lower the score on held-out DCLM text. Here a byte-level model
(from ../looped-pretraining-fixed-compute/looped_tiny.py) is trained on a stated mix of
the corpora fetched by corpora.py and scored on held-out text from every corpus.

Each training sequence comes from one corpus, drawn with the mix's probabilities.
Every mix sees the same number of bytes.

Usage: python mix_tiny.py <data dir> <dense-4|looped> <steps> <mix> <seed> [<seed> ...]
       mix is written like web=0.6,code=0.2,math=0.2
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "looped-pretraining-fixed-compute"))
import looped_tiny as lt

EVAL = ("web", "dclm", "code", "math")


def load(folder, name, part):
    path = os.path.join(folder, f"{name}.{part}.txt")
    return np.frombuffer(open(path, "rb").read(), dtype=np.uint8) if os.path.exists(path) else None


def mixed_batches(corpora, probs, steps, seed):
    rng = np.random.default_rng(seed)
    names = list(corpora)
    for _ in range(steps):
        rows = []
        for c in rng.choice(len(names), lt.BATCH, p=probs):
            d = corpora[names[c]]
            j = rng.integers(0, len(d) - lt.CTX - 1)
            rows.append(d[j:j + lt.CTX + 1])
        x = torch.from_numpy(np.stack(rows).astype(np.int64))
        yield x[:, :-1], x[:, 1:]


def main():
    folder, kind, steps, spec = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    mix = {k: float(v) for k, v in (p.split("=") for p in spec.split(",")) if float(v) > 0}
    total = sum(mix.values())
    corpora = {k: load(folder, k, "train") for k in mix}
    probs = [mix[k] / total for k in corpora]
    held = {k: load(folder, k, "heldout") for k in EVAL}
    held = {k: v for k, v in held.items() if v is not None}
    torch.set_num_threads(4)
    for seed in map(int, sys.argv[5:]):
        torch.manual_seed(seed)
        model = lt.LM(kind)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for step, (x, y) in enumerate(mixed_batches(corpora, probs, steps, 1000 + seed), 1):
            loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % max(1, steps // 4) == 0 and step < steps:
                print(f"  {spec} seed {seed} step {step}: train loss {loss.item() / math.log(2):.3f} bits/byte, {time.time() - t0:.0f}s", flush=True)
        scores = {k: lt.evaluate(model, v) for k, v in held.items()}
        print(f"RESULT {kind} {spec} seed {seed}: " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items())
              + f", mean of {len(scores)} {np.mean(list(scores.values())):.4f} bits/byte, train time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
