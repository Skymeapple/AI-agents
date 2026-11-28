"""Variable loop counts and early exit at toy scale, on CPU.

Scaled-down experiment for HFM's "Looped Transformer Pretraining with Recurrent Depth
Architectures" track, which needs GPUs. That track's idea list includes learned exits
and dynamic computation, and its best proposal trains with a balanced 1/2/4-loop
schedule. Three questions can be checked with byte-level models on TinyStories:

  1. Does training with a mix of loop counts cost anything at the usual loop count?
  2. Does a model trained that way improve when given more loops at test time?
  3. How much compute could per-token early exit save, at what loss?

Model: the looped model of the study next door (1 layer, 2 shared layers applied N
times, 1 layer; width 192). Training arms, same bytes in the same order:
  fixed-3      always 3 loops
  mix-124      1, 2 or 4 loops, drawn per step (mean 2.33 loops; HFM's proposed schedule)
  mix-234      2, 3 or 4 loops, drawn per step (mean 3 loops: same training compute as fixed-3)
Each model is scored on held-out text at 1, 2, 3, 4, 6 and 8 loops.

Early exit is simulated without training an exit head: a token leaves after the first
loop count (1, 2, 3, then 4) at which the model's top probability passes a threshold.
The "oracle" row takes each token's best loop count in hindsight; it is not achievable
and only bounds what any exit rule could do.

Usage: python loops_tiny.py <train.txt> <valid.txt> <arm> <steps> <seed> [<seed> ...]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "looped-pretraining-fixed-compute"))
import looped_tiny as lt

ARMS = {"fixed-3": (3,), "mix-124": (1, 2, 4), "mix-234": (2, 3, 4)}
LN2 = float(np.log(2))


@torch.no_grad()
def token_losses(model, valid, loops, n=60):
    """Per-token loss in bits and top probability, on the same held-out batches for every loop count."""
    model.eval()
    model.loops = loops
    rng = np.random.default_rng(12345)
    losses, conf = [], []
    for _ in range(n):
        i = rng.integers(0, len(valid) - lt.CTX - 1, lt.BATCH)
        x = torch.from_numpy(np.stack([valid[j:j + lt.CTX + 1] for j in i]).astype(np.int64))
        logits = model(x[:, :-1])
        losses.append(F.cross_entropy(logits.view(-1, 256), x[:, 1:].reshape(-1), reduction="none") / LN2)
        conf.append(logits.softmax(-1).max(-1).values.view(-1))
    model.train()
    return torch.cat(losses).numpy(), torch.cat(conf).numpy()


def main():
    train = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint8)
    valid = np.frombuffer(open(sys.argv[2], "rb").read()[:4_000_000], dtype=np.uint8)
    arm, steps = sys.argv[3], int(sys.argv[4])
    torch.set_num_threads(4)
    for seed in map(int, sys.argv[5:]):
        torch.manual_seed(seed)
        model = lt.LM("looped")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=0.05)
        pick = np.random.default_rng(777 + seed)
        t0, spent = time.time(), 0
        for step, (x, y) in enumerate(lt.batches(train, steps, 1000 + seed), 1):
            model.loops = int(pick.choice(ARMS[arm]))
            spent += model.loops
            loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % (steps // 4) == 0 and step < steps:
                print(f"  {arm} seed {seed} step {step}: {token_losses(model, valid, 3, 20)[0].mean():.4f} bits/byte at 3 loops, {time.time() - t0:.0f}s", flush=True)
        per = {n: token_losses(model, valid, n) for n in (1, 2, 3, 4, 6, 8)}
        print(f"RESULT {arm} seed {seed}: mean training loops {spent / steps:.2f}, train time {time.time() - t0:.0f}s; held-out bits/byte at loops "
              + ", ".join(f"{n}: {per[n][0].mean():.4f}" for n in per), flush=True)
        L = np.stack([per[n][0] for n in (1, 2, 3, 4)])            # (4, tokens)
        C = np.stack([per[n][1] for n in (1, 2, 3, 4)])
        best = L.argmin(0)
        print(f"RESULT {arm} seed {seed} oracle exit: {L.min(0).mean():.4f} bits/byte at {best.mean() + 1:.2f} mean loops "
              f"(share of tokens best at 1/2/3/4 loops: " + "/".join(f"{(best == k).mean():.2f}" for k in range(4)) + ")")
        for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            exit_at = np.where((C[:3] >= tau).any(0), (C[:3] >= tau).argmax(0), 3)
            bits = L[exit_at, np.arange(L.shape[1])].mean()
            print(f"RESULT {arm} seed {seed} exit when top probability >= {tau}: {bits:.4f} bits/byte at {exit_at.mean() + 1:.2f} mean loops", flush=True)


if __name__ == "__main__":
    main()
