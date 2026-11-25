"""Looped against standard transformers at toy scale, on CPU.

Scaled-down experiment for HFM's looped-pretraining tracks. HFM trains on 4 to 8
GPUs for days; this trains byte-level models of a few million parameters for about
an hour on four CPU cores, on the public TinyStories corpus
(https://huggingface.co/datasets/roneneldan/TinyStories). It cannot say what happens
at HFM's scale. It can say whether an idea helps at all when everything else is equal.

Models (width 192, 4 heads, context 128 bytes)
  dense-8        8 separate layers                    (most parameters, full compute)
  dense-4        4 separate layers                    (same parameters as the looped models)
  looped         1 layer, then 2 shared layers applied 3 times, then 1 layer
                 (8 layers of compute, 4 layers of parameters)
  looped+index   the same, plus a learned embedding added at the start of each loop
                 and a separate normalisation per loop. This is the idea HFM's agents
                 proposed for letting a shared block specialise by depth.

All models see the same bytes in the same order for the same number of steps.
Score: bits per byte on held-out text (lower is better).

Usage: python looped_tiny.py <train.txt> <valid.txt> <model> [steps] [seeds]
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, HEADS, CTX, BATCH = 192, 4, 128, 32


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(), nn.Linear(4 * D, D, bias=False))

    def forward(self, x, n1=None, n2=None):
        b, t, _ = x.shape
        q, k, v = self.qkv((n1 or self.n1)(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, D)
        x = x + self.proj(a)
        return x + self.ff((n2 or self.n2)(x))


class LM(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb, self.pos = nn.Embedding(256, D), nn.Embedding(CTX, D)
        self.out_norm, self.head = RMSNorm(D), nn.Linear(D, 256, bias=False)
        if kind.startswith("dense"):
            self.blocks = nn.ModuleList(Block() for _ in range(int(kind.split("-")[1])))
        else:
            self.pre, self.post = Block(), Block()
            self.core = nn.ModuleList(Block() for _ in range(2))
            self.loops = 3
            if kind == "looped+index":
                self.loop_emb = nn.Parameter(torch.zeros(self.loops, D))
                self.loop_norms = nn.ModuleList(nn.ModuleList(RMSNorm(D) for _ in range(4)) for _ in range(self.loops))

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        if self.kind.startswith("dense"):
            for b in self.blocks:
                x = b(x)
        else:
            x = self.pre(x)
            for l in range(self.loops):
                if self.kind == "looped+index":
                    x = x + self.loop_emb[l]
                    n = self.loop_norms[l]
                    x = self.core[0](x, n[0], n[1])
                    x = self.core[1](x, n[2], n[3])
                else:
                    for b in self.core:
                        x = b(x)
            x = self.post(x)
        return self.head(self.out_norm(x))


def batches(data, steps, seed):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = rng.integers(0, len(data) - CTX - 1, BATCH)
        x = np.stack([data[j:j + CTX + 1] for j in i]).astype(np.int64)
        yield torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


@torch.no_grad()
def evaluate(model, valid, n=60):
    model.eval()
    loss = 0.0
    for x, y in batches(valid, n, 12345):
        loss += F.cross_entropy(model(x).view(-1, 256), y.reshape(-1)).item()
    model.train()
    return loss / n / math.log(2)


def main():
    train = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint8)
    valid = np.frombuffer(open(sys.argv[2], "rb").read()[:4_000_000], dtype=np.uint8)
    kind = sys.argv[3]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    torch.set_num_threads(4)
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = LM(kind)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for step, (x, y) in enumerate(batches(train, steps, 1000 + seed), 1):   # same data order for every model with this seed
            loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step in (steps // 4, steps // 2, 3 * steps // 4):
                print(f"  {kind} seed {seed} step {step}: held-out {evaluate(model, valid, 20):.4f} bits/byte, {time.time() - t0:.0f}s", flush=True)
        print(f"RESULT {kind} seed {seed}: parameters {params / 1e6:.2f}M, steps {steps}, bytes seen {steps * BATCH * CTX / 1e6:.0f}M, "
              f"held-out {evaluate(model, valid):.4f} bits/byte, train time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
