"""Do separate training-recipe gains add up? A 2 x 2 x 2 factorial at toy scale, on CPU.

Scaled-down experiment for HFM's "Agentic Pretraining with a Frozen Corpus and
Tokenizer" track, which needs GPUs. That track's best proposal stacks three changes
that each helped alone (mixture-of-experts, longer sequences with rotary positions,
and a warmup-stable-decay schedule) and asks whether the gains add. The same question
can be asked of byte-level models on TinyStories with three cheap switches:

  rope   rotary position encoding inside attention instead of a learned position table
  wsd    warmup, then a constant rate, then a linear decay over the second half,
         instead of a one-cycle cosine schedule (same peak rate)
  long   256-byte sequences in batches of 16 instead of 128-byte sequences in batches
         of 32 (the same bytes per step)

Model: 4 layers, width 192, 4 heads (1.9M parameters). Every arm sees the same number
of bytes. Held-out text is scored in 256-byte windows; a model trained on 128-byte
sequences reads each window as two halves, which is how it would be used.

An arm is named by its switches joined with "+", or "base" for none.

Usage: python factorial_tiny.py <train.txt> <valid.txt> <arm> <steps> <seed> [<seed> ...]
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, HEADS = 192, 4
EVAL_WINDOW = 256


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


def rotate(x):
    """Rotary position encoding of (batch, heads, time, dim)."""
    t, d = x.shape[-2], x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))
    ang = torch.arange(t)[:, None] * freq[None]
    cos, sin = ang.cos(), ang.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(), nn.Linear(4 * D, D, bias=False))

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotate(q), rotate(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, D)
        x = x + self.proj(a)
        return x + self.ff(self.n2(x))


class LM(nn.Module):
    def __init__(self, rope, ctx, layers=4):
        super().__init__()
        self.emb = nn.Embedding(256, D)
        self.pos = None if rope else nn.Embedding(ctx, D)
        self.blocks = nn.ModuleList(Block(rope) for _ in range(layers))
        self.out_norm, self.head = RMSNorm(D), nn.Linear(D, 256, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.out_norm(x))


def batches(data, steps, seed, ctx, batch):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = rng.integers(0, len(data) - ctx - 1, batch)
        x = np.stack([data[j:j + ctx + 1] for j in i]).astype(np.int64)
        yield torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


@torch.no_grad()
def evaluate(model, valid, ctx, n=60):
    """Bits per byte over the same 256-byte windows for every arm."""
    model.eval()
    loss, count = 0.0, 0
    for x, y in batches(valid, n, 12345, EVAL_WINDOW, 16):
        for s in range(0, EVAL_WINDOW, ctx):
            loss += F.cross_entropy(model(x[:, s:s + ctx]).reshape(-1, 256), y[:, s:s + ctx].reshape(-1), reduction="sum").item()
            count += y[:, s:s + ctx].numel()
    model.train()
    return loss / count / math.log(2)


def rate(step, steps, wsd, peak=1.5e-3, warm=0.05):
    """Learning rate at a step (1-based)."""
    f = step / steps
    if f < warm:
        return peak * f / warm
    if wsd:
        return peak if f < 0.5 else peak * (1 - f) / 0.5
    return peak * 0.5 * (1 + math.cos(math.pi * (f - warm) / (1 - warm)))


def train(train_data, valid, arm, steps, seed, log=print):
    sw = set(arm.split("+")) - {"base"}
    assert sw <= {"rope", "wsd", "long"}, arm
    ctx, batch = (256, 16) if "long" in sw else (128, 32)
    torch.manual_seed(seed)
    model = LM("rope" in sw, ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    t0 = time.time()
    for step, (x, y) in enumerate(batches(train_data, steps, 1000 + seed, ctx, batch), 1):
        for g in opt.param_groups:
            g["lr"] = rate(step, steps, "wsd" in sw)
        loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (steps // 4, steps // 2, 3 * steps // 4):
            log(f"  {arm} seed {seed} step {step}: held-out {evaluate(model, valid, ctx, 20):.4f} bits/byte, {time.time() - t0:.0f}s")
    return evaluate(model, valid, ctx), time.time() - t0


def main():
    train_data = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint8)
    valid = np.frombuffer(open(sys.argv[2], "rb").read()[:4_000_000], dtype=np.uint8)
    arm, steps = sys.argv[3], int(sys.argv[4])
    torch.set_num_threads(4)
    for seed in map(int, sys.argv[5:]):
        bpb, secs = train(train_data, valid, arm, steps, seed, lambda s: print(s, flush=True))
        print(f"RESULT {arm} seed {seed}: steps {steps}, held-out {bpb:.4f} bits/byte, train time {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
