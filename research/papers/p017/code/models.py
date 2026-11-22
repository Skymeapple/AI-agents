"""Sequence encoders: trained GRU LM, untrained (random) GRU, echo-state reservoir, small transformer."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULM(nn.Module):
    def __init__(self, vocab, H=128, E=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, E)
        self.rnn = nn.GRU(E, H, batch_first=True)
        self.out = nn.Linear(H, vocab)
        self.H = H

    def step(self, h, x):
        _, hn = self.rnn(self.emb(x)[:, None], h[None].contiguous())
        return hn[0]

    def forward(self, toks, h0=None):
        hs, _ = self.rnn(self.emb(toks), None if h0 is None else h0[None])
        return self.out(hs), hs


class ESN(nn.Module):
    """h_t = tanh(W h_{t-1} + U x_t + b), with ||W||_2 = lam  =>  the state map is a
    lam-contraction in h for every input (tanh is 1-Lipschitz). Fixed random weights."""

    def __init__(self, vocab, H=128, lam=0.9, seed=0, in_scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(H, H, generator=g)
        W = W / torch.linalg.matrix_norm(W, ord=2) * lam
        self.register_buffer("W", W)
        self.register_buffer("U", torch.randn(vocab, H, generator=g) * in_scale)
        self.register_buffer("b", torch.randn(H, generator=g) * 0.2)
        self.H = H
        self.lam = lam
        self.out = nn.Linear(H, vocab)  # trained readout (set later)

    def step(self, h, x):
        return torch.tanh(h @ self.W.T + self.U[x] + self.b)

    def forward(self, toks, h0=None):
        N, L = toks.shape
        h = torch.zeros(N, self.H) if h0 is None else h0
        hs = []
        for t in range(L):
            h = self.step(h, toks[:, t])
            hs.append(h)
        hs = torch.stack(hs, 1)
        return self.out(hs), hs


class TinyTransformer(nn.Module):
    def __init__(self, vocab, L, d=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d)
        self.outp = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, toks):
        L = toks.shape[1]
        x = self.emb(toks) + self.pos[:L]
        x = self.enc(x, mask=self.mask[:L, :L], is_causal=True)
        hs = self.ln(x)
        return self.outp(hs), hs


def train_lm(model, toks, epochs=10, bs=256, lr=3e-3, seed=0, log=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(toks)
    N = X.shape[0]
    steps = epochs * (N // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(N // bs):
            b = X[perm[i * bs:(i + 1) * bs]]
            logits, _ = model(b[:, :-1])
            # loss on observation tokens only (targets at even positions >=2 of the full seq)
            tgt = b[:, 1:]
            lg = logits[:, 1::2]   # positions of move tokens -> predict obs
            tg = tgt[:, 1::2]
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
            tot += loss.item()
        if log is not None:
            log(f"  epoch {ep} loss {tot / (N // bs):.4f}")
    return model


def fit_readout(features, targets, n_classes, l2=1e-4, iters=300):
    """Multinomial logistic regression with standardisation; returns (W_raw, b_raw)
    such that logits = features @ W_raw.T + b_raw (standardisation folded in)."""
    Xf = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.long)
    mu = Xf.mean(0)
    sd = Xf.std(0) + 1e-6
    Z = (Xf - mu) / sd
    lin = nn.Linear(Z.shape[1], n_classes)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(Z), y) + l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    W = (lin.weight / sd).detach()
    b = (lin.bias - (lin.weight * (mu / sd)).sum(1)).detach()
    return W, b
