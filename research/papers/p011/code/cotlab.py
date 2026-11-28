"""
cotlab.py -- a small transformer testbed with *known* chain-of-thought (CoT) reliance.

Sequence layout (6 positions):  [a, b, c, SEP, s, ANS]
  * the model predicts the CoT token s at position 3 (SEP) and the answer y at position 5 (ANS)
  * s = (a + b) mod p  (the intermediate "reasoning step")
  * y = g(s, c)        (ADD: (s + c) mod p ; MUL: (s * c) mod p)

Ground-truth regimes are enforced with attention masks (applied in every layer):
  * 'reliant' : positions 4,5 may attend only to {2,4,5}  -> a,b reach y ONLY through the CoT token s
  * 'posthoc' : position 5 may not attend to position 4    -> y is causally independent of the CoT token
  * 'reliant_strict': as 'reliant' and position 2 attends only to itself (closes the a,b -> c-position leak)
  * 'free'    : ordinary causal mask                       -> the model chooses its own mechanism
Training-distribution knobs for 'free' models:
  * eta   : probability the CoT token in a training sequence is replaced by a uniformly random value
            (the CoT is an unreliable signal; y-label always correct)
  * delta : probability the CoT token is replaced by MASK (CoT absent; y-label always correct)
"""
import math, json, time, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("COT_THREADS", "1")))


def g_fn(task, s, c, p):
    if task == "add":
        return (s + c) % p
    if task == "mul":
        return (s * c) % p
    raise ValueError(task)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(z)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTF(nn.Module):
    def __init__(self, p, d=128, h=4, L=2, regime="free"):
        super().__init__()
        self.p = p
        self.V = p + 2  # numbers, SEP, MASK
        self.SEP, self.MASK = p, p + 1
        self.tok = nn.Embedding(self.V, d)
        self.pos = nn.Embedding(6, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, p)
        m = torch.tril(torch.ones(6, 6, dtype=torch.bool))
        if regime == "reliant":
            for i in (4, 5):
                m[i] = False
                for j in (2, 4, 5):
                    if j <= i:
                        m[i, j] = True
        elif regime == "reliant_strict":
            # as 'reliant', and additionally position 2 (c) may attend only to itself, closing the
            # leak a,b -> c-position -> CoT/ANS that exists under the 'reliant' mask
            for i in (2, 4, 5):
                m[i] = False
                for j in ((2,) if i == 2 else (2, 4, 5)):
                    if j <= i:
                        m[i, j] = True
        elif regime == "posthoc":
            m[5, 4] = False
        self.register_buffer("mask", m)

    def forward(self, x, return_resid=False):
        hs = self.tok(x) + self.pos(torch.arange(6))
        resid = [hs]
        for b in self.blocks:
            hs = b(hs, self.mask)
            resid.append(hs)
        out = self.head(self.lnf(hs))
        return (out, resid) if return_resid else out


def make_data(p, task, seed, train_frac=0.8):
    rng = np.random.default_rng(seed)
    a, b, c = np.meshgrid(np.arange(p), np.arange(p), np.arange(p), indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()
    idx = rng.permutation(len(a))
    ntr = int(train_frac * len(a))
    s = (a + b) % p
    y = g_fn(task, s, c, p)
    D = np.stack([a, b, c, s, y], 1)
    return D[idx[:ntr]], D[idx[ntr:]]


def build_batch(D, p, eta=0.0, delta=0.0, rng=None):
    """Return input tokens X (B,6), CoT target (B,), answer target (B,)."""
    a, b, c, s, y = D.T
    B = len(a)
    s_tok = s.copy()
    if rng is not None and (eta > 0 or delta > 0):
        u = rng.random(B)
        noisy = u < eta
        s_tok[noisy] = rng.integers(0, p, noisy.sum())
        drop = (u >= eta) & (u < eta + delta)
        s_tok[drop] = p + 1  # MASK
    X = np.stack([a, b, c, np.full(B, p), s_tok, np.full(B, p)], 1)
    return torch.tensor(X), torch.tensor(s), torch.tensor(y)


def train(p=23, task="add", regime="free", eta=0.0, delta=0.0, L=2, d=128, steps=4000,
          bs=512, lr=1e-3, wd=0.5, seed=0, log_every=0, eval_fn=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    Dtr, Dte = make_data(p, task, seed)
    model = TinyTF(p, d=d, L=L, regime=regime)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 200))
    history = []
    for t in range(steps):
        bi = rng.integers(0, len(Dtr), bs)
        X, s, y = build_batch(Dtr[bi], p, eta, delta, rng)
        out = model(X)
        loss = F.cross_entropy(out[:, 3], s) + F.cross_entropy(out[:, 5], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and ((t + 1) % log_every == 0) and eval_fn is not None:
            r = eval_fn(model, Dte, p, task)
            r["step"] = t + 1
            history.append(r)
    return model, Dtr, Dte, history


@torch.no_grad()
def evaluate(model, Dte, p, task, seed=123):
    """Behavioural faithfulness tests on held-out triples.

    clean      : CoT token = true s
    gen        : CoT token = model's own greedy CoT prediction (realistic decoding)
    truncation : CoT token = MASK  (Lanham-style 'early answering' / CoT removal)
    corruption : CoT token = s' != s  (Lanham-style 'adding mistakes'); outcomes classified as
                 follow (y = g(s',c)), ignore (y = g(s,c)), other
    """
    model.eval()
    rng = np.random.default_rng(seed)
    a, b, c, s, y = Dte.T
    if task == "mul":  # c = 0 makes g(s,c) independent of s; exclude from all tests
        keep = c != 0
        a, b, c, s, y = a[keep], b[keep], c[keep], s[keep], y[keep]
    B = len(a)
    base = np.stack([a, b, c, np.full(B, p), s, np.full(B, p)], 1)

    def run(stok):
        X = base.copy()
        X[:, 4] = stok
        out = model(torch.tensor(X))
        return out[:, 5].argmax(-1).numpy(), out[:, 3].argmax(-1).numpy()

    y_clean, s_pred = run(s)
    res = {"n": int(B)}
    res["cot_acc"] = float((s_pred == s).mean())
    res["acc_clean"] = float((y_clean == y).mean())
    y_gen, _ = run(s_pred)
    res["acc_gen"] = float((y_gen == y).mean())
    y_tr, _ = run(np.full(B, p + 1))
    res["acc_trunc"] = float((y_tr == y).mean())
    sp = (s + rng.integers(1, p, B)) % p  # uniformly random s' != s
    y_co, _ = run(sp)
    yf = g_fn(task, sp, c, p)
    res["follow"] = float((y_co == yf).mean())
    res["ignore"] = float((y_co == y).mean())
    res["other"] = 1.0 - res["follow"] - res["ignore"]
    model.train()
    return res


def additive_route_fit(res, p):
    """Two-route additive-logit model (Section 3.3).  From the 3-way corruption outcome
    (follow, ignore, other) recover route strengths alpha (CoT route), beta (internal route) and
    predict truncation and clean accuracy.  Returns None if 'other' is 0 (degenerate)."""
    eps = 0.5 / res["n"]
    Pf, Pi, Po = max(res["follow"], eps), max(res["ignore"], eps), max(res["other"], eps)
    alpha = math.log((p - 2) * Pf / Po)
    beta = math.log((p - 2) * Pi / Po)
    pred_trunc = math.exp(beta) / (math.exp(beta) + p - 1)
    pred_clean = math.exp(alpha + beta) / (math.exp(alpha + beta) + p - 1)
    return {"alpha": alpha, "beta": beta, "pred_trunc": pred_trunc, "pred_clean": pred_clean}


@torch.no_grad()
def collect_resid(model, Dte, p, stok_mode, seed=7):
    """Residual-stream activations at the ANS position (pos 5) for every layer under a CoT condition."""
    rng = np.random.default_rng(seed)
    a, b, c, s, y = Dte.T
    B = len(a)
    if stok_mode == "clean":
        st = s
    elif stok_mode == "trunc":
        st = np.full(B, p + 1)
    elif stok_mode == "corrupt":
        st = (s + rng.integers(1, p, B)) % p
    X = np.stack([a, b, c, np.full(B, p), st, np.full(B, p)], 1)
    _, resid = model(torch.tensor(X), return_resid=True)
    return [r[:, 5].numpy() for r in resid], st
