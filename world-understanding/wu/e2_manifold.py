"""E2 -- The manifold hypothesis: complexity is in the ambient space, not the world.

Observations live in R^64 but are generated from d latent factors through a
random smooth nonlinear map. We ask:
  (a) Can the intrinsic dimension d be recovered from samples alone (TwoNN)?
  (b) Does the sample efficiency of a neural network depend on the intrinsic
      dimension d rather than the ambient dimension D? Theory (nonparametric
      rates; Sharma & Kaplan 2022) predicts test error ~ n^(-c/d).
  (c) Real data: what is the intrinsic dimension of handwritten digits (D=64)?
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import Timer, powerlaw_fit, save_json, twonn_id
from .mlp import init_mlp, mlp_apply

D_AMBIENT = 64


def make_task(d, seed, D=D_AMBIENT, J=32, freq=1.5):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(d, 128)) * 2.0
    b1 = rng.uniform(-np.pi, np.pi, 128)
    W2 = rng.normal(size=(128, D)) / np.sqrt(128)
    # frequency norm independent of d, so per-direction wiggliness is matched across d
    omega = rng.normal(size=(d, J)) * freq / np.sqrt(d)
    phase = rng.uniform(0, 2 * np.pi, J)
    amp = rng.normal(size=J) / np.sqrt(J)

    def sample(n, r):
        z = r.uniform(0, 1, (n, d))
        x = np.tanh(z @ W1 + b1) @ W2
        y = np.cos(2 * np.pi * z @ omega + phase) @ amp
        return x.astype(np.float32), y.astype(np.float32)
    return sample


def fit_mlp(xtr, ytr, xva, yva, xte, yte, seed, steps=4000, width=256, lr=1e-3):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xva, xte = [(a - mu) / sd for a in (xtr, xva, xte)]
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    params = init_mlp(jax.random.PRNGKey(seed), [xtr.shape[1], width, width, 1])
    opt = optax.adam(lr)
    st = opt.init(params)

    def loss(p, x, y):
        return ((mlp_apply(p, x)[:, 0] - (y - ym) / ys) ** 2).mean()

    @jax.jit
    def step(p, st, x, y):
        l, g = jax.value_and_grad(loss)(p, x, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    pred = jax.jit(lambda p, x: mlp_apply(p, x)[:, 0] * ys + ym)
    rng = np.random.default_rng(seed)
    bs = min(256, len(xtr))
    best_va, best_te = np.inf, np.inf
    for s in range(1, steps + 1):
        idx = rng.choice(len(xtr), bs, replace=False) if bs < len(xtr) else np.arange(len(xtr))
        params, st = step(params, st, xtr[idx], ytr[idx])
        if s % 100 == 0:
            va = float(((pred(params, xva) - yva) ** 2).mean())
            if va < best_va:
                best_va = va
                best_te = float(((pred(params, xte) - yte) ** 2).mean())
    return best_te / float(yte.var())


def run(quick=False):
    dims = [2, 4, 8] if quick else [2, 3, 4, 6, 8]
    ns = [128, 512, 2048] if quick else [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    seeds = [0] if quick else [0, 1, 2]
    steps = 600 if quick else 4000
    out = {"dims": dims, "ns": ns, "seeds": seeds, "D": D_AMBIENT, "err": {}, "twonn": {}, "fits": {}}
    for d in dims:
        errs = np.zeros((len(ns), len(seeds)))
        ids = []
        for j, sd in enumerate(seeds):
            sample = make_task(d, 100 * d + sd)
            r = np.random.default_rng(sd)
            xte, yte = sample(4000, r)
            xva, yva = sample(1000, r)
            ids.append(twonn_id(xte, seed=sd))
            for i, n in enumerate(ns):
                xtr, ytr = sample(n, r)
                errs[i, j] = fit_mlp(xtr, ytr, xva, yva, xte, yte, sd, steps=steps)
            print(f"d={d} seed={sd} err={np.round(errs[:, j], 4)} id={ids[-1]:.2f}", flush=True)
        out["err"][d] = errs
        out["twonn"][d] = ids
        slope, c, r2 = powerlaw_fit(ns, errs.mean(1))
        boot = []
        rb = np.random.default_rng(0)
        for _ in range(1000):
            boot.append(powerlaw_fit(ns, errs[:, rb.integers(0, len(seeds), len(seeds))].mean(1))[0])
        out["fits"][d] = dict(slope=slope, r2=r2, ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
    # slope * d ~ const?  regress |slope| on 1/d through origin
    inv = np.array([1 / d for d in dims])
    sl = np.array([-out["fits"][d]["slope"] for d in dims])
    c = float(inv @ sl / (inv @ inv))
    out["slope_vs_invd"] = dict(c=c, r2=float(1 - ((sl - c * inv) ** 2).sum() / ((sl - sl.mean()) ** 2).sum()))
    # ambient-dimension control: same d, different D
    amb = {}
    for D in ([16, 64] if quick else [16, 64, 256]):
        sample = make_task(4, 400, D=D)
        r = np.random.default_rng(0)
        xte, yte = sample(4000, r)
        xva, yva = sample(1000, r)
        es = []
        for n in ns:
            xtr, ytr = sample(n, r)
            es.append(fit_mlp(xtr, ytr, xva, yva, xte, yte, 0, steps=steps))
        amb[D] = dict(err=es, slope=powerlaw_fit(ns, es)[0], twonn=twonn_id(xte))
        print(f"ambient D={D}: slope {amb[D]['slope']:.3f}", flush=True)
    out["ambient_control"] = amb
    # real data
    from sklearn.datasets import load_digits
    X = load_digits().data.astype(float)
    out["digits_twonn"] = twonn_id(X + 1e-3 * np.random.default_rng(0).normal(size=X.shape))
    out["digits_pca90"] = int(np.searchsorted(np.cumsum(np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2)
                                              / (np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2).sum(), 0.9) + 1)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    with Timer() as t:
        res = run(quick)
    res["seconds"] = t.elapsed
    save_json("e2_manifold" + ("_quick" if quick else ""), res)
    print({d: res["fits"][d] for d in res["dims"]}, res["slope_vs_invd"], res["digits_twonn"])
