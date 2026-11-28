"""E6 -- In-context learning as implicit Bayesian inference, and the
memorise-vs-generalise transition with task diversity.

Each training sequence (x_1, y_1, ..., x_k, y_k) comes from a fresh linear world
y = w.x + noise. A transformer trained to predict each y_i from the preceding
pairs must infer w *within the context*, with no weight updates.

  (a) With unlimited task diversity, does its error track the Bayes-optimal
      estimator for a Gaussian prior (ridge, lambda = sigma^2)?
  (b) With a finite pool of M training worlds, does it instead behave like the
      Bayes estimator for that discrete pool (dMMSE) -- memorising the worlds it
      has seen -- and is there a diversity threshold beyond which it switches to
      the general rule (Raventos et al., 2023)?
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from .common import Timer, init_transformer, save_json, train, transformer_apply

DX, K, SIGMA = 5, 16, 0.25


def make_batch(B, rng, pool=None):
    W = pool[rng.integers(0, len(pool), B)] if pool is not None else rng.normal(size=(B, DX))
    X = rng.normal(size=(B, K, DX))
    Y = np.einsum("bkd,bd->bk", X, W) + SIGMA * rng.normal(size=(B, K))
    return X.astype(np.float32), Y.astype(np.float32), W


def tokens(X, Y):
    B = X.shape[0]
    t = np.zeros((B, 2 * K, DX + 1), np.float32)
    t[:, 0::2, :DX] = X
    t[:, 1::2, DX] = Y
    return t


def ridge_pred(X, Y, lam):
    """Predict y_i from (x_1..x_{i-1}, y_1..y_{i-1}) for i = 1..K. Returns [B, K]."""
    B = X.shape[0]
    P = np.zeros((B, K))
    for i in range(1, K):
        Xi, Yi = X[:, :i], Y[:, :i]
        A = np.einsum("bkd,bke->bde", Xi, Xi) + lam * np.eye(DX)
        w = np.linalg.solve(A, np.einsum("bkd,bk->bd", Xi, Yi)[..., None])[..., 0]
        P[:, i] = np.einsum("bd,bd->b", X[:, i], w)
    return P


def ols_pred(X, Y):
    B = X.shape[0]
    P = np.zeros((B, K))
    for i in range(1, K):
        for b in range(B):
            w = np.linalg.pinv(X[b, :i]) @ Y[b, :i]
            P[b, i] = X[b, i] @ w
    return P


def dmmse_pred(X, Y, pool):
    """Bayes posterior mean when w is uniform over the finite pool."""
    B = X.shape[0]
    P = np.zeros((B, K))
    preds_all = np.einsum("bkd,md->bkm", X, pool)  # [B, K, M]
    for i in range(K):
        if i == 0:
            logw = np.zeros((B, len(pool)))
        else:
            logw = -((Y[:, :i, None] - preds_all[:, :i]) ** 2).sum(1) / (2 * SIGMA ** 2)
        logw -= logw.max(1, keepdims=True)
        wts = np.exp(logw)
        wts /= wts.sum(1, keepdims=True)
        P[:, i] = (wts * preds_all[:, i]).sum(1)
    return P


CFG = dict(d_model=64, n_layers=4, n_heads=4, batch=64, steps=5000, lr=1e-3)


def run_one(M, seed=0, quick=False, cfg=CFG):
    rng = np.random.default_rng(seed)
    pool = rng.normal(size=(M, DX)) if M is not None else None
    H = cfg["n_heads"]
    params = init_transformer(jax.random.PRNGKey(seed), cfg["d_model"], cfg["n_layers"], H,
                              2 * K, d_in=DX + 1, d_out=1)

    def loss_fn(p, batch):
        t, y = batch
        out = transformer_apply(p, t, H)[:, 0::2, 0]
        return ((out - y) ** 2).mean()

    def batch_fn(step, r):
        X, Y, _ = make_batch(cfg["batch"], r, pool)
        return tokens(X, Y), Y

    steps = 300 if quick else cfg["steps"]
    with Timer() as tm:
        params, hist = train(params, loss_fn, batch_fn, steps, cfg["lr"], 0.0, log_every=250, seed=seed)
    fwd = jax.jit(lambda p, t: transformer_apply(p, t, H)[:, 0::2, 0])
    ev = {}
    r = np.random.default_rng(999)
    n_ev = 300 if quick else 2000
    for name, pl in [("fresh", None), ("pool", pool)]:
        if name == "pool" and pool is None:
            continue
        X, Y, W = make_batch(n_ev, r, pl)
        pm = np.asarray(fwd(params, tokens(X, Y)))
        res = {"model": ((pm - Y) ** 2).mean(0), "ridge": ((ridge_pred(X, Y, SIGMA ** 2) - Y) ** 2).mean(0),
               "zero": (Y ** 2).mean(0)}
        if not quick:
            res["ols"] = ((ols_pred(X[:300], Y[:300]) - Y[:300]) ** 2).mean(0)
        if pool is not None:
            res["dmmse"] = ((dmmse_pred(X, Y, pool) - Y) ** 2).mean(0)
            # which estimator does the model's *prediction* resemble?
            res["dist_to_dmmse"] = ((pm - dmmse_pred(X, Y, pool)) ** 2).mean(0)
        res["dist_to_ridge"] = ((pm - ridge_pred(X, Y, SIGMA ** 2)) ** 2).mean(0)
        ev[name] = res
    return dict(M=M if M is not None else "inf", seed=seed, eval=ev, history=hist, seconds=tm.elapsed)


def run(quick=False):
    Ms = [8, None] if quick else [8, 64, 512, 4096, None]
    out = {"DX": DX, "K": K, "sigma": SIGMA, "config": CFG, "runs": []}
    for M in Ms:
        r = run_one(M, quick=quick)
        out["runs"].append(r)
        f = r["eval"]["fresh"]
        print(f"M={r['M']}: fresh model MSE(last) {f['model'][-1]:.4f} ridge {f['ridge'][-1]:.4f} "
              f"dist_to_ridge {f['dist_to_ridge'][-4:].mean():.4f}  [{r['seconds']:.0f}s]", flush=True)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    res = run(quick)
    save_json("e6_incontext" + ("_quick" if quick else ""), res)
