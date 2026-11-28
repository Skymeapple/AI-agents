"""E7 -- The limits: shortcut learning under spurious correlation.

Understanding fails in a characteristic way. If a training set contains a
feature that predicts the label but is not what the label *means*, a network
can rely on it instead. We add a 2x2 "watermark" to handwritten digits whose
position correlates with the label (digit >= 5) with probability rho, then test
on images where the correlation is reversed.
  (a) How does reliance on the shortcut grow with rho?
  (b) Simplicity bias: at rho = 1 both the digit shape and the watermark are
      perfectly predictive. Which does the network use?
  (c) Do interventions that break the correlation (group reweighting;
      randomising the watermark = causal data augmentation) restore robustness?
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import load_digits

from .common import Timer, save_json
from .mlp import init_mlp, mlp_apply


def watermark(X, side):
    """side: array of 0/1, 0 = top-left patch, 1 = top-right patch, -1 = none."""
    X = X.reshape(-1, 8, 8).copy()
    for i, s in enumerate(side):
        if s == 0:
            X[i, 0:2, 0:2] = 1.0
        elif s == 1:
            X[i, 0:2, 6:8] = 1.0
    return X.reshape(-1, 64)


def make_split(seed):
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = (d.target >= 5).astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    ntr = int(0.6 * len(X))
    return X[perm[:ntr]], y[perm[:ntr]], X[perm[ntr:]], y[perm[ntr:]]


def train(X, y, w, seed, steps=3000, lr=1e-3, wd=1e-4):
    params = init_mlp(jax.random.PRNGKey(seed), [64, 128, 2])
    opt = optax.adamw(lr, weight_decay=wd)
    st = opt.init(params)
    Xj, yj, wj = jnp.asarray(X), jnp.asarray(y), jnp.asarray(w / w.sum(), dtype=jnp.float32)

    def loss(p):
        lp = jax.nn.log_softmax(mlp_apply(p, Xj), -1)
        return -(wj * jnp.take_along_axis(lp, yj[:, None], -1)[:, 0]).sum()

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    for _ in range(steps):
        params, st = step(params, st)
    return params


def acc(params, X, y):
    return float((np.asarray(mlp_apply(params, jnp.asarray(X))).argmax(-1) == y).mean())


def evaluate(params, Xte, yte):
    return dict(aligned=acc(params, watermark(Xte, yte), yte),
                clean=acc(params, Xte, yte),
                conflict=acc(params, watermark(Xte, 1 - yte), yte),
                patch_only=acc(params, watermark(np.zeros_like(Xte), yte), yte))


def run(quick=False):
    rhos = [0.5, 0.9, 1.0] if quick else [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    seeds = [0] if quick else [0, 1, 2, 3, 4]
    steps = 800 if quick else 3000
    out = {"rhos": rhos, "seeds": seeds, "sweep": {}, "remedies": {}}
    for rho in rhos:
        rs = []
        for sd in seeds:
            Xtr, ytr, Xte, yte = make_split(sd)
            rng = np.random.default_rng(100 + sd)
            agree = rng.random(len(ytr)) < rho
            side = np.where(agree, ytr, 1 - ytr)
            p = train(watermark(Xtr, side), ytr, np.ones(len(ytr)), sd, steps)
            rs.append(evaluate(p, Xte, yte))
        out["sweep"][rho] = {k: [r[k] for r in rs] for k in rs[0]}
        print("rho", rho, {k: round(np.mean(v), 3) for k, v in out["sweep"][rho].items()}, flush=True)
    # remedies at rho = 0.95
    rho = 0.95
    for name in ["erm", "group_reweight", "randomise_watermark", "no_watermark"]:
        rs = []
        for sd in seeds:
            Xtr, ytr, Xte, yte = make_split(sd)
            rng = np.random.default_rng(100 + sd)
            agree = rng.random(len(ytr)) < rho
            side = np.where(agree, ytr, 1 - ytr)
            w = np.ones(len(ytr))
            if name == "group_reweight":  # equalise total weight of the 4 (label, side) groups
                for a in (0, 1):
                    for b in (0, 1):
                        g = (ytr == a) & (side == b)
                        if g.any():
                            w[g] = 1.0 / g.sum()
            if name == "randomise_watermark":
                side = rng.integers(0, 2, len(ytr))
            if name == "no_watermark":
                side = -np.ones(len(ytr), int)
            p = train(watermark(Xtr, side), ytr, w, sd, steps)
            rs.append(evaluate(p, Xte, yte))
        out["remedies"][name] = {k: [r[k] for r in rs] for k in rs[0]}
        print("remedy", name, {k: round(np.mean(v), 3) for k, v in out["remedies"][name].items()}, flush=True)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    with Timer() as t:
        res = run(quick)
    res["seconds"] = t.elapsed
    save_json("e7_shortcuts" + ("_quick" if quick else ""), res)
