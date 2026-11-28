"""E4 -- Superposition: how a finite network represents a world with more
concepts than it has dimensions.

Toy model of superposition (Elhage et al., 2022): n sparse features are squeezed
through m < n hidden dimensions and reconstructed with a ReLU readout,
  h = W x,  x_hat = ReLU(W^T h + b).
Each feature is active with probability (1 - S). Prediction: as features get
sparser, the model stores more than m of them, in nearly-orthogonal
superposition; a linear model never does (it keeps the top-m principal
features). Real-world concepts are sparse (any given scene/sentence involves
few of them), so this is a candidate mechanism for how bounded networks hold
very large concept inventories.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import Timer, save_json


def train_toy(n, m, S, importance, seed, relu=True, steps=8000, batch=1024, lr=1e-3):
    key = jax.random.PRNGKey(seed)
    k1, key = jax.random.split(key)
    params = {"W": jax.random.normal(k1, (m, n)) * np.sqrt(2.0 / (m + n)), "b": jnp.zeros(n)}
    imp = jnp.asarray(importance)
    opt = optax.adam(lr)
    st = opt.init(params)

    def loss(p, x):
        h = x @ p["W"].T
        out = h @ p["W"] + p["b"]
        out = jax.nn.relu(out) if relu else out
        return (imp * (x - out) ** 2).sum(-1).mean()

    @jax.jit
    def step(p, st, k):
        ka, kb = jax.random.split(k)
        x = jax.random.uniform(ka, (batch, n)) * (jax.random.uniform(kb, (batch, n)) > S)
        l, g = jax.value_and_grad(loss)(p, x)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    for _ in range(steps):
        key, k = jax.random.split(key)
        params, st, l = step(params, st, k)
    # final loss on fresh batch
    key, ka, kb = jax.random.split(key, 3)
    x = jax.random.uniform(ka, (20000, n)) * (jax.random.uniform(kb, (20000, n)) > S)
    return np.asarray(params["W"]), np.asarray(params["b"]), float(loss(params, x))


def metrics(W):
    norms = np.linalg.norm(W, axis=0)
    Wn = W / (norms + 1e-9)
    G = Wn.T @ W  # [n, n] projections of each feature on the direction of feature i
    interference = ((G ** 2).sum(1) - np.diag(G) ** 2)  # sum_{j!=i} (W_i_hat . W_j)^2
    represented = norms > 0.5
    return dict(norms=norms, represented=int(represented.sum()),
                dims_per_feature=float(W.shape[0] / (np.linalg.norm(W) ** 2 + 1e-9)),
                mean_interference_represented=float(interference[represented].mean()) if represented.any() else 0.0,
                gram=(W.T @ W))


def run(quick=False):
    n, m = 20, 5
    importance = 0.85 ** np.arange(n)
    sparsities = [0.0, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.997]
    seeds = [0] if quick else [0, 1, 2]
    steps = 1500 if quick else 8000
    out = {"n": n, "m": m, "sparsities": sparsities, "importance": importance, "relu": {}, "linear": {}}
    for kind, relu in [("relu", True), ("linear", False)]:
        for S in sparsities:
            runs = []
            for sd in seeds:
                W, b, l = train_toy(n, m, S, importance, sd, relu=relu, steps=steps)
                mt = metrics(W)
                mt.update(loss=l, W=W, b=b)
                runs.append(mt)
            out[kind][S] = dict(
                represented=[r["represented"] for r in runs],
                dims_per_feature=[r["dims_per_feature"] for r in runs],
                interference=[r["mean_interference_represented"] for r in runs],
                loss=[r["loss"] for r in runs],
                norms=runs[0]["norms"], gram=runs[0]["gram"], W=runs[0]["W"])
            print(kind, S, np.mean([r["represented"] for r in runs]), flush=True)
    # capacity: larger model, how many features per dimension at high sparsity
    cap = {}
    n2, m2 = 100, 10
    imp2 = np.ones(n2)
    for S in ([0.9, 0.99] if quick else [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]):
        W, b, l = train_toy(n2, m2, S, imp2, 0, steps=steps)
        mt = metrics(W)
        cap[S] = dict(represented=mt["represented"], per_dim=mt["represented"] / m2,
                      interference=mt["mean_interference_represented"])
        print("capacity", S, mt["represented"], flush=True)
    out["capacity"] = dict(n=n2, m=m2, results=cap)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    with Timer() as t:
        res = run(quick)
    res["seconds"] = t.elapsed
    save_json("e4_superposition" + ("_quick" if quick else ""), res)
