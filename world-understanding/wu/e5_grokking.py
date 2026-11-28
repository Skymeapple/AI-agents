"""E5 -- From memorisation to mechanism: grokking modular arithmetic.

A network trained on a fraction of the table a + b (mod p) first memorises the
training pairs, then -- long after training accuracy saturates, and only with
weight decay -- abruptly generalises to the unseen pairs. Mechanistic work
(Nanda et al., 2023) shows the generalising solution represents numbers as
points on circles (a sparse set of Fourier frequencies) and implements addition
as rotation. We test:
  (a) the delayed-generalisation phenomenon and its dependence on weight decay
      and training fraction;
  (b) whether generalisation coincides with the emergence of a sparse Fourier
      (i.e. structured, compressed) representation;
  (c) the parameter-norm signature: generalising solutions are lower-norm
      (simpler) than memorising ones.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import Timer, save_json

P = 97


def fourier_concentration(E, top=6):
    """Fraction of embedding power (excluding the DC term) in the top-`top` frequencies."""
    F = np.fft.rfft(E - E.mean(0), axis=0)  # [P//2+1, d]
    power = (np.abs(F[1:]) ** 2).sum(1)
    power = power / power.sum()
    return float(np.sort(power)[::-1][:top].sum()), power


def run_one(frac, wd, seed, steps=20000, d=128, hidden=256, lr=1e-3, log_every=100):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    labels = (pairs[:, 0] + pairs[:, 1]) % P
    perm = rng.permutation(len(pairs))
    ntr = int(frac * len(pairs))
    tr, te = perm[:ntr], perm[ntr:]
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "E": jax.random.normal(k1, (P, d)) / np.sqrt(d),
        "W1": jax.random.normal(k2, (2 * d, hidden)) / np.sqrt(2 * d),
        "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k3, (hidden, P)) / np.sqrt(hidden),
    }

    def logits(p, ab):
        x = jnp.concatenate([p["E"][ab[:, 0]], p["E"][ab[:, 1]]], -1)
        return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"]

    Xtr, Ytr = jnp.asarray(pairs[tr]), jnp.asarray(labels[tr])
    Xte, Yte = jnp.asarray(pairs[te]), jnp.asarray(labels[te])

    def loss(p):
        lp = jax.nn.log_softmax(logits(p, Xtr), -1)
        return -jnp.take_along_axis(lp, Ytr[:, None], -1).mean()

    opt = optax.adamw(lr, b1=0.9, b2=0.98, weight_decay=wd)
    st = opt.init(params)

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    @jax.jit
    def evaluate(p):
        tra = (logits(p, Xtr).argmax(-1) == Ytr).mean()
        tea = (logits(p, Xte).argmax(-1) == Yte).mean()
        lte = -jnp.take_along_axis(jax.nn.log_softmax(logits(p, Xte), -1), Yte[:, None], -1).mean()
        norm = jnp.sqrt(sum((v ** 2).sum() for v in jax.tree_util.tree_leaves(p)))
        return tra, tea, lte, norm

    hist = []
    for s in range(1, steps + 1):
        params, st, l = step(params, st)
        if s % log_every == 0 or s == 1:
            tra, tea, lte, norm = evaluate(params)
            fc, _ = fourier_concentration(np.asarray(params["E"]))
            hist.append((s, float(l), float(tra), float(tea), float(lte), float(norm), fc))
    h = np.array(hist)
    t_train = next((int(r[0]) for r in hist if r[2] >= 0.99), None)
    t_test = next((int(r[0]) for r in hist if r[3] >= 0.99), None)
    fc, power = fourier_concentration(np.asarray(params["E"]))
    return dict(frac=frac, wd=wd, seed=seed, hist=h, t_train99=t_train, t_test99=t_test,
                final_test=float(h[-1, 3]), final_fourier=fc, power=power)


def run(quick=False):
    steps = 3000 if quick else 20000
    out = {"P": P, "main": [], "wd_sweep": [], "frac_sweep": [], "steps": steps}
    for sd in ([0] if quick else [0, 1, 2]):
        r = run_one(0.3, 1.0, sd, steps)
        out["main"].append(r)
        print("main", sd, r["t_train99"], r["t_test99"], r["final_test"], round(r["final_fourier"], 3), flush=True)
    for wd in ([0.0] if quick else [0.0, 0.1, 0.3, 3.0]):
        for sd in ([0] if quick else [0, 1]):
            r = run_one(0.3, wd, sd, steps)
            r.pop("power")
            out["wd_sweep"].append(r)
            print("wd", wd, sd, r["t_train99"], r["t_test99"], r["final_test"], flush=True)
    for frac in ([0.5] if quick else [0.2, 0.25, 0.4, 0.5, 0.7]):
        for sd in ([0] if quick else [0, 1]):
            r = run_one(frac, 1.0, sd, steps)
            r.pop("power")
            out["frac_sweep"].append(r)
            print("frac", frac, sd, r["t_train99"], r["t_test99"], r["final_test"], flush=True)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    with Timer() as t:
        res = run(quick)
    res["seconds"] = t.elapsed
    save_json("e5_grokking" + ("_quick" if quick else ""), res)
