"""E3 -- Scaling laws as a fingerprint of a heavy-tailed world.

The world contains many independent "facts" (key -> value) whose frequency of
occurrence follows a Zipf law p_k ~ k^-alpha. A capacity-limited learner that
stores the most frequent facts first should show power-law scaling whose
exponent is set by the *world* (alpha), not by the architecture:

  capacity:  error(N) ~ N^-(alpha-1)          (N = parameters; quantization model,
                                               Michaud et al. 2023)
  data:      error(D) ~ D^-(alpha-1)/alpha     (D = samples; Hutter 2021)

We test both predictions with MLPs across three values of alpha.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import Timer, powerlaw_fit, save_json
from .mlp import init_mlp, mlp_apply

K, DX, V = 2048, 32, 32


def make_world(alpha, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(K, DX)) / np.sqrt(DX)).astype(np.float32)
    Y = rng.integers(0, V, K)
    p = np.arange(1, K + 1, dtype=float) ** -alpha
    return X, Y, p / p.sum()


def n_params(w):
    return DX * w + w + w * V + V


def train_mlp(X, Y, weights, width, steps, seed, lr=3e-3, batch=None):
    """Minimise weighted cross-entropy over keys (weights = sampling distribution)."""
    params = init_mlp(jax.random.PRNGKey(seed), [DX, width, V])
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, 0.05))
    st = opt.init(params)
    Xj, Yj, Wj = jnp.asarray(X), jnp.asarray(Y), jnp.asarray(weights, dtype=jnp.float32)

    def loss(p):
        lp = jax.nn.log_softmax(mlp_apply(p, Xj), -1)
        return -(Wj * jnp.take_along_axis(lp, Yj[:, None], -1)[:, 0]).sum()

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    for _ in range(steps):
        params, st, l = step(params, st)
    correct = np.asarray(mlp_apply(params, Xj).argmax(-1) == Yj)
    return correct


def run(quick=False):
    alphas = [1.5, 2.0, 2.5]
    widths = [4, 8, 16] if quick else [4, 8, 16, 32, 64, 128, 256]
    Ds = [256, 1024, 4096] if quick else [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    seeds = [0] if quick else [0, 1, 2]
    steps = 1500 if quick else 6000
    out = {"alphas": alphas, "widths": widths, "params": [n_params(w) for w in widths], "Ds": Ds,
           "K": K, "V": V, "capacity": {}, "data": {}}
    for a in alphas:
        err = np.zeros((len(widths), len(seeds)))
        nlearn = np.zeros_like(err)
        rank_curves = {}
        for j, sd in enumerate(seeds):
            X, Y, p = make_world(a, sd)
            # Expected-loss training (the infinite-data limit): weight each key by p_k.
            for i, w in enumerate(widths):
                c = train_mlp(X, Y, p, w, steps, sd)
                err[i, j] = float((p * (1 - c)).sum())
                nlearn[i, j] = c.sum()
                if sd == 0:
                    rank_curves[w] = np.convolve(c.astype(float), np.ones(32) / 32, "valid")[::16]
            print(f"alpha={a} seed={sd} capacity err {np.round(err[:, j], 4)}", flush=True)
        fit_range = slice(0, len(widths) - 1) if not quick else slice(None)
        P = np.array(out["params"])
        slope, _, r2 = powerlaw_fit(P[fit_range], err.mean(1)[fit_range])
        sl_n, _, r2n = powerlaw_fit(P[fit_range], nlearn.mean(1)[fit_range])
        out["capacity"][a] = dict(err=err, n_learned=nlearn, slope=slope, r2=r2,
                                  predicted=-(a - 1), learned_vs_params_slope=sl_n,
                                  learned_vs_params_r2=r2n, rank_curves=rank_curves)

        # data scaling: big model, fixed dataset of D draws; train on empirical distribution
        errD = np.zeros((len(Ds), len(seeds)))
        for j, sd in enumerate(seeds):
            X, Y, p = make_world(a, sd)
            r = np.random.default_rng(10 + sd)
            for i, Dn in enumerate(Ds):
                counts = np.bincount(r.choice(K, Dn, p=p), minlength=K)
                c = train_mlp(X, Y, counts / counts.sum(), 256, steps // 2, sd)
                errD[i, j] = float((p * (1 - c)).sum())
            print(f"alpha={a} seed={sd} data err {np.round(errD[:, j], 4)}", flush=True)
        # theory curve: expected unseen mass sum_k p_k (1-p_k)^D  (x (1 - 1/V) chance)
        theory = [float((p * (1 - p) ** Dn).sum() * (1 - 1 / V)) for Dn in Ds]
        slope, _, r2 = powerlaw_fit(Ds, errD.mean(1))
        out["data"][a] = dict(err=errD, slope=slope, r2=r2, predicted=-(a - 1) / a, theory=theory,
                              theory_slope=powerlaw_fit(Ds, theory)[0])
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    with Timer() as t:
        res = run(quick)
    res["seconds"] = t.elapsed
    save_json("e3_zipf_scaling" + ("_quick" if quick else ""), res)
    for a in res["alphas"]:
        c, d = res["capacity"][a], res["data"][a]
        print(f"alpha {a}: capacity slope {c['slope']:.3f} (pred {c['predicted']:.3f}); "
              f"data slope {d['slope']:.3f} (pred {d['predicted']:.3f}, finite-K theory {d['theory_slope']:.3f})")
