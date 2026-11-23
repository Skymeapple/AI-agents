"""Toy model for Proposition 3: coordination gap vs number of near-frontier developers.

k coalition developers have lags d_j ~ U(0, D) behind a linearly growing closed frontier;
one outside developer (the fallback) has lag D. Each developer releases at Poisson times
(rate r per month) a model equal to the closed frontier at (t - d_j). We compute the frontier
setback metric (same definition as analysis.py) for removing each coalition developer alone
and for removing the whole coalition, and report E[sum unilateral]/E[joint].
Prediction (continuous-release limit, Prop. 3): ratio = 1/k.
"""
import json
import numpy as np
from common import RES, SEED

rng = np.random.default_rng(SEED)
D, T, dt = 12.0, 120.0, 0.05          # months
tgrid = np.arange(0, T, dt)


def front(release_times, lag):
    """Open frontier (capability units = closed-frontier months) from a set of releases."""
    f = np.full(len(tgrid), -np.inf)
    idx = np.searchsorted(tgrid, release_times)
    idx = idx[idx < len(tgrid)]
    np.maximum.at(f, idx, tgrid[idx] - lag)
    return np.maximum.accumulate(f)


def setback(Ocf, O):
    a = np.searchsorted(O, O, side="left")
    b = np.searchsorted(O, np.where(np.isfinite(Ocf), Ocf, -1e9), side="left")
    return (a - b) * dt


def one(k, r):
    lags = rng.uniform(0, D, k)
    rel = [np.sort(rng.uniform(0, T, rng.poisson(r * T))) for _ in range(k + 1)]
    lag_all = list(lags) + [D]
    fr = [front(rel[j], lag_all[j]) for j in range(k + 1)]
    O = np.maximum.reduce(fr)
    w = tgrid > 2 * D
    joint = setback(fr[k], O)[w].mean()
    uni = 0.0
    for j in range(k):
        Om = np.maximum.reduce([fr[i] for i in range(k + 1) if i != j])
        uni += setback(Om, O)[w].mean()
    return uni, joint


out = {}
for r in [0.25, 0.5, 1.0, 4.0]:
    for k in [1, 2, 3, 4, 6, 8]:
        sims = np.array([one(k, r) for _ in range(150)])
        out[f"r={r}|k={k}"] = {"ratio_of_means": float(sims[:, 0].mean() / sims[:, 1].mean()),
                               "pred_1_over_k": 1.0 / k,
                               "mean_joint_mo": float(sims[:, 1].mean()),
                               "mean_sum_unilateral_mo": float(sims[:, 0].mean())}
        print(r, k, round(out[f"r={r}|k={k}"]["ratio_of_means"], 3), round(1 / k, 3))
json.dump(out, open(f"{RES}/toy_model.json", "w"), indent=1)
