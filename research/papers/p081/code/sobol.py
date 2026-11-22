"""Variance-based global sensitivity (Saltelli sampling, Jansen estimators) for the exogenous variant.
Outputs results/sobol.json and results/fig5_sobol.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import NAMES, from_unit, PRIORS
from model import simulate, summarize

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(7)
N, k = 8192, len(NAMES)
A, B = rng.random((N, k)), rng.random((N, k))


def evaluate(U):
    ys = {"log10C_2030": [], "log10C_2035": [], "slow2": []}
    for c in range(0, len(U), 4096):
        u = U[c:c + 4096]
        p = {nm: from_unit(nm, u[:, i]) for i, nm in enumerate(NAMES)}
        s = summarize(simulate(p))
        ys["log10C_2030"].append(s["log10C_2030"]); ys["log10C_2035"].append(s["log10C_2035"])
        ys["slow2"].append(np.minimum(s["slow2"], 2046.0))
    return {kk: np.concatenate(v) for kk, v in ys.items()}


fA, fB = evaluate(A), evaluate(B)
res = {}
for out in fA:
    mu = np.mean(np.concatenate([fA[out], fB[out]]))
    a, b = fA[out] - mu, fB[out] - mu                     # centre outputs (reduces estimator variance)
    var = np.var(np.concatenate([a, b]))
    S1, ST, S1ci, STci = {}, {}, {}, {}
    brng = np.random.default_rng(99)
    boots = [brng.integers(0, N, N) for _ in range(200)]
    for i, nm in enumerate(NAMES):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        c = evaluate(ABi)[out] - mu
        s1 = lambda ix: np.mean(b[ix] * (c[ix] - a[ix])) / var          # Saltelli et al. 2010
        st = lambda ix: 0.5 * np.mean((a[ix] - c[ix]) ** 2) / var        # Jansen 1999
        allix = np.arange(N)
        S1[nm], ST[nm] = float(s1(allix)), float(st(allix))
        S1ci[nm] = [float(x) for x in np.percentile([s1(ix) for ix in boots], [2.5, 97.5])]
        STci[nm] = [float(x) for x in np.percentile([st(ix) for ix in boots], [2.5, 97.5])]
    res[out] = {"S1": S1, "ST": ST, "S1_ci95": S1ci, "ST_ci95": STci, "var": float(var)}
    print(out, sorted(ST.items(), key=lambda x: -x[1])[:6])
json.dump(res, open(os.path.join(OUT, "sobol.json"), "w"), indent=2)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
order = sorted(NAMES, key=lambda nm: -res["slow2"]["ST"][nm])
y = np.arange(k)
for ax, out in zip(axes, ("log10C_2030", "log10C_2035", "slow2")):
    ax.barh(y - .2, [res[out]["ST"][nm] for nm in order], .4, label="total ST", color="#4C72B0")
    ax.barh(y + .2, [max(res[out]["S1"][nm], 0) for nm in order], .4, label="first-order S1", color="#C44E52")
    ax.set_title(out, fontsize=10); ax.grid(alpha=.3, axis="x")
axes[0].set_yticks(y); axes[0].set_yticklabels([f"{nm} [{PRIORS[nm][3]}]" for nm in order], fontsize=8)
axes[0].invert_yaxis(); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_sobol.png"), dpi=150)
