"""(i) Numerical checks of Propositions 2 and 4; (ii) re-analysis of the published CHC domain scores.

CHC scores: Hendrycks et al. (2025), arXiv:2510.18212, table of ten domains (each out of 10 points).
Order: K, RW, M, R, WM, MS, MR, V, A, S.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import power_mean_log2

OUT = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results")
rng = np.random.default_rng(1)
res = {}

# ---------- Proposition 2: asymptotic growth rate of power means of exponentially growing horizons
n = 10
a = rng.normal(0, 2, n)          # log2 horizon offsets
s = rng.uniform(0.8, 2.4, n)     # log2 units per year (doubling times 5-15 months)
ts = np.linspace(0, 30, 3001)
ps = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
rates = {}
for p in ps:
    L = np.array([power_mean_log2(a + s * t, p) for t in ts])
    r = np.gradient(L, ts)
    rates[p] = r
    ax[0].plot(ts, 12 / r, label=f"p={p:g}")
for v, lab in [(s.max(), "max s_i"), (s.mean(), "mean s_i"), (s.min(), "min s_i")]:
    ax[0].axhline(12 / v, ls=":", c="k", lw=0.8)
    ax[0].text(30.3, 12 / v, lab, fontsize=7, va="center")
ax[0].set_xlabel("years"); ax[0].set_ylabel("doubling time of aggregate (months)")
ax[0].set_title("Prop. 2: aggregate doubling time converges\nto fastest / mean / slowest domain")
ax[0].legend(fontsize=7, ncol=2)
res["prop2"] = dict(s_max=s.max(), s_mean=s.mean(), s_min=s.min(),
                    rate_at_T30={str(p): rates[p][-1] for p in ps})

# ---------- Proposition 4: downward bias of min under noise; compare with sigma*sqrt(2 ln n)
sig = 1.0
ns = [2, 5, 10, 20, 50, 80, 200]
bias, bound = [], []
for k in ns:
    true = np.zeros(k)  # worst case: all domains tied at the minimum
    est = (true + rng.normal(0, sig, (20000, k))).min(1)
    bias.append(est.mean()); bound.append(-sig * np.sqrt(2 * np.log(k)))
# spread-out true values reduce the bias
true_spread = np.linspace(0, 3, 20)
b_spread = (true_spread + rng.normal(0, sig, (20000, 20))).min(1).mean()
ax[1].plot(ns, bias, "o-", label="E[min of noisy] - true min (ties)")
ax[1].plot(ns, bound, "k--", label=r"$-\sigma\sqrt{2\ln n}$ bound")
ax[1].set_xscale("log"); ax[1].set_xlabel("number of domains n"); ax[1].set_ylabel("bias (units of sigma)")
ax[1].set_title("Prop. 4: bottleneck estimates are biased low")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_theory.png", dpi=160); plt.close()
res["prop4"] = dict(n=ns, bias=bias, bound=bound, bias_spread_n20_range3sigma=b_spread)

# ---------- CHC re-analysis
dom = ["K", "RW", "M", "R", "WM", "MS", "MR", "V", "A", "S"]
g4 = np.array([8, 6, 4, 0, 2, 0, 4, 0, 0, 3], float) / 10
g5 = np.array([9, 10, 10, 7, 4, 0, 4, 4, 6, 3], float) / 10
t4, t5 = 2023 + (31 + 28 + 14) / 365.25, 2025 + (31 + 28 + 31 + 30 + 31 + 30 + 31 + 7) / 365.25


def pmean(x, p):
    x = np.asarray(x, float)
    if np.isinf(p):
        return x.max() if p > 0 else x.min()
    if p == 0:
        return 0.0 if np.any(x == 0) else float(np.exp(np.mean(np.log(x))))
    if p < 0 and np.any(x == 0):
        return 0.0
    return float(np.mean(x ** p) ** (1 / p))


pgrid = [1, 0.5, 0, -1, -np.inf]
chc = {"scores": {str(p): (pmean(g4, p), pmean(g5, p)) for p in pgrid}}
# eps-floor sensitivity for p<=0 (zeros replaced by eps)
chc["eps_floor_geo"] = {str(e): (pmean(np.maximum(g4, e), 0), pmean(np.maximum(g5, e), 0))
                        for e in [0.005, 0.01, 0.05, 0.1]}
# linear per-domain extrapolation, capped at 1; find crossing year of aggregate thresholds
rate = (g5 - g4) / (t5 - t4)
years = np.linspace(t5, 2100, 200001)
cross = {}
for tau in [0.7, 0.9, 1.0]:
    for p in pgrid:
        hit = None
        for y in years[::50]:
            x = np.clip(g5 + rate * (y - t5), 0, 1)
            if pmean(x, p) >= tau - 1e-9:
                hit = y; break
        cross[f"tau={tau},p={p}"] = hit
chc["linear_crossing_year"] = cross
chc["stagnant_domains"] = [d for d, r in zip(dom, rate) if r <= 0]
res["chc"] = chc

fig, ax = plt.subplots(figsize=(5, 3.4))
pp = np.linspace(-3, 2, 101)
for x, lab in [(g4, "GPT-4"), (g5, "GPT-5")]:
    for e, ls in [(0.01, "-"), (0.05, "--")]:
        ax.plot(pp, [pmean(np.maximum(x, e), p) for p in pp], ls, label=f"{lab}, zero-floor={e}")
ax.set_xlabel("aggregation exponent p"); ax.set_ylabel("AGI score")
ax.set_title("CHC scores (Hendrycks et al. 2025)\nunder power-mean aggregation")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/fig_chc.png", dpi=160); plt.close()

json.dump(res, open(f"{OUT}/theory_chc.json", "w"), indent=2, default=lambda o: None if o is None else float(o))
print(json.dumps(res, indent=1, default=lambda o: None if o is None else float(o)))
