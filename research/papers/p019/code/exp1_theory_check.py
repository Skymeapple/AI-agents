"""E1: analytic FDR vs sample size for four verification pipelines + Monte Carlo check.
E1b: K* (number of independent-bias modalities needed) vs prior odds."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory import fdr, fdr_limit, simulate, k_star
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
PI1, MU, SB, ALPHA = 0.2, 0.2, 0.1, 0.05
ns = np.unique(np.logspace(1, 6, 26).astype(int))
pipes = [("self", 0, "self-verification (same data)"),
         ("replicate", 2, "2 fresh replications, same modality"),
         ("orthogonal", 2, "2 modalities, independent bias"),
         ("intervene", 1, "1 randomized intervention")]
res = {"params": dict(pi1=PI1, mu=MU, sb=SB, alpha=ALPHA), "curves": {}, "mc": {}, "limits": {}}
rng = np.random.default_rng(1)
for p, K, lab in pipes:
    res["curves"][p] = [fdr(PI1, MU, p, K, n, SB, alpha=ALPHA)[0] for n in ns]
    res["limits"][p] = fdr_limit(PI1, MU, p, K, SB, ALPHA)
    mc_n = [30, 300, 3000, 30000, 300000]
    res["mc"][p] = {int(n): simulate(PI1, MU, p, K, n, SB, rng=rng) for n in mc_n}
res["ns"] = ns.tolist()

fig, ax = plt.subplots(figsize=(6.4, 4.2))
cols = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
for (p, K, lab), c in zip(pipes, cols):
    ax.plot(ns, res["curves"][p], color=c, label=lab)
    mcx = list(res["mc"][p].keys()); mcy = [v[0] for v in res["mc"][p].values()]
    ax.scatter(mcx, mcy, color=c, s=18, zorder=3)
    ax.axhline(res["limits"][p], color=c, ls=":", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("units per dataset n"); ax.set_ylabel("false discovery rate")
ax.set_title(f"pi1={PI1}, effect mu={MU}, bias sd={SB}, alpha={ALPHA}\nlines: analytic; dots: Monte Carlo; dotted: n->inf limit (Thm 1)", fontsize=8)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_fdr_vs_n.png"), dpi=150)

# E1b: K* vs prior odds for several signal-to-bias ratios (large-n regime: f=1/2, p=Phi(mu/sb))
odds = np.logspace(-3, 0, 61)
res["kstar"] = {}
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for r in [0.5, 1.0, 2.0, 3.0]:
    lam = norm.cdf(r) / 0.5
    ks = k_star(odds, 0.1, lam)
    res["kstar"][str(r)] = ks.tolist()
    ax.step(odds, ks, where="mid", label=f"mu/sigma_b={r} (Lambda={lam:.2f})")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("prior odds pi1/pi0 among candidates (lower = more original)")
ax.set_ylabel("K*: independent-bias modalities for FDR<=0.1")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_kstar.png"), dpi=150)
res["kstar_odds"] = odds.tolist()
json.dump(res, open(os.path.join(OUT, "exp1_theory.json"), "w"), indent=1)
print("limits", res["limits"])
for p in res["mc"]:
    for n, (fd, pw) in res["mc"][p].items():
        print(p, n, "MC FDR %.3f power %.3f | analytic FDR %.3f" % (fd, pw, fdr(PI1, MU, p, dict((a, b) for a, b, _ in pipes)[p], n, SB)[0]))
