"""Conditional analysis: how slow must AI revenue growth be before it, rather than the share caps,
determines when frontier compute growth falls below 2x/yr?  Grid over initial revenue growth gR0
and decay time tau; all other parameters drawn from priors; lam_max fixed at 1.5.
Outputs results/revenue_stress.json and fig6_revenue_stress.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import sample
from model import simulate, summarize

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N = 4000
gR0s = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
taus = [1.0, 2.0, 4.0, 8.0]
base_rng = np.random.default_rng(21)
p = sample(N, base_rng)
s_exo = summarize(simulate(p))
base = float(np.median(np.minimum(s_exo["slow2"], 2046)))
grid, gridC = np.zeros((len(taus), len(gR0s))), np.zeros((len(taus), len(gR0s)))
for i, tau in enumerate(taus):
    for j, g in enumerate(gR0s):
        rev = dict(gR0=np.full(N, np.log(g)), gRinf=np.full(N, 0.10), tau=np.full(N, tau), lam_max=np.full(N, 1.5))
        s = summarize(simulate(p, variant="revenue", rev=rev))
        grid[i, j] = np.median(np.minimum(s["slow2"], 2046))
        gridC[i, j] = np.median(s["log10C_2030"])
res = {"baseline_exogenous_median_slow2": base, "baseline_exogenous_median_log10C_2030": float(np.median(s_exo["log10C_2030"])),
       "gR0_factors": gR0s, "taus": taus, "median_slow2": grid.tolist(), "median_log10C_2030": gridC.tolist()}
json.dump(res, open(os.path.join(OUT, "revenue_stress.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
fig, ax = plt.subplots(figsize=(6, 3.6))
im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(gR0s))); ax.set_xticklabels([f"{g}x" for g in gR0s])
ax.set_yticks(range(len(taus))); ax.set_yticklabels([str(t) for t in taus])
for i in range(len(taus)):
    for j in range(len(gR0s)):
        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
ax.set_xlabel("AI revenue growth in 2025 (factor/yr)"); ax.set_ylabel("decay time of revenue growth (yr)")
ax.set_title(f"median year growth < 2x/yr (exogenous baseline {base:.1f})", fontsize=9)
plt.colorbar(im); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_revenue_stress.png"), dpi=150)
