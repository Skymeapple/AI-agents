"""Figures for the paper (PNG, results/fig*.png)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import R, eps_star

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
FAM = ["linear", "threshold", "oring", "ces_automate", "ces_labelled", "ces_augment"]
FCOL = dict(zip(FAM, PAL))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

H = pd.read_csv(os.path.join(R, "multiverse_headline.csv"))
# Fig 1: specification distribution by rule family (strip plot)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, fam in enumerate(FAM):
    x = H[H.family == fam]
    ax.scatter(i + rng.uniform(-0.3, 0.3, len(x)), 100 * x.H_net, s=6, color=FCOL[fam], linewidths=0, alpha=0.7)
    ax.plot([i - 0.35, i + 0.35], [100 * x.H_net.median()] * 2, color="#0b0b0b", lw=1.5)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xticks(range(len(FAM))); ax.set_xticklabels([f"{f}\n(n={int((H.family==f).sum())})" for f in FAM], fontsize=8)
ax.set_ylabel("net employment displaced\n(% of 2024 jobs)")
ax.set_title("Fig. 1  Headline 'share of jobs displaced' across 1,680 specifications (bar = median)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_spec_curve.png"), dpi=200); plt.close(fig)

# Fig 2: variance decomposition
A = json.load(open(os.path.join(R, "analysis_summary.json")))["anova"]
fig, ax = plt.subplots(figsize=(6, 2.6))
labs = ["H_net", "H_gross", "H_half"]
comp = ["measure", "weight", "rule", "interactions+residual"]
cc = [PAL[0], PAL[3], PAL[1], "#bdbcb6"]
left = np.zeros(3)
for c, col in zip(comp, cc):
    v = np.array([A[l][c] for l in labs])
    ax.barh(labs, v, left=left, color=col, label=c, edgecolor="#fcfcfb", linewidth=2)
    left += v
ax.set_xlim(0, 1); ax.set_xlabel("share of variance across 1,680 specifications (main-effect $\\eta^2$)")
ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_variance_decomposition.png"), dpi=200); plt.close(fig)

# Fig 3: sector rank ranges
T = pd.read_csv(os.path.join(R, "sector_tau_vs_linear.csv"))
SC = pd.read_csv(os.path.join(R, "multiverse_sector.csv"), dtype={"sector": str})
SC["key"] = SC.measure + "|" + SC.weight + "|" + SC.rule
piv = SC.pivot_table(index="key", columns="sector", values="D")
piv = piv[piv.std(axis=1) > 1e-9]
fam = SC.drop_duplicates("key").set_index("key").loc[piv.index, "family"]
ranks = piv.rank(axis=1, ascending=False)
titles = pd.read_csv(os.path.join(R, "emp_sector.csv"), dtype={"NAICS": str}).drop_duplicates("NAICS").set_index("NAICS")["NAICS_TITLE"]
short = {k: (v if len(v) < 34 else v[:32] + "…") for k, v in titles.items()}
short["99"] = "Government (OEWS designation)"
order = ranks.median().sort_values().index
inc = fam.isin(["linear", "threshold", "oring", "ces_labelled"]) | (fam == "ces_automate")
fig, ax = plt.subplots(figsize=(7, 5))
for i, k in enumerate(order):
    allr = ranks[k]
    ax.plot([allr.quantile(.05), allr.quantile(.95)], [i, i], color="#bdbcb6", lw=4, solid_capstyle="round")
    for j, f in enumerate(["linear", "threshold", "oring", "ces_labelled"]):
        rr = ranks.loc[fam == f, k]
        ax.scatter(rr.median(), i + (j - 1.5) * 0.15, s=14, color=FCOL[f], zorder=3,
                   label=f if i == 0 else None)
ax.set_yticks(range(len(order))); ax.set_yticklabels([short[k] for k in order], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("rank of sector by displacement (1 = largest job loss)")
ax.set_title("Fig. 3  Sector ranks: 5–95% range over all non-degenerate specs (grey);\nmedian rank within rule family (dots)", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_sector_ranks.png"), dpi=200); plt.close(fig)

# Fig 4: employment-neutral demand elasticity
s = np.linspace(0.01, 0.8, 200)
fig, ax = plt.subplots(figsize=(6, 3.2))
i = 0
for sig in [0.5, 2.0]:
    for th in [0.5, 0.9]:
        ax.plot(s, eps_star(s, sig, th), lw=2, color=PAL[i], label=f"σ={sig}, θ={th}"); i += 1
ax.axhline(1, color="#52514e", lw=0.8, ls="--")
ax.set_xlabel("exposed cost share s"); ax.set_ylabel("ε* (employment-neutral demand elasticity)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Fig. 4  Pure automation: employment rises iff ε > ε*(s, σ, θ)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_eps_star.png"), dpi=200); plt.close(fig)

# Fig 5: identification power
P = pd.read_csv(os.path.join(R, "power_sim.csv"))
P = P[P.phi > 0]
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0), sharex=True)
for i, (r, x) in enumerate(P.groupby("true_rule")):
    axs[0].plot(x.phi, x.p_correct, marker="o", ms=4, lw=2, color=PAL[i], label=r)
    axs[1].plot(x.phi, x.power_detect_size_adjusted, marker="o", ms=4, lw=2, color=PAL[i])
axs[0].axhline(1 / 6, color="#52514e", ls="--", lw=0.8)
axs[0].set_ylabel("P(true rule fits best)"); axs[1].set_ylabel("size-adjusted power")
for a in axs:
    a.set_xscale("log"); a.set_xlabel("share φ of long-run displacement realised")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.suptitle("Fig. 5  Can 2021→2024 occupational employment changes discriminate aggregation rules?", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig5_power.png"), dpi=200); plt.close(fig)
print("figures written")
