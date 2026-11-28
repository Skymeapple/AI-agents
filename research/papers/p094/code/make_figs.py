"""Figures for Exp 3 (simulation coverage) and Exp 4 (METR backtest)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M = ["naive", "powerlaw", "betabin", "lp0"]
LAB = {"naive": "naive pass@n", "powerlaw": "power-law extrap.", "betabin": "beta-binomial extrap.", "lp0": "model-free LP"}
COL = {"naive": "#888888", "powerlaw": "#d95f02", "betabin": "#7570b3", "lp0": "#1b9e77"}

s = json.load(open(os.path.join(OUT, "exp3_sim_summary.json")))["summary"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cells = [r for r in s]
x = np.arange(len(cells))
for j, m in enumerate(M):
    ax[0].scatter(x + (j - 1.5) * 0.18, [r[m + "_cover"] for r in cells], s=14, color=COL[m], label=LAB[m])
    ax[1].scatter(x + (j - 1.5) * 0.18, [r[m + "_mean"] - r["C_true"] for r in cells], s=14, color=COL[m])
ax[0].axhline(0.95, color="k", ls="--", lw=0.8)
ax[0].set_ylabel("coverage of 95% UCB"); ax[1].set_ylabel("mean UCB minus true C(B)")
ax[1].axhline(0, color="k", lw=0.8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels([f"{r['pop'][:6]} n{r['n']} T{r['T']} B{r['B']}" for r in cells], rotation=90, fontsize=6)
ax[0].legend(fontsize=7, loc="lower left")
ax[0].set_title("Simulation: false-safe protection"); ax[1].set_title("Simulation: conservativeness")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_sim_coverage.png"), dpi=150)

p = os.path.join(OUT, "exp4_metr.json")
if os.path.exists(p):
    d = json.load(open(p))
    bt = d["backtest"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(bt))
    for j, m in enumerate(M):
        ax[0].scatter(x + (j - 1.5) * 0.18, [r[m + "_cover"] for r in bt], s=14, color=COL[m], label=LAB[m])
        ax[1].scatter(x + (j - 1.5) * 0.18, [r[m + "_mean"] - r["target"] for r in bt], s=14, color=COL[m])
    ax[0].axhline(0.95, color="k", ls="--", lw=0.8); ax[1].axhline(0, color="k", lw=0.8)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([f"{r['model'][:18]} k{r['k']}->B{r['n_full']}" for r in bt], rotation=90, fontsize=6)
    ax[0].set_ylabel("share of splits with UCB >= realised coverage")
    ax[1].set_ylabel("mean UCB minus realised coverage")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title("METR TH1.1 backtest"); ax[1].set_title("METR TH1.1 backtest: conservativeness")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_metr_backtest.png"), dpi=150)
    de = d["descriptive"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    de = sorted(de, key=lambda r: r["pass4"])
    y = np.arange(len(de))
    ax.hlines(y, [r["lp_lo64"] for r in de], [r["lp_hi64"] for r in de], color=COL["lp0"], lw=3, label="model-free 95% set, C(64)")
    ax.scatter([r["pass1"] for r in de], y, color="k", s=12, label="pass@1")
    ax.scatter([r["pass4"] for r in de], y, color="#888888", s=12, label="pass@4 (observed)")
    ax.scatter([r["bb64"] for r in de], y, color=COL["betabin"], marker="x", s=20, label="beta-binomial C(64)")
    ax.scatter([r["pl64"] for r in de], y, color=COL["powerlaw"], marker="+", s=30, label="power-law C(64)")
    ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in de], fontsize=6)
    ax.set_xlabel("fraction of METR TH1.1 tasks"); ax.legend(fontsize=6, loc="lower right")
    ax.set_title("What 4 runs/task say about a 64-attempt adversary")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_metr_identified.png"), dpi=150)
