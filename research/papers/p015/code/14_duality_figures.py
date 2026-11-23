"""Figures for the replicate-length duality analysis (reads outputs of 12_duality.py)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import npmle
import importlib.util

spec = importlib.util.spec_from_file_location("dual", f"{HERE}/12_duality.py")
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)

d = load_runs()
K = pd.read_csv(f"{RES}/duality_npmle_k.csv")
F = pd.read_csv(f"{RES}/duality_forecast.csv")
B = pd.read_csv(f"{RES}/duality_bounds.csv")
rd = release_dates()

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
# (a) duality diagnostic for two agents
for ax, alias in zip([axes[0, 0], axes[0, 1]], ["Claude 4 Opus (Inspect)", "Claude Opus 4.6 (Inspect)"]):
    tt = task_table(d[d.alias == alias])
    n, s, T = tt.n.values.astype(float), tt.s.values.astype(float), tt["T"].values.astype(float)
    x = np.log(T)
    grid = np.linspace(np.log(2), np.log(1200), 80)
    kr = K[(K.alias == alias) & (K.variant == "all tasks")].iloc[0]
    f = npmle.fit(n, s, T)
    cols = {1: "#1f4e79", 2: "#2e86c1", 4: "#85c1e9"}
    for j in (1, 2, 4):
        y = s / n if j == 1 else dual.ustat(s, n, j)
        sm = dual.loclin(x, y, grid, 0.7)
        ax.plot(np.exp(grid), sm, color=cols[j], lw=2, label=f"data: E[p^{j}] (smoothed)")
        eta = f["k"] * np.log(np.exp(grid) / T_REF)[:, None] + npmle.U_GRID[None, :]
        model = np.exp(-j * np.exp(np.clip(eta, -40, 8))) @ f["pi"]
        ax.plot(np.exp(grid), model, color=cols[j], lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("human task length (min)")
    ax.set_ylabel("success moment")
    ax.set_title(f"(a) {alias.replace(' (Inspect)', '')}: pass^j by length\n(dashed: NPMLE frailty, k = {f['k']:.2f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(-0.02, 1.02)

# (b) out-of-window forecast errors
ax = axes[1, 0]
meth = ["log-logistic (METR)", "Weibull", "CM mixture", "Gaussian frailty", "NPMLE frailty", "NPMLE frailty, k=1"]
short = ["logistic", "Weibull", "CM", "Gauss fr.", "NPMLE fr.", "NPMLE k=1"]
g = F[F.band == "[C,4C)"]
pos = 0
xt, xl = [], []
colors = ["#7f8c8d", "#95a5a6", "#bdc3c7", "#e67e22", "#c0392b", "#8e44ad"]
for ci, C in enumerate(sorted(g.cut_min.unique())):
    gc = g[g.cut_min == C]
    for mi, m in enumerate(meth):
        e = gc[m] - gc.obs
        ax.scatter(np.full(len(e), pos) + np.random.default_rng(mi).uniform(-0.15, 0.15, len(e)), e, s=8,
                   color=colors[mi], alpha=0.6)
        ax.plot([pos - 0.3, pos + 0.3], [e.mean()] * 2, color=colors[mi], lw=2.5)
        pos += 1
    xt.append(pos - 3.5)
    xl.append(f"train <{C / 60:.0f} h\npredict [{C / 60:.0f} h, {4 * C / 60:.0f} h)")
    pos += 1
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=8)
ax.set_ylabel("predicted − observed success")
ax.set_title("(b) Short→long forecasts, 20 agents (bars = means)", fontsize=10)
for mi, m in enumerate(short):
    ax.scatter([], [], color=colors[mi], label=m)
ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")

# (c) LR bounds at 8 h / 40 h / 160 h
ax = axes[1, 1]
B["date"] = B.alias.map(rd)
Bs = B.sort_values(["date", "alias"])
order = list(dict.fromkeys(Bs.alias))
for k_, (Th, col) in enumerate([(8, "#1f4e79"), (40, "#c0392b"), (160, "#7f8c8d")]):
    b = Bs[Bs.Tstar_h == Th].set_index("alias").reindex(order)
    yy = np.arange(len(order)) + (k_ - 1) * 0.25
    ax.hlines(yy, b.lo, b.hi, color=col, lw=2.5, label=f"T* = {Th} h")
    ax.plot(b.point, yy, "o", color=col, ms=3)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([a.replace(" (Inspect)", "") for a in order], fontsize=7)
ax.set_xlabel("success at length T* (95% LR bounds)")
ax.set_title("(c) What the TH1.1 replicates allow at day/week/month lengths", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(f"{RES}/fig_duality.png", dpi=150)
print("saved")
