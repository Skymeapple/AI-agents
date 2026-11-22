"""Figures for the threshold-drift paper. Run after analysis.py."""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, load_large_scale, alpha_from_halving

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "legend.frameon": False})

R = json.load(open(os.path.join(RES, "results.json")))
f_coef = np.load(os.path.join(RES, "frontier_coef.npy"))
d = load_large_scale()
A_C = alpha_from_halving(8.0)
g = R["frontier"]["g_oom_per_year"]

# ---------------- Fig 1: data + rule paths ------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
k = d[d["x"].notna() & (d["t"] >= 2020)]
ax.scatter(k.loc[k.open_weights, "t"], k.loc[k.open_weights, "x"], s=14, color=BLUE,
           edgecolor="white", linewidth=0.5, label="open weights (public estimate)", zorder=3)
ax.scatter(k.loc[~k.open_weights, "t"], k.loc[~k.open_weights, "x"], s=14, color=ORANGE,
           edgecolor="white", linewidth=0.5, label="not open weights (public estimate)", zorder=3)
m = d[d["x"].isna() & (d["t"] >= 2020)]
ax.plot(m["t"], np.full(len(m), 22.75), "|", color=INK2, ms=7, alpha=0.6,
        label="no public compute estimate (date only)")
tt = np.linspace(2020, 2030, 200)
ax.plot(tt, f_coef[0] + f_coef[1] * (tt - 2024), color=INK, lw=1.2, ls="--", label=f"frontier trend ({10**g:.1f}x/yr)")
ax.axhline(25, color=VIOLET, lw=1.5, label="static 1e25 (EU AI Act)")
ax.axhline(26, color=MAGENTA, lw=1.5, label="static 1e26 (CA SB 53 / NY RAISE)")
ax.plot(tt, 25 - A_C * (tt - 2024), color=GREEN, lw=1.8, label="capability-indexed from 1e25 @2024 (8-mo halving)")
ax.fill_between(tt, 25 - alpha_from_halving(5) * (tt - 2024), 25 - alpha_from_halving(14) * (tt - 2024),
                color=GREEN, alpha=0.12, lw=0)
ax.plot(tt, f_coef[0] + f_coef[1] * (tt - 2024) - 1, color=YELLOW, lw=1.8, label="frontier-relative (trend - 1 OOM)")
ax.axhspan(21.5, 23.0, color=GRID, alpha=0.5, lw=0)
ax.text(2026.6, 22.15, "below dataset floor (1e23)", color=INK2, fontsize=7.5)
ax.set_xlim(2020, 2030); ax.set_ylim(21.5, 29)
ax.set_xlabel("publication date"); ax.set_ylabel("log10 training compute (FLOP)")
ax.legend(fontsize=7, loc="upper left", ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_threshold_paths.png"), dpi=180); plt.close(fig)

# ---------------- Fig 2: projected annual coverage ---------------------------
p = pd.read_csv(os.path.join(RES, "projected_coverage.csv"))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for rule, col, lab in [("static_1e25", VIOLET, "static 1e25"), ("static_1e26", MAGENTA, "static 1e26"),
                       ("capability_indexed", GREEN, "capability-indexed (8-mo)"),
                       ("frontier_minus_1OOM", YELLOW, "frontier - 1 OOM")]:
    s = p[p.rule == rule].sort_values("year")
    ax.plot(s.year, s.exp_known, color=col, marker="o", ms=4, label=lab)
    ax.fill_between(s.year, s.known_lo, s.known_hi, color=col, alpha=0.15, lw=0)
    fl = s[s.below_data_floor]
    if len(fl):
        ax.plot(fl.year, fl.exp_known, "o", ms=8, mfc="none", mec=col)
ax.set_yscale("log"); ax.set_xlabel("calendar year"); ax.set_ylabel("expected releases covered per year")
ax.set_title("Model M2, known-compute calibration (open circles: threshold below 1e23 => lower bound)",
             fontsize=8, color=INK2)
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_projected_coverage.png"), dpi=180); plt.close(fig)

# ---------------- Fig 3: optimal indexing --------------------------------------
L = pd.read_csv(os.path.join(RES, "loss_ratios.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
rho = np.linspace(0, 1, 101)
for h, col in [(5, AQUA), (8, GREEN), (14, BLUE)]:
    al = alpha_from_halving(h)
    axs[0].plot(rho, -al + rho * (al + g), color=col, label=f"halving {h} mo")
axs[0].axhline(0, color=INK2, lw=1)
lo, hi = R["rho_star"]["range_over_halving_5_14_and_g_boot_5_95"]
axs[0].axvspan(lo, hi, color=GRID, alpha=0.8, lw=0)
axs[0].text((lo + hi) / 2, 0.55, "rho* band\n(static optimal)", ha="center", fontsize=7, color=INK2)
axs[0].set_xlabel("rho (share of risk that is frontier-relative)")
axs[0].set_ylabel("optimal threshold drift (OOM/yr)")
axs[0].legend(fontsize=7)
for eta, col in [(0.5, BLUE), (1.0, ORANGE), (2.0, AQUA)]:
    s = L[L.eta == eta].sort_values("rho")
    axs[1].plot(s.rho, s.ratio_static1e25_to_opt, color=col, marker="o", ms=3, label=f"static 1e25, eta={eta}")
axs[1].set_yscale("log"); axs[1].set_xlabel("rho"); axs[1].set_ylabel("loss(static) / loss(optimal), 2025-30")
axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_optimal_indexing.png"), dpi=180); plt.close(fig)

# ---------------- Fig 4: who is covered vs threshold --------------------------
J = pd.read_csv(os.path.join(RES, "jurisdiction_by_threshold.csv"))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
xs = J.threshold_log10.values
axs[0].stackplot(xs, J.share_US, J.share_China, J.share_EU, J.share_Other,
                 colors=[BLUE, ORANGE, AQUA, INK2], labels=["US", "China", "EU", "other/unknown"],
                 edgecolor="#fcfcfb", linewidth=0.8)
axs[0].set_xlabel("threshold (log10 FLOP)"); axs[0].set_ylabel("share of covered models, 2024+")
axs[0].legend(fontsize=7, loc="lower left"); axs[0].set_ylim(0, 1)
axs[1].plot(xs, J.n_orgs, color=BLUE, label="distinct developers")
axs[1].plot(xs, J.n_countries, color=ORANGE, label="distinct countries")
axs[1].set_yscale("log"); axs[1].set_xlabel("threshold (log10 FLOP)"); axs[1].set_ylabel("count (known-compute models)")
axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_who_is_covered.png"), dpi=180); plt.close(fig)

# ---------------- Fig 5: observability gap ------------------------------------
O = pd.read_csv(os.path.join(RES, "observability_gap.csv"))
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(O.year, O.missing_share_closed, color=ORANGE, marker="o", label="not open weights")
ax.plot(O.year, O.missing_share_open, color=BLUE, marker="o", label="open weights")
ax.set_ylim(0, 1); ax.set_xlabel("publication year (2026 partial)")
ax.set_ylabel("share without public compute estimate"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_observability_gap.png"), dpi=180); plt.close(fig)

# ---------------- Fig 6: bunching histogram (open weights, 2024+) -------------
from scipy.optimize import minimize
k = d[d["x"].notna() & (d["t"] >= 2024) & d["open_weights"]]
edges = np.round(np.arange(23.5, 26.0 + 1e-9, 0.1), 4)
cnt, _ = np.histogram(k["x"], bins=edges)
mids = 0.5 * (edges[1:] + edges[:-1])
T, w = 25.0, 0.4
excl = (mids >= T - w) & (mids < T + w)
X = np.vstack([(mids - T) ** j for j in range(3)]).T
nll = lambda b: np.sum(np.exp(X[~excl] @ b) - cnt[~excl] * (X[~excl] @ b))
b = minimize(nll, np.array([np.log(cnt.mean()), 0, 0]), method="BFGS").x
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.bar(mids, cnt, width=0.092, color=[BLUE if not e else AQUA for e in excl], edgecolor="#fcfcfb")
ax.plot(mids, np.exp(X @ b), color=INK, lw=1.4, ls="--", label="counterfactual (quadratic Poisson, window excluded)")
ax.axvline(25, color=VIOLET, lw=1.5, label="1e25 threshold")
ax.set_xlabel("log10 training compute (FLOP)"); ax.set_ylabel("open-weights models, 2024+")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig6_bunching_open_weights.png"), dpi=180); plt.close(fig)

# cumulative 1e26 count for comparison with Cottier & Owen (2025)
s = p[p.rule == "static_1e26"].sort_values("year")
known_before = int(((d["x"] >= 26) & (d["t"] < 2024)).sum())
cum = known_before + s.exp_known.cumsum()
cum_m = known_before + s.exp_mar.cumsum()
with open(os.path.join(RES, "cumulative_1e26.json"), "w") as fh:
    json.dump({"years": s.year.tolist(), "cum_known": cum.tolist(), "cum_mar": cum_m.tolist()}, fh, indent=1)
print("cumulative 1e26:", dict(zip(s.year, np.round(cum, 1))), dict(zip(s.year, np.round(cum_m, 1))))
print("figures written")
