"""Paper figures. Run after analysis.py and mc.py."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "lines.linewidth": 2})


def fig_diurnal(p):
    sites = ["San Jose CA", "Phoenix AZ", "Ashburn VA", "Boston MA"]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, s in zip(axs, sites):
        q = p[p.site == s].copy()
        q["C"] = q.pue * q.ci_prod
        q["Woff"] = q.pue * q.ewif
        g = q.groupby("local_hour")[["C", "wue_adiab", "wue_tower", "Woff"]].mean()
        for col, c, lab in [("C", INK, "carbon $C_t$"), ("Woff", BLUE, "off-site water"),
                            ("wue_tower", AQUA, "on-site WUE (tower)"), ("wue_adiab", ORANGE, "on-site WUE (adiabatic)")]:
            ax.plot(g.index, g[col] / g[col].mean(), color=c, label=lab)
        ax.set_title(s, fontsize=9, color=INK)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("local hour")
    axs[0].set_ylabel("value / daily mean (annual avg.)")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_diurnal.png"), dpi=200)


def fig_boundary(R):
    r = R[(R.ci == "ci_prod") & (R.phi == 0.3)]
    sites = list(dict.fromkeys(r.site))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, arch, ttl in [(axs[0], "tower", "Cooling-tower archetype"), (axs[1], "adiab", "Adiabatic archetype")]:
        q = r[r.arch == arch].set_index("site").loc[sites]
        y = np.arange(len(sites))
        ax.barh(y + 0.2, q.dWon_rel_carbon_only, height=0.38, color=ORANGE, label="on-site boundary (% of on-site water)")
        ax.barh(y - 0.2, q.dW_carbon_only, height=0.38, color=BLUE, label="on-site + off-site boundary (% of total water)")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, sites)
        ax.invert_yaxis()
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("water change under carbon-only shifting (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(RES, "fig_boundary.png"), dpi=200)


def fig_frontiers(F, R):
    cases = [("Phoenix AZ", "adiab"), ("The Dalles OR", "adiab"), ("Boston MA", "tower"), ("Charlotte NC", "tower")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (s, a) in zip(axs, cases):
        f = F[(F.site == s) & (F.arch == a)]
        ax.plot(f.dC, f.dW, color=BLUE, marker="o", ms=2.5)
        ax.scatter([f.dC.iloc[-1]], [f.dW.iloc[-1]], color=ORANGE, s=40, zorder=5, label="carbon-only")
        ax.scatter([f.dC.iloc[0]], [f.dW.iloc[0]], color=AQUA, s=40, zorder=5, label="water-only")
        k = R[(R.site == s) & (R.arch == a) & (R.ci == "ci_prod") & (R.phi == 0.3)].iloc[0]
        ax.scatter([k.knee_dC], [k.knee_dW], color=INK, marker="x", s=40, zorder=6, label="water-neutral knee")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{s} ({a})", fontsize=9)
        ax.set_xlabel("carbon change (%)")
    axs[0].set_ylabel("total water change (%)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_frontiers.png"), dpi=200)


def fig_pred(R):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for a, c in [("tower", AQUA), ("adiab", ORANGE)]:
        q = R[R.arch == a]
        ax.scatter(q.pred_dW_pct, q.real_dW_pct, s=12, color=c, label=a)
    lim = [min(R.pred_dW_pct.min(), R.real_dW_pct.min()) - 0.3, max(R.pred_dW_pct.max(), R.real_dW_pct.max()) + 0.3]
    ax.plot(lim, lim, color=INK2, lw=0.8)
    ax.set_xlabel("predicted $\\Delta W$ = $\\Sigma_d b_d \\Delta C_d$ (%)")
    ax.set_ylabel("realized $\\Delta W$ (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_prop2.png"), dpi=200)


def fig_mc(D):
    sites = list(dict.fromkeys(D.site))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, a in zip(axs, ["tower", "adiab"]):
        data = [D[(D.site == s) & (D.arch == a)].dW.values for s in sites]
        bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK))
        for b in bp["boxes"]:
            b.set_facecolor("#cde2fb"); b.set_edgecolor(BLUE)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(1, len(sites) + 1), sites)
        ax.invert_yaxis()
        ax.set_title(f"{a} archetype: Monte Carlo, 300 draws", fontsize=9)
        ax.set_xlabel("total water change under carbon-only shifting (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_mc.png"), dpi=200)


def main():
    p = pd.read_pickle(os.path.join(RES, "panel_2023.pkl"))
    R = pd.read_csv(os.path.join(RES, "shifting_summary.csv"))
    F = pd.read_csv(os.path.join(RES, "frontiers.csv"))
    fig_diurnal(p); fig_boundary(R); fig_frontiers(F, R); fig_pred(R)
    mcp = os.path.join(RES, "mc_draws.csv")
    if os.path.exists(mcp):
        fig_mc(pd.read_csv(mcp))


if __name__ == "__main__":
    main()
