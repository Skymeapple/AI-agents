"""Figures for p075 from results/ (run after analysis.py)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
COL = {"closed": "#3b4cc0", "open": "#d1495b", "noCN": "#edae49", "noUS": "#00798c", "grey": "#888888"}
r = json.load(open(f"{RES}/results.json"))

# Fig 1: frontiers
f = pd.read_csv(f"{RES}/frontier_series.csv", parse_dates=["day"])
f = f[f.day >= "2023-06-01"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.step(f.day, f.closed, where="post", color=COL["closed"], label="Closed-weight frontier")
ax.step(f.day, f.open, where="post", color=COL["open"], label="Open-weight frontier (actual)")
ax.step(f.day, f["open_minus_CN"], where="post", color=COL["noCN"], ls="--",
        label="Open frontier without CN developers")
ax.step(f.day, f["open_minus_US"], where="post", color=COL["noUS"], ls=":",
        label="Open frontier without US developers")
ax.set_ylabel("ECI (point estimate)"); ax.set_ylim(100, 178)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("Fig. 1  Frontiers and leave-coalition-out counterfactuals", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{RES}/fig1_frontiers.png"); plt.close(fig)

# Fig 2: monthly lag
m = pd.read_csv(f"{RES}/monthly_lag.csv", parse_dates=["month"])
m = m[m.month >= "2024-01-01"]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.fill_between(m.month, m.lag_lo, m.lag_hi, color=COL["open"], alpha=0.2, lw=0, label="90% band (score noise)")
ax.plot(m.month, m.lag_median, color=COL["open"], label="Monthly mean horizontal lag (median draw)")
ax.set_ylabel("months"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("Fig. 2  Open-weight lag behind the overall frontier", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{RES}/fig2_lag.png"); plt.close(fig)

# Fig 3: coordination gap (CN coalition) and unilateral setbacks
rem = pd.read_csv(f"{RES}/removal_counterfactuals.csv")
dev = rem[(rem.kind == "developer") & (rem.stale_mo > 0)].sort_values("stale_mo")
cg = r["coordination_gap"]["CN developers"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
ax.barh(dev.removed, dev.stale_mo, xerr=[dev.stale_mo - dev.stale_mo_lo, dev.stale_mo_hi - dev.stale_mo],
        color=COL["grey"], ecolor="black", capsize=2)
ax.set_xlabel("frontier setback if developer withheld (months)")
ax.set_title("(a) unilateral removal", fontsize=8.5, loc="left")
ax = axs[1]
vals = [cg["sum_unilateral_stale_mo"]["median"], cg["joint_stale_mo"]["median"]]
lo = [cg["sum_unilateral_stale_mo"]["lo90"], cg["joint_stale_mo"]["lo90"]]
hi = [cg["sum_unilateral_stale_mo"]["hi90"], cg["joint_stale_mo"]["hi90"]]
ax.bar(["Sum of\nunilateral", "Joint\n(all CN)"], vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)],
       color=[COL["grey"], COL["noCN"]], capsize=3)
ax.set_ylabel("setback (months)")
ax.set_title("(b) coordination gap", fontsize=8.5, loc="left")
fig.suptitle("Fig. 3  Restraint is superadditive (window 2025-01-01 to 2026-11-17)", fontsize=9, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{RES}/fig3_coordination_gap.png"); plt.close(fig)

# Fig 4: release rule
rr = r["release_rule"]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for J, c, mk in [("all", "black", "o"), ("CN only", COL["noCN"], "s"), ("US only", COL["noUS"], "^"),
                 ("non-CN", COL["closed"], "v")]:
    ds = sorted({int(k.split("|")[1]) for k in rr if k.startswith(J + "|")})
    y = [rr[f"{J}|{d}"]["lag_closed_mo"]["median"] for d in ds]
    lo = [rr[f"{J}|{d}"]["lag_closed_mo"]["lo90"] for d in ds]
    hi = [rr[f"{J}|{d}"]["lag_closed_mo"]["hi90"] for d in ds]
    ax.errorbar(ds, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=c, marker=mk, capsize=2,
                label=f"rule adopted by: {J}", ms=4)
ax.plot([0, 12], [0, 12], color=COL["grey"], lw=0.6, ls="--")
ax.set_xlabel("required closed-frontier lead before open release, Delta (months)")
ax.set_ylabel("resulting open lag (months)")
ax.legend(frameon=False, fontsize=7.5)
ax.set_title("Fig. 4  Staleness rule: who adopts it matters", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{RES}/fig4_release_rule.png"); plt.close(fig)

# Fig 5: Shapley vs marginal
sh = pd.read_csv(f"{RES}/shapley.csv")
sh = sh[sh.shapley_eci_months > 1].sort_values("shapley_eci_months")
fig, ax = plt.subplots(figsize=(5.6, 3.2))
y = np.arange(len(sh))
ax.barh(y + 0.2, sh.shapley_eci_months, height=0.4, color=COL["open"], label="Shapley value")
ax.barh(y - 0.2, sh.marginal_eci_months, height=0.4, color=COL["grey"], label="Unilateral marginal")
ax.set_yticks(y); ax.set_yticklabels([f"{d} ({c})" for d, c in zip(sh.developer, sh.country)], fontsize=7.5)
ax.set_xlabel("ECI-months of open-frontier access above Jan-2025 level")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("Fig. 5  Attribution: Shapley vs leave-one-out", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{RES}/fig5_shapley.png"); plt.close(fig)
print("figures written")
