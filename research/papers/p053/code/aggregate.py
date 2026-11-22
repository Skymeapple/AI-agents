"""Aggregate raw_results.json into tables and figures. Usage: python3 aggregate.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
raw = json.load(open(os.path.join(R, "raw_results.json")))
ORDER = ["D", "X25", "X50", "X75", "G", "M"]
LABEL = {"D": "D (w=0, dissociated)", "X25": "w=0.25", "X50": "w=0.5", "X75": "w=0.75",
         "G": "G (w=1, grounded)", "M": "M (mimic, no state)"}
COL = {"D": "#86b6ef", "X25": "#5598e7", "X50": "#2a78d6", "X75": "#1c5cab", "G": "#0d366b", "M": "#eb6834"}
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

by = {k: [r for r in raw if r["world"] == k] for k in ORDER}


def ms(vals):
    v = np.array(vals, float); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


# ---------- Table 1: behaviour + probes ----------
keys = ["act_acc_rho0.6", "rep_agree_cue_rho0.0", "rep_agree_mood_rho0.0", "conflict_follow_mood",
        "neutral_under_hide_rho0.0", "probe_report", "probe_mood", "probe_cue", "probe_resource",
        "probe_mood_hidden", "act_acc_hidden", "GI", "GI_naive_rho0.6"]
lines = ["| metric | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
summary = {}
for k in keys:
    row = []
    for w in ORDER:
        m, s = ms([r[k] for r in by[w]])
        summary.setdefault(w, {})[k] = (m, s)
        row.append(f"{m:.3f} ± {s:.3f}")
    lines.append(f"| {k} | " + " | ".join(row) + " |")
# sweep summary
for q in ["da_z", "dr_z", "delayed_da_z", "delayed_dr_z", "cci", "cos_pm"]:
    for rho in [-0.6, 0.0, 0.6, 0.9]:
        row = []
        for w in ORDER:
            m, s = ms([[x[q] for x in r["sweep"] if x["rho"] == rho][0] for r in by[w]])
            summary[w][f"{q}@{rho}"] = (m, s)
            row.append(f"{m:.3f} ± {s:.3f}")
        lines.append(f"| {q} @ rho_est={rho} | " + " | ".join(row) + " |")
open(os.path.join(R, "table_main.md"), "w").write("\n".join(lines) + "\n")
json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("\n".join(lines))

# slope of da_z against rho_est per model (linear fit)
slopes = {}
for w in ORDER:
    sl = []
    for r in by[w]:
        x = np.array([s["rho"] for s in r["sweep"]]); y = np.array([s["da_z"] for s in r["sweep"]])
        sl.append(np.polyfit(x, y, 1)[0])
    slopes[w] = ms(sl)
json.dump(slopes, open(os.path.join(R, "slopes_da_z.json"), "w"), indent=1)
print("slopes d(da_z)/d(rho):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in slopes.items()})

# ---------- Figure 1: cross-channel effect vs rho_est ----------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, q, title in zip(axes, ["dr_z", "da_z", "delayed_da_z"],
                        ["Report shift (immediate)", "Action shift (immediate)", "Action shift (3 steps later)"]):
    for w in ORDER:
        rhos = [s["rho"] for s in by[w][0]["sweep"]]
        Y = np.array([[s[q] for s in r["sweep"]] for r in by[w]])
        m, s = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(rhos, m - s, m + s, color=COL[w], alpha=0.15, lw=0)
        ax.plot(rhos, m, color=COL[w], lw=2, marker="o", ms=4, label=LABEL[w])
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(r"cue–situation correlation in estimation set, $\rho_{est}$")
    ax.axhline(0, color="#c3c2b7", lw=1)
axes[0].set_ylabel("shift (SD of logit) per unit steer")
h, l = axes[2].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=6)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(R, "fig1_cross_channel_vs_rho.png"), dpi=180)

# ---------- Figure 2: GI vs behavioural ground truth ----------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
for ax, k, title in zip(axes, ["GI", "GI_naive_rho0.6"],
                        [r"Grounding index, decorrelated ($\rho_{est}=0$)", r"Same index, naive ($\rho_{est}=0.6$)"]):
    for w in ORDER[:-1]:
        x = [r["conflict_follow_mood"] for r in by[w]]; y = [r[k] for r in by[w]]
        ax.scatter(x, y, s=36, color=COL[w], edgecolor="white", linewidth=1, label=LABEL[w], zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("behavioural ground truth:\nP(report follows situation | conflict)")
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.1, 1.15)
axes[0].set_ylabel("index value")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(R, "fig2_grounding_index.png"), dpi=180)

# correlation stats
xs = np.array([r["conflict_follow_mood"] for w in ORDER[:-1] for r in by[w]])
for k in ["GI", "GI_naive_rho0.6"]:
    ys = np.array([r[k] for w in ORDER[:-1] for r in by[w]])
    print(k, "pearson r =", np.corrcoef(xs, ys)[0, 1], "MAE =", np.abs(xs - ys).mean())
json.dump({k: dict(pearson=float(np.corrcoef(xs, np.array([r[k] for w in ORDER[:-1] for r in by[w]]))[0, 1]),
                   mae=float(np.abs(xs - np.array([r[k] for w in ORDER[:-1] for r in by[w]])).mean()))
           for k in ["GI", "GI_naive_rho0.6"]}, open(os.path.join(R, "gi_validation.json"), "w"), indent=1)

# ---------- Figure 3: theory check ----------
th = json.load(open(os.path.join(R, "theory_check.json")))
fig, ax = plt.subplots(figsize=(4.2, 3.3))
cols = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
for i, (bm, bc) in enumerate([(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5), (1.0, 0.0)]):
    rs = [t for t in th if t["latent"] == "gaussian" and t["beta_m"] == bm and t["beta_c"] == bc]
    ax.plot([t["rho"] for t in rs], [t["cci_pred"] for t in rs], color=cols[i], lw=2,
            label=fr"$\beta_m$={bm}, $\beta_c$={bc}")
    ax.scatter([t["rho"] for t in rs], [t["cci_emp"] for t in rs], color=cols[i], s=30, edgecolor="white", zorder=3)
ax.set_xlabel(r"$\rho_{est}$"); ax.set_ylabel("cross-channel coupling (CCI)")
ax.set_title("Prop. 1: lines = closed form, dots = Monte Carlo", fontsize=9, loc="left", color=INK)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(R, "fig3_theory_check.png"), dpi=180)

# ---------- Robustness table (GRU-32, LSTM-64) ----------
rp = os.path.join(R, "robustness_raw.json")
if os.path.exists(rp):
    rob = json.load(open(rp))
    rk = ["conflict_follow_mood", "probe_mood", "probe_mood_hidden", "GI"]
    L = ["| arch | world | " + " | ".join(rk) + " | da_z@rho0 | da_z@rho0.6 | delayed_da_z@rho0.6 |", "|---" * (len(rk) + 5) + "|"]
    allx, ally = [], []
    for a in ("gru32", "lstm64"):
        for wd in ("D", "X50", "G", "M"):
            rs = [r for r in rob if r["arch"] == a and r["world"] == wd]
            cells = [f"{ms([r[k] for r in rs])[0]:.3f} ± {ms([r[k] for r in rs])[1]:.3f}" for k in rk]
            for q, rho in (("da_z", 0.0), ("da_z", 0.6), ("delayed_da_z", 0.6)):
                v = [[s[q] for s in r["sweep"] if s["rho"] == rho][0] for r in rs]
                cells.append(f"{ms(v)[0]:.3f} ± {ms(v)[1]:.3f}")
            L.append(f"| {a} | {wd} | " + " | ".join(cells) + " |")
            if wd != "M":
                allx += [r["conflict_follow_mood"] for r in rs]; ally += [r["GI"] for r in rs]
    L.append(f"\nGI vs conflict_follow_mood (non-M, n={len(allx)}): pearson r = {np.corrcoef(allx, ally)[0,1]:.4f}, "
             f"MAE = {np.abs(np.array(allx) - np.array(ally)).mean():.4f}")
    open(os.path.join(R, "table_robustness.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
