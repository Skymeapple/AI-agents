"""Figures from ../results/summary.json. Usage: python3 plots.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

S = json.load(open("../results/summary.json"))
OUT = "../results"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
sc = S["scenarios"]

# ---- Fig 1: regime hypotheses -> product structure
hyp = ["H0_no_conflict", "H1_baseline", "H4_USF_runtime", "H2_CN_runtime", "H3_CN_vs_US_wide"]
lab = ["H0 no conflict", "H1 CN–USF conflict\n(weights)", "H4 conflict, US side\naccepts runtime", "H2 CN norm\nruntime-assessable", "H3 CN vs all\nUS segments"]
metrics = [("p_single_global", "single global weights\nserving all six segments", BLUE),
           ("p_cn_separate_weights", "CN served by its own\nweight variant", ORANGE),
           ("served_USF", "USF segment served", AQUA)]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(hyp)); w = 0.26
for k, (key, name, col) in enumerate(metrics):
    vals = [sc[h]["served_frac"]["USF"] if key == "served_USF" else sc[h][key] for h in hyp]
    ax.bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=name)
    for xi, v in zip(x + (k - 1) * w, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5)
ax.set_ylabel("share of 20,000 parameter draws"); ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_regimes.png", dpi=200); plt.close(fig)

# ---- Fig 2: coordination instruments
ins = ["I1_MR_incident", "I3_harmonize_training_summary", "I2_harmonize_frameworks", "I4_harmonize_marking",
       "I6_harmonize_provenance", "I5_harmonize_visible", "I_bundle_all_MR_harmonization", "H4_USF_runtime", "H2_CN_runtime"]
ilab = ["MR: incident reports (process)", "Harmonize training-data summary (process)", "Harmonize safety frameworks (process)",
        "Harmonize machine-readable marking (runtime)", "Harmonize data-provenance policy (weights)",
        "Harmonize visible labels (runtime)", "Bundle of all six MR/harmonization", "US side accepts runtime compliance (layer)",
        "CN accepts runtime compliance (layer)"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
y = np.arange(len(ins))[::-1]
gain = [sc[i]["mean_net_gain_vs_H1"] for i in ins]
axs[0].barh(y, gain, 0.6, color=BLUE)
for yi, g in zip(y, gain):
    axs[0].text(g + 0.03, yi, f"{g:.2f}", va="center", fontsize=6.5, color=INK2)
axs[0].set_yticks(y); axs[0].set_yticklabels(ilab, fontsize=7)
axs[0].set_xlabel("mean net-value gain vs H1 (model units)")
axs[0].xaxis.grid(True, color=GRID, lw=0.6); axs[0].set_axisbelow(True)
pw = [sc[i]["p_weight_structure_changed_vs_H1"] for i in ins]
pr = [sc[i]["p_runtime_structure_changed_vs_H1"] for i in ins]
axs[1].barh(y + 0.17, pw, 0.32, color=ORANGE, label="weight partition changes")
axs[1].barh(y - 0.17, pr, 0.32, color=AQUA, label="runtime partition changes")
axs[1].set_xlim(0, 1.05); axs[1].set_xlabel("share of draws")
axs[1].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
axs[1].xaxis.grid(True, color=GRID, lw=0.6); axs[1].set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_instruments.png", dpi=200); plt.close(fig)

# ---- Fig 3: phase diagram (Proposition 1)
ph = S["phase"]
Fw = np.array(ph["Fw_grid"]); q = np.array(ph["q_grid"]); fork = np.array(ph["fork"])
qs = np.array(ph["q_star"])
pred = (q[:, None] > qs[None, :]).astype(int)
mask2 = fork <= 1
agree = float((pred[mask2] == fork[mask2]).mean())
share_other = float((fork == 2).mean()); share_exit = float((fork == 3).mean())
fig, ax = plt.subplots(figsize=(4.6, 3.4))
from matplotlib.colors import ListedColormap
ax.pcolormesh(Fw, q, fork, cmap=ListedColormap(["#cde2fb", "#1c5cab", "#eda100", "#e87ba4"]), vmin=-0.5, vmax=3.5, shading="nearest")
ax.plot(Fw, qs, color=ORANGE, lw=2, label=r"closed form $q^*=(F_w+\kappa)/M_{\mathrm{rest}}$")
ax.set_xscale("log"); ax.set_ylim(0, 0.4)
ax.set_xlabel(r"weight-fork cost $F_w$"); ax.set_ylabel(r"value loss $q$ of CN norm for non-CN users")
ax.text(0.7, 0.33, "CN gets own weights\n(fork)", color="white", fontsize=8)
ax.text(8, 0.03, "single weights;\nCN norm exported", color=INK, fontsize=8)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_title(f"H0, medians; agreement {agree:.3f} on single/CN-fork cells", fontsize=8, color=INK2)
ax.text(0.55, 0.004, "yellow: lenient-bloc split; pink: exit CN", color=INK, fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_phase.png", dpi=200); plt.close(fig)
json.dump({"phase_closed_form_agreement_on_single_or_CNfork_cells": agree, "share_cells_other_split": share_other, "share_cells_exit_CN": share_exit}, open(f"{OUT}/phase_check.json", "w"))

# ---- Fig 4: rule export ("Brussels effect") per obligation, by regime
segs = S["segments"]
imp = ["EU", "CN", "USF", "CA", "NY", "ROW"]
regs = [("H0_no_conflict", "H0 no conflict", BLUE), ("H1_baseline", "H1 CN–USF conflict", ORANGE),
        ("H2_CN_runtime", "H2 CN norm at runtime", AQUA)]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, (dim, exporter, title) in zip(axs, [("D8_political_content_norm", "CN", "CN content norm (D8) binds product sold in ..."),
                                           ("D3_model_evaluation", "EU", "EU evaluation level (D3) binds product sold in ...")]):
    xi = np.arange(len(imp)); w = 0.26
    for k, (nm, lb, col) in enumerate(regs):
        M = np.array(S["export_matrices"][nm]["per_dim"][dim])
        v = [M[segs.index(exporter), segs.index(t)] for t in imp]
        ax.bar(xi + (k - 1) * w, v, w - 0.03, color=col, label=lb)
        for x0, vv in zip(xi + (k - 1) * w, v):
            if vv > 0.005:
                ax.text(x0, vv + 0.015, f"{vv:.2f}", ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(xi); ax.set_xticklabels(imp); ax.set_title(title, fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 1.12)
axs[0].set_ylabel("share of draws")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_exports.png", dpi=200); plt.close(fig)
print("phase agreement", agree)
