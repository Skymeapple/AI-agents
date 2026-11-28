"""Figures for the revision: fig4 (certificate vs margin / collapse controls over training, MLP and
transformer), fig5 (digits error-detection AUROC with 95% CIs).  Usage: python3 make_figures2.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
S = json.load(open(os.path.join(RES, "controls_summary.json")))
runs = S["runs"]


def pick(arch, frac, wd, seed):
    for r in runs:
        if r["arch"] == arch and r["frac"] == frac and r["wd"] == wd and r["seed"] == seed:
            return r


panels = [("mlp", 0.5, 1.0, 0, "MLP, frac 0.5, wd 1 (seed 0)"), ("mlp", 0.5, 3.0, 0, "MLP, frac 0.5, wd 3 (seed 0)"),
          ("mlp", 0.5, 0.3, 0, "MLP, frac 0.5, wd 0.3 (seed 0)"), ("transformer1L", 0.5, 1.0, 0, "1-layer transformer, frac 0.5 (seed 0)"),
          ("transformer1L", 0.3, 1.0, 0, "1-layer transformer, frac 0.3 (seed 0)")]
panels = [p for p in panels if pick(*p[:4]) is not None]
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), squeeze=False)
for ax, (a, f, wd, s, title) in zip(axes[0], panels):
    r = pick(a, f, wd, s)["series"]
    st = np.array(r["step"])
    ax.plot(st, r["test_acc_full"], "k-", label="test acc")
    ax.plot(st, r["C"], color="C3", label="C (full space)")
    ax.plot(st, r["C_vis"], color="C3", ls="--", label="C_vis (readout subspace)")
    ax.plot(st, r["M_abs5"], color="C0", label="margin > 5")
    ax.plot(st, r["M_q10"], color="C0", ls=":", label="margin > train 10th pct")
    nc = np.array(r["NC1_test"]); ax.plot(st, nc / np.nanmax(nc), color="C2", ls="-.", label="NC1 (test) / max")
    tl = np.array(r["test_loss"]); ax.plot(st, tl / np.nanmax(tl), color="0.5", lw=0.8, label="test loss / max")
    ax.set_title(title, fontsize=9); ax.set_xlabel("step"); ax.set_ylim(-0.03, 1.03)
axes[0][0].legend(fontsize=6.5, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig4_controls.png"), dpi=140); plt.close()

D = json.load(open(os.path.join(RES, "digits2_summary.json")))
sc = ["msp", "logit_margin", "energy", "mahalanobis", "knn10", "hull_dist", "hull_contrast", "certvis_slack"]
perts = ["none", "right1", "right2", "down1", "noise0.25", "rot15"]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, model in zip(axes, ["mlp", "cnn"]):
    w = 0.1
    for j, k in enumerate(sc):
        m = [D[f"{model}/{p}"]["auroc"][k]["mean"] for p in perts]
        lo = [D[f"{model}/{p}"]["auroc"][k]["mean"] - D[f"{model}/{p}"]["auroc"][k]["lo"] for p in perts]
        ax.bar(np.arange(len(perts)) + (j - len(sc) / 2) * w, m, w, yerr=lo, label=k, capsize=1.5)
    ax.axhline(0.5, color="k", lw=0.6); ax.set_xticks(range(len(perts))); ax.set_xticklabels(perts)
    ax.set_ylim(0.3, 1.0); ax.set_title(f"digits {model}: error-detection AUROC (mean over 5 seeds, 95% CI)", fontsize=9)
axes[0].legend(fontsize=6.5, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig5_digits_auroc.png"), dpi=140); plt.close()
print("ok")
