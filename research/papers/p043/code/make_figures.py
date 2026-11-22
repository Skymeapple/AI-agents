"""Aggregate JSON outputs into tables (results/summary.md) and figures."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CONDS = ["lieA_instructed", "lieB_incentive", "lieC_sleeper", "cue_no_lie", "adv_no_secret"]
SHORT = {"lieA_instructed": "A: instructed lie (ID)", "lieB_incentive": "B: incentive lie",
         "lieC_sleeper": "C: sleeper lie", "cue_no_lie": "cue, no lie", "adv_no_secret": "adv. listener, no lie"}
lines = []


def ms(x):
    x = np.asarray(x, float); return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


for variant in ["dep", "indep"]:
    files = sorted(glob.glob(os.path.join(RES, f"main_{variant}_s*.json")))
    if not files:
        continue
    R = [json.load(open(f)) for f in files]
    lines.append(f"\n## Main results, variant={variant}, n_seeds={len(R)}\n")
    lines.append("Behavioural accuracy (all conditions, mean over seeds): " +
                 ", ".join(f"{c}={np.mean([r['behaviour_acc'][c] for r in R]):.3f}" for c in R[0]["behaviour_acc"]))
    lines.append("\nAUROC vs honest (test entities). For lie conditions higher is better; for no-lie conditions 0.5 is ideal.\n")
    dets = list(R[0]["summary_auroc"].keys())
    lines.append("| detector | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    lines.append("|---" * (len(CONDS) + 1) + "|")
    for d in dets:
        lines.append(f"| {d} | " + " | ".join(ms([r["summary_auroc"][d][c] for r in R]) for c in CONDS) + " |")
    lines.append("\nFlag rate at threshold giving 1% FPR on held-out honest calibration data:\n")
    lines.append("| detector | honest(test) | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    lines.append("|---" * (len(CONDS) + 2) + "|")
    for d in R[0]["flag_rate_at_1pct_fpr"]:
        lines.append(f"| {d} | {ms([r['flag_rate_at_1pct_fpr'][d]['honest_test'] for r in R])} | " +
                     " | ".join(ms([r["flag_rate_at_1pct_fpr"][d][c] for r in R]) for c in CONDS) + " |")
    lines.append(f"\nAdmissible layers per seed: {[r['admissible_layers'] for r in R]}; "
                 f"LR val-best layer per seed: {[r['LR_valbest_layer'] for r in R]}")
    # belief decodability table
    L1 = len(R[0]["belief_val_acc"])
    lines.append("\nBelief-decoder accuracy for the TRUE value, per layer (mean over seeds):\n")
    lines.append("| condition | " + " | ".join(f"L{l}" for l in range(L1)) + " |")
    lines.append("|---" * (L1 + 1) + "|")
    for c in ["honest"] + CONDS:
        lines.append(f"| {c} | " + " | ".join(f"{np.mean([r['belief_acc_truth'][c][l] for r in R]):.2f}" for l in range(L1)) + " |")
    lines.append("\nGeometry of the LR probe (mean over seeds): cos(w, cue dir), cos(w, act dir), ||cue||, ||act||\n")
    for l in range(L1):
        g = [r["geometry"][l] for r in R]
        lines.append(f"- L{l}: cos_w_cue={np.mean([x['cos_w_cue'] for x in g]):.2f}, cos_w_act={np.mean([x['cos_w_act'] for x in g]):.2f}, "
                     f"norm_cue={np.mean([x['norm_cue'] for x in g]):.2f}, norm_act={np.mean([x['norm_act'] for x in g]):.2f}")
    # per-layer figure
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    cols = {"lieA_instructed": "#1f77b4", "lieB_incentive": "#ff7f0e", "lieC_sleeper": "#2ca02c",
            "cue_no_lie": "#d62728", "adv_no_secret": "#9467bd"}
    for ax, det in zip(axs, ["LR", "DiffMeans", "Mahalanobis", "Mismatch_l"]):
        for c in CONDS:
            M = np.array([r["per_layer_auroc"][det][c] for r in R])
            ax.errorbar(range(L1), M.mean(0), yerr=M.std(0, ddof=1) if len(R) > 1 else None,
                        marker="o", color=cols[c], label=SHORT[c], capsize=2,
                        ls="-" if c.startswith("lie") else "--")
        ax.axhline(0.5, color="gray", lw=.8)
        ax.set_title({"Mismatch_l": "Mismatch (single layer)"}.get(det, det)); ax.set_xlabel("layer (0 = embedding)")
    axs[0].set_ylabel("AUROC vs honest"); axs[-1].legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(RES, f"fig_layers_{variant}.png"), dpi=150); plt.close()
    # belief decodability figure
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for c in ["honest"] + CONDS:
        M = np.array([r["belief_acc_truth"][c] for r in R])
        ax.plot(range(L1), M.mean(0), marker="o", label=c, color=cols.get(c, "k"),
                ls="-" if c.startswith("lie") else "--")
    ax.set_xlabel("layer"); ax.set_ylabel("acc. of honest-trained decoder vs TRUE value")
    ax.set_title(f"Belief decodability ({variant})"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, f"fig_belief_{variant}.png"), dpi=150); plt.close()

for variant in ["dep", "indep"]:
    f = os.path.join(RES, f"controls_{variant}.json")
    if not os.path.exists(f):
        continue
    C = json.load(open(f))
    lines.append(f"\n## Probe training with controls (variant={variant}, n={len(C)}), best-layer-in-hindsight AUROC\n")
    lines.append("| probe | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    lines.append("|---" * (len(CONDS) + 1) + "|")
    for k in ["LR_plain", "LR_controls", "LR_AB_plus_controls"]:
        vals = []
        for c in CONDS:
            agg = [max(r[k][c]) if c.startswith("lie") else min(r[k][c]) for r in C]
            vals.append(ms(agg))
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    lines.append("\nSame, at a fixed layer (L3):\n")
    lines.append("| probe | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    lines.append("|---" * (len(CONDS) + 1) + "|")
    for k in ["LR_plain", "LR_controls", "LR_AB_plus_controls"]:
        lines.append(f"| {k} | " + " | ".join(ms([r[k][c][3] for r in C]) for c in CONDS) + " |")

for f in sorted(glob.glob(os.path.join(RES, "obfuscation_*.json"))):
    O = json.load(open(f))
    lines.append(f"\n## Obfuscation: {os.path.basename(f)} (n_seeds={len(O)})\n")
    lines.append("| attack | detector | version | " + " | ".join(SHORT[c] for c in CONDS[:4]) + " | behaviour (A/B/C/honest) |")
    lines.append("|---" * 8 + "|")
    for att in ["none", "vsLR", "vsMismatch", "joint"]:
        for det in ["LR", "Mismatch"]:
            for ver in ["frozen", "retrained"]:
                vals = [ms([o[att][ver][det][c] for o in O]) for c in CONDS[:4]]
                beh = "/".join(f"{np.mean([o[att][ver]['behaviour'][c] for o in O]):.2f}"
                               for c in ["lieA_instructed", "lieB_incentive", "lieC_sleeper", "honest"])
                lines.append(f"| {att} | {det} | {ver} | " + " | ".join(vals) + f" | {beh} |")

open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines))
print("\n".join(lines))

# ---- Probit conservation test (Prop. 1 corollary) on the trained transformers ----
from scipy.stats import norm
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "main_dep_s*.json")))]
pts = []
for r in R:
    for l in range(1, len(r["LR_val_auroc"])):
        clip = lambda a: norm.ppf(np.clip(a, 0.005, 0.995))
        idv = clip(r["per_layer_auroc"]["LR"]["lieA_instructed"][l])
        for j in ["lieB_incentive", "lieC_sleeper"]:
            pts.append((idv, clip(r["per_layer_auroc"]["LR"][j][l]) + clip(r["per_layer_auroc"]["LR"]["cue_no_lie"][l]), j))
pts_arr = np.array([(a, b) for a, b, _ in pts])
rr = np.corrcoef(pts_arr[:, 0], pts_arr[:, 1])[0, 1]
txt = (f"\n## Probit conservation check (LR probe, layers 1-4, all dep seeds)\n"
       f"Prop.1 predicts probit(AUROC_j)+probit(AUROC_cue) = probit(AUROC_ID) when the trigger-j offset is orthogonal to the probe.\n"
       f"n={len(pts)} points; Pearson r={rr:.3f}; mean(LHS)={pts_arr[:,1].mean():.3f}, mean(RHS)={pts_arr[:,0].mean():.3f}; "
       f"mean residual LHS-RHS={np.mean(pts_arr[:,1]-pts_arr[:,0]):.3f} (sd {np.std(pts_arr[:,1]-pts_arr[:,0], ddof=1):.3f})\n")
for j in ["lieB_incentive", "lieC_sleeper"]:
    sub = np.array([(a, b) for a, b, k in pts if k == j])
    txt += f"- {j}: mean residual {np.mean(sub[:,1]-sub[:,0]):.3f} (sd {np.std(sub[:,1]-sub[:,0], ddof=1):.3f}), r={np.corrcoef(sub[:,0], sub[:,1])[0,1]:.3f}\n"
fig, ax = plt.subplots(figsize=(4.2, 3.8))
for j, col in [("lieB_incentive", "#ff7f0e"), ("lieC_sleeper", "#2ca02c")]:
    sub = np.array([(a, b) for a, b, k in pts if k == j])
    ax.scatter(sub[:, 0], sub[:, 1], color=col, label=j, s=18)
lim = [min(pts_arr.min(), 0) - .2, pts_arr.max() + .2]
ax.plot(lim, lim, "k:", label="Prop. 1 prediction")
ax.set_xlabel(r"$\Phi^{-1}$(AUROC$_{ID}$)"); ax.set_ylabel(r"$\Phi^{-1}$(AUROC$_j$) + $\Phi^{-1}$(AUROC$_{cue}$)")
ax.legend(fontsize=7); ax.set_title("Probit conservation (LR probe)")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_probit.png"), dpi=150); plt.close()
with open(os.path.join(RES, "summary.md"), "a") as fh:
    fh.write(txt)
print(txt)
