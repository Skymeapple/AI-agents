"""Make figures and a summary table from results/*.json. Usage: python3 make_figures.py"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
summary = {}

# ---------- Experiment 1 ----------
runs = {}
for f in sorted(glob.glob(os.path.join(RES, "modadd_f*.json"))):
    r = json.load(open(f))
    runs[(r["frac"], r["seed"], r["random_labels"])] = r

def series(r, k):
    return np.array([h["step"] for h in r["history"]]), np.array([h[k] for h in r["history"]])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
keys = [("test_acc", "network test acc", "k", "-"),
        ("cert_frac", "readout certificate (L2)", "C3", "-"),
        ("L2_hull_nearest_acc", "hull-nearest acc, layer 2", "C0", "-"),
        ("L1_hull_nearest_acc", "hull-nearest acc, layer 1", "C2", "--"),
        ("L2_1nn_acc", "1-NN acc, layer 2", "C1", ":"),
        ("L0_hull_nearest_acc", "hull-nearest / 1-NN acc, input", "C7", "-")]
for ax, cfg in zip(axes, [(0.5, 0, False), (0.4, 0, False), (0.5, 0, True)]):
    if cfg not in runs:
        continue
    r = runs[cfg]
    for k, lab, c, ls in keys:
        x, y = series(r, k)
        ax.plot(x, y, color=c, ls=ls, label=lab)
    x, y = series(r, "train_acc"); ax.plot(x, y, color="0.6", lw=0.8, label="train acc")
    ax.set_title(f"p=53, train frac={cfg[0]}" + (", random labels" if cfg[2] else ""))
    ax.set_xlabel("step"); ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel("fraction of held-out pairs"); axes[0].legend(fontsize=7, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig1_modadd_dynamics.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4))
for (fr, s, rl), r in sorted(runs.items()):
    if rl:
        continue
    x, ta = series(r, "test_acc"); _, cf = series(r, "cert_frac")
    ax.plot(ta, cf, "o-", ms=3, label=f"frac={fr}, seed={s}")
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("network test accuracy"); ax.set_ylabel("certified fraction (readout hull certificate)")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(RES, "fig2_cert_vs_acc.png"), dpi=150); plt.close()

mod = {}
for (fr, s, rl), r in sorted(runs.items()):
    H = r["history"]; last = H[-1]
    def first_step(k, thr):
        for h in H:
            if h[k] >= thr:
                return h["step"]
        return None
    mod[f"f{fr}_s{s}_rl{int(rl)}"] = {
        "input_lp_in_hull_frac": r["input_lp_in_hull_frac"], "n_lp_checked": r["n_lp_checked"],
        "final": {k: last[k] for k in ["train_acc", "test_acc", "cert_frac", "L0_hull_nearest_acc", "L0_1nn_acc",
                                        "L1_hull_nearest_acc", "L2_hull_nearest_acc", "L1_1nn_acc", "L2_1nn_acc",
                                        "L0_rel_dist_same_median", "L1_rel_dist_same_median", "L2_rel_dist_same_median"]},
        "max_L0_hull_nearest_acc_over_training": max(h["L0_hull_nearest_acc"] for h in H),
        "max_L0_1nn_acc_over_training": max(h["L0_1nn_acc"] for h in H),
        "step_train_acc_ge_0.99": first_step("train_acc", 0.99),
        "step_test_acc_ge_0.5": first_step("test_acc", 0.5),
        "step_cert_ge_0.5": first_step("cert_frac", 0.5),
        "step_L2hull_ge_0.5": first_step("L2_hull_nearest_acc", 0.5),
        "step_test_acc_ge_0.95": first_step("test_acc", 0.95),
        "step_cert_ge_0.95": first_step("cert_frac", 0.95),
    }
summary["modadd"] = mod

# ---------- Experiment 2 ----------
fp = os.path.join(RES, "digits.json")
if os.path.exists(fp):
    d = json.load(open(fp))
    R = d["runs"]
    dig = {"input_in_hull_uncond": [r["input_in_hull_uncond"] for r in R],
           "input_in_hull_same_class": [r["input_in_hull_same_class"] for r in R],
           "pooled_auroc": d["pooled_auroc"]}
    for dx in [0, 1, 2]:
        b = [r[f"shift{dx}"] for r in R]
        ent = {k: (float(np.mean([x[k] for x in b])), float(np.std([x[k] for x in b])))
               for k in ["acc", "cert_true_frac", "cert_pred_frac"]}
        for k in ["acc_on_certified", "acc_on_uncertified"]:
            v = [x[k] for x in b if x[k] is not None]
            ent[k] = (float(np.mean(v)), float(np.std(v)), len(v)) if v else None
        for l in ["0", "1", "2"]:
            for k in ["hull_nearest_acc", "nn1_acc", "rel_same_median"]:
                ent[f"L{l}_{k}"] = (float(np.mean([x["layers"][l][k] for x in b])), float(np.std([x["layers"][l][k] for x in b])))
        # pooled accuracy on certified vs uncertified
        err = np.concatenate([x["scores"]["err"] for x in b]); cp = np.concatenate([x["scores"]["cert_pred"] for x in b]).astype(bool)
        ent["pooled_cert_count"] = int(cp.sum()); ent["pooled_n"] = int(len(cp))
        ent["pooled_acc_certified"] = float(1 - err[cp].mean()) if cp.any() else None
        ent["pooled_acc_uncertified"] = float(1 - err[~cp].mean()) if (~cp).any() else None
        dig[f"shift{dx}"] = ent
    summary["digits"] = dig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    shifts = [0, 1, 2]; w = 0.15
    for j, (k, lab) in enumerate([("acc", "network acc"), ("L0_hull_nearest_acc", "hull-nearest, input"),
                                  ("L1_hull_nearest_acc", "hull-nearest, layer 1"), ("L2_hull_nearest_acc", "hull-nearest, layer 2"),
                                  ("cert_pred_frac", "certified fraction")]):
        vals = [dig[f"shift{s}"][k][0] for s in shifts]; err_ = [dig[f"shift{s}"][k][1] for s in shifts]
        axes[0].bar(np.arange(3) + (j - 2) * w, vals, w, yerr=err_, label=lab)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(["no shift", "shift 1px", "shift 2px"])
    axes[0].set_ylabel("fraction of test set"); axes[0].legend(fontsize=7); axes[0].set_title("8x8 digits, 5 seeds")
    sc = ["auroc_neg_maxprob", "auroc_nn_dist", "auroc_hull_dist", "auroc_hull_contrast"]
    labs = ["-max softmax", "1-NN dist (L2)", "hull dist (L2)", "hull contrast (L2)"]
    for j, (k, lab) in enumerate(zip(sc, labs)):
        vals = [d["pooled_auroc"].get(f"shift{s}", {}).get(k, np.nan) for s in shifts]
        axes[1].bar(np.arange(3) + (j - 1.5) * 0.2, vals, 0.2, label=lab)
    axes[1].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["no shift", "shift 1px", "shift 2px"])
    axes[1].set_ylabel("AUROC for detecting errors"); axes[1].legend(fontsize=7); axes[1].set_title("error detection (pooled over seeds)")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig3_digits.png"), dpi=150); plt.close()

json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
