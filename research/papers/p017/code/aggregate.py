"""Aggregate raw_*.json into tables (results/summary.md, results/summary.json) and figures."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODES = ["unique", "balanced2", "landmark"]
MODE_LABEL = {"unique": "Unique (1-local)", "balanced2": "Two-colour", "landmark": "3 landmarks"}
MODELS = ["gru_trained", "gru_random", "esn_l0.9_h64", "esn_l0.99_h512", "tf_trained", "tf_random"]
LABEL = {"gru_trained": "GRU (trained)", "gru_random": "GRU (random)", "esn_l0.9_h64": "ESN λ=0.9, d=64",
         "esn_l0.99_h512": "ESN λ=0.99, d=512", "tf_trained": "Transformer (trained)",
         "tf_random": "Transformer (random)"}
COL = {"gru_trained": "#1f5fa8", "gru_random": "#8fb3e0", "esn_l0.9_h64": "#d9822b",
       "esn_l0.99_h512": "#a34d00", "tf_trained": "#2f8f4e", "tf_random": "#9fd4ae"}
EDITS = ["none", "random_subspace", "probe_subspace", "full_swap"]


def load():
    data = {m: [] for m in MODES}
    for f in sorted(glob.glob(os.path.join(RES, "raw_*.json"))):
        d = json.load(open(f))
        data[d["mode"]].append(d)
    return data


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), float(np.std(xs)), len(xs)


def fmt(m, s, n, p=3):
    if m is None:
        return "–"
    return f"{m:.{p}f} ± {s:.{p}f}" if n > 1 else f"{m:.{p}f}"


def capd(t):
    m, s, n = t
    return (None if m is None else min(m, 1.0)), s, n


def dynamics(lines):
    files = sorted(glob.glob(os.path.join(RES, "dyn_*.json")))
    if not files:
        return
    lines.append("\n## Training dynamics (GRU, 40 epochs, constant lr 3e-3)\n")
    lines.append("| world | seed | epoch | next-obs acc | landmark recall | probe acc | k_eff | CRA probe | CRA random | CRA full |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for f in files:
        d = json.load(open(f))
        ep = [c["epoch"] for c in d["checkpoints"]]
        ls = "-" if d["mode"] == "landmark" else "--"
        lab = f"{d['mode']} s{d['seed']}"
        axes[0].plot(ep, [c["probe_acc"] for c in d["checkpoints"]], ls, marker="o", ms=3, label=lab)
        axes[1].plot(ep, [c["cra_probe"] - c["cra_random"] for c in d["checkpoints"]], ls, marker="o", ms=3, label=lab)
        for c in d["checkpoints"]:
            lines.append(f"| {d['mode']} | {d['seed']} | {c['epoch']} | {c['obs_acc']:.3f} | {c.get('landmark_recall', float('nan')):.3f} | "
                         f"{c['probe_acc']:.3f} | {'∞' if c['k_eff'] is None else c['k_eff']} | {c['cra_probe']:.3f} | {c['cra_random']:.3f} | {c['cra_full']:.3f} |")
    axes[0].set_ylabel("state probe accuracy"); axes[1].set_ylabel("CRA(probe) − CRA(random)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.set_xscale("log"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig4_dynamics.png"), dpi=160); plt.close()


def main():
    data = load()
    summary = {}
    lines = ["# Summary (mean ± sd over seeds; n = number of seeds)\n"]
    for mode in MODES:
        runs = data[mode]
        if not runs:
            continue
        curve = np.mean([r["suffix_curve"] for r in runs], 0)
        summary[mode] = {"suffix_curve": curve.tolist(), "models": {}}
        lines.append(f"\n## World: {mode}  (seeds: {len(runs)})\n")
        lines.append("Acc*_k (k-suffix Bayes ceiling, t∈[12,24]): " +
                     ", ".join(f"k={k}: {curve[k]:.3f}" for k in [1, 2, 4, 8, 12, 16, 24]) + "\n")
        lines.append("| Encoder | n | next-obs acc | probe acc | k_eff | shuffled-label control | CF agreement (probe-subspace edit) | CF (random-subspace) | CF (no edit) | CF (full swap) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            rs = [r["models"][m] for r in runs if m in r["models"]]
            if not rs:
                continue
            row = {}
            for key in ["obs_acc", "probe_acc", "k_eff", "probe_control_acc", "landmark_recall"]:
                row[key] = ms([r.get(key) for r in rs])
            for e in EDITS:
                row["cf_" + e] = ms([r["rollout"][e]["cf_disc"] for r in rs if "rollout" in r])
            summary[mode]["models"][m] = row
            lines.append(f"| {LABEL[m]} | {len(rs)} | {fmt(*row['obs_acc'])} | {fmt(*row['probe_acc'])} | "
                         f"{fmt(*row['k_eff'], p=1)} | {fmt(*row['probe_control_acc'])} | {fmt(*row['cf_probe_subspace'])} | "
                         f"{fmt(*row['cf_random_subspace'])} | {fmt(*row['cf_none'])} | {fmt(*row['cf_full_swap'])} |")
        lines.append("\nPer-seed values (probe acc / k_eff, ∞ = above every window's ceiling / CRA probe-subspace):")
        for m in MODELS:
            rs = [(r["seed"], r["models"][m]) for r in runs if m in r["models"]]
            if not rs:
                continue
            lines.append(f"- {LABEL[m]}: " + "; ".join(
                f"s{sd}: {x['probe_acc']:.3f} / {('∞' if x['k_eff'] is None else x['k_eff'])}"
                + (f" / {x['rollout']['probe_subspace']['cf_disc']:.3f}" if 'rollout' in x else "")
                + f" (obs {x['obs_acc']:.3f})" for sd, x in rs))
        if mode == "landmark":
            lines.append("\nLandmark-hit recall: " + "; ".join(
                f"{LABEL[m]} {fmt(*summary[mode]['models'][m]['landmark_recall'])}"
                for m in MODELS if m in summary[mode]["models"]))
        # truncation table
        lines.append("\nTruncation test (t∈{16,20,24}): probe accuracy on h_t^(k) (encoder restarted k steps back), "
                     "full probe accuracy, Acc*_k, data-dependent bound Acc*_k + P(margin ≤ L·δ), a-priori ESN bound.\n")
        lines.append("| Encoder | k | trunc acc | full acc | Acc*_k | bound (data) | bound (a priori) | mean δ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in MODELS:
            rs = [r["models"][m] for r in runs if m in r["models"] and "trunc" in r["models"][m]]
            if not rs:
                continue
            for k in rs[0]["trunc"].keys():
                g = lambda key: ms([r["trunc"][k][key] for r in rs])
                lines.append(f"| {LABEL[m]} | {k} | {fmt(*g('trunc_probe_acc'))} | {fmt(*g('full_probe_acc'))} | "
                             f"{fmt(*g('acc_star_k'))} | {fmt(*capd(g('bound_data')))} | {fmt(*capd(g('bound_apriori')))} | {fmt(*g('mean_delta'))} |")
    dynamics(lines)
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

    # ---------- Figure 1: suffix ceilings with probe accuracies ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        if mode not in summary:
            continue
        c = summary[mode]["suffix_curve"]
        ax.plot(range(len(c)), c, color="black", lw=2, label="Acc*$_k$ (suffix Bayes ceiling)")
        for m in MODELS:
            if m in summary[mode]["models"]:
                pa = summary[mode]["models"][m]["probe_acc"][0]
                ax.axhline(pa, color=COL[m], lw=1.5, ls="--", label=LABEL[m])
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("suffix length k (steps)")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("state (position) accuracy")
    axes[2].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig1_suffix_ceiling.png"), dpi=160)
    plt.close()

    # ---------- Figure 2: truncation curves ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        runs = data[mode]
        if not runs:
            continue
        ks = None
        for m in ["gru_trained", "gru_random", "esn_l0.9_h64", "esn_l0.99_h512"]:
            rs = [r["models"][m] for r in runs if m in r["models"]]
            ks = [int(k) for k in rs[0]["trunc"].keys()]
            tr = np.mean([[r["trunc"][str(k)]["trunc_probe_acc"] for k in ks] for r in rs], 0)
            ax.plot(ks, tr, "o-", color=COL[m], label=LABEL[m] + " (truncated)", ms=3)
        cs = np.mean([[runs[0]["models"]["gru_trained"]["trunc"][str(k)]["acc_star_k"] for k in ks]], 0)
        ax.plot(ks, cs, color="black", lw=2, label="Acc*$_k$")
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("restart depth k (steps)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("probe accuracy on $h_t^{(k)}$")
    axes[2].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig2_truncation.png"), dpi=160)
    plt.close()

    # ---------- Figure 3: counterfactual rollout ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    rec = ["gru_trained", "gru_random", "esn_l0.9_h64", "esn_l0.99_h512"]
    w = 0.2
    for ax, mode in zip(axes, MODES):
        if mode not in summary:
            continue
        for i, e in enumerate(EDITS):
            vals = [summary[mode]["models"][m]["cf_" + e][0] for m in rec]
            errs = [summary[mode]["models"][m]["cf_" + e][1] for m in rec]
            ax.bar(np.arange(len(rec)) + (i - 1.5) * w, vals, w, yerr=errs, label=e.replace("_", " "),
                   color=["#cccccc", "#999999", "#1f5fa8", "#222222"][i])
        ax.set_xticks(range(len(rec)))
        ax.set_xticklabels([LABEL[m].replace(", ", "\n") for m in rec], fontsize=7)
        ax.set_title(MODE_LABEL[mode])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("counterfactual agreement\n(discriminating steps)")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig3_rollout.png"), dpi=160)
    plt.close()
    print("\n".join(lines))


if __name__ == "__main__":
    main()


def bound_tightness():
    """inf_k of the data-dependent Theorem-1 bound vs. actual probe accuracy (t in {16,20,24})."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "raw_*.json"))):
        d = json.load(open(f))
        for m, r in d["models"].items():
            if "trunc" not in r:
                continue
            tr = r["trunc"]
            best_k = min(tr, key=lambda k: tr[k]["bound_data"])
            rows.append({"mode": d["mode"], "seed": d["seed"], "model": m,
                         "probe_acc_trunc_t": tr[best_k]["full_probe_acc"],
                         "inf_bound_data": min(1.0, tr[best_k]["bound_data"]), "argmin_k": int(best_k),
                         "inf_bound_apriori": min([1.0] + [v["bound_apriori"] for v in tr.values() if v["bound_apriori"] is not None]),
                         "max_violation_trunc_minus_accstar": max(v["trunc_probe_acc"] - v["acc_star_k"] for v in tr.values())})
    json.dump(rows, open(os.path.join(RES, "bound_tightness.json"), "w"), indent=1)
    lines = ["\n## Theorem-1 bound tightness (inf over k of data-dependent bound)\n",
             "| world | model | probe acc (mean) | inf_k bound (mean) | gap | max(trunc acc − Acc*_k) over all k, seeds |", "|---|---|---|---|---|---|"]
    for mode in MODES:
        for m in MODELS:
            rs = [r for r in rows if r["mode"] == mode and r["model"] == m]
            if not rs:
                continue
            pa = np.mean([r["probe_acc_trunc_t"] for r in rs]); b = np.mean([r["inf_bound_data"] for r in rs])
            lines.append(f"| {mode} | {LABEL[m]} | {pa:.3f} | {b:.3f} | {b - pa:.3f} | {max(r['max_violation_trunc_minus_accstar'] for r in rs):+.3f} |")
    open(os.path.join(RES, "summary.md"), "a").write("\n".join(lines) + "\n")
    print("\n".join(lines))


bound_tightness()


def cra_vs_probe():
    """Trained-GRU evaluations in non-local worlds (main runs + training checkpoints)."""
    pts = []
    for f in glob.glob(os.path.join(RES, "raw_*.json")):
        d = json.load(open(f))
        if d["mode"] == "unique":
            continue
        r = d["models"]["gru_trained"]
        pts.append({"src": os.path.basename(f), "probe_acc": r["probe_acc"], "k_eff": r["k_eff"],
                    "cra_gain": r["rollout"]["probe_subspace"]["cf_disc"] - r["rollout"]["random_subspace"]["cf_disc"]})
    for f in glob.glob(os.path.join(RES, "dyn_*.json")):
        d = json.load(open(f))
        for c in d["checkpoints"]:
            pts.append({"src": os.path.basename(f) + f"@ep{c['epoch']}", "probe_acc": c["probe_acc"], "k_eff": c["k_eff"],
                        "cra_gain": c["cra_probe"] - c["cra_random"]})
    a = np.array([[p["probe_acc"], p["cra_gain"]] for p in pts])
    out = {"n": len(pts), "pearson_r": float(np.corrcoef(a.T)[0, 1]),
           "n_probe_lt_0.4": int((a[:, 0] < 0.4).sum()), "max_gain_probe_lt_0.4": float(a[a[:, 0] < 0.4, 1].max()),
           "n_probe_ge_0.75": int((a[:, 0] >= 0.75).sum()), "min_gain_probe_ge_0.75": float(a[a[:, 0] >= 0.75, 1].min()),
           "points": pts}
    json.dump(out, open(os.path.join(RES, "cra_vs_probe.json"), "w"), indent=1)
    plt.figure(figsize=(4.5, 3.5))
    plt.scatter(a[:, 0], a[:, 1], s=14, color="#1f5fa8")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("state probe accuracy"); plt.ylabel("CRA(probe) − CRA(random)")
    plt.title(f"trained GRU, non-local worlds (n={len(pts)}, r={out['pearson_r']:.2f})", fontsize=9)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(RES, "fig5_cra_vs_probe.png"), dpi=160); plt.close()
    print({k: v for k, v in out.items() if k != "points"})


cra_vs_probe()
