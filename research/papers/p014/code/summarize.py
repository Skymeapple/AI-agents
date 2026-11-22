"""Aggregate results/*.json into a markdown table and figures."""
import glob, json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": INK, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})

runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(R, "*_s*.json"))):
    d = json.load(open(f))
    runs[d["name"]].append(d)


def agg(name, path):
    vals = []
    for d in runs.get(name, []):
        x = d
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if x is not None:
            vals.append(x)
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(name, path, pct=True):
    m, s, k = agg(name, path)
    if m is None:
        return "-"
    if pct:
        return f"{100*m:.1f}" + (f" ± {100*s:.1f}" if k > 1 else "")
    return f"{m:.3f}"


order = ["obs_p0", "obs_p05", "obs_p09", "obs_p1", "laplace_p0", "graph_p0",
         "int8_p0", "int32_p0", "int128_p0"]
lines = ["| Regime | seeds | i.i.d. acc | i.i.d. acc (MEC-identified) | i.i.d. acc (unidentified) | "
         "Twin acc (matched) | LR ceiling | Causal surplus |",
         "|---|---|---|---|---|---|---|---|"]
for nm in order:
    if nm not in runs:
        continue
    lines.append(f"| {nm} | {len(runs[nm])} | {fmt(nm, ['iid','iid_acc'])} | "
                 f"{fmt(nm, ['iid','iid_acc_identified'])} | {fmt(nm, ['iid','iid_acc_unidentified'])} | "
                 f"{fmt(nm, ['twins_matched','twin_acc'])} | {fmt(nm, ['twins_matched','oracle_ceiling'])} | "
                 f"{fmt(nm, ['twins_matched','causal_surplus'], pct=False)} |")
lines += ["", "Cross-regime probes (twin accuracy, %):", "",
          "| Model | probe | twin acc | LR ceiling | extra |", "|---|---|---|---|---|"]
for nm in order:
    for d in runs.get(nm, [])[:1]:
        pass
    for probe in ["twins_laplace", "twins_gauss", "twins_corrupt25", "twins_conflict"]:
        m, s, k = agg(nm, [probe, "twin_acc"])
        if m is None:
            continue
        extra = ""
        if probe == "twins_conflict":
            extra = "follows description: " + fmt(nm, [probe, "follows_description"])
        lines.append(f"| {nm} | {probe} | {fmt(nm, [probe,'twin_acc'])} | "
                     f"{fmt(nm, [probe,'oracle_ceiling'])} | {extra} |")
lines += ["", "Oracle reference accuracies on the i.i.d. test set (%):", "",
          "| Regime | frac. identified | MEC-uniform oracle | MEC+label-order oracle |", "|---|---|---|---|"]
for nm in ["obs_p0", "obs_p05", "obs_p09", "obs_p1"]:
    if nm in runs:
        lines.append(f"| {nm} | {fmt(nm, ['iid','frac_identified'])} | "
                     f"{fmt(nm, ['iid','oracle_mec_uniform_acc'])} | {fmt(nm, ['iid','oracle_mec_labelorder_acc'])} |")
lines += ["", "Within-pair statistics (matched twins):", "",
          "| Regime | mean P(yes) on M | mean P(yes) on M' | mean abs within-pair diff | both-correct pairs |",
          "|---|---|---|---|---|"]
for nm in order:
    if nm in runs:
        lines.append(f"| {nm} | {fmt(nm, ['twins_matched','mean_p_yes_on_A'], False)} | "
                     f"{fmt(nm, ['twins_matched','mean_p_yes_on_B'], False)} | "
                     f"{fmt(nm, ['twins_matched','mean_abs_p_diff_within_pair'], False)} | "
                     f"{fmt(nm, ['twins_matched','pair_both_correct'])} |")
open(os.path.join(R, "summary_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- Figure 1
ps = [("obs_p0", 0.0), ("obs_p05", 0.5), ("obs_p09", 0.9), ("obs_p1", 1.0)]
ps = [(nm, p) for nm, p in ps if nm in runs]
if ps:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(ps)); w = 0.36
    for off, key, col, lab in [(-w/2, ["iid", "iid_acc"], BLUE, "i.i.d. test accuracy"),
                               (w/2, ["twins_matched", "twin_acc"], ORANGE, "twin accuracy")]:
        ms = [agg(nm, key)[0] for nm, _ in ps]; ss = [agg(nm, key)[1] for nm, _ in ps]
        ax.bar(x + off, ms, w - 0.03, yerr=ss, color=col, label=lab, capsize=3, error_kw={"lw": 1})
        for xi, m in zip(x + off, ms):
            ax.text(xi, m + 0.02, f"{100*m:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    orc = [agg(nm, ["iid", "oracle_mec_labelorder_acc"])[0] for nm, _ in ps]
    ax.plot(x - w/2, orc, "o--", color=GRAY, ms=5, lw=1.2, label="MEC + label-order oracle (i.i.d.)")
    ax.axhline(0.5, color=INK, lw=1, ls=":")
    ax.text(len(ps) - 0.5, 0.505, "twin bound 1/2", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"p = {p}" for _, p in ps])
    ax.set_xlabel("probability that variable labels follow causal order (training & i.i.d. test)")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_iid_vs_twin.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
ints = [("obs_p0", 0), ("int8_p0", 8), ("int32_p0", 32), ("int128_p0", 128)]
ints = [(nm, k) for nm, k in ints if nm in runs]
if ints:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ks = [k for _, k in ints]
    xs = np.arange(len(ks))
    m = [agg(nm, ["twins_matched", "twin_acc"])[0] for nm, _ in ints]
    s = [agg(nm, ["twins_matched", "twin_acc"])[1] for nm, _ in ints]
    c = [agg(nm, ["twins_matched", "oracle_ceiling"])[0] for nm, _ in ints]
    ax.plot(xs, c, "s--", color=GRAY, lw=2, ms=8, label="likelihood-ratio ceiling (1+TV)/2")
    ax.errorbar(xs, m, yerr=s, fmt="o-", color=BLUE, lw=2, ms=8, capsize=3, label="learned model")
    ax.axhline(0.5, color=INK, lw=1, ls=":")
    ax.set_xticks(xs, [f"{k}" for k in ks])
    ax.set_xlabel("interventional samples per dataset (of 256; random single target)")
    ax.set_ylabel("twin accuracy"); ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_interventions.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
bars = [("obs (Gauss) on Gauss twins", "obs_p0", "twins_matched"),
        ("obs (Gauss) on Laplace twins", "obs_p0", "twins_laplace"),
        ("obs (Laplace) on Laplace twins", "laplace_p0", "twins_matched"),
        ("obs (Laplace) on Gauss twins", "laplace_p0", "twins_gauss"),
        ("graph text, clean", "graph_p0", "twins_matched"),
        ("graph text, 25% edges reversed", "graph_p0", "twins_corrupt25"),
        ("graph text of the other twin", "graph_p0", "twins_conflict")]
bars = [b for b in bars if agg(b[1], [b[2], "twin_acc"])[0] is not None]
if bars:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(bars))[::-1]
    m = [agg(nm, [pr, "twin_acc"])[0] for _, nm, pr in bars]
    s = [agg(nm, [pr, "twin_acc"])[1] for _, nm, pr in bars]
    c = [agg(nm, [pr, "oracle_ceiling"])[0] for _, nm, pr in bars]
    ax.barh(y, m, 0.6, xerr=s, color=BLUE, capsize=3, label="learned model")
    ax.scatter(c, y, marker="|", s=250, color=ORANGE, linewidths=2.5, zorder=3,
               label="LR ceiling (data only)")
    for yi, mi in zip(y, m):
        ax.text(mi + 0.02, yi, f"{100*mi:.0f}", va="center", fontsize=8)
    ax.axvline(0.5, color=INK, lw=1, ls=":")
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 1.1); ax.set_xlabel("twin accuracy")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_regimes.png"), dpi=200); plt.close(fig)
print("figures written")
