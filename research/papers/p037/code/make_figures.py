"""Make all figures from saved CSVs in results/. Run after exp1-exp4."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS

C = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID = "#1f1f1f", "#6b6b66", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "axes.spines.top": False, "axes.spines.right": False})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def fig1():
    rows = read("exp3b_vanishing.csv")
    pops = list(dict.fromkeys(r["population"] for r in rows))
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for ax, pop in zip(axs, pops):
        for i, m in enumerate(["2", "4", "8"]):
            rr = [r for r in rows if r["population"] == pop and r["m"] == m]
            K = [int(r["K"]) for r in rr]
            ax.plot(K, [float(r["U"]) for r in rr], color=C[i], label=f"upper bound $U_K$, m={m}")
            ax.plot(K, [float(r["L"]) for r in rr], color=C[i], ls=":", lw=1.5)
        rr = [r for r in rows if r["population"] == pop and r["m"] == "4"]
        ax.plot([int(r["K"]) for r in rr], [float(r["truth"]) for r in rr], color=INK, ls="--", lw=1.2,
                label="true pass@K")
        ax.plot([], [], color=MUTED, ls=":", lw=1.5, label="lower bound $L_K$ (same colours)")
        ax.set_xscale("log")
        ax.set_title(pop, fontsize=9)
        ax.set_xlabel("attacker attempts K")
    axs[0].set_ylabel("pass@K (identified set)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig1_identified_sets.png"), dpi=180)


def fig2():
    rows = read("exp2_extrapolation.csv")
    models = ["GPT-4o (Inspect)", "Claude 3.7 Sonnet (Inspect)", "o3 (Inspect)", "GPT-5.2"]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, mo in zip(axs, models):
        rr = [r for r in rows if r["model"] == mo]
        K = np.array([int(r["K"]) for r in rr])
        ax.fill_between(K, [float(r["lp_ci_L"]) for r in rr], [float(r["lp_ci_U"]) for r in rr],
                        color=C[0], alpha=0.18, lw=0, label="90% identified-set CI")
        ax.plot(K, [float(r["lp_ci_U"]) for r in rr], color=C[0], label="upper confidence bound")
        ax.plot(K, [float(r["beta_binomial"]) for r in rr], color=C[1], label="beta-binomial extrapolation")
        ax.set_xscale("log")
        ax.set_title(mo.replace(" (Inspect)", ""), fontsize=9)
        ax.set_xlabel("attempts K")
    axs[0].set_ylabel("suite pass@K (m=4 runs/task)")
    axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig2_metr_extrapolation.png"), dpi=180)


def fig3():
    rows = [r for r in read("exp3a_required_attempts.csv") if r["alpha"] == "0.05"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, tau in enumerate(["0.01", "0.1", "0.5"]):
        rr = [r for r in rows if r["tau"] == tau]
        ax.plot([int(r["K"]) for r in rr], [int(r["m_required"]) for r in rr], color=C[i], marker="o",
                ms=4, label=f"rule out pass@K > {tau}")
    # rev. 1: randomized tests with power beta < 1 at p = 0 (Prop. 3b, general form)
    pw = [r for r in read("exp3a_power_requirements.csv") if r["tau"] == "0.1" and r["power_at_p0"] == "0.5"]
    if pw:
        ratio = float(pw[0]["m_over_K_exact"])
        Ks = np.array([1, 10, 100, 1000, 10000])
        ax.plot(Ks, np.ceil(ratio * Ks), color=C[1], ls="--", lw=1.3,
                label="pass@K > 0.1, power 0.5 at p=0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attacker attempts K")
    ax.set_ylabel("evaluator attempts m needed\n(level 0.05)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_required_attempts.png"), dpi=180)


def fig4():
    rows = read("exp1_clustering.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    p = [float(r["mean_success"]) for r in rows]
    ax.scatter(p, [float(r["se_naive"]) for r in rows], color=C[0], s=22, label="naive i.i.d.-runs SE")
    ax.scatter(p, [float(r["se_task"]) for r in rows], color=C[2], s=22, label="task-cluster bootstrap SE")
    ax.scatter(p, [float(r["se_family"]) for r in rows], color=C[1], s=22, label="family-cluster bootstrap SE")
    ax.set_xlabel("suite mean success (20 models)")
    ax.set_ylabel("standard error")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_clustering.png"), dpi=180)


def fig5():
    rows = read("exp4_floor.csv")
    keys = ["none", "0.001", "0.01", "0.05", "0.2", "0.3", "0.5"]
    xs = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, K in enumerate(["10", "100", "1000"]):
        rr = [r for r in rows if r["K"] == K]
        med = [np.median([float(r[f"U_floor_{k}"]) for r in rr]) for k in keys]
        ax.plot(xs, med, color=C[i], marker="os^"[i], ms=5, ls=["-", "-", "--"][i], label=f"K={K}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["no floor"] + keys[1:])
    ax.set_xlabel("declared resolution floor $p_{min}$")
    ax.set_ylabel("median upper bound on pass@K\n(20 models, m=4)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_floor.png"), dpi=180)


def fig6():
    import json
    summ = json.load(open(os.path.join(RESULTS, "exp6_summary.json")))
    structs = list(dict.fromkeys(d["structure"] for d in summ))
    fig, axs = plt.subplots(1, len(structs), figsize=(3.6 * len(structs), 3.1), sharey=True)
    meths = [("boot", "percentile cluster bootstrap", C[1]), ("hyb_nc", "hybrid box (n = families)", C[0]),
             ("hyb_kish", "hybrid box (Kish n_eff)", C[2])]
    for ax, st in zip(axs, structs):
        rr = [d for d in summ if d["structure"] == st]
        x = np.arange(len(rr))
        for j, (k, lab, col) in enumerate(meths):
            ax.bar(x + (j - 1) * 0.27, [d[f"cov_pop_{k}_K100"] for d in rr], width=0.27, color=col, label=lab)
        ax.axhline(0.9, color=INK, lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([d["scenario"].split("_")[0] for d in rr])
        ax.set_title(f"{st.replace('_', '-')}", fontsize=9)
        ax.set_xlabel("scenario")
    axs[0].set_ylabel("coverage of 90% upper bound\non population pass@100")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(RESULTS, "fig6_clustered_coverage.png"), dpi=180)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("figures written")
