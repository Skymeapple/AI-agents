"""Experiment 0 (fixed-lambda baseline): learned schemers vs. the Neyman-Pearson frontier (Theorem 1).

For each penalty lambda we train a policy to maximise deployment defection minus
lambda x eval defection, then plot its (alpha, beta) against the Bayes ROC of the
eval-vs-deployment discrimination problem. Two environments:
  (a) GaussianShift: linear cue, d' = 1.5
  (b) XORCue: nonlinear cue; capacity (hidden width) limits the reachable frontier.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (GaussianShift, XORCue, train_schemer, evaluate, empirical_roc,
                    RESULTS)

LAMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SEEDS = [0, 1, 2]


def run_env(env, widths, name, rng_master):
    out = {}
    for h in widths:
        pts = []
        for lam in LAMS:
            for s in SEEDS:
                rng = np.random.default_rng(1000 * s + lam + 7 * h)
                m, _ = train_schemer(env, h, lam, rng, steps=1500, lr=2e-2)
                a, b = evaluate(m, env, np.random.default_rng(99 + s))
                pts.append({"lam": lam, "seed": s, "alpha": a, "beta": b})
        out[str(h)] = pts
        print(name, "width", h, "done", flush=True)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}

    # (a) Gaussian
    g = GaussianShift(k=8, dprime=1.5, seed=0)
    res["gaussian"] = run_env(g, [0, 16], "gaussian", rng)
    # gap to frontier
    gaps = []
    for h, pts in res["gaussian"].items():
        for p in pts:
            if 1e-4 < p["alpha"] < 0.999:
                gaps.append(g.bayes_roc(p["alpha"]) - p["beta"])
    res["gaussian_gap"] = {"mean": float(np.mean(gaps)), "max": float(np.max(gaps)),
                           "min": float(np.min(gaps)), "n": len(gaps)}

    # (b) XOR
    x = XORCue(k=8, a=1.2)
    res["xor"] = run_env(x, [0, 2, 8, 32], "xor", rng)
    r2 = np.random.default_rng(5)
    se = x.log_lr(x.sample(400000, "E", r2))
    sd = x.log_lr(x.sample(400000, "D", r2))
    alphas = np.logspace(-4, 0, 60)
    bayes_xor = empirical_roc(se, sd, alphas)
    # AUC of Bayes discriminator (Mann-Whitney on subsample)
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(sd[:50000], se[:50000]).statistic / (50000.0 ** 2)
    res["xor_bayes_auc"] = float(u)
    res["xor_bayes_roc"] = {"alpha": alphas.tolist(), "beta": bayes_xor.tolist()}

    # summary: beta at lam=64 by width (mean over seeds)
    summ = {}
    for h, pts in res["xor"].items():
        for lam in [8, 64, 512]:
            sel = [p for p in pts if p["lam"] == lam]
            summ[f"w{h}_lam{lam}"] = {"alpha": float(np.mean([p["alpha"] for p in sel])),
                                       "beta": float(np.mean([p["beta"] for p in sel]))}
    res["xor_summary"] = summ
    with open(os.path.join(RESULTS, "exp0_fixed_lambda.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    al = np.logspace(-5, 0, 200)
    ax[0].plot(al, g.bayes_roc(al), "k-", lw=1.5, label="Bayes frontier (Thm 1), d'=1.5")
    mk = {"0": ("o", "#1f77b4", "linear policy"), "16": ("s", "#d62728", "MLP, 16 hidden")}
    for h, pts in res["gaussian"].items():
        m_, c_, l_ = mk[h]
        ax[0].scatter([p["alpha"] for p in pts], [p["beta"] for p in pts], marker=m_,
                      s=18, color=c_, alpha=0.7, label=l_)
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"eval defection rate $\alpha$")
    ax[0].set_ylabel(r"deployment defection rate $\beta$"); ax[0].set_xlim(1e-5, 1)
    ax[0].set_title("(a) Linear cue: learned schemers sit on the frontier")
    ax[0].legend(fontsize=8)
    ax[1].plot(alphas, bayes_xor, "k-", lw=1.5, label=f"Bayes frontier (AUC={u:.2f})")
    cols = {"0": "#1f77b4", "2": "#ff7f0e", "8": "#2ca02c", "32": "#d62728"}
    for h, pts in res["xor"].items():
        ax[1].scatter([p["alpha"] for p in pts], [p["beta"] for p in pts], s=16,
                      color=cols[h], alpha=0.7, label=f"width {h}" if h != "0" else "linear")
    ax[1].plot(al, al, ":", color="gray", lw=1, label="chance (beta = alpha)")
    ax[1].set_xscale("log"); ax[1].set_xlim(1e-5, 1)
    ax[1].set_xlabel(r"eval defection rate $\alpha$")
    ax[1].set_title("(b) Nonlinear cue: capacity bounds the frontier")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig0_fixed_lambda.png"), dpi=150)
    print(json.dumps({"gaussian_gap": res["gaussian_gap"], "xor_auc": u,
                      "xor_summary": summ}, indent=1))


if __name__ == "__main__":
    main()
