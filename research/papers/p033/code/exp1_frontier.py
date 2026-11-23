"""Experiment 1: learned conditional defectors vs. the Neyman-Pearson frontier (Thm 1).

Policies are trained (annealed-penalty method) to maximise deployment defection
E_D[pi] minus lambda x evaluation defection E_E[pi]. We compare each trained
policy's (alpha, beta) with the Bayes ROC of the eval-vs-deployment problem.
  (a) GaussianShift, linear cue, d' = 1.5: policies of width 0 (linear) and 16.
  (b) XORCue, nonlinear cue: widths 0, 2, 8, 32 -- capacity limits the frontier.
Run: OMP_NUM_THREADS=1 python3 exp1_frontier.py   (~5 min on 4 cores)
"""
import json, os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import GaussianShift, XORCue, train_anneal, empirical_roc, RESULTS

LAMS = [1, 2, 4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2]


def job(args):
    envname, h, lam, s = args
    env = GaussianShift(8, 1.5, 0) if envname == "gaussian" else XORCue(8, 1.2)
    rng = np.random.default_rng(10007 * s + 31 * lam + h + (0 if envname == "gaussian" else 5))
    m = train_anneal(env, h, lam, rng, steps=3000, batch=4096, lr=1e-2)
    r = np.random.default_rng(424242 + s)
    n = 400000
    pe = m.prob(env.sample(n, "E", r))
    pd = m.prob(env.sample(n, "D", r))
    return dict(env=envname, width=h, lam=lam, seed=s, alpha=float(pe.mean()),
                beta=float(pd.mean()))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    jobs = [("gaussian", h, l, s) for h in [0, 16] for l in LAMS for s in SEEDS]
    jobs += [("xor", h, l, s) for h in [0, 2, 8, 32] for l in LAMS for s in SEEDS]
    with Pool(4) as p:
        pts = p.map(job, jobs)

    g = GaussianShift(8, 1.5, 0)
    x = XORCue(8, 1.2)
    r = np.random.default_rng(5)
    se = x.log_lr(x.sample(2_000_000, "E", r))
    sd = x.log_lr(x.sample(2_000_000, "D", r))
    grid = np.logspace(-5, 0, 80)
    xor_roc = empirical_roc(se, sd, grid)
    # Bayes AUC by sorting (Mann-Whitney) on a subsample
    from scipy.stats import mannwhitneyu
    k = 100000
    xor_auc = float(mannwhitneyu(sd[:k], se[:k]).statistic / k ** 2)

    def frontier(env, a):
        return float(g.bayes_roc(a)) if env == "gaussian" else float(np.interp(np.log(a), np.log(grid), xor_roc))

    for p_ in pts:
        a = min(max(p_["alpha"], 1e-5), 1.0)
        p_["frontier_beta"] = frontier(p_["env"], a)
        p_["efficiency"] = p_["beta"] / p_["frontier_beta"] if p_["frontier_beta"] > 0 else None
        p_["advantage"] = p_["beta"] - p_["alpha"]   # deployment-minus-eval defection

    summary = {}
    for env, widths in [("gaussian", [0, 16]), ("xor", [0, 2, 8, 32])]:
        for h in widths:
            sel = [p_ for p_ in pts if p_["env"] == env and p_["width"] == h and p_["alpha"] > 1e-4]
            eff = [p_["efficiency"] for p_ in sel]
            ratio = [p_["beta"] / p_["alpha"] for p_ in sel]
            summary[f"{env}_w{h}"] = dict(n=len(sel), eff_mean=float(np.mean(eff)),
                                          eff_min=float(np.min(eff)), eff_max=float(np.max(eff)),
                                          beta_over_alpha_median=float(np.median(ratio)))
            # per-lambda means
            for lam in [4, 16, 64]:
                s2 = [p_ for p_ in pts if p_["env"] == env and p_["width"] == h and p_["lam"] == lam]
                summary[f"{env}_w{h}_lam{lam}"] = dict(alpha=float(np.mean([q["alpha"] for q in s2])),
                                                       beta=float(np.mean([q["beta"] for q in s2])),
                                                       frontier=float(np.mean([q["frontier_beta"] for q in s2])))
    out = dict(points=pts, xor_bayes_auc=xor_auc, gaussian_dprime=1.5,
               xor_roc=dict(alpha=grid.tolist(), beta=xor_roc.tolist()), summary=summary)
    json.dump(out, open(os.path.join(RESULTS, "exp1_frontier.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), "\nxor AUC", xor_auc)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    al = np.logspace(-5, 0, 200)
    ax[0].plot(al, g.bayes_roc(al), "k-", lw=1.5, label="Bayes frontier R(α), d'=1.5")
    for h, mk, c, lab in [(0, "o", "#1f77b4", "linear policy"), (16, "s", "#d62728", "MLP, 16 hidden")]:
        sel = [p_ for p_ in pts if p_["env"] == "gaussian" and p_["width"] == h]
        ax[0].scatter([q["alpha"] for q in sel], [q["beta"] for q in sel], marker=mk, s=20,
                      color=c, alpha=0.7, label=lab)
    ax[0].plot(al, al, ":", color="gray", lw=1, label="no discrimination (β=α)")
    ax[0].set_xscale("log"); ax[0].set_xlim(1e-5, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("evaluation defection rate α"); ax[0].set_ylabel("deployment defection rate β")
    ax[0].set_title("(a) Linear cue"); ax[0].legend(fontsize=8)
    ax[1].plot(grid, xor_roc, "k-", lw=1.5, label=f"Bayes frontier (AUC={xor_auc:.2f})")
    cols = {0: "#1f77b4", 2: "#ff7f0e", 8: "#2ca02c", 32: "#d62728"}
    for h in [0, 2, 8, 32]:
        sel = [p_ for p_ in pts if p_["env"] == "xor" and p_["width"] == h]
        ax[1].scatter([q["alpha"] for q in sel], [q["beta"] for q in sel], s=18, color=cols[h],
                      alpha=0.7, label="linear" if h == 0 else f"width {h}")
    ax[1].plot(al, al, ":", color="gray", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xlim(1e-5, 1); ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("evaluation defection rate α")
    ax[1].set_title("(b) Nonlinear (XOR) cue: capacity limits the frontier"); ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig1_frontier.png"), dpi=150)


if __name__ == "__main__":
    main()
