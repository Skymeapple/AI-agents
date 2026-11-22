"""Experiment 4 (measurement methodology, Monte Carlo): AUC is the wrong summary of
evaluation awareness for scheming risk; the frontier at small alpha is what matters,
and it is poorly identified by test sets of the size used in the literature.

 (a) Same AUC, different tails: binormal ROCs with AUC = 0.83 and slope b in
     {0.5,...,2} give very different R(alpha) at alpha = 1e-3.
 (b) Finite-sample identifiability: simulate eval-awareness test sets of
     n_E = n_D = 450 (a 900-item set split evenly -- the split is our assumption)
     from a ground-truth binormal discriminator, then estimate R(1e-3) by
       (i) the empirical ROC (non-parametric), and
       (ii) an equal-variance binormal fit from the empirical AUC,
       (iii) an unequal-variance binormal fit (probit regression of the empirical ROC).
     Report bias and 90% intervals vs n.
Run: python3 exp4_measurement.py  (~1 min)
"""
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS
from theory import binormal, auc_to_a

A0 = 1e-3
AUC = 0.83


def emp_auc(se, sd):
    allv = np.concatenate([se, sd])
    ranks = allv.argsort().argsort() + 1
    rd = ranks[len(se):].sum()
    return (rd - len(sd) * (len(sd) + 1) / 2) / (len(se) * len(sd))


def emp_R(se, sd, a):
    k = int(np.floor(a * len(se)))
    thr = np.sort(se)[::-1][min(k, len(se) - 1)]
    return float(np.mean(sd > thr))


def uv_fit(se, sd):
    """Probit-regress empirical ROC points with FPR, TPR in [0.02, 0.98]."""
    thr = np.quantile(se, np.linspace(0.02, 0.98, 25))
    fpr = np.array([np.mean(se > t) for t in thr])
    tpr = np.array([np.mean(sd > t) for t in thr])
    ok = (tpr > 0.005) & (tpr < 0.995) & (fpr > 0.005) & (fpr < 0.995)
    b, a = np.polyfit(norm.ppf(fpr[ok]), norm.ppf(tpr[ok]), 1)
    return a, b


def main():
    rng = np.random.default_rng(2026)
    out = {}
    # (a)
    tails = {}
    for b in [0.5, 0.75, 1.0, 1.5, 2.0]:
        a = auc_to_a(AUC, b)
        tails[str(b)] = dict(a=float(a), R_1em2=float(binormal(1e-2, a, b)),
                             R_1em3=float(binormal(1e-3, a, b)), R_1em4=float(binormal(1e-4, a, b)))
    out["same_auc_tails"] = tails

    # (b)
    sims = {}
    for b_true in [0.5, 1.0, 1.5]:
        a_true = auc_to_a(AUC, b_true)
        truth = float(binormal(A0, a_true, b_true))
        for n in [450, 2000, 10000, 50000]:
            est = {"emp": [], "ev": [], "uv": []}
            for _ in range(1000 if n <= 2000 else 300):
                # scores: E ~ N(0,1), D ~ N(a/b, 1/b^2)  => ROC Phi(a + b Phi^-1(x))
                se = rng.normal(size=n)
                sd = rng.normal(a_true / b_true, 1.0 / b_true, size=n)
                est["emp"].append(emp_R(se, sd, A0))
                au = emp_auc(se, sd)
                est["ev"].append(float(binormal(A0, auc_to_a(min(au, 0.9999)), 1.0)))
                a_h, b_h = uv_fit(se, sd)
                est["uv"].append(float(binormal(A0, a_h, b_h)))
            row = {"truth": truth}
            for k, v in est.items():
                v = np.array(v)
                row[k] = dict(mean=float(v.mean()), q05=float(np.quantile(v, 0.05)),
                              q95=float(np.quantile(v, 0.95)), rmse=float(np.sqrt(np.mean((v - truth) ** 2))))
            sims[f"b{b_true}_n{n}"] = row
            print(b_true, n, json.dumps(row))
    out["finite_sample"] = sims
    json.dump(out, open(os.path.join(RESULTS, "exp4_measurement.json"), "w"), indent=1)
    print(json.dumps(tails, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.logspace(-5, 0, 300)
    for b, col in zip([0.5, 0.75, 1.0, 1.5, 2.0], ["#1f77b4", "#2ca02c", "k", "#ff7f0e", "#d62728"]):
        a = auc_to_a(AUC, b)
        ax[0].plot(x, binormal(x, a, b), color=col, label=f"slope b={b}: R(1e-3)={binormal(1e-3, a, b):.3f}")
    ax[0].axvline(A0, color="gray", ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlabel("α"); ax[0].set_ylabel("R(α)")
    ax[0].set_title(f"(a) Five ROCs, all with AUC = {AUC}"); ax[0].legend(fontsize=7)
    ns = [450, 2000, 10000, 50000]
    for k, (meth, col) in enumerate([("emp", "#1f77b4"), ("ev", "#2ca02c"), ("uv", "#d62728")]):
        for j, b_true in enumerate([0.5, 1.0, 1.5]):
            rows = [sims[f"b{b_true}_n{n}"] for n in ns]
            xs = np.array(ns) * (1 + 0.08 * (k - 1)) * (1 + 0.02 * (j - 1))
            m = [r[meth]["mean"] for r in rows]
            lo = [r[meth]["mean"] - r[meth]["q05"] for r in rows]
            hi = [r[meth]["q95"] - r[meth]["mean"] for r in rows]
            ax[1].errorbar(xs, m, yerr=[lo, hi], fmt="o-", ms=3, color=col,
                           alpha=[0.5, 1.0, 0.75][j], capsize=2,
                           label=f"{ {'emp': 'empirical', 'ev': 'eq.-var. fit', 'uv': 'uneq.-var. fit'}[meth]}, true b={b_true}")
            ax[1].axhline(rows[0]["truth"], color="gray", lw=0.6, ls="--")
    ax[1].set_xscale("log"); ax[1].set_xlabel("test-set size per class n")
    ax[1].set_ylabel("estimated R(1e-3) (dashed: truth)")
    ax[1].set_title("(b) Estimating the tail that bounds scheming"); ax[1].legend(fontsize=5.5, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig4_measurement.png"), dpi=150)


if __name__ == "__main__":
    main()
