"""Exp 3: Simulation study of 'slowdown' detection.
Ground truth: the refit Chinchilla law (exp1), optionally with a genuine break delta in the reducible
loss beyond the geometric midpoint Cb of the design:  L = E + (A N^-a + B D^-b) * max(1,C/Cb)^delta.
Design: K iso-FLOP budgets log-uniform over W decades starting at 1e18 FLOP, M model sizes per budget
with tokens/param log-uniform in [2, 200]; Gaussian noise sigma on log-loss.
Tests compared (one-sided, nominal 5%):
  NAIVE: quadratic fit of log L*_k vs log C_k on the per-budget minimum losses; 'slowdown' if the
         quadratic coefficient is positive with p<0.05 (raw log-log curve flattening).
  BREAK: likelihood-ratio test of delta=0 vs delta>0 in the nested parametric model (Cb fixed at the
         design midpoint, i.e. an optimistic, known-breakpoint version).
Also records extrapolation error of the no-break fit 2 decades beyond the design."""
import json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

t0 = time.time()
rng = np.random.default_rng(2)
P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
SIG0 = json.load(open(f"{RES}/exp1_fit.json"))["residual_sd_log"]
ptrue = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])
BIG = 50.0  # Huber threshold so large that the objective is plain least squares


def obj_brk(q, lN, lD, lC, lL, lCb):
    a, b, e, al, be, dl = q
    h = np.maximum(0.0, lC - lCb)
    T = np.stack([a - al * lN + dl * h, b - be * lD + dl * h, np.full_like(lN, e)])
    lp = logsumexp(T, axis=0); w = np.exp(T - lp); r = lp - lL
    g = np.array([(r * w[0]).sum(), (r * w[1]).sum(), (r * w[2]).sum(),
                  -(r * w[0] * lN).sum(), -(r * w[1] * lD).sum(), (r * (w[0] + w[1]) * h).sum()])
    return 0.5 * (r ** 2).sum(), g


def design(W, K=12, M=20):
    lC = np.repeat(np.linspace(np.log(1e18), np.log(1e18) + W * np.log(10), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(2), np.log(200), M)), K)  # D/N
    # C = 6 N D = 6 N^2 ratio
    lN = 0.5 * (lC - np.log(6) - np.log(ratio))
    lD = lC - np.log(6) - lN
    return lN, lD, lC


def simulate(lN, lD, lC, delta, sigma, lCb):
    a, b, e, al, be = ptrue
    h = np.maximum(0, lC - lCb)
    lp = logsumexp(np.stack([a - al * lN + delta * h, b - be * lD + delta * h, np.full_like(lN, e)]), axis=0)
    return lp + sigma * rng.standard_normal(lp.shape)


def run_one(W, delta, sigma, K=12, M=20):
    lN, lD, lC = design(W, K, M)
    lCb = 0.5 * (lC.min() + lC.max())
    lL = simulate(lN, lD, lC, delta, sigma, lCb)
    n = len(lL)
    # base fit (least squares)
    f0 = min((minimize(obj_base, x0, args=(lN, lD, lL, BIG), method="BFGS", jac=True)
              for x0 in [ptrue, ptrue + np.array([0.5, -0.5, 0.05, 0.02, -0.02])]), key=lambda r: r.fun)
    f1 = min((minimize(obj_brk, np.r_[f0.x, d0], args=(lN, lD, lC, lL, lCb), method="BFGS", jac=True)
              for d0 in [0.0, 0.05]), key=lambda r: r.fun)
    rss0, rss1 = 2 * f0.fun, 2 * f1.fun
    lr = n * np.log(rss0 / max(rss1, 1e-300))
    p_break = 0.5 * stats.chi2.sf(lr, 1) if f1.x[5] > 0 else 1.0  # one-sided
    # naive test on per-budget minima
    lCk = lC.reshape(K, M)[:, 0]; lLk = lL.reshape(K, M).min(1)
    x = (lCk - lCk.mean()) / np.log(10)
    X = np.stack([np.ones_like(x), x, x ** 2], 1)
    beta_, res_, *_ = np.linalg.lstsq(X, lLk, rcond=None)
    s2 = ((lLk - X @ beta_) ** 2).sum() / (K - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
    p_naive = stats.t.sf(beta_[2] / se, K - 3)
    # extrapolation of no-break fit to 2 decades above the design max, along the true frontier
    Cx = np.exp(lC.max()) * 100
    q = dict(A=np.exp(f0.x[0]), B=np.exp(f0.x[1]), E=np.exp(f0.x[2]), alpha=f0.x[3], beta=f0.x[4])
    Lhat = frontier(q, np.array([Cx]))[0][0]
    Ltrue_nobreak = frontier(P, np.array([Cx]))[0][0]
    return p_break, p_naive, f1.x[5], float(np.log(Lhat / Ltrue_nobreak)), q["E"], gamma(q)


if __name__ == "__main__":
    Ws = [2, 3, 4, 6]
    deltas = [0.0, 0.01, 0.02, 0.04, 0.08]
    sigmas = [SIG0, 2 * SIG0]
    R = 100
    results = []
    for sigma in sigmas:
        for W in Ws:
            for d in deltas:
                out = np.array([run_one(W, d, sigma) for _ in range(R)])
                row = dict(sigma=float(sigma), W=W, delta=d, reps=R,
                           power_break=float((out[:, 0] < 0.05).mean()),
                           power_naive=float((out[:, 1] < 0.05).mean()),
                           delta_hat_mean=float(out[:, 2].mean()), delta_hat_sd=float(out[:, 2].std()))
                if d == 0.0:
                    row.update(extrap_logerr_sd=float(out[:, 3].std()), extrap_logerr_mean=float(out[:, 3].mean()),
                               E_hat_p10_p90=[float(np.percentile(out[:, 4], 10)), float(np.percentile(out[:, 4], 90))],
                               gamma_hat_p10_p90=[float(np.percentile(out[:, 5], 10)), float(np.percentile(out[:, 5], 90))])
                results.append(row)
                print(row, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(gamma_true=gamma(P), sigma0=SIG0, results=results), open(f"{RES}/exp3_simulation.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for j, sigma in enumerate(sigmas):
        for i, W in enumerate(Ws):
            rs = [r for r in results if r["sigma"] == sigma and r["W"] == W]
            ax[j].plot([r["delta"] for r in rs], [r["power_break"] for r in rs], "-o", color=f"C{i}", label=f"break LR, W={W} dec")
            ax[j].plot([r["delta"] for r in rs], [r["power_naive"] for r in rs], "--x", color=f"C{i}", alpha=.7, label=f"naive curvature, W={W}")
        ax[j].axhline(0.05, color="k", lw=.8, ls=":")
        ax[j].set_xlabel(r"true exponent loss $\delta$ (reducible-loss exponent $\gamma$=%.3f)" % gamma(P))
        ax[j].set_title(f"noise sd = {sigma:.4f} (log-loss)", fontsize=9)
    ax[0].set_ylabel("rejection rate (one-sided, 5%)"); ax[1].legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(f"{RES}/fig3_power.png", dpi=150)
