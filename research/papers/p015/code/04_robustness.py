"""Robustness of the conditional Weibull shape k_cond (task-level random effect, cloglog link).

Variants (all task-level random effect only, for speed and comparability):
  base         : Gaussian frailty, TH1.1, all runs
  hetero       : Gaussian frailty with log-sd linear in log T (frailty spread may change with length)
  latent3      : nonparametric-style frailty: 3-point discrete mixture (latent classes)
  drop_usage   : exclude runs that ended with a usage-limit fatal error
  baseline_T   : only tasks whose human time comes from human baselines (not estimates)
  th10         : METR Time Horizon 1.0 data
"""
import numpy as np
import pandas as pd
from scipy import optimize, special
from common import *

GH = np.polynomial.hermite_e.hermegauss(30)


def nll_hetero(th, tt):
    mu, k, a, c = th
    x, w = GH
    w = w / w.sum()
    lt = np.log(tt["T"].values / T_REF)
    sd = np.exp(a + c * lt)
    eta = mu + k * lt[:, None] + sd[:, None] * x[None, :]
    lp, lq = (-np.exp(np.clip(eta, -30, 6)), None)
    lq = np.log(-np.expm1(lp) + 1e-300)
    ll = tt.s.values[:, None] * lp + (tt.n - tt.s).values[:, None] * lq
    return -special.logsumexp(ll + np.log(w)[None, :], axis=1).sum()


def nll_latent(th, tt, K=3):
    mus = th[:K]
    k = th[K]
    logits = np.concatenate([[0.0], th[K + 1:]])
    lpi = logits - special.logsumexp(logits)
    lt = np.log(tt["T"].values / T_REF)
    eta = mus[None, :] + k * lt[:, None]
    lp = -np.exp(np.clip(eta, -30, 6))
    lq = np.log(-np.expm1(lp) + 1e-300)
    ll = tt.s.values[:, None] * lp + (tt.n - tt.s).values[:, None] * lq
    return -special.logsumexp(ll + lpi[None, :], axis=1).sum()


def best_of(f, starts, args):
    best = None
    for s in starts:
        r = optimize.minimize(f, np.array(s, float), args=args, method="Nelder-Mead",
                              options=dict(maxiter=6000, xatol=1e-6, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    return best


rows = []
datasets = {
    "base": load_runs("th11"),
    "drop_usage": load_runs("th11", drop_usage_limit=True),
    "th10": load_runs("th10"),
}
base = datasets["base"]
datasets["baseline_T"] = base[base.human_source == "baseline"].copy()

for name, d in datasets.items():
    for alias, da in d.groupby("alias"):
        tt = task_table(da)
        if len(tt) < 40 or tt.s.sum() < 10:
            continue
        r = fit_glmm(tt, "cloglog", nested=False)
        r0 = optimize.minimize(lambda th: glmm_nll(np.array([th[0], th[1], np.log(1e-6)]), tt, "cloglog", False),
                               r.x[:2], method="Nelder-Mead")
        row = dict(variant=name, alias=alias, ntask=len(tt), k_cond=r.x[1], k_marg=r0.x[1],
                   sigma=np.exp(r.x[2]))
        rows.append(row)
        if name == "base":
            h = best_of(nll_hetero, [[r.x[0], r.x[1], r.x[2], 0.0], [r.x[0], 1.0, r.x[2], 0.2],
                                     [r.x[0], 1.0, r.x[2], -0.2]], (tt,))
            rows.append(dict(variant="hetero", alias=alias, ntask=len(tt), k_cond=h.x[1], k_marg=r0.x[1],
                             sigma=np.exp(h.x[2]), sd_slope=h.x[3], LR_vs_base=2 * (r.fun - h.fun)))
            sd = np.exp(r.x[2])
            lc = best_of(nll_latent, [[r.x[0] - 1.5 * sd, r.x[0], r.x[0] + 1.5 * sd, r.x[1], 0, 0],
                                      [r.x[0] - 3, r.x[0] - 0.5, r.x[0] + 3, 1.0, 0.5, -0.5],
                                      [r.x[0] - 2, r.x[0] + 1, r.x[0] + 4, 0.7, 0, 0]], (tt,))
            rows.append(dict(variant="latent3", alias=alias, ntask=len(tt), k_cond=lc.x[3], k_marg=r0.x[1],
                             LR_vs_base=2 * (r.fun - lc.fun)))
        print(name, alias, round(r.x[1], 3), flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/robustness.csv", index=False)
S = R.groupby("variant").apply(lambda g: pd.Series(dict(
    n_agents=len(g), k_marg_median=g.k_marg.median(), k_cond_median=g.k_cond.median(),
    k_cond_q25=g.k_cond.quantile(.25), k_cond_q75=g.k_cond.quantile(.75),
    frac_kcond_gt_1=np.mean(g.k_cond > 1), frac_kcond_gt_kmarg=np.mean(g.k_cond > g.k_marg))))
with open(f"{RES}/robustness_summary.txt", "w") as f:
    f.write(S.round(3).to_string() + "\n\n")
    het = R[R.variant == "hetero"]
    f.write(f"hetero: median sd slope on log T {het.sd_slope.median():.3f}; median LR vs base {het.LR_vs_base.median():.2f}\n")
    lat = R[R.variant == "latent3"]
    f.write(f"latent3: median LR vs Gaussian {lat.LR_vs_base.median():.2f}\n")
print(open(f"{RES}/robustness_summary.txt").read())
