"""Profile-likelihood 95% intervals for the per-agent conditional k (task-level Gaussian frailty,
cloglog), complementing the Wald intervals of 06; also notes that LR tests of sigma = 0 lie on
the boundary (reference distribution 0.5*chi2_0 + 0.5*chi2_1 for one variance component).
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from common import *

d = load_runs()
rows = []
for alias, da in d.groupby("alias"):
    tt = task_table(da)
    r = fit_glmm(tt, "cloglog", nested=False)
    kh, fmin = r.x[1], r.fun

    def prof(k, x0=[r.x[0], r.x[2]]):
        f = lambda z: glmm_nll(np.array([z[0], k, z[1]]), tt, "cloglog", False)
        best = None
        for s in [x0, [r.x[0] + (k - kh) * 0.5, r.x[2]]]:
            o = optimize.minimize(f, np.array(s, float), method="Nelder-Mead",
                                  options=dict(xatol=1e-6, fatol=1e-7, maxiter=3000))
            if best is None or o.fun < best.fun:
                best = o
        return 2 * (best.fun - fmin) - stats.chi2.ppf(0.95, 1)

    lo = hi = np.nan
    try:
        lo = optimize.brentq(prof, max(0.05, kh - 3), kh - 1e-4, xtol=1e-3)
    except ValueError:
        pass
    try:
        hi = optimize.brentq(prof, kh + 1e-4, kh + 4, xtol=1e-3)
    except ValueError:
        pass
    r0 = optimize.minimize(lambda th: glmm_nll(np.array([th[0], th[1], np.log(1e-6)]), tt, "cloglog", False),
                           r.x[:2], method="Nelder-Mead")
    LR = 2 * (r0.fun - fmin)
    rows.append(dict(alias=alias, k=kh, prof_lo=lo, prof_hi=hi, LR_sigma=LR,
                     p_boundary=0.5 * stats.chi2.sf(LR, 1)))
    print(alias, round(kh, 3), round(lo, 3), round(hi, 3), flush=True)
R = pd.DataFrame(rows)
R.to_csv(f"{RES}/profile_ci.csv", index=False)
with open(f"{RES}/profile_ci_summary.txt", "w") as f:
    f.write(R.round(3).to_string() + "\n\n")
    f.write(f"profile CI entirely above 1: {(R.prof_lo > 1).sum()}/20; includes 1: "
            f"{((R.prof_lo <= 1) & (R.prof_hi >= 1)).sum()}/20; entirely below 1: {(R.prof_hi < 1).sum()}/20\n")
    f.write(f"min boundary LR for sigma=0: {R.LR_sigma.min():.1f}; max boundary p-value {R.p_boundary.max():.2e}\n")
print(open(f"{RES}/profile_ci_summary.txt").read())
