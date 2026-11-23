"""Where does short->long extrapolation fail?  Two decompositions (addresses the "location drift" caveat).

(a) Piecewise hazard-length map with an unrestricted (NPMLE) frailty distribution:
      log Lambda_0(T) = k1*min(l, c1) + k2*(clip(l, c1, c2) - c1) + k3*(max(l, c2) - c2),  l = log(T/60 min)
    with knots c1 = log(1) (1 h) and c2 = log(4) (4 h).  Local elasticities k1, k2, k3 per agent;
    LR test of k1 = k2 = k3.
(b) Within-family elasticity: cloglog with family fixed effects (ridge-penalised) and Gaussian task frailty,
    so k is identified only by length contrasts among tasks of the same family.

Outputs: results/hazard_map.csv, results/within_family.csv, results/hazard_map_summary.txt
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import optimize, special, stats
from common import *
import npmle

d = load_runs()
AGENTS = sorted(d.alias.unique())
C1, C2 = np.log(60 / T_REF), np.log(240 / T_REF)


def basis(T):
    l = np.log(T / T_REF)
    return np.stack([np.minimum(l, C1), np.clip(l, C1, C2) - C1, np.maximum(l, C2) - C2], 1)


def ll_piecewise(ks, n, s, T, pi0=None):
    logL0 = basis(T) @ ks
    eta = logL0[:, None] + npmle.U_GRID[None, :]
    e = np.exp(np.clip(eta, -40, 8))
    LL = s[:, None] * (-e) + (n - s)[:, None] * np.log(-np.expm1(-e) + 1e-300)
    return npmle.em_weights(LL, iters=1500, tol=1e-9, pi0=pi0)


def fit_piecewise(n, s, T, k0):
    f = lambda ks: -ll_piecewise(np.asarray(ks), n, s, T)[1]
    best = None
    for st in ([k0, k0, k0], [k0 * 0.7, k0, k0 * 1.4], [k0 * 1.3, k0, k0 * 0.7]):
        r = optimize.minimize(f, st, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return best


# ------------------------------------------------------------- within-family model
def within_family_nll(theta, lt, n, s, fam_idx, nfam, ridge=0.01, xg=None, wg=None):
    k, lsig = theta[0], theta[1]
    a = theta[2:]
    eta = (a[fam_idx] + k * lt)[:, None] + np.exp(lsig) * xg[None, :]
    e = np.exp(np.clip(eta, -30, 8))
    ll = s[:, None] * (-e) + (n - s)[:, None] * np.log(-np.expm1(-e) + 1e-300)
    li = special.logsumexp(ll + np.log(wg)[None, :], axis=1)
    return -li.sum() + ridge * np.sum(a ** 2)


def fit_within_family(tt):
    fam = tt.family.values
    # keep families with >= 2 tasks spanning at least a factor 2 in length
    gstat = tt.groupby("family")["T"].agg(["size", "min", "max"])
    keepf = gstat.index[(gstat["size"] >= 2) & (gstat["max"] / gstat["min"] >= 2)]
    m = np.isin(fam, keepf)
    t = tt[m]
    codes, idx = np.unique(t.family.values, return_inverse=True)
    lt = np.log(t["T"].values / T_REF)
    n, s = t.n.values.astype(float), t.s.values.astype(float)
    xg, wg = np.polynomial.hermite_e.hermegauss(25)
    wg = wg / wg.sum()
    best = None
    for k0 in (0.8, 1.5):
        x0 = np.concatenate([[k0, np.log(1.5)], np.zeros(len(codes))])
        r = optimize.minimize(within_family_nll, x0, args=(lt, n, s, idx, len(codes), 0.01, xg, wg), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    # profile CI for k
    kh = best.x[0]
    prof = []
    for kk in np.arange(max(0.1, kh - 1.5), kh + 1.51, 0.05):
        f = lambda th: within_family_nll(np.concatenate([[kk], th]), lt, n, s, idx, len(codes), 0.01, xg, wg)
        r = optimize.minimize(f, best.x[1:], method="L-BFGS-B")
        prof.append((kk, r.fun))
    P = np.array(prof)
    inside = P[2 * (P[:, 1] - best.fun) <= 3.84, 0]
    return dict(k_within=kh, k_within_lo=inside.min(), k_within_hi=inside.max(), sigma_within=np.exp(best.x[1]),
                n_tasks=int(m.sum()), n_families=len(codes))


def work(alias):
    tt = task_table(d[d.alias == alias])
    n, s, T = tt.n.values.astype(float), tt.s.values.astype(float), tt["T"].values.astype(float)
    f1 = npmle.fit(n, s, T)
    r = fit_piecewise(n, s, T, f1["k"])
    LR = 2 * (-r.fun - f1["ll"])
    rec = dict(alias=alias, k_single=f1["k"], ll_single=f1["ll"], k1_lt1h=r.x[0], k2_1to4h=r.x[1], k3_gt4h=r.x[2],
               ll_piecewise=-r.fun, LR_piecewise=LR, p_LR=float(stats.chi2.sf(max(LR, 0), 2)))
    wf = fit_within_family(tt)
    wf["alias"] = alias
    print(alias, "done", flush=True)
    return rec, wf


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(nproc) as p:
        res = p.map(work, AGENTS)
    H = pd.DataFrame([r[0] for r in res])
    W = pd.DataFrame([r[1] for r in res])
    H.to_csv(f"{RES}/hazard_map.csv", index=False)
    W.to_csv(f"{RES}/within_family.csv", index=False)
    pd.set_option("display.width", 250)
    L = ["(a) Piecewise hazard-length map, NPMLE frailty (knots 1 h, 4 h)", H.round(3).to_string(index=False),
         f"medians: k1 {H.k1_lt1h.median():.3f}, k2 {H.k2_1to4h.median():.3f}, k3 {H.k3_gt4h.median():.3f}; "
         f"k3 > k1 for {(H.k3_gt4h > H.k1_lt1h).sum()}/20; k3 > k2 for {(H.k3_gt4h > H.k2_1to4h).sum()}/20; "
         f"sum of LR over agents {H.LR_piecewise.sum():.1f} on {2 * len(H)} df "
         f"(p = {stats.chi2.sf(H.LR_piecewise.clip(lower=0).sum(), 2 * len(H)):.2e}); agents with p<0.05: {(H.p_LR < 0.05).sum()}",
         "\n(b) Within-family elasticity (family fixed effects + Gaussian task frailty)", W.round(3).to_string(index=False),
         f"median k_within {W.k_within.median():.3f}; k_within > 1 for {(W.k_within > 1).sum()}/20; "
         f"profile CI above 1 for {(W.k_within_lo > 1).sum()}/20; below 1 for {(W.k_within_hi < 1).sum()}/20"]
    open(f"{RES}/hazard_map_summary.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
