"""Exp 6: (a) Confound: when a genuine break is present but an analyst fits the no-break law, how
much does the irreducible-loss estimate E absorb, and how wrong is a 2-decade extrapolation?
(b) Wall detection: data simulated from the data-capped law (Muennighoff et al. eq.17, U=1e14
tokens) in a 3-decade design window ending at m*C_wall; how often does the break test (Bonferroni
over 5 candidate breakpoints) detect deceleration, versus the naive curvature test?"""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import stats
import exp3_simulation_power as S          # reuses design/simulate/objectives (no main run on import)
from common import RES, obj_base, frontier, gamma, SCALE, OPT
import exp4_data_wall as W4                # noqa: F401 (constants), re-run cheap

rng = np.random.default_rng(3)
S.rng = rng
BIG = S.BIG

# (a) confound
conf = []
for d in [0.0, 0.01, 0.02, 0.04]:
    Eh, ex = [], []
    for _ in range(60):
        lN, lD, lC = S.design(4)
        lCb = 0.5 * (lC.min() + lC.max())
        lL = S.simulate(lN, lD, lC, d, S.SIG0, lCb)
        f0 = minimize(obj_base, S.ptrue, args=(lN, lD, lL, BIG), method="BFGS", jac=True, options=OPT)
        q = dict(A=np.exp(f0.x[0]), B=np.exp(f0.x[1]), E=np.exp(f0.x[2]), alpha=f0.x[3], beta=f0.x[4])
        Cx = np.exp(lC.max()) * 100
        # true frontier with break: E + (reducible frontier) * (Cx/Cb)^d
        Lt = S.P["E"] + (frontier(S.P, np.array([Cx]))[0][0] - S.P["E"]) * (Cx / np.exp(lCb)) ** d
        Eh.append(q["E"]); ex.append(np.log(frontier(q, np.array([Cx]))[0][0] / Lt))
    conf.append(dict(delta=d, E_hat_mean=float(np.mean(Eh)), E_hat_p10_p90=[float(np.percentile(Eh, 10)), float(np.percentile(Eh, 90))],
                     extrap_logerr_mean=float(np.mean(ex)), extrap_logerr_sd=float(np.std(ex))))
    print(conf[-1], flush=True)

# (b) wall detection
A_, B_, E_ = 521.0, 1488.0, 1.87
U = 1e14
Cwall = json.load(open(f"{RES}/exp4_data_wall.json"))["caps"]["1e+14"]["C_wall"]
p_mu = np.array([np.log(A_), np.log(B_), np.log(E_), 0.35, 0.35])


def wall_one(m, sigma, Wd=3.0, K=12, M=20):
    lC = np.repeat(np.linspace(np.log(m * Cwall) - Wd * np.log(10), np.log(m * Cwall), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(2), np.log(200), M)), K)
    lN = 0.5 * (lC - np.log(6) - np.log(ratio)); lD = lC - np.log(6) - lN
    L = W4.loss(np.exp(lN), np.exp(lD), U)
    lL = np.log(L) + sigma * rng.standard_normal(L.shape)
    n = len(lL)
    f0 = min((minimize(obj_base, x0, args=(lN, lD, lL, BIG), method="BFGS", jac=True, options=OPT)
              for x0 in [p_mu, p_mu + np.array([0.3, -0.3, 0.02, 0.02, -0.02])]), key=lambda r: r.fun)
    ps = []
    for qq in [0.3, 0.4, 0.5, 0.6, 0.7]:
        lCb = np.quantile(lC, qq)
        f1 = min((minimize(S.obj_brk, np.r_[f0.x, d0], args=(lN, lD, lC, lL, lCb), method="BFGS", jac=True, options=OPT)
                  for d0 in [0.0, 0.05]), key=lambda r: r.fun)
        lr = n * np.log((2 * f0.fun) / max(2 * f1.fun, 1e-300))
        ps.append(0.5 * stats.chi2.sf(lr, 1) if f1.x[5] > 0 else 1.0)
    p_break = min(1.0, 5 * min(ps))
    lCk = lC.reshape(K, M)[:, 0]; lLk = lL.reshape(K, M).min(1)
    x = (lCk - lCk.mean()) / np.log(10); X = np.stack([np.ones_like(x), x, x ** 2], 1)
    bb, *_ = np.linalg.lstsq(X, lLk, rcond=None)
    s2 = ((lLk - X @ bb) ** 2).sum() / (K - 3)
    p_naive = stats.t.sf(bb[2] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2]), K - 3)
    return p_break, p_naive, np.exp(f0.x[2])


wall = []
for sigma in [S.SIG0, 2 * S.SIG0]:
    for m in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        o = np.array([wall_one(m, sigma) for _ in range(60)])
        wall.append(dict(sigma=float(sigma), window_end_over_Cwall=m, power_break_bonf=float((o[:, 0] < 0.05).mean()),
                         power_naive=float((o[:, 1] < 0.05).mean()), E_hat_median=float(np.median(o[:, 2]))))
        print(wall[-1], flush=True)
json.dump(dict(confound=conf, wall_detection=wall, U=U, C_wall=Cwall), open(f"{RES}/exp6_confound_wall.json", "w"), indent=2)
