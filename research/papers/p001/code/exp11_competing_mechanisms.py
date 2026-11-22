"""Exp 11 (revision 1): competing break mechanisms, so that the 'next growth point' answer is not fixed by
construction (reviewer W4).  All three mechanisms share the Muennighoff et al. base law
(E=1.87, A=521, B=1488, alpha=beta=0.35) and the same onset compute C_x = C_wall(U=1e14).

  M1 'unique-data cap'   : Muennighoff et al. eq. 17 with U = 1e14 unique tokens (repetition decays in value).
  M2 'quality decline'   : unlimited tokens, but the marginal token beyond the best Q = 1e14 is worth less:
                           D_eff = D (D <= Q);  Q [1 + ((D/Q)^(1-theta) - 1)/(1-theta)] (D > Q),  theta = 0.5
                           (the q-th token has value (q/Q)^-theta).  Lever: raise Q (filtering / better sources).
  M3 'non-data exponent break': reducible loss multiplied by max(1, C/C_x)^delta, delta = 0.05 (an
                           architecture/optimisation limit unrelated to data).  Lever: algorithmic change.

For each mechanism and C/C_x we compute: compute efficiency eta vs the no-break law; the compute-equivalent
value of doubling the data lever (U for M1, Q for M2, U for M3); the log-loss excess; the paired data-doubling
gap at the optimal (N, D); and the power of the frontier loss-curve break test (profiled Huber test,
240 runs over 3 decades ending at 10 or 100 C_x, sigma_w = 3e-4 or 1e-3, critical value from the no-break null).
"""
import json, os, time
import numpy as np
from multiprocessing import Pool
from breaktest import profiled, wilson
from common import RES

A, B, E, a, b = 521.0, 1488.0, 1.87, 0.35, 0.35
RDs, RNs, kN = 15.4, 5.3, 0.051
U0 = 1e14
THETA, DELTA = 0.5, 0.05
G = (a * A / (b * B)) ** (1 / (a + b))
CX = 6.0 * (U0 * G) ** ((a + b) / a)
PT = np.array([np.log(A), np.log(B), np.log(E), a, b])


def loss(mech, N, D, C, lever=1.0):
    if mech == "none":
        return E + A * N ** -a + B * D ** -b
    if mech == "M1":
        U = U0 * lever
        UD = np.minimum(D, U); RD = np.maximum(0.0, D / UD - 1.0)
        Dp = UD * (1 + RDs * (1 - np.exp(-RD / RDs)))
        UN = np.minimum(N, kN * UD); RN = np.maximum(0.0, N / UN - 1.0)
        Np = UN * (1 + RNs * (1 - np.exp(-RN / RNs)))
        return E + A * Np ** -a + B * Dp ** -b
    if mech == "M2":
        Q = U0 * lever
        x = np.maximum(D / Q, 1.0)
        Deff = np.where(D <= Q, D, Q * (1 + (x ** (1 - THETA) - 1) / (1 - THETA)))
        return E + A * N ** -a + B * Deff ** -b
    if mech == "M3":  # lever (data) has no effect
        return E + (A * N ** -a + B * D ** -b) * np.maximum(1.0, C / CX) ** DELTA
    raise ValueError(mech)


Cs = CX * np.logspace(-2, 4, 601)
lC = np.log(Cs)
Nopt = G * (Cs / 6) ** (b / (a + b))
lNg = np.log(Nopt)[:, None] + np.linspace(-8, 8, 3000)[None, :]


def front(mech, lever=1.0):
    Ng = np.exp(lNg); Dg = Cs[:, None] / (6 * Ng)
    Lg = loss(mech, Ng, Dg, Cs[:, None], lever)
    j = Lg.argmin(1)
    return np.minimum.accumulate(Lg[np.arange(len(Cs)), j]), Ng[np.arange(len(Cs)), j], Dg[np.arange(len(Cs)), j]


def inv(Lc, t):
    return np.inf if t < Lc[-1] - 1e-15 else float(np.exp(np.interp(-t, -Lc, lC)))


Lnone = front("none")[0]
summary = {}
for mech in ["M1", "M2", "M3"]:
    Lm, Nm, Dm = front(mech)
    L2 = front(mech, 2.0)[0]
    rows = []
    for m in [1, 3, 10, 30, 100, 1000]:
        i = int(np.argmin(np.abs(lC - np.log(m * CX))))
        eta = inv(Lnone, Lm[i]) / Cs[i]
        kap = inv(Lm, L2[i]) / Cs[i]
        gap = float(np.log(loss(mech, Nm[i], Dm[i], Cs[i], 1.0)) - np.log(loss(mech, Nm[i], Dm[i], Cs[i], 2.0)))
        rows.append(dict(C_over_Cx=m, eta=float(eta), value_of_2x_data=(None if np.isinf(kap) else float(kap)),
                         log_loss_excess=float(np.log(Lm[i] / Lnone[i])), paired_doubling_gap=gap))
    summary[mech] = rows
    print(mech, json.dumps(rows), flush=True)

# loss-curve detectability of each mechanism
def window(mend, K=12, M=20):
    lCw = np.repeat(np.linspace(np.log(mend * CX) - 3 * np.log(10), np.log(mend * CX), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(2), np.log(200), M)), K)
    lN = 0.5 * (lCw - np.log(6) - np.log(ratio)); lD = lCw - np.log(6) - lN
    return lN, lD, lCw


def job(args):
    mech, mend, sigma, seed = args
    rng = np.random.default_rng(seed)
    lN, lD, lCw = window(mend)
    mu = np.log(loss(mech, np.exp(lN), np.exp(lD), np.exp(lCw)))
    y = mu + sigma * rng.standard_normal(mu.shape)
    grid = np.quantile(lCw, np.linspace(0.1, 0.9, 9))
    r = profiled(lN, lD, lCw, y, grid, [PT, PT + np.array([0.3, -0.3, 0.01, 0.01, -0.01])], d0s=(0.0, 0.05))
    x = r["x0"]
    return r["T1"], float(np.exp(x[2]))


if __name__ == "__main__":
    t0 = time.time()
    det = []
    with Pool(os.cpu_count()) as pool:
        for mend in [10, 100]:
            for sigma in [3e-4, 1e-3]:
                nul = np.array(pool.map(job, [("none", mend, sigma, 11000000 + i + int(1e5 * sigma) + mend) for i in range(150)]))
                crit = float(np.quantile(nul[:, 0], 0.95))
                for mech in ["M1", "M2", "M3"]:
                    alt = np.array(pool.map(job, [(mech, mend, sigma, 12000000 + 1000 * ["M1", "M2", "M3"].index(mech) + i + mend)
                                                  for i in range(100)]))
                    k = int((alt[:, 0] > crit).sum())
                    det.append(dict(window_end_over_Cx=mend, sigma=sigma, mech=mech, power=k / 100, ci95=wilson(k, 100),
                                    E_hat_median=float(np.median(alt[:, 1])), E_hat_null_median=float(np.median(nul[:, 1]))))
                    print(det[-1], f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(C_x=CX, theta=THETA, delta=DELTA, summary=summary, loss_curve_detection=det),
              open(f"{RES}/exp11_competing_mechanisms.json", "w"), indent=2)
