"""Exp 9b (revision 1): is the departure from the constant-exponent law a slowdown at the top of the data?
Fits a two-break model to the 240 Chinchilla runs,
   L = E + (A N^-a + B D^-b) * max(1, C/C1)^d1 * max(1, C/C2)^d2,
with C1 at the 10th and C2 at the 70th compute percentile (the locations favoured by the one-break profile on
each side), Huber delta_H = 1e-3.  d1 > 0 with d1 + d2 < d1 would mean the exponent falls above C1 and recovers
above C2.  Uncertainty: cluster (pairs) bootstrap over 0.1-dex compute bins, 300 resamples.
Also reports the local reducible-loss exponent gamma_local in each segment, gamma - d1 and gamma - d1 - d2."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from common import load_chinchilla, epoch_keep, RES, huber, SCALE, OPT

N, D, C, L = load_chinchilla()
k = epoch_keep(L)
N, D, C, L = N[k], D[k], C[k], L[k]
lN, lD, lC, lL = np.log(N), np.log(D), np.log(C), np.log(L)
base = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
p0 = np.array([np.log(base["A"]), np.log(base["B"]), np.log(base["E"]), base["alpha"], base["beta"]])
l1, l2 = np.log(np.quantile(C, 0.1)), np.log(np.quantile(C, 0.7))


def obj(q, lN, lD, lC, lL, dh=1e-3):
    a, b, e, al, be, d1, d2 = q
    h1 = np.maximum(0, lC - l1); h2 = np.maximum(0, lC - l2)
    s = d1 * h1 + d2 * h2
    T = np.stack([a - al * lN + s, b - be * lD + s, np.full_like(lN, e)])
    lp = logsumexp(T, axis=0); w = np.exp(T - lp); r = lp - lL; hp = np.clip(r, -dh, dh)
    wr = hp * (w[0] + w[1])
    g = np.array([(hp * w[0]).sum(), (hp * w[1]).sum(), (hp * w[2]).sum(), -(hp * w[0] * lN).sum(),
                  -(hp * w[1] * lD).sum(), (wr * h1).sum(), (wr * h2).sum()])
    return SCALE * huber(r, dh).sum(), SCALE * g


def fit(ii):
    return min((minimize(obj, np.r_[p0, s1, s2], args=(lN[ii], lD[ii], lC[ii], lL[ii]), method="BFGS", jac=True, options=OPT)
                for s1, s2 in [(0, 0), (0.03, -0.05), (-0.03, 0.03)]), key=lambda r: r.fun)


al_, be_ = base["alpha"], base["beta"]
g0 = al_ * be_ / (al_ + be_)
r = fit(np.arange(len(L)))
g0 = r.x[3] * r.x[4] / (r.x[3] + r.x[4])
rng = np.random.default_rng(91)
cl = np.unique(np.round(np.log10(C), 1), return_inverse=True)[1]
bs = []
for i in range(300):
    pick = rng.integers(0, cl.max() + 1, cl.max() + 1)
    ii = np.concatenate([np.where(cl == c)[0] for c in pick])
    bs.append(fit(ii).x[5:7])
bs = np.array(bs)
out = dict(C1=float(np.exp(l1)), C2=float(np.exp(l2)), d1=float(r.x[5]), d2=float(r.x[6]),
           n_runs_above_C2=int((C >= np.exp(l2)).sum()),
           gamma_segments_pointwise=[g0, g0 - float(r.x[5]), g0 - float(r.x[5]) - float(r.x[6])],
           d1_ci95=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
           d2_ci95=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
           d1_plus_d2_ci95=[float(np.percentile(bs.sum(1), 2.5)), float(np.percentile(bs.sum(1), 97.5))],
           frac_boot_d2_negative=float((bs[:, 1] < 0).mean()))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{RES}/exp9b_two_break.json", "w"), indent=2)
