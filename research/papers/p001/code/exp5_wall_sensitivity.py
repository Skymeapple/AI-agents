"""Exp 5: Sensitivity of the data-wall transition to the repetition half-lives (R_D*, R_N*), to the
base law (Muennighoff eq.17 vs. Epoch's refit of Chinchilla with alpha != beta), and a check of the
self-similarity result (Prop. 3): with alpha = beta, every threshold scales as C_x = kappa_x * U^2.
All thresholds are reported as multiples of C_wall (compute at which uncapped-optimal D equals U)."""
import json
import numpy as np
from common import RES

lNgrid = np.linspace(np.log(1e6), np.log(1e18), 5000)


def make_loss(A, B, E, a, b, RDs, RNs, kN):
    def loss(N, D, U):
        if np.isinf(U):
            return E + A * N ** -a + B * D ** -b
        UD = np.minimum(D, U); RD = np.maximum(0.0, D / UD - 1.0)
        Dp = UD + UD * RDs * (1 - np.exp(-RD / RDs))
        UN = np.minimum(N, kN * UD); RN = np.maximum(0.0, N / UN - 1.0)
        Np = UN + UN * RNs * (1 - np.exp(-RN / RNs))
        return E + A * Np ** -a + B * Dp ** -b
    return loss


def frontier(loss, C, U):
    Ng = np.exp(lNgrid)[None, :]
    Dg = C[:, None] / (6 * Ng)
    Lg = loss(Ng, Dg, U)
    j = Lg.argmin(1)
    return Lg[np.arange(len(C)), j], Dg[np.arange(len(C)), j]


def thresholds(params, U):
    loss = make_loss(**params)
    E = params["E"]
    Cs = np.logspace(np.log10(U) * 2 - 4, np.log10(U) * 2 + 6, 601)
    Linf, Dinf = frontier(loss, Cs, np.inf)
    Lu, _ = frontier(loss, Cs, U)
    L2, _ = frontier(loss, Cs, 2 * U)
    lC = np.log(Cs)
    ginf = -np.gradient(np.log(Linf - E), lC)
    geff = -np.gradient(np.log(Lu - E), lC)
    Cwall = float(np.exp(np.interp(np.log(U), np.log(Dinf), lC)))
    def inv(Lc, t):
        return np.inf if t < Lc[-1] else float(np.exp(np.interp(-t, -Lc, lC)))
    eta = np.array([inv(Linf, Lu[i]) / c for i, c in enumerate(Cs)])
    km = np.array([inv(Lu, L2[i]) / c for i, c in enumerate(Cs)])
    def first(cond):
        return float(Cs[np.argmax(cond)] / Cwall) if cond.any() else None
    return dict(C_wall=Cwall, eff50=first(eta < 0.5), eff10=first(eta < 0.1),
                gamma_half=first(geff < 0.5 * ginf), shadow_inf=first(np.isinf(km)),
                shadow_2x=first(km > 2.0))


MU = dict(A=521.0, B=1488.0, E=1.87, a=0.35, b=0.35, RDs=15.4, RNs=5.3, kN=0.051)
fit = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
# compute-optimal N/D for Epoch refit at its frontier around 1e24 FLOP gives the kN analogue
a_, b_ = fit["alpha"], fit["beta"]
EP = dict(A=fit["A"], B=fit["B"], E=fit["E"], a=a_, b=b_, RDs=15.4, RNs=5.3, kN=0.051)
out = dict(selfsimilarity_check={}, RD_RN_grid=[], epoch_refit_law={})
for U in [3e13, 1e14, 3e14, 1e15]:
    out["selfsimilarity_check"][f"{U:.0e}"] = thresholds(MU, U)
for RDs in [5.0, 10.0, 15.4, 25.0, 40.0]:
    for RNs in [2.0, 5.3, 10.0]:
        p = dict(MU); p["RDs"] = RDs; p["RNs"] = RNs
        out["RD_RN_grid"].append(dict(RDs=RDs, RNs=RNs, **thresholds(p, 1e14)))
for U in [3e13, 1e14, 3e14, 1e15]:
    out["epoch_refit_law"][f"{U:.0e}"] = thresholds(EP, U)
json.dump(out, open(f"{RES}/exp5_wall_sensitivity.json", "w"), indent=2)
print(json.dumps(out, indent=1))
