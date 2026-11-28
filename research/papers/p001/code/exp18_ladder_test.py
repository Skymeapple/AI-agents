"""Exp 18 (revision 2): an E-free frontier "ladder" test (Proposition 4) and its resolution.

Three compute-optimal runs at C0, 10 C0, 100 C0 trained with one recipe.  Under the constant-exponent law,
whatever E and whatever constant compute penalty P applies to all rungs, the successive loss gaps satisfy
g2 = r g1 with r = 10^-gamma.  Statistic T = g2 - r_hat g1, r_hat = 10^-gamma_hat with gamma_hat taken from the
small sweep (SD sd_gamma).  One-sided 5% test with a delta-method SD.
Truth: Chinchilla refit frontier with an exponent change delta from C0 on: R(C) (C/C0)^delta.
Reports closed-form power and delta_min (50%/80% power), Monte Carlo power (20,000 replicates per cell), and
size under (i) a constant compute penalty P = 5 on all rungs, (ii) independent per-rung recipe offsets.
For comparison: the delta that the forecast-residual test resolves at the same top compute with the public
Chinchilla forecast SD (Exp. 16, compute-bin bootstrap s(C)).
"""
import json
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from common import RES, load_chinchilla, epoch_keep, obj_base, OPT
from scipy.optimize import minimize
import horizon as H

P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
TH = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])
GAM = H.gam(TH); E = float(np.exp(TH[2])); Z = norm.ppf(0.95)


def rungs(Ctop, delta, P_pen=1.0):
    C = Ctop / np.array([100.0, 10.0, 1.0]) / P_pen
    _, R, _, _ = H.frontier_parts(TH, C)
    return E + R * (C * P_pen / C[0] / P_pen) ** delta


def stat_sd(L, sigma, sd_g):
    r = 10 ** -GAM
    g1 = L[..., 0] - L[..., 1]
    v = sigma ** 2 * (L[..., 1] ** 2 * (1 + r) ** 2 + L[..., 0] ** 2 * r ** 2 + L[..., 2] ** 2) + (g1 * r * np.log(10) * sd_g) ** 2
    return np.sqrt(v)


def power_cf(Ctop, delta, sigma, sd_g, alpha_z=Z):
    L = rungs(Ctop, delta); r = 10 ** -GAM
    mu = (L[1] - L[2]) - r * (L[0] - L[1])
    return float(norm.cdf(mu / stat_sd(L, sigma, sd_g) - alpha_z))


def delta_min(Ctop, sigma, sd_g, power=0.5):
    """Smallest delta (first crossing on a fine grid up to gamma/2) reaching the target power.  The signal of
    this statistic vanishes again as delta -> gamma (the gaps themselves vanish), so we search delta <= gamma/2."""
    ds = np.linspace(1e-4, GAM / 2, 2000)
    pw = np.array([power_cf(Ctop, d, sigma, sd_g) for d in ds])
    i = np.where(pw >= power)[0]
    return float(ds[i[0]]) if len(i) else float("inf")


def mc(Ctop, delta, sigma, sd_g, n=20000, seed=0, P_pen=1.0, sigma_off=0.0):
    rng = np.random.default_rng(seed)
    L = rungs(Ctop, delta, P_pen)
    obs = np.exp(np.log(L)[None, :] + sigma * rng.standard_normal((n, 3)) + sigma_off * rng.standard_normal((n, 3)))
    g_hat = GAM + sd_g * rng.standard_normal(n)
    T = (obs[:, 1] - obs[:, 2]) - 10 ** -g_hat * (obs[:, 0] - obs[:, 1])
    sd = stat_sd(obs, sigma, sd_g)
    return float(np.mean(T > Z * sd))


if __name__ == "__main__":
    # sd of gamma_hat from a compute-bin cluster bootstrap of the 240 runs (200 replicates)
    N, D, C, L = load_chinchilla(); k = epoch_keep(L)
    lN, lD, lL = np.log(N[k]), np.log(D[k]), np.log(L[k])
    cl = np.unique(np.round(np.log10(C[k]), 1), return_inverse=True)[1]
    sets = [np.where(cl == g)[0] for g in np.unique(cl)]
    rng = np.random.default_rng(18)
    gs = []
    for b in range(200):
        ii = np.concatenate([sets[j] for j in rng.integers(0, len(sets), len(sets))])
        x = minimize(obj_base, TH, args=(lN[ii], lD[ii], lL[ii]), method="BFGS", jac=True, options=OPT).x
        gs.append(H.gam(x))
    sd_g = float(np.std(gs))
    print("sd_gamma (compute-bin bootstrap) =", sd_g)
    ex16 = json.load(open(f"{RES}/exp16_audit_ceiling.json"))
    out = dict(gamma=GAM, sd_gamma=sd_g, cells=[], dmin=[])
    for Ctop in [1e25, 1e26, 5e26, 1e27, 1e28]:
        for sigma in [3e-5, 1e-4, 3e-4, 1e-3]:
            for sg, lab in [(sd_g, "sweep"), (0.0, "gamma_known")]:
                out["dmin"].append(dict(C_top=Ctop, sigma=sigma, sd_gamma=sg, label=lab,
                                        delta_min_50=delta_min(Ctop, sigma, sg), delta_min_80=delta_min(Ctop, sigma, sg, 0.8)))
            for d in [0.0, 0.02, 0.05, 0.1]:
                out["cells"].append(dict(C_top=Ctop, sigma=sigma, delta=d, power_cf=power_cf(Ctop, d, sigma, sd_g),
                                         power_mc=mc(Ctop, d, sigma, sd_g, seed=int(Ctop % 997 + 1e5 * sigma + 100 * d))))
    out["size_robustness"] = []
    for Ctop in [1e26, 1e27]:
        for sigma in [1e-4, 3e-4]:
            out["size_robustness"].append(dict(C_top=Ctop, sigma=sigma,
                                               size_const_penalty_P5=mc(Ctop, 0.0, sigma, sd_g, seed=7, P_pen=5.0),
                                               size_rung_offsets_1em3=mc(Ctop, 0.0, sigma, sd_g, seed=8, sigma_off=1e-3)))
    # forecast-residual comparison: delta resolvable with the public Chinchilla s(C) (break at C_top/100)
    s_pub = {}
    for key in ["240:compute_bins", "240:iid"]:
        row = ex16["b_chinchilla"][key]
        s5 = row["s_at_5e26"]
        # power of residual test at 5e26 for a break delta at 5e24: penalty P = 100^(delta/gamma)
        s_pub[key] = {str(d): float(H.residual_power(TH, 5e26, s5, 100 ** (d / GAM))) for d in [0.02, 0.05, 0.1]}
    out["residual_test_power_at_5e26_public_s"] = s_pub
    json.dump(out, open(f"{RES}/exp18_ladder.json", "w"), indent=2)
    for r in out["dmin"]:
        print(f"C_top={r['C_top']:.0e} sigma={r['sigma']:g} {r['label']}: delta_min50={r['delta_min_50']:.4f} delta_min80={r['delta_min_80']:.4f}")
    for c in out["cells"]:
        print(f"C_top={c['C_top']:.0e} sigma={c['sigma']:g} delta={c['delta']}: cf={c['power_cf']:.3f} mc={c['power_mc']:.3f}")
    print(out["size_robustness"]); print(s_pub)
