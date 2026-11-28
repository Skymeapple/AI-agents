"""Analytic long-tail coverage model (Section 4) and sensitivity analysis.
All quantities are model-derived; the only external numbers are the Gao et al. (2024) fit
coefficients (beta_n=-0.017, gamma=-0.042) and token-to-convergence exponents (0.60, 0.65),
which are quoted, not re-estimated.
Usage: python3 extrapolate.py [mstar]  (default mstar read from results/summary.json if present)"""
import os, sys, json
import numpy as np
from scipy.special import zeta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
out = {}


def coverage_requirements(alpha, U, L0=100.0, mstar=1e3, rho=1.0):
    """Infinite Zipf p_i = C i^-alpha with sum p = L0. Smallest rank cutoff R such that
    the unrecovered activation mass fraction <= U, and the width/tokens/compute it implies.
    Tail mass fraction ~ R^(1-alpha) / ((alpha-1) zeta(alpha)) (integral approximation)."""
    z = zeta(alpha)
    C = L0 / z
    Rk = (U * (alpha - 1) * z) ** (1 / (1 - alpha))
    pmin = C * Rk ** (-alpha)
    T = mstar / pmin
    M = Rk / rho
    return dict(R=Rk, pmin=pmin, tokens=T, width=M, MT=M * T)


# --- Table: cost of reaching unrecovered mass U for several alpha ---
alphas = [1.1, 1.2, 1.3, 1.5, 2.0]
Us = [0.3, 0.1, 0.03, 0.01]
mstar = float(sys.argv[1]) if len(sys.argv) > 1 else None
if mstar is None:
    try:
        s = json.load(open(os.path.join(R, "summary.json")))
        mstar = float(s["E1"]["1.5"]["pooled"]["mstar"])
    except Exception:
        mstar = 1e3
out["mstar_used"] = mstar
tab = []
for a in alphas:
    for U in Us:
        r = coverage_requirements(a, U, mstar=mstar)
        tab.append(dict(alpha=a, U=U, **{k: float(v) for k, v in r.items()}))
out["coverage_table_L0_100"] = tab
# L0 sensitivity
out["L0_sensitivity_alpha1.2_U0.03"] = {L0: {k: float(v) for k, v in coverage_requirements(1.2, 0.03, L0=L0, mstar=mstar).items()}
                                        for L0 in (30, 100, 300)}

# --- Exponents: compute-optimal coverage ---
# Balanced allocation: (C T / m*)^(1/a) = rho M  ->  T ∝ M^a ; compute ∝ M T ∝ M^(1+a) ; U ∝ M^(1-a)
# => U ∝ compute^(-(a-1)/(a+1)),  compute ∝ U^(-(a+1)/(a-1))
out["compute_exponent_for_U"] = {a: float((a + 1) / (a - 1)) for a in alphas}

# --- Consistency check against Gao et al. (2024) published fits (quoted) ---
beta_n, gamma = -0.017, -0.042
gao = {}
for k in (16, 32, 64, 128, 256):
    slope = beta_n + gamma * np.log(k)        # d log L_reducible / d log n at fixed k
    alpha_from_L = 1 - slope                  # model: L_red ∝ n^(1-alpha)
    gao[k] = dict(n_exponent=float(slope), implied_alpha=float(alpha_from_L),
                  implied_Tconv_exponent=float(alpha_from_L))
out["gao_consistency"] = dict(per_k=gao, reported_Tconv_exponents={"GPT-2 small": 0.60, "GPT-4": 0.65},
                              note="model predicts T_conv ∝ n^alpha if convergence requires recovering the rank-n feature")

# --- Figure: compute vs target U ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Ug = np.geomspace(0.005, 0.5, 60)
for a in alphas:
    MT = [coverage_requirements(a, u, mstar=mstar)["MT"] for u in Ug]
    ax[0].loglog(Ug, MT, label=f"alpha={a}")
    W = [coverage_requirements(a, u, mstar=mstar)["width"] for u in Ug]
    ax[1].loglog(Ug, W, label=f"alpha={a}")
for a_ in ax:
    a_.invert_xaxis(); a_.set_xlabel("target unrecovered activation mass U"); a_.legend(fontsize=7)
    a_.axhline(3.4e7 if a_ is ax[1] else 1, ls=":", c="gray")
ax[0].set_ylabel("width x tokens (relative SAE compute / d)")
ax[1].set_ylabel("dictionary width needed (rho=1)")
ax[1].text(0.4, 4.2e7, "34M latents (Templeton et al. 2024)", fontsize=7, color="gray")
ax[0].set_title(f"compute to reach coverage (L0=100, m*={mstar:.0f})")
ax[1].set_title("width to reach coverage")
fig.savefig(os.path.join(R, "fig4_extrapolation.png"), dpi=130, bbox_inches="tight")
json.dump(out, open(os.path.join(R, "extrapolation.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
