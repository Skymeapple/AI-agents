"""Analytic coverage model (Section 4) and sensitivity analysis. Pure computation; no fitted
external data except numbers quoted (with citation) in the paper:
  Templeton et al. 2024: 34M-latent SAE with ~12M alive latents; L0 < 300.
  Gao et al. 2024: tokens-to-convergence ~ n^0.60 (GPT-2 small), n^0.65 (GPT-4).
Usage: python3 extrapolate.py [mstar]   (mstar default: toy estimate from results/summary.json)"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
out = {}


def H(N, a):
    """generalized harmonic number sum_{i<=N} i^-a, exact for N<=1e6, Euler-Maclaurin beyond."""
    N = float(N)
    if N <= 1e6:
        return float((np.arange(1, int(N) + 1, dtype=float) ** -a).sum())
    n0 = 1_000_000; s = float((np.arange(1, n0 + 1, dtype=float) ** -a).sum())
    if abs(a - 1) < 1e-12:
        tail = np.log(N / n0)
    else:
        tail = (N ** (1 - a) - n0 ** (1 - a)) / (1 - a)
    return s + tail + 0.5 * (N ** -a - n0 ** -a)


def unrec_mass(Rk, N, a):
    """fraction of feature-activation mass carried by features of rank > Rk (equal magnitudes)."""
    Rk = min(Rk, N)
    return 1 - H(Rk, a) / H(N, a)


def rank_for_U(U, N, a):
    lo, hi = 1.0, float(N)
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if unrec_mass(mid, N, a) > U else (lo, mid)
    return hi


try:
    s = json.load(open(os.path.join(R, "summary.json")))
    toy_m = [v["seed0"]["mstar_occ"] for v in s["E1"].values()]
    mstar_default = float(np.exp(np.mean(np.log(toy_m))))
except Exception:
    mstar_default = 300.0
mstar = float(sys.argv[1]) if len(sys.argv) > 1 else mstar_default
out["mstar_toy_geomean"] = mstar_default

# ---- Table A: coverage of a 12M-alive-latent dictionary if it captured the top-12M features (capacity limit)
alphas = [1.0, 1.1, 1.2, 1.5]; Ns = [1e8, 1e9, 1e10, 1e12]
tabA = []
for a in alphas:
    for N in Ns:
        tabA.append(dict(alpha=a, N=N, U_at_12M=unrec_mass(1.2e7, N, a),
                         R_for_U10=rank_for_U(0.10, N, a), R_for_U01=rank_for_U(0.01, N, a)))
out["tableA_capacity"] = tabA

# ---- Table B: multipliers to go from U=10% to U=1% unrecovered mass (unit-free; no m* transfer needed).
# data-limited requirement for rank R: T*(R) ∝ p_R^(-1/beta) ∝ R^(alpha/beta); width ∝ R; compute ∝ width*tokens
tabB = []
for a in alphas:
    for N in (1e9, 1e10, 1e12):
        R10, R01 = rank_for_U(0.10, N, a), rank_for_U(0.01, N, a)
        for beta in (1.0, 1.4):
            wm = R01 / R10; tm = wm ** (a / beta)
            tabB.append(dict(alpha=a, N=N, beta=beta, R_U10=R10, R_U01=R01, width_mult=wm, token_mult=tm, compute_mult=wm * tm))
out["tableB_multipliers"] = tabB
# illustrative absolute token requirement (beta = 1 only, m* in {1e2,1e3,1e4}, L0 = 100)
tabB2 = []
for a in alphas:
    for N in (1e10,):
        C = 100.0 / H(N, a)
        for U in (0.10, 0.01):
            Rk = rank_for_U(U, N, a); pR = C * Rk ** -a
            tabB2.append(dict(alpha=a, N=N, U=U, R=Rk, p_R=pR, **{f"tokens_mstar{int(m)}": m / pR for m in (1e2, 1e3, 1e4)}))
out["tableB2_absolute_beta1"] = tabB2

# ---- Exponents under balanced allocation rho*M = R_T, with T*(R) ∝ R^(alpha/beta)
out["exponents"] = {f"alpha={a},beta={b}": dict(U_vs_T=-b * (a - 1) / a, U_vs_M=-(a - 1), T_opt_vs_M=a / b,
                                                U_vs_compute=-(a - 1) / (1 + a / b), tokens_per_latent_vs_M=a / b - 1)
                    for a in (1.1, 1.2, 1.5, 2.0) for b in (1.0, 1.4)}

# ---- Gao et al. convergence budget T_conv ∝ n^g vs requirement T*(rank n) ∝ n^(alpha/beta):
# fraction of latents (relative to a width n0 where the budget just suffices) whose feature is learnable
tabC = []
for g in (0.60, 0.65):
    for a in (1.0, 1.2, 1.5):
        for b in (1.0, 1.4):
            e = g * b / a - 1
            tabC.append(dict(gamma=g, alpha=a, beta=b, exponent=e, **{f"frac_at_{k}x": min(1.0, k ** e) for k in (10, 100, 1000)}))
out["tableC_gao_budget"] = tabC
out["gao_consistency_alpha_over_beta"] = "full tail learnable within T_conv ∝ n^g iff alpha/beta <= g"

# ---- Figure: unrecovered mass vs width under capacity limit, several alpha (N = 1e10)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Ms = np.geomspace(1e4, 1e10, 40)
for a in alphas:
    ax[0].loglog(Ms, [unrec_mass(m, 1e10, a) for m in Ms], label=f"α={a}")
ax[0].axvline(1.2e7, ls=":", c="gray"); ax[0].text(1.4e7, 2e-3, "~12M alive latents\n(Templeton et al. 2024)", fontsize=7, color="gray")
ax[0].set_xlabel("features captured R (≈ ρ·alive width)"); ax[0].set_ylabel("unrecovered activation mass U")
ax[0].set_title("capacity limit, N = 1e10 true features", fontsize=10); ax[0].legend(fontsize=7)
n = np.geomspace(1, 1e4, 50)
for g in (0.60,):
    for a in (1.0, 1.2, 1.5):
        for b in (1.0, 1.4):
            ax[1].loglog(n, np.minimum(1, n ** (g * b / a - 1)), ls="-" if b == 1 else "--", label=f"γ={g}, α={a}, β={b}")
ax[1].set_xlabel("width relative to reference n/n0"); ax[1].set_ylabel("fraction of latents whose feature gets ≥ m* exposures")
ax[1].set_title("budget T ∝ n^γ vs requirement T* ∝ n^(α/β)", fontsize=10); ax[1].legend(fontsize=6)
fig.savefig(os.path.join(R, "fig5_extrapolation.png"), dpi=130, bbox_inches="tight")
json.dump(out, open(os.path.join(R, "extrapolation.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:5000])
