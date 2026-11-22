"""Numerically verify Proposition 1 (first-order visibility coefficients) and quantify
global (large-shock) deviations. Outputs: results/verify_props.csv, results/fig1_vhome.png,
results/fig2_global.png"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Economy, visibility, analytic_V

R = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(R, exist_ok=True)
rng = np.random.default_rng(0)

# ---- 1. random-parameter check of first-order formulas ----
rows = []
maxerr = 0.0
for k in range(300):
    e_s = rng.uniform(0.03, 0.25); e_x = rng.uniform(0.2, 0.6); e_c = 1 - e_s - e_x
    theta = rng.uniform(0.1, 0.9); sigma = rng.uniform(0.3, 4.0); eta = rng.uniform(0.4, 1.8)
    ec = Economy(e_c, e_s, e_x, theta, sigma, eta)
    Va = analytic_V(e_c, e_s, e_x, theta, sigma, eta)
    eps = 1e-4
    Vn = [visibility(ec, gc=eps)[2], visibility(ec, gs=eps)[2], visibility(ec, gB=eps)[2]]
    err = max(abs(a - b) for a, b in zip(Va, Vn))
    maxerr = max(maxerr, err)
    rows.append([e_c, e_s, e_x, theta, sigma, eta, *Va, *Vn, err])
rows = np.array(rows)
np.savetxt(os.path.join(R, "verify_props.csv"), rows, delimiter=",",
           header="e_c,e_s,e_x,theta,sigma,eta,Vc_an,Vs_an,Vh_an,Vc_num,Vs_num,Vh_num,abs_err",
           comments="")
# identity V_s - V_h = sigma
ident = np.max(np.abs((rows[:, 7] - rows[:, 8]) - rows[:, 4]))
summary = {"n_random_checks": len(rows), "max_abs_err_first_order": float(maxerr),
           "max_abs_dev_identity_Vs_minus_Vh_eq_sigma": float(ident)}

# ---- 2. Figure 1: V_home vs sigma (Cobb-Douglas upper nest), analytic + numeric small & large ----
sig_grid = np.linspace(0.5, 4.0, 36)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cols = {0.3: "#1f77b4", 0.5: "#d62728", 0.7: "#2ca02c"}
glob_tab = []
for th, col in cols.items():
    an = [analytic_V(0.45, 0.10, 0.45, th, s, 1.0)[2] for s in sig_grid]
    ax.plot(sig_grid, an, color=col, lw=2, label=f"analytic, θ={th}")
    nl = []
    for s in sig_grid[::5]:
        ec = Economy(0.45, 0.10, 0.45, th, s, 1.0)
        nl.append(visibility(ec, gB=np.log(2.0))[2])   # home productivity doubles
        glob_tab.append([th, s, analytic_V(0.45, 0.10, 0.45, th, s, 1.0)[2], nl[-1]])
    ax.plot(sig_grid[::5], nl, "o", color=col, mfc="none", ms=7,
            label=f"exact, B×2, θ={th}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("σ: elasticity of substitution, market vs self-provided service")
ax.set_ylabel("V_home = ΔGDP / ΔWelfare")
ax.set_title("Visibility of a consumer-side (home) AI productivity gain")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_vhome.png"), dpi=160); plt.close(fig)
np.savetxt(os.path.join(R, "vhome_global_vs_firstorder.csv"), np.array(glob_tab), delimiter=",",
           header="theta,sigma,V_home_first_order,V_home_exact_Bx2", comments="")

# ---- 3. Figure 2: aggregate visibility heatmap over (sigma, home-share of AI gain) ----
# AI raises productivity of market good (gc), market service (gs) and home (gB).
# Fix welfare-weighted market gain; vary the ratio gB/gs and sigma.
sig = np.linspace(0.5, 4, 50); ratio = np.linspace(0, 4, 50)
Vagg = np.zeros((len(ratio), len(sig)))
e_c, e_s, e_x, th = 0.45, 0.10, 0.45, 0.5
for i, r in enumerate(ratio):
    for j, s in enumerate(sig):
        Vc, Vs, Vh = analytic_V(e_c, e_s, e_x, th, s, 1.0)
        gc, gs, gB = 0.01, 0.05, 0.05 * r
        Wc, Ws, Wh = e_c * gc, e_s * (1 - th) * gs, e_s * th * gB
        Vagg[i, j] = (Wc * Vc + Ws * Vs + Wh * Vh) / (Wc + Ws + Wh)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
im = ax.contourf(sig, ratio, Vagg, levels=np.linspace(-0.6, 1.6, 23), cmap="RdBu")
cs = ax.contour(sig, ratio, Vagg, levels=[0, 0.5, 1.0], colors="k", linewidths=1)
ax.clabel(cs, fmt="V=%.1f", fontsize=8)
plt.colorbar(im, ax=ax, label="aggregate visibility V")
ax.set_xlabel("σ (market vs self-provided substitutability)")
ax.set_ylabel("g_B / g_s (AI gain in self-provision relative to market provision)")
ax.set_title("Aggregate GDP visibility of AI (first order, η=1)")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_aggregate_V.png"), dpi=160); plt.close(fig)

# ---- 4. global error for realistic-size shocks ----
errs = []
for k in range(100):
    s = rng.uniform(0.5, 4); th = rng.uniform(0.2, 0.8); eta = rng.uniform(0.6, 1.4)
    ec = Economy(0.45, 0.10, 0.45, th, s, eta)
    gB = rng.uniform(0.05, 0.7)
    errs.append(abs(visibility(ec, gB=gB)[2] - analytic_V(0.45, 0.10, 0.45, th, s, eta)[2]))
summary["global_vs_first_order_Vhome_abs_err_median"] = float(np.median(errs))
summary["global_vs_first_order_Vhome_abs_err_p90"] = float(np.quantile(errs, 0.9))
json.dump(summary, open(os.path.join(R, "verify_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
