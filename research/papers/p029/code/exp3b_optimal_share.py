"""Experiment 3b: optimal originator share m* under layered oversight (Proposition 4).

Two symmetric parties.  h(x) = b (1 + x/tau)^(-beta),  E(y) = (1 + y/sigma)^(-gamma).
Each party bears share m of losses it originates and 1 - m of losses the other originates
(m = alpha*q + (1-alpha)*(1-q) for the alpha-rule under a symmetric channel of accuracy q).

Closed form (interior symmetric equilibrium):
  Lam^(1+beta+gamma) = K m^(-beta) (1-m)^(-gamma),  K = L b tau^beta sigma^gamma beta^-beta gamma^-gamma
  x = m beta Lam - tau,  y = (1-m) gamma Lam - sigma,
  cost per party = (1 + m beta + (1-m) gamma) Lam - tau - sigma.
The numerical solver below does NOT use the closed form and handles corners (x = 0 or y = 0).
Parameters ASSUMED.  Output: results/exp3b_optimal_share.json, results/fig3_share.png
"""
import json, os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
b, tau, sigma = 0.1, 1.0, 1.0


def cf(mm, beta, gamma, L):
    K = L * b * tau**beta * sigma**gamma * beta**(-beta) * gamma**(-gamma)
    Lam = (K * mm**(-beta) * (1 - mm)**(-gamma)) ** (1 / (1 + beta + gamma))
    x = mm * beta * Lam - tau; y = (1 - mm) * gamma * Lam - sigma
    return (1 + mm * beta + (1 - mm) * gamma) * Lam - tau - sigma, x, y


def lossterm(xj, yk, beta, gamma):
    return b * (1 + xj / tau) ** (-beta) * (1 + yk / sigma) ** (-gamma)


def sym_nash(mm, beta, gamma, L, iters=300):
    """Symmetric Nash by damped best-response iteration on (x, y)."""
    x, y = 1.0, 1.0
    for _ in range(iters):
        def cost(z):
            return z[0] + z[1] + L * (mm * lossterm(z[0], y, beta, gamma) + (1 - mm) * lossterm(x, z[1], beta, gamma))
        best = None
        for z0 in ([x, y], [0.0, 0.0], [5.0, 5.0]):
            r = minimize(cost, z0, method="L-BFGS-B", bounds=[(0, None)] * 2)
            if best is None or r.fun < best.fun:
                best = r
        xn = 0.5 * x + 0.5 * best.x[0]; yn = 0.5 * y + 0.5 * best.x[1]
        d = max(abs(xn - x), abs(yn - y)); x, y = xn, yn
        if d < 1e-9:
            break
    return x + y + L * lossterm(x, y, beta, gamma), x, y, d


def first_best(beta, gamma, L):
    best = None
    for z0 in ([1.0, 1.0], [0.0, 3.0], [3.0, 0.0], [8.0, 8.0]):
        r = minimize(lambda z: z[0] + z[1] + L * lossterm(z[0], z[1], beta, gamma), z0,
                     method="L-BFGS-B", bounds=[(0, None)] * 2)
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x[0], best.x[1]


mgrid = np.round(np.concatenate([np.arange(0.05, 1.0, 0.025), [0.5344, 1.0]]), 4)
mgrid = np.unique(mgrid)
res = {"params": {"b": b, "tau": tau, "sigma": sigma}, "cases": []}
curves = {}
for L in [100.0, 1000.0]:
    for beta, gamma in [(1, 1), (1, 0.5), (1, 0.25), (1, 0.1), (0.5, 1), (1, 2), (2, 1)]:
        fb, xf, yf = first_best(beta, gamma, L)
        rows = []
        for mm in mgrid:
            c, x, y, d = sym_nash(float(mm), beta, gamma, L)
            row = {"m": float(mm), "cost": float(c), "x": float(x), "y": float(y), "resid": float(d),
                   "dwl_over_fb": float(c / fb - 1)}
            if mm < 1:
                cc, cx, cy = cf(float(mm), beta, gamma, L)
                row["interior"] = bool(cx > 0 and cy > 0)
                if row["interior"]:
                    row["closed_form_rel_dev"] = float(abs(cc - c) / c)
            rows.append(row)
        k = int(np.argmin([r["cost"] for r in rows]))
        get = lambda mv: next(r for r in rows if abs(r["m"] - mv) < 1e-9)
        devs = [r["closed_form_rel_dev"] for r in rows if "closed_form_rel_dev" in r]
        case = {"L": L, "beta": beta, "gamma": gamma,
                "first_best": {"cost": float(fb), "x": float(xf), "y": float(yf)},
                "m_star_numeric": rows[k]["m"], "dwl_at_m_star": rows[k]["dwl_over_fb"],
                "dwl_at_m_1.0": get(1.0)["dwl_over_fb"], "y_at_m_1.0": get(1.0)["y"],
                "dwl_at_m_0.8": get(0.8)["dwl_over_fb"], "dwl_at_m_0.5344": get(0.5344)["dwl_over_fb"],
                "dwl_at_m_0.5": get(0.5)["dwl_over_fb"],
                "n_interior_points": len(devs),
                "max_closed_form_rel_dev_on_interior": float(max(devs)) if devs else None,
                "max_resid": float(max(r["resid"] for r in rows)),
                "curve": [{"m": r["m"], "dwl_over_fb": r["dwl_over_fb"]} for r in rows]}
        res["cases"].append(case)
        curves[(L, beta, gamma)] = rows
with open(os.path.join(OUT, "exp3b_optimal_share.json"), "w") as f:
    json.dump(res, f, indent=1)

fig, ax = plt.subplots(figsize=(5.4, 3.7))
for (beta, gamma), st in [((1, 1), "-"), ((1, 0.25), "--"), ((1, 0.1), ":"), ((1, 2), "-.")]:
    rows = curves[(1000.0, beta, gamma)]
    ax.plot([r["m"] for r in rows], [r["dwl_over_fb"] for r in rows], st, label=f"beta={beta}, gamma={gamma}")
ax.set_ylim(0, 1.0); ax.set_xlabel("originator's effective share m"); ax.set_ylabel("deadweight loss / first-best cost")
ax.set_title("Layered oversight (L b = 100)", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_share.png"), dpi=150)

for c in res["cases"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("curve", "first_best")})
