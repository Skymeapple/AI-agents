"""Theory figures + Monte Carlo checks. Run: python3 code/run_theory.py (from p023/)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from theory import *

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)
R = {}

# ---------- E1: correlated-vote ceiling and slow approach ----------
p = 0.65
rhos = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
ns = np.array([1, 3, 5, 9, 15, 25, 41, 65, 101, 161, 257, 401, 641, 1025])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
R["E1"] = {}
for rho in rhos:
    acc = np.array([vote_acc(n, p, rho) for n in ns])
    C = ceiling(p, rho)
    gap = C - acc
    R["E1"][str(rho)] = {"ceiling": C, "acc": dict(zip(map(int, ns), acc.round(5).tolist()))}
    ax[0].plot(ns, acc, "o-", ms=3, label=f"rho={rho} (C={C:.3f})")
    if rho > 0:
        msk = (ns >= 41) & (gap > 0)
        slope = np.polyfit(np.log(ns[msk]), np.log(gap[msk]), 1)[0]
        R["E1"][str(rho)]["loglog_gap_slope_n>=41"] = float(slope)
        ax[1].loglog(ns[gap > 0], gap[gap > 0], "o-", ms=3, label=f"rho={rho}, slope={slope:.2f}")
    else:
        ax[1].semilogy([], [])
ax[0].set_xscale("log"); ax[0].set_xlabel("team size n"); ax[0].set_ylabel("majority-vote accuracy")
ax[0].set_title(f"(a) p={p}: correlation caps redundancy"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("team size n"); ax[1].set_ylabel("ceiling - accuracy")
ax[1].set_title("(b) approach to ceiling is polynomial"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_ceiling.png"), dpi=150); plt.close()

# ---------- E2: Monte Carlo validation (exact vs MC; tuple vs coord) ----------
R["E2"] = []
for (K, pp, rho, m) in [(1, 0.65, 0.1, 2), (3, 0.7, 0.1, 2), (3, 0.7, 0.1, 5), (3, 0.45, 0.1, 5), (4, 0.7, 0.2, 2)]:
    for n in [3, 9, 33, 129, 401]:
        n_items = 4000 if n <= 129 else 1500
        mt = mc_team(n, K, pp, rho, m, n_items, rng, "tuple")
        mc = mc_team(n, K, pp, rho, m, n_items, rng, "coord")
        exact = vote_acc(n, pp, rho) ** K if m == 2 else None
        lim = ceiling(pp, rho, m) ** K
        R["E2"].append(dict(K=K, p=pp, rho=rho, m=m, n=n, items=n_items, mc_tuple=mt, mc_coord=mc,
                            exact_coord_m2=exact, asymptote=lim))
        print("E2", K, pp, rho, m, n, round(mt, 3), round(mc, 3), exact and round(exact, 3), round(lim, 3))

# ---------- E3: asymptotic critical hand-off error eps* ----------
K = 4; pG = 0.6
rgrid = np.linspace(0.005, 0.8, 80)
psgrid = np.linspace(0.6, 0.95, 71)
H = np.array([[eps_star(pG, ps, r, K) for r in rgrid] for ps in psgrid])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
im = ax[0].imshow(H, origin="lower", aspect="auto", extent=[rgrid[0], rgrid[-1], psgrid[0], psgrid[-1]], cmap="viridis")
cs = ax[0].contour(rgrid, psgrid, H, levels=[0.02, 0.05, 0.1, 0.2, 0.3], colors="w", linewidths=0.8)
ax[0].clabel(cs, fontsize=7)
plt.colorbar(im, ax=ax[0], label="eps* (max tolerable hand-off error)")
ax[0].set_xlabel("error correlation rho"); ax[0].set_ylabel("specialist accuracy p_S")
ax[0].set_title(f"(a) asymptotic eps*, K={K}, p_G={pG}")
R["E3"] = {}
for ps in [0.65, 0.7, 0.8, 0.9]:
    e = [eps_star(pG, ps, r, K) for r in rgrid]
    ax[1].plot(rgrid, e, label=f"p_S={ps}")
    i = int(np.argmax(e))
    R["E3"][str(ps)] = {"rho_at_max": float(rgrid[i]), "eps_star_max": float(e[i]),
                        "eps_star_rho0.01": eps_star(pG, ps, 0.01, K), "eps_star_rho0.1": eps_star(pG, ps, 0.1, K),
                        "eps_star_rho0.3": eps_star(pG, ps, 0.3, K), "eps_star_rho_to_1_limit": float(max(0, 1 - (pG / ps) ** (K / (K - 1))))}
ax[1].set_xlabel("error correlation rho"); ax[1].set_ylabel("eps*")
ax[1].set_title("(b) eps* vanishes as rho->0 (p_G>1/2)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_eps_star.png"), dpi=150); plt.close()

# ---------- E4: finite-budget phase diagrams ----------
K = 4; pG, pS = 0.6, 0.7
egrid = np.linspace(0, 0.3, 61); rg = np.linspace(0.0, 0.6, 61)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
R["E4"] = {}
for j, B in enumerate([4, 40, 400]):
    Dm = np.array([[acc_division(B, K, pS, r, e) - acc_generalist_coord(B, K, pG, r) for r in rg] for e in egrid])
    im = ax[j].imshow(Dm, origin="lower", aspect="auto", extent=[rg[0], rg[-1], egrid[0], egrid[-1]],
                      cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax[j].contour(rg, egrid, Dm, levels=[0], colors="k")
    ax[j].set_title(f"B={B} units (n={B // K} per subtask)")
    ax[j].set_xlabel("rho"); ax[j].set_ylabel("hand-off error eps")
    # boundary eps at a few rho
    bd = {}
    for r in [0.0, 0.05, 0.1, 0.2, 0.4]:
        f = lambda e: acc_division(B, K, pS, r, e) - acc_generalist_coord(B, K, pG, r)
        es = np.linspace(0, 0.6, 6001); v = np.array([f(e) for e in es])
        bd[str(r)] = float(es[np.argmax(v < 0)]) if (v < 0).any() else None
    R["E4"][str(B)] = bd
plt.colorbar(im, ax=ax, label="acc(division) - acc(redundancy)")
plt.savefig(os.path.join(OUT, "fig4_phase_finite.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------- E5: hard-task regime (p_G < 1/2) ----------
R["E5"] = {}
for rho in [0.0, 0.1, 0.3]:
    R["E5"][str(rho)] = {str(n): {"G": acc_generalist_coord(K * n, K, 0.45, rho), "D_eps0.1": acc_division(K * n, K, 0.7, rho, 0.1)}
                         for n in [1, 5, 25, 125]}

# ---------- E6: optimal communication (interface-check) budget ----------
K = 4; pS = 0.7; eps = 0.1; d = 0.5
R["E6"] = {}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for B in [40, 200, 1000]:
    xs, ys = [], []
    for rho in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        c, a, a0 = best_check_split(B, K, pS, rho, eps, d, step=max(1, B // 200))
        xs.append(rho); ys.append(c / B)
        R["E6"][f"B{B}_rho{rho}"] = {"c_star": c, "share": c / B, "acc_opt": a, "acc_nocheck": a0}
    ax.plot(xs, ys, "o-", label=f"B={B}")
ax.set_xlabel("error correlation rho"); ax.set_ylabel("optimal share of budget on hand-off checks")
ax.set_title(f"K={K}, p_S={pS}, eps={eps}, d={d}"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_comm_share.png"), dpi=150); plt.close()

with open(os.path.join(OUT, "theory_results.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps({k: R[k] for k in ["E3", "E4", "E5"]}, indent=1))
for k, v in R["E6"].items(): print(k, v)
for k, v in R["E1"].items(): print(k, v["ceiling"], v.get("loglog_gap_slope_n>=41"))
