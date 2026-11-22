"""Generates all BOS-model results and figures. Run: python3 run_theory.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace, asdict
from bos_model import (Params, chain_exact, anchored_comonotone_exact, simulate,
                       optimal_rate, linearized_rate, floor_rate, h_rate, mu, det)

OUT = "../results/"
C = ["#2a6f97", "#c44e52", "#55a868", "#8172b2", "#dd8452", "#64b5cd"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
base = Params()
R_INF = 10**9
res = {"base_params": {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in asdict(base).items()}}


def best_n(p, nmax=400, fn=chain_exact):
    vals = [fn(n, p)[0] for n in range(1, nmax + 1)]
    i = int(np.argmax(vals))
    return i + 1, vals[i], vals


# ------------------------------------------------------------ 1. rate vs gap
r_opt, d_opt = optimal_rate(base)
r_lin, d_lin = linearized_rate(base)
res["rates"] = dict(r_opt=float(r_opt), delta_opt=float(d_opt), r_lin=float(r_lin), delta_lin=float(d_lin),
                    floor=float(floor_rate(base)), raw=float(base.lam),
                    half_life_opt=float(np.log(2) / r_opt), half_life_raw=float(np.log(2) / base.lam),
                    half_life_floor=float(np.log(2) / floor_rate(base)))
Gs = np.array([100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000])
rows = []
for G in Gs:
    p = replace(base, G=float(G))
    n_star, P_star, _ = best_n(p, nmax=max(10, int(4 * G / d_opt) + 5))
    P_one = chain_exact(1, p)[0]
    rows.append(dict(G=int(G), n_star=n_star, P_star=float(P_star), P_one_step=float(P_one),
                     P_pred_exact_rate=float(np.exp(-G * r_opt)), P_pred_lin=float(np.exp(-G * r_lin)),
                     P_floor=float(np.exp(-G * floor_rate(base))), P_raw=float(np.exp(-G * base.lam)),
                     n_lin=float(G / d_lin)))
res["gap_sweep"] = rows
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(Gs, [r["P_star"] for r in rows], "o-", c=C[0], label="optimal schedule (exact DP)")
ax[0].plot(Gs, [r["P_pred_lin"] for r in rows], "--", c=C[1], label="closed form exp(-G r_lin)")
ax[0].plot(Gs, [r["P_floor"] for r in rows], ":", c=C[2], label="floor exp(-lambda(1-d0)G)")
ax[0].plot(Gs, [r["P_raw"] for r in rows], "-.", c="gray", label="no oversight exp(-lambda G)")
ax[0].plot(Gs, [r["P_one_step"] for r in rows], "s-", c=C[3], ms=3, label="single step (n=1)")
ax[0].set_yscale("log"); ax[0].set_xlabel("capability gap G (Elo)"); ax[0].set_ylabel("P(final model aligned)")
ax[0].legend(fontsize=6.5, frameon=False)
dd = np.geomspace(2, 3000, 400)
ax[1].plot(dd, h_rate(dd, base) / base.lam, c=C[0])
ax[1].axhline(floor_rate(base) / base.lam, ls=":", c=C[2], label="floor 1-d0")
ax[1].axhline(1, ls="-.", c="gray", label="raw hazard")
ax[1].axvline(d_opt, ls="--", c=C[1], lw=0.8, label=f"optimal step {d_opt:.0f} Elo")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 3); ax[1].set_xlabel("per-step gap Delta (Elo)")
ax[1].set_ylabel("h(Delta) / lambda"); ax[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig1_rate.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ 2. P vs n
fig, ax = plt.subplots(figsize=(3.6, 2.8))
res["n_sweep"] = {}
for c, G in zip(C, [400, 1000, 2000]):
    p = replace(base, G=float(G))
    ns = np.arange(1, 121)
    v = [chain_exact(n, p)[0] for n in ns]
    ax.plot(ns, v, c=c, label=f"G={G}")
    i = int(np.argmax(v)); ax.plot(ns[i], v[i], "o", c=c)
    res["n_sweep"][G] = dict(n_star=int(ns[i]), P_star=float(v[i]), P_n1=float(v[0]), P_n120=float(v[-1]))
ax.set_xscale("log"); ax.set_xlabel("number of bootstrapping steps n"); ax.set_ylabel("P(final aligned)")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT + "fig2_steps.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ 3. property test of Theorem 1
rng = np.random.default_rng(1)
viol, worst = 0, np.inf
for t in range(3000):
    p = Params(G=float(rng.uniform(50, 5000)), lam=float(10 ** rng.uniform(-5, -2)), mu0=float(10 ** rng.uniform(-4, -0.5)),
               d0=float(rng.uniform(0.05, 0.995)), k=float(rng.uniform(0.2, 3)), gamma=1.0, kappa=1.0,
               R=int(rng.choice([1, 2, 5, R_INF])), f=float(rng.choice([0, 0.01, 0.1])))
    n = int(rng.integers(1, 200))
    S = chain_exact(n, p)[0]
    c0 = np.log((1 - p.d0 * p.mu0) / (1 - p.mu0))
    bound = p.lam * (1 - p.d0) * p.G + n * c0
    slack = -np.log(max(S, 1e-300)) - bound
    worst = min(worst, slack)
    viol += slack < -1e-9
res["theorem1_property_test"] = dict(trials=3000, violations=int(viol), min_slack=float(worst))

# ------------------------------------------------------------ 4. validation DP vs MC
val = []
for pol, fn, extra in [("chain", chain_exact, {}), ("anchored", anchored_comonotone_exact, {"f": 0.02})]:
    for R in [1, 3, R_INF]:
        for n in [5, 20]:
            p = replace(base, R=R, **extra)
            ex = fn(n, p); mc = simulate(n, p, pol, reps=200_000, seed=n + R % 97)
            val.append(dict(policy=pol, R=("inf" if R == R_INF else R), n=n, exact=[float(x) for x in ex],
                            mc=[float(x) for x in mc]))
res["validation"] = val

# ------------------------------------------------------------ 5. retries and anchoring
G = 1000.0
pG = replace(base, G=G, f=0.02)
n0 = res["n_sweep"][1000]["n_star"]
Rs = [1, 2, 3, 5, 8, 13, 20, R_INF]
retry = []
for R in Rs:
    p = replace(pG, R=R)
    ch = chain_exact(n0, p); an = anchored_comonotone_exact(n0, p)
    an_ind = simulate(n0, replace(p, rho=0.0), "anchored", reps=200_000, seed=R % 1000)
    an_toy = simulate(n0, replace(p, rho=0.44), "anchored", reps=200_000, seed=R % 1000 + 1)
    retry.append(dict(R=("inf" if R == R_INF else R), chain=list(map(float, ch)), anchored_comono=list(map(float, an)),
                      anchored_indep=list(map(float, an_ind)), anchored_rho044=list(map(float, an_toy))))
res["retry_sweep"] = dict(n=n0, G=G, f=0.02, rows=retry)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
xs = np.arange(len(Rs)); lab = [str(r["R"]) for r in retry]
for j, (key, title) in enumerate([("chain", "chain (parent only)"), ("anchored_comono", "anchored, rho=1"),
                                   ("anchored_rho044", "anchored, rho=0.44")]):
    S = np.array([r[key][0] for r in retry]); F = np.array([r[key][1] for r in retry]); A = np.array([r[key][2] for r in retry])
    ax[j].bar(xs, S, color=C[2], label="aligned (S)"); ax[j].bar(xs, A, bottom=S, color="#bbbbbb", label="abort (A)")
    ax[j].bar(xs, F, bottom=S + A, color=C[1], label="silent failure (F)")
    ax[j].set_xticks(xs); ax[j].set_xticklabels(lab, fontsize=7); ax[j].set_title(title, fontsize=8)
    ax[j].set_xlabel("max attempts per step R")
ax[0].set_ylabel("probability"); ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "fig3_retries.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ 6. dependence sweep
rhos = [0.0, 0.2, 0.44, 0.6, 0.8, 1.0]
dep = []
for rho in rhos:
    for R in [3, R_INF]:
        p = replace(pG, rho=rho, R=R)
        s = simulate(n0, p, "anchored", reps=200_000, seed=int(rho * 100) + (R % 7))
        dep.append(dict(rho=rho, R=("inf" if R == R_INF else R), S=float(s[0]), F=float(s[1]), A=float(s[2])))
    p = replace(pG, rho=rho, R=R_INF, f=0.0)
    s = simulate(n0, p, "anchored", reps=200_000, seed=int(rho * 100) + 11)
    dep.append(dict(rho=rho, R="inf_nofp", S=float(s[0]), F=float(s[1]), A=float(s[2])))
res["rho_sweep"] = dep
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for c, R in zip([C[0], C[1], C[3]], [3, "inf", "inf_nofp"]):
    rr = [d for d in dep if d["R"] == R]
    ax.plot([d["rho"] for d in rr], [d["F"] for d in rr], "o-", c=c, label=f"F, R={R}")
ax.axvline(0.44, ls=":", c="gray"); ax.text(0.45, ax.get_ylim()[1] * 0.9, "toy rho_hat", fontsize=6.5)
ax.set_xlabel("overseer error dependence rho"); ax.set_ylabel("P(silent failure)")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT + "fig4_rho.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ 7. sensitivity of the rate
d0s = np.linspace(0.5, 0.99, 25); mu0s = np.geomspace(1e-4, 0.1, 25)
H = np.zeros((len(mu0s), len(d0s))); Hlin = np.zeros_like(H)
for i, m0 in enumerate(mu0s):
    for j, d0 in enumerate(d0s):
        p = replace(base, mu0=float(m0), d0=float(d0))
        H[i, j] = optimal_rate(p)[0] / base.lam
        Hlin[i, j] = linearized_rate(p)[0] / base.lam
res["sensitivity"] = dict(max_rel_err_lin=float(np.max(np.abs(Hlin - H) / H)),
                          median_rel_err_lin=float(np.median(np.abs(Hlin - H) / H)),
                          r_over_lam_min=float(H.min()), r_over_lam_max=float(H.max()))
sens_k = []
for k in [0.25, 0.5, 1, 2, 4]:
    p = replace(base, k=k); sens_k.append(dict(k=k, r_over_lam=float(optimal_rate(p)[0] / base.lam),
                                                delta_opt=float(optimal_rate(p)[1])))
res["sensitivity"]["k"] = sens_k
sens_lam = []
for hl in [250, 500, 1000, 2000, 4000]:
    p = replace(base, lam=np.log(2) / hl); r, dlt = optimal_rate(p)
    sens_lam.append(dict(raw_half_life=hl, r_over_lam=float(r / p.lam), opt_half_life=float(np.log(2) / r), delta_opt=float(dlt)))
res["sensitivity"]["lam"] = sens_lam
fig, ax = plt.subplots(figsize=(3.8, 3.0))
cs = ax.contourf(d0s, mu0s, H, levels=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], cmap="viridis")
ax.contour(d0s, mu0s, Hlin, levels=[0.1, 0.2, 0.3, 0.5], colors="w", linewidths=0.6, linestyles="--")
ax.set_yscale("log"); ax.set_xlabel("peer detection d0"); ax.set_ylabel("per-step hazard mu0")
fig.colorbar(cs, label="r* / lambda"); fig.tight_layout(); fig.savefig(OUT + "fig5_sensitivity.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ 8. gamma / kappa (non-absorbing) sweep
gk = []
for gamma in [1.0, 0.9, 0.5, 0.0]:
    for kappa in [1.0, 0.5]:
        p = replace(base, gamma=gamma, kappa=kappa)
        n_s, P_s, _ = best_n(p, nmax=150)
        gk.append(dict(gamma=gamma, kappa=kappa, n_star=n_s, S=float(P_s), F=float(chain_exact(n_s, p)[1])))
res["gamma_kappa"] = gk

json.dump(res, open(OUT + "theory_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
