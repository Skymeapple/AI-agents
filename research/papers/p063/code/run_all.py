"""Runs every experiment in the paper. Usage:  python3 run_all.py   (about 2-5 min on one CPU core)
Outputs go to ../results/ (JSON, CSV, PNG)."""
import json, os, time, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY, RED = "#2a6fb0", "#d9822b", "#3a9e5c", "#888888", "#c0392b"
OUT = {}


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)


def channels(pop, gamma_b=0.5, kappa=0.43, eta=0.8):
    y = pop["y"]; o = np.argsort(y, kind="stable")
    zL, a = M.labor_gain(pop, gamma_b, kappa)
    zK, zC = M.capital_gain(pop), M.consumer_gain(pop, eta)
    d = dict(G=M.gini(y), C_L=M.conc(zL, y, o), C_K=M.conc(zK, y, o), C_C=M.conc(zC, y, o),
             C_lab=M.conc(pop["lab"], y, o), G_L=M.gini(zL), G_K=M.gini(zK), G_C=M.gini(zC),
             top10=M.top_share(y), T10_L=M.top_share_of_z(zL, y, 0.1, o),
             T10_K=M.top_share_of_z(zK, y, 0.1, o), T10_C=M.top_share_of_z(zC, y, 0.1, o))
    B, W = M.decomposition_BW(pop, a, o)
    d.update(B=B, W=W)
    d["piL_star_2ch"] = (d["C_K"] - d["G"]) / (d["C_K"] - d["C_L"])
    d["piL_star_top10_2ch"] = (d["T10_K"] - d["top10"]) / (d["T10_K"] - d["T10_L"])
    return d, dict(L=zL, K=zK, C=zC, order=o, a=a)


def piL_star(d, piC):
    """Gini-neutral labor share given consumer share piC (owners get the rest)."""
    return (d["C_K"] * (1 - piC) + piC * d["C_C"] - d["G"]) / (d["C_K"] - d["C_L"])


def piC_star(d, piL):
    """Consumer (price) share needed for Gini neutrality given worker share piL (owners get the rest)."""
    return (d["C_K"] - d["G"] - piL * (d["C_K"] - d["C_L"])) / (d["C_K"] - d["C_C"])


# =============================================================== E1 baseline
t0 = time.time()
BASE = dict(N=100_000, G_lab=0.45, between_share=0.4, K_occ=40, u=0.15, rho=0.5, S_K=0.20, S_T=0.08, seed=0)
pop = M.make_population(**BASE)
d0, Z = channels(pop)
d0["G_lab_earners_realised"] = M.gini(pop["lab"][pop["earner"]])
d0["piL_star_piC_0.3"] = piL_star(d0, 0.3)
d0["piL_star_piC_0.5"] = piL_star(d0, 0.5)
d0["piC_star_at_piL_0.07"] = piC_star(d0, 0.07)
d0["piC_star_at_piL_0.30"] = piC_star(d0, 0.30)
OUT["E1_baseline"] = {k: float(v) for k, v in d0.items()}
OUT["E1_baseline"]["params"] = BASE
print("E1", {k: round(float(v), 4) for k, v in d0.items()})

# Lorenz-dominance test at a few splits
grid = np.arange(pop["y"].size + 1) / pop["y"].size   # every household (exact piecewise-linear check)
Ly = M.lorenz(pop["y"], grid)
ld = {}
for piL in [0.07, 0.30, 0.60, 0.80, 0.95]:
    z = piL * Z["L"] + (1 - piL) * Z["K"]
    cc = M.conc_curve(z, pop["y"], grid, Z["order"])
    ld[str(piL)] = dict(C_z=float(M.conc(z, pop["y"], Z["order"])),
                        conc_curve_below_lorenz_everywhere=bool(np.all(cc[1:-1] <= Ly[1:-1] + 1e-12)),
                        conc_curve_above_lorenz_everywhere=bool(np.all(cc[1:-1] >= Ly[1:-1] - 1e-12)))
OUT["E1_lorenz_tests"] = ld

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.plot(grid, grid, color=GREY, lw=0.8, ls=":")
ax.plot(grid, Ly, color="black", lw=1.6, label=f"Lorenz curve of income (G={d0['G']:.2f})")
ax.plot(grid, M.conc_curve(Z["L"], pop["y"], grid, Z["order"]), color=BLUE, label=f"Worker-gain channel (C={d0['C_L']:.2f})")
ax.plot(grid, M.conc_curve(Z["C"], pop["y"], grid, Z["order"]), color=GREEN, label=f"Consumer-price channel (C={d0['C_C']:.2f})")
ax.plot(grid, M.conc_curve(Z["K"], pop["y"], grid, Z["order"]), color=ORANGE, label=f"Owner channel (C={d0['C_K']:.2f})")
ax.set_xlabel("Households ranked by pre-AI income (cumulative share)")
ax.set_ylabel("Cumulative share")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_concentration_curves.png")); plt.close(fig)

# Neutrality simplex
fig, ax = plt.subplots(figsize=(4.2, 3.8))
pl = np.linspace(0, 1, 401)
PL, PK = np.meshgrid(pl, pl)
PC = 1 - PL - PK
mask = PC < -1e-9
Cz = PL * d0["C_L"] + PK * d0["C_K"] + PC * d0["C_C"]
Gz_up = np.full_like(Cz, np.nan)  # upper-bound sufficient statistic computed on coarse grid below
region = np.where(Cz >= d0["G"], 1.0, 0.0)
region[mask] = np.nan
ax.contourf(PL, PK, region, levels=[-0.5, 0.5, 1.5], colors=["#d6e6f5", "#f7dcc4"])
ax.contour(PL, PK, np.where(mask, np.nan, Cz - d0["G"]), levels=[0], colors="black", linewidths=1.4)
# ambiguity zone: C_z < G but Gini of gain vector > G (large shocks could raise G)
coarse = np.linspace(0, 1, 41)
amb_pts = []
for a_ in coarse:
    for b_ in coarse:
        c_ = 1 - a_ - b_
        if c_ < -1e-9:
            continue
        z = a_ * Z["L"] + b_ * Z["K"] + c_ * Z["C"]
        if a_ * d0["C_L"] + b_ * d0["C_K"] + c_ * d0["C_C"] < d0["G"] and M.gini(z) > d0["G"]:
            amb_pts.append((a_, b_))
if amb_pts:
    amb_pts = np.array(amb_pts)
    ax.scatter(amb_pts[:, 0], amb_pts[:, 1], s=3, color=GREY, label="falls at first order, but bound does not\nrule out a rise for large shocks")
ax.plot([0, 1], [1, 0], color=GREY, lw=0.8)
ax.text(0.08, 0.55, "Gini rises for ANY\nshock size", fontsize=8)
ax.text(0.62, 0.05, "Gini falls\n(first order)", fontsize=8)
ax.axvline(0.30, color=RED, ls="--", lw=0.8); ax.text(0.31, 0.92, "KPWZ\n~0.30", fontsize=7, color=RED)
ax.axvline(0.05, color=RED, ls=":", lw=0.8); ax.text(0.06, 0.80, "HV\n0.03-0.07", fontsize=7, color=RED)
ax.set_xlabel(r"$\pi_L$: share of AI surplus to workers (wages)")
ax.set_ylabel(r"$\pi_K$: share to owners (profits)")
ax.set_title(r"Remainder $\pi_C=1-\pi_L-\pi_K$ to consumers via prices", fontsize=8)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_neutrality_simplex.png")); plt.close(fig)
OUT["E1_ambiguity_zone_points"] = int(len(amb_pts))
print("E1 done", round(time.time() - t0, 1), "s")

# =============================================================== E2 bounds validation
t0 = time.time()
rng = np.random.default_rng(1)
viol_lo, viol_hi, n_checks = 0.0, 0.0, 0
for rep in range(300):
    n = int(rng.integers(20, 600))
    y = rng.lognormal(0, rng.uniform(0.2, 1.5), n)
    kind = rep % 3
    if kind == 0:
        z = rng.lognormal(0, rng.uniform(0.2, 2.5), n)
    elif kind == 1:
        z = rng.normal(0.2, 1.0, n); z = z - z.mean() + abs(rng.normal(0.5, 0.2))  # signed gains, Z > 0
    else:
        z = y ** rng.uniform(-1, 3) * rng.lognormal(0, 0.5, n)
    Y, Zs = y.sum(), z.sum()
    G, C, Hz = M.gini(y), M.conc(z, y), M.gz_upper(z)
    for eps in np.geomspace(1e-3, 20, 25):
        yy = y + eps * z
        if yy.sum() <= 0:
            continue
        g = M.gini(yy)
        lo = (Y * G + eps * Zs * C) / (Y + eps * Zs)
        hi = (Y * G + eps * Zs * Hz) / (Y + eps * Zs)
        viol_lo = max(viol_lo, lo - g); viol_hi = max(viol_hi, g - hi); n_checks += 1
OUT["E2_random_bound_checks"] = dict(n_checks=n_checks, max_lower_violation=viol_lo, max_upper_violation=viol_hi)
print("E2 random checks", OUT["E2_random_bound_checks"])

# On calibrated population: exact vs bounds for three splits; eps = aggregate gain as share of income
y = pop["y"]; Y = y.sum()
eps_share = np.linspace(0, 1.0, 26)   # total gain / pre-AI total income
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=False)
e2 = {}
for ax, (lab, piL, piK) in zip(axes, [("Owner-heavy (pi_L=0.30, pi_K=0.70)", 0.30, 0.70),
                                       ("Near-neutral (pi_L=pi_L*, pi_K=1-pi_L*)", d0["piL_star_2ch"], 1 - d0["piL_star_2ch"]),
                                       ("Worker/consumer-heavy (0.6, 0.0, C 0.4)", 0.6, 0.0)]):
    z = piL * Z["L"] + piK * Z["K"] + (1 - piL - piK) * Z["C"]
    z = z * Y                            # sum(z) = Y, so eps = aggregate gain share
    exact = np.array([M.gini(y + e * z) for e in eps_share])
    lo, hi, fo = M.bounds(y, z, eps_share)
    ax.plot(eps_share, exact, color="black", lw=1.6, label="exact Gini")
    ax.plot(eps_share, lo, color=BLUE, ls="--", label="lower bound (Thm 1)")
    ax.plot(eps_share, hi, color=ORANGE, ls="--", label="upper bound (Thm 1)")
    ax.plot(eps_share, fo, color=GREY, ls=":", label="first-order (Lerman-Yitzhaki)")
    ax.axhline(d0["G"], color=GREY, lw=0.5)
    ax.set_title(lab, fontsize=7.5); ax.set_xlabel("aggregate AI gain / pre-AI income")
    e2[lab] = dict(eps=eps_share.tolist(), exact=exact.tolist(), lower=lo.tolist(), upper=hi.tolist(), first_order=fo.tolist())
axes[0].set_ylabel("Gini of household income"); axes[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_bounds_validation.png")); plt.close(fig)
save("E2_bounds_calibrated.json", e2)
neutral = e2["Near-neutral (pi_L=pi_L*, pi_K=1-pi_L*)"]
OUT["E2_convexity_premium_at_neutral"] = {f"{e:.2f}": float(g - d0["G"]) for e, g in zip(neutral["eps"], neutral["exact"]) if np.any(np.isclose(e, [0.04, 0.2, 0.4, 1.0]))}
OUT["E2_endpoints"] = {k: dict(G_exact_at_eps1=v["exact"][-1], lower=v["lower"][-1], upper=v["upper"][-1], first_order=v["first_order"][-1]) for k, v in e2.items()}
print("E2 done", round(time.time() - t0, 1), "s")

# =============================================================== E3 within vs between
t0 = time.time()
gammas = np.linspace(-1.0, 2.0, 25)
kappas = np.linspace(0.0, 0.95, 20)
CL = np.zeros((len(kappas), len(gammas))); BB = np.zeros_like(CL); WW = np.zeros_like(CL)
o = Z["order"]; C_lab = d0["C_lab"]
for i, k in enumerate(kappas):
    for j, g in enumerate(gammas):
        zL, a = M.labor_gain(pop, g, k)
        CL[i, j] = M.conc(zL, pop["y"], o)
        BB[i, j], WW[i, j] = M.decomposition_BW(pop, a, o)
np.savez(os.path.join(RES, "E3_grid.npz"), gammas=gammas, kappas=kappas, C_L=CL, B=BB, W=WW)
# identity check
ident_err = float(np.max(np.abs((CL - C_lab) - (BB + WW))))
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
im = axes[0].imshow(CL - C_lab, origin="lower", aspect="auto", cmap="RdBu_r",
                    extent=[gammas[0], gammas[-1], kappas[0], kappas[-1]], vmin=-0.25, vmax=0.25)
axes[0].contour(gammas, kappas, CL - C_lab, levels=[0], colors="black", linewidths=1.2)
axes[0].contour(gammas, kappas, CL - d0["G"], levels=[0], colors="black", linestyles="--", linewidths=1.0)
axes[0].axhline(0.43, color=GREEN, lw=0.8); axes[0].text(gammas[0] + 0.05, 0.45, "Dell'Acqua et al. ratio 43/17", fontsize=6.5, color=GREEN)
axes[0].set_xlabel(r"between-occupation exposure gradient $\gamma_b$")
axes[0].set_ylabel(r"within-occupation compression $\kappa$")
axes[0].set_title(r"$C_L - C_\ell$ (solid: 0; dashed: $C_L=G$)", fontsize=8)
fig.colorbar(im, ax=axes[0], fraction=0.046)
jk = np.argmin(abs(kappas - 0.43))
axes[1].plot(gammas, BB[jk], color=ORANGE, label="B (between occupations)")
axes[1].plot(gammas, WW[jk], color=BLUE, label="W (within occupations)")
axes[1].plot(gammas, BB[jk] + WW[jk], color="black", label=r"$C_L - C_\ell$ = B + W")
axes[1].axhline(0, color=GREY, lw=0.5)
axes[1].set_xlabel(r"$\gamma_b$ (at $\kappa$=0.43)"); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_within_between.png")); plt.close(fig)
# find gamma at which C_L = C_lab and C_L = G for kappa=0.43 and kappa=0.9
def crossing(row, target):
    s = np.sign(row - target)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]; x0, x1 = gammas[i], gammas[i + 1]; y0, y1 = row[i] - target, row[i + 1] - target
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))
e3 = dict(identity_max_abs_error=ident_err)
for kv in (0.0, 0.43, 0.9):
    i = np.argmin(abs(kappas - kv))
    e3[f"kappa={kappas[i]:.2f}"] = dict(gamma_where_CL_eq_Clab=crossing(CL[i], C_lab), gamma_where_CL_eq_G=crossing(CL[i], d0["G"]),
                                        CL_at_gamma0=float(np.interp(0, gammas, CL[i])), CL_at_gamma1=float(np.interp(1, gammas, CL[i])))
OUT["E3"] = e3
print("E3", e3, round(time.time() - t0, 1), "s")

# =============================================================== E4 Monte Carlo over priors
t0 = time.time()
PRI = dict(G_lab=(0.38, 0.55), between_share=(0.25, 0.55), u=(0.05, 0.25), rho=(0.2, 0.8), S_K=(0.10, 0.30),
           S_T=(0.03, 0.15), gamma_b=(-0.5, 1.5), kappa=(0.0, 0.9), eta=(0.5, 1.0))
R = 400
rng = np.random.default_rng(2)
rows = []
for r in range(R):
    th = {k: rng.uniform(*v) for k, v in PRI.items()}
    p = M.make_population(N=40_000, G_lab=th["G_lab"], between_share=th["between_share"], u=th["u"], rho=th["rho"],
                          S_K=th["S_K"], S_T=th["S_T"], seed=1000 + r)
    d, ZZ = channels(p, th["gamma_b"], th["kappa"], th["eta"])
    z30 = 0.3 * ZZ["L"] + 0.7 * ZZ["K"]
    gridc = np.arange(p["y"].size + 1) / p["y"].size
    lorenz_dom = bool(np.all(M.conc_curve(z30, p["y"], gridc, ZZ["order"])[1:-1] <= M.lorenz(p["y"], gridc)[1:-1] + 1e-12))
    rows.append(dict(**th, G=d["G"], C_L=d["C_L"], C_K=d["C_K"], C_C=d["C_C"], C_lab=d["C_lab"], B=d["B"], W=d["W"],
                     piL_star=d["piL_star_2ch"], piL_star_piC03=piL_star(d, 0.3), piL_star_piC05=piL_star(d, 0.5),
                     piL_star_top10=d["piL_star_top10_2ch"], lorenz_dominated_at_piL_0p3=lorenz_dom,
                     piC_star_at_piL_0p07=piC_star(d, 0.07), piC_star_at_piL_0p30=piC_star(d, 0.30)))
keys = list(rows[0].keys())
with open(os.path.join(RES, "E4_montecarlo.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
A = {k: np.array([r_[k] for r_ in rows], float) for k in keys}
def q(x):
    return dict(p5=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()))
e4 = dict(R=R, priors=PRI, piL_star=q(A["piL_star"]), piL_star_piC03=q(A["piL_star_piC03"]),
          piL_star_piC05=q(A["piL_star_piC05"]), piL_star_top10=q(A["piL_star_top10"]),
          C_K=q(A["C_K"]), C_L=q(A["C_L"]), G=q(A["G"]),
          share_piLstar_above_0p30=float(np.mean(A["piL_star"] > 0.30)),
          share_piLstar_above_0p50=float(np.mean(A["piL_star"] > 0.50)),
          share_piLstar_piC03_above_0p30=float(np.mean(A["piL_star_piC03"] > 0.30)),
          share_piLstar_piC05_above_0p30=float(np.mean(A["piL_star_piC05"] > 0.30)),
          share_CL_below_G=float(np.mean(A["C_L"] < A["G"])),
          share_CL_below_Clab=float(np.mean(A["C_L"] < A["C_lab"])),
          share_W_negative=float(np.mean(A["W"] < 0)),
          share_lorenz_dominated_at_piL_0p3=float(np.mean(A["lorenz_dominated_at_piL_0p3"])),
          share_piLstar_above_1=float(np.mean(A["piL_star"] > 1)),
          piC_star_at_piL_0p07=q(A["piC_star_at_piL_0p07"]), piC_star_at_piL_0p30=q(A["piC_star_at_piL_0p30"]))
# standardized regression coefficients for piL_star
Xn = np.column_stack([(A[k] - A[k].mean()) / A[k].std() for k in PRI])
yv = (A["piL_star"] - A["piL_star"].mean()) / A["piL_star"].std()
beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(R), Xn]), yv, rcond=None)
pred = np.column_stack([np.ones(R), Xn]) @ beta
e4["src_piL_star"] = {k: float(b) for k, b in zip(PRI, beta[1:])}
e4["src_R2"] = float(1 - np.sum((yv - pred) ** 2) / np.sum(yv ** 2))
OUT["E4"] = e4
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
bins = np.linspace(0, 1.2, 49)
axes[0].hist(A["piL_star"], bins=bins, color=ORANGE, alpha=0.75, label=r"$\pi_C=0$ (labor vs owners)")
axes[0].hist(A["piL_star_piC03"], bins=bins, color=GREEN, alpha=0.6, label=r"$\pi_C=0.3$ to consumers")
for v, ls, t, h in [(0.05, ":", "HV\n0.03-0.07", 0.92), (0.30, "--", "KPWZ\n~0.30", 0.75)]:
    axes[0].axvline(v, color=RED, ls=ls, lw=0.9); axes[0].text(v + 0.01, axes[0].get_ylim()[1] * h, t, fontsize=6.5, color=RED)
axes[0].set_title(f"draws with value > 1.2 not shown ({int(np.sum(A['piL_star'] > 1.2))} of {R})", fontsize=7)
axes[0].set_xlabel(r"Gini-neutral worker share $\pi_L^*$"); axes[0].set_ylabel("parameter draws")
axes[0].legend(frameon=False, fontsize=7, loc="center right")
order_k = np.argsort(np.abs(beta[1:]))
axes[1].barh(np.array(list(PRI))[order_k], beta[1:][order_k], color=[BLUE if b < 0 else ORANGE for b in beta[1:][order_k]])
axes[1].set_xlabel(r"standardised regression coefficient on $\pi_L^*$" + f"  (R²={e4['src_R2']:.2f})")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_montecarlo.png")); plt.close(fig)
print("E4", json.dumps({k: v for k, v in e4.items() if k not in ('priors',)}, indent=0)[:1500], round(time.time() - t0, 1), "s")

# =============================================================== E5 DFA history
t0 = time.time()
hist = []
with open(os.path.join(DATA, "dfa_equity_shares.csv")) as f:
    for r_ in csv.DictReader(f):
        sh = dict(bottom50=float(r_["bottom50"]), p50_90=float(r_["p50_90"]), p90_99=float(r_["p90_99"]), top1=float(r_["top1"]))
        p = M.make_population(**{**BASE, "N": 60_000}, shares=sh)
        d, _ = channels(p)
        hist.append(dict(t=int(r_["year"]) + (int(r_["quarter"]) - 1) / 4, top1=sh["top1"], C_K=d["C_K"], G=d["G"],
                         C_L=d["C_L"], piL_star=d["piL_star_2ch"], piL_star_piC03=piL_star(d, 0.3)))
with open(os.path.join(RES, "E5_dfa_history.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
H = {k: np.array([h[k] for h in hist]) for k in hist[0]}
OUT["E5"] = dict(first=hist[0], last=hist[-1], min_piL_star=float(H["piL_star"].min()), t_min=float(H["t"][H["piL_star"].argmin()]),
                 max_piL_star=float(H["piL_star"].max()), t_max=float(H["t"][H["piL_star"].argmax()]),
                 corr_top1_piL_star=float(np.corrcoef(H["top1"], H["piL_star"])[0, 1]))
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(H["t"], H["piL_star"], color=ORANGE, label=r"$\pi_L^*$ ($\pi_C=0$)")
ax.plot(H["t"], H["piL_star_piC03"], color=GREEN, label=r"$\pi_L^*$ ($\pi_C=0.3$)")
ax.set_ylabel("Gini-neutral worker share"); ax.set_xlabel("quarter (equity distribution from Fed DFA)")
ax2 = ax.twinx(); ax2.plot(H["t"], H["top1"], color=GREY, lw=0.8, ls="--"); ax2.set_ylabel("top-1% equity share (%, dashed)", color=GREY)
ax2.spines["top"].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig6_dfa_history.png")); plt.close(fig)
print("E5", OUT["E5"]["first"], OUT["E5"]["last"], round(time.time() - t0, 1), "s")

# =============================================================== E6 dynamics
t0 = time.time()
def simulate(pop0, piL, T=20, g=0.02, saving="equal", gamma_b=0.5, kappa=0.43):
    lab = pop0["lab"].copy(); cap = pop0["cap"].copy(); trans = pop0["trans"].copy()
    eq = pop0["eq"].copy()
    N = lab.size
    wpct = (np.argsort(np.argsort(eq)) + 0.5) / N
    s = np.where(wpct >= 0.99, 0.5, np.where(wpct >= 0.9, 0.3, np.where(wpct >= 0.5, 0.15, 0.05))) if saving == "rising" else np.full(N, 0.2)
    Gs, piS, capshare, top1own = [], [], [], []
    p = dict(pop0)
    for t in range(T + 1):
        y = lab + cap + trans
        p.update(y=y, lab=lab, eq=eq)
        d, ZZ = channels(p, gamma_b, kappa)
        Gs.append(d["G"]); piS.append(d["piL_star_2ch"]); capshare.append(cap.sum() / y.sum())
        top1own.append(np.sort(eq)[-N // 100:].sum() / eq.sum())
        if t == T:
            break
        gain = g * y.sum()
        lab = lab + piL * gain * ZZ["L"]
        prof = (1 - piL) * gain * ZZ["K"]
        cap = cap + prof                       # permanent profit stream to current owners (returns to wealth rise)
        # reinvestment: saved capital income buys new equity at a price proportional to capital income per share
        newsh = s * cap / (cap.sum() / eq.sum())
        newsh *= 0.05 / max(newsh.sum() / eq.sum(), 1e-12) if saving != "none" else 0.0  # 5% net issuance per period, allocated by savings
        eq = eq + newsh
        cap = eq * (cap.sum() / eq.sum())      # capital income follows ownership
    return dict(G=Gs, piL_star=piS, cap_share=capshare, top1_own=top1own)
pop6 = M.make_population(**{**BASE, "N": 60_000})
d6, _ = channels(pop6)
pstar = d6["piL_star_2ch"]
sims = {}
for name, piL, sv in [("piL=pi*_0, equal saving", pstar, "equal"), ("piL=pi*_0, saving rising in wealth", pstar, "rising"),
                      ("piL=0.30, equal saving", 0.30, "equal"), ("piL=0.30, saving rising in wealth", 0.30, "rising")]:
    sims[name] = simulate(pop6, piL, saving=sv)
save("E6_dynamics.json", dict(pi_star0=pstar, sims=sims))
OUT["E6"] = dict(pi_star0=pstar, **{k: dict(G0=v["G"][0], G20=v["G"][-1], piL_star20=v["piL_star"][-1],
                                           cap_share0=v["cap_share"][0], cap_share20=v["cap_share"][-1],
                                           top1_own0=v["top1_own"][0], top1_own20=v["top1_own"][-1]) for k, v in sims.items()})
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
cols = [BLUE, GREEN, ORANGE, RED]
for c, (k, v) in zip(cols, sims.items()):
    axes[0].plot(v["G"], color=c, label=k); axes[1].plot(v["piL_star"], color=c)
axes[0].set_xlabel("period (2% AI gain per period)"); axes[0].set_ylabel("Gini"); axes[0].legend(frameon=False, fontsize=6.5)
axes[1].set_xlabel("period"); axes[1].set_ylabel(r"current Gini-neutral $\pi_L^*$")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig7_dynamics.png")); plt.close(fig)
print("E6", json.dumps(OUT["E6"], indent=0)[:1200], round(time.time() - t0, 1), "s")

# =============================================================== E7 broad ownership of AI capital
t0 = time.time()
bs = np.linspace(0, 1, 21)
e7 = {}
for mode in ("equal_per_capita", "prop_to_labor"):
    vals = []
    for b in bs:
        broad = np.full(pop["y"].size, 1.0 / pop["y"].size) if mode == "equal_per_capita" else pop["lab"] / pop["lab"].sum()
        zK = (1 - b) * Z["K"] + b * broad
        CK = M.conc(zK, pop["y"], Z["order"])
        vals.append((CK - d0["G"]) / (CK - d0["C_L"]) if CK > d0["C_L"] else float("nan"))
    e7[mode] = dict(b=bs.tolist(), piL_star=vals)
    # smallest b at which owner channel itself is Gini-neutral or better (pi_L* <= 0)
    CKs = [M.conc((1 - b) * Z["K"] + b * (np.full(pop["y"].size, 1.0 / pop["y"].size) if mode == "equal_per_capita" else pop["lab"] / pop["lab"].sum()), pop["y"], Z["order"]) for b in bs]
    e7[mode]["C_K"] = [float(c) for c in CKs]
    idx = [i for i, c in enumerate(CKs) if c <= d0["G"]]
    e7[mode]["min_b_owner_channel_neutral"] = float(bs[idx[0]]) if idx else None
OUT["E7"] = {m: dict(min_b_owner_channel_neutral=v["min_b_owner_channel_neutral"],
                     piL_star_at_b0p25=v["piL_star"][5], piL_star_at_b0p5=v["piL_star"][10]) for m, v in e7.items()}
save("E7_broad_ownership.json", e7)
fig, ax = plt.subplots(figsize=(4.2, 2.9))
ax.plot(bs, e7["equal_per_capita"]["piL_star"], color=BLUE, label="broad stake equal per capita")
ax.plot(bs, e7["prop_to_labor"]["piL_star"], color=ORANGE, label="broad stake proportional to earnings")
ax.axhline(0.30, color=RED, ls="--", lw=0.8); ax.axhline(0.05, color=RED, ls=":", lw=0.8)
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel("share of AI profits accruing to broadly held stake, b")
ax.set_ylabel(r"Gini-neutral worker share $\pi_L^*$")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig8_broad_ownership.png")); plt.close(fig)
print("E7", OUT["E7"], round(time.time() - t0, 1), "s")

save("summary.json", OUT)
print("all done")
