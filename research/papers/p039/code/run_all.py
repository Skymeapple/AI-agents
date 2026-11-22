"""Run every experiment in the paper. Usage:  python3 code/run_all.py
Outputs: results/*.json, results/*.csv, results/fig*.png. Seeded; ~1-3 min CPU."""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import *  # noqa

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(20261122)
T0 = time.time()
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})

# ---------------- sourced inputs (see paper Table 1) ----------------
VSL = 13.7e6            # US DOT VSL, 2024 base year
N = 8.2e9               # UN WPP 2024, world population 2024
r, H = 0.03, 75.0       # discount rate; horizon 2026-2100 (catastrophe timing uniform) [assumption]
D = (1 - np.exp(-r * H)) / (r * H)
L_BASE = VSL * N * D    # present-value loss, people alive today only, US VSL
L_LOW, L_HIGH = L_BASE / 10, VSL * N
ANN = (1 - 1.03 ** -25) / 0.03                 # 25-year annuity factor at 3%
S_CUR = 189e6 * ANN                            # McAleese (2025 update): ~$189M/yr in 2024
P_XPT_SF, P_XPT_EX = 0.0038, 0.03              # XPT medians, AI extinction by 2100
G_MED, G_MEAN = 0.05, 0.162                    # Grace et al. extinction/disempowerment Q
K0, B0 = 1e9, 0.3                              # illustrative tractability point [assumption]
out = {"inputs": dict(VSL=VSL, N=N, discount_factor=D, L_base=L_BASE, L_low=L_LOW, L_high=L_HIGH,
                      annuity=ANN, S_cur_PV=S_CUR, k0=K0, beta0=B0)}

# ---------------- E0: verify closed forms ----------------
errs = []
for _ in range(300):
    p = 10 ** rng.uniform(-6, -0.3); L = 10 ** rng.uniform(15, 17.1)
    k = 10 ** rng.uniform(7, 12); b = 10 ** rng.uniform(-1.5, 0.5); m = rng.uniform(0.05, 1)
    f = lambda u: cost(np.expm1(u), p, L, k, b, m)
    res = minimize_scalar(f, bounds=(0, np.log1p(1e16)), method="bounded", options={"xatol": 1e-10})
    s_num = np.expm1(res.x); s_cf = float(s_star(p, L, k, b, m))
    c_num = min(f(res.x), cost(0.0, p, L, k, b, m)); c_cf = float(c_star(p, L, k, b, m))
    errs.append(dict(rel_cost_err=abs(c_num - c_cf) / c_cf,
                     cf_not_worse=bool(c_cf <= c_num * (1 + 1e-9))))
out["E0_verification"] = dict(n=len(errs), max_rel_cost_err=max(e["rel_cost_err"] for e in errs),
                              all_closed_form_at_least_as_good=all(e["cf_not_worse"] for e in errs))

# ---------------- E1: spending vs belief ----------------
ps = np.geomspace(1e-9, 0.5, 400)
fig, ax = plt.subplots(figsize=(6.4, 4))
E1 = {}
for i, b in enumerate([0.1, 0.3, 1.0]):
    S = s_star(ps, L_BASE, K0, b)
    ax.loglog(ps, np.maximum(S, 1e5) / ANN, color=C[i], lw=2, label=f"β = {b}")
    hi = s_star(P_XPT_EX, L_BASE, K0, b); lo = s_star(P_XPT_SF, L_BASE, K0, b)
    gm = s_star(G_MEAN, L_BASE, K0, b)
    E1[str(b)] = dict(S_sf_PV=float(lo), S_ex_PV=float(hi), S_gmean_PV=float(gm),
                      ratio_ex_over_sf=float(hi / lo), belief_ratio=P_XPT_EX / P_XPT_SF,
                      p_threshold=float(p_threshold(L_BASE, K0, b)),
                      elasticity_theory=1 / (1 + b),
                      elasticity_numeric=float(np.log(hi / lo) / np.log(P_XPT_EX / P_XPT_SF)),
                      residual_risk_sf=float(P_XPT_SF * g(lo, K0, b)),
                      residual_risk_ex=float(P_XPT_EX * g(hi, K0, b)))
ax.axhline(189e6, color="#52514e", ls="--", lw=1)
ax.text(2e-9, 189e6 * 1.4, "est. 2024 AI-safety spend ($189M/yr)", color="#52514e", fontsize=8)
for pv, lab in [(P_XPT_SF, "XPT superforecasters"), (P_XPT_EX, "XPT experts"), (G_MEAN, "AI-author mean")]:
    ax.axvline(pv, color="#9a9994", lw=0.8, ls=":")
    ax.text(pv * 1.1, 2e5 * 1.5, lab, rotation=90, fontsize=7, color="#52514e", va="bottom")
ax.set_xlabel("baseline loss-of-control probability p (by 2100)")
ax.set_ylabel("optimal spend, annualised $/yr (25 yr)")
ax.set_ylim(1e5, 1e13); ax.legend(frameon=False, loc="upper left")
ax.set_title("Optimal spending vs belief (k = $1B, L = baseline)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_spend_vs_belief.png"), dpi=200); plt.close(fig)
out["E1_spend_vs_belief"] = E1

# ---------------- belief distributions ----------------
mu_g, sg_g = fit_logitnormal(G_MED, G_MEAN)
pg, wg = ln_nodes(mu_g, sg_g)
# XPT two-group mixture: group medians from XPT, within-group dispersion borrowed from the Grace fit [assumption]
_ps, _ws = ln_nodes(logit(P_XPT_SF), sg_g); _pe, _we = ln_nodes(logit(P_XPT_EX), sg_g)
px = np.concatenate([_ps, _pe]); wx = np.concatenate([_ws, _we]) / 2
beliefs = {"AI-authors (logit-normal fit)": (pg, wg),
           "XPT two-group mixture": (px, wx),
           "XPT two-point (medians)": (np.array([P_XPT_SF, P_XPT_EX]), np.array([0.5, 0.5]))}
out["belief_fit"] = dict(grace_mu=float(mu_g), grace_sigma=float(sg_g),
                         grace_q10=float(expit(mu_g - 1.2816 * sg_g)), grace_q90=float(expit(mu_g + 1.2816 * sg_g)),
                         grace_share_ge_10pct=float(1 - 0.5 * (1 + __import__("math").erf((logit(0.1) - mu_g) / sg_g / 2 ** 0.5))))

# ---------------- E2: pooling ----------------
E2 = {}
for name, (p, w) in beliefs.items():
    pl = pools(p, w); E2[name] = {"pooled_p": {k_: float(v) for k_, v in pl.items()}, "by_beta": {}}
    for b in [0.1, 0.3, 1.0]:
        Ss = {k_: float(s_star(v, L_BASE, K0, b)) for k_, v in pl.items()}
        Ss["mean_of_individual_optima"] = float(np.sum(w / w.sum() * s_star(p, L_BASE, K0, b)))
        Ss["ratio_linear_over_logodds"] = Ss["linear"] / Ss["log_odds"]
        Ss["belief_ratio_linear_over_logodds"] = pl["linear"] / pl["log_odds"]
        E2[name]["by_beta"][str(b)] = Ss
out["E2_pooling"] = E2
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
labels = ["log_odds", "median", "mean_of_individual_optima", "linear"]
nice = ["log-odds pool", "median", "mean of individual optima", "linear pool"]
for j, name in enumerate(["AI-authors (logit-normal fit)", "XPT two-group mixture"]):
    ax = axs[j]
    for i, b in enumerate([0.1, 0.3, 1.0]):
        vals = [E2[name]["by_beta"][str(b)][l] / ANN for l in labels]
        ax.bar(np.arange(4) + (i - 1) * 0.26, vals, width=0.24, color=C[i], label=f"β = {b}")
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(nice, rotation=20, fontsize=8)
    ax.set_title(name, fontsize=9)
axs[0].set_ylabel("optimal spend, $/yr"); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_pooling.png"), dpi=200); plt.close(fig)

# ---------------- E3: robust decision rules ----------------
E3 = {}
for lab, pset in {"XPT medians [0.0038,0.03]": [P_XPT_SF, P_XPT_EX],
                  "AI-authors 10-90% [q10,q90]": [out["belief_fit"]["grace_q10"], out["belief_fit"]["grace_q90"]]}.items():
    E3[lab] = {}
    for b in [0.1, 0.3, 1.0]:
        pgrid = np.geomspace(pset[0], pset[1], 60)
        smr, worst = minimax_regret_spend(pgrid, L_BASE, K0, b)
        E3[lab][str(b)] = dict(S_low=float(s_star(pset[0], L_BASE, K0, b)),
                               S_midpoint_linear=float(s_star(np.mean(pset), L_BASE, K0, b)),
                               S_minimax_regret=smr, max_regret=worst,
                               S_maxmin=float(maxmin_spend(pset, L_BASE, K0, b)),
                               regret_of_low_at_high=float(cost(s_star(pset[0], L_BASE, K0, b), pset[1], L_BASE, K0, b) - c_star(pset[1], L_BASE, K0, b)),
                               regret_of_high_at_low=float(cost(s_star(pset[1], L_BASE, K0, b), pset[0], L_BASE, K0, b) - c_star(pset[0], L_BASE, K0, b)))
out["E3_robust"] = E3

# ---------------- E4: value of information ----------------
E4 = {}
for name, (p, w) in beliefs.items():
    E4[name] = {}
    for b in [0.1, 0.3, 1.0]:
        v, c0, s0 = evpi(p, w, L_BASE, K0, b)
        E4[name][str(b)] = dict(EVPI_PV=v, EVPI_annualised=v / ANN, expected_cost_no_info=c0,
                                S_at_pbar=s0, EVPI_over_S=v / s0, EVPI_over_cost=v / c0,
                                lognormal_formula_share=float(evpi_share_lognormal(sg_g, b)))
out["E4_voi"] = E4
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
sig = np.linspace(0, 4, 200)
for i, b in enumerate([0.1, 0.3, 1.0, 3.0]):
    axs[0].plot(sig, evpi_share_lognormal(sig, b), color=C[i], lw=2, label=f"β = {b}")
axs[0].axvline(sg_g, color="#9a9994", ls=":", lw=1)
axs[0].text(sg_g + 0.05, 0.02, "AI-author fit σ", fontsize=7, color="#52514e")
axs[0].set_xlabel("dispersion of beliefs σ (log scale)"); axs[0].set_ylabel("EVPI / optimal expected cost")
axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("Value of resolving disagreement", fontsize=9)
frac = np.linspace(0, 1, 200)
for i, b in enumerate([0.1, 0.3, 1.0]):
    axs[1].plot(frac, evsi_lognormal_ratio(sg_g, sg_g * np.sqrt(1 - frac), b), color=C[i], lw=2, label=f"β = {b}")
axs[1].plot(frac, frac, color="#9a9994", ls="--", lw=1)
axs[1].set_xlabel("fraction of log-variance resolved"); axs[1].set_ylabel("EVSI / EVPI")
axs[1].set_title("Partial information (σ₀ = AI-author fit)", fontsize=9); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_voi.png"), dpi=200); plt.close(fig)

# ---------------- E5: global sensitivity (Sobol, Jansen estimators) ----------------
def sample(n):
    z = rng.standard_normal(n)
    return np.column_stack([mu_g + sg_g * z,                                    # logit p
                            rng.uniform(np.log(1e8), np.log(1e11), n),          # ln k
                            rng.uniform(np.log(0.05), np.log(1.0), n),          # ln beta
                            rng.uniform(0.1, 1.0, n),                           # m
                            rng.uniform(np.log(L_LOW), np.log(L_HIGH), n)])     # ln L
def fY(X):
    S = s_star(expit(X[:, 0]), np.exp(X[:, 4]), np.exp(X[:, 1]), np.exp(X[:, 2]), X[:, 3])
    return np.log10(np.maximum(S, 1e6))
names = ["belief p", "scale k", "elasticity β", "reducible share m", "stakes L"]
n = 2 ** 17
A, B = sample(n), sample(n)
yA, yB = fY(A), fY(B); V = np.var(np.concatenate([yA, yB]))
S1, ST = {}, {}
for i, nm in enumerate(names):
    ABi = A.copy(); ABi[:, i] = B[:, i]; yABi = fY(ABi)
    S1[nm] = float((V - 0.5 * np.mean((yB - yABi) ** 2)) / V)
    ST[nm] = float(0.5 * np.mean((yA - yABi) ** 2) / V)
# bootstrap CIs
boot = {nm: [] for nm in names}
for _ in range(100):
    idx = rng.integers(0, n, n)
    for i, nm in enumerate(names):
        ABi = A[idx].copy(); ABi[:, i] = B[idx, i]
        boot[nm].append(0.5 * np.mean((yA[idx] - fY(ABi)) ** 2) / np.var(np.concatenate([yA[idx], yB[idx]])))
out["E5_sobol"] = dict(n=n, var_log10S=float(V), S1=S1, ST=ST,
                       ST_ci95={nm: [float(np.percentile(boot[nm], 2.5)), float(np.percentile(boot[nm], 97.5))] for nm in names},
                       share_zero_spend=float(np.mean(yA <= 6.0)))
fig, ax = plt.subplots(figsize=(6, 3.2))
yy = np.arange(len(names))
ax.barh(yy + 0.18, [S1[nm] for nm in names], height=0.34, color=C[0], label="first-order S₁")
ax.barh(yy - 0.18, [ST[nm] for nm in names], height=0.34, color=C[1], label="total S_T")
ax.set_yticks(yy); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlabel("share of variance in log₁₀ S*")
ax.legend(frameon=False, fontsize=8); ax.set_title("What drives disagreement about optimal spend?", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_sobol.png"), dpi=200); plt.close(fig)

# ---------------- E6: consensus on direction ----------------
M = 20000
Xp = sample(M)
kk, bb, mm, LL = np.exp(Xp[:, 1]), np.exp(Xp[:, 2]), Xp[:, 3], np.exp(Xp[:, 4])
fr_cur, fr_cur5 = [], []
for j in range(M):
    Sj = s_star(pg, LL[j], kk[j], bb[j], mm[j])
    fr_cur.append(np.sum(wg * (Sj > S_CUR))); fr_cur5.append(np.sum(wg * (Sj > 5 * S_CUR)))
fr_cur, fr_cur5 = np.array(fr_cur), np.array(fr_cur5)
S_sf = s_star(P_XPT_SF, LL, kk, bb, mm)
out["E6_consensus"] = dict(
    n_param_draws=M, S_cur_PV=S_CUR,
    median_share_forecasters_above_current=float(np.median(fr_cur)),
    p10_share_forecasters_above_current=float(np.percentile(fr_cur, 10)),
    prob_majority_above_current=float(np.mean(fr_cur > 0.5)),
    prob_majority_above_5x_current=float(np.mean(fr_cur5 > 0.5)),
    superforecaster_median_prob_S_above_current=float(np.mean(S_sf > S_CUR)),
    superforecaster_median_prob_S_above_10x_current=float(np.mean(S_sf > 10 * S_CUR)),
    superforecaster_median_S_annual_quantiles={q: float(np.percentile(S_sf, q) / ANN) for q in [10, 25, 50, 75, 90]})
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(fr_cur, bins=40, color=C[0], edgecolor="#fcfcfb", linewidth=0.5)
ax.set_xlabel("share of AI-author belief distribution whose optimum exceeds est. current spend")
ax.set_ylabel("parameter draws"); ax.set_title("Agreement on direction across tractability draws", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_consensus.png"), dpi=200); plt.close(fig)

# ---------------- E7: functional-form-free marginal test ----------------
# Current spending is too low iff the marginal dollar at S_cur buys more than 1/L of absolute risk reduction.
E7 = {"absolute_risk_reduction_per_$1B_breakeven": {}, "relative_reduction_per_$1B_breakeven": {},
      "value_of_one_basis_point_PV": {}}
for lab, L in [("L_low", L_LOW), ("L_base", L_BASE), ("L_high", L_HIGH)]:
    E7["absolute_risk_reduction_per_$1B_breakeven"][lab] = 1e9 / L
    E7["value_of_one_basis_point_PV"][lab] = 1e-4 * L
    E7["relative_reduction_per_$1B_breakeven"][lab] = {str(p): 1e9 / (p * L) for p in [P_XPT_SF, P_XPT_EX, G_MED, G_MEAN]}
# frontier: for the isoelastic family, which (k, beta) make current spending (weakly) optimal?
kgrid = np.geomspace(1e7, 1e16, 300); bgrid = np.geomspace(1e-3, 3, 300)
Kg, Bg = np.meshgrid(kgrid, bgrid)
fig, ax = plt.subplots(figsize=(6, 3.8))
fr = {}
for i, (p, lab) in enumerate([(P_XPT_SF, "XPT superforecasters 0.38%"), (P_XPT_EX, "XPT experts 3%"), (G_MEAN, "AI-author mean 16.2%")]):
    for L, ls in [(L_BASE, "-"), (L_LOW, "--")]:
        mp = p * Bg * L / Kg * (1 + S_CUR / Kg) ** (-(1 + Bg))   # marginal value of $1 at S_cur (m = 1)
        cs = ax.contour(Kg, Bg, np.log10(mp), levels=[0], colors=[C[i]], linestyles=ls, linewidths=1.8)
        fr[f"{lab}|{'base' if L == L_BASE else 'low'}"] = float(np.mean(mp > 1))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("mitigation scale k ($ PV)"); ax.set_ylabel("elasticity β")
ax.set_title("Where est. current spend is already enough (below/right of curve)", fontsize=9)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=C[i]) for i in range(3)] + [Line2D([], [], color="#52514e", ls="--")],
          ["p = 0.38%", "p = 3%", "p = 16.2%", "dashed: L/10"], frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig6_frontier.png"), dpi=200); plt.close(fig)
E7["share_of_logk_logbeta_box_where_more_spending_optimal"] = fr
E7["box"] = "k in [1e7,1e16], beta in [1e-3,3], log-uniform, m=1"
out["E7_marginal_test"] = E7

# ---------------- E8: Sobol robustness to prior widths ----------------
def sobol_ST(sampler, n=2 ** 15):
    A_, B_ = sampler(n), sampler(n); yA_, yB_ = fY(A_), fY(B_); V_ = np.var(np.concatenate([yA_, yB_]))
    res = {}
    for i, nm in enumerate(names):
        AB = A_.copy(); AB[:, i] = B_[:, i]
        res[nm] = float(0.5 * np.mean((yA_ - fY(AB)) ** 2) / V_)
    return res
def make_sampler(belief="grace", kr=(1e8, 1e11), br=(0.05, 1.0)):
    def s(n):
        if belief == "grace":
            lp = mu_g + sg_g * rng.standard_normal(n)
        else:  # XPT between-group only: superforecaster or expert median, 50/50
            lp = np.where(rng.random(n) < 0.5, logit(P_XPT_SF), logit(P_XPT_EX))
        return np.column_stack([lp, rng.uniform(np.log(kr[0]), np.log(kr[1]), n),
                                rng.uniform(np.log(br[0]), np.log(br[1]), n), rng.uniform(0.1, 1.0, n),
                                rng.uniform(np.log(L_LOW), np.log(L_HIGH), n)])
    return s
E8 = {"grace_baseline": sobol_ST(make_sampler()),
      "grace_wide_tractability(k 1e7-1e13, beta 0.01-3)": sobol_ST(make_sampler(kr=(1e7, 1e13), br=(0.01, 3.0))),
      "xpt_between_group_baseline": sobol_ST(make_sampler("xpt")),
      "xpt_between_group_wide_tractability": sobol_ST(make_sampler("xpt", kr=(1e7, 1e13), br=(0.01, 3.0)))}
out["E8_sobol_variants"] = E8

out["runtime_sec"] = time.time() - T0
with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=1))

# ---------------- E9: explicit frontier points (m = 1) ----------------
from scipy.optimize import brentq
E9 = {}
for p in [P_XPT_SF, P_XPT_EX, G_MEAN]:
    for lab, L in [("L_base", L_BASE), ("L_low", L_LOW)]:
        mv = lambda b, k: np.log(p * b * L / k) - (1 + b) * np.log1p(S_CUR / k)
        # max k at beta = 0.3 (search on log k above S_cur scale)
        ks = np.geomspace(1e6, 1e20, 20000); ok = mv(0.3, ks) > 0
        kmax = float(ks[ok].max()) if ok.any() else None
        E9[f"p={p}|{lab}"] = dict(k_max_at_beta_0_3=kmax)
out["E9_frontier_points"] = E9
with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(E9, indent=1))
