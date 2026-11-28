"""Run every experiment in the paper.  Usage:  python3 run_all.py
Outputs go to ../results/.  Deterministic (seeded).  Runtime < 1 min on CPU."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import params as P
from model import (G_from, months_per_efold, lag_months, leakage_gain_months,
                   stock_response, simulate_x, status_quo_paths, intervene)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)
N = 200_000
Z90 = 1.6448536


def lognormal(median, ci, n):
    lo, hi = ci
    sd = (np.log(hi) - np.log(lo)) / (2 * Z90)
    return np.exp(np.log(median) + sd * rng.standard_normal(n))


def q(x, ps=(5, 50, 95)):
    return [float(np.percentile(x, p)) for p in ps]


R = {}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ------------------------------------------------------------------ E1 ---------
growth = lognormal(P.COMPUTE_GROWTH_MEDIAN, P.COMPUTE_GROWTH_CI90, N)
halv = lognormal(P.ALGO_HALVING_MONTHS_MEDIAN, P.ALGO_HALVING_MONTHS_CI90, N)
halv_n = lognormal(P.ALGO_HALVING_MONTHS_MEDIAN, P.ALGO_HALVING_MONTHS_CI90_NARROW, N)
G = G_from(growth, halv)
Gn = G_from(growth, halv_n)
G_med = float(G_from(P.COMPUTE_GROWTH_MEDIAN, P.ALGO_HALVING_MONTHS_MEDIAN))
S = rng.uniform(*P.S_RANGE, N)
E1 = {
    "G_point": G_med,
    "K10_point_months": 12 * P.LN10 / G_med,
    "halving_point_months_A": 12 * P.LN2 / G_med,
    "G_q": q(G), "G_narrow_q": q(Gn),
    "K10_q_wide": q(12 * P.LN10 / G), "K10_q_narrow": q(12 * P.LN10 / Gn),
    "per_halving_A_q": q(P.LN2 * months_per_efold(Gn, "A")),
    "per_halving_M_q": q(P.LN2 * months_per_efold(Gn, "M", S=S)),
    "per_halving_E1_q": q(P.LN2 * months_per_efold(Gn, "E", eta_over_kappa=1.0)),
    "note": "Gn uses the narrow algorithmic-progress prior (5-14 month halving)",
}
R["E1_rates"] = E1

# ------------------------------------------------------------------ E2 ---------
Lobs = lognormal(P.LAG_OBS_MEDIAN_MONTHS, P.LAG_OBS_CI90_MONTHS, N)
Rtr = lognormal(P.R_TRAIN_MEDIAN, P.R_TRAIN_CI90, N)
lnPhi = np.log(Rtr) - Gn * Lobs / 12.0           # H-A inversion
S_impl = 1.0 - Gn * Lobs / (12.0 * np.log(Rtr))  # H-M inversion
lag_pred_noimit = lag_months(np.log(Rtr), Gn)
E2 = {
    "lag_predicted_without_imitation_q": q(lag_pred_noimit),
    "Phi_implied_q": q(np.exp(lnPhi)),
    "P_Phi_gt_1": float(np.mean(lnPhi > 0)),
    "P_Phi_gt_3": float(np.mean(lnPhi > np.log(3))),
    "S_implied_q": q(S_impl),
    "point": {
        "lag_no_imitation": float(lag_months(np.log(16), G_med)),
        "Phi": float(np.exp(np.log(16) - G_med * 7 / 12)),
        "S": float(1 - G_med * 7 / (12 * np.log(16))),
    },
}
R["E2_inversion"] = E2

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].hist(np.clip(np.exp(lnPhi), 0, 30), bins=120, color="#4C72B0")
ax[0].axvline(1, color="k", lw=0.8, ls="--")
ax[0].set_xlabel("implied follower advantage $\\Phi$ (H-A)")
ax[0].set_ylabel("MC draws")
ax[1].hist(np.clip(S_impl, -1, 1), bins=120, color="#DD8452")
ax[1].axvline(P.S_IFP, color="k", lw=0.8, ls="--")
ax[1].set_xlabel("implied recapture share $S$ (H-M); dashed = IFP 0.5")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_inversion.png"); plt.close(fig)

# ------------------------------------------------------------------ E3 ---------
# Policy values.  (i) retrospective: controls vs no-controls via shares.
cf_share = rng.uniform(*P.CN_SHARE_COUNTERFACTUAL, N)
us_cf = P.US_SHARE_ARRIVALS * (1 - cf_share) / (1 - P.CN_SHARE_ARRIVALS)
R_act = P.US_SHARE_ARRIVALS / P.CN_SHARE_ARRIVALS
dlnR_retro = np.log(R_act) - np.log(us_cf / cf_share)

# (ii) prospective interventions on flows from 2027.0, evaluated at end-2027/28.
t = np.round(np.arange(2025.0, 2031.0 + 1e-9, 1 / 48), 6)
U_sq, C_sq = status_quo_paths(t, P.US_STOCK_END2026, P.US_STOCK_END2027,
                              P.R_STOCK_END2026, P.R_STOCK_END2027)
U_sq2, C_sq2 = status_quo_paths(t, P.US_STOCK_END2026, P.US_STOCK_END2027,
                                P.R_STOCK_END2026, P.R_STOCK_END2027,
                                ratio_growth_after=0.0)
smug_rate = np.array(P.SMUGGLED_CUM_2025) / P.SMUGGLING_YEARS
interventions = {
    "H200_cap_0.75M/yr": P.H200_FLOW_CAP,
    "H200_orders_2M/yr": P.H200_FLOW_ORDERS,
    "end_smuggling_median": -smug_rate[1],
    "end_smuggling_high": -smug_rate[2],
}
eval_t = {"end2027": 2028.0, "end2028": 2029.0, "end2030": 2031.0}
idx = {k: int(np.argmin(np.abs(t - v))) for k, v in eval_t.items()}
pol = {}
paths = {}
for name, dF in interventions.items():
    for d in (0.0, P.DELTA, 0.4):
        for sqname, Csq in (("ratio_keeps_rising", C_sq), ("ratio_flat_after_2028", C_sq2)):
            Cn = intervene(t, Csq, dF, 2027.0, d)
            dlnR = -np.log(Cn / Csq)  # positive = US lead grows
            key = f"{name}|delta={d}|{sqname}"
            pol[key] = {k: float(dlnR[i]) for k, i in idx.items()}
            if d == P.DELTA and sqname == "ratio_keeps_rising":
                paths[name] = dlnR


def months(dlnR, hyp, Gs, Ss=None, ek=1.0):
    if hyp == "M":
        return dlnR * months_per_efold(Gs, "M", S=Ss)
    return dlnR * months_per_efold(Gs, hyp, eta_over_kappa=ek)


E3 = {"retro_dlnR_q": q(dlnR_retro),
      "retro_ratio_actual": float(R_act),
      "retro_months": {h: q(months(dlnR_retro, h, Gn, S)) for h in ("M", "A")},
      "retro_months_E_longrun_ek1": q(months(dlnR_retro, "E", Gn, ek=1.0)),
      "prospective_dlnR": pol, "prospective_months": {}}
for key, v in pol.items():
    if "|delta=0.25|ratio_keeps_rising" not in key:
        continue
    nm = key.split("|")[0]
    E3["prospective_months"][nm] = {
        ev: {h: q(months(v[ev], h, Gn, S)) for h in ("M", "A")} for ev in v}
# deployment margin (Prop. 5): ratio of China inference capacity change
E3["deployment_change_pct_end2028"] = {
    nm: float(100 * (np.exp(-pol[f"{nm}|delta=0.25|ratio_keeps_rising"]["end2028"]) - 1))
    for nm in interventions}
R["E3_policy"] = E3

# ------------------------------------------------------------------ E4 ---------
d = np.linspace(0, 1, 201)
fig, ax = plt.subplots(figsize=(3.6, 2.6))
E4 = {}
for leak, c in ((0.05, "#4C72B0"), (0.2, "#55A868"), (0.5, "#C44E52")):
    gain = leakage_gain_months(d, leak, G_med)
    ax.plot(d, gain, color=c, label=f"leak share $\\ell$={leak}")
    E4[f"ceiling_months_leak{leak}"] = float(12 * np.log(1 / leak) / G_med)
    E4[f"gain_at_d0.5_leak{leak}"] = float(leakage_gain_months(0.5, leak, G_med))
    E4[f"share_of_ceiling_from_last_10pct_d_leak{leak}"] = float(
        (leakage_gain_months(1, leak, G_med) - leakage_gain_months(0.9, leak, G_med))
        / leakage_gain_months(1, leak, G_med))
ax.set_xlabel("enforcement of controllable channel $d$")
ax.set_ylabel("lag gained (months, H-A)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_leakage.png"); plt.close(fig)
R["E4_leakage"] = E4

# ------------------------------------------------------------------ E5 ---------
# verify Prop. 3 numerically and report half-lives
gam = float(np.log(P.US_STOCK_END2027 / P.US_STOCK_END2026) - np.log(P.R_STOCK_END2027 / P.R_STOCK_END2026))
E5 = {"china_stock_growth_gamma_2027": gam}
for dd in (0.0, 0.25, 0.4):
    lam = gam + dd
    E5[f"half_life_months_delta{dd}"] = float(12 * P.LN2 / lam)
# numerical check: balanced-growth stock, flow multiplied by theta=0.5
tt = np.arange(0, 30, 1 / 400)
F0, theta, dd, t0 = 1.0, 0.5, 0.25, 20.0
K = np.zeros_like(tt); K[0] = F0 / (gam + dd)
Kb = K.copy()
for i in range(1, len(tt)):
    h = tt[i] - tt[i - 1]
    f = F0 * np.exp(gam * tt[i - 1])
    K[i] = K[i - 1] + h * ((theta * f if tt[i - 1] >= t0 else f) - dd * K[i - 1])
    Kb[i] = Kb[i - 1] + h * (f - dd * Kb[i - 1])
m = tt >= t0
err = np.max(np.abs(np.log(K[m] / Kb[m]) - stock_response(theta, gam + dd, tt[m] - t0)))
E5["prop3_max_abs_error_numeric_vs_closed_form"] = float(err)
R["E5_stockflow"] = E5

# ------------------------------------------------------------------ E6 ---------
# Endogenous follower R&D: lag response to the H200-orders intervention path
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
E6 = {}
dl = paths["H200_orders_2M/yr"]  # >0 would be gain; here negative (loss)
for (ek, kap), c in zip(((0, 1), (1, 2), (1, 1), (1, 0.5), (2, 1)),
                        ("k", "#4C72B0", "#55A868", "#C44E52", "#8172B2")):
    x = simulate_x(t, dl, ek * kap, kap)
    L = 12 * (dl + x) / G_med
    lbl = "H-A (no endogenous R&D)" if ek == 0 else f"H-E $\\eta/\\kappa$={ek}, $\\kappa$={kap}"
    ax[0].plot(t, L, color=c, label=lbl)
    E6[f"ek{ek}_kappa{kap}"] = {k: float(L[i]) for k, i in idx.items()}
ax[0].axvline(2027, color="grey", lw=0.6, ls=":")
ax[0].set_xlim(2026.5, 2031); ax[0].set_ylabel("change in US lead (months)")
ax[0].set_title("H200 opening, 2M H100e/yr from 2027", fontsize=8)
ax[0].legend(frameon=False, fontsize=6, loc="upper right")
# step in lnR of +ln2 at t=0 with kappa=0: divergence
ts = np.linspace(0, 6, 601)
for kap, c in ((0.0, "k"), (0.5, "#C44E52"), (1.0, "#55A868"), (2.0, "#4C72B0")):
    eta = 0.5 if kap == 0 else 1.0 * kap
    x = simulate_x(ts, np.full_like(ts, P.LN2), eta, kap)
    ax[1].plot(ts, 12 * (P.LN2 + x) / G_med, color=c,
               label=("$\\kappa$=0, $\\eta$=0.5" if kap == 0 else f"$\\eta/\\kappa$=1, $\\kappa$={kap}"))
ax[1].set_xlabel("years after a permanent halving of China compute")
ax[1].set_ylabel("lag gained (months)")
ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_endogenous.png"); plt.close(fig)
E6["halving_step_kappa0_eta0.5_gain_after_5y"] = float(12 * (P.LN2 + 0.5 * P.LN2 * 5) / G_med)
R["E6_endogenous"] = E6

# ------------------------------------------------------------------ fig1 -------
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for nm, c in (("H200_cap_0.75M/yr", "#DD8452"), ("H200_orders_2M/yr", "#C44E52"),
              ("end_smuggling_median", "#55A868"), ("end_smuggling_high", "#4C72B0")):
    ax.plot(t, 12 * paths[nm] / G_med, color=c, label=nm.replace("_", " "))
ax.axhline(0, color="k", lw=0.5); ax.set_xlim(2026.5, 2031)
ax.set_ylabel("change in US lead, months (H-A)")
ax.legend(frameon=False, fontsize=6); fig.tight_layout()
fig.savefig(f"{OUT}/fig1_interventions.png"); plt.close(fig)

# ------------------------------------------------------------------ E7 ---------
# Identification / power: can lag data distinguish beta_M from beta_A?
bA = 12 / G_med; bM = 0.5 * bA
shock = paths["H200_orders_2M/yr"]  # policy-induced deviation of lnR (negative)
def lnR_path(tr, scale):
    base = np.log(5.0) + (np.log(13.0) - np.log(5.0)) * (tr - 2023.0) / 5.0
    return base + scale * np.interp(tr, t, shock, left=0.0)
E7 = {"beta_A": bA, "beta_M": bM,
      "shock_dlnR_end2028_scale1": float(np.interp(2029.0, t, shock))}
reps = 1000
for T_end in (2029.0, 2031.0):
    for sigma in (1.5, 2.5, 3.5):
        for scale in (0.0, 1.0, 4.0):
            for trend_ctrl in (False, True):
                out = {}
                for truth, btrue in (("A", bA), ("M", bM)):
                    rej = 0; ses = []; ests = []
                    for r in range(reps):
                        n = rng.poisson(8 * (T_end - 2023.0))
                        tr = np.sort(rng.uniform(2023.0, T_end, n))
                        lr = lnR_path(tr, scale)
                        drift = rng.normal(0, 0.5)  # unknown secular drift, months/yr
                        y = 2.0 + btrue * lr + drift * (tr - 2023.0) + rng.normal(0, sigma, n)
                        X = np.column_stack([np.ones(n), lr] + ([tr - 2023.0] if trend_ctrl else []))
                        if np.linalg.matrix_rank(X, tol=1e-8) < X.shape[1]:
                            ses.append(np.inf); continue  # beta not identified
                        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                        s2 = np.sum((y - X @ beta) ** 2) / (n - X.shape[1])
                        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
                        ses.append(se); ests.append(beta[1])
                        if (beta[1] - bM) / se > 1.6448536:  # one-sided test against H-M
                            rej += 1
                    ests = np.array(ests)
                    out[truth] = {"reject_HM_rate": rej / reps,
                                  "median_SE": float(np.median(ses)),
                                  "estimate_sd": float(ests.std()) if len(ests) else None,
                                  "estimate_mean": float(ests.mean()) if len(ests) else None}
                E7[f"T{int(T_end)}|sigma{sigma}|shockx{scale}|trend{trend_ctrl}"] = {
                    "power(H-A true)": out["A"]["reject_HM_rate"],
                    "size(H-M true)": out["M"]["reject_HM_rate"],
                    "median_SE": out["A"]["median_SE"],
                    "estimate_sd_under_A": out["A"]["estimate_sd"],
                    "estimate_mean_under_A": out["A"]["estimate_mean"]}
R["E7_power"] = E7
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for sigma, c in ((1.5, "#4C72B0"), (2.5, "#55A868"), (3.5, "#C44E52")):
    ys = [E7[f"T2031|sigma{sigma}|shockx{s}|trendTrue"]["power(H-A true)"] for s in (0.0, 1.0, 4.0)]
    ax.plot([0, 1, 4], ys, "o-", color=c, label=f"noise sd {sigma} mo")
ax.axhline(0.8, color="k", lw=0.5, ls=":")
ax.set_xlabel("policy shock size (x H200-2M scenario)")
ax.set_ylabel("power vs H-M (trend ctrl.)")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_power.png"); plt.close(fig)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps({k: R[k] for k in ("E1_rates", "E2_inversion", "E4_leakage", "E5_stockflow", "E6_endogenous")}, indent=1))
print(json.dumps(R["E3_policy"]["retro_months"], indent=1))
print(json.dumps(R["E3_policy"]["prospective_months"], indent=1))
print(json.dumps(R["E3_policy"]["deployment_change_pct_end2028"], indent=1))
for k, v in R["E7_power"].items():
    print(k, v)
