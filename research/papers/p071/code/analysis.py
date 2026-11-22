"""Threshold drift analysis: main pipeline.

Run:  python3 code/analysis.py      (from the p071 directory; ~1-3 min CPU)
Outputs: results/*.json, results/*.csv, results/fig*.png
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RES, load_large_scale, load_all_models, alpha_from_halving)

RNG = np.random.default_rng(20261122)
OUT = {}
X_FLOOR = 23.0          # large-scale dataset floor (1e23 FLOP)
TAU0 = 2023.0           # centring for intensity model
U0 = 25.0               # centring for log-compute
N_BOOT = 300

# ----------------------------------------------------------------------------
# S0 data
# ----------------------------------------------------------------------------
d = load_large_scale()
T_CUT = float(d["t"].max())
OUT["data"] = {
    "n_large_scale": int(len(d)),
    "n_with_compute": int(d["x"].notna().sum()),
    "latest_publication_decimal_year": T_CUT,
    "latest_publication_date": str(d["date"].max().date()),
}

# ----------------------------------------------------------------------------
# S1 frontier trend: top-10-at-release among all models with compute
# ----------------------------------------------------------------------------
a = load_all_models()
a = a[a["x"].notna()].sort_values("date").reset_index(drop=True)
flags = []
seen = []
for xv in a["x"].values:
    seen.append(xv)
    # top-10 among all models released up to and including this one
    k = min(10, len(seen))
    thr = np.partition(np.array(seen), -k)[-k]
    flags.append(xv >= thr)
a["frontier"] = flags
fr = a[(a["frontier"]) & (a["t"] >= 2018.0)]


def ols(t, x):
    A = np.vstack([np.ones_like(t), t - 2024.0]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return coef  # intercept at 2024.0, slope (OOM / yr)


f_coef = ols(fr["t"].values, fr["x"].values)
boots = []
for _ in range(2000):
    idx = RNG.integers(0, len(fr), len(fr))
    boots.append(ols(fr["t"].values[idx], fr["x"].values[idx]))
boots = np.array(boots)
g_hat = float(f_coef[1])
g_ci = np.percentile(boots[:, 1], [5, 95]).tolist()
fr20 = fr[fr["t"] >= 2020.0]
f20 = ols(fr20["t"].values, fr20["x"].values)
OUT["frontier"] = {
    "n_frontier_models_2018plus": int(len(fr)),
    "f2024_log10": float(f_coef[0]),
    "g_oom_per_year": g_hat,
    "g_multiplier_per_year": float(10 ** g_hat),
    "g_90ci_oom": g_ci,
    "g_90ci_multiplier": [float(10 ** v) for v in g_ci],
    "g_2020plus_oom": float(f20[1]),
    "g_2020plus_multiplier": float(10 ** f20[1]),
}
G_BOOT = boots[:, 1]


def frontier(t, coef=f_coef):
    return coef[0] + coef[1] * (np.asarray(t) - 2024.0)


# ----------------------------------------------------------------------------
# S2 inhomogeneous Poisson intensity over (time, log10 compute), x >= 23
#   lambda(t,x) = exp(th0 + th1*tau + (th2 + th3*tau)*u),  tau=t-2023, u=x-25
# ----------------------------------------------------------------------------
def tgrid(t0, t1, n=600):
    edges = np.linspace(t0, t1, n + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    return mids, (t1 - t0) / n


def tail_integral(th, tau, xlo):
    """int_{xlo}^{inf} lambda(t,x) dx at times tau (vector)."""
    b = th[2] + th[3] * tau
    b = np.minimum(b, -1e-3)
    return np.exp(th[0] + th[1] * tau + b * (xlo - U0)) / (-b)


def negll(th, tau_i, u_i, w_i, t0, t1):
    mids, dt = tgrid(t0, t1)
    integ = tail_integral(th, mids - TAU0, X_FLOOR).sum() * dt
    loglam = th[0] + th[1] * tau_i + (th[2] + th[3] * tau_i) * u_i
    return -(np.sum(w_i * loglam) - integ)


def _wrap(free3):
    if free3:
        return lambda p, *a: negll(p, *a), (lambda p: p)
    return (lambda p, *a: negll(np.append(p, 0.0), *a)), (lambda p: np.append(p, 0.0))


def fit_intensity(df, t0, t1, weights=None, free3=False):
    """free3=False (primary, 'M2'): constant tail slope; free3=True ('M1'): slope drifts."""
    s = df[(df["x"] >= X_FLOOR) & (df["t"] >= t0) & (df["t"] < t1)]
    tau = s["t"].values - TAU0
    u = s["x"].values - U0
    w = np.ones(len(s)) if weights is None else weights(s)
    f, expand = _wrap(free3)
    th0 = np.array([np.log(max(len(s), 1) / (t1 - t0)), 0.5, -1.5, 0.0][: 4 if free3 else 3])
    r = minimize(f, th0, args=(tau, u, w, t0, t1), method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
    r = minimize(f, r.x, args=(tau, u, w, t0, t1), method="BFGS")
    return expand(r.x), s


def boot_intensity(df, t0, t1, weights=None, n=N_BOOT, free3=False):
    s = df[(df["x"] >= X_FLOOR) & (df["t"] >= t0) & (df["t"] < t1)]
    tau = s["t"].values - TAU0
    u = s["x"].values - U0
    base_w = np.ones(len(s)) if weights is None else weights(s)
    out = []
    f, expand = _wrap(free3)
    th_init, _ = fit_intensity(df, t0, t1, weights, free3)
    p0 = th_init if free3 else th_init[:3]
    for _ in range(n):
        pw = RNG.poisson(1.0, len(s)) * base_w   # Poisson bootstrap
        r = minimize(f, p0, args=(tau, u, pw, t0, t1), method="BFGS")
        out.append(expand(r.x))
    return np.array(out)


def expected_count(th, t0, t1, xfun):
    """Expected number of releases in [t0,t1) with log10 C >= xfun(t)."""
    mids, dt = tgrid(t0, t1, 400)
    xs = np.maximum(np.asarray(xfun(mids), dtype=float), X_FLOOR) * np.ones_like(mids)
    return float(tail_integral(th, mids - TAU0, xs).sum() * dt)


known = d[d["x"].notna()].copy()
A_C_PRE = alpha_from_halving(8.0)

# imputation weights: per calendar year, n_all / n_known (missing-at-random within year)
yr_counts = d.groupby(d["date"].dt.year).agg(n=("Model", "size"), nk=("x", "count"))
yr_w = (yr_counts["n"] / yr_counts["nk"]).to_dict()


def mar_weights(s):
    return s["date"].dt.year.map(yr_w).values.astype(float)


# --- out-of-sample validation: fit 2020-2024, predict 2025 --------------------
# Observed 2025 counts are bracketed: lower = known-compute models; upper = known plus
# all 2025 large-scale models with missing compute (any org), i.e. every model that
# could be above the threshold.
oos = []
obs25 = known[(known["t"] >= 2025.0) & (known["t"] < 2026.0)]
n_missing25 = int(((d["t"] >= 2025.0) & (d["t"] < 2026.0) & d["x"].isna()).sum())
fo_all = set(known.loc[known["x"] >= 25.0, "org"])
n_missing25_fo = int(((d["t"] >= 2025.0) & (d["t"] < 2026.0) & d["x"].isna() & d["org"].isin(fo_all)).sum())
OUT["oos_2025"] = {"n_missing_2025": n_missing25, "n_missing_2025_frontier_orgs": n_missing25_fo}
for lab, free3, wts in [("M2_constant_slope", False, None), ("M1_drifting_slope", True, None),
                        ("M2_mar_weighted", False, "mar")]:
    wf = None if wts is None else (lambda s: s["date"].dt.year.map(yr_w).values.astype(float))
    th_tr, s_tr = fit_intensity(known, 2020.0, 2025.0, wf, free3)
    bt_tr = boot_intensity(known, 2020.0, 2025.0, wf, n=200, free3=free3)
    for thr in [23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0]:
        mu = expected_count(th_tr, 2025.0, 2026.0, lambda t, v=thr: v + 0 * t)
        mus = np.array([expected_count(b, 2025.0, 2026.0, lambda t, v=thr: v + 0 * t) for b in bt_tr])
        sims = RNG.poisson(np.repeat(mus, 50))
        lo, hi = np.percentile(sims, [5, 95])
        o = int((obs25["x"] >= thr).sum())
        ub = o + (n_missing25_fo if thr >= 25.0 else n_missing25)
        oos.append({"model": lab, "threshold_log10": thr, "predicted_mean": mu, "pi90_lo": float(lo),
                    "pi90_hi": float(hi), "observed_known_lower": o, "observed_upper": ub,
                    "pi_overlaps_bracket": bool(hi >= o and lo <= ub)})
    OUT["oos_2025"]["theta_train_" + lab] = th_tr.tolist()
OUT["oos_2025"]["table"] = oos

# --- main fits on 2020-2025 ----------------------------------------------------
th_k, s_k = fit_intensity(known, 2020.0, 2026.0)
th_w, _ = fit_intensity(known, 2020.0, 2026.0, weights=mar_weights)
th_m1, _ = fit_intensity(known, 2020.0, 2026.0, free3=True)
bt_k = boot_intensity(known, 2020.0, 2026.0, n=N_BOOT)
bt_w = boot_intensity(known, 2020.0, 2026.0, weights=mar_weights, n=N_BOOT)
# closed-form growth rates of covered counts (Proposition 3), natural-log units per year
gr = {}
for lab, al in [("static", 0.0), ("capability_indexed", -A_C_PRE), ("frontier_relative", g_hat)]:
    v = bt_k[:, 1] + bt_k[:, 2] * al
    gr[lab] = {"dlnN_dt": float(th_k[1] + th_k[2] * al), "ci90": np.percentile(v, [5, 95]).tolist(),
               "annual_multiplier": float(np.exp(th_k[1] + th_k[2] * al))}
wv = -bt_k[:, 2] * (g_hat + A_C_PRE)
gr["wedge_capidx_over_frontier"] = {"dln_ratio_dt": float(-th_k[2] * (g_hat + A_C_PRE)),
                                    "ci90": np.percentile(wv, [5, 95]).tolist(),
                                    "annual_multiplier": float(np.exp(-th_k[2] * (g_hat + A_C_PRE)))}
OUT["growth_rates_prop3"] = gr
OUT["intensity"] = {
    "form": "lambda(t,x)=exp(th0+th1*(t-2023)+(th2+th3*(t-2023))*(x-25)), x=log10 FLOP>=23",
    "theta_known": th_k.tolist(),
    "theta_known_90ci": np.percentile(bt_k, [5, 95], axis=0).tolist(),
    "theta_mar_weighted": th_w.tolist(),
    "theta_mar_90ci": np.percentile(bt_w, [5, 95], axis=0).tolist(),
    "theta_M1_drifting_slope": th_m1.tolist(),
    "count_reduction_per_OOM": float(np.exp(-th_k[2])),
    "n_points": int(len(s_k)),
    "year_weights": {int(k): float(v) for k, v in yr_w.items()},
}

# 2026 partial-year check (expected undercount due to reporting lag)
obs26 = known[known["t"] >= 2026.0]
chk26 = []
for thr in [24.0, 25.0, 26.0]:
    mu = expected_count(th_k, 2026.0, T_CUT, lambda t, v=thr: v + 0 * t)
    chk26.append({"threshold_log10": thr, "predicted_mean_to_cutoff": mu,
                  "observed_known": int((obs26["x"] >= thr).sum())})
OUT["check_2026_partial"] = chk26

# ----------------------------------------------------------------------------
# S3 threshold rules: retrospective and projected coverage
# ----------------------------------------------------------------------------
H_MONTHS = {"fast(5mo)": 5.0, "central(8mo)": 8.0, "slow(14mo)": 14.0}
ALPHA = {k: alpha_from_halving(v) for k, v in H_MONTHS.items()}
OUT["alpha_oom_per_year"] = ALPHA
A_C = ALPHA["central(8mo)"]
T_ANCHOR = 2024.0


def rules(g_coef=f_coef, alpha=A_C):
    return {
        "static_1e25": lambda t: 25.0 + 0 * np.asarray(t),
        "static_1e26": lambda t: 26.0 + 0 * np.asarray(t),
        "capability_indexed": lambda t: 25.0 - alpha * (np.asarray(t) - T_ANCHOR),
        "frontier_minus_1OOM": lambda t: frontier(t, g_coef) - 1.0,
        "frontier_minus_0.5OOM": lambda t: frontier(t, g_coef) - 0.5,
    }


R = rules()
years = list(range(2024, 2031))
proj_rows = []
for name, xf in R.items():
    for y in years:
        mk = expected_count(th_k, y, y + 1, xf)
        bk = np.array([expected_count(b, y, y + 1, xf) for b in bt_k[:150]])
        mw = expected_count(th_w, y, y + 1, xf)
        bw = np.array([expected_count(b, y, y + 1, xf) for b in bt_w[:150]])
        m1 = expected_count(th_m1, y, y + 1, xf)
        proj_rows.append({"rule": name, "year": y, "threshold_log10_midyear": float(xf(y + 0.5)), "exp_M1": m1,
                          "below_data_floor": bool(xf(y + 0.5) < X_FLOOR),
                          "exp_known": mk, "known_lo": float(np.percentile(bk, 5)),
                          "known_hi": float(np.percentile(bk, 95)),
                          "exp_mar": mw, "mar_lo": float(np.percentile(bw, 5)),
                          "mar_hi": float(np.percentile(bw, 95))})
# capability-indexed under fast/slow halving
for hk, al in ALPHA.items():
    xf = lambda t, al=al: 25.0 - al * (np.asarray(t) - T_ANCHOR)
    for y in years:
        proj_rows.append({"rule": f"capability_indexed_{hk}", "year": y,
                          "threshold_log10_midyear": float(xf(y + 0.5)),
                          "below_data_floor": bool(xf(y + 0.5) < X_FLOOR),
                          "exp_known": expected_count(th_k, y, y + 1, xf),
                          "exp_mar": expected_count(th_w, y, y + 1, xf)})
proj = pd.DataFrame(proj_rows)
proj.to_csv(os.path.join(RES, "projected_coverage.csv"), index=False)

# Retrospective (observed) coverage with distinct orgs / countries
retro = []
for name, xf in R.items():
    for y in [2023, 2024, 2025, 2026]:
        s = known[(known["date"].dt.year == y)]
        cov = s[s["x"] >= xf(s["t"].values)]
        retro.append({"rule": name, "year": y, "n_models": int(len(cov)),
                      "n_orgs": int(cov["org"].nunique()),
                      "n_countries": int(cov["country"].nunique()),
                      "share_US": float((cov["bucket"] == "US").mean()) if len(cov) else np.nan,
                      "share_China": float((cov["bucket"] == "China").mean()) if len(cov) else np.nan})
retro = pd.DataFrame(retro)
retro.to_csv(os.path.join(RES, "retrospective_coverage.csv"), index=False)

# false negatives of static 1e25 relative to capability target (observed)
fn = []
for y in [2024, 2025, 2026]:
    s = known[known["date"].dt.year == y]
    xc = R["capability_indexed"](s["t"].values)
    fnm = s[(s["x"] >= xc) & (s["x"] < 25.0)]
    fn.append({"year": y, "capability_target_models": int((s["x"] >= xc).sum()),
               "covered_by_static_1e25": int(((s["x"] >= xc) & (s["x"] >= 25.0)).sum()),
               "missed_by_static_1e25": int(len(fnm)),
               "missed_open_weights": int(fnm["open_weights"].sum()),
               "missed_orgs": int(fnm["org"].nunique())})
OUT["false_negatives_static_1e25"] = fn

# ----------------------------------------------------------------------------
# S4 optimal threshold path (Proposition 1) and loss of static rule
#   H(u,t) = kappa * 10^{eta[(u-25) + (1-rho)alpha*tau - rho*(f(t)-f(2024))]}
#   x*(t) = 25 - alpha*tau + rho*(alpha+g)*tau
# ----------------------------------------------------------------------------
rho_star = A_C / (A_C + g_hat)
rho_star_boot = []
for gb in G_BOOT:
    for al in [alpha_from_halving(h) for h in RNG.uniform(5, 14, 3)]:
        rho_star_boot.append(al / (al + gb))
OUT["rho_star"] = {"central": rho_star,
                   "range_over_halving_5_14_and_g_boot_5_95": np.percentile(rho_star_boot, [5, 95]).tolist(),
                   "at_fast_5mo": ALPHA["fast(5mo)"] / (ALPHA["fast(5mo)"] + g_hat),
                   "at_slow_14mo": ALPHA["slow(14mo)"] / (ALPHA["slow(14mo)"] + g_hat)}


def loss_year(th, y, xfun, rho, eta, alpha=A_C, g=g_hat):
    """Annual loss (units of kappa) = covered count + harm from uncovered models."""
    mids, dt = tgrid(y, y + 1, 60)
    us = np.linspace(X_FLOOR, 29.0, 1200)
    du = us[1] - us[0]
    tot = 0.0
    for t in mids:
        tau = t - TAU0
        lam = np.exp(th[0] + th[1] * tau + (th[2] + th[3] * tau) * (us - U0))
        tt = t - T_ANCHOR
        H = 10 ** (eta * ((us - 25.0) + (1 - rho) * alpha * tt - rho * g * tt))
        xt = xfun(t)
        cov = us >= xt
        tot += (np.sum(lam[cov]) + np.sum(H[~cov] * lam[~cov])) * du * dt
    return tot


loss_rows = []
for rho in [0.0, 0.25, rho_star, 0.5, 0.75, 1.0]:
    xopt = lambda t, rho=rho: 25.0 - A_C * (t - T_ANCHOR) + rho * (A_C + g_hat) * (t - T_ANCHOR)
    for eta in [0.5, 1.0, 2.0]:
        Ls = sum(loss_year(th_k, y, lambda t: 25.0, rho, eta) for y in range(2025, 2031))
        Lo = sum(loss_year(th_k, y, xopt, rho, eta) for y in range(2025, 2031))
        Lf = sum(loss_year(th_k, y, lambda t: frontier(t) - 1.0, rho, eta) for y in range(2025, 2031))
        Lc = sum(loss_year(th_k, y, lambda t: 25.0 - A_C * (t - T_ANCHOR), rho, eta) for y in range(2025, 2031))
        loss_rows.append({"rho": rho, "eta": eta, "slope_opt_oom_per_yr": -A_C + rho * (A_C + g_hat),
                          "loss_opt": Lo, "ratio_static1e25_to_opt": Ls / Lo,
                          "ratio_frontier-1_to_opt": Lf / Lo, "ratio_capidx_to_opt": Lc / Lo})
loss_df = pd.DataFrame(loss_rows)
loss_df.to_csv(os.path.join(RES, "loss_ratios.csv"), index=False)

# numerical check of Proposition 1: grid-minimise loss in x at a few (t, rho, eta)
chk = []
for t in [2026.5, 2029.5]:
    for rho in [0.0, 0.5, 1.0]:
        for eta in [0.5, 2.0]:
            xs = np.linspace(22.0, 30.0, 801)
            us = np.linspace(X_FLOOR - 3, 30.0, 3000)
            du = us[1] - us[0]
            tau = t - TAU0
            lam = np.exp(th_k[0] + th_k[1] * tau + (th_k[2] + th_k[3] * tau) * (us - U0))
            tt = t - T_ANCHOR
            H = 10 ** (eta * ((us - 25.0) + (1 - rho) * A_C * tt - rho * g_hat * tt))
            cumH = np.cumsum(H * lam) * du
            cumL = np.cumsum(lam[::-1])[::-1] * du
            Ls = [np.interp(xv, us, cumL) + np.interp(xv, us, cumH) for xv in xs]
            xnum = float(xs[int(np.argmin(Ls))])
            xan = 25.0 - A_C * tt + rho * (A_C + g_hat) * tt
            chk.append({"t": t, "rho": rho, "eta": eta, "x_numeric": xnum, "x_analytic": xan})
OUT["prop1_numeric_check"] = chk

# ----------------------------------------------------------------------------
# S5 who: jurisdictional concentration as a function of threshold (2024+)
# ----------------------------------------------------------------------------
rec = known[known["t"] >= 2024.0]
frontier_orgs = set(known.loc[known["x"] >= 25.0, "org"])
miss = d[(d["x"].isna()) & (d["t"] >= 2024.0) & (d["org"].isin(frontier_orgs))].copy()
miss["x"] = 25.5   # upper-bound scenario: assume >=1e25 (but unknown re 1e26)
rec_ub = pd.concat([rec, miss])


def jur_table(df):
    rows = []
    for thr in np.round(np.arange(23.0, 26.51, 0.1), 2):
        s = df[df["x"] >= thr]
        if len(s) == 0:
            continue
        cc = s["country"].value_counts()
        need90 = int(np.searchsorted(np.cumsum(cc.values) / cc.sum(), 0.9) + 1)
        oc = s["org"].value_counts(normalize=True)
        rows.append({"threshold_log10": thr, "n_models": int(len(s)),
                     "n_orgs": int(s["org"].nunique()), "n_countries": int(s["country"].nunique()),
                     "countries_for_90pct": need90,
                     "share_US": float((s["bucket"] == "US").mean()),
                     "share_China": float((s["bucket"] == "China").mean()),
                     "share_EU": float((s["bucket"] == "EU").mean()),
                     "share_Other": float((s["bucket"].isin(["Other", "Unknown"])).mean()),
                     "org_HHI": float((oc ** 2).sum()),
                     "share_open_weights": float(s["open_weights"].mean())})
    return pd.DataFrame(rows)


jur = jur_table(rec)
jur.to_csv(os.path.join(RES, "jurisdiction_by_threshold.csv"), index=False)
jur_ub = jur_table(rec_ub)
jur_ub.to_csv(os.path.join(RES, "jurisdiction_by_threshold_upperbound.csv"), index=False)
OUT["frontier_orgs"] = sorted(frontier_orgs)
OUT["n_missing_compute_frontier_orgs_2024plus"] = int(len(miss))

# ----------------------------------------------------------------------------
# S6 observability gap and classification ambiguity
# ----------------------------------------------------------------------------
obs_rows = []
for y in range(2022, 2027):
    s = d[d["date"].dt.year == y]
    fo = s["org"].isin(frontier_orgs)
    obs_rows.append({"year": y, "n": int(len(s)),
                     "missing_share_all": float(s["x"].isna().mean()),
                     "missing_share_frontier_orgs": float(s.loc[fo, "x"].isna().mean()) if fo.any() else np.nan,
                     "n_frontier_orgs_models": int(fo.sum()),
                     "missing_share_other_orgs": float(s.loc[~fo, "x"].isna().mean()) if (~fo).any() else np.nan,
                     "missing_share_open": float(s.loc[s["open_weights"], "x"].isna().mean()) if s["open_weights"].any() else np.nan,
                     "missing_share_closed": float(s.loc[~s["open_weights"], "x"].isna().mean()) if (~s["open_weights"]).any() else np.nan})
obs = pd.DataFrame(obs_rows)
obs.to_csv(os.path.join(RES, "observability_gap.csv"), index=False)

SIG = {"Confident": 0.5 / 1.645, "Likely": 1.0 / 1.645, "Speculative": 1.5 / 1.645}
amb = []
r24 = known[known["t"] >= 2024.0].copy()
r24["sigma"] = r24["Confidence"].map(SIG).fillna(1.5 / 1.645)
for thr in [25.0, 26.0]:
    p = norm.cdf((r24["x"] - thr) / r24["sigma"])
    amb.append({"threshold_log10": thr, "n_point_estimate_above": int((r24["x"] >= thr).sum()),
                "expected_above": float(p.sum()),
                "n_ambiguous_p_0.1_0.9": int(((p > 0.1) & (p < 0.9)).sum()),
                "n_ambiguous_p_0.25_0.75": int(((p > 0.25) & (p < 0.75)).sum())})
OUT["classification_ambiguity_2024plus"] = amb

# ----------------------------------------------------------------------------
# S7 bunching below 1e25 (open-weights vs closed), with placebos
# ----------------------------------------------------------------------------
def poisson_poly_fit(xc, y, deg=2):
    X = np.vstack([xc ** k for k in range(deg + 1)]).T

    def nll(b):
        eta = X @ b
        return np.sum(np.exp(eta) - y * eta)
    b0 = np.zeros(deg + 1)
    b0[0] = np.log(max(y.mean(), 0.1))
    r = minimize(nll, b0, method="BFGS")
    return r.x, X


def bunching(df, T, width=0.4, lo=23.5, hi=26.0, binw=0.1, deg=2):
    edges = np.round(np.arange(lo, hi + 1e-9, binw), 4)
    cnt, _ = np.histogram(df["x"].values, bins=edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    excl = (mids >= T - width) & (mids < T + width)
    b, _ = poisson_poly_fit(mids[~excl] - T, cnt[~excl].astype(float), deg)
    X = np.vstack([(mids - T) ** k for k in range(deg + 1)]).T
    cf = np.exp(X @ b)
    below = (mids >= T - width) & (mids < T)
    above = (mids >= T) & (mids < T + width)
    B = cnt[below].sum() - cf[below].sum()
    M = cnt[above].sum() - cf[above].sum()
    return {"obs_below": int(cnt[below].sum()), "cf_below": float(cf[below].sum()),
            "obs_above": int(cnt[above].sum()), "cf_above": float(cf[above].sum()),
            "excess_below": float(B), "excess_above": float(M),
            "ratio_below_above_obs": float(cnt[below].sum() / max(cnt[above].sum(), 1)),
            "ratio_below_above_cf": float(cf[below].sum() / cf[above].sum())}


def bunch_boot(df, T, n=500):
    res = bunching(df, T)
    stat = []
    for _ in range(n):
        s = df.sample(len(df), replace=True, random_state=int(RNG.integers(1e9)))
        r = bunching(s, T)
        stat.append(r["excess_below"])
    res["excess_below_90ci"] = np.percentile(stat, [5, 95]).tolist()
    res["frac_boot_excess_below_gt0"] = float(np.mean(np.array(stat) > 0))
    return res


bun = {}
post = known[known["t"] >= 2024.0]
for lab, sub in [("all", post), ("open_weights", post[post["open_weights"]]),
                 ("not_open_weights", post[~post["open_weights"]])]:
    bun[lab] = {}
    for T in [24.0, 24.5, 25.0, 25.5]:
        if lab == "not_open_weights" and T > 25.0:
            continue
        bun[lab][str(T)] = bunch_boot(sub, T, n=400)
        bun[lab][str(T)]["n"] = int(len(sub))
OUT["bunching_2024plus"] = bun
# pre-2024 comparison for the placebo-in-time (frontier was ~1e25 so weak)
pre = known[(known["t"] < 2024.0) & (known["t"] >= 2021.0)]
OUT["bunching_pre2024_counts"] = {
    "n": int(len(pre)),
    "n_24.6_25.0": int(((pre["x"] >= 24.6) & (pre["x"] < 25.0)).sum()),
    "n_25.0_25.4": int(((pre["x"] >= 25.0) & (pre["x"] < 25.4)).sum())}

with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=float)

# save objects for figures
np.save(os.path.join(RES, "theta_known.npy"), th_k)
np.save(os.path.join(RES, "theta_mar.npy"), th_w)
np.save(os.path.join(RES, "frontier_coef.npy"), f_coef)
print(json.dumps({k: OUT[k] for k in ["data", "frontier", "intensity", "rho_star"]}, indent=1, default=float))
print("OOS", pd.DataFrame(oos).to_string()); print(json.dumps(OUT["growth_rates_prop3"], indent=1)); print(OUT["intensity"])
print(retro.to_string())
print(pd.DataFrame(fn).to_string())
print(loss_df.to_string())
print(pd.DataFrame(chk).to_string())
print(obs.to_string())
print(json.dumps(amb, indent=1))
print(json.dumps(bun, indent=1))
print(OUT["bunching_pre2024_counts"])
print(proj[proj.year.isin([2024, 2026, 2028, 2030])][["rule", "year", "threshold_log10_midyear", "exp_known", "known_lo", "known_hi", "exp_mar", "exp_M1"]].to_string())
print(jur.iloc[::5].to_string())
print(jur_ub[jur_ub.threshold_log10.isin([24.0, 25.0, 26.0])].to_string())
