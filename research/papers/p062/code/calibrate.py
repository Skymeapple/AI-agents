"""Calibrated Monte Carlo of measured vs welfare-relevant AI effects on US GDP, 2025-2035,
plus statistical detectability of AI in aggregate GDP growth.

Every parameter is tagged SOURCED (with citation key, see paper) or ASSUMPTION.
Outputs in results/: mc_summary.json, mc_draws.csv, fig3_fan.png, fig4_sensitivity.png,
fig5_detectability.png, scenarios.csv
"""
import os, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import analytic_V

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "results"); D = os.path.join(HERE, "..", "data")
rng = np.random.default_rng(20261122)
N = 20000
YEARS = np.arange(2025, 2036)
T = len(YEARS)

# ------------------------------------------------------------------ sourced anchors
GDP2025 = 30.7697e12          # SOURCED: World Bank WDI NY.GDP.MKTP.CD, 2025 (data/wb_us_gdp_nominal.json)
GDP2024 = 29.298013e12        # SOURCED: World Bank WDI
CS_2024 = 97e9                # SOURCED: Collis & Brynjolfsson (2025, WSJ) consumer surplus of gen-AI, 2024
CS_LATEST = 172e9             # SOURCED: Brynjolfsson, Collis, Eggers, Kazinnik & Nguyen (2026) latest annual surplus
cs_share_2025 = CS_LATEST / GDP2025   # ~0.56% of GDP
TFP_LO, TFP_HI = 0.066, 1.3   # SOURCED: Acemoglu (2025) 0.66%/10y; Aghion & Bunel (2024) upper 1.3 pp/yr
NONWORK_LO, NONWORK_HI = 0.53, 0.73   # SOURCED: Chatterji et al. (2025) non-work share of ChatGPT messages

def logu(lo, hi, n): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

P = {}
P["gM"] = logu(TFP_LO, TFP_HI, N) / 100          # annual market TFP contribution (fraction)
P["nu"] = rng.uniform(NONWORK_LO, NONWORK_HI, N)  # share of consumer surplus from non-work use
P["haircut"] = logu(0.2, 1.0, N)                  # ASSUMPTION: WTA surveys may overstate
P["gH"] = rng.uniform(0.05, 0.35, N)              # ASSUMPTION: growth of surplus/GDP after 2025
P["phi"] = rng.uniform(0.1, 0.5, N)               # ASSUMPTION: share of non-work value with market substitutes
P["sigma"] = rng.uniform(1.0, 3.0, N)             # ASSUMPTION informed by Rupert-Rogerson-Wright (1995)
P["theta"] = rng.uniform(0.3, 0.8, N)             # ASSUMPTION: home share of substitutable composite
P["eta"] = rng.uniform(0.7, 1.3, N)               # ASSUMPTION: upper-nest elasticity (labour supply)
P["e_s"] = rng.uniform(0.03, 0.10, N)             # ASSUMPTION: substitutable services in full income
P["mu"] = rng.uniform(0.0, 0.3, N)                # ASSUMPTION: unmeasured-quality share of market gains
P["k_s"] = rng.uniform(0.02, 0.15, N)             # ASSUMPTION: share of market gains in substitutable services
P["Ipk"] = rng.uniform(0.0, 0.01, N)              # ASSUMPTION: peak unmeasured intangible investment / GDP
E_X = 0.40                                        # ASSUMPTION: leisure share of full income

def simulate(P, n):
    e_s = P["e_s"]; e_c = 1 - E_X - e_s
    Vc, Vs, Vh = analytic_V(e_c, e_s, E_X, P["theta"], P["sigma"], P["eta"])
    t = np.arange(T)[None, :]
    GM = P["gM"][:, None] * (t + 1)                          # cumulative market TFP level gain
    H = (cs_share_2025 * P["haircut"])[:, None] * (1 + P["gH"][:, None]) ** t   # consumer surplus / GDP
    Hn = P["nu"][:, None] * H                                # non-work part (work part assumed inside GM)
    I = P["Ipk"][:, None] * np.minimum((t + 1) / 6.0, 1.0)   # ramps to peak by 2030
    mkt_meas = GM * (1 - P["mu"][:, None]) * ((1 - P["k_s"][:, None]) * Vc[:, None] + P["k_s"][:, None] * Vs[:, None])
    home_meas = Hn * P["phi"][:, None] * Vh[:, None]
    M = mkt_meas + home_meas - I
    Tw = GM + Hn
    return dict(M=M, T=Tw, V=M / Tw, mkt=mkt_meas, home=home_meas, I=I, GM=GM, Hn=Hn,
                Vc=Vc, Vs=Vs, Vh=Vh)

S = simulate(P, N)
q = lambda x: [float(np.quantile(x, a)) for a in (0.05, 0.25, 0.5, 0.75, 0.95)]
i30, i35 = list(YEARS).index(2030), list(YEARS).index(2035)
summ = {
    "N": N,
    "cs_share_2025_of_GDP": cs_share_2025,
    "cs_share_2024_of_GDP": CS_2024 / GDP2024,
    "V_home_quantiles": q(S["Vh"]),
    "P_V_home_negative": float(np.mean(S["Vh"] < 0)),
    "V_c_quantiles": q(S["Vc"]), "V_s_quantiles": q(S["Vs"]),
}
for name, i in (("2030", i30), ("2035", i35)):
    summ[f"measured_pctGDP_{name}"] = q(100 * S["M"][:, i])
    summ[f"welfare_pctGDP_{name}"] = q(100 * S["T"][:, i])
    summ[f"V_{name}"] = q(S["V"][:, i])
    summ[f"P_V_lt_1_{name}"] = float(np.mean(S["V"][:, i] < 1))
    summ[f"P_V_lt_0.5_{name}"] = float(np.mean(S["V"][:, i] < 0.5))
    summ[f"home_meas_pctGDP_{name}"] = q(100 * S["home"][:, i])
    summ[f"intangible_pctGDP_{name}"] = q(100 * S["I"][:, i])
    gap = S["T"][:, i] - S["M"][:, i]
    # decomposition of the welfare-GDP gap (shares, medians of component/gap)
    comp = {
        "free_home_value_unmeasured": S["Hn"][:, i] * (1 - P["phi"]),
        "substitutable_home_value": S["Hn"][:, i] * P["phi"] * (1 - S["Vh"]),
        "market_quality_unmeasured": S["GM"][:, i] * P["mu"],
        "market_realloc_and_laborsupply": S["GM"][:, i] * (1 - P["mu"]) * (1 - ((1 - P["k_s"]) * S["Vc"] + P["k_s"] * S["Vs"])),
        "intangible_investment": S["I"][:, i],
    }
    summ[f"gap_pctGDP_{name}"] = q(100 * gap)
    summ[f"gap_component_mean_pctGDP_{name}"] = {k: float(np.mean(100 * v)) for k, v in comp.items()}
    chk = sum(comp.values()) - gap
    summ[f"gap_decomp_max_residual_{name}"] = float(np.max(np.abs(chk)))

# annual measured growth contribution (pp)
dM = np.diff(np.concatenate([np.zeros((N, 1)), S["M"]], axis=1), axis=1) * 100
summ["measured_growth_contrib_pp_2026"] = q(dM[:, 1])
summ["measured_growth_contrib_pp_2030"] = q(dM[:, i30])
summ["measured_growth_contrib_pp_2035"] = q(dM[:, i35])

# ---------------------------------------------------- sensitivity (Spearman rank corr with V_2035)
sens = {k: float(stats.spearmanr(v, S["V"][:, i35])[0]) for k, v in P.items()}
summ["spearman_V2035"] = sens

# ---------------------------------------------------- scenarios (deterministic)
def scen(gM, haircut, sigma, phi, label):
    Q = {k: np.array([np.median(v)]) for k, v in P.items()}
    Q["gM"] = np.array([gM / 100]); Q["haircut"] = np.array([haircut])
    Q["sigma"] = np.array([sigma]); Q["phi"] = np.array([phi])
    Ss = simulate(Q, 1)
    return [label, gM, haircut, sigma, phi, 100 * Ss["M"][0, i35], 100 * Ss["T"][0, i35],
            Ss["V"][0, i35], 100 * Ss["home"][0, i35]]
sc = [scen(0.066, 1.0, 2.5, 0.4, "Acemoglu-low TFP, full surplus, high substitution"),
      scen(0.066, 0.3, 1.5, 0.2, "Acemoglu-low TFP, haircut surplus, mild substitution"),
      scen(0.40, 0.6, 2.0, 0.3, "OECD-mid TFP, central consumer side"),
      scen(1.0, 0.6, 2.0, 0.3, "Aghion-Bunel-high TFP, central consumer side"),
      scen(1.0, 1.0, 3.0, 0.5, "High TFP, full surplus, strong substitution")]
with open(os.path.join(R, "scenarios.csv"), "w") as f:
    f.write("scenario,gM_pp_per_yr,haircut,sigma,phi,measured_pctGDP_2035,welfare_pctGDP_2035,V_2035,home_channel_measured_pctGDP_2035\n")
    for r in sc: f.write(",".join(str(x) if isinstance(x, str) else f"{x:.4f}" for x in r) + "\n")

# ---------------------------------------------------- detectability in aggregate GDP growth
wb = json.load(open(os.path.join(D, "wb_us_gdp_growth.json")))[1]
g = {int(r["date"]): r["value"] for r in wb if r["value"] is not None}
hist = np.array([g[y] for y in range(1990, 2020)])
mu_g, sd_g = hist.mean(), hist.std(ddof=1)
phi_ar = np.corrcoef(hist[:-1], hist[1:])[0, 1]
innov_sd = sd_g * np.sqrt(1 - phi_ar ** 2)
pre_years = [y for y in range(2010, 2025) if y not in (2020, 2021)]
pre_actual = np.array([g[y] for y in pre_years])
summ["growth_1990_2019_mean_sd_ar1"] = [float(mu_g), float(sd_g), float(phi_ar)]
summ["pre_sample_years"] = pre_years
summ["pre_sample_mean_sd"] = [float(pre_actual.mean()), float(pre_actual.std(ddof=1))]

def _ar1_paths(n, npre, sd, phi):
    innov = sd * np.sqrt(1 - phi ** 2)
    eps = rng.normal(0, innov, (n, npre + T + 20))
    x = np.zeros_like(eps); x[:, 0] = rng.normal(0, sd, n)
    for k in range(1, eps.shape[1]):
        x[:, k] = phi * x[:, k - 1] + eps[:, k]
    return x[:, 20:]

def _tstats(x, npre, delta):
    pre = x[:, :npre]; post = x[:, npre:] + delta
    out = np.full((x.shape[0], T), np.nan)
    for j in range(1, T):
        out[:, j] = stats.ttest_ind(post[:, :j + 1], pre, axis=1, equal_var=False)[0]
    return out

def power_curve(delta_paths, sd, phi, nsim=400, alpha=0.05):
    """One-sided Welch t-test of post-2025 vs pre-sample mean growth, with critical values
    SIZE-CORRECTED by simulation for AR(1) serial correlation (so size = alpha exactly)."""
    npre = len(pre_years)
    null = _tstats(_ar1_paths(20000, npre, sd, phi), npre, 0.0)
    crit = np.nanquantile(null, 1 - alpha, axis=0)
    n = delta_paths.shape[0]; pw = np.zeros((n, T))
    for s in range(nsim):
        ts = _tstats(_ar1_paths(n, npre, sd, phi), npre, delta_paths)
        pw[:, 1:] += ts[:, 1:] > crit[None, 1:]
    return pw / nsim, crit

sub = rng.choice(N, 400, replace=False)
regimes = {"volatile_1990_2019": (sd_g, phi_ar),
           "calm_2010_2024_ex_covid": (float(pre_actual.std(ddof=1)),
                                       float(np.corrcoef(pre_actual[:-1], pre_actual[1:])[0, 1]))}
summ["noise_regimes_sd_phi"] = regimes
shifts = np.linspace(0, 3, 61)
pw_store = {}
for rname, (sd, ph) in regimes.items():
    pw, crit = power_curve(dM[sub], sd, ph)
    pw_store[rname] = pw
    summ[f"detect_{rname}_median_power_by_year"] = {int(y): float(np.median(pw[:, j])) for j, y in enumerate(YEARS)}
    summ[f"detect_{rname}_share_power_ge_0.8_2030"] = float(np.mean(pw[:, i30] >= 0.8))
    summ[f"detect_{rname}_share_power_ge_0.8_2035"] = float(np.mean(pw[:, i35] >= 0.8))
    pwc, _ = power_curve(np.repeat(shifts[:, None], T, axis=1), sd, ph, nsim=1000)
    def mde(j):
        ok = np.where(pwc[:, j] >= 0.8)[0]
        return float(shifts[ok[0]]) if len(ok) else None
    summ[f"MDE_{rname}_pp_80pct_2030"] = mde(i30)
    summ[f"MDE_{rname}_pp_80pct_2035"] = mde(i35)
    summ[f"size_check_{rname}_power_at_zero_2035"] = float(pwc[0, i35])

json.dump(summ, open(os.path.join(R, "mc_summary.json"), "w"), indent=2)
cols = list(P.keys())
np.savetxt(os.path.join(R, "mc_draws_sample.csv"),
           np.column_stack([P[k][:2000] for k in cols] + [S["M"][:2000, i35], S["T"][:2000, i35], S["V"][:2000, i35]]),
           delimiter=",", header=",".join(cols + ["M2035", "T2035", "V2035"]), comments="")

# ---------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for arr, col, lab in ((S["T"], "#2c7fb8", "welfare-relevant gain (market TFP + non-work consumer value)"),
                      (S["M"], "#d95f0e", "measured real GDP effect")):
    qq = np.quantile(100 * arr, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax[0].fill_between(YEARS, qq[0], qq[4], color=col, alpha=0.15)
    ax[0].fill_between(YEARS, qq[1], qq[3], color=col, alpha=0.3)
    ax[0].plot(YEARS, qq[2], color=col, lw=2, label=lab)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_ylabel("% of GDP (level, vs no-AI counterfactual)")
ax[0].set_title("Measured vs welfare-relevant AI effect (5-25-50-75-95%)")
ax[0].legend(fontsize=7, loc="upper left")
qv = np.quantile(S["V"], [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
ax[1].fill_between(YEARS, qv[0], qv[4], color="gray", alpha=0.2)
ax[1].fill_between(YEARS, qv[1], qv[3], color="gray", alpha=0.4)
ax[1].plot(YEARS, qv[2], "k", lw=2)
ax[1].axhline(1, color="r", ls="--", lw=1)
ax[1].set_ylim(-0.6, 1.4)
ax[1].set_title("Visibility ratio V = measured / welfare-relevant")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_fan.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ks = sorted(sens, key=lambda k: abs(sens[k]))
ax.barh(ks, [sens[k] for k in ks], color=["#d95f0e" if sens[k] < 0 else "#2c7fb8" for k in ks])
ax.set_xlabel("Spearman rank correlation with V(2035)")
ax.set_title("Global sensitivity of GDP visibility")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_sensitivity.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for (rname, pw), col in zip(pw_store.items(), ("#54278f", "#e6550d")):
    qp = np.quantile(pw, [0.25, 0.5, 0.75], axis=0)
    ax.fill_between(YEARS[1:], qp[0][1:], qp[2][1:], alpha=0.25, color=col)
    ax.plot(YEARS[1:], qp[1][1:], color=col, lw=2, label=f"noise: {rname} (median, IQR)")
ax.axhline(0.8, color="k", ls="--", lw=0.8)
ax.set_ylim(0, 1); ax.set_xlabel("last year of post-2025 sample")
ax.set_ylabel("power (size-corrected one-sided test, 5%)")
ax.set_title("Detectability of AI in aggregate US real GDP growth")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig5_detectability.png"), dpi=160); plt.close(fig)
print(json.dumps(summ, indent=1))
