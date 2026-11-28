"""Theory + calibration module: two-channel catch-up model and handicapped
Tullock racing contest.

Requires results/arena_fit.json and results/arena_gap_series.csv (run arena_gap.py first).
Outputs (results/): contest_check.json, calibration.json, scenarios.csv,
scenarios_summary.json, prcc.json, fig_racing_index.png, fig_scenarios.png,
fig_reversal_region.png, fig_racing_history.png
"""
import json, os, csv
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
rng = np.random.default_rng(20261122)
BT_BETA = 400 / np.log(10)          # Bradley-Terry / Elo scale (=173.7 Elo per e-fold of odds)

# ---------------------------------------------------------------- contest primitives
def eq_efforts(h, rho, VL=1.0):
    """Closed-form Nash efforts in the handicapped Tullock contest
    p_L = h xL / (h xL + xF), payoffs V_i p_i - x_i, V_F = rho V_L."""
    xL = h * rho * VL / (h + rho) ** 2
    xF = rho * xL
    return xL, xF


def racing_index(h, rho):
    """Total equilibrium effort divided by its maximum over h (attained at h = rho)."""
    return 4 * h * rho / (h + rho) ** 2


def numeric_eq(h, rho, VL=1.0):
    """Independent check of eq_efforts: solve the best-response fixed point
    xL = BR_L(BR_F(xL)) by bracketing root-finding (brentq), and confirm with a
    grid search that no unilateral deviation is profitable."""
    from scipy.optimize import brentq
    VF = rho * VL
    brL = lambda xF: max(np.sqrt(VL * xF / h) - xF / h, 0.0)
    brF = lambda xL: max(np.sqrt(VF * h * xL) - h * xL, 0.0)
    f = lambda x: brL(brF(x)) - x
    grid = np.exp(np.linspace(np.log(1e-8), np.log(VL), 4000))
    vals = np.array([f(x) for x in grid])
    i = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][-1]
    xL = brentq(f, grid[i], grid[i + 1], xtol=1e-14)
    xF = brF(xL)
    dev = np.linspace(0, VL, 200001)
    uL = VL * h * dev / (h * dev + xF + 1e-300) - dev
    uF = VF * dev / (h * xL + dev + 1e-300) - dev
    gainL = uL.max() - (VL * h * xL / (h * xL + xF) - xL)
    gainF = uF.max() - (VF * xF / (h * xL + xF) - xF)
    return xL, xF, max(gainL, gainF)


def contest_check():
    errs, gains = [], []
    rows = []
    for h in [0.5, 1.0, 1.3, 2.0, 5.0, 10.0]:
        for rho in [0.5, 1.0, 1.5, 3.0]:
            a = np.array(eq_efforts(h, rho)); xl, xf, gain = numeric_eq(h, rho)
            errs.append(np.max(np.abs(a - np.array([xl, xf]))))
            gains.append(gain)
            rows.append(dict(h=h, rho=rho, xL_closed=a[0], xF_closed=a[1], xL_num=xl, xF_num=xf,
                             max_deviation_gain=float(gain)))
    # check argmax of total effort over h equals rho
    hs = np.exp(np.linspace(-3, 3, 6001))
    argmax = {str(r): float(hs[np.argmax([sum(eq_efforts(h, r)) for h in hs])]) for r in [0.5, 1, 2]}
    out = dict(max_abs_error=float(max(errs)), max_deviation_gain=float(max(gains)), n_cases=len(errs), argmax_total_effort_over_h=argmax, cases=rows)
    json.dump(out, open(os.path.join(RES, "contest_check.json"), "w"), indent=2)
    return out


# ---------------------------------------------------------------- calibration
def calibration(fit, N=200_000):
    slope = rng.normal(fit["frontier_slope_elo_per_yr"],
                       (fit["frontier_slope_ci90"][1] - fit["frontier_slope_ci90"][0]) / (2 * 1.645), N)
    comp = rng.uniform(3.7, 4.6, N)                     # Sevilla & Roldan 2024: 4.1x/yr (90% CI 3.7-4.6)
    H = np.exp(rng.uniform(np.log(5), np.log(14), N))   # Ho et al. 2024: halving 8 mo (95% CI ~5-14)
    oom_per_yr = np.log10(comp) + (12 / H) * np.log10(2)
    kappa = slope / oom_per_yr                         # Elo per order of magnitude of effective compute
    ratio = np.exp(rng.uniform(np.log(2), np.log(10), N))  # US/PRC frontier-relevant compute ratio; Pilz et al. 75%/15% = 5
    ell = np.log10(ratio)
    Gobs = rng.uniform(30, 60, N)                      # observed steady gap band (computed + AI Index 2026)
    psi_max = np.minimum(Gobs / (kappa * ell), 1.0)
    q = lambda x: [float(v) for v in np.percentile(x, [5, 50, 95])]
    out = dict(kappa_elo_per_oom=q(kappa), oom_per_yr=q(oom_per_yr), ell=q(ell),
               compute_only_gap_elo=q(kappa * ell), psi_max=q(psi_max),
               share_psi_max_below_0p5=float(np.mean(psi_max < 0.5)),
               point=dict(kappa=float(fit["frontier_slope_elo_per_yr"] / (np.log10(4.1) + 1.5 * np.log10(2))),
                          ell=float(np.log10(5))))
    out["point"]["compute_only_gap"] = out["point"]["kappa"] * out["point"]["ell"]
    out["point"]["psi_max_at_G45"] = 45 / out["point"]["compute_only_gap"]
    json.dump(out, open(os.path.join(RES, "calibration.json"), "w"), indent=2)
    return out, dict(slope=slope, kappa=kappa, ell=ell, Gobs=Gobs, psi_max=psi_max)


# ---------------------------------------------------------------- scenarios
def scenarios(fit, cal, N=200_000):
    slope, kappa, ell, Gobs, psi_max = (cal[k] for k in ("slope", "kappa", "ell", "Gobs", "psi_max"))
    psi = rng.uniform(0, 1, N) * psi_max                          # compute pass-through
    X0 = Gobs - psi * kappa * ell                                  # knowledge-channel gap Delta_a/delta >= 0
    rho0 = np.exp(rng.uniform(np.log(2 / 3), np.log(1.5), N))     # V_F / V_L at baseline
    eta = rng.uniform(0, 1.0, N)                                   # d ln rho / d ell (securitization)
    beta = BT_BETA * np.exp(rng.uniform(np.log(0.5), np.log(2), N))
    dtight = 0.3                                                   # tighten: compute ratio x2
    drelax = -rng.uniform(0.1, 0.3, N)                             # relax (e.g. H200 licences)
    mdiff = rng.uniform(0.4, 0.7, N)                               # diffusion restriction: delta multiplier
    mopen = rng.uniform(1.5, 2.0, N)                               # more openness
    sc = {
        "S0 status quo": (0.0, 1.0),
        "S1 tighten compute (ratio x2)": (dtight, 1.0),
        "S2 relax compute (H200-type)": (drelax, 1.0),
        "S3 restrict diffusion": (0.0, mdiff),
        "S4 tighten + restrict": (dtight, mdiff),
        "S5 more openness": (0.0, mopen),
    }
    res = {}
    base = None
    for name, (dl, md) in sc.items():
        G = psi * kappa * (ell + dl) + X0 / md
        rho = rho0 * np.exp(eta * dl)
        h = np.exp(G / beta)
        RI = racing_index(h, rho)
        E = h * rho * (1 + rho) / (h + rho) ** 2           # total effort, V_L = 1
        pL = h / (h + rho)
        tau = 12 * G / slope
        res[name] = dict(G=G, tau=tau, RI=RI, E=E, pL=pL)
    base = res["S0 status quo"]
    summ = {}
    q = lambda x: [float(v) for v in np.percentile(x, [5, 50, 95])]
    for name, r in res.items():
        summ[name] = dict(G_elo=q(r["G"]), lead_months=q(r["tau"]), racing_index=q(r["RI"]),
                          p_leader=q(r["pL"]),
                          effort_change_pct=q(100 * (r["E"] / base["E"] - 1)),
                          prob_effort_up=float(np.mean(r["E"] > base["E"] * (1 + 1e-9))))
    # conditional on the securitization elasticity eta (the pivotal unknown)
    up = res["S1 tighten compute (ratio x2)"]["E"] > base["E"]
    thr = psi * kappa / beta
    cond = {}
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]:
        m = (eta >= lo) & (eta < hi)
        cond[f"eta_{lo:.1f}_{hi:.1f}"] = float(up[m].mean())
    cond["threshold_psi_kappa_over_beta_p5_p50_p95"] = q(thr)
    zpos = (np.log(np.exp(Gobs / beta)) - np.log(rho0)) > 0
    cond["share_z_positive_baseline"] = float(zpos.mean())
    cond["prop4_sign_rule_accuracy"] = float(np.mean(up == ((np.sign(Gobs / beta - np.log(rho0)) * (thr - eta)) < 0)))
    # marginal rule for absolute effort E (Corollary 2), evaluated at the scenario midpoint
    Gm = Gobs + 0.5 * psi * kappa * dtight; rm = rho0 * np.exp(eta * 0.5 * dtight); hm = np.exp(Gm / beta)
    dlnE = thr * (rm - hm) / (hm + rm) + eta * ((hm - rm) / (hm + rm) + rm / (1 + rm))
    cond["cor2_marginal_rule_accuracy"] = float(np.mean(up == (dlnE > 0)))
    up_ri = res["S1 tighten compute (ratio x2)"]["RI"] > base["RI"]
    zm = Gm / beta - np.log(rm)
    cond["prop4_RI_rule_accuracy_midpoint"] = float(np.mean(up_ri == ((np.sign(zm) * (thr - eta)) < 0)))
    cond["prob_RI_up_tighten"] = float(up_ri.mean())
    m0 = eta < 0.05
    cond["prob_E_up_given_eta_lt_0.05"] = float(up[m0].mean())
    cond["prob_RI_up_given_eta_lt_0.05"] = float(up_ri[m0].mean())
    summ["_conditional_S1"] = cond
    json.dump(summ, open(os.path.join(RES, "scenarios_summary.json"), "w"), indent=2)
    with open(os.path.join(RES, "scenarios.csv"), "w") as f:
        f.write("scenario,stat,p5,p50,p95\n")
        for n, s in summ.items():
            if n.startswith("_"):
                continue
            for k, v in s.items():
                if isinstance(v, list):
                    f.write(f"{n},{k},{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}\n")
                else:
                    f.write(f"{n},{k},{v:.3f},,\n")
    # PRCC for effort change under tightening
    params = dict(psi=psi, eta=eta, log_beta=np.log(beta), log_rho0=np.log(rho0), ell=ell,
                  kappa=kappa, Gobs=Gobs)
    y = res["S1 tighten compute (ratio x2)"]["E"] / base["E"] - 1
    prcc = prcc_calc(params, y, n=20000)
    y2 = res["S3 restrict diffusion"]["tau"] - base["tau"]
    prcc2 = prcc_calc(dict(params, mdiff=mdiff), y2, n=20000)
    json.dump(dict(effort_change_tighten=prcc, lead_gain_restrict=prcc2),
              open(os.path.join(RES, "prcc.json"), "w"), indent=2)
    # decomposition: lead gain per policy as function of psi share
    return res, summ, dict(psi=psi, psi_max=psi_max, eta=eta, beta=beta, rho0=rho0, kappa=kappa, ell=ell,
                           X0=X0, mdiff=mdiff), prcc, prcc2


def prcc_calc(params, y, n=20000):
    idx = rng.choice(len(y), n, replace=False)
    names = list(params)
    R = np.column_stack([rankdata(params[k][idx]) for k in names])
    ry = rankdata(y[idx])
    out = {}
    for j, k in enumerate(names):
        others = np.column_stack([np.ones(n), np.delete(R, j, axis=1)])
        bx = np.linalg.lstsq(others, R[:, j], rcond=None)[0]
        by = np.linalg.lstsq(others, ry, rcond=None)[0]
        ex, ey = R[:, j] - others @ bx, ry - others @ by
        out[k] = float(np.corrcoef(ex, ey)[0, 1])
    return out


# ---------------------------------------------------------------- figures
def figures(fit, summ, draws):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (1) racing index vs gap
    G = np.linspace(-150, 400, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for rho, c in [(2 / 3, "#7f7f7f"), (1.0, "#1f4e79"), (1.5, "#c55a11"), (2.0, "#548235")]:
        ax.plot(G, racing_index(np.exp(G / BT_BETA), rho), color=c, label=f"V_F/V_L = {rho:.2f}")
    ax.axvspan(30, 60, color="#1f4e79", alpha=.12, label="observed 2024-26 gap band")
    ax.axvline(356, color="k", ls=":", lw=1); ax.text(300, .15, "May 2023\ngap", fontsize=8)
    ax.set_xlabel("capability gap G (Arena Elo, leader minus follower)")
    ax.set_ylabel("racing index RI = E / E_max")
    ax.set_title("Equilibrium racing intensity vs. capability gap (beta = 400/ln10)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_racing_index.png"), dpi=150)

    # (2) racing history from data
    rows = list(csv.DictReader(open(os.path.join(RES, "arena_gap_series.csv"))))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    hist = {}
    for cat, c in (("overall", "#1f4e79"), ("overall_style_control", "#c55a11")):
        rr = [r for r in rows if r["category"] == cat]
        t = np.array([int(r["date"][:4]) + (int(r["date"][4:6]) - 1) / 12 + (int(r["date"][6:]) - 1) / 365 for r in rr])
        g = np.array([float(r["gap_elo"]) for r in rr])
        ri = racing_index(np.exp(g / BT_BETA), 1.0)
        ax.plot(t, ri, "o-", ms=2.5, lw=.8, color=c, label=cat)
        hist[cat] = dict(first=float(ri[0]), last=float(ri[-1]),
                         mean_last12m=float(ri[t >= t.max() - 1].mean()), first_date=rr[0]["date"])
    ax.set_ylim(0, 1.02); ax.set_ylabel("racing index (rho = 1)"); ax.set_xlabel("year")
    ax.set_title("Implied racing index from observed US-PRC Arena gap")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_racing_history.png"), dpi=150)
    json.dump(hist, open(os.path.join(RES, "racing_history.json"), "w"), indent=2)

    # (3) scenarios
    names = [n for n in summ if not n.startswith('_')]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, n in enumerate(names):
        s = summ[n]
        for k, a in ((("lead_months"), ax[0]), (("racing_index"), ax[1])):
            lo, md, hi = s[k]
            a.plot([lo, hi], [i, i], color="#1f4e79"); a.plot(md, i, "o", color="#c55a11")
    for a, t in zip(ax, ["steady-state lead (months)", "racing index"]):
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8); a.set_xlabel(t); a.invert_yaxis()
    ax[1].set_yticklabels([])
    ax[0].set_title("Policy scenarios: median and 90% interval"); fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_scenarios.png"), dpi=150)

    # (4) reversal region in (G, eta) for psi = 0.28 (calibrated median psi_max), kappa = 246, rho0 = 1
    fig, ax = plt.subplots(figsize=(6.4, 4))
    psi_, kap_, rho_ = 0.28, 246.0, 1.0
    thr_ = psi_ * kap_ / BT_BETA
    Gs = np.linspace(-60, 300, 400); etas = np.linspace(0, 1.2, 300)
    GG, EE = np.meshgrid(Gs, etas)
    hh = np.exp(GG / BT_BETA)
    z = GG / BT_BETA - np.log(rho_)
    ri_up = (np.sign(z) * (thr_ - EE)) < 0
    dlnE = thr_ * (rho_ - hh) / (hh + rho_) + EE * ((hh - rho_) / (hh + rho_) + rho_ / (1 + rho_))
    ax.contourf(GG, EE, ri_up.astype(float), levels=[-.5, .5, 1.5], colors=["#ffffff", "#f4c7a1"], alpha=.8)
    ax.contour(GG, EE, dlnE, levels=[0], colors="#1f4e79", linewidths=1.5)
    ax.axhline(thr_, color="k", ls="--", lw=.8)
    ax.axvspan(30, 60, color="#1f4e79", alpha=.10)
    ax.text(120, thr_ + .03, "eta = psi*kappa/beta", fontsize=8)
    ax.text(150, 0.95, "shaded: tightening raises\nracing index RI (Prop. 4)", fontsize=8)
    ax.text(150, 0.05, "below blue curve: tightening\nlowers absolute effort E (Cor. 2)", fontsize=8)
    ax.set_xlabel("baseline gap G (Elo)"); ax.set_ylabel("securitization elasticity eta")
    ax.set_title("When do tighter compute controls intensify racing? (psi=0.28, rho0=1)")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_reversal_region.png"), dpi=150)
    return hist


def main():
    fit = json.load(open(os.path.join(RES, "arena_fit.json")))
    cc = contest_check()
    print("contest check max abs err", cc["max_abs_error"], cc["argmax_total_effort_over_h"])
    cal_s, cal = calibration(fit)
    print(json.dumps(cal_s, indent=1))
    res, summ, draws, prcc, prcc2 = scenarios(fit, cal)
    print(json.dumps(summ, indent=1)); print("PRCC tighten effort", prcc); print("PRCC restrict lead", prcc2)
    hist = figures(fit, summ, draws)
    print(hist)
    # analytic quantities used in text
    a = dict(G_dagger_rho1p5=float(BT_BETA * np.log(1.5)), G_dagger_rho2=float(BT_BETA * np.log(2)),
             RI_G356_rho1=float(racing_index(np.exp(356 / BT_BETA), 1)),
             RI_G45_rho1=float(racing_index(np.exp(45 / BT_BETA), 1)),
             RI_G39_rho1=float(racing_index(np.exp(39 / BT_BETA), 1)),
             pL_G45_rho1=float(np.exp(45 / BT_BETA) / (np.exp(45 / BT_BETA) + 1)),
             halflife_months=float(12 * np.log(2) / fit["fits"]["overall"]["delta"]),
             implied_delta_g_elo_per_yr=float(fit["fits"]["overall"]["delta"] * fit["fits"]["overall"]["Gstar"]),
             implied_follower_intrinsic_rate_share=float(1 - fit["fits"]["overall"]["delta"] * fit["fits"]["overall"]["Gstar"] / fit["frontier_slope_elo_per_yr"]),
             lead_months_at_Gstar=float(12 * fit["fits"]["overall"]["Gstar"] / fit["frontier_slope_elo_per_yr"]),
             psi_kappa_over_beta_at_psi0p28=float(0.28 * 246 / BT_BETA))
    json.dump(a, open(os.path.join(RES, "analytic_quantities.json"), "w"), indent=2)
    print(a)


if __name__ == "__main__":
    main()
