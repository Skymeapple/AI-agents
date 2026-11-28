"""Run all experiments for the TB-MHC paper. Usage: python3 experiments.py
Outputs go to ../results/. Deterministic (fixed seeds). Runtime: ~1-3 min CPU."""
import json, os, itertools, time
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)
RES = {}
C = ["#2a6f97", "#d1495b", "#edae49", "#66a182", "#6c757d", "#8e5572"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- E1
def e1_validate(n=1_000_000):
    """Monte Carlo check of closed-form rates and d* thresholds."""
    rows = []
    for d in [0.25, 0.5, 1.0, 2.0, 3.0]:
        for h in [0.9, 0.95, 0.99]:
            c = d / 2 - norm.ppf(h)
            xv = rng.normal(d / 2, 1, n); xi = rng.normal(-d / 2, 1, n)
            hit_mc, fa_mc = (xv > c).mean(), (xi > c).mean()
            hit_cf, fa_cf = M.quota_rates(d, h)
            rows.append(dict(regime="quota", d=d, h=h, fa_mc=fa_mc, fa_cf=float(fa_cf),
                             hit_mc=hit_mc, hit_cf=float(hit_cf)))
        for q, r, a in [(0.9, 1, 0), (0.9, 5, 0), (0.9, 20, 0), (0.9, 5, 1.0), (0.7, 2, 0)]:
            lb = M.log_beta(q, r, a)
            c = lb / d
            xv = rng.normal(d / 2, 1, n); xi = rng.normal(-d / 2, 1, n)
            hit_cf, fa_cf = M.bayes_rates(d, q, r, a)
            rows.append(dict(regime="bayes", d=d, q=q, r=r, alpha=a, fa_mc=(xi > c).mean(),
                             fa_cf=float(fa_cf), hit_mc=(xv > c).mean(), hit_cf=float(hit_cf)))
    err = max(max(abs(r_["fa_mc"] - r_["fa_cf"]), abs(r_["hit_mc"] - r_["hit_cf"])) for r_ in rows)
    # threshold check: at d*, catch rate equals kappa
    thr = []
    for kappa in [0.5, 0.8, 0.9]:
        for h in [0.9, 0.95]:
            ds = M.quota_dstar(kappa, h); _, fa = M.quota_rates(ds, h)
            thr.append(abs((1 - fa) - kappa))
        for q, r, a in [(0.9, 1, 0), (0.9, 5, 0), (0.95, 2, 0.5)]:
            ds = M.bayes_dstar(kappa, q, r, a); _, fa = M.bayes_rates(ds, q, r, a)
            thr.append(abs((1 - fa) - kappa))
    RES["E1"] = dict(n_configs=len(rows), mc_samples_per_config=n,
                     max_abs_error_rates=float(err), max_abs_error_threshold=float(max(thr)))
    with open(os.path.join(OUT, "e1_validation.json"), "w") as f:
        json.dump(rows, f, indent=1)


# ---------------------------------------------------------------- E2
D_GRID = [1.0, 2.0, 3.0, 4.0]
TAU_GRID = [15.0, 60.0, 240.0]
DELTA_GRID = [2.0, 5.0]


def e2_catch_vs_time():
    t = np.linspace(0.5, 600, 1200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    out = {}
    for ax, tau in zip(axes, TAU_GRID):
        for i, D in enumerate(D_GRID):
            d = M.d_of_t(t, D, tau, 2.0)
            _, fa = M.quota_rates(d, 0.95)
            ax.plot(t, 1 - fa, color=C[i], label=f"D={D:g}")
            d20 = float(M.d_of_t(20.0, D, tau, 2.0)); _, fa20 = M.quota_rates(d20, 0.95)
            out[f"tau={tau:g},D={D:g}"] = dict(d_at_20s=d20, catch_at_20s=float(1 - fa20))
        ax.axvline(20, color="k", ls=":", lw=1)
        ax.axhline(0.05, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_title(f"tau = {tau:g} s, delta = 2 s, h = 0.95")
        ax.set_xlabel("review time per nomination t (s)")
    axes[0].set_ylabel("catch rate (share of machine errors vetoed)")
    axes[0].legend(frameon=False)
    save(fig, "fig1_catch_vs_time.png")
    RES["E2_catch_at_20s_quota_h0.95"] = out


# ---------------------------------------------------------------- E3
def e3_lambda_max():
    rows = []
    for D, tau, delta, h, kappa in itertools.product(D_GRID, TAU_GRID, DELTA_GRID,
                                                     [0.9, 0.95, 0.99], [0.5, 0.8, 0.9]):
        ds = M.quota_dstar(kappa, h)
        tstar = float(M.t_of_d(ds, D, tau, delta))
        lm = float(M.lambda_max(ds, D, tau, delta, eta=1.0))
        rows.append(dict(form="exp", D=D, tau=tau, delta=delta, h=h, kappa=kappa,
                         dstar=float(ds), tstar_s=tstar, lambda_max_per_hr=lm))
    # robustness: sqrt (drift-diffusion-like) form, k chosen so d(60s)=D_ref
    for k_ref, delta, h, kappa in itertools.product([1.0, 2.0, 3.0], DELTA_GRID,
                                                    [0.9, 0.95, 0.99], [0.5, 0.8, 0.9]):
        k = k_ref / np.sqrt(60.0)
        ds = M.quota_dstar(kappa, h)
        tstar = delta + (ds / k) ** 2
        rows.append(dict(form="sqrt", D=k_ref, tau=np.nan, delta=delta, h=h, kappa=kappa,
                         dstar=float(ds), tstar_s=float(tstar),
                         lambda_max_per_hr=float(3600 / tstar)))
    import csv
    with open(os.path.join(OUT, "e3_lambda_max_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    exp_rows = [r for r in rows if r["form"] == "exp"]
    feas = [r for r in exp_rows if np.isfinite(r["tstar_s"])]
    infeasible_share = 1 - len(feas) / len(exp_rows)
    # headline config
    head = [r for r in exp_rows if r["h"] == 0.95 and r["kappa"] == 0.5 and r["delta"] == 2.0]
    t_needed = [r["tstar_s"] for r in feas]
    under20 = np.mean([t <= 20 for t in t_needed])
    RES["E3"] = dict(n_configs_exp=len(exp_rows), share_infeasible_exp=infeasible_share,
                     median_tstar_feasible_s=float(np.median(t_needed)),
                     share_feasible_with_tstar_le_20s=float(under20),
                     headline_h95_k50_delta2=[{k: r[k] for k in ("D", "tau", "dstar", "tstar_s",
                                                                 "lambda_max_per_hr")} for r in head],
                     dstar_values={f"h={h},kappa={k}": float(M.quota_dstar(k, h))
                                   for h in [0.9, 0.95, 0.99] for k in [0.5, 0.8, 0.9]},
                     sqrt_form=[{k: r[k] for k in ("D", "delta", "h", "kappa", "tstar_s")}
                                for r in rows if r["form"] == "sqrt" and r["h"] == 0.95
                                and r["kappa"] == 0.5 and r["delta"] == 2.0])
    # figure: lambda_max heat map over (D, tau) for h=0.95,kappa=0.5,delta=2
    Ds = np.linspace(1.0, 5.0, 81); taus = np.geomspace(5, 600, 81)
    ds = M.quota_dstar(0.5, 0.95)
    L = np.array([[float(M.lambda_max(ds, D, tau, 2.0)) for D in Ds] for tau in taus])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    Lp = np.where(L > 0, L, np.nan)
    im = ax.pcolormesh(Ds, taus, np.log10(Lp), shading="auto", cmap="viridis")
    cs = ax.contour(Ds, taus, Lp, levels=[10, 30, 60, 180], colors="w", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:g}/h", fontsize=8)
    ax.axvline(ds, color="r", lw=1, ls="--")
    ax.text(ds + 0.05, 7, "D = d* (infeasible to the left)", color="r", fontsize=8, rotation=90)
    ax.set_yscale("log"); ax.set_xlabel("asymptotic incremental discriminability D")
    ax.set_ylabel("time constant tau (s)")
    ax.set_title("max nominations / operator-hour\n(catch >= 50%, hit quota 95%, delta = 2 s)")
    fig.colorbar(im, ax=ax, label="log10 lambda_max")
    save(fig, "fig2_lambda_max.png")


# ---------------------------------------------------------------- E4
def e4_precision_paradox():
    qs = np.linspace(0.5, 0.995, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    out = {}
    for i, r in enumerate([1, 5, 20, 100]):
        ds = np.array([M.bayes_dstar(0.5, q, r) for q in qs])
        axes[0].plot(qs, ds, color=C[i], label=f"r = C_F/C_M = {r}")
        d = 1.0
        hit, fa = M.bayes_rates(d, qs, r)
        axes[1].plot(qs, 1 - fa, color=C[i])
        axes[2].plot(qs, (1 - qs) * fa, color=C[i])
        out[f"r={r}"] = dict(catch_at_q0p8=float(1 - M.bayes_rates(d, 0.8, r)[1]),
                             catch_at_q0p9=float(1 - M.bayes_rates(d, 0.9, r)[1]),
                             catch_at_q0p99=float(1 - M.bayes_rates(d, 0.99, r)[1]),
                             unlawful_per_nom_q0p8=float(0.2 * M.bayes_rates(d, 0.8, r)[1]),
                             unlawful_per_nom_q0p9=float(0.1 * M.bayes_rates(d, 0.9, r)[1]),
                             unlawful_per_nom_q0p99=float(0.01 * M.bayes_rates(d, 0.99, r)[1]),
                             dstar_k0p5_q0p9=float(M.bayes_dstar(0.5, 0.9, r)),
                             dstar_k0p5_q0p99=float(M.bayes_dstar(0.5, 0.99, r)))
    axes[0].set_ylabel("d* for 50% catch (Bayes reviewer)"); axes[0].set_ylim(0, 6)
    axes[1].set_ylabel("catch rate at d' = 1")
    axes[2].set_ylabel("unlawful approvals per nomination at d' = 1")
    for ax in axes: ax.set_xlabel("machine precision q")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig3_precision_paradox.png")
    # automation bias effect
    ab = {}
    for a in [0, 0.5, 1.0, 2.0]:
        ab[f"alpha={a}"] = dict(dstar_k0p5_q0p9_r5=float(M.bayes_dstar(0.5, 0.9, 5, a)),
                                catch_d1_q0p9_r5=float(1 - M.bayes_rates(1.0, 0.9, 5, a)[1]))
    RES["E4"] = dict(bayes_d1=out, automation_bias=ab,
                     sign_switch_q_for_r={r: r / (1 + r) for r in [1, 5, 20, 100]})


# ---------------------------------------------------------------- E5
def e5_volume_elasticity():
    lam = np.geomspace(1, 2000, 400)  # nominations per operator-hour
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    out = {}
    q = 0.9
    for i, (D, tau) in enumerate([(2.0, 15.0), (3.0, 60.0), (4.0, 240.0)]):
        t = M.review_time(lam)
        d = M.d_of_t(t, D, tau, 2.0)
        _, fa = M.quota_rates(d, 0.95)
        u = M.unlawful_per_hour(lam, q, fa)
        el = np.gradient(np.log(u), np.log(lam))
        axes[0].loglog(lam, u, color=C[i], label=f"D={D:g}, tau={tau:g}s")
        axes[1].semilogx(lam, el, color=C[i])
        out[f"D={D:g},tau={tau:g}"] = dict(max_elasticity=float(el.max()),
                                           lam_at_max=float(lam[el.argmax()]),
                                           elasticity_at_180ph=float(np.interp(np.log(180), np.log(lam), el)))
    axes[0].loglog(lam, lam * (1 - q), "k:", lw=1, label="no human filter")
    axes[0].set_xlabel("nominations per operator-hour"); axes[0].set_ylabel("unlawful approvals per operator-hour")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("nominations per operator-hour"); axes[1].set_ylabel("elasticity d ln U / d ln lambda")
    save(fig, "fig4_volume_elasticity.png")
    RES["E5"] = out


# ---------------------------------------------------------------- E6
def e6_surge_simulation(days=2000, N_ops=10, mean_per_hr=300.0, eta=0.8, t_pref=120.0):
    """Hourly time-slotted queue. Each hour, arrivals A_h ~ NegBin with mean
    mean_per_hr and dispersion k (k=inf -> Poisson). Operators must clear the
    hour's arrivals (no backlog allowed: targets are time-sensitive), so review
    time t_h = min(t_pref, 3600*eta*N_ops/A_h). Catch rate under quota h=0.95."""
    D, tau, delta, q = 3.0, 60.0, 2.0, 0.9
    hours = days * 24
    res = {}
    for label, k in [("poisson", None), ("negbin_k=5", 5.0), ("negbin_k=1", 1.0), ("negbin_k=0.3", 0.3)]:
        if k is None:
            A = rng.poisson(mean_per_hr, hours)
        else:
            p = k / (k + mean_per_hr)
            A = rng.negative_binomial(k, p, hours)
        A = np.maximum(A, 0)
        t = np.where(A > 0, np.minimum(t_pref, 3600 * eta * N_ops / np.maximum(A, 1)), t_pref)
        d = M.d_of_t(t, D, tau, delta)
        _, fa = M.quota_rates(d, 0.95)
        errors = A * (1 - q)
        caught = (errors * (1 - fa)).sum()
        res[label] = dict(mean_arrivals=float(A.mean()), cv=float(A.std() / A.mean()),
                          error_weighted_catch=float(caught / errors.sum()),
                          unlawful_per_day=float((errors * fa).sum() / days))
    # constant-load reference
    t0 = min(t_pref, 3600 * eta * N_ops / mean_per_hr)
    _, fa0 = M.quota_rates(M.d_of_t(t0, D, tau, delta), 0.95)
    res["constant_load"] = dict(t=float(t0), catch=float(1 - fa0),
                                unlawful_per_day=float(24 * mean_per_hr * (1 - q) * fa0))
    RES["E6"] = dict(params=dict(N_ops=N_ops, mean_per_hr=mean_per_hr, eta=eta, t_pref=t_pref,
                                 D=D, tau=tau, delta=delta, q=q, h=0.95, days=days), results=res)


# ---------------------------------------------------------------- E7
def e7_reported_scenarios():
    """Illustrative calculations using publicly reported figures (see paper):
    (a) ~20 s per Lavender nomination, ~90% sampled precision, up to 37,000
    nominations; (b) Gospel: ~100 targets/day (Kochavi). Parameters D,tau are
    NOT known: we report the full sensitivity grid."""
    out = {"a_20s_catch_rate_quota95": {}, "a_operator_hours_for_37000": {}, "b_100_per_day": {}}
    for D, tau in itertools.product(D_GRID, TAU_GRID):
        d = float(M.d_of_t(20.0, D, tau, 2.0)); _, fa = M.quota_rates(d, 0.95)
        out["a_20s_catch_rate_quota95"][f"D={D:g},tau={tau:g}"] = float(1 - fa)
        for kappa in [0.5, 0.9]:
            ts = float(M.t_of_d(M.quota_dstar(kappa, 0.95), D, tau, 2.0))
            out["a_operator_hours_for_37000"][f"D={D:g},tau={tau:g},kappa={kappa}"] = (
                37000 * ts / 3600 if np.isfinite(ts) else None)
            out["b_100_per_day"][f"D={D:g},tau={tau:g},kappa={kappa}"] = (
                100 * ts / 3600 if np.isfinite(ts) else None)
    out["a_reported_20s_total_hours_37000"] = 37000 * 20 / 3600
    out["a_expected_errors_at_q0.9"] = 37000 * 0.1
    RES["E7"] = out


# ---------------------------------------------------------------- E8
def e8_audit(n_rep=4000):
    """Audit protocol: estimate catch rate kappa from a stratified ground-truth
    audit of n_a approved and n_v vetoed nominations, combined with the logged
    veto rate V. Reports bias and 95% interval coverage via parametric bootstrap."""
    q, d, h = 0.9, 1.0, 0.95
    hit, fa = M.quota_rates(d, h); hit, fa = float(hit), float(fa)
    true_kappa = 1 - fa
    V = M.veto_rate(q, hit, fa)
    p_inv_given_app = (1 - q) * fa / (1 - V)
    p_inv_given_veto = (1 - q) * (1 - fa) / V
    rows = []
    for n in [50, 100, 200, 400, 800]:
        est = []; cover = 0
        for _ in range(n_rep):
            a = rng.binomial(n, p_inv_given_app) / n
            v = rng.binomial(n, p_inv_given_veto) / n
            inv_app = (1 - V) * a; inv_veto = V * v
            k_hat = inv_veto / (inv_app + inv_veto) if inv_app + inv_veto > 0 else np.nan
            # delta-method-free: bootstrap 200 inner reps
            bs = []
            for _ in range(200):
                a2 = rng.binomial(n, a) / n; v2 = rng.binomial(n, v) / n
                ia, iv = (1 - V) * a2, V * v2
                if ia + iv > 0: bs.append(iv / (ia + iv))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cover += lo <= true_kappa <= hi
            est.append(k_hat)
        est = np.array(est)
        rows.append(dict(n_per_stratum=n, mean_est=float(np.nanmean(est)),
                         sd=float(np.nanstd(est)), coverage95=cover / n_rep))
        if n == 200: break_marker = True
    RES["E8"] = dict(true_kappa=true_kappa, veto_rate=float(V),
                     lower_bound_check=dict(V=float(V), bound=(1 - q) * true_kappa,
                                            holds=bool(V >= (1 - q) * true_kappa)),
                     rows=rows)


# ---------------------------------------------------------------- E4b
def e4b_nonmonotone_region():
    """Region of (q, r) where a more precise machine INCREASES unlawful approvals
    per nomination under a Bayes reviewer with fixed d'. Analytic condition
    (Prop. 4): dU/dq > 0  iff  phi(a)/Phi(a) > q d,  a = -d/2 - log_beta/d.
    Checked against a numerical derivative."""
    qs = np.linspace(0.5, 0.995, 300); rs = np.geomspace(1, 1000, 300)
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, d in zip(axes, [0.5, 1.0, 2.0]):
        Q, R = np.meshgrid(qs, rs)
        lb = np.log((1 - Q) / Q) + np.log(R)
        a = -d / 2 - lb / d
        analytic = norm.pdf(a) / np.maximum(norm.cdf(a), 1e-300) > Q * d
        U = lambda q: (1 - q) * norm.cdf(-d / 2 - (np.log((1 - q) / q) + np.log(R)) / d)
        h = 1e-5
        numeric = (U(Q + h) - U(Q - h)) / (2 * h) > 0
        agree = float((analytic == numeric).mean())
        out[f"d={d}"] = dict(share_region_increasing=float(analytic.mean()), agreement=agree)
        ax.pcolormesh(qs, rs, analytic, shading="auto", cmap="Greys", vmin=0, vmax=2.2)
        ax.plot(qs, qs / (1 - qs), "r--", lw=1, label="log_beta = 0")
        ax.set_yscale("log"); ax.set_ylim(1, 1000); ax.set_xlabel("machine precision q")
        ax.set_title(f"d' = {d}: grey = more precise machine -> more unlawful approvals", fontsize=8)
    axes[0].set_ylabel("cost ratio r = C_F / C_M"); axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig5_nonmonotone_region.png")
    RES["E4b"] = out


if __name__ == "__main__":
    t0 = time.time()
    e1_validate(); print("E1", RES["E1"])
    e2_catch_vs_time(); print("E2 done")
    e3_lambda_max(); print("E3 done")
    e4_precision_paradox(); print("E4 done")
    e4b_nonmonotone_region(); print("E4b", RES["E4b"])
    e5_volume_elasticity(); print("E5 done")
    e6_surge_simulation(); print("E6 done")
    e7_reported_scenarios(); print("E7 done")
    e8_audit(n_rep=1000); print("E8 done")
    RES["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(RES, f, indent=1, default=float)
    print(json.dumps(RES, indent=1, default=float))
