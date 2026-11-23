"""Run every experiment of the paper. Usage:  python3 run_all.py   (approx. a few minutes on CPU)
Outputs: ../results/*.json, ../results/*.png
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import BASE, h, dh, a_star, U_star, G, steady_states, simulate, elasticity

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
RES = {}
C = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b7791f"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def P(**kw):
    p = dict(BASE); p.update(kw); return p


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


TYPES = {"secure (S0*=1.0)": 0.020, "fragile (S0*=0.6)": 0.012}

# ---------------------------------------------------------------- E1 bifurcation
def e1():
    qs = np.linspace(0.0, 1.0, 201)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    out = {}
    for ax, (lab, eta) in zip(axs, TYPES.items()):
        p = P(eta=eta)
        out[lab] = {}
        for k, rho in enumerate([0.0, 0.5, 1.0]):
            b = rho * eta
            st, un = [], []
            folds = []
            prev_n = None
            for q in qs:
                ss = steady_states(q, b, p)
                for s, stable in ss:
                    (st if stable else un).append((q, s))
                nst = len(ss)
                if prev_n is not None and nst != prev_n:
                    folds.append(round(float(q), 3))
                prev_n = nst
            out[lab][f"rho={rho}"] = {"n_ss_change_at_q": folds}
            if st:
                x, y = zip(*st); ax.plot(x, y, ".", ms=2, color=C[k], label=f"ρ={rho} stable")
            if un:
                x, y = zip(*un); ax.plot(x, y, "x", ms=2, color=C[k], alpha=.5)
        ax.set_title(lab); ax.set_xlabel("AI companion quality q")
    axs[0].set_ylabel("steady-state social capital S*")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_bifurcation.png"), dpi=160); plt.close(fig)
    RES["E1_folds"] = out


# ---------------------------------------------------------------- E2 phase diagram
def long_run(S0, q, b, p, T=4000):
    S, A, U = simulate(S0, q, b, T, p)
    return S[-1], A[-1], U[-1]


def e2():
    qs = np.linspace(0.01, 1.0, 100)
    rhos = np.linspace(-0.5, 1.5, 81)
    fig, axs = plt.subplots(2, 2, figsize=(8, 6.4))
    out = {}
    for j, (lab, eta) in enumerate(TYPES.items()):
        p = P(eta=eta); S0 = eta / p["delta"]; h0 = float(h(S0, p))
        QQ, RR = np.meshgrid(qs, rhos)
        S_inf, A_inf, U_inf = long_run(np.full(QQ.size, S0), QQ.ravel(), (RR * eta).ravel(), p)
        dU = (U_inf - h0).reshape(QQ.shape)
        lam = 0.3
        dW = dU + lam * ((S_inf - S0).reshape(QQ.shape))
        eps = float(elasticity(S0, p))
        for i, (Z, nm) in enumerate([(dU, "ΔU∞ (experienced connection)"), (dW, "ΔW∞ = ΔU∞ + 0.3·ΔS∞")]):
            ax = axs[i, j]
            lim = np.nanmax(np.abs(Z))
            im = ax.pcolormesh(qs, rhos, Z, cmap="RdBu", vmin=-lim, vmax=lim, shading="auto")
            ax.contour(qs, rhos, Z, levels=[0], colors="k", linewidths=1)
            ax.axhline(1 - 0.5 / eps, color="k", ls="--", lw=.8)
            ax.axvline(h0, color="gray", ls=":", lw=.8)
            ax.set_title(f"{lab}: {nm}", fontsize=8)
            ax.set_xlabel("q"); ax.set_ylabel("bridging ratio ρ = b/η")
            fig.colorbar(im, ax=ax, fraction=.046)
        harm_frac_U = float((dU < 0).mean()); harm_frac_W = float((dW < 0).mean())
        # harm valley: for rho=0, q range with dU<0
        r0 = np.argmin(np.abs(rhos - 0.0))
        qneg = qs[dU[r0] < 0]
        out[lab] = dict(S0=S0, h_S0=h0, elasticity=eps, local_threshold_rho=1 - 0.5 / eps,
                        frac_grid_harm_U=harm_frac_U, frac_grid_harm_W=harm_frac_W,
                        rho0_harm_q_range=[float(qneg.min()), float(qneg.max())] if qneg.size else None,
                        rho0_min_dU=float(dU[r0].min()), rho0_argmin_q=float(qs[np.argmin(dU[r0])]))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_phase.png"), dpi=160); plt.close(fig)
    RES["E2_phase"] = out


# ---------------------------------------------------------------- E2b local-criterion check
def e2b():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(4000):
        p = P(delta=rng.uniform(0.005, 0.05), kappa=rng.uniform(0.3, 1.2), n=rng.uniform(1.5, 6),
              h0=rng.uniform(0.05, 0.4))
        p["eta"] = p["delta"] * rng.uniform(0.3, 1.5)  # S0* in [0.3,1.5]
        S0 = p["eta"] / p["delta"]
        rho = rng.uniform(-0.5, 1.5)
        q = 0.05 * h(S0, p)  # small-q regime
        S, A, U = simulate(S0, q, rho * p["eta"], int(30 / p["delta"]), p)
        dU = U[-1, 0] - h(S0, p)
        crit = (1 - rho) * elasticity(S0, p) > 0.5
        rows.append((bool(dU < 0), bool(crit)))
    rows = np.array(rows)
    agree = float((rows[:, 0] == rows[:, 1]).mean())
    RES["E2b_local_criterion_small_q_agreement"] = dict(n=len(rows), agreement=agree,
                                                        frac_harm=float(rows[:, 0].mean()))


# ---------------------------------------------------------------- E3 time paths & sign flip
def e3():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    out = {}
    T = 520
    for ax, (lab, eta) in zip(axs, TYPES.items()):
        p = P(eta=eta); S0 = eta / p["delta"]; h0 = h(S0, p)
        out[lab] = {}
        for k, (q, rho) in enumerate([(0.3, 0.0), (0.3, 0.5), (0.3, 1.0), (0.5, 0.0)]):
            S, A, U = simulate(S0, q, rho * eta, T, p)
            g = U[:, 0] - h0
            neg = np.where(g < 0)[0]
            tflip = int(neg[0]) if neg.size else None
            out[lab][f"q={q},rho={rho}"] = dict(gain_week0=float(g[0]), gain_week4=float(g[4]),
                                                gain_week52=float(g[52]), gain_week520=float(g[-1]),
                                                sign_flip_week=tflip)
            ax.plot(np.arange(T), g, color=C[k], label=f"q={q}, ρ={rho}")
        ax.axhline(0, color="k", lw=.6); ax.set_xscale("symlog", linthresh=10)
        ax.set_title(lab); ax.set_xlabel("weeks since adoption")
    axs[0].set_ylabel("U_t − U_no-AI (experienced connection)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_timepaths.png"), dpi=160); plt.close(fig)
    # table of sign-flip weeks (deterministic, fragile + secure)
    tab = {}
    for lab, eta in TYPES.items():
        p = P(eta=eta); S0 = eta / p["delta"]; h0 = h(S0, p)
        tab[lab] = {}
        for rho in [0.0, 0.25, 0.5, 0.75]:
            for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                S, A, U = simulate(S0, q, rho * eta, 2000, p)
                neg = np.where(U[:, 0] - h0 < 0)[0]
                tab[lab][f"rho={rho},q={q}"] = int(neg[0]) if neg.size else None
    out["sign_flip_table_weeks"] = tab
    RES["E3_timepaths"] = out


# ---------------------------------------------------------------- population helper
def population(N, rng, eta_mean=0.016, eta_cv=0.3, qbar=0.35, qsd=0.1):
    sig = np.sqrt(np.log(1 + eta_cv ** 2))
    eta_i = eta_mean * np.exp(sig * rng.standard_normal(N) - sig ** 2 / 2)
    q_i = np.clip(qbar + qsd * rng.standard_normal(N), 0.02, 1.0)
    return eta_i, q_i


def burn_in(eta_i, p, rng, sigma, T=600):
    S0 = eta_i / p["delta"]
    S, _, _ = simulate(S0, 0.0, 0.0, T, p, eta_i=eta_i, sigma=sigma, rng=rng)
    return S[-1]


# ---------------------------------------------------------------- E4 heterogeneity
def e4():
    rng = np.random.default_rng(4)
    N = 20000; p = P(); sigma = 0.01
    eta_i, q_i = population(N, rng)
    S0 = burn_in(eta_i, p, rng, sigma)
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, rho in zip(axs, [0.0, 0.5]):
        b = rho * eta_i
        seed = rng.integers(1 << 30)
        S, A, U = simulate(S0, q_i, b, 260, p, eta_i=eta_i, sigma=sigma, rng=np.random.default_rng(seed))
        Sc, Ac, Uc = simulate(S0, 0.0, 0.0, 260, p, eta_i=eta_i, sigma=sigma, rng=np.random.default_rng(seed))
        sr = U[0] - Uc[0]
        lr = U[-52:].mean(0) - Uc[-52:].mean(0)
        r = float(np.corrcoef(sr, lr)[0, 1])
        lonely = S0 < np.quantile(S0, 0.25)
        out[f"rho={rho}"] = dict(corr_shortrun_longrun=r, frac_longrun_harm=float((lr < 0).mean()),
                                 frac_harm_lonelyQ1=float((lr[lonely] < 0).mean()),
                                 frac_harm_topQ=float((lr[S0 > np.quantile(S0, .75)] < 0).mean()),
                                 mean_sr_gain_lonelyQ1=float(sr[lonely].mean()),
                                 mean_sr_gain_others=float(sr[~lonely].mean()),
                                 mean_lr=float(lr.mean()))
        idx = rng.choice(N, 3000, replace=False)
        sc = ax.scatter(sr[idx], lr[idx], c=S0[idx], s=3, cmap="viridis")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("short-run gain (week 0)"); ax.set_ylabel("long-run change (years 4-5)")
        ax.set_title(f"ρ={rho}: corr={r:.2f}")
    fig.colorbar(sc, ax=axs[-1], label="pre-adoption S")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_heterogeneity.png"), dpi=160); plt.close(fig)
    RES["E4_heterogeneity"] = out


# ---------------------------------------------------------------- E5 platform design
def e5():
    rng = np.random.default_rng(5)
    N = 4000; p = P(); sigma = 0.01
    eta_i, _ = population(N, rng)
    S0 = burn_in(eta_i, p, rng, sigma)
    qs = np.linspace(0.05, 1.0, 20); rhos = np.linspace(-0.5, 1.5, 21)
    T = 260; disc = 0.995 ** np.arange(T)
    E = np.zeros((len(rhos), len(qs))); W = np.zeros_like(E); WL = np.zeros_like(E)
    seed = 123
    Sc, _, Uc = simulate(S0, 0.0, 0.0, T, p, eta_i=eta_i, sigma=sigma, rng=np.random.default_rng(seed))
    base_W = float((disc[:, None] * Uc).sum(0).mean() / disc.sum())
    for i, rho in enumerate(rhos):
        for j, q in enumerate(qs):
            S, A, U = simulate(S0, q, rho * eta_i, T, p, eta_i=eta_i, sigma=sigma, rng=np.random.default_rng(seed))
            E[i, j] = A.mean()
            W[i, j] = (disc[:, None] * U).sum(0).mean() / disc.sum() - base_W
            WL[i, j] = W[i, j] + 0.3 * ((disc[:, None] * (S[1:] - Sc[1:])).sum(0).mean() / disc.sum())
    out = {}
    for nm, Z in [("engagement", E), ("welfare_U", W), ("welfare_U_plus_0.3S", WL)]:
        i, j = np.unravel_index(np.argmax(Z), Z.shape)
        out[nm] = dict(argmax_rho=float(rhos[i]), argmax_q=float(qs[j]), max=float(Z[i, j]))
    # engagement-maximiser restricted to rho in [0,1]
    sub = (rhos >= 0) & (rhos <= 1)
    i, j = np.unravel_index(np.argmax(E[sub]), E[sub].shape)
    out["engagement_rho_in_0_1"] = dict(argmax_rho=float(rhos[sub][i]), argmax_q=float(qs[j]),
                                        welfare_U_at_that=float(W[sub][i, j]))
    # monotonicity checks
    out["engagement_decreasing_in_rho_all_q"] = bool(np.all(np.diff(E, axis=0) <= 1e-9))
    out["welfare_U_increasing_in_rho_all_q"] = bool(np.all(np.diff(W, axis=0) >= -1e-9))
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
    for ax, (nm, Z, cm) in zip(axs, [("engagement (mean a)", E, "viridis"),
                                     ("Δ welfare (disc. mean U)", W, "RdBu"),
                                     ("Δ welfare incl. 0.3·S", WL, "RdBu")]):
        if cm == "RdBu":
            lim = np.abs(Z).max(); im = ax.pcolormesh(qs, rhos, Z, cmap=cm, vmin=-lim, vmax=lim, shading="auto")
            ax.contour(qs, rhos, Z, levels=[0], colors="k", linewidths=.8)
        else:
            im = ax.pcolormesh(qs, rhos, Z, cmap=cm, shading="auto")
        i, j = np.unravel_index(np.argmax(Z), Z.shape); ax.plot(qs[j], rhos[i], "w*", ms=10, mec="k")
        ax.set_title(nm, fontsize=8); ax.set_xlabel("q"); ax.set_ylabel("ρ")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_design.png"), dpi=160); plt.close(fig)
    RES["E5_design"] = out


# ---------------------------------------------------------------- E6 forward-looking user (DP)
def e6():
    out = {}
    for lab, eta in TYPES.items():
        p = P(eta=eta); S0 = eta / p["delta"]
        Sg = np.linspace(0, 1.6, 801); ag = np.linspace(0, 1, 201); beta = 0.995
        for q, rho in [(0.3, 0.0), (0.5, 0.0), (0.3, 0.5)]:
            b = rho * eta
            hS = h(Sg, p)[:, None]
            Uf = q * np.sqrt(ag)[None, :] + hS * np.sqrt(1 - ag)[None, :]
            Sn = (1 - p["delta"]) * Sg[:, None] + eta * (1 - ag)[None, :] + b * ag[None, :]
            V = np.zeros(len(Sg))
            for it in range(6000):
                Vn = (Uf + beta * np.interp(Sn, Sg, V)).max(1)
                if np.max(np.abs(Vn - V)) < 1e-9: V = Vn; break
                V = Vn
            pol = ag[(Uf + beta * np.interp(Sn, Sg, V)).argmax(1)]
            # simulate forward-looking from S0
            S = S0; Ufl = []
            for t in range(3000):
                a = np.interp(S, Sg, pol); Ufl.append(q * np.sqrt(a) + h(S, p) * np.sqrt(1 - a))
                S = (1 - p["delta"]) * S + eta * (1 - a) + b * a
            Sm, Am, Um = simulate(S0, q, b, 3000, p)
            dsc = beta ** np.arange(3000)
            out[f"{lab}|q={q},rho={rho}"] = dict(
                S_inf_forward=float(S), a_inf_forward=float(np.interp(S, Sg, pol)),
                S_inf_myopic=float(Sm[-1, 0]), a_inf_myopic=float(Am[-1, 0]),
                Wdisc_forward=float((dsc * np.array(Ufl)).sum() * (1 - beta)),
                Wdisc_myopic=float((dsc * Um[:, 0]).sum() * (1 - beta)),
                U_noAI=float(h(S0, p)))
    RES["E6_forward_vs_myopic"] = out


# ---------------------------------------------------------------- E7 empirical designs
def e7():
    rng = np.random.default_rng(7)
    p = P(); sigma = 0.01; noiseL = 0.05
    out = {}
    # (a) cross-sectional correlation usage vs loneliness, (b) RCT by duration, (c) cross-lagged panel
    N = 6000
    eta_i, q_i = population(N, rng)
    S0 = burn_in(eta_i, p, rng, sigma)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    durations = [1, 2, 4, 8, 13, 26, 52, 104, 156, 260]
    for k, rho in enumerate([0.0, 0.5, 1.0, 1.2]):
        b = rho * eta_i
        treat = rng.random(N) < 0.5
        seed = int(rng.integers(1 << 30))
        q_eff = np.where(treat, q_i, 0.0)
        # time-varying quality (novelty/updates) to create within-person variation in use
        T = 261
        qt = q_eff[None, :] * np.exp(0.25 * rng.standard_normal((T, N)))
        S = np.empty((T + 1, N)); S[0] = S0; A = np.empty((T, N)); U = np.empty((T, N))
        r2 = np.random.default_rng(seed)
        for t in range(T):
            a = a_star(S[t], qt[t], p); hh = h(S[t], p)
            A[t] = a; U[t] = qt[t] * np.sqrt(a) + hh * np.sqrt(1 - a)
            S[t + 1] = np.maximum(0, (1 - p["delta"]) * S[t] + eta_i * (1 - a) + b * a + sigma * r2.standard_normal(N))
        L = 1 - U + noiseL * rng.standard_normal(U.shape)
        ate, se = [], []
        for d in durations:
            x1, x0 = L[d, treat], L[d, ~treat]
            ate.append(float(x1.mean() - x0.mean()))
            se.append(float(np.sqrt(x1.var() / x1.size + x0.var() / x0.size)))
        # cross-sectional among treated at week 52 and 260
        cs = {f"week{w}": float(np.corrcoef(A[w, treat], L[w, treat])[0, 1]) for w in [52, 260]}
        # cross-lagged panel among treated, waves every 16 weeks starting week 16 (4 waves)
        waves = [16, 32, 48, 64]
        def z(x): return (x - x.mean()) / x.std()
        bl, bu = [], []
        for w0, w1 in zip(waves[:-1], waves[1:]):
            Lw0 = L[w0, treat]; Lw1 = L[w1, treat]
            Aw0 = A[w0 - 4:w0, treat].mean(0); Aw1 = A[w1 - 4:w1, treat].mean(0)
            X = np.column_stack([np.ones(treat.sum()), z(Lw0), z(Aw0)])
            bl.append(np.linalg.lstsq(X, z(Lw1), rcond=None)[0][2])   # use -> later loneliness
            bu.append(np.linalg.lstsq(X, z(Aw1), rcond=None)[0][1])   # loneliness -> later use
        # true long-run causal effect on S
        out[f"rho={rho}"] = dict(rct_ate_L=dict(zip(map(str, durations), ate)), rct_se=dict(zip(map(str, durations), se)),
                                 crosssec_corr_use_loneliness=cs,
                                 crosslag_use_to_later_L=[float(x) for x in bl],
                                 crosslag_L_to_later_use=[float(x) for x in bu],
                                 true_mean_dS_week260=float(S[260, treat].mean() - S[260, ~treat].mean()))
        axs[0].errorbar(durations, ate, yerr=1.96 * np.array(se), color=C[k], marker="o", ms=3, label=f"ρ={rho}")
    axs[0].axhline(0, color="k", lw=.6); axs[0].set_xscale("log")
    axs[0].set_xlabel("trial duration (weeks)"); axs[0].set_ylabel("ATE on loneliness (1−U)")
    axs[0].legend(frameon=False, fontsize=7); axs[0].set_title("RCT estimate depends on duration")
    for x in [4, 52]: axs[0].axvline(x, color="gray", ls=":", lw=.7)
    # (d) estimating displacement D with randomized availability
    est = {}
    reps = 60
    for rho in [0.0, 0.5, 1.0, 1.5]:
        D_true = p["eta"] * (1 - rho)  # with eta_i heterogeneous, use mean eta normalisation below
        ols, wald, xs = [], [], []
        for r in range(reps):
            rr = np.random.default_rng(1000 + r)
            n, T = 1500, 52
            eta_u = np.full(n, 0.016); bb = rho * 0.016
            Dt = 0.016 * (1 - rho)
            S = rr.uniform(0.3, 1.2, n); q = np.clip(0.35 + 0.1 * rr.standard_normal(n), .02, 1)
            rowsY, rowsA, rowsS, rowsZ = [], [], [], []
            em = 0.05
            St = S + em * rr.standard_normal(n)
            for t in range(T):
                xi = rr.standard_normal(n)
                zt = (rr.random(n) < 0.8).astype(float)
                hh = h(S, p) * np.exp(0.3 * xi)
                a = zt * q ** 2 / (q ** 2 + hh ** 2)
                Snew = np.maximum(0, (1 - p["delta"]) * S + eta_u * (1 - a) + bb * a + 0.01 * xi + 0.005 * rr.standard_normal(n))
                Stn = Snew + em * rr.standard_normal(n)
                rowsY.append(Stn - St); rowsA.append(a); rowsS.append(St); rowsZ.append(zt)
                S, St = Snew, Stn
            Y = np.concatenate(rowsY); Aa = np.concatenate(rowsA); Ss = np.concatenate(rowsS); Zz = np.concatenate(rowsZ)
            X = np.column_stack([np.ones_like(Y), Aa, Ss])
            ols.append(-np.linalg.lstsq(X, Y, rcond=None)[0][1])
            wald.append(-(Y[Zz == 1].mean() - Y[Zz == 0].mean()) / (Aa[Zz == 1].mean() - Aa[Zz == 0].mean()))
        est[f"rho={rho}"] = dict(D_true=Dt, ols_mean=float(np.mean(ols)), ols_sd=float(np.std(ols)),
                                 wald_mean=float(np.mean(wald)), wald_sd=float(np.std(wald)),
                                 ols_sign_correct=float(np.mean(np.sign(ols) == np.sign(Dt))) if Dt != 0 else None,
                                 wald_sign_correct=float(np.mean(np.sign(wald) == np.sign(Dt))) if Dt != 0 else None)
    out["D_estimation"] = est
    keys = list(est)
    xt = np.arange(len(keys))
    axs[1].bar(xt - 0.25, [est[k]["D_true"] for k in keys], 0.25, color="gray", label="true D")
    axs[1].bar(xt, [est[k]["ols_mean"] for k in keys], 0.25, yerr=[est[k]["ols_sd"] for k in keys], color=C[1], label="naive OLS")
    axs[1].bar(xt + 0.25, [est[k]["wald_mean"] for k in keys], 0.25, yerr=[est[k]["wald_sd"] for k in keys], color=C[0], label="randomized-availability Wald")
    axs[1].set_xticks(xt); axs[1].set_xticklabels(keys); axs[1].axhline(0, color="k", lw=.6)
    axs[1].set_ylabel("displacement D = η − b"); axs[1].legend(frameon=False, fontsize=7)
    axs[1].set_title("Identifying displacement")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_designs.png"), dpi=160); plt.close(fig)
    RES["E7_designs"] = out


if __name__ == "__main__":
    t0 = time.time()
    for f in [e1, e2, e2b, e3, e4, e5, e6, e7]:
        t = time.time(); f(); print(f.__name__, f"{time.time()-t:.1f}s", flush=True)
        save("results.json", RES)
    RES["runtime_s"] = time.time() - t0
    save("results.json", RES)
    print("done", RES["runtime_s"])
