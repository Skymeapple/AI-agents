"""Revision (R1) linear experiments.
 A. E1 re-done with a genuine ODE integrator (scipy DOP853, rtol 1e-13) instead of the eigen-solution on a grid.
 B. Recovery time (Thm 7): last-exit time T_eps = sup{t : L1(t) >= eps*F_inf}, eps = 0.1, over rho in [1e-5, 0.1];
    checked against the proven lower/upper bounds and the singular-perturbation constant T*_eps (Prop. 8).
 C. SGD with fixed-composition replay mini-batches (Thm 9): exact second-moment recursion, Monte-Carlo check,
    one-step law, gap depth vs batch size.
 D. Finite replay buffer (Prop. 10): m of the n1 = 40 task-1 samples kept; terminal + transient forgetting.
 E. Audit of the prose numbers flagged in review R1, recomputed from results/linear_results.json.
Output: ../results/linear_revision.json, fig5_linear_revision.png
"""
import numpy as np, json, os, time, sys
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(here, "..", "results")
from linear_theory_lib import null_proj, L1_traj_flow
d, n1, n2, k_shared = 100, 40, 40, 20
out = {}; t00 = time.time()


def make_tasks_X(seed):
    """identical RNG sequence to linear_theory_lib.make_tasks(False, seed), but also returns X1, X2"""
    rg_ = np.random.default_rng(seed)
    B = np.linalg.qr(rg_.standard_normal((d, d)))[0]
    U1 = B[:, :60]; U2 = np.concatenate([B[:, 60-k_shared:60], rg_.standard_normal((d, 60-k_shared))], 1)
    U2 = np.linalg.qr(U2)[0]
    X1 = rg_.standard_normal((n1, 60)) * np.linspace(0.3, 2, 60) @ U1.T
    X2 = rg_.standard_normal((n2, 60)) * np.linspace(0.3, 2, 60) @ U2.T
    return X1, X2


def setup(seed):
    X1, X2 = make_tasks_X(seed); S1, S2 = X1.T @ X1 / n1, X2.T @ X2 / n2
    e0 = null_proj(S1) @ np.random.default_rng(200 + seed).standard_normal(d); e0 /= np.linalg.norm(e0)
    return X1, X2, S1, S2, e0

PARTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A", "B", "C", "D", "E", "FIG"]
def save(part, obj):
    json.dump(obj, open(os.path.join(R, f"linear_revision_{part}.json"), "w"), indent=1, default=str)

# ---------------- A. E1 with a real ODE solver ----------------
if "A" in PARTS:
    def closed_form_2d(theta, rho):
        u = np.array([1.0, 0.0]); v = np.array([np.cos(theta), np.sin(theta)])
        H = (1-rho)*np.outer(v, v) + rho*np.outer(u, u); lam, Q = eigh(H)
        s = (u @ Q[:, 0]) * Q[1, 0]; r = lam[0]/lam[1]
        return 0.5*s**2*(1-r)**2*r**(2*r/(1-r)), np.log(lam[1]/lam[0])/(lam[1]-lam[0])

    th = np.pi/4; u = np.array([1.0, 0.0]); v = np.array([np.cos(th), np.sin(th)])
    S1_2, S2_2 = np.outer(u, u), np.outer(v, v)
    rowsA = []
    for rho in [0.01, 0.03, 0.1, 0.3, 0.5]:
        H = (1-rho)*S2_2 + rho*S1_2
        sol = solve_ivp(lambda t, e: -H @ e, (0, 60), [0.0, 1.0], method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
        L = lambda t: 0.5*(sol.sol(t)[0])**2
        tg = np.linspace(0, 60, 60001); i = int(np.argmax(L(tg)))
        res = minimize_scalar(lambda t: -L(t), bounds=(tg[max(i-1, 0)], tg[i+1]), method="bounded", options=dict(xatol=1e-12))
        G, ts = closed_form_2d(th, rho)
        rowsA.append(dict(rho=rho, G_closed=G, G_ode=float(-res.fun), tstar_closed=ts, tstar_ode=float(res.x),
                          relerr_G=float(abs(-res.fun-G)/G), abserr_tstar=float(abs(res.x-ts))))
    save("A", {"A_E1_ode": rowsA})
    print("A done", f"{time.time()-t00:.0f}s", flush=True)

# ---------------- B. recovery time ----------------
if "B" in PARTS:
    EPS = 0.1
    rhosB = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


    def last_exit(f, level, tmax):
        tg = np.concatenate([[0], np.geomspace(1e-3, tmax, 6000)]); vals = f(tg)
        above = np.where(vals >= level)[0]
        if len(above) == 0: return 0.0
        j = above[-1]
        if j == len(tg) - 1: return np.inf
        return brentq(lambda t: f(np.array([t]))[0] - level, tg[j], tg[j+1], xtol=1e-10, rtol=1e-12)

    rowsB = []
    for seed in range(10):
        X1, X2, S1, S2, e0 = setup(seed)
        l1 = eigh(S1, eigvals_only=True)[-1]
        lam2 = eigh(S2, eigvals_only=True); mu = lam2[lam2 > 1e-9*lam2[-1]].min()
        PN = null_proj(S2); z0 = PN @ e0; Finf = 0.5*z0 @ S1 @ z0
        beta = np.sqrt(2*Finf/l1)                       # ||e0|| = 1
        lamS = eigh(S1 + S2, eigvals_only=True); nu = lamS[lamS > 1e-9*lamS[-1]].min()
        # slow system (Prop. 8): z' = -M z, M = P_N S1 P_N
        M = PN @ S1 @ PN; lm, Qm = eigh(M); lm = np.clip(lm, 0, None); cm = Qm.T @ z0
        def Lslow(s):
            Z = Qm @ (np.exp(-np.outer(lm, s)) * cm[:, None]); return 0.5*np.einsum('it,ij,jt->t', Z, S1, Z)
        Tstar = last_exit(Lslow, EPS*Finf, 1e7)
        for rho in rhosB:
            H = (1-rho)*S2 + rho*S1; lam = eigh(H, eigvals_only=True); lmin = lam[lam > 1e-12*lam[-1]].min()
            f = lambda t: L1_traj_flow(S1, S2, e0, rho, t)
            T = last_exit(f, EPS*Finf, 50/lmin)
            tb = beta*(1-np.sqrt(EPS))/(2*rho*l1)
            lb_valid = bool(np.exp(-mu*(1-rho)*tb) <= beta*(1-np.sqrt(EPS))/2)
            ub = np.log(1/(EPS*beta**2))/(2*lmin); ub_nu = np.log(1/(EPS*beta**2))/(2*min(rho, 1-rho)*nu)
            # old metric: time after the peak to fall below 10% of the peak
            tg = np.concatenate([[0], np.geomspace(1e-3, 50/lmin, 6000)]); Lg = f(tg); ip = int(Lg.argmax())
            rowsB.append(dict(seed=seed, rho=rho, T_eps=float(T), rhoT=float(rho*T), Tstar_slow=float(Tstar),
                              lower=float(tb), lower_valid=lb_valid, upper=float(ub), upper_nu=float(ub_nu),
                              lmin=float(lmin), beta=float(beta), F_inf=float(Finf), G=float(Lg.max())))
    save("B", {"B_recovery": rowsB})
    print("B done", f"{time.time()-t00:.0f}s", flush=True)

# ---------------- C. SGD with fixed-composition replay batches ----------------
if "C" in PARTS:
    def Qop(X, S):
        w = np.einsum('ij,jk,ik->i', X, S, X); return (X.T * w) @ X / X.shape[0]


    def sgd_moments(X1, X2, S1, S2, e0, rho, B, eta, kmax, stop_frac=0.05):
        """exact E[e e^T] recursion; B=None means full-batch GD. Returns E[L1_k] array."""
        H = (1-rho)*S2 + rho*S1; Mi = np.eye(d) - eta*H; S = np.outer(e0, e0); L = [0.0]
        B1 = 0 if B is None else round(rho*B); B2 = 0 if B is None else B - B1
        for k in range(kmax):
            Snew = Mi @ S @ Mi
            if B is not None:
                noise = np.zeros((d, d))
                if B1: noise += B1*(Qop(X1, S) - S1 @ S @ S1)
                if B2: noise += B2*(Qop(X2, S) - S2 @ S @ S2)
                Snew += eta**2/B**2*noise
            S = 0.5*(Snew + Snew.T); L.append(0.5*np.trace(S1 @ S))
            if k > 50 and L[-1] < stop_frac*max(L): break
        return np.array(L)

    rowsC = []; mcC = {}
    Bs = [10, 50, None]
    for seed in range(5):
        X1, X2, S1, S2, e0 = setup(seed)
        R2 = max((X1**2).sum(1).max(), (X2**2).sum(1).max()); eta = 0.5/R2
        a = float(e0 @ S2 @ S1 @ S2 @ e0); b = float(np.trace(S1 @ Qop(X2, np.outer(e0, e0))))
        for rho in [0.1, 0.3, 0.5]:
            res = {}
            for B in Bs:
                L = sgd_moments(X1, X2, S1, S2, e0, rho, B, eta, 200000, stop_frac=0.5)
                res[B] = L
                pred1 = 0.5*eta**2*(1-rho)**2*a if B is None else 0.5*eta**2*(1-rho)*((1-rho)*a + (b-a)/B)
                rowsC.append(dict(seed=seed, rho=rho, B=B if B else "GD", eta=float(eta), Gexp=float(L.max()),
                                  k_peak=int(L.argmax()), EL1_step1=float(L[1]), pred_step1=float(pred1), a=a, b=b))
            for B in Bs[:-1]:
                assert np.all(res[B][:min(len(res[B]), len(res[None]))] >= res[None][:min(len(res[B]), len(res[None]))] - 1e-15)
        if seed == 0:
            # Monte-Carlo check of the recursion: rho=0.1, B=10, 2000 trajectories, 3000 steps
            rng = np.random.default_rng(7); K = 2000; steps = 3000; rho, B = 0.1, 10; B1 = 1; B2 = 9
            E = np.repeat(e0[:, None], K, 1); Lmc = [np.zeros(K)]
            for k in range(steps):
                i1 = rng.integers(0, n1, (K, B1)); i2 = rng.integers(0, n2, (K, B2))
                Xb = np.concatenate([X1[i1], X2[i2]], 1)          # K x B x d
                r = np.einsum('kbd,dk->kb', Xb, E)
                E = E - eta/B*np.einsum('kbd,kb->dk', Xb, r)
                Lmc.append(0.5*np.einsum('dk,de,ek->k', E, S1, E))
            Lmc = np.array(Lmc); Lex = sgd_moments(X1, X2, S1, S2, e0, rho, B, eta, steps, stop_frac=-1)
            se = Lmc.std(1)/np.sqrt(K); z = (Lmc.mean(1)[1:] - Lex[1:])/se[1:]
            mcC = dict(K=K, steps=steps, rho=rho, B=B, max_abs_z=float(np.abs(z).max()), mean_abs_z=float(np.abs(z).mean()),
                       frac_abs_z_gt2=float((np.abs(z) > 2).mean()), E_peak_exact=float(Lex.max()), E_peak_mc=float(Lmc.mean(1).max()),
                       E_of_max_mc=float(Lmc.max(0).mean()), sd_of_max_mc=float(Lmc.max(0).std()),
                       curve_k=list(range(0, steps+1, 10)), curve_exact=Lex[::10].tolist(), curve_mc=Lmc.mean(1)[::10].tolist(),
                       curve_mc_se=se[::10].tolist())
            gdcurve = res[None]; sgdcurves = {str(B): res[B][:len(gdcurve)].tolist()[::10] for B in Bs[:-1]}
            mcC["fig_curves"] = dict(gd=gdcurve[::10].tolist(), **sgdcurves)
        print("C seed", seed, f"{time.time()-t00:.0f}s", flush=True)
    save("C", {"C_sgd": rowsC, "C_mc": mcC})

# ---------------- D. finite replay buffer ----------------
if "D" in PARTS:
    rowsD = []
    for seed in range(10):
        X1, X2, S1, S2, e0 = setup(seed)
        perm = np.random.default_rng(300 + seed).permutation(n1)
        for m in [5, 10, 20, 30, 40]:
            XM = X1[perm[:m]]; SM = XM.T @ XM / m
            K = null_proj(SM + S2); Linf = float(0.5*(K @ e0) @ S1 @ (K @ e0))
            for rho in [0.01, 0.1, 0.3]:
                H = (1-rho)*S2 + rho*SM; lam, Q = eigh(H); lam = np.clip(lam, 0, None); c = Q.T @ e0
                ts = np.concatenate([[0], np.geomspace(1e-3, 1e6, 4000)])
                E = Q @ (np.exp(-np.outer(lam, ts)) * c[:, None]); L = 0.5*np.einsum('it,ij,jt->t', E, S1, E)
                rowsD.append(dict(seed=seed, m=m, rho=rho, G=float(L.max()), L1_inf_theory=Linf, L1_end_numeric=float(L[-1]),
                                  init_coef=float(L[1]/ts[1]**2), init_coef_pred=float(0.5*(1-rho)**2*(e0 @ S2 @ S1 @ S2 @ e0))))
    save("D", {"D_buffer": rowsD})
    print("D done", f"{time.time()-t00:.0f}s", flush=True)

# ---------------- E. audit of R1-flagged prose numbers ----------------
if "E" in PARTS:
    lr_ = json.load(open(os.path.join(R, "linear_results.json")))
    e1 = lr_["E1"]["rows"]
    aud = {"E1_grid_relerr": {str(r["rho"]): abs(r["G_numeric"]-r["G_closed"])/r["G_closed"] for r in e1 if r["rho"] > 0}}
    rows = lr_["E2"]["rows"]
    aud["terminal_L1_max_rho>=0.003"] = max(r["L1_final"] for r in rows if not r["commuting"] and r["rho"] >= 0.003)
    aud["terminal_L1_mean_rho=0.003"] = float(np.mean([r["L1_final"] for r in rows if not r["commuting"] and r["rho"] == 0.003]))
    aud["commuting_max_L1"] = max(r["G"] for r in rows if r["commuting"])
    rr = [x for x in sorted(set(r["rho"] for r in rows)) if 0 < x <= 0.1]
    mt = [np.mean([r["t_recover10"] for r in rows if not r["commuting"] and r["rho"] == x]) for x in rr]
    aud["recovery10_means"] = dict(zip(map(str, rr), mt))
    aud["slope_logmean_rho<=0.1"] = float(np.polyfit(np.log(rr), np.log(mt), 1)[0])
    aud["slope_two_smallest"] = float(np.log(mt[1]/mt[0])/np.log(rr[1]/rr[0]))
    pre = lr_["precond"]["rows"]
    for meth in ["GD", "damped_newton_delta=0.1", "damped_newton_delta=0.01", "damped_newton_delta=0.001",
                 "damped_newton_delta=0.0001", "exact_newton_pinv", "diag_precond"]:
        ks = [r["steps_to_L2_1pct"] for r in pre if r["method"] == meth]
        means = [float(np.mean([r["steps_to_L2_1pct"] for r in pre if r["method"] == meth and r["rho"] == x])) for x in [0.01, 0.1, 0.3]]
        aud[f"steps_to_L2_1pct[{meth}]"] = dict(min=min(ks), max=max(ks), per_rho_means=means)
    save("E", {"E_audit": aud})

# ---------------- figure ----------------
if "FIG" in PARTS:
    for part in "ABCDE":
        out.update(json.load(open(os.path.join(R, f"linear_revision_{part}.json"))))
    json.dump(out, open(os.path.join(R, "linear_revision.json"), "w"), indent=1)
    rowsB, rowsD, mcC = out["B_recovery"], out["D_buffer"], out["C_mc"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    import collections
    g = collections.defaultdict(list)
    for r in rowsB: g[r["rho"]].append(r)
    xs = sorted(g)
    ax[0].loglog(xs, [np.mean([r["T_eps"] for r in g[x]]) for x in xs], "o-", label="exact $T_{0.1}$ (mean of 10 seeds)")
    ax[0].loglog(xs, [np.mean([r["lower"] for r in g[x]]) for x in xs], "v--", label="Thm 7 lower bound")
    ax[0].loglog(xs, [np.mean([r["upper"] for r in g[x]]) for x in xs], "^--", label="Thm 7 upper bound")
    ax[0].loglog(xs, [np.mean([r["Tstar_slow"] for r in g[x]])/x for x in xs], "k:", label="$T^*/\\rho$ (Prop. 8)")
    ax[0].set_xlabel("ρ"); ax[0].set_ylabel("recovery time"); ax[0].legend(fontsize=7); ax[0].set_title("(a) recovery time is Θ(1/ρ)")
    fc = mcC["fig_curves"]; kk = np.arange(len(fc["gd"]))*10
    ax[1].plot(kk, fc["gd"], "k", label="full-batch GD")
    for B in ["50", "10"]: ax[1].plot(kk[:len(fc[B])], fc[B], label=f"SGD B={B}")
    ax[1].plot(mcC["curve_k"], mcC["curve_mc"], "r:", lw=2, label="Monte-Carlo B=10 (2000 runs)")
    ax[1].set_xlim(0, 3000); ax[1].set_xlabel("step k"); ax[1].set_ylabel("E[L1_k]"); ax[1].set_title("(b) SGD replay, ρ=0.1, seed 0"); ax[1].legend(fontsize=7)
    gD = collections.defaultdict(list)
    for r in rowsD: gD[(r["rho"], r["m"])].append(r)
    for rho in [0.01, 0.1, 0.3]:
        ms = [5, 10, 20, 30, 40]
        ax[2].plot(ms, [np.mean([r["G"] for r in gD[(rho, m)]]) for m in ms], "o-", label=f"peak, ρ={rho}")
    ax[2].plot(ms, [np.mean([r["L1_inf_theory"] for r in gD[(0.1, m)]]) for m in ms], "k--", label="terminal (any ρ>0)")
    ax[2].set_xlabel("buffer size m (of 40)"); ax[2].set_ylabel("old-task loss"); ax[2].set_title("(c) finite buffer"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig5_linear_revision.png"), dpi=150); plt.close()
    print("all done", f"{time.time()-t00:.0f}s")
