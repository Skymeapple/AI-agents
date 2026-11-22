"""Finite-task minimax bracket for pass@k (revision r1; replaces the Hoeffding/chi^2 analysis of exp_design.py
as the paper's headline finite-task result).

Model: M i.i.d. tasks, p_i ~ F (any law on [0,1]), X_i ~ Bin(n, p_i).  Target theta = E_F[1-(1-p)^k].
R*(n,k,M) := inf over procedures with coverage >= 1-eps for EVERY F of  sup_F E_F[half-length].

Key object: the Hellinger modulus of continuity
    omega(eta) = sup { theta(G) - theta(F) :  1 - BC(pi_F, pi_G) <= eta },   BC = sum_x sqrt(pi_F(x) pi_G(x)).
We compute W(eta) = sup { theta(G)-theta(F) : sum_x d_x^2/(4 m_x) <= eta } (d = pi_G - pi_F, m = (pi_F+pi_G)/2).
Because d^2/(8m) <= (sqrt P - sqrt Q)^2/2 <= d^2/(4m) termwise,  W(eta) <= omega(eta) <= W(2 eta).

LOWER bound (any procedure).  A feasible pair (F,G) for W(eta) has 1-BC <= eta (checked exactly, repaired if
needed).  Let mid = (F+G)/2 and F_lam^{+-} = mid +- lam (G-F)/2.  By Pratt's identity
E_mid[length] = int P_mid(theta' in CI) d theta', and P_mid(theta(F_lam) in CI) >= 1 - eps - TV(mid^M, F_lam^M),
with TV <= sqrt(1 - BC^{2M}).  Hence  E_mid[length] >= (Delta/2) sum_{+-} int_0^1 (1-eps-TV_{+-}(lam))_+ d lam.
The integrand is non-increasing in lam, so a right-endpoint Riemann sum is a rigorous lower bound.
R_lo = max over eta in {a/M} of half that length.  Grid laws are genuine laws, so R_lo is rigorous.

UPPER bound (achieved).  Linear estimator theta_hat = mean_i h[X_i] with a Chernoff certificate:
  P(theta_hat - theta >= t_+) <= eps/2 with t_+ = sup_F [alpha ln E_F e^{h[X]/alpha} - theta(F)] + alpha ln(2/eps)/M,
  and symmetrically t_-.  Interval [theta_hat - t_+, theta_hat + t_-], valid for every F (and for fixed,
  non-identical tasks, since the bound holds for each task's p_i).  By convex duality,
     sup_F [alpha ln sum_j w_j A_j - w.g] = min_s max_p [alpha (expm1(s) - s + e^s a(p)) - g(p)],  a(p) = E_p expm1(h/alpha),
  so ANY s gives a valid bound; the max over p in [0,1] is taken on a 280k-point grid plus a rigorous
  between-point Lipschitz slack.  Optimising (h, alpha) exactly gives length = omega(1 - (eps/2)^{1/M}) (Prop. 4);
  we search h along candidate directions (the modulus pair's likelihood-ratio direction, the minimax-Hoeffding
  polynomial, and for k <= n the unbiased estimator), so R_up is a valid (possibly slightly loose) upper bound.
  We also report W(2 eta_up) >= omega(eta_up), an upper bound on what the best Chernoff-linear interval achieves.
Outputs: results/minimax_bounds.csv, results/minimax_horizon.csv, results/fig_minimax.png
"""
import sys, os, time
import numpy as np, pandas as pd
import cvxpy as cp
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import cheb_grid, bernstein_matrix, g_passk, best_approx_error, FINE_GRID, fine_bernstein, \
    local_grid_slack, LinearCertificate

RES = os.path.join(os.path.dirname(__file__), "..", "results")
EPS = 0.05
L_ = np.log(2 / EPS)
MS = [240, 1000, 10000, 100000, 1000000]
A_GRID = [1.0, 4.0, 16.0, 64.0]          # eta = a/M for the lower bound
LAM = np.linspace(0, 1, 401)[1:]   # path grid for the Pratt integral


def law_grid():
    return np.unique(np.concatenate([cheb_grid(400, extra_log=False), np.logspace(-9, -1, 150)]))


# ----------------------------------------------------------------------------- modulus
def W_modulus(gv, B, eta):
    """max g.(wG - wF) s.t. sum_x d_x^2/(4 m_x) <= eta.  Parametrised wF/wG = wm -/+ s v, s = sqrt(eta):
    the constraint becomes sum_x (Bv)_x^2/(Bwm)_x <= 1 and the objective 2 s g.v (well scaled)."""
    K, mm = B.shape
    s = np.sqrt(eta)
    wm = cp.Variable(mm, nonneg=True); v = cp.Variable(mm)
    Bv = B @ v; Bw = B @ wm
    cons = [cp.sum(wm) == 1, cp.sum(v) == 0, wm - s * v >= 0, wm + s * v >= 0,
            cp.sum(cp.hstack([cp.quad_over_lin(Bv[x], Bw[x]) for x in range(K)])) <= 1]
    prob = cp.Problem(cp.Maximize(gv @ v), cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        try:
            prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
        except Exception:
            return None
    if v.value is None:
        return None
    wmv, vv = np.maximum(wm.value, 0), v.value
    wF = np.maximum(wmv - s * vv, 0); wF /= wF.sum()
    wG = np.maximum(wmv + s * vv, 0); wG /= wG.sum()
    return dict(W=float(2 * s * gv @ vv), wF=wF, wG=wG, u_dir=(B @ vv) / np.maximum(B @ wmv, 1e-300))


def one_minus_bc(P, Q):
    sp, sq = np.sqrt(np.maximum(P, 0)), np.sqrt(np.maximum(Q, 0))
    den = (sp + sq) ** 2
    return float(np.sum(np.where(den > 0, (P - Q) ** 2 / (2 * np.maximum(den, 1e-300)), 0.0)))


def lower_bound(gv, B, sol, eta, M):
    wF, wG = sol["wF"], sol["wG"]
    h = one_minus_bc(B @ wF, B @ wG)
    if h > eta:  # repair: shrink towards the midpoint (1-BC is jointly convex, zero at the midpoint)
        lam = eta / h; mid = 0.5 * (wF + wG)
        wF, wG = mid + lam * (wF - mid), mid + lam * (wG - mid)
        h = one_minus_bc(B @ wF, B @ wG)
    Delta = float(gv @ (wG - wF))
    mid = 0.5 * (wF + wG); u = 0.5 * (wG - wF); Pm = B @ mid; Bu = B @ u
    tot = 0.0
    for sgn in (+1, -1):
        hs = np.array([one_minus_bc(Pm, Pm + sgn * l * Bu) for l in LAM])
        bcM2 = np.exp(2 * M * np.log1p(-np.minimum(hs, 1 - 1e-16)))
        tv = np.sqrt(np.maximum(0, 1 - bcM2))
        integrand = np.maximum(0, 1 - EPS - tv)       # non-increasing in lam
        tot += np.sum(integrand) * (LAM[1] - LAM[0])  # right-endpoint sum = rigorous lower bound
    length = 0.5 * Delta * tot
    # two-point (Le Cam) bound for reference
    tv1 = np.sqrt(max(0.0, 1 - np.exp(2 * M * np.log1p(-min(h, 1 - 1e-16)))))
    return dict(R_lo=length / 2, R_lo_twopoint=max(0.0, 1 - 2 * EPS - tv1) * Delta / 2, Delta=Delta, oneminusBC=h)


# ----------------------------------------------------------------------------- Chernoff-certified linear estimator
def tail(alpha, e, Bc, gsig, smax=2.0):
    """min_s max_p alpha*(expm1(s)-s+e^s a(p)) + gsig(p);  a = Bc @ e.  Returns (value, s)."""
    a = Bc @ e
    f = lambda s: np.max(alpha * (np.expm1(s) - s + np.exp(s) * a) + gsig)
    r = minimize_scalar(f, bounds=(-smax, smax), method="bounded", options=dict(xatol=1e-10))
    return float(r.fun), float(r.x)


def chernoff_len(kappa, u, Bc, gv, M):
    """min over alpha of t_+ + t_- for h = kappa*alpha*u (so h/alpha = kappa*u), coarse grid."""
    ep, em = np.expm1(kappa * u), np.expm1(-kappa * u)

    def F(la):
        al = np.exp(la)
        tp, _ = tail(al, ep, Bc, -gv)
        tm, _ = tail(al, em, Bc, +gv)
        return tp + tm + 2 * al * L_ / M
    r = minimize_scalar(F, bounds=(np.log(1e-5), np.log(1e7)), method="bounded", options=dict(xatol=1e-4))
    return float(r.fun), float(np.exp(r.x))


def rigorous_len(kappa, alpha, u, n, g, M):
    """Rigorous t_+ + t_- on the fine grid with between-point slack, at the given (kappa, alpha)."""
    Fb = fine_bernstein(n); gf = g(FINE_GRID)
    out = 0.0
    for sgn in (+1, -1):
        e = np.expm1(sgn * kappa * u)
        a = Fb @ e
        f = lambda s: np.max(alpha * (np.expm1(s) - s + np.exp(s) * a) - sgn * gf)
        r = minimize_scalar(f, bounds=(-2, 2), method="bounded", options=dict(xatol=1e-12))
        s = float(r.x)
        val = f(s) + local_grid_slack(g, n, alpha * np.exp(s) * e)
        out += val + alpha * L_ / M
    return out


def upper_bound(g, gv_c, Bc, n, k, M, dirs):
    best = (np.inf, None)
    for name, u in dirs.items():
        u = np.asarray(u, float)
        sc = np.max(np.abs(u)) if np.max(np.abs(u)) > 0 else 1.0
        u = u / sc
        r = minimize_scalar(lambda lk: chernoff_len(np.exp(lk), u, Bc, gv_c, M)[0],
                            bounds=(np.log(1e-6), np.log(50.0)), method="bounded", options=dict(xatol=1e-3))
        kap = float(np.exp(r.x)); ln_c, al = chernoff_len(kap, u, Bc, gv_c, M)
        if ln_c < best[0]:
            best = (ln_c, (name, kap, al, u))
    name, kap, al, u = best[1]
    L_rig = rigorous_len(kap, al, u, n, g, M)
    return dict(R_up=L_rig / 2, R_up_coarse=best[0] / 2, up_dir=name, up_alpha=al, up_kappa=kap)


def unbiased_coef(n, k):
    x = np.arange(n + 1)
    lf = gammaln(n - x + 1) - gammaln(n - x - k + 1) - gammaln(n + 1) + gammaln(n - k + 1)
    return 1 - np.where(n - x >= k, np.exp(lf), 0.0)


# ----------------------------------------------------------------------------- driver
def job(args):
    n, k = args
    t0 = time.time()
    g = g_passk(k)
    p = law_grid(); B = bernstein_matrix(n, p).T; gv = g(p)
    pc = np.unique(np.concatenate([cheb_grid(1200, extra_log=False), np.logspace(-10, -1, 300)]))
    Bc = bernstein_matrix(n, pc); gv_c = g(pc)
    E = best_approx_error(g, n, m=1500)[1]
    rows = []
    for M in MS:
        row = dict(n=n, k=k, M=M, pop_halfwidth=E)
        # lower bound
        best = None
        for a in A_GRID:
            sol = W_modulus(gv, B, a / M)
            if sol is None:
                continue
            lb = lower_bound(gv, B, sol, a / M, M)
            if best is None or lb["R_lo"] > best["R_lo"]:
                best = dict(lb, lo_a=a)
        row.update(best or {})
        # upper bound
        eta_up = -np.expm1(np.log(EPS / 2) / M)   # 1 - (eps/2)^{1/M}
        sol_up = W_modulus(gv, B, 2 * eta_up)
        dirs = {}
        if sol_up is not None:
            row["W_2eta_up_half"] = sol_up["W"] / 2   # >= omega(eta_up)/2 >= best Chernoff-linear half-length
            uu = sol_up["u_dir"]; uu = np.where(np.isfinite(uu), uu, 0.0)
            dirs["modulus"] = np.clip(uu, -1e3 * np.median(np.abs(uu)) - 1e-12, 1e3 * np.median(np.abs(uu)) + 1e-12)
        dirs["hoeffding_lp"] = LinearCertificate(g, n, M, alpha=EPS, m=1500).h
        if k <= n:
            dirs["unbiased"] = unbiased_coef(n, k)
        row.update(upper_bound(g, gv_c, Bc, n, k, M, dirs))
        rows.append(row)
    print(n, k, f"{time.time()-t0:.0f}s",
          [(r["M"], round(r.get("R_lo", np.nan), 5), round(r["R_up"], 5), r["up_dir"]) for r in rows], flush=True)
    return rows


def interp_cross(kk, vv, delta):
    lk, lv = np.log(kk), np.log(np.maximum(vv, 1e-300))
    for i in range(len(kk) - 1):
        if vv[i] <= delta < vv[i + 1]:
            return float(np.exp(lk[i] + (np.log(delta) - lv[i]) * (lk[i + 1] - lk[i]) / (lv[i + 1] - lv[i])))
    return np.nan


def main():
    jobs = []
    for n in [16, 32, 64]:
        ks = sorted(set(int(v) for v in np.round(np.logspace(np.log10(n), np.log10(4 * n * n), 22))))
        jobs += [(n, k) for k in ks]
    jobs = sorted(jobs, key=lambda j: -j[0])
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        out = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows]).sort_values(["n", "M", "k"])
    df["ratio_up_lo"] = df.R_up / df.R_lo
    df.to_csv(os.path.join(RES, "minimax_bounds.csv"), index=False, float_format="%.6g")
    tab = []
    for delta in [0.01, 0.02, 0.05]:
        for (n, M), d in df.groupby(["n", "M"]):
            d = d.sort_values("k"); kk = d.k.values.astype(float)
            ok = d[d.R_up <= delta]; bad = d[d.R_lo > delta]; okp = d[d.pop_halfwidth <= delta]
            tab.append(dict(delta=delta, n=n, M=M,
                            k_achievable_grid=ok.k.max() if len(ok) else np.nan,
                            k_impossible_grid=bad.k.min() if len(bad) else np.nan,
                            k_population_grid=okp.k.max() if len(okp) else np.nan,
                            k_achievable_interp=interp_cross(kk, d.R_up.values, delta),
                            k_impossible_interp=interp_cross(kk, d.R_lo.values, delta),
                            k_population_interp=interp_cross(kk, d.pop_halfwidth.values, delta)))
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(RES, "minimax_horizon.csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(tab.to_string())
    print(df.groupby(["n", "M"]).ratio_up_lo.describe().to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    cols = dict(zip(MS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))
    for ax, (n, d) in zip(axs, df.groupby("n")):
        d0 = d[d.M == MS[0]]
        ax.loglog(d0.k, np.maximum(d0.pop_halfwidth, 1e-7), "k-", lw=2, label=r"population $E_n$ (M=$\infty$)")
        for M, d2 in d.groupby("M"):
            ax.fill_between(d2.k, np.maximum(d2.R_lo, 1e-7), d2.R_up, color=cols[M], alpha=0.25)
            ax.loglog(d2.k, d2.R_up, "-", color=cols[M], lw=1.3, label=f"M={M:g}: achieved (solid) / lower bound (dashed)")
            ax.loglog(d2.k, np.maximum(d2.R_lo, 1e-7), "--", color=cols[M], lw=1.1)
        ax.axvline(n * n, color="gray", ls=":"); ax.axhline(0.02, color="gray", lw=0.5)
        ax.set_title(f"n = {n}"); ax.set_xlabel("k"); ax.set_ylim(1e-5, 0.6)
    axs[0].set_ylabel("minimax 95% half-width for pass@k"); axs[0].legend(fontsize=6, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_minimax.png"), dpi=150)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(job((int(sys.argv[1]), int(sys.argv[2]))))
    else:
        main()
