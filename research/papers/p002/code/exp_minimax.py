"""Finite-task minimax bracket for pass@k (revision r1).

For n rollouts per task and M tasks we bracket the minimax expected half-width of a 95% confidence
interval for theta = E_F[1-(1-p)^k] over ALL mixing laws F on [0,1]:

  R_lo(n,k,M)  (lower bound, any procedure)   Le Cam two-point bound with the HELLINGER modulus
        omega(eps2) = max { E_F g - E_G g : H^2(pi_F, pi_G) <= eps2 },  H^2 = 1 - sum_x sqrt(pi_F pi_G),
        computed as a convex (second-order-cone) program over pairs (F, G) of laws on a grid.  The two laws
        are free (not tied to reference laws).  For M i.i.d. tasks BC(P^M,Q^M) = BC(P,Q)^M and
        TV <= sqrt(1 - BC^(2M)).  A procedure with 95% coverage under both laws contains both values with
        probability >= 0.9 - TV under F, so its expected half-width is >= (0.9 - TV) * Delta / 2.
        We maximise over eps2 = a/M, a in A_GRID.  Every reported value is recomputed exactly from the
        normalised solution (grid laws are genuine laws, so the bound is rigorous; it can only be loose).

  R_up(n,k,M)  (upper bound, achieved)  linear estimator mean_i h[X_i] with a Bernstein-inequality
        guarantee.  With V(h) = min_c sup_p sum_x (h_x - c)^2 b_{n,x}(p) (a bound on Var h[X] valid for every
        F and for fixed, non-identical tasks), range R(h), L = ln(2/alpha):
            |theta_hat - theta| <= bias(h) + (R L/3 + sqrt((R L/3)^2 + 2 M V L)) / M   w.p. >= 1 - alpha.
        h is chosen by an SOCP minimising bias + sqrt(2L/M) sqrt(V) + 2 R L/(3M); bias and V are re-verified
        on a 260k-point grid with explicit between-point slack.

The Hellinger modulus is the quantity that characterises minimax rates for linear functionals of a convex
family (Donoho & Liu 1991); the gap R_up/R_lo reported here is the constant-factor slack of our explicit
bounds.  Outputs: results/minimax_bounds.csv, results/minimax_horizon.csv, results/fig_minimax.png
"""
import sys, os, json, time
import numpy as np, pandas as pd
import cvxpy as cp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import cheb_grid, bernstein_matrix, g_passk, best_approx_error, FINE_GRID, fine_bernstein

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
MS = [240, 1000, 10000, 1000000]
A_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def hellinger_modulus(g, n, eps2, p, B):
    gv = g(p); mm = len(p)
    mu = cp.Variable(mm, nonneg=True); nu = cp.Variable(mm, nonneg=True)
    P = B @ mu; Q = B @ nu
    t = cp.Variable(n + 1, nonneg=True)
    cons = [cp.sum(mu) == 1, cp.sum(nu) == 1,
            cp.SOC(P + Q, cp.vstack([2 * t, P - Q]), axis=0),
            cp.sum(t) >= 1 - eps2]
    prob = cp.Problem(cp.Maximize(gv @ (mu - nu)), cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        return np.nan, np.nan
    if mu.value is None:
        return np.nan, np.nan
    m1 = np.maximum(mu.value, 0); m1 /= m1.sum()
    n1 = np.maximum(nu.value, 0); n1 /= n1.sum()
    P1, Q1 = B @ m1, B @ n1
    H2 = 1 - np.sum(np.sqrt(np.maximum(P1, 0) * np.maximum(Q1, 0)))
    D = float(gv @ (m1 - n1))
    if H2 > eps2:  # repair: shrink both laws towards their midpoint; H^2 is jointly convex => H2(lam) <= lam*H2
        lam = eps2 / H2
        mid = 0.5 * (m1 + n1)
        m1, n1 = mid + lam * (m1 - mid), mid + lam * (n1 - mid)
        P1, Q1 = B @ m1, B @ n1
        H2 = 1 - np.sum(np.sqrt(P1 * Q1)); D = float(gv @ (m1 - n1))
    return D, H2


def bernstein_estimator(g, n, M, p, B, pv, Bv):
    L = np.log(2 / ALPHA); K = n + 1
    gv = g(p)
    h = cp.Variable(K); c = cp.Variable(); b = cp.Variable(nonneg=True); s = cp.Variable(nonneg=True)
    lo = cp.Variable(); hi = cp.Variable()
    W = np.sqrt(Bv)  # (mv, K)
    cons = [cp.abs(B @ h - gv) <= b, h >= lo, h <= hi,
            cp.SOC(s * np.ones(len(pv)), cp.multiply(W, cp.reshape(h - c, (1, K), order="C")), axis=1)]
    obj = b + np.sqrt(2 * L / M) * s + 2 * L / (3 * M) * (hi - lo)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver="CLARABEL")
    hv = h.value; cv = float(c.value)
    # a-posteriori verification on the fine grid with between-grid slack
    Fb = fine_bernstein(n)
    err = np.abs(g(FINE_GRID) - Fb @ hv)
    gaps = np.diff(FINE_GRID)
    d = 1e-10
    gp = np.abs((g(np.minimum(FINE_GRID + d, 1)) - g(FINE_GRID)) / d)
    bias = float(err.max() + np.max(0.5 * gaps * (gp[:-1] + n * np.max(np.abs(np.diff(hv))))))
    sq = (hv - cv) ** 2
    V = float((Fb @ sq).max() + np.max(0.5 * gaps) * n * np.max(np.abs(np.diff(sq))))
    R = float(hv.max() - hv.min())
    dev = (R * L / 3 + np.sqrt((R * L / 3) ** 2 + 2 * M * V * L)) / M
    return dict(R_up=bias + dev, up_bias=bias, up_sd=np.sqrt(V), up_range=R)


def job(args):
    n, k = args
    t0 = time.time()
    g = g_passk(k)
    p = cheb_grid(500)  # ~900 points incl. log-spaced points near 0
    B = bernstein_matrix(n, p).T
    pv = cheb_grid(150, extra_log=False); pv = np.unique(np.concatenate([pv, np.logspace(-6, -1, 60)]))
    Bv = bernstein_matrix(n, pv)
    Bu = bernstein_matrix(n, p)
    _, E = best_approx_error(g, n, m=1500)
    # modulus at every eps2 needed
    eps_list = sorted(set(a / M for a in A_GRID for M in MS))
    om = {e: hellinger_modulus(g, n, e, p, B) for e in eps_list}
    rows = []
    for M in MS:
        best = (0.0, None)
        for a in A_GRID:
            D, H2 = om[a / M]
            if not np.isfinite(D):
                continue
            tv = np.sqrt(max(0.0, 1 - (1 - H2) ** (2 * M)))
            lb = max(0.0, 0.9 - tv) * D / 2
            if lb > best[0]:
                best = (lb, dict(a=a, Delta=D, H2=H2, TV=tv))
        up = bernstein_estimator(g, n, M, p, Bu, pv, Bv)
        rows.append(dict(n=n, k=k, M=M, pop_halfwidth=E, R_lo=best[0], **{f"lo_{kk}": v for kk, v in (best[1] or {}).items()}, **up))
    print(n, k, f"{time.time()-t0:.0f}s", [(r["M"], round(r["R_lo"], 5), round(r["R_up"], 5)) for r in rows], flush=True)
    return rows


def main():
    jobs = []
    for n in [16, 32, 64]:
        ks = sorted(set(int(v) for v in np.round(np.logspace(np.log10(n), np.log10(16 * n * n), 15))))
        jobs += [(n, k) for k in ks]
    with Pool(4) as pool:
        out = pool.map(job, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows]).sort_values(["n", "M", "k"])
    df["ratio_up_lo"] = df.R_up / df.R_lo
    df.to_csv(os.path.join(RES, "minimax_bounds.csv"), index=False, float_format="%.6g")
    # horizons: achievable (largest k with R_up <= delta), impossible (smallest k with R_lo > delta),
    # population (largest k with pop half-width <= delta); grid values and log-log interpolation
    def interp_cross(kk, vv, delta):
        lk, lv = np.log(kk), np.log(np.maximum(vv, 1e-300))
        for i in range(len(kk) - 1):
            if vv[i] <= delta < vv[i + 1]:
                return float(np.exp(lk[i] + (np.log(delta) - lv[i]) * (lk[i + 1] - lk[i]) / (lv[i + 1] - lv[i])))
        return np.nan
    tab = []
    for delta in [0.02, 0.05]:
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
    print(tab.to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (n, d) in zip(axs, df.groupby("n")):
        d0 = d[d.M == MS[0]]
        ax.loglog(d0.k, d0.pop_halfwidth, "k-", lw=2, label=r"population (M=$\infty$)")
        for M, d2 in d.groupby("M"):
            l, = ax.loglog(d2.k, d2.R_up, "-", lw=1.5, label=f"achievable, M={M:g}")
            ax.loglog(d2.k, np.maximum(d2.R_lo, 1e-7), "--", color=l.get_color(), lw=1.2)
        ax.axvline(n * n, color="gray", ls=":"); ax.axhline(0.02, color="gray", lw=0.5)
        ax.set_title(f"n = {n}"); ax.set_xlabel("k"); ax.set_ylim(1e-5, 1.2)
    axs[0].set_ylabel("minimax half-width bracket, pass@k"); axs[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_minimax.png"), dpi=150)


if __name__ == "__main__":
    main()
