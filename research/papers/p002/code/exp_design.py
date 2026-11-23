"""Finite-task design study (no data needed): for rollouts-per-task n and number of tasks M,
what is the best distribution-free guarantee for pass@k, k > n?

Upper bound  R_up(n,k,M)  = min_h  bias(h) + range(h)*sqrt(log(2/alpha)/(2M))   (achieved by
             the certified linear estimator; Hoeffding, alpha = 0.05)
Lower bound  R_lo(n,k,M)  = omega(1/(2M)) / 4, Le Cam two-point bound with the TV modulus
             omega(eps) = sup{ |E_F g - E_G g| : TV(count law F, count law G) <= eps }
             (TV of M-fold products <= M*eps = 1/2).
Population   2*E_n(g) (worst-case identified-set width, M = infinity).
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ttc import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def tv_modulus(g, n, eps, m=1200):
    p = cheb_grid(m)
    B = bernstein_matrix(n, p).T
    gv = g(p); mm = len(p); K = n + 1
    # vars mu (mm), nu (mm), u (K) with |B(mu-nu)| <= u, sum u <= 2 eps
    c = np.concatenate([-gv, gv, np.zeros(K)])
    A_ub = np.vstack([np.hstack([B, -B, -np.eye(K)]), np.hstack([-B, B, -np.eye(K)]),
                      np.concatenate([np.zeros(2 * mm), np.ones(K)])[None, :]])
    b_ub = np.concatenate([np.zeros(2 * K), [2 * eps]])
    A_eq = np.vstack([np.concatenate([np.ones(mm), np.zeros(mm + K)]),
                      np.concatenate([np.zeros(mm), np.ones(mm), np.zeros(K)])])
    for meth in ("highs", "highs-ds", "highs-ipm"):
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1, 1], bounds=[(0, None)] * (2 * mm + K), method=meth)
        if r.status == 0:
            return -r.fun
    return np.nan


REFS = {
    "uniform": lambda: np.random.default_rng(3).uniform(size=400000),
    "beta(0.3,3)": lambda: np.random.default_rng(3).beta(0.3, 3.0, size=400000),
    "0.3d0+0.7beta(.5,2)": lambda: np.where(np.random.default_rng(4).uniform(size=400000) < 0.3, 0.0,
                                             np.random.default_rng(3).beta(0.5, 2.0, size=400000)),
}


def chi2_lower(g, n, M, m=1200):
    """Le Cam / chi-square two-point bound (superseded in revision r1 by exp_minimax.py, which uses the
    Hellinger modulus with both laws free).  For a reference law F0 with count law Q, maximise
    |E_F g - E_F0 g| over F whose count law P satisfies the chi-square SUM constraint
    sum_x (P_x - Q_x)^2 / Q_x <= 1/(8M)  (i.e. chi2(P||Q) <= 1/(8M)), hence H^2(P^M,Q^M) <= M*chi2 <= 1/8
    and TV(P^M,Q^M) <= 0.354.  Any procedure giving 95% coverage under both laws then has
    E_P[width] >= (0.95-0.354-0.05)*Delta = 0.546*Delta.
    Returns max over references of 0.546*Delta/2 (a lower bound on the expected half-width)."""
    p = cheb_grid(m)
    B = bernstein_matrix(n, p).T
    gv = g(p); mm = len(p)
    best = 0.0
    for name, smp in REFS.items():
        s = smp()[:100000]
        Q = binom.pmf(np.arange(n + 1)[None, :], n, s[:, None]).mean(0)
        th0 = float(np.mean(g(s)))
        keep = Q > 1e-14
        import cvxpy as cp
        w = cp.Variable(mm, nonneg=True)
        P = B @ w
        cons = [cp.sum(w) == 1,
                cp.sum(cp.multiply(8.0 * M / Q[keep], cp.square(P[keep] - Q[keep]))) <= 1.0]
        if (~keep).any():
            cons.append(P[~keep] <= 1e-14)
        for sgn in (+1, -1):
            prob = cp.Problem(cp.Maximize(sgn * (gv @ w)), cons)
            try:
                prob.solve(solver="CLARABEL")
            except Exception:
                continue
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                # a-posteriori repair: the mixture (1-lam) F0 + lam F has count law Q + lam (P-Q),
                # chi2 scaled by lam^2 and Delta scaled by lam, so shrink until the constraint holds.
                wv = np.maximum(w.value, 0); wv /= wv.sum()
                Pv = B @ wv
                if (~keep).any() and np.any(Pv[~keep] > 1e-12):
                    continue
                r = 8.0 * M * np.sum((Pv[keep] - Q[keep]) ** 2 / Q[keep])
                lam = min(1.0, 1.0 / np.sqrt(r)) if r > 0 else 1.0
                delta = lam * (sgn * (gv @ wv) - sgn * th0)
                best = max(best, delta)
    return 0.546 * best / 2


def main():
    rows = []
    prev = None
    fn = os.path.join(RES, "design_bounds.csv")
    if os.path.exists(fn):
        prev = pd.read_csv(fn).set_index(["n", "k", "M"])
    for n in [8, 16, 32, 64]:
        ks = sorted(set(int(v) for v in np.round(np.logspace(np.log10(n), np.log10(64 * n * n), 8))))
        for k in ks:
            g = g_passk(k)
            _, E = best_approx_error(g, n, m=1500)
            for M in [100, 1000, 10000, 1000000]:
                if prev is not None and (n, k, M) in prev.index:
                    r0 = prev.loc[(n, k, M)]
                    class C: pass
                    cert = C(); cert.bias, cert.range = r0.bias, r0.range
                    up = r0.R_up
                else:
                    cert = LinearCertificate(g, n, M, alpha=0.05, m=1500)
                    up = cert.bias + cert.grid_slack + cert.range * cert.kappa
                lo = max(tv_modulus(g, n, 1.0 / (2 * M)) / 4, chi2_lower(g, n, M))
                rows.append(dict(n=n, k=k, M=M, pop_width=2 * E, R_up=up, R_lo=lo,
                                 bias=cert.bias, range=cert.range))
                print(n, k, M, f"pop={2*E:.4f} up={up:.4f} lo={lo:.4f} bias={cert.bias:.4f} range={cert.range:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "design_bounds.csv"), index=False, float_format="%.5g")
    # horizon table: largest k with guaranteed half-width <= delta
    tab = []
    for delta in [0.02, 0.05]:
        for (n, M), d in df.groupby(["n", "M"]):
            ok = d[d.R_up <= delta]
            kmax = ok.k.max() if len(ok) else np.nan
            okp = d[d.pop_width / 2 <= delta]
            tab.append(dict(delta=delta, n=n, M=M, kmax_certified=kmax, kmax_population=okp.k.max() if len(okp) else np.nan))
    pd.DataFrame(tab).to_csv(os.path.join(RES, "design_horizon.csv"), index=False)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, (n, d) in zip(axs, df.groupby("n")):
        dd = d[d.M == 100]
        ax.loglog(dd.k, dd.pop_width / 2, "k-", lw=2, label=r"population $E_n$ (M=$\infty$)")
        for M, d2 in d.groupby("M"):
            l, = ax.loglog(d2.k, d2.R_up, "o-", ms=3, label=f"upper, M={M:g}")
            ax.loglog(d2.k, np.maximum(d2.R_lo, 1e-6), "--", color=l.get_color(), lw=1)
        ax.axvline(n * n, color="gray", ls=":")
        ax.set_title(f"n = {n}"); ax.set_xlabel("k"); ax.set_ylim(1e-4, 1.2)
    axs[0].set_ylabel("guaranteed half-width (pass@k)"); axs[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_design.png"), dpi=150)


if __name__ == "__main__":
    main()
