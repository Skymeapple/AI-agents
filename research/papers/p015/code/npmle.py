"""Nonparametric (NPMLE) frailty model with a Weibull task-level cumulative hazard.

  P(success of a run on task i | u_i) = exp(-exp(k * log(T_i/60) + u_i)),   u_i ~ G (unrestricted),

i.e. multiplicative frailty Z_i = exp(u_i) on Lambda_0(T) = T^k with Z independent of T.
G is estimated as a discrete distribution on a fine fixed grid (Kiefer-Wolfowitz NPMLE, EM), and k is
profiled out.  By the replicate-length duality (paper, Theorem 2) k is identified without any
parametric assumption on G, as long as tasks carry >= 2 replicate runs.

Also: likelihood-ratio bounds on the pooled success S(T*) = int exp(-exp(k log(T*/60) + u)) dG(u)
over all (k, G) whose log-likelihood is within q/2 of the maximum (convex program in G for fixed k).
"""
import numpy as np
from scipy import optimize, special
from common import T_REF

U_GRID = np.linspace(-16.0, 10.0, 105)  # log-frailty support (step 0.25)


def loglik_matrix(n, s, T, k, grid=U_GRID):
    """log P(s_i | n_i, u_m) without the binomial constant, shape [I, M]."""
    eta = k * np.log(T / T_REF)[:, None] + grid[None, :]
    e = np.exp(np.clip(eta, -40, 8))
    lp = -e
    lq = np.log(-np.expm1(-e) + 1e-300)
    return s[:, None] * lp + (n - s)[:, None] * lq


def em_weights(LL, iters=3000, tol=1e-10, pi0=None):
    I, M = LL.shape
    mx = LL.max(1, keepdims=True)
    L = np.exp(LL - mx)
    pi = np.full(M, 1.0 / M) if pi0 is None else pi0.copy()
    for _ in range(iters):
        f = L @ pi
        new = pi * (L.T @ (1.0 / f)) / I
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    ll = float(np.sum(np.log(L @ pi) + mx[:, 0]))
    return pi, ll


def profile_k(n, s, T, kgrid=None):
    kgrid = np.round(np.arange(0.3, 3.01, 0.05), 3) if kgrid is None else kgrid
    out = []
    pi = None
    for k in kgrid:
        pi, ll = em_weights(loglik_matrix(n, s, T, k), pi0=None if pi is None else 0.5 * pi + 0.5 / len(pi))
        out.append((k, ll, pi))
    return out


def fit(n, s, T, kgrid=None):
    """Profile likelihood over k; returns dict with k_hat, 95% profile CI, G weights and the profile."""
    prof = profile_k(n, s, T, kgrid)
    ks = np.array([p[0] for p in prof])
    lls = np.array([p[1] for p in prof])
    j = int(np.argmax(lls))
    # refine by bounded scalar search around the grid maximum
    lo_b, hi_b = ks[max(j - 1, 0)], ks[min(j + 1, len(ks) - 1)]
    pi_j = prof[j][2]
    if hi_b > lo_b:
        r = optimize.minimize_scalar(lambda k: -em_weights(loglik_matrix(n, s, T, k), pi0=pi_j)[1],
                                     bounds=(lo_b, hi_b), method="bounded", options=dict(xatol=1e-3))
        k_hat, ll_hat = float(r.x), float(-r.fun)
    else:
        k_hat, ll_hat = ks[j], lls[j]
    if ll_hat < lls[j]:
        k_hat, ll_hat = ks[j], lls[j]
    pi_hat, _ = em_weights(loglik_matrix(n, s, T, k_hat), pi0=pi_j)
    dev = 2 * (ll_hat - lls)
    inside = ks[dev <= 3.84]
    return dict(k=k_hat, ll=ll_hat, pi=pi_hat, k_lo=inside.min() if len(inside) else np.nan,
                k_hi=inside.max() if len(inside) else np.nan, k_lo_at_edge=bool(len(inside) and inside.min() == ks[0]),
                k_hi_at_edge=bool(len(inside) and inside.max() == ks[-1]), prof=(ks, lls, [p[2] for p in prof]))


def S_pooled(pi, k, Tstar, grid=U_GRID):
    Tstar = np.atleast_1d(np.asarray(Tstar, float))
    eta = k * np.log(Tstar / T_REF)[:, None] + grid[None, :]
    return np.exp(-np.exp(np.clip(eta, -40, 8))) @ pi


def p_draws(pi, k, Tstar, grid=U_GRID):
    """Distribution of per-task success at length T*: (support values, weights)."""
    return np.exp(-np.exp(np.clip(k * np.log(Tstar / T_REF) + grid, -40, 8))), pi


def lr_bounds(n, s, T, Tstars, ll_max, ks, q=3.84, grid=U_GRID):
    """For each k in ks: min and max of S(T*) over G with loglik(k, G) >= ll_max - q/2.
    Returns array [len(ks), len(Tstars), 2] (nan where the k is outside the LR set)."""
    import cvxpy as cp
    out = np.full((len(ks), len(Tstars), 2), np.nan)
    for a, k in enumerate(ks):
        LL = loglik_matrix(n, s, T, k, grid)
        mx = LL.max(1, keepdims=True)
        L = np.exp(LL - mx)
        keep = L.max(0) > 1e-12
        Lk = L[:, keep]
        const = float(mx.sum())
        pi = cp.Variable(Lk.shape[1], nonneg=True)
        thr = ll_max - q / 2 - const
        cons = [cp.sum(pi) == 1, cp.sum(cp.log(Lk @ pi)) >= thr]
        for b, Ts in enumerate(Tstars):
            av = np.exp(-np.exp(np.clip(k * np.log(Ts / T_REF) + grid, -40, 8)))[keep]
            for c, sense in enumerate([cp.Minimize, cp.Maximize]):
                prob = cp.Problem(sense(av @ pi), cons)
                try:
                    prob.solve(solver=cp.CLARABEL)
                    if prob.status in ("optimal", "optimal_inaccurate"):
                        out[a, b, c] = float(prob.value)
                except Exception:
                    pass
    return out
