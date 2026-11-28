"""Core library: polynomial-approximation view of test-time-compute (TTC) extrapolation.

Setting. Task i has latent per-rollout success probability p_i ~ F on [0,1].
An evaluation draws n rollouts per task and records X_i ~ Bin(n, p_i).
A TTC functional is theta(F) = E_F[g(p)] for a known g, e.g.
  pass@k:            g(p) = 1 - (1-p)^k
  verifier ceiling:  g(p) = t p / (t p + q (1-p))      (resample until accept)
  BoN-with-verifier: g(p) = (1-(1-r)^k) t p / r, r = t p + q (1-p)
  majority@k (binary): g(p) = P(Bin(k,p) > k/2)

Everything is computed on a fine grid of [0,1] that is clustered at both
endpoints (Chebyshev-like), and verified a posteriori on a much finer grid.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom
from scipy.special import gammaln


# ----------------------------------------------------------------------------- grids
def cheb_grid(m, extra_log=True):
    """Grid on [0,1] clustered at endpoints; optional log-spaced points near 0."""
    j = np.arange(m)
    x = 0.5 * (1 - np.cos(np.pi * j / (m - 1)))
    if extra_log:
        x = np.concatenate([x, np.logspace(-9, -1, 400)])
    return np.unique(np.clip(x, 0, 1))


def bernstein_matrix(n, p):
    """B[j, x] = C(n,x) p_j^x (1-p_j)^(n-x)."""
    x = np.arange(n + 1)
    return binom.pmf(x[None, :], n, np.asarray(p)[:, None])


# ----------------------------------------------------------------------------- targets
def g_passk(k):
    return lambda p: 1.0 - np.power(1.0 - p, k)


def g_verifier_ceiling(t, q):
    def g(p):
        den = t * p + q * (1 - p)
        out = np.where(den > 0, t * p / np.maximum(den, 1e-300), 0.0)
        return out
    return g


def g_bon_verifier(k, t, q):
    """Draw k samples; verifier accepts correct w.p. t, wrong w.p. q; return a
    uniformly random accepted sample; fail if none accepted."""
    def g(p):
        r = t * p + q * (1 - p)
        acc = 1.0 - np.power(1.0 - r, k)
        return np.where(r > 0, acc * t * p / np.maximum(r, 1e-300), 0.0)
    return g


def g_majority(k):
    """Binary majority vote with k (odd) samples: P(Bin(k,p) > k/2)."""
    return lambda p: binom.sf(k // 2, k, p)


# ----------------------------------------------------------------------------- best approximation
def best_approx_error(g, n, m=3000, verify_m=200001):
    """E_n(g) = min_{deg P <= n} sup_[0,1] |g - P|, by LP in a Chebyshev basis on a grid.
    Returns (E_grid, E_verified_on_fine_grid)."""
    p = cheb_grid(m)
    x = 2 * p - 1
    T = np.polynomial.chebyshev.chebvander(x, n)  # (m, n+1)
    gv = g(p)
    # variables: c (n+1, free), e >= 0 ; minimize e ; |gv - T c| <= e
    nv = n + 2
    cobj = np.zeros(nv); cobj[-1] = 1
    A1 = np.hstack([T, -np.ones((len(p), 1))])      # T c - e <= g
    A2 = np.hstack([-T, -np.ones((len(p), 1))])     # -T c - e <= -g
    A = np.vstack([A1, A2]); b = np.concatenate([gv, -gv])
    bounds = [(None, None)] * (n + 1) + [(0, None)]
    res = linprog(cobj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    c = res.x[:-1]
    pf = np.linspace(0, 1, verify_m); pf = np.unique(np.concatenate([pf, cheb_grid(20001)]))
    Ef = np.max(np.abs(g(pf) - np.polynomial.chebyshev.chebval(2 * pf - 1, c)))
    return res.x[-1], Ef


# ----------------------------------------------------------------------------- sharp identified set
def identified_interval(g, n, count_law, m=4000):
    """Sharp population identified interval for E_F[g(p)] given the exact
    count law pi_x = P(X = x), X ~ Bin(n, p), p ~ F.  LP over F on a grid."""
    p = cheb_grid(m)
    B = bernstein_matrix(n, p).T  # (n+1, m)
    gv = g(p)
    out = []
    for sgn in (+1, -1):
        res = linprog(sgn * gv, A_eq=B, b_eq=count_law, bounds=[(0, None)] * len(p),
                      method="highs")
        if res.status != 0:
            # project count law to closest feasible (L1) if numerically infeasible
            return (np.nan, np.nan)
        out.append(sgn * res.fun)
    return out[0], out[1]


def identified_interval_robust(g, n, count_law, m=4000, slack=0.0):
    """Like identified_interval but allows L1 slack (sum |B w - pi| <= slack).
    With slack chosen from a simultaneous confidence region this gives a valid
    finite-task interval (used only for comparison)."""
    p = cheb_grid(m)
    B = bernstein_matrix(n, p).T
    gv = g(p); mm = len(p); K = n + 1
    # vars: w (mm), u (K) >= |B w - pi|
    A_ub = np.vstack([
        np.hstack([B, -np.eye(K)]),
        np.hstack([-B, -np.eye(K)]),
        np.hstack([np.zeros((1, mm)), np.ones((1, K))]),
    ])
    b_ub = np.concatenate([count_law, -count_law, [slack]])
    A_eq = np.hstack([np.ones((1, mm)), np.zeros((1, K))])
    out = []
    for sgn in (+1, -1):
        c = np.concatenate([sgn * gv, np.zeros(K)])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * (mm + K), method="highs")
        out.append(sgn * res.fun if res.status == 0 else np.nan)
    return out[0], out[1]




# ----------------------------------------------------------------------------- a-posteriori verification
FINE_GRID = np.unique(np.concatenate([np.linspace(0, 1, 200001), cheb_grid(20001, extra_log=False),
                                      np.logspace(-12, 0, 40001)]))


_FB = {}


def fine_bernstein(n):
    if n not in _FB:
        _FB[n] = bernstein_matrix(n, FINE_GRID)
    return _FB[n]


def local_grid_slack(g, n, h, grid=None):
    """Bound on how much |g - P_h| can exceed its grid maximum between consecutive grid points:
    max_j  (p_{j+1}-p_j)/2 * ( |g'(p_j)| + n*max|diff(h)| ).  Uses |P_h'| <= n max|h_{x+1}-h_x|
    (Bernstein derivative formula) and assumes |g'| is non-increasing in p, which holds for
    pass@k, the verifier ceiling and best-of-k with a verifier (all concave, increasing)."""
    grid = FINE_GRID if grid is None else grid
    d = 1e-10
    gp = np.abs((g(np.minimum(grid + d, 1)) - g(grid)) / d)
    gp = np.maximum(gp, np.abs((g(grid) - g(np.maximum(grid - d, 0))) / d))
    gaps = np.diff(grid)
    Ph = n * np.max(np.abs(np.diff(h))) if len(h) > 1 else 0.0
    return float(np.max(0.5 * gaps * (gp[:-1] + Ph)))


# ----------------------------------------------------------------------------- certified linear estimator
class LinearCertificate:
    """Linear estimator theta_hat = mean_i h[X_i] with a distribution-free
    finite-task guarantee:
        |theta_hat - mean_i g(p_i)| <= bias(h) + range(h) * sqrt(log(2/alpha)/(2M))
    with probability >= 1-alpha (Hoeffding, independent tasks), where
        bias(h) = sup_{p in [0,1]} |g(p) - sum_x h_x C(n,x) p^x (1-p)^(n-x)|.
    h is chosen by an LP minimising bias + kappa * range, kappa = sqrt(log(2/alpha)/(2M)).
    The bias is re-verified on a fine grid and a Markov-inequality grid slack is added.
    """

    def __init__(self, g, n, M, alpha=0.05, m=3000, lipschitz_g=None):
        self.n, self.M, self.alpha = n, M, alpha
        self.kappa = np.sqrt(np.log(2 / alpha) / (2 * M))
        p = cheb_grid(m)
        B = bernstein_matrix(n, p)
        gv = g(p)
        K = n + 1; mm = len(p)
        # vars: h (K), b, lo, hi
        nv = K + 3
        c = np.zeros(nv); c[K] = 1.0; c[K + 1] = -self.kappa; c[K + 2] = self.kappa
        A = [np.hstack([B, -np.ones((mm, 1)), np.zeros((mm, 2))]),
             np.hstack([-B, -np.ones((mm, 1)), np.zeros((mm, 2))]),
             np.hstack([-np.eye(K), np.zeros((K, 1)), np.ones((K, 1)), np.zeros((K, 1))]),
             np.hstack([np.eye(K), np.zeros((K, 2)), -np.ones((K, 1))])]
        bvec = np.concatenate([gv, -gv, np.zeros(K), np.zeros(K)])
        bounds = [(None, None)] * K + [(0, None), (None, None), (None, None)]
        res = linprog(c, A_ub=np.vstack(A), b_ub=bvec, bounds=bounds, method="highs")
        assert res.status == 0, res.message
        self.h = res.x[:K]
        # verify bias on a fine grid
        pf = FINE_GRID
        Pf = fine_bernstein(n) @ self.h
        err = np.abs(g(pf) - Pf)
        self.bias = float(err.max())
        self.grid_slack = local_grid_slack(g, n, self.h)
        self.range = float(self.h.max() - self.h.min())

    def estimate(self, counts):
        counts = np.asarray(counts)
        th = float(np.mean(self.h[counts]))
        half = self.bias + self.grid_slack + self.range * self.kappa
        return th, th - half, th + half, half


# ----------------------------------------------------------------------------- baselines
def unbiased_passk(counts, n, k):
    """Chen et al. (2021) unbiased estimator, valid for k <= n."""
    c = np.asarray(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = gammaln(n - c + 1) - gammaln(n - c - k + 1) - gammaln(n + 1) + gammaln(n - k + 1)
        fail = np.where(n - c >= k, np.exp(lf), 0.0)
    return float(np.mean(1 - fail))


def betabinom_fit(counts, n, init=None):
    """MLE of Beta(a,b) mixing law from Binomial(n) counts (Kazdan et al.-style)."""
    from scipy.optimize import minimize
    from scipy.special import betaln
    c = np.asarray(counts)

    def nll(th):
        a, b = np.exp(th)
        return -np.sum(betaln(c + a, n - c + b) - betaln(a, b))
    best = None
    starts = [tuple(init)] if init is not None else [(a0, b0) for a0 in (0.1, 0.5, 2.0) for b0 in (0.5, 2.0, 10.0)]
    for a0, b0 in starts:
        if True:
            r = minimize(nll, np.log([a0, b0]), method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
            if best is None or r.fun < best.fun:
                best = r
    return np.exp(best.x)


def betabinom_passk(a, b, k):
    from scipy.special import betaln
    return 1.0 - np.exp(betaln(a, b + k) - betaln(a, b))


def powerlaw_passk(counts, n, ks, kmin=1):
    """Fit log(1 - pass@k) = c - beta log k on k in [kmin, n] (unbiased estimates), extrapolate."""
    kk = np.unique(np.round(np.logspace(np.log10(kmin), np.log10(n), 12)).astype(int))
    f = np.array([1 - unbiased_passk(counts, n, k) for k in kk])
    ok = f > 0
    if ok.sum() < 2:
        return np.ones_like(np.asarray(ks, float))
    A = np.vstack([np.ones(ok.sum()), np.log(kk[ok])]).T
    coef, *_ = np.linalg.lstsq(A, np.log(f[ok]), rcond=None)
    return 1 - np.exp(coef[0] + coef[1] * np.log(np.asarray(ks, float)))


# ----------------------------------------------------------------------------- adaptive one-sided bounds
def _fine_grid():
    return FINE_GRID


_FINE = None


def one_sided_poly(g, n, weights, kappa, upper=True, m=800):
    """LP: choose Bernstein coefficients h minimising weights.h + kappa*range(h)
    subject to P_h >= g (upper) or P_h <= g (lower) on a grid.  Returns (h, violation)
    where violation = sup over a fine grid of the one-sided constraint violation plus
    a Markov-inequality slack for points between fine-grid nodes."""
    global _FINE
    if _FINE is None:
        _FINE = _fine_grid()
    p = cheb_grid(m)
    B = bernstein_matrix(n, p)
    gv = g(p); K = n + 1; mm = len(p)
    s = 1.0 if upper else -1.0
    c = np.concatenate([s * weights, [-kappa, kappa]])
    # s*(B h) >= s*g  ->  -s*B h <= -s*g
    A = [np.hstack([-s * B, np.zeros((mm, 2))]),
         np.hstack([-np.eye(K), np.ones((K, 1)), np.zeros((K, 1))]),
         np.hstack([np.eye(K), np.zeros((K, 1)), -np.ones((K, 1))])]
    b = np.concatenate([-s * gv, np.zeros(2 * K)])
    res = linprog(c, A_ub=np.vstack(A), b_ub=b, bounds=[(None, None)] * (K + 2), method="highs")
    if res.status != 0:
        return None, None
    h = res.x[:K]
    Pf = fine_bernstein(n) @ h
    viol = np.max(np.maximum(0.0, s * (g(_FINE) - Pf)))
    viol += local_grid_slack(g, n, h)
    return h, viol


def split_bounds(g, n, counts, alpha=0.05, rng=None, m=800, lipschitz_g=0.0):
    """Finite-task (1-alpha) interval for mean_i g(p_i) (and for E_F g) by sample splitting:
    half A selects a polynomial majorant / minorant adapted to the observed count law,
    half B evaluates it with Hoeffding.  No parametric assumption on F."""
    rng = np.random.default_rng(0) if rng is None else rng
    counts = np.asarray(counts)
    perm = rng.permutation(len(counts))
    A, Bc = counts[perm[: len(counts) // 2]], counts[perm[len(counts) // 2:]]
    MB = len(Bc)
    kappa = np.sqrt(np.log(2 / alpha) / (2 * MB))  # alpha/2 per side
    wA = np.bincount(A, minlength=n + 1) / len(A)
    hu, vu = one_sided_poly(g, n, wA, kappa, upper=True, m=m)
    hl, vl = one_sided_poly(g, n, wA, kappa, upper=False, m=m)
    gap_slack = 0.0
    up = np.mean(hu[Bc]) + (hu.max() - hu.min()) * kappa + vu
    lo = np.mean(hl[Bc]) - (hl.max() - hl.min()) * kappa - vl
    return max(lo, 0.0), min(up, 1.0)


def weissman_eps(K, M, alpha):
    """L1 radius of a (1-alpha) multinomial confidence region (Weissman et al. 2003)."""
    return np.sqrt(2 * (K * np.log(2) + np.log(1 / alpha)) / M)


def emp_bernstein_dev(vals, R, delta):
    """Maurer & Pontil (2009) empirical Bernstein deviation for the mean of i.i.d.
    variables with range R:  mean - E <= dev  w.p. >= 1-delta (one-sided)."""
    nB = len(vals)
    V = np.var(vals, ddof=1)
    L = np.log(2 / delta)
    return np.sqrt(2 * V * L / nB) + 7 * R * L / (3 * (nB - 1))


def split_bounds_eb(g, n, counts, alpha=0.05, rng=None, m=800,
                    lams=(1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """Split-sample adaptive interval with empirical-Bernstein evaluation.
    Half A: for each penalty lam, fit a polynomial majorant/minorant (Bernstein coefficients h)
    minimising mean_A h(X) + lam*range(h); pick the lam whose bound, computed on A, is best.
    Half B: evaluate the chosen h with an empirical Bernstein bound at level alpha/2 per side."""
    rng = np.random.default_rng(0) if rng is None else rng
    counts = np.asarray(counts)
    perm = rng.permutation(len(counts))
    A, Bc = counts[perm[: len(counts) // 2]], counts[perm[len(counts) // 2:]]
    wA = np.bincount(A, minlength=n + 1) / len(A)
    out = {}
    for upper in (True, False):
        best = None
        for lam in lams:
            h, v = one_sided_poly(g, n, wA, lam, upper=upper, m=m)
            if h is None:
                continue
            R = h.max() - h.min()
            dev = emp_bernstein_dev(h[A], R, alpha / 2) if R > 0 else 0.0
            sc = (np.mean(h[A]) + dev + v) if upper else -(np.mean(h[A]) - dev - v)
            if best is None or sc < best[0]:
                best = (sc, h, v)
        _, h, v = best
        R = h.max() - h.min()
        dev = emp_bernstein_dev(h[Bc], R, alpha / 2) if R > 0 else 0.0
        out[upper] = (np.mean(h[Bc]) + dev + v) if upper else (np.mean(h[Bc]) - dev - v)
    return max(out[False], 0.0), min(out[True], 1.0)


def cp_box(counts_hist, M, alpha):
    """Bonferroni-simultaneous Clopper-Pearson intervals for all K multinomial cells."""
    from scipy.stats import beta as Beta
    K = len(counts_hist); a = alpha / K
    c = np.asarray(counts_hist)
    lo = np.where(c > 0, Beta.ppf(a / 2, c, M - c + 1), 0.0)
    hi = np.where(c < M, Beta.ppf(1 - a / 2, c + 1, M - c), 1.0)
    return lo, hi


def box_interval(g, n, counts, alpha=0.05, m=1500):
    """(1-alpha) finite-task interval for E_F[g(p)]: optimise over all mixing laws F on a grid
    whose induced count law lies in a simultaneous Clopper-Pearson box around the observed
    histogram.  Valid without parametric assumptions (up to grid discretisation of F, see paper)."""
    counts = np.asarray(counts); M = len(counts)
    hist = np.bincount(counts, minlength=n + 1)
    lo, hi = cp_box(hist, M, alpha)
    p = cheb_grid(m)
    B = bernstein_matrix(n, p).T  # (K, mm)
    gv = g(p); mm = len(p)
    A_ub = np.vstack([B, -B]); b_ub = np.concatenate([hi, -lo])
    out = []
    for sgn in (+1, -1):
        r = linprog(sgn * gv, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, mm)), b_eq=[1.0],
                    bounds=[(0, None)] * mm, method="highs")
        out.append(sgn * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]
