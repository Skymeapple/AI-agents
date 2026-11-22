"""Core tools: binomial-mixture model of repeated-attempt capability evaluation.

Notation
--------
Each task i has a per-attempt success probability p_i drawn from a mixing
distribution G on [0,1].  The evaluator observes X_i ~ Bin(n, p_i) for T tasks.
The target is the budget-B coverage  C(B) = E_G[1 - (1-p)^B]
(the expected fraction of tasks an actor with B i.i.d. attempts solves).
"""
import numpy as np
from scipy import optimize, stats, special
from scipy.optimize import linprog

# ---------------------------------------------------------------- basics

def pass_at_k_chen(c, n, k):
    """Unbiased pass@k estimator of Chen et al. (2021) for one task (k<=n)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.exp(special.gammaln(n - c + 1) - special.gammaln(n - c - k + 1)
                        - special.gammaln(n + 1) + special.gammaln(n - k + 1))


def mean_pass_at_k(counts, n, k):
    return float(np.mean([pass_at_k_chen(int(c), n, k) for c in counts]))


def histogram(counts, n):
    return np.bincount(np.asarray(counts, int), minlength=n + 1)[: n + 1]


def binom_matrix(n, grid):
    """A[j, g] = P(X=j | p=grid[g]) for X~Bin(n,p)."""
    j = np.arange(n + 1)[:, None]
    return stats.binom.pmf(j, n, grid[None, :])


def coverage_true(p_samples, B):
    return float(np.mean(1.0 - (1.0 - p_samples) ** B))


def default_grid(pmin=0.0, m_log=260, m_lin=140, lo=1e-9):
    """Grid on [0,1] dense on a log scale near 0 (where extrapolation lives)."""
    lg = np.logspace(np.log10(max(lo, pmin if pmin > 0 else lo)), 0, m_log)
    ln = np.linspace(max(pmin, 1e-3), 1, m_lin)
    g = np.unique(np.concatenate([lg, ln, [1.0]]))
    g = g[g >= pmin] if pmin > 0 else g
    return np.concatenate([[0.0], g])


# ---------------------------------------------------- partial identification

def cp_bounds(h, alpha):
    """Simultaneous (Bonferroni) Clopper-Pearson bounds on the n+1 histogram cells."""
    T = h.sum()
    a = alpha / len(h)
    lo = np.where(h > 0, stats.beta.ppf(a / 2, h, T - h + 1), 0.0)
    hi = np.where(h < T, stats.beta.ppf(1 - a / 2, h + 1, T - h), 1.0)
    return lo, hi


def lp_bounds(h, n, B, alpha=0.05, pmin=0.0, grid=None, exact_q=None, slack=0.0):
    """Lower/upper bounds on C(B) over all mixing distributions G supported on
    {0} U [pmin, 1] whose induced histogram probabilities lie in a simultaneous
    1-alpha confidence box (or equal exact_q when given = population limit).
    Returns (lower, upper)."""
    if grid is None:
        grid = default_grid(pmin)
    A = binom_matrix(n, grid)
    f = 1.0 - (1.0 - grid) ** B
    M = len(grid)
    A_eq = [np.ones(M)]
    b_eq = [1.0]
    if exact_q is not None:
        A_ub = np.vstack([A, -A])
        b_ub = np.concatenate([exact_q + slack, -(exact_q - slack)])
    else:
        lo, hi = cp_bounds(h, alpha)
        A_ub = np.vstack([A, -A])
        b_ub = np.concatenate([hi, -lo])
    out = []
    for sgn in (+1, -1):  # +1 -> minimise f (lower), -1 -> maximise f (upper)
        r = linprog(sgn * f, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                    bounds=(0, None), method="highs")
        out.append(sgn * r.fun if r.status == 0 else np.nan)
    return out[0], out[1]


# ------------------------------------------------------ parametric methods

def betabin_fit(h, n):
    """MLE of a beta-binomial (a,b) from a success-count histogram."""
    j = np.arange(n + 1)

    def nll(theta):
        a, b = np.exp(theta)
        lp = (special.gammaln(n + 1) - special.gammaln(j + 1) - special.gammaln(n - j + 1)
              + special.betaln(j + a, n - j + b) - special.betaln(a, b))
        return -np.sum(h * lp)

    best = None
    for x0 in ([0.0, 0.0], [-1.5, 0.5]):
        r = optimize.minimize(nll, x0, method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(np.clip(best.x, -12, 12))
    return a, b


def betabin_coverage(a, b, B):
    return float(1.0 - np.exp(special.betaln(a, b + B) - special.betaln(a, b)))


def pass_at_k_hist(h, n, k):
    """Mean Chen et al. pass@k computed from a success-count histogram (k<=n)."""
    j = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = (special.gammaln(n - j + 1) - special.gammaln(np.maximum(n - j - k, 0) + 1)
              - special.gammaln(n + 1) + special.gammaln(n - k + 1))
        fail = np.where(n - j >= k, np.exp(lr), 0.0)
    return float(np.sum(h * (1 - fail)) / h.sum())


def powerlaw_coverage(h, n, B):
    """Fit -log pass@k = A k^{-b} on k=1..n (Brown et al. 2024 / Hughes et al. 2024
    style) by least squares in log-log space and extrapolate to k=B."""
    ks = np.arange(1, n + 1)
    pk = np.array([pass_at_k_hist(h, n, k) for k in ks])
    ok = (pk > 0) & (pk < 1)
    if ok.sum() < 2:
        return float(pk[-1])
    y = np.log(-np.log(pk[ok]))
    X = np.log(ks[ok])
    slope, icpt = np.polyfit(X, y, 1)
    return float(np.exp(-np.exp(icpt + slope * np.log(B))))


def bootstrap_ucb(h, n, B, method, reps=100, alpha=0.05, rng=None):
    """Nonparametric bootstrap over tasks (multinomial resampling of histogram)."""
    rng = np.random.default_rng(rng)
    T = int(h.sum())
    vals = []
    for _ in range(reps):
        hb = rng.multinomial(T, h / T)
        if method == "betabin":
            a, b = betabin_fit(hb, n)
            vals.append(betabin_coverage(a, b, B))
        else:
            vals.append(powerlaw_coverage(hb, n, B))
    return float(np.quantile(vals, 1 - alpha))


def naive_ucb(h, n, alpha=0.05):
    """Treat observed pass@n (any success in n attempts) as the capability."""
    T = int(h.sum()); s = T - int(h[0])
    return float(stats.beta.ppf(1 - alpha, s + 1, T - s)) if s < T else 1.0
