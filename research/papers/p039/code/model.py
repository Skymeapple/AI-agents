"""Core model: optimal loss-of-control (LoC) mitigation spending under
isoelastic mitigation returns, belief pooling, robustness, and value of information.

Notation (all money in present-value US dollars):
  p    : baseline probability of an LoC catastrophe over the horizon (no extra mitigation)
  L    : present-value loss if it occurs
  S    : mitigation spending (PV)
  g(S) = (1 - m) + m * (1 + S/k)^(-beta)   residual-risk multiplier
         m    : reducible fraction of the risk (0 < m <= 1)
         k    : spending scale at which mitigation starts to bite
         beta : elasticity of reducible risk w.r.t. spending (for S >> k)
Expected social cost: C(S; p) = p * L * g(S) + S.
"""
import numpy as np
from scipy.special import expit, logit


def g(S, k, beta, m=1.0):
    return (1.0 - m) + m * (1.0 + S / k) ** (-beta)


def cost(S, p, L, k, beta, m=1.0):
    return p * L * g(S, k, beta, m) + S


def s_star(p, L, k, beta, m=1.0):
    """Closed-form optimal spending (Prop. 1)."""
    A = m * p * beta * L / k
    a = 1.0 / (1.0 + beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(A > 1.0, A ** a, 1.0)
    return k * (x - 1.0)


def c_star(p, L, k, beta, m=1.0):
    """Closed-form minimised expected cost (Lemma 2): concave in p."""
    A = m * p * beta * L / k
    a = 1.0 / (1.0 + beta)
    interior = A > 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = p * L * (1.0 - m) + k * ((1.0 + 1.0 / beta) * np.where(interior, A, 1.0) ** a - 1.0)
    return np.where(interior, ci, p * L)


def p_threshold(L, k, beta, m=1.0):
    """Belief below which zero additional spending is optimal (Prop. 1(ii))."""
    return k / (m * beta * L)


# ---------------- belief distributions ----------------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(200)  # probabilists' Hermite
GH_W = GH_W / GH_W.sum()


def logitnormal_mean(mu, sigma):
    return float(np.sum(GH_W * expit(mu + sigma * GH_X)))


def fit_logitnormal(median, mean):
    """Fit logit-normal with given median and mean (median pins mu)."""
    from scipy.optimize import brentq
    mu = logit(median)
    sigma = brentq(lambda s: logitnormal_mean(mu, s) - mean, 1e-4, 20.0)
    return mu, sigma


def ln_nodes(mu, sigma, n=4000):
    """Equal-weight quantile grid for p ~ logit-normal(mu, sigma) (a synthetic
    forecaster population); clipped away from 0/1 so logits stay finite."""
    from scipy.stats import norm
    z = norm.ppf((np.arange(n) + 0.5) / n)
    return np.clip(expit(mu + sigma * z), 1e-12, 1 - 1e-12), np.full(n, 1.0 / n)


# ---------------- pooling ----------------
def pools(p, w):
    p = np.asarray(p); w = np.asarray(w) / np.sum(w)
    lin = np.sum(w * p)
    lo = np.sum(w * logit(p))
    order = np.argsort(p); cw = np.cumsum(w[order])
    med = p[order][np.searchsorted(cw, 0.5)]
    geo_p = np.exp(np.sum(w * np.log(p)))
    return dict(linear=lin, log_odds=expit(lo), median=med, geometric_p=geo_p)


# ---------------- value of information ----------------
def evpi(p, w, L, k, beta, m=1.0):
    """EVPI of learning p exactly before choosing S, prior = discrete (p, w)."""
    w = np.asarray(w) / np.sum(w)
    pbar = np.sum(w * p)
    no_info = c_star(pbar, L, k, beta, m)  # C(S*(pbar); p) is linear in p -> evaluate at pbar
    full = np.sum(w * c_star(p, L, k, beta, m))
    return float(no_info - full), float(no_info), float(s_star(pbar, L, k, beta, m))


def evsi_lognormal_ratio(sigma0, sigma1, beta):
    """Closed-form EVSI/EVPI for a lognormal belief whose log-variance is reduced
    from sigma0^2 to sigma1^2 by an unbiased Gaussian signal (Prop. 4), interior case,
    ignoring the -k constant (which cancels)."""
    a = 1.0 / (1.0 + beta)
    c = a * (1 - a) / 2.0
    return (1 - np.exp(-c * (sigma0 ** 2 - sigma1 ** 2))) / (1 - np.exp(-c * sigma0 ** 2))


def evpi_share_lognormal(sigma0, beta):
    """Closed-form EVPI / C*(pbar) for lognormal beliefs, large-A limit."""
    a = 1.0 / (1.0 + beta)
    return 1 - np.exp(-a * (1 - a) * sigma0 ** 2 / 2.0)


# ---------------- robust rules ----------------
def maxmin_spend(p_set, L, k, beta, m=1.0):
    return s_star(np.max(p_set), L, k, beta, m)


def minimax_regret_spend(p_set, L, k, beta, m=1.0, grid=None):
    p_set = np.asarray(p_set)
    smax = float(s_star(p_set.max(), L, k, beta, m))
    if grid is None:
        grid = np.concatenate([[0.0], np.geomspace(max(k * 1e-3, 1.0), max(smax * 1.5, k), 4000)])
    cs = c_star(p_set, L, k, beta, m)
    R = cost(grid[:, None], p_set[None, :], L, k, beta, m) - cs[None, :]
    worst = R.max(axis=1)
    i = int(np.argmin(worst))
    return float(grid[i]), float(worst[i])
