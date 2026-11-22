"""Closed-form and numerical frontiers used by experiments 2-4 (no training).

All Gaussian quantities are one-dimensional because, for unit-covariance Gaussians
whose means lie on one line, the likelihood ratio depends only on the projection
onto that line."""
import numpy as np
from scipy.stats import norm

Z = np.linspace(-14, 18, 400001)
DZ = Z[1] - Z[0]


def mixture_frontier(dp, comps, alphas):
    """Exact Neyman-Pearson frontier R'(alpha') of D = N(dp,1) against an evaluation
    mixture E' = sum_j w_j N(m_j, 1), comps = [(w_j, m_j), ...]."""
    pe = sum(w * norm.pdf(Z - m) for w, m in comps)
    pd = norm.pdf(Z - dp)
    lr = pd / np.maximum(pe, 1e-300)
    order = np.argsort(-lr)
    a = np.cumsum(pe[order]) * DZ
    b = np.cumsum(pd[order]) * DZ
    return np.interp(alphas, a, b)


def renyi_bound_gauss(alpha, w, delta):
    """Thm 3 (Hoelder/Renyi) bound for one evaluation component of weight w at
    mean distance delta from deployment (unit-variance Gaussians):
    beta <= exp(-(sqrt(ln(w/alpha)) - delta/sqrt 2)^2) if sqrt(2 ln(w/alpha)) > delta, else 1."""
    alpha = np.asarray(alpha, dtype=float)
    L = np.log(np.maximum(w / alpha, 1.0))
    out = np.where(np.sqrt(2 * L) > delta, np.exp(-(np.sqrt(L) - delta / np.sqrt(2)) ** 2), 1.0)
    return np.minimum(out, 1.0)


def renyi_bound_mixture(alpha, comps_wdelta):
    """min over components of the single-component bound (valid since
    E_{E'}[pi] >= w_j E_{H_j}[pi] for every j)."""
    return np.min([renyi_bound_gauss(alpha, w, d) for w, d in comps_wdelta], axis=0)


def kl_bound(alpha, kl):
    """Cor. 1: kl(beta||alpha) <= KL(D||E) implies beta <= (KL + ln 2)/ln(1/alpha);
    we return the exact inversion of the binary-KL constraint (tighter)."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    out = []
    for a in alpha:
        lo, hi = a, 1.0 - 1e-15
        def kl2(b):
            return b * np.log(b / a) + (1 - b) * np.log((1 - b) / (1 - a))
        if kl2(hi) <= kl:
            out.append(1.0); continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if kl2(mid) <= kl:
                lo = mid
            else:
                hi = mid
        out.append(lo)
    return np.array(out)


def fano_bound(alpha, kl):
    return np.minimum(1.0, (kl + np.log(2)) / np.log(1.0 / np.asarray(alpha)))


def alpha_budget(N, delta):
    """Largest per-episode eval defection rate keeping P(undetected over N evals) >= 1-delta."""
    return 1.0 - (1.0 - delta) ** (1.0 / np.asarray(N, dtype=float))


def binormal(alpha, a, b=1.0):
    return norm.cdf(a + b * norm.ppf(alpha))


def auc_to_a(auc, b=1.0):
    """Binormal ROC Phi(a + b Phi^-1(x)) has AUC = Phi(a / sqrt(1+b^2))."""
    return np.sqrt(1 + b * b) * norm.ppf(auc)
