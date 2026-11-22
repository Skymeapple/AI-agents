"""Audit designs for the on-policy covariance test of reward hacking.

Key identity (exponential tilting, any gold g and proxy r1):
    d/dlam E_{pi_lam}[g] = Cov_{pi_lam}(g, r1).
So gold is still improving iff the on-policy covariance between gold and the
optimised proxy is positive. We estimate it from a batch of N on-policy samples
(proxies r1, r2 observed for all) with gold audited on m of them.
"""
import numpy as np


def design_probs(r1, r2, design):
    h = r1 - r1.mean()
    a = np.abs(h)
    if design == "uniform":
        w = np.ones_like(r1)
    elif design == "leverage":
        w = a + 0.1 * a.mean()
    elif design == "disagree":
        gap = r1 - r2
        z = (gap - np.median(gap)) / (gap.std() + 1e-12)
        w = (a + 0.1 * a.mean()) * (1.0 + np.maximum(z, 0.0))
    else:
        raise ValueError(design)
    return w / w.sum()


def cov_estimate(g, r1, r2, m, design, cv, rng, a_cv=1.0):
    """Unbiased (over audit randomness) estimate of the batch covariance Cov(g, r1).
    cv=True uses the held-out proxy r2 as a control variate: exact batch term
    mean(a*r2*h) plus an audited correction for the residual g - a*r2."""
    N = len(r1)
    h = r1 - r1.mean()
    qp = design_probs(r1, r2, design)
    idx = rng.choice(N, size=m, replace=True, p=qp)
    if cv:
        base = np.mean(a_cv * r2 * h)
        y = g[idx] - a_cv * r2[idx]
    else:
        base = 0.0
        y = g[idx]
    terms = y * h[idx] / (N * qp[idx])
    est = base + terms.mean()
    se = terms.std(ddof=1) / np.sqrt(m) if m > 1 else np.inf
    return est, se, idx


def batch_cov(g, r1):
    return float(np.mean(g * (r1 - r1.mean())))
