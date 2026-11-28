"""Shared helpers for the revision experiments (round 1 response).

- sample_world: exact on-policy sampler for the exploit-mixture model with K proxies,
  optional twin world W' (shared exploits have gold c_twin + g0; default c_twin = mu),
  optional gold-label noise / grader bias (labels differ from true gold).
- stopping rules that consume a stream of checkpoints and return the stopping index
  together with the cumulative number of gold labels used.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm
from model import Params, closed_form


def sample_world(P, n, lam, rng, K=2, twin=False, c_twin=None):
    pi = float(expit(logit(P.p) + P.mu * lam))
    Z = rng.random(n) < pi
    g0 = rng.normal(P.sg2 * lam, np.sqrt(P.sg2), n)
    c = rng.normal(0, 1, n)
    U = rng.normal(0, 1, (n, K))
    E = np.sqrt(P.se2) * (np.sqrt(P.rho) * c[:, None] + np.sqrt(1 - P.rho) * U)
    shift = np.full(K, P.rho * P.se2 * lam); shift[0] = P.se2 * lam
    E = E + shift[None, :]
    B = np.zeros((n, K)); B[:, 0] = 1.0
    if K > 1:
        B[:, 1:] = rng.random((n, K - 1)) < P.q
    R = g0[:, None] + E + (Z[:, None] * B) * P.mu
    gold = np.where(Z, -P.delta, g0)
    if twin:
        shared = Z & (B[:, 1:].min(axis=1) > 0.5) if K > 1 else Z
        ct = P.mu if c_twin is None else c_twin
        gold = np.where(shared, ct + g0, gold)
    return dict(Z=Z, gold=gold, R=R, B=B, g0=g0)


def gold_curve_twin(P, lam, c_twin=None, K=2):
    """Closed-form gold in W' with shared-exploit gold c_twin + g0 (K proxies)."""
    cf = closed_form(P, lam)
    ct = P.mu if c_twin is None else c_twin
    qs = P.q ** (K - 1)
    return cf["gold"] + cf["pi"] * qs * (ct + P.sg2 * lam + P.delta)


# ---------------- estimators ----------------
def hh_cov(lab, h, m, w, rng):
    """Hansen-Hurwitz estimate of mean(lab * h) with sampling probs w (sum to 1)."""
    N = len(h)
    idx = rng.choice(N, size=m, replace=True, p=w)
    t = lab[idx] * h[idx] / (N * w[idx])
    return t.mean(), (t.std(ddof=1) / np.sqrt(m) if m > 1 else np.inf)


def lev_probs(h):
    a = np.abs(h); w = a + 0.1 * a.mean(); return w / w.sum()


def gold_mean_est(lab, r1, m, rng, reg=False):
    """Uniform audit of m labels; optional regression estimator using r1 known for the whole batch."""
    N = len(r1)
    idx = rng.choice(N, size=m, replace=False)
    y = lab[idx]
    if not reg or m < 3:
        return y.mean(), y.std(ddof=1) / np.sqrt(m)
    x = r1[idx]
    b = np.cov(x, y, ddof=1)[0, 1] / (np.var(x, ddof=1) + 1e-12)
    est = y.mean() + b * (r1.mean() - x.mean())
    res = y - y.mean() - b * (x - x.mean())
    return est, res.std(ddof=2) / np.sqrt(m)


def ols_slope(x, y):
    x = np.asarray(x); y = np.asarray(y)
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def z_anytime(k, alpha=0.1):
    """Union-bound alpha spending alpha_k = alpha/((k+1)(k+2)) -> one-sided z."""
    return float(norm.isf(alpha / ((k + 1) * (k + 2))))
