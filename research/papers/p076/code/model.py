"""Throughput-bounded meaningful human control (TB-MHC): core model.

A machine nominates items (proposed targets). A fraction q of nominations are
valid (lawful, correctly identified); 1-q are machine errors. A human reviewer
spends t seconds per nomination and observes evidence
    x ~ N(+d(t)/2, 1) if valid,   x ~ N(-d(t)/2, 1) if invalid,
and approves iff x > c.  d(t) is the reviewer's *incremental* discriminability
given t seconds, modelled by the shifted-exponential speed-accuracy function
(Wickelgren 1977; McElree & Dosher 1989):
    d(t) = D * (1 - exp(-(t - delta)/tau))  for t > delta, else 0.
Two criterion-setting regimes:
  * quota regime: c is set so that a fixed fraction h of valid nominations is
    approved (operational tempo constraint);
  * Bayes regime: c minimises expected cost with prior q (possibly biased by
    automation-bias shift alpha in log-odds) and cost ratio r = C_F / C_M.
Catch rate kappa = P(veto | invalid) = 1 - FA.
"""
import numpy as np
from scipy.stats import norm

LOG_EPS = 1e-300


def d_of_t(t, D, tau, delta=0.0):
    t = np.asarray(t, dtype=float)
    return np.where(t > delta, D * (1.0 - np.exp(-(t - delta) / tau)), 0.0)


def d_of_t_sqrt(t, k, delta=0.0):
    """Alternative non-saturating form (drift-diffusion: d' grows like sqrt(t))."""
    t = np.asarray(t, dtype=float)
    return np.where(t > delta, k * np.sqrt(np.maximum(t - delta, 0.0)), 0.0)


def t_of_d(d, D, tau, delta=0.0):
    """Inverse of d_of_t; inf if d >= D."""
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = delta + tau * np.log(D / (D - d))
    return np.where(d < D, np.where(d <= 0, delta, out), np.inf)


# ---------------- quota regime ----------------
def quota_rates(d, h):
    """Return (hit, false-approval) with criterion set so that hit rate = h."""
    zh = norm.ppf(h)
    fa = norm.cdf(zh - np.asarray(d, float))
    return np.full_like(fa, h), fa


def quota_dstar(kappa, h):
    """Minimum incremental d' for catch rate >= kappa under hit quota h."""
    return norm.ppf(h) + norm.ppf(kappa)


# ---------------- Bayes regime ----------------
def log_beta(q, r, alpha=0.0):
    """Subjective decision threshold on log-likelihood ratio.
    approve iff LLR > log_beta. alpha>0 = automation bias (inflated trust)."""
    return np.log((1.0 - q) / q) + np.log(r) - alpha


def bayes_rates(d, q, r, alpha=0.0):
    d = np.asarray(d, float)
    lb = log_beta(q, r, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(d > 0, lb / np.where(d > 0, d, 1.0), np.sign(lb) * np.inf)
    hit = norm.cdf(d / 2 - c)
    fa = norm.cdf(-d / 2 - c)
    # d = 0 limit: approve all if lb<0, veto all if lb>0, coin flip if lb==0
    zero = d <= 0
    hit = np.where(zero, np.where(lb < 0, 1.0, np.where(lb > 0, 0.0, 0.5)), hit)
    fa = np.where(zero, np.where(lb < 0, 1.0, np.where(lb > 0, 0.0, 0.5)), fa)
    return hit, fa


def bayes_dstar(kappa, q, r, alpha=0.0):
    """Minimum d' for catch rate >= kappa in the approve-default regime
    (log_beta < 0).  Closed form: z + sqrt(z^2 + 2|lb|), z = Phi^-1(kappa).
    If log_beta >= 0 the reviewer vetoes by default; returns the (possibly 0)
    d' needed, computed numerically for generality."""
    lb = log_beta(q, r, alpha)
    z = norm.ppf(kappa)
    if lb < 0:
        return z + np.sqrt(z * z + 2.0 * abs(lb))
    # veto-default regime: catch rate = Phi(lb/d + d/2) >= kappa
    # solve lb/d + d/2 >= z  <=> d^2/2 - z d + lb >= 0
    disc = z * z - 2.0 * lb
    if disc < 0:
        return 0.0  # satisfied for all d>0
    lo, hi = z - np.sqrt(disc), z + np.sqrt(disc)
    # inequality fails on (lo, hi); small d (<lo) already satisfies it
    return 0.0 if lo > 0 else max(hi, 0.0)


# ---------------- throughput ----------------
def review_time(lam, eta=1.0):
    """Seconds per nomination for one operator handling lam nominations/hour."""
    return 3600.0 * eta / np.asarray(lam, float)


def lambda_max(dstar, D, tau, delta=0.0, eta=1.0):
    t = t_of_d(dstar, D, tau, delta)
    return np.where(np.isfinite(t), 3600.0 * eta / t, 0.0)


def veto_rate(q, hit, fa):
    return q * (1 - hit) + (1 - q) * (1 - fa)


def unlawful_per_hour(lam, q, fa):
    return np.asarray(lam, float) * (1 - q) * fa
