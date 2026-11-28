"""Closed-form results for the verify-checkpoint-rollback (VCR) model of
long-running agents with silent errors and imperfect verifiers.

Model summary (see paper Section 3):
  * The task needs N units of committed progress ("steps").
  * Step executed at context age a corrupts the state with prob. p(a).
  * Each error e carries a detectability d_e ~ D (i.i.d.), the probability
    that one verifier call flags it.  rho = P(d_e = 0) is the blind-spot mass.
  * Work is committed in segments of T steps.  A segment is verified m times
    (independent calls, each also raising a false alarm w.p. f).  If every call
    passes, the segment is checkpointed (cost C); otherwise the agent rolls back
    to the last checkpoint (cost R) and re-executes the segment.
  * Each verifier call costs V; each step costs 1.
  * Context policy  'fresh'   : every (re)start from a checkpoint begins with a
                                fresh context (age 0) and pays K re-orientation.
                    'snapshot': the context is restored with the checkpoint, so
                                age keeps growing with committed progress.
"""
import numpy as np
from scipy.special import betaln


# ---------------------------------------------------------------------------
# detectability distributions:  mu_m = E[(1-d)^m]
# ---------------------------------------------------------------------------
def mu_point(r, m, rho=0.0):
    """d = r with prob 1-rho, d = 0 with prob rho."""
    return rho + (1.0 - rho) * (1.0 - r) ** m


def mu_beta(a, b, m, rho=0.0):
    """d ~ Beta(a,b) with prob 1-rho, d = 0 with prob rho.
    E[(1-d)^m] = B(a, b+m)/B(a,b)."""
    m = np.asarray(m, dtype=float)
    return rho + (1.0 - rho) * np.exp(betaln(a, b + m) - betaln(a, b))


def mu_empirical(dvals, m):
    dvals = np.asarray(dvals, dtype=float)
    m = np.atleast_1d(m)
    return np.array([np.mean((1.0 - dvals) ** k) for k in m])


# ---------------------------------------------------------------------------
# Theorem 1: interval invariance of end-to-end correctness (constant hazard)
# ---------------------------------------------------------------------------
def theta(p, mu):
    """Probability that one committed step is clean."""
    return (1.0 - p) / (1.0 - p + p * mu)


def p_clean_const(p, mu, N):
    return theta(p, mu) ** N


def floor_const(p, rho, N):
    """Theorem 2 upper bound for ANY verifier-driven policy."""
    return theta(p, rho) ** N


# ---------------------------------------------------------------------------
# Segment pass probabilities and expected makespan (general hazard)
# ---------------------------------------------------------------------------
def hazard_linear(p0, beta, ages):
    return np.minimum(1.0, p0 * (1.0 + beta * np.asarray(ages, dtype=float)))


def segment_stats(pa, mu, m, f):
    """pa: array of per-step hazards inside the segment.
    Returns (pass prob s, P(clean & pass), P(clean | pass))."""
    pa = np.asarray(pa, dtype=float)
    s = (1.0 - f) ** m * np.prod(1.0 - pa * (1.0 - mu))
    cp = (1.0 - f) ** m * np.prod(1.0 - pa)
    return s, cp, cp / s if s > 0 else np.nan


def makespan_and_correctness(N, T, p0, beta, mu, m, f, V, C, R, K=0.0,
                             policy="fresh"):
    """Exact expected makespan and exact P(final state clean) for the VCR
    model with linear hazard p(a)=p0(1+beta a).  N must be a multiple of T
    (the last partial segment is handled as a shorter segment otherwise)."""
    seg_lens = [T] * (N // T) + ([N % T] if N % T else [])
    E = 0.0
    logc = 0.0
    start_age = 0
    for L in seg_lens:
        if policy == "fresh":
            ages = np.arange(L)
            k = K
        else:  # snapshot: age continues with committed progress
            ages = start_age + np.arange(L)
            k = 0.0
        pa = hazard_linear(p0, beta, ages)
        s, cp, cgp = segment_stats(pa, mu, m, f)
        if s <= 1e-300:
            return np.inf, 0.0
        attempt = L + m * V + k
        E += attempt / s + R * (1.0 / s - 1.0) + C
        logc += np.log(cgp)
        start_age += L
    return E, np.exp(logc)


def tstar_approx(p0, beta, mu, m, V, C, K=0.0):
    """First-order optimal interval: c/T^2 = (1-mu) p0 (1 + beta T)."""
    c = m * V + C + K
    lam = (1.0 - mu) * p0
    # solve lam*beta*T^3 + lam*T^2 - c = 0 for T>0
    roots = np.roots([lam * beta, lam, 0.0, -c]) if beta > 0 else \
        np.array([np.sqrt(c / lam)])
    roots = roots[np.isreal(roots)].real
    return roots[roots > 0].max()


def horizon_multiplier(mu):
    """Ratio of 50%-correctness horizons with and without verification,
    small-p limit: h_ver / h_none = 1/mu."""
    return 1.0 / mu
