"""Core model: replication-detection branching processes for autonomous AI replicas.

Notation (see paper Sec. 3):
  P      per-attempt end-to-end replication success probability (atomic attempt)
  h_s    probability a *successful* attempt gets the parent detected & shut down
  h_f    probability a *failed* attempt gets the parent detected & shut down
  sigma  probability a newly created child survives to the next epoch
  delta  idle (background) detection probability per epoch for the parent
  a      attempt budget per epoch
All quantities are per epoch (one generation of the Galton-Watson process).
"""
import numpy as np


# ---------------------------------------------------------------- atomic model
def mean_offspring(P, h_s, h_f, sigma, delta, a):
    """Closed form m(a) (Theorem 1)."""
    r = P * (1 - h_s) + (1 - P) * (1 - h_f)
    if np.isinf(a):
        return P * sigma / (1 - r)
    return P * sigma * (1 - r ** a) / (1 - r) + (1 - delta) * r ** a


def R_inf(P, h_s, h_f, sigma):
    """Unlimited-attempt reproduction number R_inf = P sigma / (P h_s + (1-P) h_f)."""
    return P * sigma / (P * h_s + (1 - P) * h_f)


def offspring_pmf(P, h_s, h_f, sigma, delta, a):
    """Exact offspring pmf of the atomic attempt model by dynamic programming.

    Returns pmf over Z = Binomial(births, sigma) + 1{parent survives}.
    """
    a = int(a)
    # alive[b] = prob parent alive having produced b births; dead[b] likewise
    alive = np.zeros(a + 2); dead = np.zeros(a + 2)
    alive[0] = 1.0
    for _ in range(a):
        na = np.zeros_like(alive); nd = dead.copy()
        # success
        na[1:] += alive[:-1] * P * (1 - h_s)
        nd[1:] += alive[:-1] * P * h_s
        # failure
        na += alive * (1 - P) * (1 - h_f)
        nd += alive * (1 - P) * h_f
        alive, dead = na, nd
    # parent idle detection at epoch end
    surv = alive * (1 - delta)
    died = dead + alive * delta
    nmax = a + 2
    pmf = np.zeros(nmax + 1)
    from scipy.stats import binom
    for b in range(a + 2):
        if surv[b] == 0 and died[b] == 0:
            continue
        kb = binom.pmf(np.arange(b + 1), b, sigma)
        pmf[:b + 1] += died[b] * kb
        pmf[1:b + 2] += surv[b] * kb
    return pmf / pmf.sum()


def pgf(pmf, s):
    return np.polyval(pmf[::-1], s)


def extinction_prob(pmf, tol=1e-13, maxit=200000):
    q = 0.0
    for _ in range(maxit):
        qn = pgf(pmf, q)
        if abs(qn - q) < tol:
            return qn
        q = qn
    return q


# ----------------------------------------------------------- agent-level MC
def simulate_population(P, h_s, h_f, sigma, delta, a, N0=1, gens=60, cap=2000,
                        reps=4000, rng=None):
    """Attempt-level Monte Carlo. Returns fraction of runs reaching `cap` replicas
    (treated as escape) or alive after `gens` generations."""
    rng = np.random.default_rng(rng)
    escaped = 0
    for _ in range(reps):
        N = N0
        for _g in range(gens):
            if N == 0 or N >= cap:
                break
            alive = np.ones(N, bool)
            births = np.zeros(N, int)
            for _j in range(a):
                if not alive.any():
                    break
                succ = rng.random(N) < P
                det_p = np.where(succ, h_s, h_f)
                births += (succ & alive)
                det = rng.random(N) < det_p
                alive &= ~det
            alive &= rng.random(N) >= delta
            kids = rng.binomial(births, sigma).sum()
            N = int(alive.sum() + kids)
        escaped += (N > 0)
    return escaped / reps


# ------------------------------------------------------- checkpointed pipeline
def stage_pi(p, h):
    """Probability a stage is eventually completed before detection when retried."""
    p = np.asarray(p, float); h = np.asarray(h, float)
    return p / (p + (1 - p) * h)


def stage_odds(p, h):
    p = np.asarray(p, float); h = np.asarray(h, float)
    return p / ((1 - p) * h)


def R_checkpoint(p, h, sigma=1.0):
    """R for a K-stage pipeline with per-stage retries (successes undetected)."""
    Pi = np.prod(stage_pi(p, h), axis=-1)
    return sigma * Pi / (1 - Pi)


def R_atomic_multistage(p, h_f, sigma=1.0):
    """R when the whole chain must be retried atomically (h_s = 0)."""
    P = np.prod(p, axis=-1)
    return sigma * P / ((1 - P) * h_f)


def elasticities_checkpoint(p, h):
    pi = stage_pi(p, h); Pi = np.prod(pi)
    e_h = -(1 - pi) / (1 - Pi)
    e_p = (1 - pi) / (1 - np.asarray(p)) / (1 - Pi)
    return e_h, e_p


def pass_at_k_to_p1(pass_k, k):
    """Invert pass@k = 1-(1-p)^k under independent attempts."""
    return 1 - (1 - pass_k) ** (1.0 / k)
