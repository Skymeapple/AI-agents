"""Extensions added in revision r1.

  * fast exact offspring PGF of the atomic model via the linear recursion
        phi_j(s) = A(s) + B(s) phi_{j-1}(s),  phi_0(s) = delta + (1-delta) s
    (used for Theorem 2: survival probability is monotone in the attempt budget)
  * closed-form unlimited-retry survival probabilities (Theorem 2(iii), Corollary 3)
  * attempt-level agent simulation of K-stage pipelines (atomic / checkpointed,
    finite or unlimited per-epoch budget, progress carried across epochs)
  * continuous-time model with attempt duration and defender latency (Prop. 6)
"""
import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------- atomic PGF
def _AB(s, P, h_s, h_f, sigma):
    c = 1 - sigma + sigma * s
    A = P * h_s * c + (1 - P) * h_f
    B = P * (1 - h_s) * c + (1 - P) * (1 - h_f)
    return A, B


def pgf_atomic(s, P, h_s, h_f, sigma, delta, a):
    """Exact offspring PGF f_a(s) of the atomic model (a may be np.inf)."""
    A, B = _AB(s, P, h_s, h_f, sigma)
    phistar = A / (1 - B)
    if np.isinf(a):
        return phistar
    phi0 = delta + (1 - delta) * s
    return phistar + B ** a * (phi0 - phistar)


def extinction_atomic(P, h_s, h_f, sigma, delta, a):
    """Smallest fixed point of f_a on [0,1] by root bracketing."""
    from model import mean_offspring
    m = mean_offspring(P, h_s, h_f, sigma, delta, a)
    if m <= 1:
        return 1.0
    g = lambda s: pgf_atomic(s, P, h_s, h_f, sigma, delta, a) - s
    if g(0.0) <= 0:
        return 0.0
    hi = 1 - 1e-12
    for eps in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
        if g(1 - eps) < 0:
            hi = 1 - eps
            break
    return brentq(g, 0.0, hi, xtol=1e-15, rtol=1e-14)


def survival_atomic_unlimited(P, h_s, h_f, sigma):
    """Closed form: u_inf = max(0, (1 - 1/R_inf)/(1 - h_s))."""
    R = P * sigma / (P * h_s + (1 - P) * h_f)
    return max(0.0, (1 - 1 / R) / (1 - h_s))


def survival_geometric(R):
    """Survival of a GW process with geometric offspring of mean R: max(0, 1-1/R)."""
    return np.maximum(0.0, 1 - 1 / np.asarray(R, float))


# ------------------------------------------------- K-stage agent simulation
def staged_epoch(stage, p, h, sigma, mode, budget, rng, max_steps=10**7):
    """Run one epoch for a vector of replicas.

    stage : int array, current stage index of each replica (0..K-1); a replica
            starting a fresh chain has stage 0.
    mode  : 'checkpoint' (retry only the current stage) or 'atomic'
            (any failure restarts the chain from stage 0).
    budget: tries per replica per epoch (np.inf = until detected).
    Every try is simulated explicitly. A successful try at the last stage creates
    a child (kept with prob. sigma) and the replica starts a new chain.
    Failed tries at stage k are detected with prob. h[k]; detected replicas are removed.
    Returns (alive mask, new stage array, number of surviving children per replica).
    """
    K = len(p)
    stage = stage.copy()
    N = len(stage)
    alive = np.ones(N, bool)
    kids = np.zeros(N, int)
    steps = 0
    while alive.any() and steps < budget and steps < max_steps:
        idx = np.flatnonzero(alive)
        st = stage[idx]
        u = rng.random(len(idx))
        succ = u < p[st]
        # success
        si = idx[succ]
        last = stage[si] == K - 1
        born = si[last]
        kids[born] += (rng.random(len(born)) < sigma)
        stage[born] = 0
        stage[si[~last]] += 1
        # failure
        fi = idx[~succ]
        det = rng.random(len(fi)) < h[stage[fi]]
        alive[fi[det]] = False
        if mode == "atomic":
            stage[fi[~det]] = 0
        steps += 1
    return alive, stage, kids


def staged_population(p, h, sigma, mode, budget, delta=0.0, reps=2000, cap=300,
                      gens=400, rng=None):
    """Fraction of runs (one founder) that reach `cap` replicas or are alive after `gens`."""
    rng = np.random.default_rng(rng)
    p = np.asarray(p, float); h = np.asarray(h, float)
    esc = 0
    for _ in range(reps):
        stage = np.zeros(1, int)
        for _g in range(gens):
            if len(stage) == 0 or len(stage) >= cap:
                break
            alive, st, kids = staged_epoch(stage, p, h, sigma, mode, budget, rng)
            if delta > 0:
                alive &= rng.random(len(alive)) >= delta
            stage = np.concatenate([st[alive], np.zeros(int(kids.sum()), int)])
        esc += len(stage) > 0
    return esc / reps


def atomic_effective(p, h):
    """Map a K-stage atomic pipeline with stage-specific failure detection to the
    single-stage model: P = prod p_k, h_f = P(detected | failed attempt)."""
    p = np.asarray(p, float); h = np.asarray(h, float)
    P = np.prod(p)
    reach = np.concatenate([[1.0], np.cumprod(p)[:-1]])   # prob. of reaching stage k
    fail_k = reach * (1 - p)
    return P, float(np.sum(fail_k * h) / (1 - P))


# ------------------------------------------ continuous time, defender latency
def R_latency(P, h_f, sigma, L):
    """Mean offspring with attempt rate lambda, flag-to-removal latency ell (L=lambda*ell
    expected attempts during latency), no background removal, successes unflagged."""
    return sigma * P / ((1 - P) * h_f) + sigma * P * L


def surv_latency(P, h_f, sigma, L):
    """Exact survival: children ~ Poisson(sigma*P*(T_U+L)) with T_U ~ Exp((1-P)h_f) in attempt
    units, so f(s) = exp(-sigma P L (1-s)) * theta/(theta + sigma P (1-s)), theta=(1-P)h_f."""
    th = (1 - P) * h_f; b = sigma * P
    f = lambda s: np.exp(-b * L * (1 - s)) * th / (th + b * (1 - s))
    if R_latency(P, h_f, sigma, L) <= 1:
        return 0.0
    g = lambda s: f(s) - s
    hi = 1 - 1e-9
    for eps in [1e-3, 1e-5, 1e-7, 1e-9]:
        if g(1 - eps) < 0:
            hi = 1 - eps; break
    return 1 - brentq(g, 0.0, hi, xtol=1e-14)


def malthus_latency(P, h_f, sigma, L, lam=1.0):
    """Malthusian growth rate r (per unit time, attempt rate lam) from the Lotka-Euler
    equation  sigma*lam*P*[ (1-e^{-r ell})/r + e^{-r ell}/(r+theta) ] = 1,
    theta = lam (1-P) h_f, ell = L/lam. Returns r (negative if subcritical)."""
    ell = L / lam; th = lam * (1 - P) * h_f; b = sigma * lam * P
    def F(r):
        if abs(r) < 1e-12:
            return b * (ell + 1 / th) - 1
        return b * ((1 - np.exp(-r * ell)) / r + np.exp(-r * ell) / (r + th)) - 1
    lo = -th + 1e-9
    return brentq(F, lo, 50 * lam)
