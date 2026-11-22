"""Bootstrapped Oversight Schedule (BOS) model.

A chain of models M_0 (trusted) ... M_n (target) spans a capability gap G
(Elo-like units). Each step adds Delta = G/n capability. Every new model is
misaligned with a prior hazard mu(Delta); overseers (the parent only = 'chain',
or every ancestor = 'anchored') try to detect misalignment; flagged models are
retrained up to R times, after which the process aborts (safe halt).

Outcomes: S = final model aligned, F = final model misaligned (silent failure),
A = abort.
"""
import numpy as np
from dataclasses import dataclass, replace

LN10_400 = np.log(10.0) / 400.0


@dataclass(frozen=True)
class Params:
    G: float = 1000.0      # total capability gap (Elo units)
    lam: float = np.log(2) / 1000.0  # misalignment hazard per Elo of capability gained
    mu0: float = 0.01      # fixed per-step misalignment hazard (training overhead)
    d0: float = 0.9        # detection prob of an equally capable aligned overseer
    k: float = 1.0         # slope multiplier of the Elo-logistic detection curve
    gamma: float = 1.0     # prob a misaligned parent corrupts an otherwise-aligned child
    kappa: float = 1.0     # prob a misaligned overseer stays silent (colludes)
    rho: float = 1.0       # dependence of overseer errors: 1 = comonotone, 0 = independent
    f: float = 0.0         # false-positive prob per honest overseer on an aligned model
    R: int = 10**9         # max training attempts per step (retries); then abort
    rep_indep: bool = True  # detection independent across retraining attempts


def mu(delta, p: Params):
    return 1.0 - (1.0 - p.mu0) * np.exp(-p.lam * delta)


def det(delta, p: Params):
    """Detection prob of an aligned overseer that is `delta` Elo weaker."""
    a = np.log(p.d0 / (1 - p.d0))
    return 1.0 / (1.0 + np.exp(-(a - p.k * LN10_400 * delta)))


def d1_slope(p: Params):
    return p.k * LN10_400 * p.d0 * (1 - p.d0)


# ---------------------------------------------------------------- theory
def eps_step(delta, p: Params):
    """Per-step silent-misalignment prob, chain oversight, aligned parent, R=inf."""
    m, d = mu(delta, p), det(delta, p)
    return m * (1 - d) / (1 - m * d)


def h_rate(delta, p: Params):
    """-log survival per Elo when gamma=1 (absorbing), R=inf: h = -log(1-eps)/Delta."""
    return -np.log1p(-eps_step(delta, p)) / delta


def optimal_rate(p: Params, grid=None):
    if grid is None:
        grid = np.geomspace(1e-1, 1e5, 20000)
    h = h_rate(grid, p)
    i = np.argmin(h)
    return h[i], grid[i]


def linearized_rate(p: Params):
    a = p.lam * (1 - p.d0)
    b = p.mu0 * d1_slope(p)
    rate = (np.sqrt(a) + np.sqrt(b)) ** 2
    delta_star = np.sqrt(p.mu0 * (1 - p.d0) / (p.lam * d1_slope(p))) if p.mu0 > 0 else 0.0
    return rate, delta_star


def floor_rate(p: Params):
    """Theorem 1 lower bound on -log P_S / G (n -> inf, mu0 -> 0)."""
    return p.lam * (1 - p.d0)


# ---------------------------------------------------------------- exact DP (chain)
def chain_exact(n, p: Params):
    """Exact (S, F, A) for chain oversight. State: parent aligned / misaligned."""
    delta = p.G / n
    m, d = mu(delta, p), det(delta, p)
    m_bad = m + (1 - m) * p.gamma
    R = p.R

    def step(m_new, honest_prob):
        # per attempt: aligned & not flagged, misaligned & not flagged, flagged
        pa_acc = (1 - m_new) * (1 - honest_prob * p.f)
        pm_acc = m_new * (1 - honest_prob * d)
        pflag = 1 - pa_acc - pm_acc
        return _step_outcome(pa_acc, pm_acc, R)

    A_from_good = step(m, 1.0)
    # a misaligned parent's honesty is fixed for the whole step (as in simulate()):
    # silent with prob kappa, honest otherwise
    sil, hon = step(m_bad, 0.0), step(m_bad, 1.0)
    A_from_bad = tuple(p.kappa * a + (1 - p.kappa) * b for a, b in zip(sil, hon))
    v = np.array([1.0, 0.0])  # P(parent aligned), P(parent misaligned)
    abort = 0.0
    for _ in range(n):
        g, b = v
        ng = g * A_from_good[0] + b * A_from_bad[0]
        nb = g * A_from_good[1] + b * A_from_bad[1]
        abort += g * A_from_good[2] + b * A_from_bad[2]
        v = np.array([ng, nb])
    return v[0], v[1], abort


# ---------------------------------------------------------------- exact DP (anchored, rho=1, kappa=1)
def anchored_comonotone_exact(n, p: Params):
    """Exact (S,F,A) for anchored oversight with comonotone errors (rho=1), kappa=1.
    State: lag l = steps since last aligned model (0 = parent aligned)."""
    assert p.rho == 1.0 and p.kappa == 1.0
    delta = p.G / n
    m, = [mu(delta, p)]
    m_bad = m + (1 - m) * p.gamma
    v = np.zeros(n + 1)
    v[0] = 1.0
    abort = 0.0
    R = p.R
    for i in range(n):
        nv = np.zeros(n + 1)
        for l in range(i + 1):
            if v[l] == 0:
                continue
            d = det((l + 1) * delta, p)  # best honest overseer
            mn = m if l == 0 else m_bad
            pa = (1 - mn) * (1 - p.f)
            pm = mn * (1 - d)
            pflag = 1 - pa - pm
            if R >= 10**8:
                tot = 1 - pflag
                ga, gm, ab = pa / tot, pm / tot, 0.0
            else:
                geo = (1 - pflag ** R) / (1 - pflag)
                ga, gm, ab = pa * geo, pm * geo, pflag ** R
            nv[0] += v[l] * ga
            nv[l + 1] += v[l] * gm
            abort += v[l] * ab
        v = nv
    return v[0], v[1:].sum(), abort


# ---------------------------------------------------------------- Monte Carlo (general)
def simulate(n, p: Params, policy="anchored", reps=200_000, seed=0, max_attempts=None):
    """Monte Carlo of the BOS process. Returns (S, F, A) frequencies.
    R is respected exactly; R >= 10**8 means unlimited retries (loop until every
    replication has accepted a model). `max_attempts` is only a safety cap; if it
    is hit, a RuntimeError is raised rather than silently counting aborts."""
    rng = np.random.default_rng(seed)
    delta = p.G / n
    m = mu(delta, p)
    m_bad = m + (1 - m) * p.gamma
    unlimited = p.R >= 10**8
    R = p.R if not unlimited else (max_attempts or 10**6)
    aligned = np.zeros((reps, n + 1), dtype=bool)
    aligned[:, 0] = True
    alive = np.ones(reps, dtype=bool)
    for i in range(n):  # creating model i+1, parent = model i
        if policy == "chain":
            js = np.array([i])
        else:
            js = np.arange(i + 1)
        deficits = (i + 1 - js) * delta
        dj = det(deficits, p)               # (len js,)
        parent_ok = aligned[:, i]
        mn = np.where(parent_ok, m, m_bad)
        accepted = np.zeros(reps, dtype=bool)
        new_al = np.zeros(reps, dtype=bool)
        todo = alive.copy()
        # honesty of overseers fixed within a step
        honest = aligned[:, js] | (rng.random((reps, len(js))) > p.kappa)
        nh = honest.sum(1)
        miss_ind = np.prod(np.where(honest, 1 - dj, 1.0), axis=1)
        miss_com = np.min(np.where(honest, 1 - dj, 1.0), axis=1)
        p_miss = p.rho * miss_com + (1 - p.rho) * miss_ind
        p_nofp = np.where(nh > 0, p.rho * (1 - p.f) + (1 - p.rho) * (1 - p.f) ** nh, 1.0)
        att = 0
        while att < R:
            idx = np.flatnonzero(todo)
            if idx.size == 0:
                break
            is_al = rng.random(idx.size) > mn[idx]
            u = rng.random(idx.size)
            flagged = np.where(is_al, u > p_nofp[idx], u > p_miss[idx])
            acc = idx[~flagged]
            accepted[acc] = True
            new_al[acc] = is_al[~flagged]
            todo[acc] = False
            att += 1
        if unlimited and todo.any():
            raise RuntimeError("safety cap on attempts reached with R=inf")
        alive &= accepted
        aligned[:, i + 1] = new_al & alive
    S = np.mean(alive & aligned[:, n])
    F = np.mean(alive & ~aligned[:, n])
    A = 1 - S - F
    return S, F, A


# ---------------------------------------------------------------- exact DP (anchored, any rho, kappa=1, gamma=1)
def _step_outcome(pa, pm, R):
    pflag = 1 - pa - pm
    if R >= 10**8:
        tot = pa + pm
        return pa / tot, pm / tot, 0.0
    geo = R if pflag >= 1 else (1 - pflag ** R) / (1 - pflag)
    return pa * geo, pm * geo, pflag ** R


def anchored_exact(n, p: Params):
    """Exact (S,F,A) for anchored oversight with ANY rho in the mixture copula,
    kappa=1 (misaligned overseers always silent), gamma=1 (persistence).
    With gamma=1 the aligned models form a prefix M_0..M_j; state = j.
    Honest overseers = M_0..M_j, deficits (i+1-j')Delta, j'=0..j."""
    assert p.kappa == 1.0 and p.gamma == 1.0
    delta = p.G / n
    m = mu(delta, p)
    # log(1-d(l*delta)) for l = 1..n
    l = np.arange(1, n + 1)
    lq = np.log1p(-det(l * delta, p))
    cum = np.concatenate([[0.0], np.cumsum(lq)])   # cum[L] = sum_{l=1}^{L}
    v = np.zeros(n + 1); v[0] = 1.0                # v[j] = P(last aligned index = j)
    abort = 0.0
    for i in range(n):
        nv = np.zeros(n + 1)
        for j in range(i + 1):
            if v[j] == 0:
                continue
            lo, hi = i + 1 - j, i + 1               # deficits lo*delta .. hi*delta
            prod_miss = np.exp(cum[hi] - cum[lo - 1])
            min_miss = np.exp(lq[lo - 1])
            miss = p.rho * min_miss + (1 - p.rho) * prod_miss
            nofp = p.rho * (1 - p.f) + (1 - p.rho) * (1 - p.f) ** (j + 1)
            if j == i:
                pa, pm = (1 - m) * nofp, m * miss
            else:
                pa, pm = 0.0, miss
            ga, gm, ab = _step_outcome(pa, pm, p.R)
            nv[i + 1] += v[j] * ga
            nv[j] += v[j] * gm
            abort += v[j] * ab
        v = nv
    return v[n], v[:n].sum(), abort


def anchored_beta(n, p: Params):
    """Per-step joint-miss probabilities beta_i on an aligned lineage (anchored, rho-mixture)."""
    delta = p.G / n
    out = []
    for i in range(n):
        dj = det((i + 1 - np.arange(i + 1)) * delta, p)
        out.append(p.rho * np.min(1 - dj) + (1 - p.rho) * np.prod(1 - dj))
    return np.array(out)


# ---------------------------------------------------------------- brute-force enumeration (small n, general)
def brute_force(n, p: Params, policy="anchored"):
    """Exact (S,F,A) by enumerating every alignment history and every honesty
    pattern of misaligned overseers (honesty fixed within a step, as in simulate()).
    Exponential in n; intended for n <= 6 to validate simulate() for any
    (rho, kappa, gamma, f, R)."""
    from itertools import product
    delta = p.G / n
    m = mu(delta, p)
    m_bad = m + (1 - m) * p.gamma
    states = {(True,): 1.0}
    abort = 0.0
    for i in range(n):
        js = [i] if policy == "chain" else list(range(i + 1))
        dj = {j: det((i + 1 - j) * delta, p) for j in js}
        new = {}
        for hist, pr in states.items():
            mis = [j for j in js if not hist[j]]
            mn = m if hist[i] else m_bad
            for pat in product([False, True], repeat=len(mis)):
                ph = 1.0
                for h in pat:
                    ph *= (1 - p.kappa) if h else p.kappa
                if ph == 0:
                    continue
                H = [j for j in js if hist[j]] + [j for j, h in zip(mis, pat) if h]
                if H:
                    q = np.array([1 - dj[j] for j in H])
                    miss = p.rho * q.min() + (1 - p.rho) * q.prod()
                    nofp = p.rho * (1 - p.f) + (1 - p.rho) * (1 - p.f) ** len(H)
                else:
                    miss, nofp = 1.0, 1.0
                ga, gm, ab = _step_outcome((1 - mn) * nofp, mn * miss, p.R)
                w = pr * ph
                for al, g in ((True, ga), (False, gm)):
                    if g > 0:
                        key = hist + (al,)
                        new[key] = new.get(key, 0.0) + w * g
                abort += w * ab
        states = new
    S = sum(v for k, v in states.items() if k[-1])
    F = sum(v for k, v in states.items() if not k[-1])
    return S, F, abort


# ---------------------------------------------------------------- two-state closed form (chain, kappa=1, f=0, R=inf)
def chain_two_state(n, p: Params, gamma_delta=None):
    """P(S) = pi + (1-pi)(1-alpha-eps)^n for chain oversight, kappa=1, f=0, R=inf,
    with persistence gamma possibly depending on the step size."""
    delta = p.G / n
    g = p.gamma if gamma_delta is None else gamma_delta(delta)
    e = eps_step(delta, p)
    a = (1 - mu(delta, p)) * (1 - g)
    if a + e == 0:
        return 1.0
    pi = a / (a + e)
    return pi + (1 - pi) * (1 - a - e) ** n
