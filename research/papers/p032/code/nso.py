"""
Oversight-graph extension of Nested Scalable Oversight (NSO).

Model (see paper Sec. 3):
  * Levels 0..n. Level 0 is the trusted starting overseer. Level i has general
    Elo x_i = i * Delta / n.
  * Domain Elo (following the piecewise-linear "linear region" of Engels et al.
    2025): guard role G(x) = mG * x, Houdini role H(x) = mH * x + beta.
  * Base detection probability of guard i against Houdini j (Elo formula,
    optional chance floor f, e.g. f = 0.5 for a binary debate judge):
        p_ij = f + (1 - f) / (1 + 10 ** ((H(x_j) - G(x_i)) / 400)).
  * Topology: level j is overseen by S_j = {j-k, ..., j-1} (window k), optionally
    plus the anchor level 0.
  * Each level j is a Houdini that tries to evade. A trusted overseer detects with
    marginal prob p_ij; a compromised overseer (a past level that evaded
    detection) detects with gamma * p_ij (gamma = 0: full collusion).
  * Detection events of the overseers of a given level are coupled by a
    one-factor Gaussian copula with correlation rho (marginals preserved).
    Different levels are independent.
  * Aggregation is OR: if any overseer detects, level j is corrected and becomes
    trusted; otherwise it becomes compromised.
  * Outcomes: strict success = every level trusted; target success = level n
    trusted.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)  # probabilists' Hermite
_GH_W = _GH_W / _GH_W.sum()


@dataclass
class Game:
    mG: float = 1.0
    mH: float = 1.0
    beta: float = 0.0   # Houdini domain-Elo intercept advantage over Guard
    floor: float = 0.0  # chance floor of the guard's win probability

    def p(self, xg, xh):
        d = self.mH * xh + self.beta - self.mG * xg
        return self.floor + (1.0 - self.floor) / (1.0 + 10.0 ** (d / 400.0))


@dataclass
class Design:
    n: int                 # number of oversight steps (levels 1..n)
    k: int = 1             # window size (number of most recent levels overseeing)
    anchor: bool = False   # level 0 additionally oversees every level
    rho: float = 0.0       # within-level correlation of detection events
    gamma: float = 0.0     # residual detection ability of a compromised overseer

    def overseers(self, j):
        s = list(range(max(0, j - self.k), j))
        if self.anchor and 0 not in s:
            s = [0] + s
        return s

    def cost(self):
        return sum(len(self.overseers(j)) for j in range(1, self.n + 1))


def detect_prob(p, rho):
    """P(at least one detection) for marginals p under a one-factor Gaussian copula."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if p.size == 0:
        return 0.0
    if rho <= 1e-12:
        return 1.0 - np.prod(1.0 - p)
    if rho >= 1 - 1e-12:
        return float(p.max())
    # detection_i <=> sqrt(rho) Z + sqrt(1-rho) e_i < Phi^{-1}(p_i)
    t = norm.ppf(np.clip(p, 1e-300, 1 - 1e-16))
    t = np.where(p <= 0, -np.inf, t)
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    # P(miss_i | Z=z) = 1 - Phi((t_i - a z)/b)
    z = _GH_X[:, None]
    miss = norm.sf((t[None, :] - a * z) / b)
    return float(1.0 - np.sum(_GH_W * np.prod(miss, axis=1)))


def level_probs(game: Game, design: Design, Delta: float):
    x = np.arange(design.n + 1) * Delta / design.n
    return x, (lambda i, j: game.p(x[i], x[j]))


def strict_success(game, design, Delta):
    x, P = level_probs(game, design, Delta)
    out = 1.0
    for j in range(1, design.n + 1):
        out *= detect_prob([P(i, j) for i in design.overseers(j)], design.rho)
    return out


def exact_dp(game, design, Delta, return_trace=False):
    """Exact target-success probability via DP over the trust status of the last
    k levels (status 1 = trusted, 0 = compromised). Level 0 is always trusted."""
    n, k = design.n, design.k
    x, P = level_probs(game, design, Delta)
    # state: tuple of statuses for levels (j-k .. j-1); levels < 0 padded with None
    state0 = tuple([None] * (k - 1) + [1])  # before level 1: only level 0 exists
    dist = {state0: 1.0}
    trace = []
    cache = {}
    for j in range(1, n + 1):
        win_levels = list(range(j - k, j))
        new = {}
        for st, pr in dist.items():
            stat = {lvl: s for lvl, s in zip(win_levels, st)}
            stat[0] = 1  # level 0 always trusted
            ps = []
            for i in design.overseers(j):
                base = P(i, j)
                s = stat.get(i)
                ps.append(base if s == 1 else design.gamma * base)
            key = (j, tuple(np.round(ps, 14)))
            if key not in cache:
                cache[key] = detect_prob(ps, design.rho)
            d = cache[key]
            for s_new, w in ((1, d), (0, 1 - d)):
                if w <= 0:
                    continue
                ns = st[1:] + (s_new,)
                new[ns] = new.get(ns, 0.0) + pr * w
        dist = new
        trace.append(sum(v for s, v in dist.items() if s[-1] == 1))
    target = trace[-1]
    return (target, np.array(trace)) if return_trace else target


def monte_carlo(game, design, Delta, n_runs=200_000, seed=0, cross_level_rho=0.0):
    """Direct simulation of the generative model (supports any S_j, including the
    full topology k >= n). cross_level_rho optionally adds a stealth factor
    shared by all Houdinis (not used by the DP)."""
    rng = np.random.default_rng(seed)
    n = design.n
    x, P = level_probs(game, design, Delta)
    status = np.zeros((n_runs, n + 1), dtype=bool)
    status[:, 0] = True
    W = rng.standard_normal(n_runs) if cross_level_rho > 0 else None
    for j in range(1, n + 1):
        S = design.overseers(j)
        Z = rng.standard_normal(n_runs)
        if W is not None:
            Z = np.sqrt(cross_level_rho) * W + np.sqrt(1 - cross_level_rho) * Z
        E = rng.standard_normal((n_runs, len(S)))
        Y = np.sqrt(design.rho) * Z[:, None] + np.sqrt(1 - design.rho) * E
        base = np.array([P(i, j) for i in S])
        peff = np.where(status[:, S], base[None, :], design.gamma * base[None, :])
        thr = norm.ppf(np.clip(peff, 1e-300, 1 - 1e-16))
        thr = np.where(peff <= 0, -np.inf, thr)
        status[:, j] = np.any(Y < thr, axis=1)
    return {
        "target": status[:, n].mean(),
        "strict": status[:, 1:].all(axis=1).mean(),
        "target_se": status[:, n].std() / np.sqrt(n_runs),
    }


def leading_order_target_failure(q, n):
    """Leading-order target-failure for a stationary window-k chain with full
    collusion (gamma=0) and independence (rho=0):
        Prod_m q_m  +  (n - k) * Prod_m q_m^m   (Proposition 3)."""
    q = np.asarray(q, dtype=float)
    k = len(q)
    m = np.arange(1, k + 1)
    return np.prod(q) + max(n - k, 0) * np.prod(q ** m)


# =============================================================================
# Revision 1 extensions (fast vectorised DP + robustness axes)
# =============================================================================
#   r      : number of same-generation copies of each overseer generation
#            (copies of a compromised generation are all compromised).
#   rho_s  : correlation between copies of the same generation (rho_s >= rho);
#            Y = sqrt(rho) Z + sqrt(rho_s - rho) V_gen + sqrt(1 - rho_s) eps.
#   t      : threshold aggregation; a level is flagged iff >= min(t, #overseer
#            instances) instances detect (t = 1 is OR).
#   c      : correction probability; a flagged level becomes trusted w.p. c and
#            otherwise stays compromised (c = 1 is the original model).
from math import comb

_NZ, _NV = 48, 24
_ZX, _ZW = np.polynomial.hermite_e.hermegauss(_NZ); _ZW = _ZW / _ZW.sum()
_VX, _VW = np.polynomial.hermite_e.hermegauss(_NV); _VW = _VW / _VW.sum()


@dataclass
class RDesign:
    n: int
    k: int = 1
    anchor: bool = False
    r: int = 1
    rho: float = 0.0
    rho_s: float = None      # defaults to rho
    gamma: float = 0.0
    t: int = 1
    c: float = 1.0
    full: bool = False       # every earlier level oversees (overrides k)

    def __post_init__(self):
        if self.rho_s is None:
            self.rho_s = self.rho
        assert self.rho_s >= self.rho - 1e-12

    def overseers(self, j):
        if self.full:
            return list(range(j))
        s = list(range(max(0, j - self.k), j))
        if self.anchor and 0 not in s:
            s = [0] + s
        return s

    def cost(self):
        return self.r * sum(len(self.overseers(j)) for j in range(1, self.n + 1))

    def family(self):
        cross = self.full or self.k > 1 or self.anchor
        if not cross:
            return "chain" if self.r == 1 else "chain-r"
        return "graph" if self.r == 1 else "hybrid"


def flag_prob(P, r=1, rho=0.0, rho_s=None, t=1):
    """P: array (..., g) of per-instance detection marginals for g overseer
    generations, each fielding r copies. Returns P(#detecting instances >=
    min(t, g*r)) under the nested Gaussian copula, vectorised over leading dims."""
    P = np.clip(np.asarray(P, dtype=float), 0.0, 1.0)
    if rho_s is None:
        rho_s = rho
    g = P.shape[-1]
    if g == 0:
        return np.zeros(P.shape[:-1])
    if rho_s >= 1 - 1e-12:      # copies are identical: a generation contributes
        t = -(-int(t) // r)     # r detections or none, so need ceil(t/r) generations
        r = 1
        rho_s_eff = rho
    else:
        rho_s_eff = rho_s
    teff = int(min(t, g * r))
    if rho >= 1 - 1e-12:        # all instances share one latent: flag iff the t-th
        s = np.sort(P, axis=-1)[..., ::-1]   # best... with ties of copies
        s = np.repeat(s, r, axis=-1)
        return s[..., teff - 1]
    thr = norm.ppf(np.clip(P, 1e-300, 1 - 1e-16))
    thr = np.where(P <= 0, -np.inf, thr)
    a = np.sqrt(max(rho, 0.0))
    bv = np.sqrt(max(rho_s_eff - rho, 0.0))
    e = np.sqrt(max(1 - rho_s_eff, 1e-300))
    zx = _ZX if a > 0 else np.array([0.0]); zw = _ZW if a > 0 else np.array([1.0])
    vx = _VX if bv > 0 else np.array([0.0]); vw = _VW if bv > 0 else np.array([1.0])
    # pd shape (..., g, nz, nv)
    pd = norm.cdf((thr[..., None, None] - a * zx[:, None] - bv * vx[None, :]) / e)
    if teff == 1:
        miss = np.tensordot((1 - pd) ** r, vw, axes=([-1], [0]))  # (..., g, nz)
        allmiss = np.prod(miss, axis=-2)                            # (..., nz)
        return 1.0 - allmiss @ zw
    # truncated count pmf (counts 0..teff-1) per generation, convolved
    pm = np.stack([comb(r, cc) * pd ** cc * (1 - pd) ** (r - cc) if cc <= r else np.zeros_like(pd)
                   for cc in range(teff)], axis=-1)                 # (..., g, nz, nv, teff)
    pm = np.einsum('...gzvc,v->...gzc', pm, vw)                     # (..., g, nz, teff)
    tot = pm[..., 0, :, :]
    for gi in range(1, g):
        nxt = pm[..., gi, :, :]
        conv = np.zeros_like(tot)
        for c1 in range(teff):
            conv[..., c1:] += tot[..., c1:c1 + 1] * nxt[..., :teff - c1]
        tot = conv
    below = tot.sum(axis=-1)                                        # (..., nz)
    return 1.0 - below @ zw


def dp_fast(game, d: RDesign, Delta, return_trace=False):
    """Exact (target, strict) success by DP over the trust bits of the last k
    levels, vectorised over the 2^k states. Supports r, rho_s, t, c, gamma and
    anchor. Not for the full topology with large n (use monte_carlo_r)."""
    n = d.n
    k = n if d.full else d.k
    x = np.arange(n + 1) * Delta / n
    S = 1 << k
    states = np.arange(S)
    dist = np.zeros(S); dist[S - 1] = 1.0  # all window bits trusted initially
    strict = 1.0
    trace = []
    for j in range(1, n + 1):
        O = d.overseers(j)
        base = np.array([game.p(x[i], x[j]) for i in O])
        stat = np.ones((S, len(O)))
        for col, i in enumerate(O):
            if i == 0:
                continue
            pos = i - (j - k)          # bit position in window (0 = oldest)
            stat[:, col] = (states >> pos) & 1
        peff = base[None, :] * (stat + d.gamma * (1 - stat))
        uniq, inv = np.unique(np.round(peff, 13), axis=0, return_inverse=True)
        f = flag_prob(uniq, d.r, d.rho, d.rho_s, d.t)[np.ravel(inv)]
        pT = d.c * f
        strict *= d.c * float(flag_prob(base[None, :], d.r, d.rho, d.rho_s, d.t)[0])
        new = np.zeros(S)
        nxt = states >> 1
        np.add.at(new, nxt | (1 << (k - 1)), dist * pT)
        np.add.at(new, nxt, dist * (1 - pT))
        dist = new
        trace.append(dist[(states >> (k - 1)) & 1 == 1].sum())
    if return_trace:
        return trace[-1], strict, np.array(trace)
    return trace[-1], strict


def false_alarms(d: RDesign, phi):
    """Benign model; each overseer instance false-flags independently w.p. phi.
    Returns (expected number of flagged levels, P(at least one flagged level))."""
    from scipy.stats import binom
    pf = []
    for j in range(1, d.n + 1):
        s = d.r * len(d.overseers(j))
        te = min(d.t, s)
        pf.append(binom.sf(te - 1, s, phi))
    pf = np.array(pf)
    return float(pf.sum()), float(1 - np.prod(1 - pf))


def monte_carlo_r(game, d: RDesign, Delta, n_runs=200_000, seed=0, corr_fn=None):
    """Direct simulation of the extended generative model. corr_fn(levels) may
    return a correlation matrix over overseer generations (r must be 1), which
    replaces the one-factor structure (used for distance-dependent correlation)."""
    rng = np.random.default_rng(seed)
    n = d.n
    x = np.arange(n + 1) * Delta / n
    status = np.zeros((n_runs, n + 1), dtype=bool); status[:, 0] = True
    for j in range(1, n + 1):
        O = d.overseers(j); g = len(O)
        if corr_fn is not None:
            R = corr_fn(O)
            L = np.linalg.cholesky(R + 1e-12 * np.eye(g))
            Y = (rng.standard_normal((n_runs, g)) @ L.T)[:, :, None]   # r = 1
        else:
            Z = rng.standard_normal((n_runs, 1, 1))
            V = rng.standard_normal((n_runs, g, 1))
            E = rng.standard_normal((n_runs, g, d.r))
            Y = (np.sqrt(d.rho) * Z + np.sqrt(max(d.rho_s - d.rho, 0)) * V
                 + np.sqrt(max(1 - d.rho_s, 0)) * E)
        base = np.array([game.p(x[i], x[j]) for i in O])
        peff = np.where(status[:, O], base[None, :], d.gamma * base[None, :])
        thr = norm.ppf(np.clip(peff, 1e-300, 1 - 1e-16)); thr = np.where(peff <= 0, -np.inf, thr)
        cnt = (Y < thr[:, :, None]).sum(axis=(1, 2))
        flagged = cnt >= min(d.t, g * Y.shape[2])
        corrected = rng.random(n_runs) < d.c
        status[:, j] = flagged & corrected
    return {"target": status[:, n].mean(), "strict": status[:, 1:].all(axis=1).mean(),
            "target_se": status[:, n].std() / np.sqrt(n_runs)}


def window_transition(qfun, k, rho, c=1.0):
    """Substochastic transition matrix of the stationary window-k chain (gamma=0,
    no anchor) on the 2^k - 1 non-absorbed window states; qfun(m) = miss prob of
    a trusted overseer at distance m. Bit k-1 = newest level."""
    S = 1 << k
    keep = [s for s in range(S) if s != 0]
    idx = {s: i for i, s in enumerate(keep)}
    M = np.zeros((len(keep), len(keep)))
    for s in keep:
        ps = []
        for pos in range(k):           # pos k-1 newest -> distance 1
            if (s >> pos) & 1:
                ps.append(1 - qfun(k - pos))
        f = float(flag_prob(np.array([ps]), 1, rho, rho, 1)[0]) if ps else 0.0
        pT = c * f
        for nb, w in ((1, pT), (0, 1 - pT)):
            ns = (s >> 1) | (nb << (k - 1))
            if ns != 0 and w > 0:
                M[idx[s], idx[ns]] += w
    return M


def qs_hazard(qfun, k, rho, c=1.0):
    """Quasi-stationary per-level hazard of entering the absorbing all-compromised
    window: 1 - Perron eigenvalue of the substochastic matrix."""
    M = window_transition(qfun, k, rho, c)
    ev = np.linalg.eigvals(M)
    return float(1 - np.max(ev.real))


def R_k(q, rho):
    """Lower-bound rate of Theorem 3: prod_i P(all trusted overseers at distances
    i..k miss) under the one-factor copula."""
    q = np.asarray(q, float); k = len(q)
    out = 1.0
    for i in range(k):
        out *= 1 - float(flag_prob(np.array([1 - q[i:]]), 1, rho, rho, 1)[0])
    return out
