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
