"""Monte Carlo simulator of an agent building a project whose modules form a DAG.

Process for each module i (in a topological order):
  1. Build: attempts until the module-level verifier accepts (max K attempts).
     Each attempt is correct w.p. p_eff, where p_eff = p*(1-kappa) if some
     dependency of i currently carries an *unexposed* latent defect
     (contamination), else p.  A defective attempt is in the verifier blind spot
     w.p. b; otherwise the verifier detects it w.p. c_i.
  2. Integration exposure: for every dependency u of i that is latently
     defective, i's tests expose u's defect w.p. delta.  The agent then repairs
     u (a fresh build loop for u) and pays rework rho * (#already-built
     descendants of u, including i).
Final acceptance: each remaining latent defect is detected by end-to-end
acceptance tests w.p. a_acc and repaired (rework rho*|desc|).
Project success = no latent defect remains.
Cost = attempts + rework, reported per module.
"""
import bisect
import numpy as np
import networkx as nx


class DagProject:
    def __init__(self, D):
        order = list(nx.topological_sort(D))
        idx = {v: k for k, v in enumerate(order)}
        self.N = len(order)
        self.parents = [[idx[u] for u in D.predecessors(v)] for v in order]
        self.desc = [sorted(idx[w] for w in nx.descendants(D, v)) for v in order]
        self.reach = np.array([len(d) for d in self.desc])
        self.outdeg = np.array([D.out_degree(v) for v in order])
        self.depth = np.zeros(self.N, int)
        for k in range(self.N):
            for u in self.parents[k]:
                self.depth[k] = max(self.depth[k], self.depth[u] + 1)


def _build(rng, p_eff, b, c, K):
    """Returns (attempts, defective)."""
    for t in range(1, K + 1):
        if rng.random() < p_eff:
            return t, False
        if rng.random() < b or rng.random() >= c:  # escapes verifier
            return t, True
    return K, True  # budget exhausted: ship last attempt (defective)


def run_once(P, rng, p, b, cvec, delta, kappa, rho, a_acc, K=10):
    N = P.N
    defect = np.zeros(N, bool)
    cost = 0.0
    for i in range(N):
        contaminated = any(defect[u] for u in P.parents[i])
        t, d = _build(rng, p * (1 - kappa) if contaminated else p, b, cvec[i], K)
        cost += t; defect[i] = d
        for u in P.parents[i]:
            if defect[u] and rng.random() < delta:
                t, d = _build(rng, p, b, cvec[u], K)
                cost += t; defect[u] = d
                cost += rho * bisect.bisect_right(P.desc[u], i)
    # final acceptance round
    for u in np.where(defect)[0]:
        if rng.random() < a_acc:
            t, d = _build(rng, p, b, cvec[u], K)
            cost += t + rho * len(P.desc[u]); defect[u] = d
    return (not defect.any()), cost / N, defect.sum()


def simulate(P, R, seed, **kw):
    rng = np.random.default_rng(seed)
    S = np.empty(R, bool); C = np.empty(R); L = np.empty(R)
    for r in range(R):
        S[r], C[r], L[r] = run_once(P, rng, **kw)
    return S, C, L


def waterfill(w, B, beta, b, emax=30.0):
    """min sum_i w_i (1-b) exp(-beta e_i)  s.t. sum e_i = B, 0<=e_i<=emax.
    KKT: e_i = clip(log(beta (1-b) w_i / lam)/beta, 0, emax).  Bisection on lam."""
    w = np.maximum(np.asarray(w, float), 1e-300)
    lo, hi = -800.0, 800.0  # log lam
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        e = np.clip((np.log(beta * (1 - b) * w) - mid) / beta, 0, emax)
        if e.sum() > B:
            lo = mid
        else:
            hi = mid
    return np.clip((np.log(beta * (1 - b) * w) - hi) / beta, 0, emax)
