"""Core model: liability apportionment under noisy failure attribution.

Notation (matches paper.md, Section 3)
  n parties. An incident has exactly one origin j (rare-event model).
  Party i chooses care x_i >= 0 at unit cost; origin hazard h_i(x) = b_i exp(-x/tau_i).
  A forensic process emits a signal a in {1..n} with confusion matrix
      C[a, j] = P(signal = a | origin = j)          (columns sum to 1).
  An apportionment rule is S[i, a] = share of the loss L paid by party i when the
  signal is a.  Budget balance: columns of S sum to 1.  No rebates: S >= 0.
  M = S @ C ;  M[i, j] = expected share of i in losses that originate at j.
  theta_i = M[i, i] is party i's internalisation rate.
"""
import itertools
import numpy as np
from scipy.optimize import minimize, linprog


# ---------------------------------------------------------------- channels
def symmetric_C(n, q):
    """Symmetric channel: correct with prob q, else uniform over the others."""
    r = (1.0 - q) / (n - 1)
    return (q - r) * np.eye(n) + r * np.ones((n, n))


def budget(C):
    """Incentive budget B(C) = sum_a max_j C[a, j]  (Theorem 1)."""
    return float(np.max(C, axis=1).sum())


def theta_of(S, C):
    return np.diag(S @ C).copy()


def max_weighted_theta(C, w, cap=None):
    """LP: max sum_i w_i theta_i over column-stochastic S >= 0 (S[i,a] <= cap_i)."""
    n = C.shape[0]
    # variable S[i,a] flattened row-major; theta_i = sum_a S[i,a] C[a,i]
    c = np.zeros(n * n)
    for i in range(n):
        for a in range(n):
            c[i * n + a] = -w[i] * C[a, i]
    A_eq = np.zeros((n, n * n))
    for a in range(n):
        for i in range(n):
            A_eq[a, i * n + a] = 1.0
    bounds = [(0.0, 1.0 if cap is None else min(1.0, cap[i]))
              for i in range(n) for _ in range(n)]
    res = linprog(c, A_eq=A_eq, b_eq=np.ones(n), bounds=bounds, method="highs")
    return -res.fun, res.x.reshape(n, n)


# ---------------------------------------------------------------- care technology
def care(theta, L, b, tau):
    """Privately optimal care when a party internalises theta*L per own-origin loss."""
    theta = np.asarray(theta, float)
    z = np.maximum(theta * L * b / tau, 1.0)
    return tau * np.log(z)


def hazard(x, b, tau):
    return b * np.exp(-x / tau)


def social_cost(theta, L, b, tau):
    x = care(theta, L, b, tau)
    return x + L * hazard(x, b, tau)


def dwl(theta, L, b, tau):
    """Per-party deadweight loss relative to first best (theta = 1)."""
    return social_cost(theta, L, b, tau) - social_cost(np.ones_like(b), L, b, tau)


def dwl_closed_form(theta, L, b, tau):
    """Proposition 3: tau*(1/theta - 1 + ln theta) in the interior, corner otherwise."""
    theta = np.asarray(theta, float)
    rho = L * b / tau
    interior = tau * (1.0 / np.maximum(theta, 1e-300) - 1.0 + np.log(np.maximum(theta, 1e-300)))
    corner = tau * (rho - 1.0 - np.log(rho))
    return np.where(theta * rho > 1.0, interior, corner)


# ---------------------------------------------------------------- rules
def rule_channel(n, i):
    S = np.zeros((n, n)); S[i, :] = 1.0
    return S


def rule_origin(n):
    """The party named by the forensic signal pays the whole loss."""
    return np.eye(n)


def rule_equal(n):
    return np.ones((n, n)) / n


def rule_debiased(C):
    """Theorem 2: S = C^{-1}. Ex-post budget balanced; needs rebates (negative entries)."""
    return np.linalg.inv(C)


def optimal_rule(C, L, b, tau, cap=None, n_starts=40, seed=0):
    """Minimise total DWL over no-rebate budget-balanced rules (optionally asset-capped).

    The objective is not globally convex (corner regions), so we use multi-start
    SLSQP seeded with all pure-channelling vertices, the origin rule and random points.
    Returns (S, total_dwl).
    """
    n = C.shape[0]
    rng = np.random.default_rng(seed)
    ub = np.ones(n) if cap is None else np.minimum(1.0, np.asarray(cap, float))
    if ub.sum() < 1.0 - 1e-12:
        raise ValueError("caps make full compensation infeasible")

    def obj(s):
        S = s.reshape(n, n)
        return float(dwl_closed_form(theta_of(S, C), L, b, tau).sum())

    cons = [{"type": "eq", "fun": (lambda s, a=a: s.reshape(n, n)[:, a].sum() - 1.0)}
            for a in range(n)]
    bounds = [(0.0, ub[i]) for i in range(n) for _ in range(n)]

    def project(S):
        S = np.clip(S, 0, ub[:, None])
        for a in range(n):          # simple feasibility repair
            for _ in range(50):
                gap = 1.0 - S[:, a].sum()
                if abs(gap) < 1e-12:
                    break
                room = (ub - S[:, a]) if gap > 0 else S[:, a]
                if room.sum() <= 0:
                    break
                S[:, a] += gap * room / room.sum()
                S[:, a] = np.clip(S[:, a], 0, ub)
        return S

    starts = [rule_origin(n), rule_equal(n)]
    starts += [rule_channel(n, i) for i in range(n)]
    # every deterministic assignment signal -> payer
    for assign in itertools.product(range(n), repeat=n):
        S = np.zeros((n, n))
        for a, i in enumerate(assign):
            S[i, a] = 1.0
        starts.append(S)
    for _ in range(n_starts):
        starts.append(rng.dirichlet(np.ones(n), size=n).T)

    best = (None, np.inf)
    for S0 in starts:
        S0 = project(S0.copy())
        f0 = obj(S0.ravel())
        if f0 < best[1]:
            best = (S0, f0)
        res = minimize(obj, S0.ravel(), method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12})
        S1 = project(res.x.reshape(n, n))
        f1 = obj(S1.ravel())
        if f1 < best[1]:
            best = (S1, f1)
    return best
