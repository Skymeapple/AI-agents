"""Core model: redundancy vs. division of labor for multi-agent teams with
correlated errors (beta-binomial latent difficulty) and lossy hand-offs.

All quantities here are exact (beta-binomial sums / regularized incomplete
beta functions) except tuple-plurality, which is Monte Carlo.
"""
import numpy as np
from scipy import stats, special


def beta_params(p, rho):
    """Beta(a,b) with mean p and pairwise correctness correlation rho=1/(a+b+1)."""
    s = (1.0 - rho) / rho
    return p * s, (1.0 - p) * s


def vote_acc(n, p, rho):
    """Exact majority-vote accuracy for m=2, n agents, random tie-breaking.

    Agents are conditionally i.i.d. Bernoulli(theta) given item difficulty
    theta ~ Beta(mean p, ICC rho). rho=0 reduces to the binomial (Condorcet) case.
    """
    n = int(n)
    if n <= 0:
        return 0.5  # no information: coin flip between two answers
    k = np.arange(n + 1)
    if rho <= 0:
        pmf = stats.binom.pmf(k, n, p)
    else:
        a, b = beta_params(p, rho)
        pmf = stats.betabinom.pmf(k, n, a, b)
    acc = pmf[k > n / 2].sum()
    if n % 2 == 0:
        acc += 0.5 * pmf[k == n // 2].sum()
    return float(acc)


def ceiling(p, rho, m=2):
    """Infinite-team plurality accuracy: P(theta > 1/m) (Theorem 1)."""
    if rho <= 0:
        return 1.0 if p > 1.0 / m else (0.5 if p == 1.0 / m else 0.0)
    a, b = beta_params(p, rho)
    return float(special.betainc(a, b, 1.0) - special.betainc(a, b, 1.0 / m))


def acc_division(B, K, pS, rho, eps):
    """Division of labor: B/K specialist calls per subtask, per-subtask vote,
    K-1 lossy hand-offs each failing w.p. eps."""
    n = B // K
    return (1 - eps) ** (K - 1) * vote_acc(n, pS, rho) ** K


def acc_generalist_coord(B, K, pG, rho):
    """Redundancy: B/K whole-task generalist calls (cost K each), coordinate-wise vote."""
    n = B // K
    return vote_acc(n, pG, rho) ** K


def mc_team(n, K, p, rho, m, n_items, rng, agg="tuple"):
    """Monte Carlo accuracy of n whole-task agents with plurality over full K-tuples
    ('tuple') or coordinate-wise plurality ('coord'), m-ary answers, wrong answers
    uniform over m-1 options. Returns fraction of items fully correct."""
    if rho > 0:
        a, b = beta_params(p, rho)
        theta = rng.beta(a, b, size=(n_items, K))
    else:
        theta = np.full((n_items, K), p)
    correct = rng.random((n_items, n, K)) < theta[:, None, :]
    wrong = rng.integers(1, m, size=(n_items, n, K))  # answer 0 is correct
    ans = np.where(correct, 0, wrong)
    if agg == "coord":
        ok = np.ones(n_items, bool)
        for k in range(K):
            counts = np.stack([(ans[:, :, k] == v).sum(1) for v in range(m)], 1)
            noise = rng.random(counts.shape) * 1e-3  # random tie-break
            ok &= np.argmax(counts + noise, 1) == 0
        return ok.mean()
    # tuple plurality: encode tuple as integer
    code = np.zeros((n_items, n), dtype=np.int64)
    for k in range(K):
        code = code * m + ans[:, :, k]
    res = np.empty(n_items, bool)
    for i in range(n_items):
        vals, cnt = np.unique(code[i], return_counts=True)
        best = np.flatnonzero(cnt == cnt.max())
        pick = vals[rng.choice(best)]
        res[i] = pick == 0
    return res.mean()


def eps_star(pG, pS, rho, K, m=2):
    """Asymptotic critical hand-off error (Theorem 2)."""
    cG, cS = ceiling(pG, rho, m), ceiling(pS, rho, m)
    if cS <= 0:
        return 0.0
    r = cG / cS
    return max(0.0, 1.0 - r ** (K / (K - 1)))


def best_check_split(B, K, pS, rho, eps, d, step=1):
    """Divide budget B between voting calls and interface-check calls.
    c check units spread evenly over K-1 interfaces; each check independently
    catches+repairs a hand-off error w.p. d. Returns (c*, acc*, acc_nocheck)."""
    best = (0, acc_division(B, K, pS, rho, eps))
    base = best[1]
    for c in range(step, B - K + 1, step):
        eff = eps * (1 - d) ** (c / (K - 1))
        a = acc_division(B - c, K, pS, rho, eff)
        if a > best[1]:
            best = (c, a)
    return best[0], best[1], base
