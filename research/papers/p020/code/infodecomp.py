"""Authorship-information decomposition of population novelty.

Population novelty  H(X)  =  G  +  I(X;A)
    G      = sum_i w_i H(X | A=i)   (generative / within-agent entropy)
    I(X;A) = Jensen-Shannon divergence of the agents' idea distributions
             ("authorship information": how much knowing the author tells you
             about the idea).

All entropies in bits. Plug-in estimates are biased upward for I; we report the
permutation-corrected excess  I* = I_hat - E_perm[I_hat]  where the permutation
shuffles ideas across agents (within strata for conditional terms), which is the
exact null of an exchangeable generator with the same agent sizes.
"""
import numpy as np


def H_counts(c):
    c = np.asarray(c, dtype=float)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def row_entropies(M):
    """Entropy (bits) of each row of a count matrix."""
    M = np.asarray(M, dtype=float)
    n = M.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(M > 0, M / n, 1.0)
        h = -(np.where(M > 0, P * np.log2(P), 0.0)).sum(1)
    return h, n.ravel()


def plugin_mi(x, a, K, nA):
    """Plug-in I(X;A) for integer arrays x in [0,K), a in [0,nA)."""
    N = len(x)
    Hx = H_counts(np.bincount(x, minlength=K))
    M = np.bincount(a * K + x, minlength=nA * K).reshape(nA, K)
    h, n = row_entropies(M)
    Hxa = float((n * h).sum() / N)
    return Hx - Hxa, Hx, Hxa


def cond_mi(x, a, s, K):
    """Plug-in I(X;A|S): a must be nested in strata s (ints)."""
    N = len(x)
    # H(X|S)
    _, s_idx = np.unique(s, return_inverse=True)
    nS = s_idx.max() + 1
    MS = np.bincount(s_idx * K + x, minlength=nS * K).reshape(nS, K)
    hS, nS_ = row_entropies(MS)
    HxS = float((nS_ * hS).sum() / N)
    _, a_idx = np.unique(a, return_inverse=True)
    nA = a_idx.max() + 1
    MA = np.bincount(a_idx * K + x, minlength=nA * K).reshape(nA, K)
    hA, nA_ = row_entropies(MA)
    HxA = float((nA_ * hA).sum() / N)
    return HxS - HxA


def permute_within(x, s, rng):
    """Shuffle x within strata s."""
    if s is None:
        return rng.permutation(x)
    y = x.copy()
    order = np.argsort(s, kind="stable")
    ss = s[order]
    bounds = np.flatnonzero(np.diff(ss)) + 1
    for blk in np.split(order, bounds):
        y[blk] = x[rng.permutation(blk)]
    return y


def excess_mi(x, a, K, n_perm=500, rng=None, strata=None):
    """Return dict with plug-in I, null mean/sd, excess I*, H(X), p-value.

    If strata is given, computes I(X;A|S) with permutation within strata.
    """
    rng = np.random.default_rng(rng)
    _, a_idx = np.unique(a, return_inverse=True)
    nA = a_idx.max() + 1
    if strata is None:
        I, Hx, _ = plugin_mi(x, a_idx, K, nA)
        f = lambda y: plugin_mi(y, a_idx, K, nA)[0]
    else:
        I = cond_mi(x, a_idx, strata, K)
        Hx = H_counts(np.bincount(x, minlength=K))
        f = lambda y: cond_mi(y, a_idx, strata, K)
    null = np.array([f(permute_within(x, strata, rng)) for _ in range(n_perm)])
    return dict(I=I, null_mean=float(null.mean()), null_sd=float(null.std()),
                I_star=float(I - null.mean()), H=Hx,
                pi_star=float((I - null.mean()) / Hx) if Hx > 0 else np.nan,
                p=float((1 + (null >= I).sum()) / (1 + n_perm)))


def list_perm_mi(x, list_id, group_of_list, K, n_perm=500, rng=None, strata_of_list=None):
    """I(X; G) (or I(X; G | S)) where G is a property of whole lists.

    Null: permute group labels across *lists* (within strata of lists). This keeps
    every list intact, so it is the exact null of exchangeable list generators,
    valid under any within-list dependence (repetition avoidance, persistence).
    list_id: per-response integer list index 0..L-1
    group_of_list: length-L integer array; strata_of_list: length-L or None.
    """
    rng = np.random.default_rng(rng)
    g = np.asarray(group_of_list)
    s = None if strata_of_list is None else np.asarray(strata_of_list)

    def stat(gl):
        a = gl[list_id]
        if s is None:
            return plugin_mi(x, pd_factor(a), K, int(pd_factor(a).max() + 1))[0]
        return cond_mi(x, a, s[list_id], K)

    I = stat(g)
    null = np.array([stat(permute_within(g, s, rng)) for _ in range(n_perm)])
    Hx = H_counts(np.bincount(x, minlength=K))
    return dict(I=I, null_mean=float(null.mean()), null_sd=float(null.std()),
                I_star=float(I - null.mean()), H=Hx, pi_star=float((I - null.mean()) / Hx),
                p=float((1 + (null >= I).sum()) / (1 + n_perm)))


def pd_factor(a):
    _, idx = np.unique(a, return_inverse=True)
    return idx
