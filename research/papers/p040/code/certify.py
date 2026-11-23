"""Statistical certification of public-input constitutions.

Given a vote matrix V (participants x statements; 1 agree, -1 disagree, 0 pass,
NaN unseen) and a per-statement score (agreement rate, or min-group agreement for
a fixed partition), we ask which statements the votes *statistically determine*
to be in (or out of) the top-k, accounting for the fact that a different draw of
participants would give different estimates.

Inference for ranks follows the pairwise-difference construction of Mogstad,
Romano, Shaikh & Wilhelm (2024, Review of Economic Studies): simultaneous
bootstrap confidence intervals for all pairwise score differences theta_s - theta_j
(max-|t| studentised over pairs), then
   rank lower bound of s = 1 + #{j : diff_js significantly > 0}
   rank upper bound of s = M - #{j : diff_sj significantly > 0}.
s is *certified in* the top-k if its rank upper bound <= k, *certified out* if its
rank lower bound > k, and *uncertified* otherwise.  "simultaneous" controls the
familywise error over all pairs (the certified set is jointly valid at 1-alpha);
"marginal" controls it only over the pairs that involve s (per-statement validity).
Participants are the resampling unit (their votes are dependent).
"""
import numpy as np


def rates(V, est='seen'):
    a = (V == 1).sum(0).astype(float)
    d = (V == -1).sum(0).astype(float)
    s = (~np.isnan(V)).sum(0).astype(float)
    if est == 'seen':
        den = s
    elif est == 'decided':
        den = a + d
    else:
        raise ValueError(est)
    return np.divide(a, den, out=np.full_like(a, np.nan), where=den > 0)


def make_score(kind='av', est='seen', groups=None):
    """Return f(V, idx) -> per-statement score computed on rows idx of V."""
    if kind == 'av':
        return lambda V, idx: rates(V[idx], est)
    if kind == 'consmin':
        assert groups is not None
        G = np.unique(groups)
        def f(V, idx):
            g = groups[idx]
            Vi = V[idx]
            return np.vstack([rates(Vi[g == h], est) for h in G]).min(0)
        return f
    raise ValueError(kind)


def bootstrap_scores(V, score, B, rng):
    n = V.shape[0]
    full = score(V, np.arange(n))
    T = np.empty((B, V.shape[1]))
    for b in range(B):
        T[b] = score(V, rng.integers(0, n, n))
    # a statement unseen in a bootstrap draw gets its full-sample value (rare)
    bad = np.isnan(T)
    if bad.any():
        T[bad] = np.broadcast_to(full, T.shape)[bad]
    return np.nan_to_num(full, nan=0.0), T


def rank_bounds(full, T, alpha=0.05, mode='simultaneous'):
    """Pairwise-difference rank confidence bounds (Mogstad et al. 2024)."""
    M = len(full)
    D = full[:, None] - full[None, :]                    # D[s,j] = th_s - th_j
    # bootstrap SEs of differences, computed blockwise to save memory
    B = T.shape[0]
    mu = T.mean(0)
    Tc = T - mu
    cov = Tc.T @ Tc / (B - 1)
    var = np.diag(cov)
    se = np.sqrt(np.maximum(var[:, None] + var[None, :] - 2 * cov, 1e-12))
    np.fill_diagonal(se, np.inf)
    # bootstrap distribution of max |t| over pairs (centred at the full estimate)
    if mode == 'simultaneous':
        mx = np.empty(B)
        for b in range(B):
            d = T[b] - full
            Z = np.abs(d[:, None] - d[None, :]) / se
            mx[b] = Z.max()
        c = np.quantile(mx, 1 - alpha)
        crit = np.full(M, c)
    elif mode == 'marginal':   # max over pairs involving s, per s
        mxs = np.empty((B, M))
        for b in range(B):
            d = T[b] - full
            Z = np.abs(d[:, None] - d[None, :]) / se
            mxs[b] = Z.max(1)
        crit = np.quantile(mxs, 1 - alpha, axis=0)
    else:
        raise ValueError(mode)
    lo_diff = D - crit[:, None] * se        # lower bound on th_s - th_j
    sig_above = lo_diff > 0                 # s significantly above j
    np.fill_diagonal(sig_above, False)
    if mode == 'marginal':
        # for the rank of s we need pairs (s, j) at s's critical value
        n_below = sig_above.sum(1)          # j significantly below s
        n_above = (D.T - crit[:, None] * se.T > 0)   # j significantly above s, at s's crit
        np.fill_diagonal(n_above, False)
        n_above = n_above.sum(1)
    else:
        n_below = sig_above.sum(1)
        n_above = sig_above.sum(0)
    rlo = 1 + n_above
    rhi = M - n_below
    return rlo, rhi, crit


def certify(V, score, k, B=1000, alpha=0.05, rng=None, mode='simultaneous', T=None, full=None):
    rng = rng or np.random.default_rng(0)
    if T is None:
        full, T = bootstrap_scores(V, score, B, rng)
    rlo, rhi, crit = rank_bounds(full, T, alpha, mode)
    cin = rhi <= k
    cout = rlo > k
    # bootstrap stability: share of draws in which s is in the top-k
    order = np.argsort(-T, axis=1, kind='stable')[:, :k]
    pi = np.zeros(T.shape[1]); np.add.at(pi, order.ravel(), 1); pi /= T.shape[0]
    return dict(full=full, T=T, rlo=rlo, rhi=rhi, crit=crit, cin=cin, cout=cout,
                unc=~(cin | cout), pi=pi)


def certify_threshold(full, T, tau, alpha=0.05):
    """Simultaneous (max-|t| over statements) bootstrap CIs; certified above tau
    if the lower bound exceeds tau, certified below if the upper bound is below."""
    se = T.std(0, ddof=1) + 1e-12
    mx = (np.abs(T - full) / se).max(1)
    c = np.quantile(mx, 1 - alpha)
    return dict(above=(full - c * se) > tau, below=(full + c * se) < tau, crit=c)
