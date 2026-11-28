"""Core estimators for separating persistent evaluator heterogeneity (pluralism)
from transient noise in subjective evaluations, and for certifying how
population-relative a system ranking is.

Data format used throughout: a pandas DataFrame with columns
    item   : item id (one output being judged)
    sys    : system / condition id that produced the item
    rater  : rater id (linked across items)
    y      : numeric score
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize


# ---------------------------------------------------------------- agreement
def krippendorff_alpha(df, metric="interval"):
    """Krippendorff's alpha from per-item multisets of labels (rater ids ignored).
    metric: 'interval' (squared difference) or 'nominal'."""
    groups = [np.asarray(g.to_numpy(dtype=object if metric == "nominal" else float)) for _, g in df.groupby("item")["y"] if len(g) >= 2]
    vals = np.concatenate(groups)
    n = len(vals)
    if metric == "interval":
        d = lambda a, b: (a - b) ** 2
    else:
        d = lambda a, b: (a != b).astype(float)
    Do = 0.0
    for g in groups:
        m = len(g)
        D = d(g[:, None], g[None, :])
        Do += D.sum() / (m - 1)
    Do /= n
    # expected disagreement over all pairable values
    if metric == "interval":
        De = 2 * n * vals.var() / (n - 1)  # mean over ordered pairs i!=j of (vi-vj)^2
    else:
        _, c = np.unique(vals, return_counts=True)
        De = (n * n - (c ** 2).sum()) / (n * (n - 1))
    return 1.0 - Do / De


# ------------------------------------------------------- pairwise heterogeneity
def rater_contrasts(df, k, l, min_per_cell=1):
    """Per-rater contrast D_r = mean_r(sys k) - mean_r(sys l) and its
    sampling variance v_r computed from pooled within-cell variances."""
    sub = df[df.sys.isin([k, l])]
    cell = sub.groupby(["rater", "sys"])["y"].agg(["mean", "var", "count"]).reset_index()
    pooled = {}
    for s in (k, l):
        c = cell[(cell.sys == s) & (cell["count"] >= 2)]
        if len(c) == 0:
            pooled[s] = sub[sub.sys == s].y.var()
        else:
            pooled[s] = ((c["var"] * (c["count"] - 1)).sum() / (c["count"] - 1).sum())
    wide = cell.pivot(index="rater", columns="sys")
    ok = (wide[("count", k)] >= min_per_cell) & (wide[("count", l)] >= min_per_cell)
    wide = wide[ok.fillna(False)]
    D = (wide[("mean", k)] - wide[("mean", l)]).values
    nk = wide[("count", k)].values
    nl = wide[("count", l)].values
    v = pooled[k] / nk + pooled[l] / nl
    return D, v, nk + nl, np.asarray(wide.index)


def dersimonian_laird(D, v):
    """Random-effects meta-analysis over raters. Returns (mu_RE, tau2, Q, df).
    tau2 = variance of *true* rater-level contrasts (noise removed)."""
    w = 1.0 / v
    mu_f = np.sum(w * D) / np.sum(w)
    Q = np.sum(w * (D - mu_f) ** 2)
    dfree = len(D) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfree) / C)
    ws = 1.0 / (v + tau2)
    mu = np.sum(ws * D) / np.sum(ws)
    return mu, tau2, Q, dfree


def eb_shrink(D, v, mu, tau2):
    """Empirical-Bayes (BLUP) rater contrasts."""
    if tau2 <= 0:
        return np.full_like(D, mu)
    return mu + tau2 / (tau2 + v) * (D - mu)


def _cells(rater, sys, y, R, K):
    idx = rater * K + sys
    n = np.bincount(idx, minlength=R * K).astype(float)
    s1 = np.bincount(idx, weights=y, minlength=R * K)
    s2 = np.bincount(idx, weights=y * y, minlength=R * K)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        ss = s2 - n * mean ** 2          # within-cell sum of squares
    return n.reshape(R, K), np.nan_to_num(mean).reshape(R, K), np.nan_to_num(ss).reshape(R, K)


def omnibus_Q(df, pairs, min_per_cell=1, codes=None):
    """Sum over system pairs of DL excess heterogeneity statistic (Q - df).
    (Vectorised; equivalent to calling rater_contrasts + dersimonian_laird.)"""
    if codes is None:
        codes = encode(df)
    rater, sys, y, R, K = codes
    n, mean, ss = _cells(rater, sys, y, R, K)
    dfw = np.clip(n - 1, 0, None)
    pooled = ss.sum(0) / np.maximum(dfw.sum(0), 1)
    tot = 0.0
    for k, l in pairs:
        ok = (n[:, k] >= min_per_cell) & (n[:, l] >= min_per_cell)
        if ok.sum() < 2:
            continue
        D = mean[ok, k] - mean[ok, l]
        v = pooled[k] / n[ok, k] + pooled[l] / n[ok, l]
        w = 1 / v
        mu = (w * D).sum() / w.sum()
        tot += (w * (D - mu) ** 2).sum() - (ok.sum() - 1)
    return tot


def encode(df):
    rc = pd.factorize(df["rater"])[0]
    sc_vals = df["sys"].values
    return rc, sc_vals.astype(int), df["y"].values.astype(float), rc.max() + 1, int(sc_vals.max()) + 1


def permute_within_item(df, rng):
    """Shuffle rater ids among the labels of each item: preserves every
    per-item label multiset (hence every agreement statistic) but destroys
    rater persistence across items (the null of Theorem 1)."""
    out = df.copy()
    r = out["rater"].values.copy()
    for idx in out.groupby("item").indices.values():
        r[idx] = r[rng.permutation(idx)]
    out["rater"] = r
    return out


def persistence_test(df, pairs, n_perm=500, seed=0, min_per_cell=1):
    """Permutation test of H0: raters are exchangeable within items (no
    persistent rater effects). sys must be integer-coded 0..K-1."""
    rng = np.random.default_rng(seed)
    rater, sys, y, R, K = encode(df)
    item = pd.factorize(df["item"])[0]
    obs = omnibus_Q(df, pairs, min_per_cell, (rater, sys, y, R, K))
    null = np.empty(n_perm)
    for b in range(n_perm):
        order = np.lexsort((rng.random(len(item)), item))   # random order within item
        base = np.lexsort((np.arange(len(item)), item))      # original order within item
        rp = np.empty_like(rater)
        rp[base] = rater[order]
        null[b] = omnibus_Q(None, pairs, min_per_cell, (rp, sys, y, R, K))
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return obs, null, p


# ------------------------------------------------------- robustness certificates
def gamma_star_gaussian(kappa):
    """Sensitivity Gamma* at which a pairwise ordering with standardized
    consensus margin kappa = m / tau flips, when rater-level contrasts are
    N(m, tau^2) and the evaluator population may be reweighted by any
    density ratio in [1/Gamma, Gamma] (Proposition 3)."""
    if not np.isfinite(kappa):
        return np.inf
    kappa = abs(kappa)
    if kappa == 0:
        return 1.0
    f = lambda G: stats.norm.pdf(stats.norm.ppf(1.0 / (1.0 + G))) * (G - 1.0 / G) - kappa
    hi = 2.0
    while f(hi) < 0:
        hi *= 2
        if hi > 1e300:
            return np.inf
    return optimize.brentq(f, 1.0, hi)


def worst_mean_ratio(d, w, G):
    """min_{w'} sum w' d s.t. w'_r in [w_r/G, G w_r], sum w' = 1 (greedy LP)."""
    order = np.argsort(d)
    wp = w / G
    budget = 1.0 - wp.sum()
    for r in order:
        add = min(budget, (G - 1.0 / G) * w[r])
        wp[r] += add
        budget -= add
        if budget <= 1e-15:
            break
    return np.sum(wp * d)


def gamma_star_discrete(d, w=None):
    """Smallest Gamma such that reweighting raters (types) within ratio
    Gamma flips the sign of the weighted mean contrast."""
    d = np.asarray(d, float)
    w = np.full(len(d), 1.0 / len(d)) if w is None else np.asarray(w, float) / np.sum(w)
    m = np.sum(w * d)
    s = np.sign(m)
    if s == 0:
        return 1.0
    dd = s * d
    if dd.min() >= 0:
        return np.inf
    lo, hi = 1.0, 2.0
    while worst_mean_ratio(dd, w, hi) > 0:
        hi *= 2
        if hi > 1e12:
            return np.inf
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if worst_mean_ratio(dd, w, mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def tv_radius(d, w=None):
    """Minimal total-variation shift of the type mixture that makes the
    weighted mean contrast <= 0 (Theorem 2, greedy fractional knapsack).
    Returns inf when all types agree in sign (ordering is consensus)."""
    d = np.asarray(d, float)
    w = np.full(len(d), 1.0 / len(d)) if w is None else np.asarray(w, float) / np.sum(w)
    m = np.sum(w * d)
    s = np.sign(m)
    if s == 0:
        return 0.0
    d = s * d
    m = abs(m)
    dmin = d.min()
    if dmin >= 0:
        return np.inf
    need = m
    eps = 0.0
    for r in np.argsort(-d):
        gain = d[r] - dmin
        if gain <= 0:
            break
        take = min(w[r], need / gain)
        eps += take
        need -= take * gain
        if need <= 1e-15:
            return eps
    return np.inf
