"""Shared utilities: data loading, pass@k estimators, partial-identification LP,
beta-binomial extrapolation, weighted logistic horizon fits.

All functions are deterministic given an explicit numpy Generator.
"""
import json
import os
from collections import defaultdict

import numpy as np
import yaml
from scipy.optimize import linprog, minimize
from scipy.special import betaln, comb, expit, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_runs(drop_fatal=True):
    """Load METR Time-Horizon 1.1 runs (public release). Excludes human baselines.

    drop_fatal: drop runs that ended in an infrastructure/usage-limit error
    (`fatal_error_from` not null); these are not model failures.
    """
    rows = []
    with open(os.path.join(DATA, "metr_th11_runs.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["alias"] == "human":
                continue
            if drop_fatal and r.get("fatal_error_from") is not None:
                continue
            rows.append(r)
    return rows


def release_dates():
    with open(os.path.join(DATA, "release_dates.yaml")) as f:
        d = yaml.safe_load(f)["date"]
    return {k: np.datetime64(str(v)) for k, v in d.items()}


def by_model_task(rows):
    """dict model -> dict task_id -> dict(outcomes=list, family, minutes)."""
    out = defaultdict(dict)
    for r in rows:
        m = out[r["alias"]]
        t = m.setdefault(r["task_id"], {"y": [], "family": r["task_family"],
                                         "minutes": r["human_minutes"],
                                         "source": r["task_source"]})
        t["y"].append(int(r["score_binarized"]))
    return out


# ----------------------------------------------------------------------------
# pass@k
# ----------------------------------------------------------------------------
def pass_at_k_unbiased(n, c, k):
    """Chen et al. (2021) unbiased estimator of pass@k from n samples with c successes."""
    if k > n:
        raise ValueError("k > n: not unbiasedly estimable")
    if n - c < k:
        return 1.0
    return 1.0 - np.exp(gammaln(n - c + 1) - gammaln(n - c - k + 1)
                        - gammaln(n + 1) + gammaln(n - k + 1))


def binom_pmf_matrix(m, pgrid):
    """A[c, j] = P(C=c | p_j), C ~ Binomial(m, p_j)."""
    c = np.arange(m + 1)[:, None]
    p = pgrid[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        A = comb(m, c) * np.where(c == 0, 1.0, p ** c) * np.where(m - c == 0, 1.0, (1 - p) ** (m - c))
    return np.nan_to_num(A)


def default_grid(J=600, pmin=1e-7):
    g = np.concatenate([[0.0], np.logspace(np.log10(pmin), 0, J)])
    return np.unique(g)


def lp_bounds(lo, hi, m, K, pgrid=None, pmin=None, relax=True, return_slack=False):
    """Sharp bounds on E_G[1-(1-p)^K] over all mixing distributions G on [0,1]
    whose induced Binomial(m) count pmf lies in the box [lo, hi] (elementwise).

    pmin: optional 'resolution floor' assumption, support(G) in {0} U [pmin, 1].
    relax: if the box is infeasible (the empirical pmf lies outside the binomial-
      mixture moment space, which happens through sampling noise), enlarge the box
      uniformly by the smallest slack s* that restores feasibility (computed by an
      auxiliary LP) and report s*. Returns (L, U) or (L, U, s*).
    Discretised on pgrid (includes p=0 and p down to 1e-7).
    """
    if pgrid is None:
        pgrid = default_grid()
    if pmin is not None:
        pgrid = pgrid[(pgrid == 0) | (pgrid >= pmin)]
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    A = binom_pmf_matrix(m, pgrid)
    f = 1.0 - (1.0 - pgrid) ** K
    J = len(pgrid)
    A_eq = np.ones((1, J))
    b_eq = np.array([1.0])

    def solve(c, slack):
        A_ub = np.vstack([A, -A])
        b_ub = np.concatenate([hi + slack, -(lo - slack)])
        return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=(0, None), method="highs")

    slack = 0.0
    r0 = solve(np.zeros(J), 0.0)
    if r0.status != 0:
        if not relax:
            return (np.nan, np.nan, np.nan) if return_slack else (np.nan, np.nan)
        # min s  s.t. lo - s <= A g <= hi + s, sum g = 1, g >= 0, s >= 0
        Ab = np.vstack([np.hstack([A, -np.ones((m + 1, 1))]),
                        np.hstack([-A, -np.ones((m + 1, 1))])])
        bb = np.concatenate([hi, -lo])
        rs = linprog(np.r_[np.zeros(J), 1.0], A_ub=Ab, b_ub=bb,
                     A_eq=np.hstack([A_eq, [[0.0]]]), b_eq=b_eq,
                     bounds=(0, None), method="highs")
        slack = float(rs.x[-1]) * (1 + 1e-6) + 1e-12
    res_min = solve(f, slack)
    res_max = solve(-f, slack)
    if res_min.status != 0 or res_max.status != 0:
        out = (np.nan, np.nan)
    else:
        out = (float(res_min.fun), float(-res_max.fun))
    return (out[0], out[1], slack) if return_slack else out


def count_pmf(counts, m):
    return np.bincount(counts, minlength=m + 1)[: m + 1] / len(counts)


def clopper_pearson(x, n, a):
    """Exact two-sided (1-a) Clopper-Pearson interval; x may be non-integer (effective counts)."""
    from scipy.stats import beta as _beta
    lo = 0.0 if x <= 0 else _beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x >= n else _beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def pmf_confidence_box(counts, clusters, m, alpha, rng, B=2000, method="hybrid"):
    """Simultaneous (Bonferroni over m+1 cells) box for the count pmf.

    method="bootstrap": cluster-bootstrap percentile box (resample clusters).
    method="hybrid" (default): elementwise hull of the bootstrap box and an exact
      Clopper-Pearson box computed with the number of clusters as the sample size
      (x = phat * n_clusters). The exact part repairs the percentile bootstrap's
      degeneracy at the boundary: if no task ever succeeds, every bootstrap
      replicate has P(C=0)=1 and the bootstrap box collapses to a point.
    counts: per-task success counts out of m; clusters: labels."""
    counts = np.asarray(counts)
    clusters = np.asarray(clusters)
    labs = np.unique(clusters)
    idx_by = [np.where(clusters == l)[0] for l in labs]
    boots = np.empty((B, m + 1))
    for b in range(B):
        pick = rng.integers(0, len(labs), len(labs))
        ii = np.concatenate([idx_by[i] for i in pick])
        boots[b] = count_pmf(counts[ii], m)
    a = alpha / (m + 1)
    lo = np.quantile(boots, a / 2, axis=0)
    hi = np.quantile(boots, 1 - a / 2, axis=0)
    if method == "bootstrap":
        return lo, hi
    ph = count_pmf(counts, m)
    nc = len(labs)
    cp = np.array([clopper_pearson(ph[c] * nc, nc, a) for c in range(m + 1)])
    return np.minimum(lo, cp[:, 0]), np.maximum(hi, cp[:, 1])


# ----------------------------------------------------------------------------
# Beta-binomial (parametric) extrapolation
# ----------------------------------------------------------------------------
def fit_beta_binomial(counts, m, lo=1e-3, hi=1e4):
    """MLE of Beta(a,b) mixing distribution from Binomial(m) counts.
    a, b are box-constrained to [lo, hi] (log scale, L-BFGS-B); without the box the
    MLE can run off to a,b -> 0 or infinity for small m (e.g. m = 2), where the
    likelihood is flat and pass@K becomes numerically meaningless."""
    counts = np.asarray(counts)

    def nll(theta):
        a, b = np.exp(theta)
        return -np.sum(betaln(counts + a, m - counts + b) - betaln(a, b))

    bnds = [(np.log(lo), np.log(hi))] * 2
    best = None
    for a0 in (0.2, 1.0, 3.0):
        for b0 in (0.2, 1.0, 3.0):
            r = minimize(nll, np.log([a0, b0]), method="L-BFGS-B", bounds=bnds)
            if best is None or r.fun < best.fun:
                best = r
    a, b = np.exp(best.x)
    return float(a), float(b)


def beta_pass_at_k(a, b, K):
    """E[1-(1-p)^K] for p ~ Beta(a,b):  1 - B(a, b+K)/B(a, b), computed stably."""
    log_ratio = gammaln(b + K) - gammaln(b) + gammaln(a + b) - gammaln(a + b + K)
    return float(np.clip(1.0 - np.exp(log_ratio), 0.0, 1.0))


# ----------------------------------------------------------------------------
# Weighted logistic horizon fit
# ----------------------------------------------------------------------------
def fit_horizon(x_log2min, y, w, q=0.5, reg=1e-3):
    """Fit P(success) = sigmoid(b0 + b1 * log2(minutes)), with fractional y allowed
    (quasi-binomial), weights w. Return log2 horizon at success prob q."""
    x = np.asarray(x_log2min, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    w = w / w.sum()

    def nll(beta):
        z = beta[0] + beta[1] * x
        p = np.clip(expit(z), 1e-12, 1 - 1e-12)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) + reg * beta[1] ** 2

    r = minimize(nll, np.array([2.0, -0.5]), method="BFGS")
    b0, b1 = r.x
    if b1 >= 0:
        return np.nan
    logit_q = np.log(q / (1 - q))
    return float((logit_q - b0) / b1)
