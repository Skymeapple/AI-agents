"""Core estimation utilities for the definition-sensitivity analysis of METR run-level data.

Model (per AI system m):
    logit P(success | run r) = a_m + d_{m,f(r)} + b_m * x_r ,   x_r = log2(human_minutes)
with an L2 (ridge) penalty lam * sum_f d_{m,f}^2 on family offsets (d = 0 gives the pooled METR-style model).
Horizon at success level q for family f:  log2 H_{m,f}(q) = (logit(q) - a_m - d_{m,f}) / b_m.
"""
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data")
RESULTS = str(ROOT / "results")


def load_runs(version="1-1"):
    d = pd.read_json(f"{DATA}/runs_{version}.jsonl", lines=True)
    d = d[d.alias != "human"].copy()
    rel = yaml.safe_load(open(f"{DATA}/release_dates.yaml"))["date"]
    rel = {k: pd.Timestamp(str(v)) for k, v in rel.items()}
    d["release"] = d.alias.map(rel)
    d = d[~d.release.isna()].copy()
    d["x"] = np.log2(d.human_minutes.astype(float))
    d["y"] = d.score_binarized.astype(float)
    w = d.invsqrt_task_weight.astype(float).values
    d["w"] = w
    # normalise weights within each model to sum to the number of runs
    d["w"] = d.groupby("alias").w.transform(lambda s: s / s.sum() * len(s))
    return d


def irls(X, y, w, pen, iters=60, tol=1e-9):
    """Weighted penalised logistic regression by Newton/IRLS. pen: vector of ridge coefs per column."""
    beta = np.zeros(X.shape[1])
    P = np.diag(pen)
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        g = X.T @ (w * (y - p)) - P @ beta
        W = w * p * (1 - p)
        H = (X * W[:, None]).T @ X + P + 1e-8 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def fit_model(df, lam=None, tiny=1e-3, return_hessian=False):
    """Fit one AI system. lam=None -> pooled model. Returns dict with a, b, and family offsets."""
    # Collapse runs sharing (family, task length): the weighted Bernoulli log-likelihood depends on the data
    # only through sum(w) and sum(w*y) per covariate pattern, so this is exact and ~5x faster.
    keys = df.fam_key.values if lam is not None else np.zeros(len(df), int)
    g = pd.DataFrame({"k": keys, "x": df.x.values, "wy": df.w.values * df.y.values, "w": df.w.values})
    g = g.groupby(["k", "x"], sort=True, as_index=False)[["wy", "w"]].sum()
    x = g.x.values
    w = g.w.values
    y = np.divide(g.wy.values, w, out=np.zeros_like(w), where=w > 0)
    if lam is None:
        X = np.column_stack([np.ones_like(x), x])
        beta = irls(X, y, w, np.array([tiny, tiny]))
        return dict(a=beta[0], b=beta[1], fam=None, delta=None)
    fams, idx = np.unique(g.k.values, return_inverse=True)
    F = len(fams)
    Z = np.zeros((len(x), F))
    Z[np.arange(len(x)), idx] = 1.0
    X = np.column_stack([np.ones_like(x), x, Z])
    pen = np.concatenate([[tiny, tiny], np.full(F, lam)])
    beta = irls(X, y, w, pen)
    out = dict(a=beta[0], b=beta[1], fam=fams, delta=beta[2:])
    if return_hessian:  # quantities for a Laplace approximation of the marginal likelihood
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        ll = float(np.sum(w * (y * np.log(np.clip(p, 1e-300, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-300, 1)))))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(pen)
        out.update(loglik=ll, H=H, F=F)
    return out


def logit(q):
    return np.log(q / (1 - q))


def log2_horizon(a, b, q, delta=0.0):
    return (logit(q) - a - delta) / b


def power_mean_log2(l2h, p):
    """Power mean of horizons H=2^l2h, returned in log2 units. p=0 -> geometric; +-inf -> max/min."""
    l2h = np.asarray(l2h, float)
    if p == 0:
        return l2h.mean()
    if np.isinf(p):
        return l2h.max() if p > 0 else l2h.min()
    z = p * l2h * np.log(2)
    m = z.max()
    return (m + np.log(np.mean(np.exp(z - m)))) / (p * np.log(2))


def hier_bootstrap(df, rng):
    """Resample task families, then tasks within families (runs kept). Duplicated families relabelled."""
    fams = df.task_family.unique()
    pick = rng.choice(fams, size=len(fams), replace=True)
    parts = []
    by_fam = {f: g for f, g in df.groupby("task_family")}
    for j, f in enumerate(pick):
        g = by_fam[f]
        tasks = g.task_id.unique()
        tp = rng.choice(tasks, size=len(tasks), replace=True)
        by_task = {t: h for t, h in g.groupby("task_id")}
        for k, t in enumerate(tp):
            h = by_task[t].copy()
            h["fam_key"] = f"{f}#{j}"
            parts.append(h)
    return pd.concat(parts, ignore_index=True)
