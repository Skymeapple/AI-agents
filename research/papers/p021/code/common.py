"""Shared utilities: data loading and psychometric (logistic +/- lapse) fits.

Data: METR eval-analysis-public, reports/time-horizon-1-1/data/raw/runs.jsonl
(commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2, cloned into ../data/).
"""
import json
import os
import datetime as dt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "eval-analysis-public")
RUNS = os.path.join(DATA, "reports", "time-horizon-1-1", "data", "raw", "runs.jsonl")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Release dates copied from data/external/release_dates.yaml in the METR repo.
RELEASE = {
    "GPT-4 0314": "2023-03-14", "GPT-4 1106 (Inspect)": "2023-11-06",
    "Claude 3 Opus (Inspect)": "2024-03-04", "GPT-4 Turbo (Inspect)": "2024-04-09",
    "GPT-4o (Inspect)": "2024-05-13", "Claude 3.5 Sonnet (Old) (Inspect)": "2024-06-20",
    "o1-preview": "2024-09-12", "Claude 3.5 Sonnet (New) (Inspect)": "2024-10-22",
    "o1 (Inspect)": "2024-12-05", "Claude 3.7 Sonnet (Inspect)": "2025-02-24",
    "o3 (Inspect)": "2025-04-16", "Claude 4 Opus (Inspect)": "2025-05-22",
    "Claude 4.1 Opus (Inspect)": "2025-08-05", "GPT-5 (Inspect)": "2025-08-07",
    "Gemini 3 Pro": "2025-11-18", "GPT-5.1-Codex-Max (Inspect)": "2025-11-19",
    "Claude Opus 4.5 (Inspect)": "2025-11-24", "GPT-5.2": "2025-12-11",
    "Claude Opus 4.6 (Inspect)": "2026-02-05", "GPT-5.3-Codex": "2026-02-05",
}


def short(alias):
    return alias.replace(" (Inspect)", "")


def date_float(alias):
    d = dt.date.fromisoformat(RELEASE[alias])
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def load_runs():
    rows = [json.loads(l) for l in open(RUNS)]
    return [r for r in rows if r["alias"] != "human"]


def task_table(rows, alias):
    """Aggregate runs of one agent to per-task (x=log2 minutes, n, s, family, weight)."""
    agg = {}
    for r in rows:
        if r["alias"] != alias:
            continue
        t = agg.setdefault(r["task_id"], dict(task=r["task_id"], fam=r["task_family"],
                                              minutes=r["human_minutes"], n=0, s=0,
                                              w=r["equal_task_weight"], src=r["task_source"]))
        t["n"] += 1
        t["s"] += int(r["score_binarized"])
    T = sorted(agg.values(), key=lambda d: d["task"])
    out = {k: np.array([d[k] for d in T]) for k in T[0]}
    out["x"] = np.log2(out["minutes"])
    return out


# ---------------- psychometric model -----------------
# p(x) = (1 - lam) * sigmoid(a - b * x),  x = log2(human minutes)
# lam = lapse (asymptotic failure probability on arbitrarily short tasks)

def p_model(theta, x, lapse=True):
    a, b = theta[0], theta[1]
    lam = 0.5 * expit(theta[2]) if lapse else 0.0
    return (1 - lam) * expit(a - b * x)


def nll(theta, x, n, s, w=None, lapse=True):
    p = np.clip(p_model(theta, x, lapse), 1e-12, 1 - 1e-12)
    ll = s * np.log(p) + (n - s) * np.log1p(-p)
    if w is not None:
        ll = ll * w
    return -np.sum(ll)


def fit(x, n, s, w=None, lapse=True, fixed_lam=None):
    """ML fit. Returns dict(a, b, lam, nll). fixed_lam fixes lapse (profile likelihood)."""
    best = None
    if fixed_lam is not None:
        lam = fixed_lam
        f = lambda th: nll_fixed(th, x, n, s, w, lam)
        starts = [(1.0, 0.5), (3.0, 0.8), (0.0, 0.3), (5.0, 1.0)]
        for st in starts:
            r = minimize(f, np.array(st), method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
            if best is None or r.fun < best.fun:
                best = r
        return dict(a=best.x[0], b=best.x[1], lam=lam, nll=best.fun)
    starts = [(1.0, 0.5, -3.0), (3.0, 0.8, -5.0), (0.0, 0.3, -1.0), (5.0, 1.0, -8.0)]
    for st in starts:
        st = np.array(st if lapse else st[:2])
        f = lambda th: nll(th, x, n, s, w, lapse)
        r = minimize(f, st, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=6000))
        if best is None or r.fun < best.fun:
            best = r
    lam = 0.5 * expit(best.x[2]) if lapse else 0.0
    return dict(a=best.x[0], b=best.x[1], lam=lam, nll=best.fun)


def nll_fixed(th, x, n, s, w, lam):
    p = np.clip((1 - lam) * expit(th[0] - th[1] * x), 1e-12, 1 - 1e-12)
    ll = s * np.log(p) + (n - s) * np.log1p(-p)
    if w is not None:
        ll = ll * w
    return -np.sum(ll)


def horizon(f, q):
    """Task length (minutes) at which modelled success equals q; nan if unreachable."""
    lam = f["lam"]
    if q >= 1 - lam:
        return np.nan
    z = q / (1 - lam)
    x = (f["a"] - np.log(z / (1 - z))) / f["b"]
    return 2.0 ** x


def cluster_bootstrap_indices(fam, rng):
    fams = np.unique(fam)
    pick = rng.choice(fams, size=len(fams), replace=True)
    idx = np.concatenate([np.where(fam == f)[0] for f in pick])
    return idx


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
