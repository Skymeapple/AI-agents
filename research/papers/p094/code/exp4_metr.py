"""Exp 4: real-data backtest on METR Time-Horizon v1.1 public run data.

Data: ext/metr/reports/time-horizon-1-1/data/raw/runs.jsonl (METR
eval-analysis-public repository).  Each row is one agent run on one task with a
binarised score.  Many (model, task) pairs have 4, 6 or 8 independent runs.

(A) Subsample-and-extrapolate backtest.  For a model, take tasks with >= n_full
runs, keep n_full of them (seeded random choice).  The target is the realised
budget-n_full coverage: share of tasks with >=1 success among those n_full runs.
The evaluator sees only a random k-subset of each task's runs (k < n_full) and
must produce a 95% upper confidence bound (UCB) on the target.  Repeated over
random subsets.  A UCB below the target is a 'false-safe' miss.
(B) Descriptive: for all 20 models with 4 runs per task, model-free identified
interval for C(B), B in {16, 64}, vs parametric point extrapolations.
"""
import json, os, collections, time
import numpy as np
from multiprocessing import Pool
from core import (histogram, betabin_fit, betabin_coverage, powerlaw_coverage,
                  naive_ucb, lp_bounds, default_grid, pass_at_k_hist)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "ext", "metr", "reports", "time-horizon-1-1", "data", "raw", "runs.jsonl")
OUT = os.path.join(HERE, "..", "results")
NBOOT, NSPLIT = 100, 60


def load():
    d = collections.defaultdict(list)
    for line in open(DATA):
        r = json.loads(line)
        if r["alias"] == "human":
            continue
        d[(r["alias"], r["task_id"])].append(int(r["score_binarized"]))
    return d


def matrix(d, model, n_full, rng):
    rows = []
    for (a, t), v in sorted(d.items()):
        if a == model and len(v) >= n_full:
            v = np.array(v); rows.append(rng.choice(v, n_full, replace=False))
    return np.array(rows)


def ucbs(h, k, B, rng):
    T = int(h.sum())
    r = {"naive": naive_ucb(h, k)}
    bb, pl = [], []
    for _ in range(NBOOT):
        hb = rng.multinomial(T, h / T)
        a, b = betabin_fit(hb, k); bb.append(betabin_coverage(a, b, B))
        pl.append(powerlaw_coverage(hb, k, B))
    r["betabin"] = float(np.quantile(bb, 0.95))
    r["powerlaw"] = float(np.quantile(pl, 0.95))
    a, b = betabin_fit(h, k)
    r["betabin_pt"] = betabin_coverage(a, b, B)
    r["powerlaw_pt"] = powerlaw_coverage(h, k, B)
    r["lp0"] = lp_bounds(h, k, B, alpha=0.05, grid=default_grid(0.0))[1]
    return r


def job(args):
    model, n_full, k, X, seed = args
    rng = np.random.default_rng(seed)
    target = float(np.mean(X.max(axis=1)))
    out = []
    for s in range(NSPLIT):
        idx = np.array([rng.choice(n_full, k, replace=False) for _ in range(len(X))])
        sub = np.take_along_axis(X, idx, axis=1)
        h = histogram(sub.sum(axis=1), k)
        out.append(ucbs(h, k, n_full, rng))
    return dict(model=model, n_full=n_full, k=k, T=len(X), target=target, splits=out)


if __name__ == "__main__":
    t0 = time.time()
    d = load()
    rng = np.random.default_rng(2026)
    models = sorted(set(a for a, _ in d))
    jobs = []
    for n_full, k_list, min_T in [(8, [2, 4], 60), (6, [2, 3], 140)]:
        for m in models:
            X = matrix(d, m, n_full, rng)
            if len(X) >= min_T:
                for k in k_list:
                    jobs.append((m, n_full, k, X, int(rng.integers(1 << 31))))
    print(len(jobs), "backtest jobs", flush=True)
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    summ = []
    for r in res:
        row = dict(model=r["model"], n_full=r["n_full"], k=r["k"], T=r["T"], target=r["target"])
        for mth in ["naive", "betabin", "powerlaw", "lp0", "betabin_pt", "powerlaw_pt"]:
            v = np.array([s[mth] for s in r["splits"]])
            row[mth + "_mean"] = float(v.mean())
            row[mth + "_cover"] = float(np.mean(v >= r["target"] - 1e-9))
        summ.append(row); print(row, flush=True)
    # (B) descriptive, all models, 4 runs per task
    desc = []
    for m in models:
        X = matrix(d, m, 4, rng)
        h = histogram(X.sum(axis=1), 4)
        row = dict(model=m, T=len(X), pass1=pass_at_k_hist(h, 4, 1), pass4=pass_at_k_hist(h, 4, 4))
        a, b = betabin_fit(h, 4)
        for B in [16, 64]:
            lo, hi = lp_bounds(h, 4, B, alpha=0.05, grid=default_grid(0.0))
            row[f"lp_lo{B}"], row[f"lp_hi{B}"] = lo, hi
            row[f"bb{B}"] = betabin_coverage(a, b, B)
            row[f"pl{B}"] = powerlaw_coverage(h, 4, B)
        desc.append(row); print(row, flush=True)
    json.dump(dict(backtest=summ, descriptive=desc, nboot=NBOOT, nsplit=NSPLIT),
              open(os.path.join(OUT, "exp4_metr.json"), "w"), indent=1)
    print("elapsed", time.time() - t0)
