"""Choose the shrinkage strength lambda for family offsets by K-fold cross-validation over tasks.
Held-out tasks whose family is absent from the training fold get offset 0 (pooled prediction).
Usage: python3 cv_lambda.py --version 1-1
"""
import argparse, json
import numpy as np
import pandas as pd
from core import load_runs, fit_model

OUT = "/home/user/AI-agents/research/papers/p010/results"
ap = argparse.ArgumentParser(); ap.add_argument("--version", default="1-1"); ap.add_argument("--K", type=int, default=5)
ap.add_argument("--reps", type=int, default=3)
A = ap.parse_args()
df = load_runs(A.version)
df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
df["fam_key"] = df.task_family
lams = [0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, None]
tasks = df.task_id.unique()
rng = np.random.default_rng(0)
loss = {str(l): 0.0 for l in lams}
wsum = 0.0
for rep in range(A.reps):
    fold = dict(zip(tasks, rng.integers(0, A.K, len(tasks))))
    df["fold"] = df.task_id.map(fold)
    for k in range(A.K):
        tr, te = df[df.fold != k], df[df.fold == k]
        for m, gte in te.groupby("alias"):
            gtr = tr[tr.alias == m]
            for l in lams:
                f = fit_model(gtr, lam=l)
                eta = f["a"] + f["b"] * gte.x.values
                if l is not None:
                    dmap = dict(zip(f["fam"], f["delta"]))
                    eta = eta + np.array([dmap.get(fk, 0.0) for fk in gte.fam_key.values])
                p = np.clip(1 / (1 + np.exp(-eta)), 1e-6, 1 - 1e-6)
                y, w = gte.y.values, gte.w.values
                loss[str(l)] += float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))
            wsum += gte.w.sum() / len(lams) * 0 + 0  # placeholder (weights identical across lambdas)
tot_w = df.w.sum() * A.reps
res = {"weighted_logloss_per_run": {k: v / tot_w for k, v in loss.items()}}
best = min((v, k) for k, v in res["weighted_logloss_per_run"].items())
res["best_lambda"] = best[1]
json.dump(res, open(f"{OUT}/cv_lambda_{A.version}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
