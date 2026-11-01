"""Task 3.5 (simulated) and RP4: a closed-loop campaign with no human step,
comparing three agents on the same simulated lab:

  random      - random compositions
  bo_reject   - Bayesian optimisation that drops |z|>3 points as outliers
                (common practice: the surprising result is treated as noise)
  bo_surprise - the same optimiser plus RP4's surprise score: a point far above
                prediction is replicated and, if it holds, explored locally

Everything is logged through the Lab API into the DataStore.
"""
from __future__ import annotations

import json

import numpy as np

from .datastore import DataStore
from .lab import LabAPI, SimulatedLab, make_protocol

SPIKE = np.array([0.15, 0.15, 0.9])


def to_proto(x, pid):
    fa, fb, t = x
    return make_protocol(float(fa), float(fb), float(20 + 580 * t), pid)


def sample(rng, n):
    pts = []
    while len(pts) < n:
        fa, fb, t = rng.random(3)
        if fa + fb <= 1:
            pts.append((fa, fb, t))
    return np.array(pts)


def gp(X, y, Xs, ls=0.15, noise=0.05):
    mu_y, sd_y = y.mean(), y.std() + 1e-9
    yn = (y - mu_y) / sd_y
    K = lambda A, B: np.exp(-((A[:, None, :] - B[None]) ** 2).sum(-1) / (2 * ls * ls))
    Kxx = K(X, X) + (noise / sd_y) ** 2 * np.eye(len(X)) + 1e-8 * np.eye(len(X))
    L = np.linalg.cholesky(Kxx)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    Ks = K(X, Xs)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(1 - (v * v).sum(0), 1e-9, None)
    return mu * sd_y + mu_y, np.sqrt(var) * sd_y


def run_agent(kind, seed, budget=100, anomaly=True, z_flag=3.0, beta=2.0, center=None):
    rng = np.random.default_rng(seed + 1000)
    store = DataStore()
    center = tuple(SPIKE) if center is None else tuple(center)
    api = LabAPI(SimulatedLab(seed=seed, anomaly=anomaly, center=center), store, campaign=f"{kind}-{seed}")
    X, Y, kept = [], [], []
    flags = confirmed = 0
    queue: list = []
    while api.cycle < budget:
        if queue:
            x = queue.pop(0)
        elif kind == "random" or len(X) < 10:
            x = sample(rng, 1)[0]
        else:
            Xa, Ya = np.array([X[i] for i in kept]), np.array([Y[i] for i in kept])
            cand = sample(rng, 800)
            mu, sd = gp(Xa, Ya, cand)
            x = cand[np.argmax(mu + beta * sd)]
        out = api.run(to_proto(x, f"p{api.cycle}"))
        if out["outcome"] != "success":
            continue
        y = out["result"]["value"]
        surprising = False
        if kind != "random" and len(kept) >= 10:
            Xa, Ya = np.array([X[i] for i in kept]), np.array([Y[i] for i in kept])
            mu, sd = gp(Xa, Ya, np.array([x]))
            z = (y - mu[0]) / np.sqrt(sd[0] ** 2 + 0.05 ** 2)
            surprising = z > z_flag
        X.append(x); Y.append(y)
        if surprising and kind == "bo_reject":
            continue  # dropped as an outlier
        kept.append(len(X) - 1)
        if surprising and kind == "bo_surprise":
            flags += 1
            rep = api.run(to_proto(x, f"rep{api.cycle}"))
            if rep["outcome"] == "success" and abs(rep["result"]["value"] - y) < 0.3:
                confirmed += 1
                X.append(x); Y.append(rep["result"]["value"]); kept.append(len(X) - 1)
                local = x + rng.normal(0, 0.05, size=(6, 3))
                queue += [p for p in np.clip(local, 0, 1) if p[0] + p[1] <= 1]
    Y = np.array(Y)
    best = int(np.argmax(Y))
    found = bool(np.linalg.norm(np.array(X[best]) - np.array(center)) < 0.1 and Y[best] > 3.6)
    return {"best": float(Y[best]), "found_anomaly": found, "flags": flags,
            "false_alarms": flags - confirmed, "experiments": api.cycle,
            "human_steps": api.human_steps, "store": store.audit()}


def run(seeds=range(12)):
    res = {}
    for anomaly in (True, False):
        for kind in ("random", "bo_reject", "bo_surprise"):
            rs = [run_agent(kind, s, anomaly=anomaly) for s in seeds]
            res[f"{kind}|anomaly={anomaly}"] = {
                "mean_best": float(np.mean([r["best"] for r in rs])),
                "anomaly_found_rate": float(np.mean([r["found_anomaly"] for r in rs])),
                "false_alarms_per_100": float(np.mean([r["false_alarms"] for r in rs])),
                "experiments": int(np.mean([r["experiments"] for r in rs])),
                "human_steps": int(sum(r["human_steps"] for r in rs)),
                "store_missing": int(sum(len(r["store"]["missing"]) for r in rs)),
                "chain_intact": all(r["store"]["chain_intact"] for r in rs),
                "negative_or_failed_logged": int(sum(r["store"]["by_outcome"].get("failed_run", 0) for r in rs)),
            }
    return res


def powered_rp4(n=45, beta=0.3):
    """RP4 re-run: 45 planted anomalies per arm (power analysis needs 42), random anomaly
    positions, and an exploit-heavy optimiser (UCB beta 0.3) that explores little."""
    rng = np.random.default_rng(2026)
    centers = []
    while len(centers) < n:
        c = rng.uniform([0.05, 0.05, 0.1], [0.6, 0.6, 0.95])
        if c[0] + c[1] <= 0.9:
            centers.append(c)
    out = {}
    for kind in ("bo_reject", "bo_surprise"):
        rs = [run_agent(kind, 500 + i, beta=beta, center=centers[i]) for i in range(n)]
        out[kind] = {"recall": float(np.mean([r["found_anomaly"] for r in rs])),
                     "found": int(sum(r["found_anomaly"] for r in rs)), "n": n,
                     "false_alarms_per_100": float(np.mean([r["false_alarms"] for r in rs])),
                     "flags_per_run": float(np.mean([r["flags"] for r in rs]))}
    a, b = out["bo_surprise"]["found"], out["bo_reject"]["found"]
    p1, p2 = a / n, b / n
    pp = (a + b) / (2 * n)
    z = (p1 - p2) / max(np.sqrt(pp * (1 - pp) * 2 / n), 1e-12)
    from scipy.stats import norm
    out["test"] = {"diff": p1 - p2, "z": float(z), "p_two_sided": float(2 * norm.sf(abs(z)))}
    return out


if __name__ == "__main__":
    import sys
    if "--rp4" in sys.argv:
        print(json.dumps(powered_rp4(), indent=2))
        raise SystemExit
    print(json.dumps(run(), indent=2))
