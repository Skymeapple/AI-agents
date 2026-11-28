"""Task 4.4 (simulated domain): train a model on the experiment store and test it
on held-out experiments against simple baselines.

Training data comes only from the store's records (protocol + result), as a
real program would use its shared corpus. Held-out = experiments from separate
campaigns (different seeds), so this tests generalisation to new runs.
"""
from __future__ import annotations

import json

import numpy as np

from .campaign import gp
from .datastore import DataStore


def features(protocol):
    f = {s["reagent"]: s["fraction"] for s in protocol["steps"] if s["op"] == "dispense"}
    temp = next((s["temp_c"] for s in protocol["steps"] if s["op"] == "anneal"), 20.0)
    return [f.get("A", 0), f.get("B", 0), (temp - 20) / 580]


def collect(seeds):
    from .lab import LabAPI, SimulatedLab
    from .campaign import sample, to_proto
    X, y = [], []
    for s in seeds:
        store = DataStore()
        api = LabAPI(SimulatedLab(seed=s), store, campaign=f"dm-{s}")
        rng = np.random.default_rng(s + 77)
        for x in sample(rng, 100):
            api.run(to_proto(x, "p"))
        for r in store.records:
            if r["outcome"] == "success":
                X.append(features(r["protocol"]))
                y.append(r["result"]["value"])
    return np.array(X), np.array(y)


def metrics(y, p):
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": round(rmse, 4), "r2": round(r2, 4)}


def run():
    Xtr, ytr = collect(range(0, 8))
    Xte, yte = collect(range(100, 104))
    out = {"n_train": len(ytr), "n_test": len(yte)}
    out["mean_baseline"] = metrics(yte, np.full_like(yte, ytr.mean()))
    A = np.c_[np.ones(len(Xtr)), Xtr]
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    out["linear_baseline"] = metrics(yte, np.c_[np.ones(len(Xte)), Xte] @ w)
    Aq = lambda X: np.c_[np.ones(len(X)), X, X ** 2, X[:, [0]] * X[:, [1]], X[:, [0]] * X[:, [2]], X[:, [1]] * X[:, [2]]]
    wq, *_ = np.linalg.lstsq(Aq(Xtr), ytr, rcond=None)
    out["quadratic_baseline"] = metrics(yte, Aq(Xte) @ wq)
    mu, _ = gp(Xtr, ytr, Xte, ls=0.12)
    out["domain_model_gp"] = metrics(yte, mu)
    out["beats_all_baselines"] = all(out["domain_model_gp"]["rmse"] < out[k]["rmse"]
                                     for k in ("mean_baseline", "linear_baseline", "quadratic_baseline"))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
