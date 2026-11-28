"""Simulation checks (synthetic; no real data).
(1) Proposition 1 identity on random belief distributions.
(2) Bias of estimators of the task-level stop-loss curve pi_task(k)=E[(p_i-k)_+] from n runs per task:
    naive plug-in vs the split-half policy-value estimator used in usefulness.py.
Run: python3 code/sim_estimator.py   (a few seconds)  -> results/sim_estimator.json
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)

# (1) identity
def s(q, rho, h, c):
    return np.maximum(0, np.maximum(q - rho - c, 1 - c - (1 - q) * h))
def pi(q, k):
    return np.mean(np.maximum(q - k, 0))
err = 0.0
for _ in range(2000):
    q = rng.beta(rng.uniform(.2, 3), rng.uniform(.2, 3), 500)
    q[rng.random(500) < 0.2] = 1.0            # atom at 1 (deterministic successes)
    rho, c, h = rng.uniform(0, .6), rng.uniform(0, .3), 1 + rng.exponential(4)
    k1, k2 = rho + c, 1 - rho / (h - 1)
    rhs = pi(q, k1) + (h - 1) * pi(q, k2) if k2 >= k1 else h * pi(q, 1 - (1 - c) / h)
    err = max(err, abs(np.mean(s(q, rho, h, c)) - rhs))

# (2) estimator bias
def one(n_runs, n_tasks, m_prior, k, reps=400):
    out = dict(truth=[], naive=[], split=[])
    for _ in range(reps):
        # bimodal task difficulty, as in agent benchmarks: many tasks near 0 or 1, some mixed
        p = rng.beta(0.5, 0.5, n_tasks)
        y = rng.random((n_tasks, n_runs)) < p[:, None]
        out["truth"].append(np.mean(np.maximum(p - k, 0)))
        out["naive"].append(np.mean(np.maximum(y.mean(1) - k, 0)))
        h = n_runs // 2
        prior = y.mean()                       # global mean plays the role of the length model
        vals = []
        for A, B in [(y[:, :h], y[:, h:]), (y[:, h:], y[:, :h])]:
            q = (A.sum(1) + m_prior * prior) / (A.shape[1] + m_prior)
            vals.append(np.mean((q > k) * (B.mean(1) - k)))
        out["split"].append(np.mean(vals))
    return {a: float(np.mean(v)) for a, v in out.items()} | {a + "_sd": float(np.std(v)) for a, v in out.items()}

res = dict(prop1_max_abs_error=float(err), bias={})
for n_runs in [4, 6, 12]:
    for k in [0.25, 0.5, 0.75]:
        res["bias"][f"n{n_runs}_k{k}"] = one(n_runs, 228, 2.0, k)
json.dump(res, open(os.path.join(ROOT, "results/sim_estimator.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
