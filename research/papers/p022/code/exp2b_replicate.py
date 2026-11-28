"""Replication of the E3 Monte-Carlo check at m=4 (which deviated by ~3 SE in exp2)."""
import json, numpy as np, theory as th
from simulator import DetSampler, run_many
d = DetSampler('beta', a=0.5, b=1.0)
res = {"theory": float(th.p_clean_const(0.002, th.mu_beta(0.5, 1, 4), 1000)), "reps": []}
for s in [7, 8]:
    _, c = run_many(4000, s, N=1000, T=10, p0=0.002, beta=0, det=d, m=4, f=0.02, V=1, C=.5, R=1)
    res["reps"].append(dict(seed=s, n=4000, sim=float(c.mean()), se=float(np.sqrt(c.mean()*(1-c.mean())/4000))))
print(res); json.dump(res, open("../results/e2b_replication.json", "w"), indent=1)
