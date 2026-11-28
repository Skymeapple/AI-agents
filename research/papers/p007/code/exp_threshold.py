"""Pooling shifts spectral detection thresholds (Remark/Prop. 4).
Unified vs specialist at FULL capacity (no capacity competition), sweeping n.
Case 'private': k_s=0 (all structure private): pooling can only hurt via noise.
Case 'shared' : k_p=0 (all structure shared): pooling helps.
Also reports the fraction of signal energy captured by each estimator.
Run: OMP_NUM_THREADS=1 python3 exp_threshold.py"""
import json, os
import numpy as np
from common import make_task, sim_losses, theory_losses
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
d, q, M, r = 64, 16, 4, 40
rng = np.random.default_rng(3)
res = {"d": d, "q": q, "M": M, "r": r,
       "bbp_specialist_n": float(np.sqrt(q * d)), "bbp_unified_private_n": float(np.sqrt(M * q * d)),
       "bbp_unified_shared_n": float(np.sqrt(q * d / M)), "rows": []}
for case, ks, kp in [("private", 0, 8), ("shared", 8, 0)]:
    for n in [8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 2048]:
        U, S = [], []
        for rep in range(40):
            Ws = make_task(d, q, ks, kp, M, rng=rng.integers(1 << 31))
            lu, ls = sim_losses(Ws, 1.0, n, r, rng); U.append(lu); S.append(ls)
        tu, ts = theory_losses(Ws, 1.0, n, r)
        row = dict(case=case, n=n, sim_u=float(np.mean(U)), sim_s=float(np.mean(S)),
                   sim_ratio=float(np.mean(U) / np.mean(S)), th_ratio=float(tu / ts),
                   total_energy=8.0)
        res["rows"].append(row); print(row, flush=True)
json.dump(res, open(os.path.join(OUT, "threshold_results.json"), "w"), indent=1)
