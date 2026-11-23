"""Revision r2 (reviewer W5): budget-matched comparison of the capped first-success design with fixed-n evaluation.
For WalkSAT configurations A/B/C the capped design (D = 1000) costs 413 / 451 / 162 rollouts per task
(results/capped_design_cost.csv).  We give the fixed-n design the SAME per-task budget n = cost and compute the
certified half-width of the block estimator (Theorem 5a; an achievable, rigorous worst-case bound) for pass@1000,
against the DKW half-width of the capped design, which holds simultaneously for all k <= D.
Output: results/capped_fair.csv"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from block import best_design
RES = os.path.join(os.path.dirname(__file__), "..", "results")
rows = []
for cfg, n in [("A", 413), ("B", 451), ("C", 162)]:
    for M in [240, 10000]:
        h, d = best_design(n, 1000, M, 0.05, Lmax=16)
        rows.append(dict(cfg=cfg, n_fixed=n, k=1000, M=M, block_half=h, L=d["L"], dkw_half=np.sqrt(np.log(40) / (2 * M))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "capped_fair.csv"), index=False, float_format="%.4f"); print(df)
