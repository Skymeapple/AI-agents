"""Revision r2: the n*log(M) horizon law.  Recomputes the finite-task minimax bracket of exp_minimax.py
(same code, same rigorous R_lo / R_up) on a grid of k = s*n with s fixed across n, for n in {8,...,128}
and M up to 1e8, to test whether R*(n, s n, M) depends on n only through s = k/n (Theorem 5 / Sec. 5.2).
Output: results/horizon_law_bounds.csv"""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import exp_minimax as em
em.MS = [128, 1000, 10000, 100000, 1000000, 100000000]
RES = os.path.join(os.path.dirname(__file__), "..", "results")
S = [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16]

if __name__ == "__main__":
    jobs = sorted({(n, int(round(s * n))) for n in [8, 16, 32, 64, 128] for s in S}, key=lambda j: -j[0] * j[1])
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        out = pool.map(em.job, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows]).sort_values(["n", "M", "k"])
    df["s"] = df.k / df.n
    # free strengthening (reviewer W5): any valid CI must cover both members of a population-indistinguishable pair
    df["R_lo_strong"] = np.maximum(df.R_lo, (1 - 2 * em.EPS) * df.pop_halfwidth)
    df.to_csv(os.path.join(RES, "horizon_law_bounds.csv"), index=False, float_format="%.6g")
    print(df[["n", "k", "M", "pop_halfwidth", "R_lo", "R_up"]].to_string())
