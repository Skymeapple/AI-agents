"""Revision r2: the n*log(M) horizon law (Theorem 5 / Sec. 5.2).
Recomputes the finite-task minimax bracket of exp_minimax.py (same code, same rigorous R_lo / R_up) on a grid
k = s*n with s fixed across n, for n in {8,...,128} and M up to 1e8, to test whether R*(n, s n, M) depends on
n only through s = k/n and how the horizon grows with M.
Usage: python3 exp_horizon_law.py N1 [N2 ...]   -> results/horizon_law_bounds_n{N}.csv  (one file per n)
       python3 exp_horizon_law.py merge         -> results/horizon_law_bounds.csv"""
import os, sys, glob, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import exp_minimax as em
em.MS = [128, 1000, 10000, 100000, 1000000, 100000000]
RES = os.path.join(os.path.dirname(__file__), "..", "results")
S = [1, 2, 3, 4, 6, 8, 12, 16]

if __name__ == "__main__":
    if sys.argv[1] == "merge":
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "horizon_law_bounds_n*.csv")))])
        df.to_csv(os.path.join(RES, "horizon_law_bounds.csv"), index=False, float_format="%.6g")
        sys.exit()
    ns = [int(a) for a in sys.argv[1:]]
    if ns == [128]:
        S = [1, 2, 4, 8, 16]   # reduced grid for n = 128 (CPU budget)
    jobs = sorted({(n, int(round(s * n))) for n in ns for s in S}, key=lambda j: -j[0] * j[1])
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        out = pool.map(em.job, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows]).sort_values(["n", "M", "k"])
    df["s"] = df.k / df.n
    # free strengthening (reviewer W5): any valid CI must cover both members of a population-indistinguishable pair
    df["R_lo_strong"] = np.maximum(df.R_lo, (1 - 2 * em.EPS) * df.pop_halfwidth)
    for n, d in df.groupby("n"):
        d.to_csv(os.path.join(RES, f"horizon_law_bounds_n{n}.csv"), index=False, float_format="%.6g")
    print(df[["n", "k", "M", "pop_halfwidth", "R_lo", "R_up"]].to_string())
