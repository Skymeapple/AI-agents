"""Revision r2: achievable horizon of the block estimator (Theorem 5(a)) over a wide (n, M) range.
For each n and s = k/n on a geometric grid, and each degree L, the design (bias beta, range of Z) is computed once
(it does not depend on M); the certified Hoeffding half-width beta + range*sqrt(ln(2/eps)/(2M)) is then minimised
over L for every M.  The achievable horizon k_ach(n, M, delta) = largest grid k with half-width <= delta.
Output: results/block_law.csv (half-widths), results/block_horizon.csv"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from block import design
RES = os.path.join(os.path.dirname(__file__), "..", "results")
EPS = 0.05
MS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12]
SS = np.unique(np.round(np.geomspace(1, 96, 34), 3))

rows = []
for n in [16, 64, 256, 1024]:
    for s in SS:
        k = int(round(s * n))
        des = [d for d in (design(n, k, L) for L in range(0, min(16, n - 1) + 1)) if d is not None]
        for M in MS:
            hs = [d["beta"] + d["zrange"] * np.sqrt(np.log(2 / EPS) / (2 * M)) for d in des]
            i = int(np.argmin(hs))
            rows.append(dict(n=n, k=k, s=k / n, M=M, half=hs[i], L=des[i]["L"], beta=des[i]["beta"], zrange=des[i]["zrange"]))
    print(n, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "block_law.csv"), index=False, float_format="%.6g")
tab = []
for delta in [0.02, 0.05, 0.1]:
    for (n, M), d in df.groupby(["n", "M"]):
        ok = d[d.half <= delta]
        tab.append(dict(delta=delta, n=n, M=M, k_ach=ok.k.max() if len(ok) else np.nan,
                        s_ach=(ok.k.max() / n) if len(ok) else np.nan))
tab = pd.DataFrame(tab)
tab.to_csv(os.path.join(RES, "block_horizon.csv"), index=False, float_format="%.4g")
print(tab.pivot_table(index=["delta", "M"], columns="n", values="s_ach").to_string())
