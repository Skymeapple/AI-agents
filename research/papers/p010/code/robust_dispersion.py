"""Robustness check: is the growth in cross-family dispersion of offsets a ceiling artefact?
Recomputes dispersion (i) excluding families a model saturates (all runs succeed) or floors (all fail),
(ii) as the lower semi-deviation (spread of the below-median tail only).
Usage: python3 robust_dispersion.py --version 1-1
"""
import argparse, json
import numpy as np
import pandas as pd
from core import load_runs

R = "/home/user/AI-agents/research/papers/p010/results"
ap = argparse.ArgumentParser(); ap.add_argument("--version", default="1-1"); A = ap.parse_args()
df = load_runs(A.version); df = df[df.release >= pd.Timestamp("2023-01-01")]
P = pd.read_csv(f"{R}/profiles_{A.version}_delta.csv", index_col=0)
rate = df.groupby(["alias", "task_family"]).y.mean().unstack()
t = df.groupby("alias").release.first()
rows = []
for m in P.index:
    d = P.loc[m].dropna()
    r = rate.loc[m, d.index]
    inner = d[(r > 0) & (r < 1)]
    med = np.median(d)
    low = d[d < med]
    rows.append(dict(alias=m, t=(t[m] - pd.Timestamp("2023-01-01")).days / 365.25, sd_all=d.std(ddof=0),
                     sd_unsaturated=inner.std(ddof=0), n_unsat=len(inner), n_sat=int((r == 1).sum()),
                     n_floor=int((r == 0).sum()), lower_semidev=np.sqrt(np.mean((low - med) ** 2))))
T = pd.DataFrame(rows).sort_values("t")
out = {}
for c in ["sd_all", "sd_unsaturated", "lower_semidev", "n_sat", "n_floor"]:
    X = np.column_stack([np.ones(len(T)), T.t]); co = np.linalg.lstsq(X, T[c].values.astype(float), rcond=None)[0]
    # model-level bootstrap of the slope (resample models)
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(T), len(T)); Xi = X[i]
        if np.ptp(Xi[:, 1]) == 0: continue
        bs.append(np.linalg.lstsq(Xi, T[c].values.astype(float)[i], rcond=None)[0][1])
    out[c] = dict(slope_per_year=float(co[1]), lo=float(np.percentile(bs, 5)), hi=float(np.percentile(bs, 95)))
T.to_csv(f"{R}/robust_dispersion_{A.version}.csv", index=False)
json.dump(out, open(f"{R}/robust_dispersion_{A.version}.json", "w"), indent=1)
print(T.round(2).to_string()); print(json.dumps(out, indent=1))
