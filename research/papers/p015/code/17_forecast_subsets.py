"""Sensitivity of the short->long forecast errors to suspect target tasks (reads 15's per-task predictions).

Subsets of TARGET tasks: all; never-solved tasks removed (outcome-selected, a diagnostic only);
baseline-timed tasks only.  Family-cluster bootstrap (2000 draws) as in 15_forecast_bootstrap.py.
Output: results/forecast_subsets.txt
"""
import numpy as np
import pandas as pd
from common import RES

P = pd.read_csv(f"{RES}/forecast_task_preds.csv")
A = pd.read_csv(f"{RES}/task_audit.csv", index_col=0)
never = set(A.index[A.never_solved])
est = set(A.index[A.estimated_T])
rng = np.random.default_rng(11)
L = ["Mean-over-tasks (task = mean over agents) forecast error, pred - obs, with 95% family-bootstrap CI"]
for C in (60.0, 120.0, 240.0):
    for band, hi in (("[C,4C)", 4 * C), (">=C", np.inf)):
        g = P[(P.cut_min == C) & (P["T"] < hi)]
        for lab, keep in (("all targets", np.ones(len(g), bool)), ("never-solved removed*", ~g.task_id.isin(never).values),
                          ("baseline-timed only", ~g.task_id.isin(est).values)):
            tk = g[keep].groupby(["task_id", "family"])[["obs", "pred_logistic", "pred_npmle"]].mean().reset_index()
            fams = tk.family.unique()
            byf = {f: tk[tk.family == f] for f in fams}
            out = []
            for m in ("pred_logistic", "pred_npmle"):
                bs = []
                for _ in range(2000):
                    s = pd.concat([byf[f] for f in rng.choice(fams, len(fams))])
                    bs.append(float((s[m] - s.obs).mean()))
                lo, hi_ = np.quantile(bs, [0.025, 0.975])
                out.append(f"{m.replace('pred_', '')} {float((tk[m] - tk.obs).mean()):+.3f} [{lo:+.3f}, {hi_:+.3f}]")
            L.append(f"train < {C:.0f} min, target {band}, {lab}: {len(tk)} tasks/{len(fams)} families; " + "; ".join(out))
open(f"{RES}/forecast_subsets.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
