"""Held-out-task test of retry (pass@k) predictions.

5-fold task-grouped CV per agent: fit (i) task-level frailty GLMM (cloglog, Gaussian task effect) and
(ii) homogeneous pooled Weibull on 4/5 of tasks; predict pass@k on held-out tasks with >=4 runs from
their human length alone; compare with the unbiased empirical pass@k (Chen et al., 2021).
"""
import numpy as np
import pandas as pd
from math import comb
from scipy import optimize
from common import *

d = load_runs()
x, wq = np.polynomial.hermite_e.hermegauss(60)
wq = wq / wq.sum()
rng = np.random.default_rng(5)
rows = []
for alias, da in d.groupby("alias"):
    tt = task_table(da).reset_index(drop=True)
    perm = rng.permutation(len(tt))
    fold = np.empty(len(tt), int)
    fold[perm] = np.arange(len(tt)) % 5
    for f in range(5):
        tr, te = tt[fold != f], tt[(fold == f) & (tt.n >= 4)]
        if len(te) == 0:
            continue
        r = fit_glmm(tr, "cloglog", nested=False)
        r0 = optimize.minimize(lambda th: glmm_nll(np.array([th[0], th[1], np.log(1e-6)]), tr, "cloglog", False),
                               r.x[:2], method="Nelder-Mead")
        lt = np.log(te["T"].values / T_REF)
        pn = np.exp(-np.exp(np.clip(r.x[0] + r.x[1] * lt[:, None] + np.exp(r.x[2]) * x[None, :], -30, 6)))
        fh = np.exp(-np.exp(np.clip(r0.x[0] + r0.x[1] * lt, -30, 6)))
        for k in [1, 2, 3, 4]:
            for (i, row), pi_n, fhi in zip(te.iterrows(), pn, fh):
                rows.append(dict(alias=alias, fold=f, k=k, T=row["T"],
                                 emp=1 - comb(int(row.n - row.s), k) / comb(int(row.n), k),
                                 pred_frailty=float(((1 - (1 - pi_n) ** k) * wq).sum()),
                                 pred_homog=float(1 - (1 - fhi) ** k)))
    print(alias, flush=True)
R = pd.DataFrame(rows)
R.to_csv(f"{RES}/passk_heldout.csv", index=False)
A = R.groupby(["alias", "k"])[["emp", "pred_frailty", "pred_homog"]].mean().reset_index()
A["gain_emp"] = A.emp - A.groupby("alias").emp.transform("first")
A["gain_frailty"] = A.pred_frailty - A.groupby("alias").pred_frailty.transform("first")
A["gain_homog"] = A.pred_homog - A.groupby("alias").pred_homog.transform("first")
S = A.groupby("k")[["emp", "pred_frailty", "pred_homog", "gain_emp", "gain_frailty", "gain_homog"]].mean()
L = R.assign(long=R["T"] >= 240).groupby(["long", "k"])[["emp", "pred_frailty", "pred_homog"]].mean()
with open(f"{RES}/passk_heldout_summary.txt", "w") as f:
    f.write("Held-out tasks, mean over 20 agents (agent-level means first)\n" + S.round(4).to_string() + "\n\n")
    ae = A.assign(ef=(A.pred_frailty - A.emp).abs(), eh=(A.pred_homog - A.emp).abs()).groupby("k")[["ef", "eh"]].mean()
    f.write("Mean absolute agent-level error\n" + ae.round(4).to_string() + "\n\n")
    f.write("Split by task length (>=240 min), pooled over task-agent pairs\n" + L.round(4).to_string() + "\n")
print(open(f"{RES}/passk_heldout_summary.txt").read())
