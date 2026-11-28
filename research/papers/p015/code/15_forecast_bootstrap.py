"""Uncertainty for the short->long forecast errors of 12_duality.py.

Because all agents share the same target tasks, agent counts ("19/20 agents over-predict") overstate the
evidence.  Here per-task predictions are stored and the mean-over-agents error is bootstrapped by
resampling task FAMILIES among the target tasks (cluster bootstrap, 2000 draws), plus leave-one-family-out.

Outputs: results/forecast_task_preds.csv, results/forecast_bootstrap.txt
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from common import *
import npmle

d = load_runs()
AGENTS = sorted(d.alias.unique())
CUTS = [60.0, 120.0, 240.0]


def work(alias):
    da = d[d.alias == alias]
    tt = task_table(da)
    n, s, T = tt.n.values.astype(float), tt.s.values.astype(float), tt["T"].values.astype(float)
    w_run = (1.0 / da.groupby("task_id").y.transform("size")).values
    Trun, yrun = da["T"].values, da["y"].values.astype(float)
    rows = []
    for C in CUTS:
        tr, te = T < C, T >= C
        th, _ = fit_family("log-logistic (METR)", Trun[Trun < C], yrun[Trun < C], w_run[Trun < C])
        f = npmle.fit(n[tr], s[tr], T[tr])
        pl = FAMILIES["log-logistic (METR)"][0](th, T[te])
        pn = npmle.S_pooled(f["pi"], f["k"], T[te])
        for tid, fam, Ti, o, a, b in zip(tt.task_id[te], tt.family[te], T[te], (s / n)[te], pl, pn):
            rows.append(dict(alias=alias, cut_min=C, task_id=tid, family=fam, T=Ti, obs=o, pred_logistic=a, pred_npmle=b))
    print(alias, "done", flush=True)
    return rows


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(nproc) as p:
        res = p.map(work, AGENTS)
    P = pd.DataFrame([r for o in res for r in o])
    P.to_csv(f"{RES}/forecast_task_preds.csv", index=False)
    rng = np.random.default_rng(7)
    L = ["Family-cluster bootstrap (2000 draws) of the mean-over-agents forecast error (pred - obs)"]
    for C in CUTS:
        for band, hi in (("[C,4C)", 4 * C), (">=C", np.inf)):
            g = P[(P.cut_min == C) & (P["T"] < hi)]
            # per-task mean over agents
            tk = g.groupby(["task_id", "family"])[["obs", "pred_logistic", "pred_npmle"]].mean().reset_index()
            fams = tk.family.unique()
            base = {m: float((tk[m] - tk.obs).mean()) for m in ("pred_logistic", "pred_npmle")}
            bs = {m: [] for m in base}
            byf = {f_: tk[tk.family == f_] for f_ in fams}
            for _ in range(2000):
                samp = pd.concat([byf[f_] for f_ in rng.choice(fams, len(fams))])
                for m in base:
                    bs[m].append(float((samp[m] - samp.obs).mean()))
            lofo = {m: [float((tk[tk.family != f_][m] - tk[tk.family != f_].obs).mean()) for f_ in fams] for m in base}
            L.append(f"train < {C:.0f} min, target {band}: {len(tk)} tasks in {len(fams)} families")
            for m in base:
                lo, hi_ = np.quantile(bs[m], [0.025, 0.975])
                L.append(f"   {m}: mean error {base[m]:+.3f}, 95% family-bootstrap CI [{lo:+.3f}, {hi_:+.3f}], "
                         f"leave-one-family-out range [{min(lofo[m]):+.3f}, {max(lofo[m]):+.3f}]")
    open(f"{RES}/forecast_bootstrap.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
