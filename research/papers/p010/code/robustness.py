"""Robustness of the definition-sensitivity results to shrinkage (lambda) and to suite version (TH1.0 vs TH1.1).

Usage (after the analysis_metr.py runs listed in README):  python3 robustness.py
Output: results/robustness.json, results/robustness_table.csv
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = "/home/user/AI-agents/research/papers/p010/results"
K = "1 work-month (167h)"
TAGS = ["metr_1-1_lam1", "metr_1-1_lam0.25", "metr_1-1_lam4", "metr_1-0_lam1"]
PORDER = ["pmean+1", "geo(p=0)", "pmean-1", "minfam"]  # Prop. 1: arrival non-decreasing along this list

summ, out = {}, {}
for t in TAGS:
    try:
        summ[t] = pd.read_csv(f"{R}/{t}_summary.csv").set_index("metric")
    except FileNotFoundError:
        print("missing", t)
ref = summ["metr_1-1_lam1"]
rows = []
for t, s in summ.items():
    a = s[K].replace(np.inf, np.nan)
    fin = a.dropna()
    o = dict(n_def=len(s), n_finite=int(len(fin)),
             arrival_range=[float(fin.min()), float(fin.max())],
             argmin=fin.idxmin(), argmax=fin.idxmax(),
             dt_range=[float(s.dt_months[np.isfinite(s.dt_months)].min()), float(s.dt_months[np.isfinite(s.dt_months)].max())],
             pooled50_arrival=float(s.loc["pooled|0.5", K]),
             pooled50_ci=[float(s.loc["pooled|0.5", K + "_lo"]), float(s.loc["pooled|0.5", K + "_hi"])],
             pooled50_dt=float(s.loc["pooled|0.5", "dt_months"]),
             delay_q80_vs_q50_pooled=float(s.loc["pooled|0.8", K] - s.loc["pooled|0.5", K]),
             minfam50_arrival=float(s.loc["minfam|0.5", K]),
             q10fam50_arrival=float(s.loc["q10fam|0.5", K]))
    common = ref.index.intersection(s.index)
    x, y = ref.loc[common, K].replace(np.inf, np.nan), s.loc[common, K].replace(np.inf, np.nan)
    m = x.notna() & y.notna()
    o["spearman_vs_main"] = float(spearmanr(x[m], y[m]).correlation)
    # Prop. 1 ordering check: fraction of (q, adjacent p pair) where arrival is non-decreasing as p falls
    ok, tot = 0, 0
    for q in ["0.5", "0.8", "0.95"]:
        v = [s.loc[f"{p}|{q}", K] for p in PORDER]
        for i in range(len(v) - 1):
            tot += 1; ok += int(v[i + 1] >= v[i] - 1e-9)
    o["prop1_order_pairs_satisfied"] = f"{ok}/{tot}"
    # does q raise arrival for every aggregator?
    qok = sum(int(s.loc[f"{g}|0.8", K] >= s.loc[f"{g}|0.5", K] and s.loc[f"{g}|0.95", K] >= s.loc[f"{g}|0.8", K])
              for g in sorted(set(i.split("|")[0] for i in s.index)))
    o["q_monotone_aggregators"] = f"{qok}/8"
    out[t] = o
    for mtr in s.index:
        rows.append(dict(setting=t, metric=mtr, dt_months=s.loc[mtr, "dt_months"], arrival=s.loc[mtr, K],
                         lo=s.loc[mtr, K + "_lo"], hi=s.loc[mtr, K + "_hi"], level_latest_log2=s.loc[mtr, "level_latest"]))
pd.DataFrame(rows).to_csv(f"{R}/robustness_table.csv", index=False)
json.dump(out, open(f"{R}/robustness.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
