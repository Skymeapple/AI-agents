"""Robustness of the HANNA persistence result: min-cell threshold and
leave-one-heavy-rater-out.  Run: python3 robustness_hanna.py"""
import itertools, numpy as np, pandas as pd
from core import *
d = pd.read_csv("../data/hanna_stories_annotations.csv")
CRIT = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]
d["Mean6"] = d[CRIT].mean(1)
models = sorted(d.Model.unique()); mcode = {m: i for i, m in enumerate(models)}
pairs = list(itertools.combinations(range(len(models)), 2))
df0 = pd.DataFrame(dict(item=d["Story ID"], sys=d.Model.map(mcode), rater=d["Worker ID"], y=d["Mean6"]))
rows = []
for minc in [2, 3, 5, 10]:
    obs, null, p = persistence_test(df0, pairs, n_perm=499, seed=3, min_per_cell=minc)
    rows.append(dict(check=f"min_per_cell={minc}", Qex=obs, null_mean=null.mean(), null_q95=np.quantile(null, .95), p=p))
heavy = d["Worker ID"].value_counts().index[:6]
for h in heavy:
    df = df0[df0.rater != h]
    obs, null, p = persistence_test(df, pairs, n_perm=499, seed=4, min_per_cell=3)
    rows.append(dict(check=f"drop rater {list(heavy).index(h)+1} (n={int((df0.rater==h).sum())})", Qex=obs,
                     null_mean=null.mean(), null_q95=np.quantile(null, .95), p=p))
R = pd.DataFrame(rows); R.to_csv("../results/hanna_robustness.csv", index=False); print(R.to_string())
