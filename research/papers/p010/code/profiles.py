"""Structure and reliability of task-family 'jaggedness' profiles in METR run-level data.

  * split-half reliability of family offsets d_{m,f} (runs of each task split at random into halves)
  * cross-model correlation of profiles (is the jaggedness shared across systems?)
  * which families are the bottleneck for the latest frontier systems
  * sensitivity of the bottleneck (min over families) to the shrinkage strength lambda
Usage: python3 profiles.py --version 1-1
"""
import argparse, json
import numpy as np
import pandas as pd
from core import load_runs, fit_model, log2_horizon

OUT = "/home/user/AI-agents/research/papers/p010/results"
ap = argparse.ArgumentParser(); ap.add_argument("--version", default="1-1"); ap.add_argument("--splits", type=int, default=20)
A = ap.parse_args()
df = load_runs(A.version)
df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
df["fam_key"] = df.task_family
models = df.groupby("alias").release.first().sort_values().index.tolist()
res = {}

# full-data profiles
D = {}
for m in models:
    f = fit_model(df[df.alias == m], lam=1.0)
    D[m] = pd.Series(f["delta"], index=f["fam"])
P = pd.DataFrame(D).T  # models x families
P.to_csv(f"{OUT}/profiles_{A.version}_delta.csv")
C = np.corrcoef(P.fillna(0).values)
iu = np.triu_indices(len(models), 1)
res["mean_pairwise_profile_corr"] = float(np.mean(C[iu]))
X = P.fillna(0).values - P.fillna(0).values.mean(0)
sv = np.linalg.svd(X - X.mean(1, keepdims=True), compute_uv=False)
res["pc1_var_share"] = float(sv[0] ** 2 / np.sum(sv ** 2))
# profile correlation between consecutive frontier generations (first 5 vs last 5 models)
early, late = P.iloc[:5].mean(), P.iloc[-5:].mean()
res["corr_early5_vs_late5_mean_profile"] = float(np.corrcoef(early, late)[0, 1])

# bottleneck families for the latest 5 systems at q=0.5
rows = []
for m in models[-5:]:
    g = df[df.alias == m]
    f = fit_model(g, lam=1.0)
    l2 = log2_horizon(f["a"], f["b"], 0.5, f["delta"])
    order = np.argsort(l2)
    pooled = fit_model(g)
    rows.append(dict(model=m, pooled_h50_min=2 ** log2_horizon(pooled["a"], pooled["b"], 0.5),
                     bottlenecks=[(f["fam"][i], round(float(2 ** l2[i]), 1)) for i in order[:3]]))
res["bottlenecks_latest"] = rows
fam_src = df.drop_duplicates("task_family").set_index("task_family").task_source
res["bottleneck_sources"] = {m["model"]: [fam_src[b[0]] for b in m["bottlenecks"]] for m in rows}

# split-half reliability
rng = np.random.default_rng(0)
rel = []
for s in range(A.splits):
    half = np.zeros(len(df), int)
    for _, idx in df.groupby(["alias", "task_id"]).indices.items():
        perm = rng.permutation(idx)
        half[perm[: len(perm) // 2]] = 1
    rs = []
    for m in models:
        g = df[df.alias == m]
        h = half[g.index.values]
        f0 = fit_model(g[h == 0], lam=1.0); f1 = fit_model(g[h == 1], lam=1.0)
        s0 = pd.Series(f0["delta"], index=f0["fam"]); s1 = pd.Series(f1["delta"], index=f1["fam"])
        j = s0.index.intersection(s1.index)
        rs.append(np.corrcoef(s0[j], s1[j])[0, 1])
    rel.append(rs)
rel = np.array(rel)
r_half = np.nanmean(rel, 0)
res["split_half_r_by_model"] = dict(zip(models, map(float, r_half)))
res["split_half_r_mean"] = float(np.nanmean(r_half))
res["spearman_brown_full"] = float(np.nanmean(2 * r_half / (1 + r_half)))
res["note"] = "split-half splits runs within tasks: measures run-level (sampling) noise only, not task-sampling noise"

# lambda sensitivity of min over families (latest model and earliest model)
lam_tab = {}
for m in [models[0], models[len(models) // 2], models[-1]]:
    g = df[df.alias == m]
    row = {}
    for lam in [0.01, 0.1, 0.25, 1, 4, 16]:
        f = fit_model(g, lam=lam)
        l2 = log2_horizon(f["a"], f["b"], 0.5, f["delta"])
        row[str(lam)] = dict(min_h50_min=float(2 ** l2.min()), sd_log2=float(np.std(l2)))
    lam_tab[m] = row
res["lambda_sensitivity"] = lam_tab
json.dump(res, open(f"{OUT}/profiles_{A.version}.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
