"""Nonparametric between-task heterogeneity index H(T) from replicate runs.

For tasks with n_i >= 2 runs, s_i successes:
  unbiased E[p_i]   : s_i / n_i
  unbiased E[p_i^2] : s_i (s_i - 1) / (n_i (n_i - 1))
Within a length stratum B with m tasks:
  Pbar  = mean_i s_i/n_i
  M2    = mean_i s_i(s_i-1)/(n_i(n_i-1))
  Pbar2 = unbiased estimate of Pbar^2 from cross-task products (i != j)
  H     = (M2 - Pbar2) / (Pbar (1 - Pbar))      in [0, 1] up to sampling noise
H = 0: every task in the stratum has the same success probability (all variation is run-level
noise, as in a homogeneous compounding-error model). H = 1: each task is either always or never
solved (outcome fully determined by task identity).
Also: pass@2 identity  pass@2 = 2 Pbar - M2 = [1-(1-Pbar)^2] - H Pbar (1-Pbar).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

d = load_runs()
EDGES = [0, 1, 4, 15, 60, 240, 960, 1e9]
LABELS = ["<1m", "1-4m", "4-15m", "15-60m", "1-4h", "4-16h", ">16h"]
rng = np.random.default_rng(7)


def H_stats(tt):
    p = tt.s / tt.n
    m2 = tt.s * (tt.s - 1) / (tt.n * (tt.n - 1))
    m = len(tt)
    S1 = p.sum()
    Pbar = S1 / m
    Pbar2 = (S1 ** 2 - (p ** 2).sum()) / (m * (m - 1))
    M2 = m2.mean()
    den = Pbar * (1 - Pbar)
    H = (M2 - Pbar2) / den if den > 1e-9 else np.nan
    pass2_emp = (1 - (tt.n - tt.s) * (tt.n - tt.s - 1) / (tt.n * (tt.n - 1))).mean()
    return Pbar, H, M2, pass2_emp


rows = []
for alias, da in d.groupby("alias"):
    n_i = da.groupby("task_id").y.transform("size")
    th_ll, _ = fit_family("log-logistic (METR)", da["T"].values, da["y"].values.astype(float), (1.0 / n_i).values)
    tt = task_table(da)
    tt["f"] = S_loglogistic(th_ll, tt["T"].values)
    tt = tt[tt.n >= 2].copy()
    tt["bin"] = pd.cut(tt["T"], EDGES, labels=LABELS, right=False)
    for b, g in list(tt.groupby("bin", observed=True)) + [("all", tt)]:
        if len(g) < 8:
            continue
        Pbar, H, M2, p2 = H_stats(g)
        H_len = g.f.var(ddof=0) / (Pbar * (1 - Pbar)) if 0 < Pbar < 1 else np.nan
        if not (0.03 < Pbar < 0.97):
            rows.append(dict(alias=alias, bin=b, m=len(g), Pbar=Pbar, H=np.nan, lo=np.nan, hi=np.nan,
                             pass2_emp=p2, pass2_homog=1 - (1 - Pbar) ** 2))
            continue
        bs = []
        idx = np.arange(len(g))
        for _ in range(500):
            gb = g.iloc[rng.choice(idx, len(idx))]
            bs.append(H_stats(gb)[1])
        rows.append(dict(alias=alias, bin=b, m=len(g), Pbar=Pbar, H=H, H_len=H_len,
                         lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                         pass2_emp=p2, pass2_homog=1 - (1 - Pbar) ** 2))
R = pd.DataFrame(rows)
R.to_csv(f"{RES}/heterogeneity_index.csv", index=False)

# Pooled over agents, per length stratum (task-count-weighted mean of H where defined)
Rv = R[(R.bin != "all") & R.H.notna()]
pool = Rv.groupby("bin").apply(lambda g: pd.Series(dict(
    n_agent_strata=len(g), H_mean=np.average(g.H, weights=g.m), H_median=g.H.median(),
    frac_H_gt_half=np.mean(g.H > 0.5), H_len_mean=np.average(g.H_len, weights=g.m)))).reindex(LABELS)
allr = R[(R.bin == "all") & R.H.notna()]
with open(f"{RES}/heterogeneity_summary.txt", "w") as f:
    f.write("H by length stratum, pooled over agents (strata with 0.03<Pbar<0.97, >=8 tasks)\n")
    f.write(pool.to_string() + "\n\n")
    f.write("Agent-level H over all strata (includes between-length variation, upper bound)\n")
    f.write(allr[["alias", "m", "Pbar", "H", "lo", "hi"]].to_string(index=False) + "\n\n")
    f.write(f"Overall strata-level H: median {Rv.H.median():.3f}, IQR "
            f"[{Rv.H.quantile(.25):.3f}, {Rv.H.quantile(.75):.3f}], n={len(Rv)} agent-strata\n")
    f.write(f"Fraction of agent-strata whose 95% CI excludes 0: {np.mean(Rv.lo > 0):.3f}\n")
    gain_emp = (Rv.pass2_emp - Rv.Pbar).sum()
    gain_hom = (Rv.pass2_homog - Rv.Pbar).sum()
    f.write(f"pass@2 gain over pass@1, summed over agent-strata: empirical {gain_emp:.3f} vs "
            f"homogeneous prediction {gain_hom:.3f} (ratio {gain_emp / gain_hom:.3f})\n")
print(open(f"{RES}/heterogeneity_summary.txt").read())

# Figure: H vs stratum for each agent, coloured by agent capability (overall Pbar)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs = {b: i for i, b in enumerate(LABELS)}
for alias, g in Rv.groupby("alias"):
    x = [xs[b] for b in g.bin]
    ax[0].plot(x, g.H, "-o", ms=3, lw=0.8, alpha=0.6, color="#4a6fa5")
ax[0].plot(range(len(LABELS)), pool.H_median.values, "-s", color="#c0392b", lw=2.2, label="median over agents")
ax[0].set_xticks(range(len(LABELS)))
ax[0].set_xticklabels(LABELS)
ax[0].set_ylim(-0.1, 1.05)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xlabel("human task length stratum")
ax[0].set_ylabel("between-task heterogeneity H")
ax[0].legend(frameon=False)
ax[0].set_title("(a) H by length stratum (20 agents)")
ax[1].scatter(Rv.pass2_homog - Rv.Pbar, Rv.pass2_emp - Rv.Pbar, s=12, color="#4a6fa5", alpha=0.7)
mx = (Rv.pass2_homog - Rv.Pbar).max()
ax[1].plot([0, mx], [0, mx], "k--", lw=0.8, label="homogeneous tasks (H=0)")
ax[1].plot([0, mx], [0, 0], "k:", lw=0.8, label="deterministic tasks (H=1)")
ax[1].set_xlabel("pass@2 - pass@1 predicted if tasks homogeneous")
ax[1].set_ylabel("pass@2 - pass@1 observed")
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) retry gain: observed vs homogeneous")
plt.tight_layout()
plt.savefig(f"{RES}/fig_heterogeneity.png", dpi=160)
