"""Analysis 2: anatomy of failures on short tasks (the high-reliability regime).

For each agent and short-task set (human length < T minutes):
  * failure rate with Clopper-Pearson CI (runs) and task-cluster bootstrap CI
  * classification of failed runs: intermittent (task solved on another run) vs consistent
    (all runs failed), crossed with consensus-hard (pooled leave-one-out success of the
    frontier agents < 50%)
  * unbiased within/between-task variance decomposition and intra-task correlation (ICC)
  * unbiased pass^k (k=1..4), pass@4, and the identified upper bound E[(1-p)^4] >= P(p=0)
Outputs: results/decomp_T{T}.csv, results/consensus_tasks.csv, fig2_decomp.png, fig3_passk.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

rows = load_runs()
agents = sorted(RELEASE, key=date_float)
FRONTIER = [a for a in agents if RELEASE[a] >= "2025-04-01"]
tabs = {a: task_table(rows, a) for a in agents}
rng = np.random.default_rng(1)

# pooled frontier success per task, per agent (leave that agent out)
def pooled_loo(task, exclude):
    k = N = 0
    for a in FRONTIER:
        if a == exclude:
            continue
        T = tabs[a]
        m = T["task"] == task
        if m.any():
            k += T["s"][m][0]; N += T["n"][m][0]
    return k / N if N else np.nan


def comb_ratio(a, b, m):
    """C(a,m)/C(b,m) elementwise (0 if a<m)."""
    out = np.zeros(len(a))
    ok = a >= m
    out[ok] = np.exp(log_comb(a[ok], m) - log_comb(b[ok], m))
    return out


summary = {}
for TH in (4, 15, 60):
    recs = []
    for al in agents:
        T = tabs[al]
        m = T["minutes"] < TH
        n, s, task, fam = T["n"][m], T["s"][m], T["task"][m], T["fam"][m]
        f = n - s
        F = f.sum(); N = n.sum()
        lo, hi = clopper_pearson(F, N)
        # task-level (equal-weight) failure and cluster bootstrap
        p_hat = s / n
        mu_t = p_hat.mean()
        bs = []
        for b in range(2000):
            idx = cluster_bootstrap_indices(fam, rng)
            bs.append(1 - (s[idx] / n[idx]).mean())
        blo, bhi = np.percentile(bs, [2.5, 97.5])
        cons = np.array([pooled_loo(t, al) < 0.5 for t in task])
        consistent = (s == 0) & (f > 0)
        inter = (s > 0) & (f > 0)
        def share(mask):
            return f[mask].sum() / F if F else np.nan
        # variance decomposition (tasks with n>=2)
        k2 = n >= 2
        within = np.mean(s[k2] * (n[k2] - s[k2]) / (n[k2] * (n[k2] - 1)))
        mu2 = (s[k2] / n[k2]).mean()
        # unbiased estimate of E[p^2]: s(s-1)/(n(n-1)); Var(p) = E[p^2]-E[p]^2 (plug-in for mean)
        Ep2 = np.mean(s[k2] * (s[k2] - 1) / (n[k2] * (n[k2] - 1)))
        between = Ep2 - mu2 ** 2
        total = mu2 * (1 - mu2)
        icc = between / total if total > 0 else np.nan
        k4 = n >= 4
        passk = [np.mean(comb_ratio(s[n >= k], n[n >= k], k)) for k in (1, 2, 3, 4)]
        e1mp4 = np.mean(comb_ratio(f[k4], n[k4], 4))
        rec = dict(agent=short(al), date=RELEASE[al], T=TH, tasks=int(m.sum()), runs=int(N), fails=int(F),
                   fail_rate_runs=F / N, cp_lo=lo, cp_hi=hi, fail_rate_tasks=1 - mu_t,
                   boot_lo=blo, boot_hi=bhi,
                   sh_inter_noncons=share(inter & ~cons), sh_inter_cons=share(inter & cons),
                   sh_consist_noncons=share(consistent & ~cons), sh_consist_cons=share(consistent & cons),
                   n_tasks_failed=int((f > 0).sum()), n_tasks_consistent=int(consistent.sum()),
                   within=within, between=between, icc=icc,
                   pass1=passk[0], pass2=passk[1], pass3=passk[2], pass4=passk[3],
                   pass_at4=1 - e1mp4, ub_p0=e1mp4)
        recs.append(rec)
    summary[TH] = recs
    keys = list(recs[0].keys())
    with open(os.path.join(RESULTS, f"decomp_T{TH}.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in recs:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"\n=== tasks < {TH} min ===")
    for r in recs:
        print(f"{r['agent']:26s} n={r['runs']:4d} fail={r['fail_rate_runs']:.4f} [{r['cp_lo']:.4f},{r['cp_hi']:.4f}] "
              f"inter={r['sh_inter_noncons']+r['sh_inter_cons']:.2f} cons={r['sh_consist_cons']+r['sh_inter_cons']:.2f} "
              f"icc={r['icc']:.2f} pass^4={r['pass4']:.3f} pass@4={r['pass_at4']:.3f}")

# consensus-hard short tasks
T0 = tabs[FRONTIER[0]]
allt = {}
for a in FRONTIER:
    T = tabs[a]
    for t, mi, n, s in zip(T["task"], T["minutes"], T["n"], T["s"]):
        d = allt.setdefault(t, [mi, 0, 0, 0]); d[1] += n; d[2] += s; d[3] += int(s == 0)
with open(os.path.join(RESULTS, "consensus_tasks.csv"), "w") as fh:
    fh.write("task,human_minutes,frontier_runs,frontier_success,frontier_agents_all_fail\n")
    for t, (mi, n, s, z) in sorted(allt.items(), key=lambda kv: kv[1][0]):
        if mi < 15 and s / n < 0.9:
            fh.write(f"{t},{mi:.2f},{n},{s/n:.3f},{z}\n")
            print("short task with frontier success<0.9:", t, round(mi, 2), n, round(s / n, 3), z)

# ---- Figure 2: failure decomposition (T=15)
recs = summary[15]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
lab = [r["agent"] for r in recs]
X = np.arange(len(recs))
rate = np.array([r["fail_rate_runs"] for r in recs])
parts = [("sh_inter_noncons", "intermittent, ordinary task", "#9ecae1"),
         ("sh_inter_cons", "intermittent, consensus-hard task", "#3182bd"),
         ("sh_consist_noncons", "consistent, agent-specific", "#fdae6b"),
         ("sh_consist_cons", "consistent, consensus-hard task", "#e6550d")]
bottom = np.zeros(len(recs))
for k, l, c in parts:
    v = rate * np.array([r[k] for r in recs])
    axs[0].bar(X, v, bottom=bottom, color=c, label=l)
    bottom += v
axs[0].errorbar(X, rate, yerr=[rate - [r["cp_lo"] for r in recs], [r["cp_hi"] for r in recs] - rate],
                fmt="none", ecolor="k", capsize=2)
axs[0].set_yscale("log"); axs[0].set_ylim(5e-3, 1)
axs[0].set_xticks(X); axs[0].set_xticklabels(lab, rotation=75, fontsize=7)
axs[0].set_ylabel("run failure rate, tasks < 15 human-min")
axs[0].legend(fontsize=7, loc="upper right")
axs[0].set_title("Failure anatomy in the short-task regime")
axs[1].plot([date_float(a) for a in agents], [r["icc"] for r in recs], "o", color="k")
for a, r in zip(agents, recs):
    if RELEASE[a] >= "2025-04-01":
        axs[1].annotate(r["agent"], (date_float(a), r["icc"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
axs[1].set_ylabel("intra-task correlation of failure (ICC)")
axs[1].set_xlabel("release date"); axs[1].set_ylim(0, 1.05)
axs[1].set_title("Share of failure variance that is task-level")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig2_decomp.png"), dpi=150)

# ---- Figure 3: pass^k on short tasks (T=15)
fig, ax = plt.subplots(figsize=(5.5, 4))
for a, r in zip(agents, recs):
    if RELEASE[a] >= "2025-04-01":
        ax.plot([1, 2, 3, 4], [r[f"pass{k}"] for k in (1, 2, 3, 4)], "o-", label=r["agent"], ms=3)
ax.set_xlabel("k"); ax.set_ylabel("pass^k (unbiased), tasks < 15 min")
ax.set_xticks([1, 2, 3, 4]); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig3_passk.png"), dpi=150)
print("done")
