"""Task-validity audit + joint crossed agent x task frailty model (addresses review r1).

Outputs
  results/task_audit.csv            per-task audit table (never-solved, estimated length, near-threshold scores)
  results/crossed_agents.csv        per-agent k_a, Wald s.e., lam_a (shared), tau_a (agent-specific) per variant
  results/crossed_profile.csv       profile log-likelihood for a common k across agents
  results/crossed_invalid_post.csv  posterior P(task is in the 'invalid' class)
  results/crossed_summary.txt       summary
"""
import numpy as np
import pandas as pd
import torch
from common import *
from crossed import Crossed, build_arrays

torch.manual_seed(0)
d = load_runs()

# ----------------------------------------------------------------- task-validity audit
g = d.groupby("task_id").agg(T=("T", "first"), family=("task_family", "first"), source=("task_source", "first"),
                             human_source=("human_source", "first"), human_score=("human_score", "first"),
                             n=("y", "size"), s=("y", "sum"), score_cont_max=("score_cont", "max"),
                             score_cont_mean=("score_cont", "mean"))
g["agents_solving"] = d[d.y == 1].groupby("task_id").alias.nunique().reindex(g.index).fillna(0).astype(int)
g["never_solved"] = g.s == 0
g["always_solved"] = g.s == g.n
g["estimated_T"] = g.human_source != "baseline"
g["near_threshold"] = g.never_solved & (g.score_cont_max >= 0.8)
g["all_zero_cont"] = g.never_solved & (g.score_cont_max <= 0.02)
g.to_csv(f"{RES}/task_audit.csv")
ns = g[g.never_solved]
audit_lines = [
    f"Tasks: {len(g)}; never solved by any of 20 agents: {len(ns)} ({(ns['T'] >= 240).sum()} with T >= 4 h)",
    f"  of which length is an estimate (not a human baseline): {ns.estimated_T.sum()} / {len(ns)}",
    f"  (base rate of estimated lengths among all tasks: {g.estimated_T.sum()} / {len(g)};"
    f" among tasks >= 4 h: {g[g['T'] >= 240].estimated_T.sum()} / {(g['T'] >= 240).sum()})",
    f"  never solved but best continuous score >= 0.8 (near threshold): {ns.near_threshold.sum()}",
    f"  never solved with continuous score <= 0.02 in every run: {ns.all_zero_cont.sum()}",
    f"  never solved and shorter than 1 h: {(ns['T'] < 60).sum()}",
    f"  sources of never-solved tasks: {ns.source.value_counts().to_dict()}",
    f"Always solved by every run of every agent: {g.always_solved.sum()}",
    f"RE-Bench tasks: {(g.source == 'RE-Bench').sum()}; never solved among them: {(ns.source == 'RE-Bench').sum()}",
]

# ----------------------------------------------------------------- crossed model variants
ALL = sorted(d.alias.unique())
tasks_all = sorted(d.task_id.unique())
never = set(g.index[g.never_solved])
always = set(g.index[g.always_solved])
baseline = set(g.index[~g.estimated_T])
VARIANTS = {
    "gauss": dict(tasks=tasks_all, G="gauss", invalid=False),
    "gauss+invalid": dict(tasks=tasks_all, G="gauss", invalid=True),
    "discrete": dict(tasks=tasks_all, G="discrete", invalid=False),
    "discrete+invalid": dict(tasks=tasks_all, G="discrete", invalid=True),
    "gauss, baseline-timed only": dict(tasks=[t for t in tasks_all if t in baseline], G="gauss", invalid=False),
    "gauss, drop never-solved*": dict(tasks=[t for t in tasks_all if t not in never], G="gauss", invalid=False),
    "gauss, drop never- and always-solved*": dict(tasks=[t for t in tasks_all if t not in never | always],
                                                  G="gauss", invalid=False),
}
GRID = np.linspace(-5, 5, 21)
rows, summ, prof_rows, post_rows = [], [], [], []
base_fit = {}
for vname, cfg in VARIANTS.items():
    agents, tasks, n, s, T = build_arrays(d, ALL, cfg["tasks"])
    m = Crossed(n, s, T, G=cfg["G"], invalid=cfg["invalid"], grid=GRID)
    th0 = m.init()
    if "gauss" in base_fit and cfg["tasks"] is tasks_all:
        gb = base_fit["gauss"]
        for key in ["mu", "k", "loglam", "logtau"]:
            th0[m.sl[key]] = gb[key]
        if cfg["G"] == "discrete":
            lw = -0.5 * GRID ** 2
            th0[m.sl["wlog"]] = torch.tensor(lw[1:] - lw[0])
    best = None
    for jitter in [0.0, 0.3]:
        t0 = th0.clone()
        t0[m.sl["k"]] = t0[m.sl["k"]] * (1 - jitter)
        th, f = m.fit(t0)
        if best is None or f < best[1]:
            best = (th, f)
    th, f = best
    if vname == "gauss":
        base_fit["gauss"] = {key: th[m.sl[key]].clone() for key in ["mu", "k", "loglam", "logtau"]}
    k = th[m.sl["k"]].numpy()
    lam, tau = np.exp(th[m.sl["loglam"]].numpy()), np.exp(th[m.sl["logtau"]].numpy())
    if cfg["G"] == "discrete":  # report lam on the scale of a unit-variance shared effect
        w = torch.exp(m.lwu(th)).numpy()
        sdG = np.sqrt(np.sum(w * GRID ** 2) - np.sum(w * GRID) ** 2)
        lam = lam * sdG
    se = m.wald_se(th)["k"] if cfg["G"] == "gauss" else np.full(len(k), np.nan)
    for a, ka, sa, la, ta in zip(agents, k, se, lam, tau):
        rows.append(dict(variant=vname, alias=a, k=ka, k_se=sa, lam_shared=la, tau_agent=ta,
                         share_agent_specific=ta ** 2 / (ta ** 2 + la ** 2)))
    pi0 = float(torch.sigmoid(th[m.sl["logit_pi0"]])[0]) if cfg["invalid"] else np.nan
    if cfg["invalid"]:
        post = m.task_posterior_invalid(th)
        for t, p in zip(tasks, post):
            if p > 1e-4:
                post_rows.append(dict(variant=vname, task_id=t, T=g.loc[t, "T"], post_invalid=p,
                                      estimated_T=g.loc[t, "estimated_T"]))
    # common-k fit and profile likelihood (Gaussian / discrete main variants)
    ck = dict(k_common=np.nan, k_common_lo=np.nan, k_common_hi=np.nan)
    if vname in ("gauss", "gauss+invalid", "discrete", "gauss, baseline-timed only"):
        mc = Crossed(n, s, T, G=cfg["G"], invalid=cfg["invalid"], grid=GRID, common_k=True)
        tc = th.clone()
        tc[m.sl["k"]] = float(np.median(k))
        thc, fc = mc.fit(tc)
        kc = float(thc[mc.sl["k"]][0])
        prof = []
        tw = thc.clone()
        for kk in np.round(np.arange(0.6, 2.41, 0.1), 2):
            mf = Crossed(n, s, T, G=cfg["G"], invalid=cfg["invalid"], grid=GRID, fixed_k=kk)
            tw2, fk = mf.fit(tw, iters=200, outer=2)
            tw = tw2
            prof.append((kk, fk))
            prof_rows.append(dict(variant=vname, k=kk, nll=fk))
        P = np.array(prof)
        fmin = min(fc, P[:, 1].min())
        dev = 2 * (P[:, 1] - fmin) - 3.84
        lo_i = [j for j in range(1, len(P)) if dev[j - 1] > 0 >= dev[j]]
        hi_i = [j for j in range(1, len(P)) if dev[j - 1] <= 0 < dev[j]]
        cross = lambda j: P[j - 1, 0] + (P[j, 0] - P[j - 1, 0]) * dev[j - 1] / (dev[j - 1] - dev[j])
        ck = dict(k_common=kc, k_common_lo=cross(lo_i[0]) if lo_i else np.nan,
                  k_common_hi=cross(hi_i[-1]) if hi_i else np.nan,
                  LR_common_vs_free=2 * (fc - f))
    kd = pd.Series(k, index=agents)
    summ.append(dict(variant=vname, ntask=len(tasks), nll=f, pi0_invalid=pi0, k_median=np.median(k),
                     k_q25=np.quantile(k, .25), k_q75=np.quantile(k, .75), n_k_gt_1=int((k > 1).sum()),
                     n_ci_above_1=int(((k - 1.96 * se) > 1).sum()) if cfg["G"] == "gauss" else -1,
                     n_ci_below_1=int(((k + 1.96 * se) < 1).sum()) if cfg["G"] == "gauss" else -1,
                     lam_median=np.median(lam), tau_median=np.median(tau),
                     share_agent_specific_median=np.median(tau ** 2 / (tau ** 2 + lam ** 2)),
                     k_opus46=kd.get("Claude Opus 4.6 (Inspect)"), k_gpt52=kd.get("GPT-5.2"),
                     k_gpt53=kd.get("GPT-5.3-Codex"), k_opus45=kd.get("Claude Opus 4.5 (Inspect)"), **ck))
    print(vname, round(f, 2), "median k", round(np.median(k), 3), ck, flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/crossed_agents.csv", index=False)
pd.DataFrame(prof_rows).to_csv(f"{RES}/crossed_profile.csv", index=False)
pd.DataFrame(post_rows).to_csv(f"{RES}/crossed_invalid_post.csv", index=False)
S = pd.DataFrame(summ).set_index("variant")
with open(f"{RES}/crossed_summary.txt", "w") as fh:
    fh.write("TASK-VALIDITY AUDIT\n" + "\n".join(audit_lines) + "\n\n")
    fh.write("Never-solved tasks:\n" + ns[["T", "family", "source", "human_source", "score_cont_max",
                                            "near_threshold", "all_zero_cont"]].sort_values("T").to_string() + "\n\n")
    fh.write("CROSSED AGENT x TASK MODEL (cloglog; * = outcome-selected filter)\n")
    pd.set_option("display.width", 250)
    fh.write(S.round(3).to_string() + "\n")
print(open(f"{RES}/crossed_summary.txt").read())
