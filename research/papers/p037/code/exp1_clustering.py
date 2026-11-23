"""Exp 1: How much does task/family clustering inflate uncertainty of a suite-level
success rate, relative to the naive 'all runs i.i.d.' binomial standard error?

Data: METR Time-Horizon 1.1 public runs. Estimand: equal-task-weighted mean success.
Output: results/exp1_clustering.csv, results/exp1_summary.json
"""
import csv
import json
import os

import numpy as np

from common import RESULTS, by_model_task, load_runs

rng = np.random.default_rng(1)
B = 2000


def main():
    data = by_model_task(load_runs())
    out = []
    for model, tasks in sorted(data.items()):
        tids = sorted(tasks)
        tmeans = np.array([np.mean(tasks[t]["y"]) for t in tids])
        fams = np.array([tasks[t]["family"] for t in tids])
        nruns = np.array([len(tasks[t]["y"]) for t in tids])
        N = nruns.sum()
        est = tmeans.mean()
        allruns = np.concatenate([tasks[t]["y"] for t in tids])
        se_naive = np.sqrt(allruns.mean() * (1 - allruns.mean()) / N)
        # task-level bootstrap
        bt = np.array([tmeans[rng.integers(0, len(tids), len(tids))].mean() for _ in range(B)])
        # family-level bootstrap
        ufam = np.unique(fams)
        fidx = [np.where(fams == f)[0] for f in ufam]
        bf = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(ufam), len(ufam))
            ii = np.concatenate([fidx[i] for i in pick])
            bf[b] = tmeans[ii].mean()
        se_task, se_fam = bt.std(ddof=1), bf.std(ddof=1)
        # (b) runs-as-replicates SE for the task-weighted estimand (tasks fixed,
        #     only within-task Bernoulli noise): sqrt(sum_t p_t(1-p_t)/r_t)/n
        se_within = np.sqrt(np.sum(tmeans * (1 - tmeans) / np.maximum(nruns - 1, 1))) / len(tids)
        # (c') analytic cluster-robust (CR1) SEs of the task-weighted mean, task and family level
        resid = tmeans - est
        nT = len(tids)
        se_task_cr = np.sqrt(np.sum(resid ** 2) / (nT * (nT - 1)))
        fsum = np.array([resid[i].sum() for i in fidx])
        G = len(ufam)
        se_fam_cr = np.sqrt(G / (G - 1) * np.sum(fsum ** 2)) / nT
        out.append(dict(model=model, n_tasks=len(tids), n_families=len(ufam), n_runs=int(N),
                        mean_success=est, se_naive=se_naive, se_within=se_within,
                        se_task=se_task, se_family=se_fam,
                        se_task_cr1=se_task_cr, se_family_cr1=se_fam_cr,
                        deff_family_vs_task=(se_fam / se_task) ** 2,
                        deff_family_vs_task_cr1=(se_fam_cr / se_task_cr) ** 2,
                        deff_family=(se_fam / se_naive) ** 2,
                        n_eff_family=N * (se_naive / se_fam) ** 2))
    keys = list(out[0].keys())
    with open(os.path.join(RESULTS, "exp1_clustering.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    # restrict summary to models with interior success (0.1<p<0.9) for meaningful SE ratios
    sel = [r for r in out if 0.1 < r["mean_success"] < 0.9]
    ratio = np.array([r["se_family"] / r["se_naive"] for r in sel])
    summ = dict(n_models=len(out), n_models_interior=len(sel),
                se_ratio_family_over_naive_median=float(np.median(ratio)),
                se_ratio_min=float(ratio.min()), se_ratio_max=float(ratio.max()),
                deff_median=float(np.median(ratio ** 2)),
                n_runs_median=float(np.median([r["n_runs"] for r in sel])),
                n_eff_median=float(np.median([r["n_eff_family"] for r in sel])),
                n_families_median=float(np.median([r["n_families"] for r in sel])),
                margin95_naive_median=float(np.median([1.645 * r["se_naive"] for r in sel])),
                margin95_family_median=float(np.median([1.645 * r["se_family"] for r in sel])),
                margin95_task_median=float(np.median([1.645 * r["se_task"] for r in sel])),
                # headline (between-family clustering only): family vs task resampling
                deff_family_vs_task_median=float(np.median([r["deff_family_vs_task"] for r in sel])),
                deff_family_vs_task_min=float(np.min([r["deff_family_vs_task"] for r in sel])),
                deff_family_vs_task_max=float(np.max([r["deff_family_vs_task"] for r in sel])),
                se_ratio_family_over_task_median=float(np.median([r["se_family"] / r["se_task"] for r in sel])),
                deff_family_vs_task_cr1_median=float(np.median([r["deff_family_vs_task_cr1"] for r in sel])),
                n_tasks_median=float(np.median([r["n_tasks"] for r in sel])),
                n_eff_tasks_family_median=float(np.median([r["n_tasks"] / r["deff_family_vs_task"] for r in sel])),
                # decomposition of the naive->family ratio
                se_ratio_within_over_naive_median=float(np.median([r["se_within"] / r["se_naive"] for r in sel])),
                se_ratio_task_over_within_median=float(np.median([r["se_task"] / r["se_within"] for r in sel])),
                se_naive_median=float(np.median([r["se_naive"] for r in sel])),
                se_within_median=float(np.median([r["se_within"] for r in sel])),
                se_task_median=float(np.median([r["se_task"] for r in sel])),
                se_family_median=float(np.median([r["se_family"] for r in sel])))
    with open(os.path.join(RESULTS, "exp1_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    for r in out:
        print(f"{r['model']:40s} p={r['mean_success']:.3f} naive={r['se_naive']:.4f} "
              f"task={r['se_task']:.4f} fam={r['se_family']:.4f} deff={r['deff_family']:.1f} "
              f"neff={r['n_eff_family']:.0f}/{r['n_runs']}")


if __name__ == "__main__":
    main()
