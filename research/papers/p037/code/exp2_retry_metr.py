"""Exp 2: Retry (best-of-K) extrapolation on real agent data (METR TH1.1).

An evaluator observes m runs per task; an attacker makes K > m attempts.
We compare estimators/bounds of suite-level pass@K, computed from the m
evaluation runs, against an unbiased estimate from disjoint held-out runs.

Part A (validation, K small enough to check):   results/exp2_validation.csv
Part B (extrapolation to large K, no ground truth): results/exp2_extrapolation.csv
"""
import csv
import json
import os

import numpy as np

from common import (RESULTS, beta_pass_at_k, by_model_task, count_pmf, fit_beta_binomial,
                    load_runs, lp_bounds, pass_at_k_unbiased, pmf_confidence_box)

rng = np.random.default_rng(2)
CONFIGS = [(1, [2, 3], 4), (2, [3, 4], 6), (2, [3, 4, 5, 6], 8)]  # (m, Ks, min runs)
SPLITS = 60
BOOT = 300
ALPHA = 0.10
MIN_TASKS = 40


def one_split(tasks, m, Ks):
    counts, fams, truth = [], [], {K: [] for K in Ks}
    for t in tasks:
        y = np.array(t["y"])
        perm = rng.permutation(len(y))
        ev, ho = y[perm[:m]], y[perm[m:]]
        counts.append(int(ev.sum()))
        fams.append(t["family"])
        for K in Ks:
            truth[K].append(pass_at_k_unbiased(len(ho), int(ho.sum()), K))
    counts = np.array(counts)
    pm = count_pmf(counts, m)
    pbar = counts.mean() / m
    a, b = fit_beta_binomial(counts, m)
    lo, hi = pmf_confidence_box(counts, fams, m, ALPHA, rng, B=BOOT)
    rows = []
    for K in Ks:
        Lp, Up = lp_bounds(pm, pm, m, K)
        Lc, Uc = lp_bounds(lo, hi, m, K)
        rows.append(dict(K=K, truth=float(np.mean(truth[K])),
                         plugin_task=float(np.mean(1 - (1 - counts / m) ** K)),
                         homogeneous=float(1 - (1 - pbar) ** K),
                         beta_binomial=float(beta_pass_at_k(a, b, K)),
                         lp_point_L=Lp, lp_point_U=Up, lp_ci_L=Lc, lp_ci_U=Uc,
                         pass_at_m=float(np.mean([pass_at_k_unbiased(m, c, m) for c in counts]))))
    return rows


def part_a(data):
    out = []
    for m, Ks, nmin in CONFIGS:
        for model, tdict in sorted(data.items()):
            tasks = [t for t in tdict.values() if len(t["y"]) >= nmin]
            if len(tasks) < MIN_TASKS:
                continue
            for s in range(SPLITS):
                for r in one_split(tasks, m, Ks):
                    r.update(model=model, m=m, nmin=nmin, n_tasks=len(tasks), split=s)
                    out.append(r)
        print("done config", m, Ks, nmin, flush=True)
    keys = ["model", "m", "nmin", "n_tasks", "split", "K", "truth", "plugin_task", "homogeneous",
            "beta_binomial", "lp_point_L", "lp_point_U", "lp_ci_L", "lp_ci_U", "pass_at_m"]
    with open(os.path.join(RESULTS, "exp2_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: r[k] for k in keys})
    # summary: mean signed error over splits (unbiased comparison) and coverage of truth
    summ = []
    for m, Ks, nmin in CONFIGS:
        for K in Ks:
            rs = [r for r in out if r["m"] == m and r["nmin"] == nmin and r["K"] == K]
            if not rs:
                continue
            models = sorted(set(r["model"] for r in rs))
            row = dict(m=m, K=K, nmin=nmin, n_models=len(models))
            # per-model mean over splits, then summarise across models
            for meth in ["plugin_task", "homogeneous", "beta_binomial", "lp_point_U", "lp_ci_U"]:
                errs, under = [], []
                for mo in models:
                    rr = [r for r in rs if r["model"] == mo and not np.isnan(r[meth])]
                    if not rr:
                        continue
                    errs.append(np.mean([r[meth] - r["truth"] for r in rr]))
                    under.append(np.mean([r[meth] < r["truth"] for r in rr]))
                row[f"{meth}_mean_err"] = float(np.mean(errs))
                row[f"{meth}_frac_models_biased_low"] = float(np.mean(np.array(errs) < 0))
                row[f"{meth}_undercover_rate"] = float(np.mean(under))
            row["lp_point_feasible_rate"] = float(np.mean([not np.isnan(r["lp_point_U"]) for r in rs]))
            row["lp_ci_width_mean"] = float(np.nanmean([r["lp_ci_U"] - r["lp_ci_L"] for r in rs]))
            row["lp_ci_contains_truth"] = float(np.nanmean(
                [(r["lp_ci_L"] <= r["truth"] <= r["lp_ci_U"]) for r in rs if not np.isnan(r["lp_ci_U"])]))
            summ.append(row)
    with open(os.path.join(RESULTS, "exp2_validation_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


def part_b(data):
    """Extrapolate from m=4 runs (tasks with >=4 runs, first 4 runs in random order)."""
    m = 4
    Ks = [1, 4, 10, 30, 100, 1000]
    out = []
    for model, tdict in sorted(data.items()):
        tasks = [t for t in tdict.values() if len(t["y"]) >= m]
        counts = np.array([int(np.array(t["y"])[rng.permutation(len(t["y"]))[:m]].sum()) for t in tasks])
        fams = [t["family"] for t in tasks]
        pm = count_pmf(counts, m)
        a, b = fit_beta_binomial(counts, m)
        lo, hi = pmf_confidence_box(counts, fams, m, ALPHA, rng, B=1000)
        for K in Ks:
            Lc, Uc = lp_bounds(lo, hi, m, K)
            Lp, Up = lp_bounds(pm, pm, m, K)
            out.append(dict(model=model, n_tasks=len(tasks), K=K,
                            frac_zero=float(pm[0]),
                            beta_binomial=float(beta_pass_at_k(a, b, K)),
                            lp_point_L=Lp, lp_point_U=Up, lp_ci_L=Lc, lp_ci_U=Uc))
    keys = list(out[0].keys())
    with open(os.path.join(RESULTS, "exp2_extrapolation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow(r)
    for r in out:
        if r["K"] in (1, 100, 1000):
            print(f"{r['model']:36s} K={r['K']:5d} zero={r['frac_zero']:.3f} beta={r['beta_binomial']:.3f} "
                  f"point=[{r['lp_point_L']:.3f},{r['lp_point_U']:.3f}] ci=[{r['lp_ci_L']:.3f},{r['lp_ci_U']:.3f}]")


if __name__ == "__main__":
    data = by_model_task(load_runs())
    part_b(data)
    part_a(data)
