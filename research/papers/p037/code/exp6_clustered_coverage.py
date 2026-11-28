"""Exp 6 (revision 1): coverage of the LP confidence upper bound on suite pass@K under
genuine cluster sampling (task families of unequal size, correlated within-family
difficulty), including the all-fail ("rule-out") regime.

Design
  Structures:  "METR-like": the 79 real METR TH1.1 family sizes (228 tasks), m = 4.
               "CTF-like":  40 tasks in 14 families of sizes 8,6,5,4,3,3,2,2,2,1,1,1,1,1, m = 5.
               "Skewed":    100 tasks: one family of 60 plus 40 singletons, m = 4 (Kish n_eff 2.8).
  Difficulty:  logit p_t = mu + u_f + e_t, u_f ~ N(0, s_f^2), e_t ~ N(0, s_t^2); in some
               scenarios whole families are impossible (p = 0) with probability q.
               S1 moderate        mu=-0.5, s_f=1.5, s_t=1.0, q=0
               S2 rare-tail       mu=-3.5, s_f=2.0, s_t=1.5, q=0
               S3 clustered-zero  mu=-1.0, s_f=1.5, s_t=1.0, q=0.6
               S4 near-threshold  mu=-5.0, s_f=1.0, s_t=1.0, q=0.85
               S5 sparse-rare     mu=-4.0, s_f=0.5, s_t=0.5, q=0.90  (few solvable families;
                                  which ones, and their size, dominate the count pmf)
               S6 sparse-solvable mu=-1.0, s_f=0.5, s_t=0.5, q=0.95
  Targets:     population theta_K = E[1-(1-p)^K] (the estimand of the LP over a box for the
               population count pmf; Monte Carlo, 2e6 draws) and the finite-suite value
               mean_t [1-(1-p_t)^K] for the drawn tasks.
  Boxes (all 90% simultaneous, Bonferroni over m+1 cells; family-cluster bootstrap B=400):
               boot   : percentile cluster bootstrap only
               hyb_nc : hull with Clopper-Pearson using n = number of families (the paper's box)
               hyb_kish: hull with Clopper-Pearson using Kish's effective number of families
                        n_eff = (sum n_f)^2 / sum n_f^2  (32.2 for METR sizes, 7.7 for CTF-like)
               cp_nc  : the Clopper-Pearson part alone (diagnostic, not recommended)
  300 replications per (structure, scenario).
Outputs: results/exp6_clustered_coverage.csv (per replication), results/exp6_summary.json
"""
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.special import expit

from common import RESULTS, by_model_task, clopper_pearson, count_pmf, load_runs, lp_bounds

KS = [1, 10, 100, 1000]
ALPHA = 0.10
B = 400
REPS = int(os.environ.get("EXP6_REPS", 300))
SCEN = {"S1_moderate": (-0.5, 1.5, 1.0, 0.0), "S2_rare_tail": (-3.5, 2.0, 1.5, 0.0),
        "S3_clustered_zero": (-1.0, 1.5, 1.0, 0.6), "S4_near_threshold": (-5.0, 1.0, 1.0, 0.85),
        "S5_sparse_rare": (-4.0, 0.5, 0.5, 0.90), "S6_sparse_solvable": (-1.0, 0.5, 0.5, 0.95)}


def metr_sizes():
    rows = load_runs()
    fam = {}
    for r in rows:
        fam[r["task_id"]] = r["task_family"]
    from collections import Counter
    return sorted(Counter(fam.values()).values(), reverse=True)


STRUCT = {"METR_like": (None, 4), "CTF_like": ([8, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1], 5),
          "Skewed": ([60] + [1] * 40, 4)}


def draw_p(rng, sizes, scen):
    mu, sf, st, q = SCEN[scen]
    ps, labs = [], []
    for f, n in enumerate(sizes):
        if rng.random() < q:
            p = np.zeros(n)
        else:
            p = expit(mu + sf * rng.standard_normal() + st * rng.standard_normal(n))
        ps.append(p)
        labs += [f] * n
    return np.concatenate(ps), np.array(labs)


def pop_truth(scen, rng, N=2_000_000):
    mu, sf, st, q = SCEN[scen]
    p = expit(mu + sf * rng.standard_normal(N) + st * rng.standard_normal(N))
    p[rng.random(N) < q] = 0.0
    return {K: float(np.mean(1 - (1 - p) ** K)) for K in KS}


def boxes(counts, labs, m, rng):
    ul = np.unique(labs)
    idx = [np.where(labs == l)[0] for l in ul]
    sizes = np.array([len(i) for i in idx])
    boots = np.empty((B, m + 1))
    for b in range(B):
        pick = rng.integers(0, len(ul), len(ul))
        boots[b] = count_pmf(counts[np.concatenate([idx[i] for i in pick])], m)
    a = ALPHA / (m + 1)
    lo = np.quantile(boots, a / 2, axis=0)
    hi = np.quantile(boots, 1 - a / 2, axis=0)
    ph = count_pmf(counts, m)
    out = {"boot": (lo, hi)}
    for name, n in (("hyb_nc", len(ul)), ("hyb_kish", sizes.sum() ** 2 / np.sum(sizes ** 2))):
        cp = np.array([clopper_pearson(ph[c] * n, n, a) for c in range(m + 1)])
        out[name] = (np.minimum(lo, cp[:, 0]), np.maximum(hi, cp[:, 1]))
        if name == "hyb_nc":
            out["cp_nc"] = (cp[:, 0], cp[:, 1])  # exact part alone (diagnostic)
    return out


_PMF = {}


def pop_pmf(scen, m, N=1_000_000):
    """Population count pmf E[Binom(m, p)] (Monte Carlo, fixed seed), cached per worker."""
    if (scen, m) not in _PMF:
        from scipy.stats import binom
        r = np.random.default_rng([99, list(SCEN).index(scen)])
        mu, sf, st, q = SCEN[scen]
        p = expit(mu + sf * r.standard_normal(N) + st * r.standard_normal(N))
        p[r.random(N) < q] = 0.0
        _PMF[(scen, m)] = binom.pmf(np.arange(m + 1)[:, None], m, p[None, :]).mean(axis=1)
    return _PMF[(scen, m)]


def one(args):
    struct, scen, rep, sizes, m = args
    rng = np.random.default_rng([7, list(STRUCT).index(struct), list(SCEN).index(scen), rep])
    p, labs = draw_p(rng, sizes, scen)
    counts = rng.binomial(m, p)
    bx = boxes(counts, labs, m, rng)
    row = dict(structure=struct, scenario=scen, rep=rep, all_zero=int(counts.sum() == 0),
               frac_zero=float(np.mean(counts == 0)))
    pmf = pop_pmf(scen, m)
    for name, (lo, hi) in bx.items():
        inside = (lo <= pmf + 1e-12) & (pmf <= hi + 1e-12)
        row[f"box_simul_{name}"] = int(inside.all())
        row[f"box_cells_{name}"] = float(inside.mean())
    for K in KS:
        row[f"fin_truth_K{K}"] = float(np.mean(1 - (1 - p) ** K))
        for name, (lo, hi) in bx.items():
            row[f"U_{name}_K{K}"] = lp_bounds(lo, hi, m, K)[1]
    return row


def main():
    sizes_metr = metr_sizes()
    jobs = []
    for struct, (sz, m) in STRUCT.items():
        sizes = sizes_metr if sz is None else sz
        for scen in SCEN:
            for rep in range(REPS):
                jobs.append((struct, scen, rep, sizes, m))
    with Pool(4) as pool:
        rows = pool.map(one, jobs, chunksize=10)
    rng = np.random.default_rng(77)
    truth = {s: pop_truth(s, rng) for s in SCEN}
    with open(os.path.join(RESULTS, "exp6_clustered_coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for struct in STRUCT:
        for scen in SCEN:
            rr = [r for r in rows if r["structure"] == struct and r["scenario"] == scen]
            d = dict(structure=struct, scenario=scen, n_reps=len(rr),
                     frac_all_zero=float(np.mean([r["all_zero"] for r in rr])),
                     mean_frac_zero_tasks=float(np.mean([r["frac_zero"] for r in rr])))
            for name in ("boot", "cp_nc", "hyb_nc", "hyb_kish"):
                d[f"box_simul_cov_{name}"] = float(np.mean([r[f"box_simul_{name}"] for r in rr]))
            for K in KS:
                d[f"pop_truth_K{K}"] = truth[scen][K]
                for name in ("boot", "cp_nc", "hyb_nc", "hyb_kish"):
                    U = np.array([r[f"U_{name}_K{K}"] for r in rr])
                    fin = np.array([r[f"fin_truth_K{K}"] for r in rr])
                    d[f"cov_pop_{name}_K{K}"] = float(np.mean(U >= truth[scen][K] - 1e-9))
                    d[f"cov_fin_{name}_K{K}"] = float(np.mean(U >= fin - 1e-9))
                    d[f"meanU_{name}_K{K}"] = float(np.mean(U))
                    az = np.array([r["all_zero"] for r in rr]) == 1
                    if az.any():
                        d[f"cov_pop_allzero_{name}_K{K}"] = float(np.mean(U[az] >= truth[scen][K] - 1e-9))
            summ.append(d)
    with open(os.path.join(RESULTS, "exp6_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    for d in summ:
        print(d["structure"], d["scenario"], "allzero=%.2f" % d["frac_all_zero"],
              "box-simul: " + " ".join(f"{n}={d['box_simul_cov_' + n]:.2f}" for n in ("boot", "cp_nc", "hyb_nc", "hyb_kish")),
              " ".join(f"K{K}: truth={d[f'pop_truth_K{K}']:.3f} boot={d[f'cov_pop_boot_K{K}']:.2f} "
                       f"cp={d[f'cov_pop_cp_nc_K{K}']:.2f} nc={d[f'cov_pop_hyb_nc_K{K}']:.2f} kish={d[f'cov_pop_hyb_kish_K{K}']:.2f}"
                       for K in KS))


if __name__ == "__main__":
    main()
