"""Exp 4: How much does a declared 'resolution floor' assumption buy on real data?

For each METR TH1.1 model, take m=4 runs per task (tasks with >=4 runs), build the
family-cluster bootstrap simultaneous 90% box for the count pmf, and compute the
sharp upper bound on suite pass@K under support(G) in {0} U [pmin, 1] for several
pmin, plus no floor. Output: results/exp4_floor.csv, results/exp4_summary.json
"""
import csv
import json
import os

import numpy as np

from common import (RESULTS, beta_pass_at_k, by_model_task, count_pmf, fit_beta_binomial,
                    load_runs, lp_bounds, pmf_confidence_box)

rng = np.random.default_rng(4)
M = 4
KS = [1, 10, 100, 1000]
PMINS = [None, 1e-3, 1e-2, 5e-2, 0.2, 0.3, 0.5]


def main():
    data = by_model_task(load_runs())
    rows = []
    for model, td in sorted(data.items()):
        tasks = [t for t in td.values() if len(t["y"]) >= M]
        counts = np.array([int(np.array(t["y"])[rng.permutation(len(t["y"]))[:M]].sum()) for t in tasks])
        fams = [t["family"] for t in tasks]
        lo, hi = pmf_confidence_box(counts, fams, M, 0.10, rng, B=1000)
        a, b = fit_beta_binomial(counts, M)
        for K in KS:
            P0 = float(count_pmf(counts, M)[0])
            r = dict(model=model, n_tasks=len(tasks), K=K, frac_zero=P0,
                     beta_binomial=beta_pass_at_k(a, b, K))
            for pm in PMINS:
                _, U, s = lp_bounds(lo, hi, M, K, pmin=pm, return_slack=True)
                key = "none" if pm is None else f"{pm:g}"
                r[f"U_floor_{key}"] = U
                r[f"slack_floor_{key}"] = s
                if pm is not None:  # closed-form Cor.: pass@inf <= 1 - G0_min, point P0
                    q = (1 - pm) ** M
                    r[f"cf_passinf_floor_{key}"] = 1 - max(0.0, (P0 - q) / (1 - q))
            rows.append(r)
        print(model, flush=True)
    with open(os.path.join(RESULTS, "exp4_floor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for K in KS:
        rr = [r for r in rows if r["K"] == K]
        summ[f"K={K}"] = {k: float(np.median([r[k] for r in rr]))
                          for k in rr[0] if k.startswith("U_floor") or k.startswith("cf_") or k == "beta_binomial" or k == "frac_zero"}
    with open(os.path.join(RESULTS, "exp4_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
