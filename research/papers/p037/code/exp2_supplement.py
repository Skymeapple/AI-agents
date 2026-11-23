"""Exp 2 supplement (revision 1).

(i)  How often, and by how much, the minimal-slack relaxation was used in Exp. 2
     (separately for the LP at the empirical pmf and for the LP over the confidence box),
     and whether undercoverage of the point bound differs between relaxed and exact splits.
(ii) Sensitivity of the METR m = 4 LP-CI upper bounds to the grid floor (1e-5, 1e-7, 1e-9)
     and grid density, on one fixed draw of 4 runs per task (own seed).
(iii) Validation results aggregated per model (one number per model, so the unit of
     replication is the model rather than the dependent split).

Outputs: results/exp2_supplement.json, results/exp2_grid_sensitivity.csv
"""
import csv
import json
import os

import numpy as np

from common import (RESULTS, by_model_task, count_pmf, default_grid, load_runs, lp_bounds,
                    pmf_confidence_box)


def slack_analysis():
    v = list(csv.DictReader(open(os.path.join(RESULTS, "exp2_validation.csv"))))
    e = list(csv.DictReader(open(os.path.join(RESULTS, "exp2_extrapolation.csv"))))
    out = {}
    sp = np.array([float(r["slack_point"]) for r in v])
    sc = np.array([float(r["slack_ci"]) for r in v])
    m = np.array([int(r["m"]) for r in v])
    under_pt = np.array([float(r["lp_point_U"]) < float(r["truth"]) for r in v])
    out["validation_rows"] = len(v)
    out["validation_frac_point_relaxed"] = float((sp > 0).mean())
    out["validation_frac_point_relaxed_m2"] = float((sp[m == 2] > 0).mean())
    out["validation_max_point_slack"] = float(sp.max())
    out["validation_frac_ci_relaxed"] = float((sc > 0).mean())
    sel2 = m == 2
    out["m2_point_undercover_relaxed"] = float(under_pt[sel2 & (sp > 0)].mean())
    out["m2_point_undercover_exact"] = float(under_pt[sel2 & (sp == 0)].mean())
    se = np.array([float(r["slack_ci"]) for r in e])
    spe = np.array([float(r["slack_point"]) for r in e])
    out["extrap_frac_ci_relaxed"] = float((se > 0).mean())
    out["extrap_frac_point_relaxed"] = float((spe > 0).mean())
    out["extrap_max_point_slack"] = float(spe.max())
    # (iii) per-model aggregation of LP-CI coverage and margin
    per = []
    for (mm, nmin, K) in sorted(set((int(r["m"]), int(r["nmin"]), int(r["K"])) for r in v)):
        rs = [r for r in v if int(r["m"]) == mm and int(r["nmin"]) == nmin and int(r["K"]) == K]
        models = sorted(set(r["model"] for r in rs))
        cov = [np.mean([float(r["lp_ci_U"]) >= float(r["truth"]) for r in rs if r["model"] == mo]) for mo in models]
        marg = [np.mean([float(r["lp_ci_U"]) - float(r["truth"]) for r in rs if r["model"] == mo]) for mo in models]
        mmarg = [np.min([float(r["lp_ci_U"]) - float(r["truth"]) for r in rs if r["model"] == mo]) for mo in models]
        per.append(dict(m=mm, nmin=nmin, K=K, n_models=len(models),
                        min_model_coverage=float(np.min(cov)),
                        median_mean_margin=float(np.median(marg)),
                        min_split_margin=float(np.min(mmarg))))
    out["per_model"] = per
    return out


def grid_sensitivity():
    rng = np.random.default_rng(22)
    data = by_model_task(load_runs())
    m, alpha = 4, 0.10
    grids = {"J600_floor1e-5": default_grid(600, 1e-5), "J600_floor1e-7": default_grid(600, 1e-7),
             "J600_floor1e-9": default_grid(600, 1e-9), "J2000_floor1e-9": default_grid(2000, 1e-9)}
    rows = []
    for model, tdict in sorted(data.items()):
        tasks = [t for t in tdict.values() if len(t["y"]) >= m]
        counts = np.array([int(np.array(t["y"])[rng.permutation(len(t["y"]))[:m]].sum()) for t in tasks])
        fams = [t["family"] for t in tasks]
        lo, hi = pmf_confidence_box(counts, fams, m, alpha, rng, B=1000)
        pm = count_pmf(counts, m)
        for K in (10, 100, 1000, 10000, 100000):
            r = dict(model=model, K=K)
            for gname, g in grids.items():
                L, U = lp_bounds(lo, hi, m, K, pgrid=g)
                Lp, Up = lp_bounds(pm, pm, m, K, pgrid=g)
                r[f"U_ci_{gname}"] = U
                r[f"L_ci_{gname}"] = L
                r[f"U_point_{gname}"] = Up
            rows.append(r)
    with open(os.path.join(RESULTS, "exp2_grid_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for K in (10, 100, 1000, 10000, 100000):
        rr = [r for r in rows if r["K"] == K]
        base = np.array([r["U_ci_J600_floor1e-7"] for r in rr])
        bp = np.array([r["U_point_J600_floor1e-7"] for r in rr])
        d = {}
        for gname in grids:
            d[f"max_abs_diff_U_ci_{gname}"] = float(np.max(np.abs(np.array([r[f"U_ci_{gname}"] for r in rr]) - base)))
            d[f"max_abs_diff_U_point_{gname}"] = float(np.max(np.abs(np.array([r[f"U_point_{gname}"] for r in rr]) - bp)))
        d["min_U_ci_base"] = float(base.min())
        d["median_U_point_base"] = float(np.median(bp))
        summ[f"K={K}"] = d
    return summ


if __name__ == "__main__":
    out = {"slack": slack_analysis(), "grid_sensitivity": grid_sensitivity()}
    with open(os.path.join(RESULTS, "exp2_supplement.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
