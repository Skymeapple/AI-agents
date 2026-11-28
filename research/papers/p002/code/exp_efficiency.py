"""Efficiency of the finite-task intervals (revision r1).
(1) WalkSAT: box-interval coverage split into two-sided targets and the one-sided pass@inf cells;
    box half-width vs a CLT interval (unbiased estimator, 1.96*sd/sqrt(M)) where k <= n;
    box width vs the oracle (population) identified width where k > n.
(2) Synthetic: beta-binomial bootstrap coverage under the well-specified law, by M;
    certified-estimator coverage and its half-width relative to the population identified half-width.
Reads results/walksat_raw.csv, results/walksat_*.npz, results/synthetic_summary.csv.
Writes results/efficiency_walksat.csv, results/efficiency_summary.json."""
import sys, os, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ttc import unbiased_passk
from scipy.special import gammaln

RES = os.path.join(os.path.dirname(__file__), "..", "results")
NS = [8, 16, 32, 64]


def per_task_passk(c, n, k):
    lf = gammaln(n - c + 1) - gammaln(n - c - k + 1) - gammaln(n + 1) + gammaln(n - k + 1)
    return 1 - np.where(n - c >= k, np.exp(lf), 0.0)


def main():
    raw = pd.read_csv(os.path.join(RES, "walksat_raw.csv"))
    raw["cover"] = (raw.box_lo - 1e-9 <= raw.truth) & (raw.truth <= raw.box_hi + 1e-9)
    two = raw[raw.target != "pass@inf"]; one = raw[raw.target == "pass@inf"]
    out = dict(box_cells_total=len(raw), box_cover_total=int(raw.cover.sum()),
               two_sided_cells=len(two), two_sided_covered=int(two.cover.sum()),
               pass_inf_cells=len(one), pass_inf_lower_end_covered=int((one.box_lo <= one.truth + 1e-9).sum()))
    # reproduce the counterfactual subsets of exp_walksat.py (same RNG stream) for the CLT comparison
    rng = np.random.default_rng(2026)
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "walksat_*.npz"))):
        d = np.load(f); runs = d["runs"]; name = os.path.basename(f)[8:-4]
        M, N = runs.shape
        for n in NS:
            perm = rng.permutation(N)
            for s in range(10):
                idx = perm[s * n:(s + 1) * n]
                X = runs[:, idx].sum(1).astype(int)
                for k in [32, 100, 300, 1000, 2000]:
                    r = raw[(raw.cfg == name) & (raw.n == n) & (raw.subset == s) & (raw.target == f"pass@{k}")].iloc[0]
                    row = dict(cfg=name, n=n, subset=s, k=k, box_half=(r.box_hi - r.box_lo) / 2,
                               oracle_half=(r.oracle_hi - r.oracle_lo) / 2)
                    if k <= n:
                        v = per_task_passk(X, n, k)
                        row["clt_half"] = 1.96 * v.std(ddof=1) / np.sqrt(M)
                        row["est"] = v.mean(); row["truth"] = r.truth
                    rows.append(row)
    e = pd.DataFrame(rows)
    e.to_csv(os.path.join(RES, "efficiency_walksat.csv"), index=False, float_format="%.5f")
    kn = e[e.k <= e.n]
    out["k_le_n_cells"] = len(kn)
    out["k_le_n_box_half_median"] = float(kn.box_half.median())
    out["k_le_n_clt_half_median"] = float(kn.clt_half.median())
    out["k_le_n_ratio_box_over_clt_median"] = float((kn.box_half / kn.clt_half).median())
    out["k_le_n_ratio_box_over_clt_range"] = [float((kn.box_half / kn.clt_half).min()), float((kn.box_half / kn.clt_half).max())]
    out["k_le_n_clt_coverage"] = float((np.abs(kn.est - kn.truth) <= kn.clt_half).mean())
    kg = e[(e.k > e.n) & (e.oracle_half > 0)]
    tab = kg.groupby(["n", "k"]).apply(lambda d: pd.Series(dict(box_half=d.box_half.median(), oracle_half=d.oracle_half.median(),
                                                                excess=(d.box_half - d.oracle_half).median()))).reset_index()
    out["k_gt_n_box_vs_oracle"] = tab.round(4).to_dict(orient="records")
    # synthetic
    syn = pd.read_csv(os.path.join(RES, "synthetic_summary.csv"))
    b = syn[(syn.law == "Beta(0.3,3)") & syn.bb_cover.notna()]
    out["bb_boot_cover_wellspecified_by_M"] = {int(M): float(d.bb_cover.mean()) for M, d in b.groupby("M")}
    out["bb_boot_cover_wellspecified_by_M_range"] = {int(M): [float(d.bb_cover.min()), float(d.bb_cover.max())] for M, d in b.groupby("M")}
    syn["id_half"] = syn.id_width / 2
    out["cert_half_over_pop_id_half_median"] = float((syn.cert_half / np.maximum(syn.id_half, 1e-3)).median())
    json.dump(out, open(os.path.join(RES, "efficiency_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
