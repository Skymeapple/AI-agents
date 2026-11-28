"""Per-system empirical-Bayes shrinkage (random-effects variance component) for family offsets.

Treat d_{m,f} ~ N(0, 1/lambda_m) as random effects and choose lambda_m per system by maximising the Laplace
approximation of the (weighted pseudo-) marginal likelihood
    log p(y | lambda) ~ l(theta_hat) - lambda/2 ||d_hat||^2 + (F/2) log lambda - 1/2 log det H(theta_hat)
(flat prior on a_m, b_m; constants dropped). This is the GLMM / REML-style alternative the reviewer asked for to a
single global ridge on weight-normalised runs. We then recompute the 24-definition grid with lambda_m and
compare with the global lambda=1 grid and with a run-count-scaled lambda (lambda_m = n_m / mean n).
Usage: python3 eb_lambda.py --version 1-1
Output: results/eb_lambda_{version}.json
"""
import argparse, json
import numpy as np
import pandas as pd
from core import load_runs, fit_model, RESULTS
from analysis_metr import years
from decompose import metrics, arrivals_and_slopes, perturbations, DEFS

GRID = np.exp(np.linspace(np.log(0.03), np.log(30), 31))


def laplace(g, lam):
    f = fit_model(g, lam=lam, return_hessian=True)
    sign, logdet = np.linalg.slogdet(f["H"])
    return f["loglik"] - 0.5 * lam * float(np.sum(f["delta"] ** 2)) + 0.5 * f["F"] * np.log(lam) - 0.5 * logdet, f


def rep_for(df, lam_of):
    out = []
    for m, g in df.groupby("alias"):
        po = fit_model(g); fa = fit_model(g, lam=lam_of(m, g))
        rate = g.groupby("fam_key").y.mean().reindex(fa["fam"]).values
        out.append(dict(alias=m, t=years(g.release.iloc[0]), pa=po["a"], pb=po["b"], a=fa["a"], b=fa["b"],
                        fam=fa["fam"], key=fa["fam"], delta=fa["delta"], rate=rate))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="1-1"); a = ap.parse_args()
    df = load_runs(a.version)
    df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
    df["fam_key"] = df.task_family
    nbar = df.groupby("alias").size().mean()
    lam_eb, lam_ess, nruns, curve = {}, {}, {}, {}
    for m, g in df.groupby("alias"):
        ml = np.array([laplace(g, l)[0] for l in GRID])
        lam_eb[m] = float(GRID[int(np.argmax(ml))]); nruns[m] = int(len(g))
        curve[m] = (ml - ml.max()).round(3).tolist()
        # design-effect correction: runs of one task are not independent; rescale weights so they sum to the
        # number of tasks (each task counts once), then re-maximise. lambda on this scale is converted back
        # to the run scale (penalty relative to data information is what matters).
        k = g.task_id.nunique() / len(g)
        g2 = g.assign(w=g.w * k)
        ml2 = np.array([laplace(g2, l)[0] for l in GRID])
        lam_ess[m] = float(GRID[int(np.argmax(ml2))] / k)
    res = dict(grid=GRID.tolist(), lambda_eb=lam_eb, lambda_eb_task_ess=lam_ess, n_runs=nruns,
               lambda_eb_task_ess_median=float(np.median(list(lam_ess.values()))),
               lambda_eb_task_ess_range=[float(min(lam_ess.values())), float(max(lam_ess.values()))],
               lambda_eb_median=float(np.median(list(lam_eb.values()))),
               lambda_eb_range=[float(min(lam_eb.values())), float(max(lam_eb.values()))],
               corr_log_lambda_vs_log_runs=float(np.corrcoef(np.log(list(lam_eb.values())), np.log(list(nruns.values())))[0, 1]))
    variants = {"global_lam1": lambda m, g: 1.0,
                "scaled_lam1 (lam*n_m/mean n)": lambda m, g: len(g) / nbar,
                "empirical_bayes_per_system": lambda m, g: lam_eb[m],
                "empirical_bayes_task_ess": lambda m, g: lam_ess[m]}
    pert = perturbations(df.alias.nunique(), None, ())
    grids = {}
    for name, fn in variants.items():
        M, t = metrics(rep_for(df, fn))
        A, S = arrivals_and_slopes(M, t, pert)
        grids[name] = dict(arrival=A, dt_months={d: (12 / S[d] if S[d] > 0 else np.inf) for d in DEFS},
                           latest_level_min={d: float(2 ** M[d].values[-1]) for d in DEFS})
    res["grids"] = grids
    ref = pd.Series(grids["global_lam1"]["arrival"])
    for name in grids:
        s = pd.Series(grids[name]["arrival"])
        res[f"summary::{name}"] = dict(range=[float(s.min()), float(s.max())], spread=float(s.max() - s.min()),
                                       spearman_vs_global=float(s.rank().corr(ref.rank())),
                                       max_abs_shift_vs_global=float((s - ref).abs().max()),
                                       defensible_range=[float(s[[d for d in DEFS if not d.startswith(("pmean+1", "minfam"))]].min()),
                                                         float(s[[d for d in DEFS if not d.startswith(("pmean+1", "minfam"))]].max())])
    json.dump(res, open(f"{RESULTS}/eb_lambda_{a.version}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("grids", "grid")}, indent=1, default=float))


if __name__ == "__main__":
    main()
