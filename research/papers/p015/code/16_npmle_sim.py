"""Simulation check of the NPMLE-frailty estimator and of the short->long forecast as a diagnostic.

Design: Claude 4 Opus (Inspect) task lengths and run counts.  For each scenario we report
  * NPMLE k on the full design (is it consistent under non-Gaussian frailty? what does drift do?)
  * the NPMLE forecast error when training on T < 60 min and predicting the mean success of tasks in
    [60, 240) min (prediction vs the TRUE mean p of those tasks), and whether the truth lies in the 95% LR bounds.
Scenarios with an invariant frailty law should give unbiased forecasts; drift or a kink in the
hazard-length map should reproduce the over-prediction seen in the data.

Outputs: results/npmle_sim.csv, results/npmle_sim_summary.txt
"""
import sys
import importlib.util
import numpy as np
import pandas as pd
from multiprocessing import Pool
from common import *
import npmle

spec = importlib.util.spec_from_file_location("ms", f"{HERE}/09_misspecification_sim.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
T0, N0, LT0 = ms.T0, ms.N0, ms.LT0
C1, C2 = np.log(60 / T_REF), np.log(240 / T_REF)


def kink_logL(lt, k1=1.0, k2=3.0, k3=1.0):
    return k1 * np.minimum(lt, C1) + k2 * (np.clip(lt, C1, C2) - C1) + k3 * (np.maximum(lt, C2) - C2)


SCEN = {
    "B: homogeneous, k=0.63": ms.SCEN["B: homogeneous, k=0.63"],
    "C: Gaussian, k=1.5, s=3.2": ms.SCEN["C: Gaussian, k=1.5, s=3.2"],
    "M1: two-point frailty, k=0.63": ms.SCEN["M1: two-point frailty, k=0.63"],
    "M3: skewed 3-point frailty, k=0.63": ms.SCEN["M3: skewed 3-point frailty, k=0.63"],
    "M4: Gaussian s=2 + 10% impossible, k=0.63": ms.SCEN["M4: Gaussian s=2 + 10% impossible (any T), k=0.63"],
    "M7: frailty location drift +0.8 per log T, k=0.63": ms.SCEN["M7: frailty location drifts +0.8 per log T, k=0.63"],
    "K: invariant Gaussian s=2.5, kinked map (k=1 | 3 on 1-4 h | 1)": "kink",
}
NREP = 10


def one(args):
    name, rep = args
    rng = np.random.default_rng(777 + 100 * list(SCEN).index(name) + rep)
    sc = SCEN[name]
    if sc == "kink":
        mu = ms.calib_gauss(1.0, 2.5)
        p = np.exp(-np.exp(np.clip(mu + kink_logL(LT0) + 2.5 * rng.standard_normal(len(LT0)), -30, 6)))
        k_true = np.nan
    else:
        mu = ms.calib(sc)
        p = ms.p_tasks(sc, mu, LT0, rng)
        k_true = sc["k"]
    s = rng.binomial(N0, p).astype(float)
    n = N0.astype(float)
    f = npmle.fit(n, s, T0)
    tr, te = T0 < 60, (T0 >= 60) & (T0 < 240)
    fw = npmle.fit(n[tr], s[tr], T0[tr])
    pred = float(np.mean(npmle.S_pooled(fw["pi"], fw["k"], T0[te])))
    truth = float(np.mean(p[te]))
    return dict(scenario=name, rep=rep, k_true=k_true, k_npmle=f["k"], k_lo=f["k_lo"], k_hi=f["k_hi"],
                k_window=fw["k"], forecast=pred, truth=truth, err=pred - truth)


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(nproc) as pool:
        R = pd.DataFrame(pool.map(one, [(nm, r) for nm in SCEN for r in range(NREP)]))
    R.to_csv(f"{RES}/npmle_sim.csv", index=False)
    R["covers_true_k"] = (R.k_lo <= R.k_true) & (R.k_true <= R.k_hi)
    S = R.groupby("scenario", sort=False).agg(k_true=("k_true", "first"), k_npmle_mean=("k_npmle", "mean"),
                                              k_npmle_sd=("k_npmle", "std"), ci_covers=("covers_true_k", "mean"),
                                              k_window_mean=("k_window", "mean"), forecast_err_mean=("err", "mean"),
                                              forecast_err_sd=("err", "std"))
    pd.set_option("display.width", 250)
    txt = (f"NPMLE-frailty simulation, design = Claude 4 Opus TH1.1, {NREP} reps per scenario\n"
           "forecast: train T < 60 min, predict mean true p on tasks in [60, 240) min\n" + S.round(3).to_string() + "\n")
    open(f"{RES}/npmle_sim_summary.txt", "w").write(txt)
    print(txt)
