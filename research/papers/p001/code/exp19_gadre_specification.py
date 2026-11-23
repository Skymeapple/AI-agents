"""Exp 19 (revision 2): specification sensitivity of out-of-sample forecasts on the Gadre et al. (2024) suite,
expressed in compute-equivalent units.

For each training distribution / in-distribution eval (as in Exp. 15), fit the 5-parameter law (Huber 1e-3 on
log-loss, Hoffmann-style initial grid) to the small runs under five reasonable selection rules, and forecast the
three held-out large runs (1.4B at M=1, 1.4B over-trained, 6.9B at M=1).  Report the log residual and the
implied compute-equivalent factor P_hat = ((L_obs - E_hat)/(L_pred - E_hat))^(1/gamma_hat), plus a 100-replicate
pairs bootstrap sd(ln E_hat) and the implied audit ceiling C_dagger (50% power) for each specification.
Note: this is the Chinchilla functional form, not the law Gadre et al. fit; it is not an assessment of their
forecasts.  The 6.9B run used a different learning rate and weight decay from the small runs (models.csv).
"""
import csv, json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm
from common import RES, ROOT, obj_base, OPT, INIT_GRID, pred_log_base
import horizon as H

rows = list(csv.DictReader(open(f"{ROOT}/data/gadre2024/models.csv")))
EVAL = {"c4_original": "c4_val", "rpj": "paloma_redpajama", "rw_original": "paloma_falcon-refinedweb"}
BIG = ("open_lm_1b", "open_lm_7b")
SPECS = {
    "S0_all": lambda r: True,
    "S1_M_ge_0.5": lambda r: float(r["multiplier"]) >= 0.5,
    "S2_M_ge_1": lambda r: float(r["multiplier"]) >= 1.0,
    "S3_no_11M": lambda r: float(r["params"]) > 2e7,
    "S4_M_ge_1_no_11M": lambda r: float(r["multiplier"]) >= 1.0 and float(r["params"]) > 2e7,
}


def fit_grid(lN, lD, lL, inits):
    return min((minimize(obj_base, np.array(x0, float), args=(lN, lD, lL), method="BFGS", jac=True, options=OPT)
                for x0 in inits), key=lambda r: r.fun).x


def job(args):
    ds, spec = args
    ev = EVAL[ds]
    small = [r for r in rows if r["dataset"] == ds and r["model"] not in BIG and SPECS[spec](r)]
    big = [r for r in rows if r["dataset"] == ds and r["model"] in BIG]
    lN = np.log([float(r["params"]) for r in small]); lD = np.log([float(r["tokens"]) for r in small])
    lL = np.log([float(r[ev]) for r in small])
    th = fit_grid(lN, lD, lL, INIT_GRID[::3] + [[5, 7, 0.3, 0.3, 0.3]])
    rng = np.random.default_rng(1919)
    bE = []
    for b in range(100):
        ii = rng.integers(0, len(lL), len(lL))
        inits = [th, th + np.array([0.5, -0.5, 0.03, 0.03, -0.03]), th + np.array([-0.5, 0.5, -0.03, -0.03, 0.03])]
        bE.append(fit_grid(lN[ii], lD[ii], lL[ii], inits)[2])
    sdE = float(np.std(bE)); gam = H.gam(th); E = float(np.exp(th[2]))
    E_, R_, _, _ = H.frontier_parts(th, 1.0)
    k = np.expm1((norm.ppf(0.95)) * sdE)
    Cdag = float((R_ / (k * E)) ** (1 / gam))
    tg = []
    for r in big:
        n_, t_ = float(r["params"]), float(r["tokens"]); Lobs = float(r[ev])
        Lp = float(np.exp(pred_log_base(th, np.log([n_]), np.log([t_]))[0]))
        Ph = ((Lobs - E) / (Lp - E)) ** (1 / gam) if Lobs > E else 0.0
        tg.append(dict(name=r["name"], C=6 * n_ * t_, L_obs=Lobs, L_pred=Lp, log_res=float(np.log(Lobs / Lp)), P_hat=float(Ph)))
    return dict(dataset=ds, spec=spec, n_fit=len(small), E=E, alpha=float(th[3]), beta=float(th[4]), gamma=float(gam),
                resid_sd=float(np.std(lL - pred_log_base(th, lN, lD))), sd_lnE=sdE, C_dagger_50=Cdag, targets=tg)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as pool:
        res = pool.map(job, [(ds, s) for ds in EVAL for s in SPECS])
    for r in res:
        print(r["dataset"], r["spec"], r["n_fit"], f"E={r['E']:.3f} g={r['gamma']:.3f} sdlnE={r['sd_lnE']:.3f} Cdag={r['C_dagger_50']:.2e}",
              [(t["name"].split("-", 1)[1], round(t["log_res"], 4), round(t["P_hat"], 2)) for t in r["targets"]], flush=True)
    summ = {}
    for ds in EVAL:
        for t in res[0]["targets"]:
            pass
    for ds in EVAL:
        names = [t["name"] for t in next(r for r in res if r["dataset"] == ds)["targets"]]
        for nm in names:
            vals = [(t["log_res"], t["P_hat"]) for r in res if r["dataset"] == ds for t in r["targets"] if t["name"] == nm]
            summ[nm] = dict(log_res_range=[min(v[0] for v in vals), max(v[0] for v in vals)],
                            P_hat_range=[min(v[1] for v in vals), max(v[1] for v in vals)])
    print(json.dumps(summ, indent=1))
    json.dump(dict(fits=res, summary=summ), open(f"{RES}/exp19_gadre_spec.json", "w"), indent=2)
    print("done", f"{time.time()-t0:.0f}s")
