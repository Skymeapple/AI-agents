"""Exp 17 (revision 2): reviewer round-2 checks on the in-sweep (profiled) break test of Exp. 8.

 (a) Power vs number of budgets K at fixed n = 240 runs (K x M = 6x40, 12x20, 24x10, 48x5, 120x2),
     sigma_w = 1e-4, per-budget offset sigma_b = 1e-3, 3-decade window ending at 1e26, delta in {0.02, 0.05}.
     150 null replicates (oracle critical value) and 100 alternative replicates per delta.
 (b) Fitted-null calibration (the procedure used on real data): main design, sigma_w = 1e-3, window ending 1e26.
     40 null datasets and 40 datasets with delta = 0.02; each calibrated with 60 simulations from its own
     fitted null.  Compare with the oracle power (0.82, Exp. 8).
 (c) Huber threshold scaled with noise: sigma_w = 2.5e-3, window ending 1e26, delta_H in {1e-3 (Exp. 8), 2.5e-3,
     5e-3}; 150 null and 100 alternative replicates (delta = 0.02, 0.05).
"""
import json, time, os
import numpy as np
from multiprocessing import Pool
from breaktest import profiled, wilson
from exp8_detectability_realistic import design, mean_log, PT
from common import RES


def one(args):
    cell, delta, seed, p_null = args
    rng = np.random.default_rng(seed)
    lN, lD, lC = design(cell["Cend"], cell["K"], cell["M"])
    lCb = 0.5 * (lC.min() + lC.max())
    mu = mean_log(PT if p_null is None else p_null, lN, lD, lC, delta, lCb)
    off = np.repeat(cell["sigma_b"] * rng.standard_normal(cell["K"]), cell["M"])
    lL = mu + off + cell["sigma_w"] * rng.standard_normal(mu.shape)
    grid = np.quantile(lC, np.linspace(0.1, 0.9, 9))
    inits = [PT, PT + np.array([0.3, -0.3, 0.01, 0.01, -0.01])]
    r = profiled(lN, lD, lC, lL, grid, inits, delta_h=cell.get("dH", 1e-3), d0s=(0.0, 0.05))
    return r["T1"], r["delta_hat"], r["x0"].tolist()


def power_cell(pool, cell, seed0, deltas, nnull=150, nalt=100):
    T0 = np.array([x[0] for x in pool.map(one, [(cell, 0.0, seed0 + i, None) for i in range(nnull)])])
    crit = float(np.quantile(T0, 0.95))
    row = dict(cell, crit95=crit, power={})
    for d in deltas:
        T = np.array([x[0] for x in pool.map(one, [(cell, d, seed0 + 50000 + int(d * 1e4) * 7 + i, None) for i in range(nalt)])])
        k = int((T > crit).sum())
        row["power"][str(d)] = dict(rate=k / nalt, ci95=wilson(k, nalt))
    return row


if __name__ == "__main__":
    t0 = time.time(); out = {}
    with Pool(os.cpu_count()) as pool:
        rows = []
        for i, K in enumerate([6, 12, 24, 48, 120]):
            cell = dict(sigma_w=1e-4, sigma_b=1e-3, Cend=1e26, K=K, M=240 // K)
            rows.append(power_cell(pool, cell, 1700000 + 100000 * i, [0.02, 0.05]))
            print("(a)", K, {d: v["rate"] for d, v in rows[-1]["power"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        out["a_power_vs_K"] = rows

        rows = []
        for i, dH in enumerate([1e-3, 2.5e-3, 5e-3]):
            cell = dict(sigma_w=2.5e-3, sigma_b=0.0, Cend=1e26, K=12, M=20, dH=dH)
            rows.append(power_cell(pool, cell, 1900000 + 100000 * i, [0.02, 0.05]))
            print("(c)", dH, {d: v["rate"] for d, v in rows[-1]["power"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        out["c_huber_threshold"] = rows

        cell = dict(sigma_w=1e-3, sigma_b=0.0, Cend=1e26, K=12, M=20)
        res = {}
        for d in [0.0, 0.02]:
            rej = 0; nrep = 40
            for rep in range(nrep):
                T_obs, _, x0 = one((cell, d, 2100000 + int(d * 1e5) + rep, None))
                sims = pool.map(one, [(cell, 0.0, 2200000 + int(d * 1e7) + 1000 * rep + i, np.array(x0)) for i in range(60)])
                rej += int(T_obs > np.quantile([s[0] for s in sims], 0.95))
            res[str(d)] = dict(reps=nrep, rejections=rej, rate=rej / nrep, ci95=wilson(rej, nrep))
            print("(b)", d, res[str(d)], f"{time.time()-t0:.0f}s", flush=True)
        out["b_fitted_null"] = dict(cell=cell, n_sims_per_dataset=60, results=res, oracle_power_exp8=0.82)
    json.dump(out, open(f"{RES}/exp17_insweep_design.json", "w"), indent=2)
    print("done", f"{time.time()-t0:.0f}s")
