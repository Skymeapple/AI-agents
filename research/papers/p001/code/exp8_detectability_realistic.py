"""Exp 8 (revision 1): detectability of a genuine break at realistic per-run noise, using the exact procedure
applied to the real data (Huber delta_H=1e-3, breakpoint profiled over the 10th-90th compute percentiles,
one-sided sup statistic, parametric-bootstrap calibration).  Responds to reviewer W3, M1, M2, M3.

Truth: refit Chinchilla law (Exp. 1) with a break delta at the geometric midpoint of the design.
Design (default): 12 iso-FLOP budgets over 3 decades ending at C_end, 20 sizes/budget, tokens/param in [2,200].
Noise: white Gaussian sigma_w on log-loss per run, plus optional per-budget shared offset sigma_b
(models recipe/hyper-parameter drift shared by all runs of a budget).
Each cell: 200 null replicates give the 95% critical value (oracle parametric bootstrap); 100 replicates per
delta give power.  All rates are reported with Wilson 95% Monte Carlo intervals.
A check compares the oracle critical value with one obtained by simulating from a *fitted* null.
"""
import json, time, os
import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp
from breaktest import profiled, wilson
from common import RES

P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
PT = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])
GAMMA = P["alpha"] * P["beta"] / (P["alpha"] + P["beta"])
NNULL, NALT = 200, 100
DELTAS = [0.02, 0.05, 0.10]


def design(Cend, K=12, M=20, W=3.0):
    lC = np.repeat(np.linspace(np.log(Cend) - W * np.log(10), np.log(Cend), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(2), np.log(200), M)), K)
    lN = 0.5 * (lC - np.log(6) - np.log(ratio)); lD = lC - np.log(6) - lN
    return lN, lD, lC


def mean_log(p, lN, lD, lC, delta, lCb):
    a, b, e, al, be = p
    h = np.maximum(0.0, lC - lCb)
    return logsumexp(np.stack([a - al * lN + delta * h, b - be * lD + delta * h, np.full_like(lN, e)]), axis=0)


def one(args):
    cell, delta, seed, p_null = args
    rng = np.random.default_rng(seed)
    lN, lD, lC = design(cell["Cend"], cell["K"], cell["M"])
    lCb = 0.5 * (lC.min() + lC.max())
    mu = mean_log(PT if p_null is None else p_null, lN, lD, lC, delta, lCb)
    K = cell["K"]
    off = np.repeat(cell["sigma_b"] * rng.standard_normal(K), cell["M"])
    lL = mu + off + cell["sigma_w"] * rng.standard_normal(mu.shape)
    grid = np.quantile(lC, np.linspace(0.1, 0.9, 9))
    inits = [PT, PT + np.array([0.3, -0.3, 0.01, 0.01, -0.01])]
    r = profiled(lN, lD, lC, lL, grid, inits, d0s=(0.0, 0.05))
    return r["T1"], r["delta_hat"], r["x0"].tolist()


CELLS = []
for sw in [1e-4, 3e-4, 1e-3, 2.5e-3, 7.5e-3]:
    for Cend in [1e22, 1e24, 1e26, 1e28]:
        CELLS.append(dict(tag="main", sigma_w=sw, sigma_b=0.0, Cend=Cend, K=12, M=20))
for sb in [1e-3, 3e-3]:
    for Cend in [1e26, 1e28]:
        CELLS.append(dict(tag="grouped", sigma_w=1e-4, sigma_b=sb, Cend=Cend, K=12, M=20))
for sw in [3e-4, 1e-3]:
    for Cend in [1e26, 1e28]:
        CELLS.append(dict(tag="small_design", sigma_w=sw, sigma_b=0.0, Cend=Cend, K=6, M=8))

if __name__ == "__main__":
    t0 = time.time()
    out = []
    with Pool(os.cpu_count()) as pool:
        for ci, cell in enumerate(CELLS):
            base_seed = 800000 + 10000 * ci
            nul = pool.map(one, [(cell, 0.0, base_seed + i, None) for i in range(NNULL)])
            T0 = np.array([x[0] for x in nul])
            crit = float(np.quantile(T0, 0.95))
            row = dict(cell, n_runs=cell["K"] * cell["M"], crit95=crit, null_frac_T_zero=float((T0 == 0).mean()), power={})
            for d in DELTAS:
                alt = pool.map(one, [(cell, d, base_seed + 5000 + int(d * 1e4) * 7 + i, None) for i in range(NALT)])
                T = np.array([x[0] for x in alt]); k = int((T > crit).sum())
                row["power"][str(d)] = dict(rate=k / NALT, ci95=wilson(k, NALT),
                                            delta_hat_median=float(np.median([x[1] for x in alt])))
            out.append(row)
            print(cell["tag"], cell["sigma_w"], cell["sigma_b"], f"{cell['Cend']:.0e}", cell["K"] * cell["M"],
                  {d: row["power"][str(d)]["rate"] for d in DELTAS}, f"{time.time()-t0:.0f}s", flush=True)

        # size of the full procedure when calibrated from a fitted null (not the oracle), 2 cells
        size_check = []
        for ci in [CELLS.index(next(c for c in CELLS if c["tag"] == "main" and c["sigma_w"] == s and c["Cend"] == 1e26))
                   for s in [3e-4, 2.5e-3]]:
            cell = CELLS[ci]
            rej = 0; nrep = 20
            for rep in range(nrep):
                T_obs, _, x0 = one((cell, 0.0, 990000 + 1000 * ci + rep, None))
                sims = pool.map(one, [(cell, 0.0, 7000000 + 100000 * ci + 1000 * rep + i, np.array(x0)) for i in range(60)])
                crit_fit = np.quantile([s[0] for s in sims], 0.95)
                rej += int(T_obs > crit_fit)
            size_check.append(dict(sigma_w=cell["sigma_w"], Cend=cell["Cend"], reps=nrep, rejections=rej,
                                   size=rej / nrep, ci95=wilson(rej, nrep)))
            print("size check", size_check[-1], f"{time.time()-t0:.0f}s", flush=True)

    json.dump(dict(gamma=GAMMA, n_null=NNULL, n_alt=NALT, cells=out, fitted_null_size_check=size_check),
              open(f"{RES}/exp8_detectability.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for j, d in enumerate([0.02, 0.05]):
        for i, sw in enumerate([1e-4, 3e-4, 1e-3, 2.5e-3, 7.5e-3]):
            rs = [r for r in out if r["tag"] == "main" and r["sigma_w"] == sw]
            ax[j].plot([r["Cend"] for r in rs], [r["power"][str(d)]["rate"] for r in rs], "-o", color=f"C{i}",
                       label=f"$\\sigma$={sw:g}")
        ax[j].set_xscale("log"); ax[j].axhline(0.8, color="k", ls=":", lw=.8)
        ax[j].set_xlabel("end of 3-decade design window (FLOP)")
        ax[j].set_title(f"power to detect $\\delta$={d} (profiled Huber test, 240 runs)", fontsize=9)
    ax[0].set_ylabel("power (5% level)"); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{RES}/fig6_detectability_noise.png", dpi=150)
