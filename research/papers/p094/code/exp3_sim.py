"""Exp 3: Monte-Carlo comparison of upper confidence bounds (UCBs) on budget-B
coverage C(B), from n attempts on each of T tasks, across task populations.

A UCB is 'safe' if UCB >= C(B).  An evaluator that declares 'below threshold tau'
whenever UCB < tau commits a false-safe error with probability <= P(UCB < C(B)).
Methods:
  naive      - Clopper-Pearson UCB on observed pass@n (ignores B)
  powerlaw   - -log pass@k = A k^-b fit on k<=n, bootstrap 95% UCB
  betabin    - beta-binomial MLE extrapolation, bootstrap 95% UCB
  lp0        - partial-identification LP, no assumption (support [0,1])
  lp_floor   - LP with elicitation floor: support {0} U [pmin,1]
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from core import (histogram, betabin_fit, betabin_coverage, powerlaw_coverage,
                  naive_ucb, lp_bounds, default_grid)
from exp1_theory import populations

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BS = [100, 1000]
PMINS = [1e-2, 1e-3]
NBOOT = 40
REPS = int(os.environ.get("REPS", 80))


def one_rep(args):
    pop, n, T, seed = args
    rng = np.random.default_rng(seed)
    ps_all = POPS[pop]
    p = rng.choice(ps_all, size=T, replace=True)
    h = histogram(rng.binomial(n, p), n)
    res = dict(pop=pop, n=n, T=T, seed=seed)
    # bootstrap fits shared across B
    bb_fits, pl_h = [], []
    for _ in range(NBOOT):
        hb = rng.multinomial(T, h / T)
        bb_fits.append(betabin_fit(hb, n)); pl_h.append(hb)
    a, b = betabin_fit(h, n)
    for B in BS:
        r = {}
        r["naive"] = naive_ucb(h, n)
        r["betabin_pt"] = betabin_coverage(a, b, B)
        r["betabin"] = float(np.quantile([betabin_coverage(x, y, B) for x, y in bb_fits], 0.95))
        r["powerlaw_pt"] = powerlaw_coverage(h, n, B)
        r["powerlaw"] = float(np.quantile([powerlaw_coverage(hb, n, B) for hb in pl_h], 0.95))
        r["lp0"] = lp_bounds(h, n, B, alpha=0.05, grid=GRIDS[0.0])[1]
        for pm in PMINS:
            r[f"lp_floor{pm:g}"] = lp_bounds(h, n, B, alpha=0.05, grid=GRIDS[pm])[1]
        res[B] = r
    return res


def init():
    global POPS, GRIDS
    POPS = populations(np.random.default_rng(0))
    GRIDS = {0.0: default_grid(0.0)}
    for pm in PMINS:
        GRIDS[pm] = default_grid(pm)


if __name__ == "__main__":
    t0 = time.time()
    init()
    truth = {pop: {B: float(np.mean(1 - (1 - ps) ** B)) for B in BS} for pop, ps in POPS.items()}
    # population-level 'floor violation mass' (share of tasks with 0<p<pmin)
    viol = {pop: {f"{pm:g}": float(np.mean((ps > 0) & (ps < pm))) for pm in PMINS} for pop, ps in POPS.items()}
    jobs = []
    for pop in POPS:
        for n in [4, 16]:
            for T in [200, 1000]:
                for r in range(REPS):
                    jobs.append((pop, n, T, 1000003 * len(jobs) + 17))
    with Pool(4, initializer=init) as pool:
        out = pool.map(one_rep, jobs, chunksize=4)
    json.dump(dict(truth=truth, floor_violation_mass=viol, runs=out),
              open(os.path.join(OUT, "exp3_sim_raw.json"), "w"))
    # summarise
    methods = ["naive", "powerlaw", "betabin", "lp0"] + [f"lp_floor{pm:g}" for pm in PMINS]
    summ = []
    for pop in POPS:
        for n in [4, 16]:
            for T in [200, 1000]:
                for B in BS:
                    rr = [o[B] for o in out if o["pop"] == pop and o["n"] == n and o["T"] == T]
                    C = truth[pop][B]
                    row = dict(pop=pop, n=n, T=T, B=B, C_true=C)
                    for m in methods:
                        v = np.array([x[m] for x in rr])
                        row[m + "_cover"] = float(np.mean(v >= C - 1e-9))
                        row[m + "_mean"] = float(np.mean(v))
                    row["betabin_pt_mean"] = float(np.mean([x["betabin_pt"] for x in rr]))
                    row["powerlaw_pt_mean"] = float(np.mean([x["powerlaw_pt"] for x in rr]))
                    summ.append(row)
    json.dump(dict(truth=truth, floor_violation_mass=viol, summary=summ),
              open(os.path.join(OUT, "exp3_sim_summary.json"), "w"), indent=1)
    print("elapsed", time.time() - t0)
