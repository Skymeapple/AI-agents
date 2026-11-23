"""Revision r2: real-data analysis.  (1) Best-of-N jailbreaking (Hughes et al. 2024; 9 text-modality models incl.
GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro; 159 HarmBench prompts; capped first-success runs, see build_bon.py).
(2) Large Language Monkeys MATH / Pythia (as exp_llm.py; first counterfactual subset).
For each (model, n, k): parametric forecasts (BoN only; MATH parametric results are in exp_llm.py), the box interval,
the block-estimator interval (Theorem 5a; Hoeffding and empirical-Bernstein deviations; degree L chosen without
looking at the data, to minimise the worst-case Hoeffding half-width at this (n, k, M)), and the instance-dependent
lower bound of exp_local.local_lb at the grid NPMLE of the observed counts.
Outputs: results/realdata_bon.csv, results/realdata_math.csv"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ttc import bernstein_matrix, g_passk, box_interval, powerlaw_passk
from exp_baselines import fit_all
from exp_llm import subsets, unbiased, N as NLLM
from exp_local import grid, npmle, local_lb
import block

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EPS = 0.05
_cache = {}


def block_design(n, k, M):
    key = (n, k, M)
    if key not in _cache:
        _cache[key] = block.best_design(n, k, M, EPS)[1]
    return _cache[key]


def common(X, n, ks, extra):
    M = len(X); p = grid(); B = bernstein_matrix(n, p).T
    hist = np.bincount(X, minlength=n + 1).astype(float)
    wF = npmle(hist, B)
    rows = []
    for k in ks:
        gv = g_passk(k)(p)
        lb = local_lb(gv, B, wF, M)["lb"]
        blo, bhi = box_interval(g_passk(k), n, X, alpha=EPS)
        d = block_design(n, k, M)
        hlo, hhi = block.interval(d, X, EPS, "hoeffding")
        elo, ehi = block.interval(d, X, EPS, "bernstein")
        rows.append(dict(extra, n=n, k=k, M=M, s=k / n, frac_zero=float(np.mean(X == 0)), npmle_theta=float(gv @ wF),
                         local_lb_half=lb, box_lo=blo, box_hi=bhi, box_half=(bhi - blo) / 2,
                         blk_L=d["L"], blk_beta=d["beta"], blk_lo=max(hlo, 0), blk_hi=min(hhi, 1), blk_half=(hhi - hlo) / 2,
                         blkeb_lo=max(elo, 0), blkeb_hi=min(ehi, 1), blkeb_half=(ehi - elo) / 2))
    return rows


def bon_job(args):
    model, first60, T, n, ks = args
    X = np.array([s[:n].count("1") for s in first60])
    rows = common(X, n, ks, dict(data="BoN", model=model))
    par = fit_all(X, np.full(len(X), n), ks, seed=0)
    pl = powerlaw_passk(X, n, ks)
    Tn = np.asarray(T, float)
    for j, r in enumerate(rows):
        k = r["k"]
        r["truth"] = float(np.mean(np.where(np.isnan(Tn), False, Tn <= k)))
        r["powerlaw"] = float(pl[j])
        for mo in ["BB", "ZIBB", "BMix2", "BMix3", "BIC"]:
            r[mo] = par[mo][k]
        r["BIC_sel"] = par["BIC"]["selected"]
    print("BoN", model, n, flush=True)
    return rows


def math_job(args):
    model, c, n, mi, ks, subs = args
    rows = []
    for s, X in enumerate(subsets(c, n, 1000 * mi + n)):
        if s not in subs:
            continue
        rr = common(X, n, ks, dict(data="MATH", model=model, subset=s))
        for r in rr:
            r["truth"] = unbiased(c, NLLM, r["k"])
        rows += rr
    print("MATH", model, n, flush=True)
    return rows


def main(which):
    nproc = int(os.environ.get("NPROC", 4))
    t0 = time.time()
    if which in ("bon", "all"):
        d = pd.read_csv(os.path.join(DATA, "bon_text_first_success.csv"), dtype={"first60": str})
        jobs = []
        for model, g in d.groupby("model"):
            cap = int(g.n_recorded.max())
            for n in [15, 30, 60]:
                ks = sorted(set([n, 2 * n, 4 * n, 8 * n, 16 * n, 1000, min(10000, cap)]))
                jobs.append((model, g.first60.tolist(), g.first_success.values, n, ks))
        with Pool(nproc) as pool:
            bon = [r for rr in pool.map(bon_job, jobs, chunksize=1) for r in rr]
        pd.DataFrame(bon).to_csv(os.path.join(RES, "realdata_bon.csv"), index=False, float_format="%.5f")
        print("BoN done", time.time() - t0, flush=True)
    if which in ("math", "all"):
        dm = pd.read_csv(os.path.join(RES, "llm_monkeys_math_pythia_counts.csv"))
        models = sorted(dm.model.unique(), key=lambda m: float(m.split()[1].rstrip("MB")) * (1e-3 if m.endswith("M") else 1))
        C = {m: dm[dm.model == m].sort_values("problem_idx").n_correct.values.astype(int) for m in models}
        jobs = []
        for mi, m in enumerate(models):
            for n in [16, 32, 64, 128]:
                ks = sorted(set(k for k in [n, 2 * n, 4 * n, 8 * n, 16 * n, 10000] if k <= 10000))
                jobs.append((m, C[m], n, mi, ks, {0}))
        with Pool(nproc) as pool:
            loc = [r for rr in pool.map(math_job, jobs, chunksize=1) for r in rr]
        pd.DataFrame(loc).to_csv(os.path.join(RES, "realdata_math.csv"), index=False, float_format="%.5f")
        print("MATH done", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
