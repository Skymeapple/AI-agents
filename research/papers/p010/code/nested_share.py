"""Nested (double) hierarchical bootstrap for the definitional variance share.

Outer level: resample task families / tasks (as in the main bootstrap) -> a pseudo-dataset.
Inner level: on that pseudo-dataset, run B_in hierarchical bootstrap replicates and compute the share
   share = Var_d(mean_b A_bd) / (Var_d(mean_b A_bd) + mean_d Var_b A_bd)
exactly as for the real data. The spread of the outer shares is a direct bootstrap CI for the share, used to
check the cheaper single-level CI in decompose.py.
Usage: python3 nested_share.py --version 1-1 --Bout 40 --Bin 100 --procs 3
"""
import argparse, json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool
from core import load_runs, RESULTS
from analysis_metr import build_index, boot_sample
import ext_bootstrap as eb
from decompose import metrics, arrivals_and_slopes, perturbations, MENUS, DEFS

G = {}


def init(version, lam):
    df = load_runs(version)
    df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
    df["fam_key"] = df.task_family
    G.update(df=df)
    eb.G.update(lam=lam, lam_mode="global")


def outer(args):
    seed, Bin = args
    rng = np.random.default_rng(seed)
    df = G["df"]
    fam_tasks = build_index(df)
    fams = list(fam_tasks)
    parts = []
    for j, fi_ in enumerate(rng.choice(len(fams), size=len(fams), replace=True)):
        tasks = list(fam_tasks[fams[fi_]].items())
        for k, ti in enumerate(rng.choice(len(tasks), size=len(tasks), replace=True)):
            part = df.iloc[tasks[ti][1]].copy()
            part["task_family"] = f"{fams[fi_]}#{j}"            # duplicated families stay distinct
            part["task_id"] = f"{tasks[ti][0]}#{j}.{k}"         # duplicated tasks stay distinct
            parts.append(part)
    d0 = pd.concat(parts, ignore_index=True)
    d0["fam_key"] = d0.task_family
    fi = build_index(d0)
    A = []
    for _ in range(Bin):
        M, t = metrics(eb.fit_all(boot_sample(d0, fi, rng)))
        A.append(arrivals_and_slopes(M, t, perturbations(len(t), None, ()))[0])
    A = pd.DataFrame(A)[DEFS]
    res = {}
    for mn, ds in MENUS.items():
        X = A[ds].values
        W = float(np.mean(X.var(0))); Bt = float(X.mean(0).var())
        res[mn] = Bt / (Bt + W)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1-1"); ap.add_argument("--Bout", type=int, default=40)
    ap.add_argument("--Bin", type=int, default=100); ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.procs, initializer=init, initargs=(a.version, a.lam)) as pool:
        R = pool.map(outer, [(1000 + i, a.Bin) for i in range(a.Bout)], chunksize=1)
    R = pd.DataFrame(R)
    out = {mn: dict(mean=float(R[mn].mean()), lo=float(R[mn].quantile(0.05)), hi=float(R[mn].quantile(0.95)))
           for mn in R.columns}
    out["_settings"] = dict(Bout=a.Bout, Bin=a.Bin, secs=time.time() - t0)
    json.dump(out, open(f"{RESULTS}/nested_share_{a.version}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
