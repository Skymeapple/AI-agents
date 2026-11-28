"""Extended bootstrap that stores family-level fits per replicate, so that any definition, any trend procedure
and any dispersion statistic can be recomputed afterwards without refitting.

Modes
  task : hierarchical bootstrap over task families, then tasks within families (as in analysis_metr.py)
  time : as `task`, plus measurement error in human task lengths (log2 units), applied per task before fitting:
         eps_task ~ N(0, 0.5) for tasks timed by human baselines, N(0, 1.0) for tasks whose length METR estimated,
         plus one shared shift N(0, 0.5) for all estimated-length tasks (systematic mis-estimation of long tasks).
         These magnitudes are assumptions (stated in the paper), not estimates.
  point: no resampling (one replicate = the data).
Per replicate and system it stores: pooled (a,b), family-model (a,b), family keys, family offsets, and
per-family success rates (to flag saturated/floored families).

Usage: python3 ext_bootstrap.py --version 1-1 --mode task --B 1000 --lam 1 --procs 4
Output: results/ext_{version}_{mode}_lam{lam}.pkl.gz
"""
import argparse, gzip, pickle, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool
from core import load_runs, fit_model, RESULTS
from analysis_metr import build_index, boot_sample, years

G = {}
SIG_BASE, SIG_EST, SIG_SHARED = 0.5, 1.0, 0.5


def init(version, lam, mode, lam_mode):
    df = load_runs(version)
    df = df[df.release >= pd.Timestamp("2023-01-01")].reset_index(drop=True)
    df["fam_key"] = df.task_family
    G.update(df=df, lam=lam, mode=mode, fam_tasks=build_index(df), lam_mode=lam_mode,
             tasks=df.task_id.unique(), est=df.drop_duplicates("task_id").set_index("task_id").human_source == "estimate",
             nbar=df.groupby("alias").size().mean())


def fit_all(d):
    out = []
    for m, g in d.groupby("alias"):
        lam = G["lam"] * (len(g) / G["nbar"] if G["lam_mode"] == "scaled" else 1.0)
        po = fit_model(g)
        fa = fit_model(g, lam=lam)
        rate = g.groupby("fam_key").y.mean().reindex(fa["fam"]).values
        out.append(dict(alias=m, t=years(g.release.iloc[0]), pa=po["a"], pb=po["b"], a=fa["a"], b=fa["b"],
                        fam=np.array([k.split("#")[0] for k in fa["fam"]]), key=fa["fam"],
                        delta=fa["delta"].astype(np.float32), rate=rate.astype(np.float32)))
    return out


def one(seed):
    rng = np.random.default_rng(seed)
    df = G["df"]
    if G["mode"] == "point":
        d = df
    else:
        d = boot_sample(df, G["fam_tasks"], rng)
    if G["mode"] == "time":
        est = G["est"]
        sig = np.where(est.values, SIG_EST, SIG_BASE)
        eps = pd.Series(rng.normal(0, 1, len(est)) * sig + est.values * rng.normal(0, SIG_SHARED), index=est.index)
        d = d.copy()
        d["x"] = d.x.values + eps.reindex(d.task_id.values).values
    return fit_all(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1-1"); ap.add_argument("--mode", default="task")
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam_mode", default="global", help="global | scaled (lambda proportional to runs per system)")
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    t0 = time.time()
    B = 1 if a.mode == "point" else a.B
    seeds = [a.seed + i for i in range(B)]
    with Pool(a.procs, initializer=init, initargs=(a.version, a.lam, a.mode, a.lam_mode)) as pool:
        reps = pool.map(one, seeds, chunksize=8)
    tag = f"ext_{a.version}_{a.mode}_lam{a.lam:g}" + ("_scaled" if a.lam_mode == "scaled" else "")
    with gzip.open(f"{RESULTS}/{tag}.pkl.gz", "wb") as f:
        pickle.dump(dict(args=vars(a), reps=reps), f)
    print(tag, "B", B, "secs", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
