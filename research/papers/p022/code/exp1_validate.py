"""E1: validate closed forms (Thm 1, Prop 1) against the step-level simulator."""
import json, itertools, time
import numpy as np
import theory as th
from simulator import DetSampler, run_many

RES = "../results/"
configs = []
dets = {
    "point r=0.8 rho=0": (DetSampler("point", rho=0.0, r=0.8), lambda m: th.mu_point(0.8, m, 0.0)),
    "point r=0.9 rho=0.1": (DetSampler("point", rho=0.1, r=0.9), lambda m: th.mu_point(0.9, m, 0.1)),
    "beta(0.5,1) rho=0.05": (DetSampler("beta", rho=0.05, a=0.5, b=1.0), lambda m: th.mu_beta(0.5, 1.0, m, 0.05)),
}
grid = list(itertools.product(
    [("fresh", 0.0), ("fresh", 0.05), ("snapshot", 0.0), ("snapshot", 0.002)],
    [5, 20], [1, 3], list(dets.keys()), [0.0, 0.05]))
rows = []
t0 = time.time()
rng_seed = 0
for (policy, beta), T, m, dname, f in grid:
    det, mufun = dets[dname]
    kw = dict(N=400, T=T, p0=0.004, beta=beta, det=det, m=m, f=f, V=1.0,
              C=0.5, R=1.0, K=2.0 if policy == "fresh" else 0.0, policy=policy)
    rng_seed += 1
    times, clean = run_many(3000, rng_seed, **kw)
    mu = float(mufun(m))
    E, Pc = th.makespan_and_correctness(400, T, 0.004, beta, mu, m, f, 1.0, 0.5,
                                         1.0, kw["K"], policy)
    rows.append(dict(policy=policy, beta=beta, T=T, m=m, det=dname, f=f, mu=mu,
                     sim_time=float(times.mean()),
                     sim_time_se=float(times.std() / np.sqrt(len(times))),
                     th_time=float(E), sim_clean=float(clean.mean()),
                     sim_clean_se=float(np.sqrt(clean.mean() * (1 - clean.mean()) / len(clean))),
                     th_clean=float(Pc)))
print("elapsed", time.time() - t0)
zt = np.array([(r["sim_time"] - r["th_time"]) / r["sim_time_se"] for r in rows])
zc = np.array([(r["sim_clean"] - r["th_clean"]) / max(r["sim_clean_se"], 1e-9) for r in rows])
summary = dict(n_configs=len(rows), runs_per_config=3000,
               max_rel_err_time=float(max(abs(r["sim_time"] / r["th_time"] - 1) for r in rows)),
               max_abs_err_clean=float(max(abs(r["sim_clean"] - r["th_clean"]) for r in rows)),
               frac_time_within_2se=float(np.mean(np.abs(zt) < 2)),
               frac_clean_within_2se=float(np.mean(np.abs(zc) < 2)),
               mean_z_time=float(zt.mean()), mean_z_clean=float(zc.mean()))
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=rows), open(RES + "e1_validation.json", "w"), indent=1)
