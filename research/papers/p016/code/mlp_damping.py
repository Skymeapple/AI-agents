"""Test of the damping-scaling prediction (Thm 6b + Prop 5): the damping needed to suppress
the gap should shrink with the replay weight.  Flow-like damped GGN (lr=0.2, 150 steps) at
rho in {0.01, 0.1} with delta in {1e-3, 1e-4, 1e-5} (relative to lambda_max at phase-2 start).
Usage: python mlp_damping.py <scenario> <seed>  -> ../results/mlpdamp_<scen>_<seed>.json"""
import sys, os, json, time
from mlp_experiments import data, phase1, phase2, loss, acc, R
scen, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
T = data(scen, seed); th0 = phase1(seed, T[0]); runs = []
for rho in [0.01, 0.1]:
    for delta in [1e-3, 1e-4, 1e-5]:
        r = phase2(th0, T, rho, "ggn", 150, delta=delta, lr=0.2); r.update(rho=rho, delta=delta)
        runs.append(dict(rho=rho, delta=delta, L1=r["L1"], L2=r["L2"], A1=r["A1"], A2=r["A2"]))
        print(scen, seed, rho, delta, f"G={max(r['L1'])-r['L1'][0]:.4f}", f"{time.time()-t0:.0f}s", flush=True)
json.dump(dict(scenario=scen, seed=seed, runs=runs), open(os.path.join(R, f"mlpdamp_{scen}_{seed}.json"), "w"))
