"""E4: context-age hazard.  (a) optimal interval scaling T* ~ c^(1/2) vs c^(1/3);
(b) fresh-context resumption vs context-snapshot restoration across horizons;
(c) correctness/time Pareto front over T under age hazard."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory as th
from simulator import DetSampler, run_many

RES = "../results/"
out = {}
mu = th.mu_point(0.8, 1, 0.05)  # r=0.8, rho=0.05, m=1


def best_T(N, p0, beta, c_over, f=0.0, K=0.0, policy="fresh", Tmax=2000):
    Ts = np.unique(np.round(np.logspace(0, np.log10(Tmax), 400)).astype(int))
    best = (np.inf, None)
    for T in Ts:
        Nn = (N // T) * T  # exact multiple to avoid edge effects
        if Nn == 0:
            continue
        E, _ = th.makespan_and_correctness(Nn, T, p0, beta, mu, 1, f, c_over, 0.0, 0.0, K, policy)
        if E / Nn < best[0]:
            best = (E / Nn, T)
    return best[1], best[0]


# (a) scaling of T* with overhead c = V (C=R=K=0)
cs = np.logspace(-1, 2, 13)
scal = {"c": cs.tolist()}
for beta in [0.0, 0.01, 0.1]:
    Tb, Ta = [], []
    for c in cs:
        T, _ = best_T(200000, 1e-3, beta, c)
        Tb.append(int(T)); Ta.append(float(th.tstar_approx(1e-3, beta, mu, 1, c, 0.0)))
    sl = np.polyfit(np.log(cs[6:]), np.log(Tb[6:]), 1)[0]
    scal[f"beta={beta}"] = dict(T_exact=Tb, T_approx=Ta, slope_upper_half=float(sl))
out["E4a"] = scal
print(json.dumps({k: v["slope_upper_half"] for k, v in scal.items() if k != "c"}, indent=1))

# (b) fresh vs snapshot across horizons N (p0=1e-3, beta=0.01, V=2, C=1, R=1, K=5)
Ns = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
fs = {"N": Ns, "fresh": [], "snapshot": [], "fresh_clean": [], "snapshot_clean": [],
      "fresh_T": [], "snapshot_T": [], "no_ckpt_clean": []}
p0, beta = 1e-3, 0.01
for N in Ns:
    for pol, K in [("fresh", 5.0), ("snapshot", 0.0)]:
        best = (np.inf, None, None)
        for T in np.unique(np.round(np.logspace(0, np.log10(N), 120)).astype(int)):
            E, Pc = th.makespan_and_correctness(N, int(T), p0, beta, mu, 1, 0.01, 2.0, 1.0, 1.0, K, pol)
            if E < best[0]:
                best = (E, int(T), Pc)
        fs[pol].append(best[0] / N); fs[pol + "_T"].append(best[1]); fs[pol + "_clean"].append(best[2])
    # no checkpointing, no verification: one shot
    ages = np.arange(N)
    fs["no_ckpt_clean"].append(float(np.prod(1 - th.hazard_linear(p0, beta, ages))))
out["E4b"] = fs
print(json.dumps(fs, indent=1, default=float))

# simulator spot-check of E4b at N=2000
det = DetSampler("point", rho=0.05, r=0.8)
spot = {}
for pol, K in [("fresh", 5.0), ("snapshot", 0.0)]:
    T = fs[pol + "_T"][Ns.index(2000)]
    tms, cl = run_many(1000, 11 if pol == "fresh" else 12, N=2000, T=T, p0=p0, beta=beta, det=det, m=1,
                       f=0.01, V=2.0, C=1.0, R=1.0, K=K, policy=pol)
    spot[pol] = dict(T=T, sim_time_per_step=float(tms.mean() / 2000),
                     th_time_per_step=fs[pol][Ns.index(2000)], sim_clean=float(cl.mean()),
                     th_clean=fs[pol + "_clean"][Ns.index(2000)])
out["E4b_sim_check"] = spot
print(json.dumps(spot, indent=1))

# (c) Pareto over T (fresh, N=5000)
N = 5000
par = {"T": [], "time": [], "clean": []}
for T in np.unique(np.round(np.logspace(0, 3, 50)).astype(int)):
    E, Pc = th.makespan_and_correctness(N, int(T), p0, 0.05, mu, 1, 0.01, 2.0, 1.0, 1.0, 5.0, "fresh")
    par["T"].append(int(T)); par["time"].append(E / N); par["clean"].append(Pc)
out["E4c"] = par
json.dump(out, open(RES + "e4_age_hazard.json", "w"), indent=1, default=float)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for beta, col in [(0.0, "#1f77b4"), (0.01, "#ff7f0e"), (0.1, "#d62728")]:
    d = scal[f"beta={beta}"]
    ax[0].loglog(cs, d["T_exact"], "o", color=col, label=f"exact optimum, beta={beta} (slope {d['slope_upper_half']:.2f})")
    ax[0].loglog(cs, d["T_approx"], "-", color=col, lw=1)
ax[0].set_xlabel("per-segment overhead c = mV + C + K"); ax[0].set_ylabel("optimal interval T*")
ax[0].set_title("(a) T* scaling: 1/2 (constant) to 1/3 (age hazard)"); ax[0].legend(fontsize=7)
ax[1].semilogx(Ns, fs["fresh"], "o-", label="fresh-context resumption (K=5)")
ax[1].semilogx(Ns, fs["snapshot"], "s-", label="context-snapshot restoration")
ax[1].set_xlabel("task length N (steps)"); ax[1].set_ylabel("optimal expected time per step")
ax[1].set_title("(b) resumption policy vs horizon"); ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
sc = ax[2].scatter(par["time"], par["clean"], c=np.log10(par["T"]), cmap="viridis", s=14)
plt.colorbar(sc, ax=ax[2], label="log10 T")
ax[2].set_xlabel("expected time per step"); ax[2].set_ylabel("P(final state clean)")
ax[2].set_title("(c) age hazard couples T to correctness"); ax[2].set_xscale("log")
plt.tight_layout(); plt.savefig(RES + "fig_age_hazard.png", dpi=150)
