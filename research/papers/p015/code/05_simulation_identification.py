"""Identification simulation on METR's actual design (task lengths and run counts of one agent).

Scenarios (task-level cumulative hazard Lambda_i(T) = exp(mu + k log(T/60) + sigma u_i)):
  A 'frailty + constant hazard' : k = 1.0, sigma = 2.5
  B 'homogeneous + Lindy'       : k = 0.63, sigma = 0   (pooled Weibull shape seen in data)
  C 'empirical'                 : k = 1.5, sigma = 3.2  (median estimates from 03_frailty_glmm)
mu is calibrated so that every scenario has the same pooled 50% horizon (60 min).
For each replicate we compute: pooled Weibull shape k_marg (what single-run, pooled analyses
see), the stratified heterogeneity index H, and the random-effects estimate of k and sigma.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from common import *

d = load_runs()
design = task_table(d[d.alias == "Claude 4 Opus (Inspect)"])
T = design["T"].values
n = design["n"].values
rng = np.random.default_rng(11)
x, wq = np.polynomial.hermite_e.hermegauss(80)
wq = wq / wq.sum()


def pooled_S(mu, k, sig, TT):
    return (np.exp(-np.exp(np.clip(mu + k * np.log(TT / T_REF)[:, None] + sig * x[None, :], -30, 6))) * wq).sum(1)


def calib_mu(k, sig):
    return optimize.brentq(lambda m: pooled_S(m, k, sig, np.array([60.0]))[0] - 0.5, -20, 20)


SCEN = {"A: frailty, k=1": (1.0, 2.5), "B: homogeneous, k=0.63": (0.63, 0.0), "C: empirical, k=1.5": (1.5, 3.2)}
EDGES = [0, 1, 4, 15, 60, 240, 960, 1e9]


def H_strat(tt):
    Hs, ws = [], []
    tt = tt.copy()
    tt["bin"] = pd.cut(tt["T"], EDGES, right=False)
    for _, g in tt.groupby("bin", observed=True):
        if len(g) < 8:
            continue
        p = g.s / g.n
        m = len(g)
        Pb = p.mean()
        if not (0.03 < Pb < 0.97):
            continue
        Pb2 = (p.sum() ** 2 - (p ** 2).sum()) / (m * (m - 1))
        M2 = (g.s * (g.s - 1) / (g.n * (g.n - 1))).mean()
        Hs.append((M2 - Pb2) / (Pb * (1 - Pb)))
        ws.append(m)
    return np.average(Hs, weights=ws)


rows = []
NREP = 30
for name, (k, sig) in SCEN.items():
    mu = calib_mu(k, sig)
    for rep in range(NREP):
        u = rng.standard_normal(len(T))
        p = np.exp(-np.exp(np.clip(mu + k * np.log(T / T_REF) + sig * u, -30, 6)))
        s = rng.binomial(n, p)
        tt = pd.DataFrame(dict(task_id=np.arange(len(T)), n=n, s=s, T=T, family=np.arange(len(T))))
        # pooled Weibull (equal task weights, run level)
        Tr = np.repeat(T, n)
        yr = np.concatenate([np.r_[np.ones(si), np.zeros(ni - si)] for si, ni in zip(s, n)])
        wr = np.repeat(1.0 / n, n)
        th, _ = fit_family("Weibull", Tr, yr, wr)
        r = fit_glmm(tt, "cloglog", nested=False)
        rows.append(dict(scenario=name, rep=rep, k_true=k, sig_true=sig, k_marg=np.exp(th[1]),
                         H=H_strat(tt), k_cond=r.x[1], sig_hat=np.exp(r.x[2])))
    print(name, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{RES}/simulation_identification.csv", index=False)
S = R.groupby("scenario")[["k_true", "sig_true", "k_marg", "H", "k_cond", "sig_hat"]].agg(["mean", "std"])
with open(f"{RES}/simulation_summary.txt", "w") as f:
    f.write(f"Design: {len(T)} tasks, {n.sum()} runs (Claude 4 Opus TH1.1 lengths and run counts); {NREP} reps\n")
    f.write(S.round(3).to_string() + "\n")
print(open(f"{RES}/simulation_summary.txt").read())

# Figure: pooled curves (indistinguishable) vs task-level distributions (distinguishable)
TT = np.logspace(-1, 4, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = ["#4a6fa5", "#c0392b", "#2e8b57"]
for (name, (k, sig)), c in zip(SCEN.items(), cols):
    mu = calib_mu(k, sig)
    ax[0].plot(TT, pooled_S(mu, k, sig, TT), color=c, lw=2, label=name)
    for q, ls in [(0.1, ":"), (0.5, "-"), (0.9, ":")]:
        z = np.quantile(np.random.default_rng(0).standard_normal(100000), q) if sig > 0 else 0
        ax[1].plot(TT, np.exp(-np.exp(np.clip(mu + k * np.log(TT / T_REF) + sig * z, -30, 6))), color=c, ls=ls, lw=1.5)
ax[0].set_xscale("log")
ax[0].set_xlabel("human task length T (min)")
ax[0].set_ylabel("pooled P(success)")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) pooled curves: near-identical")
ax[1].set_xscale("log")
ax[1].set_xlabel("human task length T (min)")
ax[1].set_ylabel("task-level P(success)")
ax[1].set_title("(b) 10th/50th/90th-percentile task: distinct")
plt.tight_layout()
plt.savefig(f"{RES}/fig_identification.png", dpi=160)
