"""E2: interval invariance (Thm 1) and the blind-spot floor (Thm 2).
E3: power-law returns to re-verification under heterogeneous detectability (Prop 3),
    and the time cost of re-verification under false alarms."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory as th
from simulator import DetSampler, run_many

RES = "../results/"
out = {}
N, p = 1000, 0.002
common = dict(N=N, p0=p, beta=0.0, f=0.02, V=1.0, C=0.5, R=1.0, K=0.0, policy="fresh")

# ---------------- E2a: correctness vs T for several m (point det, rho=0.1)
det = DetSampler("point", rho=0.1, r=0.7)
Ts = [1, 2, 5, 10, 25, 50, 100]
e2 = {"T": Ts}
seed = 100
for m in [1, 3]:
    sims, ses = [], []
    for T in Ts:
        seed += 1
        _, c = run_many(2000, seed, T=T, det=det, m=m, **common)
        sims.append(c.mean()); ses.append(np.sqrt(c.mean() * (1 - c.mean()) / len(c)))
    e2[f"m{m}_sim"] = sims; e2[f"m{m}_se"] = ses
    e2[f"m{m}_theory"] = float(th.p_clean_const(p, th.mu_point(0.7, m, 0.1), N))
# adaptive policies
adap = []
for T in [5, 20, 50]:
    seed += 1
    _, c = run_many(2000, seed, T=T, det=det, m=1, adaptive=True, **common)
    adap.append(c.mean())
e2["adaptive_T"] = [5, 20, 50]; e2["adaptive_sim"] = adap
# perfect-on-visible verifier (r=1) attains the floor
det1 = DetSampler("point", rho=0.1, r=1.0)
r1 = []
for T in [5, 50]:
    seed += 1
    _, c = run_many(2000, seed, T=T, det=det1, m=1, **common)
    r1.append(c.mean())
e2["r1_T"] = [5, 50]; e2["r1_sim"] = r1
e2["floor"] = float(th.floor_const(p, 0.1, N))
e2["no_verification"] = float((1 - p) ** N)
out["E2"] = e2
print(json.dumps(e2, indent=1))

# ---------------- E3: power-law returns
ms = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
e3 = {"m": ms.tolist()}
fam = {"Beta(0.3,1)": (0.3, 1.0, 0.0), "Beta(0.5,1)": (0.5, 1.0, 0.0),
       "Beta(1,1)": (1.0, 1.0, 0.0), "Beta(0.5,1)+atom 0.02": (0.5, 1.0, 0.02)}
for k, (a, b, rho) in fam.items():
    e3[k] = th.mu_beta(a, b, ms, rho).tolist()
e3["point r=0.5 (homogeneous)"] = th.mu_point(0.5, ms, 0.0).tolist()
# fitted slopes for m in [30,1000]
sl = {}
sel = ms >= 30
for k, (a, b, rho) in fam.items():
    y = th.mu_beta(a, b, ms[sel], rho)
    sl[k] = float(np.polyfit(np.log(ms[sel]), np.log(y), 1)[0])
e3["loglog_slope_m30_1000"] = sl
# Monte-Carlo check of mu_m for Beta(0.5,1) at a few m via the simulator
detb = DetSampler("beta", rho=0.0, a=0.5, b=1.0)
mc = {}
for m in [1, 4, 16]:
    seed += 1
    _, c = run_many(3000, seed, T=10, det=detb, m=m, **common)
    mc[m] = dict(sim=float(c.mean()), theory=float(th.p_clean_const(p, th.mu_beta(0.5, 1.0, m), N)))
e3["mc_check_Beta(0.5,1)"] = mc

# cost of re-verification with false alarms f: exact makespan vs m (T=10)
cost = {}
for f in [0.0, 0.02, 0.1]:
    E, Cc = [], []
    for m in ms[ms <= 200]:
        mu = float(th.mu_beta(0.5, 1.0, m, 0.0))
        e, c = th.makespan_and_correctness(N, 10, p, 0.0, mu, int(m), f, 1.0, 0.5, 1.0)
        E.append(e); Cc.append(c)
    cost[f"f={f}"] = dict(m=ms[ms <= 200].tolist(), time=E, clean=Cc)
e3["cost"] = cost
out["E3"] = e3
json.dump(out, open(RES + "e2_e3_floor_powerlaw.json", "w"), indent=1, default=float)

# ---------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a0 = ax[0]
for m, c in [(1, "#1f77b4"), (3, "#d62728")]:
    a0.errorbar(Ts, e2[f"m{m}_sim"], yerr=2 * np.array(e2[f"m{m}_se"]), fmt="o", color=c,
                label=f"simulated, m={m}")
    a0.axhline(e2[f"m{m}_theory"], color=c, ls="--", lw=1, label=f"Thm 1, m={m}")
a0.plot(e2["adaptive_T"], adap, "s", color="#2ca02c", label="adaptive policy (sim)")
a0.plot(e2["r1_T"], r1, "^", color="k", label="r=1 verifier (sim)")
a0.axhline(e2["floor"], color="k", lw=1.5, label="blind-spot ceiling (Thm 2)")
a0.axhline(e2["no_verification"], color="gray", ls=":", label="no verification")
a0.set_xscale("log"); a0.set_xlabel("checkpoint interval T (steps)")
a0.set_ylabel("P(final state clean)"); a0.set_title("(a) correctness is invariant to T")
a0.legend(fontsize=7, loc="center right")
a1 = ax[1]
for k in list(fam) + ["point r=0.5 (homogeneous)"]:
    a1.loglog(ms, e3[k], label=k)
a1.set_xlabel("verifier calls per segment m"); a1.set_ylabel(r"miss rate $\mu_m=E[(1-d)^m]$")
a1.set_title("(b) re-verification: power law vs exponential"); a1.legend(fontsize=7)
a1.set_ylim(1e-6, 1.1)
a2 = ax[2]
for k, v in cost.items():
    a2.plot(v["time"], v["clean"], "-o", ms=3, label=k)
a2.set_xscale("log"); a2.set_xlabel("expected makespan (time units)")
a2.set_ylabel("P(final state clean)"); a2.set_title("(c) time/correctness frontier, m=1..200")
a2.legend(fontsize=8)
plt.tight_layout(); plt.savefig(RES + "fig_floor_powerlaw.png", dpi=150)
print(json.dumps({"slopes": sl, "mc": mc}, indent=1))
