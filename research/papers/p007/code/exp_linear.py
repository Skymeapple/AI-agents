"""Linear shared-core experiments: theory (df model) vs Monte Carlo.
Outputs JSON + figures in ../results/.  Run: python3 exp_linear.py"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import make_task, theory_losses, sim_losses

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SIG2 = 1.0
REPS = 30
t0 = time.time()
res = {}

def mc(d, q, ks, kp, M, n, r, theta=0.0, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    U, S, T = [], [], []
    for i in range(reps):
        Ws = make_task(d, q, ks, kp, M, theta=theta, rng=rng.integers(1 << 31))
        lu, ls = sim_losses(Ws, SIG2, n, r, rng)
        U.append(lu); S.append(ls)
    tu, ts = theory_losses(Ws, SIG2, n, r)  # theory is task-rotation invariant
    U, S = np.array(U), np.array(S)
    ratio = U / S
    return dict(sim_u=U.mean(), sim_s=S.mean(), sim_ratio=ratio.mean(),
                sim_ratio_se=ratio.std(ddof=1) / np.sqrt(reps),
                th_u=tu, th_s=ts, th_ratio=tu / ts)

# ---------------- (A) capacity sweep: competition barrier crossing ----------
d, q, ks, kp, M = 64, 16, 4, 4, 4
ns = [200, 1000, 5000, 100000]
rs = list(range(2, 25))
A = {}
for n in ns:
    A[n] = [mc(d, q, ks, kp, M, n, r) for r in rs]
res["A"] = dict(d=d, q=q, ks=ks, kp=kp, M=M, rs=rs,
                runs={str(n): A[n] for n in ns})
# crossover capacities
cross = {}
for n in ns:
    # competition window: first r with ratio > 1, and first r after it with ratio < 1
    def window(key):
        lo = next((r for r, o in zip(rs, A[n]) if o[key] > 1), None)
        hi = next((r for r, o in zip(rs, A[n]) if lo and r > lo and o[key] < 1), None)
        return lo, hi
    cross[str(n)] = dict(sim_window=window("sim_ratio"), theory_window=window("th_ratio"),
                         population_window=(ks + 1, ks + M * kp))
res["A"]["crossover"] = cross
print("A done", time.time() - t0, cross)

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))
for c, n in zip(cols, ns):
    sr = [o["sim_ratio"] for o in A[n]]; se = [o["sim_ratio_se"] for o in A[n]]
    tr = [o["th_ratio"] for o in A[n]]
    ax.errorbar(rs, sr, yerr=se, fmt="o", ms=3, color=c, label=f"sim n={n}")
    ax.plot(rs, tr, "-", color=c, lw=1.2)
ax.axhline(1, color="k", ls="--", lw=0.8)
ax.set_yscale("log"); ax.set_xlabel("core capacity r")
ax.set_ylabel("L_unified / L_specialist")
ax.set_title(f"Competition barrier (M={M}, k_s={ks}, k_p={kp}, d={d})", fontsize=9)
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(os.path.join(OUT, "figA_capacity_sweep.png"), dpi=150)

# ---------------- (B) modality budget: sweep M at fixed capacity ------------
d, q, ks, kp, r = 128, 16, 4, 4, 16
Ms = list(range(1, 15))
B = {}
for n in [1000, 5000, 100000]:
    B[str(n)] = [mc(d, q, ks, kp, Mm, n, r, reps=20) for Mm in Ms]
res["B"] = dict(d=d, q=q, ks=ks, kp=kp, r=r, Ms=Ms, runs=B)
Mstar = {}
for n, runs in B.items():
    def last_ok(key):
        ok = [o[key] <= 1 for o in runs]
        mm = 1
        for Mm, o in zip(Ms, ok):
            if Mm == 1: continue
            if o: mm = Mm
            else: break
        return mm
    Mstar[n] = dict(sim=last_ok("sim_ratio"), theory=last_ok("th_ratio"),
                    infinite_data_limit=(r - ks) / kp)
res["B"]["Mstar"] = Mstar
print("B done", time.time() - t0, Mstar)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
cols = plt.cm.plasma(np.linspace(0, 0.8, 3))
for c, (n, runs) in zip(cols, B.items()):
    ax[0].plot(Ms, [o["sim_u"] for o in runs], "o", color=c, ms=3, label=f"unified sim n={n}")
    ax[0].plot(Ms, [o["th_u"] for o in runs], "-", color=c, lw=1)
    ax[0].plot(Ms, [o["sim_s"] for o in runs], "x", color=c, ms=3)
    ax[1].plot(Ms, [o["sim_ratio"] for o in runs], "o", color=c, ms=3, label=f"sim n={n}")
    ax[1].plot(Ms, [o["th_ratio"] for o in runs], "-", color=c, lw=1)
ax[0].set_yscale("log"); ax[0].set_xlabel("number of modalities M")
ax[0].set_ylabel("per-modality excess risk"); ax[0].legend(fontsize=7)
ax[0].set_title("o: unified, x: specialists, line: theory", fontsize=8)
ax[1].axhline(1, color="k", ls="--", lw=.8); ax[1].axvline((r - ks) / kp, color="gray", ls=":", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel("number of modalities M"); ax[1].set_ylabel("ratio")
ax[1].set_title(f"modality budget, r={r}; dotted: (r-k_s)/k_p", fontsize=8); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_modality_budget.png"), dpi=150)

# ---------------- (C) misalignment phase diagram ----------------------------
d, q, ks, kp, M, n = 64, 16, 4, 4, 4, 1000
thetas = np.linspace(0, np.pi / 2, 10)
rs_c = list(range(4, 31, 2))
Cs = np.zeros((len(thetas), len(rs_c))); Ct = np.zeros_like(Cs)
for i, th in enumerate(thetas):
    for j, r in enumerate(rs_c):
        o = mc(d, q, ks, kp, M, n, r, theta=th, reps=12)
        Cs[i, j] = np.log(o["sim_ratio"]); Ct[i, j] = np.log(o["th_ratio"])
res["C"] = dict(d=d, q=q, ks=ks, kp=kp, M=M, n=n, thetas=thetas.tolist(), rs=rs_c,
                log_ratio_sim=Cs.tolist(), log_ratio_theory=Ct.tolist(),
                sign_agreement=float(np.mean(np.sign(Cs) == np.sign(Ct))))
print("C done", time.time() - t0, res["C"]["sign_agreement"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
vm = np.abs(Cs).max()
for a, Z, ttl in [(ax[0], Cs, "simulation"), (ax[1], Ct, "theory (df model)")]:
    im = a.imshow(Z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm,
                  extent=[rs_c[0] - 1, rs_c[-1] + 1, 0, 90])
    a.contour(np.array(rs_c), np.degrees(thetas), Z, levels=[0], colors="k", linewidths=1)
    a.set_xlabel("core capacity r"); a.set_title(f"log(L_u/L_s): {ttl}", fontsize=9)
ax[0].set_ylabel("misalignment angle theta (deg)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(os.path.join(OUT, "figC_misalignment.png"), dpi=150, bbox_inches="tight")

# ---------------- (D) theory accuracy vs data (where df model breaks) -------
d, q, ks, kp, M = 64, 16, 4, 4, 4
D = []
for n in [50, 100, 200, 500, 1000, 3000, 10000, 100000]:
    o = mc(d, q, ks, kp, M, n, 24, reps=20)
    o["n"] = n; o["snr_private"] = n / (SIG2 * np.sqrt(d * q))  # s^2 n / (sigma^2 sqrt(dq))
    D.append(o)
res["D"] = D
print("D done", time.time() - t0)
json.dump(res, open(os.path.join(OUT, "linear_results.json"), "w"), indent=1, default=float)
print("total s", time.time() - t0)
