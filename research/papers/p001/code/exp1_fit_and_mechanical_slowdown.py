"""Exp 1: Refit the Chinchilla parametric law on Epoch AI's reconstructed data,
bootstrap uncertainty, and quantify the *mechanical* deceleration implied by a
constant-exponent law (Proposition 1)."""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

rng = np.random.default_rng(0)
N, D, C, L = load_chinchilla()
t0 = time.time()
keep = epoch_keep(L)   # Epoch AI rule: drop the 5 highest-loss runs -> 240 points
fit_240 = fit_base(N[keep], D[keep], L[keep])
fit_all = fit_base(N, D, L, grid=coarse_grid() + [list(fit_240["raw"])])  # sensitivity: all 245 points
res = np.log(L) - pred_log_base(fit_240["raw"], np.log(N), np.log(D))
print("fit245", {k: fit_all[k] for k in "A B E alpha beta".split()})
print("fit240 (Epoch exclusion rule)", {k: fit_240[k] for k in "A B E alpha beta".split()})
sigma = float(np.std(res[keep]))
print("residual sd (log loss), 240 pts:", sigma, "time", time.time() - t0)

# bootstrap
B = 300
boots = []
for i in range(B):
    idx = rng.integers(0, len(keep), len(keep))
    ii = keep[idx]
    f = fit_base(N[ii], D[ii], L[ii], init=fit_240["raw"])
    boots.append([f["A"], f["B"], f["E"], f["alpha"], f["beta"]])
boots = np.array(boots)
names = ["A", "B", "E", "alpha", "beta"]
ci = {n: [float(np.percentile(boots[:, j], 10)), float(np.percentile(boots[:, j], 90))] for j, n in enumerate(names)}
gam = boots[:, 3] * boots[:, 4] / (boots[:, 3] + boots[:, 4])
ci["gamma"] = [float(np.percentile(gam, 10)), float(np.percentile(gam, 90))]
print("80% CIs", ci)

p = fit_240
g = gamma(p)
# Mechanical slowdown quantities along the frontier
Cs = np.logspace(18, 30, 400)
Ls, Ns, Ds = frontier(p, Cs)
red = Ls - p["E"]
s_raw = -np.gradient(np.log(Ls), np.log(Cs))          # local log-log slope of raw loss
s_red = -np.gradient(np.log(red), np.log(Cs))         # local slope of reducible loss (constant = gamma)
gain_per_decade = Ls - frontier(p, Cs * 10)[0]

# bootstrap bands for s_raw at a few compute levels
Cq = np.array([1e21, 1e23, 1e25, 1e27, 1e29])
s_raw_b = []
for row in boots:
    q = dict(A=row[0], B=row[1], E=row[2], alpha=row[3], beta=row[4])
    Lq = frontier(q, Cq)[0]; Lq2 = frontier(q, Cq * 1.01)[0]
    s_raw_b.append(-(np.log(Lq2) - np.log(Lq)) / np.log(1.01))
s_raw_b = np.array(s_raw_b)

out = dict(
    fit245={k: float(fit_all[k]) for k in names}, fit240={k: float(p[k]) for k in names},
    gamma=float(g), residual_sd_log=sigma, ci80=ci, n_boot=B,
    frontier_table=[dict(C=float(c), Lstar=float(frontier(p, np.array([c]))[0][0]),
                         reducible=float(frontier(p, np.array([c]))[0][0] - p["E"]),
                         s_raw=float(np.interp(np.log(c), np.log(Cs), s_raw)),
                         s_raw_ci80=[float(np.percentile(s_raw_b[:, k], 10)), float(np.percentile(s_raw_b[:, k], 90))],
                         gain_next_decade=float(np.interp(np.log(c), np.log(Cs), gain_per_decade)))
                    for k, c in enumerate(Cq)],
    decade_gain_ratio=float(10 ** (-g)),
    compute_multiplier_for_equal_abs_gain_per_decade=float(10 ** g),
)
json.dump(out, open(f"{RES}/exp1_fit.json", "w"), indent=2)
np.save(f"{RES}/exp1_boots.npy", boots)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].loglog(C, L, ".", ms=3, alpha=.5, color="gray", label="Chinchilla runs (Epoch recon.)")
ax[0].loglog(Cs, Ls, color="C0", label="fitted compute-optimal $L^*(C)$")
ax[0].axhline(p["E"], ls="--", color="C3", lw=1, label=f"E={p['E']:.2f}")
ax[0].set_xlabel("training FLOP"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=7); ax[0].set_ylim(1.7, 5.2)
ax[1].semilogx(Cs, s_raw, label=r"raw: $-d\log L/d\log C$")
ax[1].semilogx(Cs, s_red, label=r"reducible: $-d\log(L-E)/d\log C$")
ax[1].set_xlabel("training FLOP"); ax[1].set_ylabel("local scaling exponent"); ax[1].legend(fontsize=7)
ax[1].set_title("constant-exponent law, declining raw slope", fontsize=9)
ax[2].semilogx(Cs, gain_per_decade); ax[2].set_xlabel("training FLOP")
ax[2].set_ylabel("loss reduction from next 10x compute"); ax[2].set_title(f"each decade buys {10**(-g):.2f}x the previous", fontsize=9)
for a in ax: a.axvspan(C.min(), C.max(), color="gray", alpha=.1)
plt.tight_layout(); plt.savefig(f"{RES}/fig1_mechanical_slowdown.png", dpi=150)
print(json.dumps(out, indent=1))
