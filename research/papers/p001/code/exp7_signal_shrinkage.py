"""Exp 7: Signal shrinkage near the irreducible loss.
(a) Power of the break LR test for the same exponent loss delta when the 3-decade design window is
    placed at increasing compute (ending at 1e22 ... 1e28 FLOP); truth = refit Chinchilla law (+break).
(b) Reducible fraction rho(C) = (L*-E)/L* along the frontier (governs the log-loss signature of a break).
(c) Paired data-doubling protocol under the data-capped law (Muennighoff eq.17, U=1e14): loss gap
    between training with U and 2U unique tokens at the same (N, D), and the number of paired runs
    needed to detect it (one-sided 5%, power 80%) for a given per-run log-loss noise."""
import json, time
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import exp3_simulation_power as S
import exp4_data_wall as W4
from common import RES, obj_base, frontier

t0 = time.time()
rng = np.random.default_rng(7)
S.rng = rng


def run_window(Cend, delta, sigma, Wd=3.0, K=12, M=20):
    lC = np.repeat(np.linspace(np.log(Cend) - Wd * np.log(10), np.log(Cend), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(2), np.log(200), M)), K)
    lN = 0.5 * (lC - np.log(6) - np.log(ratio)); lD = lC - np.log(6) - lN
    lCb = 0.5 * (lC.min() + lC.max())
    lL = S.simulate(lN, lD, lC, delta, sigma, lCb)
    n = len(lL)
    f0 = min((minimize(obj_base, x0, args=(lN, lD, lL, S.BIG), method="BFGS", jac=True)
              for x0 in [S.ptrue, S.ptrue + np.array([0.5, -0.5, 0.02, 0.02, -0.02])]), key=lambda r: r.fun)
    f1 = min((minimize(S.obj_brk, np.r_[f0.x, d0], args=(lN, lD, lC, lL, lCb), method="BFGS", jac=True)
              for d0 in [0.0, 0.05, 0.15]), key=lambda r: r.fun)
    lr = n * np.log((2 * f0.fun) / max(2 * f1.fun, 1e-300))
    return 0.5 * stats.chi2.sf(lr, 1) if f1.x[5] > 0 else 1.0


res_a = []
for sigma in [0.0025, S.SIG0]:
    for Cend in [1e22, 1e24, 1e26, 1e28]:
        for d in [0.0, 0.02, 0.05, 0.1, round(S.gamma(S.P), 3)]:
            ps = np.array([run_window(Cend, d, sigma) for _ in range(60)])
            res_a.append(dict(sigma=float(sigma), C_end=Cend, delta=d, power=float((ps < 0.05).mean())))
            print(res_a[-1], f"{time.time()-t0:.0f}s", flush=True)

Cs = np.array([1e20, 1e22, 1e24, 1e26, 1e28, 1e30])
Ls = frontier(S.P, Cs)[0]
rho = [dict(C=float(c), Lstar=float(l), rho=float((l - S.P["E"]) / l)) for c, l in zip(Cs, Ls)]
print(rho)

U = 1e14
Cwall = json.load(open(f"{RES}/exp4_data_wall.json"))["caps"]["1e+14"]["C_wall"]
z = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
res_c = []
for m in [1, 3, 10, 30, 100]:
    C = np.array([m * Cwall])
    Lu, Nu, Du = W4.frontier(C, U)
    L2 = W4.loss(Nu, Du, 2 * U)[0]
    gap = float(np.log(Lu[0]) - np.log(L2))   # log-loss gap at identical (N, D)
    row = dict(C_over_Cwall=m, C=float(C[0]), N=float(Nu[0]), D=float(Du[0]), epochs=float(Du[0] / U),
               log_loss_gap=gap)
    for sigma in [0.001, 0.0025, S.SIG0]:
        row[f"pairs_needed_sigma{sigma:.4f}"] = float(np.ceil((z * np.sqrt(2) * sigma / gap) ** 2)) if gap > 0 else None
    res_c.append(row)
    print(row)
json.dump(dict(window_power=res_a, rho=rho, paired_doubling=res_c), open(f"{RES}/exp7_signal_shrinkage.json", "w"), indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for j, sigma in enumerate([0.0025, S.SIG0]):
    for i, Cend in enumerate([1e22, 1e24, 1e26, 1e28]):
        rs = [r for r in res_a if r["sigma"] == sigma and r["C_end"] == Cend]
        ax[j].plot([r["delta"] for r in rs], [r["power"] for r in rs], "-o", color=f"C{i}", label=f"window ends {Cend:.0e} FLOP")
    ax[j].axhline(0.05, color="k", ls=":", lw=.8)
    ax[j].set_xlabel(r"true exponent loss $\delta$ ($\delta=\gamma$ means reducible loss stops falling)")
    ax[j].set_title(f"per-run noise sd {sigma:.4f} (log-loss), 240 runs/3 decades", fontsize=9)
ax[0].set_ylabel("power of break test (5%)"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{RES}/fig5_signal_shrinkage.png", dpi=150)
