"""Power-law feature spectra: scale-invariant 'capacity tax' of unification.
(E1) population ratio vs capacity r vs closed form R = (phi M^{1/a} + (1-phi) M)^a / M
(E2) finite-data df-model trajectories + a Monte Carlo check at d=512.
Run: OMP_NUM_THREADS=1 python3 exp_powerlaw.py"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import orth, rrr_risk_curve

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
t0 = time.time()


def spectra(K, phi, M, alpha):
    Ks = int(round(phi * K)); Kp = K - Ks
    shared = (np.arange(1, Ks + 1) / phi) ** (-alpha) if Ks else np.array([])
    priv = (np.arange(1, Kp + 1) / (1 - phi)) ** (-alpha) if Kp else np.array([])
    spec = np.sort(np.concatenate([shared, priv]))[::-1]           # one modality
    pooled = np.sort(np.concatenate([M * shared] + [priv] * M))[::-1]
    return spec, pooled


def pop_ratio(spec, pooled, M, r):
    Lu = pooled[r:].sum() / M
    Ls = spec[r:].sum()
    return Lu / Ls


def R_closed(phi, M, alpha):
    return (phi * M ** (1 / alpha) + (1 - phi) * M) ** alpha / M


res = {"E1": [], "E2": {}}
K = 200000
rs = np.unique(np.logspace(1, 4, 25).astype(int))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for alpha, ax in zip([1.5, 2.0, 3.0], axs):
    for phi, ls in zip([0.25, 0.5, 0.75], ["-", "--", ":"]):
        for M, c in zip([2, 4, 8], ["C0", "C1", "C2"]):
            spec, pooled = spectra(K, phi, M, alpha)
            rat = [pop_ratio(spec, pooled, M, r) for r in rs]
            Rc = R_closed(phi, M, alpha)
            res["E1"].append(dict(alpha=alpha, phi=phi, M=M, rs=rs.tolist(), ratio=rat,
                                  R_closed=Rc, kappa=Rc ** (1 / (alpha - 1)),
                                  saving_vs_M_specialists=M / Rc ** (1 / (alpha - 1)),
                                  rel_err_at_r1000=abs(rat[np.searchsorted(rs, 1000)] - Rc) / Rc))
            ax.plot(rs, rat, ls, color=c, lw=1)
            ax.plot([rs[-1] * 1.3], [Rc], "<", color=c, ms=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("core capacity r")
    ax.set_title(f"alpha={alpha}  (colors M=2,4,8; style phi=.25,.5,.75)", fontsize=8)
axs[0].set_ylabel("population L_u / L_s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figE1_powerlaw_tax.png"), dpi=150)
print("E1", time.time() - t0)

# ---- E2: finite data, df model (theory) + Monte Carlo at d=512 -------------
d, q, M, phi, alpha = 512, 96, 4, 0.5, 2.0
Ks = Kp = 48
rng = np.random.default_rng(0)
shared = (np.arange(1, Ks + 1) / phi) ** (-alpha)
priv = (np.arange(1, Kp + 1) / (1 - phi)) ** (-alpha)
scale = 25.0  # overall signal energy multiplier
basis = orth(rng.standard_normal((d, Ks + M * Kp)))
S = basis[:, :Ks]
Ws = []
for m in range(M):
    P = basis[:, Ks + m * Kp: Ks + (m + 1) * Kp]
    AB = orth(rng.standard_normal((q, Ks + Kp)))
    sv = np.sqrt(scale * np.concatenate([shared, priv]))
    Ws.append(AB @ (sv[:, None] * np.concatenate([S, P], 1).T))
spec_u = np.clip(np.linalg.eigvalsh(sum(W.T @ W for W in Ws)), 0, None)
spec_s = [np.clip(np.linalg.eigvalsh(W.T @ W), 0, None) for W in Ws]
rs2 = [2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 200, 240]
ns = [500, 2000, 8000, 32000, 128000, 10 ** 7]
E2 = {}
for n in ns:
    th, sim = [], []
    for r in rs2:
        Lu = rrr_risk_curve(spec_u, 1.0, n, d, M * q, r) / M
        Ls = np.mean([rrr_risk_curve(e, 1.0, n, d, q, r) for e in spec_s])
        th.append(Lu / Ls)
    # Monte Carlo (orthogonal design), 4 reps, oracle rank chosen on a grid
    for r in rs2:
        vals = []
        for rep in range(4):
            sd = np.sqrt(1.0 / n)
            Wh = [W + sd * rng.standard_normal(W.shape) for W in Ws]
            _, vec = np.linalg.eigh(sum(w.T @ w for w in Wh)); vec = vec[:, ::-1]
            grid = sorted(set(list(range(0, min(r, 40) + 1)) + list(range(0, r + 1, 8)) + [r]))
            bu = min(sum(np.sum((w @ vec[:, :k] @ vec[:, :k].T - W) ** 2) for w, W in zip(Wh, Ws)) for k in grid) / M
            bs = 0
            for w, W in zip(Wh, Ws):
                u_, s_, vt = np.linalg.svd(w, full_matrices=False)
                bs += min(np.sum(((u_[:, :k] * s_[:k]) @ vt[:k] - W) ** 2) for k in grid if k <= len(s_))
            vals.append(bu / (bs / M))
        sim.append(float(np.mean(vals)))
    E2[str(n)] = dict(theory=th, sim=sim)
    print("E2 n", n, time.time() - t0, np.round(th, 2).tolist(), np.round(sim, 2).tolist(), flush=True)
pop = [pop_ratio(np.sort(spec_s[0])[::-1], np.sort(spec_u)[::-1], M, r) for r in rs2]
res["E2"] = dict(d=d, q=q, M=M, phi=phi, alpha=alpha, Ks=Ks, Kp=Kp, scale=scale, rs=rs2,
                 runs=E2, population_ratio=pop, R_closed=R_closed(phi, M, alpha))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
cols = plt.cm.viridis(np.linspace(0, .9, len(ns)))
for c, n in zip(cols, ns):
    ax.plot(rs2, E2[str(n)]["theory"], "-", color=c, lw=1)
    ax.plot(rs2, E2[str(n)]["sim"], "o", color=c, ms=3, label=f"n={n:g}")
ax.axhline(R_closed(phi, M, alpha), color="r", ls=":", lw=1, label="closed-form R")
ax.axhline(1, color="k", ls="--", lw=.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("core capacity r"); ax.set_ylabel("L_u / L_s")
ax.set_title("power-law spectrum, M=4, phi=.5, alpha=2 (line: df theory, o: MC)", fontsize=8)
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(OUT, "figE2_powerlaw_finite_data.png"), dpi=150)
json.dump(res, open(os.path.join(OUT, "powerlaw_results.json"), "w"), indent=1, default=float)
print("done", time.time() - t0)
