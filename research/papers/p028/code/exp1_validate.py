"""Exp 1: Monte Carlo validation of Lemma 1 / Theorem 1 / Theorem 2 on chains
and the verification-ceiling figure."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import latent_defect_rate, project_success, horizon, ceiling_q, escape

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(0)


def simulate_module(p, c, b, K, n):
    """Vectorised simulation of n independent modules. Returns (accepted, defective)."""
    accepted = np.zeros(n, bool); defective = np.zeros(n, bool)
    active = np.ones(n, bool)
    for _ in range(K):
        idx = np.where(active)[0]
        if idx.size == 0: break
        u = rng.random(idx.size)
        correct = u < p
        blind = rng.random(idx.size) < b
        detected = (~correct) & (~blind) & (rng.random(idx.size) < c)
        acc = ~detected
        accepted[idx[acc]] = True
        defective[idx[acc & ~correct]] = True
        active[idx[acc]] = False
    return accepted, defective


rows = []
for p, c, b, K in [(0.9, 0.9, 0.0, 5), (0.9, 0.9, 0.1, 5), (0.7, 0.95, 0.05, 3),
                   (0.5, 0.99, 0.2, 10), (0.95, 0.5, 0.0, 2)]:
    acc, dfc = simulate_module(p, c, b, K, 400_000)
    q_mc = dfc[acc].mean()
    q_th = latent_defect_rate(p, c, b)
    # project of N modules (chain, no propagation): success = all accepted & none defective
    rows.append(dict(p=p, c=c, b=b, K=K, q_mc=float(q_mc), q_theory=float(q_th),
                     abs_err=float(abs(q_mc - q_th))))

# direct project-level check: N-module chains
proj = []
for p, c, b, N in [(0.9, 0.9, 0.0, 30), (0.9, 0.99, 0.05, 30), (0.8, 0.95, 0.02, 60)]:
    R = 50_000
    acc, dfc = simulate_module(p, c, b, 1000, R * N)  # K large -> always accepted
    succ = (~dfc.reshape(R, N)).all(1).mean()
    proj.append(dict(p=p, c=c, b=b, N=N, success_mc=float(succ),
                     success_theory=float(project_success(N, latent_defect_rate(p, c, b)))))

# Figure: 50% horizon vs verifier coverage, for several blind-spot masses
p = 0.9
cs = np.linspace(0.0, 0.999, 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for b, col in zip([0.0, 0.01, 0.05, 0.2], ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b"]):
    q = latent_defect_rate(p, cs, b)
    ax[0].semilogy(cs, horizon(q), color=col, label=f"b = {b}")
    if b > 0:
        ax[0].axhline(horizon(ceiling_q(p, b)), color=col, ls=":", lw=0.8)
ax[0].set_xlabel("verifier coverage c on detectable defects")
ax[0].set_ylabel("50% horizon N$_{50}$ (modules)")
ax[0].set_title("(a) Retries + tests: ceiling set by blind spot b (p=0.9)")
ax[0].legend(frameon=False)
# panel b: horizon vs p for c=0.99 at different b
ps = np.linspace(0.3, 0.99, 300)
for b, col in zip([0.0, 0.01, 0.05, 0.2], ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b"]):
    ax[1].semilogy(ps, horizon(latent_defect_rate(ps, 0.99, b)), color=col, label=f"b = {b}")
ax[1].semilogy(ps, horizon(1 - ps), color="gray", ls="--", label="no verifier")
ax[1].set_xlabel("per-attempt correctness p")
ax[1].set_title("(b) Horizon vs generator skill (c=0.99)")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_ceiling.png"), dpi=150)

tbl = {}
for b in [0.0, 0.01, 0.05, 0.2]:
    tbl[str(b)] = {f"c={c}": float(horizon(latent_defect_rate(0.9, c, b))) for c in [0.5, 0.9, 0.99, 1.0 - 1e-12]}
json.dump(dict(module_level=rows, project_level=proj, horizon_table_p09=tbl),
          open(os.path.join(OUT, "exp1_validate.json"), "w"), indent=2)
print(json.dumps(dict(module_level=rows, project_level=proj, horizon_table_p09=tbl), indent=1))
