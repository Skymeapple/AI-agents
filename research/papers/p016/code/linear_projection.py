"""Revision (R1): straight-path / projection baselines in the d=100 linear model (answers reviewer Q2).
GD on the replay objective vs.
  OGD-style:  e <- e - eta * P_{ker S1} grad   (project out the span of old-task inputs; Farajtabar et al. 2020 analogue)
  GEM-style:  e <- e - eta * g~,  g~ = g - min(0,<g,g1>)/||g1||^2 g1  (aggregate old-task gradient constraint)
  exact GN:   e <- e - 0.5 * H^+ grad
rho in {0.01, 0.1, 0.3}, 10 seeds. Output: ../results/linear_projection.json"""
import numpy as np, json, os
from scipy.linalg import eigh
from linear_theory_lib import make_tasks, null_proj
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); d = 100
rows = []
for seed in range(10):
    S1, S2 = make_tasks(False, seed); P1n = null_proj(S1)
    e0 = P1n @ np.random.default_rng(200 + seed).standard_normal(d); e0 /= np.linalg.norm(e0)
    for rho in [0.01, 0.1, 0.3]:
        H = (1 - rho) * S2 + rho * S1; lmax = eigh(H, eigvals_only=True)[-1]; Hp = np.linalg.pinv(H, rcond=1e-10)
        lmo = eigh(P1n @ S2 @ P1n, eigvals_only=True)[-1]
        for meth in ["GD", "OGD", "GEM", "GN"]:
            e = e0.copy(); L1 = [0.0]; L2 = [0.5 * e @ S2 @ e]
            steps = 400 if meth == "GN" else 20000
            for _ in range(steps):
                g = H @ e
                if meth == "GD": e = e - g / lmax
                elif meth == "OGD": e = e - P1n @ g / lmo
                elif meth == "GEM":
                    g1 = S1 @ e; ip = g @ g1
                    if ip < 0: g = g - ip / (g1 @ g1) * g1
                    e = e - g / lmax
                else: e = e - 0.5 * Hp @ g
                L1.append(0.5 * e @ S1 @ e); L2.append(0.5 * e @ S2 @ e)
            L1 = np.array(L1); L2 = np.array(L2); k = np.where(L2 < 0.01 * L2[0])[0]
            rows.append(dict(seed=seed, rho=rho, method=meth, G=float(L1.max()), L2_final_rel=float(L2[-1] / L2[0]),
                             steps_to_L2_1pct=int(k[0]) if len(k) else -1))
json.dump(rows, open(os.path.join(R, "linear_projection.json"), "w"), indent=1)
import collections
g = collections.defaultdict(list)
for r in rows: g[(r["method"], r["rho"])].append(r)
for k, rs in sorted(g.items()):
    print(k, "G mean %.3g max %.3g" % (np.mean([r["G"] for r in rs]), max(r["G"] for r in rs)),
          "L2rel max %.2g" % max(r["L2_final_rel"] for r in rs), "steps1%%", [r["steps_to_L2_1pct"] for r in rs])
