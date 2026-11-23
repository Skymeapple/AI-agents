"""Optimal per-content cap a* of Theorem 5 on the E2 instance (q=0.8), grid of (eps, U); and the cap
realised by the trained E2 policies, split by U < q vs U >= q."""
import json
import numpy as np
import exp2_partial as e
from verify_partial import V_greedy, beta_K, f_K

KL, KA = e.kernels(0.8)
theory = {}
for eps in [0.01, 0.03, 0.1, 0.25, 0.5]:
    for U in [0.3, 0.5, 0.7, 0.78]:
        grid = np.linspace(f_K(KL, KA, U), 1, 2001)
        v = [(1 - eps) * V_greedy(e.Lc, e.A0c, KL, KA, U, beta_K(KL, KA, a)) + eps * a for a in grid]
        theory[f"eps{eps}_U{U}"] = float(grid[int(np.argmin(v))])
d = json.load(open("../results/e2_partial.json"))
emp = {}
for reg in ["STATIC", "ROBUST"]:
    for eps in [0.1, 0.25, 0.5]:
        rs = [r for r in d["rows"] if r["q"] == 0.8 and r["regime"] == reg and r["eps_eval"] == eps and 0.05 < r["U"] < 0.99]
        lo = [r for r in rs if r["U"] < 0.8]; hi = [r for r in rs if r["U"] >= 0.8]
        emp[f"{reg}_eps{eps}"] = dict(n_U_below_q=len(lo), n_cap_eq_1mq=sum(r["gmax"] < 0.2 + 1e-9 for r in lo),
                                     n_U_at_least_q=len(hi), n_cap_1=sum(r["gmax"] > 1 - 1e-9 for r in hi))
out = dict(theory_optimal_cap_q0_8=theory, trained_caps_q0_8=emp)
json.dump(out, open("../results/e2_cap.json", "w"), indent=1)
print(json.dumps(out, indent=1))
