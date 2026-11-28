"""Experiment 3: layered oversight. Origin-based liability and monitoring incentives.

Loss probability (rare-event):  P = sum_j h_j(x_j) * prod_{k != j} E_k(y_k)
  h_j(x) = b_j / (1 + x / tau_j)      origination hazard of party j (own care x_j)
  E_k(y) = 1 / (1 + y / sigma_k)      probability that party k's monitoring misses another's error
Rule family (signal a, accuracy q, symmetric channel): the attributed party pays share alpha,
the remaining 1 - alpha is split equally among the other parties.
Parameters are ASSUMED (illustrative). Outputs: results/exp3_oversight.json, fig2_oversight.png
"""
import json, os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
L = 100.0


def loss_terms(x, y, b, tau, sigma):
    n = len(b)
    h = b / (1 + x / tau)
    E = 1 / (1 + y / sigma)
    esc = np.array([np.prod(np.delete(E, j)) for j in range(n)])
    return h * esc                                   # per-origin loss probability


def social(x, y, b, tau, sigma):
    return x.sum() + y.sum() + L * loss_terms(x, y, b, tau, sigma).sum()


def first_best(b, tau, sigma):
    n = len(b)
    best = None
    for s in range(8):
        z0 = np.random.default_rng(s).uniform(0, 5, 2 * n)
        r = minimize(lambda z: social(z[:n], z[n:], b, tau, sigma), z0, method="L-BFGS-B",
                     bounds=[(0, None)] * (2 * n))
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:n], best.x[n:], best.fun


def nash(M, b, tau, sigma, iters=500, damp=0.5):
    n = len(b)
    x = np.ones(n); y = np.ones(n)
    for it in range(iters):
        xo, yo = x.copy(), y.copy()
        for i in range(n):
            def cost(z):
                xx = x.copy(); yy = y.copy(); xx[i], yy[i] = z
                return z[0] + z[1] + L * float(M[i] @ loss_terms(xx, yy, b, tau, sigma))
            best = None
            for z0 in ([x[i], y[i]], [0.0, 0.0], [3.0, 3.0]):
                r = minimize(cost, z0, method="L-BFGS-B", bounds=[(0, None)] * 2)
                if best is None or r.fun < best.fun:
                    best = r
            x[i] = damp * x[i] + (1 - damp) * best.x[0]
            y[i] = damp * y[i] + (1 - damp) * best.x[1]
        resid = max(np.abs(x - xo).max(), np.abs(y - yo).max())
        if resid < 1e-8:
            break
    return x, y, resid


def run(tag, b, tau, sigma, qs, alphas):
    n = len(b)
    xf, yf, fb = first_best(b, tau, sigma)
    out = {"b": b.tolist(), "tau": tau.tolist(), "sigma": sigma.tolist(), "L": L,
           "first_best": {"x": xf.tolist(), "y": yf.tolist(), "cost": float(fb)}, "sweeps": {}}
    for q in qs:
        C = m.symmetric_C(n, q)
        rows = []
        for al in alphas:
            S = al * np.eye(n) + (1 - al) / (n - 1) * (np.ones((n, n)) - np.eye(n))
            M = S @ C
            x, y, resid = nash(M, b, tau, sigma)
            sc = social(x, y, b, tau, sigma)
            rows.append({"alpha": float(al), "x": x.tolist(), "y": y.tolist(), "resid": float(resid),
                         "social_cost": float(sc), "dwl_over_fb": float(sc / fb - 1),
                         "loss_prob": float(loss_terms(x, y, b, tau, sigma).sum())})
        k = int(np.argmin([r["social_cost"] for r in rows]))
        out["sweeps"][f"q={q}"] = {"rows": rows, "alpha_star": rows[k]["alpha"],
                                   "min_dwl_over_fb": rows[k]["dwl_over_fb"],
                                   "dwl_at_alpha_1": rows[-1]["dwl_over_fb"],
                                   "monitoring_at_alpha_1": rows[-1]["y"]}
    return out


alphas = np.round(np.linspace(0.30, 1.0, 29), 3)
res = {}
# symmetric two-party case
res["symmetric_n2"] = run("sym", np.array([0.1, 0.1]), np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                          [1.0, 0.8, 0.5344], alphas)
# developer originates most errors; user is the natural overseer
res["dev_originates_user_oversees"] = run("asym", np.array([0.15, 0.02]), np.array([1.0, 1.0]),
                                          np.array([3.0, 0.5]), [1.0, 0.8, 0.5344], alphas)
# three parties
res["symmetric_n3"] = run("sym3", np.full(3, 0.1), np.ones(3), np.ones(3), [1.0, 0.5344],
                          np.round(np.linspace(0.34, 1.0, 23), 3))
with open(os.path.join(OUT, "exp3_oversight.json"), "w") as f:
    json.dump(res, f, indent=1)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for q, st in [(1.0, "-"), (0.8, "--"), (0.5344, ":")]:
    rows = res["symmetric_n2"]["sweeps"][f"q={q}"]["rows"]
    ax.plot([r["alpha"] for r in rows], [r["dwl_over_fb"] for r in rows], st, label=f"q = {q}")
ax.set_xlabel("share alpha borne by the attributed party"); ax.set_ylabel("deadweight loss / first-best cost")
ax.set_title("Layered oversight, two symmetric parties", fontsize=10); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_oversight.png"), dpi=150)

for k, v in res.items():
    print(k, "FB", np.round(v["first_best"]["x"], 3), np.round(v["first_best"]["y"], 3), round(v["first_best"]["cost"], 3))
    for q, s in v["sweeps"].items():
        print("  ", q, "alpha*", s["alpha_star"], "minDWL", round(s["min_dwl_over_fb"], 4),
              "DWL@1", round(s["dwl_at_alpha_1"], 4), "y@1", np.round(s["monitoring_at_alpha_1"], 4),
              "maxresid", max(r["resid"] for r in s["rows"]))
