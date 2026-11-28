"""Analytical results and their numerical checks (Props 2-5 in the paper).
Writes ../results/theory.json."""
import os, json
import numpy as np
from scipy.stats import norm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng = np.random.default_rng(0)
out = {}

# ---------- Prop 2: disagreement rate kappa and mimic ceiling ----------
alphas = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
N = 2_000_000
a = rng.normal(size=N)
rows = []
for al in alphas:
    be = np.sqrt(1 - al ** 2)
    s = be * a + al * rng.normal(size=N) > 0
    sc = be * a + al * rng.normal(size=N) > 0
    kap_mc = float((s != sc).mean())
    ceil_mc = float((s == (a > 0)).mean())
    rows.append(dict(alpha=float(al), kappa_theory=float(np.arccos(be ** 2) / np.pi), kappa_mc=kap_mc,
                     mimic_ceiling_theory=float(1 - np.arccos(be) / np.pi), mimic_ceiling_mc=ceil_mc))
out["prop2"] = rows

# ---------- Prop 3: Bayes-optimal reporter under mixed labels ----------
def da_theory(lam, al, n=1_000_000):
    be = np.sqrt(1 - al ** 2)
    a = rng.normal(size=n)
    s = (be * a + al * rng.normal(size=n) > 0).astype(float)
    p = norm.cdf(be * a / al)
    q = lam * s + (1 - lam) * p
    r = (q > 0.5).astype(float)
    typ = (p > 0.5).astype(float)
    idio = s != typ
    return float((r[idio] == s[idio]).mean()), float((r == s).mean())

out["prop3"] = [dict(alpha=al, lam=l, DA_theory=da_theory(l, al)[0], acc_theory=da_theory(l, al)[1])
                for al in (0.2, 0.4, 0.6, 0.8) for l in np.round(np.linspace(0, 1, 21), 3).tolist()]

# ---------- Prop 4/5: linear-Gaussian ridge reporter, closed form vs sample ----------
def ridge_theory(al, rho):
    be = np.sqrt(1 - al ** 2)
    C = np.array([[1, be], [be, 1]])
    c_m = np.array([be, be * be]) / np.sqrt(2 * np.pi)          # Cov([a,z], s_c)
    c_s = np.array([be, 1.0]) / np.sqrt(2 * np.pi)              # Cov([a,z], s)
    A = np.linalg.inv(C + rho * np.eye(2))
    wm, ws = A @ c_m, A @ c_s
    vdom = np.array([be, 1.0]) / np.sqrt(1 + be ** 2)           # diff-of-means direction (a,z)
    return dict(w2_mimic=wm[1], w2_self=ws[1], dom_mimic=wm @ vdom, dom_self=ws @ vdom,
                w2_mimic_closed=rho * be ** 2 / (np.sqrt(2 * np.pi) * (al ** 2 + 2 * rho + rho ** 2)),
                w2_self_closed=(rho * be ** 2 + (1 + rho) * al ** 2) / (np.sqrt(2 * np.pi) * (al ** 2 + 2 * rho + rho ** 2)))

def ridge_sample(al, rho, n=200_000):
    be = np.sqrt(1 - al ** 2)
    a = rng.normal(size=n); z = be * a + al * rng.normal(size=n)
    s = (z > 0).astype(float); sc = (be * a + al * rng.normal(size=n) > 0).astype(float)
    X = np.stack([a, z], 1)
    def fit(y):
        yc = y - y.mean()
        return np.linalg.solve(X.T @ X / n + rho * np.eye(2), X.T @ yc / n)
    wm, ws = fit(sc), fit(s)
    vdom = np.array([be, 1.0]) / np.sqrt(1 + be ** 2)
    return dict(w2_mimic=wm[1], w2_self=ws[1], dom_mimic=wm @ vdom, dom_self=ws @ vdom)

rows = []
for rho in (1e-3, 1e-2, 1e-1):
    for al in np.round(np.concatenate([[0.0, 0.01, 0.02], np.linspace(0.05, 1.0, 20)]), 4):
        th, sm = ridge_theory(al, rho), ridge_sample(al, rho)
        rows.append(dict(rho=rho, alpha=float(al), **{k + "_th": float(v) for k, v in th.items()},
                         **{k + "_sm": float(v) for k, v in sm.items()}))
out["prop4"] = rows
err = max(max(abs(r["w2_mimic_th"] - r["w2_mimic_sm"]), abs(r["w2_self_th"] - r["w2_self_sm"]),
              abs(r["w2_mimic_th"] - r["w2_mimic_closed_th"]), abs(r["w2_self_th"] - r["w2_self_closed_th"]))
          for r in rows)
out["prop4_max_abs_err"] = float(err)
print("prop4 max abs error (theory vs sample vs closed form):", err)
json.dump(out, open(os.path.join(RES, "theory.json"), "w"), indent=1)
for r in out["prop2"]:
    print(r)
