"""Extra linear checks: (a) Theorem 3 lower bound vs exact G(rho);
(b) damped-Newton bound (Prop. 5) for delta << lambda_min^+;
(c) non-realizable (noisy, under-parameterised) case: transient overshoot above the endpoint."""
import numpy as np, json, os
from scipy.linalg import eigh
here = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(here, "..", "results")
from linear_theory_lib import make_tasks, null_proj, L1_traj_flow
out = {"bound": [], "damped": [], "noisy": []}
for seed in range(10):
    S1, S2 = make_tasks(False, seed); d = S1.shape[0]
    e0 = null_proj(S1) @ np.random.default_rng(200 + seed).standard_normal(d); e0 /= np.linalg.norm(e0)
    l1 = eigh(S1, eigvals_only=True)[-1]
    lam2 = eigh(S2, eigvals_only=True); mu = lam2[lam2 > 1e-9 * lam2[-1]].min()
    einf = null_proj(S2) @ e0; Finf = 0.5 * einf @ S1 @ einf
    for rho in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        ts = np.concatenate([np.linspace(0, 50, 2001), np.logspace(np.log10(50.1), 6, 800)])
        G = L1_traj_flow(S1, S2, e0, rho, ts).max()
        # Theorem 3: sqrt(2G) >= sup_tau [ sqrt(2F(tau)) - rho*tau/(1-rho)*l1^{1.5} ]
        taus = np.logspace(-2, 4, 2000)
        F = L1_traj_flow(S1, S2, e0, 0.0, taus)
        lb = np.max(np.sqrt(2 * F) - rho * taus / (1 - rho) * l1 ** 1.5)
        lb = max(lb, 0) ** 2 / 2
        out["bound"].append(dict(seed=seed, rho=rho, G=float(G), lower_bound=float(lb), F_inf=float(Finf)))
    # damped Newton (continuous-time) bound: G <= 0.5*||S1|| (delta/(e*lam_min^+))^2
    for rho in [0.1, 0.3]:
        H = (1 - rho) * S2 + rho * S1; lam, Q = eigh(H); pos = lam > 1e-9 * lam[-1]; lmin = lam[pos].min()
        for frac in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            delta = frac * lmin
            rates = np.where(pos, lam / (lam + delta), 0.0)
            ts = np.concatenate([np.linspace(0, 20, 2001), np.logspace(np.log10(20.1), 4, 400)])
            c = Q.T @ e0; E = Q @ (np.exp(-np.outer(rates, ts)) * c[:, None])
            L = 0.5 * np.einsum('it,ij,jt->t', E, S1, E)
            bound = 0.5 * l1 * (delta / (np.e * lmin)) ** 2
            out["damped"].append(dict(seed=seed, rho=rho, delta_over_lmin=frac, G=float(L.max()), bound=float(bound)))
# (c) noisy under-parameterised: d=30, n=40 per task
for seed in range(10):
    rg = np.random.default_rng(seed); d = 30
    A = rg.standard_normal((d, d)); X1 = rg.standard_normal((40, d)) * np.linspace(0.2, 2, d)
    X2 = rg.standard_normal((40, d)) @ np.linalg.qr(A)[0] * np.linspace(0.2, 2, d)
    w1t, w2t = rg.standard_normal(d), rg.standard_normal(d)
    y1 = X1 @ w1t + 0.3 * rg.standard_normal(40); y2 = X2 @ (0.5 * w1t + 0.5 * w2t) + 0.3 * rg.standard_normal(40)
    S1, S2 = X1.T @ X1 / 40, X2.T @ X2 / 40; b1, b2 = X1.T @ y1 / 40, X2.T @ y2 / 40
    w0 = np.linalg.solve(S1, b1); L1min = 0.5 * np.mean((X1 @ w0 - y1) ** 2)
    for rho in [0.01, 0.1, 0.3, 0.5]:
        H = (1 - rho) * S2 + rho * S1; winf = np.linalg.solve(H, (1 - rho) * b2 + rho * b1)
        lam, Q = eigh(H); ts = np.linspace(0, 400, 8001)
        W = winf[:, None] + Q @ (np.exp(-np.outer(lam, ts)) * (Q.T @ (w0 - winf))[:, None])
        L1 = 0.5 * np.mean((X1 @ W - y1[:, None]) ** 2, 0) - L1min
        out["noisy"].append(dict(seed=seed, rho=rho, peak=float(L1.max()), final=float(L1[-1]),
                                 overshoot=float(L1.max() - L1[-1])))
json.dump(out, open(os.path.join(R, "linear_extras.json"), "w"), indent=1)
import collections
for key in ["bound", "damped", "noisy"]:
    g = collections.defaultdict(list)
    for r in out[key]:
        k = tuple(v for kk, v in r.items() if kk in ("rho", "delta_over_lmin"))
        g[k].append(r)
    for k, rs in sorted(g.items()):
        print(key, k, {kk: round(float(np.mean([r[kk] for r in rs])), 6) for kk in rs[0] if kk not in ("seed", "rho", "delta_over_lmin")},
              "bound_holds_all" if key != "noisy" and all((r["G"] >= r["lower_bound"] - 1e-12) if key == "bound" else (r["G"] <= r["bound"] + 1e-12) for r in rs) else "")
