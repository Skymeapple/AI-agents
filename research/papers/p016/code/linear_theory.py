"""Linear-regression theory of the stability gap: numerical verification.
E1: 2-D rank-one tasks, closed form vs gradient flow.
E2: high-dimensional random tasks: gap depth G(rho), recovery time, commuting control,
    preconditioning (exact Newton on H_rho, damped, diagonal).
All outputs -> ../results/
"""
import numpy as np, json, os
from scipy.linalg import expm, eigh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(R, exist_ok=True)
out = {}

# ---------------- E1: 2-D closed form ----------------
def closed_form_2d(a, b, theta, rho, c=1.0):
    u = np.array([1.0, 0.0]); v = np.array([np.cos(theta), np.sin(theta)])
    H = (1-rho)*b*np.outer(v, v) + rho*a*np.outer(u, u)
    lam, Q = eigh(H)  # ascending
    l1, l2 = lam
    s = (u @ Q[:, 0]) * (Q[:, 0] @ np.array([0.0, 1.0]))
    if l1 <= 0:  # rho=0: no recovery
        return 0.5*a*c**2*s**2, np.inf, lam, s
    r = l1/l2
    G = 0.5*a*c**2*s**2*(1-r)**2*r**(2*r/(1-r))
    tstar = np.log(l2/l1)/(l2-l1)
    return G, tstar, lam, s

def L1_traj_flow(S1, S2, e0, rho, ts):
    H = (1-rho)*S2 + rho*S1
    lam, Q = eigh(H); lam = np.clip(lam, 0, None)
    c = Q.T @ e0
    E = (Q * 1.0) @ (np.exp(-np.outer(lam, ts)) * c[:, None])  # d x T
    return 0.5*np.einsum('it,ij,jt->t', E, S1, E)

a, b, theta = 1.0, 1.0, np.pi/4
u = np.array([1.0, 0.0]); v = np.array([np.cos(theta), np.sin(theta)])
S1 = a*np.outer(u, u); S2 = b*np.outer(v, v); e0 = np.array([0.0, 1.0])
ts = np.linspace(0, 400, 40001)
rhos = [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]
e1 = {"a": a, "b": b, "theta_deg": 45, "rows": []}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for rho in rhos:
    L = L1_traj_flow(S1, S2, e0, rho, ts)
    G, tstar, lam, s = closed_form_2d(a, b, theta, rho)
    num_peak = L.max(); num_t = ts[L.argmax()]
    e1["rows"].append({"rho": rho, "G_closed": G, "G_numeric": float(num_peak),
                       "tstar_closed": float(tstar), "tstar_numeric": float(num_t)})
    ax[0].plot(ts[:3001], L[:3001], label=f"ρ={rho}")
ax[0].set_xlabel("gradient-flow time t"); ax[0].set_ylabel("old-task loss L1(t)")
ax[0].set_title("(a) 2-D tasks, θ=45°"); ax[0].legend(fontsize=7)
rg = np.logspace(-4, np.log10(0.99), 200)
for th in [15, 30, 45, 60, 75]:
    Gs = [closed_form_2d(a, b, np.deg2rad(th), r)[0] for r in rg]
    ax[1].semilogx(rg, Gs, label=f"θ={th}°")
ax[1].set_xlabel("replay weight ρ"); ax[1].set_ylabel("gap depth G(ρ)")
ax[1].set_title("(b) closed-form gap depth"); ax[1].legend(fontsize=7)
ts_ = [closed_form_2d(a, b, theta, r)[1] for r in rg]
lam_min = [closed_form_2d(a, b, theta, r)[2][0] for r in rg]
ax[2].loglog(rg, ts_, label="peak time t*")
ax[2].loglog(rg, 1/np.array(lam_min), '--', label="1/λ_min(H_ρ) (recovery scale)")
ax[2].set_xlabel("replay weight ρ"); ax[2].set_title("(c) time scales"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig1_linear2d.png"), dpi=150); plt.close()
# slope of recovery scale for small rho
sl = np.polyfit(np.log(rg[:60]), np.log(1/np.array(lam_min[:60])), 1)[0]
e1["recovery_scale_loglog_slope_small_rho"] = float(sl)
e1["noreplay_forgetting_sin2cos2"] = 0.5*a*np.sin(theta)**2*np.cos(theta)**2
out["E1"] = e1

# ---------------- E2: high-dimensional ----------------
rng = np.random.default_rng(0)
d, n1, n2, k_shared = 100, 40, 40, 20

def make_tasks(commuting=False, seed=0):
    rg_ = np.random.default_rng(seed)
    if commuting:
        # both tasks diagonal in the same basis (shared eigenbasis), different spectra
        Qb, _ = np.linalg.qr(rg_.standard_normal((d, d)))
        idx1 = np.arange(0, 60); idx2 = np.arange(30, 90)
        D1 = np.zeros(d); D2 = np.zeros(d)
        D1[idx1] = rg_.uniform(0.2, 2, len(idx1)); D2[idx2] = rg_.uniform(0.2, 2, len(idx2))
        return Qb @ np.diag(D1) @ Qb.T, Qb @ np.diag(D2) @ Qb.T
    # task inputs lie in random subspaces sharing k_shared directions, anisotropic scales
    B = np.linalg.qr(rg_.standard_normal((d, d)))[0]
    U1 = B[:, :60]; U2 = np.concatenate([B[:, 60-k_shared:60], rg_.standard_normal((d, 60-k_shared))], 1)
    U2 = np.linalg.qr(U2)[0]
    X1 = rg_.standard_normal((n1, 60)) * np.linspace(0.3, 2, 60) @ U1.T
    X2 = rg_.standard_normal((n2, 60)) * np.linspace(0.3, 2, 60) @ U2.T
    return X1.T @ X1 / n1, X2.T @ X2 / n2

def null_proj(S, tol=1e-9):
    lam, Q = eigh(S); Z = Q[:, lam < tol*lam.max()]
    return Z @ Z.T

def run_precond(S1, S2, e0, rho, P, eta, steps):
    """discrete GD e <- e - eta P H e ; returns L1 and L2 trajectories"""
    H = (1-rho)*S2 + rho*S1
    M = np.eye(len(e0)) - eta * P @ H
    e = e0.copy(); L1 = []; L2 = []
    for _ in range(steps):
        L1.append(0.5*e@S1@e); L2.append(0.5*e@S2@e); e = M @ e
    L1.append(0.5*e@S1@e); L2.append(0.5*e@S2@e)
    return np.array(L1), np.array(L2)

e2 = {"d": d, "n1": n1, "n2": n2, "seeds": 10, "rows": []}
rho_grid = [0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7]
T = 200.0; ts = np.linspace(0, T, 4001)
curves_store = {}
for seed in range(10):
    for comm in [False, True]:
        S1, S2 = make_tasks(comm, seed)
        P1n = null_proj(S1)
        wstar = np.random.default_rng(100+seed).standard_normal(d)
        e0 = P1n @ np.random.default_rng(200+seed).standard_normal(d)  # e0 in ker S1
        e0 /= np.linalg.norm(e0)
        # no-replay terminal forgetting (Evron-type)
        P2n = null_proj(S2); einf = P2n @ e0; F_inf = 0.5*einf@S1@einf
        for rho in rho_grid:
            ts_long = np.concatenate([ts, np.logspace(np.log10(T)+1e-3, 5, 600)])
            L = L1_traj_flow(S1, S2, e0, rho, ts_long)
            G = float(L.max()); i = int(L.argmax())
            # recovery time: first time after peak where L1 < 0.1*G
            after = np.where(L[i:] < 0.1*G)[0]
            trec = float(ts_long[i+after[0]]) if (len(after) and rho > 0) else np.inf
            # initial curvature check: L1(t) ~ 0.5 (1-rho)^2 t^2 ||S1^{1/2} S2 e0||^2
            pred_c = 0.5*(1-rho)**2*(e0@S2@S1@S2@e0)
            tt = 1e-3; num_c = L1_traj_flow(S1, S2, e0, rho, np.array([tt]))[0]/tt**2
            e2["rows"].append({"seed": seed, "commuting": comm, "rho": rho, "G": G,
                               "F_inf_noreplay": float(F_inf), "t_peak": float(ts_long[i]),
                               "t_recover10": trec, "init_curv_pred": float(pred_c),
                               "init_curv_num": float(num_c), "L1_final": float(L[-1])})
            if seed == 0: curves_store[(comm, rho)] = (ts, L[:len(ts)])
out["E2"] = e2

# preconditioning study (discrete steps)
pre = {"rows": []}
for seed in range(10):
    S1, S2 = make_tasks(False, seed)
    e0 = null_proj(S1) @ np.random.default_rng(200+seed).standard_normal(d); e0 /= np.linalg.norm(e0)
    for rho in [0.01, 0.1, 0.3]:
        H = (1-rho)*S2 + rho*S1
        lmax = eigh(H, eigvals_only=True)[-1]
        configs = {"GD": (np.eye(d), 1.0/lmax)}
        for delta in [1e-4, 1e-3, 1e-2, 1e-1]:
            configs[f"damped_newton_delta={delta}"] = (np.linalg.inv(H + delta*lmax*np.eye(d)), 0.5)
        configs["exact_newton_pinv"] = (np.linalg.pinv(H, rcond=1e-10), 0.5)
        configs["diag_precond"] = (np.diag(1/(np.diag(H)+1e-3*lmax)), None)
        for name, (P, eta) in configs.items():
            if eta is None:
                eta = 1.0/np.abs(eigh(np.sqrt(P)@H@np.sqrt(P), eigvals_only=True)).max()
            steps = 20000 if name in ("GD", "diag_precond") else 400
            L1, L2 = run_precond(S1, S2, e0, rho, P, eta, steps)
            # gap = max L1; also L1 at the step where L2 first drops below 1% of its start
            L2_0 = L2[0]
            k = np.where(L2 < 0.01*L2_0)[0]; k = int(k[0]) if len(k) else -1
            pre["rows"].append({"seed": seed, "rho": rho, "method": name, "G": float(L1.max()),
                                "L1_final": float(L1[-1]), "L2_final_rel": float(L2[-1]/L2_0),
                                "steps_to_L2_1pct": k})
out["precond"] = pre

json.dump(out, open(os.path.join(R, "linear_results.json"), "w"), indent=1, default=float)

# figure 2: high-dim curves + G(rho) + trec(rho)
import collections
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for rho in [0.0, 0.01, 0.03, 0.1, 0.3]:
    t_, L_ = curves_store[(False, rho)]
    ax[0].plot(t_[:1500], L_[:1500], label=f"ρ={rho}")
t_, L_ = curves_store[(True, 0.1)]
ax[0].plot(t_[:1500], L_[:1500], 'k:', label="commuting, ρ=0.1")
ax[0].set_xlabel("t"); ax[0].set_ylabel("L1(t)"); ax[0].set_title("(a) d=100 random tasks (seed 0)"); ax[0].legend(fontsize=7)
rows = [r for r in e2["rows"] if not r["commuting"]]
Gm = collections.defaultdict(list); Tm = collections.defaultdict(list); Fm = []
for r in rows:
    Gm[r["rho"]].append(r["G"]/max(r["F_inf_noreplay"], 1e-12))
    if r["rho"] > 0: Tm[r["rho"]].append(r["t_recover10"])
xs = [x for x in rho_grid if x > 0]
ax[1].errorbar(xs, [np.mean(Gm[x]) for x in xs], [np.std(Gm[x]) for x in xs], marker='o')
ax[1].axhline(1, color='gray', ls='--', lw=0.8)
ax[1].set_xscale('log'); ax[1].set_xlabel("ρ"); ax[1].set_ylabel("G(ρ) / F∞(no replay)")
ax[1].set_title("(b) gap depth relative to no-replay forgetting")
ax[2].loglog(xs, [np.mean(Tm[x]) for x in xs], 'o-')
ax[2].loglog(xs, np.mean(Tm[xs[0]])*xs[0]/np.array(xs), 'k--', lw=0.8, label="∝1/ρ")
ax[2].set_xlabel("ρ"); ax[2].set_ylabel("time to recover to 10% of peak"); ax[2].legend(); ax[2].set_title("(c) recovery time")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig2_linear_highdim.png"), dpi=150); plt.close()
print("done")
