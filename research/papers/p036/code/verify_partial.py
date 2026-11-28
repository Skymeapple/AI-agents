"""Exact-LP checks of the partial-mimicry results (Theorem 4, Theorem 5) and the graded-response
proposition (Proposition 6) on random finite instances.

Theorem 4 (mimicry gap).  Kernels K_L, K_A independent of content.  For any attack family Q with
    delta = inf_{Q in conv(Q)} TV(Q, L),
  min_{phi: U>=u} max_{Q} ASR(phi,Q) >= f_K(u) - delta,  and the s-only LR test attains f_K(u).
Theorem 5 (eps-contamination frontier).  Q_eps = {(1-eps) A0 + eps R : R arbitrary}.  Then
  ASR*_eps(u) = min_a (1-eps) V(u,a) + eps a,
  V(u,a) = min { E_A0 f_K(u(c)) : E_L u(c) >= u, 0 <= u(c) <= beta_K(a) }.
  V is computed here by an independent greedy (fractional-knapsack over the linear pieces of f_K),
  the full minimax by an LP over phi(c,s).
Proposition 6 (graded responses).  Under exact mimicry with no unforgeable signal, the minimax expected
  harm at helpfulness H equals the lower convex envelope of the (b(r), h(r)) points at H.
"""
import json
import numpy as np
from scipy.optimize import linprog, minimize_scalar

rng = np.random.default_rng(1)


def roc_pieces(KL, KA):
    """Linear pieces of the NP frontier f_K of (K_L,K_A): list of (dU, dASR) in increasing slope order."""
    lr = KL / np.maximum(KA, 1e-300)
    order = np.argsort(-lr)
    return [(KL[i], KA[i]) for i in order]


def f_K(KL, KA, u):
    acc_u = acc_a = 0.0
    for du, da in roc_pieces(KL, KA):
        if acc_u >= u - 1e-15: break
        t = min(1.0, (u - acc_u) / du) if du > 0 else 1.0
        acc_u += t * du; acc_a += t * da
    return acc_a


def beta_K(KL, KA, a):
    """max power u with f_K(u) <= a."""
    acc_u = acc_a = 0.0
    for du, da in roc_pieces(KL, KA):
        if da <= 0:
            acc_u += du; continue
        if acc_a + da <= a + 1e-15:
            acc_u += du; acc_a += da
        else:
            acc_u += du * (a - acc_a) / da; break
    return min(acc_u, 1.0)


def V_greedy(L, A0, KL, KA, u, cap):
    """min sum_c A0[c] f_K(u_c) s.t. sum_c L[c] u_c >= u, u_c <= cap.  f_K piecewise linear convex, so
    fill (content, piece) items in increasing order of cost-per-legit-mass A0[c]*slope/L[c]."""
    items = []
    for c in range(len(L)):
        acc = 0.0
        for du, da in roc_pieces(KL, KA):
            take = max(0.0, min(du, cap - acc)); acc += du
            if take <= 0 or L[c] <= 0: continue
            slope = da / du if du > 0 else 0.0
            items.append((A0[c] * slope / L[c], L[c] * take, A0[c] * slope * take))
    items.sort(key=lambda z: z[0])
    got = cost = 0.0
    for _, gain, c_full in items:
        if got >= u - 1e-15: break
        t = min(1.0, (u - got) / gain)
        got += t * gain; cost += t * c_full
    return cost if got >= u - 1e-9 else np.inf


def formula_T5(L, A0, KL, KA, u, eps):
    fU = f_K(KL, KA, u)
    obj = lambda a: (1 - eps) * V_greedy(L, A0, KL, KA, u, beta_K(KL, KA, a)) + eps * a
    grid = np.linspace(fU, 1.0, 801)
    vals = np.array([obj(a) for a in grid])
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
    return min(vals[i], r.fun)


def lp_T5(L, A0, KL, KA, u, eps):
    nc, ns = len(L), len(KL); nv = nc * ns
    cobj = np.zeros(nv + 1); cobj[:nv] = (1 - eps) * np.outer(A0, KA).ravel(); cobj[-1] = eps
    Aub, bub = [], []
    for c in range(nc):
        row = np.zeros(nv + 1); row[c * ns:(c + 1) * ns] = KA; row[-1] = -1; Aub.append(row); bub.append(0)
    row = np.zeros(nv + 1); row[:nv] = -np.outer(L, KL).ravel(); Aub.append(row); bub.append(-u)
    r = linprog(cobj, A_ub=Aub, b_ub=bub, bounds=[(0, 1)] * (nv + 1), method="highs")
    return r.fun


def lp_minimax_finite(L, Qs, KL, KA, u):
    nc, ns = len(L), len(KL); nv = nc * ns
    cobj = np.zeros(nv + 1); cobj[-1] = 1
    Aub, bub = [], []
    for Q in Qs:
        row = np.zeros(nv + 1); row[:nv] = np.outer(Q, KA).ravel(); row[-1] = -1; Aub.append(row); bub.append(0)
    row = np.zeros(nv + 1); row[:nv] = -np.outer(L, KL).ravel(); Aub.append(row); bub.append(-u)
    return linprog(cobj, A_ub=Aub, b_ub=bub, bounds=[(0, 1)] * (nv + 1), method="highs").fun


def tv_to_hull(L, Qs):
    """min_lambda TV(sum_i lambda_i Q_i, L) by LP with slack variables."""
    k, n = len(Qs), len(L)
    # vars: lambda (k), t (n);  min 0.5 sum t ; t >= +-(sum lambda Q - L)
    c = np.r_[np.zeros(k), 0.5 * np.ones(n)]
    Qm = np.array(Qs).T
    Aub = np.block([[Qm, -np.eye(n)], [-Qm, -np.eye(n)]]); bub = np.r_[L, -L]
    Aeq = np.r_[np.ones(k), np.zeros(n)][None]; beq = [1.0]
    return linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (k + n), method="highs").fun


if __name__ == "__main__":
    res = dict(T4_min_slack=np.inf, T4_n=0, T4_sonly_max_err=0.0,
               T5_max_abs_err=0.0, T5_n=0, T5_endpoint_eps1_err=0.0, T5_endpoint_eps0_err=0.0,
               T5_concavity_violations=0, T5_bounds_violations=0, P6_max_abs_err=0.0, P6_n=0)
    for trial in range(200):
        nc, ns = int(rng.integers(2, 9)), int(rng.integers(2, 5))
        L = rng.dirichlet(np.ones(nc) * rng.uniform(0.3, 2)); KL = rng.dirichlet(np.ones(ns)); KA = rng.dirichlet(np.ones(ns))
        u = float(rng.uniform(0.05, 0.98)); fU = f_K(KL, KA, u)
        # ---- Theorem 4: random finite attack family ----
        Qs = [rng.dirichlet(np.ones(nc) * rng.uniform(0.3, 2)) for _ in range(int(rng.integers(1, 4)))]
        delta = tv_to_hull(L, Qs)
        mm = lp_minimax_finite(L, Qs, KL, KA, u)
        res["T4_min_slack"] = min(res["T4_min_slack"], mm - (fU - delta)); res["T4_n"] += 1
        # s-only LR policy value against each Q equals f_K(u)
        res["T4_sonly_max_err"] = max(res["T4_sonly_max_err"], abs(lp_minimax_finite(np.ones(1), [np.ones(1)], KL, KA, u) - fU))
        # ---- Theorem 5 ----
        A0 = rng.dirichlet(np.ones(nc) * rng.uniform(0.3, 2))
        vals = []
        for eps in [0.0, 0.1, 0.3, 0.6, 1.0]:
            lp = lp_T5(L, A0, KL, KA, u, eps); fo = formula_T5(L, A0, KL, KA, u, eps)
            res["T5_max_abs_err"] = max(res["T5_max_abs_err"], abs(lp - fo)); res["T5_n"] += 1
            vals.append(lp)
            static = lp_minimax_finite(L, [A0], KL, KA, u)
            if lp < eps * fU + (1 - eps) * static - 1e-8 or lp > fU + 1e-8: res["T5_bounds_violations"] += 1
        res["T5_endpoint_eps1_err"] = max(res["T5_endpoint_eps1_err"], abs(vals[-1] - fU))
        res["T5_endpoint_eps0_err"] = max(res["T5_endpoint_eps0_err"], abs(vals[0] - lp_minimax_finite(L, [A0], KL, KA, u)))
        e = np.array([0.0, 0.1, 0.3, 0.6, 1.0]); v = np.array(vals)
        for i in range(1, 4):  # concavity in eps (chord check)
            w = (e[i] - e[i - 1]) / (e[i + 1] - e[i - 1])
            if v[i] < (1 - w) * v[i - 1] + w * v[i + 1] - 1e-8: res["T5_concavity_violations"] += 1
        # ---- Proposition 6: graded responses, exact mimicry, no signal ----
        nr = int(rng.integers(2, 6))
        b = np.r_[0.0, rng.random(nr - 1)]; h = np.r_[0.0, rng.random(nr - 1)]   # level 0 = refuse
        H = float(rng.uniform(0, b.max()))
        # LP over pi(r|c): min E_Q E_pi h, Q = L (mimicry), s.t. E_L E_pi b >= H
        nv = nc * nr
        cobj = np.outer(L, h).ravel()
        Aub = [-np.outer(L, b).ravel()]; bub = [-H]
        Aeq = np.zeros((nc, nv))
        for c in range(nc): Aeq[c, c * nr:(c + 1) * nr] = 1
        lp = linprog(cobj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.ones(nc), bounds=[(0, 1)] * nv, method="highs").fun
        # lower convex envelope of points (b_r, h_r) at H: LP over mixture weights (content-free)
        env = linprog(h, A_ub=[-b], b_ub=[-H], A_eq=[np.ones(nr)], b_eq=[1], bounds=[(0, 1)] * nr, method="highs").fun
        res["P6_max_abs_err"] = max(res["P6_max_abs_err"], abs(lp - env)); res["P6_n"] += 1

    # tightness instance for Theorem 4 (no signal, singleton attack family)
    L1 = np.array([1.0, 0.0]); d = 0.3; Q1 = np.array([1 - d, d])
    tight = lp_minimax_finite(L1, [Q1], np.array([1.0]), np.array([1.0]), 1.0)
    res["T4_tight_example"] = dict(delta=d, minimax=tight, bound=1.0 - d)
    res = {k: (float(v) if not isinstance(v, dict) else v) for k, v in res.items()}
    print(json.dumps(res, indent=1))
    json.dump(res, open("../results/partial_theory_checks.json", "w"), indent=1)
