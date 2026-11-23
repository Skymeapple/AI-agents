"""E1b (revision r1): exact recency-retention frontier over ALL age-dependent score functions.

Replaces the old Fig. 1b, which swept only a one-parameter linear-recency family and used the
alpha >= 0.5 criterion (for which L_eps = 0 and Theorem 1 is vacuous).

Head family: logit f(a) in [-B, B] on a write of the queried key at age a (f arbitrary, not
necessarily monotone), logit -B on every other position (optimal for non-matching positions).
In units u_a = exp(f(a) + B) in [1, exp(2B)], the faithfulness condition alpha_latest >= 1-eps
on a history is LINEAR in u:
     rho * u_lat >= (T-1) + sum_{older versions}(u - 1),       rho = eps/(1-eps).
Requirements (deterministic, worst case):
  * hot horizon A_h : every history whose latest version has age a <= A_h and whose versions
    are >= Delta apart (any number of versions) is retrieved faithfully.  The worst case over
    older-version placements is a max-weight Delta-spaced selection, which we encode exactly
    with dynamic-programming variables V_b >= max(V_{b+1}, u_b - 1 + V_{b+Delta}).
  * cold horizon C : a never-revised fact (single write) at any age c <= C is retrieved
    faithfully:  rho * u_c >= T - 1.
For each A_h we binary-search the largest feasible C (LP feasibility, HiGHS). Each LP solution
is re-checked exactly (worst-case DP evaluated in float64 on the returned scores).
We also evaluate, under the same criteria, the linear-recency family (Prop. 2) and two-level
step heads (the reviewer's counter-example family), and record the Theorem-1 limit.
Usage: python3 frontier_lp.py
"""
import numpy as np, json, os, time
import scipy.sparse as sp
from scipy.optimize import linprog

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def worst_case_ok(u, T, Delta, rho, A_h, C, tol=1e-7):
    """Exact check of all hot (latest age <= A_h) and cold (age <= C) constraints for scores u."""
    V = np.zeros(T + Delta + 1)
    for b in range(T - 1, -1, -1):
        V[b] = max(V[b + 1], u[b] - 1 + V[b + Delta])
    hot = all(rho * u[a] >= (T - 1 + V[a + Delta]) * (1 - tol) for a in range(A_h + 1))
    cold = all(rho * u[c] >= (T - 1) * (1 - tol) for c in range(C + 1))
    return hot and cold


def feasible(T, B, Delta, eps, A_h, C):
    rho = eps / (1 - eps)
    nb = T + Delta                     # V_b for b = 0..T+Delta-1, V_b = 0 for b >= T (bounds)
    nv = T + nb
    iu = lambda a: a
    iV = lambda b: T + b
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for b in range(T):                 # V_b >= V_{b+1}   ->  V_{b+1} - V_b <= 0
        rows += [r, r]; cols += [iV(b + 1), iV(b)]; vals += [1, -1]; rhs.append(0); r += 1
        # V_b >= u_b - 1 + V_{b+Delta}  ->  u_b + V_{b+D} - V_b <= 1
        rows += [r, r, r]; cols += [iu(b), iV(b + Delta), iV(b)]; vals += [1, 1, -1]; rhs.append(1); r += 1
    for a in range(A_h + 1):           # V_{a+D} - rho u_a <= -(T-1)
        rows += [r, r]; cols += [iV(a + Delta), iu(a)]; vals += [1, -rho]; rhs.append(-(T - 1)); r += 1
    for c in range(C + 1):             # -rho u_c <= -(T-1)
        rows += [r]; cols += [iu(c)]; vals += [-rho]; rhs.append(-(T - 1)); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    bounds = [(1.0, np.exp(2 * B))] * T + [(0, None)] * T + [(0, 0)] * Delta
    res = linprog(np.zeros(nv), A_ub=A, b_ub=np.array(rhs, float), bounds=bounds, method="highs")
    if res.status != 0:
        return False, None
    u = np.clip(res.x[:T], 1.0, np.exp(2 * B))
    return worst_case_ok(u, T, Delta, rho, A_h, C, tol=1e-6), u


def max_cold(T, B, Delta, eps, A_h):
    ok, _ = feasible(T, B, Delta, eps, A_h, A_h)
    if not ok:
        return -1
    lo, hi = A_h, T          # lo feasible, hi infeasible (C = T-1 may be feasible)
    ok, _ = feasible(T, B, Delta, eps, A_h, T - 1)
    if ok:
        return T - 1
    hi = T - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _ = feasible(T, B, Delta, eps, A_h, mid)
        if ok: lo = mid
        else: hi = mid
    return lo


def horizons_of_scores(u, T, Delta, rho):
    """Largest hot horizon A_h and cold horizon C (contiguous from age 0) of a given score u."""
    V = np.zeros(T + Delta + 1)
    for b in range(T - 1, -1, -1):
        V[b] = max(V[b + 1], u[b] - 1 + V[b + Delta])
    A_h = -1
    for a in range(T):
        if rho * u[a] >= T - 1 + V[a + Delta]: A_h = a
        else: break
    C = -1
    for c in range(T):
        if rho * u[c] >= T - 1: C = c
        else: break
    return A_h, C


if __name__ == "__main__":
    t0 = time.time()
    T, Delta, eps = 4096, 32, 0.05
    rho = eps / (1 - eps); L = np.log(1 / rho)
    out = {"T": T, "Delta": Delta, "eps": eps, "frontiers": []}
    for B in [6, 7, 8, 9, 10]:
        R_ub = (2 * B - np.log(T - 1)) / L
        A_thm = int(Delta * np.floor(R_ub)) - 1 if R_ub >= 1 else -1   # Thm 1: A_h <= Delta*floor(R_ub) - 1
        pts = []
        for A_h in [0, 16, 31, 32, 48, 63, 64, 80, 95, 96, 112, 127, 128]:
            C = max_cold(T, B, Delta, eps, A_h)
            if C < 0:
                pts.append(dict(A_h=A_h, C=-1)); print("B", B, "A_h", A_h, "infeasible", flush=True); break
            pts.append(dict(A_h=A_h, C=int(C))); print("B", B, "A_h", A_h, "C", C, flush=True)
        # linear-recency family (Prop. 2) and two-level step heads, same criteria
        lin = []
        for lam in np.geomspace(1e-4, 2.0, 120):
            f = np.maximum(B - lam * np.arange(T), -B); u = np.exp(f + B)
            Ah, C = horizons_of_scores(u, T, Delta, rho); lin.append(dict(lam=float(lam), A_h=int(Ah), C=int(C)))
        step = []
        for w in range(1, 129, 4):
            for lowv in np.linspace(-B, B, 41):
                f = np.where(np.arange(T) < w, B, lowv); u = np.exp(f + B)
                Ah, C = horizons_of_scores(u, T, Delta, rho); step.append(dict(w=w, low=float(lowv), A_h=int(Ah), C=int(C)))
        out["frontiers"].append(dict(B=B, R_ub=float(R_ub), A_h_thm1_max=A_thm, lp=pts, linear=lin, step=step))
    # the reviewer's exact counter-example criterion (eps -> 1/2) for B = 10: frontier is not binding
    rho_half = 1.0
    f = np.where(np.arange(T) < 24, 10.0, -1.6); u = np.exp(f + 10.0)
    out["reviewer_step_head_eps_half"] = dict(zip(["A_h", "C"], map(int, horizons_of_scores(u, T, Delta, rho_half))))
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(OUT, "frontier_lp.json"), "w"), indent=1)
    print("saved results/frontier_lp.json in %.0fs" % out["runtime_s"])
