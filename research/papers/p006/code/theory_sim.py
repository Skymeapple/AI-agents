"""E1 (numerical, no training).

E1a: Theorem-1 upper bound on the revision-robust horizon vs. the horizon achieved by the
     Proposition-2 head (exact-match + linear recency, logits clipped to [-B, B]).
E1c: delta-rule fast-weight memory with random keys: latest-value accuracy vs M/d and R.

The recency-retention frontier over *general* age-score functions (replacing the old
one-parameter Fig. 1b) is computed separately in frontier_lp.py.

Revision r1: the worst-case weight of the Prop-2 head is computed in closed form
(geometric sums), so the whole script runs in well under a minute; it writes every field
of results/theory_sim.json itself (no post-hoc edits).
Usage: python3 theory_sim.py
"""
import numpy as np, json, os, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)


def L_of(eps):
    return np.log((1 - eps) / eps)


def upper_bound_R(B, T, eps):
    """Theorem 1: R * L_eps <= 2B - log(T-1) with R = floor(A/Delta)+1."""
    return (2 * B - np.log(T - 1)) / L_of(eps)


def worst_weight_latest(B, lam, A, Delta, T):
    """Exact worst case over H(T, A, Delta) of the weight on the latest version for the head
    s = max(B - lam*age, -B) on writes of the queried key and -B elsewhere.
    Worst case: latest at age A, older versions at ages A + j*Delta, j = 1..J (J = all that fit).
    Closed form: unclipped older versions form a geometric series; clipped ones and all
    other positions contribute exp(-B) each."""
    J = (T - 1 - A) // Delta                      # number of older versions that fit
    s_lat = max(B - lam * A, -B)
    if lam > 0:
        # older version j is unclipped iff B - lam*(A + j*Delta) > -B
        jc = int(np.clip(np.ceil((2 * B / lam - A) / Delta) - 1, 0, J))
    else:
        jc = J
    r = np.exp(-lam * Delta)
    if jc > 0:
        geo = r * (1 - r ** jc) / (1 - r) if r < 1 else float(jc)
        old_unclipped = np.exp(B - lam * A - s_lat) * geo   # relative to exp(s_lat)
    else:
        old_unclipped = 0.0
    rest = (T - 1 - jc) * np.exp(-B - s_lat)
    return 1.0 / (1.0 + old_unclipped + rest)


def achievable_A(B, T, eps, Delta, n_lam=400):
    """Largest A such that some lam on a log grid gives worst-case weight >= 1 - eps."""
    best = -1
    for lam in np.geomspace(1e-4, 50, n_lam) / Delta:
        lo, hi = -1, T - 1                         # weight is non-increasing in A
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if worst_weight_latest(B, lam, mid, Delta, T) >= 1 - eps:
                lo = mid
            else:
                hi = mid
        best = max(best, lo)
    return best


def staircase_R(B, T, eps):
    """Prop. 2' (staircase head): largest R with log(T-1) + L_eps + (R-1) log(1+1/rho) <= 2B.
    The head gives logit f = log(U_k) - B on a match in age block k = floor(age/Delta), k < R,
    U_k = (T-1)(1+1/rho)^(R-1-k)/rho, and -B on everything else; hot and cold horizons = R*Delta - 1."""
    rho = eps / (1 - eps)
    x = (2 * B - np.log(T - 1) - L_of(eps)) / np.log(1 + 1 / rho)
    return int(np.floor(x)) + 1 if x >= 0 else 0


def staircase_scores(B, T, eps, Delta):
    rho = eps / (1 - eps); R = staircase_R(B, T, eps)
    u = np.ones(T)
    for k in range(R):
        u[k * Delta:(k + 1) * Delta] = (T - 1) * (1 + 1 / rho) ** (R - 1 - k) / rho
    return u, R


def delta_eval(M, d, R, rng, nV=64, trials=20):
    accs = []
    for _ in range(trials):
        Kk = rng.standard_normal((M, d)); Kk /= np.linalg.norm(Kk, axis=1, keepdims=True)
        Vb = rng.standard_normal((nV, d)); Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
        W = np.zeros((d, d)); latest = np.zeros(M, int)
        for k in rng.permutation(np.repeat(np.arange(M), R)):
            v = rng.integers(nV); latest[k] = v
            W += np.outer(Vb[v] - W @ Kk[k], Kk[k])
        pred = np.argmax((Kk @ W.T) @ Vb.T, axis=1)
        accs.append(np.mean(pred == latest))
    return float(np.mean(accs)), float(np.std(accs) / np.sqrt(trials))


if __name__ == "__main__":
    t0 = time.time()
    res = {"E1": [], "E1c_delta": []}
    eps = 0.05
    for T in [10**3, 10**5, 10**7]:
        for B in [4, 6, 8, 10, 12, 16, 20]:
            Delta = 16
            Rub = upper_bound_R(B, T, eps)
            A_ach = achievable_A(B, T, eps, Delta)
            Rst = staircase_R(B, T, eps)
            st_verified = None
            if T <= 10**5 and Rst >= 1:   # exact worst-case check of the staircase head (DP over ages)
                from frontier_lp import horizons_of_scores
                u, _ = staircase_scores(B, T, eps, Delta)
                assert u.max() <= np.exp(2 * B) * (1 + 1e-12)
                Ah, C = horizons_of_scores(u, T, Delta, eps / (1 - eps))
                st_verified = bool(Ah >= Rst * Delta - 1 and C >= Rst * Delta - 1)
            row = dict(T=T, B=B, Delta=Delta, eps=eps, R_ub=float(Rub),
                       # Thm 1: floor(A/Delta)+1 <= R_ub  =>  A/Delta < floor(R_ub)
                       A_ub_over_Delta_strict=float(np.floor(Rub)) if Rub >= 1 else -1.0,
                       A_ach=int(A_ach),
                       A_ach_over_Delta=float(A_ach / Delta) if A_ach >= 0 else -1.0,
                       staircase_R=Rst, staircase_A_over_Delta=float(Rst - 1 / Delta) if Rst >= 1 else -1.0,
                       staircase_verified_numerically=st_verified)
            res["E1"].append(row); print(row, flush=True)
    rng = np.random.default_rng(1)
    d = 64
    for M in [8, 16, 32, 48, 64, 96, 128, 256]:
        for R in [1, 8, 32]:
            a, se = delta_eval(M, d, R, rng, trials=20)
            res["E1c_delta"].append(dict(M=M, d=d, R=R, acc=a, se=se, trials=20))
            print("delta", M, R, round(a, 3), flush=True)
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "theory_sim.json"), "w"), indent=1)
    print("saved results/theory_sim.json in %.1fs" % res["runtime_s"])
