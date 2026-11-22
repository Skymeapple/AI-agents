"""Numerical sanity checks of Theorems 1-3 on random finite instances (exact LPs with scipy).
T1: ASR >= U - TV for every policy.  T2: LP optimum == Neyman-Pearson (likelihood-ratio) frontier.
T3: with attacker-chosen content Q covering the legit content marginal, min_pi max_Q ASR at utility u equals
    the NP frontier of the unforgeable channel alone (K_L vs K_A)."""
import json
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)


def np_frontier(L, A, u):
    """min E_A phi s.t. E_L phi >= u, by likelihood-ratio ordering (fractional knapsack)."""
    order = np.argsort(-(L / np.maximum(A, 1e-300)))
    acc_u = acc_a = 0.0
    for i in order:
        if acc_u >= u - 1e-15: break
        take = min(1.0, (u - acc_u) / L[i]) if L[i] > 0 else 1.0
        acc_u += take * L[i]; acc_a += take * A[i]
    return acc_a


def lp_frontier(L, A, u):
    r = linprog(A, A_ub=[-L], b_ub=[-u], bounds=[(0, 1)] * len(L), method="highs")
    return r.fun


res = {"T1_min_slack": np.inf, "T2_max_abs_err": 0.0, "T3_max_abs_err": 0.0, "n_instances": 0}
for trial in range(300):
    n = rng.integers(2, 30)
    L = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3)); A = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3))
    tv = 0.5 * np.abs(L - A).sum()
    for _ in range(20):  # random policies
        phi = rng.random(n)
        res["T1_min_slack"] = min(res["T1_min_slack"], A @ phi - (L @ phi - tv))
    for u in np.linspace(0, 0.999, 11):
        res["T2_max_abs_err"] = max(res["T2_max_abs_err"], abs(lp_frontier(L, A, u) - np_frontier(L, A, u)))
    # T3: content c in C (nc values), unforgeable signal s in S (ns values)
    nc, ns = rng.integers(2, 6), rng.integers(2, 5)
    Lc = rng.dirichlet(np.ones(nc)); KL = rng.dirichlet(np.ones(ns)); KA = rng.dirichlet(np.ones(ns))
    u = rng.uniform(0.05, 0.95)
    # attacker's best response set: point masses on content + the legit marginal; defender LP over phi(c,s):
    # min t s.t. sum_s KA[s] phi(c,s) <= t for all c (attacker may put all mass on any c; covers mixtures),
    #            sum_{c,s} Lc[c] KL[s] phi(c,s) >= u
    nv = nc * ns
    cobj = np.zeros(nv + 1); cobj[-1] = 1
    Aub, bub = [], []
    for c in range(nc):
        row = np.zeros(nv + 1); row[c * ns:(c + 1) * ns] = KA; row[-1] = -1; Aub.append(row); bub.append(0)
    row = np.zeros(nv + 1); row[:nv] = -np.outer(Lc, KL).ravel(); Aub.append(row); bub.append(-u)
    r = linprog(cobj, A_ub=Aub, b_ub=bub, bounds=[(0, 1)] * nv + [(0, 1)], method="highs")
    minimax = r.fun
    res["T3_max_abs_err"] = max(res["T3_max_abs_err"], abs(minimax - np_frontier(KL, KA, u)))
    res["n_instances"] += 1
res = {k: float(v) for k, v in res.items()}
print(json.dumps(res, indent=1))
json.dump(res, open("../results/theory_checks.json", "w"), indent=1)
