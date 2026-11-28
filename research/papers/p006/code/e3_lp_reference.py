"""Reference for E3: the best worst-case weight achievable by ANY age-score head (logit f(age) in [-B, B]
on writes of the queried key, -B elsewhere) that must simultaneously
  * select never-revised facts at every age 0..N-1 (the training distribution asks for this), and
  * select the latest version of every revised fact with spacing >= Delta and latest age <= A - Delta,
with weight >= 1 - eps.  We bisect on eps with the exact LP of frontier_lp.py.
Also reported: the same without the all-ages cold requirement (cold only at age A, i.e. Theorem 1's chain
family), which is what the Theorem-1 ceiling refers to.  Writes results/e3_lp_reference.json.
"""
import json, os, numpy as np
from frontier_lp import feasible

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def best_weight(N, B, Delta, A, cold_all):
    lo, hi = 1e-6, 0.5          # eps: hi always "feasible enough"? we search the smallest feasible eps
    A_h = A - Delta
    def ok(eps):
        C = N - 1 if cold_all else A
        if A_h < 0:
            return _cold_only(N, B, eps, C)
        return feasible(N, B, Delta, eps, A_h, C)[0]
    if not ok(0.5 - 1e-9): return 0.5
    for _ in range(30):
        mid = (lo + hi) / 2
        if ok(mid): hi = mid
        else: lo = mid
    return 1 - hi


def _cold_only(N, B, eps, C):
    rho = eps / (1 - eps)
    return rho * np.exp(2 * B) >= N - 1          # u = e^{2B} at every age


res = []
for B in [4, 6]:
    for A in [0, 16, 48, 96, 192]:
        r = dict(N=256, B=B, Delta=16, A=A,
                 best_m_cold_all_ages=best_weight(256, B, 16, A, True),
                 best_m_chain_family=best_weight(256, B, 16, A, False))
        res.append(r); print(r, flush=True)
json.dump(res, open(os.path.join(OUT, "e3_lp_reference.json"), "w"), indent=1)
