"""Exp 5 (revised): exact value of the proxy-only stopping game and its dependence on the twin.

Any proxy-only rule induces the SAME law F of the stopping point lambda_tau in W and W'
(Theorem 3a). Its worst-case expected regret is max(E_F R_W, E_F R_W'). The best achievable
value over ALL proxy-only rules is therefore at least
    V = min_F max(E_F R_W, E_F R_W')  =  max_{a in [0,1]} min_lambda [a R_W + (1-a) R_W']
(minimax theorem on a finite grid), with
    1/2 inf (R_W + R_W')  <=  V  <=  min_lambda max(R_W, R_W').
We compute all three, for the 'proxy-consistent' twin (shared exploits have gold c + g0 with
c = mu, i.e. exactly what the agreeing proxies say) and for a sweep of the twin constant c.
Run: python3 exp_lowerbound_v2.py   (seconds)
"""
import json, os
import numpy as np
from scipy.optimize import linprog
from model import Params, closed_form
from rev_common import gold_curve_twin

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
LAM_MAX = 2.0
grid = np.linspace(0, LAM_MAX, 4001)
alphas = np.linspace(0, 1, 2001)


def game(gW, gT):
    RW, RT = gW.max() - gW, gT.max() - gT
    det = float(np.min(np.maximum(RW, RT)))
    half = float(0.5 * np.min(RW + RT))
    vals = [np.min(a * RW + (1 - a) * RT) for a in alphas]
    ia = int(np.argmax(vals))
    # primal LP over stopping distributions F on the grid: min t s.t. F.RW <= t, F.RT <= t
    n = len(grid)
    cvec = np.r_[np.zeros(n), 1.0]
    A = np.vstack([np.r_[RW, -1.0], np.r_[RT, -1.0]])
    lp = linprog(cvec, A_ub=A, b_ub=[0, 0], A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[1.0],
                 bounds=[(0, None)] * n + [(None, None)], method="highs")
    F = lp.x[:n]; supp = [(float(grid[i]), float(F[i])) for i in np.where(F > 1e-6)[0]]
    return dict(V_randomised=float(lp.fun), V_dual_check=float(vals[ia]), half_inf_sum=half,
                deterministic_minmax=det, optimal_mixture=supp, peak_gain_W=float(gW.max() - gW[0]))


res = {"proxy_consistent_twin_c=mu": {}, "twin_constant_sweep_q=1.0": {}, "twin_constant_sweep_q=0.5": {}}
for q in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    P = Params(q=q)
    res["proxy_consistent_twin_c=mu"][f"q={q}"] = game(closed_form(P, grid)["gold"], gold_curve_twin(P, grid))
for q in [1.0, 0.5]:
    P = Params(q=q)
    for c in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]:
        res[f"twin_constant_sweep_q={q}"][f"c={c}"] = game(closed_form(P, grid)["gold"], gold_curve_twin(P, grid, c_twin=c))
json.dump(res, open(os.path.join(OUT, "lowerbound_v2_results.json"), "w"), indent=1)
for k, v in res.items():
    print(k)
    for kk, vv in v.items():
        print("  ", kk, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in vv.items()})
