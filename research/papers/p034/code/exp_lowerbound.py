"""Exp 5: minimax lower bound for proxy-only stopping rules (Theorem 3) and the
offline-audit indistinguishability bound. Pure closed-form; runs in seconds.
Run: python3 exp_lowerbound.py
"""
import json, os
import numpy as np
from model import Params, closed_form

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
LAM_MAX = 2.0
grid = np.linspace(0, LAM_MAX, 20001)
res = {}
for q in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    P = Params(q=q)
    cf = closed_form(P, grid)
    gW, gT = cf["gold"], cf["gold_twin"]
    RW, RT = gW.max() - gW, gT.max() - gT
    # any proxy-only rule stops at the same (random) lambda in both worlds, so
    # max_world E[regret] >= (1/2) inf_lambda [RW + RT]   (and >= inf_lambda max(RW,RT))
    i = int(np.argmin(np.maximum(RW, RT)))
    j = int(np.argmin(RW + RT))
    res[f"q={q}"] = dict(lam_star_W=float(grid[np.argmax(gW)]), lam_star_twin=float(grid[np.argmax(gT)]),
                         gold_star_W=float(gW.max()), gold_star_twin=float(gT.max()),
                         minimax_regret_deterministic=float(np.maximum(RW, RT)[i]), lam_minimax=float(grid[i]),
                         half_inf_sum_regret=float(0.5 * (RW + RT)[j]))
# offline audit on base policy pi0: probability of observing >=1 shared exploit among m labelled samples
P = Params()
res["offline_TV_upper_bound"] = {str(m): float(1 - (1 - P.p * 1.0) ** m) for m in [100, 1000, 10000]}
json.dump(res, open(os.path.join(OUT, "lowerbound_results.json"), "w"), indent=1)
for k, v in res.items():
    print(k, v)
