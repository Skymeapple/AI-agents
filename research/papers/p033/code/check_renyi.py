"""Check that the Thm 3 / Cor 3b bound upper-bounds the exact mixture frontier on a
dense alpha grid (d'=2) and report bound/exact at alpha=1e-4. Run: python3 check_renyi.py"""
import json, os
import numpy as np
from theory import mixture_frontier, renyi_bound_mixture
from common import RESULTS
al = np.logspace(-8, -0.3, 40)
i = np.argmin(abs(al - 1e-4))
out = {}
for rho, t in [(1.0, 0.0), (0.02, 1.0), (0.02, 0.5), (0.1, 0.75), (0.1, 0.0)]:
    comps = [(1 - rho, 0.0), (rho, t * 2.0)] if rho < 1 else [(1.0, 0.0)]
    wd = [(1 - rho, 2.0), (rho, (1 - t) * 2.0)] if rho < 1 else [(1.0, 2.0)]
    ex = mixture_frontier(2.0, comps, al); rb = renyi_bound_mixture(al, wd)
    out[f"rho{rho}_t{t}"] = dict(violations=int(np.sum(ex > rb + 1e-9)), ratio_at_alpha_1em4=float(rb[i] / ex[i]))
json.dump(out, open(os.path.join(RESULTS, "check_renyi.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
