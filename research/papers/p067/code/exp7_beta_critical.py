"""Exp 7: critical reliance sharpness beta_c below which no eps yields bistability (smooth model)."""
import json, numpy as np
from model import *

def bistable_any(P, grid=np.linspace(0.05, 0.7, 261)):
    return any(sum(st for _, st in equilibria(e, P, n=4001)) >= 2 for e in grid)

def beta_c(P, lo=1.0, hi=2000.0):
    if not bistable_any(P.with_(beta=hi)):
        return None
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if bistable_any(P.with_(beta=mid)):
            hi = mid
        else:
            lo = mid
    return hi

P = Params()
out = {}
for k in [0.0, 0.15, 0.3, 0.5, 0.7]:
    out[f"kappa={k}"] = beta_c(P.with_(kappa=k))
    print(k, out[f"kappa={k}"])
json.dump(out, open("../results/exp7_beta_critical.json", "w"), indent=2)
