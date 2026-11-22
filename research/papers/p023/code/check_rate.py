"""Check Proposition 2: C - A_n ~ f'(1/2)/(8n) for odd n (beta density f)."""
import json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from theory import vote_acc, ceiling, beta_params
out = {}
for p, rho in [(0.65, 0.05), (0.65, 0.1), (0.65, 0.2), (0.7, 0.1), (0.55, 0.1)]:
    a, b = beta_params(p, rho)
    h = 1e-6
    fprime = (stats.beta.pdf(0.5 + h, a, b) - stats.beta.pdf(0.5 - h, a, b)) / (2 * h)
    pred = fprime / 8
    rows = {}
    for n in [101, 401, 1601, 6401]:
        rows[n] = n * (ceiling(p, rho) - vote_acc(n, p, rho))
    out[f"p{p}_rho{rho}"] = {"fprime_half_over_8": pred, "n_times_gap": rows}
    print(p, rho, round(pred, 5), {k: round(v, 5) for k, v in rows.items()})
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "rate_check.json"), "w"), indent=1)
