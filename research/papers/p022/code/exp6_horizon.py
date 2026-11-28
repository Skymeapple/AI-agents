"""E6: horizon multiplier implied by Thm 1/2 (analytic, small-p limit and exact).
If an unverified agent has per-step error prob p, its 50%-correctness horizon is
h0 = ln2/-ln(1-p).  With verification the horizon is h = ln2/-ln(theta_m).
We report h/h0 (exact, p=1e-3) and the small-p value 1/mu_m."""
import json, numpy as np, theory as th
p = 1e-3
h0 = np.log(2) / -np.log(1 - p)
rows = []
for r in [0.5, 0.8, 0.95]:
    for rho in [0.0, 0.05, 0.2]:
        for m in [1, 3, 10]:
            mu = th.mu_point(r, m, rho)
            h = np.log(2) / -np.log(th.theta(p, mu))
            rows.append(dict(r=r, rho=rho, m=m, mu=float(mu), mult_exact=float(h / h0),
                             mult_smallp=float(1 / mu), ceiling=float(1 / rho) if rho > 0 else None))
for x in rows:
    print(x)
json.dump(rows, open("../results/e6_horizon_multiplier.json", "w"), indent=1)
