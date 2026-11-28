"""Round 3 (review_r2 m3): direct test of Cor. 3 (head interference) on the fine footing.

phi = 0 (no shared features), ample capacity (r >= M k_p), oracle width, orthogonal-design model.
Cor. 3: ratio = (d + M q - M k_p) / (d + q - k_p).  20 repetitions per cell.
Writes ../results/headint_results.json   (~20 s)
"""
import json, os
import numpy as np
from common import make_task, theory_losses, sim_losses

out = []
for (d, q, kp, M) in [(64, 16, 8, 4), (64, 16, 8, 2), (64, 16, 4, 4), (128, 16, 4, 8), (64, 8, 8, 4)]:
    r = M * kp + 8
    for n in [1000, 10000, 100000]:
        rng = np.random.default_rng(0); rat = []
        for rep in range(20):
            Ws = make_task(d, q, 0, kp, M, rng=rng.integers(1 << 31))
            lu, ls = sim_losses(Ws, 1.0, n, r, rng); rat.append(lu / ls)
        rat = np.array(rat); g = (d + M * q - M * kp) / (d + q - kp)
        out.append(dict(d=d, q=q, kp=kp, M=M, r=r, n=n, sim_ratio=rat.mean(), sim_ratio_se=rat.std(ddof=1) / np.sqrt(len(rat)),
                        cor3=g, df_ratio=float(np.divide(*theory_losses(Ws, 1.0, n, r))), rel_err=abs(rat.mean() - g) / g))
        print(out[-1], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "headint_results.json"), "w"), indent=1, default=float)
