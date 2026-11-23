"""E2b (revision): Part A finite-group bias.  With centred advantages r_i - mean(r) the group
mean includes r_i itself, so the expected update is (G-1)/G times the population gradient and
the KL-regularised fixed point is the tilt at t = (G-1)/(G*lam), not 1/lam (review W5).
Recomputes the Part A predictions with and without the correction against the stored GRPO
results (results/exp2.json).  For std-normalised advantages the same factor is applied to the
self-consistent t_eff as an approximation (the ddof=0 group std adds a further small bias)."""
import json, os
import numpy as np
from theory import tilted_accuracy
from exp2_tabular_grpo import make_u20

R = os.path.join(os.path.dirname(__file__), "..", "results")
G = 16; c = (G - 1) / G
w, y, f = make_u20(0.02)
d = json.load(open(os.path.join(R, "exp2.json")))["partA_kl_regularised"]


def tstd(lam, scale):
    t = scale / lam
    for _ in range(300):
        lw = np.log(w) + t * f; p = np.exp(lw - lw.max()); p /= p.sum(); fb = p @ f
        t = 0.5 * t + 0.5 * scale / (lam * np.sqrt(fb * (1 - fb)))
    return t


rows = []
for r in d:
    lam = r["lam"]
    row = {"lam": lam}
    for key, tf in [("centred", lambda s: s / lam), ("std", lambda s: tstd(lam, s))]:
        A = r[key]["A_grpo_mean"]
        p1 = float(tilted_accuracy(f, y, w, tf(1.0))[0]); pc = float(tilted_accuracy(f, y, w, tf(c))[0])
        row[key] = {"A_grpo": A, "A_pred_uncorrected": p1, "A_pred_corrected": pc,
                    "err_uncorrected": A - p1, "err_corrected": A - pc}
    rows.append(row); print(row)
json.dump({"G": G, "factor": c, "rows": rows}, open(os.path.join(R, "exp2b.json"), "w"), indent=1)
