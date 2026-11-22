"""Three-mode closed form: contrast of a math-like and a judge-like verifier."""
import json, os
from theory import three_mode_peak
R = os.path.join(os.path.dirname(__file__), "..", "results")
out = {}
for name, (a, b) in {"math-like": (0.8, 0.1), "judge-like": (0.6, 0.4)}.items():
    for h0 in [1e-2, 1e-3, 1e-4]:
        t, A, k = three_mode_peak(0.2, 0.8 - h0, h0, a, b, 0.95)
        out[f"{name} h0={h0:g}"] = {"alpha": a, "beta": b, "eta": 0.95, "c0": 0.2, "t_star": t, "A_star": A, "kappa": k}
json.dump(out, open(os.path.join(R, "exp1b.json"), "w"), indent=2)
for k, v in out.items(): print(k, round(v["t_star"], 2), round(v["A_star"], 4), round(v["kappa"], 3))
