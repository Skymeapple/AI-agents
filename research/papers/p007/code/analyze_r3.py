"""Round-3 re-analyses of existing result files (no new simulation).

1. Exp. E3: fragility of the crossover slopes - leave-one-out slopes and the slope
2. Exp. F round 2: the oracle-width ReLU ratio at h=64 against the Cor. 2 formula value
   (relative excess, and whether the bootstrap 95% CI contains the formula value).
Writes ../results/analysis_r3.json
"""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(__file__), "..", "results")
c = json.load(open(os.path.join(R, "crossover_results.json")))
out = {"E3_slope_fragility": {}}
for a, v in c.items():
    use = [r for r in v["rows"] if r["r_cross_sim"] and r["rho_u_pred"] < 240]
    assert len(use) == v["n_points_used"]
    x = np.log([r["n"] for r in use]); y = np.log([r["r_cross_sim"] for r in use])
    full = np.polyfit(x, y, 1)[0]; assert abs(full - v["slope_sim"]) < 1e-9
    loo = [float(np.polyfit(np.delete(x, i), np.delete(y, i), 1)[0]) for i in range(len(x))]
    out["E3_slope_fragility"][a] = dict(n_points=len(use), n_values=[r["n"] for r in use], r_cross=[r["r_cross_sim"] for r in use],
                                        slope=float(full), leave_one_out=loo, loo_min=min(loo), loo_max=max(loo),
                                        slope_asymptotic=1 / float(a))
m = json.load(open(os.path.join(R, "mlp2_summary.json")))
out["relu_vs_gamma_formula"] = {k: dict(oracle_at_hmax=v["oracle_at_hmax"], ci95=v["oracle_at_hmax_ci95"], gamma_formula=v["gamma_formula"],
                                        rel_excess=v["oracle_at_hmax"] / v["gamma_formula"] - 1,
                                        ci_contains_formula=bool(v["oracle_at_hmax_ci95"][0] <= v["gamma_formula"] <= v["oracle_at_hmax_ci95"][1]),
                                        in_competition=bool(v["oracle_at_hmax_ci95"][0] > 1))
                                for k, v in m.items() if isinstance(v, dict) and "oracle_at_hmax" in v}
json.dump(out, open(os.path.join(R, "analysis_r3.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
