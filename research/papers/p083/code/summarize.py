"""Headline statistics quoted in the paper -> results/headline.json"""
import json, os
import numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
R = pd.read_csv(os.path.join(RES, "shifting_summary.csv"))
A = pd.read_csv(os.path.join(RES, "accounting_bias.csv"))
M = pd.read_csv(os.path.join(RES, "mc_summary.csv"))
D = pd.read_csv(os.path.join(RES, "mc_draws.csv"))
r = R[(R.ci == "ci_prod") & (R.phi == 0.3)]
h = {}
h["n_configs_prop2"] = int(len(R))
h["prop2_corr"] = float(np.corrcoef(R.pred_dW_pct, R.real_dW_pct)[0, 1])
h["prop2_sign_agree"] = float((np.sign(R.pred_dW_pct) == np.sign(R.real_dW_pct)).mean())
h["prop2_median_abs_err_pp"] = float((R.pred_dW_pct - R.real_dW_pct).abs().median())
h["boundary_sign_flip"] = {a: [s for s, x, y in zip(q.site, q.dWon_rel_carbon_only, q.dW_carbon_only) if np.sign(x) != np.sign(y)]
                           for a, q in r.groupby("arch")}
h["total_water_penalty_sites"] = {a: q[q.dW_carbon_only > 0].site.tolist() for a, q in r.groupby("arch")}
h["onsite_penalty_sites"] = {a: q[q.dWon_rel_carbon_only > 0].site.tolist() for a, q in r.groupby("arch")}
pen = r[r.dW_carbon_only > 0]
h["knee_retention"] = {f"{s}|{a}": float(k / c) for s, a, k, c in zip(pen.site, pen.arch, pen.knee_dC, pen.dC_carbon_only)}
h["dC_carbon_only_range"] = [float(r.dC_carbon_only.min()), float(r.dC_carbon_only.max())]
h["acc_bias_Won_adiab_range"] = [float(A[A.arch == "adiab"].bias_Won_pct.min()), float(A[A.arch == "adiab"].bias_Won_pct.max())]
h["acc_bias_Won_tower_range"] = [float(A[A.arch == "tower"].bias_Won_pct.min()), float(A[A.arch == "tower"].bias_Won_pct.max())]
h["acc_bias_C_range"] = [float(A.bias_C_pct.min()), float(A.bias_C_pct.max())]
h["acc_bias_W_range"] = [float(A.bias_W_pct.min()), float(A.bias_W_pct.max())]
h["mc_P_dW_pos_overall"] = float((D.dW > 0).mean())
h["mc_P_dWon_pos_overall"] = float((D.dWon_rel > 0).mean())
h["mc_retained_pen_draws_median"] = float(D[D.dW > 0].retained.median())
h["mc_retained_pen_draws_p05"] = float(D[D.dW > 0].retained.quantile(0.05))
h["mc_retained_pen_draws_min"] = float(D[D.dW > 0].retained.min())
h["mc_by_window_P_pos"] = D.groupby("window").dW.apply(lambda v: float(np.mean(v > 0))).to_dict()
h["mc_by_ci_P_pos"] = D.groupby("ci").dW.apply(lambda v: float(np.mean(v > 0))).to_dict()
h["mc_by_hydro_P_pos"] = {str(k): float(v) for k, v in D.groupby("hydro").dW.apply(lambda v: np.mean(v > 0)).items()}
# Spearman of within-day rho vs dW sign
h["corr_rhoCW_vs_dW_per_dC"] = float(np.corrcoef(r.rho_CW, r.dW_carbon_only / r.dC_carbon_only)[0, 1])
json.dump(h, open(os.path.join(RES, "headline.json"), "w"), indent=1)
print(json.dumps(h, indent=1))

# Corollary 3 bound check: carbon cost of water neutrality <= P * (C_w - C_c) / (W_c - W_w)
F = pd.read_csv(os.path.join(RES, "frontiers.csv"))
rows = []
for (s, a), f in F.groupby(["site", "arch"]):
    f = f.sort_values("lam")
    Cc, Wc = f.dC.iloc[-1], f.dW.iloc[-1]
    Cw, Ww = f.dC.iloc[0], f.dW.iloc[0]
    if Wc <= 0:
        continue
    # exact neutral point by linear interpolation along the frontier
    g = f.iloc[::-1].reset_index(drop=True)
    k = int(np.argmax(g.dW.values <= 0))
    c0, w0, c1, w1 = g.dC[k - 1], g.dW[k - 1], g.dC[k], g.dW[k]
    c_neutral = c0 + (c1 - c0) * (w0 / (w0 - w1))
    bound = Wc * (Cw - Cc) / (Wc - Ww)
    rows.append(dict(site=s, arch=a, P=Wc, cost_actual=c_neutral - Cc, cost_bound=bound,
                     retained_actual=c_neutral / Cc, retained_lower_bound=1 - bound / abs(Cc)))
B = pd.DataFrame(rows)
B.to_csv(os.path.join(RES, "corollary3_bound.csv"), index=False)
print(B.round(3).to_string())
