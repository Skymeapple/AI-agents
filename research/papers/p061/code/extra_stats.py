"""Extra descriptive statistics quoted in the paper -> results/extra_stats.json"""
import os, pickle, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from lib import R, rule_table, ces_ratio
st = pickle.load(open(os.path.join(R, "occ_stats.pkl"), "rb"))
E = pd.read_csv(os.path.join(R, "emp_national.csv"), dtype={"soc": str}).set_index("soc")
E = E[E.has_tasks & E.emp2024.notna()]
out = {}
for meas in ["hum_beta", "gpt_beta"]:
    S = st[(meas, "im")].reindex(E.index)
    rho = spearmanr(S["s"], S["prod"]).statistic
    out[f"{meas}_spearman_linear_vs_oring"] = rho
    out[f"{meas}_spearman_linear_vs_labelled_a"] = spearmanr(S["s"], S["a"]).statistic
    top = S["s"].sort_values(ascending=False).head(20).index
    topo = S["prod"].sort_values(ascending=False).head(20).index
    out[f"{meas}_top20_overlap_linear_oring"] = len(set(top) & set(topo))
    out[f"{meas}_share_occ_prod_zero"] = float((S["prod"] == 0).mean())
    out[f"{meas}_emp_weighted_mean_s"] = float(np.average(S["s"], weights=E.emp2024))
    out[f"{meas}_emp_weighted_mean_a_share_of_s"] = float(np.average(S["a"], weights=E.emp2024) / np.average(S["s"], weights=E.emp2024))
    T = pd.DataFrame({"title": E.OCC_TITLE, "emp": E.emp2024, "s": S["s"], "oring": S["prod"], "bott": S["bott"]})
    T.sort_values("s", ascending=False).head(15).round(3).to_csv(os.path.join(R, f"top_occ_linear_{meas}.csv"))
    T.sort_values("oring", ascending=False).head(15).round(3).to_csv(os.path.join(R, f"top_occ_oring_{meas}.csv"))
# knife-edge check: pure automation with sigma == eps reproduces linear rule exactly
s = np.linspace(0, .9, 10)
out["knife_edge_max_abs_err"] = float(max(abs((1 - ces_ratio(s, 0 * s, sg, sg, th)) - s).max()
                                       for sg in [0.5, 1.0, 2.0] for th in [0.3, 0.9]))
out["total_emp_2024_matched"] = float(E.emp2024.sum())
json.dump(out, open(os.path.join(R, "extra_stats.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
print(pd.read_csv(os.path.join(R, "top_occ_linear_hum_beta.csv")).to_string())
print(pd.read_csv(os.path.join(R, "top_occ_oring_hum_beta.csv")).to_string())
