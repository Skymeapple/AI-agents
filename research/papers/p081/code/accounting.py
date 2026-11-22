"""Growth accounting of the 2020-2025 frontier-compute trend and the headroom ('OOM budget') calculation.
Outputs results/accounting.json and results/fig1_accounting.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import sample, FIXED

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(1)
N = 200_000
p = sample(N, rng)

gC, gK, gh, gY, ge, gT = p["gC0"], p["gK0"], p["gh"], p["gY"], p["ge"], p["gT"]
comp = {
    "world GDP (secular)": gY,
    "hardware FLOP/$ (secular)": gh,
    "spending share of GDP (bounded)": gK - gY,
    "precision/utilisation residual (bounded)": gC - gK - gh,
}
q = lambda a: [float(np.percentile(a, x)) for x in (5, 50, 95)]
res = {"total_log_growth": q(gC), "components_log": {k: q(v) for k, v in comp.items()},
       "components_share": {k: q(v / gC) for k, v in comp.items()}}
sec_share = (gY + gh) / gC
res["secular_share"] = q(sec_share)
res["bounded_share"] = q(1 - sec_share)
res["long_run_financial_growth_factor"] = q(np.exp(gY + gh))
# consistency check: implied growth of power drawn by largest run = gC - ge - gT  (duration growth lowers power)
gP = gC - ge - gT
res["implied_power_growth_factor"] = q(np.exp(gP))
res["implied_power_growth_after_duration_cap"] = q(np.exp(gC - ge))

# OOM budget: remaining orders of magnitude above the secular trend from bounded factors
phi0 = FIXED["K0_usd"] / FIXED["I0_usd"]
sig0 = FIXED["I0_usd"] / FIXED["Y0_usd"]
B = np.log10(p["phi_max"] / phi0) + np.log10(p["sigma_max"] / sig0) + np.log10(p["x_max"])
excess = (gC - gY - gh) / np.log(10)        # OOM / yr above secular trend today
res["phi0"] = phi0; res["sigma0"] = sig0
res["OOM_budget_financial"] = q(B)
res["excess_OOM_per_year"] = q(excess)
res["naive_runway_years"] = q(B / excess)
json.dump(res, open(os.path.join(OUT, "accounting.json"), "w"), indent=2)
print(json.dumps(res, indent=2))

# figure
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
labels = list(comp.keys())
med = [np.median(comp[k]) / np.log(10) for k in labels]
lo = [np.percentile(comp[k], 5) / np.log(10) for k in labels]
hi = [np.percentile(comp[k], 95) / np.log(10) for k in labels]
cols = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
y = np.arange(len(labels))
ax[0].barh(y, med, color=cols, xerr=[np.subtract(med, lo), np.subtract(hi, med)], capsize=3)
ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8); ax[0].invert_yaxis()
ax[0].set_xlabel("contribution to growth (OOM / year)")
ax[0].set_title("Decomposition of 4-5x/yr (blue: secular, orange: bounded)", fontsize=9)
ax[1].hist(B / excess, bins=80, range=(0, 20), color="#DD8452", alpha=.8)
ax[1].set_xlabel("naive runway (years) = OOM budget / excess growth")
ax[1].set_ylabel("samples"); ax[1].set_title("Years of above-trend growth left (financial channel)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_accounting.png"), dpi=150)
