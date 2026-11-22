"""Probabilistic predictions from the exogenous variant (main specification). Outputs results/predictions.json"""
import json, os
import numpy as np
from params import sample
from model import simulate, summarize
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(11)
N = 20000
p = sample(N, rng)
sim = simulate(p); s = summarize(sim)
k = 12; tt = s["tt"]
def g_at(yr): return np.exp(s["g12"][:, np.argmin(np.abs(tt - yr))])
res = {
 "P(C2028>1e28)": float(np.mean(s["log10C_2028"] > 28)),
 "P(C2030>1e29)": float(np.mean(s["log10C_2030"] > 29)),
 "P(C2030>1e30)": float(np.mean(s["log10C_2030"] > 30)),
 "P(C2035>1e30)": float(np.mean(s["log10C_2035"] > 30)),
 "P(C2035>1e31)": float(np.mean(s["log10C_2035"] > 31)),
 "P(slow2<2030.5)": float(np.mean(s["slow2"] < 2030.5)),
 "P(slow2<2032.5)": float(np.mean(s["slow2"] < 2032.5)),
 "P(growth in yr to mid-2028 >= 3x)": float(np.mean(g_at(2028.5) >= 3)),
 "growth_to_mid2028_pct": [float(np.percentile(g_at(2028.5), x)) for x in (5, 50, 95)],
 "growth_to_mid2030_pct": [float(np.percentile(g_at(2030.5), x)) for x in (5, 50, 95)],
}
# implied power of largest run (GW) = C / (e * T)  -- use the model's power channel inputs
T_days = np.minimum(90 * np.exp(p["gT"][:, None] * (sim["t"][None, :] - 2025.5)), p["T_max_days"][:, None])
e = (10 ** p["log10_C0"] / (310e9 * 3600))[:, None] * np.exp(p["ge"][:, None] * (sim["t"][None, :] - 2025.5))
x = 1.0  # residual x does not enter the power channel
P_GW = np.exp(sim["logC"]) / (e * x * T_days * 86400) / 1e9
for yr in (2028, 2030, 2035):
    j = np.argmin(np.abs(sim["t"] - (yr + .5)))
    res[f"largest_run_power_GW_{yr}"] = [float(np.percentile(P_GW[:, j], q)) for q in (5, 50, 95)]
gP = np.log(P_GW[:, 12:] / P_GW[:, :-12])
for yr in (2026.5, 2027.5, 2028.5, 2029.5):
    j = np.argmin(np.abs(sim["t"][12:] - yr))
    res[f"power_growth_factor_to_{yr}"] = [float(np.exp(np.percentile(gP[:, j], q))) for q in (5, 50, 95)]
json.dump(res, open(os.path.join(OUT, "predictions.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
