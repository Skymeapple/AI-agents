"""Numerical checks of Propositions 1-3 and cap-exhaustion timing.
Outputs results/theory_check.json and fig7_frontloading.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import sample, PRIORS, FIXED
from model import simulate

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
res = {}

# --- Prop. 2 (front-loading / invariance): same caps, different growth speeds of bounded factors ---
med = {k: np.array([np.median(sample(20001, np.random.default_rng(3))[k])]) for k in PRIORS}
med["log10_Clat"] = np.array([60.0])          # remove latency cap to isolate financial/power
med["P2030_GW"] = np.array([1e3])             # remove power cap
paths = {}
for gI0 in (0.35, 0.50, 0.70):
    p = {k: v.copy() for k, v in med.items()}
    p["gI0"] = np.array([gI0]); p["gK0"] = np.array([gI0 + 0.35])   # keep phi growth fixed, vary sigma speed
    sim = simulate(p)
    paths[gI0] = sim
t = paths[0.35]["t"]
j2040 = np.argmin(np.abs(t - 2040.5))
res["prop2_log10C_2040_by_gI0"] = {str(g): float(paths[g]["logC"][0, j2040] / np.log(10)) for g in paths}
res["prop2_log10C_2028_by_gI0"] = {str(g): float(paths[g]["logC"][0, np.argmin(np.abs(t - 2028.5))] / np.log(10)) for g in paths}
# analytic: after all caps bind, log C = log(phi_max sigma_max Y(t) q0 x_max) + gh t  -- independent of speeds
p = med
anal = (np.log10(p["phi_max"][0]) + np.log10(p["sigma_max"][0]) + np.log10(FIXED["Y0_usd"]) + p["gY"][0] * 15 / np.log(10)
        + np.log10(10 ** p["log10_C0"][0] / FIXED["K0_usd"]) + p["gh"][0] * 15 / np.log(10) + np.log10(p["x_max"][0]))
res["prop2_analytic_log10C_2040"] = float(anal)

fig, ax = plt.subplots(figsize=(6, 3.6))
for g, ls in zip(paths, ("-", "--", ":")):
    ax.plot(t, paths[g]["logC"][0] / np.log(10), ls, label=f"investment growth {np.exp(g):.2f}x/yr")
ax.axhline(anal, color="grey", lw=.8)
ax.set_xlabel("year"); ax.set_ylabel("log10 training FLOP"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Prop. 2: speed of bounded factors shifts timing, not the eventual level", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig7_frontloading.png"), dpi=150)

# --- cap exhaustion times for each bounded factor (from priors) ---
rng = np.random.default_rng(5)
s = sample(200_000, rng)
phi0 = FIXED["K0_usd"] / FIXED["I0_usd"]; sig0 = FIXED["I0_usd"] / FIXED["Y0_usd"]
gphi = np.clip(s["gK0"] - s["gI0"], 1e-9, None)
gx = np.clip(s["gC0"] - s["gK0"] - s["gh"], 1e-9, None)
tau = {
    "sigma (investment/GDP)": np.log(s["sigma_max"] / sig0) / (s["gI0"] - s["gY"]),
    "phi (largest run/investment)": np.log(s["phi_max"] / phi0) / gphi,
    "x (precision/utilisation)": np.log(s["x_max"]) / gx,
    "T (run duration)": np.log(s["T_max_days"] / FIXED["T0_days"]) / s["gT"],
}
q = lambda a: [float(np.percentile(np.minimum(a, 100), x)) for x in (5, 50, 95)]
res["cap_exhaustion_years_after_2025.5"] = {k: q(v) for k, v in tau.items()}
first = np.argmin(np.stack([tau[k] for k in list(tau)[:3]]), axis=0)
res["first_financial_cap_to_bind"] = {k: float(np.mean(first == i)) for i, k in enumerate(list(tau)[:3])}
# power: year the power channel would bind if financial channel were unconstrained at 4.5x/yr
json.dump(res, open(os.path.join(OUT, "theory_check.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
