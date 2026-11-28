"""Illustrative dispatch: Abilene 2012, no-firm mix, CFE 100%, v = 500 $/MWh (2-h steps).
Plots the week containing the largest curtailment event."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model
from experiments import cf, NOFIRM
RES = os.path.join(HERE, "..", "results")
cs, cw = cf("abilene_tx", 2012); P = model.params()
o = model.solve(cs, cw, P, phi=1.0, v=500, allow=NOFIRM, dt=2, return_series=True)
s = o["series"]; dt = 2
csa = cs.reshape(-1, dt).mean(1); cwa = cw.reshape(-1, dt).mean(1)
pv = o["caps"]["pv"] * csa; wd = o["caps"]["wind"] * cwa
k = int(np.argmax(np.convolve(s["u"], np.ones(84), "same"))); a = max(0, k - 42); b = a + 84
t = np.arange(a, b) * dt / 24.0
fig, axs = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax = axs[0]
ax.stackplot(t, np.minimum(wd[a:b], 1e9), pv[a:b], s["d"][a:b], colors=["#2a78d6", "#eda100", "#1baf7a"],
             labels=["wind", "solar", "battery discharge"], alpha=0.85, edgecolor="#fcfcfb", linewidth=0.5)
ax.plot(t, 1 - s["u"][a:b], color="#0b0b0b", lw=1.5, label="served compute load")
ax.axhline(1, color="#52514e", lw=0.8, ls=":")
ax.set_ylim(0, 3.2); ax.set_ylabel("MW per MW of load"); ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper right")
ax.set_title("Abilene 2012, no firm, CFE 100%, v = $500/MWh: week with largest curtailment", fontsize=9)
axs[1].plot(t, s["e"][a:b] / max(o["caps"]["bat_e"], 1e-9), color="#1baf7a", lw=1.5, label="battery state of charge (fraction)")
axs[1].bar(t, s["u"][a:b], width=dt / 24 * 0.9, color="#e34948", label="curtailed load (MW)")
axs[1].set_xlabel("Day of year 2012"); axs[1].legend(fontsize=7, frameon=False); axs[1].set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig6_dispatch_week.png"))
json.dump({k2: (float(v2) if not isinstance(v2, dict) else v2) for k2, v2 in o.items() if k2 != "series"},
          open(os.path.join(RES, "fig6_case.json"), "w"), indent=1, default=float)
print({k2: v2 for k2, v2 in o.items() if k2 != "series"})
