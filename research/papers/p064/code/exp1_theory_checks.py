"""Numerical verification of Propositions 1-3 against the cohort simulator."""
import json
import numpy as np
from common import *
from model import *

out = {}

# --- Prop 2: shortfall path after a permanent intake shock
errs = {}
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
for j, xe in enumerate([0.13, 0.19, x_eff(0.19, 0.17)]):
    o = simulate(T=40, g=SD_G, mu_S=SD_MU, mu_J=SD_MU, tau=TAU, hire_rule=lambda t, st, xe=xe: st["h0"] * (1 - xe))
    b = simulate(T=40, g=SD_G, mu_S=SD_MU, mu_J=SD_MU, tau=TAU)
    th = shortfall_path(o["t"], xe, TAU, SD_MU, SD_G)
    errs[f"{xe:.4f}"] = float(np.abs((o["G"] - b["G"]) - th).max())
    ax[0].plot(o["t"], 100 * (o["G"] - b["G"]), color=C[j], label=f"x_eff={xe:.2f}")
    ax[0].plot(o["t"], 100 * th, color=INK, lw=0.8, ls=":")
out["prop2_max_abs_error"] = errs
ax[0].axvline(TAU, color=INK2, lw=0.8, ls="--")
ax[0].set_xlabel("years since intake shock"); ax[0].set_ylabel("senior shortfall G(t), % of demand")
ax[0].set_title("A. Shortfall: simulation vs closed form (dotted)", fontsize=9, loc="left")
ax[0].legend(fontsize=7)

# --- Prop 3: conservation of senior-years
xe = 0.19
b = simulate(T=200, g=SD_G, mu_S=SD_MU, mu_J=SD_MU, tau=TAU)
cons = []
for tp in [1, 2, 5, 10, 15, 20]:
    for tau in [3.0, 5.0, 8.0]:
        bb = simulate(T=200, g=SD_G, mu_S=SD_MU, mu_J=SD_MU, tau=tau)
        o = simulate(T=200, g=SD_G, mu_S=SD_MU, mu_J=SD_MU, tau=tau,
                     hire_rule=lambda t, st, tp=tp: st["h0"] * (1 - xe) if t < tp else st["h0"])
        cons.append(dict(tp=tp, tau=tau, sim=excess_deficit(o, bb), theory=xe * tp))
out["prop3_conservation"] = cons
tps = sorted(set(c["tp"] for c in cons))
for k, tau in enumerate([3.0, 5.0, 8.0]):
    ax[1].plot(tps, [c["sim"] for c in cons if c["tau"] == tau], ["o", "s", "^"][k], ms=[9, 6, 4][k], color=C[k], label=f"sim, tau={tau:g}")
ax[1].plot(tps, [xe * t for t in tps], color=INK, lw=0.8, label="x_eff * T_gap")
ax[1].set_xlabel("duration of hiring gap T_gap (years)"); ax[1].set_ylabel("integrated shortfall (share-years)")
ax[1].set_title("B. Conservation of senior-years", fontsize=9, loc="left"); ax[1].legend(fontsize=7)

# --- Prop 1: Lotka growth rate in the mentor-capacity regime
rows = []
for tau in [3.0, 5.0, 8.0]:
    for R in [0.6, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0]:
        mu = SD_MU
        mbar = R * mu / (P * np.exp(-mu * tau))
        o = simulate(T=120, g=0, mu_S=mu, mu_J=mu, tau=tau, hire_rule=lambda t, st: ("abs", 1e12), m_bar=mbar)
        t = o["t"]; S = o["S"]; i0 = np.searchsorted(t, 80); i1 = np.searchsorted(t, 110)
        r_sim = np.log(S[i1] / S[i0]) / (t[i1] - t[i0])
        r_th = lotka_root(R, mu, tau)
        lo, hi = lotka_bounds(R, mu, tau) if R > 1 else (np.nan, np.nan)
        rows.append(dict(R=R, tau=tau, r_sim=float(r_sim), r_theory=float(r_th), lower=float(lo), upper=float(hi)))
out["prop1_lotka"] = rows
for k, tau in enumerate([3.0, 5.0, 8.0]):
    rr = [r for r in rows if r["tau"] == tau]
    ax[2].plot([r["R"] for r in rr], [r["r_theory"] for r in rr], color=C[k], label=f"root, tau={tau:g}")
    ax[2].plot([r["R"] for r in rr], [r["r_sim"] for r in rr], "o", ms=4, color=C[k])
    up = [r for r in rr if r["R"] > 1]
    ax[2].plot([r["R"] for r in up], [r["upper"] for r in up], color=C[k], lw=0.8, ls="--")
ax[2].axhline(0, color=INK2, lw=0.8); ax[2].axvline(1, color=INK2, lw=0.8, ls=":")
ax[2].set_xscale("log"); ax[2].set_xticks([0.6, 1, 2, 3, 5]); ax[2].set_xticklabels(["0.6", "1", "2", "3", "5"]); ax[2].minorticks_off(); ax[2].set_xlabel("expertise reproduction number R"); ax[2].set_ylabel("growth rate r (1/yr)")
ax[2].set_title("C. Renewal growth rate (dots: sim; dashed: upper bound)", fontsize=9, loc="left"); ax[2].legend(fontsize=7)
fig.tight_layout()
save(fig, "fig1_theory_checks.png")
json.dump(out, open(os.path.join(RES, "exp1_theory_checks.json"), "w"), indent=1)
print(json.dumps(out["prop2_max_abs_error"], indent=1))
print("max |cons err|", max(abs(c["sim"] - c["theory"]) for c in cons))
print("max |lotka err|", max(abs(r["r_sim"] - r["r_theory"]) for r in rows))
print("bounds hold", all((r["lower"] - 1e-9 <= r["r_theory"] <= r["upper"] + 1e-9) for r in rows if r["R"] > 1))
