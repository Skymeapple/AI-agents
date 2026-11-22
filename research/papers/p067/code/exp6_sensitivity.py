"""Exp 6: one-at-a-time sensitivity of closed-form step-model quantities."""
import json, numpy as np
from model import *
from plotstyle import *

P0 = Params()
def q(P):
    if not condition_A(P):
        return None
    sH, sL = s_high(P), s_low(P)
    ec, er = eps_collapse(P), eps_recover(P)
    return dict(s_L_over_s_H=sL / sH, eps_c=ec, eps_r=er, width=er - ec,
                jump=P.dc + ec * P.rho * (sH - sL), outage_gap=P.h1 * (sH - sL),
                valley=[h(sH, P) / (1 - P.rho * sL), ec])
rows = {"baseline": q(P0)}
for name in ["kappa", "rho", "dc", "h1", "h0", "delta"]:
    for f in [0.5, 1.5]:
        v = getattr(P0, name) * f
        kw = {name: v}
        if name == "delta":
            kw["alpha"] = P0.alpha  # keep alpha: s_H changes too
        rows[f"{name}x{f}"] = q(P0.with_(**kw))
ks = np.linspace(0, 0.9, 91)
kap_sweep = [(float(k), q(P0.with_(kappa=k))) for k in ks]
rows["kappa_where_trap_vanishes"] = float(next((k for k, r in kap_sweep if r["width"] <= 1e-9), np.nan))
json.dump(rows, open("../results/exp6_sensitivity.json", "w"), indent=2, default=float)
for k, v in rows.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(ks, [r["eps_c"] for _, r in kap_sweep], color=C1, label="eps_c (collapse)")
ax.plot(ks, [r["eps_r"] for _, r in kap_sweep], color=C2, label="eps_r (recovery)")
ax.fill_between(ks, [r["eps_c"] for _, r in kap_sweep], [r["eps_r"] for _, r in kap_sweep], color=C1, alpha=0.12)
ax.set_xlabel("kappa"); ax.set_ylabel("AI error rate eps"); ax.set_title("Bistable (hysteresis) window vs. kappa")
ax.legend(); plt.tight_layout(); plt.savefig("../results/fig8_sensitivity_kappa.png")
