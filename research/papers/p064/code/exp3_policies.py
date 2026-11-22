"""Rebuilding pathways compared in the calibrated software-developer occupation.
Shock at t=0: extensive hiring cut x=0.19 and intensive learning loss d=0.17 for AI-era cohorts.
Policies start at t_p and act only on cohorts hired from t_p onward."""
import json
import numpy as np
from common import *
from model import *

X, Dd = 0.19, 0.17
T = 80.0
mu, g, tau, p = SD_MU, SD_G, TAU, P
h0 = required_h(g, tau, p, mu, mu)
base = simulate(T=T, g=g, mu_S=mu, mu_J=mu, tau=tau, p=p)
XE = x_eff(X, Dd)


def run(policy, tp, catch=0.25, mbar_mult=1.3):
    def hire(t, st):
        h = h0 * (1 - X)
        if t >= tp and policy in ("A", "A+B", "A+B+catch", "A+B+C", "A+B+D"):
            h = h0
        if t >= tp and policy == "A+B+catch":
            # temporary over-hiring until the locked-in deficit (Prop. 3) is repaid
            T_rep = XE * tp / catch if tp > 0 else 0.0
            if t < tp + T_rep:
                h = h0 * (1 + catch)
        return h

    def prule(t):
        if t >= tp and policy in ("B", "A+B", "A+B+catch", "A+B+C", "A+B+D"):
            return p
        return p * (1 - Dd)

    def taurule(t):
        if t >= tp and policy in ("C", "A+B+C"):
            return 0.7 * tau
        return tau

    def murule(t):
        if t >= tp and policy in ("D", "A+B+D"):
            return 0.8 * mu
        return mu

    mbar = mbar_mult * h0 if policy == "A+B+catch" else None
    return simulate(T=T, g=g, mu_S=mu, mu_J=mu, tau=tau, p=p, hire_rule=hire, p_rule=prule,
                    tau_rule=taurule, m_bar=mbar, mu_S_rule=murule)


POL = ["none", "A", "B", "C", "D", "A+B", "A+B+C", "A+B+D", "A+B+catch"]
POL_FIG = ["none", "A", "B", "C", "D", "A+B", "A+B+C", "A+B+D"]
LAB = {"none": "no policy", "A": "A: protected entry slots (h)", "B": "B: guarded AI training (p)",
       "C": "C: accelerated training (tau x0.7)", "D": "D: senior retention (mu_S x0.8)", "A+B": "A+B", "A+B+C": "A+B+C", "A+B+D": "A+B+D",
       "A+B+catch": "A+B + 25% catch-up hiring (mentor cap)"}
t_det = detection_time(0.05, XE, tau, mu, g)
starts = [0.0, 2.0, 5.0, round(t_det, 1)]
res = {}
for tp in starts:
    for pol in POL:
        o = run(pol, tp)
        G = o["G"] - base["G"]
        t = o["t"]
        res[f"{pol}|{tp}"] = dict(policy=pol, tp=tp,
                                  integrated_60y=float(np.sum(np.clip(G[t < 60], 0, None)) * (1 / 12)),
                                  surplus_60y=float(np.sum(np.clip(-G[t < 60], 0, None)) * (1 / 12)),
                                  peak=float(G.max()), G_at_40=float(G[np.searchsorted(t, 40)]),
                                  t_peak=float(t[G.argmax()]))
json.dump(dict(h0=h0, x_eff=XE, t_detect=t_det, results=res), open(os.path.join(RES, "exp3_policies.json"), "w"), indent=1)

print(f"x_eff={XE:.4f}  detection time={t_det:.2f}y  h0={h0:.4f}")
print(f"{'policy':12s} {'tp':>5s} {'int60':>7s} {'surp':>6s} {'peak%':>7s} {'G40%':>7s} {'tpeak':>6s}")
for k, v in res.items():
    print(f"{v['policy']:12s} {v['tp']:5.1f} {v['integrated_60y']:7.3f} {v['surplus_60y']:6.2f} {100*v['peak']:7.2f} {100*v['G_at_40']:7.2f} {v['t_peak']:6.1f}")

# Figure 3: trajectories at tp = 5, and integrated deficit vs start time
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for j, pol in enumerate(POL_FIG):
    o = run(pol, 5.0)
    ax[0].plot(o["t"], 100 * (o["G"] - base["G"]), color=C[j], lw=1.6, label=LAB[pol])
ax[0].axvline(5, color=INK2, lw=0.8, ls="--"); ax[0].axhline(0, color=INK2, lw=0.6)
ax[0].text(5.3, 30, "policy starts", fontsize=7, color=INK2)
ax[0].set_xlim(0, 60); ax[0].set_xlabel("years since intake shock"); ax[0].set_ylabel("senior shortfall, % of demand")
ax[0].set_title("A. Shortfall paths, policy start at year 5", fontsize=9, loc="left")
ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
width = 0.8 / len(starts)
for k, tp in enumerate(starts):
    vals = [res[f"{pol}|{tp}"]["integrated_60y"] for pol in POL]
    ax[1].bar(np.arange(len(POL)) + (k - 1.5) * width, vals, width * 0.92, color=C[k],
              label=f"start year {tp:g}" + (" (=5% detection)" if tp == starts[-1] else ""))
ax[1].set_xticks(np.arange(len(POL))); ax[1].set_xticklabels(["none", "A", "B", "C", "D", "A+B", "A+B\n+C", "A+B\n+D", "A+B\n+catch"], fontsize=8)
ax[1].set_ylabel("integrated shortfall (positive part),\nyears 0-60 (share-years)")
ax[1].set_title("B. Cost of each pathway by start time", fontsize=9, loc="left")
ax[1].legend(fontsize=7); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig3_rebuilding_pathways.png")
