"""Reactive (shortage-triggered) versus anticipatory (pipeline-targeting) intake rules.
Reactive top-up: u = kappa*h0*D*G(t - delta). Linearisation: y' = -lam y - kappa*lam*y(t - tau - delta).
Hayes (1950): stable for every delay iff kappa <= 1, otherwise iff tau+delta < arccos(-1/kappa)/(lam sqrt(kappa^2-1))."""
import json
import numpy as np
from common import *
from model import *

mu, g, p = SD_MU, SD_G, P
lam = mu + g
dt = 1 / 12


def reactive(kappa, tau, x=0.19, delta=1.0, tp=0.0, T=120.0):
    h0 = required_h(g, tau, p, mu, mu)
    lagk = int(round(delta / dt))
    hist = []

    def hire(t, st):
        G = 1 - st["S"] / st["D"]
        hist.append(G)
        Gobs = hist[-1 - lagk] if len(hist) > lagk else 0.0
        u = kappa * h0 * Gobs if t >= tp else 0.0
        return max(0.0, h0 * (1 - x) + u)

    return simulate(T=T, g=g, mu_S=mu, mu_J=mu, tau=tau, p=p, hire_rule=hire)


def anticipatory(tau, x=0.19, tp=0.0, T=120.0, cap=3.0):
    """Choose entrants so that the projected senior stock one training-duration ahead equals
    projected demand, using only the observable pipeline (cohorts already in training)."""
    h0 = required_h(g, tau, p, mu, mu)
    k = int(round(tau / dt))
    l = np.exp(-mu * tau)
    dec = np.exp(-mu * dt)

    def hire(t, st):
        if t < tp:
            return h0 * (1 - x)
        i, gr, S, D = st["i"], st["grads"], st["S"], st["D"]
        js = np.arange(i, i + k)
        proj = S * dec ** k + np.sum(gr[js] * np.exp(-mu * dt / 2) * dec ** (i + k - 1 - js))
        proj = proj * dec + gr[i + k] * np.exp(-mu * dt / 2)
        target = D * np.exp(g * (k + 1) * dt)
        B = (target - proj) / (np.exp(-mu * dt / 2) * dt * p * l)
        return ("abs", float(np.clip(B, 0.0, cap * h0 * D)))

    return simulate(T=T, g=g, mu_S=mu, mu_J=mu, tau=tau, p=p, hire_rule=hire)


out = {}
TAU = 5.0
base = simulate(T=120, g=g, mu_S=mu, mu_J=mu, tau=TAU, p=p)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
series = []
for j, kappa in enumerate([1.0, 4.0, 10.0]):
    o = reactive(kappa, TAU, delta=1.0, tp=0.0)
    G = o["G"] - base["G"]
    ax[0].plot(o["t"], 100 * G, color=C[j], lw=1.6, label=f"reactive, kappa={kappa:g}")
    late = np.abs(G[o["t"] > 80]).max(); mid = np.abs(G[(o["t"] > 40) & (o["t"] < 80)]).max()
    series.append(dict(rule="reactive", kappa=kappa, max_abs_G=float(np.abs(G).max()), amp_40_80=float(mid), amp_80_120=float(late),
                       tau_c=float(hayes_critical_delay(lam, kappa * lam))))
o = anticipatory(TAU, tp=0.0)
G = o["G"] - base["G"]
ax[0].plot(o["t"], 100 * G, color=C[3], lw=2.2, label="anticipatory (pipeline-targeting)")
series.append(dict(rule="anticipatory", max_abs_G=float(np.abs(G).max()), amp_80_120=float(np.abs(G[o["t"] > 80]).max())))
oa5 = anticipatory(TAU, tp=5.0); Ga5 = oa5["G"] - base["G"]
or5 = reactive(4.0, TAU, tp=5.0); Gr5 = or5["G"] - base["G"]
series.append(dict(rule="anticipatory_start5", peak=float(Ga5.max()), t_back_below_1pct=float(oa5["t"][(oa5["t"] > 10) & (np.abs(Ga5) < 0.01)][0]),
                   integrated_pos=float(np.clip(Ga5, 0, None).sum() * dt), integrated_neg=float(np.clip(-Ga5, 0, None).sum() * dt)))
series.append(dict(rule="reactive4_start5", peak=float(Gr5.max()), integrated_pos=float(np.clip(Gr5, 0, None).sum() * dt),
                   integrated_neg=float(np.clip(-Gr5, 0, None).sum() * dt)))
ax[0].axhline(0, color=INK2, lw=0.6)
ax[0].set_xlabel("years since intake shock (x=0.19, rules active from year 0)"); ax[0].set_ylabel("senior shortfall, % of demand")
ax[0].set_title("A. Shortage-reactive vs pipeline-targeting intake", fontsize=9, loc="left")
ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

# stability map: simulate small shock without reporting lag, classify growth of oscillation
kap = np.array([0.5, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
taus = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12])
grid = []
for kk in kap:
    for tt in taus:
        o = reactive(kk, float(tt), x=0.01, delta=0.0, T=400.0)
        b = simulate(T=400, g=g, mu_S=mu, mu_J=mu, tau=float(tt), p=p)
        G = o["G"] - b["G"]; t = o["t"]
        # oscillation amplitude (peak-to-peak, offset-free) in two late windows
        a1 = np.ptp(G[(t > 200) & (t < 300)]); a2 = np.ptp(G[t > 300])
        stable_sim = bool(a2 < a1 * 0.999 or a2 < 1e-6)
        tc = hayes_critical_delay(lam, kk * lam)
        grid.append(dict(kappa=float(kk), tau=float(tt), stable_sim=stable_sim, stable_theory=bool(tt < tc)))
agree = np.mean([gk["stable_sim"] == gk["stable_theory"] for gk in grid])
kk_line = np.linspace(1.01, 26, 300)
ax[1].plot(kk_line, [hayes_critical_delay(lam, k_ * lam) for k_ in kk_line], color=INK, lw=1.2, label="Hayes boundary tau_c(kappa)")
for gk in grid:
    ax[1].plot(gk["kappa"], gk["tau"], "o" if gk["stable_sim"] else "x", color=C[2] if gk["stable_sim"] else C[7], ms=5)
ax[1].plot([], [], "o", color=C[2], label="sim: converges"); ax[1].plot([], [], "x", color=C[7], label="sim: growing cycles")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 13); ax[1].set_xlabel("reactivity kappa (% extra intake per 1% shortfall)")
ax[1].set_ylabel("feedback delay tau (years)")
ax[1].set_title(f"B. Stability map (lam={lam:.4f}); sim/theory agree {100*agree:.0f}%", fontsize=9, loc="left")
ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); save(fig, "fig4_feedback_stability.png")
out["series"] = series; out["grid"] = grid; out["agreement"] = float(agree); out["lam"] = lam
json.dump(out, open(os.path.join(RES, "exp4_feedback.json"), "w"), indent=1)
print(json.dumps(series, indent=1)); print("agreement", agree)
