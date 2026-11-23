"""Exp 3: myopic vs forward-looking reliance. The forward-looking agent solves
min E int e^{-r_d t} C(r,s) dt, C = r*a(s) + (1-r)*(h(s)+dc), with skill dynamics,
by value iteration on a grid (semi-Lagrangian, dt = 0.1 month)."""
import json, numpy as np
from model import *
from plotstyle import *

P = Params()
S = np.linspace(0, 1, 401)
R = np.linspace(0, 1, 21)
dt = 0.1


def solve(eps, rd, iters=20000, tol=1e-9):
    disc = np.exp(-rd * dt)
    Sg, Rg = np.meshgrid(S, R, indexing="ij")
    p = (1 - Rg) + P.kappa * Rg
    snext = np.clip(Sg + (P.alpha * p * (1 - Sg) - P.delta * Sg) * dt, 0, 1)
    cost = (Rg * a(Sg, eps, P) + (1 - Rg) * (h(Sg, P) + P.dc)) * dt
    V = np.zeros_like(S)
    for _ in range(iters):
        Q = cost + disc * np.interp(snext, S, V)
        Vn = Q.min(axis=1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    pol = R[Q.argmin(axis=1)]
    return V, pol, cost, snext, disc


def run_policy(policy_fn, eps, rd, s0=0.9, T=600):
    s, J, disc = s0, 0.0, 1.0
    for k in range(int(T / dt)):
        r = policy_fn(s)
        c = (r * a(s, eps, P) + (1 - r) * (h(s, P) + P.dc)) * dt
        J += disc * c
        disc *= np.exp(-rd * dt)
        s = np.clip(s + (P.alpha * practice(r, P) * (1 - s) - P.delta * s) * dt, 0, 1)
    return s, J


eps_list = [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.30, 0.35, 0.40, 0.45, 0.50]
annual = {"5%/yr": 0.05, "30%/yr": 0.30, "200%/yr": 2.0}
out = {"rows": []}
for eps in eps_list:
    s_my, J_my = run_policy(lambda s: float(reliance(s, eps, P)), eps, 0.05 / 12)
    row = dict(eps=eps, myopic_s_inf=float(s_my))
    for name, ra in annual.items():
        rd = np.log(1 + ra) / 12
        V, pol, *_ = solve(eps, rd)
        s_fl, J_fl = run_policy(lambda s: float(np.interp(s, S, pol)), eps, rd)
        _, J_my2 = run_policy(lambda s: float(reliance(s, eps, P)), eps, rd)
        row[f"fl_s_inf[{name}]"] = float(s_fl)
        row[f"myopic_cost_excess_pct[{name}]"] = float(100 * (J_my2 - J_fl) / J_fl)
        row[f"fl_reliance_at_s_inf[{name}]"] = float(np.interp(s_fl, S, pol))
        # starting deskilled (novice / already-reliant worker)
        s_fl0, J_fl0 = run_policy(lambda s: float(np.interp(s, S, pol)), eps, rd, s0=s_low(P))
        s_my0, J_my0 = run_policy(lambda s: float(reliance(s, eps, P)), eps, rd, s0=s_low(P))
        row[f"fl_s_inf_from_sL[{name}]"] = float(s_fl0)
        row[f"myopic_s_inf_from_sL"] = float(s_my0)
        row[f"myopic_cost_excess_from_sL_pct[{name}]"] = float(100 * (J_my0 - J_fl0) / J_fl0)
    out["rows"].append(row)
    print({k: round(v, 3) for k, v in row.items()})
json.dump(out, open("../results/exp3_planner.json", "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
xs = [r["eps"] for r in out["rows"]]
ax[0].plot(xs, [r["myopic_s_inf"] for r in out["rows"]], "o-", color=INK, label="myopic, from s0=0.9")
ax[0].plot(xs, [r["myopic_s_inf_from_sL"] for r in out["rows"]], "s--", color=INK2, label="myopic, from s0=s_L")
for (name, _), col in zip(annual.items(), [C1, C3, C2]):
    ax[0].plot(xs, [r[f"fl_s_inf[{name}]"] for r in out["rows"]], "o-", color=col, ms=4, label=f"forward-looking {name}, s0=0.9")
    ax[1].plot(xs, [r[f"myopic_cost_excess_pct[{name}]"] for r in out["rows"]], "o-", color=col, label=name)
    ax[2].plot(xs, [r[f"myopic_cost_excess_from_sL_pct[{name}]"] for r in out["rows"]], "o-", color=col, label=name)
ax[0].set_xlabel("AI error rate eps"); ax[0].set_ylabel("long-run skill")
ax[0].set_title("(a) Long-run skill"); ax[0].invert_xaxis(); ax[0].legend(fontsize=6.5)
for k, t in [(1, "(b) Excess cost of myopia, s0 = 0.9"), (2, "(c) Excess cost of myopia, s0 = s_L")]:
    ax[k].set_xlabel("AI error rate eps"); ax[k].set_ylabel("excess discounted cost (%)")
    ax[k].set_title(t); ax[k].invert_xaxis(); ax[k].legend(fontsize=8, title="discount")
plt.tight_layout(); plt.savefig("../results/fig4_planner.png")
