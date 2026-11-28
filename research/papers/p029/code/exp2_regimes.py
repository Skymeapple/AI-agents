"""Experiment 2: liability regimes under noisy attribution (three parties).

All technology parameters here are ASSUMED (illustrative), not calibrated; the
sensitivity block varies them widely.  Outputs: results/exp2_regimes.json, fig1_regimes.png
"""
import json, os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
res = {}
names = ["developer", "deployer", "user"]
n = 3
L = 100.0


def evaluate(C, b, tau, cap=None, with_opt=True, n_starts=10):
    fb = m.social_cost(np.ones(n), L, b, tau).sum()
    out = {"first_best_cost": float(fb)}
    ch = [float(m.dwl_closed_form(m.theta_of(m.rule_channel(n, i), C), L, b, tau).sum()) for i in range(n)]
    out["channel_dwl"] = ch
    out["best_channel"] = names[int(np.argmin(ch))]
    out["best_channel_dwl"] = float(min(ch))
    out["origin_dwl"] = float(m.dwl_closed_form(m.theta_of(m.rule_origin(n), C), L, b, tau).sum())
    out["equal_dwl"] = float(m.dwl_closed_form(m.theta_of(m.rule_equal(n), C), L, b, tau).sum())
    if with_opt:
        S, f = m.optimal_rule(C, L, b, tau, cap=cap, n_starts=n_starts)
        out["optimal_dwl"] = float(f)
        out["optimal_rule"] = np.round(S, 4).tolist()
        out["optimal_theta"] = np.round(m.theta_of(S, C), 4).tolist()
    Sd = m.rule_debiased(C)
    out["debiased_max_payment_over_L"] = float(Sd.max())
    out["debiased_min_payment_over_L"] = float(Sd.min())
    return out


# ---- (a) symmetric baseline, q sweep
b = np.array([0.10, 0.10, 0.10]); tau = np.array([1.0, 1.0, 1.0])
qs = [0.34, 0.40, 0.45, 0.50, 0.5344, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
sweep = []
for q in qs:
    r = evaluate(m.symmetric_C(n, q), b, tau)
    r["q"] = q
    sweep.append(r)
res["a_symmetric_baseline"] = {"L": L, "b": b.tolist(), "tau": tau.tolist(), "sweep": sweep}

# ---- (b) heterogeneous baseline (care-elastic developer, inelastic user), q sweep
b2 = np.array([0.10, 0.10, 0.10]); tau2 = np.array([3.0, 1.0, 0.3])
sweep2 = []
for q in qs:
    r = evaluate(m.symmetric_C(n, q), b2, tau2)
    r["q"] = q
    sweep2.append(r)
res["b_heterogeneous"] = {"L": L, "b": b2.tolist(), "tau": tau2.tolist(), "sweep": sweep2}


# ---- (c) threshold q* at which the origin rule overtakes the best single-party channelling
def qstar_closed(nn, rho):
    """Symmetric parties: n(1/q - 1 + ln q) = (n-1)(rho - 1 - ln rho); valid while q*rho > 1."""
    rhs = (nn - 1) * (rho - 1 - np.log(rho))
    f = lambda q: nn * (1 / q - 1 + np.log(q)) - rhs
    lo = max(1.0 / rho, 1e-9) * (1 + 1e-9)
    if f(lo) < 0:
        return None            # origin rule already better at the corner boundary
    return float(brentq(f, lo, 1.0))


def qstar_numeric(nn, rho):
    bb = np.full(nn, 0.1); tt = np.full(nn, L * 0.1 / rho)
    def gap(q):
        C = m.symmetric_C(nn, q)
        o = m.dwl_closed_form(m.theta_of(m.rule_origin(nn), C), L, bb, tt).sum()
        c = m.dwl_closed_form(m.theta_of(m.rule_channel(nn, 0), C), L, bb, tt).sum()
        return o - c
    grid = np.linspace(1.0 / nn + 1e-6, 1.0, 20001)
    g = np.array([gap(q) for q in grid[::20]])
    idx = np.where(g <= 0)[0]
    return float(grid[::20][idx[0]]) if idx.size else None


thr = []
for nn in [2, 3, 5]:
    for rho in [2, 5, 10, 30, 100]:
        thr.append({"n": nn, "rho": rho, "qstar_closed_form": qstar_closed(nn, rho),
                    "qstar_numeric_grid": qstar_numeric(nn, rho)})
res["c_channelling_threshold"] = thr

# ---- (d) judgment-proof user: cap on the user's share, optimal capped rule
capres = []
for q in [0.5344, 0.80, 1.00]:
    for capU in [1.0, 0.5, 0.2, 0.05, 0.0]:
        C = m.symmetric_C(n, q)
        S, f = m.optimal_rule(C, L, b, tau, cap=[1.0, 1.0, capU], n_starts=10)
        capres.append({"q": q, "user_cap_over_L": capU, "optimal_dwl": float(f),
                       "theta": np.round(m.theta_of(S, C), 4).tolist(),
                       "user_dwl_lower_bound": float(m.dwl_closed_form(np.array([capU]), L, b[2:], tau[2:])[0])})
res["d_user_asset_cap"] = capres

# ---- (e) sensitivity: random technologies and random diagonally dominant channels
rng = np.random.default_rng(7)
draws = 300
rec = []
for t in range(draws):
    rho = np.exp(rng.uniform(np.log(2), np.log(100), n))
    tt = np.exp(rng.uniform(np.log(0.3), np.log(3), n))
    bb = rho * tt / L
    acc = rng.uniform(0.40, 0.95)
    C = acc * np.eye(n) + (1 - acc) * rng.dirichlet(np.ones(n), size=n).T
    r = evaluate(C, bb, tt, n_starts=6)
    S_raw, _ = m.optimal_rule(C, L, bb, tt, n_starts=6)
    Bc = m.budget(C)
    best_simple = min(r["origin_dwl"], r["best_channel_dwl"])
    S = np.array(r["optimal_rule"])
    rec.append({"acc_diag_mean": float(np.mean(np.diag(C))), "budget": Bc,
                "origin_beats_channel": r["origin_dwl"] < r["best_channel_dwl"],
                "optimal_is_pure_channel": bool(np.allclose(S, m.rule_channel(n, int(np.argmax(S.sum(axis=1)))), atol=1e-3)),
                "optimal_is_origin": bool(np.allclose(S, np.eye(n), atol=1e-3)),
                "opt_over_fb": r["optimal_dwl"] / r["first_best_cost"],
                "origin_over_fb": r["origin_dwl"] / r["first_best_cost"],
                "channel_over_fb": r["best_channel_dwl"] / r["first_best_cost"],
                "gain_of_opt_vs_best_simple": (best_simple - r["optimal_dwl"]) / r["first_best_cost"],
                "sum_theta_opt": float(np.sum(m.theta_of(S_raw, C)))})
A = lambda k: np.array([x[k] for x in rec], float)
res["e_sensitivity"] = {
    "draws": draws,
    "share_origin_beats_best_channel": float(A("origin_beats_channel").mean()),
    "share_optimal_is_pure_channel": float(A("optimal_is_pure_channel").mean()),
    "share_optimal_is_origin_rule": float(A("optimal_is_origin").mean()),
    "median_dwl_over_firstbest": {"optimal": float(np.median(A("opt_over_fb"))),
                                  "origin": float(np.median(A("origin_over_fb"))),
                                  "best_channel": float(np.median(A("channel_over_fb")))},
    "median_gain_opt_vs_best_simple_over_fb": float(np.median(A("gain_of_opt_vs_best_simple"))),
    "max_sum_theta_minus_budget": float(np.max(A("sum_theta_opt") - A("budget"))),
    "corr_budget_vs_optimal_dwl": float(np.corrcoef(A("budget"), A("opt_over_fb"))[0, 1]),
}

# ---- (f) break-even forensic cost per incident: attribution-based rule vs channelling
#      origin_dwl + k * incident_rate_origin = channel_dwl + 0   (channelling needs no attribution)
be = []
for rho in [2, 5, 10, 30, 100]:
    for q in [0.40, 0.5344, 0.80]:
        bb = np.full(n, 0.1); tt = np.full(n, L * 0.1 / rho)
        C = m.symmetric_C(n, q)
        th = m.theta_of(m.rule_origin(n), C)
        rate = float(m.hazard(m.care(th, L, bb, tt), bb, tt).sum())
        o = float(m.dwl_closed_form(th, L, bb, tt).sum())
        c = float(m.dwl_closed_form(m.theta_of(m.rule_channel(n, 0), C), L, bb, tt).sum())
        be.append({"rho": rho, "q": q, "origin_dwl": o, "channel_dwl": c,
                   "incident_rate_origin_rule": rate,
                   "breakeven_forensic_cost_over_L": (c - o) / rate / L})
res["f_breakeven_forensic_cost"] = be

with open(os.path.join(OUT, "exp2_regimes.json"), "w") as f:
    json.dump(res, f, indent=1)

# ---- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
for ax, sw, title in [(axes[0], sweep, "symmetric parties (rho = 10)"),
                      (axes[1], sweep2, "heterogeneous (tau = 3, 1, 0.3)")]:
    q = [r["q"] for r in sw]; fb = sw[0]["first_best_cost"]
    ax.plot(q, [r["best_channel_dwl"] / fb for r in sw], "--", label="best single-party channelling")
    ax.plot(q, [r["origin_dwl"] / fb for r in sw], "-o", ms=3, label="attributed party pays all")
    ax.plot(q, [r["equal_dwl"] / fb for r in sw], ":", label="equal shares")
    ax.plot(q, [r["optimal_dwl"] / fb for r in sw], "-", lw=2, label="optimal no-rebate rule")
    ax.axvline(0.5344, color="grey", lw=0.8)
    ax.set_xlabel("attribution accuracy q"); ax.set_title(title, fontsize=10)
    ax.set_ylabel("deadweight loss / first-best cost")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_regimes.png"), dpi=150)

for r in sweep:
    print(r["q"], round(r["best_channel_dwl"], 3), round(r["origin_dwl"], 3), round(r["equal_dwl"], 3),
          round(r["optimal_dwl"], 3), r["optimal_theta"], round(r["debiased_max_payment_over_L"], 2))
print("het")
for r in sweep2:
    print(r["q"], r["best_channel"], round(r["best_channel_dwl"], 3), round(r["origin_dwl"], 3),
          round(r["optimal_dwl"], 3), r["optimal_theta"])
print(json.dumps(res["c_channelling_threshold"]))
print(json.dumps(res["d_user_asset_cap"]))
print(json.dumps(res["e_sensitivity"], indent=1)); print(json.dumps(res["f_breakeven_forensic_cost"]))
