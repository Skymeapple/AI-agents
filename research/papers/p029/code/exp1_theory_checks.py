"""Experiment 1: numerical checks of Theorems 1-2 and of the rare-event approximation.

(a) Budget bound: LP maximum of sum_i theta_i equals B(C) on random channels.
(b) Blackwell monotonicity: garbling a channel never increases any weighted LP value.
(c) Debiased rule: S = C^{-1} gives M = I and ex-post budget balance.
(d) Exact (non-rare) pipeline: Nash care levels vs the closed form x_i = tau ln(theta L b / tau).
(e) Monte Carlo of payments under origin rule and debiased rule with unequal hazards.
"""
import json, os
import numpy as np
from scipy.optimize import minimize_scalar
import model as m

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(20261128)
res = {}

# ---- (a) and (b)
gaps, viol = [], 0
n_trials = 2000
for t in range(n_trials):
    n = int(rng.integers(2, 7))
    C = rng.dirichlet(np.ones(n) * rng.uniform(0.3, 3), size=n).T   # columns sum to 1
    val, _ = m.max_weighted_theta(C, np.ones(n))
    gaps.append(abs(val - m.budget(C)))
    G = rng.dirichlet(np.ones(n), size=n).T                          # garbling, column-stochastic
    C2 = G @ C
    w = rng.uniform(0, 1, n)
    v1, _ = m.max_weighted_theta(C, w)
    v2, _ = m.max_weighted_theta(C2, w)
    if v2 > v1 + 1e-9:
        viol += 1
res["a_budget_lp_max_abs_gap"] = float(np.max(gaps))
res["a_n_trials"] = n_trials
res["b_blackwell_violations"] = viol

# ---- (c)
errs, bal, minentry = [], [], []
for t in range(2000):
    n = int(rng.integers(2, 7))
    q = rng.uniform(1.0 / n + 0.05, 1.0)
    C = m.symmetric_C(n, q) if t % 2 == 0 else None
    if C is None:
        C = 0.6 * np.eye(n) + 0.4 * rng.dirichlet(np.ones(n), size=n).T
    S = m.rule_debiased(C)
    errs.append(np.abs(S @ C - np.eye(n)).max())
    bal.append(np.abs(S.sum(axis=0) - 1).max())
    if t % 2 == 0:
        r = (1 - q) / (n - 1)
        errs.append(abs(S[0, 0] - (1 - r) / (q - r)))
        errs.append(abs(S[1, 0] + r / (q - r)))
res["c_debias_max_err_M_minus_I"] = float(np.max(errs))
res["c_debias_max_budget_imbalance"] = float(np.max(bal))

# ---- (d) exact pipeline model, n = 3
L = 100.0
b = np.array([0.10, 0.10, 0.10])
tau = np.array([1.0, 1.0, 1.0])
n = 3


def origin_probs(x):
    """Exact: each layer fails independently with prob h_i; origin = first failing layer."""
    h = m.hazard(x, b, tau)
    p = np.zeros(n); surv = 1.0
    for j in range(n):
        p[j] = surv * h[j]; surv *= 1 - h[j]
    return p


def nash(S, C, iters=200):
    M = S @ C
    x = np.ones(n)
    for _ in range(iters):
        x_old = x.copy()
        for i in range(n):
            def cost(xi):
                xx = x.copy(); xx[i] = xi
                return xi + L * float(M[i] @ origin_probs(xx))
            x[i] = minimize_scalar(cost, bounds=(0, 30), method="bounded",
                                   options={"xatol": 1e-10}).x
        if np.abs(x - x_old).max() < 1e-9:
            break
    return x


rows = []
for q in [0.4, 0.5344, 0.7, 0.9, 1.0]:
    C = m.symmetric_C(n, q)
    S = m.rule_origin(n)
    x_exact = nash(S, C)
    x_cf = m.care(m.theta_of(S, C), L, b, tau)
    sc_exact = x_exact.sum() + L * origin_probs(x_exact).sum()
    sc_cf = m.social_cost(m.theta_of(S, C), L, b, tau).sum()
    rows.append({"q": q, "x_exact": x_exact.tolist(), "x_closed_form": x_cf.tolist(),
                 "max_rel_dev_care": float(np.max(np.abs(x_exact - x_cf) / x_cf)),
                 "social_cost_exact": float(sc_exact), "social_cost_closed_form": float(sc_cf),
                 "rel_dev_social_cost": float(abs(sc_exact - sc_cf) / sc_cf)})
res["d_exact_vs_rare_event"] = rows

# ---- (e) Monte Carlo payments, unequal hazards, q = 0.5344 (published all-at-once accuracy)
q = 0.5344
C = m.symmetric_C(n, q)
h = np.array([0.02, 0.01, 0.005])      # deliberately unequal hazards (assumed, for the check)
N = 4_000_000
u = rng.random((N, n))
fails = u < h
any_fail = fails.any(axis=1)
origin = np.argmax(fails, axis=1)[any_fail]
k = origin.size
sig = np.empty(k, dtype=int)
for j in range(n):
    idx = np.where(origin == j)[0]
    sig[idx] = rng.choice(n, size=idx.size, p=C[:, j])
mc = {"N_tasks": N, "n_incidents": int(k), "q": q, "hazards_assumed": h.tolist()}
for name, S in [("origin_rule", m.rule_origin(n)), ("debiased_rule", m.rule_debiased(C))]:
    pay = L * S[:, sig]                       # shape n x k
    per_task = pay.sum(axis=1) / N
    true_share = np.array([(origin == i).sum() for i in range(n)]) * L / N
    mc[name] = {
        "rule_matrix": S.tolist(),
        "mean_payment_per_task": per_task.tolist(),
        "own_origin_loss_per_task": true_share.tolist(),
        "max_abs_expost_imbalance": float(np.abs(pay.sum(axis=0) - L).max()),
        "std_payment_per_incident_party0": float(pay[0].std()),
        "max_payment_over_L": float(pay.max() / L),
        "min_payment_over_L": float(pay.min() / L),
    }
res["e_monte_carlo"] = mc

with open(os.path.join(OUT, "exp1_theory_checks.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)[:3500])
