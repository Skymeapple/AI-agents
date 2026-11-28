"""Exp 3 supplement (revision 1).

(a') Corrected Prop. 3(b): attempts needed for a level-alpha test with power beta at p = 0
     to rule out pass@K >= tau:  m >= K ln(beta/alpha) / ln(1/(1-tau)).
     Tabulate m/K (exact ratio and ceiling at K=1, K=100) for beta in {1, 0.9, 0.8, 0.5},
     and check numerically, by solving the LP for the most powerful (possibly randomized)
     test phi: {0..m} -> [0,1] against the composite null {p : 1-(1-p)^K >= tau}, that
     the maximal power at p = 0 equals min(1, alpha / (1-tau)^(m/K)).
(b') Tightness of the Prop. 2 rate bound: K at which the exact U_K first reaches 0.9 / 0.99
     vs the K implied by 1-(1-eps*)^K (from results/exp3b_vanishing.csv).
Outputs: results/exp3a_power_requirements.csv, results/exp3_supplement.json
"""
import csv
import json
import os

import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom

from common import RESULTS


def required_ratio(alpha, beta, tau):
    return np.log(beta / alpha) / np.log(1 / (1 - tau))


def max_power_lp(m, K, alpha, tau, ngrid=400):
    p0 = 1 - (1 - tau) ** (1 / K)
    pnull = np.unique(np.r_[p0, np.linspace(p0, 1, ngrid)])
    A = binom.pmf(np.arange(m + 1)[None, :], m, pnull[:, None])  # rows: null p; cols: c
    c = np.zeros(m + 1)
    c[0] = -1.0  # maximise phi(0) = power at p = 0
    r = linprog(c, A_ub=A, b_ub=np.full(len(pnull), alpha), bounds=[(0, 1)] * (m + 1), method="highs")
    return float(-r.fun)


def part_a():
    rows = []
    for alpha in (0.05,):
        for tau in (0.01, 0.05, 0.10):
            for beta in (1.0, 0.9, 0.8, 0.5):
                ratio = required_ratio(alpha, beta, tau)
                rows.append(dict(alpha=alpha, tau=tau, power_at_p0=beta, m_over_K_exact=ratio,
                                 m_required_K1=int(np.ceil(ratio)),
                                 m_required_K100=int(np.ceil(100 * ratio)),
                                 m_required_K1000=int(np.ceil(1000 * ratio))))
    with open(os.path.join(RESULTS, "exp3a_power_requirements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    checks = []
    for (m, K, tau) in [(30, 1, 0.1), (100, 3, 0.1), (300, 10, 0.1), (60, 2, 0.05), (300, 100, 0.1)]:
        lp = max_power_lp(m, K, 0.05, tau)
        formula = min(1.0, 0.05 / (1 - tau) ** (m / K))
        checks.append(dict(m=m, K=K, tau=tau, lp_max_power=lp, formula=formula))
    return dict(table=rows, lp_checks=checks,
                max_abs_lp_minus_formula=float(max(abs(c["lp_max_power"] - c["formula"]) for c in checks)))


def k_first(pmf, m, lev, grid):
    """Smallest integer K with sharp U_K >= lev (U_K is nondecreasing in K), by bisection."""
    from common import lp_bounds
    lo, hi = 1, 2
    while lp_bounds(pmf, pmf, m, hi, pgrid=grid)[1] < lev:
        lo, hi = hi, hi * 2
        if hi > 10 ** 6:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lp_bounds(pmf, pmf, m, mid, pgrid=grid)[1] >= lev:
            hi = mid
        else:
            lo = mid
    return hi if lp_bounds(pmf, pmf, m, lo, pgrid=grid)[1] < lev else lo


def part_b_exact_k():
    from scipy.stats import betabinom
    from common import default_grid
    out = []
    for name, (a, b) in {"Beta(0.5,1.5)": (0.5, 1.5), "Beta(2,2)": (2.0, 2.0)}.items():
        for m in (2, 4, 8):
            pmf = betabinom.pmf(np.arange(m + 1), m, a, b)
            es = [float(r["eps_star"]) for r in csv.DictReader(open(os.path.join(RESULTS, "exp3b_vanishing.csv")))
                  if r["population"].startswith(name) and int(r["m"]) == m][0]
            d = dict(population=name, m=m, eps_star=es)
            for lev in (0.9, 0.99):
                d[f"K_exact_{lev}"] = k_first(pmf, m, lev, default_grid())
                d[f"K_bound_{lev}"] = int(np.ceil(np.log(1 - lev) / np.log(1 - es)))
                d[f"ratio_bound_over_exact_{lev}"] = d[f"K_bound_{lev}"] / d[f"K_exact_{lev}"]
            out.append(d)
    return out


def part_b():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp3b_vanishing.csv"))))
    out = []
    for pop in sorted(set(r["population"] for r in rows)):
        for m in sorted(set(int(r["m"]) for r in rows)):
            rr = [r for r in rows if r["population"] == pop and int(r["m"]) == m]
            if not rr:
                continue
            es = float(rr[0]["eps_star"])
            d = dict(population=pop, m=m, eps_star=es)
            for lev in (0.9, 0.99):
                hit = [int(r["K"]) for r in rr if float(r["U"]) >= lev]
                d[f"K_exact_U_ge_{lev}"] = min(hit) if hit else None
                d[f"K_bound_U_ge_{lev}"] = (float(np.log(1 - lev) / np.log(1 - es)) if es > 0 else None)
            # U_K and bound at K = 100
            r100 = [r for r in rr if int(r["K"]) == 100][0]
            d["U_K100"] = float(r100["U"])
            d["bound_K100"] = float(r100["lower_bound_on_U"])
            out.append(d)
    return out


if __name__ == "__main__":
    out = {"a_power": part_a(), "b_rate_tightness": part_b(), "b_exact_K": part_b_exact_k()}
    with open(os.path.join(RESULTS, "exp3_supplement.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
