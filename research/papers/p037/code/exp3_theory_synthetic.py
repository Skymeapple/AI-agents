"""Exp 3: Checks of the theory on synthetic task populations.

(a) 'Rule of three for attackers': zero successes in m attempts gives the exact
    (1-alpha) upper bound 1 - alpha^(K/m) on pass@K; tabulate the attempts needed
    to rule out pass@K <= tau and verify coverage by simulation.
(b) Vanishing identification (Prop. 2): with the count pmf known exactly, the
    sharp upper bound U_K -> 1 for interior moment points, and U_K >= 1-(1-eps*)^K,
    where eps* = sup{eps : moments representable on [eps,1]}. Boundary case (finite
    atoms incl. 0) stays bounded.
(c) Misspecification: plug-in, beta-binomial and LP confidence bound (with and
    without a resolution floor) vs. truth, for populations with rare-success tasks.

Outputs: results/exp3a_required_attempts.csv, results/exp3a_coverage.csv,
         results/exp3b_vanishing.csv, results/exp3c_misspec.csv, results/exp3_summary.json
"""
import csv
import json
import os

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as beta_dist

from common import (RESULTS, beta_pass_at_k, binom_pmf_matrix, count_pmf, default_grid,
                    fit_beta_binomial, lp_bounds, pmf_confidence_box)

rng = np.random.default_rng(3)


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- (a)
def part_a():
    rows = []
    for alpha in (0.05, 0.10):
        for tau in (0.01, 0.05, 0.10, 0.50):
            for K in (1, 10, 100, 1000, 10000):
                m_req = int(np.ceil(K * np.log(1 / alpha) / np.log(1 / (1 - tau))))
                rows.append(dict(alpha=alpha, tau=tau, K=K, m_required=m_req,
                                 ratio_m_over_K=m_req / K))
    write("exp3a_required_attempts.csv", rows)
    # coverage: single task, true p; evaluator runs m attempts; if zero successes use
    # closed form, else exact Clopper-Pearson upper bound; bound on pass@K.
    cov = []
    alpha = 0.05
    for m in (30, 300):
        for K in (10, 100):
            for p in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                R = 20000
                x = rng.binomial(m, p, R)
                pu = np.where(x == m, 1.0, beta_dist.ppf(1 - alpha, x + 1, m - x))
                ub = 1 - (1 - pu) ** K
                truth = 1 - (1 - p) ** K
                zero = x == 0
                cov.append(dict(m=m, K=K, p=p, true_passK=truth,
                                closed_form_bound=1 - alpha ** (K / m),
                                frac_zero=float(zero.mean()),
                                coverage=float((ub >= truth).mean()),
                                rule_out_0p1_rate=float((ub < 0.1).mean())))
    write("exp3a_coverage.csv", cov)
    return dict(min_coverage_a=min(r["coverage"] for r in cov),
                m_over_K_alpha05_tau10=[r["ratio_m_over_K"] for r in rows
                                        if r["alpha"] == 0.05 and r["tau"] == 0.10][0])


# ---------------------------------------------------------------- (b)
def moments_from_G(atoms, w, m):
    """Exact Binomial(m) count pmf of mixture sum_j w_j delta_{atoms_j}."""
    A = binom_pmf_matrix(m, np.asarray(atoms, float))
    return A @ np.asarray(w, float)


def beta_pmf(a, b, m, J=4000):
    # exact beta-binomial pmf
    from scipy.stats import betabinom
    return betabinom.pmf(np.arange(m + 1), m, a, b)


def eps_star(pmf, m, tol=1e-7):
    """sup eps s.t. pmf is a Binomial(m) mixture with support in [eps, 1] (bisection on LP feasibility)."""
    def feasible(eps):
        g = default_grid()
        g = g[g >= eps]
        g = np.unique(np.r_[eps, g])
        A = binom_pmf_matrix(m, g)
        r = linprog(np.zeros(len(g)), A_eq=np.vstack([A, np.ones(len(g))]),
                    b_eq=np.r_[pmf, 1.0], bounds=(0, None), method="highs")
        return r.status == 0
    if not feasible(1e-7):
        return 0.0
    lo, hi = 1e-7, 1.0
    while hi / lo > 1 + 1e-3:
        mid = np.sqrt(lo * hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def part_b():
    pops = {
        "Beta(0.5,1.5) [interior]": ("beta", (0.5, 1.5)),
        "Beta(2,2) [interior]": ("beta", (2.0, 2.0)),
        "0.4*delta0+0.6*delta0.5 [boundary]": ("atoms", ([0.0, 0.5], [0.4, 0.6])),
    }
    Ks = [1, 2, 4, 8, 16, 32, 100, 316, 1000, 3162, 10000]
    rows = []
    for name, (kind, par) in pops.items():
        for m in (2, 4, 8):
            if kind == "beta":
                pmf = beta_pmf(*par, m)
            else:
                pmf = moments_from_G(par[0], par[1], m)
            es = eps_star(pmf, m)
            for K in Ks:
                if kind == "beta":
                    truth = beta_pass_at_k(*par, K)
                else:
                    truth = float(np.sum(np.array(par[1]) * (1 - (1 - np.array(par[0])) ** K)))
                grid = default_grid() if kind == "beta" else np.union1d(default_grid(), par[0])
                L, U, s = lp_bounds(pmf, pmf, m, K, pgrid=grid, return_slack=True)
                rows.append(dict(population=name, m=m, K=K, truth=truth, L=L, U=U,
                                 eps_star=es, lower_bound_on_U=1 - (1 - es) ** K, slack=s))
    write("exp3b_vanishing.csv", rows)
    tol = 1e-4  # LP discretisation tolerance
    viol = [r for r in rows if r["U"] < r["lower_bound_on_U"] - tol or not (r["L"] - tol <= r["truth"] <= r["U"] + tol)]
    return dict(prop2_violations=len(viol))


# ---------------------------------------------------------------- (c)
def sample_population(name, n):
    if name == "beta_correct":  # Beta(0.3,1.0): beta-binomial well specified
        return rng.beta(0.3, 1.0, n)
    if name == "rare_success":  # 50% Beta(2,2), 30% impossible, 20% rare p in [1e-3,1e-2]
        u = rng.random(n)
        p = np.where(u < 0.5, rng.beta(2, 2, n), 0.0)
        rare = u >= 0.8
        p[rare] = 10 ** rng.uniform(-3, -2, rare.sum())
        return p
    if name == "mostly_impossible":  # 85% impossible, 15% p in [3e-3, 3e-2]
        u = rng.random(n)
        p = np.zeros(n)
        r = u < 0.15
        p[r] = 10 ** rng.uniform(np.log10(3e-3), np.log10(3e-2), r.sum())
        return p
    raise ValueError(name)


def true_passK(name, K, Nmc=400000):
    p = sample_population(name, Nmc)
    return float(np.mean(1 - (1 - p) ** K))


def part_c(reps=150, n=200, m=4, alpha=0.10):
    Ks = [1, 10, 100, 1000]
    scen = ["beta_correct", "rare_success", "mostly_impossible"]
    truth = {(s, K): true_passK(s, K) for s in scen for K in Ks}
    rows = []
    for s in scen:
        for rep in range(reps):
            p = sample_population(s, n)
            counts = rng.binomial(m, p)
            a, b = fit_beta_binomial(counts, m)
            lo, hi = pmf_confidence_box(counts, np.arange(n), m, alpha, rng, B=400)
            lob, hib = pmf_confidence_box(counts, np.arange(n), m, alpha, rng, B=400, method="bootstrap")
            for K in Ks:
                _, U, _ = lp_bounds(lo, hi, m, K, return_slack=True)
                _, Ub, _ = lp_bounds(lob, hib, m, K, return_slack=True)
                _, Uf, _ = lp_bounds(lo, hi, m, K, pmin=1e-3, return_slack=True)
                rows.append(dict(scenario=s, rep=rep, K=K, truth=truth[(s, K)],
                                 plugin=float(np.mean(1 - (1 - counts / m) ** K)),
                                 beta_binomial=beta_pass_at_k(a, b, K),
                                 lp_ci_U=U, lp_ci_U_bootstrap_only=Ub, lp_ci_U_floor1e3=Uf,
                                 all_zero=int(counts.sum() == 0)))
        print("done", s, flush=True)
    write("exp3c_misspec.csv", rows)
    summ = []
    for s in scen:
        for K in Ks:
            rr = [r for r in rows if r["scenario"] == s and r["K"] == K]
            d = dict(scenario=s, K=K, truth=truth[(s, K)])
            d["frac_reps_all_zero"] = float(np.mean([r["all_zero"] for r in rr]))
            for meth in ["plugin", "beta_binomial", "lp_ci_U", "lp_ci_U_bootstrap_only", "lp_ci_U_floor1e3"]:
                v = np.array([r[meth] for r in rr])
                d[meth + "_mean"] = float(np.mean(v))
                d[meth + "_coverage"] = float(np.mean(v >= truth[(s, K)] - 1e-9))
            summ.append(d)
    return summ


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "b":
        p = os.path.join(RESULTS, "exp3_summary.json")
        out = json.load(open(p))
        out["b"] = part_b()
        print(out["b"])
        json.dump(out, open(p, "w"), indent=2)
        sys.exit(0)
    out = {}
    out["a"] = part_a()
    print(out["a"])
    out["b"] = part_b()
    print(out["b"])
    out["c"] = part_c()
    for r in out["c"]:
        print(r)
    with open(os.path.join(RESULTS, "exp3_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
