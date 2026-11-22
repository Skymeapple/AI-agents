"""Exp 1: numerical companion to Theorems 2-3.
(a) best uniform approximation error E_n of f_B(x)=x^B on [0,1] by degree-n
    polynomials (x = 1-p), computed by LP on a Chebyshev grid, vs. the bound
    2 exp(-n^2/B) proved in the paper.  Worst-case identified-set width = 2E_n.
(b) identified interval for C(B) in the population limit (T -> infinity,
    histogram known exactly) for concrete task populations, as a function of n.
"""
import json, os, time
import numpy as np
from numpy.polynomial import chebyshev as Ch
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import binom_matrix, default_grid, lp_bounds

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def best_approx_error(B, n, M=3000):
    # grid in x in [0,1]; Chebyshev-extrema grid plus extra points near x=1
    t = np.cos(np.pi * np.arange(M + 1) / M)
    x = np.unique(np.concatenate([(t + 1) / 2, 1 - np.logspace(-9, -0.3, 600)]))
    f = x ** B
    V = Ch.chebvander(2 * x - 1, n)  # (len x, n+1)
    K = n + 1
    # variables: c (K, free), e (>=0); minimise e s.t. |f - Vc| <= e
    c_obj = np.zeros(K + 1); c_obj[-1] = 1
    A_ub = np.vstack([np.hstack([-V, -np.ones((len(x), 1))]),
                      np.hstack([V, -np.ones((len(x), 1))])])
    b_ub = np.concatenate([-f, f])
    bounds = [(None, None)] * K + [(0, None)]
    r = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return r.x[-1]


def part_a():
    Bs = [16, 64, 256, 1024, 4096]
    rows = []
    for B in Bs:
        ns = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]))
        for n in ns:
            if n >= B:
                continue
            e = best_approx_error(B, n)
            rows.append(dict(B=B, n=n, E_n=float(e), width=float(2 * e),
                             bound=float(min(1.0, 4 * np.exp(-n ** 2 / B)))))
            print(f"B={B:5d} n={n:4d}  2E_n={2*e:.4g}  bound={min(1,4*np.exp(-n**2/B)):.4g}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "exp1a_approx_error.json"), "w"), indent=1)
    # minimal n for worst-case width <= eps
    summ = []
    for eps in [0.1, 0.05, 0.01]:
        for B in Bs:
            ok = [r["n"] for r in rows if r["B"] == B and r["width"] <= eps]
            summ.append(dict(eps=eps, B=B, n_min_grid=min(ok) if ok else None,
                             sqrt_bound=float(np.sqrt(B * np.log(4 / eps)))))
    json.dump(summ, open(os.path.join(OUT, "exp1a_nmin.json"), "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for B in Bs:
        rr = [r for r in rows if r["B"] == B]
        l, = ax[0].semilogy([r["n"] for r in rr], [max(r["width"], 1e-12) for r in rr], "o-", label=f"B={B}")
        ax[1].semilogy([r["n"] / np.sqrt(B) for r in rr], [max(r["width"], 1e-12) for r in rr], "o-",
                       color=l.get_color(), label=f"B={B}")
    z = np.linspace(0, 3, 50)
    ax[1].semilogy(z, np.minimum(1, 4 * np.exp(-z ** 2)), "k--", label=r"bound $4e^{-n^2/B}$")
    ax[0].set_xlabel("attempts per task n"); ax[0].set_ylabel(r"worst-case identified width $2E_n$")
    ax[1].set_xlabel(r"$n/\sqrt{B}$"); ax[1].set_ylim(1e-8, 2); ax[0].set_ylim(1e-8, 2)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    ax[0].set_title("Width of identified set for C(B), infinite tasks")
    ax[1].set_title(r"Collapse onto $n/\sqrt{B}$")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_identification_width.png"), dpi=150)


def populations(rng, size=400000):
    pops = {}
    pops["beta(0.2,2)"] = rng.beta(0.2, 2.0, size)
    p = rng.beta(0.8, 0.6, size); p[rng.random(size) < 0.35] = 0.0
    pops["zero+beta"] = p
    p = rng.beta(0.8, 0.8, size); m = rng.random(size) < 0.4
    p[m] = 10 ** rng.uniform(-4, -1, m.sum()); pops["hard-tail"] = p
    p = rng.beta(1, 1, size); u = rng.random(size)
    p[u < 0.25] = 0.0; p[(u >= 0.25) & (u < 0.5)] = 3e-3; pops["locked"] = p
    return pops


def part_b():
    rng = np.random.default_rng(0)
    pops = populations(rng)
    rows = []
    for name, ps in pops.items():
        for B in [100, 1000]:
            ps = ps[:40000]  # G := empirical law of these atoms (q and C(B) exact for G)
            CB = float(np.mean(1 - (1 - ps) ** B))
            for n in [2, 4, 8, 16, 32, 64]:
                q = np.mean(binom_matrix(n, ps), axis=1)  # exact-ish histogram probs
                # population-limit identified set (tiny slack for MC error in q)
                lo, hi = lp_bounds(None, n, B, exact_q=q, slack=1e-4)
                rows.append(dict(pop=name, B=B, n=n, C_true=CB, lo=lo, hi=hi))
                print(name, B, n, f"C={CB:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    json.dump(rows, open(os.path.join(OUT, "exp1b_population_identified.json"), "w"), indent=1)


if __name__ == "__main__":
    t0 = time.time(); part_a(); part_b(); print("done", time.time() - t0)
