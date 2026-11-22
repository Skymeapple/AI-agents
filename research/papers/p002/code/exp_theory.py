"""Theory experiments: best-approximation errors E_n(g) that govern the worst-case
width of the identified set for TTC functionals.  Outputs results/theory_*.csv and figures."""
import sys, os, json
import numpy as np
from scipy.special import erfc
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ttc import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)


def worst_case_width_lp(g, n, m=1500):
    """max_{mu,nu} int g d(mu - nu) s.t. mu, nu probability on grid with equal count laws."""
    p = cheb_grid(m, extra_log=True)
    B = bernstein_matrix(n, p).T
    gv = g(p); mm = len(p)
    c = np.concatenate([-gv, gv])
    A_eq = np.vstack([np.hstack([B, -B]),
                      np.hstack([np.ones((1, mm)), np.zeros((1, mm))]),
                      np.hstack([np.zeros((1, mm)), np.ones((1, mm))])])
    b_eq = np.concatenate([np.zeros(n + 1), [1, 1]])
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (2 * mm), method="highs")
    return -r.fun


def main():
    rows = []
    # (1) pass@k collapse on n^2/k
    for n in [4, 8, 16, 32, 64, 128]:
        ks = np.unique(np.round(np.logspace(np.log10(n + 1), np.log10(min(400 * n * n, 2e6)), 14)).astype(int))
        for k in ks:
            _, E = best_approx_error(lambda p, k=k: np.power(1 - p, k), n, m=2500)
            rows.append(dict(family="passk", n=n, k=int(k), x=n * n / k, E=E))
            print("passk", n, k, E, flush=True)
    # (2) majority collapse
    for n in [4, 8, 16, 32, 64]:
        ks = np.unique((np.round(np.logspace(np.log10(n + 1), np.log10(min(200 * n * n, 3e5)), 12)) // 2 * 2 + 1).astype(int))
        for k in ks:
            _, E = best_approx_error(g_majority(int(k)), n, m=2500)
            rows.append(dict(family="majority", n=n, k=int(k), x=n * n / k, E=E))
            print("maj", n, k, E, flush=True)
    # (3) verifier ceiling vs closed form
    t = 0.9
    for q in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
        s = np.sqrt(q / t)
        for n in [2, 4, 8, 16, 32, 64, 128]:
            _, E = best_approx_error(g_verifier_ceiling(t, q), n, m=2500)
            closed = 0.5 * ((1 - s) / (1 + s)) ** n
            rows.append(dict(family="ceiling", n=n, k=-1, q=q, t=t, E=E, closed=closed))
            print("ceil", q, n, E, closed, flush=True)
    # (4) BoN with verifier: horizon is the min of the two mechanisms
    for q in [0.0, 0.001, 0.01, 0.1]:
        for n in [16]:
            for k in [16, 32, 64, 128, 256, 512, 1024, 4096, 16384]:
                _, E = best_approx_error(g_bon_verifier(k, t, q), n, m=2500)
                rows.append(dict(family="bon_verifier", n=n, k=k, q=q, t=t, E=E))
    import csv
    keys = sorted({kk for r in rows for kk in r})
    with open(os.path.join(RES, "theory_En.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

    # (5) sanity: worst-case LP width == 2 E_n
    chk = []
    for (name, g, n) in [("pass@64,n=8", g_passk(64), 8), ("pass@256,n=16", g_passk(256), 16),
                         ("ceiling q=.01,n=8", g_verifier_ceiling(0.9, 0.01), 8),
                         ("maj@65,n=8", g_majority(65), 8)]:
        W = worst_case_width_lp(g, n)
        _, E = best_approx_error(g, n)
        chk.append(dict(case=name, lp_width=W, two_En=2 * E))
        print(chk[-1], flush=True)
    json.dump(chk, open(os.path.join(RES, "theory_duality_check.json"), "w"), indent=1)

    # ---------------- figures
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    d = df[df.family == "passk"]
    for n, dd in d.groupby("n"):
        ax[0].loglog(dd.x, 2 * dd.E, "o-", ms=3, label=f"n={n}")
    xs = np.logspace(-3, np.log10(20), 200)
    ax[0].loglog(xs, erfc(np.sqrt(xs)), "k--", label=r"erfc$(n/\sqrt{k})$")
    ax[0].loglog(xs, np.minimum(1, 4 * np.exp(-xs)), "k:", label=r"bound $4e^{-n^2/k}$")
    ax[0].set_ylim(1e-6, 1.5); ax[0].set_xlabel(r"$n^2/k$"); ax[0].set_ylabel("worst-case width  $2E_n$")
    ax[0].set_title("pass@k extrapolation"); ax[0].legend(fontsize=7)
    d = df[df.family == "majority"]
    for n, dd in d.groupby("n"):
        ax[1].loglog(dd.x, 2 * dd.E, "o-", ms=3, label=f"n={n}")
    ax[1].set_ylim(1e-6, 1.5); ax[1].set_xlabel(r"$n^2/k$"); ax[1].set_title("majority@k (binary) extrapolation")
    ax[1].legend(fontsize=7)
    d = df[df.family == "ceiling"]
    for q, dd in d.groupby("q"):
        l, = ax[2].semilogy(dd.n, 2 * dd.E, "o", ms=4, label=f"q={q}")
        ax[2].semilogy(dd.n, 2 * dd.closed, "-", color=l.get_color(), lw=1)
    ax[2].set_ylim(1e-12, 1.5); ax[2].set_xlabel("rollouts per task n"); ax[2].set_title("verifier-limited ceiling (t=0.9)\nmarkers: LP, lines: closed form")
    ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_theory.png"), dpi=150)


if __name__ == "__main__":
    main()
