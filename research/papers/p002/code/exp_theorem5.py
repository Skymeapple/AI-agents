"""Revision r2: numerical checks of Theorem 5 (forecast horizon k* = Theta(n min{log M, n})).

(A) Markov constant.  For lambda = mu L^2 and f(t) = (1 - lambda t / k)^k on [0,1], Lemma 5.1 gives
    E_L(f) >= eta_mu := max_c (1 - e^{-c} - 2c/mu) / (2 + 2c/mu)   (eta_6 = 0.1281, eta_4 = 0.0658).
    Checked by LP for several (L, k).
(B) Converse construction.  The extremal signed measure sigma (L+2 alternation points, sigma ⊥ P_L, ||sigma|| = 2),
    pushed to p = lambda t / k, gives two laws with |theta_F - theta_G| = 2 E_L(f) and
    TV(pi_F, pi_G) <= tau := sum_{m>L} C(n,m) (2 lambda / k)^m.  Computed in 60-digit arithmetic.
(C) Explicit converse horizon: smallest k such that some L has k >= mu L^2 and M tau <= 1/4, for mu in {4, 6}
    (then R* >= eta_mu (1 - 2 eps - 1/4)); this is a closed-form, fully rigorous (if loose) bound.
(D) Achievability ingredients.  (i) The monomial coefficients of T_j(2w-1) alternate in sign, so their absolute sum
    is T_j(3) (checked exactly, j <= 40).  (ii) Block estimator bias <= E_L(w^q) <= 2 exp(-L^2/q) (checked on a grid).
Output: results/theorem5_checks.json, results/theorem5_converse_horizon.csv
"""
import os, sys, json
import numpy as np, mpmath as mp
from scipy.optimize import linprog
from scipy.special import comb
sys.path.insert(0, os.path.dirname(__file__))
from ttc import cheb_grid
RES = os.path.join(os.path.dirname(__file__), "..", "results")
mp.mp.dps = 60


def best_poly(fv, t, L):
    """Minimax degree-L approximation on grid t (LP in a Chebyshev basis). Returns (E, coeffs_cheb, alternation pts)."""
    x = 2 * t - 1
    V = np.polynomial.chebyshev.chebvander(x, L)
    m = len(t)
    # variables: c (L+1), e ; min e s.t. -e <= f - Vc <= e
    A = np.vstack([np.hstack([-V, -np.ones((m, 1))]), np.hstack([V, -np.ones((m, 1))])])
    b = np.concatenate([-fv, fv])
    r = linprog(np.r_[np.zeros(L + 1), 1.0], A_ub=A, b_ub=b, bounds=[(None, None)] * (L + 1) + [(0, None)], method="highs")
    c = r.x[:L + 1]; E = r.x[-1]
    res = fv - V @ c
    return E, c, res


def alternation(t, res, E, L):
    """Pick L+2 alternating extremal points of the residual."""
    idx = np.where(np.abs(np.abs(res) - E) < 1e-6 * max(E, 1e-12) + 1e-12)[0]
    pts = []
    for i in idx:
        s = np.sign(res[i])
        if pts and np.sign(res[pts[-1]]) == s:
            if abs(res[i]) > abs(res[pts[-1]]):
                pts[-1] = i
        else:
            pts.append(i)
    return pts




def eta_mu(mu):
    c = np.linspace(1e-4, 12, 200000)
    return float(np.max((1 - np.exp(-c) - 2 * c / mu) / (2 + 2 * c / mu)))


def markov_check(L, k, mu):
    t = np.unique(np.concatenate([cheb_grid(6000, extra_log=False), np.linspace(0, 1, 4001)]))
    E, _, _ = best_poly((1 - mu * L * L * t / k) ** k, t, L)
    return dict(L=L, k=k, mu=mu, E_L=float(E), eta_mu=eta_mu(mu), holds=bool(E >= eta_mu(mu) - 1e-9))


def log_tau(n, k, L, mu):
    from scipy.special import gammaln
    a2 = 2 * mu * L * L / k
    m = np.arange(L + 1, n + 1)
    if len(m) == 0:
        return -np.inf
    lt = gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1) + m * np.log(a2)
    mx = lt.max()
    return float(mx + np.log(np.exp(lt - mx).sum()))


def converse_horizon(n, M, mu):
    best = np.inf
    for L in range(1, n + 1):
        lo = int(np.ceil(2 * mu * L * L))   # 2 lambda/k <= 1 (also gives p <= 1)
        if L >= n:                            # tau = 0: population case, k >= mu n^2 suffices
            best = min(best, int(np.ceil(mu * n * n))); continue
        hi = 10 ** 12
        if np.log(M) + log_tau(n, hi, L, mu) > np.log(0.25):
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if np.log(M) + log_tau(n, mid, L, mu) <= np.log(0.25):
                hi = mid
            else:
                lo = mid
        best = min(best, hi)
    return best


def shifted_cheb_alternation(jmax=40):
    ok = True; worst = 0.0
    for j in range(1, jmax + 1):
        # monomial coefficients of T_j(2w-1), exact integers via mpmath
        c = [mp.mpf(0)] * (j + 1)
        T0, T1 = [mp.mpf(1)], [mp.mpf(-1), mp.mpf(2)]
        Ts = [T0, T1]
        for _ in range(2, j + 1):
            a, b = Ts[-1], Ts[-2]
            nx = [mp.mpf(0)] * (len(a) + 1)
            for i, v in enumerate(a):
                nx[i] += -2 * v; nx[i + 1] += 4 * v
            for i, v in enumerate(b):
                nx[i] -= v
            Ts.append(nx)
        cj = Ts[j]
        signs = [mp.sign(v) for v in cj]
        alt = all(signs[i] == (-1) ** (j - i) for i in range(j + 1))
        l1 = mp.fsum(abs(v) for v in cj)
        T3 = mp.chebyt(j, 3)
        ok &= alt and abs(l1 - T3) < 1e-30 * T3
    return ok

def converse_check(n, L, k):
    lam = 6 * L * L
    a = lam / k
    t = np.unique(np.concatenate([cheb_grid(6000, extra_log=False), np.linspace(0, 1, 4001)]))
    fv = (1 - a * t) ** k
    E, c, res = best_poly(fv, t, L)
    pts = alternation(t, res, E, L)
    out = dict(n=n, L=L, k=k, lam=lam, a=a, E_L=E, E_L_gt_0p1=bool(E > 0.1), n_alt=len(pts))
    if len(pts) < L + 2:
        return out
    pts = pts[:L + 2]
    tp = [mp.mpf(float(t[i])) for i in pts]
    # sigma ⊥ P_L on L+2 points: null vector of the (L+1) x (L+2) Vandermonde system, normalised to ||sigma|| = 2
    A = mp.matrix(L + 1, L + 2)
    for m in range(L + 1):
        for j in range(L + 2):
            A[m, j] = tp[j] ** m
    # null vector via Lagrange/divided-difference weights: w_j = 1 / prod_{i != j}(t_j - t_i)
    w = [1 / mp.fprod([tp[j] - tp[i] for i in range(L + 2) if i != j]) for j in range(L + 2)]
    s = mp.fsum(abs(v) for v in w)
    sig = [2 * v / s for v in w]
    orth = max(abs(mp.fsum(sig[j] * tp[j] ** m for j in range(L + 2))) for m in range(L + 1))
    am, km = mp.mpf(lam) / k, k
    f = lambda tt: (1 - am * tt) ** km
    dtheta = abs(mp.fsum(sig[j] * f(tp[j]) for j in range(L + 2)))
    # count-law L1 difference: d_x = sum_j sig_j b_{n,x}(a t_j)
    d = []
    for x in range(n + 1):
        d.append(mp.fsum(sig[j] * mp.binomial(n, x) * (am * tp[j]) ** x * (1 - am * tp[j]) ** (n - x) for j in range(L + 2)))
    L1 = mp.fsum(abs(v) for v in d)
    bound = 2 * mp.fsum(mp.binomial(n, m) * (2 * am) ** m for m in range(L + 1, n + 1))
    out.update(orth_err=float(orth), delta_theta=float(dtheta), two_E_L=2 * E, count_L1=float(L1), count_L1_bound=float(bound),
               bound_holds=bool(L1 <= bound * (1 + 1e-12)), H2_upper=float(L1) / 2)
    return out


def achievability_check(n, L, k):
    b = n // (L + 1); q, r = divmod(k, b)
    w = np.unique(np.concatenate([cheb_grid(4000, extra_log=False), np.linspace(0, 1, 2001)]))
    E, cc, _ = best_poly(w ** q, w, L)
    # monomial coefficients of Q(w) = sum cc_j T_j(2w-1)
    Pm = np.polynomial.chebyshev.Chebyshev(cc, domain=[0, 1]).convert(kind=np.polynomial.Polynomial).coef
    x = np.arange(n + 1)
    U = np.array([comb(n - x, r + j * b) / comb(n, r + j * b) for j in range(L + 1)])  # (L+1, n+1)
    h = 1 - Pm @ U
    p = np.unique(np.concatenate([cheb_grid(20000, extra_log=True)]))
    from scipy.stats import binom
    Bm = binom.pmf(x[None, :], n, p[:, None])
    bias = float(np.max(np.abs(Bm @ h - (1 - (1 - p) ** k))))
    TL3 = float(np.polynomial.chebyshev.chebval(3.0, np.r_[np.zeros(L), 1.0]))
    Qnorm = float(np.max(np.abs(np.polynomial.Polynomial(Pm)(w))))
    return dict(n=n, L=L, k=k, b=b, q=q, r=r, E_L_wq=E, SV_bound=2 * np.exp(-L * L / q), exact_max_bias=bias,
                bias_le_E=bool(bias <= E * (1 + 1e-5) + 1e-9), range_exact=float(np.max(np.abs(Pm @ U))),
                coef_l1=float(np.sum(np.abs(Pm))), markov_bound=Qnorm * TL3)


if __name__ == "__main__":
    import pandas as pd
    out = dict(eta_6=eta_mu(6), eta_4=eta_mu(4), markov=[], converse=[], achievability=[])
    for (L, k) in [(1, 6), (2, 24), (2, 1000), (3, 54), (4, 400), (6, 216), (8, 2000), (10, 600)]:
        for mu in (4, 6):
            r = markov_check(L, max(k, int(np.ceil(mu * L * L))), mu); out["markov"].append(r); print(r, flush=True)
    out["markov_all_hold"] = bool(all(r["holds"] for r in out["markov"]))
    for (n, L, k) in [(16, 2, 4000), (16, 3, 6000), (32, 3, 9000), (64, 4, 25000), (64, 6, 40000)]:
        r = converse_check(n, L, k); print(r, flush=True); out["converse"].append(r)
    for (n, L, k) in [(64, 3, 100), (64, 5, 200), (128, 5, 400), (128, 7, 640), (64, 7, 320)]:
        r = achievability_check(n, L, k); print(r, flush=True); out["achievability"].append(r)
    out["shifted_chebyshev_alternation_l1_eq_T3"] = bool(shifted_cheb_alternation())
    rows = []
    for mu in (4, 6):
        for n in [16, 64, 256, 1024, 4096]:
            for M in [1e2, 1e3, 1e4, 1e6, 1e8, 1e12]:
                k = converse_horizon(n, M, mu)
                rows.append(dict(mu=mu, halfwidth_lb=eta_mu(mu) * (1 - 2 * 0.05 - 0.25), n=n, M=M, k_conv=k, s_conv=k / n))
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(RES, "theorem5_converse_horizon.csv"), index=False, float_format="%.5g")
    print(tab.pivot_table(index=["mu", "M"], columns="n", values="s_conv").to_string())
    json.dump(out, open(os.path.join(RES, "theorem5_checks.json"), "w"), indent=1, default=float)
