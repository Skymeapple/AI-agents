"""Revision r2: numerical checks of Theorem 5 (the n*log M horizon) -- both explicit constructions.

(A) Converse construction.  For L >= 1 and lambda = 6 L^2, f(t) = (1 - lambda t / k)^k on [0,1].
    Claim 1: E_L(f) > 0.1 (Markov-inequality argument).  Checked by LP for several (L, k).
    Claim 2: the extremal signed measure sigma (L+2 alternation points, sigma ⊥ P_L, ||sigma|| = 2), pushed to
    p = lambda t / k, gives two laws F = sigma^+, G = sigma^- with |theta_F - theta_G| = 2 E_L(f) and
    sum_j |pi_F(j) - pi_G(j)| <= 2 sum_{m>L} C(n,m) (2 a)^m, a = lambda/k.  Computed in 60-digit arithmetic.
(B) Achievability construction.  Block estimator Z = 1 - sum_j c_j U_{r+jb}(X), U_m(X) = C(n-X,m)/C(n,m),
    b = floor(n/(L+1)), k = q b + r.  Claim: |E Z - pass@k(p)| <= E_L(w^q) <= 2 exp(-L^2/(2q)) for all p,
    and |sum_j c_j U| <= sum |c_j| <= ||Q|| T_L(3).  Checked exactly on a fine p grid.
Output: results/theorem5_checks.json
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
    return dict(n=n, L=L, k=k, b=b, q=q, r=r, E_L_wq=E, SV_bound=2 * np.exp(-L * L / (2 * q)), exact_max_bias=bias,
                bias_le_E=bool(bias <= E * (1 + 1e-5) + 1e-9), range_exact=float(np.max(np.abs(Pm @ U))),
                coef_l1=float(np.sum(np.abs(Pm))), markov_bound=Qnorm * TL3)


if __name__ == "__main__":
    out = dict(converse=[], achievability=[])
    for (n, L, k) in [(16, 2, 4000), (16, 3, 6000), (32, 3, 9000), (64, 4, 25000), (64, 6, 40000)]:
        r = converse_check(n, L, k); print(r, flush=True); out["converse"].append(r)
    for (n, L, k) in [(64, 3, 100), (64, 5, 200), (128, 5, 400), (128, 7, 640), (64, 7, 320)]:
        r = achievability_check(n, L, k); print(r, flush=True); out["achievability"].append(r)
    json.dump(out, open(os.path.join(RES, "theorem5_checks.json"), "w"), indent=1, default=float)
