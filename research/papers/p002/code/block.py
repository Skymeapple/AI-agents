"""Block estimator for pass@k with k > n (Theorem 5(a)), a binomial analogue of the smoothed Good-Toulmin idea.

Unbiased building blocks: for m <= n, U_m(X) = C(n-X, m) / C(n, m) satisfies E_p U_m(X) = (1-p)^m and 0 <= U_m <= 1.
Choose a degree L, block length b = floor(n/(L+1)), and write k = q b + r (0 <= r < b).  With w = (1-p)^b,
(1-p)^k = (1-p)^r w^q.  Let Q(w) = sum_j c_j w^j be a degree-L polynomial approximating w^q on [0,1].  Then

    Z(X) = 1 - sum_{j=0}^{L} c_j U_{r + j b}(X)        (all indices r + j b <= n)

has |E_p Z - pass@k(p)| = (1-p)^r |Q(w) - w^q| <= ||Q - w^q||_[0,1] =: beta  for every p, and Z takes the n+1 known
values Z(0..n).  A 1-eps interval valid for every F and for fixed non-identical tasks is
    mean_i Z(X_i) -/+ (beta + dev),   dev = (max Z - min Z) sqrt(ln(2/eps) / (2M))       (Hoeffding),
or with an empirical-Bernstein deviation (Maurer & Pontil 2009) for data-adaptive widths.
beta is evaluated rigorously: fine grid in w plus a Markov-inequality Lipschitz slack.
"""
import numpy as np
import mpmath as mp
from scipy.optimize import linprog

mp.mp.dps = 50


def best_poly_monomial(q, L, m=6000):
    """Minimax degree-L approximation of w^q on [0,1] (LP in Chebyshev basis); returns monomial coeffs (mp)."""
    j = np.arange(m)
    w = 0.5 * (1 - np.cos(np.pi * j / (m - 1)))
    x = 2 * w - 1
    V = np.polynomial.chebyshev.chebvander(x, L)
    fv = w ** q
    A = np.vstack([np.hstack([-V, -np.ones((m, 1))]), np.hstack([V, -np.ones((m, 1))])])
    b = np.concatenate([-fv, fv])
    r = linprog(np.r_[np.zeros(L + 1), 1.0], A_ub=A, b_ub=b, bounds=[(None, None)] * (L + 1) + [(0, None)],
                method="highs")
    cc = r.x[:L + 1]
    # exact (mp) conversion Chebyshev(2w-1) -> monomials in w
    coef = [mp.mpf(0)] * (L + 1)
    # T_j(2w-1) monomial coefficients by recurrence, in mp
    Tprev, Tcur = [mp.mpf(1)], [mp.mpf(-1), mp.mpf(2)]
    Ts = [Tprev, Tcur]
    for _ in range(2, L + 1):
        nxt = [mp.mpf(0)] * (len(Tcur) + 1)
        for i, v in enumerate(Tcur):          # 2(2w-1) T_j - T_{j-1}
            nxt[i] += -2 * v
            nxt[i + 1] += 4 * v
        for i, v in enumerate(Ts[-2]):
            nxt[i] -= v
        Ts.append(nxt); Tcur = nxt
    for jj in range(L + 1):
        for i, v in enumerate(Ts[jj]):
            coef[i] += mp.mpf(float(cc[jj])) * v
    return coef


def rigorous_beta(coef, q, ngrid=200001):
    """sup_{w in [0,1]} |Q(w) - w^q| on a fine grid + Lipschitz slack (Markov: |Q'| <= 2 L^2 ||Q||)."""
    L = len(coef) - 1
    w = np.linspace(0, 1, ngrid)
    cf = np.array([float(c) for c in coef])
    Qv = np.polynomial.polynomial.polyval(w, cf)
    err = np.max(np.abs(Qv - w ** q))
    Qn = np.max(np.abs(Qv))
    lip = 2 * L * L * (Qn + err + 1e-9) * 1.01 + q
    return float(err + lip * 0.5 / (ngrid - 1))


def z_values(coef, n, k, L):
    """Z(x) for x = 0..n (mp evaluation of the combination, returned as float)."""
    b = n // (L + 1)
    q, r = divmod(k, b)
    Z = []
    for x in range(n + 1):
        s = mp.mpf(0)
        for j, c in enumerate(coef):
            m = r + j * b
            s += c * (mp.binomial(n - x, m) / mp.binomial(n, m) if n - x >= m else 0)
        Z.append(float(1 - s))
    return np.array(Z), q, r, b


def design(n, k, L):
    b = n // (L + 1)
    if b < 1:
        return None
    q, r = divmod(k, b)
    coef = best_poly_monomial(q, L)
    beta = rigorous_beta(coef, q)
    Z, q, r, b = z_values(coef, n, k, L)
    return dict(L=L, b=b, q=q, r=r, beta=beta, Z=Z, zrange=float(Z.max() - Z.min()))


def hoeffding_halfwidth(des, M, eps=0.05):
    return des["beta"] + des["zrange"] * np.sqrt(np.log(2 / eps) / (2 * M))


def best_design(n, k, M, eps=0.05, Lmax=14):
    best = None
    for L in range(0, min(Lmax, n - 1) + 1):
        d = design(n, k, L)
        if d is None:
            continue
        h = hoeffding_halfwidth(d, M, eps)
        if best is None or h < best[0]:
            best = (h, d)
    return best


def interval(des, X, eps=0.05, method="hoeffding"):
    """Certified interval from observed counts X (array of ints in 0..n)."""
    z = des["Z"][np.asarray(X)]
    M = len(z)
    lo_z, R = des["Z"].min(), des["zrange"]
    if method == "hoeffding":
        dev = R * np.sqrt(np.log(2 / eps) / (2 * M))
    else:  # empirical Bernstein (Maurer & Pontil 2009, Thm 4), applied to (z - lo_z)/R in [0,1], each side eps/2
        u = (z - lo_z) / max(R, 1e-300)
        V = np.var(u, ddof=1)
        lg = np.log(4 / eps)
        dev = R * (np.sqrt(2 * V * lg / M) + 7 * lg / (3 * (M - 1)))
    c = float(np.mean(z))
    return c - des["beta"] - dev, c + des["beta"] + dev
