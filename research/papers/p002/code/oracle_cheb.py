"""Revision r2: better-conditioned oracle identified intervals (reviewer Q3).
The identified set {theta_g(F): pi_F = pi} depends only on the degree-n moment vector of F.  Instead of equality
constraints on the Bernstein count law (condition number ~ binomial coefficients), we impose equality of the
Chebyshev moments int T_j(2p-1) dF, j = 0..n (an equivalent constraint set: same polynomial span), computed
directly from the plug-in law, on a grid that contains the plug-in atoms p_hat_i (so the system is exactly
feasible).  Solved with HiGHS; a residual check is reported."""
import numpy as np
from scipy.optimize import linprog
from ttc import cheb_grid


def oracle_interval_cheb(g, n, ph, m=4000):
    p = np.unique(np.concatenate([cheb_grid(m), ph]))
    V = np.polynomial.chebyshev.chebvander(2 * p - 1, n).T          # (n+1, mm), entries in [-1, 1]
    mom = np.polynomial.chebyshev.chebvander(2 * ph - 1, n).mean(0)
    gv = g(p); out = []; res = []
    for sgn in (+1, -1):
        r = linprog(sgn * gv, A_eq=V, b_eq=mom, bounds=[(0, None)] * len(p), method="highs",
                    options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if r.status != 0:
            out.append(np.nan); res.append(np.nan); continue
        out.append(sgn * r.fun); res.append(float(np.max(np.abs(V @ r.x - mom))))
    return out[0], out[1], max(res) if res else np.nan
