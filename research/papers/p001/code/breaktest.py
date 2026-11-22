"""Shared implementation of the break test exactly as applied to the real Chinchilla data (Exp. 2 / Exp. 9):

  * Huber loss (delta_H = 1e-3) on log-loss, log-sum-exp parametrisation;
  * nested break model L = E + (A N^-a + B D^-b) * max(1, C/Cb)^delta;
  * breakpoint Cb profiled over a grid of compute quantiles of the design;
  * one-sided statistic T = max over Cb of [f0 - f1(Cb)] restricted to grid points with delta_hat(Cb) > 0
    (T = 0 if no grid point has delta_hat > 0);  two-sided statistic T2 = f0 - min_Cb f1(Cb);
  * calibration by parametric bootstrap: simulate the null at the same design and noise model.

Revision 1 addition (responds to reviewer M2: size/power must be established for the procedure actually used).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from common import huber, obj_base, SCALE, OPT


def obj_break(q, lN, lD, lC, lL, lCb, delta_h=1e-3):
    a, b, e, al, be, dl = q
    h = np.maximum(0.0, lC - lCb)
    T = np.stack([a - al * lN + dl * h, b - be * lD + dl * h, np.full_like(lN, e)])
    lp = logsumexp(T, axis=0)
    w = np.exp(T - lp)
    r = lp - lL
    hp = np.clip(r, -delta_h, delta_h)
    g = np.array([(hp * w[0]).sum(), (hp * w[1]).sum(), (hp * w[2]).sum(),
                  -(hp * w[0] * lN).sum(), -(hp * w[1] * lD).sum(), (hp * (w[0] + w[1]) * h).sum()])
    return SCALE * huber(r, delta_h).sum(), SCALE * g


def fit0(lN, lD, lL, inits, delta_h=1e-3):
    return min((minimize(obj_base, np.asarray(x0, float), args=(lN, lD, lL, delta_h), method="BFGS", jac=True, options=OPT)
                for x0 in inits), key=lambda r: r.fun)


def profiled(lN, lD, lC, lL, lCb_grid, inits, delta_h=1e-3, d0s=(0.0, 0.05, -0.05)):
    """Returns dict with f0, per-Cb (f1, delta_hat), the one-sided and two-sided statistics,
    and the profiled delta_hat / Cb_hat."""
    f0 = fit0(lN, lD, lL, inits, delta_h)
    rows = []
    for lCb in lCb_grid:
        r = min((minimize(obj_break, np.r_[f0.x, d0], args=(lN, lD, lC, lL, lCb, delta_h), method="BFGS", jac=True, options=OPT)
                 for d0 in d0s), key=lambda r: r.fun)
        rows.append((float(r.fun), float(r.x[5]), float(lCb)))
    f1s = np.array([r[0] for r in rows]); dls = np.array([r[1] for r in rows])
    j = int(np.argmin(f1s))
    gain = np.maximum(f0.fun - f1s, 0.0)
    T1 = float(np.max(np.where(dls > 0, gain, 0.0)))
    T2 = float(gain.max())
    return dict(f0=float(f0.fun), x0=f0.x, rows=rows, T1=T1, T2=T2, delta_hat=float(dls[j]), lCb_hat=float(rows[j][2]),
                j_hat=j)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]
