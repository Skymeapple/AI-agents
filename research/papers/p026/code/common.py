"""Shared closed-form expressions for the identity half-life paper (p026)."""
import numpy as np

def stationary_fidelity(p, pi_x, h=0.0):
    """Fixed point c* of c_{t+1} = h + (1-h)[(1-p)c_t + p*pi_x]."""
    lam = (1 - h) * (1 - p)
    return (h + (1 - h) * p * pi_x) / (1 - lam)

def fidelity_curve(t, p, pi_x, h=0.0, c0=1.0):
    lam = (1 - h) * (1 - p)
    cs = stationary_fidelity(p, pi_x, h)
    return cs + (c0 - cs) * lam ** np.asarray(t, dtype=float)

def half_life(lam):
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(lam >= 1, np.inf, np.log(2) / -np.log(np.clip(lam, 1e-300, 1)))
