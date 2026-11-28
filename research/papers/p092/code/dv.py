"""Delegation-Value (DV) metric library.

Model (see paper Sec. 3). For task i with human replacement cost H_i (USD),
AI success probability p_i, verification cost V_i, inference cost K_i,
verifier sensitivity s and damage multiplier delta for undetected failures,
the expected net saving of one delegated attempt is

    S_i = H_i * [p_i + (1 - p_i)(1 - s)(1 - delta)] - V_i - K_i.

A delegation policy chooses a set D of tasks to delegate; realised value is
sum_{i in D} omega_i S_i.  We normalise by sum_i omega_i H_i (the labour cost
of the whole portfolio), giving the Delegation-Value Rate (DVR).
"""
import numpy as np
from scipy.optimize import minimize


def net_saving(p, H, V, K, s=1.0, delta=1.0):
    return H * (p + (1 - p) * (1 - s) * (1 - delta)) - V - K


def fit_logistic(x, y, w=None, l2=1e-3):
    """Weighted logistic regression y ~ sigmoid(a + b x). Returns (a, b)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    w = w / w.sum()

    def nll(theta):
        a, b = theta
        z = a + b * x
        # log(1+exp(z)) stable
        lse = np.logaddexp(0, z)
        return np.sum(w * (lse - y * z)) + l2 * (a * a + b * b)

    def grad(theta):
        a, b = theta
        z = a + b * x
        pr = 1 / (1 + np.exp(-z))
        g = w * (pr - y)
        return np.array([g.sum() + 2 * l2 * a, (g * x).sum() + 2 * l2 * b])

    res = minimize(nll, np.array([0.0, -0.5]), jac=grad, method="L-BFGS-B")
    return res.x


def horizon(a, b, level=0.5):
    """Human-minutes at which fitted success equals `level` (x = log2 minutes)."""
    if b >= 0:
        return np.inf
    logit = np.log(level / (1 - level))
    return 2 ** ((logit - a) / b)


def stop_loss(p, weight, rhos):
    """E_mu[(P - rho)_+] for the weight-normalised distribution of p."""
    p = np.asarray(p); weight = np.asarray(weight) / np.sum(weight)
    return np.array([np.sum(weight * np.clip(p - r, 0, None)) for r in rhos])
