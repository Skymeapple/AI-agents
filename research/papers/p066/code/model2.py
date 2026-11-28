"""
Two-state (skill s, persistence g) model of adolescent learning with AI help.
Shared by calibrate.py and project.py.

Per period (one practice session), learner chooses offload share a in {0,.1,...,1}:
  assisted practice performance  P = (1-a) g s + a q
  unassisted performance          E = g s      (prob. persist x prob. correct)
  choice rule (one-step lookahead, present-biased weight on next unassisted test):
     a* = argmax_a  w P(a) - c (1-a)(1 - theta s) + beta * Omega * g'(a) s'(a)
  skill        s' = s + eta*phi*[(1-a) g + sigma a](eps + s)(1 - s) - lam s
  persistence  g' = g + kappa*[(1-a) s (1-g) - rho a (1-sigma) g] + mu (gbar - g)
"""
import numpy as np

A = np.linspace(0, 1, 11)
THETA, EPS, W = 0.5, 0.05, 1.0


def transitions(s, g, a, P):
    """s,g arrays broadcastable with a. P: dict of arrays/scalars (broadcastable)."""
    sig = P["sigma"]
    L = (1 - a) * g + sig * a
    s2 = s + P["eta"] * P.get("phi", 1.0) * L * (EPS + s) * (1 - s) - P["lam"] * s
    g2 = g + P["kappa"] * ((1 - a) * s * (1 - g) - P["rho"] * a * (1 - sig) * g) \
        + P.get("mu", 0.0) * (P.get("gbar", g) - g)
    return np.clip(s2, 0, 1), np.clip(g2, 0, 1)


def choose(s, g, P, forced=None):
    """Returns chosen a (same shape as s). s,g shape (..., n). Params shape (..., 1)."""
    if forced is not None:
        return np.full_like(s, forced)
    a = A.reshape((1,) * s.ndim + (-1,))
    S, Gg = s[..., None], g[..., None]
    Pe = {k: (v[..., None] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    s2, g2 = transitions(S, Gg, a, Pe)
    U = W * ((1 - a) * Gg * S + a * Pe["q"]) - Pe["c"] * (1 - a) * (1 - THETA * S) \
        + Pe["beta"] * Pe["Omega"] * g2 * s2
    return A[np.argmax(U, axis=-1)]


def period(s, g, P, forced=None):
    a = choose(s, g, P, forced)
    prac = (1 - a) * g * s + a * P["q"]
    s2, g2 = transitions(s, g, a, P)
    return a, prac, s2, g2
