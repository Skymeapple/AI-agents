"""Displacement-bridging model of human-AI emotional relationships.

State   S_t >= 0 : human social capital (normalised so that the no-AI steady state is eta/delta).
Choice  a_t in [0,1] : share of discretionary social time/attention spent with the AI companion.
Flow    U = q*sqrt(a) + h(S)*sqrt(1-a)      (experienced connection; loneliness L = 1 - U)
Myopic  a*(S) = q^2/(q^2+h(S)^2),  U*(S) = sqrt(q^2+h(S)^2)
Law     S_{t+1} = (1-delta) S_t + eta (1-a_t) + b a_t + noise
D = eta - b  (net displacement per unit of AI time); rho = b/eta (bridging ratio).
"""
import numpy as np
from scipy.optimize import brentq

BASE = dict(delta=0.02, eta=0.02, h0=0.15, kappa=0.6, n=4.0)


def h(S, p=BASE):
    S = np.maximum(S, 0.0)
    return p["h0"] + (1 - p["h0"]) * S ** p["n"] / (S ** p["n"] + p["kappa"] ** p["n"])


def dh(S, p=BASE):
    S = np.maximum(S, 1e-12)
    k, n = p["kappa"], p["n"]
    return (1 - p["h0"]) * n * S ** (n - 1) * k ** n / (S ** n + k ** n) ** 2


def a_star(S, q, p=BASE):
    hh = h(S, p)
    return q ** 2 / (q ** 2 + hh ** 2)


def U_star(S, q, p=BASE):
    return np.sqrt(q ** 2 + h(S, p) ** 2)


def G(S, q, b, p=BASE):
    """Drift: S_{t+1}-S_t (deterministic)."""
    a = a_star(S, q, p)
    return p["eta"] * (1 - a) + b * a - p["delta"] * S


def steady_states(q, b, p=BASE, Smax=None, ngrid=4000):
    """Return list of (S*, stable_bool)."""
    if Smax is None:
        Smax = 3 * max(p["eta"], b, 1e-9) / p["delta"] + 0.5
    Sg = np.linspace(0.0, Smax, ngrid)
    g = G(Sg, q, b, p)
    out = []
    for i in range(len(Sg) - 1):
        if g[i] == 0:
            out.append(Sg[i])
        elif g[i] * g[i + 1] < 0:
            out.append(brentq(G, Sg[i], Sg[i + 1], args=(q, b, p)))
    res = []
    for s in out:
        eps = 1e-6
        slope = (G(s + eps, q, b, p) - G(max(s - eps, 0), q, b, p)) / (2 * eps)
        res.append((float(s), bool(slope < 0)))
    return res


def simulate(S0, q, b, T, p=BASE, eta_i=None, sigma=0.0, rng=None, avail=None):
    """Vectorised simulation for arrays of users. Returns S (T+1,N), a (T,N), U (T,N)."""
    S0 = np.atleast_1d(np.asarray(S0, float))
    N = S0.size
    q = np.broadcast_to(np.asarray(q, float), (N,))
    b = np.broadcast_to(np.asarray(b, float), (N,))
    eta = np.full(N, p["eta"]) if eta_i is None else np.asarray(eta_i, float)
    S = np.empty((T + 1, N)); S[0] = S0
    A = np.empty((T, N)); Uo = np.empty((T, N))
    for t in range(T):
        a = a_star(S[t], q, p)
        if avail is not None:
            a = a * avail[t]
        hh = h(S[t], p)
        A[t] = a
        Uo[t] = q * np.sqrt(a) + hh * np.sqrt(1 - a)
        noise = sigma * rng.standard_normal(N) if sigma > 0 else 0.0
        S[t + 1] = np.maximum(0.0, (1 - p["delta"]) * S[t] + eta * (1 - a) + b * a + noise)
    return S, A, Uo


def elasticity(S, p=BASE):
    return S * dh(S, p) / h(S, p)
