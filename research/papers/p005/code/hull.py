"""Convex-hull utilities: projection onto a hull (NNLS with sum-to-one penalty row),
exact LP membership, and the linear-readout interpolation certificate."""
import numpy as np
from scipy.optimize import nnls, linprog


def hull_project(phi, P, M=None):
    """Project vector phi onto conv(rows of P). Returns (distance, lambda).
    Solves min ||P^T l - phi|| s.t. l>=0, sum l = 1 via NNLS with a heavily weighted
    sum-to-one row; lambda is renormalised and the distance recomputed exactly."""
    if M is None:
        M = 1e3 * (1.0 + np.abs(P).max())
    A = np.vstack([P.T, M * np.ones((1, P.shape[0]))])
    b = np.concatenate([phi, [M]])
    lam, _ = nnls(A, b, maxiter=50 * P.shape[0])
    s = lam.sum()
    lam = lam / s if s > 0 else np.full(P.shape[0], 1.0 / P.shape[0])
    return float(np.linalg.norm(P.T @ lam - phi)), lam


def in_hull_lp(phi, P):
    """Exact LP feasibility: is phi in conv(rows of P)?"""
    n = P.shape[0]
    A_eq = np.vstack([P.T, np.ones((1, n))])
    b_eq = np.concatenate([phi, [1.0]])
    res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.status == 0


def readout_certificate(phi, y, Py, W, b):
    """Proposition 2/3 certificate for a linear readout z = W^T h + b.
    Py: training representations with label y (rows). Returns (certified, delta, cert_margin).
    certified => the network's logit margin for y at phi is provably positive, and the
    decision is explained by a convex combination of same-label training representations."""
    delta, lam = hull_project(phi, Py)
    h = Py.T @ lam
    zh = h @ W + b
    wy = W[:, y]
    diff = W - wy[:, None]                 # columns w_c - w_y
    lip = np.linalg.norm(diff, axis=0)     # ||w_y - w_c||
    marg = zh[y] - zh - lip * delta
    marg[y] = np.inf
    m = float(marg.min())
    return m > 0, delta, m
