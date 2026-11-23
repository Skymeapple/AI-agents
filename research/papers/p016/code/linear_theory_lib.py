"""Shared helpers (extracted from linear_theory.py)."""
import numpy as np
from scipy.linalg import eigh
d, n1, n2, k_shared = 100, 40, 40, 20

def L1_traj_flow(S1, S2, e0, rho, ts):
    H = (1-rho)*S2 + rho*S1
    lam, Q = eigh(H); lam = np.clip(lam, 0, None)
    c = Q.T @ e0
    E = (Q * 1.0) @ (np.exp(-np.outer(lam, ts)) * c[:, None])  # d x T
    return 0.5*np.einsum('it,ij,jt->t', E, S1, E)


def make_tasks(commuting=False, seed=0):
    rg_ = np.random.default_rng(seed)
    if commuting:
        # both tasks diagonal in the same basis (shared eigenbasis), different spectra
        Qb, _ = np.linalg.qr(rg_.standard_normal((d, d)))
        idx1 = np.arange(0, 60); idx2 = np.arange(30, 90)
        D1 = np.zeros(d); D2 = np.zeros(d)
        D1[idx1] = rg_.uniform(0.2, 2, len(idx1)); D2[idx2] = rg_.uniform(0.2, 2, len(idx2))
        return Qb @ np.diag(D1) @ Qb.T, Qb @ np.diag(D2) @ Qb.T
    # task inputs lie in random subspaces sharing k_shared directions, anisotropic scales
    B = np.linalg.qr(rg_.standard_normal((d, d)))[0]
    U1 = B[:, :60]; U2 = np.concatenate([B[:, 60-k_shared:60], rg_.standard_normal((d, 60-k_shared))], 1)
    U2 = np.linalg.qr(U2)[0]
    X1 = rg_.standard_normal((n1, 60)) * np.linspace(0.3, 2, 60) @ U1.T
    X2 = rg_.standard_normal((n2, 60)) * np.linspace(0.3, 2, 60) @ U2.T
    return X1.T @ X1 / n1, X2.T @ X2 / n2


def null_proj(S, tol=1e-9):
    lam, Q = eigh(S); Z = Q[:, lam < tol*lam.max()]
    return Z @ Z.T

