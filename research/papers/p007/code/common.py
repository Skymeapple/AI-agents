"""Shared-core linear model of a unified multimodal network.

Modality m: x ~ N(0, I_d) (tokens embedded in a common d-dim space),
y = W_m x + eps, eps ~ N(0, sigma^2 I_q), n samples per modality.
Unified model: y_hat = V_m U^T x with one shared core U (d x rho), rho <= r.
Specialist:    y_hat = V_m U_m^T x, its own core of rank <= r per modality.
Ground truth: W_m = s * [A_m B_m] [S_m P_m]^T, S_m = shared subspace (possibly
rotated by angle theta away from a common S), P_m private subspace.
"""
import numpy as np


def orth(a):
    q, _ = np.linalg.qr(a)
    return q


def make_task(d, q, ks, kp, M, theta=0.0, s=1.0, rng=None):
    """Return list of W_m (q x d) and the population Gram spectrum info."""
    rng = np.random.default_rng(rng)
    need = ks + M * kp + (M * ks if theta != 0 else 0)  # S, P_m, misalignment dirs E_m
    assert need <= d, "d too small"
    basis = orth(rng.standard_normal((d, need)))
    S = basis[:, :ks]
    Ws = []
    for m in range(M):
        P = basis[:, ks + m * kp: ks + (m + 1) * kp]
        if theta != 0:
            E = basis[:, ks + M * kp + m * ks: ks + M * kp + (m + 1) * ks]
            Sm = np.cos(theta) * S + np.sin(theta) * E
        else:
            Sm = S
        AB = orth(rng.standard_normal((q, ks + kp)))  # orthonormal head columns
        Wm = s * AB @ np.concatenate([Sm, P], axis=1).T
        Ws.append(Wm)
    return Ws


# ---------------------------------------------------------------- theory
def rrr_risk_curve(eigs, sigma2, n, d, qeff, rmax):
    """Oracle-rank risk (summed) of rank-<=rmax reduced-rank fit.

    eigs: population eigenvalues of sum_m W_m^T W_m (descending)
    qeff: total output dimension (q for specialist, M*q for unified)
    risk(rho) = sum_{i>rho} eigs_i + sigma2 * rho (d + qeff - rho) / n
    (classical high-SNR asymptotic degrees of freedom of reduced-rank regression)
    """
    eigs = np.sort(np.asarray(eigs))[::-1]
    tot = eigs.sum()
    best = tot
    cum = 0.0
    for rho in range(1, rmax + 1):
        cum += eigs[rho - 1] if rho - 1 < len(eigs) else 0.0
        val = tot - cum + sigma2 * rho * (d + qeff - rho) / n
        best = min(best, val)
    return best


def theory_losses(Ws, sigma2, n, r):
    """Per-modality average excess risk predicted for unified and specialists."""
    M = len(Ws)
    q, d = Ws[0].shape
    G = sum(W.T @ W for W in Ws)
    eu = np.clip(np.linalg.eigvalsh(G), 0, None)
    Lu = rrr_risk_curve(eu, sigma2, n, d, M * q, r) / M
    Ls = 0.0
    for W in Ws:
        es = np.clip(np.linalg.eigvalsh(W.T @ W), 0, None)
        Ls += rrr_risk_curve(es, sigma2, n, d, q, r)
    return Lu, Ls / M


# ---------------------------------------------------------------- simulation
def sim_losses(Ws, sigma2, n, r, rng):
    """Monte Carlo: orthogonal design (X^T X = n I) so OLS = W + sigma/sqrt(n) Z.

    Excess risk under x ~ N(0, I) equals squared Frobenius error.
    Rank is chosen by oracle (min true risk over rho <= r) for both models.
    """
    M = len(Ws)
    q, d = Ws[0].shape
    sd = np.sqrt(sigma2 / n)
    What = [W + sd * rng.standard_normal(W.shape) for W in Ws]
    # unified
    G = sum(Wh.T @ Wh for Wh in What)
    _, vec = np.linalg.eigh(G)
    vec = vec[:, ::-1]
    best_u = np.inf
    for rho in range(0, r + 1):
        U = vec[:, :rho]
        err = sum(np.sum((Wh @ U @ U.T - W) ** 2) for Wh, W in zip(What, Ws))
        best_u = min(best_u, err)
    # specialists
    tot_s = 0.0
    for Wh, W in zip(What, Ws):
        u_, sv, vt = np.linalg.svd(Wh, full_matrices=False)
        best = np.inf
        for rho in range(0, min(r, len(sv)) + 1):
            approx = (u_[:, :rho] * sv[:rho]) @ vt[:rho]
            best = min(best, np.sum((approx - W) ** 2))
        tot_s += best
    return best_u / M, tot_s / M


# ---------------------------------------------------------------- closed forms
def population_interaction(r, ks, kp, M, s2=1.0):
    """Prop. 1: population per-modality Delta = L_unified - L_specialist."""
    k = ks + kp
    if r <= ks:
        return 0.0
    if r <= k:
        return s2 * (r - ks) * (1 - 1 / M)
    return s2 * max(ks + M * kp - r, 0) / M
