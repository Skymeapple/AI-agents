"""Exact sparse-coding (non-negative lasso) analysis of the two-feature hierarchy model.

Data: orthonormal parent direction e_P=(1,0), child direction e_C=(0,1) (in the plane they span).
  P-only     : x = a e_P
  P and C    : x = a e_P + c e_C
  C-only     : x = c e_C          (violations of the hierarchy, rate rho relative to P&C)
Dictionary: d1 = (cos t1, sin t1), d2 = (cos t2, sin t2), unit norm.
Objective (identical to the SAE objective with optimal, non-amortised codes):
  L(D) = E_x min_{z>=0} ||x - D z||^2 + lam * sum(z)
"""
import numpy as np


def nn_lasso2(X, D, lam):
    """Exact non-negative lasso for 2 unit-norm atoms. X:(N,2), D:(2,2) rows are atoms.
    Returns codes Z (N,2) and per-sample loss (N,)."""
    N = X.shape[0]
    b = X @ D.T                      # (N,2) correlations
    xx = np.sum(X * X, 1)
    g = D[0] @ D[1]
    best = xx.copy(); Z = np.zeros((N, 2))
    # single-atom supports
    for k in range(2):
        zk = b[:, k] - lam / 2
        f = xx - zk ** 2
        ok = (zk > 0) & (f < best)
        best = np.where(ok, f, best)
        Z[ok] = 0; Z[ok, k] = zk[ok]
    # two-atom support
    det = 1 - g * g
    if det > 1e-10:
        r = b - lam / 2
        z1 = (r[:, 0] - g * r[:, 1]) / det
        z2 = (r[:, 1] - g * r[:, 0]) / det
        f = xx - (r[:, 0] * z1 + r[:, 1] * z2)
        ok = (z1 > 0) & (z2 > 0) & (f < best)
        best = np.where(ok, f, best)
        Z[ok, 0] = z1[ok]; Z[ok, 1] = z2[ok]
    return Z, best


def sample_data(n, cv_p, cv_c, q, rho, rng, p_parent=0.5):
    """Returns X (n,2) and labels: 0 none, 1 P-only, 2 P&C, 3 C-only, plus magnitudes."""
    u = rng.random(n)
    p_pc = p_parent * q
    p_p = p_parent * (1 - q)
    p_c = p_pc * rho
    lab = np.zeros(n, int)
    lab[u < p_p] = 1
    lab[(u >= p_p) & (u < p_p + p_pc)] = 2
    lab[(u >= p_p + p_pc) & (u < p_p + p_pc + p_c)] = 3
    a = np.clip(1 + cv_p * rng.standard_normal(n), 0.05, None)
    c = np.clip(1 + cv_c * rng.standard_normal(n), 0.05, None)
    X = np.zeros((n, 2))
    X[:, 0] = np.where((lab == 1) | (lab == 2), a, 0)
    X[:, 1] = np.where((lab == 2) | (lab == 3), c, 0)
    return X, lab


def dict_from_angles(t1, t2):
    return np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])


def loss_grid(X, lam, t1s, t2s):
    L = np.zeros((len(t1s), len(t2s)))
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            _, f = nn_lasso2(X, dict_from_angles(t1, t2), lam)
            L[i, j] = f.mean()
    return L


def optimum(X, lam, n1=29, n2=91, t1_range=(-0.35, 0.35)):
    """Global optimum over (t1, t2) with t1 near the parent axis (latent 1 = 'parent latent')
    and t2 in [0, pi/2]. Coarse grid, then local refinement."""
    X = X[np.any(X != 0, 1)]          # all-zero inputs have zero loss for every dictionary
    t1s = np.linspace(*t1_range, n1)
    t2s = np.linspace(0.0, np.pi / 2, n2)
    L = loss_grid(X, lam, t1s, t2s)
    i, j = np.unravel_index(np.argmin(L), L.shape)
    # refine
    d1 = t1s[1] - t1s[0]; d2 = t2s[1] - t2s[0]
    t1f = np.linspace(t1s[i] - d1, t1s[i] + d1, 21)
    t2f = np.linspace(max(0, t2s[j] - d2), min(np.pi / 2, t2s[j] + d2), 21)
    Lf = loss_grid(X, lam, t1f, t2f)
    a, b = np.unravel_index(np.argmin(Lf), Lf.shape)
    return t1f[a], t2f[b], Lf[a, b]


def readout_stats(X, lab, D, lam, tau=0.1):
    """Recall/precision of the 'parent latent' (latent 0) as a detector of the parent concept."""
    Z, _ = nn_lasso2(X, D, lam)
    fire = Z[:, 0] > tau
    parent = (lab == 1) | (lab == 2)
    rec_all = fire[parent].mean()
    rec_pc = fire[lab == 2].mean() if np.any(lab == 2) else np.nan
    rec_ponly = fire[lab == 1].mean() if np.any(lab == 1) else np.nan
    prec = parent[fire].mean() if fire.any() else np.nan
    # mean activation of parent latent on P&C inputs relative to true parent magnitude
    rel = (Z[lab == 2, 0] / X[lab == 2, 0]).mean() if np.any(lab == 2) else np.nan
    return dict(recall=rec_all, recall_pc=rec_pc, recall_ponly=rec_ponly, precision=prec, rel_act_pc=rel)


def closed_form_fixed(a, c, lam):
    """Prop. 1: per-(P&C)-input loss of the faithful and fully absorbed dictionaries, fixed magnitudes."""
    faithful = lam * (a + c) - lam ** 2 / 2
    absorbed = lam * np.hypot(a, c) - lam ** 2 / 4
    return faithful, absorbed
