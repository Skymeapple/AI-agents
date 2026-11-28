"""Closed forms for Proposition 1 (AUROC ceiling) and Proposition 2 (familiarity shortcut)."""
import numpy as np
from scipy.stats import norm

def auroc_ceiling(alpha, delta):
    """Max AUROC of any q-only detector when a fraction alpha of 'known' items are aleatoric
    and their q-distribution is within TV delta of the 'unknown' items' q-distribution."""
    return 1 - alpha * (1 - delta) ** 2 / 2

def lda_rates(a_vec, b_vec, pi):
    """Whitened Gaussian model z = F a + K b + N(0,I); training P(K=1)=1/2, P(F=K)=pi.
    Returns (false-knowing rate on F=1,K=0 cell, missed-knowing rate on F=0,K=1 cell,
    in-distribution accuracy) of the population LDA rule."""
    a = np.asarray(a_vec, float); b = np.asarray(b_vec, float)
    mu1 = pi * a + b; mu0 = (1 - pi) * a
    S = np.eye(len(a)) + pi * (1 - pi) * np.outer(a, a)
    w = np.linalg.solve(S, mu1 - mu0); c = w @ (mu1 + mu0) / 2; s = np.linalg.norm(w)
    fk = norm.cdf((w @ a - c) / s)
    mk = 1 - norm.cdf((w @ b - c) / s)
    # in-distribution accuracy (mixture over the 4 cells with training weights)
    cells = {(1, 1): pi / 2, (0, 0): pi / 2, (1, 0): (1 - pi) / 2, (0, 1): (1 - pi) / 2}
    acc = 0
    for (F, K), wt in cells.items():
        m = F * a + K * b; pk = norm.cdf((w @ m - c) / s)
        acc += wt * (pk if K == 1 else 1 - pk)
    return fk, mk, acc

def fk_closed_form(a, b, pi, sigma=1.0):
    """Orthogonal case, scalar saliences a,b (Eq. in Prop. 2)."""
    kappa = (2 * pi - 1) * sigma**2 / (sigma**2 + a**2 * pi * (1 - pi))
    return norm.cdf((a**2 * kappa - b**2) / (2 * sigma * np.sqrt(a**2 * kappa**2 + b**2)))

def lda_rates_general(a_vec, b_vec, pi1, pi0, p1):
    """Prop. 2 without the balanced-class assumption (revision R1, reviewer m2).
    Training law: P(K=1)=p1, P(F=1|K=1)=pi1, P(F=0|K=0)=pi0; z = F a + K b + N(0,I) (whitened).
    Population LDA (pooled within-class covariance, Bayes threshold incl. log prior ratio).
    Returns (false-knowing rate on F=1,K=0; missed-knowing rate on F=0,K=1)."""
    a = np.asarray(a_vec, float); b = np.asarray(b_vec, float); p0 = 1 - p1
    mu1 = pi1 * a + b; mu0 = (1 - pi0) * a
    S = np.eye(len(a)) + (p1 * pi1 * (1 - pi1) + p0 * pi0 * (1 - pi0)) * np.outer(a, a)
    w = np.linalg.solve(S, mu1 - mu0)
    c = w @ (mu1 + mu0) / 2 + np.log(p0 / p1)
    s = np.sqrt(w @ w)   # noise is N(0, I) in whitened space
    return float(norm.cdf((w @ a - c) / s)), float(1 - norm.cdf((w @ b - c) / s))
