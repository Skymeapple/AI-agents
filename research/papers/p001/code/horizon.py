"""Audit-horizon utilities (revision 2).

Notation: theta = (a, b, e, alpha, beta) with A=e^a, B=e^b, E=e^e (log-sum-exp parametrisation of
L = E + A N^-alpha + B D^-beta).  The compute-optimal envelope is L*(C) = E + R(C), R(C) = G C^-gamma.

Core identities (paper, Sec. 3):
  * compute-equivalence: if an observed run at compute C has loss L*(C/P) (a compute penalty P, from any
    mechanism), then ln L_obs - ln L*(C) = ln(1 + rho(C) (P^gamma - 1)), rho = R/L*.
  * resolution: a one-sided forecast-residual test with total log-scale SD s(C) reaches power 1-beta at
    P_min(C) = [1 + (exp((z_a+z_b) s(C)) - 1) / rho(C)]^(1/gamma).
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def unpack(th):
    a, b, e, al, be = th
    return np.exp(a), np.exp(b), np.exp(e), al, be


def gam(th):
    return th[3] * th[4] / (th[3] + th[4])


def frontier_parts(th, C):
    A, B, E, al, be = unpack(th)
    G = (al * A / (be * B)) ** (1.0 / (al + be))
    N = G * (np.asarray(C) / 6.0) ** (be / (al + be))
    D = np.asarray(C) / 6.0 / N
    R = A * N ** -al + B * D ** -be
    return E, R, N, D


def Lstar(th, C):
    E, R, _, _ = frontier_parts(th, C)
    return E + R


def rho(th, C):
    E, R, _, _ = frontier_parts(th, C)
    return R / (E + R)


def log_pred(th, lN, lD):
    a, b, e, al, be = th
    return logsumexp(np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)]), axis=0)


def jac_log(th, lN, lD):
    """d ln L / d theta for each run (n x 5)."""
    a, b, e, al, be = th
    T = np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)])
    w = np.exp(T - logsumexp(T, axis=0))
    return np.stack([w[0], w[1], w[2], -w[0] * lN, -w[1] * lD], axis=1)


def grad_logLstar(th, C, h=1e-6):
    th = np.asarray(th, float)
    g = np.zeros(5)
    for k in range(5):
        tp = th.copy(); tm = th.copy(); tp[k] += h; tm[k] -= h
        g[k] = (np.log(Lstar(tp, C)) - np.log(Lstar(tm, C))) / (2 * h)
    return g


def cov_theta(th, lN, lD, groups, sigma_w, sigma_b, estimator="ols"):
    """Covariance of the (least-squares) estimator of theta on log-loss for a design with iid noise sigma_w
    and a shared offset sigma_b per group (budget).  'ols' gives the sandwich for ordinary least squares
    (what practitioners fit); 'gls' the efficient (Cramer-Rao) covariance under Gaussian noise."""
    J = jac_log(th, lN, lD)
    n = len(lN)
    Z = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    S = sigma_w ** 2 * np.eye(n) + sigma_b ** 2 * Z @ Z.T
    if estimator == "gls":
        return np.linalg.inv(J.T @ np.linalg.solve(S, J))
    Hi = np.linalg.inv(J.T @ J)
    return Hi @ (J.T @ S @ J) @ Hi


def forecast_sd(th, cov, C):
    """SD of ln L*_hat(C) by the delta method."""
    return np.array([np.sqrt(grad_logLstar(th, c) @ cov @ grad_logLstar(th, c)) for c in np.atleast_1d(C)])


def p_min(th, C, s, alpha=0.05, power=0.5):
    """Smallest compute penalty detected with the given power by a one-sided level-alpha residual test."""
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return (1.0 + np.expm1(z * np.asarray(s)) / rho(th, C)) ** (1.0 / gam(th))


def residual_power(th, C, s, P, alpha=0.05):
    """Power of the one-sided residual test against a compute penalty P (Gaussian approximation)."""
    sig = np.log1p(rho(th, C) * (P ** gam(th) - 1.0))
    return norm.cdf(sig / np.asarray(s) - norm.ppf(1 - alpha))


def horizon(th, Cgrid, s, alpha=0.05, power=0.5):
    """Audit horizon C_H: first compute at which rho(C) <= exp(z s(C)) - 1, i.e. P_min >= 2^(1/gamma)."""
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    r = rho(th, Cgrid) - np.expm1(z * np.asarray(s))
    idx = np.where(r <= 0)[0]
    if len(idx) == 0:
        return np.inf
    i = idx[0]
    if i == 0:
        return float(Cgrid[0])
    # log-linear interpolation of the crossing
    x0, x1 = np.log(Cgrid[i - 1]), np.log(Cgrid[i])
    return float(np.exp(x0 + (x1 - x0) * r[i - 1] / (r[i - 1] - r[i])))


def iso_design(Clo, Chi, K, M, rmin=2.0, rmax=200.0):
    """K iso-FLOP budgets log-spaced in [Clo, Chi], M sizes per budget with tokens/param log-uniform."""
    lC = np.repeat(np.linspace(np.log(Clo), np.log(Chi), K), M)
    ratio = np.tile(np.exp(np.linspace(np.log(rmin), np.log(rmax), M)), K)
    lN = 0.5 * (lC - np.log(6) - np.log(ratio)); lD = lC - np.log(6) - lN
    groups = np.repeat(np.arange(K), M)
    return lN, lD, lC, groups
