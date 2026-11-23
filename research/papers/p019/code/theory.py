"""Analytic model of AI hypothesis verification with shared systematic bias.

Model (per hypothesis h): true effect theta in {0, mu}. A measurement in modality m with
n units returns  X = theta + b_m + e,  b_m ~ N(0, sb^2) (systematic bias, fixed per
hypothesis x modality), e ~ N(0, s^2/n) (sampling noise, fresh per dataset).
A one-sided level-alpha test accepts iff X > z_alpha * s / sqrt(n).

Pipelines (generator always screens on modality A first):
  self      : verification re-uses the generator's data -> acceptance = generator pass
  replicate : K fresh datasets from the SAME modality A (shared b_A)
  orthogonal: K fresh datasets from K modalities with independent biases
  intervene : K randomized-intervention datasets (bias = 0)
"""
import numpy as np
from scipy.stats import norm

# dense quadrature over standardized bias (integrand becomes step-like as n grows,
# so Gauss-Hermite is inaccurate; a fine Riemann grid on [-9, 9] is used instead)
GH_X = np.linspace(-9, 9, 200_001)
GH_W = norm.pdf(GH_X); GH_W = GH_W / GH_W.sum()


def pass_prob_given_b(theta, b, n, s=1.0, alpha=0.05):
    c = norm.isf(alpha) * s / np.sqrt(n)
    return norm.sf((c - theta - b) / (s / np.sqrt(n)))


def accept_prob(theta, pipeline, K, n, sb, s=1.0, alpha=0.05):
    """Probability that a hypothesis with true effect theta is accepted by generator + K verifiers."""
    b = sb * GH_X
    g = pass_prob_given_b(theta, b, n, s, alpha)  # generator pass prob given b_A
    if pipeline == "self":
        return float(np.sum(GH_W * g))
    if pipeline == "replicate":
        return float(np.sum(GH_W * g ** (K + 1)))
    if pipeline == "orthogonal":
        single = float(np.sum(GH_W * g))
        return float(np.sum(GH_W * g)) * single ** K
    if pipeline == "intervene":
        v = float(pass_prob_given_b(theta, 0.0, n, s, alpha))
        return float(np.sum(GH_W * g)) * v ** K
    raise ValueError(pipeline)


def fdr(pi1, mu, pipeline, K, n, sb, s=1.0, alpha=0.05):
    a0 = accept_prob(0.0, pipeline, K, n, sb, s, alpha)
    a1 = accept_prob(mu, pipeline, K, n, sb, s, alpha)
    return (1 - pi1) * a0 / ((1 - pi1) * a0 + pi1 * a1), a1


def fdr_limit(pi1, mu, pipeline, K, sb, alpha=0.05):
    """n -> infinity limits (Theorem 1)."""
    p = norm.cdf(mu / sb)
    if pipeline in ("self", "replicate"):
        a0, a1 = 0.5, p
    elif pipeline == "orthogonal":
        a0, a1 = 0.5 ** (K + 1), p ** (K + 1)
    elif pipeline == "intervene":
        a0, a1 = 0.5 * alpha ** K, p
    pi0 = 1 - pi1
    return pi0 * a0 / (pi0 * a0 + pi1 * a1)


def k_star(prior_odds, q, lam):
    """Minimum number of independent-bias modalities so that FDR <= q (Theorem 2).
    prior_odds = pi1/pi0 among candidates, lam = per-modality likelihood ratio p/f > 1."""
    need = np.log((1 - q) / (q * prior_odds))
    return np.maximum(0, np.ceil(need / np.log(lam)))


def simulate(pi1, mu, pipeline, K, n, sb, N=200_000, s=1.0, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    truth = rng.random(N) < pi1
    theta = np.where(truth, mu, 0.0)
    c = norm.isf(alpha) * s / np.sqrt(n)
    se = s / np.sqrt(n)
    bA = rng.normal(0, sb, N)
    acc = theta + bA + rng.normal(0, se, N) > c
    for _ in range(K):
        if pipeline == "self":
            break
        if pipeline == "replicate":
            b = bA
        elif pipeline == "orthogonal":
            b = rng.normal(0, sb, N)
        else:
            b = 0.0
        acc &= theta + b + rng.normal(0, se, N) > c
    nacc = acc.sum()
    return (acc & ~truth).sum() / max(nacc, 1), (acc & truth).sum() / truth.sum()
