"""Exploit-mixture model of reward hacking: closed forms and samplers.

Base policy pi0 produces outputs of two latent types:
  Z=0 (genuine, prob 1-p):  gold g = g0,             proxy r_k = g0 + e_k
  Z=1 (exploit, prob p):    gold g = -delta,         proxy r_k = B_k*mu + g0 + e_k
with g0 ~ N(0, sg2), e ~ N(0, se2) with corr rho between proxies, B_1 = 1 (the
optimised proxy is always fooled by its own exploits) and B_k ~ Bernoulli(q)
for held-out proxies k >= 2 (q = "shared vulnerability").
Optimisation = exponential tilting pi_lam(x) ∝ pi0(x) exp(lam * r_1(x)),
(the optimum of KL-regularised reward maximisation with beta = 1/lam), or best-of-n.
"""
import numpy as np
from scipy.special import expit, logit


class Params:
    def __init__(self, p=0.002, sg2=1.0, se2=1.0, rho=0.5, mu=6.0, delta=2.0, q=0.5):
        self.p, self.sg2, self.se2, self.rho = p, sg2, se2, rho
        self.mu, self.delta, self.q = mu, delta, q

    @property
    def v(self):
        return self.sg2 + self.se2

    @property
    def c12(self):  # covariance of r1 and r2 within a component
        return self.sg2 + self.rho * self.se2


def closed_form(P, lam):
    lam = np.asarray(lam, dtype=float)
    pi = expit(logit(P.p) + P.mu * lam)
    gold = (1 - pi) * P.sg2 * lam - pi * P.delta
    r1 = P.v * lam + pi * P.mu
    r2 = P.c12 * lam + pi * P.q * P.mu
    gap = (1 - P.rho) * P.se2 * lam + pi * (1 - P.q) * P.mu
    logM = P.v * lam ** 2 / 2 + np.log1p(-P.p + P.p * np.exp(P.mu * lam))
    kl = lam * r1 - logM
    # gold in the "twin" world W' where *shared* exploits are genuinely excellent
    # (gold = mu + g0); proxy law identical to W.
    gold_twin = gold + pi * P.q * (P.mu + P.sg2 * lam + P.delta)
    return dict(pi=pi, gold=gold, r1=r1, r2=r2, gap=gap, kl=kl, gold_twin=gold_twin)


def sample_base(P, n, rng, K=2):
    """Sample n outputs from pi0. Returns dict with Z, gold, R (n x K proxies)."""
    Z = rng.random(n) < P.p
    g0 = rng.normal(0, np.sqrt(P.sg2), n)
    # correlated proxy errors: e_k = sqrt(rho)*c + sqrt(1-rho)*u_k
    c = rng.normal(0, 1, n)
    U = rng.normal(0, 1, (n, K))
    E = np.sqrt(P.se2) * (np.sqrt(P.rho) * c[:, None] + np.sqrt(1 - P.rho) * U)
    B = np.zeros((n, K))
    B[:, 0] = 1.0
    if K > 1:
        B[:, 1:] = rng.random((n, K - 1)) < P.q
    R = g0[:, None] + E + (Z[:, None] * B) * P.mu
    gold = np.where(Z, -P.delta, g0)
    return dict(Z=Z, gold=gold, R=R, B=B)


def sample_tilted(P, n, lam, rng, K=2):
    """Exact sampling from pi_lam using the closed-form component structure."""
    pi = float(expit(logit(P.p) + P.mu * lam))
    Z = rng.random(n) < pi
    # within each component the tilt shifts (g0, e) along Cov(., r1) * lam
    g0 = rng.normal(P.sg2 * lam, np.sqrt(P.sg2), n)
    c = rng.normal(0, 1, n)
    U = rng.normal(0, 1, (n, K))
    E = np.sqrt(P.se2) * (np.sqrt(P.rho) * c[:, None] + np.sqrt(1 - P.rho) * U)
    # tilt of errors: Cov(e_k, r1) = se2*rho (k>=2), se2 (k=1)
    shift = np.full(K, P.rho * P.se2 * lam)
    shift[0] = P.se2 * lam
    E = E + shift[None, :]
    B = np.zeros((n, K))
    B[:, 0] = 1.0
    if K > 1:
        B[:, 1:] = rng.random((n, K - 1)) < P.q
    R = g0[:, None] + E + (Z[:, None] * B) * P.mu
    gold = np.where(Z, -P.delta, g0)
    return dict(Z=Z, gold=gold, R=R, B=B)


def gold_peak(P, lam_grid=None):
    if lam_grid is None:
        lam_grid = np.linspace(0, 3, 30001)
    cf = closed_form(P, lam_grid)
    i = int(np.argmax(cf["gold"]))
    return lam_grid[i], cf["gold"][i], cf["pi"][i]
