"""Core model: per-request refusal rights against retrying users.

Latent harm score of a user's underlying intent: a ~ N(mu_type, tau^2) (or other law).
Each attempt (rephrasing / resample) yields signal s = a + eps, eps iid with scale sigma.
The AI refuses an attempt iff s > t. Per-attempt refusal prob for intent a: r(a) = P(eps > t - a).

Harmful users retry up to K times (stop at first acceptance).
Benign users retry after each refusal with prob beta (patience), up to K attempts.

Policies:
  memoryless(t)        : each attempt judged independently.
  strike(t, m)         : after m refusals the account is locked (all later attempts refused);
                         benign accounts that get locked additionally lose L (future value).
  escalate(t, m, qH, qB, c): after m refusals the case goes to a human reviewer who blocks a
                         harmful user w.p. qH, a benign user w.p. qB, at cost c per escalation.

Welfare (per user): W = (1-pi) v P(benign served) - pi h P(harmful served) - costs.
Value of refusal right: V = W(policy) - W(always comply) = pi h P_H(blocked) - (1-pi) v P_B(blocked) - costs.
"""
import numpy as np
from scipy import special, stats

# ---------------------------------------------------------------- noise / latent laws
def log_refuse_prob(a, t, sigma, noise="gauss"):
    """log r(a) = log P(eps > t-a); and log(1-r)."""
    z = (t - a) / sigma
    if noise == "gauss":
        return special.log_ndtr(-z), special.log_ndtr(z)
    if noise == "logistic":  # logistic with scale s chosen so that sd = sigma
        s = sigma * np.sqrt(3) / np.pi
        z = (t - a) / s
        # P(eps > x) = 1/(1+e^{x/s})
        return -np.logaddexp(0, z), -np.logaddexp(0, -z)
    raise ValueError(noise)


def latent_grid(mu, tau, latent="gauss", n=20001, span=14.0):
    """Quadrature nodes and weights for the latent intent score."""
    if tau == 0:
        return np.array([mu]), np.array([1.0])
    if latent == "gauss":
        x = np.linspace(mu - span * tau, mu + span * tau, n)
        w = stats.norm.pdf(x, mu, tau)
    elif latent == "laplace":  # Laplace with sd tau
        b = tau / np.sqrt(2)
        x = np.linspace(mu - 40 * b, mu + 40 * b, n)
        w = stats.laplace.pdf(x, mu, b)
    else:
        raise ValueError(latent)
    w = w / w.sum()
    return x, w


# ---------------------------------------------------------------- blocking probabilities
def harmful_blocked(t, K, mu_H, tau, sigma, noise="gauss", latent="gauss", grid=None):
    """P(harmful user never served within K attempts) = E_a[r(a)^K]."""
    x, w = grid if grid is not None else latent_grid(mu_H, tau, latent)
    lr, _ = log_refuse_prob(x, t, sigma, noise)
    return float(np.sum(w * np.exp(K * lr)))


def benign_blocked(t, K, beta, mu_B, tau, sigma, noise="gauss", latent="gauss", grid=None,
                   return_locked=False):
    """P(benign user never served) with patience beta and at most K attempts.
    = sum_{j=1}^{K-1} r^j beta^{j-1} (1-beta) + r^K beta^{K-1}.
    If return_locked, also returns P(refused on all K attempts) = r^K beta^{K-1}."""
    x, w = grid if grid is not None else latent_grid(mu_B, tau, latent)
    lr, _ = log_refuse_prob(x, t, sigma, noise)
    r = np.exp(lr)
    br = beta * r
    if K == 1:
        geo = np.zeros_like(r)
    else:
        # sum_{j=1}^{K-1} r^j beta^{j-1} = r * sum_{i=0}^{K-2} (br)^i
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(1 - br) > 1e-12, (1 - br ** (K - 1)) / (1 - br), K - 1.0)
        geo = r * s * (1 - beta)
    allref = r ** K * beta ** (K - 1)
    pb = float(np.sum(w * (geo + allref)))
    if return_locked:
        return pb, float(np.sum(w * allref))
    return pb


# ---------------------------------------------------------------- welfare
class Params:
    def __init__(self, pi=0.01, v=1.0, h=100.0, d=2.0, rho=0.5, beta=0.5,
                 noise="gauss", latent="gauss", L=0.0):
        self.pi, self.v, self.h, self.d, self.rho, self.beta = pi, v, h, d, rho, beta
        self.noise, self.latent, self.L = noise, latent, L
        # total single-shot signal variance normalised to 1: tau^2 + sigma^2 = 1
        self.tau = np.sqrt(rho)
        self.sigma = np.sqrt(1 - rho)
        self.mu_B, self.mu_H = 0.0, d
        self.gH = latent_grid(self.mu_H, self.tau, latent)
        self.gB = latent_grid(self.mu_B, self.tau, latent)


def value_memoryless(p, t, K):
    PH = harmful_blocked(t, K, p.mu_H, p.tau, p.sigma, p.noise, p.latent, p.gH)
    PB = benign_blocked(t, K, p.beta, p.mu_B, p.tau, p.sigma, p.noise, p.latent, p.gB)
    return p.pi * p.h * PH - (1 - p.pi) * p.v * PB, PH, PB


def value_strike(p, t, m, K):
    """m-strike lockout against a K-budget adversary (K >= m => effective budget m)."""
    k = min(m, K)
    PH = harmful_blocked(t, k, p.mu_H, p.tau, p.sigma, p.noise, p.latent, p.gH)
    PB, Plock = benign_blocked(t, k, p.beta, p.mu_B, p.tau, p.sigma, p.noise, p.latent, p.gB,
                               return_locked=True)
    if K <= m:
        Plock = 0.0  # never reaches the lock
    return p.pi * p.h * PH - (1 - p.pi) * (p.v * PB + p.L * Plock), PH, PB


def value_escalate(p, t, m, K, qH=0.9, qB=0.05, c=0.5):
    """After m refusals escalate to a human reviewer (cost c, blocks harmful w.p. qH,
    benign w.p. qB). Harmful users that reach review stop retrying (review is final)."""
    k = min(m, K)
    PH_reach = harmful_blocked(t, k, p.mu_H, p.tau, p.sigma, p.noise, p.latent, p.gH)
    PBq, Pb_reach = benign_blocked(t, k, p.beta, p.mu_B, p.tau, p.sigma, p.noise, p.latent,
                                   p.gB, return_locked=True)
    if K < m:
        # escalation never triggered; harmful exhausts budget
        return value_memoryless(p, t, K)
    PH = PH_reach * qH
    PB = (PBq - Pb_reach) + Pb_reach * qB  # quitters + blocked by reviewer
    cost = c * (p.pi * PH_reach + (1 - p.pi) * Pb_reach)
    return p.pi * p.h * PH - (1 - p.pi) * p.v * PB - cost, PH, PB


T_GRID = np.concatenate([np.linspace(-6, 12, 721)])


def optimise(fun, tgrid=T_GRID):
    """Maximise value over threshold; include t=+inf (never refuse, value 0)."""
    best = (0.0, np.inf, 0.0, 0.0)
    for t in tgrid:
        val, PH, PB = fun(t)
        if val > best[0]:
            best = (val, t, PH, PB)
    return best  # value, t*, PH, PB


def single_shot_fpr(p, t):
    """Benign single-attempt refusal rate (over-refusal) at threshold t."""
    if not np.isfinite(t):
        return 0.0
    return benign_blocked(t, 1, p.beta, p.mu_B, p.tau, p.sigma, p.noise, p.latent, p.gB)


def single_shot_tpr(p, t):
    if not np.isfinite(t):
        return 0.0
    return harmful_blocked(t, 1, p.mu_H, p.tau, p.sigma, p.noise, p.latent, p.gH)
