"""Agent timing model and fluency metrics.

Agent on a task whose human reference has n actions:
  steps K = ceil(kappa * n) nominal steps, plus recovery steps: each executed step errs
  (recoverably) with prob eps, each error costs Geometric(mean rho) extra steps.
  step j (1-indexed over all executed steps) costs  l0 * (1 + gamma*(j-1)) + a  seconds.
  Unrecoverable failure: task-level success probability p_succ (calibrated per agent),
  optionally decaying with steps via hazard h: P(success) = p0 * exp(-h * K_exec).
Human: time ~ LogNormal(median = T_h, sigma = sigma_h), success w.p. p_h.
Time-budgeted success S(tau) = P(success and T <= tau * T_h).
"""
import numpy as np
from math import ceil, log, sqrt


def agent_time_closed(K, l0, gamma, a):
    """Deterministic time for K steps with linear context growth (Prop. 1)."""
    return K * (l0 + a) + l0 * gamma * K * (K - 1) / 2.0


def parity_horizon(h, kappa, l0, gamma, a, c0=0.0):
    """Largest human-step count n with agent_time(kappa n) <= h n + c0 (gamma>0), Prop. 1.
    Solves l0*gamma*kappa^2 n^2/2 + (kappa(l0+a) - l0*gamma*kappa/2 - h) n - c0 <= 0."""
    A = l0 * gamma * kappa ** 2 / 2.0
    Bc = kappa * (l0 + a) - l0 * gamma * kappa / 2.0 - h
    if A == 0:
        return np.inf if Bc <= 0 else (c0 / Bc if c0 > 0 else 0.0)
    disc = Bc ** 2 + 4 * A * c0
    return max((-Bc + sqrt(disc)) / (2 * A), 0.0)


def required_l0(T_h, n, kappa, gamma, a):
    """Per-step model latency needed for the agent to match T_h on an n-action task."""
    K = kappa * n
    denom = K + gamma * K * (K - 1) / 2.0
    return (T_h - K * a) / denom


def simulate_agent(n_actions, T_h, agent, rng, reps=200):
    """Monte Carlo: returns arrays (times[reps, tasks], success[reps, tasks])."""
    n_actions = np.asarray(n_actions)
    m = len(n_actions)
    kappa, l0, gamma, a = agent['kappa'], agent['l0'], agent['gamma'], agent['a']
    eps, rho, p0, hz = agent.get('eps', 0.0), agent.get('rho', 1.0), agent['p0'], agent.get('h', 0.0)
    K = np.ceil(kappa * n_actions).astype(int)
    times = np.empty((reps, m)); succ = np.empty((reps, m), dtype=bool)
    for r in range(reps):
        if eps > 0:
            nerr = rng.binomial(K, eps)
            # sum of nerr geometric(mean rho) recovery costs (support >=1): NB + nerr
            pgeo = 1.0 / rho
            extra = np.where(nerr > 0, rng.negative_binomial(np.maximum(nerr, 1), pgeo) + nerr, 0)
        else:
            extra = 0
        Kx = K + extra
        times[r] = agent_time_closed(Kx, l0, gamma, a) * agent.get('noise', 1.0) ** rng.standard_normal(m)
        ps = p0 * np.exp(-hz * Kx)
        succ[r] = rng.random(m) < ps
    return times, succ


def budgeted_success(times, succ, T_h, taus):
    T_h = np.asarray(T_h)
    return np.array([np.mean(succ & (times <= tau * T_h[None, :])) for tau in taus])


def human_budgeted_success(taus, p_h, sigma_h):
    from scipy.stats import norm
    taus = np.asarray(taus, dtype=float)
    return p_h * norm.cdf(np.log(taus) / sigma_h)


def ld_rate_gaussian(mu, var, b):
    """Cramer rate for exceeding per-step budget b when mean step time mu>b (Gaussian approx)."""
    return (mu - b) ** 2 / (2 * var)
