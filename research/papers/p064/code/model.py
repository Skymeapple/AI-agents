"""Core analytics and a deterministic cohort simulator for the
expertise-renewal model ("pipeline" model) of an occupation.

Notation (continuous time, years):
  B(t)   junior entrants per year
  S(t)   senior (independently competent) workers
  D(t)   senior demand, D0*exp(g t)
  tau    time-to-seniority (training duration)
  mu_J   junior hazard of leaving the occupation during training
  mu_S   senior hazard of leaving the occupation
  p      probability that a junior who survives to tau becomes a senior
  l      = exp(-mu_J*tau) junior survival to tau
  h      entrants per unit of senior demand per year (B = h D) [demand regime]
  m_bar  maximum entrants per senior per year (mentor capacity)  [capacity regime]

Expertise reproduction number  R = (entrants per senior-year) * p * l / mu_S.
"""
import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------- analytics
def reproduction_number(h, p, tau, mu_J, mu_S):
    return h * p * np.exp(-mu_J * tau) / mu_S


def required_R(g, tau, mu_S):
    """R needed for the senior stock to track demand growing at rate g."""
    return (1.0 + g / mu_S) * np.exp(g * tau)


def required_h(g, tau, p, mu_J, mu_S):
    return (g + mu_S) * np.exp(g * tau) / (p * np.exp(-mu_J * tau))


def lotka_root(R, mu_S, tau):
    """Unique real root r of r + mu_S = mu_S R exp(-r tau) (capacity regime)."""
    f = lambda r: r + mu_S - mu_S * R * np.exp(-r * tau)
    lo, hi = -mu_S + 1e-12, 10.0
    # f(lo) = -mu_S R e^{(mu_S) tau} <0 ; f(hi)>0
    return brentq(f, lo, hi)


def lotka_bounds(R, mu_S, tau):
    """Proposition 1 bounds on the real root for R>1."""
    lower = mu_S * (R - 1) / (1 + mu_S * R * tau)
    upper = min(mu_S * (R - 1), np.log(R) / tau)
    return lower, upper


def x_eff(x, d):
    """Effective intake shock: extensive (hiring cut x) times intensive (learning loss d)."""
    return 1.0 - (1.0 - x) * (1.0 - d)


def shortfall_path(t, xe, tau, mu_S, g):
    """Proposition 2: senior shortfall G(t) = 1 - S/D after a permanent intake shock at t=0."""
    lam = mu_S + g
    u = np.clip(t - tau, 0, None)
    return xe * (1 - np.exp(-lam * u))


def detection_time(eps, xe, tau, mu_S, g):
    """First time the senior shortfall reaches eps (inf if eps >= xe)."""
    if eps >= xe:
        return np.inf
    return tau - np.log(1 - eps / xe) / (mu_S + g)


def hayes_critical_delay(a, b):
    """y' = -a y - b y(t-tau). Stable for all tau if b<=a; else stable iff tau<tau_c."""
    if b <= a:
        return np.inf
    w = np.sqrt(b * b - a * a)
    return np.arccos(-a / b) / w


# ---------------------------------------------------------------- simulator
class Params(dict):
    __getattr__ = dict.get


def simulate(T=60.0, dt=1 / 12, D0=1.0, g=0.0, mu_S=0.043, mu_J=0.043, p=0.8, tau=5.0,
             hire_rule=None, p_rule=None, tau_rule=None, m_bar=None, S0=None,
             history_h=None, mu_S_rule=None):
    """Deterministic cohort simulation starting from the balanced-growth steady state.

    hire_rule(t, st) -> h (entrants per unit demand per year), or ('abs', B) for an absolute
    entrant flow; p_rule(t)->p and tau_rule(t)->tau apply to the cohort hired at t.
    st is a dict with S, D, pipeline information. m_bar caps entrants at m_bar*S.
    Returns dict of arrays.
    """
    n = int(round(T / dt))
    max_tau_steps = int(round(3 * tau / dt)) + 2
    grads = np.zeros(n + max_tau_steps + 1)   # senior inflow (persons) scheduled per step
    l0 = np.exp(-mu_J * tau)
    h0 = required_h(g, tau, p, mu_J, mu_S)
    hist = h0 if history_h is None else history_h
    # pre-period history: cohorts hired at t<0 with h0 graduate during [0, tau)
    k_tau = int(round(tau / dt))
    for j in range(k_tau):
        t_hire = (j - k_tau) * dt
        grads[j] += hist * D0 * np.exp(g * t_hire) * dt * p * l0
    S = D0 if S0 is None else S0
    out = {k: np.zeros(n) for k in ["t", "S", "D", "B", "G", "R"]}
    for i in range(n):
        t = i * dt
        D = D0 * np.exp(g * t)
        pc = p if p_rule is None else p_rule(t)
        tc = tau if tau_rule is None else tau_rule(t)
        kc = max(1, int(round(tc / dt)))
        lc = np.exp(-mu_J * tc)
        st = dict(t=t, S=S, D=D, grads=grads, i=i, dt=dt, p=pc, tau=tc, l=lc, mu_S=mu_S, g=g,
                  h0=h0)
        hr = h0 if hire_rule is None else hire_rule(t, st)
        if isinstance(hr, tuple):
            B = max(0.0, hr[1])
        else:
            B = max(0.0, hr * D)
        if m_bar is not None:
            B = min(B, m_bar * S)
        grads[i + kc] += B * dt * pc * lc
        ms = mu_S if mu_S_rule is None else mu_S_rule(t)
        S = S * np.exp(-ms * dt) + grads[i] * np.exp(-0.5 * ms * dt)
        out["t"][i] = t; out["S"][i] = S; out["D"][i] = D; out["B"][i] = B
        out["G"][i] = 1 - S / D
        out["R"][i] = (B / max(S, 1e-12)) * pc * lc / mu_S
    return out


def cumulative_deficit(out, dt=1 / 12, discount=0.0):
    t = out["t"]
    gap = np.clip(out["G"], 0, None)   # shortfall as a share of contemporaneous demand
    return float(np.sum(gap * np.exp(-discount * t)) * dt)


def excess_deficit(out, base, dt=1 / 12):
    """Integrated senior shortfall (share-years of demand) relative to an unshocked baseline run
    with identical numerics; removes O(dt) discretisation offsets."""
    return float(np.sum(out["G"] - base["G"]) * dt)
