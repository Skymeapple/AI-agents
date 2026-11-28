"""Core reliance--skill feedback model (see paper Section 3).

State: unaided skill s in [0,1].  Time unit: months.
  h(s)   = h0 + h1*(1-s)            unaided error rate
  a(s;e) = e*(1 - rho*s)            AI-assisted error after human verification
  D(s;e) = h(s) + dc - a(s;e)       myopic advantage of relying on AI (loss units)
  r(s)   = 1[D>0] (step) or sigmoid(beta*D) (smooth)  reliance share
  p(r)   = (1-r) + kappa*r          effective practice
  ds/dt  = alpha*p*(1-s) - delta*s
"""
import numpy as np
from dataclasses import dataclass, replace

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Params:
    alpha: float = 9 * LN2 / 6.5     # learning rate (gives s_H = 0.9)
    delta: float = LN2 / 6.5         # forgetting: half-life 6.5 months (Psych Bull 2025 accuracy)
    kappa: float = 0.15              # learning retained on AI-assisted tasks
    h0: float = 0.05
    h1: float = 0.60
    rho: float = 0.50                # verification efficacy
    dc: float = 0.03                 # effort saving from relying (loss units)
    beta: float = np.inf             # reliance sharpness (inf = step)
    u_min: float = 0.0               # mandated unaided-practice floor

    def with_(self, **kw):
        return replace(self, **kw)


def h(s, P):
    return P.h0 + P.h1 * (1 - s)


def a(s, eps, P):
    return eps * (1 - P.rho * s)


def D(s, eps, P):
    return h(s, P) + P.dc - a(s, eps, P)


def reliance(s, eps, P):
    d = D(s, eps, P)
    if np.isinf(P.beta):
        r = (d > 0).astype(float) if np.ndim(d) else float(d > 0)
    else:
        r = 1.0 / (1.0 + np.exp(-P.beta * d))
    return np.minimum(r, 1 - P.u_min)


def practice(r, P):
    return (1 - r) + P.kappa * r


def drift(s, eps, P):
    p = practice(reliance(s, eps, P), P)
    return P.alpha * p * (1 - s) - P.delta * s


def team_error(s, eps, P):
    r = reliance(s, eps, P)
    return r * a(s, eps, P) + (1 - r) * h(s, P)


# ---------- closed forms for the step model (Propositions 1-4) ----------
def s_eq(p, P):
    return P.alpha * p / (P.alpha * p + P.delta)


def s_high(P):
    return s_eq(1.0, P)


def s_low(P):
    pmin = P.u_min + P.kappa * (1 - P.u_min)
    return s_eq(pmin, P)


def theta(eps, P):
    """Reliance threshold: agent relies iff s < theta."""
    return (P.h0 + P.h1 + P.dc - eps) / (P.h1 - eps * P.rho)


def eps_threshold(s_star, P):
    """eps such that theta(eps) = s_star."""
    return (P.h0 + P.h1 + P.dc - s_star * P.h1) / (1 - s_star * P.rho)


def eps_collapse(P):
    return eps_threshold(s_high(P), P)


def eps_recover(P):
    return eps_threshold(s_low(P), P)


def condition_A(P):
    """theta decreasing in eps (better AI raises reliance threshold)."""
    return P.rho * (P.h0 + P.h1 + P.dc) < P.h1


def u_floor_star(eps, P):
    """Minimal practice floor that removes the deskilled equilibrium (Prop. 4)."""
    th = theta(eps, P)
    if th >= s_high(P):
        return np.nan  # trap cannot be removed: even full practice keeps s below theta
    return u_for_target(th, P)


def u_for_target(s_bar, P):
    """Minimal floor guaranteeing every stable equilibrium has s >= s_bar."""
    if s_bar >= s_high(P):
        return np.nan
    p_needed = P.delta * s_bar / (P.alpha * (1 - s_bar))
    return float(np.clip((p_needed - P.kappa) / (1 - P.kappa), 0, 1))


# ---------- numerics ----------
def equilibria(eps, P, n=20001):
    """Return list of (s*, stable) for the 1-D ODE by sign changes of drift."""
    s = np.linspace(0, 1, n)
    f = drift(s, eps, P)
    out = []
    for i in range(n - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            # bisection / linear interp
            lo, hi = s[i], s[i + 1]
            flo = f[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fm = drift(mid, eps, P)
                if np.sign(fm) == np.sign(flo) and fm != 0:
                    lo, flo = mid, fm
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
            stable = f[i] > 0 and f[i + 1] < 0
            # for step model discontinuities: drift jumps sign at theta -> not an equilibrium
            if np.isinf(P.beta):
                fl, fr = drift(root - 1e-7, eps, P), drift(root + 1e-7, eps, P)
                if abs(fl - fr) > 1e-3 and not (abs(fl) < 1e-4 or abs(fr) < 1e-4):
                    continue
            out.append((root, stable))
    return out


def simulate(s0, eps_path, P, dt=0.05, sigma=0.0, rng=None):
    """Euler(-Maruyama) integration; eps_path is array over steps."""
    s = np.array(s0, dtype=float)
    traj = np.empty((len(eps_path),) + s.shape)
    sq = np.sqrt(dt)
    for k, e in enumerate(eps_path):
        s = s + drift(s, e, P) * dt
        if sigma > 0:
            s = s + sigma * sq * rng.standard_normal(s.shape)
        s = np.clip(s, 0, 1)
        traj[k] = s
    return traj
