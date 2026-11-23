"""
Vintage-capital model of AI compute assets under technological obsolescence.

A unit of compute capex (cost normalised to 1) earns a competitive rental flow
    pi(a) = P * exp(-g a) - c          (a = age in years)
where g is the rate at which the market rental of a fixed amount of old compute
falls because newer hardware delivers compute more cheaply, and c is the annual
operating cost (power, maintenance, space) per $ of capex.  The asset is retired
at T = min(ln(P/c)/g, Tmax).  Free entry pins P by V(0) = 1 at discount rate r.

Economic value at age a:  V(a) = int_a^T pi(s) e^{-r(s-a)} ds   (closed form below)
Economic depreciation:    delta(a) = -V'(a) = pi(a) - r V(a)
Accounting (straight-line over L): B(a) = max(0, 1 - a/L).
"""
import numpy as np
from scipy.optimize import brentq


def life(P, c, g, Tmax):
    if c <= 0:
        return Tmax
    return float(min(np.log(P / c) / g, Tmax)) if P > c else 0.0


def value(a, P, c, g, r, Tmax):
    """Closed-form V(a); vectorised in a."""
    a = np.asarray(a, dtype=float)
    T = life(P, c, g, Tmax)
    rem = np.clip(T - a, 0.0, None)
    term1 = P * np.exp(-g * a) * (1 - np.exp(-(r + g) * rem)) / (r + g)
    term2 = c * (1 - np.exp(-r * rem)) / r
    return np.where(a < T, term1 - term2, 0.0)


def solve_P(c, g, r, Tmax):
    """Rental level at age 0 (per $ capex per year) such that V(0)=1."""
    f = lambda P: float(value(0.0, P, c, g, r, Tmax)) - 1.0
    lo = max(c * 1.0000001, 1e-6)
    return brentq(f, lo, 50.0)


def econ_dep(a, P, c, g, r, Tmax):
    a = np.asarray(a, dtype=float)
    T = life(P, c, g, Tmax)
    pi = P * np.exp(-g * a) - c
    d = pi - r * value(a, P, c, g, r, Tmax)
    return np.where(a < T, d, 0.0)


def book(a, L):
    a = np.asarray(a, dtype=float)
    return np.clip(1 - a / L, 0.0, 1.0)


class Asset:
    """Convenience wrapper: one asset class (e.g. servers or shells)."""

    def __init__(self, g, c, r, Tmax, L, straight_line_econ=False):
        self.g, self.c, self.r, self.Tmax, self.L = g, c, r, Tmax, L
        self.sl = straight_line_econ
        if straight_line_econ:
            # long-lived structures: economic = straight-line over L; rental is
            # the level annuity that recovers capital plus opex.
            self.P = r / (1 - np.exp(-r * L)) + c
            self.T = L
        else:
            self.P = solve_P(c, g, r, Tmax)
            self.T = life(self.P, c, g, Tmax)

    def V(self, a):
        if self.sl:
            a = np.asarray(a, float)
            rem = np.clip(self.L - a, 0, None)
            return (self.P - self.c) * (1 - np.exp(-self.r * rem)) / self.r
        return value(a, self.P, self.c, self.g, self.r, self.Tmax)

    def Vbook_econ(self, a):
        """Economic value used for depreciation (for structures: straight-line)."""
        if self.sl:
            return book(a, self.L)
        return self.V(a)

    def B(self, a):
        return book(a, self.L)

    def rent(self, a):
        """Gross rental revenue per $ of original capex at age a."""
        a = np.asarray(a, float)
        if self.sl:
            return np.where(a < self.L, self.P, 0.0)
        return np.where(a < self.T, self.P * np.exp(-self.g * a), 0.0)
