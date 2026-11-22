"""
GDP-visibility model of AI productivity gains.

Static general-equilibrium model with:
  * market good c        (price 1/A_c)
  * market service s_m   (price 1/A_s)  -- household-substitutable service
  * home-produced s_h    (shadow price 1/B) -- self-provision with own time + free consumer AI
  * leisure x            (price 1 = wage)
Technologies are linear in time; wage w = 1 is numeraire; full income F = 1.

Preferences (homothetic, nested CES):
  U = [ a c^r + b s^r + g x^r ]^(1/r),   eta = 1/(1-r)    (eta=1 -> Cobb-Douglas)
  s = [ om s_m^rho + (1-om) s_h^rho ]^(1/rho), sigma = 1/(1-rho)

Measured real GDP = chained Fisher quantity index over market final goods {c, s_m}.
Welfare = equivalent variation (money metric, homothetic => P0/P1 - 1 of full income).
"""
import numpy as np


def ces_index(weights, prices, el):
    """Ideal price index for CES with share weights `weights` (CES 'a_i') and elasticity el."""
    weights = np.asarray(weights, float); prices = np.asarray(prices, float)
    if abs(el - 1.0) < 1e-10:
        w = weights / weights.sum()
        return np.exp(np.sum(w * np.log(prices / w)))  # Cobb-Douglas unit cost (up to const)
    return np.sum(weights ** el * prices ** (1 - el)) ** (1 / (1 - el))


class Economy:
    def __init__(self, e_c=0.45, e_s=0.10, e_x=0.45, theta0=0.5, sigma=2.0, eta=1.0):
        """Calibrate so that at A_c=A_s=B=1, upper-nest expenditure shares are (e_c,e_s,e_x)
        and the home share of the s-composite is theta0."""
        assert abs(e_c + e_s + e_x - 1) < 1e-12
        self.sigma, self.eta = sigma, eta
        # inner nest: at equal prices theta = (1-om)^sig / (om^sig + (1-om)^sig)
        if abs(sigma - 1) < 1e-10:
            self.om = 1 - theta0
        else:
            ratio = (theta0 / (1 - theta0)) ** (1 / sigma)   # (1-om)/om
            self.om = 1 / (1 + ratio)
        # upper nest: at base prices all =1 except P_s(base); choose a_i so shares match
        Ps0 = self._Ps(1.0, 1.0)
        P0 = np.array([1.0, Ps0, 1.0])
        e = np.array([e_c, e_s, e_x])
        if abs(eta - 1) < 1e-10:
            self.a = e.copy()
        else:
            # e_i = a_i^eta P_i^(1-eta) / sum  => a_i ∝ (e_i P_i^(eta-1))^(1/eta)
            self.a = (e * P0 ** (eta - 1)) ** (1 / eta)
        self.base = self.solve(1.0, 1.0, 1.0)

    def _Ps(self, ps, ph):
        s = self.sigma
        if abs(s - 1) < 1e-10:
            om = self.om
            return (ps / om) ** om * (ph / (1 - om)) ** (1 - om)
        return (self.om ** s * ps ** (1 - s) + (1 - self.om) ** s * ph ** (1 - s)) ** (1 / (1 - s))

    def solve(self, Ac, As, B):
        pc, ps, ph, px = 1 / Ac, 1 / As, 1 / B, 1.0
        s = self.sigma
        Ps = self._Ps(ps, ph)
        if abs(s - 1) < 1e-10:
            theta = 1 - self.om
        else:
            theta = (1 - self.om) ** s * ph ** (1 - s) / Ps ** (1 - s)
        P_up = np.array([pc, Ps, px]); eta = self.eta
        if abs(eta - 1) < 1e-10:
            e = self.a / self.a.sum()
            logP = np.sum(e * np.log(P_up / e))
            P = np.exp(logP)
        else:
            num = self.a ** eta * P_up ** (1 - eta)
            e = num / num.sum()
            P = num.sum() ** (1 / (1 - eta))
        e_c, e_s, e_x = e
        q = dict(c=e_c / pc, sm=e_s * (1 - theta) / ps, sh=e_s * theta / ph, x=e_x)
        h = e_s * theta
        ell = 1 - h - e_x
        return dict(prices=dict(c=pc, sm=ps), q=q, theta=theta, e=e, P=P, ell=ell,
                    gdp_nom=e_c + e_s * (1 - theta), h=h)


def chained_fisher(econ, path):
    """path: array (N,3) of (Ac,As,B) from base to final. Returns chained Fisher real GDP index."""
    idx = 1.0
    prev = econ.solve(*path[0])
    for k in range(1, len(path)):
        cur = econ.solve(*path[k])
        p0 = np.array([prev['prices']['c'], prev['prices']['sm']])
        p1 = np.array([cur['prices']['c'], cur['prices']['sm']])
        q0 = np.array([prev['q']['c'], prev['q']['sm']])
        q1 = np.array([cur['q']['c'], cur['q']['sm']])
        L = p0 @ q1 / (p0 @ q0); Pa = p1 @ q1 / (p1 @ q0)
        idx *= np.sqrt(L * Pa)
        prev = cur
    return idx


def visibility(econ, gc=0.0, gs=0.0, gB=0.0, steps=60):
    """Exact (global) visibility ratio: real GDP change ($ at base) / equivalent variation ($)."""
    path = np.exp(np.outer(np.linspace(0, 1, steps + 1), [gc, gs, gB]))
    b = econ.base
    fin = econ.solve(*path[-1])
    dgdp = (chained_fisher(econ, path) - 1) * b['gdp_nom']   # base-year dollars (F=1)
    ev = b['P'] / fin['P'] - 1                                # homothetic money metric, F=1
    return dgdp, ev, (dgdp / ev if abs(ev) > 1e-14 else np.nan)


def analytic_V(e_c, e_s, e_x, theta, sigma, eta):
    """First-order visibility coefficients (Proposition 1)."""
    Lam = (1 - eta) * (e_c - (1 - e_s) * (1 - theta))
    V_c = 1 - (1 - eta) * (e_x + e_s * theta)
    V_s = 1 + Lam + (sigma - 1) * theta
    V_h = Lam - (sigma - 1) * (1 - theta)
    return V_c, V_s, V_h
