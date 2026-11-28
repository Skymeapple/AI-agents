"""Core model: a non-rival training-data commons with market dilution.

Notation (see paper Sec. 3):
  M      mass (number) of potential creators, entry cost c ~ F
  pi0    per-work market income absent AI
  sigma  dilution: AI quality q reduces per-work market income to pi0*(1-sigma*q)
  T      total per-work training payment received by a creator (sum over firms)
  H(N)   steady-state AI quality given a flow N of fresh human works per period
  A      aggregate AI revenue scale: industry revenue = A * q
All functions are deterministic; no fitted or invented data.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ---------------------------------------------------------------- learner-based H
def kalman_P(N, phi, s2, iters=4000):
    """Steady-state per-dimension posterior variance of an optimal (Kalman) tracker
    of an AR(1) drifting parameter theta_t = phi*theta_{t-1} + sqrt(1-phi^2)*eta,
    stationary variance 1, observed through N fresh samples/period with noise s2
    and isotropic unit-variance features (information N/s2 per dimension)."""
    N = np.asarray(N, dtype=float)
    P = np.ones_like(N)
    for _ in range(iters):
        Pm = phi**2 * P + (1 - phi**2)
        Pn = Pm / (1 + Pm * N / s2)
        if np.max(np.abs(Pn - P)) < 1e-13:
            P = Pn
            break
        P = Pn
    return P


def H_kalman(N, phi=0.95, s2=50.0):
    """AI quality q = 1 - normalised tracking error (in [0,1])."""
    return 1.0 - kalman_P(N, phi, s2)


def dH_kalman(N, phi=0.95, s2=50.0, h=1e-4):
    N = np.asarray(N, dtype=float)
    return (H_kalman(N + h, phi, s2) - H_kalman(np.maximum(N - h, 0), phi, s2)) / (
        (N + h) - np.maximum(N - h, 0))


# ---------------------------------------------------------------- cost distribution
class Uniform:
    def __init__(self, cbar=1.0):
        self.cbar = cbar

    def F(self, p):
        return np.clip(p / self.cbar, 0.0, 1.0)

    def f(self, p):
        return np.where((p > 0) & (p < self.cbar), 1.0 / self.cbar, 0.0)

    def partial_mean(self, p):
        """int_0^p c dF(c)"""
        p = np.clip(p, 0, self.cbar)
        return p**2 / (2 * self.cbar)


class LogNormal:
    def __init__(self, mu=np.log(0.5), s=0.6):
        from scipy.stats import lognorm
        self.d = lognorm(s=s, scale=np.exp(mu))

    def F(self, p):
        return self.d.cdf(np.maximum(p, 0))

    def f(self, p):
        return self.d.pdf(np.maximum(p, 1e-12))

    def partial_mean(self, p):
        from scipy.integrate import quad
        return quad(lambda c: c * self.d.pdf(c), 0, max(p, 0))[0]


# ---------------------------------------------------------------- model
class Commons:
    def __init__(self, M=1000.0, pi0=0.6, sigma=0.3, A=200.0, phi=0.95, s2=50.0,
                 dist=None, lam=1.0, v=None):
        self.M, self.pi0, self.sigma, self.A = M, pi0, sigma, A
        self.phi, self.s2 = phi, s2
        self.dist = dist if dist is not None else Uniform(1.0)
        self.lam = lam                      # AI consumer-surplus multiplier for welfare
        self.v = v if v is not None else pi0  # consumer value per human work

    def H(self, N):
        return H_kalman(N, self.phi, self.s2)

    def dH(self, N):
        return dH_kalman(N, self.phi, self.s2)

    def price(self, N, T):
        return self.pi0 * (1 - self.sigma * self.H(N)) + T

    def N_star(self, T, data_share=1.0):
        """Unique steady state N = M F(pi0(1-sigma H(s N)) + T).
        data_share s<1 = fraction of works available for training (opt-out)."""
        g = lambda N: N - self.M * self.dist.F(
            self.pi0 * (1 - self.sigma * self.H(data_share * N)) + T)
        lo, hi = 0.0, self.M
        if g(lo) >= 0:
            return 0.0
        if g(hi) <= 0:
            return self.M
        return brentq(g, lo, hi, xtol=1e-10)

    def dN_dT(self, T, h=1e-5):
        return (self.N_star(T + h) - self.N_star(max(T - h, 0))) / (T + h - max(T - h, 0))

    def multiplier(self, T=0.0):
        """Analytic pass-through dN*/dT = M f / (1 + M f pi0 sigma H'(N))."""
        N = self.N_star(T)
        p = self.price(N, T)
        Mf = self.M * self.dist.f(p)
        return Mf / (1 + Mf * self.pi0 * self.sigma * self.dH(N)), N

    # ---- AI industry
    def industry_profit(self, T):
        N = self.N_star(T)
        return self.A * self.H(N) - T * N

    def firm_payoff(self, tau_i, T_others, K):
        T = tau_i + T_others
        N = self.N_star(T)
        return self.A / K * self.H(N) - tau_i * N

    def A1(self):
        """Threshold revenue scale above which a monopolist pays voluntarily."""
        mult, N0 = self.multiplier(0.0)
        return N0 / (self.dH(N0) * mult)

    def Kbar(self):
        return self.A / self.A1()

    def T_joint(self, Tmax=None):
        Tmax = Tmax if Tmax is not None else 2 * self.dist_scale()
        r = minimize_scalar(lambda T: -self.industry_profit(T), bounds=(0, Tmax),
                            method="bounded", options={"xatol": 1e-7})
        T = r.x if -r.fun > self.industry_profit(0.0) else 0.0
        return T

    def dist_scale(self):
        """Upper bound for payment searches: smallest T at which every potential
        creator enters (beyond it N*=M and payments are pure transfers)."""
        cbar = getattr(self.dist, "cbar", None)
        if cbar is None:
            cbar = float(self.dist.d.ppf(0.999))
        return max(cbar - self.pi0 * (1 - self.sigma * float(self.H(self.M))), 1e-6) / 2

    def best_response(self, T_others, K, Tmax=None):
        Tmax = Tmax if Tmax is not None else 2 * self.dist_scale()
        r = minimize_scalar(lambda t: -self.firm_payoff(t, T_others, K), bounds=(0, Tmax),
                            method="bounded", options={"xatol": 1e-8})
        return r.x if -r.fun > self.firm_payoff(0.0, T_others, K) else 0.0

    def T_nash(self, K, iters=200):
        """Symmetric voluntary-contribution Nash equilibrium (best-response iteration)."""
        if K <= 1:
            return self.T_joint()
        tau = np.full(int(K), self.T_joint() / K)
        for _ in range(iters):
            old = tau.copy()
            for i in range(len(tau)):
                tau[i] = 0.5 * tau[i] + 0.5 * self.best_response(tau.sum() - tau[i], K)
            if np.max(np.abs(tau - old)) < 1e-7:
                break
        return tau.sum()

    # ---- creators
    def creator_surplus(self, T, data_share=1.0):
        N = self.N_star(T, data_share)
        p = self.pi0 * (1 - self.sigma * self.H(data_share * N)) + T
        return self.M * (p * self.dist.F(p) - self.dist.partial_mean(p))

    # ---- welfare (payments are transfers)
    def welfare(self, T, data_share=1.0):
        N = self.N_star(T, data_share)
        q = self.H(data_share * N)
        p = self.pi0 * (1 - self.sigma * q) + T
        human_value = self.v * N * (1 - self.sigma * q)   # consumption of human works
        cost = self.M * self.dist.partial_mean(p)
        return self.lam * self.A * q + human_value - cost

    def T_welfare(self, Tmax=None):
        Tmax = Tmax if Tmax is not None else 2 * self.dist_scale()
        r = minimize_scalar(lambda T: -self.welfare(T), bounds=(0, Tmax), method="bounded",
                            options={"xatol": 1e-7})
        return r.x if -r.fun > self.welfare(0.0) else 0.0
