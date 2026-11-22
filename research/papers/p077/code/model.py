"""Core model: Significant Quantity of Compute (SQC) and material-accountancy verification.

Notation (years as decimals; chips in H100-equivalents, H100e):
  SQC(t)  chips needed to complete a threshold-crossing run within window tau
  S(t)    verified/declared global stock
  sigma   relative 1-sigma inventory uncertainty of the accountancy system
  rho(t)  = SQC(t) / (sigma * S(t))  accountancy ratio (signal-to-noise of a diversion of one SQC)
Neyman-Pearson test on material-unaccounted-for (MUF ~ N(D, (sigma S)^2)):
  P_D = Phi(rho - z_{1-alpha});  goal P_D >= p*  <=>  rho >= rho* = z_{1-alpha} + z_{p*}
"""
import numpy as np
from scipy.stats import norm
import params as P

SEC_PER_YR = 365*24*3600.0


def rho_star(alpha=P.ALPHA, pd=P.PD_GOAL):
    return norm.ppf(1-alpha) + norm.ppf(pd)


def stock(t, gS=P.G_S_MED):
    return P.S_REF * gS ** (t - P.T_S)


def chip_flop(u=P.U_MED, tau=P.TAU_BASE):
    """FLOP one H100e delivers in window tau at utilisation u."""
    return P.R_H100 * u * tau * SEC_PER_YR


def threshold_flop(t, rule, h=P.H_ALG_MED, gF=P.G_F_MED, F_cap=1e25, t_pause=2026.0):
    """Physical FLOP needed to cross the threshold at time t.
    rule='capability': fixed capability (F0 at T0), eroded by algorithmic progress (half-life h)
    rule='flopcap'   : fixed physical-FLOP cap F_cap (e.g. EU AI Act 1e25 presumption)
    rule='pause'     : capability frozen at the frontier of t_pause; eroded by algorithmic progress thereafter
    rule='frontier'  : moving cap tied to the frontier trend (grows gF x/yr)
    """
    t = np.asarray(t, dtype=float)
    if rule == 'capability':
        return P.F0 * 2.0 ** (-(t - P.T0) / h)
    if rule == 'flopcap':
        return F_cap * np.ones_like(t)
    if rule == 'pause':
        Fp = P.F0 * gF ** (t_pause - P.T0)
        return Fp * 2.0 ** (-np.maximum(t - t_pause, 0) / h)
    if rule == 'frontier':
        return P.F0 * gF ** (t - P.T0)
    raise ValueError(rule)


def sqc(t, rule, u=P.U_MED, tau=P.TAU_BASE, **kw):
    return threshold_flop(t, rule, **kw) / chip_flop(u, tau)


def rho(t, rule, sigma, gS=P.G_S_MED, u=P.U_MED, tau=P.TAU_BASE, **kw):
    return sqc(t, rule, u=u, tau=tau, **kw) / (sigma * stock(t, gS))


def decay_rate(rule, h=P.H_ALG_MED, gS=P.G_S_MED, gF=P.G_F_MED):
    """lambda in rho(t) = rho(t1) exp(-lambda (t-t1)); positive = verification gets harder."""
    if rule in ('capability', 'pause'):
        return np.log(2)/h + np.log(gS)
    if rule == 'flopcap':
        return np.log(gS)
    if rule == 'frontier':
        return np.log(gS) - np.log(gF)
    raise ValueError(rule)


def horizon(rule, sigma, h=P.H_ALG_MED, gS=P.G_S_MED, gF=P.G_F_MED, u=P.U_MED,
            tau=P.TAU_BASE, rs=None, t1=2026.0, **kw):
    """Date t* at which rho(t*) = rho* (closed form, log-linear rho).
    Returns +inf if rho never falls below rho* after t1 (lambda<=0 and rho(t1)>=rho*),
    and the (possibly past) crossing date otherwise. For 'frontier' with lambda<0 and rho(t1)<rho*,
    returns the date at which verification first becomes feasible (as a negative-direction crossing)."""
    rs = rho_star() if rs is None else rs
    lam = decay_rate(rule, h, gS, gF)
    r1 = rho(t1, rule, sigma, gS=gS, u=u, tau=tau, h=h, gF=gF, **kw)
    if rule == 'pause':
        lam = np.log(2)/h + np.log(gS)  # valid for t>=t_pause
    if np.isclose(lam, 0):
        return np.inf if r1 >= rs else -np.inf
    return t1 + np.log(r1/rs)/lam


def sigma_required(t, rule, rs=None, **kw):
    """Largest relative inventory uncertainty that still meets the detection goal at t."""
    rs = rho_star() if rs is None else rs
    gS = kw.pop('gS', P.G_S_MED)
    return sqc(t, rule, **kw) / (rs * stock(t, gS))


# ---------------- deterrence (inspection game) ---------------------------------
def deterrence_pd(gain_over_penalty):
    """Violator with gain G if undetected and penalty C if detected is deterred iff
    (1-P_D) G - P_D C <= 0  <=>  P_D >= G/(G+C)."""
    k = np.asarray(gain_over_penalty, float)
    return k/(1+k)


def rho_star_deterrence(gain_over_penalty, alpha=P.ALPHA):
    return norm.ppf(1-alpha) + norm.ppf(deterrence_pd(gain_over_penalty))


# ---------------- inspection capacity (facility regime) --------------------------
def inspection_nmax(I=P.IAEA_INFIELD, q=0.9, tau=P.TAU_BASE, pd=P.PD_GOAL):
    """Max number of candidate sites over which random inspections (Poisson, I per year, each
    detecting an ongoing covert run with prob q) achieve P_D >= pd for a run lasting tau."""
    return q*I*tau/(-np.log(1-pd))


def potential_sites(t, rule, gS=P.G_S_MED, **kw):
    """Upper bound on simultaneous threshold-crossing runs the stock could host: S/SQC."""
    return stock(t, gS)/sqc(t, rule, **kw)


# ---------------- hardware-enabled mechanisms (legacy stock) ---------------------
def uncontrolled_noise(t, sigma_legacy, eps, t_hem=2026.0, life=5.0, gS=P.G_S_MED):
    """1-sigma absolute uncertainty when chips shipped after t_hem carry HEMs.
    Legacy stock S(t_hem) retires exponentially with mean life `life`; a fraction eps of the
    post-t_hem stock is assumed effectively uncontrolled (tamper/defeat), with full-scale uncertainty."""
    t = np.asarray(t, float)
    S_h = stock(t_hem, gS)
    legacy = np.where(t < t_hem, stock(t, gS), S_h*np.exp(-(t - t_hem)/life))
    new = np.where(t < t_hem, 0.0, stock(t, gS) - S_h)
    return np.sqrt((sigma_legacy*legacy)**2 + (eps*new)**2)
