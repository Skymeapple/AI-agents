"""Core model: mapping compute denial to frontier-capability lag.

Notation (time in years, logs natural):
  g   growth rate of frontier training compute (ln of yearly multiplier)
  a   rate of frontier algorithmic efficiency progress
  G   = g + a, growth rate of frontier *effective* compute
  R   US:China ratio of compute usable for frontier training
  x   follower algorithmic-efficiency gap, ln E_US - ln E_CN
  Phi follower imitation advantage (compute multiplier)

Lag identity (Lemma 1):  G * L = ln R + x - ln Phi.

Three structural hypotheses about how x responds to R:
  H-A (additive / exogenous diffusion):  x constant  -> beta = 1/G
  H-M (multiplicative recapture, IFP 2026): L = (1-S) ln R / G -> beta = (1-S)/G
  H-E (endogenous follower R&D, Nelson-Phelps catch-up):
        dx/dt = eta * ln R - kappa * x,   steady state x* = (eta/kappa) ln R
        -> short-run beta = 1/G, long-run beta = (1 + eta/kappa)/G
"""
import numpy as np
from params import LN2, LN10


def G_from(growth_factor, halving_months):
    return np.log(growth_factor) + LN2 / (np.asarray(halving_months) / 12.0)


def months_per_efold(G, hypothesis="A", S=0.5, eta_over_kappa=1.0):
    base = 12.0 / G
    if hypothesis == "A":
        return base
    if hypothesis == "M":
        return (1.0 - S) * base
    if hypothesis == "E":
        return (1.0 + eta_over_kappa) * base
    raise ValueError(hypothesis)


def lag_months(lnR, G, x=0.0, lnPhi=0.0):
    """Lemma 1 lag, in months."""
    return 12.0 * (lnR + x - lnPhi) / G


def leakage_gain_months(d, leak, G, mult=1.0):
    """Prop. 2: lag gained by enforcing a fraction d of the controllable channel
    when a fraction `leak` of counterfactual compute is uncontrollable."""
    return -mult * 12.0 / G * np.log(1.0 - d * (1.0 - leak))


def stock_response(theta, lam, dt):
    """Prop. 3: ln(K_new/K_old) after a permanent flow multiplier theta applied at
    dt=0, in balanced growth, lam = gamma + delta."""
    return np.log(theta + (1.0 - theta) * np.exp(-lam * np.asarray(dt)))


def simulate_x(t, dlnR, eta, kappa):
    """Integrate dx/dt = eta*dlnR(t) - kappa*x with x(t0)=0 (deviation form)."""
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        if kappa == 0:
            x[i] = x[i - 1] + h * eta * dlnR[i - 1]
        else:
            # exact step for piecewise-constant forcing
            xs = eta * dlnR[i - 1] / kappa
            x[i] = xs + (x[i - 1] - xs) * np.exp(-kappa * h)
    return x


# ---------------------------------------------------------------------------
# Status-quo compute paths anchored on IFP (2026) and intervention simulation
# ---------------------------------------------------------------------------
def status_quo_paths(t, us26, us27, r26, r27, us_growth_after=np.log(2.0),
                     ratio_growth_after=None):
    """US stock and US:China ratio, log-linear between anchors at t=2027.0
    (end-2026) and t=2028.0 (end-2027).  Before 2027 the 2027 rates are
    extended backwards; after 2028 the US grows at `us_growth_after` and the log
    ratio grows at `ratio_growth_after` (default: same as 2027)."""
    gu = np.log(us27 / us26)
    gr = np.log(r27 / r26)
    if ratio_growth_after is None:
        ratio_growth_after = gr
    lnU = np.where(t <= 2028.0, np.log(us26) + gu * (t - 2027.0),
                   np.log(us27) + us_growth_after * (t - 2028.0))
    lnRr = np.where(t <= 2028.0, np.log(r26) + gr * (t - 2027.0),
                    np.log(r27) + ratio_growth_after * (t - 2028.0))
    U = np.exp(lnU)
    C = U / np.exp(lnRr)
    return U, C


def intervene(t, C_sq, dflow, t0, delta):
    """Add a constant flow change dflow (H100e/yr) from t0; return new China
    stock.  Stock added by intervention depreciates at delta."""
    add = np.zeros_like(t)
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        f = dflow if t[i - 1] >= t0 else 0.0
        add[i] = add[i - 1] * np.exp(-delta * h) + f * h
    return np.maximum(C_sq + add, 1.0)
