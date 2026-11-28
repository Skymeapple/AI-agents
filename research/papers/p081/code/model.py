"""Headroom-budget model of frontier training compute (vectorised over samples).

C(t) = min( C_fin(t), C_pow(t), C_lat )
  C_fin = phi(t) * sigma(t) * Y(t) * q0 * exp(gh t) * x(t)        (financial channel)
  C_pow = Pmax(t) * T(t) * e0 * exp(ge t)                          (power channel)
  C_lat = data-movement / latency ceiling                          (engineering channel)

Bounded "share" factors (phi, sigma, x, T) grow at their current rates until a cap;
"secular" factors (Y, h, e, Pmax) grow forever at constant rates.
Variant 'revenue': investment growth is additionally limited by AI revenue growth once the
investment/revenue ratio exceeds lambda_max times its 2025 value.
"""
import numpy as np
from params import FIXED

SEC_PER_DAY = 86400.0
CHANNELS = np.array(["financial", "power", "latency"])


def simulate(p, years=20.0, dt=1.0 / 12, variant="exogenous", rev=None):
    n = len(p["gC0"])
    t = np.arange(0.0, years + 1e-9, dt)              # years since mid-2025
    T_ = t[None, :]
    C0 = 10 ** p["log10_C0"][:, None]
    K0, I0, Y0 = FIXED["K0_usd"], FIXED["I0_usd"], FIXED["Y0_usd"]

    gh, ge, gY = p["gh"][:, None], p["ge"][:, None], p["gY"][:, None]
    gK0, gI0, gC0 = p["gK0"][:, None], p["gI0"][:, None], p["gC0"][:, None]

    # ---- precision / utilisation residual (bounded) ----
    gx0 = np.clip(gC0 - gK0 - gh, 0.0, None)
    logx = np.minimum(gx0 * T_, np.log(p["x_max"])[:, None])

    # ---- financial channel ----
    phi0 = K0 / I0
    gphi = np.clip(gK0 - gI0, 0.0, None)
    logphi = np.minimum(np.log(phi0) + gphi * T_, np.log(p["phi_max"])[:, None])
    logY = np.log(Y0) + gY * T_
    sig0 = I0 / Y0
    logsigcap = np.log(p["sigma_max"])[:, None]
    if variant == "exogenous":
        logsig = np.minimum(np.log(sig0) + (gI0 - gY) * T_, logsigcap)
    elif variant == "revenue":
        gR0, gRinf, tau, lam = (rev[k][:, None] for k in ("gR0", "gRinf", "tau", "lam_max"))
        gR = gRinf + (gR0 - gRinf) * np.exp(-T_ / tau)          # revenue growth path
        logR = np.concatenate([np.zeros((n, 1)), np.cumsum(gR[:, :-1] * dt, axis=1)], axis=1)
        logI = np.zeros_like(T_ * np.ones((n, 1)))
        for j in range(1, len(t)):
            # investment grows at gI0 unless leverage (I/R relative to 2025) would exceed lam_max
            cand = logI[:, j - 1] + gI0[:, 0] * dt
            lev_cap = logR[:, j] + np.log(lam[:, 0])
            logI[:, j] = np.minimum(cand, np.maximum(lev_cap, logI[:, j - 1] + gR[:, j - 1] * dt))
        logsig = np.minimum(np.log(sig0) + logI - gY * T_, logsigcap)
    else:
        raise ValueError(variant)
    logq = np.log(C0 / K0) + gh * T_ + logx
    logCfin = logphi + logsig + logY + logq

    # ---- power channel ----
    e0 = C0 / FIXED["E0_J"]                                   # FLOP per joule today
    logT = np.minimum(np.log(FIXED["T0_days"]) + p["gT"][:, None] * T_,
                      np.log(p["T_max_days"])[:, None]) + np.log(SEC_PER_DAY)
    logP = np.log(p["P2030_GW"][:, None] * 1e9) + p["gPmax"][:, None] * (T_ - 4.5)
    logCpow = logP + logT + np.log(e0) + ge * T_      # x excluded: see paper Sec. 4.4 (power-growth consistency)

    # ---- latency ceiling ----
    logClat = (p["log10_Clat"] * np.log(10))[:, None] * np.ones_like(T_)

    stack = np.stack([logCfin, logCpow, logClat], axis=0)
    logC = stack.min(axis=0)
    bind = stack.argmin(axis=0)
    return dict(t=t + FIXED["t0"], logC=logC, bind=bind, logCfin=logCfin,
                logCpow=logCpow, logClat=logClat, gx0=gx0[:, 0], gphi=gphi[:, 0])


def summarize(sim, dt=1.0 / 12):
    t, logC = sim["t"], sim["logC"]
    k = int(round(1 / dt))
    g12 = (logC[:, k:] - logC[:, :-k])                      # trailing 1-yr log growth
    tt = t[k:]

    def first_below(th):
        below = g12 < np.log(th)
        idx = np.where(below.any(axis=1), below.argmax(axis=1), -1)
        return np.where(idx >= 0, tt[np.clip(idx, 0, None)], np.inf)

    out = {}
    for yr in (2028, 2030, 2035, 2040, 2045):
        j = np.argmin(np.abs(t - (yr + 0.5)))
        if abs(t[j] - (yr + 0.5)) < 1e-6 or yr == 2045:
            j = min(j, len(t) - 1)
        out[f"log10C_{yr}"] = logC[:, j] / np.log(10)
        out[f"bind_{yr}"] = sim["bind"][:, j]
    out["slow3"] = first_below(3.0)
    out["slow2"] = first_below(2.0)
    out["g12"] = g12
    out["tt"] = tt
    return out
