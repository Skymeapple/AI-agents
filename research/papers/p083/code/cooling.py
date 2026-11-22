"""Hourly on-site cooling models (transparent, parametric; see paper Sec. 3.2).

wet_bulb_stull: Stull (2011), J. Appl. Meteor. Climatol. 50:2267-2269.
rh_from_dewpoint: Magnus form with Alduchov & Eskridge (1996) coefficients.
wue_tower: water-cooled chiller plant with evaporative cooling tower. All facility
  electricity (PUE per IT kWh) is rejected as heat; a fraction eps(T) is rejected
  latently (h_fg = 2.43 MJ/kg -> 1.481 L per kWh of latent heat); blowdown adds a
  factor COC/(COC-1).
wue_adiabatic: air-side economizer with direct evaporative (adiabatic) assist. Water
  is evaporated only when outdoor dry-bulb T exceeds the supply set-point T_set, to
  cool the supply air by min(T - T_set, e*(T - Twb)). Air mass per IT kWh follows
  from a server air-side temperature rise dT_srv.
"""
import numpy as np

HFG = 2.43e3   # kJ/kg
CP = 1.006     # kJ/kg/K


def rh_from_dewpoint(T, Td):
    es = lambda x: 6.1094 * np.exp(17.625 * x / (x + 243.04))
    return np.clip(100.0 * es(Td) / es(T), 5.0, 99.0)


def wet_bulb_stull(T, RH):
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def pue_t(Twb, pue0=1.10, k=0.008, tref=10.0):
    return pue0 + k * np.maximum(0.0, Twb - tref)


def wue_tower(T, pue, coc=4.0, eps0=0.55, eps_slope=0.012):
    eps = np.clip(eps0 + eps_slope * T, 0.5, 0.95)
    return coc / (coc - 1.0) * eps * pue * 3600.0 / HFG / 1.0


def wue_adiabatic(T, Twb, t_set=24.0, eff=0.85, dT_srv=12.0, blow=1.1):
    dT = np.minimum(np.maximum(T - t_set, 0.0), eff * np.maximum(T - Twb, 0.0))
    m_air = 3600.0 / (CP * dT_srv)  # kg air per IT kWh
    return blow * m_air * CP * dT / HFG  # kg (=L) per IT kWh
