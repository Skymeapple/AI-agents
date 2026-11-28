"""Problem 1 - dark matter.

Two simulations:
  (a) a rotation curve for a synthetic Milky-Way-like galaxy under three hypotheses:
      visible matter only, visible matter + an NFW dark halo, and MOND;
  (b) the expected number of WIMP recoils in a liquid-xenon detector, to show what
      "a limit of 2.2e-48 cm^2" means in events.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special

from .common import Cosmo, measurements, val
from .plotting import BLUE, MUTED, ORANGE, AQUA, figure, save

G = 4.30091e-6  # kpc (km/s)^2 / Msun
KPC_M = 3.0857e19


def rotation_curves():
    g = measurements()["dark_matter"]["galaxy"]["value"]
    r = np.linspace(0.3, 60, 300)
    # Exponential disk (Freeman 1970) and Hernquist bulge.
    y = r / (2 * g["disk_scale"])
    v2_disk = (4 * np.pi * G * g["disk_mass"] / (2 * np.pi * g["disk_scale"] ** 2) * g["disk_scale"] * y ** 2
               * (special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y)))
    v2_bulge = G * g["bulge_mass"] * r / (r + g["bulge_scale"]) ** 2
    v2_bar = v2_disk + v2_bulge
    # NFW halo defined by M200 and concentration.
    rho_crit = 3 * Cosmo().H(0) ** 2 / (8 * np.pi * G) * 1e-6  # Msun / kpc^3
    r200 = (3 * g["halo_m200"] / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)
    rs = r200 / g["halo_c"]
    m = lambda x: np.log(1 + x) - x / (1 + x)
    v2_halo = G * g["halo_m200"] * m(r / rs) / m(g["halo_c"]) / r
    # MOND, "simple" interpolating function.
    a0 = val("dark_matter.mond_a0") * KPC_M / 1e6  # (km/s)^2 / kpc
    g_n = v2_bar / r
    g_mond = g_n / 2 + np.sqrt(g_n ** 2 / 4 + g_n * a0)
    curves = {"visible matter only": np.sqrt(v2_bar), "visible + NFW halo": np.sqrt(v2_bar + v2_halo),
              "MOND": np.sqrt(g_mond * r)}
    return r, curves, r200


def wimp_rate(m_chi, sigma_n, energies_kev):
    """Differential rate on xenon, events / (kg day keV). Standard halo, Helm form factor."""
    A, amu = 131.29, 0.931494
    m_n, m_p = A * amu, 0.938272
    mu_n = m_chi * m_p / (m_chi + m_p)
    mu_N = m_chi * m_n / (m_chi + m_n)
    e_gev = energies_kev * 1e-6
    c = 2.99792458e10
    v0, ve = val("dark_matter.v0") * 1e5, val("dark_matter.v_earth") * 1e5
    vmin = np.sqrt(m_n * e_gev / (2 * mu_N ** 2)) * c
    eta = (special.erf((vmin + ve) / v0) - special.erf((vmin - ve) / v0)) / (2 * ve)  # s/cm
    q = np.sqrt(2 * m_n * e_gev) / 0.1973269  # fm^-1
    s, a = 0.9, 0.52
    cc = 1.23 * A ** (1 / 3) - 0.6
    rn = np.sqrt(cc ** 2 + 7 / 3 * np.pi ** 2 * a ** 2 - 5 * s ** 2)
    x = q * rn
    form = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3 * np.exp(-(q * s) ** 2 / 2)
    rate = val("dark_matter.local_density") * sigma_n * A ** 2 * form ** 2 * eta * c ** 2 / (2 * m_chi * mu_n ** 2)
    return rate * 5.6096e26 * 86400 * 1e-6  # per GeV(target) s GeV(recoil) -> per kg day keV


def run() -> dict:
    r, curves, r200 = rotation_curves()
    fig, (ax1, ax2) = figure(2)
    for (name, v), col in zip(curves.items(), (MUTED, BLUE, ORANGE)):
        ax1.plot(r, v, color=col, label=name)
    ax1.set(xlabel="radius (kpc)", ylabel="circular speed (km/s)", ylim=(0, 280),
            title="Rotation curve of a synthetic Milky-Way-like galaxy")
    ax1.legend(loc="lower right")

    e = np.linspace(1, 60, 400)
    sigma, eff = val("dark_matter.lz_limit_sigma"), val("dark_matter.efficiency")
    out = {}
    for mass, col in ((40.0, BLUE), (200.0, ORANGE), (1000.0, AQUA)):
        rate = wimp_rate(mass, sigma, e)
        ax2.plot(e, rate * 1000 * 365.25, color=col, label=f"{mass:.0f} GeV")
        window = (e >= 5) & (e <= 50)
        per_ty = integrate.simpson(rate[window], x=e[window]) * 1000 * 365.25 * eff
        out[f"events_per_tonne_year_m{mass:.0f}"] = per_ty
    ax2.set(yscale="log", xlabel="nuclear recoil energy (keV)", ylabel="events / (tonne year keV)",
            title=f"WIMP recoil spectrum on xenon at {sigma:.1e} cm$^2$")
    ax2.legend(title="WIMP mass")
    path = save(fig, "p01_dark_matter")

    expo = val("dark_matter.lz_exposure")
    i30 = np.argmin(abs(r - 30))
    return {
        "figure": path,
        "v_at_30kpc_kms": {k: float(v[i30]) for k, v in curves.items()},
        "halo_r200_kpc": float(r200),
        "expected_events_at_lz_limit_40GeV": out["events_per_tonne_year_m40"] * expo,
        "events_per_tonne_year_at_limit": out,
        "reading": ("Visible matter alone falls away with radius; a halo or MOND both keep the curve flat, so "
                    "rotation curves alone cannot choose between them. At the published LZ limit a 40 GeV WIMP "
                    f"would give about {out['events_per_tonne_year_m40'] * expo:.1f} events in the whole exposure: "
                    "limits of this size are statements about one or two events, which is why a single "
                    "candidate event cannot be a discovery."),
        "caveats": ["Galaxy is synthetic (assumed parameters), not a fit to SPARC data.",
                    "Flat 50% detection efficiency and no escape-velocity cut-off."],
    }
