"""Problem 5 - why is there more matter than antimatter?

Simulates thermal leptogenesis: solves the Boltzmann equations for a heavy neutrino N1
decaying out of equilibrium (decays, inverse decays, washout), following Buchmuller,
Di Bari & Plumacher (2005).  Output: the efficiency of the mechanism as a function of
the washout strength, the CP asymmetry needed to match the observed baryon abundance,
and the implied minimum heavy-neutrino mass.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special

from .common import val
from .plotting import BLUE, MUTED, ORANGE, figure, save


def efficiency(K: float, thermal_start: bool):
    """Final efficiency factor kappa for decay parameter K = Gamma_D / H(T = M1)."""
    n_eq = lambda z: 0.375 * z ** 2 * special.kn(2, z)
    D = lambda z: K * z * special.kn(1, z) / special.kn(2, z)
    W = lambda z: 0.25 * K * z ** 3 * special.kn(1, z)

    def rhs(z, y):
        n, kappa = y
        dn = -D(z) * (n - n_eq(z))
        return [dn, -dn - W(z) * kappa]  # kappa defined so N_B-L = -(3/4) eps kappa * (4/3)

    y0 = [n_eq(1e-2) if thermal_start else 0.0, 0.0]
    sol = integrate.solve_ivp(rhs, (1e-2, 60), y0, method="Radau", rtol=1e-8, atol=1e-12, dense_output=True)
    return sol, abs(sol.y[1, -1]) / 0.75


def run() -> dict:
    eta_obs, m_star = val("baryogenesis.eta_b"), val("baryogenesis.m_star")
    m_atm, vev = val("baryogenesis.m_atm"), val("baryogenesis.higgs_vev")
    Ks = np.geomspace(1e-2, 1e3, 26)
    kap_th = np.array([efficiency(k, True)[1] for k in Ks])
    kap_0 = np.array([efficiency(k, False)[1] for k in Ks])
    K_atm = m_atm / m_star
    kappa = efficiency(K_atm, False)[1]
    eps_needed = eta_obs / (0.96e-2 * kappa)
    # Davidson-Ibarra bound: |eps| <= 3 M1 m_atm / (16 pi v^2)
    m1_min = eps_needed * 16 * np.pi * vev ** 2 / (3 * m_atm * 1e-9)

    fig, (ax1, ax2) = figure(2)
    ax1.loglog(Ks * m_star, kap_th, color=BLUE, label="N1 starts in thermal equilibrium")
    ax1.loglog(Ks * m_star, kap_0, color=ORANGE, label="N1 starts at zero abundance")
    ax1.axvline(m_atm, color=MUTED, lw=1, ls="--")
    ax1.annotate("atmospheric\nmass scale", (m_atm, 2e-4), xytext=(5, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax1.set(xlabel="effective neutrino mass (eV)", ylabel="efficiency factor", ylim=(1e-4, 2),
            title="How much asymmetry survives washout")
    ax1.legend(loc="upper right", fontsize=8)
    sol, _ = efficiency(K_atm, False)
    z = np.geomspace(0.05, 50, 300)
    n, k = sol.sol(z)
    ax2.loglog(z, np.maximum(n, 1e-12), color=BLUE, label="heavy-neutrino abundance")
    ax2.loglog(z, 0.375 * z ** 2 * special.kn(2, z), color=MUTED, lw=1, ls="--", label="equilibrium abundance")
    ax2.loglog(z, np.maximum(np.abs(k), 1e-12), color=ORANGE, label="lepton asymmetry / CP asymmetry")
    ax2.set(xlabel="z = M1 / T  (time runs right)", ylabel="abundance per comoving volume", ylim=(1e-6, 2),
            title=f"Evolution at the atmospheric mass scale (K = {K_atm:.0f})")
    ax2.legend(loc="lower left", fontsize=8)
    path = save(fig, "p05_baryogenesis")
    return {
        "figure": path, "decay_parameter_K": float(K_atm), "efficiency": float(kappa),
        "analytic_strong_washout_estimate": float(2e-2 * (0.01 / m_atm) ** 1.1),
        "cp_asymmetry_needed": float(eps_needed), "min_heavy_neutrino_mass_GeV": float(m1_min),
        "eta_b_observed": eta_obs,
        "reading": ("With light-neutrino masses at the measured scale the mechanism sits in the strong-washout "
                    "regime: the answer no longer depends on initial conditions, "
                    f"about {100 * kappa:.1f}% of the asymmetry survives, and a CP asymmetry of {eps_needed:.1e} "
                    f"is enough. That needs a heavy neutrino above about {m1_min:.0e} GeV, far beyond any "
                    "collider, which is why the idea is plausible but hard to test."),
        "caveats": ["One flavour, decays and inverse decays only; no scatterings, flavour or spectator effects.",
                    "The mass bound assumes hierarchical heavy neutrinos (no resonant enhancement)."],
    }
