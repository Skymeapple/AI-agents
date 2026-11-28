"""Problem 10 - neutrino masses against cosmology.

Builds the allowed sum of neutrino masses from oscillation data for both mass orderings
and compares it with the cosmological upper limit and the laboratory limit, then shows
the size of the effect cosmology is looking for (suppression of small-scale structure).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .common import Cosmo, val
from .plotting import BLUE, MUTED, ORANGE, figure, save


def masses(lightest, ordering):
    d21 = val("neutrino.dm21")
    if ordering == "normal":
        m1 = lightest
        m2 = np.sqrt(m1 ** 2 + d21)
        m3 = np.sqrt(m1 ** 2 + val("neutrino.dm31_normal"))
    else:
        m3 = lightest
        m2 = np.sqrt(m3 ** 2 - val("neutrino.dm32_inverted"))
        m1 = np.sqrt(m2 ** 2 - d21)
    return m1, m2, m3


def run() -> dict:
    light = np.geomspace(1e-4, 0.5, 300)
    sums = {o: np.sum(masses(light, o), axis=0) for o in ("normal", "inverted")}
    mins = {o: float(np.sum(masses(0.0, o))) for o in sums}
    limit = val("neutrino.cosmo_limit_95")
    # Treat the cosmological posterior as a Gaussian peaked at zero and truncated at zero.
    sigma = limit / stats.norm.ppf(0.975)
    pull = {o: mins[o] / sigma for o in mins}
    # Posterior odds if both orderings are equally likely a priori and sit at their minimum.
    odds = float(np.exp(0.5 * (pull["inverted"] ** 2 - pull["normal"] ** 2)))
    c = Cosmo()
    om_h2 = c.omega_bc
    suppression = {o: float(8 * mins[o] / 93.14 / om_h2) for o in mins}
    # Effective electron-neutrino mass that KATRIN measures, at the minimum (mixing weights ~ NuFIT).
    u2 = np.array([0.68, 0.30, 0.022])
    m_beta = {o: float(np.sqrt(np.sum(u2 * np.array(masses(0.0, o)) ** 2))) for o in mins}

    fig, (ax1, ax2) = figure(2)
    ax1.loglog(light, sums["normal"], color=BLUE, label="normal ordering")
    ax1.loglog(light, sums["inverted"], color=ORANGE, label="inverted ordering")
    ax1.axhspan(limit, 2, color=MUTED, alpha=0.15, lw=0)
    ax1.annotate("excluded by DESI + CMB (95%)", (1.2e-4, limit * 1.15), color=MUTED, fontsize=8)
    ax1.set(xlabel="lightest neutrino mass (eV)", ylabel="sum of the three masses (eV)", ylim=(0.03, 1.5),
            title="What oscillations allow, against the cosmological limit")
    ax1.legend(loc="upper left", fontsize=8)

    s = np.linspace(0, 0.3, 200)
    ax2.plot(s, 100 * 8 * s / 93.14 / om_h2, color=BLUE)
    for o, col in (("normal", BLUE), ("inverted", ORANGE)):
        ax2.plot(mins[o], 100 * suppression[o], "o", color=col)
        ax2.annotate(f"minimum, {o}", (mins[o], 100 * suppression[o]), xytext=(8, -4), textcoords="offset points",
                     color=MUTED, fontsize=8)
    ax2.set(xlabel="sum of neutrino masses (eV)", ylabel="suppression of small-scale power (%)",
            title="The signal cosmology is looking for")
    path = save(fig, "p10_neutrino_mass")
    return {
        "figure": path, "minimum_sum_eV": mins, "cosmological_limit_95_eV": limit,
        "implied_gaussian_sigma_eV": float(sigma), "tension_of_minimum_sigma": pull,
        "odds_normal_vs_inverted": odds, "power_suppression_at_minimum": suppression,
        "effective_beta_decay_mass_eV": m_beta, "katrin_limit_eV": val("neutrino.katrin_limit"),
        "katrin_final_sensitivity_eV": val("neutrino.katrin_final"),
        "reading": ("The cosmological limit sits almost exactly on the smallest sum that oscillations allow. "
                    f"Inverted ordering is disfavoured at {pull['inverted']:.1f} sigma, and even normal ordering "
                    f"sits at {pull['normal']:.1f} sigma. The laboratory experiment cannot arbitrate: its final "
                    f"sensitivity ({val('neutrino.katrin_final')} eV) is {val('neutrino.katrin_final') / m_beta['inverted']:.0f} to "
                    f"{val('neutrino.katrin_final') / m_beta['normal']:.0f} times the effective mass these numbers imply."),
        "caveats": ["The cosmological posterior is approximated as a zero-centred Gaussian; the real one peaks "
                    "at negative effective mass, which makes the tension larger than computed here.",
                    "The limit assumes LCDM; with evolving dark energy it relaxes substantially.",
                    "The suppression formula (8 x neutrino fraction) is the standard linear-theory estimate."],
    }
