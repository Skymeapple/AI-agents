"""Problem 8 - is structure too smooth, and is the universe lopsided?

  (a) S8 comparison across surveys, plus a linear growth calculation showing how much
      the clumpiness changes if dark energy evolves as fitted in problem 2;
  (b) a Monte Carlo of the quasar number-count dipole: mock skies are generated with
      exactly the dipole our motion through the CMB should produce, observed through
      the same Galactic mask, to see how often the measured amplitude would arise.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats

from .common import C_KMS, Cosmo, measurements, val
from .plotting import BLUE, MUTED, ORANGE, figure, save

N_MOCKS = 4000


def growth_factor(c: Cosmo, z=0.0):
    """Linear growth D(z), normalised to D = a deep in matter domination."""
    def rhs(lna, y):
        a = np.exp(lna)
        zz = 1 / a - 1
        e2 = c.E(zz) ** 2
        de = 1e-4
        dlnh = (np.log(c.E(1 / np.exp(lna + de) - 1)) - np.log(c.E(1 / np.exp(lna - de) - 1))) / (2 * de)
        om = c.omega_bc * (1 + zz) ** 3 / e2
        return [y[1], -(2 + dlnh) * y[1] + 1.5 * om * y[0]]
    a0 = 1e-3
    sol = integrate.solve_ivp(rhs, (np.log(a0), np.log(1 / (1 + z))), [a0, a0], rtol=1e-8, atol=1e-10)
    return sol.y[0, -1]


def dipole_mocks(rng, amplitude, n_sources, cut_deg):
    """Poisson skies on an equal-area grid, masked, fitted for monopole + dipole."""
    nphi, nmu = 96, 48
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    mu = -1 + (np.arange(nmu) + 0.5) * 2 / nmu
    P, M = np.meshgrid(phi, mu)
    s = np.sqrt(1 - M ** 2)
    n = np.stack([s * np.cos(P), s * np.sin(P), M], axis=-1).reshape(-1, 3)
    n = n[np.abs(n[:, 2]) > np.sin(np.radians(cut_deg))]  # Galactic-latitude cut, z = Galactic pole
    # The CMB dipole points to Galactic (l, b) = (264.02, 48.25) deg.
    l, b = np.radians(264.02), np.radians(48.25)
    d_hat = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    mean = n_sources / len(n) * (1 + amplitude * n @ d_hat)
    A = np.column_stack([np.ones(len(n)), n])
    pinv = np.linalg.pinv(A)
    counts = rng.poisson(mean, size=(N_MOCKS, len(n)))
    coef = counts @ pinv.T
    return np.linalg.norm(coef[:, 1:], axis=1) / coef[:, 0]


def run() -> dict:
    s8 = measurements()["structure"]["s8"]
    planck = s8[0]
    rows = [{"name": d["name"], "S8": d["value"], "err": d["err"],
             "tension_with_planck_sigma": float((planck["value"] - d["value"]) / np.hypot(planck["err"], d["err"]))}
            for d in s8]
    lcdm = Cosmo()
    evolving = Cosmo(h=0.667, w0=-0.84, wa=-0.6)  # representative of the problem-2 fits
    growth_ratio = growth_factor(evolving) / growth_factor(lcdm)

    beta = val("structure.cmb_velocity") / C_KMS
    x, alpha = val("structure.quasar_x"), val("structure.quasar_alpha")
    expected = (2 + x * (1 + alpha)) * beta
    observed, n_src = val("structure.dipole_observed"), val("structure.dipole_n_sources")
    rng = np.random.default_rng(8)
    amps = dipole_mocks(rng, expected, n_src, val("structure.galactic_cut_deg"))
    n_exceed = int(np.sum(amps >= observed))
    z_gauss = float((observed - amps.mean()) / amps.std())

    fig, (ax1, ax2) = figure(2)
    for i, d in enumerate(s8):
        y = len(s8) - i
        ax1.errorbar(d["value"], y, xerr=d["err"], fmt="o", color=BLUE if i else ORANGE, lw=1.5)
        ax1.annotate(d["name"], (0.705, y), va="center", fontsize=8)
    ax1.axvspan(planck["value"] - planck["err"], planck["value"] + planck["err"], color=ORANGE, alpha=0.12, lw=0)
    ax1.set(xlim=(0.70, 0.87), yticks=[], xlabel="S8 (clumpiness of matter)", title="Lensing surveys against the CMB prediction")
    ax1.grid(axis="y", visible=False)

    ax2.hist(amps * 100, bins=50, color=BLUE, alpha=0.85, label=f"{N_MOCKS} mock skies, motion only")
    ax2.axvline(observed * 100, color=ORANGE)
    ax2.annotate("measured", (observed * 100, ax2.get_ylim()[1] * 0.9), xytext=(-5, 0), textcoords="offset points",
                 ha="right", color=MUTED)
    ax2.set(xlabel="dipole amplitude (%)", ylabel="mock skies", title="Quasar-count dipole: expectation against measurement")
    ax2.legend(loc="upper center", fontsize=8)
    path = save(fig, "p08_structure_and_dipole")
    return {
        "figure": path, "s8": rows,
        "growth_today_evolving_de_over_lcdm": float(growth_ratio),
        "dipole_expected_from_motion": float(expected), "dipole_observed": observed,
        "dipole_ratio": float(observed / expected),
        "mock_mean": float(amps.mean()), "mock_std": float(amps.std()),
        "mocks_exceeding_observed": n_exceed, "n_mocks": N_MOCKS,
        "significance_sigma_gaussian": z_gauss,
        "published_significance_sigma": val("structure.dipole_published_sigma"),
        "reading": ("The lensing discrepancy is now survey-dependent and at most "
                    f"{max(r['tension_with_planck_sigma'] for r in rows):.1f} sigma. The dipole is a different "
                    f"matter: {n_exceed} of {N_MOCKS} mock skies reach the measured amplitude, and the excess is "
                    f"{z_gauss:.1f} standard deviations of the mock scatter (published: "
                    f"{val('structure.dipole_published_sigma')}). If the catalogue is sound, our motion does not "
                    "explain the lopsidedness of the distant universe. Evolving dark energy as fitted in "
                    f"problem 2 changes today's growth by only {100 * (growth_ratio - 1):+.1f}%, so it does not "
                    "by itself explain a low S8."),
        "caveats": ["Mocks contain shot noise and the mask only: no clustering, no survey systematics. The test "
                    "shows the excess is not a noise or masking artefact, not that the catalogue is clean.",
                    "Source-count slope and spectral index are quoted from memory."],
    }
