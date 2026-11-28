"""Problem 9 - the missing lithium.

  (a) simulates the neutron-to-proton freeze-out in the first minutes and predicts the
      primordial helium fraction, to show the nucleosynthesis framework works;
  (b) compares predicted and observed deuterium and lithium-7 across the baryon density,
      to show that no single baryon density reconciles lithium;
  (c) asks how much of the gap the measured stellar depletion closes.

The helium calculation is a real (simplified) integration.  Deuterium and lithium use
published power-law scalings around the standard prediction, because a full reaction
network is beyond a short script.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .common import measurements, val
from .plotting import BLUE, MUTED, ORANGE, figure, save

Q = 1.2933  # neutron-proton mass difference, MeV


X_D_THRESHOLD = 1e-3  # assumed: fusion proceeds once equilibrium deuterium reaches this mass fraction


def helium_fraction(eta10: float, tau_n: float, threshold: float = X_D_THRESHOLD):
    """Y_p from n/p freeze-out plus neutron decay until the deuterium bottleneck opens."""
    def rhs(lnT, y):
        T = np.exp(lnT)
        x = Q / T
        lam_np = 255.0 / (tau_n * x ** 5) * (12 + 6 * x + x ** 2)  # Bernstein et al. 1989
        lam_pn = lam_np * np.exp(-x)
        # dt/dlnT = -2 t  with t = k / T^2 ; k drifts 0.74 -> 1.32 s MeV^2 across e+e- annihilation
        k = 0.74 + (1.32 - 0.74) / (1 + (T / 0.25) ** 3)
        dt = -2 * k / T ** 2
        return [dt * (lam_pn * (1 - y[0]) - lam_np * y[0] - y[0] / tau_n)]

    # Deuterium bottleneck: equilibrium X_D = 16.3 (T/m_N)^1.5 eta exp(B/T) X_n X_p, with X_n X_p ~ 0.1.
    eta = eta10 * 1e-10
    t_nuc = optimize.brentq(lambda T: np.log(16.3 * 0.1 * eta) + 1.5 * np.log(T / 938.0) + 2.224 / T
                            - np.log(threshold), 0.03, 0.2)
    sol = integrate.solve_ivp(rhs, (np.log(30.0), np.log(t_nuc)), [1 / (1 + np.exp(Q / 30.0))], rtol=1e-8, atol=1e-10,
                              dense_output=True)
    xn = sol.y[0, -1]
    return 2 * xn, t_nuc, sol


def run() -> dict:
    b = measurements()["bbn"]
    eta10_cmb = 273.9 * 0.02237
    tau = val("bbn.neutron_lifetime")
    yp, t_nuc, sol = helium_fraction(eta10_cmb, tau)
    yp_range = [helium_fraction(eta10_cmb, tau, th)[0] for th in (1e-4, 1e-2)]

    eta = np.linspace(2, 10, 200)
    dh = 2.53e-5 * (6.1 / eta) ** 1.6                       # scaling, e.g. Cyburt et al. 2016
    li = b["li7_predicted"]["value"] * (eta / 6.1) ** 2.1
    li_obs, li_pred = b["li7_observed"], b["li7_predicted"]
    gap = li_pred["value"] / li_obs["value"]
    eta_li = 6.1 * (li_obs["value"] / li_pred["value"]) ** (1 / 2.1)
    dh_at_eta_li = 2.53e-5 * (6.1 / eta_li) ** 1.6
    corrected = li_obs["value"] * (1 + val("bbn.cluster_initial_excess"))
    residual = li_pred["value"] / corrected
    sig = (li_pred["value"] - corrected) / np.hypot(li_pred["err"], li_obs["err"] * (1 + val("bbn.cluster_initial_excess")))

    fig, (ax1, ax2) = figure(2)
    T = np.geomspace(30, t_nuc, 200)
    xn = sol.sol(np.log(T))[0]
    ax1.loglog(T, xn, color=BLUE, label="simulated neutron fraction")
    ax1.loglog(T, 1 / (1 + np.exp(Q / T)), color=MUTED, lw=1, ls="--", label="if equilibrium held")
    ax1.invert_xaxis()
    ax1.set(xlabel="temperature (MeV, time runs right)", ylabel="neutron fraction", ylim=(0.01, 1),
            title=f"Neutron freeze-out: predicted helium fraction {yp:.3f}")
    ax1.legend(loc="lower left", fontsize=8)

    ax2.plot(eta, li / li_obs["value"], color=BLUE, label="lithium-7")
    ax2.plot(eta, dh / b["dh_observed"]["value"], color=ORANGE, label="deuterium")
    ax2.axhline(1, color=MUTED, lw=1)
    ax2.annotate("agreement", (9.9, 1), xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    ax2.axvline(eta10_cmb, color=MUTED, lw=1, ls="--")
    ax2.annotate("baryon density\nfrom the CMB", (eta10_cmb, 0.22), xytext=(5, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax2.set(yscale="log", ylim=(0.2, 10), xlabel="baryon-to-photon ratio (x 1e-10)", ylabel="predicted / observed abundance",
            title="No baryon density fits deuterium and lithium together")
    ax2.legend(loc="upper center", fontsize=8)
    path = save(fig, "p09_lithium")
    return {
        "figure": path,
        "helium_fraction_simulated": float(yp), "helium_fraction_threshold_range": [float(v) for v in yp_range],
        "helium_fraction_observed": b["yp_observed"]["value"],
        "nucleosynthesis_temperature_MeV": float(t_nuc),
        "lithium_gap_factor": float(gap),
        "eta10_that_would_fit_lithium": float(eta_li),
        "deuterium_at_that_eta": float(dh_at_eta_li), "deuterium_observed": b["dh_observed"]["value"],
        "lithium_after_measured_depletion": float(corrected),
        "residual_gap_factor_after_depletion": float(residual),
        "residual_sigma": float(sig),
        "reading": (f"The simplified freeze-out integration gives a helium fraction of {yp:.2f} against "
                    f"{b['yp_observed']['value']} observed (full reaction-network codes give 0.247), so the framework "
                    f"works; yet it overpredicts lithium by a factor of {gap:.1f}. Lowering the baryon density enough to fix lithium would more than double "
                    "deuterium, which is measured to a percent, so the fix cannot be in the cosmology. The "
                    "depletion measured in globular-cluster stars closes about half of the gap in the log; a "
                    f"factor of {residual:.1f} remains, at {sig:.1f} sigma given the quoted errors."),
        "caveats": ["Helium comes from a simplified freeze-out integration; the moment fusion starts is set by "
                    "an assumed deuterium threshold, and the quoted range shows the yield for thresholds ten "
                    "times lower and higher.",
                    "Deuterium and lithium curves are power-law scalings anchored to published predictions, "
                    "not a reaction-network calculation."],
    }
