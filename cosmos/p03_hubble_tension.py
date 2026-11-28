"""Problem 3 - the Hubble tension.

  (a) puts the measurements side by side and computes the pairwise tension;
  (b) shows the sound-horizon degeneracy with real DESI data: BAO fix H0 x r_d, so the
      only way to H0 = 73 is a smaller sound horizon;
  (c) a toy early-dark-energy model: how much extra energy before recombination is
      needed to shrink the sound horizon that far.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .common import Cosmo, bao_model, load_desi_bao, measurements
from .plotting import AQUA, BLUE, MUTED, ORANGE, figure, save

Z_BAO, KIND, V_BAO, COV_BAO = load_desi_bao()
ICOV = np.linalg.inv(COV_BAO)
KIND_COLOR = {"early": BLUE, "ladder": ORANGE, "independent": AQUA}
KIND_LABEL = {"early": "early universe + LCDM", "ladder": "distance ladder", "independent": "independent of both"}


def bao_h_rd():
    """Fit (Omega_m, h*r_d) to DESI BAO alone: the product is what BAO measure."""
    def chi2(p):
        om, hrd = p
        if not 0.1 < om < 0.6:
            return 1e9
        c = Cosmo(h=0.7, omega_bc=om * 0.49 - 0.06 / 93.14)
        d = bao_model(c, Z_BAO, KIND, rd=hrd / 0.7) - V_BAO
        return d @ ICOV @ d
    res = optimize.minimize(chi2, [0.3, 100.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6})
    # one-sigma on h*r_d from the chi2 = min + 1 profile
    def prof(hrd):
        return optimize.minimize_scalar(lambda om: chi2([om, hrd]), bounds=(0.2, 0.4), method="bounded").fun - res.fun - 1
    hi = optimize.brentq(prof, res.x[1], res.x[1] + 3)
    return res.x[0], res.x[1], hi - res.x[1]


def ede_scan():
    base = Cosmo()
    rd0 = base.sound_horizon(1059.94)
    f = np.linspace(0, 0.4, 41)
    rd = np.array([Cosmo(f_ede=x).sound_horizon(1059.94) for x in f])
    return f, rd / rd0


def run() -> dict:
    data = measurements()["h0"]
    by = {d["name"]: d for d in data}
    planck, shoes = by["Planck 2018 (LCDM)"], by["SH0ES Cepheids (HST)"]
    tension = lambda a, b: abs(a["value"] - b["value"]) / np.hypot(a["err"], b["err"])
    groups = {}
    for kind in KIND_LABEL:
        sel = [d for d in data if d["kind"] == kind]
        w = np.array([1 / d["err"] ** 2 for d in sel])
        groups[kind] = {"mean": float(np.sum(w * [d["value"] for d in sel]) / w.sum()), "err": float(w.sum() ** -0.5)}

    om, hrd, hrd_err = bao_h_rd()
    f, ratio = ede_scan()
    rd_lcdm = Cosmo().r_drag()
    h0_from_bao = 100 * hrd / rd_lcdm
    rd_needed = 100 * hrd / shoes["value"]
    f_needed = float(np.interp(rd_needed / rd_lcdm, ratio[::-1], f[::-1]))

    fig, (ax1, ax2) = figure(2, height=4.6)
    for i, d in enumerate(data):
        y = len(data) - i
        ax1.errorbar(d["value"], y, xerr=d["err"], fmt="o", color=KIND_COLOR[d["kind"]], lw=1.5)
        ax1.annotate(d["name"], (62.2, y), va="center", fontsize=8)
    for kind, col in KIND_COLOR.items():
        ax1.plot([], [], "o", color=col, label=KIND_LABEL[kind])
    ax1.set(xlim=(62, 80), yticks=[], xlabel="H0 (km/s/Mpc)", title="Measurements of the expansion rate")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="upper right", fontsize=8)

    rd = np.linspace(132, 152, 50)
    ax2.plot(rd, 100 * hrd / rd, color=BLUE, label="DESI DR2 BAO: H0 x r_d fixed")
    ax2.fill_between(rd, 100 * (hrd - hrd_err) / rd, 100 * (hrd + hrd_err) / rd, color=BLUE, alpha=0.2, lw=0)
    ax2.plot(rd_lcdm, h0_from_bao, "o", color=MUTED)
    ax2.annotate("standard sound horizon", (rd_lcdm, h0_from_bao), xytext=(-6, -14), textcoords="offset points",
                 ha="right", color=MUTED, fontsize=8)
    ax2.plot(rd_needed, shoes["value"], "o", color=ORANGE)
    ax2.annotate(f"needed for H0 = {shoes['value']}", (rd_needed, shoes["value"]), xytext=(8, 2),
                 textcoords="offset points", color=MUTED, fontsize=8)
    ax2.set(xlabel="sound horizon r_d (Mpc)", ylabel="H0 (km/s/Mpc)", title="Raising H0 requires a smaller sound horizon")
    ax2.legend(loc="upper right", fontsize=8)
    path = save(fig, "p03_hubble_tension")
    return {
        "figure": path,
        "tension_planck_vs_shoes_sigma": float(tension(planck, shoes)),
        "group_means": groups,
        "tension_early_vs_ladder_sigma": float(abs(groups["early"]["mean"] - groups["ladder"]["mean"])
                                               / np.hypot(groups["early"]["err"], groups["ladder"]["err"])),
        "bao_only": {"omega_m": float(om), "h_times_rd_mpc": float(hrd), "err": float(hrd_err)},
        "h0_with_standard_rd": float(h0_from_bao),
        "rd_standard_mpc": float(rd_lcdm),
        "rd_needed_for_shoes_mpc": float(rd_needed),
        "rd_shrink_percent": float(100 * (1 - rd_needed / rd_lcdm)),
        "toy_ede_peak_fraction_needed": f_needed,
        "reading": ("BAO alone pin the product of H0 and the sound horizon. Matching the Cepheid value means "
                    f"shrinking the sound horizon by {100 * (1 - rd_needed / rd_lcdm):.1f}%, which in the toy model "
                    f"takes a new component peaking at {100 * f_needed:.0f}% of the energy density around redshift "
                    "3500. Published early-dark-energy fits need less (about 10%) because they also let the "
                    "matter density and other parameters move."),
        "caveats": ["Grouped means treat measurements as independent; ladder results share calibrators, so the "
                    "ladder group error is too small.",
                    "The early-dark-energy model is a Gaussian bump in the energy budget, not a scalar field, "
                    "and ignores the changes to the CMB peaks that a full analysis must also fit."],
    }
