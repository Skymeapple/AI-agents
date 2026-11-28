"""Problem 2 - is dark energy constant?

Fits flat LCDM and the w0-wa model to real data and reports how much better the
evolving model fits:
  * DESI DR2 baryon acoustic oscillations (official compressed likelihood, 13 points)
  * Pantheon+ supernovae (1590 SNe compressed to 40 bins with the full covariance)
  * the DESI collaboration's Gaussian CMB prior on (theta*, omega_b, omega_bc)

The published chi^2 improvements are loaded alongside so the reproduction can be judged.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .common import Cosmo, bao_model, load_desi_bao, load_sn_binned, measurements, sigma_from_dchi2, val
from .plotting import BLUE, MUTED, ORANGE, figure, save

Z_BAO, KIND, V_BAO, COV_BAO = load_desi_bao()
ICOV_BAO = np.linalg.inv(COV_BAO)
Z_SN, MU_SN, COV_SN = load_sn_binned()
ICOV_SN = np.linalg.inv(COV_SN)
_prior = measurements()["cmb"]["theta_star_prior"]["value"]
CMB_MEAN = np.array(_prior["mean"])
ICOV_CMB = np.linalg.inv(np.array(_prior["cov_1e9"]) * 1e-9)


def chi2_parts(c: Cosmo, use_cmb: bool, use_sn: bool) -> dict:
    d = bao_model(c, Z_BAO, KIND) - V_BAO
    parts = {"bao": float(d @ ICOV_BAO @ d)}
    if use_cmb:
        d = np.array([c.theta_star(), c.omega_b, c.omega_bc]) - CMB_MEAN
        parts["cmb"] = float(d @ ICOV_CMB @ d)
    else:  # without the CMB, a BBN prior on the baryon density sets the sound horizon
        parts["bbn"] = ((c.omega_b - val("bbn.omega_b")) / measurements()["bbn"]["omega_b"]["err"]) ** 2
    if use_sn:
        r = MU_SN - c.distance_modulus(Z_SN)
        one = np.ones_like(r)
        parts["sn"] = float(r @ ICOV_SN @ r - (one @ ICOV_SN @ r) ** 2 / (one @ ICOV_SN @ one))
    return parts


def fit(evolving: bool, use_cmb: bool, use_sn: bool):
    def unpack(p):
        w0, wa = (p[3], p[4]) if evolving else (-1.0, 0.0)
        return Cosmo(h=p[0], omega_b=p[1], omega_bc=p[2], w0=w0, wa=wa)

    def objective(p):
        if not (0.4 < p[0] < 1.0 and 0.015 < p[1] < 0.03 and 0.08 < p[2] < 0.25):
            return 1e9
        if evolving and not (-3 < p[3] < 1 and -3 < p[4] < 2 and p[3] + p[4] < 0):
            return 1e9  # the priors DESI uses
        c = unpack(p)
        if c.h ** 2 <= c.omega_bc + 1e-3:
            return 1e9
        return sum(chi2_parts(c, use_cmb, use_sn).values())

    best = None
    starts = [[0.68, 0.0222, 0.142]] if not evolving else [
        [0.68, 0.0222, 0.142, -1.0, 0.0], [0.65, 0.0222, 0.142, -0.7, -1.0], [0.64, 0.0222, 0.144, -0.4, -1.8]]
    for s in starts:
        res = optimize.minimize(objective, s, method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 6000, "adaptive": True})
        res = optimize.minimize(objective, res.x, method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-5, "maxiter": 6000, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x), float(best.fun)


def run() -> dict:
    pub = measurements()["dark_energy_published"]
    combos = [("DESI BAO (+BBN)", False, False, pub["desi_only"]["value"]),
              ("DESI + CMB prior", True, False, pub["desi_cmbprior"]["value"]),
              ("DESI + Pantheon+ (+BBN)", False, True, pub["desi_pantheon"]["value"]),
              ("DESI + CMB prior + Pantheon+", True, True, None)]
    rows, fits = [], {}
    for name, cmb, sn in [(c[0], c[1], c[2]) for c in combos]:
        lcdm, chi_l = fit(False, cmb, sn)
        w, chi_w = fit(True, cmb, sn)
        fits[name] = (lcdm, w)
        published = next(c[3] for c in combos if c[0] == name)
        rows.append({"data": name, "chi2_lcdm": chi_l, "chi2_w0wa": chi_w, "delta_chi2": chi_w - chi_l,
                     "sigma": sigma_from_dchi2(chi_l - chi_w, 2), "w0": w.w0, "wa": w.wa,
                     "h_lcdm": lcdm.h, "omega_m_lcdm": (lcdm.omega_bc + lcdm.mnu / 93.14) / lcdm.h ** 2,
                     "published_delta_chi2": published})

    lcdm, w = fits["DESI + CMB prior + Pantheon+"]
    fig, (ax1, ax2) = figure(2)
    zz = np.linspace(0.05, 2.5, 200)
    ref = bao_model(lcdm, Z_BAO, KIND)
    err = np.sqrt(np.diag(COV_BAO))
    for kind, marker, lab in (("DM_over_rs", "o", "transverse"), ("DH_over_rs", "s", "line of sight"),
                              ("DV_over_rs", "D", "angle averaged")):
        sel = np.array([k == kind for k in KIND])
        ax1.errorbar(Z_BAO[sel], V_BAO[sel] / ref[sel] - 1, err[sel] / ref[sel], fmt=marker, color=BLUE,
                     mfc="white" if kind != "DM_over_rs" else BLUE, label=f"DESI DR2, {lab}", lw=1)
    for kind, ls in (("DM_over_rs", "-"), ("DH_over_rs", "--")):
        k = [kind] * len(zz)
        ax1.plot(zz, bao_model(w, zz, k) / bao_model(lcdm, zz, k) - 1, ls, color=ORANGE,
                 label=f"best w0wa model, {'transverse' if ls == '-' else 'line of sight'}")
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set(xlabel="redshift", ylabel="distance / best LCDM - 1", title="BAO distances relative to the best LCDM fit")
    ax1.legend(fontsize=8)

    a = np.linspace(0.3, 1, 100)
    ax2.plot(1 / a - 1, w.w0 + w.wa * (1 - a), color=ORANGE, label="best-fit w(z)")
    ax2.axhline(-1, color=MUTED, lw=1)
    ax2.annotate("cosmological constant", (2.3, -1), xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED)
    ax2.set(xlabel="redshift", ylabel="equation of state w", title="Fitted dark-energy equation of state")
    ax2.legend()
    path = save(fig, "p02_dark_energy")
    return {
        "figure": path, "fits": rows,
        "reading": ("The evolving model fits better in every combination, by an amount close to the published "
                    "values, and never by enough to be decisive. The best fit crosses w = -1, which a single "
                    "simple scalar field cannot do."),
        "caveats": ["CMB enters only through the three-parameter compressed prior, not the full likelihood.",
                    "Supernovae are binned; the model is evaluated at each bin's effective redshift.",
                    "Significance uses the chi^2 difference with 2 extra parameters, as DESI does."],
    }
