"""Problem 4 - which inflation model, if any?

Solves the slow-roll equations numerically for several potentials and compares the
predicted spectral tilt n_s and tensor-to-scalar ratio r with the measurements.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .common import measurements, val
from .plotting import MUTED, SERIES, figure, save

# Potentials in reduced Planck units, V(phi) up to normalisation.
MODELS = {
    "phi^2": (lambda p: p ** 2, (0.05, 40)),
    "phi^1": (lambda p: p, (0.05, 40)),
    "phi^(2/3)": (lambda p: p ** (2 / 3), (0.05, 40)),
    "Starobinsky R^2": (lambda p: (1 - np.exp(-np.sqrt(2 / 3) * p)) ** 2, (0.05, 12)),
    "natural, f = 7": (lambda p: 1 - np.cos(p / 7.0), (0.05, 7 * np.pi - 0.05)),
}


def slow_roll(V, phi):
    h = 1e-4
    v, vp, vm = V(phi), V(phi + h), V(phi - h)
    d1, d2 = (vp - vm) / (2 * h), (vp - 2 * v + vm) / h ** 2
    eps, eta = 0.5 * (d1 / v) ** 2, d2 / v
    return eps, eta, v / d1


def predict(V, bounds, n_efolds):
    lo, hi = bounds
    grid = np.linspace(lo, hi, 20000)
    eps = slow_roll(V, grid)[0]
    # inflation ends where eps = 1, approached from the flat side
    idx = np.where(np.diff(np.sign(eps - 1)))[0]
    phi_end = optimize.brentq(lambda p: slow_roll(V, p)[0] - 1, grid[idx[0]], grid[idx[0] + 1])
    flat_above = slow_roll(V, phi_end + 0.1)[0] < 1
    direction = 1 if flat_above else -1

    def efolds(phi):
        xs = np.linspace(phi_end, phi, 4000)
        return direction * integrate.simpson(slow_roll(V, xs)[2], x=xs)

    far = hi if direction > 0 else lo
    phi_n = optimize.brentq(lambda p: efolds(p) - n_efolds, phi_end + direction * 1e-3, far)
    e, eta, _ = slow_roll(V, phi_n)
    return 1 - 6 * e + 2 * eta, 16 * e


def run() -> dict:
    ns_planck, ns_act = measurements()["cmb"]["planck_ns"], measurements()["cmb"]["pact_lb_ns"]
    r_max = val("cmb.r_upper_95")
    n_lo, n_hi = measurements()["inflation"]["efolds"]["value"]
    rows = []
    fig, ax = figure(1, height=4.6)
    for (name, (V, bounds)), col in zip(MODELS.items(), SERIES):
        try:
            pts = [predict(V, bounds, n) for n in (n_lo, n_hi)]
        except ValueError:
            continue
        ns, r = zip(*pts)
        ax.plot(ns, r, "-o", color=col, label=name, mfc="white")
        ax.plot(ns[1], r[1], "o", color=col)
        mid_ns, mid_r = np.mean(ns), np.mean(r)
        rows.append({"model": name, "ns_N50": ns[0], "ns_N60": ns[1], "r_N50": r[0], "r_N60": r[1],
                     "allowed_by_r_limit": bool(r[1] < r_max),
                     "ns_pull_planck_sigma": float((mid_ns - ns_planck["value"]) / ns_planck["err"]),
                     "ns_pull_act_sigma": float((mid_ns - ns_act["value"]) / ns_act["err"]),
                     "energy_scale_1e16GeV": float(1.04 * (mid_r / 0.01) ** 0.25)})
    ax.axhline(r_max, color=MUTED, lw=1, ls="--")
    ax.annotate("gravitational-wave limit (95%): excluded above", (0.9455, r_max), xytext=(0, -11),
                textcoords="offset points", color=MUTED, fontsize=8)
    for m, lab, y in ((ns_planck, "Planck 2018", 2.2e-4), (ns_act, "ACT+Planck+DESI", 4.5e-4)):
        ax.errorbar(m["value"], y, xerr=m["err"], fmt="s", color=MUTED, lw=1.5, ms=4)
        ax.annotate(lab, (m["value"] + m["err"], y), xytext=(5, -3), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set(yscale="log", ylim=(1e-4, 0.5), xlim=(0.945, 0.982), xlabel="spectral tilt n_s",
           ylabel="tensor-to-scalar ratio r",
           title="Slow-roll predictions (open: 50 e-folds, filled: 60) against data")
    ax.legend(loc="center left", fontsize=8)
    path = save(fig, "p04_inflation")
    return {
        "figure": path, "models": rows, "r_limit": r_max,
        "reading": ("Power-law potentials are excluded by the absence of gravitational waves. Plateau models "
                    "such as Starobinsky survive that test but predict a tilt that sits below the newer "
                    "ACT-based measurement, so the surviving class is under mild pressure from the other side."),
        "caveats": ["First-order slow-roll; the number of e-folds is an assumed bracket, not derived from reheating.",
                    "Measured tilt values are quoted from memory and should be checked before reuse."],
    }
