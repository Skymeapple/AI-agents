"""Problem 7 - how did giant black holes form so early?

Grows black-hole seeds by gas accretion from an assumed formation epoch and asks, for
each observed early black hole, what seed mass or what accretion rate is required.
"""
from __future__ import annotations

import numpy as np

from .common import Cosmo, measurements
from .plotting import AQUA, BLUE, INK, MUTED, ORANGE, figure, save

T_EDD_MYR = 450.0      # sigma_T c / (4 pi G m_p)
EFFICIENCY = 0.1       # radiative efficiency (assumed, thin-disc value)
Z_SEED = 25.0          # assumed seed formation redshift
SEEDS = {"stellar remnant, 100 Msun": 1e2, "dense-cluster seed, 1e3 Msun": 1e3, "direct collapse, 1e5 Msun": 1e5}


def efold_time(eddington_ratio=1.0, duty=1.0):
    return T_EDD_MYR * EFFICIENCY / (1 - EFFICIENCY) / (eddington_ratio * duty)


def run() -> dict:
    cosmo = Cosmo()
    age = lambda z: cosmo.age_gyr(z) * 1000  # Myr
    t0 = age(Z_SEED)
    objs = measurements()["early_black_holes"]
    rows = []
    for o in objs:
        dt = age(o["z"]) - t0
        n_efolds_available = dt / efold_time()
        rows.append({
            "name": o["name"], "z": o["z"], "mass_msun": o["mass"], "time_available_myr": dt,
            "seed_needed_at_eddington_msun": o["mass"] / np.exp(n_efolds_available),
            "mean_eddington_ratio_needed": {k: float(np.log(o["mass"] / m) / n_efolds_available) for k, m in SEEDS.items()},
        })

    fig, ax = figure(1, height=4.6)
    zz = np.linspace(Z_SEED, 6, 200)
    tt = np.array([age(z) for z in zz])
    for (name, m), col in zip(SEEDS.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(tt, m * np.exp((tt - t0) / efold_time()), color=col, label=name)
    for o in objs:
        ax.plot(age(o["z"]), o["mass"], "o", color=INK, mfc="white", mew=1.5)
        dx, dy, ha = {"J0313-1806": (-6, 5, "right"), "J1342+0928": (-6, -9, "right")}.get(o["name"], (6, -3, "left"))
        ax.annotate(o["name"], (age(o["z"]), o["mass"]), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    color=MUTED, ha=ha)
    ax.set(yscale="log", ylim=(50, 1e11), xlim=(100, 1000), xlabel="age of the universe (Myr)",
           ylabel="black-hole mass (solar masses)",
           title=f"Uninterrupted Eddington-limited growth from z = {Z_SEED:.0f}")
    ax.legend(loc="upper left", fontsize=8, title="seed")
    path = save(fig, "p07_early_black_holes")
    worst = max(rows, key=lambda r: r["mean_eddington_ratio_needed"]["stellar remnant, 100 Msun"])
    return {
        "figure": path, "efold_time_myr": efold_time(), "seed_epoch_myr": t0, "objects": rows,
        "hardest_object_for_light_seeds": worst["name"],
        "reading": ("A 100-solar-mass seed can reach the observed masses only by accreting at or above the "
                    "Eddington limit without interruption for the entire available time, which feedback makes "
                    "implausible. Heavy seeds or sustained super-Eddington phases relax this to duty cycles "
                    "well below one. The objects at z > 10 are the binding constraint."),
        "caveats": ["Constant radiative efficiency of 0.1; lower efficiency (spinning-down or slim discs) "
                    "shortens the e-folding time and eases the problem.",
                    "Observed masses are single-epoch estimates uncertain by a factor of several, and for "
                    "'little red dots' may be overestimated.", "Mergers are ignored."],
    }
