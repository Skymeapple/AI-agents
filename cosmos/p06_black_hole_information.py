"""Problem 6 - does black-hole evaporation destroy information?

Simulates a toy evaporating black hole as a random pure quantum state of N qubits, of
which k have been emitted as radiation.  The entanglement entropy of the radiation is
computed directly from the state vector and compared with
  * Hawking's semiclassical answer (entropy only ever grows), and
  * Page's formula for unitary evaporation (it must turn over and return to zero).
"""
from __future__ import annotations

import numpy as np

from .plotting import BLUE, MUTED, ORANGE, figure, save

N_QUBITS = 14
N_STATES = 12


def page_entropy(m: int, n: int) -> float:
    """Mean entanglement entropy (nats) of a random pure state split into dims m <= n."""
    if m > n:
        m, n = n, m
    return float(np.sum(1.0 / np.arange(n + 1, m * n + 1)) - (m - 1) / (2 * n))


def simulated_curve(rng):
    psi = rng.normal(size=2 ** N_QUBITS) + 1j * rng.normal(size=2 ** N_QUBITS)
    psi /= np.linalg.norm(psi)
    ent = [0.0]
    for k in range(1, N_QUBITS):
        s = np.linalg.svd(psi.reshape(2 ** k, 2 ** (N_QUBITS - k)), compute_uv=False)
        p = s ** 2
        p = p[p > 1e-16]
        ent.append(float(-(p * np.log(p)).sum()))
    return np.array(ent + [0.0])


def run() -> dict:
    rng = np.random.default_rng(6)
    curves = np.array([simulated_curve(rng) for _ in range(N_STATES)])
    mean = curves.mean(axis=0)
    k = np.arange(N_QUBITS + 1)
    page = np.array([0.0] + [page_entropy(2 ** j, 2 ** (N_QUBITS - j)) for j in range(1, N_QUBITS)] + [0.0])
    hawking = k * np.log(2)

    fig, ax = figure(1)
    ax.plot(k, hawking / np.log(2), color=ORANGE, label="Hawking: thermal radiation, information lost")
    ax.plot(k, page / np.log(2), color=MUTED, lw=1, ls="--", label="Page formula (unitary)")
    ax.plot(k, mean / np.log(2), "o", color=BLUE, label=f"simulated random states (mean of {N_STATES})")
    ax.set(xlabel=f"qubits emitted as radiation (of {N_QUBITS})", ylabel="radiation entropy (bits)",
           title="Entropy of the radiation from a toy black hole")
    ax.legend(loc="upper left", fontsize=8)
    path = save(fig, "p06_black_hole_information")

    # Scales for a real black hole.
    G, hbar, c, msun, yr = 6.6743e-11, 1.054572e-34, 2.99792458e8, 1.98847e30, 3.15576e7
    t_evap = 5120 * np.pi * G ** 2 * msun ** 3 / (hbar * c ** 4) / yr
    s_bh = 4 * np.pi * G * msun ** 2 / (hbar * c)
    return {
        "figure": path,
        "max_abs_deviation_from_page_bits": float(np.max(np.abs(mean - page)) / np.log(2)),
        "page_time_fraction_of_qubits": 0.5,
        "entropy_deficit_at_half_bits": float((hawking[N_QUBITS // 2] - mean[N_QUBITS // 2]) / np.log(2)),
        "solar_mass_evaporation_time_yr": float(t_evap),
        "solar_mass_entropy_kB": float(s_bh),
        "page_time_fraction_of_lifetime": float(1 - 0.5 ** 1.5),
        "reading": ("A generic quantum-mechanical evolution reproduces Page's curve to within a fraction of a "
                    "bit: the radiation entropy rises, turns over when half the system has been emitted, and "
                    "returns to zero. Hawking's calculation keeps rising. The two only disagree after the "
                    "turnover, which for a solar-mass black hole is ~1e67 years away, so no observation of an "
                    "astrophysical black hole can distinguish them."),
        "caveats": ["A random state stands in for the black hole's dynamics; this shows what unitarity requires, "
                    "not how gravity achieves it.",
                    "14 qubits, against ~1e77 for a solar-mass black hole."],
    }
