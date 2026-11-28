"""Shared cosmology and data loading for the cosmos problem simulations.

Background model: flat FLRW with photons, neutrinos (one massive species plus two
massless), cold matter + baryons, and dark energy with the CPL equation of state
w(a) = w0 + wa (1 - a).  Everything is plain numerical integration; nothing here is
a Boltzmann code, and the limits of that are stated where they matter.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy import integrate

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"

C_KMS = 299792.458
OMEGA_GAMMA_H2 = 2.4728e-5        # photons, T_cmb = 2.7255 K
T_NU0_EV = 1.6765e-4              # neutrino temperature today, eV
NEFF = 3.044


@lru_cache(maxsize=1)
def measurements() -> dict:
    """The provenance-tagged measurement table (data/measurements.yaml)."""
    return _numbers(yaml.safe_load((DATA / "measurements.yaml").read_text()))


def _numbers(node, key=None):
    """PyYAML reads '1.6e9' as a string (it wants '1.6e+9'); coerce numeric fields."""
    if isinstance(node, dict):
        return {k: _numbers(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_numbers(v, key) for v in node]
    if isinstance(node, str) and key not in ("name", "ref", "rationale", "unit", "kind", "checked", "provenance"):
        try:
            return float(node)
        except ValueError:
            return node
    return node


def val(path: str) -> float:
    """Look up 'section.key' and return its value."""
    node = measurements()
    for part in path.split("."):
        node = node[part]
    return node["value"]


@dataclass(frozen=True)
class Cosmo:
    h: float = 0.6736
    omega_b: float = 0.02237      # Omega_b h^2
    omega_bc: float = 0.1424      # Omega_(b+cdm) h^2
    w0: float = -1.0
    wa: float = 0.0
    mnu: float = 0.06             # sum of neutrino masses, eV (one massive state)
    f_ede: float = 0.0            # peak early-dark-energy fraction (toy model)
    z_ede: float = 3500.0
    width_ede: float = 0.5        # Gaussian width in ln(1+z)

    def omega_nu(self, z):
        """Neutrino density parameter x h^2 at redshift z, radiation -> matter."""
        per_species = OMEGA_GAMMA_H2 * 0.2271 * NEFF / 3.0 * (1 + z) ** 4
        massless = 2 * per_species
        if self.mnu > 0:
            x = self.mnu / (3.151 * T_NU0_EV * (1 + z))
            massive = per_species * np.sqrt(1 + x * x)
        else:
            massive = per_species
        return massless + massive

    def E(self, z):
        """H(z) / (100 h km/s/Mpc) * h, i.e. sqrt of total omega(z); H = 100 * E."""
        z = np.asarray(z, dtype=float)
        h2 = self.h ** 2
        fixed0 = self.omega_bc + OMEGA_GAMMA_H2 + self.omega_nu(0.0)
        omega_de = h2 - fixed0
        a = 1 / (1 + z)
        de = omega_de * a ** (-3 * (1 + self.w0 + self.wa)) * np.exp(-3 * self.wa * (1 - a))
        tot = (self.omega_bc * (1 + z) ** 3 + OMEGA_GAMMA_H2 * (1 + z) ** 4
               + self.omega_nu(z) + de)
        if self.f_ede > 0:
            f = self.f_ede * np.exp(-0.5 * (np.log((1 + z) / (1 + self.z_ede)) / self.width_ede) ** 2)
            tot = tot / (1 - f)
        return np.sqrt(tot)

    def H(self, z):
        return 100.0 * self.E(z)

    def comoving_distance(self, z):
        """D_M in Mpc (flat), vectorised over z."""
        zs = np.atleast_1d(np.asarray(z, dtype=float))
        grid = np.concatenate([[0.0], np.expm1(np.linspace(np.log1p(1e-4), np.log1p(zs.max()), 2000))])
        integrand = C_KMS / self.H(grid)
        cum = integrate.cumulative_trapezoid(integrand, grid, initial=0.0)
        out = np.interp(zs, grid, cum)
        return out if np.ndim(z) else float(out[0])

    def sound_horizon(self, z_end):
        """Comoving sound horizon at z_end in Mpc, integrated numerically."""
        def f(lnap1):
            zz = np.expm1(lnap1)
            R = 0.75 * self.omega_b / OMEGA_GAMMA_H2 / (1 + zz)
            cs = C_KMS / np.sqrt(3 * (1 + R))
            return cs / self.H(zz) * (1 + zz)
        x = np.linspace(np.log1p(z_end), np.log(1e9), 6000)
        return float(integrate.simpson(f(x), x=x))

    def z_star(self):
        """Redshift of photon decoupling: Hu & Sugiyama (1996) fit, rescaled by 0.99823
        so that the Planck 2018 best fit returns its published z* = 1089.92."""
        ob, om = self.omega_b, self.omega_bc
        g1 = 0.0783 * ob ** -0.238 / (1 + 39.5 * ob ** 0.763)
        g2 = 0.560 / (1 + 21.1 * ob ** 1.81)
        return 0.99823 * 1048 * (1 + 0.00124 * ob ** -0.738) * (1 + g1 * om ** g2)

    def theta_star(self):
        zs = self.z_star()
        return self.sound_horizon(zs) / self.comoving_distance(zs)

    def r_drag(self):
        """Drag-epoch sound horizon, DESI 2024 VI eq. 2.5 fit (standard early physics)."""
        om = self.omega_bc
        return 147.05 * (self.omega_b / 0.02236) ** -0.13 * (om / 0.1432) ** -0.23 * (NEFF / 3.04) ** -0.1

    def age_gyr(self, z=0.0):
        x = np.linspace(np.log1p(z), np.log(1e8), 6000)
        zz = np.expm1(x)
        return float(integrate.simpson(1.0 / self.H(zz), x=x)) * 977.8

    def distance_modulus(self, z):
        dl = (1 + np.asarray(z)) * self.comoving_distance(z)
        return 5 * np.log10(dl) + 25


def load_desi_bao():
    """DESI DR2 BAO compressed likelihood: (z, kind, value), covariance."""
    rows = [ln.split() for ln in (DATA / "desi_dr2_bao_mean.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    z = np.array([float(r[0]) for r in rows])
    v = np.array([float(r[1]) for r in rows])
    kind = [r[2] for r in rows]
    cov = np.loadtxt(DATA / "desi_dr2_bao_cov.txt")
    return z, kind, v, cov


def bao_model(c: Cosmo, z, kind, rd=None):
    rd = c.r_drag() if rd is None else rd
    dm = c.comoving_distance(z)
    dh = C_KMS / c.H(z)
    out = np.empty(len(z))
    for i, k in enumerate(kind):
        if k == "DM_over_rs":
            out[i] = dm[i] / rd
        elif k == "DH_over_rs":
            out[i] = dh[i] / rd
        else:  # DV_over_rs
            out[i] = (z[i] * dm[i] ** 2 * dh[i]) ** (1 / 3) / rd
    return out


def load_sn_binned():
    """Pantheon+ compressed to binned distance-modulus offsets (see build_sn_binned.py)."""
    d = np.load(DATA / "pantheon_plus_binned.npz")
    return d["z_eff"], d["mu_minus_M"], d["cov"]


def sigma_from_dchi2(dchi2: float, dof: int) -> float:
    """Convert a chi^2 improvement with `dof` extra parameters to a Gaussian sigma."""
    from scipy import stats
    p = stats.chi2.sf(max(dchi2, 0.0), dof)
    return float(stats.norm.isf(p / 2))
