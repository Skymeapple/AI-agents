"""Compress Pantheon+ (1701 light curves, full stat+sys covariance) into binned
distance moduli that are small enough to commit.

Inputs are downloaded from the public releases (not committed; the covariance is 33 MB):
  Pantheon+SH0ES.dat           github.com/PantheonPlusSH0ES/DataRelease
  Pantheon+SH0ES_STAT+SYS.cov  github.com/CobayaSampler/sn_data

Method: keep zHD > 0.01 (the cut the Pantheon+ cosmology analysis uses), then solve
the generalised least squares problem  m_i = mu_fid(z_i) + delta_b(i)  for one offset
per log-spaced redshift bin.  The result is  mu_fid(z_eff) + delta  with covariance
(A^T C^-1 A)^-1.  A model is later compared at each bin's effective redshift, which is
accurate because mu_model - mu_fid is smooth across a narrow bin.  The absolute
magnitude M is NOT removed here; fits marginalise over it.

Usage: python -m cosmos.build_sn_binned <dir containing the two files>
"""
import sys
from pathlib import Path

import numpy as np

from .common import DATA, Cosmo

NBINS = 40


def main(src: Path):
    dat = src / "Pantheon+SH0ES.dat"
    cols = dat.read_text().splitlines()[0].split()
    raw = np.genfromtxt(dat, skip_header=1, usecols=(cols.index("zHD"), cols.index("m_b_corr")))
    flat = np.loadtxt(src / "Pantheon+SH0ES_STAT+SYS.cov")
    n = int(flat[0])
    cov = flat[1:].reshape(n, n)
    keep = raw[:, 0] > 0.01
    z, m = raw[keep, 0], raw[keep, 1]
    cov = cov[np.ix_(keep, keep)]

    fid = Cosmo()
    edges = np.geomspace(z.min() * 0.999, z.max() * 1.001, NBINS + 1)
    idx = np.digitize(z, edges) - 1
    used = np.unique(idx)
    A = np.zeros((len(z), len(used)))
    for j, b in enumerate(used):
        A[idx == b, j] = 1.0
    cinv = np.linalg.inv(cov)
    fisher = A.T @ cinv @ A
    bcov = np.linalg.inv(fisher)
    resid = m - fid.distance_modulus(z)
    delta = bcov @ (A.T @ cinv @ resid)
    # effective redshift: inverse-variance weighted mean of ln z
    z_eff = np.array([np.exp(np.average(np.log(z[idx == b]),
                                        weights=1 / np.diag(cov)[idx == b])) for b in used])
    mu = fid.distance_modulus(z_eff) + delta
    np.savez(DATA / "pantheon_plus_binned.npz", z_eff=z_eff, mu_minus_M=mu, cov=bcov,
             n_sn=np.array([np.sum(idx == b) for b in used]))
    print(f"{len(z)} SNe -> {len(used)} bins, z = {z_eff.min():.4f}..{z_eff.max():.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
