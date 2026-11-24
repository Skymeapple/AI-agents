"""Anti-reflection coating design under manufacturing variation.

Computational proxy for HFM's "Optical Coating Design Under Manufacturing
Variation" track. No external data: reflectance comes from the standard
transfer-matrix method at normal incidence, with fixed textbook refractive indices
(MgF2 1.38, SiO2 1.46, Al2O3 1.65, Ta2O5 2.10, TiO2 2.35).

It tests the idea in research/human-frontier-mission.md section 7.6: optimise the
*expected* reflectance under the process-variation model instead of the nominal
reflectance, with the same search budget.

Variation model: each layer thickness gets an independent Gaussian error of 2% of
its thickness plus 1 nm; each layer index gets an independent 0.5% error.

For each specification (substrate index, wavelength band), one shared pool of
candidate designs is built by random-restart nominal optimisation (4, 6 and 8
layers). Three ways of producing the final design are then compared:
  nominal            the candidate with the lowest nominal reflectance
  robust selection   the same candidates, chosen by reflectance averaged over 24
                     fixed variation samples
  robust polish      each candidate's thicknesses re-optimised on that average,
                     then the best chosen
All are scored on 4,000 fresh variation samples. Sharing the candidate pool keeps
search luck out of the comparison.

HFM's catalogue, variation model and score are private; numbers are NOT comparable
to HFM's skill_mean.

Usage: python research/hfm-experiments/robust_ar_coating.py
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize

MATERIALS = np.array([1.38, 1.46, 1.65, 2.10, 2.35])
T_REL, T_ABS, N_REL = 0.02, 1.0, 0.005
RESTARTS = 40


def reflectance(d, n, n_sub, wl):
    """d, n: (B, L) thickness in nm and index; wl: (W,). Returns (B, W) reflectance from air."""
    B, L = d.shape
    m11 = np.ones((B, len(wl)), complex)
    m12 = np.zeros_like(m11)
    m21 = np.zeros_like(m11)
    m22 = np.ones_like(m11)
    for j in range(L):
        delta = 2 * np.pi * n[:, j, None] * d[:, j, None] / wl[None, :]
        c, s = np.cos(delta), np.sin(delta)
        a11, a12, a21, a22 = c, 1j * s / n[:, j, None], 1j * s * n[:, j, None], c
        m11, m12, m21, m22 = m11 * a11 + m12 * a21, m11 * a12 + m12 * a22, m21 * a11 + m22 * a21, m21 * a12 + m22 * a22
    b = m11 + m12 * n_sub
    c_ = m21 + m22 * n_sub
    r = (b - c_) / (b + c_)
    return np.abs(r) ** 2


def perturb(d, n, z_t, z_n):
    """Apply variation samples z (S, L) to one design; returns (S, L) arrays."""
    dd = np.maximum(d[None] * (1 + T_REL * z_t) + T_ABS * z_t, 0.0)  # same draw drives both thickness terms
    return dd, n[None] * (1 + N_REL * z_n)


def mean_r(d, n, n_sub, wl, z_t=None, z_n=None):
    if z_t is None:
        return reflectance(d[None], n[None], n_sub, wl).mean()
    dd, nn = perturb(d, n, z_t, z_n)
    return reflectance(dd, nn, n_sub, wl).mean()


def search(n_sub, wl, layers, rng, keep=4):
    """Random-restart local search on nominal reflectance; returns the best `keep` designs."""
    found = []
    lows, highs = MATERIALS[:2], MATERIALS[2:]
    for _ in range(RESTARTS):
        # outermost layer low-index, then alternate; material picked at random per layer
        n = np.array([rng.choice(lows) if (j % 2 == 0) else rng.choice(highs) for j in range(layers)])
        d0 = rng.uniform(10, 160, layers)
        res = minimize(lambda d: mean_r(d, n, n_sub, wl), d0, method="L-BFGS-B", bounds=[(5, 400)] * layers,
                       options={"maxiter": 200})
        found.append((res.fun, res.x, n))
    return sorted(found, key=lambda b: b[0])[:keep]


def robust_polish(cand, n_sub, wl, z_t, z_n):
    """Re-optimise thicknesses of one design on the variation-averaged reflectance."""
    _, d, n = cand
    L = len(d)
    res = minimize(lambda x: mean_r(x, n, n_sub, wl, z_t[:, :L], z_n[:, :L]), d, method="L-BFGS-B",
                   bounds=[(5, 400)] * L, options={"maxiter": 150})
    return res.fun, res.x, n


def main():
    rng = np.random.default_rng(0)
    eval_rng = np.random.default_rng(12345)
    specs = list(itertools.product((1.52, 1.77, 2.0), ((400, 700), (450, 950))))
    print("columns: best nominal design [nominal R, R under variation, 90th pct, layers] | same candidates chosen by robust score"
          " [R under variation, change, layers] | robust-polished [nominal R, R under variation, 90th pct, layers, change, change in 90th pct]")
    gains, p90_gains, sel_gains = [], [], []
    for n_sub, (lo, hi) in specs:
        wl = np.linspace(lo, hi, 41)
        z_t, z_n = rng.standard_normal((24, 8)), rng.standard_normal((24, 8))
        cands = [c for L in (4, 6, 8) for c in search(n_sub, wl, L, rng)]
        nom = min(cands, key=lambda b: b[0])                         # usual practice: best nominal design
        score = lambda c: mean_r(c[1], c[2], n_sub, wl, z_t[:, :len(c[1])], z_n[:, :len(c[1])])
        sel = min(cands, key=score)                                  # same candidates, chosen by robust score
        rob = min((robust_polish(c, n_sub, wl, z_t, z_n) for c in cands), key=lambda b: b[0])
        out = []
        for _, d, n in (nom, sel, rob):
            L = len(d)
            zt, zn = eval_rng.standard_normal((4000, L)), eval_rng.standard_normal((4000, L))
            dd, nn = perturb(d, n, zt, zn)
            per = reflectance(dd, nn, n_sub, wl).mean(axis=1)
            out.append((mean_r(d, n, n_sub, wl), per.mean(), np.quantile(per, 0.9), L))
        (n0, n1, n9, nl), (s0, s1, s9, sl), (r0, r1, r9, rl) = out
        gains.append(r1 / n1 - 1); p90_gains.append(r9 / n9 - 1); sel_gains.append(s1 / n1 - 1)
        print(f"{n_sub:>9} {lo}-{hi:<5} | {n0:>8.3%} {n1:>8.3%} {n9:>8.3%} {nl:>3} | {s1:>8.3%} {s1 / n1 - 1:>+7.1%} {sl:>3} |"
              f" {r0:>8.3%} {r1:>8.3%} {r9:>8.3%} {rl:>3} {r1 / n1 - 1:>+7.1%} {r9 / n9 - 1:>+7.1%}")
    print(f"\nrobust selection only: mean change in expected reflectance {np.mean(sel_gains):+.1%} (better or equal in {sum(g <= 1e-9 for g in sel_gains)}/{len(specs)})")
    print(f"robust polish:         mean change in expected reflectance {np.mean(gains):+.1%} (better in {sum(g < 0 for g in gains)}/{len(specs)}); "
          f"90th percentile {np.mean(p90_gains):+.1%}")

if __name__ == "__main__":
    main()
