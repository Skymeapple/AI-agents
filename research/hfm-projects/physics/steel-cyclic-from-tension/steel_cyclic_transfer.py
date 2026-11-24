"""Predict a steel's cyclic stress response from its tension test plus other steels.

Public proxy for HFM's "Predict Cyclic Steel Stress Response from Limited Material
Tests" and "Steel Cyclic Loading History Transfer" tracks, on the public database
of Hartloper et al., "Database of Uniaxial Cyclic and Tensile Coupon Tests for
Structural Metallic Materials" (Zenodo record 6965147, CC-BY 4.0). It uses load
protocols LP1 (monotonic tension) to LP10, the names that appear in HFM's logs.

Task: for a target steel, only its LP1 tension test is known. Predict the stress
history of its cyclic tests (LP2..LP10) from their strain histories. Steels from
other test campaigns form the donor library. Steels are held out by campaign
folder, so the target's own campaign never contributes donors.

Model: one-dimensional Voce-Chaboche plasticity (Voce isotropic hardening plus two
Armstrong-Frederick back-stresses), parameters scaled by the measured yield stress.

Methods (prior widths and the ridge strength were fixed before looking at results):
  tension only      fit an isotropic-hardening model to the target's tension test
                    (what the tension test alone identifies; reference, skill 0)
  donor median      median donor parameters, scaled by the target's yield stress
  + tension update  donor median as a prior, refined on the target's tension test
  conditional       donor parameters regressed on tension-test features (ridge),
                    i.e. the "prior conditioned on monotonic features" of
                    research/human-frontier-mission.md section 7.3
  + tension update  conditional prior refined on the target's tension test
  oracle            parameters fitted to the target's own cyclic tests (the best
                    this model class can do; not available in the task)

HFM's split, score and call budget are private; numbers are NOT comparable to
HFM's skill_mean.

Usage: python research/hfm-experiments/steel_cyclic_transfer.py <dir-with-summary.csv-and-Clean_Data>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

CYCLIC = ["LP2", "LP3", "LP4", "LP5", "LP6", "LP7", "LP8", "LP9", "LP10"]
E_CAP = 0.08  # ignore tension data beyond 8% strain (necking region, outside the cyclic range)


@njit(cache=True)
def simulate(strain, E, sy0, Q, b, C1, g1, C2, g2):
    """Strain-driven Voce-Chaboche response with implicit back-stress update."""
    n = strain.shape[0]
    out = np.zeros(n)
    sig, a1, a2, p, e_prev = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        de_total = strain[i] - e_prev
        nsub = max(1, int(abs(de_total) / 2e-4) + 1)
        de = de_total / nsub
        for _ in range(nsub):
            trial = sig + E * de
            sy = sy0 + Q * (1.0 - np.exp(-b * p))
            if abs(trial - a1 - a2) <= sy:
                sig = trial
                continue
            s = 1.0 if trial - a1 - a2 > 0 else -1.0
            lo, hi = 0.0, abs(de) + 1e-9
            for _ in range(40):  # bisection on the plastic increment
                dp = 0.5 * (lo + hi)
                n1 = s * C1 / g1 - (s * C1 / g1 - a1) * np.exp(-g1 * dp) if g1 > 1e-9 else a1 + s * C1 * dp
                n2 = s * C2 / g2 - (s * C2 / g2 - a2) * np.exp(-g2 * dp) if g2 > 1e-9 else a2 + s * C2 * dp
                f = s * (trial - E * s * dp - n1 - n2) - (sy0 + Q * (1.0 - np.exp(-b * (p + dp))))
                if f > 0:
                    lo = dp
                else:
                    hi = dp
            dp = 0.5 * (lo + hi)
            a1 = s * C1 / g1 - (s * C1 / g1 - a1) * np.exp(-g1 * dp) if g1 > 1e-9 else a1 + s * C1 * dp
            a2 = s * C2 / g2 - (s * C2 / g2 - a2) * np.exp(-g2 * dp) if g2 > 1e-9 else a2 + s * C2 * dp
            sig = trial - E * s * dp
            p += dp
        e_prev = strain[i]
        out[i] = sig
    return out


# parameters in log space, stresses relative to the yield stress fy:
# [sy0/fy, Q/fy, b, C1/fy, g1, C2/fy, g2]
THETA0 = np.log(np.array([0.85, 0.25, 8.0, 40.0, 150.0, 4.0, 10.0]))
LO = np.log(np.array([0.3, 1e-3, 0.1, 0.5, 20.0, 0.05, 0.5]))
HI = np.log(np.array([1.5, 2.0, 200.0, 500.0, 2000.0, 60.0, 20.0]))


def predict(theta, strain, E, fy):
    t = np.exp(theta)
    return simulate(strain, E, t[0] * fy, t[1] * fy, t[2], t[3] * fy, t[4], t[5] * fy, t[6])


def residuals(theta, tests, E, fy, prior=None, width=None):
    r = [(predict(theta, e, E, fy) - s) / fy / np.sqrt(len(e)) for e, s in tests]
    r = np.concatenate(r) / np.sqrt(len(tests))
    if prior is not None:
        r = np.concatenate([r, 0.05 * (theta - prior) / width])  # weak Gaussian prior in log space
    return r


def fit(tests, E, fy, start, prior=None, width=None, isotropic_only=False):
    lo, hi = LO.copy(), HI.copy()
    start = np.clip(start, lo + 1e-6, hi - 1e-6)
    if isotropic_only:  # switch the back-stresses off: tiny C, fixed
        fixed = np.log(np.array([1e-6, 100.0, 1e-6, 5.0]))

        def fun(x):
            return residuals(np.concatenate([x, fixed]), tests, E, fy)
        r = least_squares(fun, start[:3], bounds=(lo[:3], np.log(np.array([1.5, 3.0, 400.0]))), max_nfev=60)
        return np.concatenate([r.x, fixed])
    r = least_squares(residuals, start, bounds=(lo, hi), args=(tests, E, fy, prior, width), max_nfev=80)
    return r.x


def load(root):
    d = pd.read_csv(os.path.join(root, "summary.csv")).dropna(subset=["file"])
    d["top"] = d.file.str.split("/").str[1]
    d["mat"] = d.top + "/" + d.file.str.split("/").str[2].str.replace(r"_(tensile|random|\d+\w*)\.csv$", "", regex=True)
    mats = {}
    for mat, g in d.groupby("mat"):
        if "LP1" not in set(g.lp):
            continue
        rec = dict(top=g.top.iloc[0], tension=None, cyclic=[])
        for _, row in g.iterrows():
            path = os.path.join(root, row.file)
            if not os.path.exists(path) or (row.lp != "LP1" and row.lp not in CYCLIC):
                continue
            x = pd.read_csv(path)
            e, s = x["e_true"].values.astype(float), x["Sigma_true"].values.astype(float)
            if row.lp == "LP1":
                keep = e <= E_CAP
                rec["tension"] = (e[keep], s[keep])
                rec["fy"], rec["E"] = float(row.yield_stress__mpa_), float(row.elastic_modulus__mpa_)
            else:
                rec["cyclic"].append((row.lp, e, s))
        if rec["tension"] is not None and len(rec["cyclic"]) >= 2 and np.isfinite(rec["fy"]) and np.isfinite(rec["E"]):
            mats[mat] = rec
    return mats


def tension_features(rec):
    e, s = rec["tension"]
    at = lambda x: np.interp(x, e, s) / rec["fy"]
    return np.array([np.log(rec["fy"] / 355.0), at(0.01), at(0.02), at(0.04), at(0.07)])


def err(theta, rec):
    return float(np.mean([np.sqrt(np.mean((predict(theta, e, rec["E"], rec["fy"]) - s) ** 2)) / rec["fy"] for _, e, s in rec["cyclic"]]))


def main():
    mats = load(sys.argv[1])
    names = sorted(mats)
    print(f"{len(names)} steels with a tension test and at least two cyclic tests, from {len({mats[m]['top'] for m in names})} campaigns")

    # donor fits: each steel's own parameters from all of its tests (also the oracle)
    own = {}
    for m in names:
        rec = mats[m]
        tests = [rec["tension"]] + [(e, s) for _, e, s in rec["cyclic"]]
        own[m] = fit(tests, rec["E"], rec["fy"], THETA0)
    width = np.std(np.array([own[m] for m in names]), axis=0) + 1e-6

    rows = []
    for m in names:
        rec = mats[m]
        donors = [d for d in names if mats[d]["top"] != rec["top"]]
        D = np.array([own[d] for d in donors])
        med = np.median(D, axis=0)
        # ridge regression of donor parameters on tension-test features
        F = np.array([tension_features(mats[d]) for d in donors])
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Z = (F - mu) / sd
        W = np.linalg.solve(Z.T @ Z + 3.0 * np.eye(Z.shape[1]), Z.T @ (D - D.mean(0)))
        cond = np.clip(D.mean(0) + ((tension_features(rec) - mu) / sd) @ W, LO, HI)
        ten = [rec["tension"]]
        res = {
            "tension only (reference)": fit(ten, rec["E"], rec["fy"], THETA0, isotropic_only=True),
            "donor median": med,
            "donor median + tension update": fit(ten, rec["E"], rec["fy"], med, prior=med, width=width),
            "conditional prior": cond,
            "conditional prior + tension update": fit(ten, rec["E"], rec["fy"], cond, prior=cond, width=width),
            "oracle (own cyclic tests)": own[m],
        }
        rows.append({"steel": m, "campaign": rec["top"], **{k: err(v, rec) for k, v in res.items()}})
    r = pd.DataFrame(rows)
    by_c = r.drop(columns="steel").groupby("campaign").mean()  # average within campaign first
    ref = by_c["tension only (reference)"]
    base = by_c["donor median"]
    print(f"\nerror = RMS stress error on the target's cyclic tests, as a share of yield stress; {len(by_c)} campaigns")
    print(f"{'method':<38} {'error':>7} {'skill vs reference':>19} {'vs donor median':>16} {'campaigns better than donor median':>35}")
    for k in by_c.columns:
        print(f"{k:<38} {by_c[k].mean():7.3f} {(1 - by_c[k] / ref).mean():>+19.3f} {by_c[k].mean() / base.mean() - 1:>+16.1%} {int((by_c[k] < base - 1e-9).sum()):>29}/{len(by_c)}")


if __name__ == "__main__":
    main()
