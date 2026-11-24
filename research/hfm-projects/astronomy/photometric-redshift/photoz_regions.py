"""Photometric redshift across disjoint sky regions, on public SDSS data.

Public proxy for HFM's "Galaxy Photometric Redshift Prediction Across Disjoint Sky
Regions" track. HFM uses Legacy Survey image cutouts with AION-1 embeddings; this
proxy uses SDSS DR17 catalogue photometry only (no images), so absolute errors are
NOT comparable to HFM's. It tests two ideas from research/human-frontier-mission.md
section 3.1 under the same kind of split (test galaxies lie in sky regions absent
from training):

  1. predict a redshift probability distribution instead of a single value;
  2. give the model the Galactic extinction of each galaxy.

Data: SDSS DR17 SpecPhoto, galaxies with zWarning = 0 and 0.01 < z < 1, sampled in
twelve 30-degree right-ascension stripes via the public SkyServer SQL service:

  SELECT TOP 40000 ra,dec,z,zErr,modelMag_u,modelMag_g,modelMag_r,modelMag_i,
         modelMag_z,psfMag_r,petroMag_r,fiberMag_r,extinction_r,survey
  FROM SpecPhoto WHERE class='GALAXY' AND zWarning=0 AND z BETWEEN 0.01 AND 1.0
    AND ra >= {ra} AND ra < {ra+30} AND specObjID % 13 = 0
    AND modelMag_r BETWEEN 10 AND 23

Usage: python research/hfm-experiments/photoz_regions.py <dir-with-ra*.csv>
"""
from __future__ import annotations

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

BANDS = "ugriz"


def load(d):
    df = pd.concat([pd.read_csv(f, comment="#") for f in sorted(glob.glob(os.path.join(d, "ra*.csv")))], ignore_index=True)
    num = [c for c in df.columns if c != "survey"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=num)
    df = df[(df[[f"modelMag_{b}" for b in BANDS]] > 0).all(axis=1) & (df[[f"modelMag_{b}" for b in BANDS]] < 30).all(axis=1)]
    # sky regions: 15-degree RA cells x north/south of dec = 20 deg
    df["region"] = (df.ra // 15).astype(int) * 2 + (df.dec > 20).astype(int)
    return df.reset_index(drop=True)


def features(df, extinction):
    X = pd.DataFrame({f"m_{b}": df[f"modelMag_{b}"] for b in BANDS})
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        X[f"{a}-{b}"] = df[f"modelMag_{a}"] - df[f"modelMag_{b}"]
    X["psf-model"] = df.psfMag_r - df.modelMag_r      # concentration / size proxies
    X["petro-model"] = df.petroMag_r - df.modelMag_r
    X["fiber-model"] = df.fiberMag_r - df.modelMag_r
    if extinction:
        X["ext_r"] = df.extinction_r
    return X


PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)


def fit_regress(Xtr, ytr, Xte, seed):
    m = lgb.LGBMRegressor(random_state=seed, **PARAMS).fit(Xtr, ytr)
    return m.predict(Xte)


def fit_pdf(Xtr, ytr, Xte, seed, nbins=40):
    """Classify into bins uniform in log(1+z); return the PDF mean and median."""
    edges = np.linspace(np.log1p(0.01), np.log1p(1.0), nbins + 1)
    centres = np.expm1((edges[:-1] + edges[1:]) / 2)
    lab = np.clip(np.digitize(np.log1p(ytr), edges) - 1, 0, nbins - 1)
    present = np.unique(lab)
    remap = {c: i for i, c in enumerate(present)}
    m = lgb.LGBMClassifier(random_state=seed, **{**PARAMS, "n_estimators": 120, "num_leaves": 31}).fit(Xtr, np.vectorize(remap.get)(lab))
    P = m.predict_proba(Xte)
    c = centres[present]
    mean = P @ c
    cdf = np.cumsum(P, axis=1)
    median = c[np.minimum((cdf < 0.5).sum(axis=1), len(c) - 1)]
    return mean, median


def metrics(pred, z):
    dz = (pred - z) / (1 + z)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    return np.sqrt(np.mean(dz ** 2)), nmad, np.mean(np.abs(dz) > 0.15)


def main():
    df = load(sys.argv[1])
    z = df.z.values
    groups = df.region.values
    print(f"{len(df)} galaxies in {df.region.nunique()} sky regions; 5-fold cross-validation with whole regions held out")
    preds = {k: np.zeros(len(df)) for k in ("regression", "regression seed 2", "regression + extinction", "PDF mean + extinction",
                                             "PDF median + extinction", "blend: regression + PDF mean (+ extinction)")}
    fold_of = np.zeros(len(df), int)
    for f, (tr, te) in enumerate(GroupKFold(5).split(df, z, groups)):
        fold_of[te] = f
        X0, X1 = features(df, False), features(df, True)
        preds["regression"][te] = fit_regress(X0.iloc[tr], z[tr], X0.iloc[te], 0)
        preds["regression seed 2"][te] = fit_regress(X0.iloc[tr], z[tr], X0.iloc[te], 1)
        preds["regression + extinction"][te] = fit_regress(X1.iloc[tr], z[tr], X1.iloc[te], 0)
        mean, med = fit_pdf(X1.iloc[tr], z[tr], X1.iloc[te], 0)
        preds["PDF mean + extinction"][te] = mean
        preds["PDF median + extinction"][te] = med
        preds["blend: regression + PDF mean (+ extinction)"][te] = 0.5 * (preds["regression + extinction"][te] + mean)
        print(f"  fold {f} done ({len(te)} test galaxies)", flush=True)

    base = metrics(preds["regression"], z)[0]
    print(f"\n{'method':<46} {'NRMSE':>8} {'vs base':>8} {'NMAD':>8} {'outliers':>9}   folds better than base")
    for k, p in preds.items():
        r, nmad, out = metrics(p, z)
        wins = sum(metrics(p[fold_of == f], z[fold_of == f])[0] < metrics(preds["regression"][fold_of == f], z[fold_of == f])[0] for f in range(5))
        print(f"{k:<46} {r:8.5f} {r / base - 1:+8.2%} {nmad:8.5f} {out:9.3%}   {wins}/5")


if __name__ == "__main__":
    main()
