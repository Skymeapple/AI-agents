"""Which extra catalogue inputs lower photometric-redshift error on unseen sky regions?

Follow-up to photoz_regions.py in this folder. That experiment found that one added
input, Galactic extinction, lowered error by 1.3% in every held-out region. This one
adds the other things an image carries that five magnitudes do not: size,
concentration, profile shape, axis ratio, per-band point-source and fibre
magnitudes, and measurement errors. All come from the public SDSS DR17 catalogue.

Data: SpecObj joined to PhotoObj through the SkyServer SQL service,

  SELECT TOP 40000 s.ra,s.dec,s.z,p.modelMag_[ugriz],p.psfMag_[ugriz],
         p.fiberMag_[gri],p.petroMag_r,p.petroR50_r,p.petroR90_r,p.deVRad_r,
         p.expRad_r,p.fracDeV_r,p.deVAB_r,p.expAB_r,p.extinction_r,p.modelMagErr_[ugriz]
  FROM SpecObj s JOIN PhotoObj p ON s.bestObjID=p.objID
  WHERE s.class='GALAXY' AND s.zWarning=0 AND s.z BETWEEN 0.01 AND 1.0
    AND s.ra >= {ra} AND s.ra < {ra+30} AND s.specObjID % 13 = 0
    AND p.modelMag_r BETWEEN 10 AND 23

Same protocol as before: boosted-tree regression, five-fold cross-validation with
whole sky regions held out, error = RMS of (z_pred - z) / (1 + z).

Usage: python photoz_features.py <dir-with-ra*.csv>
"""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

B = "ugriz"
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=4)


def load(d):
    df = pd.concat([pd.read_csv(f, comment="#") for f in sorted(glob.glob(os.path.join(d, "ra*.csv")))], ignore_index=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df[[f"modelMag_{b}" for b in B]] > 0).all(axis=1) & (df[[f"modelMag_{b}" for b in B]] < 30).all(axis=1)]
    df["region"] = (df.ra // 15).astype(int) * 2 + (df.dec > 20).astype(int)
    return df.reset_index(drop=True)


def blocks(df):
    f = {}
    base = pd.DataFrame({f"m_{b}": df[f"modelMag_{b}"] for b in B})
    for a, b in zip(B[:-1], B[1:]):
        base[f"{a}-{b}"] = df[f"modelMag_{a}"] - df[f"modelMag_{b}"]
    base["psf-model r"] = df.psfMag_r - df.modelMag_r
    base["petro-model r"] = df.petroMag_r - df.modelMag_r
    base["fiber-model r"] = df.fiberMag_r - df.modelMag_r
    f["base"] = base
    f["extinction"] = pd.DataFrame({"ext_r": df.extinction_r})
    f["size and shape"] = pd.DataFrame({"r50": np.log10(df.petroR50_r.clip(0.05)), "r90": np.log10(df.petroR90_r.clip(0.05)),
                                        "concentration": df.petroR90_r / df.petroR50_r.clip(0.05), "deV radius": np.log10(df.deVRad_r.clip(0.01)),
                                        "exp radius": np.log10(df.expRad_r.clip(0.01)), "fracDeV": df.fracDeV_r, "deV axis ratio": df.deVAB_r,
                                        "exp axis ratio": df.expAB_r, "surface brightness": df.modelMag_r + 5 * np.log10(df.petroR50_r.clip(0.05))})
    f["per-band apertures"] = pd.DataFrame({**{f"psf-model {b}": df[f"psfMag_{b}"] - df[f"modelMag_{b}"] for b in "ugiz"},
                                            "fiber g-r": df.fiberMag_g - df.fiberMag_r, "fiber r-i": df.fiberMag_r - df.fiberMag_i})
    f["magnitude errors"] = pd.DataFrame({f"err_{b}": df[f"modelMagErr_{b}"] for b in B})
    return f


def main():
    df = load(sys.argv[1])
    z, groups = df.z.values, df.region.values
    f = blocks(df)
    sets = {"base (magnitudes and colours)": ["base"], "+ extinction": ["base", "extinction"],
            "+ size and shape": ["base", "size and shape"], "+ per-band apertures": ["base", "per-band apertures"],
            "+ magnitude errors": ["base", "magnitude errors"], "all inputs": list(f)}
    print(f"{len(df)} galaxies in {df.region.nunique()} sky regions", flush=True)
    folds = list(GroupKFold(5).split(df, z, groups))
    res = {}
    for name, keys in sets.items():
        X = pd.concat([f[k] for k in keys], axis=1)
        pred = np.zeros(len(df))
        per = []
        for tr, te in folds:
            pred[te] = lgb.LGBMRegressor(random_state=0, **PARAMS).fit(X.iloc[tr], z[tr]).predict(X.iloc[te])
            per.append(np.sqrt(np.mean(((pred[te] - z[te]) / (1 + z[te])) ** 2)))
        dz = (pred - z) / (1 + z)
        res[name] = (np.sqrt(np.mean(dz ** 2)), 1.4826 * np.median(np.abs(dz - np.median(dz))), np.mean(np.abs(dz) > 0.15), per)
        print(f"  done: {name}", flush=True)
    base = res["base (magnitudes and colours)"]
    print(f"\n{'inputs':<32} {'NRMSE':>8} {'vs base':>8} {'NMAD':>8} {'outliers':>9} {'folds better':>13}")
    for name, (r, nmad, out, per) in res.items():
        print(f"{name:<32} {r:8.5f} {r / base[0] - 1:+8.2%} {nmad:8.5f} {out:9.3%} {sum(a < b for a, b in zip(per, base[3])):>9}/5")


if __name__ == "__main__":
    main()
