"""Estimating a site's wave-energy resource from a short measurement campaign.

Public proxy for HFM's "Estimating Wave-Energy Resources from Short Measurement
Campaigns" track, on NOAA NDBC buoy records (public domain):
https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz

Protocol: a 60-day campaign has paired wind and wave observations; the following
365 days have wind only. The method must estimate the wave-power resource over
those 365 days. Wave power is P = 0.49 * Hs^2 * T (kW/m, with T the average
period as a stand-in for the energy period).

It tests the idea in research/human-frontier-mission.md section 7.2: model the
resource as a wind-driven part plus a swell part that local wind cannot predict.

Methods (no settings were tuned on the results; lag windows were fixed up front):
  reference       assume the campaign's mean power holds for the year (skill 0)
  log-regression  ridge regression of log P on log wind speed at 6/24/72 h lags
  boosted trees   gradient boosting on the same features (flexible ML baseline)
  additive        P = swell constant + non-negative sum of wind-sea power terms
                  (U^2, U^3.5, U^5 of 24 h and 72 h mean wind), fitted by NNLS
  neighbour buoy  log-regression with one more input: the wave power measured at
                  another buoy in the same sea area during the assessment year.
                  This stands in for a long-term reference record or hindcast, the
                  outside information the first results pointed to

HFM's sites, campaign lengths and score are private; numbers are NOT comparable
to HFM's skill_mean.

Usage: python research/hfm-experiments/wave_resource.py <dir-with-ndbc-files>
"""
from __future__ import annotations

import glob
import os
import statistics
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
CAMPAIGN_H, ASSESS_H, STEP_H = 60 * 24, 365 * 24, 45 * 24


def load_station(d, st):
    frames = []
    for f in sorted(glob.glob(os.path.join(d, f"{st}h*.txt.gz"))):
        df = pd.read_csv(f, sep=r"\s+", skiprows=[1], na_values=["99.00", "999", "99.0", "999.0", "9999.0"])
        df["t"] = pd.to_datetime(dict(year=df["#YY"], month=df.MM, day=df.DD, hour=df.hh))
        frames.append(df[["t", "WSPD", "WVHT", "APD"]])
    df = pd.concat(frames)
    for c in ("WSPD", "WVHT", "APD"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df.WSPD > 90, "WSPD"] = np.nan
    df.loc[df.WVHT > 90, "WVHT"] = np.nan
    df.loc[df.APD > 90, "APD"] = np.nan
    h = df.groupby("t").mean().resample("1h").mean()
    h["P"] = 0.49 * h.WVHT ** 2 * h.APD
    w = h.WSPD.interpolate(limit=6)
    for lag in (6, 24, 72):
        h[f"U{lag}"] = w.rolling(lag, min_periods=lag // 2).mean()
    return h


def design_log(x):
    return np.column_stack([np.ones(len(x))] + [np.log(np.maximum(x[f"U{lag}"].values, 0.3)) for lag in (6, 24, 72)])


def design_add(x, scale):
    cols = [np.ones(len(x))]
    for lag in (24, 72):
        u = np.maximum(x[f"U{lag}"].values, 0) / scale
        cols += [u ** 2, u ** 3.5, u ** 5]
    return np.column_stack(cols)


def predict_all(camp, assess):
    out = {"reference": np.full(len(assess), camp.P.mean())}
    # ridge in log space, with the usual lognormal back-transform correction
    A, y = design_log(camp), np.log(np.maximum(camp.P.values, 1e-3))
    beta = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)
    resid = y - A @ beta
    out["log-regression"] = np.exp(design_log(assess) @ beta + 0.5 * resid.var())
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1, n_jobs=1, random_state=0)
    m.fit(A[:, 1:], y)
    r2 = y - m.predict(A[:, 1:])
    out["boosted trees"] = np.exp(m.predict(design_log(assess)[:, 1:]) + 0.5 * r2.var())
    scale = max(camp.U24.quantile(0.9), 1.0)
    coef, _ = nnls(design_add(camp, scale), camp.P.values)
    out["additive (swell + wind-sea)"] = design_add(assess, scale) @ coef
    return out


def main():
    d = sys.argv[1]
    stations = sorted({os.path.basename(f)[:5] for f in glob.glob(os.path.join(d, "*.txt.gz"))})
    data = {st: load_station(d, st) for st in stations}
    # buoys in the same sea area, by the first two digits of the station number
    area = lambda st: st[:2]
    rows = []
    for st in stations:
        h = data[st]
        others = {o: data[o].P.reindex(h.index) for o in stations if o != st and area(o) == area(st)}
        n_inst = 0
        for s in range(0, len(h) - CAMPAIGN_H - ASSESS_H, STEP_H):
            camp = h.iloc[s:s + CAMPAIGN_H].dropna(subset=["P", "U6", "U24", "U72"])
            assess = h.iloc[s + CAMPAIGN_H:s + CAMPAIGN_H + ASSESS_H].dropna(subset=["P", "U6", "U24", "U72"])
            if len(camp) < 0.7 * CAMPAIGN_H or len(assess) < 0.6 * ASSESS_H:
                continue
            truth = assess.P.values
            for name, p in predict_all(camp, assess).items():
                rows.append(dict(station=st, start=s, method=name,
                                 mean_err=abs(p.mean() - truth.mean()) / truth.mean(),
                                 rmse=float(np.sqrt(np.mean((p - truth) ** 2)) / truth.mean())))
            # outside information: the wave record of another buoy in the same sea area. The partner is
            # chosen inside the campaign (highest correlation of log power), so the choice is causal.
            best = None
            for o, ref in others.items():
                rc = np.log(ref.reindex(camp.index).clip(lower=1e-3))
                ok = rc.notna()
                if ok.sum() > 0.6 * len(camp) and ref.reindex(assess.index).notna().mean() > 0.6:
                    c = np.corrcoef(rc[ok], np.log(camp.P[ok].clip(lower=1e-3)))[0, 1]
                    if best is None or c > best[0]:
                        best = (c, o, ref)
            if best is not None:
                _, o, ref = best
                rc, ra = np.log(ref.reindex(camp.index).clip(lower=1e-3)), np.log(ref.reindex(assess.index).clip(lower=1e-3))
                okc, oka = rc.notna().values, ra.notna().values
                A = np.column_stack([design_log(camp)[okc], rc.values[okc]])
                y = np.log(np.maximum(camp.P.values[okc], 1e-3))
                beta = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)
                resid = y - A @ beta
                p = np.exp(np.column_stack([design_log(assess)[oka], ra.values[oka]]) @ beta + 0.5 * resid.var())
                t2 = truth[oka]
                wind_only = predict_all(camp, assess.iloc[oka])
                for name, pp in (("neighbour buoy + wind", p), ("log-regression, same hours", wind_only["log-regression"]), ("reference, same hours", wind_only["reference"])):
                    rows.append(dict(station=st, start=s, method=name, mean_err=abs(pp.mean() - t2.mean()) / t2.mean(),
                                     rmse=float(np.sqrt(np.mean((pp - t2) ** 2)) / t2.mean())))
            n_inst += 1
        print(f"  station {st}: {n_inst} campaign instances", flush=True)
    r = pd.DataFrame(rows)
    piv_m = r.pivot_table(index=["station", "start"], columns="method", values="mean_err")
    piv_r = r.pivot_table(index=["station", "start"], columns="method", values="rmse")
    print(f"\n{len(piv_m)} instances at {piv_m.index.get_level_values(0).nunique()} stations")
    print(f"{'method':<30} {'annual-mean error':>18} {'skill (mean)':>13} {'hourly RMSE/mean':>17} {'skill (hourly)':>15} {'stations better':>16}")
    for mth in ("reference", "log-regression", "boosted trees", "additive (swell + wind-sea)"):
        # skill computed per station first so stations with more instances do not dominate
        sk_m = (1 - piv_m[mth].groupby("station").mean() / piv_m["reference"].groupby("station").mean())
        sk_r = (1 - piv_r[mth].groupby("station").mean() / piv_r["reference"].groupby("station").mean())
        print(f"{mth:<30} {piv_m[mth].mean():>18.1%} {sk_m.mean():>+13.3f} {piv_r[mth].mean():>17.3f} {sk_r.mean():>+15.3f} {int((sk_m > 0).sum()):>10}/{len(sk_m)}")
    sub = piv_m.dropna(subset=["neighbour buoy + wind"])
    subr = piv_r.loc[sub.index]
    print(f"\nwith a neighbouring buoy's wave record as an extra input: {len(sub)} instances at {sub.index.get_level_values(0).nunique()} stations (hours where the neighbour reported)")
    for mth in ("reference, same hours", "log-regression, same hours", "neighbour buoy + wind"):
        sk_m = (1 - sub[mth].groupby("station").mean() / sub["reference, same hours"].groupby("station").mean())
        sk_r = (1 - subr[mth].groupby("station").mean() / subr["reference, same hours"].groupby("station").mean())
        better = int((sub[mth].groupby("station").mean() < sub["log-regression, same hours"].groupby("station").mean()).sum())
        print(f"{mth:<30} {sub[mth].mean():>18.1%} {sk_m.mean():>+13.3f} {subr[mth].mean():>17.3f} {sk_r.mean():>+15.3f}   stations better than wind only: {better}/{len(sk_m)}")

if __name__ == "__main__":
    main()
