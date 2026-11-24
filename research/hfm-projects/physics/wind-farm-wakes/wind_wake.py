"""Wind-farm wake calibration and per-turbine power prediction from real SCADA.

Public proxy for HFM's "Wind-Farm Wake Calibration and Power Prediction from Real
SCADA Data" track, on the same two farms: Kelmarsh (Zenodo 16807551) and
Penmanshiel (Zenodo 16807304), both CC-BY 4.0, 10-minute data for 2017.

HFM uses the FLORIS wake engine; this proxy uses a plain Jensen (top-hat) wake
model so it runs anywhere. Task: from the farm's inflow (wind speed and direction
at the upstream turbines), predict every turbine's power. Each farm has a 60-day
calibration window; the rest of the year is the test. All method choices were made
on Kelmarsh; Penmanshiel is the unseen farm and only gets its own calibration.

It tests the idea in research/human-frontier-mission.md section 7.10: calibrate
the inflow direction and wake physics first, because machine-learned residuals do
not carry to unseen conditions.

Methods:
  no wake            every turbine produces the free-stream power (reference)
  wake, default      Jensen wake, decay k = 0.075, direction as measured
  wake, calibrated   k and a single direction offset fitted on the calibration window
  + sector factors   per-turbine, per-30-degree multiplicative correction (shrunk)
  boosted trees      per-turbine gradient boosting on inflow speed and direction
  wake + trees       calibrated wake model plus boosted-tree residual

Error is mean absolute power error as a share of rated power, for inflow between
4 and 12 m/s. "New directions" are test records whose 10-degree direction bin had
fewer than 20 records in the calibration window.

HFM's records, engine and reward are private; numbers are NOT comparable.

Usage: python research/hfm-experiments/wind_wake.py <kelmarsh.csv.gz> <Kelmarsh_WT_static.csv> <penmanshiel.csv.gz> <Penmanshiel_WT_static.csv>
"""
from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RATED = 2050.0
CAL_DAYS = 60


def layout(static_csv):
    s = pd.read_csv(static_csv, encoding="utf-8-sig").dropna(subset=["Latitude"])
    num = s["Title"].str.extract(r"(\d+)$")[0].astype(int).values
    lat, lon = s["Latitude"].values, s["Longitude"].values
    x = (lon - lon.mean()) * 111320.0 * np.cos(np.deg2rad(lat.mean()))
    y = (lat - lat.mean()) * 110540.0
    return num, np.column_stack([x, y]), float(s["Rotor Diameter (m)"].iloc[0]) / 2


def load(scada_csv, nums):
    d = pd.read_csv(scada_csv, parse_dates=["t"])
    piv = {c: d.pivot_table(index="t", columns="turbine", values=c)[list(nums)] for c in ("ws", "yaw", "power")}
    ok = piv["ws"].notna().all(axis=1) & piv["power"].notna().all(axis=1) & piv["yaw"].notna().all(axis=1)
    ok &= (piv["power"] > 0).all(axis=1) | (piv["ws"] < 3.5).all(axis=1)   # drop records with a stopped turbine in wind
    return {c: p[ok].values for c, p in piv.items()}, piv["ws"].index[ok]


def circ_median(deg):
    r = np.deg2rad(deg)
    return np.rad2deg(np.arctan2(np.median(np.sin(r), axis=1), np.median(np.cos(r), axis=1))) % 360


def geometry(xy, R, k):
    """For each whole degree of wind direction: squared geometric wake factor per turbine, and upstream order."""
    n = len(xy)
    G = np.zeros((360, n))
    front = np.zeros((360, n))
    for th in range(360):
        d = np.array([-np.sin(np.deg2rad(th)), -np.cos(np.deg2rad(th))])   # downwind unit vector
        along = xy @ d
        cross = xy @ np.array([-d[1], d[0]])
        dx = along[:, None] - along[None, :]          # i downstream of j by dx
        dy = np.abs(cross[:, None] - cross[None, :])
        hit = (dx > 1.0) & (dy < R + k * dx)
        G[th] = np.where(hit, 1.0 / (1.0 + k * dx / R) ** 4, 0.0).sum(axis=1)
        front[th] = along
    return G, front


def inflow(data, xy, R):
    """Free-stream speed and direction: taken from the two most upstream turbines."""
    wd0 = circ_median(data["yaw"])
    _, front = geometry(xy, R, 0.075)
    order = np.argsort(front[np.round(wd0).astype(int) % 360], axis=1)[:, :2]
    rows = np.arange(len(wd0))[:, None]
    u = data["ws"][rows, order].mean(axis=1)
    wd = circ_median(data["yaw"][rows, order])
    return u, wd


def deficit(G, wd, offset, u):
    ct = np.where(u < 9.0, 0.8, 0.8 * (9.0 / np.maximum(u, 9.0)) ** 2)
    g = G[np.round(wd + offset).astype(int) % 360]
    return (1 - np.sqrt(1 - ct))[:, None] * np.sqrt(g)


def power_curve(u, power, unwaked):
    """Median power of geometrically unwaked turbines against inflow speed."""
    bins = np.arange(0, 26, 0.5)
    uu = np.repeat(u, power.shape[1])[unwaked.ravel()]
    pp = power.ravel()[unwaked.ravel()]
    idx = np.digitize(uu, bins)
    centres, med = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 10:
            centres.append(bins[b - 1] + 0.25)
            med.append(np.median(pp[m]))
    med = np.maximum.accumulate(np.array(med))
    return lambda x: np.interp(x, centres, med)


def run(name, scada, static, k_grid=(0.04, 0.06, 0.075, 0.1, 0.13), off_grid=range(-12, 13, 3)):
    nums, xy, R = layout(static)
    data, t = load(scada, nums)
    u, wd = inflow(data, xy, R)
    P = data["power"]
    cal = np.asarray(t < t[0] + pd.Timedelta(days=CAL_DAYS))
    band = (u >= 4) & (u <= 12)
    test = ~cal & band
    geo = {k: geometry(xy, R, k)[0] for k in k_grid}
    pc = power_curve(u[cal], P[cal], geo[0.075][np.round(wd[cal]).astype(int) % 360] == 0)

    def wake_pred(k, off):
        return pc(u[:, None] * (1 - deficit(geo[k], wd, off, u)))

    preds = {"no wake (reference)": np.repeat(pc(u)[:, None], P.shape[1], axis=1), "wake, default": wake_pred(0.075, 0)}
    fit = cal & band
    best = min(((np.abs(wake_pred(k, o)[fit] - P[fit]).mean(), k, o) for k in k_grid for o in off_grid))
    _, k, off = best
    base = wake_pred(k, off)
    preds["wake, calibrated"] = base
    # per-turbine, per-sector multiplicative factors, shrunk towards 1
    sector = (wd // 30).astype(int)
    fac = np.ones((12, P.shape[1]))
    for s in range(12):
        m = fit & (sector == s)
        if m.sum():
            ratio = np.median(P[m] / np.maximum(base[m], 50.0), axis=0)
            fac[s] = 1 + (ratio - 1) * m.sum() / (m.sum() + 50.0)
    preds["+ sector factors"] = base * fac[sector]
    F = np.column_stack([u, np.sin(np.deg2rad(wd)), np.cos(np.deg2rad(wd))])
    trees, resid = np.zeros_like(P), np.zeros_like(P)
    for j in range(P.shape[1]):
        kw = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=30, verbose=-1, n_jobs=1, random_state=0)
        trees[:, j] = lgb.LGBMRegressor(**kw).fit(F[fit], P[fit, j]).predict(F)
        resid[:, j] = lgb.LGBMRegressor(**kw).fit(F[fit], P[fit, j] - base[fit, j]).predict(F)
    preds["boosted trees"] = trees
    preds["wake + trees"] = base + resid

    counts = np.bincount((wd[fit] // 10).astype(int), minlength=36)
    new_dir = test & (counts[(wd // 10).astype(int)] < 20)
    waked = geo[k][np.round(wd + off).astype(int) % 360] > 0
    print(f"\n{name}: {P.shape[1]} turbines, {cal.sum()} calibration records, {test.sum()} test records "
          f"({new_dir.sum()} in new directions); calibrated k = {k}, direction offset = {off} deg")
    print(f"  {'method':<22} {'error, all':>11} {'vs reference':>13} {'waked turbines only':>20} {'new directions':>15}")
    out = {}
    ref = np.abs(preds["no wake (reference)"][test] - P[test]).mean() / RATED
    for m, p in preds.items():
        e = np.abs(p - P) / RATED
        out[m] = (e[test].mean(), e[test][waked[test]].mean(), e[new_dir].mean() if new_dir.sum() else np.nan)
        print(f"  {m:<22} {out[m][0]:>11.4f} {out[m][0] / ref - 1:>+13.1%} {out[m][1]:>20.4f} {out[m][2]:>15.4f}")
    return out


def main():
    run("Kelmarsh (development farm)", sys.argv[1], sys.argv[2])
    run("Penmanshiel (unseen farm)", sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
