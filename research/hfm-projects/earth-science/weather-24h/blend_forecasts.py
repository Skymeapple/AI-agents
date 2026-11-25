"""24-hour global surface forecasts: correct one model, or blend several?

CPU-only experiment for HFM's "24-Hour Global Near-Surface Weather Forecasting
with ArchesWeather" track. It uses forecasts that WeatherBench 2 has already
computed and published for several models on the challenge's own grid (1.5 degrees,
240 x 121, poles included), so no GPU is needed:
  gs://weatherbench2/datasets/<model>/...240x121_equiangular_with_poles_conservative.zarr
  (public bucket, read anonymously; see https://weatherbench2.readthedocs.io)

HFM's agents only post-process the output of one frozen model (ArchesWeather) and
have gained 0.75%. research/human-frontier-mission.md section 2.1 proposed blending
model families instead. ArchesWeather forecasts are not in the bucket, so this
tests the idea on the models that are: Pangu-Weather, IFS HRES, GraphCast,
NeuralGCM, FuXi and Keisler's model, for the year 2020.

Protocol
  * Variables: 2 m temperature, 10 m wind (u, v), mean sea-level pressure; 24 h lead;
    truth is ERA5, as in the challenge.
  * Forecasts started at 00 UTC every second day of 2020. Alternating two-week
    blocks are used for fitting and for testing, so both cover all seasons.
  * Score: latitude-weighted RMSE per variable, divided by that variable's
    standard deviation over the test targets, averaged over the four variables
    (the same construction as HFM's "area-weighted normalised RMSE"; HFM's exact
    normalisation is not published).

Methods
  raw                the model's forecast as published
  bias-corrected     subtract the mean error at every grid point (fitted blocks)
  equal blend        mean of the bias-corrected models
  weighted blend     non-negative weights per variable fitted on the fitted blocks

Numbers are NOT comparable to HFM's score (different model, year and normalisation).

Usage: python blend_forecasts.py [step-in-days]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import xarray as xr
from scipy.optimize import nnls

BUCKET = "gs://weatherbench2/datasets"
GRID = "240x121_equiangular_with_poles_conservative.zarr"
MODELS = {
    "pangu": f"{BUCKET}/pangu/2018-2022_0012_{GRID}",
    "ifs-hres": f"{BUCKET}/hres/2016-2022-0012-{GRID}",
    "graphcast": f"{BUCKET}/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-{GRID}",
    "neuralgcm": f"{BUCKET}/neuralgcm_deterministic/2020-{GRID}",
    "fuxi": f"{BUCKET}/fuxi/2020-{GRID}",
    "keisler": f"{BUCKET}/keisler/2020-{GRID}",
}
ERA5 = f"{BUCKET}/era5/1959-2022-6h-{GRID}"
VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
LEAD = np.timedelta64(24, "h")


def open_zarr(path):
    return xr.open_zarr(path, storage_options={"token": "anon"}, chunks=None)


def to_grid(a):
    """Put fields on (time, latitude ascending, longitude)."""
    a = a.transpose("time", "latitude", "longitude")
    return a.sortby("latitude")


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    inits = np.arange(np.datetime64("2020-01-01T00"), np.datetime64("2020-12-30T00"), np.timedelta64(24 * step, "h")).astype("datetime64[ns]")
    valid = inits + LEAD
    t0 = time.time()
    era = open_zarr(ERA5)
    truth = {v: to_grid(era[v].sel(time=valid)).values.astype(np.float32) for v in VARS}
    lat = np.sort(era.latitude.values)
    print(f"ERA5 targets loaded: {len(inits)} dates, grid {truth[VARS[0]].shape[1:]}, {time.time() - t0:.0f}s", flush=True)

    fc = {}
    for name, path in MODELS.items():
        t1 = time.time()
        try:
            ds = open_zarr(path)
            missing = [v for v in VARS if v not in ds]
            if missing:
                print(f"{name}: skipped, missing {missing}; has {[v for v in ds.data_vars][:12]}", flush=True)
                continue
            have = np.isin(inits, ds.time.values)
            if have.mean() < 0.95:
                print(f"{name}: skipped, only {have.mean():.0%} of dates present", flush=True)
                continue
            lead = ds.prediction_timedelta.values
            hours = lead / np.timedelta64(1, "h") if np.issubdtype(lead.dtype, np.timedelta64) else lead.astype(float)
            if hours.max() > 1e6:                      # stored as nanoseconds
                hours = hours / 3.6e12
            k = int(np.argmin(np.abs(hours - 24)))
            if abs(hours[k] - 24) > 0.01:
                print(f"{name}: skipped, no 24 h lead (has {hours[:6]} ...)", flush=True)
                continue
            sel = ds[VARS].isel(prediction_timedelta=k).sel(time=inits)
            fc[name] = {v: to_grid(sel[v]).values.astype(np.float32) for v in VARS}
            print(f"{name}: loaded in {time.time() - t1:.0f}s", flush=True)
        except Exception as e:  # keep going with the models that work
            print(f"{name}: failed ({type(e).__name__}: {str(e)[:200]})", flush=True)
    names = list(fc)
    if len(names) < 2:
        print("fewer than two models available; nothing to blend")
        return

    block = (np.arange(len(inits)) * step // 14) % 2          # alternating two-week blocks
    fit, test = block == 0, block == 1
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[None, :, None]

    def score(pred, v):
        err = pred[test] - truth[v][test]
        return float(np.sqrt((w * err ** 2).mean()) / truth[v][test].std())

    rows = {}
    corrected = {n: {} for n in names}
    for n in names:
        raw, cor = [], []
        for v in VARS:
            bias = (fc[n][v][fit] - truth[v][fit]).mean(axis=0)
            corrected[n][v] = fc[n][v] - bias
            raw.append(score(fc[n][v], v))
            cor.append(score(corrected[n][v], v))
        rows[f"{n}, raw"] = raw
        rows[f"{n}, bias-corrected"] = cor
    eq, wt, weights = [], [], {}
    sub = (slice(None), slice(None, None, 3), slice(None, None, 3))     # thin the grid for the weight fit
    for v in VARS:
        stack = np.stack([corrected[n][v] for n in names])
        eq.append(score(stack.mean(axis=0), v))
        A = np.stack([(corrected[n][v][fit][sub] * np.sqrt(w[:, ::3])).ravel() for n in names], axis=1)
        b = (truth[v][fit][sub] * np.sqrt(w[:, ::3])).ravel()
        coef, _ = nnls(A.astype(np.float64), b.astype(np.float64))
        weights[v] = coef
        wt.append(score(np.tensordot(coef, stack, axes=1), v))
    rows["equal blend of bias-corrected models"] = eq
    rows["weighted blend of bias-corrected models"] = wt

    best_single = min((np.mean(r), k) for k, r in rows.items() if "bias-corrected" in k and "blend" not in k)
    print(f"\nnormalised RMSE at 24 h on test blocks ({int(test.sum())} forecasts); lower is better")
    print(f"{'method':<42} {'t2m':>7} {'u10':>7} {'v10':>7} {'mslp':>7} {'mean':>7} {'vs best corrected single model':>32}")
    for k, r in rows.items():
        print(f"{k:<42} " + " ".join(f"{x:7.4f}" for x in r) + f" {np.mean(r):7.4f} {np.mean(r) / best_single[0] - 1:>+32.1%}")
    print(f"\nbest corrected single model: {best_single[1]}")
    print("blend weights per variable (" + ", ".join(names) + "):")
    for v in VARS:
        print(f"  {v:<28} " + " ".join(f"{x:5.2f}" for x in weights[v]))
    for n in names:
        gain = np.mean(rows[f"{n}, bias-corrected"]) / np.mean(rows[f"{n}, raw"]) - 1
        print(f"bias correction alone, {n}: {gain:+.1%}")


if __name__ == "__main__":
    main()
