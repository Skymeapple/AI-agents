"""Low-cost PM2.5 sensor calibration, transferred to sites and cities it was not fitted on.

Experiment for HFM's "PM2.5 Sensor Calibration Transfer" track. Data: SensEURCity
(Zenodo 17858205, CC-BY 4.0): 85 AirSensEUR sensor boxes in Antwerp, Oslo and Zagreb,
2020-2021, with co-located reference PM2.5. Each box carries two optical particle
sensors (a Plantower PMS5003 and an Alphasense OPC-N3) and humidity and temperature.

One-minute data are averaged to hours. Only hours with a reference value are used.

Two transfer tests:
  across cities   fit on two cities, test on the third (three rotations)
  across sites    fit at each city's main co-location station, test at that city's
                  other reference stations

Methods (input is the sensor's own PM2.5 reading x, relative humidity and temperature)
  raw             the sensor reading as it is
  linear          a x + c                                    (reference, skill 0)
  linear + RH     a x + b RH + c   (the form of the US EPA correction)
  hygroscopic     x / (1 + k RH / (100 - RH)), then linear; k fitted. This is the
                  kappa-Koehler growth form of research/human-frontier-mission.md 7.7
  boosted trees   gradient boosting on x, RH, temperature

Skill = 1 - RMSE / RMSE of the linear calibration, on the test city or sites.
HFM's stations, months and score are private; numbers are NOT comparable.

Usage: python pm25_transfer.py <senseurcity_data_v03.zip>
"""
import io
import sys
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd

SENSORS = {"PMS5003": "5325CAT", "OPC-N3": "OPCN3PM25"}
COLS = ["date", "Location.ID", "SHT31HE", "SHT31TE", "5325CAT", "OPCN3PM25", "Ref.PM2.5"]


def load(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (name.startswith("dataset/") and name.endswith(".csv")):
                continue
            city, box = name.split("/")[-1][:-4].split("_", 1)
            head = pd.read_csv(io.TextIOWrapper(z.open(name), encoding="latin-1"), nrows=0).columns
            use = [c for c in COLS if c in head]
            if "Ref.PM2.5" not in use:
                continue
            df = pd.read_csv(io.TextIOWrapper(z.open(name), encoding="latin-1"), usecols=use, low_memory=False)
            df = df.dropna(subset=["Ref.PM2.5"])
            if df.empty:
                continue
            df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("h")
            num = [c for c in use if c not in ("date", "Location.ID")]
            df[num] = df[num].apply(pd.to_numeric, errors="coerce")
            g = df.groupby(["Location.ID", "hour"])
            h = g[num].mean()
            h = h[g.size() >= 30].reset_index()
            h["city"], h["box"] = city, box
            out.append(h)
            print(f"  {name}: {len(h)} hours with reference", flush=True)
    d = pd.concat(out, ignore_index=True).rename(columns={"SHT31HE": "rh", "SHT31TE": "temp", "Ref.PM2.5": "ref", "Location.ID": "site"})
    return d[(d.ref > 0) & (d.ref < 500) & d.rh.between(1, 99.5)]


def fit_predict(tr, te, method):
    x, xt = tr.x.values, te.x.values
    if method == "raw":
        return xt
    if method == "linear":
        a = np.polyfit(x, tr.ref.values, 1)
        return np.polyval(a, xt)
    if method == "linear + RH":
        A = np.column_stack([x, tr.rh.values, np.ones(len(x))])
        c, *_ = np.linalg.lstsq(A, tr.ref.values, rcond=None)
        return np.column_stack([xt, te.rh.values, np.ones(len(xt))]) @ c
    if method == "hygroscopic":
        best = None
        for k in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8):
            dry = x / (1 + k * tr.rh.values / (100 - tr.rh.values))
            a = np.polyfit(dry, tr.ref.values, 1)
            e = np.sqrt(np.mean((np.polyval(a, dry) - tr.ref.values) ** 2))
            if best is None or e < best[0]:
                best = (e, k, a)
        _, k, a = best
        return np.polyval(a, xt / (1 + k * te.rh.values / (100 - te.rh.values)))
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=4, random_state=0)
    f = ["x", "rh", "temp"]
    return m.fit(tr[f], tr.ref).predict(te[f])


METHODS = ("raw", "linear", "linear + RH", "hygroscopic", "boosted trees")


def evaluate(splits, label):
    print(f"\n{label}")
    print(f"  {'sensor':<8} {'test':<22} {'hours':>7} " + " ".join(f"{m:>14}" for m in METHODS) + "   (RMSE in ug/m3)")
    skills = {m: [] for m in METHODS}
    for sensor, name, tr, te in splits:
        rm = {m: float(np.sqrt(np.mean((fit_predict(tr, te, m) - te.ref.values) ** 2))) for m in METHODS}
        for m in METHODS:
            skills[m].append(1 - rm[m] / rm["linear"])
        print(f"  {sensor:<8} {name:<22} {len(te):>7} " + " ".join(f"{rm[m]:>14.2f}" for m in METHODS))
    print("  mean skill against linear: " + ", ".join(f"{m} {np.mean(v):+.3f} (better in {sum(x > 1e-9 for x in v)}/{len(v)})" for m, v in skills.items() if m != "linear"))


def main():
    d = load(sys.argv[1])
    print(f"\n{len(d)} sensor-hours with reference; cities: {d.groupby('city').size().to_dict()}; reference sites: {d.site.nunique()}")
    city_splits, site_splits = [], []
    for sensor, col in SENSORS.items():
        if col not in d:
            continue
        s = d.dropna(subset=[col, "rh", "temp"]).rename(columns={col: "x"})
        s = s[(s.x >= 0) & (s.x < 1000)]
        for city in sorted(s.city.unique()):
            tr, te = s[s.city != city], s[s.city == city]
            if len(tr) > 500 and len(te) > 500:
                city_splits.append((sensor, f"city {city}", tr, te))
            c = s[s.city == city]
            main_site = c.site.value_counts().idxmax()
            tr, te = c[c.site == main_site], c[c.site != main_site]
            if len(tr) > 500 and len(te) > 500:
                site_splits.append((sensor, f"{city}: other sites ({te.site.nunique()})", tr, te))
    evaluate(city_splits, "transfer across cities: fitted on two cities, tested on the third")
    evaluate(site_splits, "transfer across sites: fitted at the main co-location station, tested at the city's other reference stations")


if __name__ == "__main__":
    main()
