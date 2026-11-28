"""Run the shared invariance comparison (invariance.py) on several projects' public data.

  pm25      SensEURCity low-cost PM2.5 sensors (physics/pm25-calibration): inputs are the
            sensor reading, humidity and temperature; target the reference PM2.5.
            Environments: city (3), then reference site (each station with enough hours).
  photoz    SDSS DR17 galaxies (astronomy/photometric-redshift): magnitudes, colours,
            size proxies and extinction; target redshift. Environments: six 60-degree
            right-ascension bands of the sky.

Usage: python run_invariance.py pm25 <senseurcity_data_v03.zip>
       python run_invariance.py photoz <folder for the downloaded catalogue>
"""
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import invariance as inv


def pm25(path):
    sys.path.insert(0, os.path.join(HERE, "..", "physics", "pm25-calibration"))
    import pm25_transfer as pt
    d = pt.load(path)
    for sensor, col in pt.SENSORS.items():
        if col not in d:
            continue
        s = d.dropna(subset=[col, "rh", "temp"]).rename(columns={col: "x"})
        s = s[(s.x >= 0) & (s.x < 1000)]
        X = s[["x", "rh", "temp"]].values
        inv.compare(X, s.ref.values, s.city.values, f"pm25 {sensor} by city")
        site_counts = s.site.value_counts()
        keep = s.site.isin(site_counts[site_counts >= 1000].index)
        inv.compare(X[keep.values], s.ref.values[keep.values], s.site.values[keep.values], f"pm25 {sensor} by site", min_rows=1000)


SQL = ("SELECT TOP 40000 ra,dec,z,modelMag_u,modelMag_g,modelMag_r,modelMag_i,modelMag_z,psfMag_r,petroMag_r,fiberMag_r,extinction_r "
       "FROM SpecPhoto WHERE class='GALAXY' AND zWarning=0 AND z BETWEEN 0.01 AND 1.0 AND ra >= {a} AND ra < {b} "
       "AND specObjID % 13 = 0 AND modelMag_r BETWEEN 10 AND 23")


def photoz(folder):
    os.makedirs(folder, exist_ok=True)
    parts = []
    for a in range(0, 360, 30):
        f = os.path.join(folder, f"ra{a:03d}.csv")
        if not os.path.exists(f):
            url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?" + urllib.parse.urlencode(
                {"cmd": SQL.format(a=a, b=a + 30), "format": "csv"})
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            open(f, "wb").write(urllib.request.urlopen(req, timeout=600).read())
        parts.append(pd.read_csv(f, comment="#"))
    df = pd.concat(parts, ignore_index=True).apply(pd.to_numeric, errors="coerce").dropna()
    B = "ugriz"
    df = df[(df[[f"modelMag_{b}" for b in B]] > 0).all(1) & (df[[f"modelMag_{b}" for b in B]] < 30).all(1)]
    X = pd.DataFrame({f"m_{b}": df[f"modelMag_{b}"] for b in B})
    for a, b in zip(B[:-1], B[1:]):
        X[f"{a}-{b}"] = df[f"modelMag_{a}"] - df[f"modelMag_{b}"]
    X["psf-model"] = df.psfMag_r - df.modelMag_r
    X["petro-model"] = df.petroMag_r - df.modelMag_r
    X["fiber-model"] = df.fiberMag_r - df.modelMag_r
    X["ext_r"] = df.extinction_r
    env = (df.ra // 60).astype(int).values
    print(f"{len(df)} galaxies; rows per 60-degree band: {np.bincount(env).tolist()}", flush=True)
    inv.compare(X.values, df.z.values, env, "photoz by sky band")


if __name__ == "__main__":
    {"pm25": pm25, "photoz": photoz}[sys.argv[1]](sys.argv[2])
