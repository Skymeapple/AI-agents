"""Build an hourly 2023 panel of carbon- and water-intensity signals for 13 US
data-center hubs.

Inputs (downloaded by code/fetch_data.sh):
  data/processed/EIA930_2023_{Jan_Jun,Jul_Dec}_elec.csv  -- physics-reconciled EIA-930
      hourly generation by fuel per balancing authority (de Chalendar & Benson 2021,
      gridemissions bulk dataset)
  data/processed/EIA930_2023_*_co2i.csv  -- consumption-based CO2 intensity (gCO2e/kWh)
  data/isd/<USAF-WBAN>-2023.gz  -- NOAA ISD-Lite hourly weather (air temp, dew point)

Output: results/panel_2023.parquet (or .pkl) with one row per (site, UTC hour).
All emission/water factors are documented in factors.py.
"""
import gzip
import os
import numpy as np
import pandas as pd

from factors import CARBON_LCA, EWIF_CENTRAL
from cooling import wet_bulb_stull, rh_from_dewpoint, wue_tower, wue_adiabatic, pue_t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

# site name, balancing authority, ISD station, UTC offset of local standard time
SITES = [
    ("Ashburn VA", "PJM", "724030-93738", -5),
    ("Dallas TX", "ERCO", "722590-03927", -6),
    ("San Jose CA", "CISO", "724945-23293", -8),
    ("Phoenix AZ", "AZPS", "722780-23183", -7),
    ("The Dalles OR", "BPAT", "726988-24219", -8),
    ("Council Bluffs IA", "MISO", "725500-14942", -6),
    ("Pryor OK", "SWPP", "723560-13968", -6),
    ("Atlanta GA", "SOCO", "722190-13874", -5),
    ("Memphis TN", "TVA", "723340-13893", -6),
    ("Las Vegas NV", "NEVP", "723860-23169", -8),
    ("Salt Lake UT", "PACE", "725720-24127", -7),
    ("Charlotte NC", "DUK", "723140-13881", -5),
    ("Boston MA", "ISNE", "725090-14739", -5),
]
FUELS = ["COL", "GAS", "NUC", "OIL", "OTH", "SUN", "UNK", "WAT", "WND"]


def load_grid():
    parts = []
    for half in ["Jan_Jun", "Jul_Dec"]:
        e = pd.read_csv(os.path.join(DATA, "processed", f"EIA930_2023_{half}_elec.csv"), index_col=0)
        c = pd.read_csv(os.path.join(DATA, "processed", f"EIA930_2023_{half}_co2i.csv"), index_col=0)
        parts.append(e.join(c, how="left"))
    g = pd.concat(parts)
    g.index = pd.to_datetime(g.index, utc=True)
    g = g[~g.index.duplicated()].sort_index()
    return g


def load_isd(station):
    rows = []
    with gzip.open(os.path.join(DATA, "isd", f"{station}-2023.gz"), "rt") as f:
        for line in f:
            p = line.split()
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])))
    w = pd.DataFrame(rows, columns=["y", "m", "d", "h", "T", "Td"])
    w.index = pd.to_datetime(dict(year=w.y, month=w.m, day=w.d, hour=w.h), utc=True)
    w = w[~w.index.duplicated()]
    w = w[["T", "Td"]].replace(-9999, np.nan) / 10.0
    full = pd.date_range("2023-01-01", "2024-01-01", freq="h", tz="UTC", inclusive="left")
    w = w.reindex(full).interpolate(limit=6).ffill().bfill()
    return w


def build():
    g = load_grid()
    out = []
    for name, ba, st, utc_off in SITES:
        w = load_isd(st)
        gen = g[[f"E_{ba}_{f}" for f in FUELS]].clip(lower=0).copy()
        gen.columns = FUELS
        tot = gen.sum(axis=1)
        share = gen.div(tot, axis=0)
        df = share.add_prefix("sh_")
        df["ci_prod"] = sum(share[f] * CARBON_LCA[f] for f in FUELS)  # gCO2e/kWh, production-based
        df["ci_cons"] = g[f"CO2i_{ba}_D"]  # consumption-based (gridemissions)
        df["ewif"] = sum(share[f] * EWIF_CENTRAL[f] for f in FUELS)  # L/kWh, central factors
        df["ewif_nohydro"] = sum(share[f] * EWIF_CENTRAL[f] for f in FUELS if f != "WAT")
        df = df.join(w, how="inner")
        df["RH"] = rh_from_dewpoint(df["T"], df["Td"])
        df["Twb"] = wet_bulb_stull(df["T"], df["RH"])
        df["pue"] = pue_t(df["Twb"])
        df["wue_tower"] = wue_tower(df["T"], df["pue"])
        df["wue_adiab"] = wue_adiabatic(df["T"], df["Twb"])
        df["site"] = name
        df["ba"] = ba
        df["utc_off"] = utc_off
        df["local_hour"] = (df.index + pd.Timedelta(hours=utc_off)).hour
        df["local_date"] = (df.index + pd.Timedelta(hours=utc_off)).date
        df = df.dropna(subset=["ci_prod", "ewif", "T"])
        out.append(df)
    panel = pd.concat(out)
    panel.index.name = "utc"
    os.makedirs(RES, exist_ok=True)
    panel.to_pickle(os.path.join(RES, "panel_2023.pkl"))
    return panel


if __name__ == "__main__":
    p = build()
    print(p.groupby("site")[["ci_prod", "ci_cons", "ewif", "T", "Twb", "pue", "wue_tower", "wue_adiab"]].mean().round(3))
    print(len(p))
