"""Download the observed series the world model is calibrated on.

Only sources that publish machine-readable global aggregates are used: the World
Bank indicators API (the ``WLD`` aggregate) and two public GitHub datasets. Each
series is written in long form with a manifest recording the exact URL and the
date it was retrieved, so a calibration can be traced to the bytes behind it.

Figures that exist only inside agency reports (UCDP conflict deaths, UNHCR
displacement totals, the Lancet resistance forecast) are *not* fetched here. They
live in ``data/world/sourced.yaml`` with a citation each, and are graded
``sourced`` rather than ``observed`` -- we did not ingest the underlying data.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "world"

WORLD_BANK = "https://api.worldbank.org/v2/country/WLD/indicator/{code}?format=json&per_page=500"
OWID_CO2 = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
GLOBAL_TEMP = "https://raw.githubusercontent.com/datasets/global-temp/main/data/annual.csv"

# series name -> (World Bank code, unit)
WORLD_BANK_SERIES: dict[str, tuple[str, str]] = {
    "population": ("SP.POP.TOTL", "people"),
    "gdp_per_capita": ("NY.GDP.PCAP.KD", "constant 2015 US$"),
    "poverty_rate": ("SI.POV.DDAY", "% of population under $3.00/day (2021 PPP)"),
    "undernourishment_rate": ("SN.ITK.DEFC.ZS", "% of population"),
    "food_insecurity_rate": ("SN.ITK.MSFI.ZS", "% of population, moderate or severe"),
    "safe_water_rate": ("SH.H2O.SMDW.ZS", "% of population with safely managed drinking water"),
    "safe_sanitation_rate": ("SH.STA.SMSS.ZS", "% of population with safely managed sanitation"),
    "forest_share": ("AG.LND.FRST.ZS", "% of land area"),
    "protected_share": ("ER.PTD.TOTL.ZS", "% of territorial area, land and sea"),
    "military_spending_share": ("MS.MIL.XPND.GD.ZS", "% of GDP"),
    "refugees_unhcr_mandate": ("SM.POP.RHCR.EA", "people, by country of asylum"),
    "under5_mortality": ("SH.DYN.MORT", "deaths per 1,000 live births"),
    "life_expectancy": ("SP.DYN.LE00.IN", "years"),
    "homicide_rate": ("VC.IHR.PSRC.P5", "per 100,000 people"),
}

# series name -> (OWID column, unit)
OWID_SERIES: dict[str, tuple[str, str]] = {
    "co2_emissions": ("co2_including_luc", "MtCO2 per year, fossil plus land-use change"),
    "ghg_emissions": ("total_ghg", "MtCO2e per year"),
    "cumulative_co2": ("cumulative_co2_including_luc", "MtCO2 since 1850"),
}


def _get(url: str) -> requests.Response:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response


def fetch_all(out_dir: Path = DATA_DIR) -> dict:
    """Fetch every series, write ``observed.csv`` and ``manifest.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    rows: list[tuple[str, int, float]] = []
    manifest: dict[str, dict] = {}

    for name, (code, unit) in WORLD_BANK_SERIES.items():
        url = WORLD_BANK.format(code=code)
        payload = _get(url).json()
        points = [(int(r["date"]), float(r["value"])) for r in payload[1] if r["value"] is not None]
        if not points:
            raise RuntimeError(f"World Bank returned no world aggregate for {code}")
        rows += [(name, year, value) for year, value in sorted(points)]
        manifest[name] = {
            "publisher": "World Bank, World Development Indicators",
            "indicator": code,
            "title": payload[1][0]["indicator"]["value"],
            "unit": unit,
            "url": url,
            "retrieved": today,
        }

    world = [r for r in csv.DictReader(io.StringIO(_get(OWID_CO2).text)) if r["country"] == "World"]
    for name, (column, unit) in OWID_SERIES.items():
        points = [(int(r["year"]), float(r[column])) for r in world if r[column] and int(r["year"]) >= 1950]
        rows += [(name, year, value) for year, value in points]
        manifest[name] = {
            "publisher": "Our World in Data CO2 dataset (Global Carbon Project and others)",
            "indicator": column,
            "unit": unit,
            "url": OWID_CO2,
            "retrieved": today,
        }

    temps = [r for r in csv.DictReader(io.StringIO(_get(GLOBAL_TEMP).text)) if r["Source"] == "GISTEMP"]
    if not temps:
        raise RuntimeError("no GISTEMP rows in the global temperature dataset")
    rows += [("temperature_anomaly", int(r["Year"][:4]), float(r["Mean"])) for r in temps]
    manifest["temperature_anomaly"] = {
        "publisher": "NASA GISTEMP v4 via the datasets/global-temp mirror",
        "indicator": "annual mean",
        "unit": "degC relative to 1951-1980 (not pre-industrial)",
        "url": GLOBAL_TEMP,
        "retrieved": today,
    }

    rows.sort()
    with (out_dir / "observed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "year", "value"])
        writer.writerows(rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    fetched = fetch_all()
    print(f"fetched {len(fetched)} series into {DATA_DIR}")
