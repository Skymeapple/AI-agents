"""Shared data loading and helpers for the threshold-drift analysis.

Data: Epoch AI 'Data on AI models' snapshot (CC BY 4.0), files in ../data/.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

LOG10_2 = np.log10(2.0)

# Snapshot data cutoff: latest publication date in large_scale_ai_models.csv
T_CUTOFF = None


def dec_year(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    start = pd.to_datetime(ts.dt.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    end = pd.to_datetime((ts.dt.year + 1).astype("Int64").astype(str) + "-01-01", errors="coerce")
    return ts.dt.year + (ts - start).dt.total_seconds() / (end - start).dt.total_seconds()


def primary_country(s):
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    first = s.split(",")[0].strip()
    return first


def country_bucket(c):
    if c == "United States of America":
        return "US"
    if c in ("China", "Hong Kong"):
        return "China"
    if c in ("France", "Germany", "Italy", "Spain", "Netherlands", "Sweden", "Finland",
             "Belgium", "Austria", "Denmark", "Poland", "Ireland", "Portugal", "Czechia",
             "Greece", "Luxembourg", "Estonia", "Lithuania", "Latvia", "Slovenia",
             "Slovakia", "Hungary", "Romania", "Bulgaria", "Croatia", "Malta", "Cyprus"):
        return "EU"
    if c == "Unknown":
        return "Unknown"
    return "Other"


def load_large_scale():
    d = pd.read_csv(os.path.join(DATA, "large_scale_ai_models.csv"))
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d = d[d["date"].notna()].copy()
    d["t"] = dec_year(d["date"])
    d["C"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d["x"] = np.log10(d["C"])
    d["country"] = d["Country (of organization)"].map(primary_country)
    d["bucket"] = d["country"].map(country_bucket)
    d["org"] = d["Organization"].fillna("Unknown").map(lambda s: s.split(",")[0].strip())
    acc = d["Model accessibility"].fillna("")
    d["open_weights"] = acc.str.startswith("Open weights")
    return d


def load_all_models():
    d = pd.read_csv(os.path.join(DATA, "all_ai_models.csv"), low_memory=False)
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d = d[d["date"].notna()].copy()
    d["t"] = dec_year(d["date"])
    d["C"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d["x"] = np.log10(d["C"])
    return d


def alpha_from_halving(months):
    """Algorithmic efficiency in OOM (log10) per year given compute-halving time in months."""
    return LOG10_2 * 12.0 / months
