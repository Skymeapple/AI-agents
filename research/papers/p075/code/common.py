"""Shared data loading and frontier primitives for p075 (open-weight substitution analysis)."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

SNAP_MAIN = os.path.join(DATA, "epoch_eci_2026-11-18", "eci_scores.csv")
SNAP_OLD = os.path.join(DATA, "epoch_eci_2026-11-11", "eci_scores.csv")

GRID_START = pd.Timestamp("2023-01-01")
SEED = 20261122

# Merge co-authored / sub-unit labels into one developer (decision-making unit).
DEV_MAP = {
    "DeepSeek,Peking University": "DeepSeek",
    "Microsoft Research": "Microsoft",
    "Hugging Face,ServiceNow,Nvidia,BigCode": "BigCode (HF/ServiceNow/Nvidia)",
}
COUNTRY_SHORT = {"United States of America": "US", "China": "CN", "France": "FR",
                 "United Arab Emirates": "AE", "United Kingdom": "UK"}


def load(path=SNAP_MAIN, ci_level=0.90):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["Accessibility group"].isin(["Open weights", "Closed weights"])].copy()
    d["open"] = d["Accessibility group"].eq("Open weights")
    d["dev"] = d["Organization"].replace(DEV_MAP)
    d["cc"] = d["Country (of organization)"].map(COUNTRY_SHORT).fillna("other")
    z = {0.90: 1.6449, 0.95: 1.9600}[ci_level]
    sd = (d["eci_ci_high"] - d["eci_ci_low"]) / (2 * z)
    d["sd"] = sd.fillna(sd.median())
    d = d.sort_values("date").reset_index(drop=True)
    return d


class Grid:
    """Daily time grid; frontiers are step functions evaluated on it."""

    def __init__(self, end):
        self.days = pd.date_range(GRID_START, end, freq="D")
        self.n = len(self.days)

    def idx(self, dates):
        return np.asarray((pd.DatetimeIndex(pd.to_datetime(dates)) - GRID_START).days)


def frontier(grid, idx, scores, mask):
    """Running max of scores over releases in mask; -inf before first release."""
    arr = np.full(grid.n, -np.inf)
    ii = idx[mask]
    np.maximum.at(arr, ii, scores[mask])
    return np.maximum.accumulate(arr)


def first_reach(front, levels):
    """Index of first grid day where nondecreasing `front` >= level (n if never)."""
    return np.searchsorted(front, levels, side="left")


def staleness(grid, X, Y):
    """S_{X|Y}(t) = t - T_Y(X(t)) in days; nan where X is -inf."""
    t = np.arange(grid.n)
    T = first_reach(Y, X)
    s = (t - T).astype(float)
    s[~np.isfinite(X)] = np.nan
    return s


def hgap(grid, X, Y):
    """Horizontal gap at X's current level: T_X(X(t)) - T_Y(X(t)) in days (>=0 if Y >= X)."""
    TX = first_reach(X, X)
    TY = first_reach(Y, X)
    h = (TX - TY).astype(float)
    h[~np.isfinite(X)] = np.nan
    return h


def setback(grid, Xcf, X):
    """Frontier setback: T_X(X(t)) - T_X(Xcf(t)) in days, for a counterfactual Xcf <= X.
    Equals v(N)-v(N\\R) for v(S)=int G(O_S) with G=T_X monotone (Prop. 1 applies)."""
    a = first_reach(X, X).astype(float)
    b = first_reach(X, np.where(np.isfinite(Xcf), Xcf, -1e9)).astype(float)
    return a - b


def window_mask(grid, start, end=None):
    m = grid.days >= pd.Timestamp(start)
    if end is not None:
        m &= grid.days <= pd.Timestamp(end)
    return m
