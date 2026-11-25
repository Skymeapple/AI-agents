"""Spill at eight Snake and Columbia dams: a dissolved-gas model fitted to public data,
and dam-by-dam against joint spill decisions.

Experiment for HFM's "Columbia and Snake River Dam Spill Optimization" track.
Data: Columbia Basin Research DART, daily river conditions at each dam's forebay
and tailwater gauge (flow, spill, dissolved gas; courtesy of the US Army Corps of
Engineers). Spring season, 1 April to 20 June.

Part 1 (measured on held-out years). Predict each dam's tailwater dissolved gas
from that day's spill, flow and forebay gas:
  linear       gas = a + b * spill + c * forebay gas
  saturating   gas = 100 + a * (1 - exp(-b * spill)) + c * (forebay gas - 100) * (1 - spill share)
               (the form used in the Corps' SYSTDG model: gas from spill levels off)
  reference    yesterday's measured value (not usable for planning a different spill)
and the next dam's forebay gas from the mixed water released upstream.

Part 2 (model study). With the fitted gas model, choose each day's spill share at
all eight dams so that no tailwater exceeds the gas cap:
  dam by dam   from upstream, each dam spills as much as its own cap allows
  joint        all eight chosen together (exact dynamic programme along the river)
Fish survival per dam uses ASSUMED round values (spillway 0.98, other routes 0.94,
share of fish using the spillway = spill share ** 0.6); they are not fitted here.
Part 2 therefore shows whether coordination matters under this gas model, not a
survival forecast. HFM's instances and score are private; numbers are NOT comparable.

Usage: python dam_spill.py <cache directory>
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DAMS = [("LWG", "LGNW"), ("LGS", "LGSW"), ("LMN", "LMNW"), ("IHR", "IDSW"), ("MCN", "MCPW"), ("JDA", "JHAW"), ("TDA", "TDDO"), ("BON", "CCIW")]
YEARS = list(range(2012, 2025))
TEST = [2022, 2023, 2024]
URL = "https://www.cbr.washington.edu/dart/cs/php/rpt/river_daily.php?sc=1&outputFormat=csv&year={y}&proj={p}&span=no&startdate=4%2F1&enddate=6%2F20"
CAP, S_SPILL, S_OTHER, MAX_SHARE = 120.0, 0.98, 0.94, 0.8


def fetch(cache, proj, year):
    path = os.path.join(cache, f"{proj}-{year}.csv")
    if not os.path.exists(path):
        req = urllib.request.Request(URL.format(y=year, p=proj), headers={"User-Agent": "research-script"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=180).read())
    txt = open(path, encoding="latin-1").read().split("Notes:")[0]
    d = pd.read_csv(io.StringIO(txt))
    d = d[d["Project"] == proj]
    out = pd.DataFrame({"date": pd.to_datetime(d["Date"]), "flow": pd.to_numeric(d["Outflow (kcfs)"], errors="coerce"),
                        "spill": pd.to_numeric(d["Spill (kcfs)"], errors="coerce"), "gas": pd.to_numeric(d["Dissolved Gas Percent (%)"], errors="coerce")})
    return out.set_index("date")


def sat(X, a, b, c):
    spill, flow, fb = X
    return 100 + a * (1 - np.exp(-b * spill)) + c * (fb - 100) * (1 - spill / np.maximum(flow, 1e-6))


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    cache = sys.argv[1]
    os.makedirs(cache, exist_ok=True)
    data = []
    for fb, tw in DAMS:
        f = pd.concat([fetch(cache, fb, y) for y in YEARS])
        t = pd.concat([fetch(cache, tw, y) for y in YEARS])
        d = pd.DataFrame({"flow": f.flow, "spill": f.spill, "fb": f.gas, "tw": t.gas})
        d["year"] = d.index.year
        data.append(d)
        print(f"{fb}: {len(d)} days, {d[['spill', 'fb', 'tw']].notna().all(axis=1).sum()} complete, median flow {d.flow.median():.0f} kcfs, median spill share {(d.spill / d.flow).median():.2f}", flush=True)

    # ---------- Part 1: tailwater gas ----------
    print("\nPart 1a. Tailwater dissolved gas (% saturation), error on held-out years", TEST)
    print(f"  {'dam':<5} {'days':>5} {'linear':>8} {'saturating':>11} {'yesterday':>10}")
    par, tot = [], {"lin": [], "sat": [], "prev": []}
    for (fb, _), d in zip(DAMS, data):
        d = d.assign(prev=d.groupby("year").tw.shift(1)).dropna()
        d = d[(d.fb > 90) & (d.fb < 145) & (d.tw > 90) & (d.tw < 145) & (d.flow > 0)]
        tr, te = d[~d.year.isin(TEST)], d[d.year.isin(TEST)]
        A = lambda x: np.column_stack([np.ones(len(x)), x.spill, x.fb])
        beta = np.linalg.lstsq(A(tr), tr.tw, rcond=None)[0]
        X = lambda x: (x.spill.values, x.flow.values, x.fb.values)
        p, _ = curve_fit(sat, X(tr), tr.tw.values, p0=[20, 0.02, 0.5], bounds=([0, 1e-4, 0], [60, 1, 2]), maxfev=20000)
        par.append(p)
        e = dict(lin=te.tw - A(te) @ beta, sat=te.tw - sat(X(te), *p), prev=te.tw - te.prev)
        for k in tot:
            tot[k].append(rmse(e[k]))
        print(f"  {fb:<5} {len(te):>5} {rmse(e['lin']):>8.2f} {rmse(e['sat']):>11.2f} {rmse(e['prev']):>10.2f}   fitted: gas from spill levels off at {100 + p[0]:.0f}%, forebay carry-over {p[2]:.2f}")
    print(f"  {'mean':<5} {'':>5} {np.mean(tot['lin']):>8.2f} {np.mean(tot['sat']):>11.2f} {np.mean(tot['prev']):>10.2f}   dams where saturating beats linear: {sum(s < l for s, l in zip(tot['sat'], tot['lin']))}/8")

    # ---------- Part 1b: next forebay from upstream release ----------
    print("\nPart 1b. Next dam's forebay gas from the upstream release, error on held-out years")
    print(f"  {'reach':<10} {'lag (days)':>10} {'from upstream':>14} {'season mean':>12}")
    link = []
    for i in range(len(DAMS) - 1):
        u, dn = data[i], data[i + 1]
        mix = (u.spill * u.tw + (u.flow - u.spill) * u.fb) / u.flow          # gas of the mixed release, using the measured tailwater for spilled water
        best = None
        for lag in range(0, 5):
            z = pd.DataFrame({"x": mix.groupby(mix.index.year).shift(lag), "y": dn.fb, "year": dn.year}).dropna()
            z = z[(z.x > 90) & (z.x < 145) & (z.y > 90) & (z.y < 145)]
            tr, te = z[~z.year.isin(TEST)], z[z.year.isin(TEST)]
            c = np.polyfit(tr.x - 100, tr.y - 100, 1)
            err_tr = rmse(tr.y - 100 - np.polyval(c, tr.x - 100))
            if best is None or err_tr < best[0]:
                best = (err_tr, lag, c, rmse(te.y - 100 - np.polyval(c, te.x - 100)), rmse(te.y - tr.y.mean()))
        link.append(best[2])
        print(f"  {DAMS[i][0]}-{DAMS[i + 1][0]:<6} {best[1]:>10} {best[3]:>14.2f} {best[4]:>12.2f}   forebay excess = {best[2][0]:.2f} x upstream excess + {best[2][1]:.1f}")

    # ---------- Part 2: dam by dam against joint ----------
    shares = np.linspace(0, MAX_SHARE, 33)
    grid = np.linspace(95, 135, 161)                                          # forebay gas states

    def tail(i, share, flow, fbgas):
        return sat((share * flow, flow, fbgas), *par[i])

    def next_fb(i, share, flow, fbgas):
        mix = share * tail(i, share, flow, fbgas) + (1 - share) * fbgas
        return 100 + np.polyval(link[i], mix - 100)

    def logsurv(share):
        pass_spill = share ** 0.6
        return np.log(pass_spill * S_SPILL + (1 - pass_spill) * S_OTHER)

    def greedy(flows, fb0):
        g, tot_, plan = fb0, 0.0, []
        for i in range(8):
            ok = shares[tail(i, shares, flows[i], g) <= CAP]
            s = ok.max() if len(ok) else 0.0
            plan.append(s)
            tot_ += logsurv(s)
            if i < 7:
                g = float(np.clip(next_fb(i, s, flows[i], g), grid[0], grid[-1]))
        return tot_, plan

    def joint(flows, fb0):
        V = np.zeros(len(grid))                                               # value-to-go as a function of forebay gas
        choice = []
        for i in range(7, -1, -1):
            G, S = np.meshgrid(grid, shares, indexing="ij")
            tw = tail(i, S, flows[i], G)
            val = logsurv(S) + (np.interp(np.clip(next_fb(i, S, flows[i], G), grid[0], grid[-1]), grid, V) if i < 7 else 0.0)
            val = np.where(tw <= CAP, val, -1e9)
            val[:, 0] = np.maximum(val[:, 0], logsurv(0.0) + (np.interp(np.clip(next_fb(i, 0.0, flows[i], grid), grid[0], grid[-1]), grid, V) if i < 7 else 0.0))  # no spill is always allowed
            choice.insert(0, shares[val.argmax(axis=1)])
            V = val.max(axis=1)
        g, tot_, plan = fb0, 0.0, []
        for i in range(8):
            s = float(choice[i][np.abs(grid - g).argmin()])
            plan.append(s)
            tot_ += logsurv(s)
            if i < 7:
                g = float(np.clip(next_fb(i, s, flows[i], g), grid[0], grid[-1]))
        return tot_, plan

    flows = pd.DataFrame({fb: d.flow for (fb, _), d in zip(DAMS, data)})
    fb0 = data[0].fb
    days = [t for t in flows.index if t.year in TEST and flows.loc[t].notna().all() and np.isfinite(fb0.get(t, np.nan))]
    for cap in (120.0, 125.0):
        global CAP
        CAP = cap
        rows = []
        for t in days:
            a, pa = greedy(flows.loc[t].values, float(fb0[t]))
            b, pb = joint(flows.loc[t].values, float(fb0[t]))
            rows.append((np.exp(a), np.exp(b), np.mean(pa), np.mean(pb), t.year))
        r = np.array(rows)
        loss_a, loss_b = 1 - r[:, 0].mean(), 1 - r[:, 1].mean()
        print(f"\nPart 2. Tailwater gas cap {cap:.0f}%, {len(days)} held-out days (model study, assumed survival values)")
        print(f"  dam by dam: survival through eight dams {r[:, 0].mean():.4f}, mean spill share {r[:, 2].mean():.2f}")
        print(f"  joint:      survival through eight dams {r[:, 1].mean():.4f}, mean spill share {r[:, 3].mean():.2f}")
        print(f"  dam-passage loss {loss_a:.4f} -> {loss_b:.4f} ({loss_b / loss_a - 1:+.1%}); days joint is better {int((r[:, 1] > r[:, 0] + 1e-6).sum())}/{len(days)}; "
              f"by year: " + ", ".join(f"{y} {1 - r[r[:, 4] == y, 1].mean() - (1 - r[r[:, 4] == y, 0].mean()):+.4f}" for y in TEST))
        lo, hi = np.log(S_OTHER) * 8, 8 * logsurv(MAX_SHARE)
        print(f"  bounds: no spill {np.exp(lo):.4f}, maximum spill everywhere (ignoring the cap) {np.exp(hi):.4f}")


if __name__ == "__main__":
    main()
