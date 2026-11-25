"""Ride-hail matching and repositioning on New York City trip records.

Simulator and experiment for HFM's "Ride-Hail Dispatch Policy for Matching and
Repositioning" track. Data: NYC Taxi and Limousine Commission high-volume for-hire
trip records (public, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
January 2024: request time, pickup and drop-off zone, trip time and fare.

Model
  * A 4% sample of real requests is replayed at their real request times.
  * A fleet of drivers starts spread over the zones in proportion to demand. Every
    30 seconds, waiting riders are matched to idle drivers. Driving time between
    zones is the median real trip time for that pair of zones (shortest path
    through observed pairs where a pair was never observed).
  * A rider is lost if not matched within 3 minutes or if no driver can arrive
    within 8 minutes. Score = share of requests served, and fares served.
  * Every 5 minutes idle drivers may be repositioned. Repositioning takes the real
    driving time, during which the driver is unavailable.

Policies
  nearest driver      each rider, in order of request, takes the nearest idle driver
  batch assignment    all waiting riders and idle drivers are matched together to
                      serve as many as possible, then minimise pickup time
  + repositioning     batch assignment, and idle drivers move towards zones where
                      forecast demand over the next 15 minutes exceeds idle supply.
                      The forecast is the same hour on earlier days of the same type
  + perfect forecast  the same with the real coming demand (not available; shows
                      what a perfect demand forecast would be worth)

The fleet size is set so the simple policy serves roughly 85% of requests, which
leaves room to see differences. HFM's replay, fleet, constraints and score are
private; numbers are NOT comparable.

Usage: python ridehail_sim.py <fhvhv_tripdata_2024-01.parquet> [drivers]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

TICK, PATIENCE, MAX_ETA, REPO_EVERY, LOOK = 0.5, 3.0, 8.0, 5.0, 15.0     # minutes
SAMPLE, NZ = 0.04, 266


def load(path):
    cols = ["request_datetime", "PULocationID", "DOLocationID", "trip_time", "base_passenger_fare"]
    d = pq.read_table(path, columns=cols).to_pandas().dropna()
    d = d[(d.trip_time > 60) & (d.trip_time < 3 * 3600) & (d.PULocationID < NZ) & (d.DOLocationID < NZ)]
    d["day"] = d.request_datetime.dt.day
    d["t"] = d.request_datetime.dt.hour * 60 + d.request_datetime.dt.minute + d.request_datetime.dt.second / 60
    d["tt"] = d.trip_time / 60
    med = d.groupby(["PULocationID", "DOLocationID"]).tt.agg(["median", "size"]).reset_index()
    med = med[med["size"] >= 20]
    T = np.full((NZ, NZ), np.inf)
    T[med.PULocationID.values, med.DOLocationID.values] = med["median"].values
    T = np.minimum(T, T.T)
    np.fill_diagonal(T, 3.0)
    for k in range(NZ):
        T = np.minimum(T, T[:, [k]] + T[[k], :])
    T[~np.isfinite(T)] = 90.0
    return d, T


def simulate(day, T, drivers, policy, forecast=None, seed=0):
    rng = np.random.default_rng(seed)
    pu, do, t, tt, fare = (day[c].values for c in ("PULocationID", "DOLocationID", "t", "tt", "base_passenger_fare"))
    n = len(t)
    start = rng.choice(pu, size=drivers)                 # drivers start where demand is
    zone, free_at = start.copy(), np.zeros(drivers)
    waiting, k = [], 0
    served = np.zeros(n, bool)
    pickup_wait, deadhead = [], 0.0
    next_repo = 0.0
    for now in np.arange(0.0, 24 * 60, TICK):
        while k < n and t[k] <= now:
            waiting.append(k)
            k += 1
        waiting = [r for r in waiting if now - t[r] <= PATIENCE]
        idle = np.flatnonzero(free_at <= now)
        if waiting and len(idle):
            W = np.array(waiting)
            eta = T[zone[idle][:, None], pu[W][None, :]]                    # (drivers, riders)
            pairs = []
            if policy == "nearest":
                free = np.ones(len(idle), bool)
                for j in range(len(W)):
                    c = np.where(free, eta[:, j], np.inf)
                    i = int(np.argmin(c))
                    if c[i] <= MAX_ETA:
                        pairs.append((i, j))
                        free[i] = False
            else:
                cost = np.where(eta <= MAX_ETA, eta, 1e4)                   # serve as many as possible, then least waiting
                ri, cj = linear_sum_assignment(cost)
                pairs = [(i, j) for i, j in zip(ri, cj) if eta[i, j] <= MAX_ETA]
            done = set()
            for i, j in pairs:
                d_, r = idle[i], W[j]
                served[r] = True
                pickup_wait.append(now - t[r] + eta[i, j])
                free_at[d_] = now + eta[i, j] + tt[r]
                zone[d_] = do[r]
                done.add(r)
            waiting = [r for r in waiting if r not in done]
        if forecast is not None and now >= next_repo:
            next_repo = now + REPO_EVERY
            idle = np.flatnonzero(free_at <= now - 1.0)
            if len(idle):
                if isinstance(forecast, str):                               # perfect forecast: real requests in the window
                    a, b = np.searchsorted(t, now), np.searchsorted(t, now + LOOK)
                    need = np.bincount(pu[a:b], minlength=NZ).astype(float)
                else:
                    need = forecast[:, min(int(now // 60), 23)] * LOOK / 60.0
                supply = np.bincount(zone[free_at <= now], minlength=NZ).astype(float)
                soon = (free_at > now) & (free_at <= now + LOOK)
                supply += 0.5 * np.bincount(zone[soon], minlength=NZ)
                gap = need - supply
                order = idle[np.argsort(gap[zone[idle]])]                    # move drivers out of the most over-supplied zones first
                for d_ in order:
                    z = zone[d_]
                    if gap[z] > -1.0:
                        break
                    cand = np.flatnonzero((gap > 0.5) & (T[z] <= 12.0))
                    if len(cand) == 0:
                        continue
                    dest = cand[np.argmax(gap[cand] / T[z, cand])]
                    gap[z] += 1; gap[dest] -= 1
                    free_at[d_] = now + T[z, dest]
                    deadhead += T[z, dest]
                    zone[d_] = dest
    return dict(served=served.mean(), fares=fare[served].sum() / fare.sum(), wait=float(np.mean(pickup_wait)), deadhead=deadhead / drivers)


def main():
    d, T = load(sys.argv[1])
    rng = np.random.default_rng(0)
    d = d[rng.random(len(d)) < SAMPLE].sort_values(["day", "t"])
    by_day = {k: g for k, g in d.groupby("day")}
    dow = {k: pd.Timestamp(2024, 1, k).weekday() for k in by_day}
    test = [k for k in sorted(by_day) if 15 <= k <= 28]
    drivers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if drivers is None:                                  # size the fleet on a day outside the test set
        for drivers in range(300, 3000, 100):
            if simulate(by_day[9], T, drivers, "nearest")["served"] >= 0.85:
                break
    print(f"{len(test)} test days, {np.mean([len(by_day[k]) for k in test]):.0f} requests a day in the sample, {drivers} drivers")
    rows = []
    for k in test:
        hist = [x for x in by_day if x < k and (dow[x] < 5) == (dow[k] < 5)]
        fc = np.zeros((NZ, 24))
        for x in hist:
            g = by_day[x]
            np.add.at(fc, (g.PULocationID.values, (g.t // 60).astype(int).clip(0, 23).values), 1.0 / len(hist))
        res = {"nearest driver": simulate(by_day[k], T, drivers, "nearest"),
               "batch assignment": simulate(by_day[k], T, drivers, "batch"),
               "+ repositioning (forecast)": simulate(by_day[k], T, drivers, "batch", forecast=fc),
               "+ repositioning (perfect forecast)": simulate(by_day[k], T, drivers, "batch", forecast="truth")}
        rows += [dict(day=k, policy=p, **v) for p, v in res.items()]
        print(f"  day {k}: " + ", ".join(f"{p} {v['served']:.3f}" for p, v in res.items()), flush=True)
    r = pd.DataFrame(rows)
    piv = r.pivot(index="day", columns="policy", values="served")
    base = piv["nearest driver"]
    print(f"\n{'policy':<36} {'served':>7} {'lost requests vs nearest':>25} {'fares served':>13} {'pickup wait':>12} {'empty driving per driver':>25} {'days better':>12}")
    for p in ("nearest driver", "batch assignment", "+ repositioning (forecast)", "+ repositioning (perfect forecast)"):
        x = r[r.policy == p]
        lost = (1 - piv[p]).mean() / (1 - base).mean() - 1
        print(f"{p:<36} {x.served.mean():>7.3f} {lost:>+25.1%} {x.fares.mean():>13.3f} {x.wait.mean():>9.1f} min {x.deadhead.mean():>21.0f} min {int((piv[p] > base + 1e-9).sum()):>8}/{len(piv)}")


if __name__ == "__main__":
    main()
