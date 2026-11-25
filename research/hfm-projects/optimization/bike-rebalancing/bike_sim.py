"""Bike-share rebalancing replay on Helsinki city-bike trips.

Simulator and experiment for HFM's "Citywide Bicycle Rebalancing Policy for
Station-Based Bike-Sharing" track. Data: HSL city-bike origin-destination trips
(https://dev.hsl.fi/citybikes/, CC-BY 4.0), one CSV per month. HFM replays a
nextbike system; this is a different city with the same structure.

Model
  * Each trip rents a bike at its origin at its real departure time and returns it
    at its real destination after its real duration.
  * A rental fails if the origin is empty. A return fails if the destination is
    full (the bike is still accepted so bikes are conserved). Cost = failed
    rentals + failed returns in the day.
  * Vans (4, capacity 20, shift 06:00-22:00) move bikes. When a van finishes a
    stop the policy picks its next station and how many bikes to load or unload.
    Driving time comes from the median cycled distance between stations at 25 km/h
    plus 4 minutes of handling.
  * As in HFM's task, the policy sees station stocks and past events, not future
    rentals and not the destinations of bikes in use.

Assumptions that are NOT in the data: dock counts (set from each station's traffic,
12 to 44 docks), the fleet size (half the docks at 05:00), and the van fleet.

Policies
  no rebalancing    vans stay parked
  reactive          serve the emptiest or fullest station per minute of driving
  forecast          same, but on stock projected two hours ahead with hourly
                    demand rates learned from earlier weekdays
  look-ahead        rollout on top of "forecast": try the best few moves, replay
                    the next three hours on last week's same weekday, keep the best
  look-ahead,       the same, but each move is scored on four past weekdays and the
  4 past days       average is used
  perfect forecast  the look-ahead policy given the real future (not available in
                    the task; shows what better forecasts would be worth)

Days up to the 7th build the first forecasts; later weekdays are the test.
HFM's replay, constraints and score are private; numbers are NOT comparable.

Usage: python bike_sim.py <trips-month.csv> [max-test-days]
"""
from __future__ import annotations

import heapq
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_sim"))
from core import RolloutPolicy, Simulator, compare  # noqa: E402

VANS, VAN_CAP, SHIFT = 4, 20, (6 * 60.0, 22 * 60.0)
HANDLING, SPEED = 4.0, 25.0 / 60.0          # minutes per stop, km per minute
MARGIN, LOOK, HORIZON = 3, 120.0, 180.0     # keep this many bikes/docks free; forecast and rollout horizons (min)


def load(path):
    d = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 2, 4, 6, 7])
    d.columns = ["dep", "o", "d", "dist", "dur"]
    d = d.dropna()
    d["dep"] = pd.to_datetime(d["dep"], format="ISO8601")
    d = d[(d.dur > 60) & (d.dur < 5 * 3600)]
    ids = sorted(set(d.o) | set(d.d))
    idx = {s: i for i, s in enumerate(ids)}
    d["o"], d["d"] = d.o.map(idx), d.d.map(idx)
    d["day"] = d.dep.dt.day
    d["t"] = d.dep.dt.hour * 60 + d.dep.dt.minute + d.dep.dt.second / 60
    d["dur"] = d.dur / 60
    n = len(ids)
    # driving time: median cycled distance for observed pairs, shortest path through observed pairs otherwise
    med = d[d.o != d.d].groupby(["o", "d"]).dist.agg(["median", "size"]).reset_index()
    med = med[med["size"] >= 5]
    km = np.full((n, n), np.inf)
    km[med.o.values, med.d.values] = med["median"].values / 1000
    km = np.minimum(km, km.T)
    np.fill_diagonal(km, 0)
    for k in range(n):
        km = np.minimum(km, km[:, [k]] + km[[k], :])
    km[~np.isfinite(km)] = 15.0
    drive = km / SPEED + HANDLING
    daily = d.groupby("o").size().reindex(range(n), fill_value=0).values / d.day.nunique()
    cap = np.clip(np.round(10 + 0.12 * daily), 12, 44).astype(int)
    return d, n, drive, cap


def rates(d, n, days):
    """Mean net outflow per station per hour (rentals minus returns) over the given days."""
    x = d[d.day.isin(days)]
    out = np.zeros((n, 25))
    np.add.at(out, (x.o.values, (x.t // 60).astype(int).values), 1.0)
    np.subtract.at(out, (x.d.values, np.minimum(((x.t + x.dur) // 60).astype(int).values, 24)), 1.0)
    return out / max(len(days), 1)


class BikeSim(Simulator):
    def __init__(self, trips, n, drive, cap, net_rate, scenario=None, truth=None):
        self.o, self.d, self.t, self.dur = (trips[c].values for c in ("o", "d", "t", "dur"))
        self.n, self.drive, self.cap, self.net_rate = n, drive, cap, net_rate
        self.scenario, self.truth = scenario, truth      # demand used for look-ahead copies
        self.stock = (cap // 2).astype(int)
        self.returns = []                    # heap of (time, station)
        self.k = 0                           # next rental
        self.now, self.end = 300.0, 24 * 60.0 + 300
        self.cost_ = 0
        self.van_at = [int(np.argmax(cap))] * VANS
        self.van_load = [0] * VANS
        self.van_free = [SHIFT[0]] * VANS    # time the van is next free
        self.van_job = [None] * VANS         # (station, change in van load) to apply on arrival
        self.use_forecast = True

    def clone(self):
        c = BikeSim.__new__(BikeSim)
        c.__dict__.update(self.__dict__)
        c.stock, c.returns = self.stock.copy(), list(self.returns)
        c.van_at, c.van_load, c.van_free, c.van_job = self.van_at[:], self.van_load[:], self.van_free[:], self.van_job[:]
        return c

    def belief(self, k=0):
        """Look-ahead copy: future rentals and bikes in use come from a past day, not the real one.

        `scenario` may be one day or a list of past days; k picks which one.
        """
        c = self.clone()
        scen = self.scenario[k % len(self.scenario)] if isinstance(self.scenario, list) else self.scenario
        s = self.truth if self.truth is not None else scen
        so, sd, st, sdur = (s[x].values for x in ("o", "d", "t", "dur"))
        c.o, c.d, c.t, c.dur = so, sd, st, sdur
        c.k = int(np.searchsorted(st, self.now))
        riding = (st < self.now) & (st + sdur > self.now)
        c.returns = [(float(a), int(b)) for a, b in zip((st + sdur)[riding], sd[riding])]
        heapq.heapify(c.returns)
        c.end = min(self.end, self.now + HORIZON)
        c.scenario = c.truth = None
        return c

    def done(self):
        return self.now >= self.end

    def _advance(self, until):
        o, d, t, dur, stock, cap = self.o, self.d, self.t, self.dur, self.stock, self.cap
        m = len(t)
        while True:
            t_rent = t[self.k] if self.k < m else np.inf
            t_ret = self.returns[0][0] if self.returns else np.inf
            if min(t_rent, t_ret) > until:
                break
            if t_ret <= t_rent:
                _, s = heapq.heappop(self.returns)
                if stock[s] >= cap[s]:
                    self.cost_ += 1
                stock[s] += 1
            else:
                s = o[self.k]
                if stock[s] > 0:
                    stock[s] -= 1
                    heapq.heappush(self.returns, (t_rent + dur[self.k], d[self.k]))
                else:
                    self.cost_ += 1
                self.k += 1
        self.now = until

    def decisions(self):
        while True:
            nxt = min(self.van_free)
            if nxt >= min(SHIFT[1], self.end):
                self._advance(self.end)
                return []
            self._advance(max(nxt, self.now))
            v = int(np.argmin(self.van_free))
            if self.van_job[v] is not None:          # arrive: load or unload what is actually possible
                s, change = self.van_job[v]
                change = int(np.clip(change, -min(self.van_load[v], self.cap[s] - self.stock[s]), min(VAN_CAP - self.van_load[v], self.stock[s])))
                self.stock[s] -= change
                self.van_load[v] += change
                self.van_at[v], self.van_job[v] = s, None
            opts = self._options(v)
            if not opts:
                self.van_free[v] = self.now + 10.0   # nothing useful to do: check again in ten minutes
                continue
            return [(v, opts)]

    def _options(self, v):
        """Best few moves for a van: (station, change in van load), ranked by bikes fixed per minute."""
        proj = self.stock.astype(float)
        if self.use_forecast:
            h = int(self.now // 60)
            proj = proj - self.net_rate[:, min(h, 24)] * (LOOK / 60)
        need = np.maximum(MARGIN - proj, 0)                       # bikes missing
        surplus = np.maximum(proj - (self.cap - MARGIN), 0)       # docks missing
        target = self.cap / 2
        taken = {j[0] for j in self.van_job if j is not None}
        time = self.drive[self.van_at[v]]
        opts = []
        if self.van_load[v] > 0:
            for s in np.argsort(-need / time)[:3]:
                if need[s] > 0 and s not in taken:
                    drop = int(min(self.van_load[v], max(target[s] - proj[s], 1)))
                    opts.append((need[s] / time[s], (int(s), -drop)))
        if self.van_load[v] < VAN_CAP:
            for s in np.argsort(-surplus / time)[:3]:
                if surplus[s] > 0 and s not in taken:
                    take = int(min(VAN_CAP - self.van_load[v], max(proj[s] - target[s], 1)))
                    opts.append((surplus[s] / time[s], (int(s), take)))
            if self.van_load[v] == 0 and need.sum() > 0 and not any(o[1][1] > 0 for o in opts):
                # empty van and only shortages: fetch bikes from the best-stocked nearby station
                spare = np.maximum(proj - target, 0)
                s = int(np.argmax(spare / time))
                if spare[s] >= 2 and s not in taken:
                    opts.append((0.5 * spare[s] / time[s], (s, int(min(VAN_CAP, spare[s])))))
        return [o for _, o in sorted(opts, key=lambda x: -x[0])]

    def can_apply(self, v, option):
        return True

    def apply(self, v, option):
        s, change = option
        self.van_job[v] = (s, change)
        self.van_free[v] = self.now + self.drive[self.van_at[v], s]

    def cost(self):
        return float(self.cost_)


def best_first(sim, v, options):
    return options[0]


class NoVans(BikeSim):
    def decisions(self):
        self._advance(self.end)
        return []


class Reactive(BikeSim):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.use_forecast = False


def main():
    d, n, drive, cap = load(sys.argv[1])
    days = sorted(d.day.unique())
    weekday = {day: pd.Timestamp(d.dep.iloc[0].year, d.dep.iloc[0].month, day).weekday() for day in days}
    test = [day for day in days if day > 7 and weekday[day] < 5 and (day - 7) in days]
    if len(sys.argv) > 2:
        test = test[: int(sys.argv[2])]
    by_day = {day: g.sort_values("t") for day, g in d.groupby("day")}
    print(f"{n} stations, {cap.sum()} docks, {len(test)} test weekdays, {np.mean([len(by_day[x]) for x in test]):.0f} trips a day")

    def inst(day):
        hist = [x for x in days if x < day and weekday[x] < 5]
        past = [by_day[x] for x in sorted(hist, reverse=True) if x != day - 7][:3]      # three most recent other weekdays
        return dict(trips=by_day[day], net=rates(d, n, hist), scen=by_day[day - 7], scens=[by_day[day - 7]] + past)

    def mk(cls, truth=False, many=False):
        return lambda i: cls(i["trips"], n, drive, cap, i["net"], scenario=i["scens"] if many else i["scen"], truth=i["trips"] if truth else None)

    insts = [inst(day) for day in test]
    table = {"no rebalancing": (mk(NoVans), best_first), "reactive": (mk(Reactive), best_first), "forecast": (mk(BikeSim), best_first),
             "look-ahead (rollout)": (mk(BikeSim), RolloutPolicy(best_first)),
             "look-ahead, 4 past days": (mk(BikeSim, many=True), RolloutPolicy(best_first, scenarios=4)),
             "perfect forecast (not available)": (mk(BikeSim, True), RolloutPolicy(best_first))}
    # compare() takes one simulator factory, so wrap each policy with its own
    from core import run
    costs = {name: [run(make(i), pol) for i in insts] for name, (make, pol) in table.items()}
    ref = costs["forecast"]
    trips = np.mean([len(i["trips"]) for i in insts])
    print(f"\ncost = failed rentals + failed returns per day ({len(insts)} test days)")
    print(f"  {'policy':<34} {'mean cost':>10} {'share of trips':>15} {'vs forecast':>12} {'days better':>12} {'worse':>6}")
    for name, c in costs.items():
        print(f"  {name:<34} {np.mean(c):>10.0f} {np.mean(c) / trips:>15.2%} {np.mean(c) / np.mean(ref) - 1:>+12.1%} "
              f"{sum(a < b for a, b in zip(c, ref)):>8}/{len(c)} {sum(a > b for a, b in zip(c, ref)):>6}")


if __name__ == "__main__":
    main()
