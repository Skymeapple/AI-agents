"""Single-track railway dispatching on the Lielahti - Pohjois-Louko line.

Simulator and experiment for HFM's "Online Single-Track Railway Dispatcher for
Finnish Main-Line Replay" track, built on Fintraffic's open data (Digitraffic,
https://rata.digitraffic.fi, CC-BY 4.0): every train's planned and actual times.

The corridor has 17 stations and so 16 single-track sections, which is HFM's
description ("16-section single-track line ... with 15 passing loops").

Model
  * A section holds one train at a time, in either direction.
  * Stations (passing loops) have no capacity limit. That is a simplification; it
    removes deadlocks and leaves the meet-and-pass ordering problem.
  * Trains enter the corridor at their *actual* entry time (the real disturbance)
    and run each section in the planned running time. Passenger trains do not leave
    a commercial stop before the planned departure time.
  * Cost is weighted delay at the corridor exit in minutes (passenger trains weigh
    3, others 1), against the planned exit time.
  * "Disrupted" days add extra entry delays to a random 30% of trains (exponential,
    mean 12 minutes), drawn once per day so every policy sees the same day.

Policies
  first come       the train that has waited longest gets a free section
  priority         passenger before freight, then most delayed
  look-ahead       rollout on top of "priority": try each option (including
                   waiting), finish the day with "priority", keep the cheapest.
                   Trains that have not yet entered are assumed to be on plan.
  hindsight bound  mixed-integer programme over the whole day with all real entry
                   times known (HiGHS, time-limited), which no online policy can beat

HFM's replay, cost weights and score are private; numbers are NOT comparable.

Usage: python rail_sim.py <dir-with-digitraffic-day-json-files> [max-days]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_sim"))
from core import RolloutPolicy, Simulator, compare  # noqa: E402

CORRIDOR = ["LLH", "YLÖ", "LAK", "MJJ", "KRR", "STÖ", "VJR", "PKK", "PKO", "LMK", "KIS", "RLÄ", "MD", "YLV", "JAL", "PSJ", "PLU"]
POS = {c: i for i, c in enumerate(CORRIDOR)}


def minutes(ts, day0):
    return (datetime.fromisoformat(ts.replace("Z", "+00:00")) - day0).total_seconds() / 60.0


def load_day(path):
    """Trains that run on the corridor, each as a list of (section, running time, earliest departure)."""
    day0 = datetime.fromisoformat(os.path.basename(path)[:10] + "T00:00:00+00:00")
    trains = []
    for t in json.load(open(path)):
        if t.get("cancelled"):
            continue
        rows = [r for r in t["timeTableRows"] if r["stationShortCode"] in POS and not r.get("cancelled")]
        if len(rows) < 2:
            continue
        stops = []  # per station: [pos, planned arrival, planned departure, actual arrival or departure, commercial]
        for r in rows:
            p = POS[r["stationShortCode"]]
            if not stops or stops[-1][0] != p:
                stops.append([p, None, None, None, False])
            s = minutes(r["scheduledTime"], day0)
            a = minutes(r["actualTime"], day0) if r.get("actualTime") else None
            if r["type"] == "ARRIVAL":
                stops[-1][1] = s
                stops[-1][3] = a if stops[-1][3] is None else stops[-1][3]
            else:
                stops[-1][2] = s
                stops[-1][3] = a if stops[-1][3] is None else stops[-1][3]
            stops[-1][4] = stops[-1][4] or bool(r.get("trainStopping") and r.get("commercialStop"))
        if len(stops) < 2 or any(abs(b[0] - a[0]) != 1 for a, b in zip(stops, stops[1:])):
            continue
        legs = []
        for a, b in zip(stops, stops[1:]):
            dep = a[2] if a[2] is not None else a[1]
            arr = b[1] if b[1] is not None else b[2]
            legs.append((min(a[0], b[0]), max(arr - dep, 1.0), dep if a[4] else -1e9))
        first = stops[0]
        plan_in = first[1] if first[1] is not None else first[2]
        actual_in = first[3] if first[3] is not None else plan_in
        last = stops[-1]
        trains.append(dict(id=t["trainNumber"], w=3.0 if t["trainCategory"] in ("Long-distance", "Commuter") else 1.0,
                           legs=legs, plan_in=plan_in, actual_in=max(actual_in, plan_in - 30), plan_out=last[1] if last[1] is not None else last[2]))
    return sorted(trains, key=lambda x: x["actual_in"])


def disrupt(trains, seed, share=0.3, mean=12.0):
    rng = np.random.default_rng(seed)
    out = []
    for t in trains:
        extra = rng.exponential(mean) if rng.random() < share else 0.0
        out.append(dict(t, actual_in=t["actual_in"] + extra))
    return sorted(out, key=lambda x: x["actual_in"])


class RailSim(Simulator):
    def __init__(self, trains, known_at=None):
        self.trains = trains
        n = len(trains)
        self.leg = [0] * n                       # index of the next leg
        self.ready = [t["actual_in"] for t in trains]   # time the train can next request a section
        self.wait_since = list(self.ready)
        self.running = [False] * n
        self.finish = [None] * n
        self.free_at = {}                        # section -> time it clears
        self.now = min(self.ready) if n else 0.0
        self.held = set()                        # trains told to wait until the next event

    def clone(self):
        c = RailSim.__new__(RailSim)
        c.trains = self.trains
        c.leg, c.ready, c.wait_since = self.leg[:], self.ready[:], self.wait_since[:]
        c.running, c.finish = self.running[:], self.finish[:]
        c.free_at, c.now, c.held = dict(self.free_at), self.now, set(self.held)
        return c

    def belief(self):
        """Trains that have not entered yet are assumed to enter on plan (or now, if already overdue)."""
        c = self.clone()
        for i, t in enumerate(self.trains):
            if self.leg[i] == 0 and not self.running[i] and t["actual_in"] > self.now:
                c.ready[i] = c.wait_since[i] = max(t["plan_in"], self.now + 1e-6)
        return c

    def done(self):
        return all(f is not None for f in self.finish)

    def _requests(self):
        """Sections that are free now, each with the trains ready to enter it."""
        req = {}
        for i, t in enumerate(self.trains):
            if self.finish[i] is None and not self.running[i] and self.ready[i] <= self.now + 1e-9 and i not in self.held:
                sec = t["legs"][self.leg[i]][0]
                if self.free_at.get(sec, -1e18) <= self.now + 1e-9:
                    req.setdefault(sec, []).append(i)
        return req

    def _next_time(self):
        times = [r for i, r in enumerate(self.ready) if self.finish[i] is None and r > self.now + 1e-9]
        times += [f for f in self.free_at.values() if f > self.now + 1e-9]
        return min(times) if times else None

    def decisions(self):
        while True:
            req = self._requests()
            if req:
                sec = min(req)
                # a wait option is only offered if something else is still moving, so the run cannot stall
                opts = sorted(req[sec], key=lambda i: self.wait_since[i])
                return [(sec, opts + ([None] if self._next_time() is not None else []))]
            nxt = self._next_time()
            if nxt is None:
                if self.held:               # nothing else can happen: release held trains
                    self.held.clear()
                    continue
                return []
            self.now = nxt
            self.held.clear()

    def can_apply(self, key, option):
        return True

    def apply(self, sec, i):
        if i is None:
            for j in self._requests().get(sec, []):
                self.held.add(j)
            return
        t = self.trains[i]
        _, run_time, _ = t["legs"][self.leg[i]]
        arrive = self.now + run_time
        self.free_at[sec] = arrive
        self.leg[i] += 1
        if self.leg[i] == len(t["legs"]):
            self.finish[i] = arrive
        else:
            self.ready[i] = max(arrive, t["legs"][self.leg[i]][2])
            self.wait_since[i] = self.ready[i]

    def cost(self):
        return sum(t["w"] * max(0.0, (f if f is not None else self.now + 600) - t["plan_out"]) for t, f in zip(self.trains, self.finish))


def first_come(sim, sec, options):
    return options[0]


def priority(sim, sec, options):
    cands = [o for o in options if o is not None]
    return max(cands, key=lambda i: (sim.trains[i]["w"], sim.now - sim.trains[i]["plan_in"] - sum(l[1] for l in sim.trains[i]["legs"][:sim.leg[i]])))


def make_sim(trains):
    sim = RailSim(trains)
    # first departure also respects the planned time at a commercial stop
    sim.ready = [max(t["actual_in"], t["legs"][0][2]) for t in trains]
    sim.wait_since = list(sim.ready)
    sim.now = min(sim.ready)
    return sim


def hindsight(trains, time_limit=60.0, window=240.0):
    """Whole-day mixed-integer programme with real entry times known."""
    idx, legs = {}, []
    for i, t in enumerate(trains):
        for k, (sec, rt, dep) in enumerate(t["legs"]):
            idx[(i, k)] = len(legs)
            legs.append((i, k, sec, rt, dep))
    nT, nL = len(trains), len(legs)
    by_sec = {}
    for v, (i, k, sec, rt, dep) in enumerate(legs):
        by_sec.setdefault(sec, []).append(v)
    plan_t = [trains[i]["plan_in"] + sum(l[1] for l in trains[i]["legs"][:k]) for i, k, *_ in legs]
    pairs = [(a, b) for vs in by_sec.values() for x, a in enumerate(vs) for b in vs[x + 1:]
             if legs[a][0] != legs[b][0] and abs(plan_t[a] - plan_t[b]) < window]
    nvar = nL + nT + len(pairs)          # entry time per leg, delay per train, order per pair
    M = 1500.0
    rows = nL + nT + 2 * len(pairs)
    A = lil_matrix((rows, nvar))
    lo, hi = np.full(rows, -np.inf), np.full(rows, np.inf)
    lb, ub = np.zeros(nvar), np.full(nvar, np.inf)
    integ = np.zeros(nvar)
    c = np.zeros(nvar)
    r = 0
    for v, (i, k, sec, rt, dep) in enumerate(legs):
        lb[v] = max(trains[i]["actual_in"] if k == 0 else 0.0, dep, 0.0)
        ub[v] = 3000.0
        if k:                                   # cannot enter before arriving from the previous section
            p = idx[(i, k - 1)]
            A[r, v] = 1; A[r, p] = -1; lo[r] = legs[p][3]; r += 1
    for i, t in enumerate(trains):              # delay >= exit time - planned exit
        last = idx[(i, len(t["legs"]) - 1)]
        A[r, nL + i] = 1; A[r, last] = -1; lo[r] = legs[last][3] - t["plan_out"]; r += 1
        c[nL + i] = t["w"]
    for q, (a, b) in enumerate(pairs):          # one of the two trains clears the section first
        y = nL + nT + q
        integ[y] = 1; ub[y] = 1
        A[r, b] = 1; A[r, a] = -1; A[r, y] = M; lo[r] = legs[a][3]; r += 1          # y = 0: a first
        A[r, a] = 1; A[r, b] = -1; A[r, y] = -M; lo[r] = legs[b][3] - M; r += 1     # y = 1: b first
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo[:r], hi[:r]), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-3})
    # the dual bound is a true lower bound even if the time limit stops the search early
    return float(getattr(res, "mip_dual_bound", res.fun if res.x is not None else 0.0))


def main():
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
    if len(sys.argv) > 2:
        files = files[: int(sys.argv[2])]
    days = [d for d in (load_day(f) for f in files) if len(d) >= 20]
    print(f"{len(days)} days, {np.mean([len(d) for d in days]):.0f} trains a day on the corridor")
    policies = {"first come": first_come, "priority": priority, "look-ahead (rollout)": RolloutPolicy(priority)}
    compare(days, make_sim, policies, "priority", bound=hindsight, label="real days, real entry delays")
    disrupted = [disrupt(d, seed=k) for k, d in enumerate(days)]
    compare(disrupted, make_sim, policies, "priority", bound=hindsight, label="disrupted days (extra entry delays on 30% of trains)")


if __name__ == "__main__":
    main()
