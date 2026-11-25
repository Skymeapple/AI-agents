"""Bus holding control on Helsinki's trunk routes.

Simulator and experiment for HFM's "Control Policy for Frequent Bus Services"
track, which replays HSL trunk routes 500, 510, 560, 570 and 200 and lets a
policy hold buses at permitted timing points.

What is real: the stop sequences, scheduled running times, morning-peak headways
and timing points of those five routes, taken from HSL's public timetable feed
(GTFS, https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip, CC-BY 4.0) and stored
in hsl_trunk_routes.json.

What is assumed, because HSL does not publish it openly per stop and bus: passenger
demand (a triangular boarding profile sized so the peak load is about half a bus),
running-time variability (15% per link plus a shared 10% per link and half hour),
late departures from the terminal (mean 45 s), and boarding times. HFM replays
recorded vehicle movements, so this is a structurally faithful stand-in, not a
replay.

Model: buses cannot overtake. Passengers arrive at a steady rate; their waiting
grows with the square of the gap between buses, which is why bunching is costly.
Cost = passenger waiting at stops + delay to passengers on a held bus, in
passenger-minutes.

Policies (holds of 0 to 120 s at timing points)
  no holding
  schedule         hold until the scheduled departure time
  forward headway  hold until the gap to the bus ahead reaches 80% of the headway
  two-way          hold to centre the bus between the one ahead and the one behind
  look-ahead       rollout: try each hold, finish the run with expected running
                   times under a simple rule ("two-way" or "no holding"), keep
                   the cheapest
  perfect info     the rollout on "no holding" given the real future running
                   times (not available in the task)

Each policy is run at the assumed variability and at double that.

HFM's replay, cost and score are private; numbers are NOT comparable.

Usage: python bus_sim.py [seeds-per-route]
"""
from __future__ import annotations

import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_sim"))
from core import RolloutPolicy, Simulator, run  # noqa: E402

HOLDS = (0.0, 30.0, 60.0, 90.0, 120.0)
CAPACITY, MIN_GAP, BASE_DWELL, BOARD_S, ALIGHT_S = 90.0, 15.0, 8.0, 2.5, 1.5
PERIOD = 3 * 3600.0


def make_instance(route, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    links = np.array(route["link_seconds"], float)
    S, H = len(links) + 1, route["headway_seconds"]
    K = int(PERIOD / H)
    x = np.linspace(0, 1, S)
    board = np.maximum(1 - x / 0.75, 0.05)                 # most boarding early on the route
    board[-1] = 0.0
    lam = board / board.sum() * (0.5 * CAPACITY * 1.6) / H  # passengers per second per stop
    alight = np.clip((x - 0.25) * 0.25, 0, 1)
    alight[-1] = 1.0
    shared = np.exp(0.10 * noise * rng.standard_normal((6, S - 1)))    # per link, per half hour
    own = np.exp(0.15 * noise * rng.standard_normal((K, S - 1)) - 0.5 * (0.15 * noise) ** 2)
    late = rng.exponential(45.0 * noise, K)
    return dict(links=links, H=H, K=K, S=S, lam=lam, alight=alight, shared=shared, own=own, late=late,
                tp=set(route["timing_points"]) - {0, S - 1})


class BusSim(Simulator):
    def __init__(self, inst):
        self.i = inst
        K, S = inst["K"], inst["S"]
        self.mult = np.array([[inst["own"][k, s] * inst["shared"][min(int(k * inst["H"] // 1800), 5), s] for s in range(S - 1)] for k in range(K)])
        self.start = np.arange(K) * inst["H"] + inst["late"]
        self.events = [(float(self.start[k]), k, 0) for k in range(K)]
        heapq.heapify(self.events)
        self.last_dep = [None] * S           # last departure time per stop
        self.queue = [0.0] * S               # passengers left behind
        self.load = [0.0] * K
        self.where = [(float(self.start[k]), 0) for k in range(K)]   # next known (time, stop) per bus
        self.cost_ = 0.0
        self.pending = None                  # (bus, stop, ready time) awaiting a holding decision
        self.cum = np.concatenate([[0.0], np.cumsum(inst["links"])])

    def clone(self):
        c = BusSim.__new__(BusSim)
        c.__dict__.update(self.__dict__)
        c.events, c.last_dep, c.queue, c.load, c.where = list(self.events), self.last_dep[:], self.queue[:], self.load[:], self.where[:]
        return c

    def belief(self):
        """Future running times are replaced by their expected values."""
        c = self.clone()
        c.mult = np.ones_like(self.mult)
        return c

    def done(self):
        return not self.events and self.pending is None

    def _depart(self, k, s, t):
        self.last_dep[s] = t
        nxt = t + self.i["links"][s] * self.mult[k, s]
        self.where[k] = (nxt, s + 1)
        heapq.heappush(self.events, (nxt, k, s + 1))

    def decisions(self):
        i = self.i
        while self.events:
            t, k, s = heapq.heappop(self.events)
            prev = self.last_dep[s]
            if prev is not None:
                t = max(t, prev + MIN_GAP)
            gap = t - prev if prev is not None else i["H"]
            waiting = self.queue[s] + i["lam"][s] * gap
            self.cost_ += i["lam"][s] * gap * gap / 2 + self.queue[s] * gap
            off = self.load[k] * i["alight"][s]
            self.load[k] -= off
            on = min(waiting, CAPACITY - self.load[k])
            self.queue[s] = waiting - on
            self.load[k] += on
            if s == i["S"] - 1:
                self.where[k] = (np.inf, s)
                continue
            ready = t + BASE_DWELL + max(BOARD_S * on, ALIGHT_S * off)
            if s in i["tp"]:
                self.pending = (k, s, ready)
                return [(k, list(HOLDS))]
            self._depart(k, s, ready)
        return []

    def can_apply(self, key, option):
        return True

    def apply(self, k, hold):
        _, s, ready = self.pending
        self.pending = None
        self.cost_ += self.load[k] * hold
        self._depart(k, s, ready + hold)

    def cost(self):
        return self.cost_ / 60.0

    # --- what a controller can observe at a holding decision ---
    def gaps(self):
        """(time since the bus ahead left this stop, expected time until the bus behind arrives here)."""
        k, s, ready = self.pending
        prev = self.last_dep[s]
        fwd = ready - prev if prev is not None else self.i["H"]
        if k + 1 < self.i["K"]:
            t2, s2 = self.where[k + 1]
            eta = t2 + (self.cum[s] - self.cum[s2]) + BASE_DWELL * (s - s2) if s2 <= s else ready
            bwd = max(eta - ready, 0.0)
        else:
            bwd = self.i["H"]
        return fwd, bwd

    def schedule_slack(self):
        k, s, ready = self.pending
        return k * self.i["H"] + self.cum[s] + BASE_DWELL * s - ready


def nearest(options, want):
    return min(options, key=lambda h: abs(h - want))


def no_holding(sim, k, options):
    return 0.0


def schedule(sim, k, options):
    return nearest(options, max(sim.schedule_slack(), 0.0))


def forward_headway(sim, k, options):
    fwd, _ = sim.gaps()
    return nearest(options, max(0.8 * sim.i["H"] - fwd, 0.0))


def two_way(sim, k, options):
    fwd, bwd = sim.gaps()
    return nearest(options, max(0.5 * (bwd - fwd), 0.0))


class PerfectInfo(BusSim):
    def belief(self):
        return self.clone()


def evaluate(routes, seeds, noise, label):
    table = {"no holding": (BusSim, no_holding), "schedule": (BusSim, schedule), "forward headway": (BusSim, forward_headway),
             "two-way": (BusSim, two_way),
             "look-ahead on two-way": (BusSim, RolloutPolicy(two_way, max_options=5)),
             "look-ahead on no holding": (BusSim, RolloutPolicy(no_holding, max_options=5)),
             "perfect info (not available)": (PerfectInfo, RolloutPolicy(no_holding, max_options=5))}
    costs = {n: [] for n in table}
    for name, route in routes.items():
        for seed in range(seeds):
            inst = make_instance(route, 1000 * int(name) + seed, noise)
            for n, (cls, pol) in table.items():
                costs[n].append(run(cls(inst), pol))
    ref = costs["no holding"]
    print(f"\n{label}: {len(ref)} instances (5 routes x {seeds} random days); cost = passenger-minutes of waiting and holding")
    print(f"  {'policy':<30} {'vs no holding':>14} {'instances better':>17} {'worse':>6}")
    for n, c in costs.items():
        print(f"  {n:<30} {np.mean(c) / np.mean(ref) - 1:>+14.1%} {sum(a < b - 1e-6 for a, b in zip(c, ref)):>13}/{len(c)} "
              f"{sum(a > b + 1e-6 for a, b in zip(c, ref)):>6}", flush=True)


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    routes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsl_trunk_routes.json")))
    evaluate(routes, seeds, 1.0, "assumed variability")
    evaluate(routes, seeds, 2.0, "doubled variability (30% per link, 90 s terminal lateness)")


if __name__ == "__main__":
    main()
