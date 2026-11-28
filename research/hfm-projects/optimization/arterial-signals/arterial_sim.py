"""Coordinated signal timing on an arterial: tune for one counted hour, or for a range?

Experiment for HFM's "Toronto Arterial Signal Timing Coordination" track, where an
engineer picks splits and offsets for six or seven signals from one counted hour.
HFM's score there is 0.989, so the open question is not how to optimise that hour
but whether a plan tuned to one hour holds up on other days.

This is a model study. The corridor geometry and volumes are generated within
typical urban ranges (they are NOT Toronto's counts), so it tests the idea, not the
city. Toronto's turning-movement counts are public and could replace the generated
volumes.

Model: 6 or 7 signals on a two-way arterial, common 90 s cycle, two phases each
(arterial and cross street). Vehicles are simulated second by second as flows:
queues discharge at the saturation rate during green and platoons arrive downstream
after the link travel time. Cost is total delay in vehicle-hours over one hour.

Plans compared, each evaluated on 30 other days whose volumes differ from the
counted hour by a day-wide factor (standard deviation 12%) and per-approach noise:
  Webster          splits from Webster's rule, all offsets zero
  + progression    Webster splits, offsets set to the travel time in the heavier direction
  tuned to hour    splits and offsets searched to minimise delay on the counted hour
  tuned to range   the same search, minimising average delay over 8 sampled days

Usage: python arterial_sim.py [corridors]
"""
from __future__ import annotations

import sys

import numpy as np

C, SAT, HOUR, LOST = 90, 0.5, 3600, 4     # cycle (s), saturation flow per lane-group (veh/s), horizon, lost time per phase


def make_corridor(rng):
    n = int(rng.integers(6, 8))
    return dict(n=n, travel=rng.integers(18, 45, n - 1),               # seconds between neighbouring signals
                east=rng.uniform(500, 1000) / 3600, west=rng.uniform(300, 800) / 3600,   # arterial through volumes (veh/s)
                cross=rng.uniform(250, 650, n) / 3600, turn_in=rng.uniform(0.03, 0.10, n), turn_out=rng.uniform(0.03, 0.10, n))


def day(cor, rng, sd=0.12):
    f = rng.normal(1.0, sd)
    return dict(east=cor["east"] * f * rng.normal(1, 0.05), west=cor["west"] * f * rng.normal(1, 0.05),
                cross=cor["cross"] * f * rng.normal(1, 0.08, cor["n"]))


def delay(cor, vol, green, offset):
    """Total delay (veh-h). green[i]: arterial green seconds; offset[i]: start of arterial green in the cycle."""
    n = cor["n"]
    t = np.arange(HOUR)
    is_green = ((t[None, :] - offset[:, None]) % C) < green[:, None]             # arterial green
    cross_green = ~is_green
    cross_green &= ((t[None, :] - offset[:, None] - green[:, None]) % C) >= LOST  # start-up lost time on each phase
    art_green = is_green & (((t[None, :] - offset[:, None]) % C) >= LOST)
    total = 0.0
    for direction in (1, -1):
        order = range(n) if direction == 1 else range(n - 1, -1, -1)
        arrivals = np.full(HOUR, vol["east"] if direction == 1 else vol["west"])
        for pos, i in enumerate(order):
            a = arrivals * (1 - cor["turn_out"][i]) + vol["cross"][i] * cor["turn_in"][i]
            q, out = 0.0, np.zeros(HOUR)
            g = art_green[i]
            for s in range(HOUR):
                q += a[s]
                d = min(q, SAT) if g[s] else 0.0
                q -= d
                out[s] = d
                total += q
            nxt = i + direction
            if 0 <= nxt < n:
                tt = int(cor["travel"][min(i, nxt)])
                arrivals = np.concatenate([np.full(tt, out[:C].mean()), out[:-tt]])
        # cross streets: uniform arrivals, both approaches lumped
    for i in range(n):
        q = 0.0
        g = cross_green[i]
        a = vol["cross"][i]
        for s in range(HOUR):
            q += a
            d = min(q, SAT) if g[s] else 0.0
            q -= d
            total += q
    return total / 3600.0


def webster(cor, vol):
    y_art = max(vol["east"], vol["west"]) / SAT
    g = np.array([(C - 2 * LOST) * y_art / (y_art + vol["cross"][i] / SAT) + LOST for i in range(cor["n"])])
    return np.clip(np.round(g), 20, C - 20)


def progression(cor, vol):
    off = np.zeros(cor["n"])
    if vol["east"] >= vol["west"]:
        off[1:] = np.cumsum(cor["travel"])
    else:
        off[:-1] = np.cumsum(cor["travel"][::-1])[::-1]
    return off % C


def tune(cor, days, green, offset, sweeps=2):
    """Coordinate search over splits and offsets on the given days."""
    cost = lambda g, o: np.mean([delay(cor, d, g, o) for d in days])
    best = cost(green, offset)
    for _ in range(sweeps):
        for i in range(cor["n"]):
            for arr, steps, lo, hi in ((offset, (-20, -10, -5, 5, 10, 20), None, None), (green, (-8, -4, 4, 8), 20, C - 20)):
                for st in steps:
                    trial = arr.copy()
                    trial[i] = (trial[i] + st) % C if lo is None else np.clip(trial[i] + st, lo, hi)
                    c = cost(trial, offset) if arr is green else cost(green, trial)
                    if c < best - 1e-9:
                        best = c
                        arr[i] = trial[i]
    return green, offset


def main():
    n_cor = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = []
    for k in range(n_cor):
        rng = np.random.default_rng(100 + k)
        cor = make_corridor(rng)
        counted = day(cor, rng)
        sample = [day(cor, rng) for _ in range(8)]
        others = [day(cor, np.random.default_rng(9000 + 31 * k + j)) for j in range(30)]
        g0 = webster(cor, counted)
        plans = {"Webster, zero offsets": (g0.copy(), np.zeros(cor["n"])), "Webster + progression": (g0.copy(), progression(cor, counted))}
        g, o = tune(cor, [counted], g0.copy(), progression(cor, counted).copy())
        plans["tuned to the counted hour"] = (g.copy(), o.copy())
        g2, o2 = tune(cor, sample, g0.copy(), progression(cor, counted).copy())
        plans["tuned to a range of days"] = (g2, o2)
        res = {name: (delay(cor, counted, *p), np.array([delay(cor, d, *p) for d in others])) for name, p in plans.items()}
        rows.append(res)
        print(f"  corridor {k} ({cor['n']} signals): " + ", ".join(f"{n_} {v[1].mean():.0f}" for n_, v in res.items()), flush=True)
    base = "tuned to the counted hour"
    print(f"\ndelay in vehicle-hours, mean over {n_cor} corridors")
    print(f"  {'plan':<28} {'on the counted hour':>20} {'on 30 other days':>17} {'vs tuned to hour':>17} {'worst tenth of days':>20} {'corridors better':>17}")
    for name in rows[0]:
        own = np.mean([r[name][0] for r in rows])
        oth = np.mean([r[name][1].mean() for r in rows])
        ref = np.mean([r[base][1].mean() for r in rows])
        p90 = np.mean([np.sort(r[name][1])[-3:].mean() for r in rows])
        better = sum(r[name][1].mean() < r[base][1].mean() - 1e-9 for r in rows)
        print(f"  {name:<28} {own:>20.1f} {oth:>17.1f} {oth / ref - 1:>+17.1%} {p90:>20.1f} {better:>13}/{n_cor}")


if __name__ == "__main__":
    main()
