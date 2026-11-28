"""Layer-4 test: choose the forecast policy's settings by forecast accuracy or by the decision outcome.

The framework's fourth layer says prediction and decision should be optimised together,
with the final utility as the objective. The bike replay of bike_sim.py has a forecast
policy with three settings: a scale alpha on the learned hourly net-outflow rates, the
look-ahead window over which stock is projected (LOOK, minutes) and the safety margin of
bikes and docks kept free (MARGIN). Two ways to set them, on the same tuning days:

  predictive   alpha minimises the squared error of the projected net outflow over the
               default window (the usual "fit the forecast, then optimise" route); LOOK
               and MARGIN stay at their defaults (120 minutes, 3)
  decision     alpha, LOOK and MARGIN chosen together by the simulated number of failed
               rentals and returns on the tuning days (the policy's own utility)
  decision, alpha only   alpha chosen by failures, LOOK and MARGIN at their defaults

The first 6 test weekdays of bike_sim.py are the tuning days; the remaining weekdays are
the evaluation. Same simulator, data and costs as bike_sim.py.

Usage: python decision_tuning.py <trips-month.csv>
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bike_sim as bs
from core import run

ALPHAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LOOKS = (30.0, 60.0, 120.0, 180.0, 240.0)
MARGINS = (1, 3, 5, 8)
G = {}


def setup(path):
    d, n, drive, cap = bs.load(path)
    days = sorted(d.day.unique())
    first = d.dep.iloc[0]
    weekday = {day: pd.Timestamp(first.year, first.month, day).weekday() for day in days}
    test = [day for day in days if day > 7 and weekday[day] < 5 and (day - 7) in days]
    by_day = {day: g.sort_values("t") for day, g in d.groupby("day")}
    nets = {day: bs.rates(d, n, [x for x in days if x < day and weekday[x] < 5]) for day in test}
    G.update(d=d, n=n, drive=drive, cap=cap, test=test, by_day=by_day, nets=nets)


def cost(args):
    day, alpha, look, margin = args
    bs.LOOK, bs.MARGIN = look, margin
    sim = bs.BikeSim(G["by_day"][day], G["n"], G["drive"], G["cap"], alpha * G["nets"][day])
    return run(sim, bs.best_first)


def forecast_error(day, alpha, look=120.0):
    """Squared error of alpha * rate * window against the real net outflow over the window, hourly 06-22."""
    x, n, net = G["by_day"][day], G["n"], G["nets"][day]
    err = 0.0
    for h in range(6, 22):
        t0, t1 = h * 60, h * 60 + look
        real = np.zeros(n)
        r = x[(x.t >= t0) & (x.t < t1)]
        np.add.at(real, r.o.values, 1.0)
        back = x[(x.t + x.dur >= t0) & (x.t + x.dur < t1)]
        np.subtract.at(real, back.d.values, 1.0)
        err += np.sum((alpha * net[:, h] * look / 60 - real) ** 2)
    return err


def main():
    setup(sys.argv[1])
    test = G["test"]
    tune, ev = test[:6], test[6:]
    print(f"{G['n']} stations; tuning days {tune}; evaluation days {ev}", flush=True)
    pred_alpha = min(ALPHAS, key=lambda a: sum(forecast_error(day, a) for day in tune))
    print(f"predictive choice: alpha {pred_alpha} (forecast squared error by alpha: "
          + ", ".join(f"{a}: {sum(forecast_error(day, a) for day in tune):.0f}" for a in ALPHAS) + ")", flush=True)
    grid = list(itertools.product(ALPHAS, LOOKS, MARGINS))
    with ProcessPoolExecutor(4, initializer=setup, initargs=(sys.argv[1],)) as ex:
        jobs = [(day, a, l, m) for (a, l, m) in grid for day in tune]
        res = dict(zip(jobs, ex.map(cost, jobs, chunksize=4)))
        tcost = {g: np.mean([res[(day,) + g] for day in tune]) for g in grid}
        dec = min(grid, key=tcost.get)
        dec_a = min((g for g in grid if g[1] == 120.0 and g[2] == 3), key=tcost.get)
        print(f"decision choice: alpha {dec[0]}, LOOK {dec[1]:.0f}, MARGIN {dec[2]} (tuning cost {tcost[dec]:.0f}); "
              f"alpha only: {dec_a[0]} (tuning cost {tcost[dec_a]:.0f}); default (1, 120, 3) tuning cost {tcost[(1.0, 120.0, 3)]:.0f}", flush=True)
        arms = {"default (alpha 1, LOOK 120, MARGIN 3)": (1.0, 120.0, 3), "predictive": (pred_alpha, 120.0, 3),
                "decision, alpha only": dec_a, "decision, all three": dec, "no forecast (alpha 0)": (0.0, 120.0, 3)}
        ev_jobs = [(day,) + g for g in set(arms.values()) for day in ev]
        ev_res = dict(zip(ev_jobs, ex.map(cost, ev_jobs)))
    ref = [ev_res[(day,) + arms["default (alpha 1, LOOK 120, MARGIN 3)"]] for day in ev]
    print(f"\nevaluation: failed rentals + returns per day over {len(ev)} weekdays")
    for name, g in arms.items():
        c = [ev_res[(day,) + g] for day in ev]
        print(f"RESULT {name:38s} settings {g}: mean {np.mean(c):.0f}, vs default {np.mean(c) / np.mean(ref) - 1:+.1%}, "
              f"better on {sum(a < b for a, b in zip(c, ref))}/{len(ev)} days", flush=True)


if __name__ == "__main__":
    main()
