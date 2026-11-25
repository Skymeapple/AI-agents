# Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_bike_rebalancing` (paused on HFM)
- **Project status:** Tested on public data

## The challenge

This work develops a dispatching policy for a replay of a station-based nextbike system. When a crew completes a stop, the policy determines the van's next destination and the number of bicycles to load or unload, subject to fixed constraints on van capacity, crew shifts, return-to-depot requirements, and bicycle conservation. The policy observes station state and past events but has no access to future rider requests or the return destinations of bicycles still in use.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.8019
- **HFM effort:** 72 proposals, 66 scored, 13 accepted improvements, about $1,766 of compute
- **HFM's current best idea:** Subtracting matched homogeneous numerical error from p1191's chronological first-passage probabilities increases whole-practice mean_instance_score over its uncorrected chronological solver.

## Existing theory and methods

Inventory models per station (user dissatisfaction functions), vehicle routing with pickups and deliveries, fluid-model look-ahead.

## Public data

Helsinki city-bike origin-destination trips, HSL open data (`https://dev.hsl.fi/citybikes/`, CC-BY 4.0), June 2021: 1.2 million trips. Dock counts are published separately by HSL and were not used.

## What limits progress

Myopic stop-by-stop scoring.

## Route to a step change

Sampled-demand look-ahead feeding a routing solver, re-planned at each stop.

## Experiment and results

Code: [`bike_sim.py`](bike_sim.py) in this folder, built on the shared simulation core in [`../../_sim/core.py`](../../_sim/core.py). The full run was executed on free GitHub Actions compute; its log is in [`research/compute/results/bike-june-2021/`](../../../compute/results/bike-june-2021/run.log).

- **Setup:** Helsinki city-bike trips for June 2021 (HSL open data), 457
  stations, about 41,700 trips a day, 17 test weekdays. Every trip is
  replayed at its real time between its real stations. Four vans of 20
  bikes work 06:00–22:00. Cost is failed rentals plus failed returns.
  Dock counts, fleet size and the van fleet are assumptions, because they
  are not in the trip file; they make this a stressed system (22% of
  trips fail without rebalancing).
- **Result:**

  | Policy | Failures per day | Share of trips | Versus forecast policy | Days better |
  | --- | --- | --- | --- | --- |
  | No rebalancing | 9,272 | 22.3% | +77.1% | 0/17 |
  | Reactive (serve the emptiest or fullest station) | 6,258 | 15.0% | +19.5% | 1/17 |
  | Forecast (stock projected two hours ahead) | 5,235 | 12.6% | — | — |
  | Look-ahead, replaying last week's same weekday | 5,275 | 12.7% | +0.8% | 7/17 |
  | Look-ahead with the real future (not available) | 4,652 | 11.2% | −11.1% | 17/17 |

- **What this shows:**
  - **Forecasting demand is the step that pays:** 16% fewer failures
    than a reactive dispatcher, on 16 of 17 days.
  - **Look-ahead on a single past day does not help.** A two-day trial
    had suggested a 2% gain; over 17 days it is +0.8% and worse on 10
    days. Last week's trips are too poor a stand-in for today's.
  - **A perfect forecast would be worth a further 11%,** on every day.
    The limit is the demand forecast, not the dispatch rule.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **With these vans:** even a perfect forecast leaves 11% of trips failing, so van capacity and fleet size bound the result more than the policy does.
- **Forecast headroom:** 11% between the forecast policy and the perfect-forecast look-ahead. A look-ahead that samples several demand scenarios should capture part of it.
- **Not measured:** the true optimum for the realised demand, which needs a routing solver.

## Next steps

- Replace the single-scenario look-ahead with several sampled days and average the outcome.
- Add weather and weekday effects to the demand rates.
- Obtain real dock counts and bike counts (HSL publishes station capacities) to remove the main assumption.
- Compute a hindsight bound with a pickup-and-delivery routing solver.
