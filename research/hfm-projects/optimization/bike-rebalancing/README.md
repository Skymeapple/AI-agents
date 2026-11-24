# Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_bike_rebalancing` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

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

Public station-based bike-share trip data (for example Citi Bike, Capital Bikeshare, Helsinki City Bikes). nextbike publishes live station feeds.

## What limits progress

Myopic stop-by-stop scoring.

## Route to a step change

Sampled-demand look-ahead feeding a routing solver, re-planned at each stop.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a replay simulator on public bike-share trips; not yet built.

## How far this can go

Hindsight-optimal routing on the realised demand gives the bound.

## Next steps

Build the replay and compute the hindsight bound.
