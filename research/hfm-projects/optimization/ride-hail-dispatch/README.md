# Ride-Hail Dispatch Policy for Matching and Repositioning

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_ridehail_dispatch` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## The challenge

This work develops a dispatch policy for a New York City ride-hail replay. The policy determines how to match available drivers to ride requests and where to reposition idle drivers. Optimization is performed through a fixed objective, subject to constraints including rider patience, driver shifts, and travel rules. The approach selects matching, prediction, and repositioning methods while the replay provides future requests.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.9895
- **HFM effort:** 40 proposals, 36 scored, 9 accepted improvements, about $899 of compute
- **HFM's current best idea:** Ordering p1075's fixed selected repositioning proposals by their exact final source-repriced, destination-rank-specific assignment weights increases full P1-P6 mean_instance_score relative to its unchanged rank-zero endpoint ordering.

## Existing theory and methods

Batched bipartite matching, value-function dispatch, demand-aware repositioning.

## Public data

NYC TLC high-volume for-hire vehicle trip records (public).

## What limits progress

The score is at 0.989, close to its ceiling.

## Route to a step change

Test robustness to demand shocks before further tuning.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a replay simulator on NYC trip records; not yet built.

## How far this can go

About 1% remains under this metric.

## Next steps

Low priority.
