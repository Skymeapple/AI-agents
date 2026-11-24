# Online Single-Track Railway Dispatcher for Finnish Main-Line Replay

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_rail_dispatch` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## The challenge

Develop a dispatcher for a 16-section single-track line between Lielahti and Pohjois-Louko, with 15 passing loops. The policy controls section access to manage meets, overtakes, and waiting while minimizing fixed disruption cost.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.3984
- **HFM effort:** 93 proposals, 89 scored, 12 accepted improvements, about $6,175 of compute
- **HFM's current best idea:** Breaking a closed three-train veto cycle among ready waiting requests by minimum opposing-price regret increases mean_instance_score relative to p2376's independent pairwise vetoes.

## Existing theory and methods

Meet-pass scheduling as a job-shop problem; alternative-graph models; rolling-horizon MILP or constraint programming; greedy priority rules as baselines.

## Public data

Digitraffic (Fintraffic) open API: timetables and actual train running for the Tampere-Pori line that contains Lielahti to Pohjois-Louko.

## What limits progress

Greedy pairwise rules. The score of 0.40 suggests a large gap.

## Route to a step change

Rolling-horizon exact optimisation of meet and pass orders over a 10 to 20 minute look-ahead.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a replay simulator built on Finnish open rail data (Digitraffic); not yet built.

## How far this can go

A perfect-information MILP over each disruption gives the bound; it is computable once a simulator exists.

## Next steps

Build the 16-section simulator from Digitraffic data and compute the perfect-information bound.
