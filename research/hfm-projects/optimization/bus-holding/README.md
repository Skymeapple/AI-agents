# Control Policy for Frequent Bus Services

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_bus_headways` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## The challenge

This work develops an online holding policy for the HSL trunk-line replay covering routes 500, 510, 560, 570, and 200. The policy uses available observations to decide holds at permitted timing points, minimizing passenger costs. The replay, action limits, and timing rules are fixed.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.7505
- **HFM effort:** 26 proposals, 26 scored, 10 accepted improvements, about $758 of compute
- **HFM's current best idea:** Replacing stride-indexed link-error draws with four deterministic empirical-quantile strata increases whole-practice mean normalized passenger score relative to p1574 v28.

## Existing theory and methods

Headway-based holding (Daganzo), two-way-looking control, predictive control with travel-time distributions.

## Public data

HSL (Helsinki) high-frequency vehicle positions and GTFS are open data.

## What limits progress

Rule-based holding without look-ahead.

## Route to a step change

Hold to equalise forecast headways over the next stops.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a replay on HSL vehicle positions; not yet built.

## How far this can go

Bounded by hindsight-optimal holding on realised travel times.

## Next steps

Build the replay for routes 500 and 510 from HSL data.
