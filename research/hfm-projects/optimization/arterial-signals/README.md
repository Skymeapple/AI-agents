# Toronto Arterial Signal Timing Coordination

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_arterial_signals_auto` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## Readiness

- **State:** Data verified; pipeline or simulator still to build.
- **Data access:** City of Toronto open data, traffic volumes at intersections: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** A corridor delay model calibrated to the counts. Low priority: HFM's score is already 0.989.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

A signal-timing engineer selects a coordinated plan for six or seven consecutive signalized intersections on a Toronto arterial, based on one counted hour of data.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.9892
- **HFM effort:** 50 proposals, 49 scored, 12 accepted improvements, about $1,162 of compute
- **HFM's current best idea:** 保持p1769全部晚期交换和局部偏移候选集合不变，仅用双向绿中心行程误差选择局部偏移正负方向的查询顺序，能提高相同50次查询上限下的mean_instance_score。

## Existing theory and methods

Webster and HCM delay, MAXBAND and MULTIBAND progression, max-pressure control, cell-transmission models.

## Public data

City of Toronto Open Data: turning-movement counts and signal locations.

## What limits progress

The score is at 0.989 on one counted hour, so the metric is nearly exhausted.

## Route to a step change

Optimise for a distribution of demand around the counted hour.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a traffic model calibrated to Toronto counts; not yet built.

## How far this can go

Under this metric, about 1% remains.

## Next steps

Evaluate the incumbent plan under plus or minus 20% demand.
