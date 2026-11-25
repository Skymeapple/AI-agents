# Toronto Arterial Signal Timing Coordination

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_arterial_signals_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU (model study on generated corridors).
- **Data access:** City of Toronto open data, traffic volumes at intersections: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** Toronto's real counts in place of generated volumes.
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

HFM's score is nearly exhausted on the counted hour (0.989). What is untested is how the plan behaves on other days.

## Route to a step change

Tune splits and offsets for a distribution of demand around the counted hour.

## Experiment and results

Code: [`arterial_sim.py`](arterial_sim.py) in this folder.

- **Setup:** a model study, not a replay. Eight generated corridors of 6 or 7 signals on a two-way arterial with a common 90-second cycle, with volumes and spacings in typical urban ranges. They are **not** Toronto's counts. Traffic is simulated second by second as queues and platoons. Each plan is built from one "counted hour" and then scored on 30 other days whose volumes differ by a day-wide factor (standard deviation 12%) plus per-approach noise. Cost is delay in vehicle-hours.
- **Result (mean over the eight corridors):**

  | Plan | Delay on the counted hour | Delay on 30 other days | Versus tuned to the hour | Worst tenth of days | Corridors better |
  | --- | --- | --- | --- | --- | --- |
  | Webster splits, zero offsets | 64.9 | 76.1 | +32.4% | 164.9 | 1/8 |
  | Webster splits + progression offsets | 48.9 | 61.0 | +6.2% | 145.5 | 3/8 |
  | Splits and offsets tuned to the counted hour | **39.9** | 57.5 | — | 145.4 | — |
  | Splits and offsets tuned to a range of days | 50.3 | **47.4** | **−17.5%** | **110.8** | 6/8 |

- **What this shows:**
  - **A plan tuned to one hour is tuned to that hour.** It is the best plan on the counted hour (39.9) and 44% worse on other days (57.5).
  - **Tuning to a range of plausible days is 17.5% better on other days** and 24% better on the worst tenth, on 6 of 8 corridors. It gives up delay on the counted hour to get it.
  - **HFM's score of 0.989 measures the counted hour.** If that is all its evaluation tests, the remaining headroom there is small and this result is invisible to it. The route in the overview (optimise for a distribution of demand) is supported in this model.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Under HFM's score:** about 1% left on the counted hour.
- **On other days:** a 17.5% reduction in delay in this model, from robustness alone.
- **Caveat:** the size depends on how much day-to-day demand varies, which was assumed (12%). Toronto's repeated counts would give the real figure.

## Next steps

- Replace the generated volumes with Toronto's public turning-movement counts, using repeat counts at the same intersection to measure real day-to-day variation.
- Check the result in an established simulator (the literature pass found SUMO models built from Toronto counts).
- Add left-turn phases and pedestrian minimums.
