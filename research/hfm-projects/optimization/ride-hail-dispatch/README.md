# Ride-Hail Dispatch Policy for Matching and Repositioning

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_ridehail_dispatch` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** NYC TLC for-hire trip records (473 MB per month): Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

NYC Taxi and Limousine Commission high-volume for-hire trip records (public), `fhvhv_tripdata_2024-01.parquet`, 473 MB.

## What limits progress

Idle drivers are in the wrong places relative to coming demand.

## Route to a step change

Reposition idle drivers towards forecast demand; then improve the forecast.

## Experiment and results

Code: [`ridehail_sim.py`](ridehail_sim.py) in this folder. Run instructions are in its header.

- **Setup:** New York City's public for-hire trip records for January 2024 (19.7 million trips). A 4% sample of real requests is replayed at real request times, about 25,000 a day, on 14 test days. Driving times between zones are the median real trip times. A rider is lost if not matched within 3 minutes or if no driver can arrive within 8 minutes. The fleet is sized so the simple policy serves about 85% on a day outside the test set.
- **Result:**

  | Policy | Requests served | Lost requests versus nearest driver | Pickup wait | Empty driving per driver per day | Days better |
  | --- | --- | --- | --- | --- | --- |
  | Nearest driver | 83.2% | — | 4.7 min | 0 | — |
  | Batch assignment (match everyone together every 30 s) | 83.3% | −0.6% | 4.5 min | 0 | 11/14 |
  | + repositioning idle drivers towards forecast demand | **92.1%** | **−52.7%** | 4.1 min | 17 min | 14/14 |
  | + repositioning with the real coming demand (not available) | 94.7% | −68.6% | 3.7 min | 27 min | 14/14 |

- **What this shows:**
  - **Repositioning on a simple demand forecast halves lost requests,** on every test day, for 17 minutes of empty driving per driver.
  - **Better matching alone does almost nothing** (−0.6%). The drivers are in the wrong place, and no assignment rule can fix that.
  - **The forecast used is crude** (the same hour on earlier days of the same type) and already captures three quarters of what a perfect forecast would.
  - HFM's score for this track is 0.989, so its agents have very likely found the same thing. This confirms the mechanism on public data and does not go beyond HFM.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Reached here:** 92% served against 83%, with a perfect-forecast ceiling of 95% for this repositioning rule.
- **Remaining 5%:** requests that no idle driver can reach in time at this fleet size.
- **Under HFM's own score** the track is at 0.989, so little headroom is left there.

## Next steps

- Add a cost for empty driving to the score and find the best trade-off.
- Test under demand shocks (events, weather), where a same-hour forecast fails.
- Compare with the open dispatch environments found in the literature pass (OpenHail, RideGym).
