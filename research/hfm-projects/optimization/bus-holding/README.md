# Control Policy for Frequent Bus Services

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_bus_headways` (paused on HFM)
- **Project status:** Tested on public data

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

HSL public timetable feed (GTFS, `https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip`, CC-BY 4.0): real stop sequences, running times, headways and timing points for routes 500, 510, 560, 570 and 200. HSL's recorded vehicle positions (HFP) are public but large and were not used.

## What limits progress

Rule-based holding without look-ahead.

## Route to a step change

The overview proposed holding to equalise forecast headways. The experiment below does not support it under these assumptions; see the results.

## Experiment and results

Code: [`bus_sim.py`](bus_sim.py) in this folder, built on the shared simulation core in [`../../_sim/core.py`](../../_sim/core.py). Route data: [`hsl_trunk_routes.json`](hsl_trunk_routes.json).

- **Setup:** the five routes HFM names (500, 510, 560, 570, 200). Stop
  sequences, scheduled running times, morning headways (8 to 10 minutes)
  and timing points are real, from HSL's public timetable feed.
  Passenger demand and running-time variability are **assumed**, because
  HSL does not publish them openly per stop and bus. So this is a
  structurally faithful stand-in and not a replay. 30 instances per
  setting (5 routes, 6 random days). Cost is passenger-minutes of waiting
  at stops plus delay to passengers on a held bus.
- **Result (change in cost against no holding):**

  | Policy | Assumed variability | Doubled variability |
  | --- | --- | --- |
  | Hold to schedule | +0.1% | +0.8% |
  | Forward-headway rule | +4.4% | +9.3% |
  | Two-way headway rule | +10.2% | +11.7% |
  | Look-ahead on the two-way rule | +2.6% | +2.2% |
  | Look-ahead on no holding | −0.2% (better on 16/30) | −0.3% (better on 21/30) |
  | Look-ahead with real future running times (not available) | −0.2% (24/30) | −0.9% (29/30) |

- **What this shows:**
  - **Classic headway rules make things worse here,** on all 30
    instances. At 8 to 10 minute headways with only two to four timing
    points, the delay to passengers already on the bus outweighs the
    waiting saved at stops.
  - **Even with perfect knowledge of future running times the gain is
    under 1%.** Under this cost and these assumptions there is very
    little for a holding policy to win.
  - This contradicts the route proposed in the overview (headway-based
    predictive holding). The proposal assumed bunching is costly on
    these lines; in this model it is not.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Under the assumptions here the ceiling is about 1% below no holding, reached only with perfect foresight. That conclusion depends on the assumed demand and variability. Holding is known to pay on lines with headways of about 5 minutes or less, heavy boarding and many control points, where bunching feeds on itself; these trunk routes at morning headways are outside that regime in this model.

## Next steps

- Replace the assumed variability and demand with HSL's recorded vehicle positions (HFP data) and stop-level boardings, which is what HFM replays. That is the only way to check whether the real lines bunch more than this model does.
- Test the same policies at 4 to 5 minute headways to find where holding starts to pay.
- Test weighting waiting time at twice in-vehicle time, a common convention that favours holding.
