# Online Single-Track Railway Dispatcher for Finnish Main-Line Replay

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_rail_dispatch` (paused on HFM)
- **Project status:** Tested on public data

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

Fintraffic Digitraffic open API (`https://rata.digitraffic.fi`, CC-BY 4.0): planned and actual times of every train, one file per day.

## What limits progress

Greedy pairwise rules cannot see that holding one train briefly lets an opposing train through and saves more later.

## Route to a step change

Rolling-horizon exact optimisation of meet and pass orders over a 10 to 20 minute look-ahead.

## Experiment and results

Code: [`rail_sim.py`](rail_sim.py) and [`fetch_days.py`](fetch_days.py) in this folder, built on the shared simulation core in [`../../_sim/core.py`](../../_sim/core.py). A 60-day run is queued on free GitHub Actions compute; its log will appear in [`research/compute/results/rail-60-days/`](../../../compute/results/).

- **Setup:** Fintraffic's open train data for the Lielahti–Pohjois-Louko
  line: 17 stations and 16 single-track sections, as in HFM's task. 19
  weekdays from January 2024, about 51 trains a day. Trains enter at
  their real entry times and run sections in planned times. A section
  holds one train. Cost is weighted exit delay in minutes (passenger
  trains weigh 3). "Disrupted" days add extra entry delays to 30% of
  trains. Passing loops have unlimited capacity, which is a
  simplification.
- **Result (change in weighted delay against the priority rule):**

  | Policy | Real days | Days better | Disrupted days | Days better |
  | --- | --- | --- | --- | --- |
  | First come, first served | +2.0% | 8/19 | +3.2% | 2/19 |
  | Priority (passenger first, then most delayed) | — | — | — | — |
  | Look-ahead (rollout on the priority rule) | **−11.0%** | 19/19 | **−7.0%** | 16/19 |
  | Hindsight bound (whole-day optimisation, real entry times known) | −34.5% | | −29.2% | |

- **What this shows:**
  - **Look-ahead beats the greedy rules on every real day.** It tests
    each option, including holding a train back for an opposing one, by
    simulating the rest of the day. This supports the route in the
    overview.
  - **A large gap remains.** The bound is 29–35% below the priority
    rule, against 7–11% captured. Part of that gap is information no
    online dispatcher has (late trains that have not yet appeared), and
    part is that the bound is a relaxation stopped after 12 seconds per
    day, so the true optimum lies above it.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Captured so far:** 7–11% of weighted delay by one-step look-ahead.
- **Upper limit on what is left:** the hindsight bound is 29–35% below the priority rule. The real ceiling for an online policy is lower than that, because the bound knows every future entry delay and is itself a relaxation.
- **How to pin it down:** solve the whole-day problem to optimality (longer solver time on the free compute), and separately measure a look-ahead that is told the real entry times. The difference between those two splits the gap into "better search" and "better forecasts of late trains".

## Next steps

- Run the 60-day job to completion and tighten the bound with longer solver time.
- Replace one-step rollout with rolling-horizon exact optimisation over the next 60–90 minutes.
- Add loop capacity and block signalling so that deadlocks and following moves are modelled.
- Use the delay causes in the open data to forecast entry delays of trains not yet on the line.
