# Rolling Unit-Commitment and Dispatch Controller under Real OPSD Load and Wind Sequences

- **Field:** optimization
- **HFM track:** `discovery_live_rsi_grid_dispatch_mpc_seedevo` (active on HFM)
- **Project status:** Tested on public data

## The challenge

This work implements a day-by-day rolling decision for unit commitment, dispatch, and storage under real OPSD load and wind sequences with day-ahead forecasts. A verifier runs a real-time balancing LP using actual values of the day, accumulating real costs over one week. The unit and storage models and real-time balancing rules are frozen; the evolving component is the Controller in methods/main. The problem formulation, constraints, and delivery format follow the instruction.md in the harbor task.

- **Metric:** `reward` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.5338
- **HFM effort:** 6 proposals, 6 scored, 0 accepted improvements, about $100 of compute
- **HFM's current best idea:** Keeping cheap baseload (nuclear/lignite, var 10-30 EUR/MWh, min_up 12-72 h) committed but letting their dispatch fall to p_min in high-renewable net-load valleys while the shortfall of thermal spinning reserve is met by credited storage discharge plus a free curtailment slack - instead of forcing the full thermal reserve M_t in every hour as v5 does - reduces curtailment (289 GWh in the clean run) and summer high-renewable over-commitment (the worst practice week 2016-08-01 scores only 0.13) WITHOUT the restart/start-up-ramp penalty that killed level-adaptive reserve (p274 log: scaling margin down at valleys scored 0.10-0.27), because baseload is never shut off, so peak-hour reliability (shed_mwh = 0) is preserved and mean weekly score rises.

## Existing theory and methods

Deterministic unit commitment with reserve rules, two-stage stochastic and robust unit commitment, scenario reduction, rolling-horizon MPC.

## Public data

Open Power System Data time series (2020-10-06 release): German load, day-ahead load forecast and wind.

## What limits progress

Fixed reserve rules ignore the shape of wind forecast errors.

## Route to a step change

Commit against wind scenarios that always include the extreme low-wind cases.

## Experiment and results

Code: [`unit_commitment.py`](unit_commitment.py) in this folder. Run instructions are in its header.

- **Setup:** German hourly load, the official day-ahead load forecast and
  actual wind from Open Power System Data. A German-scale invented fleet
  (17 slow thermal units, a fast peaker, storage) is committed each
  midnight on forecasts, then dispatched against what really happened.
  The reserve rule is tuned on 28 days of 2018. Results are for 42 days
  in six weeks of 2019. Regret is cost above perfect foresight
  (1,016 million EUR).
- **Result:**

  | Controller | Regret, million EUR | Versus tuned rule | Weeks cheaper |
  | --- | --- | --- | --- |
  | Reserve rule, no wind reserve | 370.2 | +644% | 1/6 |
  | Reserve rule, tuned on 2018 | 49.7 | — | — |
  | Stochastic, 10 scenarios **spread** from worst to best past error | 38.2 | **−23%** | 5/6 |
  | Stochastic, 20 spread scenarios | 37.6 | **−24%** | 5/6 |
  | Stochastic, 10 **random** scenarios, four draws | 23.0 / 38.6 / 265.0 / 339.9 | −54% / −22% / +433% / +583% | — |
  | Stochastic, 20 random scenarios | 190.6 | +283% | 3/6 |

- **What this shows:**
  - **Stochastic commitment beats a tuned reserve rule by about a
    quarter of the regret,** but only when the scenarios always include
    the extreme low-wind cases.
  - **With randomly drawn scenarios the method is dangerous.** Two of
    four draws were four to six times worse than the simple rule,
    because a draw that misses the bad-wind tail leaves the system short
    and it pays the load-shedding price.
  - A single run of the random version would have reported either a 54%
    gain or a 583% loss. This is the same lesson as Section 14.2: one
    scored run is not evidence.
  - The fleet and the wind forecast are invented, so the size of the
    gain will not carry over to HFM's task. The ordering of the methods
    is the finding.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Perfect foresight costs 1,016 million EUR on the test days. The tuned rule is 4.9% above that and spread-scenario commitment 3.7% above. The last 3.7% is the price of not knowing the wind; only a better forecast removes it.

## Next steps

Add a storage end-of-day value, minimum up and down times, and a real wind forecast. Test scenario reduction that keeps tail scenarios.
