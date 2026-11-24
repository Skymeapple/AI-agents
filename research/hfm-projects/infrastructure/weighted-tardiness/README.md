# Single-Machine Weighted Tardiness Scheduling with Release Times

- **Field:** infrastructure
- **HFM track:** `discovery_live_live_debug_tardiness` (active on HFM)
- **Project status:** Tested on public data

## The challenge

A single machine processes a batch of jobs. Each job j becomes available at its release time r[j], requires continuous processing time p[j], has a due date d[j], and incurs a penalty w[j] per unit time of tardiness. The machine processes one job at a time and must wait idle if the next job has not yet arrived.

- **Metric:** `normalized_tardiness_gain` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 1.211
- **HFM effort:** 45 proposals, 43 scored, 0 accepted improvements, about $101 of compute
- **HFM's current best idea:** Raising p441's internal CPU guard from 0.95 to 0.97 of the supplied budget will lower final weighted tardiness while still returning legal schedules before the one-second hard limit.

## Existing theory and methods

ATC dispatching, iterated local search, dynasearch, exact branch-and-bound and time-indexed MILP for small instances.

## Public data

No external data needed. Instances are generated with the standard OR-Library-style scheme.

## What limits progress

Search quality within a one-second budget.

## Route to a step change

Iterated local search or dynasearch seeded by ATC, with fast move evaluation.

## Experiment and results

Code: [`tardiness.py`](tardiness.py) in this folder. Run instructions are in its header.

- **Setup:** 60 generated instances (n = 40 and 100, three tardiness and
  due-date-range settings, 10 seeds each), with the same one-second
  budget as HFM.
- **Result:** iterated local search seeded by ATC reduced total weighted
  tardiness by **12.5% on average versus ATC with k = 2**, the dispatch
  rule HFM's track starts from. It reduced it by **6.8% versus ATC with
  k tuned per instance**, and it never did worse on any instance.
- **What this does and does not show:** it confirms the direction
  recommended in Section 9. It does **not** beat HFM's incumbent (gain
  1.211), which already includes local search and is scored on private
  instances with an undisclosed normalisation.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Pure-Python local search gains 6.8% over tuned ATC here. A compiled implementation with O(1) move evaluation would search roughly 100 times more moves; published results put good heuristics within about 1% of optimal on instances of this size.

## Next steps

Compute exact optima for n = 40 with a time-indexed MILP to measure the remaining gap. Port the search to compiled code.
