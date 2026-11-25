# Single-Machine Weighted Tardiness Scheduling with Release Times

- **Field:** infrastructure
- **HFM track:** `discovery_live_live_debug_tardiness` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Generated instances; no external data.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

### Follow-up: where does the search plateau?

Code: [`tardiness_fast.py`](tardiness_fast.py). The same search compiled
with numba, run at three budgets, with an independent random multi-start
descent as a cross-check. Six seeds per setting.

| Jobs | Reduction vs tuned ATC at 30 / 300 / 3,000 rounds | Multi-start finds better |
| --- | --- | --- |
| 40 | 7.1% / 8.7% / 8.7% (average of three settings) | 0 of 18 instances |
| 100 | 7.1% / 10.7% / 11.5% (average of three settings) | 1 of 18 instances |

- **40-job instances are very likely solved.** Ten times more search
  changes nothing, and an unrelated method lands on the same value in all
  18 instances.
- **100-job instances are close to flat.** Ten times more search adds
  under one percentage point.
- Across both sizes the compiled search is 10.1% below tuned ATC, against
  6.8% for the pure-Python version at one second. The remaining gain at
  this size comes from implementation speed, not from a new idea.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Pure-Python local search gains 6.8% over tuned ATC here. A compiled implementation with O(1) move evaluation would search roughly 100 times more moves; published results put good heuristics within about 1% of optimal on instances of this size.

## Next steps

Confirm the 40-job optima with an exact method. For 100 or more jobs, test dynasearch or a memetic search against the compiled search at equal time.
