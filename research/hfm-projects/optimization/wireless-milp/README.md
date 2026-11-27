# Wireless 400-Cell Synthetic MILP Case Solver

- **Field:** optimization
- **HFM track:** `discovery_live_wireless_milp_400_live_v8` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in on a self-built instance family has run. HFM's own instance format and reward are still unpublished.
- **Data access:** None published by HFM.
- **Compute:** CPU.
- **Still missing:** The instance format and reward, which HFM has not published.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

A self-contained Python program processes one wireless 400-cell synthetic MILP case from stdin and writes one answer JSON object to stdout. NumPy 2.3.2 and SciPy 1.16.1, including HiGHS, are installed; no network is available, and any installed solver or algorithm may be used.

- **Metric:** `wireless_composite_reward` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** not published
- **HFM effort:** 0 proposals, 0 scored, 0 accepted improvements, about $0 of compute

## Existing theory and methods

Lagrangian decomposition, large-neighbourhood search around a MILP solver, warm starts.

## Public data

None published. Comparable public problems are cell-association and channel-assignment benchmarks.

## What limits progress

Unknown until instances are public.

## Route to a step change

Decompose by cell cluster, warm-start HiGHS from a greedy solution, large-neighbourhood search.

## Experiment and results

HFM has not published the instance format or reward, so this builds a problem of the kind its description and the literature point to and compares the proposed methods on equal time, with the same tools as HFM's sandbox (SciPy's HiGHS, NumPy). Script: [`sleep_milp.py`](sleep_milp.py); logs in `research/compute/results/wireless-milp-floor/` (round 2) and [`round1_run.log`](round1_run.log). It is not comparable to HFM's score.

- **Instance family (seeded):** 400 cells on a 20 × 20 km square (about 20% macro cells at 400 W, small cells at 60 W) and 6,000 users clustered around 25 hotspots. Binary decisions: which cells are on, and which cell serves each user (31,000–35,000 user–cell pairs, about 38,000 variables and constraints). Each cell's load (demand over distance-dependent rate) must fit its capacity; a dropped user costs 30 W. Minimise power plus drop penalty.
- **Methods, same wall-clock budget:** greedy (switch on cells by users covered per watt); HiGHS on the whole MILP; large-neighbourhood search from the greedy solution (free 25 neighbouring cells and their users, re-solve with HiGHS for up to 20 s, keep improvements); a hybrid (HiGHS for 60% of the budget, then fast neighbourhoods of 12 cells with 4 s sub-solves around its incumbent).
- **Score:** gap to the LP-relaxation bound (lower is better), four instances.

| Method | 60 s | 300 s |
|---|---|---|
| greedy | 34–39% | 34–39% |
| greedy start + LNS | 21–27% | 6.4–11.4% |
| HiGHS alone | 2.5–4.6%, and 27% on one instance | 2.0–4.6% (mean 2.93%) |
| HiGHS 180 s, then LNS 120 s | — | **1.4–2.3% (mean 1.85%)** |

- **The route usually proposed for this track (warm start from greedy, then large-neighbourhood search) loses badly to plain HiGHS.** The greedy start is 34–39% above the bound, and 20-second sub-solves allow only about 20 neighbourhoods in five minutes.
- **HiGHS stalls.** Its incumbent at 180 seconds equals its incumbent at 300 seconds on all four instances; the last 40% of the budget is wasted. At 60 seconds it can also get stuck (27% on one instance).
- **Spending the stalled time on small neighbourhoods around HiGHS's own incumbent helps on every instance.** The hybrid lowers the gap from 2.93% to 1.85% on average, removing 37% of what remained, with 41–66 neighbourhoods and 4–15 improvements per instance.
- **For HFM's track:** once the format is published, the first things to try are a full HiGHS solve with its stall detected, followed by local search around its solution, not a greedy-seeded search.

## How far this can go

Not estimable for HFM's unpublished instances. On this family, HiGHS plus a short neighbourhood search ends within 1.4–2.3% of the LP bound in five minutes; the true optimum lies somewhere in that gap.

## Next steps

Request the instance schema through HFM's Research Exchange. On this family: detect HiGHS's stall adaptively instead of a fixed 60% split, and choose neighbourhoods by LP-solution fractionality.
