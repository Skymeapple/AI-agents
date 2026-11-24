# Wireless 400-Cell Synthetic MILP Case Solver

- **Field:** optimization
- **HFM track:** `discovery_live_wireless_milp_400_live_v8` (active on HFM)
- **Project status:** Blocked

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

No experiment has been run for this project.

- **Why:** HFM has not published the instance format or the reward, and no proposals have run.

## How far this can go

Not estimable.

## Next steps

Request the instance schema through HFM's Research Exchange.
