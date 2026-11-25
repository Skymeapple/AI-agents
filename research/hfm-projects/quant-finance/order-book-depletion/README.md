# Nasdaq Ten-Level Order Book Depth Exhaustion Ranking

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_nasdaq_depth_depletion_auto` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Blocked on licensed data.
- **Data access:** Only LOBSTER's free sample day: Verified reachable from the free compute on 2026-11-25. One day is enough to build and test code, not to draw conclusions.
- **Compute:** CPU.
- **Still missing:** Licensed order-book history.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

For 79 S&P 500 stocks, Databento's mbp-10 feed records the ten best price levels from Nasdaq TotalView-ITCH, with price, size, and order count per book event. One seeded 20-minute window per sampled weekday is reduced to ten-second buckets, capturing the book at bucket end.

- **Metric:** `instance_score_mean` (maximize)
- **HFM baseline:** 0.1997
- **HFM best so far:** 0.3766
- **HFM effort:** 58 proposals, 57 scored, 12 accepted improvements, about $1,498 of compute
- **HFM's current best idea:** 在冻结的p1690状态条件方差路径中，使用固定自由度Student-t尺度似然拟合状态系数而非高斯工作似然，能提高冻结评测的instance_score_mean。

## Existing theory and methods

Order-flow imbalance, queue-depletion hazard models, Hawkes processes.

## Public data

LOBSTER publishes a few free sample days only.

## What limits progress

Snapshot features; last gains about one part in 100,000.

## Route to a step change

Event-time models of order arrivals across levels.

## Experiment and results

No experiment has been run for this project.

- **Why:** Nasdaq TotalView order-book data (Databento) is licensed.

## How far this can go

Unknown.

## Next steps

Needs a data licence.
