# Nasdaq Ten-Level Order Book Depth Exhaustion Ranking

- **Field:** quant-finance
- **HFM track:** `discovery_live_live_quant_nasdaq_depth_depletion_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A stand-in on public crypto order books has run. The Nasdaq task itself needs licensed data.
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

Data: Bybit's public archives for four USDT perpetuals (BTC, ETH, SOL, XRP), 3–5 June 2024: full 500-level order-book snapshots and deltas (quote-saver.bycsi.com) and every public trade (public.bybit.com). The book is rebuilt and sampled once a second at ten levels. Fitting uses 3–4 June; the test is 5 June. Script: [`bybit_book_floor.py`](../order-book-depletion/bybit_book_floor.py); logs in `research/compute/results/bybit-book-*`. Licensed Nasdaq data (Databento) is not used, and LOBSTER's free equity samples now require an application. Crypto perpetuals trade around the clock with no auction and, for BTC and ETH, a tick far below one basis point, so this is an analogue, not a replica, and is not comparable to HFM's score.

Task: at the end of every ten-second bucket, for each side, rank by the log change of the ten-level depth over the next bucket (negative = depletion). Score: Spearman rank correlation on the test day (17,274 bucket-sides per instrument) and the share of the 10% most-depleting buckets among each method's 10% picks.

| Instrument | Median spread | Reversion (minus last change) | Ridge | Boosted trees | Top-decile hit, boosted |
|---|---|---|---|---|---|
| BTCUSDT | 0.01 bp | +0.406 | +0.576 | **+0.585** | 0.226 |
| ETHUSDT | 0.03 bp | +0.419 | +0.568 | **+0.577** | 0.251 |
| SOLUSDT | 0.6 bp | +0.386 | +0.485 | **+0.486** | 0.264 |
| XRPUSDT | 1.9 bp | +0.231 | +0.313 | **+0.320** | 0.244 |

- **Most of the rankable signal is mean reversion of depth.** Minus the last bucket's change alone gives 0.23–0.42. A learned model on six more features (traded volume against the side, imbalance, near-level share, spread, hour) adds 0.08–0.17.
- **Trees add almost nothing over a linear model** (+0.001 to +0.010). HFM's best idea at this point is a change of likelihood inside a frozen variance model; on this evidence, feature choice matters more than the fitting machinery.
- **Deep, fast books are the most predictable.** BTC and ETH reach 0.58; XRP, with a wide spread and fewer trades, 0.32. Ranking the deepest depletions is harder: the top-decile hit is 2.3–2.6 times random.

**Path memory: the steel-hysteresis analogy, tested.** [`memory_floor.py`](memory_floor.py) adds features that summarise the book's recent path to the eight state features, on the same data and split (logs: `book-memory-*` in [`../../../compute/results/`](../../../compute/results/)). Spearman on the test day, gradient-boosted trees:

| Features | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| state (eight features above) | 0.585 | 0.578 | 0.485 | 0.319 |
| + lags (depth changes over the previous minute) | 0.589 | 0.585 | 0.515 | 0.349 |
| + drawdown from the 5- and 30-minute peak, distance from the 30-minute minimum | 0.597 | 0.598 | 0.518 | 0.358 |
| + flow and volatility over 1 and 5 minutes | 0.595 | 0.590 | 0.489 | 0.323 |
| + lags + flow and volatility | 0.596 | 0.593 | 0.514 | 0.351 |
| + all path features | **0.601** | **0.602** | **0.528** | **0.361** |

- **The path adds on every symbol:** +0.016 to +0.043 over the state features, with ridge giving the same picture.
- **The drawdown features carry the most.** Three numbers (how far depth has fallen from its recent peak, and how far it sits above its recent minimum) are the best single group on all four symbols, and they add on top of lags, flow and volatility on all four. That is the hysteresis reading: where the book stands relative to its recent loading history predicts the next depletion, beyond its last few changes.
- Same caveat as above: one test day per symbol, crypto not Nasdaq.

## How far this can go

Unknown for Nasdaq stocks. On public crypto books, depth reversion plus a handful of flow features gives rank correlations of 0.3–0.6 from one day of fitting.

## Next steps

Needs a data licence for the Nasdaq task. On public data: more fitting days, and features from the order-message stream (cancellation bursts) rather than one-second snapshots.
