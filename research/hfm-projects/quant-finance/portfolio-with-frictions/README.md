# Cost-aware long-only portfolio construction under trading frictions on Qlib

- **Field:** quant-finance
- **HFM track:** `discovery_live_rsi_portfolio_frictions` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU, on industry portfolios.
- **Data access:** Community Qlib data release on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** Single-stock data with a clear licence, price limits and halts.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work builds long-only portfolio strategies on Qlib using real daily data and a backtester that incorporates trading costs, price-limit moves, trading halts, and position-weight and turnover constraints. The backtest period covers CSI 300 and an unseen US market, with the backtester, cost rules, and benchmarks held fixed. The evolving component is the Strategy in methods/main, which combines alpha signals, a risk model, and cost-aware portfolio construction. Evaluation follows the task specification in the harbor task's instruction.md.

- **Metric:** `window_score_mean` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.2119
- **HFM effort:** 9 proposals, 9 scored, 0 accepted improvements, about $234 of compute
- **HFM's current best idea:** 对 p219 的 Ridge alpha 仅在预测组合与当前持仓的 Jaccard 相似度低于阈值或预期换手超过 15% 时才触发再平衡，能比固定日程更自适应地捕捉信号强度变化并减少无效交易。

## Existing theory and methods

Mean-variance optimisation with transaction costs, shrinkage covariance (Ledoit-Wolf), turnover control.

## Public data

Kenneth French data library, 49 industry portfolios, daily (public). Qlib's community data release exists but states no licence.

## What limits progress

The signal sets the ceiling. Trading rules only decide how much of it survives costs.

## Route to a step change

Cost-aware optimisation with a shrinkage covariance; test whether factors carry across markets.

## Experiment and results

Code: [`portfolio_costs.py`](portfolio_costs.py) in this folder. It ran on free GitHub Actions compute; the log is in [`research/compute/results/portfolio-costs/`](../../../compute/results/portfolio-costs/run.log).

- **Setup:** Kenneth French's 49 US industry portfolios, daily returns. These are industry indices, not single stocks, with no price limits or halts, so this tests the trading rule and not HFM's backtester. The Qlib community data was not used because it states no licence.
- **Signals:** a slow one (12-month momentum) and a fast one (momentum plus 5-day reversal). The target is equal weight in the 10 best-ranked industries.
- **Rules compared:** five ways of trading towards the target, including HFM's current idea of rebalancing only when the portfolio has drifted. Each rule's one setting was chosen on 1970–1999 and tested on 2000 to September 2026 (27 years).
- **Result at 10 basis points per unit traded (return per unit of risk, net of costs):**

  | Rule | Slow signal | Fast signal | Fast: years better than monthly |
  | --- | --- | --- | --- |
  | All industries, equal weight (no signal) | 0.61 | 0.61 | 16/27 |
  | Trade fully every day | 0.59 | 0.03 | 4/27 |
  | Trade fully every 21 days | **0.70** | 0.50 | — |
  | Trade only when drifted (HFM's idea) | 0.63 | 0.39 | 12/27 |
  | Trade part-way every day | 0.64 | **0.60** | 18/27 |
  | Hold until a holding weakens | 0.66 | 0.59 | 18/27 |

- **What this shows:**
  - **For a fast signal, how you trade decides everything.** Daily trading turns a 12% gross return into −2%. Trading part-way each day, or holding until a position weakens, recovers a ratio of about 0.60 against 0.50 for monthly rebalancing, in 18 of 27 years. At 20 basis points the gap is wider (0.58 against 0.43, 22 of 27 years).
  - **For a slow signal, plain monthly rebalancing was best** at every cost level tested (5, 10 and 20 basis points). None of the cleverer rules beat it out of sample.
  - **HFM's drift-triggered idea was the weakest of the cost-aware rules** in five of six signal and cost combinations. A drift trigger still trades all the way when it fires.
  - **The fast signal never clearly beat holding everything** once costs reached 10 basis points (0.60 against 0.61). Cost-aware trading rescued the signal; it did not make it worth having.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **In this test:** cost-aware trading lifted a fast signal from 0.50 to about 0.60, which only matches holding all industries.
- **Realistic ceiling:** set by the signal, not the trading rule. HFM's 9 proposals with no accepted improvement are consistent with that.

## Next steps

- Repeat on single stocks with a clear licence, adding price limits and halts.
- Test a cost-aware optimiser with a shrinkage covariance against the part-way rule.
- Test whether a signal fitted on one market carries to another.
