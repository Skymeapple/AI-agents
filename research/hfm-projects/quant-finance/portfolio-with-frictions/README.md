# Cost-aware long-only portfolio construction under trading frictions on Qlib

- **Field:** quant-finance
- **HFM track:** `discovery_live_rsi_portfolio_frictions` (active on HFM)
- **Project status:** Open: public data exist, experiment not yet built

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

Qlib public daily data for CSI 300 and US stocks.

## What limits progress

Simple ridge alpha with rule-based rebalancing.

## Route to a step change

Cost-aware optimisation with a shrinkage covariance; test whether factors carry across markets.

## Experiment and results

No experiment has been run for this project.

- **Why:** Public data exist (Qlib daily data) but the backtest with price limits and halts was not built.

## How far this can go

Unknown; cross-market transfer of alpha factors is typically weak.

## Next steps

Build the Qlib backtest and test transfer from CSI 300 to US.
