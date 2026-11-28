# References: Intraday Trading Overlay for A-Share Holdings

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page or the repository's raw README/LICENSE). The session's web-search allowance ran out before this project's searches, so most entries here are `[P]`, and the A-share-specific literature (T+1 rule, price limits, intraday patterns) could not be confirmed and is under "Unverified leads". Nothing is cited from memory. This is research material, not investment advice.

## Theory and established methods

- **Guijarro-Ordonez, Pelger, Zanotti (2021).** Deep Learning Statistical Arbitrage. arXiv:2106.04028. https://arxiv.org/abs/2106.04028 `[P]`
  - General framework for mean-reversion trading: build residuals from a factor model, extract a signal, then form a trading policy under constraints. The overlay's intraday mean-reversion signals fit this three-step structure.
- **Boyd, Busseti, Diamond, Kahn, Koh, Nystrup, Speth (2017).** Multi-Period Trading via Convex Optimization. arXiv:1705.00109. https://arxiv.org/abs/1705.00109 `[P]`
  - Trading around base holdings with cost terms and hard constraints as a convex problem. The T+1 rule (sell only what was held at the open) and the end-of-day position requirement are linear constraints in this form.
- **Almgren, Chriss (2001).** Optimal execution of portfolio transactions. Journal of Risk. https://www.doi.org/10.21314/JOR.2001.041 `[S]`
  - Cost model for the trades the overlay places; the net-of-cost objective needs an impact term, not only a fixed fee.
- **Lehalle, Mounjid (2016).** Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency. arXiv:1610.00261. https://arxiv.org/abs/1610.00261 `[P]`
  - Shows how the value of an imbalance signal is eroded by adverse selection and latency; relevant to whether five-minute signals survive realistic fills.
- **Yang, Liu, Zhou, Bian, Liu (2020).** Qlib: An AI-oriented Quantitative Investment Platform. arXiv:2009.11189. https://arxiv.org/abs/2009.11189 `[P]`
  - Platform with A-share data handling and a nested (daily plus intraday) decision framework.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - HFM's reward is pinned at 1.0 after 38 scored proposals. Once the metric is uncapped, the best result must be deflated by the number of trials.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Combinatorially symmetric cross-validation across trading days; a direct test of whether the chosen overlay setting ranks well out of sample.
- **Harvey, Liu, Zhu (2016).** ... and the Cross-Section of Expected Returns. Review of Financial Studies 29(1), 5-68. https://nber.org/papers/w20592 `[S]`
  - Multiple-testing hurdle (t-statistic above 3.0) for newly found signals.

## Recent advances and breakthroughs (2023-2026)

- **Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025).** Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. https://arxiv.org/abs/2508.02739 `[P]`
  - Candlestick foundation model pretrained on over 12 billion bars from 45 exchanges. Its repository's quick-start uses a five-minute Shanghai-listed stock file and ships a Qlib A-share fine-tuning and backtest pipeline, so it applies to this task's bar frequency directly.
- **Zhang, Duan, Chen, Chen, Li, Zhao (2023).** Towards Generalizable Reinforcement Learning for Trade Execution. arXiv:2307.11685. https://arxiv.org/abs/2307.11685 `[P]`
  - Shows intraday trading policies trained on replayed history overfit the market context; proposes context compression. The overlay is trained in the same offline setting.
- **Li, Yang, Yang, Xu, Wang, Liu, Bian (2025).** R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. arXiv:2505.15155. https://arxiv.org/abs/2505.15155 `[P]`
  - Agent-driven factor and model search on Qlib A-share data; published counterpart of the HFM agent loop.
- **Wang, Kong, Guo, Hua, Qi, Zhou, Zheng, Wang, Ni, Guo (2025).** QuantBench: Benchmarking AI Methods for Quantitative Investment. arXiv:2504.18600. https://arxiv.org/abs/2504.18600 `[P]`
  - Benchmark whose stated lessons are distribution shift and overfitting in low signal-to-noise data.
- **Zong, Wang, Qin, Feng, Wang, An (2024).** MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading. arXiv:2406.14537. https://arxiv.org/abs/2406.14537 `[P]`
  - Conditions a minute-level trading policy on market regime (crypto data); the regime-conditioning idea transfers to an intraday overlay.
- **Qin, Sun, Zhang, Xia, Wang, An (2023).** EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading. arXiv:2309.12891. https://arxiv.org/abs/2309.12891 `[P]`
  - Hierarchical policy for very long intraday trajectories (crypto data).

## Datasets and benchmarks

- **Qlib `cn_data_1min` (one-minute A-share bars).** Download command in the Qlib README: `python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min`. https://github.com/microsoft/qlib `[P]`
  - This contradicts the project README's statement that only daily data are public: Qlib lists a public one-minute A-share set ("High-frequency data(1min)" released 2021-01-27), and one-minute bars aggregate to five-minute bars. Two cautions from the same README, fetched 2026-11-25: "The official dataset is disabled temporarily" for data-security reasons, and the data were collected from Yahoo Finance and "might not be perfect". The date range and stock coverage of the one-minute set were not confirmed. No licence for the data is stated.
- **Qlib Yahoo collector, one-minute mode.** https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo `[P]`
  - Collects one-minute A-share bars from Yahoo, but the README states that "only the last month's data is available in 1min". Good for a rolling forward sample, not for history.
- **chenditc/investment_data (repo, Apache-2.0 code).** Community A-share daily data in Qlib format. https://github.com/chenditc/investment_data `[P]`
  - The daily source the Qlib README now points to. Daily only. Underlying sources listed as Tushare, Wind and Caihui; data licence not stated.
- **Kronos example data.** A five-minute bar file for one Shanghai-listed stock (`XSHG_5min_600977.csv`) used in the repository's example. https://github.com/shiyu-coder/Kronos `[P]`
  - One stock only; useful for format checks, not for research.
- **No confirmed free source of redistributable five-minute A-share history.** The Qlib one-minute set is the only candidate found, and its official download is currently disabled. See "Unverified leads" for AKShare, Baostock and Tushare.

## Open-source code and tools

- **microsoft/qlib (repo, MIT).** Nested decision framework for joint daily and intraday strategies, high-frequency data example, backtester. https://github.com/microsoft/qlib `[P]`
- **akfamily/akshare (repo, MIT).** Python interface to public Chinese market data pages. https://github.com/akfamily/akshare `[P]`
  - Code is MIT; the README states the data it provides are for academic research use only. Its minute-bar coverage and history depth were not confirmed.
- **shiyu-coder/Kronos (repo, MIT).** Pretrained candlestick models (weights on Hugging Face) and fine-tuning scripts. https://github.com/shiyu-coder/Kronos `[P]`
- **cvxgrp/cvxportfolio (repo, GPL-3.0).** Cost-aware multi-period optimisation and backtesting. https://github.com/cvxgrp/cvxportfolio `[P]`
  - For allocating intraday effort across the stock book under T+1 and end-of-day constraints.
- **microsoft/RD-Agent (repo, MIT).** Automated factor and model research loop on Qlib. https://github.com/microsoft/RD-Agent `[P]`
- **esvhd/pypbo (repo, AGPL-3.0).** Probability of backtest overfitting. https://github.com/esvhd/pypbo `[P]`

## Breakthroughs in related fields that could transfer

- **Sun, Xue, Wang, He, Zhu, Li, An (2021).** DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities. arXiv:2201.09058. https://arxiv.org/abs/2201.09058 `[P]`
  - Intraday RL with a large action space and a hindsight bonus in the reward; a design for the overlay's decision step.
- **Fang, Ren, Liu, Zhou, Zhang, Bian, Yu, Liu (2021).** Universal Trading for Order Execution with Oracle Policy Distillation. arXiv:2103.10860. https://arxiv.org/abs/2103.10860 `[P]`
  - Distils a policy from an oracle teacher with perfect information; the overlay's intraday buy/sell timing can be trained the same way.
- **Poh, Lim, Zohren, Roberts (2020).** Building Cross-Sectional Systematic Strategies By Learning to Rank. arXiv:2012.07149. https://arxiv.org/abs/2012.07149 `[P]`
  - "Effort allocation across a stock book" is a ranking problem: which holdings get the intraday trades.
- **Cont, Cucuringu, Zhang (2021).** Cross-Impact of Order Flow Imbalance in Equity Markets. arXiv:2112.13213. https://arxiv.org/abs/2112.13213 `[P]`
  - Lagged cross-asset flow helps short-horizon forecasts and decays fast; a testable source of five-minute signals.

## Published results to beat

- No published benchmark for a T+1 intraday overlay on five-minute A-share data was found in this session.
- **HFM track (source: project README).** Reward 1.0, which is the cap; the evaluator cannot rank methods until it is uncapped.
- **Qlib benchmark table (daily, CSI 300; source: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md).** LightGBM on Alpha158: IC 0.0448, annualised return 0.0901, information ratio 1.0164.
  - Daily-frequency reference for the base holdings only; it says nothing about intraday overlay value.

## Unverified leads

- AKShare minute-bar functions, Baostock and Tushare as free sources of five-minute A-share bars, with their terms: not confirmed in this session.
- Literature on the T+1 rule and overnight/intraday return patterns in China (for example Qiao and Dam, "The overnight return puzzle and the 'T+1' trading rule in Chinese stock markets"): not confirmed in this session.
- Literature on daily price limits in China (for example Chen, Gao, He, Jiang, Xiong, "Daily price limits and destructive market behavior"): not confirmed in this session.
- Gao, Han, Li, Zhou, "Market intraday momentum" (Journal of Financial Economics): not confirmed in this session.
- Liu, Stambaugh, Yuan, "Size and Value in China": not confirmed in this session.
- Qlib documentation of the one-minute data set's coverage (qlib.readthedocs.io): host not reachable here.
