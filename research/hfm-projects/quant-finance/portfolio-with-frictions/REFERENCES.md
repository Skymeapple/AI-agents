# References: Cost-aware long-only portfolio construction under trading frictions on Qlib

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page or the repository's raw README/LICENSE). The session's web-search allowance ran out before this project's searches, so every entry here is `[P]`; several classic journal papers could therefore not be confirmed and are under "Unverified leads". Nothing is cited from memory. This is research material, not investment advice.

## Theory and established methods

- **Boyd, Busseti, Diamond, Kahn, Koh, Nystrup, Speth (2017).** Multi-Period Trading via Convex Optimization. arXiv:1705.00109. https://arxiv.org/abs/1705.00109 `[P]`
  - The standard formulation of the README's "route to a step change": maximise forecast return minus risk, transaction cost and holding cost under weight and turnover constraints, as one convex problem per rebalance.
- **Boyd, Johansson, Kahn, Schiele, Schmelzer (2024).** Markowitz Portfolio Construction at Seventy. arXiv:2401.05080. https://arxiv.org/abs/2401.05080 `[P]`
  - Current practical recipe: Markowitz with transaction cost, leverage and turnover terms, plus robustness to forecast error, still convex and fast to solve.
- **Ledoit, Wolf (2012).** Nonlinear shrinkage estimation of large-dimensional covariance matrices. Annals of Statistics 40(2), 1024-1060. https://arxiv.org/abs/1207.5322 `[P]`
  - Shrinkage covariance for the case where the number of stocks (300) is not small relative to the number of days; the risk model the README names.
- **Bun, Bouchaud, Potters (2016).** Cleaning large correlation matrices: tools from random matrix theory. arXiv:1610.08104. https://arxiv.org/abs/1610.08104 `[P]`
  - Review of eigenvalue cleaning; alternative risk-model estimators with out-of-sample risk comparisons.
- **Yang, Liu, Zhou, Bian, Liu (2020).** Qlib: An AI-oriented Quantitative Investment Platform. arXiv:2009.11189. https://arxiv.org/abs/2009.11189 `[P]`
  - The platform the task is built on: data layer, Alpha158/Alpha360 features, backtester.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - `window_score_mean` is an average over backtest windows; any strategy chosen after several tries needs this correction.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Estimates how often the selected strategy is below median out of sample; the unseen US market in this task is exactly such a test.
- **Harvey, Liu, Zhu (2016).** ... and the Cross-Section of Expected Returns. Review of Financial Studies 29(1), 5-68. https://nber.org/papers/w20592 `[S]`
  - Multiple-testing hurdle for alpha factors; relevant when choosing which Alpha158 features to keep.

## Recent advances and breakthroughs (2023-2026)

- **Wang, Kong, Guo, Hua, Qi, Zhou, Zheng, Wang, Ni, Guo (2025).** QuantBench: Benchmarking AI Methods for Quantitative Investment. arXiv:2504.18600. https://arxiv.org/abs/2504.18600 `[P]`
  - Full-pipeline benchmark; its stated findings are the need for continual learning under distribution shift, better relational modelling, and stronger protection against overfitting in low signal-to-noise data.
- **Li, Yang, Yang, Xu, Wang, Liu, Bian (2025).** R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. arXiv:2505.15155. https://arxiv.org/abs/2505.15155 `[P]`
  - Agent loop that co-optimises factors and models on Qlib; the nearest published system to what HFM agents do on this track, with code at https://github.com/microsoft/RD-Agent (MIT).
- **Li, Liu, Shen, Wang, Chen, Huang (2023).** MASTER: Market-Guided Stock Transformer for Stock Price Forecasting. arXiv:2312.15235. https://arxiv.org/abs/2312.15235 `[P]`
  - Models cross-time stock correlation with market-conditioned feature selection; a candidate replacement for the ridge alpha. Code: https://github.com/SJTU-DMTai/MASTER (MIT).
- **Yu, Xue, Ao, Pan, He, Tu, He (2023).** Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning. KDD 2023. https://arxiv.org/abs/2306.12964 `[P]`
  - Searches for a set of formula factors that work together, on Qlib data. Code: https://github.com/RL-MLDM/alphagen.
- **Shi, Song, Zhang, Shi, Luo, Ao, Arian, Seco (2024).** AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors. arXiv:2406.18394. https://arxiv.org/abs/2406.18394 `[P]`
  - Factor mining with time-varying combination weights; addresses factor decay.
- **Hu, Li, Liu, Zhu, Li, Dai, Xia, Cheng, Jiang (2025).** FinTSB: A Comprehensive and Practical Benchmark for Financial Time Series Forecasting. Frontiers of Computer Science 2026. https://arxiv.org/abs/2502.18834 `[P]`
  - Argues existing evaluations lack a unified protocol and neglect market-structure factors; a second reference for backtest protocol.
- **Nicolini, Manzi, Delatte (2025).** skfolio: Portfolio Optimization in Python. arXiv:2507.04176. https://arxiv.org/abs/2507.04176 `[P]`
  - Describes a scikit-learn style library with cross-validation for portfolio models, including combinatorial purged cross-validation.
- **Cao, Wang, Lin, Wu, Zhang, Ni, Guo (2025).** From Deep Learning to LLMs: A survey of AI in Quantitative Investment. arXiv:2503.21422. https://arxiv.org/abs/2503.21422 `[P]`
  - Survey of the alpha-to-portfolio pipeline.

## Datasets and benchmarks

- **Qlib public data (CSI 300 and US, daily).** https://github.com/microsoft/qlib `[P]`
  - README (fetched 2026-11-25) states: "Due to more restrict data security policy. The official dataset is disabled temporarily", and points to a community source. It also states the data come from Yahoo Finance and "might not be perfect". The project README's assumption that Qlib public data are available should be re-checked before building.
- **chenditc/investment_data (repo, Apache-2.0 code).** Community-maintained A-share daily data in Qlib format, released as `qlib_bin.tar.gz`. https://github.com/chenditc/investment_data `[P]`
  - The replacement source named by the Qlib README. Its own README lists the underlying sources as Tushare (current) and Wind and Caihui (static, to 2019). The licence of the data themselves is not stated; treat redistribution as unresolved.
- **Qlib benchmark table (CSI 300, Alpha158 and Alpha360).** https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md `[P]`
  - Mean and standard deviation over 20 seeds for each model: IC, Rank IC, annualised return, information ratio, maximum drawdown. See "Published results to beat".

## Open-source code and tools

- **microsoft/qlib (repo, MIT).** Backtester, Alpha158/Alpha360 handlers, benchmark model zoo, nested decision framework. https://github.com/microsoft/qlib `[P]`
- **cvxgrp/cvxportfolio (repo, GPL-3.0).** Implements the multi-period trading paper: cost-aware optimisation and backtesting. https://github.com/cvxgrp/cvxportfolio `[P]`
  - Reference implementation for the optimiser that should replace rule-based rebalancing.
- **cvxpy/cvxpy (repo, Apache-2.0).** Convex modelling language under the above. https://github.com/cvxpy/cvxpy `[P]`
- **skfolio/skfolio (repo, BSD-3).** Portfolio optimisation with scikit-learn API and cross-validation. https://github.com/skfolio/skfolio `[P]`
- **PyPortfolio/PyPortfolioOpt (repo, MIT).** Mean-variance, Black-Litterman, shrinkage covariance, hierarchical risk parity. https://github.com/PyPortfolio/PyPortfolioOpt `[P]`
- **dcajasn/Riskfolio-Lib (repo; licence file is a copyright notice by Dany Cajas, check terms).** Portfolio optimisation library. https://github.com/dcajasn/Riskfolio-Lib `[P]`
- **esvhd/pypbo (repo, AGPL-3.0).** Probability of backtest overfitting. https://github.com/esvhd/pypbo `[P]`

## Breakthroughs in related fields that could transfer

- **Zhang, Zohren, Roberts (2020).** Deep Learning for Portfolio Optimization. arXiv:2005.13665. https://arxiv.org/abs/2005.13665 `[P]`
  - Optimises portfolio weights directly for a Sharpe objective, skipping the forecast step; an end-to-end alternative that can include a cost term in the loss.
- **Poh, Lim, Zohren, Roberts (2020).** Building Cross-Sectional Systematic Strategies By Learning to Rank. arXiv:2012.07149. https://arxiv.org/abs/2012.07149 `[P]`
  - A long-only top-k portfolio depends on rank, not level; ranking losses target that directly.
- **Xu, Liu, Wang, Xia, Bian, Yin, Liu (2021).** HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information. arXiv:2110.13716. https://arxiv.org/abs/2110.13716 `[P]`
  - Uses shared-concept structure between stocks; top information ratio in the Qlib Alpha360 table.
- **Zhang, Li, Chen, Jin, Tang, Li (2020).** DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis. arXiv:2010.01265. https://arxiv.org/abs/2010.01265 `[P]`
  - Sample reweighting and feature selection for noisy financial labels; top information ratio in the Qlib Alpha158 table.
- **Lin, Zhou, Liu, Bian (2021).** Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport. arXiv:2106.12950. https://arxiv.org/abs/2106.12950 `[P]`
  - Handles several coexisting return patterns; relevant to cross-market transfer, where patterns differ by market.
- **Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025).** Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. https://arxiv.org/abs/2508.02739 `[P]`
  - Pretrained on candlestick data from 45 exchanges; a test of whether a multi-market pretrained model transfers from CSI 300 to the US better than hand-made factors. The repository includes a Qlib A-share fine-tuning and backtest example.

## Published results to beat

All figures below are from the Qlib benchmark README (CSI 300, mean of 20 seeds, Qlib's default top-k strategy and cost settings, not this task's backtester): https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md

- **Linear, Alpha158:** IC 0.0397, annualised return 0.0692, information ratio 0.9209, maximum drawdown -0.1509.
  - The closest published stand-in for the task's ridge alpha.
- **LightGBM, Alpha158:** IC 0.0448, annualised return 0.0901, information ratio 1.0164, maximum drawdown -0.1038.
- **DoubleEnsemble, Alpha158:** IC 0.0521, annualised return 0.1158, information ratio 1.3432, maximum drawdown -0.0920.
- **HIST, Alpha360:** IC 0.0522, annualised return 0.0987, information ratio 1.3726, maximum drawdown -0.0681.
- **TRA, Alpha360:** IC 0.0485, annualised return 0.0920, information ratio 1.2789 (standard deviation 0.42 across seeds).
  - Seed-to-seed spread of the information ratio is 0.2 to 0.4 for the neural models; differences smaller than that are not evidence.
- **HFM track (source: project README):** `window_score_mean` 0.2119, with 0 of 9 proposals accepted. Not comparable to the Qlib table.

## Unverified leads

- Ledoit, Wolf (2004), "A well-conditioned estimator for large-dimensional covariance matrices" (Journal of Multivariate Analysis) and "Honey, I shrunk the sample covariance matrix": not confirmed in this session.
- Gârleanu, Pedersen (2013), "Dynamic Trading with Predictable Returns and Transaction Costs" (Journal of Finance): not confirmed in this session.
- DeMiguel, Garlappi, Uppal (2009), "Optimal Versus Naive Diversification" (Review of Financial Studies): not confirmed in this session.
- Liu, Stambaugh, Yuan (2019), "Size and Value in China" (Journal of Financial Economics), on why US factors do not carry over to A-shares unchanged: not confirmed in this session.
- Qlib backtest documentation on price-limit thresholds, suspended stocks and cost parameters (qlib.readthedocs.io): host not reachable here; not confirmed.
