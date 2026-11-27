# References: Minute-Level Perpetual Trading Strategy from Price, Flow, and Funding Data

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page, the repository's raw README/LICENSE, or the data host). Nothing is cited from memory; items that could not be confirmed are under "Unverified leads". This is research material, not investment advice.

## Theory and established methods

- **He, Manela, Ross, von Wachter (2022, revised 2024).** Fundamentals of Perpetual Futures. arXiv:2212.06888. https://arxiv.org/abs/2212.06888 `[S][P]`
  - Derives no-arbitrage prices and, with trading costs, no-arbitrage bounds for perpetuals; shows "funding rate arbitrage" is a random-maturity arbitrage and is not risk-free. Explains why the funding-carry family in this folder's experiment persists and why it still carries risk.
- **Ackerer, Hugonnier, Jermann (2025).** Perpetual Futures Pricing. Mathematical Finance 36(3), 481-499 (NBER w32936; arXiv:2310.11771). https://arxiv.org/abs/2310.11771 `[S][P]`
  - Prices the perpetual as a discounted expected spot price at a random time set by the funding rule; links funding specification to the size of the futures-spot gap.
- **Angeris, Chitra, Evans, Lorig (2022).** A primer on perpetuals. arXiv:2209.03307. https://arxiv.org/abs/2209.03307 `[P]`
  - Short formal treatment of funding and replication for perpetual contracts.
- **Schmeling, Schrimpf, Todorov (2023, revised 2025).** Crypto carry. BIS Working Paper 1087 (also CEPR DP 20719). https://bis.org/publ/work1087.htm `[S]`
  - Carry is driven by leveraged demand from trend-chasing investors and scarce arbitrage capital, and high carry predicts later crashes; the funding-carry return is compensation for crash risk, so a Sharpe ratio alone understates the risk.
- **Liu, Tsyvinski, Wu (2022).** Common Risk Factors in Cryptocurrency. Journal of Finance 77(2), 1133-1177. https://econpapers.repec.org/RePEc:bla:jfinan:v:77:y:2022:i:2:p:1133-1177 `[S]`
  - Market, size and momentum factors explain the cross-section of coin returns; the baseline risk model for any market-neutral perpetual portfolio.
- **Makarov, Schoar (2020).** Trading and arbitrage in cryptocurrency markets. Journal of Financial Economics 135(2), 293-319. https://econpapers.repec.org/RePEc:eee:jfinec:v:135:y:2020:i:2:p:293-319 `[S]`
  - Signed order flow has a large common component that explains most of the return; relevant to the taker-flow signals in this track.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - Corrects a Sharpe ratio for the number of settings tried and for skew and fat tails; apply it to the 36-setting grid in `perp_funding_carry.py` (best development Sharpe 0.79).
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Combinatorially symmetric cross-validation; turns the single development/test split used here into a distribution of rank degradation.
- **Harvey, Liu, Zhu (2016).** ... and the Cross-Section of Expected Returns. Review of Financial Studies 29(1), 5-68. https://nber.org/papers/w20592 `[S]`
  - Multiple-testing argument for a t-statistic hurdle above 3.0 for a newly found factor.

## Recent advances and breakthroughs (2023-2026)

- **Bieganowski, Ślepaczuk (2026).** Explainable Patterns in Cryptocurrency Microstructure. arXiv:2602.00776. https://arxiv.org/abs/2602.00776 `[S][P]`
  - One-second Binance perpetual data, 2022 to October 2025; the same order-flow features have similar importance and effect shapes across five coins of very different size, validated with a conservative taker backtest. Supports one pooled model across contracts.
- **Borri, Liu, Tsyvinski, Wu (2025).** Cryptocurrency as an Investable Asset Class: Coming of Age. arXiv:2510.14435. https://arxiv.org/abs/2510.14435 `[S][P]`
  - Organises crypto return evidence into seven stylised facts, including carry; a search summary quotes it as saying the carry profit comes mostly from the funding rate and that funding premia are neither guaranteed nor permanent.
- **Zhu, Cai (2026).** Artificial Intelligence in Equity and Crypto Markets: Progress, Profitability Evidence, and the Limits of Automated Investing. arXiv:2609.04917. https://arxiv.org/abs/2609.04917 `[S][P]`
  - Review to 31 August 2026; finds progress is mainly in prediction and tooling and that evidence for durable net-of-cost returns is thin. A search summary notes its checklist for perpetual backtests: contract-level funding timestamps, mark-price liquidation, margin mode, fee tier, downtime, and funding intervals that changed over time.
- **Kim, Park (2025).** Designing funding rates for perpetual futures in cryptocurrency markets. arXiv:2506.08573. https://arxiv.org/abs/2506.08573 `[S][P]`
  - Shows how the funding rule determines how closely the perpetual tracks its target; useful for modelling the predictable part of funding.
- **Zheng, He, Yang (2023).** Optimal Execution Using Reinforcement Learning. arXiv:2306.17178. https://arxiv.org/abs/2306.17178 `[S][P]`
  - Cross-exchange signals improve crypto execution; relevant because HFM's reported gains in this track come from passive execution, not signals.
- **Fabre, Ragel (2023).** Interpretable ML for High-Frequency Execution. arXiv:2307.04863. https://arxiv.org/abs/2307.04863 `[S][P]`
  - Fill-probability model for limit orders on crypto exchanges; needed for the "repeat with passive execution" next step, since a passive fill assumption drives the result.

## Datasets and benchmarks

- **Binance public data archive.** USD-margined futures: `klines` (1m bars with taker-buy volume), `aggTrades`, `fundingRate`, `metrics` (open interest, long/short ratios, taker volume ratio at 5-minute steps), `bookTicker`, `bookDepth`. https://data.binance.vision/ ; documentation https://github.com/binance/binance-public-data `[S][P]`
  - Covers the next step (one-minute resolution) at no cost. Licence changed in substance: "Binance Vision Dataset Terms" v1.0, last updated 2026-08-26, grant CC BY-NC-SA 4.0. Permitted uses include academic research and personal non-production backtesting; live proprietary trading, signal distribution for compensation and commercial use are prohibited; redistributed derivative works must attribute Binance Vision and keep the same licence.
- **Tardis.dev.** Historical data for Bybit and OKX perpetuals, including funding updates. https://docs.tardis.dev/historical-data-details/okex-swap `[S]`
  - First day of each month is free without an API key; allows a cross-venue check of funding and prices on a sparse sample.
- **OKX REST `/public/funding-rate-history`.** Described at https://mintlify.com/tiagosiebler/okx-api/rest-client/market-data `[S]`
  - Free funding history for a second venue, for an out-of-venue test of the funding-carry family.

## Open-source code and tools

- **nkaz001/hftbacktest (repo, MIT).** Replay backtester with latency and queue-position fill models; Binance Futures examples. https://github.com/nkaz001/hftbacktest `[P]`
  - A passive-execution study without a queue model will overstate fills; this is the standard open tool for it.
- **esvhd/pypbo (repo, AGPL-3.0).** Python implementation of the probability of backtest overfitting. https://github.com/esvhd/pypbo `[P]`
  - Drop-in for scoring the settings grid.
- **stefan-jansen/machine-learning-for-trading (repo, MIT).** Includes a worked multiple-testing and deflated Sharpe ratio section (`08_ml4t_workflow/01_multiple_testing`). https://github.com/stefan-jansen/machine-learning-for-trading `[S][P]`
- **cvxgrp/cvxportfolio (repo, GPL-3.0).** Multi-period portfolio optimisation and backtesting with transaction-cost terms. https://github.com/cvxgrp/cvxportfolio `[P]`
  - Replaces equal-weight top/bottom sorting with a cost-aware optimiser, which matters when cost exceeds signal.
- **ccxt/ccxt (repo, MIT).** Unified exchange API for funding and bar history on other venues. https://github.com/ccxt/ccxt `[P]`

## Breakthroughs in related fields that could transfer

- **Boyd, Busseti, Diamond, Kahn, Koh, Nystrup, Speth (2017).** Multi-Period Trading via Convex Optimization. arXiv:1705.00109. https://arxiv.org/abs/1705.00109 `[P]`
  - Trading only when expected gain exceeds modelled cost is the standard remedy for "signals are weak relative to trading costs".
- **Poh, Lim, Zohren, Roberts (2020).** Building Cross-Sectional Systematic Strategies By Learning to Rank. arXiv:2012.07149. https://arxiv.org/abs/2012.07149 `[P]`
  - Ranking losses instead of regression for cross-sectional long-short portfolios; the abstract reports roughly a threefold Sharpe improvement for cross-sectional momentum in its setting (not net of crypto costs).
- **Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025).** Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. https://arxiv.org/abs/2508.02739 `[P]`
  - Pretrained on over 12 billion candlestick records from 45 exchanges; a candidate pretrained encoder for one-minute bars. Code and weights: https://github.com/shiyu-coder/Kronos (MIT).
- **Guijarro-Ordonez, Pelger, Zanotti (2021).** Deep Learning Statistical Arbitrage. arXiv:2106.04028. https://arxiv.org/abs/2106.04028 `[P]`
  - Residual portfolios from factor models plus a learned trading policy; the template for reversal signals after removing the common crypto factor.

## Published results to beat

- **He, Manela, Ross, von Wachter (source: search summary of arXiv v5 HTML).** A futures-spot gap rule on Bitcoin perpetuals earns a Sharpe ratio of 1.8 at retail-level trading costs and up to 3.5 for market makers who pay no fees. https://arxiv.org/abs/2212.06888
  - The comparison point for this folder's funding-carry result (Sharpe roughly 0.4 to 1.6 in 2024-2025 at 8-hour rebalancing, 4 bp cost). The paper's rule is time-series and uses spot; this folder's is cross-sectional and futures-only, so the numbers are not like for like.
- **Schmeling, Schrimpf, Todorov (source: BIS summary via search).** Crypto carry averages above 10% a year and has at times exceeded 40% (the abstract says up to 60%) a year. https://bis.org/publ/work1087.htm
- **Borri, Liu, Tsyvinski, Wu (source: search summary).** Funding-rate component of carry: mean about 8% a year with volatility about 0.8% over the full sample; the summary notes this may not be net of costs. https://arxiv.org/abs/2510.14435
- **HFM track (source: project README).** Best reward 0.0776, from passive execution changes with signals unchanged.

## Unverified leads

- A peer-reviewed cross-sectional funding-rate strategy study with net-of-cost returns for 2024-2025: a targeted search found none.
- Management Science publication of "Crypto Carry" (DOI 10.1287/mnsc.2024.05069 appeared as a search hit; details not confirmed).
- Christin, Routledge, Soska, Zetlin-Jones, "The crypto carry trade": not confirmed in this session.
- Bybit public archive (public.bybit.com) for funding and trades: host not reachable here, terms not found.
