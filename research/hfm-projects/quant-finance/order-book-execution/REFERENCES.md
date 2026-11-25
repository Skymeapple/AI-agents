# References: Execution Policy Development from Ten-Level Order Book Data

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page or the repository's raw README/LICENSE). Nothing is cited from memory; items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Almgren, Chriss (2001).** Optimal execution of portfolio transactions. Journal of Risk. https://www.doi.org/10.21314/JOR.2001.041 `[S]`
  - The baseline schedule: trade-off between impact cost and price risk; with linear impact and no risk aversion it reduces to TWAP, the benchmark most papers report against.
- **Cont, Kukanov (2017).** Optimal order placement in limit order markets. Quantitative Finance 17(1), 21-39. https://arxiv.org/abs/1210.1625 `[S][P]`
  - Closed-form split between limit and market orders from queue sizes, fees and fill risk; the tactical layer under a schedule.
- **Nevmyvaka, Feng, Kearns (2006).** Reinforcement Learning for Optimized Trade Execution. ICML 2006, 673-680. https://www.cis.upenn.edu/~mkearns/papers/rlexec `[S]`
  - First large-scale RL execution study, on 1.5 years of NASDAQ limit order data; defines the state (time, inventory, book features) still used.
- **Ning, Lin, Jaimungal (2018).** Double Deep Q-Learning for Optimal Execution. arXiv:1812.06600. https://arxiv.org/abs/1812.06600 `[P]`
  - Standard deep RL baseline for execution.
- **Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren, Bouchaud (2011).** Anomalous price impact and the critical nature of liquidity in financial markets. Physical Review X 1, 021006. https://arxiv.org/abs/1105.1694 `[P]`
  - Evidence for concave (square-root) impact; a replay that ignores the agent's own impact overstates savings.
- **Busseti, Lillo (2012).** Calibration of optimal execution of financial transactions in the presence of transient market impact. arXiv:1206.0682. https://arxiv.org/abs/1206.0682 `[S][P]`
  - Calibrating transient-impact execution models to data.
- **Lehalle, Mounjid (2016).** Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency. arXiv:1610.00261. https://arxiv.org/abs/1610.00261 `[P]`
  - Uses order-book imbalance to decide when to keep or cancel a passive order; directly relevant to the "pre-sweep marketable order" idea HFM reports.
- **Huang, Lehalle, Rosenbaum (2015).** Simulating and analyzing order book data: The queue-reactive model. JASA 110(509), 107-122. https://arxiv.org/abs/1312.0563 `[S][P]`
  - State-dependent book simulator used for transaction-cost analysis of execution tactics.

## Recent advances and breakthroughs (2023-2026)

- **Arroyo, Cartea, Moreno-Pino, Zohren (2024).** Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers. Quantitative Finance 24(1), 35-57. https://arxiv.org/abs/2306.05479 `[S][P]`
  - Fill-time distribution from LOB features with cancellations as right-censoring, scored with proper scoring rules; the fill-probability model the README calls for.
- **Lokin, Yu (2024).** Fill Probabilities in a Limit Order Book with State-Dependent Stochastic Order Flows. arXiv:2403.02572. https://arxiv.org/abs/2403.02572 `[P]`
  - Semi-analytical fill probabilities at the best quotes and deeper levels in a queueing model; an interpretable alternative to the neural estimate.
- **Fabre, Ragel (2023).** Interpretable ML for High-Frequency Execution. arXiv:2307.04863. https://arxiv.org/abs/2307.04863 `[S][P]`
  - Small network for fixed-horizon fill probability with a censoring-aware loss, compared across crypto exchanges and Euronext equities.
- **Zhong, Bacry, Guilloux, Muzy (2025).** KANFormer for Predicting Fill Probabilities via Survival Analysis in Limit Order Books. arXiv:2512.05734. https://arxiv.org/abs/2512.05734 `[S][P]`
  - Time-to-fill prediction that adds agent-level information to market-level features.
- **Zhang, Duan, Chen, Chen, Li, Zhao (2023).** Towards Generalizable Reinforcement Learning for Trade Execution. arXiv:2307.11685. https://arxiv.org/abs/2307.11685 `[P]`
  - Shows RL execution trained on historical replay overfits because the market context is large and samples are few; derives a generalisation bound and proposes context compression.
- **Espana, Hafsi, Lillo, Vittori (2025).** Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution. arXiv:2511.15262. https://arxiv.org/abs/2511.15262 `[S][P]`
  - Argues policy optimisation needs counterfactual feedback that historical data cannot give, so trains a Double DQN in a queue-reactive simulator with transient impact.
- **Hafsi, Vittori (2024).** Optimal Execution with Reinforcement Learning. arXiv:2411.06389. https://arxiv.org/abs/2411.06389 `[S][P]`
  - RL execution agent in a multi-agent simulator, benchmarked against TWAP and other standard schedules.
- **de Witt, Pakkanen (2026).** Diverse Approaches to Optimal Execution Schedule Generation. arXiv:2601.22113. https://arxiv.org/abs/2601.22113 `[S][P]`
  - Quality-diversity search (MAP-Elites) produces regime-specialist schedules; reports 8-10% gains inside niches and losses elsewhere.
- **Zaznov, Badii, Kunkel, Dufour (2026).** TT-DAC-PS: Twin-Target Deterministic Actor-Critic with Policy Smoothing for Optimal Trade Execution. arXiv:2606.08379. https://arxiv.org/abs/2606.08379 `[S][P]`
  - Actor-critic that combines Almgren-Chriss impact with LOB features; the search summary reports lower mean implementation shortfall than classical and RL baselines.
- **Frey, Li, Nagy, Sapora, Lu, Zohren, Foerster, Calinescu (2023).** JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading. ICAIF 2023. https://arxiv.org/abs/2308.13289 `[S][P]`
  - Runs thousands of books in parallel on GPU with an execution environment; removes the compute limit on RL execution research.

## Datasets and benchmarks

- **Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis (2017).** Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods (FI-2010). https://arxiv.org/abs/1705.03233 `[S][P]`
  - Free ten-level data for 5 Nasdaq Nordic stocks over 10 days; the LOBCAST paper states CC BY 4.0 (secondary source). Usable for a small replay, but it is normalised and event-sampled, so execution costs in currency terms cannot be recovered reliably.
- **Binance public data archive.** Futures `aggTrades`, `bookTicker`, `bookDepth` daily files. https://github.com/binance/binance-public-data `[P]`
  - Free trade and top-of-book data for a replay with real ticks. Terms (v1.0, updated 2026-08-26): CC BY-NC-SA 4.0, non-commercial.
- **Bybit historical order book archive (free L2, 200 levels at 100 ms since 2025-08-21 per a Bybit API announcement).** https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e `[S]`
  - The only free multi-level book history found; a different asset class from Nasdaq equities. Bybit's terms were not found.
- **Databento.** Usage, pricing and data credits FAQ. https://databento.com/docs/faqs/usage-pricing-and-data-credits `[S]`
  - USD 125 of free historical credits per new team; enough for a small licensed mbp-10 sample, not for redistribution.
- **No free source found** for Nasdaq ten-level book plus trade data on the task's universe.

## Open-source code and tools

- **KangOxford/jax-lob (repo).** JAX-LOB simulator and gymnax execution environment. https://github.com/KangOxford/jax-lob `[P]`
- **jpmorganchase/abides-jpmc-public (repo, BSD-3).** ABIDES agent-based market simulator with Gym environments, including an execution environment. https://github.com/jpmorganchase/abides-jpmc-public `[P]`
  - Paper: Amrouni, Moulin, Vann, Vyetrenko, Balch, Veloso (2021), ABIDES-Gym. https://arxiv.org/abs/2110.14771 `[S][P]`
- **nkaz001/hftbacktest (repo, MIT).** Replay backtester that models feed and order latency and queue position from L2/L3 feeds. https://github.com/nkaz001/hftbacktest `[P]`
  - Queue-position fill simulation is the piece most replay evaluators lack.
- **microsoft/qlib (repo, MIT).** Includes a nested decision framework and RL order execution examples. https://github.com/microsoft/qlib `[P]`
  - Paper for the execution method: Fang, Ren, Liu, Zhou, Zhang, Bian, Yu, Liu (2021), Universal Trading for Order Execution with Oracle Policy Distillation. https://arxiv.org/abs/2103.10860 `[P]`
- **Pardo, Auth, Dascalu (2022).** A Modular Framework for Reinforcement Learning Optimal Execution. arXiv:2208.06244. https://arxiv.org/abs/2208.06244 `[S][P]`
  - Reward defined as improvement over TWAP on replayed LOB snapshots; a template for the evaluator.

## Breakthroughs in related fields that could transfer

- **Nagy et al. (2025).** LOB-Bench: Benchmarking Generative AI for Finance. ICML 2025. https://arxiv.org/abs/2502.09172 `[S][P]`
  - Includes price-response (impact) tests for generated order flow; a way to check that a simulator reacts to the agent's orders realistically before training on it.
- **Li, Liu, Liu, Fang, Wang, Xu, Bian (2024).** MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model. arXiv:2409.07486. https://arxiv.org/abs/2409.07486 `[P]`
  - Learned order-level simulator that responds to injected orders; an alternative to pure replay.
- **Coletta, Moulin, Vyetrenko, Balch (2022).** Learning to simulate realistic limit order book markets from data as a World Agent. arXiv:2210.09897. https://arxiv.org/abs/2210.09897 `[P]`
  - Single learned "world agent" reproduces market response to an experimental agent.
- **Cartea, Duran-Martin, Sánchez-Betancourt (2023).** Detecting Toxic Flow. arXiv:2312.05827. https://arxiv.org/abs/2312.05827 `[P]`
  - Online prediction of adverse selection; the same signal tells an execution policy when resting passively is costly.

## Published results to beat

- **JAX-LOB (source: search summary of arXiv:2308.13289).** 550 versus 74 environment steps per second against CPU simulators (about 7x), up to 75x faster per-message processing. https://arxiv.org/abs/2308.13289
  - A throughput baseline, not a cost baseline.
- **de Witt, Pakkanen (source: arXiv abstract).** 8-10% improvement for regime specialists over a PPO baseline within their niches. https://arxiv.org/abs/2601.22113
- **TWAP / Almgren-Chriss** remain the reference policies in every paper above; no paper found in this session reports a transferable implementation-shortfall number on public Nasdaq ten-level data. A search summary noted that no 2024-2025 paper was found that trains and evaluates purely on real LOB data against TWAP.

## Evaluating results honestly

- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - Correct any reported saving for the number of policies tried; HFM's reward is pinned at 1.0, so it cannot rank policies at all.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Applies to selecting among execution policies on a fixed set of replayed parent orders.

## Unverified leads

- LOBSTER free sample files at lobsterdata.com: not returned by search, site not reachable here. Not confirmed.
- Maglaras, Moallemi, Wang, "A deep learning approach to estimating fill probabilities in a limit order book" (Quantitative Finance, 2022): mentioned in a search summary as prior work, no URL confirmed.
- Obizhaeva, Wang, "Optimal trading strategy and supply/demand dynamics": not confirmed in this session.
- Gatheral, "No-dynamic-arbitrage and market impact": not confirmed in this session.
