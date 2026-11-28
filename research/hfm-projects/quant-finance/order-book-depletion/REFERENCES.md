# References: Nasdaq Ten-Level Order Book Depth Exhaustion Ranking

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page or the repository's raw README/LICENSE). Nothing is cited from memory; items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Cont, Kukanov, Stoikov (2014).** The Price Impact of Order Book Events. Journal of Financial Econometrics 12(1), 47-88. https://arxiv.org/abs/1011.6402 `[S][P]`
  - Defines order-flow imbalance (OFI) and shows short-horizon price change is linear in OFI with slope inversely proportional to depth; depth exhaustion is the same mechanism seen from the queue side.
- **Cont, de Larrard (2013).** Price dynamics in a Markovian limit order market. SIAM Journal on Financial Mathematics 4(1), 1-25. https://arxiv.org/abs/1104.4596 `[P]`
  - Models the best queues as a Markovian queueing system and gives analytical expressions for the probability of an upward price move conditional on the state of the book. In this model a price move happens when a best queue is exhausted, so it is a direct analytical baseline for a depletion score.
- **Huang, Lehalle, Rosenbaum (2015).** Simulating and analyzing order book data: The queue-reactive model. Journal of the American Statistical Association 110(509), 107-122. https://arxiv.org/abs/1312.0563 `[S][P]`
  - Arrival, cancel and trade intensities depend on current queue sizes; estimating these per level is the event-time model the README names as the route to a step change.
- **Wu, Rambaldi, Muzy, Bacry (2019).** Queue-reactive Hawkes models for the order flow. arXiv:1901.08938. https://arxiv.org/abs/1901.08938 `[S][P]`
  - Adds self- and cross-excitation to the queue-reactive model; a search summary of the paper reports that the Hawkes term markedly improves on the pure queue-reactive model on Eurex Bund and DAX level-1 data.
- **Bacry, Mastromatteo, Muzy (2015).** Hawkes processes in finance. arXiv:1502.04592. https://arxiv.org/abs/1502.04592 `[P]`
  - Review of Hawkes models for order flow, including estimation; reference for kernels and calibration.
- **Gould, Bonart (2015).** Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book. arXiv:1512.03492. https://arxiv.org/abs/1512.03492 `[P]`
  - Logistic regressions on 10 liquid Nasdaq stocks show that bid/ask queue imbalance predicts the direction of the next mid-price move; the natural snapshot baseline.
- **Gould, Porter, Williams, McDonald, Fenn, Howison (2010).** Limit Order Books. arXiv:1012.0349. https://arxiv.org/abs/1012.0349 `[P]`
  - Survey of LOB mechanics and stylised facts.
- **Cont, Cucuringu, Zhang (2021).** Cross-Impact of Order Flow Imbalance in Equity Markets. arXiv:2112.13213. https://arxiv.org/abs/2112.13213 `[P]`
  - Combines OFI across the top levels into one integrated variable; reports that lagged cross-asset OFI improves short-horizon forecasts, which supports a cross-sectional (79-stock) model.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - HFM ran 58 proposals on one frozen evaluator; the same selection-bias logic applies to a ranking score, so gains of one part in 100,000 are within selection noise.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Combinatorially symmetric cross-validation gives a direct estimate of how often the best configuration in-sample is below median out-of-sample.

## Recent advances and breakthroughs (2023-2026)

- **Prata, Masi, Berti, Arrigoni, Coletta, Cannistraci, Vyetrenko, Velardi, Bartolini (2023).** LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study (LOBCAST). arXiv:2308.01915. https://arxiv.org/abs/2308.01915 `[S][P]`
  - Re-tests fifteen deep LOB models and finds all lose much of their performance on new data; a warning against trusting small gains on a fixed evaluator.
- **Briola, Bartolucci, Aste (2025).** Deep Limit Order Book Forecasting: a microstructural guide. Quantitative Finance 25(7), 1101-1131. https://arxiv.org/abs/2403.09267 `[S][P]`
  - Shows forecastability depends on stock microstructure (tick size, depth) and that accuracy does not equal a usable signal; proposes scoring by correctly predicted complete transactions.
- **Briola, Bartolucci, Aste (2024).** HLOB -- Information Persistence and Structure in Limit Order Books. arXiv:2405.18938. https://arxiv.org/abs/2405.18938 `[P]`
  - Models dependency structure among volume levels on three NASDAQ datasets of 15 stocks each; relevant to how information is distributed across the ten levels.
- **Berti, Kasneci (2025).** TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data. arXiv:2502.15757. https://arxiv.org/abs/2502.15757 `[S][P]`
  - Attention over both time and price levels; also reports that predictability has fallen over time and that a simple MLP matches earlier complex models.
- **Jain, Firoozye, Kochems, Treleaven (2023).** Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process. arXiv:2312.08927. https://arxiv.org/abs/2312.08927 `[S][P]`
  - Hawkes model with order sizes and a queue-depletion event at the best level, fitted to a NASDAQ stock; closest published match to "event-time model of depletion".
- **Jain, Muzy, Kochems, Bacry (2024).** No Tick-Size Too Small: A General Method for Modelling Small Tick Limit Order Books. arXiv:2410.08744. https://arxiv.org/abs/2410.08744 `[S][P]`
  - Extends Hawkes LOB models to sparse, multi-tick-spread books; needed because many S&P 500 names are small-tick.
- **Nagy, Frey, Li, Sarkar, Vyetrenko, Zohren, Calinescu, Foerster (2025).** LOB-Bench: Benchmarking Generative AI for Finance -- an Application to Limit Order Book Data. ICML 2025 (PMLR 267). https://arxiv.org/abs/2502.09172 `[S][P]`
  - Standard distributional tests (spread, volumes, imbalance, inter-arrival times, price response) for message-level models in LOBSTER format.
- **Nagy, Frey, Sapora, Li, Calinescu, Zohren, Foerster (2023).** Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network. arXiv:2309.00638. https://arxiv.org/abs/2309.00638 `[P]`
  - Autoregressive model over tokenised messages; depletion probabilities can be read off simulated continuations.
- **Jain, Firoozye, Kochems, Treleaven (2024).** Limit Order Book Simulations: A Review. arXiv:2402.17359. https://arxiv.org/abs/2402.17359 `[P]`
  - Survey of point-process, agent-based and deep generative LOB models.

## Datasets and benchmarks

- **Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis (2017).** Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods (FI-2010). arXiv:1705.03233 (Journal of Forecasting per search result). https://arxiv.org/abs/1705.03233 `[S][P]`
  - Free ten-level LOB data: 5 Nasdaq Nordic stocks, 10 trading days, about 4 million events. The LOBCAST paper states the licence as CC BY 4.0 (secondary source; confirm on the download page). It is small, old (2010) and Finnish, so it can test code and model class but not replace the Nasdaq sample.
- **Binance public data archive.** USD-margined futures `bookTicker` (best bid/ask updates) and `bookDepth` (depth at percentage bands) daily files. https://data.binance.vision/ and https://github.com/binance/binance-public-data `[P]`
  - Free order-book-derived data for event-time modelling practice. Terms (version 1.0, last updated 2026-08-26): CC BY-NC-SA 4.0, non-commercial use only. Not a ten-level book.
- **Bybit historical order book archive.** Described in "How to download and format free historical order book dataset". https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e `[S]`
  - Free daily L2 snapshots; a Bybit API announcement found by search says futures files changed from 500 levels to 200 levels at 100 ms from 2025-08-21. Bybit's own terms of use were not found.
- **Databento.** Usage, pricing and data credits FAQ. https://databento.com/docs/faqs/usage-pricing-and-data-credits `[S]`
  - New accounts get USD 125 of historical credits (expire after six months, one set per team). This is the cheapest legal way to obtain a small real mbp-10 sample; the data stay licensed and may not be redistributed without checking the licence.
- **No free source found** for Nasdaq TotalView ten-level data on the 79 S&P 500 names. LOBSTER's free sample page could not be confirmed in this session (see "Unverified leads").

## Open-source code and tools

- **LeonardoBerti00/TLOB (repo, MIT).** Official TLOB code; also trains MLPLOB, DeepLOB, BiN-CTABL. https://github.com/LeonardoBerti00/TLOB `[S][P]`
  - README states FI-2010 is bundled and a Bitcoin LOB set is downloaded from Kaggle automatically; a ready no-licence starting point.
- **FinancialComputingUCL/LOBFrame (repo).** Pipeline for LOBSTER-format data: processing, training, trading-simulation evaluation. https://github.com/FinancialComputingUCL/LOBFrame `[S][P]`
  - Reference pre-processing for LOBSTER files; no licence file was found on the default branch.
- **peernagy/lob_bench (repo, MIT).** LOB-Bench evaluation code. https://github.com/peernagy/lob_bench `[S][P]`
- **peernagy/LOBS5 (repo, MIT).** Token-level generative LOB model. https://github.com/peernagy/LOBS5 `[P]`
- **X-DataInitiative/tick (repo, BSD-3).** Point-process library with Hawkes estimation. https://github.com/X-DataInitiative/tick `[P]`
  - Standard tool for fitting multivariate Hawkes kernels to level-by-level event streams.
- **zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books (repo).** DeepLOB notebook on FI-2010. https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books `[P]`

## Breakthroughs in related fields that could transfer

- **Mei, Eisner (2016).** The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. arXiv:1612.09328. https://arxiv.org/abs/1612.09328 `[P]`
  - Neural intensity models remove the fixed-kernel assumption; candidate for cross-level excitation.
- **Zuo, Jiang, Li, Zhao, Zha (2020).** Transformer Hawkes Process. arXiv:2002.09291. https://arxiv.org/abs/2002.09291 `[P]`
  - Attention-based point process; scales to long event histories.
- **Arroyo, Cartea, Moreno-Pino, Zohren (2024).** Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers. Quantitative Finance 24(1), 35-57. https://arxiv.org/abs/2306.05479 `[S][P]`
  - Time-to-fill is a survival problem with censoring; time-to-depletion of a level has the same structure, and the paper scores with proper scoring rules.
- **Li, Liu, Liu, Fang, Wang, Xu, Bian (2024).** MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model. arXiv:2409.07486 (ICLR 2025 per the repository README). https://arxiv.org/abs/2409.07486 `[P]`
  - Order-level foundation model; released models could serve as a pretrained prior. Code: https://github.com/microsoft/MarS (MIT).

## Published results to beat

- **TLOB (source: repository README and arXiv abstract).** Reports FI-2010 gains of 3.7 F1 points on average over the prior state of the art, +1.3 (Tesla) and +7.7 (Intel) on NASDAQ data; the arXiv summary also reports a 6.68-point F1 decline in predictability over time. https://github.com/LeonardoBerti00/TLOB
  - These are mid-price trend numbers, not depletion ranking; no published benchmark for ten-level depth-exhaustion ranking was found.
- **LOBCAST (source: arXiv abstract).** All fifteen models show a significant drop on unseen data. https://arxiv.org/abs/2308.01915
  - The honest target is out-of-sample stability across days and stocks, not a higher in-sample score.

## Unverified leads

- LOBSTER free sample files (message and order-book files for a few tickers on 2012-06-21, up to ten levels) at lobsterdata.com: the page was not returned by search and the site is not reachable from this environment. Not confirmed.
- Kolm, Turiel, Westray, "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book" (Mathematical Finance): not confirmed in this session.
- Cont, Stoikov, Talreja, "A stochastic model for order book dynamics" (Operations Research, 2010): not confirmed in this session.
- Stoikov, "The micro-price": not confirmed in this session.
