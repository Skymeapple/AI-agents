# References: S&P 500 Minute Trading Activity Ranking

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page or the repository's raw README/LICENSE). The session's web-search allowance ran out before this project's searches, so most entries here are `[P]`, and the classic intraday-volume forecasting papers (which are not on arXiv) could not be confirmed and are under "Unverified leads". Nothing is cited from memory.

## Theory and established methods

- **Poh, Lim, Zohren, Roberts (2020).** Building Cross-Sectional Systematic Strategies By Learning to Rank. arXiv:2012.07149. https://arxiv.org/abs/2012.07149 `[P]`
  - Shows pairwise and listwise ranking losses beat sorting the outputs of a regression when the goal is a cross-sectional ranking. The task's score is a ranking score, and HFM's current method fits a level regression.
- **Bruch (2019).** An Alternative Cross Entropy Loss for Learning-to-Rank. arXiv:1911.09798. https://arxiv.org/abs/1911.09798 `[P]`
  - Listwise cross-entropy loss with a link to NDCG; implemented in LightGBM as `rank_xendcg`.
- **Pobrotyn, Białobrzeski (2021).** NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting. arXiv:2102.07831. https://arxiv.org/abs/2102.07831 `[P]`
  - Differentiable sorting so a neural model can be trained on the ranking metric itself.
- **Sirignano, Cont (2018).** Universal features of price formation in financial markets: perspectives from Deep Learning. arXiv:1803.06917. https://arxiv.org/abs/1803.06917 `[P]`
  - One model pooled across all US stocks outperforms per-stock models and works on unseen stocks; the case for a single pooled sequence model over 500 names.
- **Zhang, Zhang, Cucuringu, Qian (2022).** Volatility forecasting with machine learning and intraday commonality. arXiv:2202.08962. https://arxiv.org/abs/2202.08962 `[P]`
  - Pooling stocks and adding a market-wide proxy improves intraday realised-volatility forecasts, with clear time-of-day effects. Activity (volume) and volatility share these features, so the same design applies.
- **Antulov-Fantulin, Guo, Lillo (2020).** Temporal mixture ensemble models for intraday volume forecasting in cryptocurrency exchange markets. arXiv:2005.09356. https://arxiv.org/abs/2005.09356 `[P]`
  - Short-horizon intraday volume forecasting with uncertainty estimates; finds machine learning methods beat econometric models, also conditional on volume level.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - The selection-bias logic applies to any score maximised over many trials: HFM scored 98 proposals and the last gains were about one part in a million.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Combinatorially symmetric cross-validation over days gives the probability that the selected model ranks below median out of sample.

## Recent advances and breakthroughs (2023-2026)

- **Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025).** Kronos: A Foundation Model for the Language of Financial Markets. arXiv:2508.02739. https://arxiv.org/abs/2508.02739 `[P]`
  - Pretrained on over 12 billion candlestick records from 45 exchanges with a tokeniser that keeps volume and amount; the model forecasts volume as well as price. A pretrained encoder for one-minute bars that does not need the licensed training set.
- **Ansari et al. (2024).** Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815 `[P]`
  - General pretrained forecaster; a zero-shot baseline for next-30-minute volume.
- **Das, Kong, Sen, Zhou (2023).** A decoder-only foundation model for time-series forecasting (TimesFM). arXiv:2310.10688. https://arxiv.org/abs/2310.10688 `[P]`
  - Second zero-shot baseline.
- **Liu, Hu, Zhang, Wu, Wang, Ma, Long (2023).** iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv:2310.06625. https://arxiv.org/abs/2310.06625 `[P]`
  - Attention across series instead of across time steps; matches a cross-section of 500 stocks.
- **Zhang, Pu, Cucuringu, Dong (2023).** Graph Neural Networks for Forecasting Multivariate Realized Volatility with Spillover Effects. arXiv:2308.01419. https://arxiv.org/abs/2308.01419 `[P]`
  - Finds multi-hop spillovers alone do not help, nonlinear spillovers do, and a quasi-likelihood loss beats mean squared error; guidance on how to use the stock graph HFM's current method keeps fixed.
- **Li, Liu, Shen, Wang, Chen, Huang (2023).** MASTER: Market-Guided Stock Transformer for Stock Price Forecasting. arXiv:2312.15235. https://arxiv.org/abs/2312.15235 `[P]`
  - Cross-stock, cross-time attention gated by market state; an architecture for "one cross-sectional sequence model".
- **Hu, Li, Liu, Zhu, Li, Dai, Xia, Cheng, Jiang (2025).** FinTSB: A Comprehensive and Practical Benchmark for Financial Time Series Forecasting. Frontiers of Computer Science 2026. https://arxiv.org/abs/2502.18834 `[P]`
  - Unified evaluation protocol for financial sequence models.

## Datasets and benchmarks

- **No free, legally redistributable one-minute consolidated-tape history for the full S&P 500 was found.** The project README's statement stands. The partial options below were confirmed.
- **Qlib Yahoo collector, US one-minute mode.** https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo `[P]`
  - Collects one-minute US bars from Yahoo, but the README states "only the last month's data is available in 1min". The yfinance README says Yahoo's API is "intended for personal use only" and refers to Yahoo's terms. A rolling one-month sample for private prototyping, not a redistributable dataset.
- **Alpaca Market Data API (client: alpacahq/alpaca-py, Apache-2.0).** https://github.com/alpacahq/alpaca-py `[P]`
  - README: account sign-up is free and the API gives historical bars for 5000+ stocks. Which feed the free tier uses, how far back it goes, and the redistribution terms were not confirmed; read the data agreement before use.
- **Databento.** Usage, pricing and data credits FAQ. https://databento.com/docs/faqs/usage-pricing-and-data-credits `[S]`
  - USD 125 of free historical credits per new team (expire after six months). A legal way to buy a small licensed sample of US equity bars; not redistributable.
- **Massive flat files (US stocks trades and quotes by day).** https://www.massive.com/docs/flat-files/stocks/quotes/2012/06 `[S]`
  - Full-market daily files from which one-minute bars can be built; a paid, licensed service.
- **FirstRate Data, "Free Tick Data".** https://firstratedata.com/free-tick-data `[S]`
  - A free sample page appeared in search; contents, coverage and terms were not confirmed.
- **Binance public data archive, one-minute klines with volume and number of trades per contract.** https://github.com/binance/binance-public-data `[P]`
  - Free (CC BY-NC-SA 4.0, non-commercial). A different market with no opening or closing auction, but it allows the whole pipeline (cross-sectional sequence model with a ranking loss on a forward volume surprise) to be built and tested before any licence is bought.

## Open-source code and tools

- **microsoft/LightGBM (repo, MIT).** Gradient boosting with `lambdarank` and `rank_xendcg` objectives (documented in docs/Parameters.rst). https://github.com/microsoft/LightGBM `[P]`
  - Fastest way to test whether a ranking objective alone improves the score with the existing features.
- **allegro/allRank (repo, Apache-2.0).** PyTorch learning-to-rank framework with pointwise, pairwise and listwise losses and Transformer scoring functions. https://github.com/allegro/allRank `[P]`
- **shiyu-coder/Kronos (repo, MIT).** Pretrained candlestick models and fine-tuning scripts. https://github.com/shiyu-coder/Kronos `[P]`
- **amazon-science/chronos-forecasting (repo, Apache-2.0).** Chronos models. https://github.com/amazon-science/chronos-forecasting `[P]`
- **google-research/timesfm (repo, Apache-2.0).** TimesFM models. https://github.com/google-research/timesfm `[P]`
- **esvhd/pypbo (repo, AGPL-3.0).** Probability of backtest overfitting. https://github.com/esvhd/pypbo `[P]`

## Breakthroughs in related fields that could transfer

- **Nie, Nguyen, Sinthong, Kalagnanam (2022).** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). arXiv:2211.14730. https://arxiv.org/abs/2211.14730 `[P]`
  - Patching makes long minute-level histories cheap to encode; channel-independent training matches pooled per-stock modelling.
- **Moreno-Pino, Zohren (2022).** DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions. arXiv:2210.04797. https://arxiv.org/abs/2210.04797 `[P]`
  - Learns directly from raw high-frequency bars instead of hand-made aggregates; the README says hand-made features are saturated.
- **Cont, Cucuringu, Zhang (2021).** Cross-Impact of Order Flow Imbalance in Equity Markets. arXiv:2112.13213. https://arxiv.org/abs/2112.13213 `[P]`
  - Lagged cross-asset effects exist at short horizons and decay fast; supports cross-stock inputs with short lags.
- **Glasserman, Lin (2023).** Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis. arXiv:2309.17322. https://arxiv.org/abs/2309.17322 `[P]`
  - Pretrained models may have seen the test period. Any foundation model used here must be evaluated on days after its training cut-off.

## Published results to beat

- No published result for this target (cross-sectional ranking of next-30-minute versus last-30-minute volume for S&P 500 members) was found in this session.
- **HFM track (source: project README).** `instance_score_mean` 0.4625 against a baseline of 0.1674, at about USD 9,488 of compute; last gains about one part in a million.
- **Poh, Lim, Zohren, Roberts (source: arXiv abstract).** Learning-to-rank gives roughly a threefold Sharpe improvement over regress-then-sort for cross-sectional momentum. https://arxiv.org/abs/2012.07149
  - A different target; it indicates the size of gain a ranking loss can give over a regression read-out.

## Unverified leads

- Classic intraday-volume models: Bialkowski, Darolles, Le Fol, "Improving VWAP strategies: A dynamic volume approach"; Brownlees, Cipollini, Gallo, "Intra-daily Volume Modeling and Prediction for Algorithmic Trading"; Chen, Feng, Palomar, "Forecasting Intraday Trading Volume: A Kalman Filter Approach". Not confirmed in this session.
- Goyenko, Kelly, Moskowitz, Su, Zhang, "Trading Volume Alpha" (machine-learning prediction of trading volume): not confirmed in this session.
- A free AWS Marketplace listing of 2020 intraday tick and one-minute bars for five large technology stocks appeared in a search summary; the listing title and terms were not confirmed.
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview": not confirmed in this session.
