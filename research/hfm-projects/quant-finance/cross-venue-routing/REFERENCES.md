# References: Cross-Venue Allocation of Perpetual Orders

Checked 2026-11-25. `[S]` = existence, title and URL confirmed by a web search result in this session. `[P]` = confirmed by fetching the primary page (arXiv abstract page, the repository's raw README/LICENSE, or the data host). Nothing is cited from memory; items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Cont, Kukanov (2017).** Optimal order placement in limit order markets. Quantitative Finance 17(1), 21-39. https://arxiv.org/abs/1210.1625 `[S][P]`
  - The reference formulation for this task: allocation across venues as a convex problem in queue sizes, fees and fill risk, solved with a stochastic algorithm that learns from recent fills without an explicit order-flow model. This is the "small convex allocation per order" the README proposes.
- **Laruelle, Lehalle, Pagès (2009).** Optimal split of orders across liquidity pools: a stochastic algorithm approach. arXiv:0910.1166. https://arxiv.org/abs/0910.1166 `[P]`
  - Stochastic-approximation rule for splitting an order across pools with convergence results; an online alternative to refitting trees.
- **Ganchev, Kearns, Nevmyvaka, Vaughan (2009).** Censored Exploration and the Dark Pool Problem. UAI 2009. https://arxiv.org/abs/1205.2646 `[P]`
  - Allocation when only filled quantity is observed (censored liquidity); gives a learning algorithm with guarantees.
- **Baldacci, Manziuk (2020).** Adaptive trading strategies across liquidity pools. arXiv:2008.07807. https://arxiv.org/abs/2008.07807 `[P]`
  - Bayesian updating of venue parameters inside an optimal control problem across several venues.
- **Almgren, Chriss (2001).** Optimal execution of portfolio transactions. Journal of Risk. https://www.doi.org/10.21314/JOR.2001.041 `[S]`
  - Cost model (temporary and permanent impact) that each venue's cost function should reduce to in the single-venue case.
- **Makarov, Schoar (2020).** Trading and arbitrage in cryptocurrency markets. Journal of Financial Economics 135(2), 293-319. https://econpapers.repec.org/RePEc:eee:jfinec:v:135:y:2020:i:2:p:293-319 `[S]`
  - Decomposes signed volume into a common and a venue-specific component; search summaries report the common component explains about 80% of bitcoin returns (85% in the working paper) and the venue-specific part explains price gaps between exchanges.
- **Cont, Kukanov, Stoikov (2014).** The Price Impact of Order Book Events. Journal of Financial Econometrics 12(1), 47-88. https://arxiv.org/abs/1011.6402 `[S][P]`
  - Impact slope is inversely proportional to depth; gives a per-venue cost term that can be estimated from public book data.
- **Bailey, López de Prado (2014).** The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5). https://papers.ssrn.com/abstract=2460551 `[S]`
  - HFM accepted 12 of 33 proposals on one replay with last gains near one part in 100,000; such gains need a selection-bias correction before they count.
- **Bailey, Borwein, López de Prado, Zhu (2017).** The Probability of Backtest Overfitting. Journal of Computational Finance. https://papers.ssrn.com/abstract=2326253 `[S]`
  - Combinatorially symmetric cross-validation over trading days estimates how often the selected router is below median out of sample.

## Recent advances and breakthroughs (2023-2026)

- **Zheng, He, Yang (2023).** Optimal Execution Using Reinforcement Learning. arXiv:2306.17178. https://arxiv.org/abs/2306.17178 `[S][P]`
  - Aligns data from several crypto exchanges (a search summary names Binance, OKX and Bybit) and shows cross-exchange signals add information for execution; the closest published setting to this task.
- **Bieganowski, Ślepaczuk (2026).** Explainable Patterns in Cryptocurrency Microstructure. arXiv:2602.00776. https://arxiv.org/abs/2602.00776 `[S][P]`
  - Binance perpetual books and trades at one-second frequency, 2022-01-01 to 2025-10-12; feature importance and effect shapes (OFI, spread, adverse selection) are stable across assets, and results are checked with a top-of-book taker backtest.
- **Cho, Norman, Nunes (2023).** PRIME: A Price-Reverting Impact Model of a cryptocurrency Exchange. arXiv:2305.07559. https://arxiv.org/abs/2305.07559 `[S][P]`
  - Measures market impact on BTC/USD futures and builds a simulator that reproduces it; source for an impact term in the venue cost model.
- **Fabre, Ragel (2023).** Interpretable ML for High-Frequency Execution. arXiv:2307.04863. https://arxiv.org/abs/2307.04863 `[S][P]`
  - Fill-probability model estimated on crypto centralised exchanges and on equities; fill probability differs by venue and tick size, which is an input to the allocation.
- **He, Manela, Ross, von Wachter (2022, revised 2024).** Fundamentals of Perpetual Futures. arXiv:2212.06888. https://arxiv.org/abs/2212.06888 `[S][P]`
  - A search summary reports that the price impact of perpetual taker imbalance is about 0.094 bp per percentage point when the futures-spot gap is large against 0.031 when it is small; impact is state-dependent, so venue costs should be too.

## Datasets and benchmarks

- **Binance public data archive.** USD-margined futures daily files: `aggTrades`, `trades`, `klines`, `bookTicker`, `bookDepth`, `metrics`, `fundingRate`. https://data.binance.vision/ ; documentation https://github.com/binance/binance-public-data `[S][P]`
  - Free. Terms ("Binance Vision Dataset Terms" v1.0, last updated 2026-08-26, in the repository's TERMS_AND_CONDITIONS.md): CC BY-NC-SA 4.0, non-commercial; derivative works must carry attribution and the same licence. `bookDepth` gives depth at percentage bands, not a level-by-level book.
- **Bybit historical data archive.** Trades and L2 order book files. Tutorial: https://medium.com/@lu.battistoni/how-to-download-and-format-free-historical-order-book-dataset-16b3a84a8e0e ; downloader https://github.com/aoki-h-jp/bybit-bulk-downloader (MIT) `[S][P]`
  - Free and updated daily per the tutorial. A Bybit API announcement found by search says futures order book files are 200 levels at 100 ms from 2025-08-21 (500 levels before). Bybit's own terms of use for the archive were not found.
- **OKX.** No official bulk archive of historical order books was found. A client-library documentation page states that the REST order book endpoint returns the current snapshot only (max 400 levels per side), that recent trades cover several months by pagination, and that `/public/funding-rate-history` returns past funding. https://mintlify.com/tiagosiebler/okx-api/rest-client/market-data `[S]`
  - Plainly: OKX historical depth is the missing leg. Trades and funding are available free; books need forward collection or a third party.
- **Tardis.dev.** Historical data details for OKX swaps and Bybit. https://docs.tardis.dev/historical-data-details/okex-swap and https://docs.tardis.dev/historical-data-details/bybit `[S]`
  - CSV datasets for the first day of each month can be downloaded without an API key; the rest is paid. This gives one synchronised day per month across venues from a single source with consistent timestamps, enough to compute the hindsight bound the README asks for.
- **Albers, Cucuringu, Howison, Shestopaloff (2021).** Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets. arXiv:2108.09750. https://arxiv.org/abs/2108.09750 `[S][P]`
  - Multi-venue order book and trade study; a methodological benchmark for leader-lagger structure between venues.

## Open-source code and tools

- **nkaz001/hftbacktest (repo, MIT).** Replay backtester with feed latency, order latency and queue-position fill models; examples for Binance Futures and live support for Binance Futures and Bybit. https://github.com/nkaz001/hftbacktest `[P]`
  - Latency differs by venue and decides whether a routed order reaches the quote in time.
- **bmoscon/cryptofeed (repo).** Normalised websocket feed handler for many exchanges including Binance Futures and Bybit. https://github.com/bmoscon/cryptofeed `[P]`
  - For forward collection of synchronised books where no archive exists (OKX).
- **ccxt/ccxt (repo, MIT).** Unified exchange API. https://github.com/ccxt/ccxt `[P]`
- **tardis-dev/tardis-node (repo, MPL-2.0).** Client for Tardis historical and real-time normalised data. https://github.com/tardis-dev/tardis-node `[P]`
- **cvxpy/cvxpy (repo, Apache-2.0).** Convex optimisation modelling. https://github.com/cvxpy/cvxpy `[P]`
  - Solves the per-order allocation as a small convex programme.
- **nautechsystems/nautilus_trader (repo, LGPL).** Event-driven backtester with multi-venue support. https://github.com/nautechsystems/nautilus_trader `[P]`

## Breakthroughs in related fields that could transfer

- **Arroyo, Cartea, Moreno-Pino, Zohren (2024).** Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers. Quantitative Finance 24(1), 35-57. https://arxiv.org/abs/2306.05479 `[S][P]`
  - Per-venue fill-time distributions are the missing input to a passive-versus-aggressive allocation.
- **Cartea, Duran-Martin, Sánchez-Betancourt (2023).** Detecting Toxic Flow. arXiv:2312.05827. https://arxiv.org/abs/2312.05827 `[P]`
  - Online adverse-selection prediction; gives the "adverse selection" term in the venue cost model named in the README.
- **Cont, Cucuringu, Zhang (2021).** Cross-Impact of Order Flow Imbalance in Equity Markets. arXiv:2112.13213. https://arxiv.org/abs/2112.13213 `[P]`
  - Finds lagged cross-asset OFI helps forecasting at short horizons and decays fast; the same test applies to cross-venue OFI for one contract.

## Published results to beat

- **Albers et al. (source: arXiv abstract and search summary).** Linear models explain 10% to 37% of the variation in 500 ms future returns depending on the target market; a search summary reports a very low R² (0.0061) for the imbalance measure on Binance futures. https://arxiv.org/abs/2108.09750
  - Sets expectations: short-horizon predictability is real but venue-dependent, and weakest on the most liquid venue.
- **HFM track (source: project README).** Best reward 0.5927; last accepted gains about one part in 100,000. No published paper reports a comparable three-venue routing cost; the hindsight-best venue per order is the upper bound to compute first.

## Unverified leads

- An official OKX historical-data download page and Bybit's public archive host (public.bybit.com): neither was returned by search with terms of use, and neither host is reachable from this environment. Check both by hand before building the replay.
- `okx-dump` Python package on PyPI (appeared as a search hit, content not confirmed).
- Almgren, Thum, Hauptmann, Li, "Direct estimation of equity market impact": not confirmed in this session.
