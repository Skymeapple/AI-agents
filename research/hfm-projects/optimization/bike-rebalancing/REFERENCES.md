# References: Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing

Verification: every entry was confirmed in this session, either by a web search result or by fetching the arXiv record. Items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Raviv, Kolka (2013).** Optimal inventory management of a bike-sharing station. IIE Transactions 45(10), 1077-1093. https://ideas.repec.org/a/taf/uiiexx/v45y2013i10p1077-1093.html
  - Defines the user dissatisfaction function (expected failed rentals and returns as a convex function of starting stock); the per-station cost a look-ahead should use in place of projected stock.
- **Freund, Henderson, Shmoys (2022).** Minimizing Multimodular Functions and Allocating Capacity in Bike-Sharing Systems. Operations Research 70(5), 2715-2731. https://arxiv.org/abs/1611.09304
  - Extends the dissatisfaction function to system-wide dock and bike allocation; gives the target inventories a van policy should steer towards.
- **Freund, Henderson, O'Mahony, Shmoys (2019).** Analytics and Bikes: Riding Tandem with Motivate to Improve Mobility. Interfaces 49(5), 310-323. https://pubsonline.informs.org/stoken/default+domain/EC-25/full/10.1287/inte.2019.1005
  - Describes the Citi Bike practice, including de-censoring demand at empty and full stations; trip files undercount demand exactly when stations fail.
- **Brinkmann, Ulmer, Mattfeld (2019).** Dynamic Lookahead Policies for Stochastic-Dynamic Inventory Routing in Bike Sharing Systems. Computers & Operations Research 106, 260-279. https://dblp.uni-trier.de/rec/journals/cor/BrinkmannUM19.html
  - The closest published match to this task: an MDP for one van with look-ahead horizons tuned by time of day.
- **Brinkmann, Ulmer, Mattfeld (2020).** The multi-vehicle stochastic-dynamic inventory routing problem for bike sharing systems. Business Research. https://link.springer.com/article/10.1007/s40685-019-0100-z
  - Extends the look-ahead to several vans and moves anticipation to an analytical offline step to keep decisions fast.
- **Lin (2018).** A Comprehensive Framework for Dynamic Bike Rebalancing in a Large Bike Sharing Network. arXiv:1806.02890. https://arxiv.org/abs/1806.02890
  - Pipeline of station-level pick-up and drop-off prediction, station clustering and capacitated location-routing; a template for "forecast feeds routing solver".

## Recent advances and breakthroughs (2023-2026)

- **Liang, Liu, Jena, Lodi et al. (2024, revised 2025).** Single- vs. Dual-Policy Reinforcement Learning for Dynamic Bike Rebalancing. arXiv:2402.03589. https://arxiv.org/abs/2402.03589
  - Multi-van continuous-time MDP without synchronised decisions, with a simulator driven by time and weather; the same event structure as this replay.
- **Liang, Liu, Jena, Lodi et al. (2024).** Dual Policy Reinforcement Learning for Real-time Rebalancing in Bike-sharing Systems. arXiv:2406.00868. https://arxiv.org/abs/2406.00868
  - Separates the "how many bikes" decision from the "which station next" decision; maps directly onto the two outputs this task's policy must give.
- **Gammelli, Wang, Prak, Rodrigues et al. (2021).** Predictive and Prescriptive Performance of Bike-Sharing Demand Forecasts for Inventory Management. arXiv:2108.00858. https://arxiv.org/abs/2108.00858
  - Feeds a probabilistic (variational Poisson RNN) forecast of pick-up and return rates into the dissatisfaction function on New York trip data, and shows forecast accuracy and decision quality can diverge; directly addresses the README finding that the forecast is the limit.
- **Behroozi, Edrisi (2024).** Predicting travel demand of a bike sharing system using graph convolutional neural networks. arXiv:2408.09317. https://arxiv.org/abs/2408.09317
  - Station-level demand model with weather and access features (Chicago data); a stronger forecaster than "same weekday last week".
- **Scarpel, Pettena, Cederle, Chiariotti et al. (2026).** Fully Dynamic Rebalancing in Dockless Bike-Sharing Systems via Deep Reinforcement Learning. arXiv:2605.14501. https://arxiv.org/abs/2605.14501
  - Real-time single-truck routing with pick-up and drop-off actions on real data; most recent learned policy for the same decision loop.
- **Sreevatsa K, Raveendran, Mani, Shigli et al. (2025).** SmartFlow: Reinforcement Learning and Agentic AI for Bike-Sharing Optimisation. arXiv:2601.00868. https://arxiv.org/abs/2601.00868
  - Gymnasium-style environment for station-to-station transfers; shows one way to wrap a replay as an RL environment.

## Datasets and benchmarks

- **City Bike Finland / HSL.** City bike stations' Origin-Destination (OD) data. https://www.avoindata.fi/data/en/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat
  - The trip data used in `bike_sim.py`; monthly CSVs from 2016 onward under CC BY 4.0, so more seasons than June 2021 are available for training a forecaster.
- **HSL.** Helsinki Region Transport's (HSL) city bicycle stations. https://hri.fi/data/en/dataset/hsl-n-kaupunkipyoraasemat
  - Station list in CSV, GeoJSON, SHP and API form; check its attribute table for dock capacity, which the current experiment assumes (the search result did not show the field list).
- **Citi Bike.** System Data (trip histories) and real-time GBFS feed. https://citibikenyc.com/system-data
  - Second city for out-of-sample tests; the GBFS `station_information` feed carries per-station `capacity`, so dock counts need not be assumed.
- **MobilityData / NABSA.** General Bikeshare Feed Specification (GBFS). https://github.com/MobilityData/gbfs/blob/v2.3/gbfs.md
  - Defines `station_information` (capacity, location) and `station_status` (live bikes and docks); the standard way to obtain real capacities and stock levels for any operator.

## Open-source code and tools

- **Fernández, Billhardt, Ossowski, Sánchez (2020).** Bike3S: A Tool for Bike Sharing Systems Simulation. Journal of Simulation 14(4); arXiv:2402.16871. https://arxiv.org/abs/2402.16871
  - Agent-based station simulator built to test balancing strategies and station capacities; a cross-check for the in-house replay.
- **Geometrein.** helsinki-city-bikes (GitHub repository). https://www.github.com/Geometrein/helsinki-city-bikes
  - Existing analysis of the same Helsinki trip data; a source of cleaned station metadata.
- **Coretti Sánchez, Martinez, Alonso Pastor, Larson (2022).** Simulation study on the fleet performance of shared autonomous bicycles. Communications in Transportation Research 2. https://arxiv.org/abs/2106.09694
  - Reviews the open-source bike-share simulators that exist and what each was built for.

## Breakthroughs in related fields that could transfer

- **Jungel, Parmentier, Schiffer, Vidal (2023).** Learning-based Online Optimization for Autonomous Mobility-on-Demand Fleet Control. arXiv:2302.03963. https://arxiv.org/abs/2302.03963
  - Trains an online dispatch and rebalancing policy to imitate full-information optimal solutions; the README already computes a "real future" policy that is 11% better on every day, which is the training signal this method needs.
- **Ansari, Stella, Turkmen, Zhang et al. (2024).** Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
  - Pretrained probabilistic forecaster that outputs sample paths without task-specific training; a cheap source of sampled demand scenarios per station.
- **Das, Kong, Sen, Zhou (2023).** A decoder-only foundation model for time-series forecasting. arXiv:2310.10688. https://arxiv.org/abs/2310.10688
  - Second pretrained forecaster to compare against the two-hour projection.
- **Bertsekas (2019).** Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120. https://arxiv.org/abs/1910.00120
  - One-van-at-a-time rollout with a guarantee of improving on the base policy; a principled way to coordinate four vans without a joint search.
- **Pan, Cai, Fang, Tang et al. (2018).** A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems. arXiv:1802.04592. https://arxiv.org/abs/1802.04592
  - Rebalancing through user incentives; relevant if the cost model later allows demand shifting.

## Published results to beat

- Liang et al., arXiv:2402.03589 (v1 abstract as returned by search): lost demand reduced by up to 27.12% on average against their baselines, on data generated from historical trips.
- This repository's own replay (README): forecast policy 5,235 failures per day; perfect-foresight look-ahead 4,652 (−11.1%).

## Unverified leads

- Schuijbroek, Hampshire, van Hoeve (2017), "Inventory rebalancing and vehicle routing in bike sharing systems", EJOR: not confirmed in session.
- O'Mahony, Shmoys (2015), "Data analysis and optimization for (Citi)Bike sharing", AAAI: cited by Freund et al., record not opened.
- Ghosh, Varakantham, Adulyasak, Jaillet (2017), "Dynamic repositioning to reduce lost demand in bike sharing systems", JAIR: not confirmed in session.
- CIRRELT-2022-28 (Liang, Jena, Lodi), minute-level bike-share simulator on BIXI Montreal data: seen in a search result, title not captured.
- BIXI Montreal and Capital Bikeshare open trip data: not confirmed in session.
