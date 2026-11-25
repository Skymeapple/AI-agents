# References: Toronto Arterial Signal Timing Coordination

Verification: every entry was confirmed in this session, either by a web search result or by fetching the arXiv record. Items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Wei, Zheng, Gayah, Li (2019).** A Survey on Traffic Signal Control Methods. arXiv:1904.08117. https://arxiv.org/abs/1904.08117
  - Summarises Webster, MAXBAND (Little et al. 1981, MILP for two-way bandwidth with a common cycle) and MULTIBAND (Gartner et al. 1991, link-specific bands weighted by volume); the quickest entry point to the classical formulations this task uses.
- **Cabezas, García (2018).** A Heuristic Algorithm for Traffic Light Synchronization Based on the MAXBAND Model. arXiv:1805.03982. https://arxiv.org/abs/1805.03982
  - Restates the MAXBAND MILP and gives a heuristic for it; useful as a compact formulation to code an offset search against.
- **Varaiya (2013).** The Max-Pressure Controller for Arbitrary Networks of Signalized Intersections. Springer book chapter. https://link.springer.com/10.1007/978-1-4614-6243-9_2
  - Source of max-pressure control: stage choice from local queues, stable without demand knowledge but needs turn ratios.
- **Kouvelas, Lioris, Fayazi, Varaiya (2014).** Maximum Pressure Controller for Stabilizing Queues in Signalized Arterial Networks. Transportation Research Record. https://www.research-collection.ethz.ch/bitstream/20.500.11850/275877/4/Kouvelas_EtAl_TRR_2014.pdf
  - Applies max pressure to an arterial; a non-coordinated baseline to compare a fixed coordinated plan against.
- **Daganzo (1994).** The Cell Transmission Model: A Dynamic Representation of Highway Traffic Consistent with the Hydrodynamic Theory. Transportation Research Part B 28(4), 269-287. https://doi.org/10.1016/0191-2615(94)90002-7
  - The queue-and-spillback model named in the README; Part II (network traffic, 1995, Transportation Research Part B 29(2), 79-93) is listed at https://its.berkeley.edu/publications/cell-transmission-model-part-ii-network-traffic.
- **Ouyang, Zhang, Lavaei, Varaiya (2019).** Large-Scale Traffic Signal Offset Optimization. arXiv:1911.08368; IEEE Transactions on Control of Network Systems 2020. https://arxiv.org/abs/1911.08368
  - Offset optimisation posed as a tractable optimisation problem at network scale; an alternative to query-limited local offset search.
- **Arizona State University research record (year not captured).** Robust signal timing for arterials under day-to-day demand variations. https://asu.elsevierpure.com/en/publications/robust-signal-timing-for-arterials-under-day-to-day-demand-variat/
  - Scenario-based stochastic programme for pretimed arterial signals with probability-weighted demand scenarios; this is the README's "optimise for a distribution of demand" route, already formulated.
- **Caltrans research report (2010).** Robust signal timing report, f0017242 PRR-2010-07. https://dot.ca.gov/-/media/dot-media/programs/research-innovation-system-information/documents/f0017242-prr-2010-07-final-report.pdf
  - Jointly sets cycle, splits, phase sequence and offsets to minimise expected delay in the high-consequence demand scenarios; solved with a simulation-based genetic algorithm on two networks.
- **University of Minnesota CTS (year not captured).** SMART-Signal Phase II: Arterial Offset Optimization Using Archived High-Resolution Traffic Signal Data. https://www.cts.umn.edu/publications/report/smart-signal-phase-ii-arterial-offset-optimization-using-archived-high-resolution-traffic-signal-data
  - Offset optimisation over a series of demand scenarios, handling early return to green and uncertain queues.

## Recent advances and breakthroughs (2023-2026)

- **Maheshwari, Yang, Pazzi (2025).** Traffic Intersection Simulation Using Turning Movement Count Data in SUMO: A Case Study of Toronto Intersections. arXiv:2508.10733. https://arxiv.org/abs/2508.10733
  - Builds SUMO intersection simulations directly from City of Toronto turning-movement counts and checks simulated against counted flows; this is most of the missing "traffic model calibrated to Toronto counts".
- **Authors not captured (2026).** Explicit public and pairwise bandwidth optimization for arterial traffic coordination (PP-Band). Transportation Research Part E, vol. 214. https://trid.trb.org/View/2725265
  - Newer band model that optimises a global band and pairwise local bands together; a candidate replacement for MULTIBAND-style scoring.
- **Friedman, Mark, Gartner (2024).** Uninterrupted Maximum Flow on Signalized Traffic Networks. arXiv:2404.16592. https://arxiv.org/abs/2404.16592
  - Two-way arterial progression at a recommended speed with a common cycle; gives conditions under which every signal can be made in both directions.
- **Tsitsokas, Kouvelas, Geroliminis (2023).** Two-layer adaptive signal control framework for large-scale dynamically-congested networks: Combining efficient Max Pressure with Perimeter Control. Transportation Research Part C 152, 104128. https://arxiv.org/abs/2210.10453
  - Shows how max pressure is combined with a higher-level controller under congestion; relevant when testing plans at +20% demand.
- **USDOT ROSA P report (2025).** Automatic Signal Retiming for Large Scale Networks with Vehicle Trajectory Data. https://rosap.ntl.bts.gov/view/dot/89246
  - Michigan deployment that designs timing plans from vehicle trajectory data instead of turning-movement counts.
- **Wang, Yu, Yang (2026).** Fostering Data Collaboration in Digital Transportation Marketplaces: The Role of Privacy-Preserving Mechanisms. arXiv:2602.01804. https://arxiv.org/abs/2602.01804
  - Uses Webster's formula plus band-based offset optimisation as the downstream task; a recent worked example of that pipeline.
- **Lai, Xu, Zhang, Liu et al. (2023).** LLMLight: Large Language Models as Traffic Signal Control Agents. arXiv:2312.16044. https://arxiv.org/abs/2312.16044
  - Language-model signal control; background on what agent-driven control looks like outside fixed-time plans.

## Datasets and benchmarks

- **City of Toronto Open Data.** Traffic Volumes - Multimodal Intersection Turning Movement Counts. https://open.toronto.ca/dataset/traffic-volumes-at-intersections-for-all-modes/
  - The counts behind the task: vehicles, pedestrians and cyclists per movement over 8- or 14-hour count days, 1984 to present; many intersections have repeat counts, which gives an empirical demand distribution around the counted hour.
- **City of Toronto (2025).** News release on new transportation datasets. https://www.toronto.ca/news/city-of-toronto-publishes-new-transportation-datasets-that-highlight-travel-trends-and-support-traffic-management-actions/
  - Confirms the January 2025 refresh of the count data and the companion speed, volume and classification counts (one-day to one-week counts since 1993), which can calibrate link speeds.

## Open-source code and tools

- **Mei, Lei, Da, Shi et al. (2022).** LibSignal: An Open Library for Traffic Signal Control. arXiv:2211.10649. https://arxiv.org/abs/2211.10649
  - Runs the same controller on SUMO and CityFlow with published baseline results; a ready evaluation harness.
- **RESCO.** Reinforcement Learning Benchmarks for Traffic Signal Control (SUMO). https://github.com/UnivOfAlabama-BittleResearchGroup/RESCO
  - Ships fixed-time, greedy and max-pressure baselines on SUMO networks, including arterial corridors.
- **Zhang, Feng, Liu, Ding et al. (2019).** CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario. arXiv:1905.05217. https://arxiv.org/abs/1905.05217
  - Fast mesoscopic simulator suitable for thousands of plan evaluations under sampled demand.

## Breakthroughs in related fields that could transfer

- **Wei, Xu, Zhang, Zheng et al. (2019).** CoLight: Learning Network-level Cooperation for Traffic Signal Control. arXiv:1905.05717. https://arxiv.org/abs/1905.05717
  - Learned coordination between neighbouring signals; a comparison point for whether fixed offsets leave value on the table under variable demand.

## Published results to beat

- University of Arizona dissertation (2025), https://repository.arizona.edu/handle/10150/678281: trajectory-based offset coordination reduced travel time by 17.17% and stop delay by 46.85% against volume-based retiming (as reported in the search result for that record). These are field-style gains, not the HFM metric; the HFM incumbent is 0.9892 (README).

## Unverified leads

- Little, Kelson, Gartner (1981) MAXBAND and Gartner, Assmann, Lasaga, Hou (1991) MULTIBAND original papers: cited by the survey above, original records not opened.
- Webster (1958) delay formula and the Highway Capacity Manual delay model: not confirmed in session.
- City of Toronto "Traffic Signals Tabular" open dataset (signal locations): named in the README, not confirmed in session.
- SUMO simulator home page (eclipse.dev/sumo): not confirmed in session.
