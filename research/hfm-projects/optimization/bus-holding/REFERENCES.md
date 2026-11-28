# References: Control Policy for Frequent Bus Services

Verification: every entry was confirmed in this session, either by a web search result or by fetching the arXiv record. Items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Daganzo (2009).** A headway-based approach to eliminate bus bunching: Systematic analysis and comparisons. Transportation Research Part B 43(10), 913-921. https://econpapers.repec.org/RePEc:eee:transb:v:43:y:2009:i:10:p:913-921
  - The forward-headway rule tested in `bus_sim.py`; holds at control points from real-time headways with less slack than schedule holding.
- **Xuan, Argote, Daganzo (2011).** Dynamic bus holding strategies for schedule reliability: Optimal linear control and performance analysis. Transportation Research Part B 45(10), 1831-1845. https://www.ocf.berkeley.edu/~xuanyg/doc/Xuan_Argote_Daganzo_2011_TR-B.pdf
  - Shows headway-only control cannot keep a timetable and gives a linear rule on virtual-schedule deviations needing about 40% less slack than schedule holding; relevant because the HSL trunk lines are timetabled with timing points.
- **Bartholdi, Eisenstein (2012).** A self-coordinating bus route to resist bus bunching. Transportation Research Part B 46(4), 481-491. https://ideas.repec.org/a/eee/transb/v46y2012i4p481-491.html
  - Backward-headway holding at a terminal with no schedule; tested on an Atlanta route. Holding at the terminal delays no on-board passengers, which matters given the README result.
- **Berrebi, Watkins, Laval (2015).** A real-time bus dispatching policy to minimize passenger wait on a high frequency route. Transportation Research Part B 81, 377-389. https://ideas.repec.org/a/eee/transb/v81y2015ip2p377-389.html
  - Dispatching as a stochastic decision problem solved by backward induction, using predicted arrivals of following buses.
- **Berrebi et al. (2018).** Comparison of holding methods on Tri-Met Route 72, Portland (title not captured). HAL hal-01825861. https://hal.archives-ouvertes.fr/hal-01825861/file/doc00027579.pdf
  - Finds prediction-based holding gives the best trade-off between holding time and headway stability, depending on prediction accuracy; the same trade-off that made the classic rules lose in the README experiment.
- **Rezazada, Nassir, Tanin, Ceder (2024).** Bus bunching: a comprehensive review from demand, supply, and decision-making perspectives. Transport Reviews 44(4), 766-790. https://ideas.repec.org/a/taf/transr/v44y2024i4p766-790.html
  - Current review; notes that hybrid methods (holding plus other controls) beat holding alone and that evaluations are rarely realistic.

## Recent advances and breakthroughs (2023-2026)

- **Shen, Bi, Gu, Xu et al. (2026).** From Observation to Intervention: Mastering Bus Holding Control from Historical Uncontrolled Trajectories with Offline Reinforcement Learning. arXiv:2610.14632. https://arxiv.org/abs/2610.14632
  - Trains holding policies only from logged GPS data of uncontrolled operations, synthesising holding actions that were never taken; this is the method that fits HSL's recorded vehicle positions.
- **Zhang, Zhang, Zheng (2026).** Completion-Aware Cross-Fidelity Offline-to-Online Reinforcement Learning for Multi-Line Bus Holding. arXiv:2609.39868. https://arxiv.org/abs/2609.39868
  - Multi-line holding that combines logged data with a cheaper simulator and corrects for the gap between them; applies to five routes sharing a replay.
- **Zhen, Gu, Shen, Fan et al. (2026).** Stability and Optimal Design of Adaptive Linear Bus Holding on Open Chains and Loops. arXiv:2610.18253. https://arxiv.org/abs/2610.18253
  - Exact stability conditions and closed-form Pareto-optimal linear rules for the schedule-versus-headway trade-off on finite routes; gives the best linear rule to test before any learning.
- **Dong, Gayah (2026).** Mitigating Bus Bunching with Reinforcement Learning Enhanced by Semantic Stop Embedding. arXiv:2608.10207. https://arxiv.org/abs/2608.10207
  - Deep Q-learning holding with offline LLM-built stop embeddings, calibrated on two real routes and compared with a tuned Daganzo rule.
- **Zhang, Zheng, Zhang (2025).** Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control. arXiv:2508.20784. https://arxiv.org/abs/2508.20784
  - Single-agent holding on a bidirectional timetabled network with varying demand and fleet size, closer to the HSL setting than loop-line studies.
- **Shen, Gu, Cassidy, Lin et al. (2024).** A vicious cycle along busy bus corridors and how to abate it. arXiv:2403.08230. https://arxiv.org/abs/2403.08230
  - Analyses bunching where several lines share a corridor, as the HSL trunk lines partly do.
- **Nie, Ou, Zhang, Lu et al. (2023).** A Robust Integrated Multi-Strategy Bus Control System via Deep Reinforcement Learning. arXiv:2308.08179. https://arxiv.org/abs/2308.08179
  - Combines holding with other control actions in one policy; relevant because holding alone gained under 1% here.
- **Wang, Sun (2021).** Reducing Bus Bunching with Asynchronous Multi-Agent Reinforcement Learning. arXiv:2105.00376. https://arxiv.org/abs/2105.00376
  - Standard event-driven formulation where a bus acts only on reaching a stop; the baseline most later papers compare with.
- **Wang, Sun (2022).** Robust Dynamic Bus Control: A Distributional Multi-agent Reinforcement Learning Approach. IEEE Transactions on Intelligent Transportation Systems. https://arxiv.org/abs/2111.01946
  - Distributional critic for robustness to running-time and demand uncertainty.
- **Wang (2022).** Integrating Conventional Headway Control with Reinforcement Learning to Avoid Bus Bunching. arXiv:2210.00201. https://arxiv.org/abs/2210.00201
  - Uses the two-way headway rule as a prior inside PPO; a low-risk way to add learning to an existing rule.

## Datasets and benchmarks

- **HSL / Digitransit.** High-frequency positioning (HFP) API documentation. https://digitransit.fi/en/developers/apis/5-realtime-api/vehicle-positions/high-frequency-positioning/
  - Live vehicle events for all HSL vehicles over MQTT (`/hfp/v2/journey/#` on mqtt.hsl.fi); the source of real running times and dwell events that the current experiment assumes.
- **HSLdevcom.** transitlog-ui (Reittiloki) and hfp-analytics. https://github.com/HSLdevcom/transitlog-ui and https://github.com/HSLdevcom/hfp-analytics
  - HSL's tools over recorded HFP history; the search found no official public bulk download, and hfp-analytics states its API is not public, so plan to record the feed or request an extract.
- **datarttu.** hfplogger. https://github.com/datarttu/hfplogger
  - Records the HFP v2 feed to compressed CSV and a database; the practical way to build a replay archive for routes 500, 510, 560, 570 and 200.
- **Mendeley Data.** Melbourne GTFS-Realtime bus bunching dataset. https://data.mendeley.com/datasets/4dn7drrtgc/1
  - About five months of real-time telemetry from 30 high-frequency routes with headway-prediction code; a ready open set for testing forecasts before HFP is collected.
- **TransitMatters.** Data Dashboard (MBTA arrival and departure data). https://dashboard.transitmatters.org
  - Open headway and travel-time histories for Boston bus routes; a second city for checking whether holding gains appear at shorter headways.

## Open-source code and tools

- **Authors not captured (2024).** An Extensible Python Open-Source Simulation Platform for Developing and Benchmarking Bus Holding Strategies. IEEE Open Journal of Intelligent Transportation Systems. https://doaj.org/article/3c53f4bddd974e2cb16154ee08ec4ece
  - A shared environment for comparing holding rules; use it to check whether the README's negative result holds in an independent simulator.

## Breakthroughs in related fields that could transfer

- **Bertsekas (2019).** Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120. https://arxiv.org/abs/1910.00120
  - The README's look-ahead is a rollout; this gives the bus-by-bus version with an improvement guarantee over the base rule.

## Published results to beat

- Dong and Gayah, arXiv:2608.10207 (abstract): against the best calibrated Daganzo baseline, headway variability −32.0%, bunching events −69.2%, passenger waiting time −24.0%, in simulations calibrated on two real routes.
- Xuan, Argote, Daganzo (2011), abstract as returned by search: about 40% less slack than schedule-based holding.
- This repository (README): no holding policy improved passenger cost by more than 0.9%, even with known future running times. The published gains above are measured on waiting time or headway variance, not on a cost that also charges on-board delay.

## Unverified leads

- Wang, Sun (2023), multi-objective multi-agent RL for multiline services with a shared corridor, Transportation Research Part C vol. 155: seen only in reference lists.
- HSL HFP historical dumps on an Azure blob store, used by third-party projects such as ForumViriumHelsinki/hsl-rt-analysis: the file host was not confirmed.
- Chalmers "An overview of solutions to the bus bunching problem in urban bus systems" (2024), https://research.chalmers.se/publication/?id=540738: record seen, authors not captured.
- HSL passenger count (APC) data per stop: no open source found.
