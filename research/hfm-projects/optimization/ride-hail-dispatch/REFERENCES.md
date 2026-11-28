# References: Ride-Hail Dispatch Policy for Matching and Repositioning

Verification: the shared web-search budget ran out before this project was searched. RideGym and OpenHail were confirmed by a web search result; every other entry was confirmed by fetching its arXiv record (title, authors, date). Papers and datasets that are not on arXiv are under "Unverified leads".

## Theory and established methods

- **Qin, Zhu, Ye (2022).** Reinforcement Learning for Ridesharing: An Extended Survey. Transportation Research Part C 144. https://arxiv.org/abs/2105.01099
  - Survey of matching, repositioning, pricing and pooling methods; the map of the field for this task.
- **Braverman, Dai, Liu, Ying (2016).** Empty-car routing in ridesharing systems. arXiv:1609.07219. https://arxiv.org/abs/1609.07219
  - Fluid-model optimal repositioning of idle cars in a closed queueing network; gives an upper bound and a static repositioning rule to compare with.
- **Banerjee, Freund, Lykouris (2016).** Pricing and Optimization in Shared Vehicle Systems: An Approximation Framework. arXiv:1608.06819. https://arxiv.org/abs/1608.06819
  - Approximation guarantees for control of closed-network vehicle systems; a theory baseline for repositioning.
- **Kanoria, Qian (2019).** Blind Dynamic Resource Allocation in Closed Networks via Mirror Backpressure. arXiv:1903.02764. https://arxiv.org/abs/1903.02764
  - Near-optimal dispatch that needs no demand forecast; by construction robust to demand shocks, which is the README's proposed test.
- **Lin, Zhao, Xu, Zhou (2018).** Efficient Collaborative Multi-Agent Deep Reinforcement Learning for Large-Scale Fleet Management. arXiv:1802.06444. https://arxiv.org/abs/1802.06444
  - Early large-scale learned repositioning; a standard baseline.
- **Oda, Joe-Wong (2018).** MOVI: A Model-Free Approach to Dynamic Fleet Management. arXiv:1804.04758. https://arxiv.org/abs/1804.04758
  - Per-vehicle learned repositioning from trip data.
- **Shah, Lowalekar, Varakantham (2019).** Neural Approximate Dynamic Programming for On-Demand Ride-Pooling. arXiv:1911.08842. https://arxiv.org/abs/1911.08842
  - A learned value function inside the batch assignment problem; the "value-function dispatch" named in the README.
- **Jin, Zhou, Zhang, Li et al. (2019).** CoRide: Joint Order Dispatching and Fleet Management for Multi-Scale Ride-Hailing Platforms. arXiv:1905.11353. https://arxiv.org/abs/1905.11353
  - Joint matching and repositioning in one hierarchical policy.
- **Tang, Zhang, Qin, Wang et al. (2021).** Value Function is All You Need: A Unified Learning Framework for Ride Hailing Platforms. arXiv:2105.08791. https://arxiv.org/abs/2105.08791
  - One value function drives both dispatch and repositioning, with online updates; the production-style design from DiDi.
- **Feng, Gluzman, Dai (2020).** Scalable Deep Reinforcement Learning for Ride-Hailing. IEEE Control Systems Letters 5(6), 2060-2065. https://arxiv.org/abs/2009.14679
  - Policy optimisation for matching and empty-car routing that scales by decomposing the joint action.
- **Gammelli, Yang, Harrison, Rodrigues et al. (2021).** Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems. arXiv:2104.11434. https://arxiv.org/abs/2104.11434
  - Graph-network rebalancing policy that transfers across demand patterns and cities; relevant to robustness under demand shocks.

## Recent advances and breakthroughs (2023-2026)

- **Jungel, Parmentier, Schiffer, Vidal (2023).** Learning-based Online Optimization for Autonomous Mobility-on-Demand Fleet Control. arXiv:2302.03963. https://arxiv.org/abs/2302.03963
  - Learns dispatching and rebalancing from optimal full-information solutions inside a combinatorial optimisation layer; since this replay "provides future requests", the hindsight optimum is available as a training target.
- **Enders, Harrison, Pavone, Schiffer (2022, revised 2023).** Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems. arXiv:2212.07313. https://arxiv.org/abs/2212.07313
  - Per-vehicle learned scores combined through a central bipartite matching; the same structure as batched matching with learned weights.

## Datasets and benchmarks

- **Zhao, Hu, Li (2026).** RideGym: A Standardized Interface for Real-World Large-Scale Ride-Sharing System. arXiv:2607.10173. https://arxiv.org/abs/2607.10173
  - Open-source Gym-style environment for order dispatch that separates the simulator from the dispatch algorithm; could replace the unbuilt in-house replay simulator.
- **Schettini, Kullman, Mendoza (2026).** OpenHail: An Event-Driven Gymnasium Environment for Electric Ride-Hailing Fleet Control. arXiv:2609.30628. https://arxiv.org/abs/2609.30628
  - Event-driven simulator with pickup deadlines (rider patience), vehicle job queues, repositioning and seeded instances; its request model matches this task's constraints.

## Open-source code and tools

- **OpenHail repository (MIT licence).** https://github.com/tommaso-schettini/openhail
  - Code for the environment above; a template for an event-driven NYC replay.

## Breakthroughs in related fields that could transfer

- **Bertsekas (2019).** Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120. https://arxiv.org/abs/1910.00120
  - Agent-by-agent rollout on top of the incumbent policy; a low-risk improvement step when the score is already near its ceiling.
- **Gibbs, Candès (2021).** Adaptive Conformal Inference Under Distribution Shift. arXiv:2106.00170. https://arxiv.org/abs/2106.00170
  - Keeps demand-forecast intervals calibrated when demand shifts; supports the README's demand-shock robustness test.

## Published results to beat

- Jungel et al., arXiv:2302.03963 (abstract): up to 17.1% and on average 6.3% more realised profit than greedy and model-predictive control baselines on large real-world scenarios. HFM incumbent on this task: 0.9895 (README).

## Unverified leads

- NYC TLC Trip Record Data (yellow, green and high-volume for-hire vehicle Parquet files) and the TLC taxi zone lookup: named in the README, not confirmed in session.
- Xu et al. (2018), "Large-scale order dispatch in on-demand ride-hailing platforms: a learning and planning approach", KDD: not confirmed in session.
- Alonso-Mora et al. (2017), "On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment", PNAS: not confirmed in session.
- Özkan, Ward (2020), "Dynamic matching for real-time ride sharing", Stochastic Systems: not confirmed in session.
- DiDi KDD Cup 2020 dispatch and repositioning simulator and the GAIA open dataset: not confirmed in session.
