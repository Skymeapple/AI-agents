# References: Online Single-Track Railway Dispatcher for Finnish Main-Line Replay

Verification: the shared web-search budget ran out before this project was searched. Every entry below was confirmed in this session by fetching its arXiv record (title, authors, date). The core operations-research literature on train dispatching is mostly not on arXiv, so it is listed under "Unverified leads" and still needs checking. This file is incomplete.

## Theory and established methods

- **Abels, Jordi, Ostrowski, Schaub et al. (2020).** Train Scheduling with Hybrid Answer Set Programming. Theory and Practice of Logic Programming 21 (2021), 317-347. https://arxiv.org/abs/2003.08598
  - Exact routing and scheduling of real-world train instances with difference constraints for timing; an alternative exact engine to MILP for a rolling-horizon meet-pass solve.
- **Bertsekas (2019).** Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120. https://arxiv.org/abs/1910.00120
  - The README's look-ahead is a rollout on the priority rule; this gives the train-by-train version and the conditions under which rollout cannot do worse than its base rule.

## Recent advances and breakthroughs (2023-2026)

- **Yue, Jin, Dai, Feng et al. (2024).** Reinforcement Learning for Scalable Train Timetable Rescheduling with Graph Representation. arXiv:2401.06952. https://arxiv.org/abs/2401.06952
  - Graph-network policy that builds a feasible rescheduled timetable step by step, trained with a delay curriculum and finished with a simple local search; the learned-policy-plus-local-search pattern applies to meet-pass ordering.

## Datasets and benchmarks

- **Mohanty, Nygren, Laurent, Schneider et al. (2020).** Flatland-RL: Multi-Agent Reinforcement Learning on Trains. arXiv:2012.05893. https://arxiv.org/abs/2012.05893
  - Open grid-world rail benchmark with single-track conflicts and deadlocks; a public test bed for dispatch rules before the Finnish replay.
- **Laurent, Schneider, Scheller, Watson et al. (2021).** Flatland Competition 2020: MAPF and MARL for Efficient Train Coordination on a Grid World. arXiv:2103.16511. https://arxiv.org/abs/2103.16511
  - Competition report setting operations-research and path-finding planners against multi-agent reinforcement learning on train rescheduling; read its results section before investing in a learned dispatcher.

## Breakthroughs in related fields that could transfer

- **Zhang, Song, Cao, Zhang et al. (2020).** Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. arXiv:2010.12367. https://arxiv.org/abs/2010.12367
  - Learns priority rules on the disjunctive graph of a job shop; single-track meet-pass scheduling is a job-shop problem in the same graph form.
- **Li, Tinka, Kiesel, Durham et al. (2020).** Lifelong Multi-Agent Path Finding in Large-Scale Warehouses. arXiv:2005.07371. https://arxiv.org/abs/2005.07371
  - Rolling-horizon collision resolution: resolve conflicts only within a time window and replan periodically; the same windowing the README proposes at 10 to 20 minutes.
- **Jungel, Parmentier, Schiffer, Vidal (2023).** Learning-based Online Optimization for Autonomous Mobility-on-Demand Fleet Control. arXiv:2302.03963. https://arxiv.org/abs/2302.03963
  - Trains an online policy to imitate full-information optimal solutions; the README's hindsight whole-day optimum (−34.5%) is exactly such a training target.
- **Liu, Tong, Yuan, Lin et al. (2024).** Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model. arXiv:2401.02051. https://arxiv.org/abs/2401.02051
  - LLM-driven search over heuristic code; a structured alternative to the hand-proposed veto rules HFM agents are iterating on.

## Published results to beat

- No external published number on this line was confirmed. This repository (README): rollout look-ahead −11.0% weighted delay against the priority rule on real days; hindsight bound −34.5%. HFM incumbent: 0.3984.

## Unverified leads

- Fintraffic Digitraffic rail API (rata.digitraffic.fi), including track-section occupancy messages and train compositions: named in the README, not confirmed in session.
- D'Ariano, Pacciarelli, Pranzo (2007), "A branch and bound algorithm for scheduling trains in a railway network", EJOR (alternative-graph model): not confirmed in session.
- Lamorgese, Mannino (2015), "An exact decomposition approach for the real-time train dispatching problem", Operations Research: not confirmed in session.
- Törnquist, Persson (2007), "N-tracked railway traffic re-scheduling during disturbances", Transportation Research Part B: not confirmed in session.
- Cacchiani et al. (2014), "An overview of recovery models and algorithms for real-time railway rescheduling", Transportation Research Part B: not confirmed in session.
- DISPLIB 2025 train dispatching competition and instance library (SINTEF and partners): believed to exist, not confirmed in session.
- INFORMS Railway Applications Section problem-solving competition data sets (single-track dispatching instances): not confirmed in session.
- Li et al. (2021), "Scalable Rail Planning and Replanning: Winning the 2020 Flatland Challenge", ICAPS: not confirmed in session.
