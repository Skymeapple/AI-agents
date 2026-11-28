# References: Rolling Unit-Commitment and Dispatch Controller under Real OPSD Load and Wind Sequences

Verification: the shared web-search budget ran out before this project was searched. Every entry below was confirmed in this session by fetching its arXiv record (title, authors, date). Standard unit-commitment references that are not on arXiv are under "Unverified leads".

## Theory and established methods

- **Garcia, Street, Homem-de-Mello, Muñoz (2021).** Application-Driven Learning: A Closed-Loop Prediction and Optimization Approach Applied to Dynamic Reserves and Demand Forecasting. arXiv:2102.13273. https://arxiv.org/abs/2102.13273
  - Co-optimises the load forecast and the reserve requirement against realised operating cost through a bilevel problem, with an exact and a scalable heuristic method; this is a direct replacement for a reserve rule tuned by hand on 28 days.
- **Xavier, Qiu, Ahmed (2019).** Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems. arXiv:1902.01697. https://arxiv.org/abs/1902.01697
  - Uses past solves to predict warm starts and redundant constraints; relevant because the controller re-solves a similar problem every day.
- **Pineda, Morales (2021).** Is learning for the unit commitment problem a low-hanging fruit? arXiv:2106.11687. https://arxiv.org/abs/2106.11687
  - Shows a naive reuse of past commitment solutions already finds optimal or near-optimal schedules quickly; a baseline any learned method must beat.
- **Donti, Amos, Kolter (2017).** Task-based End-to-end Model Learning in Stochastic Optimization. NeurIPS 2017. https://arxiv.org/abs/1703.04529
  - Trains the forecast on the downstream dispatch cost, with a grid-scheduling example; the basis of decision-focused forecasting.
- **Elmachtoub, Grigas (2017).** Smart "Predict, then Optimize". arXiv:1710.08005. https://arxiv.org/abs/1710.08005
  - Loss function that measures decision regret instead of forecast error; regret against perfect foresight is this project's own metric.

## Recent advances and breakthroughs (2023-2026)

- **Mandi, Kotary, Berden, Mulamba et al. (2024).** Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. Journal of Artificial Intelligence Research 81, 1623-1701. https://arxiv.org/abs/2307.13565
  - Review and benchmark of methods that train predictions for decision quality, including problems with integer decisions.
- **Kempke, Koch (2025).** Fix-and-Propagate Heuristics Using Low-Precision First-Order LP Solutions for Large-Scale Mixed-Integer Linear Optimization. arXiv:2503.10344. https://arxiv.org/abs/2503.10344
  - Primal heuristic for very large unit-commitment-based dispatch models using GPU first-order LP solves; makes many-scenario commitment problems tractable.
- **de Mars (2022).** Reinforcement Learning and Tree Search Methods for the Unit Commitment Problem. PhD thesis, arXiv:2212.06001. https://arxiv.org/abs/2212.06001
  - Guided tree search (learned policy plus model-based planning) for commitment under uncertainty, aimed at cutting spinning reserve.
- **Bergmeir, de Nijs, Genov, Sriramulu et al. (2022).** Predict+Optimize Problem in Renewable Energy Scheduling. arXiv:2212.10723. https://arxiv.org/abs/2212.10723
  - Report on a forecasting-plus-scheduling competition with storage; evidence on which forecast-then-schedule pipelines worked in practice.
- **Dumouchelle, Patel, Khalil, Bodur (2022).** Neur2SP: Neural Two-Stage Stochastic Programming. arXiv:2205.12006. https://arxiv.org/abs/2205.12006
  - Replaces the scenario-wise second stage with a neural surrogate embedded in the first-stage MILP; lets the day-ahead commitment see many wind scenarios at low cost.
- **Kody, Chevalier, Chatzivasileiadis, Molzahn (2021).** Modeling the AC Power Flow Equations with Optimally Compact Neural Networks: Application to Unit Commitment. arXiv:2110.11269. https://arxiv.org/abs/2110.11269
  - Embeds a small neural network as MILP constraints inside unit commitment; the same technique can embed a learned real-time cost model.

## Datasets and benchmarks

- **Wiese, Schlecht, Bunke, Gerbaulet et al. (2019).** Open Power System Data - Frictionless data for electricity system modelling. Applied Energy 236, 401-409. https://arxiv.org/abs/1812.10405
  - Describes the data platform this project uses (time series, power plant lists, weather data); its conventional power plant list could replace the invented fleet with real German units.
- **Hörsch, Hofmann, Schlachtberger, Brown (2018).** PyPSA-Eur: An Open Optimisation Model of the European Transmission System. Energy Strategy Reviews 22, 207-215. https://arxiv.org/abs/1806.01613
  - Open European data set with German generators, renewable availability series and network; a source of realistic unit parameters and wind profiles.
- **Marot, Donnot, Dulac-Arnold, Kelly et al. (2021).** Learning to run a Power Network Challenge: a Retrospective Analysis. NeurIPS 2020 Competition Track. https://arxiv.org/abs/2103.03104
  - Sequential grid-operation benchmark with public environments; shows how a rolling grid controller is scored against realised conditions.

## Open-source code and tools

- **Brown, Hörsch, Schlachtberger (2018).** PyPSA: Python for Power System Analysis. Journal of Open Research Software 6(1). https://arxiv.org/abs/1707.09913
  - Open modelling library with unit commitment, storage and rolling-horizon optimisation; an independent implementation to check `unit_commitment.py` against.

## Breakthroughs in related fields that could transfer

- **Gibbs, Candès (2021).** Adaptive Conformal Inference Under Distribution Shift. arXiv:2106.00170. https://arxiv.org/abs/2106.00170
  - Online-calibrated prediction intervals for net-load error; sizes reserve to a target coverage that adapts by season, addressing the summer over-commitment noted in the README.
- **Chen, Wang, Kirschen, Zhang (2017).** Model-Free Renewable Scenario Generation Using Generative Adversarial Networks. arXiv:1707.09676. https://arxiv.org/abs/1707.09676
  - Generates wind scenarios with realistic temporal shape, including conditional generation; a source of the extreme low-wind scenarios the README wants in every scenario set.
- **Ansari, Stella, Turkmen, Zhang et al. (2024).** Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
  - Pretrained probabilistic forecaster that returns sample paths; a quick way to obtain day-ahead wind scenarios, since OPSD gives a day-ahead forecast for load but the README uses actual wind only.
- **Das, Kong, Sen, Zhou (2023).** A decoder-only foundation model for time-series forecasting. arXiv:2310.10688. https://arxiv.org/abs/2310.10688
  - Second pretrained forecaster for comparison.

## Published results to beat

- Kempke and Koch, arXiv:2503.10344 (abstract): primal-dual gap under 2% in under 4 hours on unit-commitment-based models with up to 8 million variables, where commercial solvers found no feasible solution in 2 days. This is a solver-scale result, not a regret figure.
- This repository (README): regret against perfect foresight of 1,016 million EUR is the reference; HFM incumbent reward 0.5338.

## Unverified leads

- Knueven, Ostrowski, Watson (2020), "On mixed-integer programming formulations for the unit commitment problem", INFORMS Journal on Computing: not confirmed in session.
- Morales-España, Latorre, Ramos (2013), "Tight and compact MILP formulation for the thermal unit commitment problem", IEEE Transactions on Power Systems: not confirmed in session.
- Bertsimas, Litvinov, Sun, Zhao, Zheng (2013), "Adaptive robust optimization for the security constrained unit commitment problem", IEEE Transactions on Power Systems: not confirmed in session.
- UnitCommitment.jl (Xavier et al., Argonne) with its benchmark instances, and the RTS-GMLC test system: believed to exist, not confirmed in session (a Zenodo record fetch failed).
- OPSD time-series package, release 2020-10-06 (data.open-power-system-data.org/time_series): named in the README, release page not opened; check whether it includes the TSO day-ahead wind forecast columns for Germany.
- ENTSO-E Transparency Platform day-ahead wind and solar forecasts: not confirmed in session.
