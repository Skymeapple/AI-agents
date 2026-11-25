# Human Frontier Mission: one project per challenge

Each of HFM's 54 challenges has its own folder here. Every folder holds the task, HFM's score so far, the existing methods, the public data, what limits progress, a route past it, the experiment and results where one was run, an estimate of how far the problem can go, and next steps.

- **Tested on public data:** 19 projects (18 experiments; the two steel tracks share one). Four use simulators built on the shared core in [`_sim/`](_sim/core.py).
- **Open:** 5 projects where public data exist and the experiment still has to be built.
- **Blocked:** 30 projects that need a GPU, a language-model key, licensed data, or details HFM has not published. See [`READINESS.md`](READINESS.md) for what each one needs, [`LITERATURE-FINDINGS.md`](LITERATURE-FINDINGS.md) for what the literature pass changed, and [`../compute/README.md`](../compute/README.md) for free compute.

The cross-project review of what worked, what stalled and where AI applies is in [`SYNTHESIS.md`](SYNTHESIS.md). The breakdown of which ingredient (information, data, algorithm, theory, reasoning) produced each gain is in [`BREAKTHROUGH-FACTORS.md`](BREAKTHROUGH-FACTORS.md).

No result here is scored on HFM's private evaluators. The overview and cross-project conclusions are in [`../human-frontier-mission.md`](../human-frontier-mission.md).

## agent-harnesses

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Building an Agent for Preserving Knowledge from Research Papers](agent-harnesses/paper-memory-agent/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | 0.2405 → 0.4794 |
| [Agent for ARC Grid Transformation Tasks](agent-harnesses/arc-grid-agent/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | 0.1213 → 0.7856 |
| [Agent for Verified Programming with Dafny](agent-harnesses/dafny-agent/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | 0.3165 → 0.8289 |
| [Agent for Formal Theorem Proving in Lean](agent-harnesses/lean-prover-agent/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | 0.3643 → 0.5661 |
| [Agent for Questions over Real Tables](agent-harnesses/table-qa-agent/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | 0.3711 → 0.877 |
| [Agent Harness Evolution Under a Frozen API Model](agent-harnesses/terminal-agent-harness/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | not published → 0.5362 |

## astronomy

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions](astronomy/photometric-redshift/README.md) | Tested on public data | Adding extinction, size, shape, aperture and error inputs: −5.8% error in every held-out fold. Redshift distributions: worse on RMSE. | 0.0364 → 0.02434 |

## biology

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA](biology/regulatory-dna-mpra/README.md) | Tested on public data | Multi-task model +0.039 Spearman on full data (three seeds agree within 0.002); reverse-complement averaging +0.02. | 0.4465 → 0.7076 |
| [Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas](biology/lung-cell-annotation/README.md) | Tested on public data | Leave-one-study-out on the atlas core: four classifier designs within 0.006 macro-F1; hierarchy-aware decoding does not help. | 0.7812 → 0.8536 |

## earth-science

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [24-Hour Global Near-Surface Weather Forecasting with ArchesWeather](earth-science/weather-24h/README.md) | Tested on public data | Blending stored forecasts of several model families: 4.6% (2020) and 8.4% (2022) lower error than the best corrected single model; bias correction alone gains about 0.5%. | 0.09823 → 0.09749 |

## infrastructure

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Single-Machine Weighted Tardiness Scheduling with Release Times](infrastructure/weighted-tardiness/README.md) | Tested on public data | Iterated local search: 12.5% less tardiness than the starting rule; 40-job instances very likely solved. | not published → 1.211 |

## materials

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim](materials/crystal-generation/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. MatterGen sampling and MatterSim relaxation need one. | 0.1364 → 0.4395 |

## medicine

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Patient-level lung cancer histology classification from H&E slides with Phikon-v2](medicine/lung-histology/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. Whole-slide images (TCGA, CPTAC, NLST) are hundreds of GB. | 0.9102 → 0.9784 |

## optimization

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Wireless 400-Cell Synthetic MILP Case Solver](optimization/wireless-milp/README.md) | Blocked | HFM has not published the instance format or the reward, and no proposals have run. | not published → not published |
| [Online Single-Track Railway Dispatcher for Finnish Main-Line Replay](optimization/rail-dispatch/README.md) | Tested on public data | Look-ahead dispatching: 10% less weighted delay than the priority rule on real days (better on 43 of 57); bound is 35% below. | 0 → 0.3984 |
| [Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing](optimization/bike-rebalancing/README.md) | Tested on public data | Forecast-based dispatch: 16% fewer failures than reactive. Look-ahead over four past days adds 2.7%; a perfect forecast would add 11%. | 0 → 0.8019 |
| [Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration](optimization/dam-spill/README.md) | Open | Needs the fish-passage and dissolved-gas models; public data exist but the model is not yet assembled. | 0 → 0.7199 |
| [Toronto Arterial Signal Timing Coordination](optimization/arterial-signals/README.md) | Open | Needs a traffic model calibrated to Toronto counts; not yet built. | 0 → 0.9892 |
| [Online Cloud Resource Allocator for Azure Trace Replay](optimization/cloud-vm-placement/README.md) | Tested on public data | Lifetime-aware packing: −13.8% server-days with true lifetimes; five policies without that knowledge do not beat first fit (−3.2%). | 0 → 0.7615 |
| [Ride-Hail Dispatch Policy for Matching and Repositioning](optimization/ride-hail-dispatch/README.md) | Open | Needs a replay simulator on NYC trip records; not yet built. | 0 → 0.9895 |
| [Control Policy for Frequent Bus Services](optimization/bus-holding/README.md) | Tested on public data | Headway-based holding raises passenger cost by 4–12% in this model; even perfect foresight gains under 1%. | 0 → 0.7505 |
| [Rolling Unit-Commitment and Dispatch Controller under Real OPSD Load and Wind Sequences](optimization/unit-commitment/README.md) | Tested on public data | Spread-scenario commitment: −23% regret against a tuned reserve rule; random scenarios are unsafe. | not published → 0.5338 |

## physics

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Battery Voltage Model for Cold Operating Conditions](physics/cold-battery-voltage/README.md) | Tested on public data | Arrhenius-form resistances: skill +0.34 against +0.15 for fitted slopes; replicated on a second cell (+0.36 against +0.16). Extrapolation now matches fitting in the cold. | 0 → 0.5991 |
| [Estimating Wave-Energy Resources from Short Measurement Campaigns](physics/wave-energy-resource/README.md) | Tested on public data | Wind alone stops at 29% error in the annual resource; adding a neighbouring buoy's wave record brings it to 19%. | 0 → 0.5602 |
| [Steel Cyclic Loading History Transfer](physics/steel-history-transfer/README.md) | Tested on public data | Shares the steel experiment: donor library cuts error 73%; 40% headroom needs a new measurement. | 0 → 0.6677 |
| [Predicting Optical Constants Beyond the Measured Wavelength Band](physics/optical-constants/README.md) | Tested on public data | Causality-only fit: +0.02 to +0.11 skill. A reference library with another measurement of the same material: +0.35. | 0 → 0.7907 |
| [Predict Cyclic Steel Stress Response from Limited Material Tests](physics/steel-cyclic-from-tension/README.md) | Tested on public data | Donor median cuts error 73%; one extra ±2% cyclic test closes 71% of the remaining gap. | 0 → 0.5453 |
| [64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well](physics/rayleigh-benard-forecast/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. The Well dataset is hundreds of GB and Walrus is a large model. | 0.3693 → 0.3129 |
| [Optical Coating Design Under Manufacturing Variation](physics/robust-ar-coating/README.md) | Tested on public data | Selecting designs by robust score: −14.5% expected reflectance. | 0 → 0.2523 |
| [PM2.5 Sensor Calibration Transfer](physics/pm25-calibration/README.md) | Open | Needs a multi-site co-location dataset; candidates are large (SensEURCity is 5.7 GB) and were not processed. | 0 → 0.6945 |
| [Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors](physics/rayleigh-benard-reconstruction/README.md) | Tested on public data | Pressure from the flow equation cuts pressure error 36% and total field error 10.8% against gappy POD; sensor-history regressions are worse. | 0.2918 → 0.2342 |
| [Wind-Farm Wake Calibration and Power Prediction from Real SCADA Data](physics/wind-farm-wakes/README.md) | Tested on public data | Direction calibration: −1.4% becomes −11.9% at the unseen farm; physics + learned residual best at −29.8%. | not published → 0.4124 |

## post-training

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [SynLogic logical reasoning training](post-training/synlogic-training/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 0.3778 |
| [Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding](post-training/embspatial-training/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 0.785 |
| [FabNER Material Manufacturing Entity Extraction](post-training/fabner-extraction/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 80.68 |
| [MapQA Map Visual Question Answering](post-training/mapqa-training/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 0.8283 |
| [ScienceWorld task training on Qwen2.5-7B-Instruct](post-training/scienceworld-training/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 35.8 |
| [TOPv2 compositional semantic parsing](post-training/topv2-parsing/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 83 |
| [Agentic RL Training Recipe Exploration with verl and agent-harness](post-training/agentic-rl-recipes/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → 0.367 |

## pretraining

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Looped pretraining with two concurrent four-GPU ideas per batch](pretraining/looped-pretraining-v2/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -1.152 |
| [Pretraining architecture development on frozen public training pool](pretraining/pretraining-architecture-pool/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -0.7048 |
| [Agentic Pretraining with a Frozen Corpus and Tokenizer](pretraining/agentic-pretraining/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -0.6941 |
| [Agentic Pretraining Under Fixed Inference-Cost Constraint](pretraining/pretraining-fixed-inference-cost/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -0.667 |
| [Looped-Transformer Pretraining Under a Fixed Compute Budget](pretraining/looped-pretraining-fixed-compute/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -0.7786 |
| [Looped Transformer Pretraining with Recurrent Depth Architectures](pretraining/looped-pretraining-v1/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. The track has ended. | not published → -1.553 |
| [Pretraining architecture development from scratch](pretraining/pretraining-from-scratch/README.md) | Blocked | Needs a GPU. This analysis ran on a 4-CPU container. | not published → -0.9624 |

## quant-finance

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [S&P 500 Minute Trading Activity Ranking](quant-finance/sp500-activity-ranking/README.md) | Blocked | One-minute consolidated-tape bars for S&P 500 members are licensed data and not freely available. | 0.1674 → 0.4625 |
| [Event-Driven Equity Strategy Based on Financial Filing Events](quant-finance/filing-event-strategy/README.md) | Open | The reward is pinned at 1.0, so HFM's evaluation cannot rank methods. Public data exist (SEC EDGAR XBRL, free daily prices) but a replay was not built. | 0 → 1 |
| [Nasdaq Ten-Level Order Book Depth Exhaustion Ranking](quant-finance/order-book-depletion/README.md) | Blocked | Nasdaq TotalView order-book data (Databento) is licensed. | 0.1997 → 0.3766 |
| [Intraday Trading Overlay for A-Share Holdings](quant-finance/ashare-intraday-overlay/README.md) | Blocked | Five-minute A-share data are not freely redistributable, and the reward is pinned at 1.0. | 0 → 1 |
| [Cross-Venue Allocation of Perpetual Orders](quant-finance/cross-venue-routing/README.md) | Open | Needs synchronised order-book data from three venues; Binance is public, Bybit and OKX archives were not fetched. | 0 → 0.5927 |
| [Minute-Level Perpetual Trading Strategy from Price, Flow, and Funding Data](quant-finance/perpetual-futures-strategy/README.md) | Tested on public data | Signal families persist to a later period; fine-tuned settings do not. | 0 → 0.0776 |
| [Execution Policy Development from Ten-Level Order Book Data](quant-finance/order-book-execution/README.md) | Blocked | Nasdaq order-book data is licensed, and the reward is pinned at 1.0. | 0 → 1 |
| [Cost-aware long-only portfolio construction under trading frictions on Qlib](quant-finance/portfolio-with-frictions/README.md) | Open | Public data exist (Qlib daily data) but the backtest with price limits and halts was not built. | not published → 0.2119 |
