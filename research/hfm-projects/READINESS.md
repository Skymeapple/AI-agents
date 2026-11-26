# Readiness of the 54 projects

Where each project stands, and what it still needs before research can proceed. Data access was checked from free GitHub Actions compute on 2026-11-25; the raw report is in [`../compute/results/data-access-probe/run.log`](../compute/results/data-access-probe/run.log).

## Runs today on CPU (25)

| Project | Still missing |
| --- | --- |
| [Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration](optimization/dam-spill/README.md) | A real fish-passage model and hourly data. |
| [Cost-aware long-only portfolio construction under trading frictions on Qlib](quant-finance/portfolio-with-frictions/README.md) | Single-stock data with a clear licence, price limits and halts. |
| [Cross-Venue Allocation of Perpetual Orders](quant-finance/cross-venue-routing/README.md) | Order-book data for a faithful replay. |
| [PM2.5 Sensor Calibration Transfer](physics/pm25-calibration/README.md) | Nothing. |
| [Toronto Arterial Signal Timing Coordination](optimization/arterial-signals/README.md) | Toronto's real counts in place of generated volumes. |
| [Ride-Hail Dispatch Policy for Matching and Repositioning](optimization/ride-hail-dispatch/README.md) | Nothing. |
| [Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors](physics/rayleigh-benard-reconstruction/README.md) | Nothing for the linear methods. A GPU for learned decoders. |
| [Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas](biology/lung-cell-annotation/README.md) | A free GPU account for UCE embeddings. The CPU route on stored embeddings runs now. |
| [Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions](astronomy/photometric-redshift/README.md) | A free GPU account for the image-based version. |
| [Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA](biology/regulatory-dna-mpra/README.md) | A free GPU account for Evo 2 features; the 7B model may exceed a 16 GB GPU, so the smaller Evo 2 checkpoints are the realistic start. |
| [24-Hour Global Near-Surface Weather Forecasting with ArchesWeather](earth-science/weather-24h/README.md) | A free GPU account to produce ArchesWeather forecasts. |
| [Single-Machine Weighted Tardiness Scheduling with Release Times](infrastructure/weighted-tardiness/README.md) | Nothing. |
| [Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing](optimization/bike-rebalancing/README.md) | Nothing. Next step is to replace assumed dock counts with the published ones. |
| [Control Policy for Frequent Bus Services](optimization/bus-holding/README.md) | A working source for recorded vehicle positions, to replace the assumed variability. |
| [Online Cloud Resource Allocator for Azure Trace Replay](optimization/cloud-vm-placement/README.md) | Nothing. |
| [Online Single-Track Railway Dispatcher for Finnish Main-Line Replay](optimization/rail-dispatch/README.md) | Nothing. |
| [Rolling Unit-Commitment and Dispatch Controller under Real OPSD Load and Wind Sequences](optimization/unit-commitment/README.md) | Nothing. |
| [Battery Voltage Model for Cold Operating Conditions](physics/cold-battery-voltage/README.md) | Nothing. |
| [Predicting Optical Constants Beyond the Measured Wavelength Band](physics/optical-constants/README.md) | Nothing. |
| [Optical Coating Design Under Manufacturing Variation](physics/robust-ar-coating/README.md) | Nothing. |
| [Predict Cyclic Steel Stress Response from Limited Material Tests](physics/steel-cyclic-from-tension/README.md) | Nothing. |
| [Steel Cyclic Loading History Transfer](physics/steel-history-transfer/README.md) | Nothing. |
| [Estimating Wave-Energy Resources from Short Measurement Campaigns](physics/wave-energy-resource/README.md) | Nothing. A swell hindcast (ERA5 waves) would need a free Copernicus account. |
| [Wind-Farm Wake Calibration and Power Prediction from Real SCADA Data](physics/wind-farm-wakes/README.md) | Nothing. |
| [Minute-Level Perpetual Trading Strategy from Price, Flow, and Funding Data](quant-finance/perpetual-futures-strategy/README.md) | Nothing. |

## Data verified; pipeline or simulator still to build (1)

| Project | Still missing |
| --- | --- |
| [Event-Driven Equity Strategy Based on Financial Filing Events](quant-finance/filing-event-strategy/README.md) | Your decision on a contact address to send to the SEC, then the event-study pipeline. |

## Ready to run once a free GPU account is connected (17)

| Project | Still missing |
| --- | --- |
| [Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim](materials/crystal-generation/README.md) | A free GPU account. |
| [Patient-level lung cancer histology classification from H&E slides with Phikon-v2](medicine/lung-histology/README.md) | A free GPU account, and an NLST data request for the held-out cohort (CPTAC can stand in until then). A CPU route also exists: pre-extracted slide features for TCGA and CPTAC are on Hugging Face behind an access form, which needs a free Hugging Face account token. |
| [64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well](physics/rayleigh-benard-forecast/README.md) | A free GPU account. |
| [Agentic RL Training Recipe Exploration with verl and agent-harness](post-training/agentic-rl-recipes/README.md) | A free GPU account. |
| [Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding](post-training/embspatial-training/README.md) | A CPU study has run. A free GPU account for the fine-tuning. |
| [FabNER Material Manufacturing Entity Extraction](post-training/fabner-extraction/README.md) | A CPU study has run. A free GPU account for the hybrid with a language model. |
| [MapQA Map Visual Question Answering](post-training/mapqa-training/README.md) | A CPU study has run. A free GPU account for the fine-tuning. |
| [ScienceWorld task training on Qwen2.5-7B-Instruct](post-training/scienceworld-training/README.md) | A CPU stand-in study has run. A free GPU account for the fine-tuning. |
| [SynLogic logical reasoning training](post-training/synlogic-training/README.md) | A CPU stand-in study has run. A free GPU account for the fine-tuning. |
| [TOPv2 compositional semantic parsing](post-training/topv2-parsing/README.md) | A CPU study has run. A free GPU account for the language model. |
| [Agentic Pretraining with a Frozen Corpus and Tokenizer](pretraining/agentic-pretraining/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Looped-Transformer Pretraining Under a Fixed Compute Budget](pretraining/looped-pretraining-fixed-compute/README.md) | A toy-scale CPU test has run. A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Looped Transformer Pretraining with Recurrent Depth Architectures](pretraining/looped-pretraining-v1/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Looped pretraining with two concurrent four-GPU ideas per batch](pretraining/looped-pretraining-v2/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Pretraining architecture development on frozen public training pool](pretraining/pretraining-architecture-pool/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Agentic Pretraining Under Fixed Inference-Cost Constraint](pretraining/pretraining-fixed-inference-cost/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |
| [Pretraining architecture development from scratch](pretraining/pretraining-from-scratch/README.md) | A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute. |

## Ready to run once a language-model key is connected (6)

| Project | Still missing |
| --- | --- |
| [Agent for ARC Grid Transformation Tasks](agent-harnesses/arc-grid-agent/README.md) | A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token. |
| [Agent for Verified Programming with Dafny](agent-harnesses/dafny-agent/README.md) | A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token. |
| [Agent for Formal Theorem Proving in Lean](agent-harnesses/lean-prover-agent/README.md) | A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token. |
| [Building an Agent for Preserving Knowledge from Research Papers](agent-harnesses/paper-memory-agent/README.md) | A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token. |
| [Agent for Questions over Real Tables](agent-harnesses/table-qa-agent/README.md) | A small-scale study with an open 3B model has run on CPU. A language-model key for HFM's frozen model. |
| [Agent Harness Evolution Under a Frozen API Model](agent-harnesses/terminal-agent-harness/README.md) | A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token. |

## Blocked on licensed data (4)

| Project | Still missing |
| --- | --- |
| [Intraday Trading Overlay for A-Share Holdings](quant-finance/ashare-intraday-overlay/README.md) | Licensed intraday data, or a redefinition on daily data. |
| [Nasdaq Ten-Level Order Book Depth Exhaustion Ranking](quant-finance/order-book-depletion/README.md) | Licensed order-book history. |
| [Execution Policy Development from Ten-Level Order Book Data](quant-finance/order-book-execution/README.md) | Licensed order-book history. |
| [S&P 500 Minute Trading Activity Ranking](quant-finance/sp500-activity-ranking/README.md) | Licensed minute bars. |

## Blocked on unpublished task details (1)

| Project | Still missing |
| --- | --- |
| [Wireless 400-Cell Synthetic MILP Case Solver](optimization/wireless-milp/README.md) | The instance format and reward, which HFM has not published. |

## What would unblock the most

- **A free GPU account (Kaggle):** the 17 GPU projects, plus image and foundation-model versions of four CPU projects.
- **A language-model key:** the 6 agent-harness projects.
- **Build work only, no outside dependency:** the 1 project with verified data.
- **Licensed market data:** 4 finance projects. No free legal substitute was found.
