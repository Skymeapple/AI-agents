# Human Frontier Mission: one project per challenge

Each of HFM's 54 challenges has its own folder here. Every folder holds the task, HFM's score so far, the existing methods, the public data, what limits progress, a route past it, the experiment and results where one was run, an estimate of how far the problem can go, and next steps.

- **Tested on public data or in a model study:** 50 projects (49 experiments; the two steel tracks share one). Five use simulators, four of them built on the shared core in [`_sim/`](_sim/core.py).
- **Open:** 1 project where public data exist and the experiment still has to be built.
- **Blocked:** 3 projects that need a GPU, a language-model key, licensed data, or details HFM has not published. See [`READINESS.md`](READINESS.md) for what each one needs, [`LITERATURE-FINDINGS.md`](LITERATURE-FINDINGS.md) for what the literature pass changed, and [`../compute/README.md`](../compute/README.md) for free compute.

The cross-project review of what worked, what stalled and where AI applies is in [`SYNTHESIS.md`](SYNTHESIS.md). The breakdown of which ingredient (information, data, algorithm, theory, reasoning) produced each gain is in [`BREAKTHROUGH-FACTORS.md`](BREAKTHROUGH-FACTORS.md). The nine problem classes behind the 54 projects, what blocks each, and how they compare with AI breakthroughs in mathematics, physics and biology (in Chinese) are in [`PROBLEM-CLASSES.md`](PROBLEM-CLASSES.md).

No result here is scored on HFM's private evaluators. The overview and cross-project conclusions are in [`../human-frontier-mission.md`](../human-frontier-mission.md).

## agent-harnesses

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Building an Agent for Preserving Knowledge from Research Papers](agent-harnesses/paper-memory-agent/README.md) | Tested on public data | With no model on QASPER: a 3,000-character note of selected paragraphs keeps 46% of the answer evidence; an oracle that knows the questions keeps 81%. The paper's opening keeps 9%. HFM's own task needs a language-model key. | 0.2405 → 0.4794 |
| [Agent for ARC Grid Transformation Tasks](agent-harnesses/arc-grid-agent/README.md) | Tested on public data | With no model, a fixed library of 34 grid operations searched two deep solves 6.5% of public training tasks and 1.0% of held-out evaluation tasks. When a program fits the examples it is usually right (30 of 36); the hard part is proposing one. | 0.1213 → 0.7856 |
| [Agent for Verified Programming with Dafny](agent-harnesses/dafny-agent/README.md) | Tested on public data | With no model on DafnyBench: 28.8% of hint-stripped programs verify unchanged; two template rules for loop invariants, pruned by the verifier, raise that to 37.0%. 7.7% of the files do not parse. HFM's own task needs a language-model key. | 0.3165 → 0.8289 |
| [Agent for Formal Theorem Proving in Lean](agent-harnesses/lean-prover-agent/README.md) | Tested on public data | With no model on miniF2F: one composed tactic (`norm_num at * <;> grind`) proves 34.4% of the test theorems; eight tactics chosen on the validation set prove 40.6%. Adding one structural step before a tactic portfolio moves that only to 42.2%. HFM's own task needs a language-model key. | 0.3643 → 0.5661 |
| [Agent for Questions over Real Tables](agent-harnesses/table-qa-agent/README.md) | Tested on public data | With a small open model on CPU, one harness detail (printing the last expression) moves the code route from 1.6% to 27.6%. Choosing between the direct and code routes with a simple rule gives 41.2% on fresh questions against 28.0% direct (perfect selector 47.6%). | 0.3711 → 0.877 |
| [Agent Harness Evolution Under a Frozen API Model](agent-harnesses/terminal-agent-harness/README.md) | Blocked | Needs paid access to a language-model API to run the frozen model. | not published → 0.5362 |

## astronomy

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions](astronomy/photometric-redshift/README.md) | Tested on public data | Adding extinction, size, shape, aperture and error inputs: −5.8% error in every held-out fold. Redshift distributions: worse on RMSE. | 0.0364 → 0.02434 |

## biology

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA](biology/regulatory-dna-mpra/README.md) | Tested on public data | Multi-task model +0.039 Spearman on full data; with strand averaging and a five-seed ensemble the total goes from 0.535 to 0.672. | 0.4465 → 0.7076 |
| [Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas](biology/lung-cell-annotation/README.md) | Tested on public data | Full atlas, 35 studies each held out: a small network averaged with nearest neighbours lifts macro-F1 from 0.684 (linear) to 0.786, better in 35 of 35 studies. On the 11-study core all methods were within 0.006. | 0.7812 → 0.8536 |

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
| [Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim](materials/crystal-generation/README.md) | Tested on public data | Selection stand-in on JARVIS-DFT, 1,468 held-out chemical systems: a cheap learned screen raises the stable share of one pick from 23% to 39% (on the hull) and from 47% to 70% (within 0.05 eV/atom), and holds when related systems are removed from training. Generation still needs a GPU. | 0.1364 → 0.4395 |

## medicine

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Patient-level lung cancer histology classification from H&E slides with Phikon-v2](medicine/lung-histology/README.md) | Tested on public data | Leakage check on public LC25000 patches (CPU): random folds give 99.5% for adenocarcinoma against squamous; keeping rotated and flipped copies of one image together drops it to 93.6%, twelve times the error. Patient-level slide work still needs a GPU. | 0.9102 → 0.9784 |

## optimization

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Wireless 400-Cell Synthetic MILP Case Solver](optimization/wireless-milp/README.md) | Blocked | HFM has not published the instance format or the reward, and no proposals have run. | not published → not published |
| [Online Single-Track Railway Dispatcher for Finnish Main-Line Replay](optimization/rail-dispatch/README.md) | Tested on public data | Look-ahead dispatching: 10% less weighted delay than the priority rule (57 days); 15% when it also uses delays visible upstream (52 days); bound is 35% below. | 0 → 0.3984 |
| [Citywide Bicycle Rebalancing Policy for Station-Based Bike-Sharing](optimization/bike-rebalancing/README.md) | Tested on public data | Forecast-based dispatch: 16% fewer failures than reactive. Look-ahead over four past days adds 2.7%; a perfect forecast would add 11%. | 0 → 0.8019 |
| [Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration](optimization/dam-spill/README.md) | Tested on public data | Gas model fitted on 2012–2021, tested on 2022–2024: a linear model beats the saturating physics form (1.80 vs 2.02 points). Joint spill planning removes only 1.4% of passage loss at a 120% cap and none at 125% (model study). | 0 → 0.7199 |
| [Toronto Arterial Signal Timing Coordination](optimization/arterial-signals/README.md) | Tested on public data | Model study: a plan tuned to a range of days has 17.5% less delay on other days than one tuned to the counted hour (6 of 8 corridors). | 0 → 0.9892 |
| [Online Cloud Resource Allocator for Azure Trace Replay](optimization/cloud-vm-placement/README.md) | Tested on public data | Lifetime-aware packing: −13.8% server-days with true lifetimes; five policies without that knowledge do not beat first fit (−3.2%). | 0 → 0.7615 |
| [Ride-Hail Dispatch Policy for Matching and Repositioning](optimization/ride-hail-dispatch/README.md) | Tested on public data | Repositioning idle drivers on a demand forecast halves lost requests (83% → 92% served, 14 of 14 days); better matching alone gives 0.6%. | 0 → 0.9895 |
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
| [64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well](physics/rayleigh-benard-forecast/README.md) | Tested on public data | Floors with no network on three flow settings: blurring towards the mean beats persistence by only 9–16% (0.56–0.57) and fails at the most turbulent setting; simple skill is gone within about 8 frames. The Walrus model itself needs a GPU. | 0.3693 → 0.3129 |
| [Optical Coating Design Under Manufacturing Variation](physics/robust-ar-coating/README.md) | Tested on public data | Selecting designs by robust score: −14.5% expected reflectance. | 0 → 0.2523 |
| [PM2.5 Sensor Calibration Transfer](physics/pm25-calibration/README.md) | Tested on public data | A learned calibration transfers 10% better to an unseen city and 29% better to unseen sites than a linear one; the physics-based humidity form adds nothing. | 0 → 0.6945 |
| [Full-field reconstruction of 2D Rayleigh–Bénard convection from 256 fixed sensors](physics/rayleigh-benard-reconstruction/README.md) | Tested on public data | Pressure from the flow equation cuts pressure error 36% and total field error 10.8% against gappy POD; sensor-history regressions are worse. | 0.2918 → 0.2342 |
| [Wind-Farm Wake Calibration and Power Prediction from Real SCADA Data](physics/wind-farm-wakes/README.md) | Tested on public data | Direction calibration: −1.4% becomes −11.9% at the unseen farm; physics + learned residual best at −29.8%. | not published → 0.4124 |

## post-training

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [SynLogic logical reasoning training](post-training/synlogic-training/README.md) | Tested on public data | An untrained 3B open model solves 4.8% of the 270 easy validation puzzles directly and 1.5% by writing programs; 20 of 27 task types score zero. Classical search, checked by the benchmark's verifiers, solves 124 of 130 puzzles in 13 task types (45.9% of the set). Six futoshiki puzzles are unsolvable as stated and one verifier accepts wrong answers. | not published → 0.3778 |
| [Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding](post-training/embspatial-training/README.md) | Tested on public data | Without the image, object boxes and plain geometry answer 89.1% (98–100% on left/right/above/under, about 69% on nearest/farthest); a blind model reaches 59.5%. The distractor options are never correct. HFM's trained model: 78.5%. | not published → 0.785 |
| [FabNER Material Manufacturing Entity Extraction](post-training/fabner-extraction/README.md) | Tested on public data | On the public split a CRF given a dictionary's tags scores 86.2 F1, a dictionary of training phrases 83.4 and a CRF 82.8, against HFM's 80.7 (different evaluator, not directly comparable). 86% of test entities are seen phrases; the classical model finds only 7% of the unseen 14%. | not published → 80.68 |
| [MapQA Map Visual Question Answering](post-training/mapqa-training/README.md) | Tested on public data | Without the map a model scores about 20% (yes/no guessing; 4–7% on state and value questions) on 65,071 questions from a public copy. The benchmark cannot be answered from language, unlike EmbSpatial-Bench. | not published → 0.8283 |
| [ScienceWorld task training on Qwen2.5-7B-Instruct](post-training/scienceworld-training/README.md) | Tested on public data | With no training, a 3B open model scores 12.9 of 100 (16.5 when its commands are snapped to the simulator's valid-action list) against 5.9 for random actions, on 30 tasks. The snap gain rests on one task, and the valid-action list omits many gold commands. | not published → 35.8 |
| [TOPv2 compositional semantic parsing](post-training/topv2-parsing/README.md) | Tested on public data | A per-word tagger on CPU scores 38.5% exact match against HFM's 83 (different test set). 90% of test parses use a tree shape seen in training; the tagger gets 2% of the rest. With a small open model and no training, grammar-constrained decoding lifts exact match from 26.7% to 45.7%, but not on unseen shapes. | not published → 83 |
| [Agentic RL Training Recipe Exploration with verl and agent-harness](post-training/agentic-rl-recipes/README.md) | Tested on public data | Toy scale (CPU, test-checked string-command tasks): GRPO on a narrow task generator gains 16 points in its own distribution but loses 30 on broader tasks and halves a longer-task benchmark; no recipe improves the longer tasks. | not published → 0.367 |

## pretraining

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [Looped pretraining with two concurrent four-GPU ideas per batch](pretraining/looped-pretraining-v2/README.md) | Tested on public data | Toy scale (1.9M parameters, CPU): held-out DCLM bits per byte rise steadily as FineWeb-Edu replaces DCLM in training (2.090 with DCLM only, 2.153 at a 75% FineWeb-Edu share), against the direction of HFM's best proposal. | not published → -1.152 |
| [Pretraining architecture development on frozen public training pool](pretraining/pretraining-architecture-pool/README.md) | Tested on public data | Toy scale (1.9M parameters, CPU): HFM's proposed 0.60/0.20/0.20 web/code/maths mix gains 0.015 bits per byte on web against 0.50/0.25/0.25 but loses 0.031 on the four-domain mean; balanced mixes are within seed noise of one another. | not published → -0.7048 |
| [Agentic Pretraining with a Frozen Corpus and Tokenizer](pretraining/agentic-pretraining/README.md) | Tested on public data | Toy scale (1.9M parameters, CPU): rotary positions, a warmup-stable-decay schedule and longer sequences each gain 0.03–0.04 bits per byte; stacked, they beat the sum of their gains by 0.023, because longer sequences need the other two. | not published → -0.6941 |
| [Agentic Pretraining Under Fixed Inference-Cost Constraint](pretraining/pretraining-fixed-inference-cost/README.md) | Tested on public data | Toy scale (3.7M parameters, CPU): at equal per-token compute, 8 layers beats 2, 4 and 16 layers, but the three reasonable shapes span only 0.017 bits per byte; doubling training gains 0.084. | not published → -0.667 |
| [Looped-Transformer Pretraining Under a Fixed Compute Budget](pretraining/looped-pretraining-fixed-compute/README.md) | Tested on public data | Toy scale (2–4M parameters, CPU): looping closes 63% of the gap between 4 and 8 layers at equal parameters, loses by 0.026 bits per byte at equal compute, and the loop-index idea adds nothing beyond seed noise. | not published → -0.7786 |
| [Looped Transformer Pretraining with Recurrent Depth Architectures](pretraining/looped-pretraining-v1/README.md) | Tested on public data | Toy scale (1.9M parameters, CPU): training with mixed loop counts costs 0.013 to 0.048 bits per byte at 3 loops, no model improves with extra loops at test time, and confidence-based early exit does not beat a fixed loop count. | not published → -1.553 |
| [Pretraining architecture development from scratch](pretraining/pretraining-from-scratch/README.md) | Tested on public data | Toy scale (1.9M parameters, CPU): runs at 5–12% of the final length rank eight recipes with correlation 0.52 and wrongly reject longer sequences; a quarter-length run picks the right winner (0.79). | not published → -0.9624 |

## quant-finance

| Project | Status | Headline result or blocker | HFM baseline → best |
| --- | --- | --- | --- |
| [S&P 500 Minute Trading Activity Ranking](quant-finance/sp500-activity-ranking/README.md) | Tested on public data | Stand-in on public Bybit trades (24 perpetuals, 7 test days): a linear model on burst and level features ranks 30-minute activity surprises at Spearman 0.52 (bursts fading alone: 0.38); HFM's momentum covariate adds +0.002; trees are worse. | 0.1674 → 0.4625 |
| [Event-Driven Equity Strategy Based on Financial Filing Events](quant-finance/filing-event-strategy/README.md) | Open | The reward is pinned at 1.0, so HFM's evaluation cannot rank methods. Public data exist (SEC EDGAR XBRL, free daily prices) but a replay was not built. | 0 → 1 |
| [Nasdaq Ten-Level Order Book Depth Exhaustion Ranking](quant-finance/order-book-depletion/README.md) | Tested on public data | Stand-in on public Bybit order books (four perpetuals, one test day): ranking ten-second depth changes reaches Spearman 0.32–0.59; most of it is depth mean reversion (0.23–0.42), and trees add almost nothing over a linear model. | 0.1997 → 0.3766 |
| [Intraday Trading Overlay for A-Share Holdings](quant-finance/ashare-intraday-overlay/README.md) | Blocked | Five-minute A-share data are not freely redistributable, and the reward is pinned at 1.0. | 0 → 1 |
| [Cross-Venue Allocation of Perpetual Orders](quant-finance/cross-venue-routing/README.md) | Tested on public data | Replay on public trade prints: routing to the venue with the cheaper last trade saves 0.7 basis points per small order (bound 0.77), before fees and with an optimistic fill proxy. | 0 → 0.5927 |
| [Minute-Level Perpetual Trading Strategy from Price, Flow, and Funding Data](quant-finance/perpetual-futures-strategy/README.md) | Tested on public data | Signal families persist to a later period; fine-tuned settings do not. | 0 → 0.0776 |
| [Execution Policy Development from Ten-Level Order Book Data](quant-finance/order-book-execution/README.md) | Tested on public data | Stand-in on public Bybit order books: resting at the touch costs 0.7–0.85 bp more than TWAP on tight books (adverse selection); an imbalance rule recovers two-thirds; only the wide-spread instrument shows a saving (0.07 bp). Slicing large orders matters most. | 0 → 1 |
| [Cost-aware long-only portfolio construction under trading frictions on Qlib](quant-finance/portfolio-with-frictions/README.md) | Tested on public data | On 49 US industry portfolios, 2000–2026: trading part-way lifts a fast signal from 0.50 to 0.60 return per unit of risk at 10 bp costs; HFM's drift-trigger idea was the weakest cost-aware rule; a slow signal gained nothing over monthly rebalancing. | not published → 0.2119 |
