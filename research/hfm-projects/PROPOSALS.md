# Proposals ready for HFM's Research Exchange

HFM's scores are private, so the only way to find out whether any result in this repository beats an HFM limit is to submit it through HFM's own evaluation. That needs the repository owner's HFM account. This page turns the evidence into proposals in the form HFM's tracks use: a hypothesis, the change, the evidence, the expected direction, and how to check it. Nothing here has been run on HFM's data.

Proposals are grouped by what they ask HFM to do. Each links to the project folder with the full experiment.

## A. Change the method (evidence that a structural change helps)

| Track | Proposal | Evidence here | Expected direction |
|---|---|---|---|
| [24-hour weather](earth-science/weather-24h/README.md) | Replace single-model post-processing with a weighted blend of several model families (GraphCast, FGN, Pangu, HRES), weights fitted per variable on an earlier year. | WeatherBench 2: −4.6% (2020) and −8.4% (2022) error against the best bias-corrected single model; bias correction alone −0.5%. | Large improvement if other models' forecasts are admissible inputs. |
| [Lung cell annotation](biology/lung-cell-annotation/README.md) | Train a small non-linear head on UCE embeddings averaged with k-nearest-neighbour votes, evaluated leave-one-study-out on the full atlas. | Macro-F1 0.684 → 0.786, better in 35 of 35 held-out studies; no effect on the 11-study core. | Improvement, provided training uses the full atlas. |
| [Regulatory DNA](biology/regulatory-dna-mpra/README.md) | Multi-task head over cell types, reverse-complement averaging, five-seed ensemble. | Spearman 0.535 → 0.672 on lentiMPRA. | Improvement. |
| [Photometric redshift](astronomy/photometric-redshift/README.md) | Add extinction, size, shape, aperture and error columns as inputs. | −5.8% error in every held-out sky region. | Improvement. |
| [Crystal generation](materials/crystal-generation/README.md) | Over-generate 4–8× the 64-structure budget and spend the budget on candidates ranked by a cheap composition-and-structure screen (or MatterSim single points). | On JARVIS-DFT, a screen trained on other systems raises the stable share of one pick from 23% to 39% (on hull), holding on unseen chemical families. | Higher stable fraction per budget; check novelty separately. |
| [Wireless 400-cell MILP](optimization/wireless-milp/README.md) | Run HiGHS until its incumbent stalls, then spend the rest of the budget on 12-cell neighbourhoods with 4-second sub-solves around that incumbent; do not seed from greedy. | Self-built 400-cell instances: gap to LP bound 2.93% → 1.85% on 4 of 4; greedy-seeded LNS 6–11%. | Improvement if HFM's instances behave like these. |
| [Convection reconstruction](physics/rayleigh-benard-reconstruction/README.md) | Compute pressure from the reconstructed velocity through the flow equation instead of regressing it. | Pressure error −36%, total field error −10.8% against gappy POD. | Improvement. |
| [Rail dispatch](optimization/rail-dispatch/README.md) | Look-ahead dispatching that uses delays already visible upstream. | −15% weighted delay against the priority rule over 52 days. | Improvement. |
| [Ride-hail dispatch](optimization/ride-hail-dispatch/README.md) | Reposition idle drivers on a demand forecast; leave matching alone. | Served 83% → 92%, 14 of 14 days; better matching alone +0.6%. | Improvement, if HFM's score is not saturated. |
| [Cold battery model](physics/cold-battery-voltage/README.md) | Arrhenius-form temperature dependence of the resistances. | Extrapolation skill +0.34 against +0.15, replicated on a second cell (+0.36). | Improvement in the cold. |
| [Lean prover](agent-harnesses/lean-prover-agent/README.md) | Before any model call, try a validation-chosen portfolio of composed tactics (starting with `norm_num at * <;> grind`), with one structural step (induction, case split) first. | No model: 42.2% of miniF2F test. A 7B open-weight prover on free CPU (4 samples) then adds 18 of the 134 left; 8 more samples add only 3; repairing model sketches with the portfolio adds 3 (2 only through repair): 138 of 244 after 24 samples (at least 53.7% with validation-chosen tactics). | A free floor under any model-based prover; close every prefix of a failed model proof with the tactic portfolio before asking for more samples. |
| [Table QA agent](agent-harnesses/table-qa-agent/README.md) | Print the last expression of generated code, and route between direct and code answers with a simple rule. | Code route 1.6% → 27.6%; routing 41.2% against 28.0% direct. | Improvement for small models; check with the frozen model. |
| [Paper memory agent](agent-harnesses/paper-memory-agent/README.md) | Build notes from whole paragraphs chosen by a learned evidence model, not from the paper's opening. | Evidence share at 3,000 characters: 0.09 (opening) → 0.46. | Improvement. |

## B. Do not spend the budget on these ideas (evidence against HFM's current best idea)

| Track | HFM's current best idea | Evidence here | Suggested instead |
|---|---|---|---|
| [Looped pretraining v2](pretraining/looped-pretraining-v2/README.md) | Shift sampling towards FineWeb-Edu (q = 0.25p + 0.75 FineWeb) to lower held-out DCLM bpb. | Toy scale: held-out DCLM worsens monotonically with the FineWeb-Edu share (2.090 → 2.153 at 75%). | Test the share sweep at small scale first; keep DCLM in the mix. |
| [Pretraining architecture pool](pretraining/pretraining-architecture-pool/README.md) | 0.60/0.20/0.20 web/code/maths beats 0.50/0.25/0.25 on the four-domain score. | Toy scale: web −0.015, four-domain mean +0.031 (worse); balanced mixes within noise. | Do not starve code or maths; report per domain with seed spread. |
| [Looped pretraining v1](pretraining/looped-pretraining-v1/README.md) | Balanced 1/2/4-loop schedule improves likelihood. | Toy scale: mixed loop counts cost 0.013–0.048 bpb at 3 loops; extra loops at test time never help. | Fixed loop count for the score; mixed counts only if flexible inference cost is wanted. |
| [Looped pretraining, fixed compute](pretraining/looped-pretraining-fixed-compute/README.md) | Loop-index embedding lets the shared block specialise. | Toy scale: no effect beyond seed noise. | Drop it. |
| [Agentic RL recipes](post-training/agentic-rl-recipes/README.md) | GRPO on procedurally synthesised terminal tasks with tests. | Toy scale: a narrow generator gains 16 points in-distribution and loses 30 on broader tasks; nothing improves longer tasks. | Widen the generator to the benchmark's task lengths; score on Terminal-Bench from the first checkpoint. |
| [S&P activity ranking](quant-finance/sp500-activity-ranking/README.md) | Add a price-momentum covariate. | Public crypto stand-in: +0.002 to a linear model. | Burst and level features in a linear model (0.52). |
| [A-share overlay](quant-finance/ashare-intraday-overlay/README.md) | Tenfold stronger ridge shrinkage. | Public crypto stand-in: within one standard error; every overlay negative after 2 bp cost. | Measure gross-to-net first; compare with not trading. |
| [Portfolio with frictions](quant-finance/portfolio-with-frictions/README.md) | Drift-triggered rebalancing. | 49 US industry portfolios: the weakest of five cost-aware rules. | Partial trading towards target. |
| [Order-book depth](quant-finance/order-book-depletion/README.md) | Student-t likelihood in a frozen variance model. | Public crypto stand-in: trees ≈ linear; depth mean reversion carries most of the signal. Adding the depth's drawdown from its 5- and 30-minute peak and its distance from the 30-minute minimum raises Spearman on 4 of 4 symbols (+0.016 to +0.043). | Path features (drawdown from recent peak, distance from recent minimum) over likelihood changes. |

## C. Fix the yardstick first

| Track | Problem | Evidence | Request |
|---|---|---|---|
| [Filing events](quant-finance/filing-event-strategy/README.md), [A-share overlay](quant-finance/ashare-intraday-overlay/README.md), [Order-book execution](quant-finance/order-book-execution/README.md) | Reward pinned at 1.0. | Stand-ins show differences of tenths of a basis point and a no-trade optimum after costs. | Report net cost or return in basis points with standard errors, against a no-trade benchmark. |
| [EmbSpatial training](post-training/embspatial-training/README.md) | Benchmark answerable without the image. | Object boxes and plain geometry: 89.1% (HFM's trained model 78.5%); distractors never correct. | Remove the layout shortcut or report against this floor. |
| [Lung histology](medicine/lung-histology/README.md) | Leakage risk between patches of one source. | LC25000: 99.5% with random folds, 93.6% with near-duplicates grouped. | Confirm folds are drawn by patient (ideally by hospital). |
| [Terminal-agent harness](agent-harnesses/terminal-agent-harness/README.md) | Harness changes accepted on noisy, in-distribution evidence. | Frozen 3B model: the best development change is the worst on held-out task kinds (+12 → −12 points). | Accept changes only on held-out task kinds, counted by kind, with repeats. |
| [From-scratch pretraining](pretraining/pretraining-from-scratch/README.md) | Ideas chosen from very short runs. | Toy scale: runs at 5–12% of final length rank recipes at 0.52 and reject a later winner; 25% length picks it. | Screen at a quarter of the final length for late-paying changes. |

## How to submit

Each project's README names the script and the log behind its numbers. A submission should restate the hypothesis in the track's own terms, reuse HFM's frozen evaluation, and report the result with HFM's noise estimate. Proposals in group B are worth submitting as small, cheap ablations; a negative result there saves the track's budget.
