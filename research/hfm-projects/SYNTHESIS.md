# What worked, what stalled, and where AI applies

A review of all 54 Human Frontier Mission (HFM) projects, written on 25 November
2026. It answers five questions:

1. Which projects made progress, or could?
2. What do the successes have in common?
3. What holds back the projects that stalled?
4. Which kinds of project could break through with AI, and under what conditions?
5. Where are the current limits of AI in this kind of research, and what makes a
   project a good fit?

## 1. The evidence, and how far it reaches

Two bodies of evidence are used.

- **HFM's own record.** 54 challenges, 2,307 experiments proposed and run by
  HFM's AI agents, about $112,000 of compute. Read from HFM's public data feed
  on its launch day. This shows what autonomous agents did when left to
  optimise a fixed score.
- **The experiments in this repository.** 17 experiments on public data covering
  18 challenges, all designed, coded and run by an AI model working with a
  person in one session, on free CPU machines. This shows what an AI does
  when asked to test specific ideas with held-out data.

Limits on what can be concluded:

- **None of the experiments here is scored on HFM's private tests.** They test
  ideas, not HFM's leaderboard.
- **HFM's agents and models are not disclosed.** "AI" below means the systems
  that produced these two records, not AI in general.
- **The record is a snapshot of one day.** Tracks that had run for weeks and
  tracks that had run for hours are in the same table.
- **Only 15 of 54 tracks have a baseline that allows a relative gain to be
  computed.** 19 score against a reference set to zero, and 20 publish no
  baseline at all.

## 2. Where progress happened

### In HFM's record

| Group | Tracks | Result | Cost per track |
| --- | --- | --- | --- |
| Agent harnesses around a frozen model | ARC grids, Dafny, tables, paper notes, Lean | Scores rose 55% to 548%. Three of five closed about three quarters of the gap to a perfect score. | $1,100–2,800 |
| Science tasks with a weak or generic starting model | Crystal generation, regulatory DNA, galaxy redshift | +222%, +58%, error −33% | $1,500–5,200 |
| Science tasks with a strong starting model | Lung histology, lung cell annotation, convection (two tracks) | +7% to +20%. Histology still closed 76% of its remaining gap (0.910 to 0.978). | $2,100–6,000 |
| Science task with a specialised state-of-the-art starting model | 24-hour weather | −0.75% error | $3,600 |
| Hand-built physical models and operating heuristics | 10 physics and 9 operations tracks | Skill 0.25 to 0.99 against a zero reference. Gains continue but in very small steps. | $800–9,000 |
| Model training recipes | 7 post-training, 7 pretraining | No baseline published; no improvement can be recorded. | $20–1,300 |

The single strongest pattern in HFM's numbers: **the lower the starting score,
the larger the gain.**

| Starting score | Relative gain |
| --- | --- |
| 0.12–0.20 | +89% to +548% |
| 0.24–0.45 | +55% to +162% |
| 0.78–0.91 | +7% to +9% |

### In the experiments here

22 specific ideas were tested. About 12 held up, 2 held under a condition, and
8 did not.

| Held up | Size of effect |
| --- | --- |
| Weather: blend several model families | 4.6% (2020) and 8.4% (2022) lower error than the best corrected single model |
| Cold battery: Arrhenius-form resistances | Skill +0.34 against +0.15 for fitted slopes, on 35 of 35 cycles |
| Regulatory DNA: one multi-task model | +0.039 correlation, three seeds within 0.002 |
| Rail: look-ahead dispatching | 9.9% less delay, better on 43 of 57 days |
| Bike sharing: dispatch on forecast stock | 16% fewer failed trips |
| Grid scheduling: plan against spread wind scenarios | 23% less excess cost |
| Optical coatings: select designs under variation | 14.5% lower expected reflectance |
| Wind farms: calibrate direction, then add a learned residual | 1.4% becomes 11.9%, then 29.8% |
| Job scheduling: local search | 10–12% less lateness; small instances very likely solved |
| Galaxy redshift: add dust extinction as an input | 1.3% lower error in every fold |
| DNA: reverse-strand averaging | +0.02 correlation |
| Crypto futures: judge on a later period | Strategy families persist; tuned settings do not |

| Did not hold | What happened |
| --- | --- |
| Bus holding by headway | Raised passenger cost 4–12%; perfect foresight gains under 1% |
| Wave energy: separate swell model | No better than plain regression |
| Steel: prior conditioned on the tension test | No reliable gain over averaging other steels |
| Steel: one extra load reversal | Closes only 8% of the gap |
| Lung cells: hierarchy-aware labels | Four classifier designs within 0.006 |
| Galaxy redshift: predict a distribution | Error up 2.3% |
| Optical constants: causality constraint alone | Small and fragile |
| Bike sharing: look-ahead on one past day | No gain over 17 days |
| Cloud: coarse lifetime classes | Worse, even with true lifetimes |
| Wind farms: "avoid learned residuals" | Wrong; the hybrid was best |

## 3. What the successes have in common

Seven factors recur. Each is stated with the evidence for it.

1. **A weak or generic starting point.** Every gain above 50% in HFM's record
   started below 0.45. The agents' strength is bringing a problem from "barely
   works" to "competent" using known methods.
2. **A score that can be computed automatically, quickly and cheaply.** The
   harness tracks cost $1,100–2,800 each and iterate in minutes. Every
   experiment here that produced a clear answer ran in under 30 minutes on four
   CPUs. The weather blend ran in 90 seconds.
3. **The information needed is already in the inputs.** In the wins, the data
   contained the answer and the method had to extract it: DNA sequence predicts
   activity, forecasts from different models carry different errors, the
   timetable shows the conflict ahead.
4. **A known method existed and had not been applied.** Almost every win is a
   transfer: multi-task learning, Arrhenius kinetics, scenario planning,
   rollout, model blending, local search. None is a new invention. The
   literature pass found most of them already published for a neighbouring
   problem.
5. **The change was to structure, not to a setting.** The wins changed what is
   being combined, chosen or simulated. HFM's stalled tracks changed
   coefficients. In the weather track the difference is a factor of ten: 0.75%
   from tuning one model's output, 8.4% from combining five models.
6. **A missing input or a missing step was added.** Extinction for redshift,
   wind-direction offset for wakes, demand forecast for bikes. Small additions,
   each found by asking what the model was not being told.
7. **Physics and learning were combined.** A known physical form carried the
   model to new conditions (cold batteries), and a learned part mopped up what
   the form missed (wind farms: physics 11.9%, learned 29.3%, both 29.8%).

One further factor applies to the experiments here and not to HFM's record:

8. **Results were judged on data held out along the real shift, with noise
   measured.** Held-out sky regions, studies, farms, steels, years. Several
   apparent gains vanished under this test (bike look-ahead, the 19-of-19 rail
   result, the wider optical search). HFM's agents accepted gains as small as
   one part in ten million.

## 4. What holds back the stalled projects

Eight distinct bottlenecks appear. Most stalled projects have one clear primary
cause.

| Bottleneck | What it looks like | Projects |
| --- | --- | --- |
| **Missing information.** The inputs do not contain what is needed, so no model can supply it. | A large gap to an "oracle" that is given the missing fact, and several different methods stopping at the same score. | Steel (both): a tension test says little about cyclic hardening; 40% gap. Wave energy: local wind says nothing about distant swell; three methods stop at 30% error. Optical constants: resonances outside the band. Cloud placement: lifetimes; 14% gap. Bike sharing: demand; 11% gap. |
| **Local search trap.** The agent edits its last winner and never changes approach. | 83% of HFM's proposals edit an earlier one. Median accepted gain 0.15%. | Weather, convection forecast, redshift, order-book ranking, S&P activity, and most physics tracks late in their runs. |
| **A strong specialised starting point.** | Post-processing has little left to give. | Weather (−0.75%), histology (0.978), lung cells. |
| **A broken or missing yardstick.** | Score pinned at its maximum, or no baseline. | Three finance tracks at 1.0; ride-hail and traffic signals at 0.989; all 14 training-recipe tracks; wireless planning (no task published). |
| **Little to win.** The problem has small headroom under its own cost measure. | Even perfect foresight gains almost nothing. | Bus holding (under 1% in this model). |
| **Noise mistaken for progress.** | Many accepted gains below run-to-run variation. | Grid scheduling with random scenarios (one method ranged from −54% to +583%); the terminal-agent track reports ±6 points of noise. |
| **Engineering reliability.** Long runs fail before producing a result. | Few proposals end with a score. | Pretraining: one track scored 2 of 20 proposals. |
| **Access.** The work needs something only a person can provide. | Nothing runs at all. | 17 projects need a GPU account, 6 a language-model key, 4 licensed market data, 1 a contact address for a data provider. |

Two of these are worth separating, because they call for opposite responses:

- **Missing information cannot be fixed by a better model.** It needs a new
  measurement, a new data source or a better forecast. More AI effort on the
  existing inputs is wasted.
- **The local search trap can be fixed by process alone.** The same agents,
  told to change structure and given a larger budget for it, have the
  ingredients. The weather result shows the size of what is being left behind.

## 5. Which projects could break through with AI, and what they need

| Condition to supply | Projects that would benefit | Why a breakthrough is plausible | Evidence |
| --- | --- | --- | --- |
| **Permission and budget to change the approach,** not only tune it | Weather, convection forecast, redshift, lung cells, order-book and activity ranking | A structural alternative is known and untried | Weather blend: 8.4% against 0.75%. Published: probabilistic retrofitting of simulators cuts rollout error scores 20–54%. |
| **A published harness design the agents have not adopted** | Lean proving, paper notes, terminal agent | The gap to published results is large | Lean: 0.566 against a published 95.9% on a standard benchmark. |
| **A better forecast of one hidden quantity** | Cloud placement (lifetimes), bike sharing (demand), rail (late trains), grid scheduling (wind) | The value of the forecast is measured and large | Cloud 14%, bikes 11%, rail up to 35% (optimistic bound). Cloud needs lifetimes right within a factor of 3. |
| **A simulator with a hindsight bound** | Rail, bikes, dam spill, ride-hail, cross-venue routing | Policies can be trained to imitate hindsight-optimal decisions | Simulators for three of these now exist here. |
| **A GPU** | Crystals, histology, convection, DNA with Evo 2, all 14 training tracks | The tasks are well posed and the data is open | Data access verified for all of them. |
| **A repaired yardstick** | Three capped finance tracks, 14 training tracks | Nothing can be learned until scores can separate methods | — |
| **One new measurement or data source** | Steel (a short cyclic test), wave energy (a swell hindcast), optical constants (material-class knowledge) | The missing information is identified and obtainable | Steel: four reversals close 30% of the gap. |

Projects where a breakthrough from AI is **unlikely** on current evidence:

- **Bus holding:** little to win under this cost measure.
- **Traffic signals and ride-hail dispatch:** already at 0.989 under HFM's score.
- **Pretraining at HFM's stated budgets:** published work finds looped models
  lose to standard ones at matched compute, and the tracks need 4–8 GPUs for
  days.
- **Trading strategies:** signals are weak against costs, and published
  anomalies decay after publication.

## 6. Current limits of AI in this kind of research

Stated as what the two records show, with the evidence for each.

**What AI does well**

- **Raises a weak solution to a competent one quickly and cheaply.** $1,000–3,000
  and tens of experiments per problem in HFM's record.
- **Builds scaffolding around an existing model.** The five harness tracks.
- **Applies a known method to a new problem,** including writing the
  simulator, the data pipeline and the evaluation. All 17 experiments here.
- **Constructs physical sub-models and operating rules by hand.** HFM's physics
  and operations tracks reach skill of 0.5–0.8 this way.
- **Tunes exhaustively.** Sometimes to the point of solving the problem
  (small scheduling instances).
- **Surveys literature and finds the relevant prior work.** The literature pass
  corrected several of this repository's own notes.

**What AI does not yet do reliably**

- **Leave a local optimum without being told to.** 83% of HFM's proposals are
  edits of an earlier winner.
- **Tell signal from noise unaided.** HFM's agents accept gains far below
  run-to-run variation. Discipline has to be imposed from outside.
- **Predict which idea will work.** Here, about 12 of 22 confidently proposed
  ideas held. The proposals were plausible and well-argued in every case. Only
  the experiment separated them.
- **Create information that is not in the inputs.** Five projects are bounded
  this way, and no modelling change moved them.
- **Beat a mature, specialised system by adjusting its output.** Weather.
- **Carry long engineering efforts to completion.** Pretraining.
- **Design or repair its own yardstick.** Saturated and baseline-free tracks
  kept running.
- **Obtain access.** Accounts, licences, compute and data agreements still
  need a person.

The summary: **AI is currently a fast, cheap, tireless applier and combiner of
known methods under a trusted score. It is not yet a reliable judge of its own
results, nor a source of information the data lacks.** The human contribution
that mattered most in both records was defining the test, choosing the held-out
split, and supplying access.

## 7. Criteria for whether a project suits AI

Eight questions. A project that passes all eight is a strong candidate. Each
"no" names the thing to fix first.

| # | Question | If the answer is no |
| --- | --- | --- |
| 1 | **Is there an automatic score that still separates good from bad?** Not saturated, with a published baseline and known noise. | Repair the score before anything else. |
| 2 | **Can one experiment run in minutes to hours for a few dollars?** | Find a cheaper proxy, or expect few iterations. |
| 3 | **Do the inputs contain the information needed?** Test: give an "oracle" the suspected missing fact and measure the gap. | Collect the measurement or data. More modelling will not help. |
| 4 | **Is the current solution weak or generic, or is there a known structural alternative to a strong one?** | Expect single-digit gains from tuning. |
| 5 | **Do published methods exist for a neighbouring problem?** | The task needs invention, which neither record shows AI doing. |
| 6 | **Are data, models and tools open and reachable?** | A person must supply access. |
| 7 | **Can the test be held out along the real shift** (new site, material, year, study)? | Results will look better than they are. |
| 8 | **Is the search allowed to change approach, with budget for it?** | Progress will stall at the first local optimum. |

### The 54 projects sorted by these criteria

**Strong fit now: passes all or all but one (18).**
ARC grids, Dafny, table questions, paper notes, Lean proving, crystal
generation, regulatory DNA, galaxy redshift, convection reconstruction, cold
battery, wind-farm wakes, optical coatings, rail dispatch, bike rebalancing,
cloud placement, grid scheduling, job scheduling, dam spill.

**Fit once the search is allowed to change approach (criterion 8) (6).**
Weather, convection forecast, lung cell annotation, lung histology, terminal
agent, air-quality sensor calibration.

**Bounded by missing information (criterion 3) until new data arrives (5).**
Steel (two tracks), wave energy, optical constants, S&P trading activity.

**Not assessable until the yardstick is repaired (criterion 1) (20).**
Seven post-training and seven pretraining tracks (no baseline); filing events,
A-share overlay and order-book execution (score at 1.0); ride-hail and traffic
signals (0.989); wireless planning (no task published).

**Weak fit (5).**
Bus holding (little to win); crypto futures strategy, cross-venue routing,
order-book depletion and portfolio construction (weak signals against costs,
with licensed data or unclear data licences for three of them).

Several projects also fail criterion 6 on access (GPU, model key or licensed
data) regardless of the group above. Those are listed in
[`READINESS.md`](READINESS.md).

## 8. Update after a further round of experiments

A second round targeted the projects that had not moved. The detailed breakdown by
ingredient is in [`BREAKTHROUGH-FACTORS.md`](BREAKTHROUGH-FACTORS.md).

**Projects that moved past their earlier limit**

| Project | Earlier limit | What moved it | New result |
| --- | --- | --- | --- |
| Wave energy | 29% error with wind alone | A neighbouring buoy's wave record | 19% error |
| Optical constants | Skill +0.02 to +0.11 | A library with another measurement of the same material | Skill +0.35 |
| Steel | Donor median, 40% short of the best possible | One extra cyclic test at ±2% strain | 71% of that gap closed |
| Galaxy redshift | −1.3% from extinction | All extra catalogue inputs together | −5.8% |
| Rail | −10% from look-ahead | Delays already visible upstream | −14% (18 days; 60-day run queued) |
| Bike sharing | Look-ahead gave nothing | Look-ahead averaged over four past days | −2.7% |
| Cold battery | Skill +0.34 on one cell | Independent second cell | Replicated: +0.36 |
| Convection reconstruction | Not started | Pressure computed from the flow equation | −10.8% against HFM's starting method |
| Weather | −4.6% on 2020 | Five model families on 2022 | −8.4%, stable when halves are swapped |

**Projects that were pushed and did not move**

| Project | What was tried | Result |
| --- | --- | --- |
| Cloud placement | Re-predicting lifetimes; tenant and batch affinity | None beats first fit. The 13.8% needs lifetime knowledge the trace does not carry. |
| Lung cell annotation | Eight classifier and adaptation methods | All within 0.006. The embedding is the limit. |
| Cold battery | Two refinements beyond the Arrhenius form | Each helped one cell and not the other. |
| Bus holding | Look-ahead with perfect foresight | Under 1%. |

**What this changes in the sorting of section 7**

- Steel, wave energy and optical constants were listed as "bounded by missing
  information". That diagnosis was right, and supplying the information moved all
  three. They now belong with the projects that fit, on the condition that the task
  allows the extra source.
- Convection reconstruction is confirmed as a strong fit.
- Cloud placement moves the other way: it is bounded by information that could not
  be obtained from the public trace.

The round also tested the review's own central claim. Of the nine projects that
moved, seven moved by adding information or data, one by theory (convection) and
one by more reasoning (bikes).

### Third round

A third round built the remaining experiments that needed only public data and
CPU, and revisited two earlier results.

| Project | What was tested | Result |
| --- | --- | --- |
| Lung cell annotation | Non-linear classifier heads on the full 35-study atlas | Macro-F1 0.684 → 0.786, better in 35 of 35 studies. This reverses the "did not move" entry above: the limit on the 11-study core was the small benchmark, not the embedding. |
| Air-quality sensors | Boosted trees for calibration transfer | Skill +0.10 across cities, +0.29 across sites. The humidity growth law did not help. |
| Ride-hail dispatch | Forecast-driven repositioning | Requests served 83% → 92%. |
| Arterial signals | Tuning to a range of days (model study) | Delay on other days −17.5%. |
| Cross-venue routing | Route to the cheaper last print | 0.70 of 0.77 basis points available, on trade prints only and before fees. |
| Portfolio under costs | Five trading rules | Part-way trading rescues a fast signal (0.50 → 0.60) but only to the level of holding everything. HFM's drift trigger was the weakest rule. |
| Dam spill | Gas model and joint spill planning | Joint planning worth 1.4% at most. A linear gas model beat the saturating physics form. |
| FabNER entity extraction | Dictionary and classical model on CPU | 83.4 and 82.8 F1 on the public split, against HFM's 80.7 from a fine-tuned language model (different evaluator). Headroom is in the 14% of entities never seen in training. |
| FabNER, dictionary plus a small open model | Untrained 3B model proposes extra entities | Worse: 83.2 → 75.7 F1 on 300 sentences. The model finds 20% of unseen-phrase entities, but only 6% of what it adds is correct. |
| FabNER, CRF given the dictionary's tags | Cross-fitted stacking on CPU | 86.2 F1 against 83.4 for the dictionary (+2.8, interval +2.5 to +3.2). The gain is precision on seen phrases; unseen phrases stay unfound. |
| SynLogic reasoning puzzles | Untrained 3B model, direct answer against writing a program | 4.8% and 1.5% of 270 puzzles. Only 11% of programs ran and printed anything. A negative result: the small model is too weak for this benchmark. |
| SynLogic, classical solvers | Search programs for 13 of 27 task types, judged by the benchmark's verifiers | 124 of 130 accepted (45.9% of the easy set). Six futoshiki puzzles have no solution; the arithmetic-target verifier accepted a wrong answer in 10 of 10. |
| EmbSpatial-Bench without the image | Blind model, and object boxes with plain geometry | Blind 59.5%; boxes 89.1% (HFM's trained model 78.5%). Left/right/above/under are 2D layout (98–100%); nearest/farthest needs depth (about 69%). Distractor options are never the answer. |
| EmbSpatial-Bench, depth for nearest/farthest | Off-the-shelf monocular depth read inside each box | 67.2% → 74.8% on those questions (88% on scanned homes, worse on simulated scenes). Depth is part of the missing input, not all of it. |
| MapQA without the map | Blind baselines on 65,071 questions, whole maps held out | About 20%, almost all from yes/no guessing. No language shortcut: HFM's 0.83 is map reading. |
| ScienceWorld with no training | Random, small-model and snapped-to-valid agents on 30 tasks | 5.9, 12.9 and 16.5 of 100. Constraining to the simulator's valid-action list is not a reliable gain (better on 3 tasks, worse on 7); the list omits many gold commands. |
| ARC without a model | Search over a fixed library of 34 grid operations, two deep | 6.5% of public training tasks, 1.0% of held-out evaluation tasks. Fitted programs are usually right (30 of 36); proposing a fitting program is the hard part. |
| Fixed inference cost (toy scale) | Four shapes at equal per-token compute; doubled training for one | 8 layers best (1.071 bits per byte); 4 and 16 layers within 0.017; 2 layers +0.075. Doubling training: −0.084. Training amount outweighs shape at this scale. |
| Convection forecasting floors | Persistence, damped, blurred and mode-regression forecasts, 64 frames, three flow settings | Best floor 0.56–0.57 at Rayleigh 1e6 and 1e8; fitted methods lose to persistence at 1e10. Simple skill is gone by lead 8. |
| Dafny without a model | DafnyBench unchanged, then template loop invariants pruned by the verifier | 28.8% verify unchanged; bounds +0.9 points; postcondition rewrite +7.3 points, to 37.0%. 60 of 782 files do not parse. |
| Lean without a model, then an open prover | miniF2F, one short tactic call per theorem (52 tactics); tactics chosen on validation, scored on test; then DeepSeek-Prover-V1.5-RL 7B on CPU on the rest; all proofs re-checked strictly | `grind` alone 30.3% of test; `norm_num at * <;> grind` 34.4%; best four 38.5%; best eight 40.6%; 13 chosen 41.4%. The open prover adds 23 theorems (32 samples each): 52.9% combined, 50.8% with validation-chosen tactics. No `imo` proof survived the strict check. An earlier, weaker acceptance test had overstated these numbers (up to 143 of 244); automatic repair of model sketches added nothing once checked strictly. |
| Paper notes without a model | QASPER, 3,000-character notes of selected sentences; builder chosen on validation, scored on test | Evidence share: paper opening 0.09, TF-IDF central 0.20, learned sentence selection 0.38, learned whole-paragraph selection 0.46, oracle that knows the questions 0.81. Word features did not help. The limit is not knowing the questions, not the budget. |
| TOPv2 semantic parsing | Tree-shape coverage and a per-word tagger on CPU | Tagger 38.5% exact match against HFM's 83 (different test). 90% of test parses use a seen tree shape; the tagger gets 2% of the rest. Here the language model's gain is real, and the open part is the unseen tenth. |
| TOPv2, constrained decoding | Small open model, retrieved examples, with and without a grammar | Exact match 26.7% → 45.7% with no training (60 fixed, 2 broken). Unseen shapes unchanged within noise (6 → 9 of 130). |
| Table question answering | Two harnesses around a small open model on CPU | Printing the code's last expression moves the code route from 1.6% to 27.6%. A simple rule choosing between the direct and code routes gives 41.2% on fresh questions against 28.0% direct; a perfect selector 47.6%. |
| Looped transformer | Toy-scale comparison | Looping closes 63% of the depth gap per parameter, loses at equal compute. The loop-index idea adds nothing. |
| Looped transformer, variable loops | Toy-scale comparison | Mixed loop counts in training cost 0.013 to 0.048 bits per byte at 3 loops and buy tolerance to running fewer loops. Extra loops at test time never help. Confidence-based early exit does not pay. |
| Looped transformer, data mix | Toy-scale share sweep | Held-out DCLM is best when training on DCLM alone and worsens steadily as FineWeb-Edu replaces it (2.090 → 2.153 at a 75% share), against HFM's best proposal. |
| Pretraining data mix, four domains | Toy-scale mix sweep | HFM's 0.60/0.20/0.20 mix helps web (−0.015) and hurts the four-domain mean (+0.031) against 0.50/0.25/0.25. Balanced mixes are flat within noise; starving a domain is costly. |
| Stacking recipe changes | Toy-scale 2×2×2 factorial | Each of three changes gains 0.03–0.04; stacked they beat additive by 0.023. Pairs that enable each other are super-additive, overlapping pairs sub-additive. |
| Short-run screening | Toy-scale rank transfer | Runs at 5–12% of final length rank recipes at 0.52 and reject a change that wins later; a quarter-length run picks the right winner (0.79). |
| Crystal pre-screening | Public DFT stand-in, held-out chemical systems | A learned screen raises the stable share of a fixed budget 1.4–1.7× (23% → 39% on the hull for one pick) and transfers to unseen chemical families. Diversity by formula trades stable share for unique compositions. |
| RL on procedural tasks with tests | Toy-scale GRPO | Narrow generator: +16 points in-distribution, −30 on broader tasks, benchmark halved. No recipe improves longer compositions. Dynamic sampling and rejection-sampling fine-tuning avoid the in-distribution loss. |
| Histology patch leakage | Public LC25000, CPU | Random folds 99.5%; grouping augmented copies of one image drops adenocarcinoma vs squamous to 93.6% (error ×12). Near-perfect patch benchmarks largely measure leakage. |
| Order-book depth ranking | Public crypto books | Spearman 0.32–0.59 on a test day; depth mean reversion gives 0.23–0.42; trees ≈ linear. |
| Order execution | Public crypto books | Passive placement loses 0.7–0.85 bp to TWAP on tight books (adverse selection); an imbalance rule recovers two-thirds; savings exist only with a wide spread and are under 0.1 bp. |
| Activity-surprise ranking | Public crypto trades | Linear model 0.52 Spearman, reversion alone 0.38; HFM's momentum covariate +0.002; boosted trees worse than linear. |
| Intraday overlay | Public crypto minute data | Reversal earns ~75 bp/day gross; 2 bp costs turn every overlay negative; tenfold shrinkage is within noise. Not trading is best. |
| Wireless MILP | Self-built 400-cell instances | Greedy-seeded LNS loses to plain HiGHS; HiGHS stalls after 180 s; LNS around HiGHS's incumbent cuts the gap to the LP bound from 2.93% to 1.85% (4 of 4). |
| Harness changes, frozen model | 3B open model, 48 shell tasks | Folder listing up front: development +12 points, held-out task kinds −12 points. Structured feedback and a check before finishing change no outcome. Effective sample is 12 task kinds. |

Three points for the conclusions of this review:

- **A null result on a small benchmark can hide a large effect.** The lung-atlas
  gain appeared only with 35 studies and more training cells. Section 6's warning
  about evaluating on too little data applies to negative results too.
- **Two more "obvious" coordination ideas failed** (joint dam planning,
  drift-triggered rebalancing). Together with bus holding and cloud placement,
  four of the operations projects show that a more global optimiser does not
  help when the simple rule is already near what the information allows.
- **HFM's agents' own ideas were testable cheaply and two did not hold:** the
  loop index and the drift trigger. A toy-scale or public-data check before
  spending GPU budget would have filtered them.

## 9. How confident to be

- **Most confident:** the starting-point pattern, the local-search trap, and
  the missing-information bound. Each rests on many tracks or on repeated
  experiments with oracle comparisons.
- **Moderately confident:** the list of what could break through. It rests on
  measured gaps and published results, but most routes are untested on HFM's
  own data.
- **Least confident:** the sorting of individual projects. Some were placed on
  one experiment, some on HFM's record alone, and the bus and trading
  judgements depend on assumptions stated in those projects.
- The "about 12 of 22" hit rate is from one AI model in one session. It is an
  observation, not a measured property of AI.
