# Human Frontier Mission: all 54 challenges, where they are stuck, and how to get past it

*Research note, 24 November 2026, the day HFM launched. The data comes from HFM's
public live API, which was read the same day.*

## 0. Summary

The **Human Frontier Mission (HFM)** is run by the **Human Frontier Mission
Committee** and launched publicly on 24 November 2026. In it, domain experts
define research challenges with fixed evaluators, and autonomous AI research
agents propose, run and score experiments against them.

All 54 challenges were read from the public API on launch day. The snapshot
is in [`hfm-challenges-2026-11-24.csv`](hfm-challenges-2026-11-24.csv). It
shows **2,307 agent proposals and about $112k of compute spent**.

Three findings drive the recommendations:

1. **The agents are hill-climbing, not exploring.**
   - 1,923 of the 2,307 proposals (83%) are explicit edits of an earlier
     winning proposal ("keep p4053 unchanged, but…").
   - Across the 450 accepted improvements, the median gain is **0.15%**. 44%
     are below 0.1%, and 16% are below 0.01%.
   - In the most mature tracks, the median of the last five accepted gains
     is 2–8×10⁻⁵ relative. That holds for weather, Rayleigh–Bénard
     forecasting, photometric redshift, order-book ranking and S&P activity
     ranking. Lung-atlas annotation and both steel tracks are at about
     10⁻⁴.
2. **Big gains came only where the starting point was weak.** Examples are
   agent harnesses (ARC 12%→79%, Dafny 32%→83%, tables 37%→88%), crystal
   generation (13.6%→43.9% stable-novel-unique), regulatory DNA (Spearman
   0.45→0.71) and photometric redshift (−33% NRMSE). Where the starter was
   already strong, the agents improved it at the margin. The clearest case is
   the 24 h weather track: **−0.75% total, and nothing beyond post-processing.**
   The one attempt to fine-tune the model itself was worse than the baseline
   and was never revisited.
3. **The breakthroughs are in moves that the search loop structurally
   avoids.** These are changing the model class, adding a new data source or
   physical structure, or changing the objective so it matches what the
   challenge is really about. Each such move costs more than one experiment
   budget to set up, and it usually scores *worse* at first. A
   greedy loop that accepts only immediate improvements will never take one.

Sections 2–13 go through all 54 challenges, grouped by HFM's 12 fields. Each
entry gives the task, the score so far, what the agents have tried, the
**limit that binds**, and **the route most likely to give a step change**,
with a test that would confirm it. Section 14 covers changes to the research
process itself, which would help every track at once.

---

## 1. Sources, method and caveats

- **Data.** The directory, per-track overviews and summaries come from
  `https://hf-mission.com/api/discovery/live/snapshot` (sections `directory`,
  `overview`, `summary`), fetched 24 Nov 2026, about 13:50 UTC.
  - "Best" is the best `metricValue` among proposals whose `scoreAudit`
    passed, in the metric's own direction.
  - Baselines are HFM's own. A baseline of `0.0` on a "skill" or "instance
    score" metric means the score is normalised so that the reference method
    scores 0.
  - `None` means HFM publishes no baseline.
- **Status.** 22 tracks are active, 31 paused and 1 ended. Paused tracks
  keep their published history.
- **What is not verified.**
  - HFM's own claims (2,391 experiments, 12 fields, the committee
    affiliations listed on its homepage) come from its press release and site.
  - The API counts differ slightly: 2,307 proposals appear in the summaries.
  - The press release names no individuals and discloses no funding.
  - The aicell.io lab newsletter calls HFM "one to watch sceptically" and
    argues that *verification, not production, is the bottleneck*. The data
    below supports that view.
- **Scores were taken as published.** Scores were not re-run, and some
  tracks report scores on a practice set, not a hidden set. Section 14.2
  covers what that means.
- **Background facts not re-verified this session** (for example published
  behaviour of particular methods) are written as general
  knowledge and should be checked before citing.

## 2. Earth science (1)

### 2.1 24-Hour Global Near-Surface Weather Forecasting with ArchesWeather

- **Task:** 24 h forecasts of 2 m temperature, 10 m wind and mean sea-level
  pressure on the WeatherBench2 1.5° ERA5 grid. ArchesWeather is the
  starter, and 2022 is held out.
- **Score:** area-weighted normalised RMSE 0.098226 → **0.097486 (−0.75%)**.
  36 proposals, about $3.6k.
- **What the agents did:** a stack of causal bias corrections on the frozen
  model's output: geographic "innovation slopes", decay memories and
  winsorised residual updates. The last accepted gains are +2×10⁻⁷ to
  +3×10⁻⁶. One fine-tuning attempt (0.09837) did worse and was dropped.
- **Binding limit:** output post-processing cannot fix errors in the model's
  dynamics. At 24 h most of the remaining error is in the forecast itself,
  not a stationary bias.
- **Routes to a step change:**
  1. **Fine-tune properly, with a budget class above one experiment.** Use
     the full public 1979–2021 record, weight the loss toward surface
     variables, and train autoregressively over 4 × 6 h steps so that
     24 h is optimised directly. The single failed attempt used only
     2000–2009.
  2. **Ensemble across model families:** ArchesWeather plus a second open
     model (GraphCast-class or a diffusion model) with learned per-variable
     weights. Cross-family ensembles typically beat any single
     model's bias correction.
  3. **Residual diffusion or flow-matching correction** conditioned on the
     forecast and the static fields (orography, land-sea mask). It learns
     error *structure*, not a slope.
- **Confirming test:** beat 0.0975 on 2022 by more than seed-to-seed
  variation (≥1% relative), and report each variable separately.

## 3. Astronomy (1)

### 3.1 Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions

- **Task:** predict spectroscopic redshift from Legacy Survey g/r/z cutouts
  and catalogue photometry. Evaluation is on held-out *sky regions*.
- **Score:** NRMSE 0.0364 → **0.0243 (−33%)**. The last gains are about
  10⁻⁶–10⁻⁷.
- **What the agents did:** readouts on AION-1 embeddings blended with
  gradient-boosted trees. The recent edits tune tree depth and blend weight.
- **Binding limit:** spatial covariate shift. Region-to-region differences
  in depth, extinction and seeing change the photometry–redshift relation,
  and point regression cannot express catastrophic-outlier modes.
- **Routes:**
  1. **Predict a full redshift PDF** (classification over redshift bins or
     a mixture density) and take its mode or median. This idea is on the
     agents' list but was not pursued.
  2. **Train with region-level augmentation:** perturb depth, PSF and
     Galactic-extinction (E(B−V)) values to mimic unseen regions, and
     feed the extinction and depth maps in as inputs.
  3. **Fine-tune the image encoder end to end on 152² cutouts** with
     region-grouped cross-validation, not a frozen embedding.
- **Test:** NRMSE plus outlier fraction (|Δz|/(1+z) > 0.15) on held-out
  regions. PDF calibration should be checked with PIT histograms.

## 4. Biology (2)

### 4.1 Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA

- **Task:** predict lentiMPRA activity (log₂ RNA/DNA) of 200 bp inserts in
  HepG2, K562 and WTC11 cells.
- **Score:** assay macro-Spearman 0.447 → **0.708 (+58%)**. 65 proposals.
- **What the agents did:** supervised CNN ensembles with many seeds, ranking
  losses, and blending with a ridge readout on frozen Evo 2 features. The
  last gains are about +5×10⁻⁴.
- **Binding limit:** each cell line is learned largely on its own, and
  frozen Evo 2 features are only a ridge input.
- **Routes:**
  1. **One multi-task model across all three cell types** with shared
     sequence features and cell-specific heads. It exploits the shared
     transcription-factor grammar and lets HepG2/K562 data inform WTC11.
  2. **Parameter-efficient fine-tuning of Evo 2** (LoRA) on the MPRA
     objective, not ridge on frozen embeddings.
  3. **Reverse-complement-equivariant architecture plus test-time augmentation
     by shifting the insert inside the adapters.** These are well-established
     wins for MPRA models.
- **Test:** macro-Spearman with a per-cell breakdown. The gain on WTC11, the
  smallest assay, shows whether transfer works.

### 4.2 Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas

- **Task:** annotate cells from *unseen studies* in the full Human Lung Cell
  Atlas (about 2.3M cells, 35 studies). Splits are by study, with no donor
  in more than one split.
- **Score:** study macro-F1 0.781 → **0.854 (+9%)**. The last gains run from
  5×10⁻⁸ to 1×10⁻⁴.
- **What the agents did:** pseudo-labelling students on frozen UCE embeddings,
  donor-stratified selection quotas and temperature matching.
- **Binding limit:** a frozen general embedding plus a shallow head. Studies
  disagree on labels, and rare types dominate macro-F1.
- **Routes:**
  1. **Predict over the Cell Ontology hierarchy with abstention.** Back off
     to the parent term when uncertain. Ontology-aware loss and decoding
     reduce near-miss errors such as "AT1 vs AT2-like".
  2. **A second view of raw counts:** a highly-variable-gene model
     (scVI/scANVI-style) ensembled with UCE. Independent benchmarks show
     simple count-based models often match or beat frozen foundation-model
     embeddings across studies.
  3. **Fine-tune UCE's last layers with study-adversarial or group-DRO
     training**, so that the representation itself becomes study-invariant.
- **Test:** macro-F1 on held-out studies, F1 on the rarest decile of types,
  and calibration.

## 5. Medicine (1)

### 5.1 Patient-level lung cancer histology classification with Phikon-v2

- **Task:** adenocarcinoma vs squamous cell carcinoma from H&E slides.
  Training is on TCGA, development on CPTAC, and held-out evaluation on NLST.
- **Score:** patient AUROC 0.910 → **0.978**. 71 proposals.
- **What the agents did:** density and redundancy weighting of slides and
  patches on frozen Phikon-v2 features. Recent gains are about 10⁻⁴.
- **Binding limit:** AUROC is near its ceiling. What remains is the
  cohort-shift tail (NLST scanners and staining) and ambiguous cases.
- **Routes:**
  1. **Stain and scanner augmentation at the feature level,** plus a
     second foundation model (UNI- or Virchow-class) as an ensemble view.
     This attacks the cohort shift directly.
  2. **Attention-based multiple-instance learning with an explicit
     "uncertain" class and calibrated abstention.** At about 0.98 AUROC,
     the useful gain is knowing which cases to send to a pathologist.
- **Test:** report sensitivity at fixed high specificity, and accuracy on the
  cases the model keeps after abstaining. Beyond about 0.98, AUROC alone
  cannot separate methods.

## 6. Materials (1)

### 6.1 Crystal Generation for Unseen Chemical Systems (MatterGen + MatterSim)

- **Task:** for unseen 2–3-element systems, propose up to 64 structures.
  A structure counts if it is stable, novel and unique after MatterSim
  relaxation.
- **Score:** 0.136 → **0.439 (+222%)**. Gains are still large: +0.006 to
  +0.021 per step.
- **What the agents did:** deterministic exploration of more reduced
  formulas per parent ("try the 7th and 8th unused formula"). Coverage
  expansion is the winning lever.
- **Binding limit:** the 64-structure budget is spent on *generation*, not
  on *selection*.
- **Routes:**
  1. **Generate thousands of candidates and pre-screen cheaply.** Use a fast
     MatterSim single-point energy, symmetry and duplicate filters, and an
     energy-above-hull proxy, then spend the 64-structure budget only on
     the top candidates. This idea is on the agents' list but not done.
  2. **Choose compositions from phase-diagram logic:** enumerate
     charge-balanced stoichiometries, rank them by distance to the
     known hull, and seed generation from prototype substitution.
- **Test:** the stable-novel-unique fraction on unseen systems, and
  separately the hit rate against DFT for a sample, to guard against
  exploiting MatterSim's errors.

## 7. Physics and engineering (10)

These tracks share a pattern. A reusable *method* is scored by a normalised
skill (reference = 0) on withheld conditions, and the agents have converged on
hand-tuned physical sub-models adjusted one coefficient at a time.

### 7.1 Battery Voltage Model for Cold Operating Conditions

- **Task:** predict terminal voltage under current and temperature profiles.
  Calibration data is warmer than the test drive cycles.
- **Score:** skill 0 → **0.599**. 100 proposals, about $9k, 41
  improvements, each now about +10⁻⁴.
- **What the agents did:** fit equivalent-circuit sub-models (RC
  polarisation, OCV, thermal ratios) for each cell chemistry, one at a time.
- **Binding limit:** extrapolating to colder temperatures. Ad hoc thermal
  slopes have no physical basis outside the calibrated range.
- **Routes:**
  1. **Arrhenius-parameterised kinetics** (charge-transfer resistance and
     diffusivity ∝ exp(Eₐ/RT)) with a Butler–Volmer overpotential. One
     activation energy per process, identified from the temperature span
     available. Physical form is what extrapolates.
  2. **A reduced electrochemical model** (SPMe-class) with the same
     Arrhenius laws, plus a small learned residual that is bounded to
     vanish outside the calibrated range.
- **Test:** skill on the coldest held-out cycles specifically, reported by
  temperature bin.

### 7.2 Estimating Wave-Energy Resources from Short Measurement Campaigns

- **Task:** extend a short paired wave–wind record to a long wind-only
  period, then estimate the wave resource.
- **Score:** **0.560**. 100 proposals, about $8k. Recent gains are about
  10⁻⁴.
- **What the agents did:** direct sea-state regressions with censoring,
  gates and shrinkage.
- **Binding limit:** swell is generated by remote storms, not local wind, so
  any model driven only by local wind has an error floor.
- **Routes:**
  1. **Split wind-sea and swell explicitly.** Use fetch- and duration-limited
     growth laws for wind-sea, and treat swell as a persistent
     autoregressive or regime process whose climatology comes from the
     campaign.
  2. **Match the distribution, not the time series:** quantile mapping of
     the resource statistic (energy-flux distribution), which is
     what assessment uses.
- **Test:** skill on long-term mean power and the exceedance curve.

### 7.3 Steel Cyclic Loading History Transfer, and 7.4 Predict Cyclic Steel Stress Response from Limited Material Tests

- **Tasks:**
  - **7.3:** calibrate a material model from tension coupon tests and
    predict stress and energy under earthquake strain histories.
  - **7.4:** predict a steel's stress response to a strain history from one
    tension test plus a library of tests on other steels.
- **Scores:** **0.668** (100 proposals) and **0.545** (71 proposals,
  paused). Recent gains are about 10⁻⁴–10⁻⁶.
- **What the agents did:** Chaboche/Armstrong–Frederick back-stress models,
  Bauschinger transitions and donor-library averaging, tuned one coefficient
  at a time.
- **Binding limit:** a monotonic tension test does not identify the cyclic
  hardening parameters. They have to be *transferred* from other steels, and
  the agents do that by averaging donors.
- **Routes:**
  1. **A hierarchical Bayesian prior over constitutive parameters** learned
     from the donor library, conditioned on monotonic features (yield,
     ultimate strength, hardening exponent, yield-plateau length), and
     updated by the target coupon. Predict with the posterior. This is
     principled transfer with honest uncertainty.
  2. **A thermodynamically consistent learned constitutive model** (for
     example input-convex or internal-variable networks) trained across
     donors. Physics guarantees stability on long earthquake histories.
- **Test:** skill on held-out steels *and* protocols (LP9/LP10 are already
  used as independent checks), plus energy-dissipation error.

### 7.5 Predicting Optical Constants Beyond the Measured Wavelength Band

- **Task:** extrapolate n and k outside the measured band.
- **Score:** **0.791**. 100 proposals.
- **What the agents did:** Drude–Lorentz, Urbach and pole models chosen by
  material class, with careful choice of which models to keep.
- **Binding limit:** fitting n and k independently breaks causality.
- **Routes:**
  1. **Fit only Kramers–Kronig-consistent models** (sums of
     Tauc–Lorentz, Cody–Lorentz and Drude oscillators), and choose the
     oscillator count by held-out-band cross-validation inside the
     measured data.
  2. **A prior over oscillators from a materials database** (by material
     class), which supplies the out-of-band resonances the data cannot
     reveal.
- **Test:** skill as a function of distance from the band edge.

### 7.6 Optical Coating Design Under Manufacturing Variation

- **Task:** an anti-reflection stack that maximises expected performance
  under a process-variation model.
- **Score:** **0.252** (paused). Gains were still large, about 5×10⁻³ per
  step, when it stopped.
- **What the agents did:** Sobol sampling and thickness polishing with
  local material swaps.
- **Binding limit:** local search in a very multimodal design space.
- **Routes:**
  1. **Needle-optimisation / global topology search** for inserting layers,
     wrapped in robust optimisation that uses the analytic sensitivities
     of the transfer-matrix method (adjoint gradients of the expected score).
  2. **Prefer designs whose error sensitivity is low by construction,**
     for example quarter-wave-stable and graded-index structures.
- **Test:** the expected score and the worst decile under the variation
  model.

### 7.7 PM2.5 Sensor Calibration Transfer

- **Task:** calibrate low-cost particle sensors to reference PM2.5, judged
  on hourly values, monthly means and advisory tallies.
- **Score:** **0.695** (paused).
- **What the agents did:** humidity growth corrections, laser-health
  selection and temperature-adjusted gain.
- **Binding limit:** the aerosol type changes the optical response (κ
  hygroscopicity, size distribution), and that varies by site and season.
- **Routes:**
  1. **κ-Köhler hygroscopic growth with κ inferred per site** from the
     relationship between the two lasers and the sensor's particle-size
     bins.
  2. **Hierarchical (site-level random-effects) calibration** shared across
     stations, so a new site borrows strength.
- **Test:** leave-one-site-out skill on the advisory tallies.

### 7.8 64-Frame Rayleigh–Bénard Convection Forecasting with Walrus (The Well)

- **Task:** forecast 64 frames from 6 input frames, over 35 Rayleigh and
  Prandtl settings.
- **Score:** rollout NRMSE 0.369 → **0.313 (−15%)**. The last gains are
  about 10⁻⁵.
- **What the agents did:** LoRA adapters on Walrus, specialised by regime,
  with physical projections in the feedback loop.
- **Binding limit:** chaotic divergence. Past a few eddy turnovers, RMSE on
  a point forecast mostly rewards blurring toward the mean.
- **Routes:**
  1. **Train on full rollouts with spectral and conservation losses**
     (incompressibility, Nusselt-number consistency) and noise injection
     to stabilise long horizons.
  2. **Generate ensembles of rollouts** (diffusion or flow-matching
     refiners such as PDE-Refiner) and score them with CRPS alongside RMSE.
- **Test:** NRMSE plus energy-spectrum error and Nusselt error at frame 64.

### 7.9 Full-field Reconstruction of Rayleigh–Bénard Convection from 256 Sensors

- **Task:** reconstruct the whole field, including pressure, which is never
  measured, from 256 point sensors over 8 frames.
- **Score:** field NRMSE 0.292 → **0.234 (−20%)**.
- **What the agents did:** kernel regressions on gappy-POD features with
  pressure mixtures.
- **Binding limit:** linear and kernel maps from sparse points cannot
  recover small-scale plumes. Pressure needs physics.
- **Routes:**
  1. **A learned reconstruction network** (U-Net or transformer over a
     Voronoi tessellation of the sensors) trained across settings. This idea
     is on the agents' list but not done.
  2. **Recover pressure from the pressure-Poisson equation** applied to the
     reconstructed velocity, not by regression.
- **Test:** NRMSE per variable, with pressure reported separately.

### 7.10 Wind-Farm Wake Calibration and Power Prediction from Real SCADA

- **Task:** calibrate FLORIS on Kelmarsh and Penmanshiel SCADA, then predict
  per-turbine power for unseen periods and an unseen farm.
- **Score:** reward **0.412**. 6 proposals, so it has barely started.
- **What the agents did:** a physics skeleton with shrinkage corrections, and
  a per-turbine gradient-boosting residual has been proposed.
- **Binding limit:** the inflow is poorly known (wind-direction bias,
  veer and stability), and residual models will not transfer to the unseen
  farm.
- **Routes:**
  1. **Calibrate inflow first:** correct yaw and nacelle-direction bias, then
     condition wake expansion on stability (turbulence intensity and shear
     taken from SCADA). Calibrate only physically meaningful parameters so
     that they transfer.
  2. **Validate the transfer with leave-one-farm-out testing** before
     adding any machine-learned residual.
- **Test:** nMAE on the unseen farm specifically.

## 8. Optimisation and operations (9)

The agents have built hand-engineered heuristic policies, still incremental
but with larger gains than the physics tracks.

| Challenge | Best | What the agents built | Route to a step change |
| --- | --- | --- | --- |
| **Toronto Arterial Signal Timing Coordination** (6–7 signals, one counted hour, 50 evaluator queries) | 0.989 | Offset and split probing with zero-sum green swaps | Near its ceiling under this metric. The value now is in **robustness**: optimise for a *distribution* of demand around the counted hour, not the one hour. The test should be the same plan under ±20% demand. |
| **Online Single-Track Railway Dispatcher** (Finnish line, 16 sections) | 0.398 | Pairwise veto and pricing rules for meets | **Rolling-horizon exact optimisation** (MILP or constraint programming over meet and pass orders, with a 10–20 minute look-ahead). Single-track meet scheduling is small enough for exact solvers, and the gap to 1.0 suggests greedy rules are leaving a lot behind. |
| **Citywide Bicycle Rebalancing** | 0.802 | First-passage probabilities and parametric order scoring | **Stochastic look-ahead routing:** a fluid model or sampled demand scenarios feeding a vehicle-routing solver for the van tour, re-planned at each stop. |
| **Columbia/Snake River Dam Spill Optimisation** (8 dams, juvenile salmon) | 0.720 | Five-action expert sets, soft-regret weighting | **Coordinated model-predictive control across the cascade.** Optimise spill jointly at all 8 dams over the travel horizon of the fish, with total-dissolved-gas constraints as hard limits. The current controller decides dam by dam. |
| **Online Cloud Resource Allocator** (Azure trace) | 0.762 | Survival-curve lifetime estimates, shortlist scoring | **Learned lifetime distributions conditioned on VM metadata,** used in a best-fit-by-expected-departure packing rule. Lifetime-aware packing is where the large wins are in the scheduling literature. |
| **Ride-Hail Matching and Repositioning** (NYC) | 0.989 | Repricing and ordering repositioning moves | Near its ceiling. Test **robustness to demand shocks** before investing further. |
| **Control Policy for Frequent Bus Services** (HSL) | 0.750 | Holding heuristics with empirical link-error strata | **Headway-based holding with predictive control:** hold to equalise *forecast* headways over the next stops, using the per-link travel-time distributions the agents already estimate. |
| **Rolling Unit-Commitment and Dispatch** (OPSD load and wind, with storage) | 0.534 (6 proposals) | Rolling MILP; let baseload drop to minimum output, with storage covering reserve | **Stochastic unit commitment over wind scenarios** sampled from historical day-ahead forecast errors, with a storage end-of-horizon value. The current plan uses fixed reserve rules. Measure regret against perfect foresight. |
| **Wireless 400-Cell Synthetic MILP** | no proposals yet | none yet | Start from the structure of the problem: decompose by cell cluster (Lagrangian relaxation), warm-start HiGHS from a greedy solution, and use a large-neighbourhood search to fix most variables and re-solve the rest. |

## 9. Infrastructure (1)

**Single-Machine Weighted Tardiness with Release Times.**
- **Score:** gain **1.211**. 45 proposals.
- **What the agents did:** the winning edit was raising a CPU-time guard
  from 95% to 97%, which is a sign the search is near its limit.
- **Binding limit:** the search under a one-second wall clock.
- **Route:** the state of the art for this problem is **iterated local search
  or dynasearch with fast O(1) move evaluation** (precomputed prefix
  completion times), seeded with ATC. Add-ons are memetic recombination
  for large instances and a bounded exact DP for small ones.
- **Test:** gap to best-known solutions on standard OR-Library-style
  instances.

## 10. Quantitative finance (8)

| Challenge | Best | Observation | Route |
| --- | --- | --- | --- |
| **S&P 500 Minute Trading Activity Ranking** | 0.167 → 0.463 | 100 proposals, about $9.5k, last gains about 10⁻⁶ | Saturated with hand-made features. Replace them with **one cross-sectional sequence model** (a transformer over the 60-minute bars of all members) trained on a ranking loss, using intraday seasonality as an input, not a hand-built normaliser. |
| **Nasdaq Order-Book Depth Exhaustion Ranking** | 0.200 → 0.377 | Last gains about 10⁻⁵ | **Event-time, order-flow-imbalance models across depth levels** (multi-level order-flow imbalance, queue depletion hazard). Model the arrivals, not the snapshots. |
| **Event-Driven Equity on Filings** | reward 1.0 | Score is pinned at the ceiling; individual gains of 0.23 / 0.68 / 0.09 | **The reward is capped or saturated**, so the evaluation can no longer rank methods. Fix the metric first: use uncapped net-of-cost Sharpe on a later out-of-sample window, with an explicit test against overfitting. |
| **A-Share Intraday Overlay** | reward 1.0 | Same pinned score | Same: fix the evaluator before searching further. |
| **Execution Policy from Ten-Level Order Book** | reward 1.0 | Same pinned score; 11 proposals | Same, then optimal-execution control (Almgren–Chriss-style with learned fill probabilities). |
| **Cross-Venue Perpetual Order Allocation** | 0.593 | Residual trees, gains about 10⁻⁵ | A **venue-cost model with fill and adverse-selection terms,** solved as a small convex allocation per parent order. |
| **Minute-Level Perpetual Strategy** | 0.078 | Funding-carry timing | Low and noisy. Pre-register a single hypothesis family (funding-settlement carry) and test it on a fresh window before tuning more. |
| **Cost-Aware Long-Only Portfolio (Qlib, CSI 300 → unseen US)** | 0.212 | Ridge alpha with threshold rebalancing | **Transaction-cost-aware mean–variance optimisation with a shrinkage covariance,** and alpha factors tested for whether they transfer across markets. |

Across finance, **three of eight tracks score exactly 1.0.** A capped or
saturated reward cannot rank methods, so those tracks need their evaluators
fixed before any further agent search. In general, gains in finance tracks
must be tested on time windows later than every experiment, because
2,000-proposal searches will find spurious patterns.

## 11. Agent harnesses (6)

These are the tracks with the largest relative gains. A frozen model is given
a better scaffold.

| Challenge | Score | Winning move | Route to the next step change |
| --- | --- | --- | --- |
| **ARC Grid Transformation** | 0.121 → **0.786** | Program synthesis with repair, multi-view augmentation | **Test-time training on each task's examples,** combined with program search. That combination is the known frontier on ARC. Also use **verified program voting** across the eight symmetry views. |
| **Verified Programming with Dafny** | 0.316 → **0.829** | Stop repair loops when stalled, restart fresh drafts | **Decompose into invariants first:** generate loop invariants and lemmas separately, check each with Dafny, then compose. Most remaining failures are proof obligations, not code. |
| **Formal Theorem Proving in Lean** | 0.364 → **0.566** | More independent repair requests | **Whole-proof sampling plus search guided by tactic state** (best-first over proof states using Lean REPL feedback), and **lemma caching across theorems**. |
| **Questions over Real Tables** | 0.371 → **0.877** | Expose exact data types to the model | **Program-of-thought with self-consistency:** generate several pandas programs and vote on executed answers. |
| **Preserving Knowledge from Research Papers** | 0.241 → **0.479** | Consolidation prompts, BM25 retrieval | **Anticipate likely questions:** build notes as dense fact tables (entity, number, condition) in the reader's likely answer format. |
| **Agent Harness Evolution (Terminal-Bench 2)** | 0.536 (8 proposals) | Wall-clock awareness and an "always have something deliverable" contract | Agree. Next: **verifier-first loops** (write and run the task's tests before declaring done). The agents also note the sealed readout has about ±6 points of noise, so require repeated runs before promoting a change. |

## 12. Post-training (7)

**SynLogic** (0.378), **EmbSpatial** (0.785), **FabNER** (80.7 F1),
**MapQA** (0.828), **ScienceWorld** (35.8), **TOPv2** (83.0 exact match), and
**Agentic RL with verl** (0.367).

- **What the agents did:** LoRA rank, batch size, label smoothing and token
  weighting, all small-step recipe search. None of these tracks has a
  published baseline, so "improvement" is undefined.
- **Binding limits:** small LoRA, a single GPU and a fixed time budget. The
  data matters more than the hyper-parameters.
- **Routes:**
  1. **Data, not knobs.** Synthesise verified training data: SynLogic
     generators with checkers, programmatic ScienceWorld trajectories,
     and grammar-sampled TOPv2 trees. The agentic RL track's own winning
     hypothesis, procedurally generated Docker tasks with executable tests,
     is the template.
  2. **Rejection-sampling fine-tuning** (generate, verify, train on what
     passes) before GRPO, so that RL starts from a policy that already
     succeeds sometimes.
  3. **Publish a base-model score for every track,** so progress can be
     measured at all.

## 13. Pretraining (7)

Seven tracks: looped and recurrent-depth transformers, architecture search
from scratch, and pretraining under an inference-cost constraint.

- **What the agents did:** many proposals fail to produce a scorable
  checkpoint. For example, 20 proposals but only 2 scored in the 168-hour
  looped track. The scored ones are function-preserving expansions,
  data-mix changes and throughput fixes. None has a baseline.
- **Binding limit:** engineering reliability and throughput, not ideas.
  One winning hypothesis is just "use the idle GPU memory to raise
  throughput 20%".
- **Routes:**
  1. **A known-good, efficient recipe as a fixed floor** (Muon or AdamW,
     QK-norm, a warmup–stable–decay schedule, fused cross-entropy, full
     `torch.compile`). Only architecture ideas are compared on top of it.
  2. **For looped models,** use loop-index embeddings with a separate norm
     for each loop, plus random loop counts and deep supervision at
     training time. The agents' own best idea in that direction targets a
     documented failure: math BPB stalls in the reference looped model.
  3. **Scaling-law extrapolation from short runs** (fit loss against
     compute on 3–4 sizes), so a 24-hour budget ranks ideas by their
     slope, not by one noisy endpoint.

## 14. Cross-cutting: changing the research process itself

These would raise the ceiling of every track at once.

### 14.1 Escape the local optimum

- **Budget a share of proposals for structural changes** (for example
  20–30%). These would change the model class, the data source or the
  physics, and must *not* be framed as edits of the current best. Judge
  them on a longer clock.
- **Give such proposals a larger budget class.** The weather fine-tune
  failed under the same budget as a coefficient tweak.
- **Keep a quality-diversity archive** of distinct approaches (MAP-Elites
  style) instead of a single champion. Recent work reports that AI research
  agents narrow exploration, and the 83% figure above is that narrowing,
  measured.
- **Stop or redirect a track automatically** when its last *k* accepted
  gains fall below the measured noise floor. Five tracks already have median
  recent gains below 10⁻⁴ relative, and three more are just above it.

### 14.2 Make every claimed gain real

- **Promote a change only when it beats seed-to-seed variation.** Gains of
  10⁻⁷ cannot be distinguished from re-run noise, and the harness track
  itself reports ±6-point noise on a sealed readout.
- **Correct for multiple comparisons.** 100 proposals against one practice
  set will find practice-set artefacts. Confirm on a fresh hidden split
  before the result is shown as "best".
- **Fix evaluators that saturate.** Three finance tracks are pinned at 1.0.
  Post-training and pretraining tracks need published baselines.

### 14.3 Use what the challenges already reward

Most challenges are defined by **a generalisation gap**: a new sky region,
new study, new cohort, colder temperature, unseen farm, unseen steel,
out-of-band wavelength, or unseen chemical system. The proposals that look
most promising in this note all put in **structure that carries across the
gap**:
- physical laws: Arrhenius kinetics, Kramers–Kronig, Poisson pressure and
  wave-growth laws;
- hierarchical priors across donors, sites and studies;
- explicit invariances: reverse-complement, symmetry views, stain
  augmentation.

The agents' default is tuning the residual on the practice set, which is
exactly the approach that does not transfer across those gaps.

## 15. Can the limits be beaten from outside HFM?

An attempt was made to go beyond analysis and beat the published results
directly. As of launch day, that is not possible for any of the 54 tracks:

- **HFM publishes no task data, evaluators or submission path.**
  - The API exposes directories, proposals, scores and agent chat, but no
    datasets, `instruction.md` files or grading code.
  - The agents' own logs show that verification runs inside private
    sandboxes ("launch the verifier exactly once").
  - Outside participation is limited to three things, all needing a
    Novix account: joining a challenge's "Research Exchange" conversation,
    proposing a new challenge for committee review, and offering funding,
    compute, data or review time. Outsiders cannot submit solutions for
    scoring.
  - The practical route to HFM's real scoreboard is therefore to post
    testable proposals for its agents to run. A ready-to-post set is in
    [`hfm-exchange-proposals.md`](hfm-exchange-proposals.md).
  - So no externally produced result can be scored on HFM's metrics, and
    any claim to have "surpassed" an HFM score would be unverifiable.
- **Compute.** More than 30 tracks need GPUs: all of pretraining and
  post-training, plus the ArchesWeather, Walrus, Evo 2, MatterGen, Phikon-v2,
  UCE and AION-1 tracks. This analysis ran on a 4-CPU container with no GPU,
  later extended with free CPU machines on GitHub Actions (see below).

What *can* be tested locally is whether a proposed route works on a
public reproduction of a fully specified task. That was done for
eleven tracks where the task can be rebuilt from public material:

**Weighted tardiness with release times**
([`hfm-projects/infrastructure/weighted-tardiness/tardiness.py`](hfm-projects/infrastructure/weighted-tardiness/tardiness.py)).
- **Setup:** 60 generated instances (n = 40 and 100, three tardiness and
  due-date-range settings, 10 seeds each), with the same one-second
  budget as HFM.
- **Result:** iterated local search seeded by ATC reduced total weighted
  tardiness by **12.5% on average versus ATC with k = 2**, the dispatch
  rule HFM's track starts from. It reduced it by **6.8% versus ATC with
  k tuned per instance**, and it never did worse on any instance.
- **What this does and does not show:** it confirms the direction
  recommended in Section 9. It does **not** beat HFM's incumbent (gain
  1.211), which already includes local search and is scored on private
  instances with an undisclosed normalisation.

**Optical constants beyond the measured band**
([`hfm-projects/physics/optical-constants/optical_extrapolation.py`](hfm-projects/physics/optical-constants/optical_extrapolation.py)).
- **Setup:** 620 tabulated (n, k) records from the public
  refractiveindex.info database. The middle of each record's wavelength
  range is treated as measured, and both ends are withheld. Settings are
  chosen on one half of the materials and results reported on the other
  half. Skill is 1 − error ÷ the error of holding the last measured value.
- **Method tested:** the Kramers–Kronig-consistent fit from Section 7.5,
  blended part-way toward the hold-last reference.
- **Result: a modest and fragile gain.**

  | Settings search | Skill on the tuning half | Mean skill, held-out | Median, held-out | Materials improved |
  | --- | --- | --- | --- | --- |
  | First search | +0.25 | **+0.11** | +0.18 | 77% |
  | Wider search | +0.31 | **+0.02** | +0.13 | 65% |

  - Naive linear extension scored −0.20, and the unblended fit scored
    about −0.06.
  - The wider search did better on the tuning half and *worse* on the
    held-out half. That is the practice-set overfitting described in
    Section 14.2, reproduced at small scale.
  - The held-out half was looked at twice, so even +0.11 is slightly
    optimistic.
- **What this shows:** causality alone helps the typical material but
  fails badly on a minority, mainly where a resonance lies outside the
  band and nothing inside the band reveals it. Route 2 in Section 7.5
  (a prior over oscillators by material class) is therefore the part
  that matters, and it is what HFM's agents have built by hand. Section
  7.5's ordering of the two routes should be read with that in mind.
  The result is not comparable to HFM's 0.791, which uses private samples
  and scoring.

**Photometric redshift across disjoint sky regions**
([`hfm-projects/astronomy/photometric-redshift/photoz_regions.py`](hfm-projects/astronomy/photometric-redshift/photoz_regions.py)).
- **Setup:** 198,350 galaxies with measured redshifts from the public SDSS
  DR17 catalogue, in 42 sky regions. Five-fold cross-validation with whole
  regions held out, so every galaxy is predicted by a model that never saw
  its region. Catalogue brightness only, no images, so errors are larger
  than HFM's.
- **Result:**

  | Method | NRMSE | Change | Core scatter (NMAD) | Outliers | Folds improved |
  | --- | --- | --- | --- | --- | --- |
  | Boosted-tree regression (base) | 0.04280 | — | 0.02342 | 1.22% | — |
  | Same, different random seed | 0.04285 | +0.10% | 0.02344 | 1.22% | 2/5 |
  | + Galactic extinction as an input | 0.04225 | **−1.29%** | 0.02251 | 1.21% | **5/5** |
  | Redshift distribution, mean | 0.04380 | +2.33% | 0.02226 | 1.37% | 0/5 |
  | Redshift distribution, median | 0.04506 | +5.27% | 0.02236 | 1.32% | 0/5 |
  | Blend of regression and distribution | 0.04238 | −0.99% | 0.02209 | 1.24% | 4/5 |

- **What this shows:**
  - **Extinction as an input is a real gain:** −1.29%, in every fold,
    against seed noise of 0.10%. It is small, but about 13 times the noise.
  - **Predicting a distribution does not lower NRMSE.** It tightens the
    core scatter by about 5% and makes the outliers worse, so on an RMSE
    metric it loses. Route 1 in Section 3.1 is not supported for HFM's
    metric. It would help only if the metric rewarded core scatter.

**Wave-energy resource from a short campaign**
([`hfm-projects/physics/wave-energy-resource/wave_resource.py`](hfm-projects/physics/wave-energy-resource/wave_resource.py)).
- **Setup:** 16 NOAA buoys (Pacific, Atlantic, Gulf of Mexico, Hawaii),
  2021–2023. A 60-day campaign has wind and waves, and the next 365 days
  have wind only. 200 campaign instances. The reference assumes the
  campaign's mean wave power holds for the year. No settings were tuned.
- **Result:**

  | Method | Error in annual mean power | Skill, annual mean | Skill, hourly | Stations improved |
  | --- | --- | --- | --- | --- |
  | Reference | 38.5% | 0 | 0 | — |
  | Log-regression on wind | 29.3% | **+0.216** | +0.128 | 14/16 |
  | Boosted trees on wind | 29.6% | +0.221 | +0.094 | 14/16 |
  | Additive swell + wind-sea model | 30.9% | +0.182 | +0.124 | 14/16 |

- **What this shows:** using the wind record cuts the error in the annual
  resource by about a fifth, at 14 of 16 sites. The physically structured
  swell-plus-wind-sea model from Section 7.2 does **no better** than a
  plain regression. With local wind as the only long-term input, swell
  can only enter as a constant, so the structure adds nothing. A real
  step change would need a swell input from outside the site, such as a
  public wave hindcast. That is a change of data source, not of model.

**Day-ahead unit commitment under wind uncertainty**
([`hfm-projects/optimization/unit-commitment/unit_commitment.py`](hfm-projects/optimization/unit-commitment/unit_commitment.py)).
- **Setup:** German hourly load, the official day-ahead load forecast and
  actual wind from Open Power System Data. A German-scale invented fleet
  (17 slow thermal units, a fast peaker, storage) is committed each
  midnight on forecasts, then dispatched against what really happened.
  The reserve rule is tuned on 28 days of 2018. Results are for 42 days
  in six weeks of 2019. Regret is cost above perfect foresight
  (1,016 million EUR).
- **Result:**

  | Controller | Regret, million EUR | Versus tuned rule | Weeks cheaper |
  | --- | --- | --- | --- |
  | Reserve rule, no wind reserve | 370.2 | +644% | 1/6 |
  | Reserve rule, tuned on 2018 | 49.7 | — | — |
  | Stochastic, 10 scenarios **spread** from worst to best past error | 38.2 | **−23%** | 5/6 |
  | Stochastic, 20 spread scenarios | 37.6 | **−24%** | 5/6 |
  | Stochastic, 10 **random** scenarios, four draws | 23.0 / 38.6 / 265.0 / 339.9 | −54% / −22% / +433% / +583% | — |
  | Stochastic, 20 random scenarios | 190.6 | +283% | 3/6 |

- **What this shows:**
  - **Stochastic commitment beats a tuned reserve rule by about a
    quarter of the regret,** but only when the scenarios always include
    the extreme low-wind cases.
  - **With randomly drawn scenarios the method is dangerous.** Two of
    four draws were four to six times worse than the simple rule,
    because a draw that misses the bad-wind tail leaves the system short
    and it pays the load-shedding price.
  - A single run of the random version would have reported either a 54%
    gain or a 583% loss. This is the same lesson as Section 14.2: one
    scored run is not evidence.
  - The fleet and the wind forecast are invented, so the size of the
    gain will not carry over to HFM's task. The ordering of the methods
    is the finding.

**Cyclic steel response from a tension test**
([`hfm-projects/physics/steel-cyclic-from-tension/steel_cyclic_transfer.py`](hfm-projects/physics/steel-cyclic-from-tension/steel_cyclic_transfer.py)).
- **Setup:** the public Hartloper et al. coupon-test database (Zenodo
  6965147), which uses the LP1–LP10 loading protocols named in HFM's logs
  and is very likely the data behind both steel tracks. 38 steels from 21
  test campaigns. For each target steel only its tension test is known;
  steels from other campaigns are the donor library. Error is the RMS
  stress error on the target's cyclic tests as a share of yield stress.
- **Result:**

  | Method | Error | Versus donor median | Campaigns better than donor median |
  | --- | --- | --- | --- |
  | Tension test alone (reference) | 0.545 | +275% | 1/21 |
  | Donor median | 0.145 | — | — |
  | Donor median, refined on the tension test | 0.146 | +0.6% | 8/21 |
  | Prior conditioned on tension-test features | 0.169 | +16% | 7/21 |
  | Conditioned prior, refined on the tension test | 0.138 | −5.2% | 10/21 |
  | Oracle (fitted to the target's own cyclic tests) | 0.087 | −40% | 21/21 |

- **What this shows:**
  - **The donor library does almost all the work.** Borrowing the median
    cyclic parameters of other steels cuts the error by 73% against using
    the tension test alone.
  - **The conditioned prior proposed in Section 7.3 is not supported.**
    Its best variant is 5% better on average but wins in only 10 of 21
    campaigns, which is not distinguishable from chance.
  - **40% of the error is still there to remove,** and a tension test
    does not carry the information to remove it. The step change would
    come from an extra measurement per steel, not from a better prior.
    The follow-up below shows it has to be several load cycles.

**Anti-reflection coatings under manufacturing variation**
([`hfm-projects/physics/robust-ar-coating/robust_ar_coating.py`](hfm-projects/physics/robust-ar-coating/robust_ar_coating.py)).
- **Setup:** standard thin-film reflectance calculation, six
  specifications (three substrates, two wavelength bands), a shared pool
  of candidate designs with 4, 6 or 8 layers, and thickness and index
  errors on every layer. Final designs are scored on 4,000 fresh
  variation samples.
- **Result:** against the usual practice of picking the design with the
  lowest nominal reflectance:
  - **Choosing among the same candidates by their score under
    variation** cut expected reflectance by **14.5% on average**. It cut
    it by 33–50% in two specifications and 5% in a third, and changed it
    by under 0.5% in the other three.
  - **Re-optimising thicknesses on the robust score added nothing more**
    (−13.8% on average). It cut the 90th-percentile reflectance by 18%.
- **What this shows:** the gain comes from picking a less sensitive
  design, usually one with fewer layers, not from polishing thicknesses.
  HFM's agents spend most of their effort on polishing. Section 7.6 is
  supported on the selection point.

**Online VM placement on the Azure packing trace**
([`hfm-projects/optimization/cloud-vm-placement/cloud_placement.py`](hfm-projects/optimization/cloud-vm-placement/cloud_placement.py)).
- **Setup:** Microsoft's public Azure Packing Trace 2020. Four disjoint
  tenant samples of about 100,000 VMs each are replayed over 14 days.
  Cost is server-days. There are no evictions, unlike HFM's objective.
- **Result:**

  | Policy | Server-days versus best fit |
  | --- | --- |
  | First fit | about −3% (better in all four samples) |
  | Lifetime-aware, lifetimes predicted from finished VMs | **+1.3%** (better in 1/4) |
  | Lifetime-aware, true lifetimes | **−13.8%** (better in 4/4) |

- **What this shows:** packing VMs that leave together is worth about
  14% of server time, and it gets within a few percent of the lower
  bound, **but only if lifetimes are known**. With a simple predictor
  (median lifetime of the same tenant and VM type) the gain disappears.
  The limit is lifetime prediction, not the packing rule. This agrees
  with what HFM's agents are doing, which is building survival-curve
  estimates.

**Wind-farm wake calibration and power prediction**
([`hfm-projects/physics/wind-farm-wakes/wind_wake.py`](hfm-projects/physics/wind-farm-wakes/wind_wake.py)).
- **Setup:** the same two farms HFM uses, Kelmarsh and Penmanshiel, 2017
  10-minute SCADA from Zenodo. A simple Jensen wake model stands in for
  FLORIS. Each farm gets a 60-day calibration window and is tested on the
  rest of the year. All choices were made on Kelmarsh; Penmanshiel is the
  unseen farm. Error is mean absolute power error as a share of rated
  power.
- **Result:**

  | Method | Kelmarsh, vs no-wake | Penmanshiel (unseen), vs no-wake | Penmanshiel, new wind directions |
  | --- | --- | --- | --- |
  | Wake model, default settings | −27.3% | −1.4% | 0.0810 |
  | Wake model, direction offset and decay calibrated | −27.3% | **−11.9%** | 0.0620 |
  | + per-turbine sector factors | −31.6% | −19.8% | 0.0560 |
  | Boosted trees per turbine (no physics) | −36.0% | −29.3% | 0.0567 |
  | Calibrated wake model + boosted-tree residual | −35.7% | **−29.8%** | **0.0523** |

- **What this shows:**
  - **Calibrating the wind direction matters most for the physics
    model.** At the unseen farm a 9° direction offset turned a 1.4% gain
    into 11.9%. This supports the first half of Section 7.10.
  - **The second half of Section 7.10 is wrong.** It predicted that
    machine-learned residuals would not carry to unseen conditions. In
    fact trees trained on each farm's own calibration window beat the
    physics model on both farms, and physics plus a learned residual was
    best, including in wind directions the calibration window barely
    covered.
  - The Jensen model is cruder than FLORIS, so physics may be
    undersold here.

**Regulatory DNA activity in three cell lines**
([`hfm-projects/biology/regulatory-dna-mpra/mpra_multitask.py`](hfm-projects/biology/regulatory-dna-mpra/mpra_multitask.py)).
- **Setup:** the public lentiMPRA tables of Agarwal et al. (Zenodo
  10558183), the same assay and cell lines as HFM's task. A small CNN
  trained from scratch on CPU, at most 40,000 training sequences per cell
  line, the authors' fold 10 as the test set, two seeds. Sequences in any
  cell line's test fold are removed from all training sets, because the
  libraries overlap and their folds differ.
- **Result (Spearman on the test fold):**

  | Model | HepG2 | K562 | WTC11 | Macro |
  | --- | --- | --- | --- | --- |
  | One model per cell line | 0.549 | 0.554 | 0.504 | 0.535 |
  | + reverse-complement averaging | 0.566 | 0.571 | 0.519 | 0.552 |
  | One multi-task model | 0.595 | 0.589 | 0.549 | 0.578 |
  | Multi-task + reverse-complement averaging | 0.610 | 0.604 | 0.563 | **0.593** |

- **What this shows:** both ideas from Section 4.1 hold. Multi-task
  training adds 0.042 and reverse-complement averaging adds 0.015. Each
  gain appears in every cell line and both seeds (seed-to-seed spread is
  about 0.005 to 0.010). The largest multi-task gain is on WTC11, the
  smallest dataset, as predicted. Absolute scores are far below HFM's
  0.708, because this model is small and uses no Evo 2 features.

**Perpetual-futures strategies on a later period**
([`hfm-projects/quant-finance/perpetual-futures-strategy/perp_funding_carry.py`](hfm-projects/quant-finance/perpetual-futures-strategy/perp_funding_carry.py)).
- **Setup:** Binance's public archive, 24 large USD-margined perpetual
  contracts, 2022–2025, 8-hour rebalancing. 36 market-neutral settings
  from three signal families (funding carry, taker-flow reversal, price
  reversal) are ranked on 2022–2023 and then scored on 2024–2025, net of
  4 basis points per unit of turnover.
- **Result:**
  - **Families persist.** Funding carry is the only family with
    profitable settings in either period. Flow and price reversal lose
    heavily in both, because trading costs exceed the signal.
  - **6 of the 8 settings that were profitable in 2022–2023 stayed
    profitable in 2024–2025.** The rank correlation between the two
    periods is +0.91 across all settings and +0.63 inside the
    funding-carry family.
  - **The single best setting is not reliable.** The best on development
    went from a Sharpe ratio of 0.79 to 0.63. The second best went from
    0.62 to −0.04.
- **What this shows:** on this small, pre-declared grid the
  later-period check confirms the family and not the fine tuning. That
  is the rule Section 10 proposes. It is a study of evaluation, not a
  trading result.

**Follow-up experiments inside three projects**

- **Steel** ([project](hfm-projects/physics/steel-cyclic-from-tension/README.md)):
  one extra load reversal on the target steel closes only 8% of the gap
  to the oracle, and four reversals close 30%. The earlier suggestion
  that one cheap reversal would be enough was too optimistic.
- **Cloud placement** ([project](hfm-projects/optimization/cloud-vm-placement/README.md)):
  lifetime predictions within about a factor of 3 would save about 8% of
  server time. The simple predictor is off by about a factor of 10.
  Coarse lifetime classes do not help even with true lifetimes.
- **Weighted tardiness** ([project](hfm-projects/infrastructure/weighted-tardiness/README.md)):
  with a compiled search, 40-job instances are very likely solved (two
  independent methods agree on all 18 instances), and 100-job instances
  gain under one point from ten times more search.

**Summary of the eleven public-data tests**

| Challenge | Idea tested | Verdict |
| --- | --- | --- |
| Weighted tardiness | Iterated local search on top of the dispatch rule | **Supported:** 12.5% less tardiness than the starting rule, never worse |
| Grid unit commitment | Commit against wind scenarios instead of a reserve rule | **Supported with a condition:** −23% regret with spread scenarios; random scenarios can be far worse |
| Optical coating design | Select designs by score under variation | **Supported:** −14.5% expected reflectance; polishing adds nothing more |
| Regulatory DNA | One multi-task model for all cell lines | **Supported:** +0.042 Spearman, every cell line and seed |
| Regulatory DNA | Reverse-complement averaging | **Supported:** +0.015 Spearman |
| Photometric redshift | Extinction as an input | **Supported, small:** −1.3% error in every fold |
| Wind-farm wakes | Calibrate wind direction before anything else | **Supported:** −1.4% becomes −11.9% at the unseen farm |
| Wind-farm wakes | Avoid machine-learned residuals | **Refuted:** physics plus learned residual is best (−29.8%) |
| Perpetual futures | Judge tuned settings on a later period | **Supported as a rule:** families persist, fine tuning does not |
| Photometric redshift | Predict a redshift distribution | **Not supported** for an RMSE metric: +2.3% error |
| Cyclic steel response | Prior conditioned on tension-test features | **Not supported:** no reliable gain over the donor median |
| Cloud VM placement | Lifetime-aware packing | **Conditional:** −13.8% with true lifetimes, no gain with a simple predictor |
| Wave energy | Separate swell and wind-sea model | **Not supported:** no better than plain regression on wind |
| Optical constants | Causality-consistent (Kramers–Kronig) fit | **Weak:** +0.02 to +0.11 skill, fragile |

Eight of fourteen ideas held up, one depends on a prediction the test
could not make well, and five did not hold. The pattern is consistent:
- **Ideas that failed added modelling structure where information was
  the limit.** Local wind says nothing about swell. A tension test says
  little about cyclic hardening. In-band data says nothing about
  out-of-band resonances.
- **Ideas that worked changed the decision procedure or the selection
  rule,** shared data across related tasks, or added a missing input:
  local search, scenario planning, choosing designs by robust score,
  multi-task training, extinction, direction calibration.
- **Physics and learning work best together.** In the wind test the
  hybrid beat both pure physics and pure learning.
- **Where the oracle gap is large** (steel 40%, cloud 14%), the next
  step is a new measurement or a better forecast, not a better model
  of the existing inputs. The follow-ups put numbers on that: steel
  needs several load cycles, and cloud needs lifetimes right within
  about a factor of 3.

**Simulators and free compute (added later)**

- **Shared simulation core:**
  [`hfm-projects/_sim/core.py`](hfm-projects/_sim/core.py) is a small
  event-driven replay framework. It gives every operations project the
  same pieces: a policy hook, look-ahead by simulating a decision forward
  ("rollout") using only what is known at that moment, and paired
  comparison of policies on identical instances.
- **Simulators built on it:** rail dispatching on the real
  Lielahti–Pohjois-Louko line, bike rebalancing on Helsinki trips, and
  bus holding on the five HSL trunk routes. Their results are in the
  project folders and in the table below.
- **Free compute:** this repository is public, so GitHub Actions gives
  free 4-core machines with open internet. A workflow runs job scripts
  from [`compute/jobs/`](compute/jobs) and commits logs to
  [`compute/results/`](compute/results). Details and the options that
  need an account (free GPUs on Kaggle, and others) are in
  [`compute/README.md`](compute/README.md).
- **What the free compute already did:** it found a working download for
  the public battery datasets that the research container could not
  reach, and it ran the full-month bike experiment.

| Challenge | Idea tested | Verdict |
| --- | --- | --- |
| Cold battery voltage | Arrhenius-form resistances instead of fitted temperature slopes | **Supported:** skill +0.34 against +0.15, better on all 35 cold drive cycles |
| Bike rebalancing | Dispatch on forecast stock instead of current stock | **Supported:** 16% fewer failed trips; a perfect forecast would add 11% more |
| Bike rebalancing | Look-ahead on one past day's demand | **Not supported:** +0.8% over 17 days, after a two-day trial had suggested −2% |
| Bus holding | Hold buses to even out headways | **Refuted in this model:** headway rules raise passenger cost by 4–12%; perfect foresight gains under 1% |
| Rail dispatching | Look-ahead instead of greedy priority rules | **Supported:** 11% less weighted delay on real days (19 of 19), 7% on disrupted days; an optimistic bound is 29–35% below |

What would allow verified attempts on HFM's actual leaderboard:
1. participation access from the HFM Committee (task bundles and a
   submission or verification route);
2. GPU compute for the model-based tracks;
3. the tracks' `instruction.md` files, so that proxies match the real
   evaluation.

## 16. Data and reproduction

- [`hfm-projects/`](hfm-projects/README.md) has one self-contained project folder for each of the 54
  challenges, with the experiment code inside the project it belongs to.

- `research/hfm-challenges-2026-11-24.csv` lists all 54 challenges: field,
  status, metric, direction, baseline, best audited score, proposal count,
  scored count, accepted improvements, spend, and the best hypothesis.
- Re-fetch a track with
  `GET https://hf-mission.com/api/discovery/live/snapshot?section=summary&source=<source_id>`.
  The directory is at
  `section=directory&directoryLimit=50` and pages with `beforeTrackId`.

## Sources

- [Human Frontier Mission](https://hf-mission.com/) and its live API (fetched 24 Nov 2026)
- PR Newswire, [The Next Frontier for AI: Human Frontier Mission Announces Global Launch](https://www.prnewswire.com/news-releases/the-next-frontier-for-ai-human-frontier-mission-announces-global-launch-302947034.html) (21 Nov 2026)
- AI Weekly, [Human Frontier Mission Opens Nov 24 With 2,391 AI Experiments](https://aiweekly.co/alerts/human-frontier-mission-opens-nov-24-with-2391-ai-experiments)
- aicell.io, [Lab Newsletter — November 24, 2026: The Verification Turn](https://aicell.io/post/newsletter-2026-11-24/)
- [AI Research Agents Narrow Scientific Exploration, arXiv 2605.27905](https://arxiv.org/pdf/2605.27905); [AI agents struggle to perform original scientific research](https://techxplore.com/news/2026-08-ai-agents-struggle-scientific.html)
- Single-cell foundation-model evaluation: [Genome Biology 2025](https://link.springer.com/article/10.1186/s13059-025-03574-x); [zero-shot limits (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.10.16.561085.full.pdf)
- Weather: [WeatherBench 2](https://sites.research.google/weatherbench); [GenCast (Nature 2024)](https://unpaywall.org/10.1038%2FS41586-024-08252-9)
- Traffic RL benchmarks: [RESCO](https://people.engr.tamu.edu/guni/papers/NeurIPS-signals.pdf)
- Fatigue and constitutive ML: [physics-guided NN under overloads](https://research.buaa.edu.cn/en/publications/a-physics-guided-neural-network-for-probabilistic-fatigue-life-pr/)
- Decision-focused energy scheduling: [arXiv 2412.10554](https://arxiv.org/html/2412.10554v1); [arXiv 2401.03680](https://arxiv.org/pdf/2401.03680v2.pdf)
