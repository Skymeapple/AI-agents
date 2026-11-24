# Human Frontier Mission: the challenges, their limits, and routes past them

*Research note, 24 November 2026 (HFM's launch day).*

## 1. What the Human Frontier Mission is

The **Human Frontier Mission (HFM)** was announced by the **Human Frontier
Mission Committee** in a press release from Seattle dated 21 November 2026. It
launched globally on 24 November 2026. Its site is `hf-mission.com` and it posts
as `@HFMission` on X.

It is **not** the Human Frontier Science Program (HFSPO, Strasbourg, founded
1989). That is an older life-sciences funder with a similar name.

How it works, as described:

- **Domain experts** define each challenge and the criteria for judging progress.
- **Autonomous AI research agents** build and test candidate approaches.
- **Results are published for public review.** The record is meant to show where
  an approach gained ground, where it fell short, and what someone could try next.
- Outside researchers can propose questions, contribute evidence or evaluation
  methods, and help assess results.

Reported scale at the launch snapshot: **2,391 experiments, 54 challenges, 12
fields, 23 studies running.** The homepage lists committee affiliations with
Novix Lab, Google, Microsoft, HKU, CMU, Tsinghua, Johns Hopkins and Stanford.

### What this note could and could not verify

| Claim | Status |
| --- | --- |
| Launch date, structure, headline counts | From the press release and its reprints. That is one source repeated, not independent confirmation. |
| 2,391 experiments | **Unverified.** AI Weekly reported that the landing page's challenge counter was still empty when it checked. |
| Committee membership | The release names no individuals, labs or models. The homepage lists affiliations. These could not be cross-checked. |
| Funding | Not disclosed. |
| The full list of 54 challenges and 12 fields | **Not obtained.** This environment's network policy blocks `hf-mission.com` and the press-release hosts, and search engines index only the examples quoted in coverage. |

The aicell.io lab newsletter for 24 November ("The Verification Turn") calls HFM
"one to watch sceptically". The newsletter likes the model but withholds
judgement on this instance of it, and argues that *production is solved;
verification is the bottleneck*. Section 7 builds on that point.

**This note therefore covers the five challenges that public coverage names.**
The method in Section 7 is meant to apply unchanged to the other 49 once their
definitions can be read.

| # | Challenge (as described publicly) | Field |
| --- | --- | --- |
| 1 | Improving global weather forecasts | Earth system science |
| 2 | Coordinating road traffic signals | Transportation |
| 3 | Predicting how steel responds to repeated loads it did not see in development | Structural materials |
| 4 | Scheduling wind power against energy storage | Energy systems |
| 5 | Identifying lung cell types | Biomedicine / single-cell biology |

For each challenge the note gives: where the frontier actually is, **what limit
is binding**, and the routes most likely to get past it. A route counts only if
it attacks the binding limit, not if it shaves a decimal off a saturated
leaderboard.

---

## 2. Global weather forecasting

### Where the frontier is

Machine-learned medium-range models now match or beat the best physics-based
systems on most standard scores. GenCast, a diffusion-based 15-day ensemble at
0.25°, was reported to outperform ECMWF's ENS on 97.2% of the variable × lead
time × level targets tested. A single forecast takes about 8 minutes on one
TPUv5. WeatherBench 2 is the shared yardstick for global 1–15 day skill. Google
released WeatherNext code and weights in August 2026, and WeatherNext 3 is now
its operational product.

### The binding limits

1. **These models depend on the analysis they start from.** Almost every leading
   ML model starts from an analysis produced by conventional data assimilation
   (ECMWF's 4D-Var). The expensive, slow, physics-dependent part of the forecast
   chain has not been replaced, only the cheap part after it.
2. **Extremes are underdispersed and smoothed.** Models trained against
   reanalysis regress toward the conditional mean. Ensemble spread and
   tail intensity (peak gusts, tropical-cyclone intensity, record heat) lag behind
   the headline RMSE gains.
3. **Behaviour outside the training climate is unknown.** Training on 1979–2018
   reanalysis says little about behaviour in a warming climate whose extremes
   lie outside that record.
4. **The benchmark is saturating.** Small gains on global mean RMSE/CRPS for Z500
   and T850 no longer track what forecast users value.

### Routes past them

| Route | Limit it attacks | Why it could be a step change |
| --- | --- | --- |
| **End-to-end learned assimilation from raw observations** (Aardvark-, GraphDOP-style), trained on satellite radiances, radiosondes and surface stations directly | 1 | Removes the dependency on NWP analyses. That makes forecasting possible wherever observations exist, at a fraction of the compute, and lets the model exploit observations that 4D-Var under-uses. |
| **Train on proper scoring rules for the full distribution** (fair CRPS on ensembles, threshold-weighted CRPS for tails) instead of MSE | 2 | Directly penalises the smoothing that kills extremes. It also produces calibrated spread without post-hoc inflation. |
| **Hybrid models with a dynamical core** (NeuralGCM-style: learned physics plus a differentiable solver) | 3 | Conservation laws and dynamics carry the model where data does not. That is the only credible route to stability under climate shift. |
| **Evaluation that reflects what users need** | 4 | Score extreme events, cyclone track and intensity, and economic-value curves on years held out *after* every model's training cut-off. |

**What an HFM agent should actually be asked to do.** Pick one regional,
extreme-weighted target, such as 24–72 h heavy-precipitation CRPS over a
held-out 2025–26 season. Require a beat over both IFS ENS and the best open ML
ensemble on it. Require that the agent does not lose global CRPS while doing so.
Global-mean leaderboard gains alone should not count as a breakthrough.

---

## 3. Coordinating road traffic signals

### Where the frontier is

Reinforcement learning for signal control has a decade of papers and almost no
field deployments. The RESCO benchmark found that methods claiming the state of
the art struggled on realistic scenarios. A well-tuned, decentralised deep
Q-learning baseline often beat them. 2025–26 work is still about generalising to
unseen demand (Vissim-validated MARL, CityLight's city-scale model) and about
corridor deployments calibrated to real traffic (PNNL's A2C work in Aimsun Next).
What is operational at scale is simpler: adaptive schedule-driven control and
offset/timing recommendations mined from probe-vehicle data.

### The binding limits

1. **Policies learned in simulation do not transfer.** They overfit to the
   simulator's demand profile, driver model and detector layout.
2. **Real controllers see little.** Loop detectors give occupancy at a point, not
   queues. The rich simulated state that RL policies learn on is not available
   on the street.
3. **Safety and acceptability come first.** Agencies need guaranteed minimum
   greens, pedestrian phases, cyclic or predictable phase order, and a fallback.
   An unconstrained policy cannot be deployed.
4. **Baselines are weak in many papers.** Gains reported against fixed-time
   plans disappear against tuned actuated control or max-pressure.

### Routes past them

| Route | Limit it attacks | Mechanism |
| --- | --- | --- |
| **Residual RL on top of max-pressure** | 3, 4 | Max-pressure has throughput-optimality guarantees. Learning only a bounded correction keeps those guarantees and safety constraints, and the gain over a strong baseline is measured directly. |
| **Train across a demand distribution, not on one scenario.** Randomise origin–destination matrices, incidents, sensor dropout and driver parameters, calibrated to real detector and probe data | 1 | Turns "it works on the network it was trained on" into a robustness claim that can be tested. |
| **Use connected-vehicle trajectories and probe data as the state**, with a detector-only fallback | 2 | Probe penetration is now high enough in many cities to estimate queues and delay. It is also where the deployed, scaled systems get their signal. |
| **Offline RL from historical controller logs, with conservative policy improvement** | 1, 3 | Learns from the real system's own data and bounds how far the policy can drift from behaviour already proven safe. |
| **Graph neural policies with shared parameters** | 1 | One policy for any intersection topology, so it can be tested zero-shot on networks it has never seen. |

**Breakthrough criterion.** The policy should cut network delay and stops on
**held-out cities and incident scenarios** while beating tuned actuated control
*and* max-pressure. It should hold every phase and safety constraint at all
times. A result on a single training grid is not a result.

---

## 4. Steel under repeated loads it has not seen

This is the challenge whose framing is most precise. It asks for
**extrapolation**: predict fatigue under loading histories outside the
development data.

### Where the frontier is

Recent ML-for-fatigue work adds physics in three ways. It builds Basquin-model
features and boundary-enforcing losses (validated on titanium and carbon steel,
but constant-amplitude only). It adds physics-informed losses for irradiated
steels (low-cycle, strain-controlled). And a 2025 physics-guided network is
built on a crack-closure model. That last one is **trained only on
constant-amplitude data and predicts lives under multiple overloads**, which is
the closest existing work to this challenge. Engineering practice still rests
on rainflow counting plus Palmgren–Miner linear damage. Miner's rule's
critical damage sum scatters over roughly an order of magnitude because it
ignores load-sequence effects.

### The binding limits

1. **Load sequence matters, and black-box models cannot extrapolate it.**
   Overloads retard crack growth and underloads accelerate it. A network
   trained on life versus summary statistics learns the training spectra, not
   the mechanism.
2. **Data is scarce and noisy.** Fatigue tests are slow. Scatter of a factor of 2–10
   on life at one stress level is normal, so point predictions are misleading.
3. **Validation splits are usually leaky.** Random splits over specimens from the
   same spectra make interpolation look like generalisation.

### Routes past them

| Route | Limit it attacks | Mechanism |
| --- | --- | --- |
| **Put the extrapolation in the physics, and learn only the parameters.** Predict cycle-by-cycle crack growth with a closure/strip-yield model whose material parameters and closure law are learned. Do not predict life directly. | 1 | The model's structure does the extrapolating. Unseen spectra are just new inputs to a mechanism, not a covariate shift. |
| **A learned damage-state ODE stepped through the load history**, constrained to be monotone in damage and to commute with how cycles are counted | 1 | Captures sequence effects that Miner cannot, while staying stable on sequences of any length. |
| **Probabilistic output**: a calibrated life distribution (for example P–S–N style lognormal), scored by coverage | 2 | A factor-of-two band that is honest beats a point estimate that is confidently wrong. Design codes already work in probabilities of failure. |
| **Active learning over tests**: choose the next physical test to cut predictive uncertainty on the target spectrum class most | 2 | Makes each slow, expensive test count for more. |
| **Spectrum-level held-out splits**: train on constant-amplitude and block loading, test on standard random spectra (for example FALSTAFF, CARLOS, WASH-type histories) never seen in training | 3 | The only split that measures what the challenge asks. |

**Breakthrough criterion.** Most predictions on **unseen spectrum families**
should fall within a factor of 2 of test life. That is stricter than the usual
factor-of-3 band. The predicted intervals should hit their nominal coverage, and
the model should beat rainflow + Miner and rainflow + a calibrated nonlinear
damage rule on the same split.

---

## 5. Scheduling wind power against energy storage

### Where the frontier is

The field is moving from *predict-then-optimise* to **decision-focused learning
(DFL)**: train the forecaster on the cost of the decision it feeds, not on
forecast error. There are recent results across the problem. One end-to-end
method tunes wind forecasts to two-stage market cost under a distributionally
robust OPF. Others apply DFL to storage arbitrage, reserve-market bidding and
battery dispatch under renewable uncertainty. Deep RL for co-located wind and
batteries is the other strand.

### The binding limits

1. **Better forecasts do not mean better decisions.** Errors at high-price hours
   and near ramps matter far more than average error.
2. **Battery degradation is left out or crudely priced.** Schedules that look
   profitable over a week consume cycle life worth more than the arbitrage.
3. **Market rules are complex.** Imbalance settlement, reserve products and
   gate-closure times make the real problem a multi-stage stochastic programme.
4. **Results do not generalise across years.** Strategies tuned on one year's price
   regime fail in the next (compare 2021–22 against 2024–26).

### Routes past them

| Route | Limit it attacks | Mechanism |
| --- | --- | --- |
| **DFL with a stochastic-MPC inner layer.** Probabilistic forecasts (quantiles or scenarios) feed a receding-horizon stochastic programme, and the forecaster is fine-tuned through it on realised profit | 1, 3 | Forecasts get pushed toward accuracy where the decision is sensitive. MPC keeps hard constraints that pure RL can break. |
| **A degradation-aware objective**: cycle-depth-dependent (rainflow-based) ageing cost inside the dispatch problem | 2 | Turns hidden capital consumption into a priced trade-off. |
| **Conformal calibration of the reserve or imbalance margin** | 1, 3 | A distribution-free guarantee on shortfall probability, which a grid operator can accept. |
| **Report regret against perfect foresight, not raw profit**, on out-of-sample years | 4 | Makes results comparable across markets and years. "Captures 85% of the perfect-foresight value in an unseen year" is a claim that means something. |

**Breakthrough criterion.** On a held-out market year, capture a higher share of
perfect-foresight value than stochastic MPC with an accuracy-trained forecaster.
Keep degradation-adjusted net value positive and keep shortfall probability at or
below target.

---

## 6. Identifying lung cell types

### Where the frontier is

The reference is the **Human Lung Cell Atlas (HLCA)**. Its core integrates
about 584k cells from 107 people across 14 datasets with a consensus
hierarchical annotation, and its extension covers about 2.4M cells. Single-cell
foundation models (scGPT, Geneformer, scFoundation) were expected to solve
annotation. Independent evaluations found otherwise. **Zero-shot they are often
beaten by simple baselines** (highly variable genes, scVI, logistic regression).
Geneformer's embedding space clusters by batch rather than by cell type. And
pre-training sometimes adds nothing for annotation after fine-tuning.

### The binding limits

1. **Batch and platform effects** dominate the biological signal across studies,
   protocols and chemistries.
2. **Rare and transitional types** are where annotation matters clinically and
   fails most: ionocytes, neuroendocrine and tuft cells, and disease-specific
   states such as aberrant basaloid cells in pulmonary fibrosis. Accuracy is
   dominated by abundant types, so it hides these failures.
3. **Classifiers must assign a known label.** Most of them cannot say "this is not
   in the reference", which is exactly the case in disease samples.
4. **Label noise.** Annotations differ across studies, so a model can be "wrong"
   where the reference is.

### Routes past them

| Route | Limit it attacks | Mechanism |
| --- | --- | --- |
| **Hierarchical prediction over the HLCA ontology with calibrated abstention** | 2, 3, 4 | Predict at the finest level the evidence supports and back off to the parent ("basal-like epithelial") when uncertain. Flag novel cells. |
| **Reference mapping with per-cell uncertainty** (scArches/scANVI-style surgery onto the HLCA), combined with marker-gene priors | 1, 2 | Corrects batch effects during mapping. Marker priors anchor rare types that have few training cells. |
| **Foundation-model embeddings only as one input to a stacked ensemble**, kept only if ablation shows they add value | 1 | Uses what FMs do well without trusting them zero-shot. The evidence says they must earn their place. |
| **Evaluation held out by study and by disease**, scored on macro-F1 across rare types, calibration error and novel-cell AUROC | 2, 3 | Measures the failures that matter rather than overall accuracy. |

**Breakthrough criterion.** On a held-out *disease* cohort from an unseen lab,
reach high macro-F1 over rare types, detect cell states missing from the
reference with high AUROC, and keep confidence calibrated. In short, a model
that knows when it does not know.

---

## 7. Across all challenges: making agent-driven research credible

The five challenges differ in domain but share one structural risk. **An
autonomous agent running thousands of experiments against a fixed metric
optimises the metric.** That is useful exactly as far as the metric measures the
real goal and no further. Recent work found that AI research agents *narrow*
scientific exploration, and other studies report that agents struggle with
open-ended original research. The aicell newsletter's diagnosis applies:
generating candidate approaches is cheap now, and verifying them is the
constraint.

What would let HFM, or anyone running agents on these challenges, claim a
breakthrough that holds up:

1. **Sealed, distribution-shifted test sets.** Hold out *future* time windows
   (weather, markets), *unseen networks* (traffic), *unseen spectrum families*
   (steel) and *unseen labs and diseases* (lung cells). Agents should never be
   able to query them. Each challenge above is defined by a generalisation gap,
   so the test split must embody that gap.
2. **A floor of strong baselines.** Every claim must beat the best simple method
   tuned with the same compute budget: max-pressure, rainflow + Miner, stochastic
   MPC, logistic regression on highly variable genes, IFS ENS. Most published
   "gains" in these fields vanish against tuned baselines.
3. **Count every experiment.** 2,391 experiments against a fixed metric is a
   multiple-comparisons problem. Report how many attempts preceded the best one,
   and confirm winners on fresh data before announcing them.
4. **Search for diversity, not only for a score.** Keep an archive of
   qualitatively different approaches (quality-diversity / MAP-Elites style)
   instead of hill-climbing one family. That counters the narrowing effect and
   is where unexpected routes come from.
5. **Keep provenance for every number.** Every reported figure should say
   whether it was measured, taken from a named source or assumed. That is the
   discipline `sciscout` enforces (see the repository README), and it is what
   lets a public record be audited rather than merely read.
6. **Put physical structure where extrapolation is needed.** In weather (a
   dynamical core), steel (crack-closure mechanics) and traffic (max-pressure
   stability), the most credible routes past the current limits keep a
   mechanism that is known to be correct and learn only the part around it.
   That gives an agent a search space where a win is likely to transfer, rather
   than one where it exploits the benchmark.

## 8. Next steps

- **Get the other 49 challenges.** Reading `hf-mission.com` would give the
  full set of definitions, metrics and current-best results, and Sections 2–6
  could then be repeated for each.
- **Check each named challenge's official metric** against the breakthrough
  criteria proposed here. If HFM's metric rewards in-distribution gains, the
  criteria above are the stronger test.
- **Track HFM's directions in `sciscout`.** Harvest papers for each challenge
  area to see which routes above are accelerating in the literature.

## Sources

- PR Newswire, [The Next Frontier for AI: Human Frontier Mission Announces Global Launch](https://www.prnewswire.com/news-releases/the-next-frontier-for-ai-human-frontier-mission-announces-global-launch-302947034.html) (21 Nov 2026), and reprints such as [wireuk.org](https://wireuk.org/the-next-frontier-for-ai-human-frontier-mission-announces-global-launch/)
- AI Weekly, [Human Frontier Mission Opens Nov 24 With 2,391 AI Experiments](https://aiweekly.co/alerts/human-frontier-mission-opens-nov-24-with-2391-ai-experiments)
- aicell.io, [Lab Newsletter — November 24, 2026: The Verification Turn](https://aicell.io/post/newsletter-2026-11-24/)
- [Human Frontier Mission homepage](https://hf-mission.com/)
- [WeatherBench 2](https://sites.research.google/weatherbench); GenCast, [Nature 2024](https://unpaywall.org/10.1038%2FS41586-024-08252-9); [WeatherNext model docs](https://developers.google.com/weathernext/guides/models-wn-gen); [WeatherNext open release coverage](https://rits.shanghai.nyu.edu/ai/google-open-sources-weathernext-cyclone-forecasts-gain-a-day/); [EGU26-17711](https://meetingorganizer.copernicus.org/EGU26/EGU26-17711.html)
- Traffic: [RESCO (NeurIPS)](https://people.engr.tamu.edu/guni/papers/NeurIPS-signals.pdf); [arXiv 2603.12096](https://arxiv.org/abs/2603.12096); [CityLight, arXiv 2406.02126](https://arxiv.org/html/2406.02126v2); [PNNL corridor MARL](https://www.pnnl.gov/publications/deep-multi-agent-reinforcement-learning-real-world-signalized-traffic-corridor-control); [arXiv 2103.16223](https://arxiv.org/abs/2103.16223v3)
- Fatigue: [Physics-guided NN under multiple overloads (BUAA)](https://research.buaa.edu.cn/en/publications/a-physics-guided-neural-network-for-probabilistic-fatigue-life-pr/); [arXiv 2503.05419](https://export.arxiv.org/pdf/2503.05419); [arXiv 2508.17303](https://web3.arxiv.org/abs/2508.17303); [arXiv 2501.15057](https://arxiv.org/pdf/2501.15057); [arXiv 2209.00628](https://arxiv.org/pdf/2209.00628)
- Wind and storage: [arXiv 2412.10554](https://arxiv.org/html/2412.10554v1); [arXiv 2401.03680](https://arxiv.org/pdf/2401.03680v2.pdf); [arXiv 2509.05772](https://www.arxiv.org/pdf/2509.05772); [arXiv 2305.00362](https://pdf.arxiv.org/pdf/2305.00362); [arXiv 2304.02239](https://arxiv.org/pdf/2304.02239)
- Lung cells: [Genome Biology 2025, zero-shot FM evaluation](https://link.springer.com/article/10.1186/s13059-025-03574-x); [Assessing the limits of zero-shot foundation models (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.10.16.561085.full.pdf); [Foundation models and imbalanced single-cell data](https://www.biorxiv.org/content/10.1101/2023.10.24.563625.full.pdf)
- Agentic research: [AI Research Agents Narrow Scientific Exploration, arXiv 2605.27905](https://arxiv.org/pdf/2605.27905); [AI agents struggle to perform original scientific research](https://techxplore.com/news/2026-08-ai-agents-struggle-scientific.html); [arXiv 2607.27191](https://arxiv.org/pdf/2607.27191)

Background knowledge not re-verified in this session, so check it against the
original before citing: HLCA cell and donor counts (Sikkema et al., *Nature
Medicine* 2023); the max-pressure stability property; the scatter of Miner's
critical damage sum; the named standard load spectra (FALSTAFF, CARLOS, WASH);
learned-assimilation systems (Aardvark Weather, GraphDOP) and NeuralGCM.
