# Review (Round 1) — p010: "The Definition Is the Forecast: How Reliability Thresholds and Cross-Domain Aggregation Determine Measured Distance to AGI"

Reviewer role: independent area-chair-level reviewer. I have no stake in the paper.

## 1. Summary

The paper asks a narrower version of "how far are we from AGI, and how should AGI be defined?": with the evaluation evidence held fixed, how much does the extrapolated arrival date depend on the definitional choices made when that evidence is reduced to one number? It treats a definition as a tuple (domain set Ω, reliability threshold q, power-mean/quantile aggregator, target τ) and makes four contributions:

1. **Four propositions.** (1) Arrival times are ordered by the power-mean exponent p. (2) The growth rate of a power mean of exponentially growing domains is a softmax-weighted average of the domain rates, and it tends to the max, mean or min rate depending on the sign of p. (3) Under a logistic model of success against log task length, a threshold q adds a delay of logit(q)/|β| doublings, and doubling times are unchanged if and only if β is stationary. (4) The minimum of noisy estimates is biased downward, by at most σ√(2 ln n).
2. **A penalised hierarchical logistic estimator** of per-task-family horizons on METR run-level data. The ridge λ is chosen by task-level CV, and uncertainty comes from a hierarchical bootstrap.
3. **A grid of 24 definitions** (3 thresholds × 8 aggregators) on METR TH1.1, replicated on TH1.0. The 167-hour arrival dates span 2027.1–2032.6, and "definition choice" accounts for 43% of arrival-date variance (29% on TH1.0).
4. **A CHC re-analysis** of Hendrycks et al.'s GPT-4/GPT-5 profiles, plus pre-registered predictions and a reporting protocol.

## 2. Verification performed

### 2.1 Re-running the code

I copied the paper directory to a scratch location, repointed the hard-coded paths, and re-ran:

- `analysis_metr.py --version 1-1 --B 500 --lam 1` (main analysis)
- `analysis_metr.py` for the TH1.0 replication and the λ ∈ {0.25, 4} sensitivity runs
- `theory_chc.py`, `reliability_tax.py`, `robust_dispersion.py`, `profiles.py`, `cv_lambda.py`

The results of these reruns are summarised in Section 2.3 below. I also checked by hand every number in Tables 1–2 and §§5.2–5.7 against the saved `results/` files.

**Agreement.** Table 1 (all 24 rows: doubling time, 167-h arrival, 90% intervals, latest level), Table 2, the slope-trend numbers, the dispersion numbers, the CV losses (λ=1: 0.3542; pooled: 0.3852, an 8.0% gain), the variance decomposition (1.77 / 2.39 yr², 43%; 61% for aggregator only; 52% for threshold only), the λ-robustness summary, the TH1.0 summary, the CHC numbers (0.27/0.57, 0.15/0.49, ε-floor 0.36/0.49, p=1 crossing of 0.7 at 2027.66, plateau at 0.77) and the Opus 4.6 λ-sensitivity of the minimum (0.17 → 287 min) all match the saved outputs. `theory_chc.json`, `reliability_tax.json` and `robust_dispersion_1-1.json` are reproduced bit-for-bit on rerun.

**Mismatches between paper text and data/code.**

- **(M1) Run counts are wrong.** §4 says TH1.1 "contains 24,008 agent runs (excluding human baselines)". 24,008 is the line count of `runs_1-1.jsonl` *including* 773 human-baseline rows. The analysed agent runs number **23,235**. For TH1.0, "41,629 runs" is again the raw line count including 793 human rows. There are 40,836 agent runs, and the analysis uses **37,539** after restricting to 2023+ systems. The abstract and Chinese abstract repeat both wrong figures. This does not change any estimate, but it is a factual misstatement of the data in the abstract.
- **(M2) The split-half reliability numbers are mislabelled.** §5.4 says "Reliability is higher for recent systems (0.92–0.94 for GPT-5.3-Codex and Opus 4.6)", right after reporting a Spearman–Brown full-data reliability of 0.88. In `profiles_1-1.json`, 0.919 and 0.943 are the *split-half r* values. The Spearman–Brown values for those systems would be about 0.96–0.97.
- **(M3) A numbering mismatch between code and paper.** `theory_chc.py` and `fig_theory.png` label the growth-rate result "Prop. 1", but the paper calls it Prop. 2. Readers comparing the figure with the text will be confused.
- **(M4) Resolved on rerun.** `cv_lambda.py` defaults to `--reps 3`, while the README and paper use 2. Re-running with `--reps 2` reproduces `cv_lambda_1-1.json` exactly, so the saved file corresponds to the documented setting. The script default should still be changed to 2 to match.

### 2.2 Reference spot-checks (web search; direct arXiv/metr.org fetches were blocked by the sandbox network)

| Reference | Status |
|---|---|
| Hendrycks et al. 2025, arXiv:2510.18212 | Exists. GPT-4 27%, GPT-5 57%. The GPT-5 per-domain row (9,10,10,7,4,0,4,4,6,3) matches the paper's table. |
| Fourati 2025, arXiv:2510.20784 | Exists. Generalised mean with a compensability exponent p, plus an AUC over p, applied to the CHC scores. FAST@AAAI 2026. The description is accurate. |
| Burnell et al. 2026, arXiv:2605.28405 | Exists. Google DeepMind, ten-faculty cognitive taxonomy, May 2026. Accurate. |
| Ord 2025, arXiv:2505.05115 | Exists. Constant-hazard / half-life model. The 212 vs 213-day doubling times are reported in the Forethought summary of Ord's piece, so the attribution is plausible. |
| Grace et al. 2025, JAIR 84 | Exists. HLMI 50% year 2047, full automation of labour 2116. Accurate. |
| METR 2025, "How does time horizon vary across domains" (14 Jul 2025) | Exists. Nine benchmarks, similar doubling rates, levels differing by large factors. Accurate. |
| METR 2026, time-horizons page | Exists. Contains "Measurements above 16 hrs are unreliable with our current task suite". TH1.1 released 29 Jan 2026 with 228 tasks. Accurate. |
| Kwa et al. 2025, arXiv:2503.14499 | Exists (well known). |

I found no fabricated or misdescribed references among those checked. The remaining references are standard and well known: Hardy–Littlewood–Pólya, Boucheron et al., Legg & Hutter, Chollet, Morris et al., Schaeffer et al., Ruan et al., Burnell et al. (2023), McGrew and Dell'Acqua et al.

Two caveats. First, per secondary sources, METR corrected a regularisation issue in its TH1.1 analysis in March 2026. The paper should state which data commit it downloaded and whether this matters. Second, METR posted a Q4-2026 capability update on 2026-11-20, two days before this snapshot, and the paper does not discuss it. For a paper whose headline is an extrapolation, newer frontier data that may already test P1–P4 should at least be acknowledged.

### 2.3 Re-run results

- **Main TH1.1 analysis** (λ=1, B=500, seed 0): the rerun summary CSV is identical to `results/metr_1-1_lam1_summary.csv` (max abs difference 0.0 across all columns). `metr_1-1_lam1_slopes.json` is identical. Recomputing `variance_decomposition.json` from the rerun bootstrap gives 0.4257 share, 2.389 within and 1.771 between, the same as the paper.
- **`profiles.py --splits 10`**: identical. Mean profile correlation 0.421, PC1 0.355, early/late correlation 0.249, split-half r 0.796, Spearman–Brown 0.884.
- **`cv_lambda.py --reps 2`**: identical. Best λ=1.
- **`theory_chc.py`, `reliability_tax.py`, `robust_dispersion.py`**: identical.
- **TH1.0 (λ=1) and TH1.1 λ=0.25**: the per-model point-estimate CSVs are bit-identical. Point arrival ranges recomputed from them are TH1.0 2027.95–2032.80 (DT 4.56–8.28 mo, pooled DT 5.77, 80% vs 50% delay 0.908 yr) and λ=0.25 2026.70–2033.99 (DT 3.45–7.46). Both match the paper. I stopped the full bootstrap reruns for TH1.0, λ=0.25 and λ=4 early because of CPU contention. Their interval endpoints were therefore checked only against the saved files, not regenerated, but they use the same code path that reproduced exactly for the main analysis.
- **Runtime.** The main analysis took about 23 min, under heavy contention, vs the stated ~10 min.

Conclusion: apart from the data-description mismatches M1–M2, every quantitative result in the paper is backed by code that reproduces exactly.

## 3. Assessment of the theory (line by line)

- **Prop. 1** is correct and, as the authors say, elementary. It is the power-mean inequality applied pointwise. Continuity is not even needed. **Error in the surrounding text:** §3.1 states that "quantile aggregators such as the 10th-percentile domain sit between p=0 and p=−∞". This is false in general. A quantile is not bounded above by the geometric mean (a single near-zero domain drives A_0 below any quantile above it). The median, also a quantile, can exceed the arithmetic mean. The paper's own Table 1 shows this: for Opus 4.6 at q=0.5 the median family is 1,973 min, against a geometric mean of 935 min. Quantiles therefore cannot be placed in the Prop. 1 chain. The robustness check (`PORDER`) correctly excludes them, but the text implies otherwise.
- **Prop. 2.** The derivative is correct: d/dt of (1/p)·logsumexp is the softmax-weighted rate. So is the limit. Implication (a), "the long-range forecast is discontinuous in p at p=0", is mathematically true only for the t→∞ limit. At any finite horizon the rate is continuous in p, and the transition time 1/(|p|Δs) diverges as p→0. So p=+0.1 and p=−0.1 give nearly identical forecasts for any practically relevant horizon. As written, the "jump" is misleading and should be stated with that qualification. The result itself is standard: it is the gradient of log-sum-exp.
- **Prop. 3.** Correct. I re-derived d/dt log₂H_q = −α′/β − log₂H_q·β′/β. The difference across thresholds is −(logit q₁ − logit q₂)β′/β², which vanishes for all t iff β′ ≡ 0. The sign claim (flattening slows high-q horizons) is right. It is also near-trivial algebra that METR and Ord already use implicitly: the H80/H50 ratio is fixed by the slope. Its value lies in the clean statement, not in novelty.
- **Prop. 4.** Correct, and a textbook extreme-value bound. "Equality to first order when tied and n large" is asymptotically correct, but convergence is slow (the simulation gives a ratio of 0.84 at n=200). The proposition ignores that the actual estimator is ridge-shrunk and that family noise is heteroscedastic (families have 1–several tasks), so it gives only qualitative guidance for §5.5.

Overall, the "four results" are correct but are standard facts dressed as propositions. They are useful framing, not a theoretical contribution at the level the abstract implies ("prove four results").

## 4. Assessment of the empirical work

**Strengths**

- The pipeline is transparent and reproducible, and it runs on real public run-level data. It includes a sensible hierarchical bootstrap over families and tasks, and λ is chosen by CV (with a clear held-out-family rule).
- The authors are honest about many limitations. They call the 167-h dates a yardstick and not forecasts. They report the λ-dependence of the spread. They run the ceiling/floor check on the dispersion trend and drop the "profiles are getting more jagged" claim accordingly. They replicate on TH1.0 and state that the variance share is weaker there.
- The closed-form reliability delay vs direct fits (Table 2) is a nice, checkable consistency test.

**Weaknesses**

1. **The headline statistic depends on the menu.** "Definition choice explains 43% of the variance" is a function of which 24 definitions are put on the menu, weighted equally. The menu includes p=+1, which the authors themselves recommend against (§5.1, §7). It includes the unshrunk-sensitive minimum, which they also recommend against without validated shrinkage, and q=0.95, which extrapolates the logistic tail. Dropping the definitions the paper disavows, or weighting by plausibility, would change the share substantially. The share also moved from 43% to 29% across suite versions and is estimated from B=60 on TH1.0. The paper presents this number in the abstract and conclusion ("29–43%... comparable in size to all sampling uncertainty") as if it were a property of the world. It is a property of the design. At minimum, report the share for a "recommended menu" (pooled/geo/q10/q25 × q∈{0.5,0.8}) and with p=+1 and min-family removed, and give a bootstrap CI on the share itself.
2. **"Sampling uncertainty" is incomplete.** The within-definition variance covers only task/family resampling. It omits uncertainty in frontier selection, in model release-date choice, in trend functional form (exponential vs super-exponential, window choice) and in human time estimates. METR itself says these dominate its own uncertainty, and 26 of the 31 long tasks have estimated rather than measured times. Comparing the definitional variance with only the task-bootstrap variance overstates the relative importance of definitions. A model-level (leave-one-system-out) or window-choice sensitivity is needed.
3. **Evidence for the Prop. 2 mechanism is weak.** §5.1 point 2 says doubling times rise "as p decreases", as Prop. 2 predicts. But the CIs overlap heavily, the pattern is non-monotone (q25 at 5.7 > minfam at 4.9 > q10 at 5.1), and no paired-bootstrap test of the DT differences is reported. Moreover, the model imposes a common slope b across families within a system, so family-level growth-rate heterogeneity is only indirectly identified through the drift of ridge offsets. A direct test is needed: per-family rate estimates, or a paired bootstrap of DT(p) − DT(pooled).
4. **The lower-end support problem is ignored.** The paper discusses extrapolating beyond the longest tasks (30 h) but not the fact that many bottleneck/high-q levels are far *below* the support. The latest-system levels are 0.1 min (6 s) for the worst family at q=0.95 and 0.3 min for q10 at q=0.95, and earlier systems are lower still. These are extrapolations of a common-slope logistic into a region with few or no tasks in the relevant family. The resulting doubling times for these rows are largely model artefacts, yet they drive the upper end of the 5.5-year spread.
5. **The link to the AGI question is thin.** Task families in METR's software/ML/cyber suite are not cognitive domains, as the authors concede. The 167-h target is about 10× beyond METR's own stated 16-h reliability limit. So the paper's empirical core is a sensitivity analysis of METR horizon extrapolations, and its relation to "distance to AGI" is by analogy. The title and abstract oversell this. The CHC part is the only analysis on AGI-relevant domains, and with two snapshots and integer scores it is necessarily thin. "Linear extrapolation of a bounded score with stagnant domains never reaches the threshold" is close to a tautology.
6. **Novelty is incremental.** The static aggregator sensitivity on CHC is Fourati (2025), as the authors acknowledge. The 50%/80% threshold comparison and its relation to the logistic slope come from Kwa et al. (2025) and Ord (2025). METR itself reports a hierarchical bootstrap, per-domain heterogeneity and sensitivity to task composition. The genuinely new elements are (a) the dynamic framing (rate vs constant delay) and (b) the definition-vs-sampling variance decomposition on run-level data. Both are reasonable but modest. For a NeurIPS D&B evaluation-methodology paper, I would want either a released benchmark/tooling artefact with broader coverage (e.g. applying the protocol to several suites, such as the nine domains in METR 2025, or the newer HCAST/TH2 data), or substantively deeper theory.
7. **Mixed bootstrap concepts.** The 0.35 /year lower-semideviation trend and the "unsaturated" 0.04 /year trend use a *model-resampling* bootstrap (`robust_dispersion.py`), while the main dispersion CI uses the *task-hierarchical* bootstrap. The paper compares these intervals directly ("not distinguishable from zero") although they quantify different uncertainty. The "100% of bootstrap replicates positive" (task bootstrap) vs the model-bootstrap CI for the unsaturated version is not an apples-to-apples ablation.
8. **The pre-registered predictions are weakly falsifiable.** P1 requires *both* criteria to fail in the same direction. P2's band (2.4 ± 0.8 doublings, i.e. H50/H80 between 3× and 9×) is very wide. P2 also states "holds per system by construction", so only the band is tested. P4 is consistent with the paper's own data, where `automatic_jailbreak` is a bottom-3 family in all five latest systems, so persistence of 1 of 3 is built in. "Pre-registered" also needs a timestamped registry entry (OSF/AsPredicted); a paragraph in a submission is not a pre-registration.
9. **Weighting and estimator details.** Rescaling weights to sum to the number of runs per system makes the ridge penalty strength effectively vary with the per-system run count, so λ=1 means different amounts of shrinkage for systems with 700 vs 1,600 runs. This interacts directly with §5.5's central finding (the minimum varies by 1,700×). A per-system or effective-sample-size-scaled λ, or a proper random-effects model with an estimated variance component (e.g. a GLMM fit by REML or a Bayesian hierarchical model), would be more principled than one CV-chosen ridge.

## 5. Questions to the authors

1. What does the 43% figure become if p=+1 and the min-family aggregator are removed (the two the paper recommends against)? Please also give a bootstrap CI for the share.
2. How do the arrival-date spreads change with leave-one-system-out frontiers, or with a 2024+ window?
3. Can you estimate family-specific growth rates, or test DT differences by paired bootstrap, to support the claim that "aggregators change rates"?
4. What fraction of the bottleneck/high-q horizon estimates lie below the shortest task in the relevant family?
5. Which commit of `METR/eval-analysis-public` was used, and does it include METR's March-2026 correction?
6. Why use a single global ridge on weight-normalised runs instead of a random-effects variance component estimated per system?

## 6. Minor issues

- Fix the run counts (23,235 agent runs for TH1.1; 37,539 analysed / 40,836 total agent runs for TH1.0) in the abstract, §4 and the Chinese abstract.
- Fix the "0.92–0.94" reliability label (split-half r, not Spearman–Brown).
- Correct the claim that quantiles lie between p=0 and p=−∞.
- Qualify the "discontinuous in p" claim with the finite-time continuity.
- Align the proposition numbering in `theory_chc.py` and `fig_theory.png` with the paper.
- §5.5 says 0.17 min "at λ = 0.01". The code's λ grid in `profiles.py` includes 0.01, but the CV grid starts at 0.03. Clarify that λ=0.01 is outside the validated range.
- The "Latest" column mixes horizons extrapolated far below and above the task support without flagging it.
- Code paths are hard-coded absolute paths, so the code is not portable as released.

## 7. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 8 |

**Overall: 5 / 10 (weak reject).**

**Decision: major revision.** Below the bar for a top-venue main or D&B track in its current form. It could reach the workshop/borderline level after the fixes above.

**Rationale.** The paper is careful, honest and reproducible, and almost all reported numbers are backed by run code. However, (i) the theoretical results are standard; (ii) the headline variance share is an artefact of the chosen menu of definitions and an incomplete notion of sampling uncertainty; (iii) the evidence for the key mechanism (aggregators change rates) is not statistically tested; (iv) the connection to AGI is by analogy only; and (v) there are factual mismatches in the data description (run counts in the abstract). Under this venue's rule, a mismatch between the paper and the code/data is a fatal issue, even though this one is easily corrected and does not affect estimates.

**Single most valuable change.** Replace the single "43% of variance" headline with a menu-robust, properly uncertain decomposition. That means: restrict to defensible definitions; add model-level, window and trend-form uncertainty to the "sampling" side; give CIs on the share; and add a paired-bootstrap test that bottleneck aggregators have slower doubling times. Then either broaden the domains (several suites) or retitle the paper as a METR-horizon methodology study.
