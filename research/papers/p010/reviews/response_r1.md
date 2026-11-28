# Response to Round-1 Review — p010

We thank the reviewer for a careful report and for re-running the full pipeline. The revision makes four kinds of change. It corrects every factual mismatch. It adds new analyses for each major point (`ext_bootstrap.py`, `decompose.py`, `nested_share.py`, `eb_lambda.py`, `fig_decomposition.py`). It withdraws claims the new analyses do not support. It retitles and reframes the paper as a METR time-horizon methodology study. Several of our headline claims are weaker as a result; we state where.

New title: *How Much Does the Definition Move the Forecast? Reliability Thresholds, Cross-Family Aggregation and METR Time-Horizon Extrapolations.*

**Engineering change that enabled the new analyses.** `core.fit_model` now fits on run data collapsed to (family, task-length) cells. This is exact for the weighted Bernoulli likelihood and about 15× faster. The main TH1.1 analysis (B = 500, seed 0) reproduces the round-1 `metr_1-1_lam1_summary.csv` to 7×10⁻¹⁵, and the point CSV to 9×10⁻¹⁵. CV and profile outputs reproduce to floating-point precision.

---

## Fatal issue

**F1. Run counts misstated (abstract, Chinese abstract, §4).** *Fixed.* We now report 23,235 agent runs for TH1.1 (24,008 rows include 773 human baselines). For TH1.0 we report 37,539 analysed agent runs from 30 systems released in 2023 or later (the file has 41,629 rows: 40,836 agent runs plus 793 human rows). These figures appear in the abstract, the Chinese abstract (23,235 / 37,539), §4 and `README.md`. We also verified the counts in code: `load_runs` gives 23,235 and 40,836, and the 2023+ restriction leaves 37,539. In addition we now state the data provenance precisely: METR commit `52cb829` (2026-03-05), byte-identical, with SHA-256 hashes in the README.

## Major issues

**M1. The "43%" headline depends on the menu, the share has no CI, and TH1.0 used B = 60.** *Addressed with new analyses; the headline is replaced.*
- We report the share for seven menus: all 24; 18 (drop p = +1 and min-family); 12 (also drop q = 0.95); a recommended 8 (pooled, geometric, q25, q10 × q ∈ {0.5, 0.8}); aggregator-only; threshold-only (0.5 vs 0.8); and 0.5/0.8/0.95 (Table 4, `decomp_{v}_table.csv`, Fig. 6).
- We add **nested-bootstrap CIs** on the share (TH1.1 40×100, TH1.0 30×80; `nested_share_{v}.json`), plus a robust (MAD-scale) share, because arrival dates are heavy-tailed.
- We reran TH1.0 and the λ ∈ {0.25, 4} grids with **B = 500**. The TH1.0 all-24 share under the round-1 method is now 0.26 (it was 0.29 at B = 60).
- Findings, reported honestly. On TH1.1 the menu matters little: the share is 0.52–0.60 for every menu, e.g. drop p = +1 and min 0.53 [0.38, 0.65]; 0.5 vs 0.8 only 0.58 [0.35, 0.68]. This partly rebuts the concern for TH1.1. On TH1.0 the menu matters a lot: all-24 0.32 [0.07, 0.54] falls to 0.15 [0.04, 0.41] once p = +1 and min are dropped. So the concern is correct there, and the abstract now states the TH1.0 range.
- We also found that the single full-data share depends on the Monte Carlo draw. The same specification gives 0.43 at B = 500 and 0.52 at B = 1,000, because of heavy tails. The paper states this (§5.3, item 2).

**M2. "Sampling uncertainty" omits frontier, release-date, window, functional-form and human-time uncertainty.** *Addressed.* The non-definitional side now has five layers (§4):
- U1: task resampling.
- U2: human-time error. Per-task SD of 0.5 log₂ for baselined tasks and 1.0 for estimated tasks, plus a shared SD 0.5 shift for estimated tasks. These are labelled as assumptions. We also verified the reviewer's fact: 5 of 31 tasks of 8 h or more are baseline-timed.
- U3: leave-one-system-out.
- U4: ±45-day release-date jitter.
- U5: 2023 vs 2024 window start, and OLS vs Theil–Sen.
With all layers, the definitional share falls to 0.30–0.43 on TH1.1 and 0.12–0.26 on TH1.0. The trend window is the largest single source: a 2024+ window compresses the 24 dates to 2026.8–2029.1. The paper now says explicitly that round 1 overstated definitional variance relative to total uncertainty, and the reporting protocol adds window sensitivity (item 7).

**M3. "Aggregators change rates" was not tested; the model imposes a common slope.** *Tested, and the claim is withdrawn.*
- Paired-bootstrap slope differences (Table 3). On TH1.1 under task resampling, p = −1 and q25 grow more slowly than pooled: −0.38 [−0.93, −0.06] and −0.72 [−1.03, −0.11] doublings/yr. Under U1–U5 every interval includes 0 (P(<0) = 0.76–0.88). On TH1.0 the differences are centred near 0.
- Family-specific growth rates (`family_rates_{v}.csv`). Rates are heterogeneous: SD 0.86 vs a permutation null of 0.35, p < 0.001 (TH1.0: 0.70 vs 0.28). But early-weak families *catch up*: r = −0.30 between the early level and the rate estimated on later systems only, which avoids regression to the mean from shared data. A plug-in from the new finite-horizon Prop. 2(ii) predicts −0.13 doublings/yr for p = −1 vs geometric against an observed −0.38 (TH1.0: −0.31 vs −0.03).
- §5.1 point 2 of round 1 is removed. §5.2 now states: "We therefore withdraw the round-1 claim that aggregators change rates on these data." The common-slope restriction is listed in Limitations. Family-specific *slopes* (b per family) cannot be estimated with a median of 2 tasks per family. Family-specific *growth rates* over time are estimated as above.

**M4. Extrapolation below the task support is ignored.** *Addressed with a new diagnostic.* For each definition and frontier system we check whether the determining family horizon lies more than 2 doublings outside that family's task lengths, and whether that family is saturated or floored (Table 1, new columns; `decomp_{v}.json["support"]`). Result: for 20 of the 21 family-level definitions, 62–100% of frontier points are off-support. For q25 and q10, 60–100% lie *below* the support. Only the pooled rows are fully in support. This is now a central finding (§5.4): the aggregator dimension is not identified on this suite, and Table 1's family rows are model outputs. The "Latest" column is now accompanied by these flags (minor issue m7).

**M5. The theory is standard; "four results proved" overclaims; the p = 0 discontinuity holds only as t → ∞.** *Accepted.*
- §3 is retitled "Framework: four standard facts". The abstract says "we collect four standard facts", and §1 lists the contributions as empirical and methodological.
- Prop. 2 gains part (ii), a finite-horizon first-order expansion, r_p(t) = s̄ + p[Cov_w(a, s) + t Var_w(s)] + O(p²). This shows the rate is continuous in p at any finite t, explicitly corrects the "jump" claim, and is what we test empirically (M3).
- Prop. 1 no longer assumes continuity, as the reviewer noted it need not. Prop. 4 now notes that shrinkage and heteroscedasticity limit it to qualitative guidance.

**M6. "Quantiles lie between p = 0 and p = −∞" is false.** *Corrected.* Prop. 1(c) now states that quantiles are not ordered relative to power means; only min ≤ Q_α ≤ max holds. It gives a counterexample and cites the reviewer's Table 1 instance (median 1,973 > geometric 935 min for Opus 4.6). Prop. 1(b) adds the separate ordering in α. The empirical ordering check already used only the power-mean chain, and the paper now says so.

**M7. The link to AGI is by analogy; families are not cognitive domains; 167 h is far beyond the 16-h limit; the CHC analysis is near-tautological.** *Accepted.*
- We retitled the paper and reframed it as a METR-horizon methodology study (abstract, §1). A new §7, "what this does and does not say about distance to AGI", limits the AGI claim to: the reliability dial moves extrapolated dates by about one year per 50%→80% step, and the breadth dial is not measurable on current run-level data.
- §5.7 (CHC) is relabelled "illustration only" and now says the result "is close to a tautology… We draw no forecasting conclusion from it."
- We did not broaden to other suites, because no other public source has run-level outcomes with human timings. This is stated in Limitations.

**M8. Novelty is incremental relative to Fourati, Kwa et al., Ord and METR.** *Accepted in the framing.* §1 now lists what is new: the decomposition with nested CIs and multi-source non-definitional uncertainty; the identification diagnostics (support, censoring, CV/EB/scaled shrinkage); the paired and family-rate tests; and the prespecified predictions. §2 credits Fourati for the static CHC analysis and says the Prop. 3 algebra is implicit in Kwa et al. and Ord. We do not claim theoretical novelty.

**M9. Mixed bootstrap types; "pre-registered" predictions are not registered and are weakly falsifiable.**
- *Bootstrap.* All three dispersion statistics (all families, unsaturated, lower semideviation) now use the same task-hierarchical bootstrap (`decompose.py`). TH1.1: +0.16 [0.07, 0.24], +0.04 [−0.06, 0.16], +0.35 [0.21, 0.47]; TH1.0 replicated. The conclusion is unchanged but the comparison is now like for like. `robust_dispersion.py` is kept for reference and labelled superseded.
- *Predictions.* Renamed "prespecified". We state that they are not registered with an independent registry and that we intend to file them on OSF. P1 is now falsified if *either* criterion fails. P2's band is narrowed to the 90% bootstrap band of 2.0–2.9 doublings (H₅₀/H₈₀ between 4.1× and 7.6×), computed from the bootstrap. P4 is replaced: the bottom-3 overlap rule was trivially satisfied by `automatic_jailbreak`. The new P4 is a lag-2 profile-correlation band of [0.40, 0.85], from the observed 0.40–0.82, falsified if 3 or more of the next 5 systems fall outside. P3 is labelled a test of the estimator, given M4.

**M10. A global ridge on weight-normalised runs gives different effective shrinkage per system; a random-effects model would be more principled.** *Tested (§5.5, `eb_lambda_1-1.json`).*
- (a) A run-count-scaled ridge (λ_m ∝ n_m) moves the 24 dates by at most 0.53 yr (Spearman 0.99). The specific concern is real but small.
- (b) Per-system empirical Bayes, with d ~ N(0, 1/λ_m) chosen by a Laplace-approximated marginal likelihood, i.e. a GLMM variance component. It gives λ_m with median 0.15, uncorrelated with run count, and a 24-date spread of 11.7 yr.
- (c) The same with a design-effect correction (each task counted once) gives median λ_m = 4.8 and a spread of 3.9 yr.
- Task-level CV prefers λ ≈ 1 and favours (c) over (b).
- Conclusion: the variance component is weakly identified. That is why the paper no longer reports a single spread for family aggregates, and why the abstract cites a shift of up to 5.3 years from the shrinkage rule.

## Minor issues

| # | Issue | Change |
|---|---|---|
| m1 | 0.92–0.94 mislabelled as Spearman–Brown | §5.4 now says split-half r of 0.92/0.94 (Spearman–Brown 0.96/0.97). |
| m2 | "Prop. 1" label in `theory_chc.py` / `fig_theory.png` | Relabelled "Prop. 2". The JSON key is now `prop2`. The figure was regenerated. |
| m3 | `cv_lambda.py` default `--reps 3` | Default changed to 2. The rerun reproduces `cv_lambda_1-1.json` to floating-point precision. |
| m4 | λ = 0.01 outside the CV grid | §5.5 now quotes the factor within the CV grid (162× for λ = 0.1–16; 17× for λ = 0.25–4) and flags λ = 0.01 as outside the validated range. |
| m5 | Data commit, March-2026 correction, November-2026 update | §4 and the README give commit `52cb829` (2026-03-05) with hashes. That commit includes METR's March-2026 "analysis updates" and Opus 4.6 / GPT-5.3-Codex. It was the latest commit on 2026-11-22, and the repository contains no run-level data for later systems. We could not find or verify a November-2026 METR update by web search, so we do not describe one. Any later update without run-level data could not be analysed with this pipeline. |
| m6 | Hard-coded absolute paths | All scripts now resolve paths relative to the repository (`Path(__file__).parents[1]`). |
| m7 | "Latest" column mixes extrapolations | Table 1 now has off-support and censoring columns. |
| m8 | Runtime claim | The README gives the measured total (about 45 min on 3–4 cores with the faster fitter); per-script times are noted. |

## Questions to the authors (answers)

1. Without p = +1 and min, the TH1.1 share is 0.53 [0.38, 0.65] (task resampling) and 0.30 (all sources). On TH1.0 it is 0.15 [0.04, 0.41] and 0.12.
2. Leave-one-system-out moves the pooled-50% date within 2027.4–2027.8. A 2024+ window gives 2026.8–2029.1 across all 24 definitions (round 1: 2027.1–2032.6).
3. Yes, see M3. The paired test is significant on TH1.1 only under task resampling. Family rates are heterogeneous but show catch-up.
4. See M4 and Table 1. For q25 and q10, 60–100% of frontier determining horizons lie more than two doublings below the family's task lengths.
5. Commit `52cb829` (2026-03-05), which includes METR's March-2026 updates.
6. See M10. We now report EB variance components alongside CV. They disagree by more than an order of magnitude depending on how within-task dependence is handled, which is itself a finding.

## Claims removed or weakened

- "Definition choice accounts for 43% of total variance" → a menu × uncertainty table with nested CIs. Task-only shares are about 0.5 (TH1.1) and 0.15–0.32 (TH1.0); with all sources they are about 0.3–0.4 (TH1.1) and 0.12–0.26 (TH1.0).
- "Aggregators change rates, as Prop. 2 predicts" → withdrawn (not robust and not replicated).
- "We prove four results" → "four standard facts".
- "The long-range forecast jumps at p = 0" → only in the t → ∞ limit; finite-horizon continuity is stated.
- "Quantiles lie between p = 0 and −∞" → corrected.
- "Pre-registered" → "prespecified, not externally registered"; the predictions are tightened.
- "The 5.5-year spread" → reported as conditional; the family-level rows are labelled as poorly identified.
