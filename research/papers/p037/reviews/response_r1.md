# Response to Review (Round 1)

We thank the reviewer for a careful report and for rerunning every script. We agree with all six major points; for one minor point (m7) we partly disagree and give evidence. Every new number below was produced by code in `code/`, run in this environment, with outputs in `results/`. New and changed files:

- **New code:**
  - `exp2_supplement.py`
  - `exp3_supplement.py`
  - `exp6_clustered_coverage.py`
- **Extended code:**
  - `exp1_clustering.py`
  - `exp2_retry_metr.py`: Part B now uses a finer K grid.
  - `exp5_suite_sizes.py`
  - `make_figures.py`: Fig. 3 updated; new Fig. 6.
- **New outputs:**
  - `exp2_supplement.json`, `exp2_grid_sensitivity.csv`
  - `exp3_supplement.json`, `exp3a_power_requirements.csv`
  - `exp5_chance_corrected.csv`, `exp5_attempt_counts.csv`
  - `exp6_summary.json`, `exp6_clustered_coverage.csv`
  - `fig6_clustered_coverage.png`

All original numbers that the paper still reports are unchanged. We checked that the refined Exp. 2 Part B reproduces the old values exactly at the old K grid (0 mismatches in 120 rows). Exp. 1 likewise reproduces its old values, because the new statistics consume no random numbers.

---

## Major issues

### M1. Prop. 3(b) is false for randomized tests

**Agreed; fixed.** The counterexample is correct. The old proof assumed that a procedure which ever declares at p = 0 must declare *on the event* C = 0 with probability 1.

**Change (Section 4, Prop. 3(b)).** The proposition now covers any test φ: {0..m} → [0, 1], randomized or not. Let β = φ(0) be the probability of declaring the rule-out when p = 0 (the power at an impossible task). Level α at p₀ (where pass@K = τ) gives α ≥ φ(0)·P_{p₀}(C = 0) = β(1 − τ)^{m/K}. Hence m ≥ K·ln(β/α)/ln(1/(1 − τ)) whenever β > α.

We also prove the bound is **attained**, by the test that declares only after C = 0 with probability min{1, α(1 − τ)^{−m/K}}. For non-randomized tests, β = 1 and the old formula is recovered. The paper now says explicitly that the first version's statement was false.

**Numerical check (new, `exp3_supplement.py`).** For five (m, K, τ) we solved, as an LP, for the most powerful randomized test against the composite null. The maximal power at p = 0 matches min{1, α(1 − τ)^{−m/K}} to within 4.7·10⁻¹⁶.

**Headline restated everywhere.** Abstract, contributions, Prop. 3, Fig. 3, Section 7 and the Conclusion now say "28.4·K for non-randomized tests (power 1 at p = 0)". New table in `exp3a_power_requirements.csv`: m/K = 28.43 / 27.43 / 26.32 / 21.85 for β = 1 / 0.9 / 0.8 / 0.5 at α = 0.05, τ = 0.1. The message, that effort is proportional to K, survives. Lowering the required power helps only logarithmically, which we now state. Fig. 3 adds a dashed β = 0.5 line.

### M2. Limited novelty; missing classical literature

**Agreed.** We now position the work honestly; we added the literature and scaled down our claims.

- **New references.** Each was verified to exist by web search:
  - Teicher 1963 [27];
  - Tian, Kong and Valiant, NIPS 2017 [28];
  - Good and Toulmin 1956 [29];
  - Orlitsky, Suresh and Wu, PNAS 2016 [30];
  - Kazdan, Schaeffer et al. 2025, arXiv:2510.05197 [31];
  - Andrews 2000, Econometrica [32];
  - Sison and Glaz 1995, JASA [33];
  - Wei et al., NeurIPS 2025 D&B [34];
  - Kish 1965 [35].
- **New paragraph "What is classical and what is ours" (Section 1).** It also adds two new Related-Work paragraphs: "Binomial mixtures and population-of-parameters estimation" and "Unseen species". These state the following:
  - Lemma 1 and the K ≤ m dichotomy are classical restatements ([27, 28]). Lemma 1 is now labelled "classical, cf. [27, 28]".
  - Tian et al.'s optimal O(1/t) rate rests on the same moment fact.
  - Unseen-species extrapolation is the structural analogue, with its log n limit [30]. Our version is per task, with small m, and exact at every n.
  - Prop. 3(a) is a monotone transform of Clopper–Pearson, stated for convenience.
  - Prop. 4 is elementary; its value is the explicit threshold.
  - Kazdan et al. and Schaeffer et al. are described as the parametric counterpart and as complementary (useful for "ruling in"), not as straw baselines.
- **Contribution, stated plainly.** (1) Applying partial identification to rule-out claims, with the sharp LP for pass@K. (2) The ε* rate (see m2). (3) The corrected, attained sample-size bound with power. (4) The explicit floor threshold. (5) The empirical measurements and the new cluster-sampling coverage study.

### M3. The hybrid box is validated only without clustering

**Agreed; new experiment, and the claim is reduced.**

**New Experiment 6 (`exp6_clustered_coverage.py`, Section 6.5, Table 4, Fig. 6).**

- **Structures.** Three family structures:
  - the 79 real METR family sizes (228 tasks, m = 4);
  - a CTF-sized suite of 40 tasks in 14 unequal families (m = 5);
  - a skewed suite of one family of 60 plus 40 singletons (m = 4; Kish n_eff 2.8).
- **Scenarios.** Six difficulty scenarios with family random effects (logit-normal), including family-level impossibility. Three of them produce all-fail data in 5–72% of replications.
- **Replications.** 300 per cell.
- **Boxes compared.** Four:
  - the percentile cluster bootstrap;
  - the Clopper–Pearson part alone;
  - the hybrid box with n = number of families (the box used on METR);
  - a new hybrid box with Kish's effective number of families.
- **Targets.** The population count pmf (simultaneous box coverage) and population pass@K for K ∈ {1, 10, 100, 1000}.

**Findings, reported as they came out:**

- **The reviewer was right about the exact part.** With n = n_families, the Clopper–Pearson box alone covers the count pmf in only 0.49–0.73 of replications under skewed family sizes.
- **The hybrid box held up.** Box coverage was 0.95–1.00 in all 18 cells. Coverage of the pass@K upper bound was ≥ 0.997 in all 72 (structure, scenario, K) cells, including the all-fail subsets.
- **The Kish variant was most conservative.** Box coverage was ≥ 0.997, at a width cost: moderate on METR and CTF structures, large under skew.
- **The percentile cluster bootstrap undercovers broadly, not only at all-zero data.** Its box coverage is 0.30–0.82 in benign scenarios. Its pass@10 bound covers 0–0.43 in 11 of 18 cells.

**Text changes.**

- "Restores nominal coverage" is replaced by "was never below nominal in our settings; no proof of validity; the CP component alone is not valid".
- We recommend the Kish variant for skewed family sizes (Section 7, item 3).
- The limitations list the box as a heuristic.
- The METR conclusions do not depend on the choice: all upper bounds are ≥ 0.98 at K = 30 even with the less conservative box.

**"Commonly used" and "finding" claims.** Removed. We cite Andrews [32] for bootstrap inconsistency at the boundary. Section 2 now says that the all-fail failure is an instance of a known phenomenon, which we include only because it occurs in the rule-out regime. The abstract no longer presents it as a finding.

### M4. Held-out validation (Table 2) is weak evidence

**Agreed; reframed, and large-K validity moved to simulation.**

- **Section 6.2 is retitled "Held-out sanity check (small K)".** It now has a "What this table does not show" paragraph listing (a) K ≤ 6, (b) width 0.40–0.47, (c) a noisy, same-task target and 5,640 dependent comparisons, and (d) a population vs observed-task target.
- **New per-model aggregation (`exp2_supplement.py`).** Every one of the 19–20 models had LP-CI ≥ held-out value in every split. We say the informative unit is the model, not the split.
- **A further caveat we found ourselves.** In the m = 2 configurations the held-out pass@K has median 0.996, since tasks with ≥ 6 runs are mostly easy. Near-zero margins there are not evidence of tightness. This is now stated.
- **Large-K coverage against a population target.** This is now provided by Experiment 6 (M3), for K up to 1000 under realistic clustering.

### M5. The design-effect headline conflates several things

**Agreed; headline replaced.** `exp1_clustering.py` now also computes a runs-as-replicates SE for the task-weighted estimand and analytic CR1 cluster-robust SEs at the task and family levels.

- **Abstract and Section 6.1 now lead with the between-family effect.** Family vs task resampling gives a median design effect of 4.26 (range 2.23–7.93), an SE ratio of 2.06 and CR1 agreement (4.24). About 227 tasks carry the information of about 54 independent tasks. The 95% margin grows from 4.7 to 9.8 points.
- **The old 4.5× / deff-20 comparison is kept only as a decomposition.** The median SEs are 0.0128 (pooled naive), 0.0062 (runs-as-replicates), 0.0285 (task) and 0.0594 (family). The text explains that only the family-over-task ratio is a substantive design effect.
- **Section 6.7 (old 6.6).** It already used the family-vs-task figure 4.26 for the transferred design effect; this is now stated consistently.

### M6. The link to bio/chem/cyber is indirect

**Partly addressed. The limitation is stated honestly. We could not obtain a cyber dataset.**

- **Attempt to obtain data.** We tried to obtain a public run-level repeated-attempt CTF dataset (e.g. the InterCode-CTF logs of the Palisade "plain agents" study). Access to that repository was not available in our sandbox, and we did not find a downloadable alternative. We do not claim to have analysed cyber data. The Limitations now say this explicitly.
- **Cyber-specific additions that do not need run-level data.**
  - Section 6.7 applies Prop. 3 to the repeated-sampling counts that Wei et al. [34] tabulate for published cyber evaluations (3–30 attempts per task) (`exp5_attempt_counts.csv`). With zero successes in m = 30 attempts, the 95% bound on pass@10 is still 0.63. No non-randomized test can rule out pass@K ≥ 0.1 for any K > 1 at m = 30. About 285 failures are needed for pass@10.
  - Experiment 6 includes a Cybench-sized (40-task, 14-family) structure.
  - The pre-registered CTF protocol (P1–P4) is kept.
  - Wei et al. [34] is cited for the empirical point that a small adversary compute budget, including repeated sampling, raises InterCode-CTF success by more than 40%.
- **Verification assumption.** It is now stated in the abstract, the model section and the limitations. The K-scaling message applies most directly where success is machine-checkable (cyber/CTF, software). For many biological and chemical steps the attacker quantity is closer to pass@1 under selection, and the K-scaling results do not transfer directly.

---

## Minor issues

- **m1 (186–228 tasks).** Fixed to "141–228 tasks per model". `exp2_extrapolation.csv` now has n_tasks ∈ {141, 186, …, 228}.
- **m2 (Prop. 2 is nearly immediate; the rate is loose).** Agreed. The text now says that the limit is nearly immediate and that the contribution is the computable rate. New tightness check (`exp3_supplement.py`, Table 3): exact integer K at which U_K first reaches 0.9, found by bisection, against the K implied by the bound. Ratios are 1.0–3.7 at the 0.9 level and 1.04–2.1 at 0.99. The bound is exact or near-exact at m = 2 and loosens as m grows and ε* shrinks. For example, Beta(2, 2) at m = 8: exact K = 6, bound K = 22.
- **m3 (slack relaxation).** Quantified (`exp2_supplement.json`). The relaxation was **never** used for the LP over a confidence box: 0 of 5,640 validation rows and 0 extrapolation rows. No LP-CI number depends on it. It was used only for the point LP:
  - in 8.1% of validation rows (14.1% at m = 2), with slack ≤ 4.1·10⁻⁴;
  - in 20% of extrapolation rows, with slack ≤ 0.0036.

  Relaxed splits undercover far more often (80% vs 15%). The paper reports this as another reason not to use the point bound for claims. Grid sensitivity (reviewer Q5): floors 10⁻⁵ / 10⁻⁷ / 10⁻⁹ and grid sizes 600 / 2,000 change the LP-CI bound by ≤ 2.2·10⁻⁵ and the point bound by ≤ 1.3·10⁻³ for K = 10–10⁵.
- **m4 (28.4 vs 29.0).** Clarified in Section 4. 28.43 is the exact ratio; 29 is ⌈28.43⌉, the integer requirement at K = 1, which is what `exp3_summary.json` stores.
- **m5 (Bonferroni is conservative).** Agreed. Sison–Glaz [33] and moment-space confidence sets are now named as tighter alternatives in Sections 2 and 8. We did not implement them in this revision. Experiment 6 shows the current box is conservative rather than anti-conservative, so tightening it is an efficiency improvement and cannot change Prop. 2's conclusion. We state this as a limitation.
- **m6 (multiple-choice suites).** Fixed. VCT is removed from the zero-success calculation. WMDP is now chance-corrected (four options; `exp5_chance_corrected.csv`): at chance-level accuracy the above-chance rate is bounded by 1.6% (independent items) or 3.4% (transferred design effect). The Cybench numbers stand, since flag capture has essentially zero chance rate.
- **m7 (duplicate models).** **Partly rebutted.** The 20 aliases are 20 distinct model checkpoints. "(Inspect)" denotes the harness, not a scaffold variant of another listed model. The reviewer's example, "o1 (Inspect)" vs "o1-preview", is two different OpenAI models. The underlying point is correct, though: the models share developers and lineages and are not statistically independent. The paper now says so (Section 5, Section 8), calls cross-model medians descriptive, and notes that no cross-model test is run.
- **m8 (coarse K grid in Fig. 2).** Fixed. 14 K values from 1 to 1,000. The finer grid revealed a sharper statement, which is now in the paper: every model's 90% bound is ≥ 0.98 by K = 30 (minimum 0.985), and ≥ 0.938 at K = 18.
- **m9 (compare the protocol with existing guidance).** Section 7 now says which items are covered by generic evaluation-statistics guidance (clustered SEs [12], exact small-n intervals [13]; items 3–4). It says which are specific to the attacker-budget problem (items 1, 2, 5, 6), and links item 1 to Wei et al.'s argument for dynamic, budget-aware cyber risk assessment [34]. We do not make claims about the contents of the developers' frameworks [1–3] beyond what we cite.

## Citations flagged as unverifiable

- **[17] METR Time Horizon 1.1 blog post (2026-01-29).** Verified by web search; the URL resolves to the METR post dated January 29, 2026.
- **[26] commit hash.** We could not reach GitHub to re-verify the hash in this environment. The reference now reads "commit as recorded at download". The Limitations section notes the hash was not re-verified online, while the file contents match the description (as the reviewer also confirmed).
- **[11] Best-of-N Jailbreaking.** Verified as NeurIPS 2025 main track (proceedings and poster page).
- **[23] AI Sandbagging.** Verified in the ICLR 2025 proceedings.

## Questions for the authors

1. **Restate Prop. 3(b)?** Yes. It now holds for all tests, with power β at p = 0, and is attained. 28.4·K is the non-randomized / β = 1 case (M1).
2. **Coverage under genuine cluster sampling?** See Experiment 6 (M3). The hybrid box was ≥ nominal everywhere tested. Its exact component alone is not valid under skew. The Kish variant is recommended for skewed family sizes.
3. **Relation to unseen species and Tian–Kong–Valiant; is the ε* rate new?** See M2. We are not aware of the ε* rate for this functional in that literature, but we present it modestly, with its looseness quantified (m2).
4. **A CTF dataset?** We could not obtain run-level data (M6). Instead, Section 6.7 uses the attempt counts reported in published cyber evaluations [34] to illustrate m/K, as the reviewer suggested.
5. **Sensitivity to the grid floor and the slack?** Negligible for the grid (≤ 2.2·10⁻⁵ on LP-CI bounds). The slack never touches LP-CI bounds (m3).
