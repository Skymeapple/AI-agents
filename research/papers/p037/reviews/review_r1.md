# Review (Round 1): "What Can a Finite Evaluation Rule Out? Partial Identification of Attacker Success in Dangerous-Capability Evaluations"

Reviewer role: area chair level, independent. Research question: "How should dangerous capabilities (biological, chemical, cyber, etc.) be evaluated and governed?"

## Summary

The paper looks at rule-out claims in dangerous-capability evaluations as upper-confidence-bound claims about an attacker's pass@K, where the evaluator observes only m attempts per task. In the theory part:

- (Lemma 1, Prop. 1) Binomial(m) count data identify exactly the first m moments of the task-difficulty distribution G. So pass@K is point-identified if and only if K <= m. Otherwise its sharp identified set is the value range of a linear program, with a closed form for m = 1.
- (Prop. 2) For moment vectors in the interior of the moment space, the sharp upper bound U_K tends to 1, with U_K >= 1 - (1 - eps*)^K.
- (Prop. 3) An "attacker rule of three": the bound 1 - alpha^{K/m}, and a sample-size lower bound m >= K ln(1/alpha) / ln(1/(1 - tau)).
- (Prop. 4) A declared "resolution floor" gives a finite bound on pass@infinity.

Empirically, on 22,236 public METR Time-Horizon 1.1 agent runs (20 model configurations, 228 tasks, 79 families), the paper:

- measures a family-cluster design effect;
- validates pass@K extrapolators against held-out runs for K <= 6;
- shows that m = 4 cannot bound suite pass@100 below 0.9999;
- shows in simulation that a percentile cluster bootstrap degenerates when every attempt fails, and that a bootstrap/Clopper-Pearson hybrid box fixes this.

It ends with a reporting protocol.

## Verification performed

**Code rerun.** I copied `code/` and `data/` to a scratch directory and reran all six scripts from scratch in the order given in the README. The rerun reproduces exactly (byte-identical CSV/JSON) every output I compared, including:

- the Exp. 1 clustering table and summary;
- the Exp. 2 validation summary and extrapolation table;
- the Exp. 3 a/b/c outputs;
- the Exp. 4 floor summary;
- the Exp. 5 suite-size table.

The details are in the "Rerun result" note at the end. I also recomputed the data-cleaning counts from the raw JSONL: 24,008 records, 773 human, 999 fatal-error, 22,236 retained, 228 tasks, 79 families, 20 aliases. All match.

**Numbers checked against outputs.** These match:

- Section 6.1: SE medians 0.0128 / 0.0286 / 0.0594; ratio 4.48 (3.13 to 5.89); design effect 20.1; n_eff 53; margins 2.1 to 9.8 pp; family-versus-task design effect 4.26 (2.23 to 7.95).
- Table 2: all cells; 5,640 split-level comparisons with 0 undercoverage.
- Section 6.3: 0.838 / 0.975 (min 0.840) / >= 0.99999; point-U median 0.995 (min 0.78); lower end 0.619; beta-binomial 0.868 (0.606 to 0.976); zero-share median 0.246 (0.06 to 0.50).
- Table 1: all cells; 31% all-zero replications.
- Section 6.4: coverage 0.952 to 1.000; the 97% versus 0% rule-out rates; eps* = 0.167 / 0.071 / 0.025.
- Section 6.5: 0.956 and 0.886.
- Section 6.6: all bounds and effective sizes.

One small mismatch: Section 6.3 says "186–228 tasks per model". The file `exp2_extrapolation.csv` gives n_tasks from **141** to 228. This is a text error, not a computational one, but it must be fixed.

**Proofs checked line by line.**

- Lemma 1: correct (Bernstein basis argument).
- Prop. 1: correct. The hypergeometric identity E[C(m-C, K)/C(m, K)] = E[(1-p)^K] holds, and the sign of the (m+1)-th derivative is computed correctly. The non-degeneracy step for K > m relies on Karlin and Studden's principal representations, which is acceptable as a citation-based sketch.
- Cor. 1: correct (Jensen for the upper bound; f_K(p) >= p for the lower bound).
- Prop. 2: correct. The support-function separation argument is fine, and closedness of the moment space at eps* follows by weak compactness. This could be stated in one line.
- Prop. 3(a): correct (one-sided Clopper-Pearson with zero successes).
- **Prop. 3(b) is false as stated.** It claims that *any* level-alpha procedure that declares "pass@K < tau" *with positive probability* when p = 0 needs m >= K ln(1/alpha) / ln(1/(1 - tau)). Counterexample: a randomized test that declares the rule-out with probability alpha whenever C = 0 (and never otherwise). Its rejection probability is at most alpha at every p, so it is level-alpha for any m, and it declares with positive probability (alpha) at p = 0. The proof's step "a procedure that ever rules out at p = 0 must rule out on the outcome C = 0" silently assumes a non-randomized test. The correct statement needs one of two fixes:
  1. restrict to non-randomized procedures, or
  2. require the procedure to declare with probability 1 when p = 0 (power 1 at p = 0). More generally, power beta at p = 0 gives m >= K ln(beta/alpha) / ln(1/(1 - tau)).

  The arithmetic is correct (28.43, 58.4, 298.1).
- Prop. 4: correct, including the threshold 1 - 0.25^{1/4} = 0.293 and 0.8^4 = 0.41.

**Reference spot-check.** I attempted web verification. The session's WebSearch budget was already exhausted, Tavily returned a plan-limit error, and arxiv.org is blocked by the sandbox's network policy. Independent online verification was therefore impossible in this session. I checked the following references against my own knowledge of the literature (cutoff mid-2026). Each exists and supports the use made of it:

- [11] Best-of-N Jailbreaking: 89% ASR on GPT-4o with 10,000 augmented samples.
- [5] VCT: 322 multimodal virology questions.
- [9] Barnett and Thiergart, "Declare and Justify".
- [13] Bowyer et al., ICML 2025 position paper on the CLT.
- [18] Schaeffer et al.: the power law arises from a heavy left tail of per-problem success probabilities.
- [19] Jones et al., Forecasting Rare Language Model Behaviors.
- [12] Miller, Adding Error Bars to Evals.
- [6] Cybench: 40 tasks, ICLR 2025.
- [15] Chen et al.: the unbiased pass@k estimator.
- [25] Hanley and Lippman-Hand 1983: rule of three.

I could not confirm two references from knowledge:

- [17] the METR "Time Horizon 1.1" blog post dated 2026-01-29;
- the exact commit hash in [26].

However, the local data file is consistent with the description in every detail I could check. I list these as unverified, not as fabricated. Two venue tags are also unconfirmed: [11] "NeurIPS 2025" and [23] "ICLR 2025".

## Strengths

1. **A sharp, well-posed question.** The question is the gap between the evaluator's m and the attacker's K, framed as partial identification. Rule-out claims are the dominant use of dangerous-capability evaluations, and the paper correctly recasts them as upper-confidence-bound claims. This framing helps the safety-framework community.
2. **Clean, mostly correct theory.** The Lemma 1 / Prop. 1 dichotomy (K <= m is identified, K > m is not) is crisp. The m = 1 closed form usefully shows that "pass@1 reused" is the lower end of the identified set and the homogeneous formula is the upper end.
3. **Fully reproducible and honest empirics.** Every number traces to a script output, and the rerun reproduced them. Assumptions such as the design-effect transfer in Section 6.6 are clearly labelled.
4. **Useful negative results.** The beta-binomial extrapolation errs in both directions under misspecification (Table 1, mostly-impossible population). Four runs per task are uninformative about pass@100. Both are useful for practitioners who currently report parametric extrapolations.
5. **Testable predictions.** The pre-registered falsifiable predictions (P1 to P4) and the limitations section are good practice.

## Weaknesses

### Major

**M1. Prop. 3(b) is incorrect as stated (randomized tests).** See the counterexample above. This is the paper's headline "28.4·K" requirement, and it appears in the abstract, the contributions list, Section 7 and the conclusion. The fix is easy (non-randomized tests, or power 1 at p = 0), but as written a stated theorem is false.

**M2. Limited novelty relative to classical statistics, and missing related work.**

- Identifiability of binomial mixtures up to m moments is classical (Teicher 1963, "Identifiability of finite mixtures"). Estimating the distribution of per-unit success probabilities from Binomial(m) counts, and the limits of doing so, was studied directly by Tian, Kong and Valiant (2017, "Learning populations of parameters"). Neither is cited.
- Extrapolating from m samples to K > m samples is the same structural problem as unseen-species / Good-Toulmin extrapolation. There the impossibility of extrapolating far beyond the sample is well known (e.g. Orlitsky, Suresh and Wu 2016). This lineage is not discussed.
- Recent pass@k-scaling work that fits beta-binomial models (e.g. Kazdan, Schaeffer et al. 2025, on efficient prediction of pass@k scaling) is the natural parametric baseline, but it is not cited. Instead the beta-binomial is presented as a generic straw baseline.
- Prop. 3(a) is a monotone transform of Clopper-Pearson, and Prop. 4 is a two-line rearrangement.

The contribution is mainly the *framing* and its application to rule-out claims, plus the eps* rate in Prop. 2. That is valuable but incremental for a NeurIPS main-track methods paper. The authors should position the work honestly against this literature.

**M3. The "bootstrap fix" is validated only in a setting that does not test it.** The hybrid box takes the Clopper-Pearson interval with x = p_hat × n_clusters and n = n_clusters. With unequal cluster sizes and within-cluster heterogeneity this is not an exact interval; it is a heuristic. Yet the paper claims that it "restores nominal coverage". Experiment 3c, the only coverage study, uses `clusters = np.arange(n)`, i.e. every task is its own cluster. So the clustered case that motivates the construction, and that is used on METR, is never tested. The finding that a percentile bootstrap degenerates when all observations are zero is well known and needs no simulation. Presenting it as a finding ("a commonly used percentile cluster bootstrap collapses", coverage 0 to 4%) overstates it. No citation shows that rule-out evaluations actually use this bootstrap. A clustered coverage simulation (families of varying size, correlated difficulty) is needed.

**M4. The held-out validation (Table 2) is weak evidence of validity.**

- (a) It covers only K <= 6. The paper's important claims are about K = 100 to 1,000.
- (b) The LP-CI bound has an average width of 0.40 to 0.47, so "0 of 5,640 undercover" is unsurprising and says little about calibration.
- (c) The "truth" is itself a noisy unbiased estimate computed from a handful of held-out runs on the *same* tasks. The 5,640 comparisons are heavily dependent (60 splits of the same data per model), so they are not 5,640 independent coverage trials.
- (d) The LP-CI targets a population over families, while the held-out target is pass@K on the observed tasks.

The validation should be framed as a sanity check, not as evidence that the procedure is valid.

**M5. The design-effect headline conflates several things.** The "naive" SE uses the pooled run mean with N runs as if the runs were independent. The estimand, however, is the equal-task-weighted mean, so the 4.5-fold ratio mixes three effects:

1. weighting mismatch;
2. repeated runs on the same task not being independent replicates of the suite (obvious to most practitioners);
3. genuine between-family clustering.

Only (3) is a statistically interesting design effect, and it is 4.26 (family versus task). The abstract headlines the 4.5-fold / deff-20 number, which is partly a straw man. The abstract should lead with the family-versus-task number, or justify the naive baseline as actually used in practice (with citations).

**M6. The link to the stated research question (bio/chem/cyber) is indirect.** No dangerous-capability data are analysed. The single real dataset is a general software suite. The model assumes that the attacker (a) runs i.i.d. attempts with the evaluator's per-attempt probability and (b) can verify success. For bio and chem pathways (b) is often false, as the limitations admit, so the central K-scaling message applies most cleanly to cyber. The paper would be stronger with at least one cyber or CTF repeated-attempt dataset; Cybench and similar suites publish multi-attempt results, and if those exist this matters. Without one, the significance for the stated question rests on theory plus an analogy.

### Minor

- m1. Section 6.3: "186–228 tasks per model" should be 141 to 228 according to `results/exp2_extrapolation.csv`.
- m2. Prop. 2's practical content, U_K -> 1 for interior points, follows almost immediately: any G with no atom at 0 and the right moments drives θ_K -> 1. State this up front and present the eps* rate as the contribution. Also, eps* is tiny in realistic settings (0.025 at m = 8 for Beta(0.5, 1.5)), so the rate bound is loose. A comparison of the bound with the actual U_K curve would be informative.
- m3. The LP slack relaxation (Section 4, "smallest uniform slack") is a heuristic with no coverage guarantee. It was needed in about 20% of m = 2 splits on the ≥6-run configuration (14% over all m = 2 rows). Its effect on the reported bounds should be quantified.
- m4. Exp. 3a reports m/K = 29.0 (a ceiling) in the summary JSON, while the text says 28.4. Both are fine, but say which one is meant.
- m5. Bonferroni over m + 1 cells of a pmf that sums to 1 is conservative. A multinomial simultaneous region (e.g. Sison-Glaz) or a direct confidence set for the moment vector would tighten the lower bounds. The lower bounds matter for the "ruling in" side.
- m6. Section 6.6 transfers METR's design effect to WMDP and VCT, which are multiple-choice with nonzero chance accuracy. The bounds are labelled as orders of magnitude, but a chance-corrected version would avoid misreading.
- m7. The 20 "models" include duplicate scaffold configurations of the same model (e.g. "o1 (Inspect)" alongside "o1-preview"). Medians across them are not across independent models.
- m8. Only K = 1 to 1,000 is reported in Fig. 2. The x-axis gap between K = 1 and K = 4 makes the curve shape partly an interpolation artefact.
- m9. The protocol (Section 7) is reasonable but admittedly "not new statistics". Consider comparing it explicitly with existing guidance in the frameworks [1–3] to show what changes.

## Questions for the authors

1. Can you restate Prop. 3(b) for non-randomized tests, or with a power requirement at p = 0, and confirm that the 28.4·K figure is stated under that condition?
2. What is the coverage of the hybrid box under genuine cluster sampling with unequal family sizes and correlated within-family difficulty?
3. How does your framework relate to the unseen-species / Good-Toulmin extrapolation limits and to Tian–Kong–Valiant's population-of-parameters estimation? Is the eps* rate new relative to these?
4. Is there any public dangerous-capability (e.g. CTF) dataset with repeated attempts on which you could run Experiment 2? If not, could you at least use the attempt counts reported in published system cards to illustrate m/K?
5. How sensitive are the METR conclusions to the grid floor of 10^-7 and to the slack relaxation?

## Scores (1–10)

- Novelty: 5
- Rigor: 5
- Correctness: 6 (one false proposition statement, which is fixable; all numbers reproduce)
- Significance: 6
- Clarity: 8
- Reproducibility: 9

**Overall: 5 / 10 (weak reject at a top venue; workshop-ready).**

## Decision

**Major revision.** The paper is well written, fully reproducible and asks the right question. Its main practical message, that rule-out claims need effort proportional to the attacker budget and that parametric pass@K extrapolations carry the conclusion, is valuable to the dangerous-capability evaluation community. To reach the bar, it must:

- fix the false general claim in Prop. 3(b);
- position the identification results against the classical binomial-mixture and unseen-species literature;
- validate the hybrid confidence box under actual clustering;
- reframe the design-effect headline.

Ideally it should also add a cyber or CTF repeated-attempt dataset, so that the evidence actually addresses biological, chemical and cyber evaluations.

---
*Rerun result:* I reran the experiments in a scratch copy using the fixed seeds. The following outputs are byte-identical to `results/`:
- exp1_clustering.csv and exp1_summary.json
- exp2_extrapolation.csv, exp2_validation.csv and exp2_validation_summary.json
- exp3_summary.json and exp3a/3b/3c CSVs
- exp4_floor.csv and exp4_summary.json
- exp5_suite_sizes.csv

I did not regenerate the figures in full. I inspected them (Figs. 1 and 2), and they agree with the CSVs. Total runtime was about 25 minutes on CPU, which is consistent with the README.
