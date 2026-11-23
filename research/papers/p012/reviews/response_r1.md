# Response to Review R1: "Familiar Is Not Known"

We thank the reviewer for a careful, reproducing review. It identified real problems. The most important were the familiarity confound in E1 and the threshold artefact in the dissociated-cell AUROC, and both changed our conclusions. Every new number below comes from code we wrote and ran in this environment:

- `code/train_cache.py`, `code/analysis_r1.py` and `code/aggregate_r1.py`;
- results in `results/r1/summary_r1.txt` and `results/r1/*_w*.json`;
- figures `results/r1/fig_r1_prop1_strata.png` and `results/r1/fig_r1_prop2_gap.png`.

R0 code and results are unchanged and still reproduce. We also renamed the paper's title to reflect its revised, mainly methodological contribution.

**Summary of what changed in the claims**

| R0 claim | R1 status |
|---|---|
| MI is flat in α "because it separates ignorance from ambiguity" | Confounded in R0. MI survives exposure matching (0.96–0.98), but a fact-exposure count gives 1.000, so MI is reported as a fact-exposure detector |
| "Only parameter disagreement remains reliable" | Withdrawn. Under imperfect memorization, MSP matches MI on seen-fact strata |
| Hidden probe dissociated AUROC 0.245 (below chance) | Reported as specific to perfect memorization and largely a threshold artefact. Within-stratum AUROC is about 0.5 in all 7 regimes |
| Prop. 1 binds as "calibration pushes δ → 0" | Refuted as stated. Prop. 1 binds under *shape matching*. We show exact tightness for a calibrated Bayes learner in a shape-matched world, and show that calibration alone gives δ = 1 |
| Prop. 2 is a "lower-bound-style diagnostic" | Dropped. It is a qualitative model. Its mechanism is validated by empirical LDA (r = 0.97), and the closed form is poor (r = 0.36) |
| Good–Turing is an accurate abstention budget | Withdrawn outside (near-)perfect memorization |
| q "fails in principle" | Replaced by a conditional statement (Prop. 1 + Corollary 1 + shape matching) |

---

## Major issues

### M1. The test bed is degenerate; no LLM experiment.

**Partly fixed, partly not fixable here. The limitation is stated.**

We first tried to run a CPU LLM pilot with a sub-1B model on PopQA, since a 7–8B model needs a GPU and none is available. The PopQA TSV was reachable (raw.githubusercontent.com returned 200). huggingface.co and its CDN, however, are blocked by this environment's network policy (HTTP 403 "no rule allows this host"), and no other host of model weights we tried was reachable. We therefore could not run any LLM, and we say so in Section 7 and the Limitations. We did not describe any LLM result.

We addressed the specific degeneracies the reviewer listed by adding six new regimes (Table 0 and Section 5):

- **Imperfect memorization.** In `early` (3 epochs), `small` (width 32) and `matched_early`, rare-seen accuracy is 0.55–0.76 and only 23–38% of errors fall on never-seen facts (Table 1). K is no longer ≈ 1[c ≥ 1].
- **Unseen names composed of trained sub-tokens** (`comp`). The first sub-token carries type information, so unseen-entity accuracy rises from 0.06 to 0.29 and unseen names are no longer random-embedding OOD.
- **A Bayes-reference learner** gives the exact predictive distribution of the generative model with oracle types. It is calibrated (ECE 0.002) and uses type structure (unseen-fact accuracy 0.62). This answers the point that the networks do worse than a type-default predictor.

Several R0 conclusions turned out to depend on the degenerate regime, and they are now reported as such (C4 in the paper):

- the below-chance dissociated AUROC;
- "only MI is reliable" (in the imperfect regimes, MSP ≈ MI within seen strata: early FS 0.956 vs 0.948, RS 0.828 vs 0.806; small RS 0.912 vs 0.921);
- the Good–Turing budget.

We accept that "K ≈ fact seen" remains true in every world we built, so MI cannot be distinguished from a fact-exposure detector. This is now stated explicitly in Section 6.3 and the Limitations.

### M2. E1 is confounded by familiarity; the count baseline was left out.

**Fixed.**

- Table 2 now includes the −log entity-count baseline (0.748 → 0.992) and a −log fact-count exposure oracle (0.924 → 1.000), recomputed over 5 worlds.
- New **exposure-matched E1** (Table 3). All items come from familiar entities (n_e ≥ 50). Positives are never-seen facts answered wrongly. Negatives are resampled to the positives' entity-count histogram. The entity-count baseline falls to 0.497 / 0.525 / 0.555 at α = 0 / ½ / 1, so matching works up to a small residual at α = 1. In this matched design, ensemble MI scores 0.984 / 0.973 / 0.962. **MI does not collapse**, so R0's MI result was not purely familiarity. The fact-count oracle is 1.000, however, and the paper now says MI is a fact-exposure detector in these worlds.
- The same pattern holds in all 7 regimes (T-B): MI ≥ 0.93, entity count ≤ 0.59, fact count 1.000.
- The sentence crediting MI with "separating ignorance from ambiguity" has been removed.

### M3. Neither proposition is validated on trained networks.

**Prop. 1: tested directly; the conjecture was refuted and replaced.**

- We added **Corollary 1**: δ_T is invariant to any order-preserving injective recalibration, such as temperature scaling, so post-hoc recalibration cannot move the ceiling.
- We built a **shape-matched world**. Ignorance about a deterministic fact and knowledge of an aleatoric fact give the same sorted predictive distribution under the true generative model. For the calibrated Bayes learner there, δ_T = 0, and the best trained q-only detector gives 1.000 / 0.752 / 0.500 at α = 0 / ½ / 1, against a ceiling of 1.0 / 0.75 / 0.5. **This is the first demonstration in the paper that Prop. 1 binds, and it binds exactly.**
- The same Bayes learner in the original world has δ_T = 1. Calibration alone therefore does *not* drive δ → 0, and we state that R0's conjecture is false as stated.
- We trained calibrated variants, as the reviewer requested (Table 4). Early stopping lowers ECE (0.267 → 0.067) and δ̂ (0.785 → 0.513). Label smoothing lowers ECE (0.261 → 0.117) but *raises* δ̂ (0.906 → 0.973). No trained MLP gets below δ̂ = 0.51.
- Table 4 now reports δ̂, the best q-only AUROC and the implied ceiling side by side for 9 learner/regime pairs.

**Prop. 2: misfit diagnosed.**

- We compared the observed probe with LDA fitted to the actual hidden-state covariance of the same training set. Across 69 runs, empirical LDA matches the probe with r = 0.973 and MAE 0.028. The two-vector closed form gives r = 0.358 and MAE 0.148 (Table 6, `fig_r1_prop2_gap.png`). The mechanism (a linear discriminant on co-occurring F and K) is therefore right, and the two-vector Gaussian summary of the geometry is what fails.
- We removed the "lower-bound-style diagnostic" wording.
- Prop. 2 is now presented as qualitative only. Its sign prediction (FK > ½ at natural π) holds in 6 of 7 regimes, and the predicted monotone benefit of mixing holds in 6 of 7. The exceptions are named: `small` for the sign and `matched_ls` for the monotonicity.

### M4. Limited novelty of the theory; missing citations.

**Accepted. Claims were reduced and citations added.**

- Prop. 1 is now presented as "an elementary corollary" of the ROC/TV inequality and of AUROC's linearity in the negative mixture.
- Prop. 2 is presented as "an instance of the spurious-feature model". We now cite Sagawa et al. (ICML 2020, PMLR 119:8346–8356), Nagarajan et al. (ICLR 2021) and Rosenfeld et al. (ICLR 2021), all verified by web search.
- The empirical familiarity/popularity confound in LLMs is attributed to prior work:
  - Cheang et al., with the updated Findings of ACL 2026 title, pp. 713–730, verified on the ACL Anthology;
  - Ni et al. 2025, arXiv:2505.17537. We verified that its later arXiv version carries the title "Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity".
- The Related Work now says explicitly that "we do not claim to discover it".
- We did not cite the AUC–TV separation results or the Kairouz et al. characterization the reviewer mentioned, because we could not pin down a specific verifiable reference. Instead we call the inequality standard/textbook.
- The paper's contribution has been reframed around **evaluation design**: exposure matching and within-stratum AUROC, plus a multi-regime controlled study with explicit negative results. The theory is described as elementary in the Contributions, Section 4 and the Limitations.

### M5. Dissociated-cell AUROC is partly guaranteed by construction.

**Fixed. The reviewer was right, and this changed a headline.**

- New within-stratum AUROC for every detector and regime (Table 5, T-D, `fig_r1_prop1_strata.png`, right).
- The hidden probe within FU (familiar, never-seen fact: correct vs wrong) is 0.45–0.52 in all 7 regimes. Within RS it is 0.50–0.57 where measurable, and within FS 0.65–0.72 (against MSP/MI 0.94–0.98).
- The pooled dissociated value of 0.239 appears only in the base regime. Elsewhere the hidden probe gives 0.50–0.57.
- We added a "type-default disagreement" baseline: one bit recording whether the model's answer differs from the entity type's default. It reaches 0.84–0.89 within FU in original-world regimes, beating every learned signal. Correctness within never-seen facts is therefore mostly prior-predictable luck. This quantifies the label issue the reviewer raised under M1/m2, and it is now a recommended control in the LLM protocol.

### M6. Statistics are thin; no hyperparameter or regime stress test.

**Fixed within compute limits.**

- The base regime now uses 5 worlds (worlds 3 and 4 are new). Every ± in the paper is now a 95% t-interval across worlds (the old values were SDs), and the paper warns that n = 3 intervals are wide.
- δ̂ now has bootstrap intervals over held-out items.
- Six additional regimes vary epochs (3/10/20/30), capacity (width 32 vs 256), label smoothing, and the world family.
- Probe ablation (T-F; Section 6.6): λ ∈ {0.01, 1, 100} × threshold {0.5, Youden on train} × π_d ∈ {0, 0.5}. In the base regime the shortcut is stable: natural AUROC 0.90–0.93, dissociated 0.24–0.28, FK 0.78–0.90. With λ ≤ 1, the threshold rule changes FK by ≤ 0.12 in every regime. At λ = 100 it can change FK by up to 0.24 (small regime). The paper says so and directs readers to the threshold-free AUROCs.
- The claims that depend on the memorization regime (E2, E4, MSP inversion) have all been stress-tested. MSP falls below 0.5 at α = 1 in 4 of 7 regimes. Good–Turing fails in the imperfect regimes.

---

## Minor issues

- **m1 (Table 2 ceiling row vs figure line).** Fixed. Section 4.1 states that the figure's dotted line is the detector-specific bound using the measured A(MSP; P₀, P_det) = 0.816, and that the detector-agnostic bound starts at 1. The confusing ceiling row has been removed from Table 2. Ceilings now appear only in Table 4, computed from δ̂.
- **m2 (class balance in Prop. 2).** Implemented: `theory.lda_rates_general` handles an arbitrary P(K=1) together with π₁ and π₀. It does **not** explain the gap: the class-prior-corrected form has MAE 0.181 vs 0.148 for the balanced form. Reported in Table 6.
- **m3 (δ̂ optimistic).** Fixed. The Youden threshold is now chosen on the training fold and applied to the held-out fold, with a 90% bootstrap interval. Base: 0.785 [0.74, 0.83], vs 0.829 with R0's estimator.
- **m4 (domain of P2).** Fixed. Section 4.2 now states that the monotonicity and floor statements hold on π ∈ [½, 1], and that for π < ½, FK can fall below Φ(−b/2σ).
- **m5 (Good–Turing guaranteed by memorization).** Confirmed and claim withdrawn. In the four imperfect regimes the error rate is 1.8–3.1× the singleton fraction, and only 23–38% of errors fall on never-seen facts (Section 6.2).
- **m6 (Cheang title).** Updated to the Findings of ACL 2026 title and pages (verified).
- **m7 (runtime).** Fixed. The paper and README now report CPU-minutes for R1 (training 43.2 CPU-min, 19.7 min wall on 4 cores; analysis about 5 CPU-min). They acknowledge the reviewer's 3.2 h/world measurement under concurrent multithreaded BLAS and recommend `OMP_NUM_THREADS=1`, with which one base world takes 216–269 s.
- **m8 (Discussion overreach on q).** Fixed. "Fails in principle" has been replaced by a conditional statement: q-only detection is limited by δ_T, which is invariant to recalibration and is not removed by calibration.
- **m9 (semantic entropy measurability).** Fixed. Remark 3 now states that semantic entropy is a randomized function of q. Proposition 1 applies to its expectation over sampling, and sampling noise cannot increase δ (data processing). Label invariance depends on the clustering. We do not test semantic entropy.

## What we could not do

- **No LLM pilot:** there is no GPU and model-weight hosts are blocked (see M1). The Section 7 protocol now includes the R1 controls (count baselines, exposure matching, within-stratum AUROC, a prior-agreement feature) and falsification criteria.
- **No world in which "knowing" differs from "having seen the fact".** MI's success therefore cannot be attributed to knowledge rather than exposure. This is stated as a limitation.
- **Few worlds per regime** (3) for the new regimes. Several within-FU comparisons (n ≈ 230 items per world) remain unresolved, and we do not interpret them.
