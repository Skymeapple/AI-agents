# Review, Round 2: "Rate, Race, or Fate: Verifier Discrimination, Exploit Prevalence, and the Limits of RL Beyond Math and Code"

**Research question:** 强化学习能否让模型在数学、代码以外的领域同样快速进步？ (Can RL make models progress just as fast in domains other than math and code?)

**Reviewer role:** independent area-chair-level reviewer, round 2.

---

## 1. Summary

The paper asks which properties of a non-rule verifier (an LLM judge or reward model) limit how fast and how durably RL improves a policy. It has four parts.

1. **Theory for the idealised objective (the tilt / replicator flow).**
   - A CGF identity for gold log-odds (Thm 1).
   - A rate law dA/dt = A(1−A)·J_t (Cor 1).
   - A tail ("fate") law (Cor 2).
   - A non-identifiability construction (Prop 2).
   - A three-mode closed form for peak accuracy and error floor (Cor 3–4).
2. **Toy softmax-PG experiments** on one prompt with at most 1,024 strings. Practical PG "races" rather than tilts. A takeover boundary J*(h0) is computed from the mean-field dynamics (E9), and robustness is checked across encodings and optimisers (E10).
3. **A matched-budget audit comparison** over 120 toy configurations (E8).
4. **A real-verifier anchor (E11).** J_pair = 2·P(prefer correct) − 1 for 33 published JudgeBench judges, by domain. The median is 0.12 on humanities/social/biology knowledge and 0.49 on math.

The conclusion is stated conditionally: judge rewards outside math sit at low J, which in the toy means slower SGD learning and a roughly 6× smaller tolerable exploit prevalence.

The revision is honest and substantial. Several round-1 headline claims are withdrawn: the "on-policy beats offline" audit claim and the power law. Every new number reproduces exactly (Section 3). However, the central link to the research question is still a toy boundary placed under a judge statistic, and that link has two new weaknesses that the revision introduces:
- The boundary is not a function of (J, h0) alone.
- E11 measures reference-free pairwise judging, not the reference-guided verifiers that non-math RLVR actually uses.

---

## 2. Status of the round-1 major issues

**M1. Missing related work and overstated novelty: largely resolved.**
- All four requested sources are now cited and positioned accurately: Egashira et al.; Moya et al.; Sparse Reward Auditing; Kwa et al. I checked each by web search.
- Thm 1 and Cor 2 are now described as elementary and textbook. The contributions are restated as quantitative.

Residual problems:
- (a) "Wrong the Same Way: Cross-Prompt Sharing Both Suppresses and Amplifies Reward Hacking in RLVR" (NeurIPS 2026; arXiv:2611.13218) is uncited.
  - It shows, in an exact-enumeration toy, that parameter sharing either suppresses or amplifies exploitation depending on a threshold.
  - It also shows that amplification requires the base representation to already encode the verifier's acceptance pattern.
  - This is essentially the mechanism behind E4/E10 (tabular never collapses, shared parameterisations do, near-miss exploits are worst).
  - The paper already cites another NeurIPS 2026 paper, so this omission is not excusable as concurrent.
- (b) Concurrent work also uses the "race" framing: "Honesty Is a Race" (arXiv:2611.15779, November 2026). It is recent enough to forgive, but it should be acknowledged in a camera-ready, because "practical PG races" is listed as a contribution.
- (c) "When Group Normalization Undermines Sparse Reward Auditing in RLVR" (NeurIPS 2026) is relevant to the std-normalised advantages used throughout.

With these, the remaining novelty is mostly the specific toy boundary and the E11 measurement.

**M2. On-policy vs offline audit: resolved in substance, with a new overstatement.**
- E8 now compares audits at matched budgets, reports CIs, analyses decision times, and restricts to undecided runs. The "on-policy beats offline" claim is withdrawn. All of this reproduces exactly.
- New overstatement: the paper now says "the informative distinction is *aggregate vs tail-aware*". Its own numbers do not support that emphasis. I ran a paired bootstrap on `exp8.json` (4,000 resamples):

| comparison | ΔAUC | 95% CI |
|---|---|---|
| tail-aware − aggregate, matched budget | +0.037 | [0.001, 0.079] |
| tail-aware − look-ahead | +0.023 | [−0.022, 0.073] |
| aggregate at 2,816 − aggregate at 256 | +0.111 | [0.044, 0.183] |

- The tail-vs-aggregate gain is barely distinguishable from zero, and tail-aware does not beat look-ahead. Most of the round-1 gap (0.77 vs 0.98) was **label budget**, not tail-awareness. The near-perfect predictors are those that know the exploit set.
- The tail-aware statistic works here partly by construction: it flags wrong responses accepted above the mean correct acceptance, and the exploit is the only wrong mode with η > α. In E10's heterogeneous-η setting, or for exploits with η < α that still win the race (E4, β = 0.4), there is no evidence that it would work.
- The "MARGIN" oracle in `exp8_offline_audits.py` still uses the withdrawn power law 12.6·h0^0.8. The paper describes it as using J*(h0). Because the statistic is monotone this is harmless, but it is inconsistent.

**M3. Evidence far from the research question: partially addressed, not resolved.**
- E11 is a real addition, and the cross-domain gap is real. An item-level bootstrap of the median-judge difference (math − humanities) gives 0.36, 95% CI [0.23, 0.49].
- Three new problems undercut E11 as an anchor for "non-math RL rewards":
  1. **Construct validity.** JudgeBench judges are *reference-free* and mostly *pairwise*. Non-math RLVR, including the paper's own protocol citation Su et al. (2025), gives the generative verifier the reference answer. MMLU-Pro knowledge items are multiple choice, so with a reference they are rule-checkable. E11 therefore measures how hard it is to judge hard, contested items *without a reference*, not the J of rewards actually used outside math. The code row (median J = 0.02, for a domain with executable tests) shows directly that the statistic tracks the judging protocol and item selection, not the domain's verifiability. The paper notes this but still draws the domain conclusion.
  2. **Pairwise J is not the pointwise J of an RL reward.** The identity J_pair = α − β holds for a class-conditional *pointwise* binary verifier scored independently on both responses. Most of the 33 judges are pairwise comparators with order averaging; for them J_pair is not the α − β of the same model used as a pointwise grader.
  3. **The statistics overstate certainty.** The sign test (33/33, p = 2×10⁻¹⁰) treats judges as independent replications. All judges are scored on the *same* 67 math and 99 humanities items, and many share base models. Only 27 judges have complete item sets in both domains. For those 27:
     - "every judge better on math" holds in only 33% of item resamples;
     - only 20 of the 27 are individually significant at item level.

     The p-value should be replaced by an item-level (or crossed item×judge) analysis.
- The toy still uses one prompt, a hand-built exploit, fixed η = 0.95, and SGD without clipping. No J or exploit tail is measured along any RL trajectory. The conclusion remains largely conditional on the model.

**M4. Power-law boundary: the power law is withdrawn, but the replacement has a new problem.**
- The power law is gone. The mean-field boundary is computed and matches the finite-sample bisection points (MAE 0.0067 / 0.0080, reproduced exactly). Curvature and saturation are reported honestly. P2 is softened.
- "Derived" overstates what was done. The boundary comes from numerical integration of the population update plus bisection, and the authors admit no closed form exists. "Computed from the mean-field dynamics" is accurate.
- **New, substantive problem: the boundary is not a function of (J, h0).** Every boundary experiment fixes β = 0.15 and η = 0.95, so J = α − β is confounded with the exploit's acceptance advantage η − α = 0.8 − J. I ran the authors' own mean-field code (`exp9_meanfield_boundary.mf_final_exploit`, factorised, h0 = 0.009) varying β:

| β | J* | α* | η − α* |
|---|---|---|---|
| 0.05 | 0.378 | 0.428 | 0.52 |
| 0.15 | 0.333 | 0.483 | 0.47 |
| 0.30 | 0.268 | 0.568 | 0.38 |
| 0.40 | 0.225 | 0.625 | 0.33 |

  - J* changes by a factor of about 1.7 at fixed h0. This is consistent with the paper's own E4 table: (β = 0.4, J = 0.2) gives 5/8 collapse, against 16/16 at (β = 0.15, J = 0.2).
  - So the claimed "boundary stated in two measurable quantities, J and h0" is false. The "≈6× smaller tolerable prevalence" placement of real judges (J = 0.12 vs 0.49) depends on the arbitrary choice β = 0.15, η = 0.95. Neither β nor η is measured for any real judge.

---

## 3. Reproduction

- I copied `code/` to a scratch directory and reran E1, E1b, E2b, E8, E9, E10, E11 and E12. For E11 I cloned the public JudgeBench repository at the stated commit, e2c52c28.
- Every regenerated JSON (`exp1`, `exp1b`, `exp2b`, `exp8`, `exp9`, `exp10`, `exp11`, `exp12`) is **bit-identical** to the shipped results.
- I spot-checked the paper's numbers against these outputs and all matched:
  - AUC table and CIs;
  - decision-time fractions (0 / 44% / 93%);
  - undecided-at-50 AUCs;
  - boundary MAE and local slopes 1.10 → 0.27 / 1.14 → 0.30;
  - J*/h0 ≈ 43–46;
  - tolerable h0 of 0.27% vs 1.56%;
  - E10 tables (encodings, optimisers, lr/G, J = 0.75);
  - t90 slopes −0.97 / −0.38;
  - E12 metastability numbers;
  - JudgeBench medians, ranges and sign-test p.
- E2, E2c and E4–E7 were reproduced in round 1 and were not rerun here.
- I found no mismatch between numbers and code.

## 4. Proofs and derivations (checked line by line)

- **Thm 1, Cor 1, Cor 2:** correct and elementary.
- **Cor 3:** I re-derived t* and A*: R = w0(α−β)/(h0(η−α)), e^{(η−β)t*} = R, and 1/A* − 1 = (w0/c0)R^{−κ}(η−β)/(η−α). Correct. The rise condition is correct.
- **Cor 4:** 2^{1/κ} = 2.33 and 6.9. Correct.
- **Prop 2 (corrected):** the construction is valid, and f on the residual wrong mass lies in [0, FPR). Small gap: "no atom larger than FPR(1−A0)" does not guarantee a set of mass strictly below FPR(1−A0) (for example, atoms all of exactly that mass). The construction also works with δ′ = FPR(1−A0) (residual f = 0), so the open interval should be closed, or the condition restated.
- **E11 identity:** 2P − 1 = α − β with ties counted as ½. Correct for independent pointwise verdicts (see M3.2 on its applicability).

## 5. Other weaknesses

1. **Adam comparison.** Adam uses lr = 0.05 against SGD at lr = 0.5, with no lr sweep for Adam. The "Adam raises collapse risk" and "weaker J-dependence" findings are confounded with step size. The weaker J-dependence follows almost by construction from per-coordinate normalisation. The reconciliation with Lv et al. is speculative.
2. **t90 at J = 0.1** rests on the 4 surviving seeds out of 16. This is acknowledged, but the slope CI ignores the survivor selection.
3. **E12.** "This run converges" (tabular, λ = 0.5) is not quite right: exploit mass is 0.064 against the tilt's 0.111. Accuracy matches, but the state has not converged. The 80k-step extrapolation assumes log-linear growth.
4. **Practical implication.** "Audit for tails, not means (AUC 0.91)" overstates a +0.04 gain over the aggregate audit at matched budget (see M2).
5. **References.** Moya et al.'s author list could not be confirmed in search snippets, although the title, arXiv id and content match. The "Checking the Checkers" survey has identifiable authors (Gupta & Joshi; accepted at AISSA 2026), which the paper says it could not retrieve.

## 6. Citation spot-check (web search)

All of the following exist and say what the paper claims:
- Egashira et al. 2026 (arXiv:2605.02909; authors and finding confirmed).
- Moya et al. (arXiv:2609.35677; title and content confirmed: non-identifiability, audit-based correction).
- Sparse Reward Auditing (NeurIPS 2026 page 164475; ~1% trusted labels confirmed).
- "Checking the Checkers" (arXiv:2610.17824; the structure-over-rate conclusion confirmed).
- Rad et al. 2026 (ICML 2026, PMLR 306; Youden-J drift confirmed).
- Kwa, Thomas & Garriga-Alonso 2024 (NeurIPS 2024; confirmed). Note that Kwa et al. also report that open reward models look *light-tailed*, which tempers the "fate" framing for real reward models.
- Lv et al. 2025 (72.02% vs 75.85% confirmed).
- Zhao et al. 2025 ("up to 80%" FPR confirmed).
- JudgeBench (repository and commit confirmed by clone).

No fabricated or misattributed citations were found.

## 7. Questions for the authors

1. Can you recompute J*(h0) over a (β, η) grid and state the boundary in the variables that actually determine it, for example J and η − α? Then, how would one place real judges without measuring η?
2. Can you rerun E11 with reference-guided pointwise grading (the judges given the gold answer)? At minimum, can you restrict to the scalar reward models and report pointwise α − β at a threshold?
3. Can you replace the judge-level sign test with an item-level or crossed random-effects analysis?
4. Does the tail-aware audit retain its edge when the exploit's η < α (as in E4 with β = 0.4) or when η is heterogeneous?
5. How do your sharing results relate to "Wrong the Same Way" (NeurIPS 2026)?

## 8. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5 (weak reject)** |

## 9. Decision

**Major revision (weak reject at a top venue).**

The revision is unusually honest and fully reproducible, and the theory is correct. But the answer to the research question still rests on two things:
- a toy boundary that, contrary to the paper's framing, is not determined by (J, h0);
- a judge measurement that captures reference-free pairwise judging rather than the rewards non-math RLVR actually uses.

The remaining novelty is incremental relative to Egashira et al., Moya et al., Rad et al. and "Wrong the Same Way".

The most valuable change would be to:
- re-state the boundary in its true variables;
- measure reference-guided, pointwise judge J and an exploit-acceptance η on a real non-math RL setting;
- ideally, run even one small LLM GRPO experiment testing P2.
