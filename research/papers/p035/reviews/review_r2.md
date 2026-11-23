# Review (Round 2): "What Can Evaluations Tell Us About Deployment? Partial Identification of the Evaluation–Deployment Gap for Evaluation-Aware Models" (Revision 1)

Reviewer role: independent area-chair-level reviewer. I did not write this paper. I read the round-1 review and the authors' response, but the judgement below is my own.

**Recommendation: major revision (borderline; workshop-ready, below the bar for a top-venue main track). Overall: 5.5/10.**

---

## 1. Summary

The paper treats "does the model behave the same in evaluation as in deployment?" as a partial-identification problem. Behaviour depends on context only through an internal evaluation-belief statistic s (A1). The paper then derives:
- Prop. 1: a Manski-type worst-case identified set, whose width is the deployment mass at unreached beliefs.
- Prop. 2 / 2′: a lower bound on θ_D from a realism rewrite, under monotonicity (A2) and evidence removal (A3), extended to beliefs ordered by a cone.
- Prop. 3: a Lipschitz bound and a total-variation (TV) bound measured through the model's belief ("model-relative realism"), plus an AUC→TV lemma.
- New in this round: Prop. 4 (A3 cannot be tested in general, but can be tested on benign traffic under task–context separability, A5) and Prop. 5 (a task-shift correction by TV of the task marginals).

The empirical part uses 408 CPU-trained "organisms":
- 110 round-0 MLPs;
- 280 new MLPs covering saturated and unsaturated regimes, an interaction detector, a deliberately non-monotone "tent" policy and κ=0 controls;
- 18 small transformers trained from scratch on token sequences.

The new MLP grid stress-tests six rewriters. It also evaluates a benign-traffic A3 check and runs a within-configuration analysis of realism diagnostics. That analysis withdraws the round-0 headline claim that the probe residual AUC tracks bias. There are still no LLM experiments. A pre-registered protocol for the Hua et al. organism is provided instead.

## 2. Reproduction check

I re-ran the revision pipeline from a clean copy of `code/` in a scratch directory:
- `run_v2.py`: 280 MLP organisms, 34 min wall-clock on a loaded 4-core machine;
- `run_token.py`: 18 transformers, 10.5 min;
- then `analyze_v2.py` and `analyze_token.py`.

Results:
- `raw_v2.json`: every float field is **identical** to the shipped file (max abs diff 0).
- `summary_v2.txt` and `analysis_v2.json`: **byte-identical**.
- `summary_token.txt`: **byte-identical**.
- Round 0 was verified exactly by the round-1 reviewer, and its re-analysis is reproduced inside `analyze_v2.py`.

I then checked the paper's text against these outputs.

Matches:
- Table 2, all entries.
- The §6.7 MAE table.
- §6.3(i)–(iv): validity 100/53/90/67%; SD of the rewrite error 0.010/0.024 vs mean item SE 0.004/0.005; largest ρ=0 excess 0.0346.
- Over-shoot, task-intensify and tent biases.
- The §6.5 confusion table, which I recomputed from `raw_v2.json`: violations 68/53/33/5, flagged 60/35/0/0, and 81 extra flags, all on over-shooting rewriters, 72 of them saturated.
- Transformer numbers.
- Figures 4 and 5 are consistent with the summaries.

Small imprecisions are listed in §6. **No unbacked or mismatched result.**

## 3. Reference spot-check (web search; arXiv pages were reached only through search snippets and listing pages)

I prioritised the references added or changed in this round.

| reference | status |
|---|---|
| Li, Zhang, Zhang, Jin, Abdelnabi, Andriushchenko (2026), "Decomposing and Measuring Evaluation Awareness", arXiv:2605.23055 | **Verified.** The authors' correction is right: the author list has no Tramèr, so the round-1 review was wrong on this point. The content matches: environment/model decomposition, eight trigger factors, recognition rarely associated with behaviour change and direction depending on evaluation type, stacking factors raising awareness (v2). NeurIPS 2026 listing. |
| arXiv:2606.23583, "Evaluation Awareness Is Not One Capability: Evidence from Open Language Models" | **Verified.** The paper describes it accurately: 37 open-weight models, seven families, three weakly coupled axes, "benchmark illusion". |
| arXiv:2608.21766, "Evaluation Awareness in Language Models: Representation, Verbalization, and Control" | **Verified.** The description ("representation, verbalisation and steerability agree only in part") matches the abstract. |
| Knecht et al. (2026), arXiv:2605.05835, NeurIPS 2026 poster | **Verified.** The authors are Knecht, Florin and Hagendorff, and the full list *is* retrievable by search. The bracketed note "full author list could not be retrieved" should be replaced by the names. |
| Abdelnabi & Salem (2025), arXiv:2505.14617, v1/v2 title "Linear Control of Test Awareness…", v3 "The Hawthorne Effect in Reasoning Models…", NeurIPS 2025 | **Verified**, including the title history. The paper's use of it (effects of test awareness on harmful-task execution vary in size and direction across models) matches the abstract. |
| OpenAI Alignment, "Sidestepping Evaluation Awareness and Anticipating Misalignment with Production Evaluations" (alignment.openai.com/prod-evals/) | **Verified.** Production-derived evals show substantially lower evaluation awareness, as described. |
| Hua, Qin, Marks, Nanda (2025), arXiv:2510.20487; release `timhua/wood_v2_sftr4_filt`; code `tim-hua-01/steering-eval-awareness-public` | **Verified** through the paper's HTML and the Alignment Forum post (Llama 3.3 Nemotron Super 49B, type hints, Wood Labs cue). |
| Nguyen, Hoang, Attubato, Hofstätter (2025), arXiv:2507.01786, ICML 2025 workshops | **Verified.** |
| Xiong et al. (2025), Probe-Rewrite-Evaluate, arXiv:2509.00591 | **Verified** (NeurIPS 2025 workshop). |

No fabricated or misdescribed references found. Relevant recent work that is missing:
- OpenAI, "Predicting LLM Safety Before Release by Simulating Deployment" (2026; arXiv:2607.07184). This is the paper-length version of the production-eval programme, and it directly targets the deployment estimand.
- "A Model Organism for Studying Unverbalized Evaluation Awareness" (arXiv:2611.09945). A second LLM organism that is relevant to the §7.3 protocol.
- "Evaluation Awareness or Prompt Content? Stress-Testing Activation Probes with Paired Conversations" (NeurIPS 2026). This is directly relevant to the paper's own finding that probe-readable evaluation information is largely irrelevant to behaviour.

## 4. Novelty

My search (partial identification, monotone bounds, realism rewriting and evaluation awareness) found no prior work that casts evaluation-awareness auditing as partial identification, or that gives conditions under which a realism-rewritten rate is a valid one-sided bound. The closest is 2606.23583, which informally calls test compliance an optimistic bound. The framing is therefore novel. The mathematics is not, and the authors now say so honestly. The novelty sits in the "what should an evaluator report, and which assumptions are testable" layer. That is useful, but it is a modest contribution for a main-track paper with no LLM evidence.

## 5. Proof check (line by line)

- **Prop. 1.** Correct. The convexity and linearity argument gives the full interval. The precise remark (lower end ≥ θ_R under A2+A3; upper end unchanged when unreached beliefs lie below the reached ones) is correct.
- **Prop. 2 and its nested version.** Correct. The nested claim needs a three-way coupling with equal tasks, which "common coupling" covers.
- **Prop. 2′.** Correct. Strassen's theorem for closed partial orders on ℝ^k is the right reference for the "consequence". One caveat: the implemented check compares *means* along one direction. That is weaker than first-order dominance along that direction, which is in turn weaker than cone dominance. The text should say "mean comparison along one dual-cone direction".
- **Prop. 3(a,b).** Correct: Jensen/A4, then the variational form of TV and data processing.
- **Lemma 1.** Correct. I re-derived it: the (M,M) term has zero mean sign by exchangeability, the remaining weight is 2τ(1−τ)+τ² = 1−(1−τ)², and the rearrangement gives τ ≥ 1−√(2−2AUC). The finite tightness construction (M = U[0,1], P′ = δ₂, Q′ = δ₋₁) is exact, and TV(P,Q) = τ because the singular parts are disjoint from each other and from M.
- **Corollary.** Correct as restated, with pathwise matching and A3-type ordering. It implicitly needs additive separability s = s_rew + s_nr, which the interaction detector violates through the E×H cross terms. That is worth one sentence.
- **Prop. 4(a): statement and proof do not match, and the construction needs repair.**
  1. The statement says θ_R > θ′_D "whenever g(t,·) is strictly decreasing and the unobserved mass is positive". The proof only shows that θ′_D < θ_D and that it "can" fall below θ_R. Since θ_D ≥ θ_R, the gap can be arbitrarily large, so a small unobserved mass need not push θ′_D below θ_R. As written, the statement is false. It should say "θ′_D < θ_D, and θ′_D < θ_R whenever the unobserved mass q satisfies q·(drop in g) > θ_D − θ_R", or simply "can".
  2. Setting s′ to "the largest belief value in the range of s" does not guarantee that A3 fails. If rewritten items also reach that maximum, an ordered coupling can still exist, and "largest" may not exist at all (a supremum that is not attained). Choose s′ strictly above sup s(R(x_E)) on the unobserved set. g is defined on all of ℝ, so A1 and A2 survive.
  These are fixable, but they are errors in a *new* proposition that is billed as answering the round-1 reviewer's main theoretical question.
- **Prop. 4(b): a missing assumption.** The proof draws "a common task t independently". That gives a task-preserving coupling of P_E and P_D only if the task marginals are equal, P_T^E = P_T^D. A5 does not assume this. Add the assumption, or combine with Prop. 5, which gives θ_D ≥ θ_R − TV(P_T^E, P_T^D). Also, "both laws are identified" requires σ itself to be observed. With a probe proxy, the identified object is the law of the proxy, and §5.3 implements only a t-test on means with a −3 threshold. Failing to reject is not evidence of dominance. The paper partly acknowledges this (caveat 2), but the gap between the proposition (FOSD of σ) and the implemented check (mean along one probe direction, at a strict rejection threshold) should be stated explicitly.
- **Prop. 5.** Correct, given conditional A1–A3. It needs support overlap, supp P_T^D ⊆ supp P_T^E, for E[b(R(x_E)) | t] to be defined. State this.

## 6. Statistics and experimental design

Strengths of the revision:
- validity is reported without slack;
- the within-configuration analysis uses stratified permutation tests;
- the authors state explicitly that the ρ>0 exact-rewriter results are non-evidence;
- the unsaturated regime is added, and the saturation-masks-violations finding is a genuinely useful point;
- the tent organism is included;
- the MRDE baseline the reviewer asked for is implemented, and its two-sided failure is reported honestly.

Remaining concerns:
1. **No uncertainty on validity rates or flag rates.** Cells have n = 10–15. For example, "53% valid" (unsaturated, ρ=0) is 8/15, with a Wilson 95% CI of about [0.30, 0.75]. "0 of 6" and "6 of 6" in the transformers are tiny samples. Report CIs for every proportion that enters the abstract.
2. **The A3-check accounting.** The 95/121 sensitivity pools regimes and depends on an arbitrary 0.02 violation threshold. The claim that the 81 flags without a large violation are "true A3 failures, not false alarms" is asserted, not measured. Ground-truth z is available in the simulator and should be used to verify it. It is also contradicted at least twice in `raw_v2.json`: overshoot_full, unsaturated, latent, ρ=0.5 and ρ=0.75 are flagged with bias −0.09 and −0.365. There, behaviour-relevant belief is clearly *above* deployment, so the flag is a false alarm along the probe axis.
3. **Task-intensification blindness is close to built in.** The check runs on benign items, whose policy is constant (0.02) whatever the task, and Prop. 4(b) excludes task changes by assumption. "0 of 33 flagged" illustrates the proposition; it does not test it.
4. **The transformer attribution to Prop. 2′ is speculative.** The ground-truth belief in the token organisms is a *scalar* (pair-count evidence plus scenario tokens). The paper attributes the 0/6 detection at ρ=0.25 to the "multi-dimensional caveat" without showing that the learned representation is multi-dimensional in the relevant sense. Plausible alternatives are a poor probe at layer 1 of the CLS stream, or an A5 violation. Either show a second direction that detects it, or soften the claim. Separately, the claim that order-dependent pair evidence is "invisible to a bag-of-tokens view" is too strong: pair counts are strongly correlated with unigram counts.
5. **Several "perfect rewrite" numbers come from two different draws.** The `full` estimator and the `perfect` stress-test column use different random draws: −0.006 vs −0.005 saturated, +0.002 vs −0.000 unsaturated, −0.022 vs −0.023 interaction. The paper mixes them in §6.2. Harmless, but say which is which.
6. **Small numerical imprecisions.**
   - §6.3(ii): "saturated over-shoot +0.006 to +0.011". The data range is +0.002 to +0.011.
   - §6.6: "D5p MAE 0.003–0.013" vs 0.003–0.012 in the MAE table.
   - The MLP over-shoot levels (0.5×, 3×) and the transformer levels (λ = 0.5, 1.0, capped at 0.9) differ, but share the names "mild/full". Clarify.
7. **Multiplicity.** Table 2 reports about 20 within-cell permutation p-values with no correction. The narrative correctly relies on R², not on p-values, but say so.
8. **Abstract framing.** "The lower bound fails in exactly the cases the theory predicts" is slightly too strong. It also fails in 30–47% of exact-rewrite runs at ρ=0 and in κ=0 controls. The paper explains these as approximate A2/A3 violations, but the theory did not "predict" them in any quantitative sense. The next bullet in the abstract qualifies this; merge the two.

## 7. Assessment of earlier issues

| round-1 issue | status |
|---|---|
| M1 theory thin, hard questions skipped | Partly resolved. Repositioned honestly. New Props. 2′/4/5 address the questions, but Prop. 4 has the statement/proof errors above and Prop. 4(b) omits equal task marginals. |
| M2 circular simulation | Largely resolved. Over-shoot, task alteration, tent and unsaturated regimes are genuine stress tests, and the trivially valid cases are labelled as non-evidence. |
| M3 pooled diagnostic artefact | Resolved, and handled well: the claim is withdrawn, the analysis is stratified, and the negative result is reported. |
| M4 weak baselines | Resolved within the toy (risky-target steering, layer-2 steering, mean matching, MRDE). |
| M5 saturation | Resolved. |
| M6 no LLM experiments | **Not resolved.** The authors give a credible explanation (49B model, no GPU, Hugging Face blocked). Still, there is no evidence on any language model, not even a small open model with a synthetic evaluation cue fine-tuned in. The research question is about LLMs. This remains the main obstacle to main-track acceptance. |
| minor m1–m10 | Resolved. The m10 author correction was right, and it overturns a statement in the round-1 review. |

## 8. Scores

| criterion | score |
|---|---|
| Novelty | 5 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5.5** |

**Decision: major revision.** The revision is careful and candid, and every number I checked is backed by the code outputs. It is now a solid workshop paper and a borderline main-track one. The remaining gap is evidential, not presentational.

## 9. Path to a higher score

1. Run *any* LLM instance of the protocol. A 1–8B open model fine-tuned (LoRA, which fits a modest GPU) with a known synthetic evaluation cue would give ground truth for P1/P2, i.e. the lower bound and the benign-traffic A3 check, including an over-shooting rewriter. This would move the paper to 6.5–7.
2. Fix Prop. 4(a): the statement/proof mismatch and the construction of s′. Add equal task marginals (or a TV correction) and the support condition to Props. 4(b) and 5. Describe the implemented check as a mean test along one direction.
3. Use the simulator's ground-truth belief to classify A3-check flags as true or false, and drop the "these are true A3 failures" claim unless that classification supports it. Add CIs to every proportion in the abstract.
4. Either test the multi-dimensional explanation of the transformer miss, or soften it.
5. Add the missing related work: OpenAI's deployment-simulation paper, the unverbalised-awareness organism, and the paired-conversation probe stress test. Fill in the Knecht et al. author list.
