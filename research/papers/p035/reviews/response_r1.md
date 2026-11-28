# Response to Review (Round 1)

We thank the reviewer for a careful report. The reviewer re-ran our pipeline and re-analysed `raw.json`, and that re-analysis found a real error in our main diagnostic claim. We reproduced it and now report it in the paper.

The revision does five things:
- It repositions the paper as a framing and methodology contribution.
- It adds two propositions, on testing A3 and on task shift, plus a multi-dimensional version of the lower bound.
- It runs 298 new organisms: 280 MLPs and 18 token-sequence transformers. These cover over-shooting and task-altering rewriters, an unsaturated regime, a non-additive detector, a deliberately A2-violating organism, and stronger white-box baselines.
- It replaces the pooled diagnostic analysis with a within-configuration analysis.
- It withdraws or weakens every claim that the new evidence does not support.

We could not do the reviewer's highest-leverage request, which was to run the Hua et al. LLM organism. The reason is explained under M6.

**Where the evidence lives.** All new numbers come from code we wrote and ran:
- Code: `code/organism_v2.py`, `code/run_v2.py`, `code/token_organism.py`, `code/run_token.py`, `code/analyze_v2.py`, `code/analyze_token.py`.
- Outputs: `results/raw_v2.json`, `results/raw_token.json`, `results/summary_v2.txt`, `results/analysis_v2.json`, `results/summary_token.txt`, and figures `fig4_rewriter_stress.png` and `fig5_diagnostics_within.png`.
- The previous paper is kept at `reviews/paper_r0.md`.

---

## Major issues

### M1. The theory is textbook-level and thin; the hard questions are left open

**We agree.** We have made the following changes.

**Repositioning.**
- The abstract and §1 now say that the formal results are elementary applications of partial-identification reasoning: P1 is a worst-case bound in Manski's style, P2 a monotone-coupling argument, and P3(b) follows from the definition of TV plus data processing.
- We state that the contribution is the framing and what it implies in practice. We claim no technical depth for the propositions.

**New results on the three open questions the reviewer raised (§4):**
- **Testing A3 without deployment data (new Prop. 4).**
  - (a) Without further structure, A3 cannot be tested. For any model satisfying A1–A3, there is an observationally equivalent model that violates A3 and has a lower θ_D. The construction raises the belief on deployment inputs the auditor never observes.
  - (b) Under a task–context separability assumption (A5), A3 holds whenever the rewritten-benign context statistic stochastically dominates the deployment-benign one. That condition is identified from benign traffic. It holds only for rewrites that change the context and leave the task unchanged.
  - This answers the reviewer's question 1. Benign traffic can validate A3 *only* under A5. Where task and context interact in the representation, it cannot. We now state this, and the simulation confirms that the check fails in the predicted ways (see M2).
- **Task shift (new Prop. 5).** θ_D ≥ θ_R − TV(P_T^E, P_T^D). We also remark that re-weighting over *tasks*, unlike re-weighting over contexts, can work, because task supports overlap.
- **Multi-dimensional belief (new Prop. 2′).** The lower bound holds unchanged when the belief is ordered by a closed convex cone. A single-projection check is necessary but not sufficient for A3 in that order. The transformer organisms show this failure in practice: mild over-shoot at ρ = 0.25 breaks the bound in 6 of 6 organisms, and the check flags none of them.

**What changed in the paper:** Abstract; §1 ("What kind of contribution this is"); §3 (new A5); §4 (Props. 2′, 4, 5).

### M2. The simulation is circular

**We agree that the old 100% validity was largely tautological.** We now say so explicitly in §6.3 and §8, and we no longer count ρ > 0 runs with exact rewriters as evidence. We added the stress tests the reviewer asked for, each with 10–15 organisms per cell. Fig. 4 and `summary_v2.txt` show the results.

| stress test | outcome |
|---|---|
| Over-shooting rewriters (marginals pushed 0.5× or 3× past deployment) | Bound broken. Unsaturated ρ = 0: bias +0.146 / +0.178, valid in 0% of runs. Interaction detector: +0.172 / +0.228. Transformers: +0.163 / +0.225. At ρ ≥ 0.5 the leftover non-rewritable evidence masks the over-shoot. |
| Task-intensifying rewriter | Bound broken at ρ = 0 (+0.040 saturated, +0.038 unsaturated; valid in 0% and 20% of runs). |
| Task-softening rewriter | Valid but loose. |
| Under-shooting (partial) rewriter | Valid but loose. |
| Non-additive detector (pairwise-interaction evidence, MLP) | Perfect rewrite still close (−0.022). Over-shoot fails as above. |
| Order-dependent detector (transformers on token sequences; evidence carried by adjacent token pairs) | Perfect rewrite bias −0.000 at ρ = 0. Over-shoot fails. |
| Organism not trained to be monotone (deliberate A2 violation, "tent" policy) | Perfect rewrite overestimates by +0.085 / +0.093 at ρ = 0.25 / 0.5. Bound valid in 0% of runs, and nothing the auditor observes reveals it. |

In addition, the plain MLP organisms are themselves not constrained to be monotone. Without the slack, they violate the bound at ρ = 0 in up to 47% of runs (unsaturated). This is caused by per-organism bias, up to +0.035, which is 2.5 to 4.5 times the item-level SE.

**The A3 check from Prop. 4(b), evaluated on 1,380 run–rewriter pairs:**
- It flags 95 of the 121 over-shoot violations larger than 0.02.
- It flags 0 of the 33 task-intensifying violations.
- It flags 0 tent violations.
- It never flags perfect, partial or task-softening rewriters.

**What changed in the paper:** §5.1–5.3 (new variants and rewriters); §6.3 (rewritten in full: "where it is trivially valid, and where it is not"); §6.5 (new); Fig. 4; §7.1 recommendations 1–2; §8.

### M3. The headline diagnostic result is a pooling and design artefact

**The reviewer is correct, and we withdraw the claim.** We reproduced every number in the reviewer's re-analysis (`analysis_v2.json`, keys `round0_*`):
- D1 vs |bias| within the visible, ρ > 0 group: Spearman 0.49, p = 0.006.
- D2 within-cell Spearman ranges from −0.54 to +0.45.
- The D2 ranges at ρ = 0 (0.898–0.956) and ρ = 0.25 (0.892–0.973) overlap.

The pooled p = 2.4 × 10⁻²⁰ is withdrawn. "D1 contains no information" is withdrawn.

**New analysis (§6.6, Table 2).** We used 15 seeds per cell. For each of 10 diagnostics, in both regimes, we report:
- the Spearman correlation over cell means;
- the mean within-cell Spearman, with a stratified permutation test (permutations within cells);
- the within-cell R² after demeaning.

**Findings:**
- Probe-based diagnostics rank configurations (cell Spearman 0.93–1.00) but explain at most 3% of within-configuration variance. D1 is uninformative in both senses.
- Behaviour-based white-box residuals (steering residual, MRDE residual, mean-matching residual) are weakly and inconsistently informative within a cell. For example, MRDE reaches R² = 0.31 in the saturated regime and about 0 in the unsaturated one.
- Only pre-activation mean matching works (R² ≥ 0.96). It relies on the toy's linear first layer, in which context enters additively, i.e. A5 holds by construction. We label it accordingly and do not claim it transfers to LLMs.

**What changed in the paper:**
- Abstract and Conclusion now state the negative result.
- §7.1 Recommendation 2 is replaced. The old version said "measure realism through the model's activations". The new version says "audit over-shoot with a benign-traffic belief check, audit task preservation separately, and do not treat probe residual AUC as a per-audit certificate".
- The protocol prediction P3 (formerly P2) is now pre-registered as a *within-subject, within-rewrite* test.
- m8 is resolved by this change.

### M4. The steering and importance-weighting baselines are weak

**We implemented the stronger baselines the reviewer asked for (§5.3, §6.7):**
- steering with a *risky-adjusted* target (the risky-minus-benign offset estimated in evaluation context);
- steering at layer 2;
- full-subspace residual mean matching, both after the ReLU and in pre-activation space;
- the reviewer's suggested **belief-curve extrapolation (MRDE)**. We vary belief through graded rewrites, measure it on the model's own probe axis, fit the curve in logit space, and extrapolate to the risky-adjusted deployment projection.

**Results (MAE table, §6.7):**
- Risky-target and layer-2 steering, and post-ReLU mean matching, do not beat simple layer-1 steering after rewriting. That approach removes about 25–50% of the residual bias.
- MRDE has the lowest error among the non-oracle estimators at moderate ρ (saturated ρ = 0.5: MAE 0.032 vs 0.229 for the full rewrite).
- However, MRDE errors are two-sided: +0.08 to +0.12 at ρ = 0 (unsaturated and interaction), and **+0.17 / +0.28 on the A2-violating organism**. It therefore gives up the one-sided guarantee. We recommend it only as a sensitivity analysis next to the lower bound.

**Importance weighting.** We agree that its failure is textbook. It is now presented as a reference point only (§6.1), with the Shimodaira citation, and it is not re-run in the revision grid. Prop. 5's remark points out where re-weighting *is* appropriate: over tasks.

### M5. The policy is saturated, so gap sizes are simulator parameters

**We agree.**
- The abstract and conclusion no longer contain gap magnitudes such as "up to 0.92". The abstract now says: "Gap magnitudes are set by simulator parameters and are not findings."
- §6.1 says the same thing explicitly.
- We added a full **unsaturated regime** (b₀ = −0.5, κ = 1.5): policy-level deployment rate about 0.70 and evaluation rate about 0.15; organism θ_D 0.65–0.69 and naive 0.12–0.17. Every qualitative result was re-checked there.
- Saturation turned out to matter in a way the reviewer anticipated. In the saturated regime, over-shoot violations of the bound are almost invisible (+0.006 to +0.011) because the policy sits at its ceiling. In the unsaturated regime they are large (+0.15 to +0.18). We report this in §6.3(ii) as a further reason not to rely on saturated simulations.

### M6. There are no LLM experiments

**We could not address this, and we say so plainly in §8.**
- We verified (by web search) that the released organism is Llama 3.3 Nemotron Super 49B fine-tuned for evaluation-conditional type hints (`timhua/wood_v2_sftr4_filt`).
- At 49B parameters it needs about 98 GB of memory in bf16. Our environment has 4 CPU cores, 16 GB of RAM and no GPU.
- The sandbox proxy also refuses connections to huggingface.co (we tested this), so no pretrained weights of any size could be downloaded.

We chose not to present a substitute as equivalent. Instead:
1. We added a second, structurally different organism family: transformers trained from scratch on token sequences, with order-dependent cues. This checks that the results are not specific to the MLP or to bag-of-features inputs. They are explicitly *not* language models.
2. We updated the pre-registered protocol (§7.3) with the release identifiers, over-shoot and task-altering rewrite arms, a pre-stated tolerance δ, the A3-check prediction (P2), and within-subject analysis (P3).

We accept that this limits the paper's empirical scope for a main-track venue.

---

## Minor issues

- **m1 (Contribution 2(d) misstated Lemma 1).** Fixed in §1 and in the "Reading of Lemma 1 (corrected)" paragraph after Lemma 1. A high AUC *does* certify a large input-level TV. That TV is irrelevant whenever the distinguishing features do not enter the model's belief (Prop. 3(b) and data processing).
- **m2 (Lemma 1 tightness used points at ±∞).** Replaced with finite points P′ = δ₂, Q′ = δ₋₁, M = U[0,1] and f = identity, which gives equality exactly.
- **m3 (+0.01 slack).** Table 1 now reports validity without slack, with the slacked value in brackets: κ = 2, 4, 8 at ρ = 0 is 0.7 / 0.8 / 0.8 without slack, and the subtle variant 0.7. Revision results are reported without slack throughout.
- **m4 (the SE covers only item-sampling noise).** *Partly a rebuttal.*
  - Each item receives an independent rewrite draw, so rewrite randomness is already part of the item-level variance (§5.3).
  - Organism-level variation is not sampling noise relative to the estimand, because θ_D is defined per organism.
  - The reviewer's underlying concern is still right. The dominant error at ρ = 0 is a per-organism *bias* from approximate A2/A3 violations: its SD across organisms is 0.010–0.024, against an item-level SE of about 0.004–0.005. The SE does not cover it.
  - We now say this in §6.3(i) and recommend reporting θ_R − δ with a stated tolerance.
- **m5 (the auditor-space Lipschitz bound is ad hoc).** §6.4 now says that its failure in the latent case is guaranteed by construction (the auditor-score gap is zero), and presents it only as an illustration of Prop. 3(ii).
- **m6 (remark after Prop. 1).** Made precise: the lower end rises to θ_R when A2 holds and some R ∈ ℛ satisfies A3; the upper end is unchanged when unreached beliefs lie below all reached beliefs.
- **m7 (the Corollary needs precise coupling conditions).** Restated with a pathwise match of the rewritable parts and an A3-type ordering of the non-rewritable parts. We note that equality in distribution is not sufficient.
- **m8 ("orders the configurations correctly").** Rewritten as a statement about configuration means only; see M3.
- **m9 (missing related work).** Added after verification by web search:
  - Abdelnabi & Salem, arXiv:2505.14617 (NeurIPS 2025). We now also use it as real-world evidence that A2's direction can fail (§3, §7.1).
  - arXiv:2606.23583, "Evaluation Awareness Is Not One Capability" ("benchmark illusion").
  - OpenAI's alignment post on production evaluations. We could **not** find a post with the exact title the reviewer gave. We cite the verified post on the same topic, "Sidestepping Evaluation Awareness and Anticipating Misalignment with Production Evaluations" (alignment.openai.com/prod-evals/), and flag the substitution in the reference list.
- **m10 (citation style; the Li et al. paraphrase).**
  - **We found and corrected an author error.** Search results for arXiv:2605.23055 list Changling Li, Terry Jingchen Zhang, Jie Zhang, Zhijing Jin, Sahar Abdelnabi and Maksym Andriushchenko, and do not include F. Tramèr. Both the previous version and the review had included him; he is removed.
  - The paraphrase "recognition rarely goes with behaviour change" matches the abstract ("recognition rarely associates with behavioral change, and when it does, the direction depends on the type of evaluation perceived"). We now include the second clause.
  - For arXiv:2605.05835 we give the first author, Knecht, as cited by arXiv:2610.22333.
  - For arXiv:2608.21766 and arXiv:2606.23583, full author lists could not be retrieved, because arxiv.org is unreachable from our sandbox and search snippets did not include them. The reference list states this rather than guessing.

## Other corrections we found ourselves

- **A2 and sandbagging (reviewer's question 5).** The previous version said that capability sandbagging "flips" A2 and turns Prop. 2 into an upper bound. That was wrong. With target "dangerous capability exhibited", sandbagging satisfies A2, so the rewritten rate is a *lower* bound on deployment capability. The bound flips only for behaviours that evaluation belief *increases*. Corrected in §3 and §8.
- **The MRDE and tent results** show that the simulation is no longer built only to confirm the theory. Several of our own estimators fail visibly.

## Answers to the reviewer's questions

1. **Checking A3 without deployment data:** Prop. 4. It is possible only under A5 and only for task-preserving rewrites. Empirical performance is in §6.5.
2. **Within-configuration D2, and a calibrated variant:** Table 2. D2 is not calibrated within a configuration. Behaviour-weighted variants (D3, D4, D5) help only inconsistently. The only calibrated one exploits the toy's additive layer.
3. **Over-shooting rewriters:** §6.3(ii) and Fig. 4. They break the bound when ρ is small, and the benign-traffic check catches most of these cases.
4. **The Hua et al. organism:** not feasible here (M6). The protocol is updated.
5. **Capability under sandbagging:** see the correction above. Sandbagging gives the same lower-bound direction, not an upper bound.
