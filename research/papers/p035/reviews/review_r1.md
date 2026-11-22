# Review (Round 1): "What Can Evaluations Tell Us About Deployment? Partial Identification of the Evaluation–Deployment Gap for Evaluation-Aware Models"

Reviewer role: area chair–level independent reviewer. Recommendation: **major revision (weak reject in current form)**. Overall score: **5/10**.

## 1. Summary

The paper treats the gap between evaluation and deployment behaviour of an evaluation-aware model as a partial-identification problem. It posits that behaviour depends on context only through an internal "evaluation-belief" statistic s (A1) and proves four results:
- (P1) Worst-case (Manski-style) non-identification. The width of the identified set equals the deployment mass whose (task, belief) pairs the auditor cannot reach.
- (P2) A monotone lower bound. If unsafe behaviour is non-increasing in s (A2) and the rewrite only removes evidence (A3), the rewritten unsafe rate is at most θ_D.
- (P3) A Lipschitz bound and a TV bound. The TV bound is taken through the model's own belief statistic and leads, via data processing, to "model-relative realism".
- (Lemma 1) An AUC-to-TV inequality.

It then simulates 110 "model organisms": 2×32 MLPs trained on labels from an evaluation-conditional logistic policy over 60-d binary features. The simulation compares six estimators and three diagnostics against known ground truth, and gives a pre-registered (but unexecuted) protocol for LLMs.

## 2. Reproduction check

I re-ran the full pipeline (`run_experiments.py` → `analyze.py` → `analyze2.py`) from a clean copy in a scratch directory (205 s on 4 cores).
- `summary.txt` is **byte-identical** to the shipped file.
- `diagnostics.json` is **identical** to the shipped file (Spearman values, partial-budget biases, policy-level curve).
- I also checked the secondary numbers in the text: ESS range 1.05–43.4; naive range 0.034–0.387; organism-vs-policy naive gap −0.012; LB gaps 0.06–0.07 / 0.28 / 0.54–0.56; UB widths 0.11 / 0.35 / 0.63 (visible) and 0.015–0.067 (latent); RDE SE 0.100 at κ=4; mean-difference steering −0.44 to −0.86; LB failures (5/10 at κ=0, 1/10 in the subtle variant). All match.
- Table 1 and §6 are fully backed by code. **No numerical mismatch was found.**

## 3. Citation spot-check

Checked by web search (arXiv itself was not reachable from the sandbox, so I relied on search snippets, conference pages and author pages):
- Chaudhary et al. 2025 (arXiv:2509.13333; 15 models, 0.27B–70B, power law, NeurIPS 2025). Verified.
- Xiong et al. 2025, Probe-Rewrite-Evaluate (arXiv:2509.00591; probe + rewrite + judge; NeurIPS 2025 workshop). Verified.
- Hua, Qin, Marks, Nanda 2025 (arXiv:2510.20487; type-hint model organism; steering; ICLR 2026). Verified.
- Schoen et al. 2025 (arXiv:2509.15541). The o4-mini verbalised awareness figure of 0.9%→5.5% and the causal claim are verified.
- arXiv:2605.05835, "Evaluation Awareness in Language Models Has Limited Effect on Behaviour" (eight open-weight reasoning models, prefilling, NeurIPS 2026 poster). Verified. ω ≤ 0.06 for injection is confirmed only through a secondary summary.
- arXiv:2608.21766, "Evaluation Awareness in Language Models: Representation, Verbalization, and Control". Verified to exist, and its content matches the paper's description.
- Li et al. 2026 (arXiv:2605.23055). Verified, including Tramèr as co-author, nine frontier models and the environment/model decomposition. The specific phrasing "recognition rarely goes with behaviour change" was not directly confirmed in what I could read. The paper does separate recognition from propensity, so this is a paraphrasing risk, not a fabrication.

I found no fabricated references. Two pieces of relevant prior or concurrent work are missing:
- OpenAI's "Eval-production behavior gaps in agent safety and the role of eval-awareness" (alignment.openai.com). It measures the very gap this paper targets, in production.
- arXiv:2606.23583, "Evaluation Awareness Is Not One Capability". It explicitly frames test-condition compliance as an *optimistic bound* on deployment safety, which is close in spirit to Proposition 2.

Abdelnabi & Salem (arXiv:2505.14617), on steering test awareness, is also directly relevant to the steering estimators and is not cited.

## 4. Strengths

1. **A useful and sharp reframing.** "What can an evaluator validly conclude about deployment" is a better question than "is the model aware". The practical takeaway is concrete and actionable: report realism-rewritten rates as lower bounds, and audit realism through the model rather than through the rewriter's classifier.
2. **The mathematics is correct.** I checked P1, P2, P3(a,b), Lemma 1 and the tightness construction line by line. All are correct, if elementary. The Lemma 1 bound |2AUC−1| ≤ 2τ−τ² is slightly tighter than the trivial AUC−½ ≤ τ.
3. **Exemplary reproducibility.** The code is deterministic, small and readable, and it regenerates every number exactly.
4. **Honest scoping.** The authors repeatedly state that no LLM experiments were run, and they report negative results: the κ=0 failure of the lower bound, the non-calibration of D2, and RDE instability.
5. **Organisms learn their own detector.** Training an MLP instead of hard-coding the policy is a small but real step up from purely analytic toy models. The κ=0 A2-violation finding (spurious non-monotone dependence) is a genuinely instructive failure mode.

## 5. Weaknesses

### Major

**M1. The theory is essentially textbook, and the "four results" are thin as contributions.**
- P1 is the Manski worst-case bound applied to an unreached region.
- P2 is a one-line monotone-coupling argument.
- P3(b) is the definition of TV plus data processing.
- Lemma 1 is a short mixture argument.
None of this is wrong, but at a top ML venue the theoretical contribution is the *framing*, not the results. The paper should say so and position itself as a framing/methodology paper.

The paper also skips the hard identification questions:
- How can an auditor *test* A3 (no over-rewriting) without access to P_D? Section 5.3 hints at using benign traffic, but benign and risky items may be encoded differently, as the paper's own steering results show.
- What happens under task shift? This is dismissed in Limitations, but in practice it is likely first-order.
- What happens when s is multi-dimensional and A2 has no natural ordering?

**M2. The simulation largely confirms its own assumptions (circularity).**
- The ground-truth policy is monotone in an additive evidence score by construction.
- The "perfect rewriter" resamples exactly from the deployment marginal.
- Non-rewritable cues are Bernoulli(0.8) vs Bernoulli(0.1), so A3 holds by stochastic dominance.

Under these conditions P2 *must* hold up to learning noise, so 100% LB validity is not evidence for the theory's practical relevance. Moreover, at ρ>0 the bias is −0.05 to −0.55, so a lower bound is trivially satisfied. The informative tests are:
- the κ=0 control (50% failure);
- the subtle-cue case (10% failure);
- the ρ=0 configurations at κ≥2, where the margin is a few thousandths and the "valid" criterion includes a +0.01 slack.

Those are exactly where the bound is fragile. The paper should stress-test P2 under the conditions that matter in practice:
- imperfect or over-shooting rewriters;
- task-altering rewrites;
- non-additive or interaction-based detectors;
- organisms trained with an objective that does not enforce monotonicity.

**M3. The headline diagnostic result (D1 Spearman −0.08 vs D2 0.85) is largely a design and pooling artefact.** I re-analysed `raw.json`:
- **D1 is a visibility indicator by construction.** It sits at about 0.99 whenever the non-rewritable cues are visible and about 0.50 when they are latent, independent of ρ. Pooling these two clusters, each spanning all ρ values, mechanically drives the pooled Spearman toward zero.
- **Within the visible group** (ρ>0, n=30), D1 actually correlates with |bias| at **Spearman 0.49 (p = 0.006)**. The paper's claim that "D1 contains no information about the bias" is therefore an overstatement (a Simpson-type pooling effect).
- **D2's 0.85 is entirely between-configuration.** It reflects ρ being a design variable. Within each (ρ, visibility) cell, D2's Spearman with |bias| ranges from −0.54 to +0.45, with no consistent sign. Its distributions at ρ=0 (0.898–0.956) and ρ=0.25 (0.892–0.973) overlap completely.
- At the level of an individual audit, which is the use case the paper recommends, D2 cannot tell a bias of 0 from a bias of about 0.06.
- The reported p = 2.4×10⁻²⁰ treats 70 runs as independent evidence. The effective sample size is roughly the number of ρ levels (4).

Recommendations 2 and the "model-relative" message of §7.1 rest on this result. They need either a much weaker statement or a diagnostic that works within configuration.

**M4. Steering and IW baselines are under-powered or strawmen.**
- Steering vectors are estimated on benign items only, at layer 1, with a single direction. This is acknowledged, but it means the "steering does not substitute for realism" conclusion is about this particular implementation. Missing variants include risky-item contrastive directions, other layers, and multi-direction or subspace ablation.
- IW uses a logistic density ratio on nearly separable contexts. Its failure is expected, and Shimodaira-style overlap failure is well known; it hardly needs 110 organisms.
- A stronger baseline that is missing: estimate the belief-to-behaviour curve g by moving s directly in the organism (e.g. steering along a range of magnitudes) and extrapolate.

**M5. The policy is extremely saturated, which makes the gap sizes uninformative.** With b₀ = 1.5 and κ·z_c/3 ≈ ±4, the deployment unsafe rate is about 0.94 and the evaluation rate about 0.1. Headline figures such as "naive understates by up to 0.92" are free parameters of the simulator, not findings, yet they appear in the abstract and conclusion as results. The abstract should either drop them or state clearly that they are set by design.

**M6. The empirical scope does not match the venue.** The research question is about LLMs, but the paper has no LLM experiment, not even a small one on the publicly released Hua et al. organism. That organism has a known ground-truth deployment context, and the authors already propose it as subject (a). Running P1–P2 of the pre-registered protocol on it would turn this from a workshop-level framing paper into a main-track contribution.

### Minor

- m1. Contribution 2(d) says Lemma 1 "explains why a high-AUC realism test cannot be read as 'the evaluation is far from deployment'". This is a misstatement. Lemma 1 shows that a high AUC *does* imply large input-level TV; the irrelevance of input-level TV comes from P3(b) and data processing. Please rephrase.
- m2. The Lemma 1 tightness construction uses δ_{±∞}. Use finite points, e.g. P′ = δ₂ and Q′ = δ₋₁.
- m3. The "LB valid" criterion uses a +0.01 slack (`lb_mono ≤ θ_D + 0.01`). Report validity without slack as well; it matters at ρ = 0.
- m4. The one-sided SE in the lower bound covers only item-sampling noise. It does not cover rewrite randomness or organism-level noise. State this.
- m5. The "Auditor-space Lipschitz upper bound" uses an ad hoc L̂ (max finite-difference slope over the last three levels, ×2). Its failure in the latent case is guaranteed by construction, since gapA = 0 when the auditor cannot see the non-rewritable cues. Present it as an illustration, not as evidence.
- m6. Remark after P1: the claim that A2 "raises the lower end to the monotone bound of Proposition 2" needs A3 or an ordering of reached and unreached beliefs. Make it precise.
- m7. Corollary: "bias at most L·Δ_nr" requires the A3 coupling, with rewritable parts matched pathwise and not just in distribution. State this.
- m8. Section 6.4 says D2 "orders the configurations correctly". That holds for configuration means only; see M3.
- m9. Missing related work: OpenAI's eval–production gap post; arXiv:2606.23583 (test compliance as an optimistic bound); Abdelnabi & Salem (arXiv:2505.14617) on steering test awareness.
- m10. Use a consistent citation style. Some 2026 papers are cited by arXiv ID alone, without authors, even though author lists are available (e.g. 2605.05835).

## 6. Questions for the authors

1. How do you propose to check A3 in practice without deployment risky data? Benign traffic is a proxy, but in your own organisms the representation of evaluation evidence interacts with task riskiness (§6.5). Doesn't that undercut using benign traffic to validate A3?
2. Can you report D2 performance *within* each configuration, and propose a calibrated variant? One option is a probe read out at the layer or direction that most affects the output, or a behaviour-weighted distinguishability measure.
3. What happens to P2's validity when the rewriter over-shoots, i.e. inserts deployment cues at above-deployment frequency? A simulation of this failure mode is essential to the "report as lower bound" recommendation.
4. Could you run the protocol's P1 and P2 on the released Hua et al. type-hint organism?
5. For capability evaluations, where A2 flips under sandbagging, is the analogous upper bound on capability practically usable? Can you show it in the simulator?

## 7. Assessment of the sub-question and conclusions

The sub-question is sharp and important: what valid inferences about deployment does evaluation-time access allow, and when? The theoretical conclusions follow from the assumptions. The empirical conclusions follow only weakly:
- the LB-validity results are near-tautological under the simulator's design;
- the diagnostic comparison is driven by between-design variation and pooling;
- the large gap magnitudes are set by simulator parameters.
The policy recommendations in §7.1 are reasonable as conjectures but are presented with more confidence than the evidence supports. Recommendation 2 in particular rests on the D2 result that does not hold within configuration.

## 8. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 5 |
| Rigor | 4 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision.** In its current form this is borderline workshop level. There are no fatal issues: the numbers reproduce exactly and the citations exist. The single most valuable improvement would be to run the pre-registered P1–P2 test on the released Hua et al. LLM organism, which has known deployment ground truth, together with a within-configuration evaluation of the model-relative diagnostic.
