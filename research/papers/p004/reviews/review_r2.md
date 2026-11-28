# Review, Round 2: p004 "Exchange Rates for Synthetic Data: Coverage, Learnability and Verification Limits on Stretching the Human Data Stock"

Reviewer role: area chair / senior reviewer (independent). Target venue: ICLR 2027 main track.

## Summary

The paper asks whether synthetic data can take over once high-quality human text runs out, and frames the question as an exchange rate: the data multiplier M, the number of human tokens a synthetic pipeline is worth. The revision has four parts.

1. **Theory.** The standard results are now labelled as standard: garbling/DPI (Fact 1), a chain-rule accounting identity for verifier-filtered generation (Lemma 3), the strong law (Remark 1), and a randomly-reinforced-urn mapping (Prop. 5). Two pieces are claimed as new. The first is Prop. 2b, threshold-storage asymptotics showing that the memoryless 1/s ceiling and its distribution invariance are fragile, together with Prop. 2c (the coverage ceiling). The second is Prop. 4: a verifier of L bits adds at most L bits about the target.
2. **E5.** A 0.67M-parameter transformer is trained on 2,000 planted facts (78 runs). It measures paraphrase and repetition multipliers, the coverage ceiling, and a fitted per-exposure retention s.
3. **E6 and E2.** A logistic-regression self-training loop is run with (a) a verifier grid that decouples information from b/a and (b) verifier designs of explicit cost (a knowledge lookup table and a quantized skill program).
4. **E4.** Horizon arithmetic converts external rephrasing speed-ups and the E5 multipliers into years of delay (about 1.3 to 3.3 years).

The revision is much more candid than round 1. Two of the authors' own earlier claims are withdrawn, and the paper labels clearly what is standard. All numbers reproduce. My concerns now are mainly about what the evidence supports. The new experiments carry the knowledge-side and verifier-cost conclusions, and I found that both have confounds large enough to change those conclusions. The theoretical contribution is also too thin for an ICLR main-track paper.

## Reproduction

I copied `code/` and `results/` to a scratch directory and reran everything except E2's 144-run sweep. That sweep was bit-identical last round, and its re-analysis scripts are bit-identical now.

- **E1, E3, E7, E2 analysis.** `e1_knowledge.json`, `e3_urn.json`, `e7_threshold.json`, `e2_table.txt` and `e2_predictors.txt` are byte-identical to the shipped files.
- **E5.** I reran all 78 tiny-transformer runs (880 s on 4 cores). `e5_tinylm.json` is byte-identical, as are `e5_analysis.json` and `e4_horizon.json`, which are derived from it.
- **E6.** I reran all 170 runs and compared them with the shipped `e6_verifier_design.json`; the result is in the addendum at the end. The table numbers in the paper (Part A Spearman values 0.783/0.811/0.853/0.863/0.821/0.972/0.993/0.958; partial correlations 0.993/0.936/0.948/0.497; and all Part B rows) match `e6_summary.txt` exactly.
- **Paper text against outputs.** I checked every E5 table entry (accuracy, M, CI, coverage, memoryless prediction), the s fits (0.029, 0.075, 0.124, 0.205; 0.0025 to 0.28), recall by occurrence (0.25, 0.49, 0.74), and the E7 values (5.70; 5.87 to 8.71; 21.0; 5.73; 32.4). All match.

I found no mismatch between the reported numbers and the code outputs.

## Citation check (web search)

I verified the following, and they say what the paper claims:

- May & Flournoy (2009), Ann. Statist. 37(2):1058–1078, doi 10.1214/08-AOS596.
- Muliere, Paganoni & Secchi (2006), JSPI 136:1853–1874.
- Kang et al. (2025), EMNLP Main, arXiv:2510.01631. The 5–10× figure is real, but it applies to a one-third rephrased mix at larger budgets, and pure rephrased data does not beat web text.
- Yi et al. (2025), arXiv:2510.16657. The "knowledge center" convergence claim is correct; the paper is an ICLR 2026 poster.
- Gan & Liu (2025), ICLR 2025, arXiv:2410.01720 (reverse bottleneck, GGMI).
- Maini et al. (2024), ACL 2024, 14044–14072. The ~3× speed-up is on C4.
- Villalobos et al. (2024). 300T tokens, 90% CI 100T–1000T, and 2026–2032 as an 80% interval.

None are fabricated. One citation is *used* incorrectly: Villalobos et al. describe their 300T stock as "quality- **and repetition**-adjusted". They state that multi-epoch training is part of why the estimate rose. E4's row "4 epochs of repetition ≈ fresh data (Muennighoff et al.), M=4 → 2.0 yr" therefore double-counts a factor that the stock estimate already includes. This is a substantive error in the horizon table (see W6), not a fabricated source.

## Status of round-1 major issues

**R1-M1 (theory is standard results in new notation).** *Acknowledged and relabelled; substantively unresolved.* The labelling is now honest. Lemma 3's first step is correctly an equality, and the Jensen step is stated. What remains as "new" is the following.

- *Prop. 2b.* This is a two-line compound-Poisson asymptotic. I checked it. For (a), plain Pr[stored] ~ (sλ')^m/m! and amplified Pr[stored] = λP_r + O(λ²), which gives M(λ) ~ (m!P_r)^{1/m} s^{-1} λ^{1/m-1}. Numerically, at m=2, s=0.1, r=7, λ=0.1 the leading term is 19.3, against an exact value of 21.0. For (b), matching exponential rates is correct, and the polynomial factors vanish in the ratio. The result is correct but elementary. Its premise (threshold storage) is, by the paper's own E5 fits, at most weakly present in a real learner.
- *Prop. 2c* is a tautology for planted random facts.
- *Prop. 4* is one line of data processing plus H(d) ≤ L. It is correct. Note, however, that I(θ;·) ≤ H(θ) always holds. For a skill target with a short description, Prop. 4 therefore never binds. Its only content is that an incompressible target requires a verifier at least as large as the target's entropy. This is the Kolmogorov/entropy truism behind "the verifier must contain the facts". It is a useful clarification, but not a contribution that carries a paper.

Net effect: the paper has no theorem of real depth. The authors concede this, but conceding it does not resolve it.

**R1-M2 (urn literature not cited).** *Resolved.* Both papers are cited and correctly characterised, and Prop. 5 is presented as an application. The mean-field rate W/T ~ t^{-(1-b/a)} matches the known n^{m2/m1} growth of the inferior colour.

**R1-M3 (E2 does not test the knowledge/skill asymmetry).** *Addressed by a new experiment, but that experiment is confounded (see W1).* E6 Part B is the right kind of experiment. However, I found that the knowledge-side numbers are driven substantially by a learner artifact, not by verifier information. The comparison also does not control for headroom.

**R1-M4 (gain vs information confounded).** *Largely resolved.* The decorrelated 3×4 grid is the right design, and the withdrawal of "gain tracks information" is correct and to the authors' credit. The remaining caveats are in W3.

**R1-M5 (horizon headline is ln10/ln2 on an unmeasured M).** *Partially resolved.* The revision anchors on WRAP and Kang et al. and labels provenance. But the anchors are compute or time speed-ups, not human-data multipliers. The Muennighoff row double-counts. The E5 multipliers are not meaningful as anchors (W2, W6). Recent work that measures effective-token multipliers directly is not used (RePro, arXiv:2510.10681; BeyondWeb, arXiv:2508.10975; FinePhrase, 2604.13977; SynPro, 2605.17849, which reports multipliers over repetition).

**R1-M6 (toy learners only; no baselines).** *Partially resolved.* The tiny transformer is a genuine improvement for the knowledge side. The verification side is still convex logistic regression, with no neural verified-self-training experiment and no STaR/ReST or Feng et al. baseline. The authors acknowledge this as the main gap.

## Strengths

1. **Exceptional candour and reproducibility.** Withdrawn claims are stated explicitly, standard results are labelled, and every number reproduces bit-for-bit from shipped code. This is rare and commendable.
2. **The E6 Part A design is clean.** Crossing a ∈ {0.25, 0.5, 1} with b/a ∈ {0, …, 0.8} separates information from purity. The finding that gain at fixed b/a barely moves with a 4× change in a (0.359 → 0.389) while information grows about 8× is a crisp, useful observation for people who design verifiers.
3. **Prop. 2b plus E7 correctly show that distribution invariance is an artifact of the memoryless assumption.** This is a fair corrective to naive 1/s reasoning.
4. **The E5 paraphrase-vs-repetition comparison is a sensible control,** and the result agrees with the Allen-Zhu & Li diversity findings.
5. **The skill learner converges to the quantized verifier's own agreement with the truth** (0.834 vs 0.836; 0.901 vs 0.921). This is a nice concrete instance of Yi et al.'s "knowledge center".

## Weaknesses

**W1 (major; affects C2/C4 and the knowledge/skill asymmetry). E6 Part B's knowledge numbers are largely a regularisation-dilution artifact.** The learner (`e2_verified_selftraining.fit`) minimises a *sample-averaged* cross-entropy plus a *fixed* L2 penalty (LAM=1e-3). With one-hot items, adding thousands of accepted synthetic examples on covered items shrinks the effective weight of the 200 human examples. The weights of human-seen but *uncovered* items are then pulled toward zero, and the learner forgets facts it already knew.

This explains the anomaly the paper waves away: at κ=0.25, a verifier with precision 1.000 and b=0 *lowers* accuracy (−0.016, s.d. 0.019, 5 seeds; about 1.9 SE below zero, not "within seed noise").

I verified this directly (script in the reviewer scratchpad; seeds 0–2, κ=0.25):

- Accuracy on human-seen, uncovered facts falls from 1.00 to 0.89, 0.83 and 0.87 in the paper's learner.
- When λ is rescaled so that the human data keeps its effective regularisation (λ·200/n_total), that accuracy stays at 1.00. Final accuracy becomes 0.835, 0.824 and 0.837, against 0.773, 0.739 and 0.770.
- The gain at 300 bits therefore rises from about −0.01 to about +0.06. That reaches the union-coverage bound of about 0.83.

So the claim "the lookup verifier is a poor channel even for facts it holds, because uniform proposals are rarely correct at first" is wrong in this setup. With a correct learner, the lookup verifier transmits about what it stores. The knowledge/skill asymmetry at equal bits is probably still present, because the 600-bit bound is 0.89 (gain ≤ 0.12) against +0.29 for skill. But it is smaller than reported, and the headline comparison ("+0.05 vs +0.29–0.35") is not trustworthy.

The same artifact likely contributes to E2's "unverified self-training hurts knowledge (0.892 → 0.750)" and to E6 Part A's negative knowledge gains at b/a=0.8, both of which the paper attributes to wrong labels. Moreover, the asymmetry itself is close to a tautology given the task construction: the knowledge target is 1,200 bits of i.i.d. uniform labels, and the skill target is a 168-parameter linear map. "An incompressible target needs a verifier as large as itself" follows without an experiment. A test that would mean something would use a knowledge domain with *partial* compressibility, or real-world verifiers such as retrieval and unit tests. Absolute gains are also compared across tasks with different headroom: skill acc₀ = 0.547 against knowledge acc₀ = 0.769. The paper notes this, but the abstract still reports the raw comparison.

**W2 (major; affects C3, the E5 headline and E4). E5 multipliers are confounded with optimizer steps in a severely undertrained regime.** Training runs for a fixed 4 epochs, so an amplified corpus N₀(1+r) receives (1+r)× more gradient steps than the plain N₀ corpus. At N=2,000 the model gets about 125 steps.

- Recall-by-occurrence shows the regime clearly. At α=0.6, N=2,000, facts seen 7 times are recalled at 0.15 (chance is 0.0625). At N=16,000, facts seen *once* are recalled at 0.76. The plain α=0.6, N₀=2,000 model is near chance (0.105).
- The "largest multiplier" M=3.75, which feeds E4, therefore mostly measures that a model given 4× more steps finally starts learning.
- The repetition control confirms the point. Repetition, which is just more epochs on the same facts, already gives M=2.01 at r=1 and 2.66 at r=3. The increment attributable to paraphrasing over repetition is 1.04× at r=1 and about 1.12× at r=3.
- So the abstract's framing, "a source-grounded paraphrase is worth about one fresh occurrence (M=2.09)", is misleading. Most of that value comes from extra compute.
- Two controls are needed: fixed optimizer steps, or training each corpus to convergence with early stopping; and the repetition-relative multiplier reported as the primary quantity. "Fitted s rises 7× with N" is the same artifact, which the authors acknowledge in Limitations but not in the claims.

**W3 (moderate). E6 Part A overstates "not by information".**

- The partial rank correlation of gain with information *given b/a* is 0.948 on the skill task. Within each b/a level, information orders gain almost perfectly. b/a has 4 tie groups of 3 and cannot resolve within-group order at all.
- The correct conclusion is that b/a dominates the between-level variance in this grid, while information predicts within-level order. The regime is also built in: the grid's a only goes down to 0.25 with 16,000 proposals. At small a·m (few accepted samples), a must matter.
- "Precision of the accepted stream" is an endogenous outcome of the same run, not an ex-ante property of the verifier, so its 0.993 is partly circular.
- The seed bootstrap (P=0.000) resamples seeds over a fixed set of 12 design points. It says nothing about generalising over designs, and with 12 points the Spearman differences of 0.78 vs 0.97 rest on very few rank swaps.

**W4 (major; significance). No neural verified-self-training experiment and no external baselines.** The verification half of the paper, which covers C2, C4, P3, P4′ and P5, rests entirely on L2-regularised logistic regression with 8 rounds. Since W1 shows that a quirk of this learner drives several knowledge-side outcomes, the claim that results "carry over to a learner that generalizes" is weak. A small STaR/ReST-style loop on a tiny transformer (for example, arithmetic with a test verifier of controlled false-positive rate) would be feasible on CPU at the scale of E5. It would test P3/P4′ in a nonconvex learner. It has not been done.

**W5 (major; novelty). The contribution is incremental relative to existing work.**

- Knowledge diversity and paraphrase effects are established in synthetic planted-fact settings (Allen-Zhu & Li, Parts 3.1 and 3.3).
- Synthetic continued pre-training with a theoretical scaling model of knowledge acquisition already exists (EntiGraph, Yang et al. 2025, with a mixture-of-exponentials toy model). The paper cites EntiGraph but does not compare with its theory.
- Verifier-filtered retraining that converges to the verifier's knowledge center is Yi et al. (2025). Verification as necessary for scaling is Feng et al. (2025).
- The information view of synthetic data is Gan & Liu (2025). Urn asymptotics are May & Flournoy.
- What remains is an organising vocabulary (M), two elementary propositions, and toy confirmations. The research question is important. This paper's increment over the literature is small.

**W6 (moderate; the answer to the research question). The horizon arithmetic still does not answer the question in a defensible way.**

- (a) The Muennighoff 4-epoch row double-counts repetition, which the 300T stock already includes.
- (b) The E5 rows (2.92, 3.75) come from an undertrained 0.67M model on 2,000 synthetic facts (W2) and should not appear next to LLM-scale anchors.
- (c) WRAP and Kang et al. report compute or time speed-ups to a loss, while human data is still being consumed. Kang et al. explicitly find that pure rephrased data does not beat web text. Neither says how many *human tokens are saved* at fixed quality.
- (d) The data-growth rate g_D = 2.0–2.24 is derived from Chinchilla-optimal scaling, while actual frontier training is heavily over-trained.
- What survives is "Δt = ln M / ln g_D". With M of a few to ten, that gives a few years. This is arithmetic on assumed inputs, as the authors themselves now say.

**W7 (minor/moderate). Statistics.**

- E5 CIs come from bootstrapping 3 seeds, which gives only 10 distinct resamples per curve and understates uncertainty.
- E2 uses 3 seeds. Inverting a mean learning curve from 3 seeds, then bootstrapping, compounds this.
- No multiplicity or design-level uncertainty is reported anywhere.
- The claim that recall "saturates exactly at the coverage ceiling" (0.813 vs 0.812) is presented as confirmation of a prediction. For planted uniform-random facts, exceeding the ceiling is impossible except through chance, so this is not a risky test.

## Minor issues

- The abstract says E6 gives "Spearman 0.96–0.99" for b/a and precision, but b/a gives 0.972 on both tasks. State the numbers per predictor.
- "within seed noise of zero gain" for κ=0.25 is inaccurate (about 1.9 SE below zero; see W1).
- Prop. 4 requires d to be a fixed-length string, or a prefix code with H(d) ≤ E|d|. State this.
- In Lemma 3, the "conditional" information in E6 uses the learner's own softmax probability, not the posterior Pr[C_j=1 | F_j], so it is not Lemma 3's quantity. It is labelled a proxy, but the abstract's "conditional" is loose.
- §5.2: the evaluation averages over all 8 templates, including templates the repetition corpora never use. Part of "paraphrase beats repetition" is therefore a template-matching effect, not a storage effect. Report recall on the original template separately.
- Yi et al. (2025) now has an ICLR 2026 venue; update the reference.
- RePro, BeyondWeb and other effective-token measurements published since WRAP are not discussed.

## Questions for the authors

1. If E6 Part B is rerun with λ scaled by the human fraction, or with a per-example (sum-reduced) loss, do the knowledge gains reach the union-coverage bound? What is the equal-bits asymmetry once headroom is normalised?
2. Does E5 with a fixed number of optimizer steps (or trained to convergence) still give M > 1.2 relative to repetition? What is the repetition-relative multiplier at each (α, N₀, r)?
3. Can P3/P4′ be tested in a tiny transformer self-training loop, for example modular arithmetic with a test-suite verifier whose false-positive rate is controlled?
4. How do the E4 conclusions change once the repetition factor already in the 300T estimate is removed and the E5 rows are dropped?
5. Can E6 Part A be extended to a·m small, for example a = 0.02 or m = 200, to locate where information starts to matter?

## Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 3 |
| Rigor | 4 |
| Correctness | 5 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **4** |

## Decision: Reject (encourage resubmission after major revision)

The paper is honest and fully reproducible, and the revision handled the round-1 points in good faith. But its contribution now rests on four things: two elementary propositions, a tiny-transformer experiment whose multipliers are confounded with optimizer steps, a verifier-cost experiment whose knowledge side is driven by a regularisation artifact, and a convex-learner study of verifier purity. That is not enough for an ICLR main-track paper on LLM data exhaustion. In its current form this is workshop material.

The most valuable single change would be one neural experiment that addresses W1, W2 and W4 together:

- Train a tiny transformer under *fixed compute*.
- Measure (i) the paraphrase multiplier relative to repetition, and (ii) a verified self-training loop with tunable (a, b).
- Compare a fact-lookup verifier with a program verifier at equal bits.

---

## Addendum: E6 rerun

