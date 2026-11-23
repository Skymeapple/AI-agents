# Review (Round 1): "Familiar Is Not Known: Two Identifiability Failures in Machine Self-Knowledge"

**Reviewer role:** independent area-chair-level reviewer
**Recommendation:** Weak reject (5/10) for ICLR main track; would be a good workshop paper as it stands.

---

## 1. Summary

The paper asks how a model can reliably "know that it does not know", and recasts this as a detection problem. It isolates two structural failure modes:

1. **Ambiguity confound (Prop. 1).** Any detector that is a function of the predictive distribution q has AUROC ≤ 1 − α(1−δ)²/2, where α is the aleatoric share of "known" items and δ = TV(P₀, P_alea). The paper also argues the bound is tight.
2. **Familiarity shortcut (Prop. 2).** In a whitened Gaussian hidden-state model z = aFu + bKv + ε, the population LDA correctness probe trained with co-occurrence π = P(F=K) has a false-"knows" rate on familiar-unknown items given in closed form. The rate increases in π and tends to Φ((a²−b²)/(2σ√(a²+b²))) as π → 1.

The authors test both on a synthetic relational world (4,000 entities, 8 relations) with 5-member ensembles of two-layer embedding MLPs over 3 worlds. A natural-data hidden-state probe has AUROC 0.930 on natural cells and 0.245 on dissociated cells. MSP AUROC falls from 0.816 to 0.169 as α goes 0 → 1. A trained q-only detector recovers 0.954 at α = 1 (δ̂ ≈ 0.83), so Prop. 1 is **not binding** in their networks. Prop. 2 gets the direction right but under-predicts the shortcut by about 0.23. Only ensemble MI is uniformly good. The Good–Turing singleton fraction matches the missing mass. There are no LLM experiments; a pre-registered LLM protocol is given instead.

## 2. Reproduction and verification

I re-ran the full pipeline in a clean copy of `code/` (theory_check.py; run_all.py for worlds 0–2; e1_qonly.py for worlds 0–2; merge; make_figures.py).

- `theory_check.json`: **bit-identical** to the submitted file. Max Prop. 2 MC error 0.0034; Prop. 1 MC within 0.0015 of 1 − α/2.
- `e1_qonly_w{0,1,2}.json`: **bit-identical**. δ̂ = 0.829 ± 0.024; q-only AUROC 0.960; Table 2b is reproduced exactly.
- `main_results.json` / `summary.txt`: identical except for last-digit floating-point noise (BLAS ordering). Two printed values differ by 0.001: missed-know for hidden at π_d = 0 is 0.418 vs 0.419, and false-know for hidden+output at π_d = 0.5 is 0.522 vs 0.523. This is immaterial.
- I cross-checked every number in Tables 1, 2, 2b and 3, the geometry (‖a‖, ‖b‖, cosines), r = 0.57, the mean absolute error of 0.233, and the E4 numbers against `summary.txt`. **All match.**
- Runtime: the README/paper claim about 6 min per world. On a 4-core machine, with the three worlds and E1b running concurrently, each world took about 3.2 h wall-clock. That is not a correctness issue, but the stated runtime is optimistic or environment-specific.

**Verdict:** the numbers are honestly produced by the submitted code, and I found no fabricated or unbacked results.

## 3. Reference spot-check (web-verified)

I verified the following, and each says what the paper claims:

- Cheang et al. 2025 (arXiv 2510.09033). Confirmed, including the later Findings of ACL 2026 version, which appeared under a new title ("Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness"). The paper should update the title. The claim that hidden states do not separate subject-associated hallucinations from correct recall is accurate.
- Ferrando et al., ICLR 2025: SAE entity-recognition directions that causally steer refusal and hallucination. Confirmed.
- Gottesman & Geva, EMNLP 2024 (KEEN). Confirmed (pp. 3994–4019).
- Reder & Ritter 1992, JEP:LMC 18, 435–451. Confirmed; the finding is described correctly.
- Kalai & Vempala, STOC 2024. Confirmed; the monofact / Good–Turing claim is accurate.
- Kapoor et al., NeurIPS 2024. Confirmed; the "≈1000 graded examples" claim is accurate.
- Wimmer et al., UAI 2023, PMLR 216:2282–2292. Confirmed.
- Allen-Zhu & Li, ICML 2024, PMLR 235:1067–1077. Confirmed.

No unverifiable citations were found.

## 4. Strengths

- **Honest reporting.** The paper states plainly that Prop. 1 is not binding in its networks, that Prop. 2 is off by about 0.23, and that there are no LLM experiments. This is rare and welcome.
- **Correct derivations.** I checked both proofs line by line (see §6) and found them correct.
- **The F × K cell framing is useful.** Reporting π and per-cell AUROC is cheap and actionable. The −log entity-count sanity baseline (0.955 natural vs 0.014 dissociated) is a good diagnostic that benchmark designers should adopt.
- **Clean reproducibility.** The code is small, the seeds are fixed, and the results reproduce to the last digit.

## 5. Weaknesses

### Major

**M1. The empirical evidence is too far from the question it claims to address.** The research question concerns models, in practice LLMs. The test bed is a two-layer embedding MLP that perfectly memorizes every seen fact (accuracy 1.000 for c ≥ 1) and puts all errors on never-seen facts (99.9%). It is also worse than a trivial type-default predictor on unseen facts (0.27–0.30 vs about 0.62 for `prior_acc`). This regime is degenerate in exactly the ways that drive the headline results:

- K ≈ 1[c ≥ 1], so any signal that detects "was this (e, r) pair trained on" solves the task. Ensemble MI does exactly that: member disagreement is driven by untrained or randomly initialized embedding/MLP regions.
- Unseen entities keep random embeddings, so "unfamiliar" is trivially out-of-distribution in input space.

The conclusion that "only parameter disagreement remains reliable" is therefore close to tautological in this world. It may not transfer to LLMs, where unseen names are composed from familiar subtokens and where seen facts are not perfectly memorized.

**M2. The E1 design confounds aleatoric-known with familiar, and the paper omits the baseline that reveals it.** The E1 negatives are aleatoric facts with c ≥ 20, so they all come from heavily trained entities. Positives include about 30% unseen-entity items. The code computes a −log entity-count baseline, and `fig_aleatoric.png` shows it rising from 0.743 to **0.992** at α = 1, above ensemble MI (0.970). Table 2 omits this row. As a result:

- MI's apparent invariance to α (0.948 → 0.970) cannot be attributed to MI "separating ignorance from ambiguity". It is also consistent with MI tracking familiarity or exposure.
- The rising count baseline shows that E1 at high α is largely a familiarity task.

The E1 negatives should be matched on entity exposure, or E1 should be reported stratified by F. Prop. 2's own lesson (state π) applies to E1 as well.

**M3. Neither proposition is validated on trained networks; both are partly falsified.**

- Prop. 1 is shown not to bind (δ̂ ≈ 0.83, ceiling about 0.99). The claim that calibration pushes models toward δ → 0 is conjecture; no experiment tests it. A calibrated or temperature-scaled model, or a label-smoothed or Bayesian variant, could have been trained cheaply in this setting.
- Prop. 2 under-predicts by 0.23 and mispredicts the slope of the remedy (predicted 0.488 vs observed 0.760 at π_d = 0.5). The fit across runs is r = 0.57. Calling it a "lower-bound-style diagnostic" is post hoc: nothing in the theory guarantees a lower bound under misspecification.

The theory thus supplies the direction of effects that are already obvious from the data design, but not their magnitude.

**M4. Limited novelty of the theory.**

- Prop. 1 is a direct corollary of the standard facts that ROC-curve sup(TPR − FPR) equals TV and that AUROC is linear in the negative mixture. The AUC–TV link is well known; see, for example, the separation results relating AUC(LR) − ½ to TV, and Kairouz et al.'s ROC/TV characterization. The idea that q-only signals cannot separate aleatoric from epistemic uncertainty is well established (Hüllermeier & Waegeman 2021, which the paper cites).
- Prop. 2 is the textbook Gaussian spurious-correlation model (a "core" feature b, a "spurious" feature a, and correlation π), analysed many times. Examples: Sagawa et al. 2020 on why overparameterization exacerbates spurious correlations; Nagarajan et al. 2021 on failure modes of OOD generalization; Rosenfeld et al. 2021 on the risks of IRM. None of these is cited. The novelty lies only in relabelling the spurious feature as entity familiarity.
- Empirically, the familiarity/popularity confound in LLM self-knowledge has been reported directly: Cheang et al. 2025 (associated vs unassociated hallucinations), "Popular but Wrong" (2025) on popularity-driven overconfidence, and the PopQA long-tail line of work. The contribution is therefore a formalization plus a toy replication, which is incremental.

**M5. Dissociated-cell AUROC is partly true by construction.** The dissociated pool is FU ∪ rare-seen, where rare-seen is 100% correct and FU is about 70% wrong, and both cells are defined by thresholds on entity count. The −log count baseline's 0.014 and the hidden probe's 0.245 are therefore largely guaranteed by how the cells are cut. The interesting question is how much fact-specific signal a probe has *within* a familiarity stratum. The paper should report within-F-stratum AUROC (for example, FU-correct vs FU-wrong), which removes the count-threshold artefact.

**M6. Statistical design.** There are 3 worlds, one architecture, one capacity, one training length and one regularization setting, and the reported ± values are SDs over 3 worlds rather than CIs. Results that depend on the memorization regime (E2, E4, and the inverted MSP in E1) are not stress-tested by varying epochs, weight decay or width. The paper says as much, but it means the generality of the qualitative claims is untested. Probe hyperparameters (L2 λ = 1, a fixed 0.5 threshold) are not tuned or ablated, and the false-"knows" rate depends directly on the threshold.

### Minor

- m1. Table 2's "Prop. 1 ceiling at δ = 0" row (1.0 → 0.5) differs from the dotted line in `fig_aleatoric.png`, which starts at the measured A_det of 0.816. State which is meant.
- m2. The Prop. 2 derivation assumes P(K=1) = ½ in probe training. The natural pool here is roughly 40/60, so the logistic probe's threshold is not the LDA midpoint. This is another source of the 0.23 gap that is not mentioned.
- m3. The δ̂ "lower bound on TV" comes from maximizing Youden J over thresholds on the evaluation fold, with about 110 aleatoric items per fold. The max-over-thresholds step is optimistically biased, so it is not strictly a lower bound in finite samples. Use a threshold fixed on the training fold, or a bootstrap CI.
- m4. Prediction P2 says mixing "cannot push FK below Φ(−b/2σ)". This holds only on π ∈ [½, 1]; with π < ½ it can go lower. State the domain.
- m5. The E4 Good–Turing "accurate abstention budget" claim is essentially guaranteed by perfect memorization (error rate 0 for c ≥ 1). Under realistic imperfect memorization the correspondence is not tested.
- m6. Update the Cheang et al. citation to its published title in Findings of ACL 2026.
- m7. The README's runtime estimate (about 6 min per world) did not hold on a 4-core machine with concurrent runs: about 3.2 h per world. Report CPU-hours.
- m8. Remark 3 on semantic entropy as q-only is correct in spirit, but semantic entropy is computed from samples, so it is a noisy estimate of a functional of q. Make the measurability statement precise.

## 6. Derivation check

- **Prop. 1.** The decomposition AUROC = (1−α)A(P₀, P_det) + αA(P₀, P_alea) is correct, because AUROC is linear in the negative law. The bound TPR − FPR ≤ TV is correct for any threshold set. The integral ∫₀¹ min(1, f+δ) df = 1 − (1−δ)²/2 is correct. Tie-handling by interpolation is safe because the bound is concave. The tightness construction is correct: P_det is disjoint from P₀ and scored lowest, and P_alea equals P₀ except on a δ-mass. ✔
- **Prop. 2.** I checked: Σ_W = σ²I + a²π(1−π)uuᵀ; the means; w ∝ aκu + bv; the threshold; the standardization; g′(κ) = a²b²(1+κ)/(a²κ²+b²)^{3/2}; the monotonicity of κ on [½, 1); the limits (ii) and (iii); the accuracy statement (iv); and the missed-knowledge rate. All are correct. ✔ The general non-orthogonal implementation in `theory.lda_rates` agrees with the closed form (fk_general = fk_closed in theory_check.json).

## 7. Does the sub-question stay sharp, and do the conclusions follow?

The sub-question is sharp: which signals can, in principle, separate "I don't know" from "it's random" and from "I'm familiar". The qualitative conclusions for the toy world follow. Two of the broader conclusions overreach:

- "A model's own predictive distribution fails the first requirement in principle" (Discussion). The paper's own E1b shows the opposite in its networks: a q-only detector reaches 0.954.
- "Only parameter disagreement remains reliable." This is confounded by M1 and M2. The count baseline beats MI on the E1 task at high α.

The evaluation recommendation (report per-F×K cell and state π) is well supported and is the most durable contribution.

## 8. Questions for the authors

1. Please add the −log-count row to Table 2, and report E1 with negatives matched on entity exposure. Does MI stay flat?
2. Can you train a calibrated variant (temperature scaling, label smoothing, mixup, or early stopping) and show δ̂ moving toward 0, so that Prop. 1 binds? This is the paper's central conjecture, and it is testable in your setting.
3. What is the within-FU AUROC (correct vs wrong among familiar-unseen-fact items) for each detector?
4. How do the results change under imperfect memorization (fewer epochs, stronger weight decay, smaller width)?
5. How does Prop. 2 differ from existing Gaussian spurious-correlation analyses (Sagawa et al. 2020; Nagarajan et al. 2021)?
6. Can even a small LLM pilot be run (for example, PopQA with Llama-class 7–8B and a linear probe over cells i–iv)? Without one, the paper's claims about LLM self-knowledge rest entirely on analogy.

## 9. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: weak reject (major revision).** The work is honest, correct and reproducible, but its theory is a light repackaging of known results (an AUC/TV bound and the Gaussian spurious-correlation model). Its only empirical evidence comes from a degenerate memorizing MLP. That evidence leaves one proposition non-binding and the other quantitatively wrong, and a key E1 comparison is confounded by familiarity, with the revealing baseline left out of the table. An LLM pilot, exposure-matched E1, within-stratum probe evaluation, and a calibrated-model test of Prop. 1 would move it toward acceptance.
