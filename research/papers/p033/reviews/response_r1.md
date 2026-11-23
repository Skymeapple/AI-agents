# Response to Review (Round 1)

We thank the reviewer for a careful and reproducible review. Every fatal and major point has been addressed. Where we agreed, we changed the text, added an experiment, or narrowed the claim. Where we could not do what was asked, chiefly running experiments on real open-weights models, we say so plainly. The paper has been reframed around the reviewer's central objection. Its title is now *"What Behavioural Evaluations Can and Cannot Certify About Conditional Defection."*

New code (all CPU, all run in this environment, outputs in `results/`):
- `exp0b_ablation.py`: clean annealing ablation.
- `exp4b_misspec.py`: tail estimation under misspecification, with sample sizes for explicit precision targets.
- `exp5_certify.py`: direction of evaluator-side bounds, and a certified bound.
- `exp6_lowalpha.py`: learned defectors at α ∈ [10⁻⁷, 10⁻³] with importance sampling.
- `exp7_diagnostic.py`: identification limits of the realism-response curve.
- `analysis_r1.py`: seed uncertainty, per-seed values, exact collapse rates.

`exp3_scaling.py` gained a d′ = 2.0 row and a ln N_req decomposition. All previously reported fields of `exp3_scaling.json` are unchanged (checked by key-wise comparison). Exp. 1, 2 and 4 were not re-run. Their JSONs are the byte-identical files the reviewer reproduced.

---

## Fatal

**F1. "β < 10⁻⁷" for incentive collapse is false for 3 of 6 policies.**
*Agreed; fixed.* §5.3 now lists all six values from `exp2_honeypots.json`, also tabulated in `analysis_r1.json`: 2.2×10⁻⁷ and 6.1×10⁻⁷ (ρ = 0.02, λ = 128), 5.8×10⁻⁸ and 2.7×10⁻⁷ (ρ = 0.1, λ = 32), and 6.5×10⁻⁸ and 4.1×10⁻⁸ (ρ = 0.1, λ = 128). The claim is now "all below 10⁻⁶", with an explicit note that the earlier statement was wrong. We also give both seeds for the λρ < 1 comparison policies: 0.098–0.10 and 0.18–0.25. We further relabel this check as a sanity check, since Cor. 3c fixes the sign of the very objective we train (point M3).

## Major

**M1. The ceiling cannot be operationalised. Evaluator estimates of R_Π are lower bounds, so the headline framing inverts.**
*Agreed. This is now the paper's central topic.* Changes:
- **New Proposition 5 (§4.5)** settles the direction question formally.
  - (a) An evaluator who observes behaviour only on E′ can certify "α′ ≤ a ⇒ β ≤ b" for all policies iff b ≥ R_{E′,D}(a). A part of D singular to E′, of mass s, makes β < s uncertifiable at any N.
  - (b) Any frontier R̂_g estimated from a probe, verbalised probability or trained discriminator is attained by the thresholded discriminator. It is therefore an *achievability (lower) bound*, an "insufficiency certificate", and gives no upper bound on R_Π without a certificate of class containment or elicitation completeness.
  - (c) **A route to a certified upper bound.** If E′ ≥ κD by construction, which is what happens when a share κ of evaluation episodes are real deployment opportunities replayed or audited under monitoring, then β ≤ α′/κ for every policy. A finite-sample version is β ≤ U_CP(k, N, γ)/κ. No frontier estimate is needed.
- **New Exp. 5 (§5.4).**
  - (a) On the XOR cue, 74/77 trained defectors lie above a logistic-probe "ceiling" and 30/77 above a width-2 MLP probe (by up to 2.0×). None lie above a width-8 probe. This demonstrates the inversion the reviewer described.
  - (b) Against adaptive best-response defectors, the deployment-resampling certificate is within 2.0×, 1.16× and 1.0× of the best achievable β at κ = 0.005, 0.02 and 0.1 (N = 10³), and within 1.27× and 1.04× at N = 10⁵. Clopper–Pearson coverage was ≥ 0.948 at nominal 0.95.
- Thm 1 now states explicitly that the model-free parts (b)–(c) are vacuous for near-singular text distributions ("Where the theorem bites, and where it does not").
- The abstract, §1, §6.1 and §8 are rewritten. Synthetic-input evaluations yield lower bounds on what can be hidden. Upper bounds need density-ratio control, which we know how to obtain only through deployment resampling. That route has stated costs: A1, audits the model cannot tell apart, and availability only at or after deployment (§4.5, §7).
- The literature conversions in §5.5 are relabelled as insufficiency numbers.

We did not find a general way to upper-bound R_Π for synthetic inputs, and we say so. The elicitation route, as used by Gardner-Challis et al. (2026) for self-recognition, is discussed as conditional on an elicitation-completeness assumption.

**M2. Novelty is modest, and close 2026 work is missing.**
*Agreed on the first part; addressed on the second.*
- §1 and §2 now say plainly that the theorems apply classical tools and that the qualitative insights circulate in AI control. The novelty paragraph limits our claim to the precise certification statement (Prop. 5) and the measurement methodology.
- The f-DP connection is made explicit: T(E, D) = 1 − R (Dong, Roth & Su, 2022; JRSS-B 84(1); verified). f-DP composition is named as the tool for the stateful case.
- Added and discussed, all verified by web search:
  - Gardner-Challis et al., arXiv:2602.20628 (§2, §6.2), including how Prop. 5 classifies their elicitation-based bound.
  - AutoHoney, NeurIPS 2026 poster.
  - arXiv:2608.21766, *Evaluation Awareness in Language Models: Representation, Verbalization, and Control*, on the partial mismatch between representations and verbalisation, which supports Prop. 5(b).
  - The upfront-auditing observation of Greenblatt et al. (2024b), now credited as the 1/N insight that Thm 2 generalises.
- We could not retrieve author lists for AutoHoney and 2608.21766, because arxiv.org, neurips.cc and huggingface.co were unreachable from our environment. The references say so rather than guess.
- The earlier "no prior work found" sentence is removed.

**M3. The experiments mostly verify identities.**
*Agreed.* We now label Exp. 1, the Cor. 3c part of Exp. 2, the probit-slope recovery (described as "near-circular by construction") and the validity part of Exp. 5b as sanity checks of code and optimiser (§1 Scope, §5.1, §5.3, §7). We added four experiments whose outcomes the theory does not fix:
- Exp. 4b: misspecification.
- Exp. 5a: probe ceilings are exceeded.
- Exp. 6: tail efficiency and its dependence on the optimiser.
- Exp. 7: identification of the realism diagnostic.

**M4. The experiments do not reach α ≈ 10⁻⁴–10⁻⁶, and 29 policies were excluded.**
*Addressed with a new experiment (Exp. 6, §5.2).*
- Design: 144 new policies at λ ∈ {64, …, 2048}. All rates are estimated by a defensive importance sampler with weights bounded by 2 (median relative SE of α̂ is 0.8%; at most 16% for α > 10⁻⁷). Two trainers differ only in the evaluation-gradient estimator.
- No policy exceeds the frontier.
- The gap to the frontier widens as α falls. XOR medians drop from 0.94–0.97 at α ~ 10⁻³ to 0.37–0.76 at α ~ 10⁻⁶·⁵.
- The gradient estimator matters a great deal. XOR width 32 at α ∈ (10⁻⁷, 10⁻⁶] reaches 0.37 with the plain trainer and 0.76 with the IS trainer.
- The linear policy collapses in all 36 runs. This is the timid collapse appearing exactly in the relevant regime, as the reviewer anticipated.
- Table 1 now shows, for each configuration, how many policies are excluded and points to §5.2 for them.

**M5. Exp. 4 does not support "~450 items cannot resolve this tail", misspecification is untested, and "several thousand" is not derived.**
*Agreed on all three counts.*
- (i) The claim is withdrawn and explicitly corrected in §5.6. Under correct specification, the unequal-variance fit at n = 450 gives non-overlapping intervals and is within 2× of the truth with probability ≥ 0.9 for b ≤ 1.
- (ii) New Exp. 4b tests four ground truths, all at AUC 0.83: binormal, a chat/agentic mixture calibrated within the ranges of Needham et al., a 5% "obviously real" deployment subset, and 2% deployment-like evaluation items.
  - Parametric fits converge to wrong values.
  - With the obvious-cue subset, equal-variance and full-range probit fits understate R(10⁻⁴) by 4.6–5.9×, and this does not improve with n.
  - A tail-restricted probit fit removes most of the bias at 10⁻³ but not at 10⁻⁴.
  - The empirical ROC is consistent.
- (iii) Sample sizes are now derived from stated targets (P(within 2×) ≥ 0.9, and within 1.25×). The empirical estimator needs n_E ≈ 0.5/α₀–20/α₀, and 5/α₀ sufficed in all AUC-0.83 scenarios. The factor-1.25 target costs 4–25× more. The recommendation in §5.6 is rewritten around these numbers.

**M6. The realism-response diagnostic is overclaimed.**
*Agreed; the claim is narrowed and a new experiment shows the confounds and a remedy (§4.4, §5.7, Exp. 7).*
- (a) With unknown realism units g(t), the slopes are 2.4 and 1.64 rather than d′ = 2, so only monotonicity is identified.
- (b) An aligned model with difficulty-correlated accidental errors gives a significantly rising probit slope (1.51) in 100% of replicates. A single ladder cannot separate it from a schemer.
- (c) A factorial realism × difficulty ladder separates the two. The realism coefficient is 1.98 vs −0.01, and false-positive rates are 2.5–3.5%. This holds provided realism can be varied at fixed difficulty, which we state as an assumption.
- The sentence claiming the curve "separates conditional from unconditional misbehaviour" is withdrawn. The protocol (§6.3, P3) now uses the factorial design.

## Minor

1. **"Tight over distribution pairs".** *Fixed.* Thm 1(d) now states that the binary-KL constraint is tight, that the Fano relaxation is loose by at most an additive ln 2/ln(1/α), and that only its 1/ln(1/α) rate is tight. The last point is proved: β*·ln(1/α)/κ → 1. The tightness claim is removed from the abstract.
2. **exp(d′²/2) scaling.** *Fixed.* Thm 2(b) now gives ln N_req = x²/2 + ln x + ln(c√2π) with x = d′ + |z_b| and the expansion d′²/2 + d′|z_b| + z_b²/2. It reports the terms 2.0, 4.65 and 2.71 at d′ = 2, b = 0.01 (`exp3_scaling.json`, `lnN_decomposition`) and the local slope x + 1/x = 4.56, against 4.54 numerically. The "Reading" paragraph states that the earlier phrasing was misleading.
3. **"92–99% when capacity matches".** *Fixed.* We now report medians of 0.94–0.997, means of 0.85–0.99 and a minimum of 0.60 (XOR width 32, 8 of 21 below 0.7).
4. **XOR λ = 16 linear example averaged a collapsed seed.** *Fixed.* Per-seed values are given in §5.1: linear 0.038, 0.039 and collapsed (α = β = 1.5×10⁻⁴); width 8 0.225, 0.250 and 0.230.
5. **"±0.020" is an SD; Table 1 lacks seed uncertainty.** *Fixed.* The probit slope is reported as SD 0.022 over 6 policies (sample SD) with a t-based 95% CI of [1.963, 2.009]. Table 1 now has bootstrap CIs, medians and the SD of the seed means.
6. **Exp. 0 is not a clean ablation.** *Fixed with new Exp. 0b.* It uses the same trainer, steps, learning rate, batch and seeds, and varies only the schedule. Fixed λ collapses at λ ≥ 16 (linear efficiency 0.027 vs 0.88; width 16 0.003 vs 0.98). Both schedules work at λ = 4. Annealing alone fails for the linear policy at λ = 64.
7. **§5.3 mixed d′ = 1.99 and d′ = 2.** *Fixed.* All honeypot comparisons now use the d′ = 2.0 baseline (N_req = 1.39×10⁴), added to `exp3_scaling.py`.
8. **Needham et al. version and the chat/agentic split.** *Fixed.* The per-model overall AUCs are attributed to v1 (28 May 2025). Later versions report chat 0.71–0.77 and agentic 0.88–0.95, which we cite. We note that this implies a non-binormal pooled ROC and model it in Exp. 4b. We could not open v1 directly (arxiv.org was unreachable), so the v1 attribution follows the reviewer's check. The version dates come from the arXiv listing as returned by web search.
9. **Thm 1(a) singular part.** *Fixed.* The theorem now uses the Lebesgue decomposition, sets L = ∞ on the singular set S, and gives R(0) = D(S).
10. **§2 "no prior work" claim.** *Fixed.* It is replaced by an explicit engagement with the AI-control honeypot and upfront-audit literature (see M2).

## Highest-leverage suggestion

> Give a credible way to upper-bound R_Π, or reframe as measurement methodology with misspecification tests and real ROC tails from open-weights models.

We did both of the parts we could do in this environment:
- **Reframe.** The paper is now about what can and cannot be certified, with the measurement methodology as a main contribution.
- **Credible upper bound.** We give one, the deployment-resampling certificate (Prop. 5c). It is valid for every policy, empirically tight against adaptive defectors, and comes with its preconditions stated.
- **Misspecification.** Tested in Exp. 4b.

**Not done:** measuring real evaluation-awareness ROC tails on open-weights models. This environment has no GPU and no model access, and huggingface.co is unreachable. We have not described any such experiment as run. It is now step 1 of the pre-registered protocol (§6.3) and is listed first under Limitations.
