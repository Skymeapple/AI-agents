# Review (Round 1): "Capacity, Not Teaching, Bounds Compression: A Skill-Frequency Theory of How Far Distillation Can Shrink Models"

Reviewer role: independent area-chair-level reviewer. Target venue as stated: NeurIPS 2027 main track.

## 1. Summary

The paper asks how far distillation can shrink a model while keeping frontier capabilities. It models a capability as M discrete skills with Zipfian usage p_i ∝ i^-(1+α) and states three propositions:

- **P1**: an information-theoretic bound. A student whose parameters take at most 2^{bN} values can retain a set S of independent skills to distortion ε only if bN ≥ Σ_{i∈S} R_i(ε). The bound holds for every training procedure.
- **P2**: under all-or-none, top-K skill retention, the aggregate-error ratio is r^α when the student keeps 1/r of the teacher's skills, while the uniform "tail" score falls to 1/r.
- **P3**: a threshold learning model (skill i is learnable once D·p_i ≥ n_m) in which distillation divides n_m by γ. This raises the data-limited frontier by γ^{1/(1+α)}, leaves the capacity frontier unchanged, and implies a crossover D* ∝ K_cap^{1+α}.

The evidence is a CPU testbed: 400 random bilinear-softmax skills routed by fixed random codes, 2-layer MLP students with 404 to 76k parameters, and 166 runs. Oracle soft labels (the true conditional distribution) are compared with sampled hard labels, plus one trained teacher. The paper ends with four predictions and a protocol for LLMs, none of which is tested.

## 2. Reproduction and number checking

I reran the code from a clean copy of `code/` in a scratch directory, without touching the authors' results. A full rerun of all 166 runs was not possible: the shared machine was heavily loaded (load average about 28 on 4 cores), so the full grid would have taken several hours. I did two things instead.

(a) I ran `analyze.py` on the shipped raw files `all_runs.json` and `e5_runs.json`. It regenerated a `summary.json` byte-for-byte identical to the shipped one, and figures that match. I inspected Figure 5 visually.

(b) I retrained 10 runs from scratch with the authors' exact configs and seeds, spanning four experiments:
- E2: h = 16, D = 10^4, hard and soft, seeds 0 and 1;
- E3: β = 8, h = 128, hard and soft, seed 0;
- E1: random family, h = 8, 45k steps, seeds 0 and 1;
- E5: h = 16, unlimited D, 45k steps, hard and soft, seed 0.

All 10 reproduced the stored K exactly and A to 4 decimal places (e.g., E5 h = 16: hard K = 18, A = 0.8560; soft K = 21, A = 0.8713). The code is deterministic in practice, and the raw files are genuine outputs of the shipped code.

Separately, I checked every number in Sections 5–6 and the abstract against the shipped `results/summary.json` and raw `all_runs.json` / `e5_runs.json`. The following match exactly or to rounding: E1 K values (5.0 → 110.0), log-log slope 1.28, parameters per skill (81 → 39), Spearman values (−0.46/−0.65/−0.82/−0.93), convergence check (21→22, 53→63), low-rank ratios (61.5/20.0, 216.5/56.5, 375.5), head–tail MAE 0.016, Table 1, Table 2 (all columns), E2 values (33.5/7.0, 68/16.5, 181/77), n_hard ≈ 188, n_soft ≈ 22.5, γ ≈ 8.4, slopes 0.62 and 0.73, trained teacher (241 skills, A = 0.988, 76,292 params; 60 vs 68, 138 vs 181), D* values, relative gaps, the E3 table, and the E5 table (including 1.14× and 1.4 aggregate points). The run count is 166 (E1 24 + 2 convergence checks, E2 84, E3 20, E4 4, teacher 1, E2T 7, E5 24).

Small textual inaccuracies:
- §6.3: the h = 128 relative gap is described as "0.57–0.79 throughout", but it is 1.0 at D = 10^3 and 3·10^3.
- The abstract's "within 5 points" is exactly 5.0 points for seed 0 (0.988 vs 0.938).

I found no fabricated or unbacked experimental number.

## 3. Reference checks

I verified the following by web search. Each exists and supports the claim attributed to it:
- Busbridge et al. 2025 (PMLR 267:5977–6045; arXiv:2502.08606).
- Xiao et al., Densing Law. The preprint gives ≈3.3 months; the journal version (Nature MI, Nov 2025) reports ≈3.5 months, which is what the paper says.
- DeepSeek-R1. The 1.5B distilled model reports 28.9 on AIME 2024 and 83.9 on MATH-500.
- Allen-Zhu & Li, ICLR 2025, 2 bits/parameter.
- Menon et al. 2021 (PMLR 139:7632–7642, Bayes-teacher variance reduction).
- Gemma 2 (distilled 2B/9B, "competitive alternatives to models 2–3× bigger").
- Sardana et al. 2024 (PMLR 235:43445–43460).

No citation was found to be unverifiable. I did not independently check page numbers for Cho & Hariharan, Mirzadeh et al. or Phuong & Lampert, but these are well-known papers.

**Missing related work.** The most important omission is Hooker et al. (2019/2020), "What Do Compressed Deep Neural Networks Forget?". It shows empirically that compression keeps top-line accuracy while hurting the long tail ("pruning identified exemplars"). This is the empirical version of the paper's central aggregate–tail message. Also missing: work tying the long-tail effect to Feldman's memorisation theory (e.g., "Understanding the Effect of the Long Tail on Neural Network Compression"), and long-tail distillation work such as BalDistill. Each of these weakens the claimed gap in §2.

## 4. Strengths

1. The framing is useful. Separating capacity, data and optimisation limits, and separating frequency-weighted from uniform evaluation, is a clean way to reconcile "small distilled models look great" with "small models lose tail knowledge."
2. The code is honest and complete. It is small and readable, and every reported number traces to code. The authors flag many of their own weaknesses (non-convergence at h ≥ 32, floor effects in γ, the logit-norm confound, the unresolved 1.14×).
3. The testbed design is sensible. Fixed non-learned skill codes make parameter count meaningful as shared capacity. The random vs low-rank families directly manipulate compressibility.
4. The LLM predictions are concrete and falsifiable, and the protocol needs only inference on public checkpoints.

## 5. Weaknesses

### W1. The theory is largely elementary or tautological, and novelty is limited
- **P1** is a standard counting / rate–distortion argument. I checked the proof: H(ω) ≤ bN, then the chain rule with independence (I(θ_k; ω | θ_<k) ≥ I(θ_k; ω)), then data processing, then the definition of R(ε). It is correct, but it is textbook. The real content is the choice of b and R_i, and the paper supplies neither for its own testbed.
  - b is never measured. Students use float32 (b = 32 nominally), so the bound is not quantitatively compared with the observed ~40 parameters per skill.
  - The distortion d_i on θ_i is never connected to the KL-based skill score s_i.
  - In the testbed, P1 is therefore never tested. The figure of 2.5 parameters per Gaussian number is descriptive, not a test of the bound.
- **P2** is arithmetic on Zipf tails under an assumed all-or-none, top-K retention.
- **P3**'s "proof" is substitution into an assumed threshold rule. The rule is essentially Hutter (2021) and Michaud et al. (2023) plus a multiplier γ on the threshold.
- The claim of a "mechanistic account of the Busbridge crossover" that "predicts its exponent" is not checked against Busbridge et al.'s fitted law in any way.

### W2. "Distillation saves data and compute, not capacity" overreaches from the evidence
- P1 is an upper bound on what any procedure can reach. It does not show that hard and soft supervision reach the same achievable capacity for a given architecture and optimiser, which is what the slogan asserts.
- The empirical test (E5) uses only h = 8 and h = 16.
  - At h = 8 every condition sits at K ≈ 4–6. This is a floor, not an informative test.
  - At h = 16 with unlimited data, soft beats hard in both seeds (21/19 vs 18/17). The minimum soft value exceeds the maximum hard value. "Within seed noise" is therefore not an accurate description, even though the effect is small.
- At 10k steps (E2) the soft advantage at unlimited data is 15 vs 13 (h = 16), 38 vs 27 (h = 32) and 221.5 vs 94 (h = 128).
- No experiment at h ≥ 32 runs long enough to tell an optimisation-limited gap from a capacity-limited one. The central conclusion rests on one non-floor width with two seeds.

### W3. P2's quantitative prediction is not tested, and it does not match the testbed
The paper's evidence for P2 is that A ≈ Σ_{i≤K} p_i, which is close to tautological once learning is frequency-ordered. The proposition's operative claim is E_S/E_T → r^α, used for the "4× cut for 2× error" headline and Table 1. I checked it against Table 2 (α = 0.5, K_T = 241):

| h | K_S | r = 241/K_S | r^α (Prop. 2) | observed E_S/E_T |
|---|---|---|---|---|
| 48 | 106–114 | 2.1–2.3 | 1.45–1.51 | 2.44–2.51 |
| 32 | 53–60 | 4.0–4.5 | 2.0–2.1 | 4.4–5.1 |
| 16 | 19–21 | 11.5–12.7 | 3.4–3.6 | 9.5–10.6 |

The asymptotic formula underpredicts the error ratio by about 1.7–3×. The cause is finite M = 400 with K_T = 241 close to the truncation; a finite-M top-K computation gives 3.0–12.3, closer to what is observed. The teacher's learned set is also not a top-K prefix (its highest learned index is 385). The paper therefore illustrates its own regime with a formula (Table 1) that is off by a large factor in that regime, and never notices. The finite-M correction belongs in the proposition, and Table 1 should be clearly scoped to M ≫ K_T.

### W4. γ is not a constant, and the fit has little support
- γ is fitted from 3 hard-label points (D = 10^4, 3·10^4, 10^5) with the exponent fixed at 1/1.5.
- The point-wise ratios K_soft/K_hard at h = 128 are 4.8, 4.1 and 3.5, which imply γ ≈ 10.5, 8.4 and 6.6. γ falls with D, consistent with the different empirical slopes (0.62 vs 0.73).
- The β sweep then shows γ varying by more than two orders of magnitude, with floor effects and the logit-norm / metric confound. KL0 also changes with β, so s_t is not comparable across β.
- The parameterisation "distillation = constant γ on the threshold" is thus not well supported even inside the toy world. No uncertainty is reported for γ.

### W5. "Distillation" in most experiments is learning from Bayes probabilities, not compressing a teacher
- E1, E3, E4, E5 and most of E2 use the oracle p(y|t,z) as the soft target.
- Table 2 compares oracle-trained students with a separately trained hard-label teacher. The students were never distilled from that teacher, so "teacher skills retained" is a comparison of two independently trained models, not a measurement of distillation compression.
- The only real teacher (E2T) appears in one 7-point sweep with a single seed.
- Teacher-size effects, the heart of the "how small?" question in practice, are not studied.

### W6. Statistical rigour
- Most cells have 2 seeds, and error bars are min/max.
- There are no confidence intervals and no tests on the trend claims (e.g., relative gap "shrinks as predicted").
- The D* predictions use the 10k-step K_cap, which the paper itself says is optimisation-limited.
- The "crossover" is never observed: at h = 16 the gap plateaus at 0.24.
- The fit for D* ∝ N^{1+α} presumes K_cap ∝ N, but the testbed shows parameters per skill falling from 81 to 39 (slope 1.28, partly from non-convergence). The scaling relation is asserted, not supported.

### W7. Distance from the research question
The question concerns frontier LLMs. The experiments have 400 four-class bilinear skills and MLPs with up to 76k parameters. No LLM measurement is made, even though the authors note their own protocol needs only inference on public checkpoints, such as a PopQA popularity-stratified comparison of Gemma 2 2B/9B/27B or the R1-Distill family. Without even one such measurement, the significance for the stated question is speculative.

## 6. Questions for the authors
1. Can you state P2 for finite M and compare the predicted error ratios with Table 2? Why is Table 1 used to interpret a regime (M = 400, K_T = 241) where it is off by 2–3×?
2. Can you run E5-style long training (e.g., 180k+ steps, ≥5 seeds) at h = 32 and h = 48, where the soft advantage is large at 10k steps? Does the gap close?
3. Can you estimate an effective b for your students, e.g. by quantising weights until K drops, so that P1 is actually tested against the ~40 parameters/skill figure?
4. For Table 2, can you distil the students from the trained teacher rather than the oracle?
5. Can you report bootstrap CIs for γ and a fit with the exponent free?
6. Would a single PopQA popularity-binned evaluation of an existing distilled family (P1/P2 of §7.2) be feasible? It would greatly strengthen the paper.
7. How does the s_t metric (normalised by KL0, which depends on β) affect the E3 conclusions? Can you report accuracy-based K as well?

## 7. Scores (1–10)
- Novelty: 4. The framing is nice, but the components are standard (counting bound, quantization model, Zipf arithmetic), and Hooker et al. already showed the head–tail compression divergence empirically.
- Rigor: 4
- Correctness: 6. The proofs are correct; the numbers reproduce; P2's asymptotic is misapplied to the finite testbed; the central slogan is overstated.
- Significance: 4
- Clarity: 7
- Reproducibility: 9

**Overall: 4.5 / 10**, leaning to a weak reject at a top venue. It could become a workshop paper with the fixes below.

## 8. Decision
**Major revision.** The submission is honest, well engineered and reproducible, but it falls short at main-track level:
- the theory is elementary;
- the headline quantitative claim of P2 is not validated (and fails quantitatively) in the paper's own testbed;
- the central "not capacity" conclusion rests on one small width with two seeds where soft labels still win in both seeds;
- there is no measurement on actual language models.

**Single most valuable change:** add a real test of the "not capacity" claim and of P2. That means long-training hard-vs-teacher-distilled runs at several non-floor widths with ≥5 seeds, plus a popularity-stratified evaluation of at least one public distilled LLM family. Together these would turn the slogan into evidence.
