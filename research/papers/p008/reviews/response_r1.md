# Response to Review Round 1 (p008)

We thank the reviewer for a careful report. The reviewer reproduced 10 runs and checked every number, and the questions were specific enough to act on directly. The most important consequence of the revision: our new long-training experiments **contradict our own round-1 headline** ("distillation saves data and compute, not capacity"). We have withdrawn that headline rather than defend it. The title, abstract, contributions, Section 6.5, discussion and conclusion are rewritten around what the evidence supports.

**New compute and code**
- 56 new training runs (E6-A: 30, E6-B: 20, E6-C: 6), bringing the total to 222.
- New run script `code/run_e6.py`.
- New analysis script `code/analyze_r2.py`, which writes `results/summary_r2.json`, `fig6_prop2_finiteM.png` and `fig7_e6_ceiling.png`.
- New raw files `results/e6_runs_{A,B,C}.json`, with model weights in `results/e6_models/`.

All new numbers in the paper are read from these files.

---

## Major issues

### M1. Prop. 2's operative claim E_S/E_T → r^α is untested and fails quantitatively at finite M = 400

**The reviewer is right, and we accept the criticism fully.**

What changed:
- **Proposition 2 is now stated for finite M (Section 4.2):** E_S/E_T = (r^α − κ^α)/(1 − κ^α), with κ = K_T/M the teacher's coverage. The admissible compression becomes r* = [(1+δ)(1−κ^α)+κ^α]^{1/α}. The round-1 formula is the κ → 0 limit and is now explicitly scoped to it.
- **Table 1 now has κ ∈ {0, 0.1, 0.6} rows.** The headline "4× cut for 2× error" (α = 0.5) holds only as κ → 0. At our teacher's κ = 0.6 it is 1.5×. The abstract and Section 7.1 now say this.
- **The operative claim is tested directly (Section 6.2, Figure 6).** The test covers 80 ordered pairs: capacity-regime students at the same α, plus each student against the trained teacher.
  - Asymptotic formula: median absolute log-error 0.235, underpredicting by about 21%.
  - Finite-M formula: median absolute log-error 0.088, overpredicting by about 6%.
  - The finite-M formula is closer in 78% of pairs.
  - The reviewer's Table 2 numbers are reproduced: at h = 32 the observed ratio is 4.1–4.8 for teacher-distilled students, against 5.0–5.7 finite-M and 1.9–2.1 asymptotic.
- **On "A ≈ top-K mass is tautological".** We agree that it mainly confirms frequency ordering. It is no longer offered as the evidence for P2; the error-ratio test replaces it.
- **The teacher's learned set is not a top-K prefix.** We now quantify this. The teacher learned 89% of its top-241 skills, its highest learned rank is 386, and its error is 0.0122 against 0.0114 for an exact prefix. Students are 75–91% prefix-consistent. This and partial credit explain the residual 6% overprediction, which is listed in the Limitations.

### M2. "Distillation saves data and compute, not capacity" overreaches

**The reviewer is right, and the new experiment the reviewer asked for confirms it.**

We ran E6 at unlimited data with world seed 0 and 5 seeds, comparing hard, oracle-soft and **real-teacher-soft** students (Section 6.5, Figure 7):

| width | steps | hard K | oracle-soft K | teacher-soft K | soft − hard [95% bootstrap CI] | teacher − hard [95% CI] |
|---|---|---|---|---|---|---|
| 16 | 45k | 18.2 | 21.2 | 20.5 (2 seeds) | +3.0 [−0.2, 6.4] | +2.3 [−0.8, 5.3] |
| 32 | 100k | 52.8 | 61.8 | 62.6 | +9.0 [6.4, 11.6], MW p = 0.012 | +9.8 [6.0, 13.4], MW p = 0.012 |
| 48 | 100k | 87.8 | 123.8 | 111.6 | +36.0 [29.2, 43.0], MW p = 0.008 | +23.8 [17.4, 30.4], MW p = 0.008 |
| 128 | 40k | 193.5 | 393.0 | — | +199.5 (2 seeds) | — |
| 48 | 250k | 114.5 (2 seeds) | — | 125.5 (2 seeds) | — | +11.0, 1.10× (MW p = 0.33) |
| 128 | 200k | 331.5 (2 seeds) | (393.0 at 40k) | — | — | — |

What changed in the paper as a result:
- **The soft-label advantage does not close at non-floor widths within our budget.** At h = 32 and 48 it is significant for both oracle and real-teacher labels. It narrows slowly with training: the hard/teacher ratio goes from 0.70 at the 10k checkpoint to 0.79 at 100k steps (h = 48). With 250k steps it narrows further, to 1.10× (2 seeds). At h = 128, where capacity is not binding, hard-label students go from 49% to 84% of the oracle-soft count when training is extended from 40k to 200k steps. The hard-label curves are still rising when every run ends.
- **Concessions on E5.** We concede both of the reviewer's points about E5. At h = 16 the 5-seed rerun gives 1.16× (CI now includes zero because the hard runs vary from 14 to 23), and "within seed noise" was inaccurate for the original 2 seeds.
- **The P1 argument cannot carry the claim.** We agree with the reviewer's logical point: P1 bounds all procedures from above and says nothing about equal *achievable* capacity. We now also show that P1 is about 16× loose in this testbed (see M3), which leaves ample room for the supervision signal to matter.
- **The headline is withdrawn.** It is gone from the title, abstract, Section 6.5, Section 7.1 and the conclusion. The new statement: no teaching signal exceeds the bit budget, and distilled students never learned a skill the teacher lacked (0 of 1,249 learned student–skill cases outside the teacher's set). Within our budget, soft labels raise the skill count reached at a given width by 1.2–1.4×, and whether that persists at convergence is stated as open. The LLM prediction L3 was rewritten accordingly.

### M3. Theory largely elementary; P1 untested; Busbridge "exponent" claim unchecked

**We agree on elementarity and have changed how the contribution is framed.**

What changed:
- **Elementarity.** Section 4.1 now says plainly that P1 is a standard counting / rate–distortion argument. P3 is the Hutter/Michaud threshold model with a supervision-dependent threshold, and its proof is substitution. The contributions list now places the weight on (i) the finite-domain correction to P2 and its validation, (ii) the calibration of P1, and (iii) the controlled measurements, not on the mathematics.
- **P1 calibration (reviewer's Q3).** Neither b nor R was supplied in round 1; both are now estimated.
  - *Rate.* We link the KL-based skill score to a distortion. Gaussian test-channel reconstructions of the 12 identifiable skill parameters are passed through the testbed's own score; half of the skills cross the threshold at relative MSE 0.136, which corresponds to **about 17 bits per skill**. We label this a heuristic estimate, not a proven lower bound for the KL-threshold distortion.
  - *Effective b.* We quantised all 30 E6-A models with per-channel uniform quantisation. Keeping ≥ 90% of learned skills needs **6–8 bits per parameter**, which is 231–344 bits per learned skill.
  - *Result.* The bound holds with at most about 16× slack. We state that this measures looseness and is not a falsification test, because P1 is a theorem.
- **Busbridge et al.** We removed the claims of a "mechanistic account" and of predicting "its exponent". Section 4.3 explains why our threshold model cannot produce an exponent comparable with their compute / teacher-cross-entropy law, and Section 2 now describes the relation as a qualitative similarity only.

### M4. γ is not a constant; weak fit; no uncertainty

**We agree.** Section 6.3 is rewritten.
- **Point-wise γ** is reported at every D: 10.5, 8.4, 6.5 and 3.6 for D = 10^4 to 3·10^5. It falls with D, as the reviewer found.
- **Free-exponent slopes** (over all 4,096 seed resamplings) are 0.64–0.82 for hard labels and 0.57–0.66 for soft labels. We state that the threshold model's exponent is misspecified.
- **Uncertainty** for the fixed-exponent γ: median 7.8, with a 2.5–97.5% range of 6.5–9.5 over seed resamplings. We label this a resampling range, not a CI, because each cell has 2 seeds.
- **New exponent-free estimator.** We compare the data needed by each method at matched skill count, D_hard(K)/D_soft(K). It is stable at 8.9, 7.7, 8.4 and 8.2 for K = 10, 20, 40 and 70. The paper now reports "about 8× data saving at matched skill count" instead of a constant threshold multiplier.
- **β sweep (reviewer's Q7 and minor point m5).** We added an accuracy-based skill count (argmax agreement with Bayes ≥ 0.9), which does not depend on KL0. The implied γ rises with entropy under both criteria (KL: 1.6 → 10.5 → ≥ 95; accuracy: 2.2 → 34.8 → ≥ 93), but the magnitudes differ by up to 3×. The text now claims only the direction and flags the floor effects for β ≤ 1.

### M5. Most "distillation" runs use oracle Bayes probabilities; Table 2 is not distillation

**We agree.**

What changed:
- **Table 2 is replaced** by students actually distilled from the trained teacher: h = 32 and 48 at 100k steps with 5 seeds, and h = 8, 16 and 24 at 45k steps with 2 seeds. For these students we report the observed error ratio, both predictions, the teacher skills retained, and the skills learned outside the teacher's set (none).
- **The real teacher is now used in the main capacity experiment** (E6-A, 5 seeds), not only in the single-seed E2T sweep.
- **Teacher-size effects** remain unstudied, and this is listed as a limitation. Training several teachers of different sizes, plus long-trained students for each, exceeded our compute budget.

### M6. Statistical rigour

What changed:
- **Seeds and statistics.** The key capacity-regime comparisons (h = 16, 32, 48) now use 5 seeds per arm, with 95% bootstrap CIs on differences, Welch tests and Mann–Whitney tests. We note that p = 0.008 is the floor attainable with 5 vs 5.
- **D\*.** D* is recomputed from the 100k-step E6 ceilings instead of the optimisation-limited 10k-step ones: 2.3·10^5 (h = 32) and 6.5·10^5 (h = 48). We removed the claim that observed relative gaps "shrink as predicted" and state that no crossover was observed.
- **D\* ∝ N^{1+α}.** This relation is now stated as conditional on K_cap ∝ N, and we say that the data do not support proportionality. Parameters per skill are 81 at h = 8, 35 at h = 48 (100k steps) and about 55 at h = 128.

The remaining 2-seed cells (E1–E5, h = 8/24/128 in E6) are listed as such in the Limitations.

### M7. Missing related work (Hooker et al.; long tail and Feldman)

**Added.** The new references, each verified by web search:
- Hooker et al. 2019 (arXiv:1911.05248);
- Dam et al. 2023, "Understanding the Effect of the Long Tail on Neural Network Compression" (arXiv:2306.06238);
- Feldman 2019 (arXiv:1906.05271);
- Feldman & Zhang, NeurIPS 2020 (arXiv:2008.03703);
- Zhou et al. 2024, BalDistill, Findings of EMNLP 2024 (arXiv:2406.13114).

Section 2 now opens with "Compression and the long tail". It states that we do **not** claim the head–tail divergence as new, and narrows the gap statement to (i) the closed-form law with teacher coverage, validated in controlled runs, and (ii) the controlled separation of the regimes in which soft labels help.

### M8. No LLM measurement

**We could not do this, and we say so plainly.** This environment has no GPU and no model access: the Hugging Face hub is blocked by the network proxy, which we checked, so no public checkpoint (Gemma 2, R1-Distill, or even DistilGPT2) can be downloaded.

What we did instead:
- **A public-data analysis (Section 6.6)** on the PopQA popularity vector (14,267 items, Wikipedia page views), using the general-p form of Prop. 2.
  - The popularity distribution is not a single power law: local slopes are −0.74, −1.96 and −3.31.
  - Under top-f knowledge, view-weighted vs entity-balanced accuracy is 84% vs 10% at f = 0.1 and 71% vs 5% at f = 0.05.

  This shows that the aggregate–tail effect is order-of-magnitude in a real knowledge distribution. It is explicitly labelled as arithmetic on public data, not a model measurement.
- **The protocol (Section 7.2)** is sharpened into concrete inference-only steps.

Significance for frontier models remains a prediction, as the Limitations state.

---

## Minor issues

- **m1 (§6.3, "0.57–0.79 throughout").** Corrected. The h = 128 relative gap is 1.0 at D = 10^3 and 3·10^3, where K_hard = 0, and 0.57–0.79 for D ≥ 10^4.
- **m2 ("within 5 points").** The sentence is removed. The new Table 2 uses teacher-distilled students, which are 3.8–4.7 points below the teacher at h = 32, and the text gives that range.
- **m3 ("about 40 parameters per skill").** Removed as a constant. We report the range: 81 (h = 8), 39 (h = 48 at 45k), 35 (h = 48 at 100k) and about 55 (h = 128).
- **m4 (scope Table 1; finite-M Prop. 2).** Done; see M1.
- **m5 (s_t not comparable across β; accuracy-based K).** Done; see M4.
- **m6 (min/max error bars).** The E6 comparisons now carry bootstrap CIs and tests. The remaining 2-seed cells are flagged.

## Reviewer's "highest-leverage change"

1. **Long-training hard vs teacher-distilled runs at h = 32, 48 and 128 with ≥ 5 seeds.** Done for h = 32 and 48 (100k steps, 5 seeds). At h = 48 we also ran 250k steps (2 seeds per arm) and at h = 128 up to 200k steps (2 seeds). We could not afford 180k+ steps with 5 seeds at every width on the shared machine. The result reversed our headline, and the paper now says so.
2. **Prop. 2 for finite M, validated against the testbed.** Done.
3. **A PopQA evaluation of a public distilled LLM.** Not possible in this environment (no model access, no GPU). It is replaced by a PopQA popularity-data analysis, with the inference protocol given for future work.
