# Review (round 1): "Where Unification Stops Paying: A Capacity–Data Theory of Competition and Synergy in Unified Multimodal Models"

Reviewer role: area chair / senior reviewer. Target venue: NeurIPS 2027 main track.

## 1. Summary

The paper builds a linear "shared-core" model of a unified multimodal network. There are M modalities. Each is a multi-output linear regression whose target depends on k_s directions shared across modalities and k_p private directions. The unified model is a rank-r bottleneck U shared by all modalities with per-modality heads. The baseline is M rank-r specialists, following the same-size convention of Aghajanyan et al. (2023). The results are:

- **Prop. 1 / Cor. 1 (population).** An exact Eckart–Young computation. The interaction is zero for r ≤ k_s, positive on (k_s, k_s+Mk_p), peaks at r = k_s+k_p, and gives a "modality budget" M* = ⌊(r−k_s)/k_p⌋.
- **Lemma 2.** The spectrum of the pooled Gram matrix under misalignment θ.
- **Prop. 2 and Cor. 2–4 (finite data).** These rest on a heuristic: excess risk is the discarded signal plus σ²/n times the dimension of the rank-ρ manifold (the reduced-rank-regression df count). They give a bounded synergy ratio Γ, a "head-interference" penalty when φ = 0, and an alignment tolerance sin²θ_c ∝ 1/n.
- **Prop. 3 (power-law spectra).** A scale-invariant population loss ratio R = (φM^{1/α}+(1−φ)M)^α/M, a "capacity tax" κ = R^{1/(α−1)}, and a predicted data-bound crossover r_× ∝ n^{1/α}.
- **Remark 4.** The rectangular BBP threshold shifts by √M under pooling: down for shared features, up for private ones.
- **Experiments.** Monte Carlo checks of all of the above in the same linear model; a small one-hidden-layer ReLU teacher–student study; and a pre-registered protocol for real models that was not executed.

## 2. Reproduction and verification of numbers

I copied `code/` to a scratch directory and reran every script into a fresh `results/` folder.

- `exp_linear.py`, `exp_gap.py`, `exp_threshold.py`, `exp_powerlaw.py` and `exp_crossover.py` reproduce the shipped JSON files **bit-for-bit**. Across 3,400 numeric leaves, I found 0 differences.
- `exp_mlp.py` was rerun as well. Every run that finished before I submitted this review is bit-identical to the shipped `mlp_results.json`. On a shared single-core machine the full sweep did not finish in my review window.
- I checked every table entry in Sections 5.1–5.7 against the JSON files. All of them match:
  - Exp A: 1.40/1.46, 5.50/5.62, 26.1/26.5, 516.5/521.3, Γ = 0.9375, windows 5–19.
  - Exp B: M* = 3; 1.58/5.22/92.5.
  - Exp C: 140/140 cells; the θ_c table.
  - Exp D: 17%, 8%, 19%, 9%, 4.6%, 1.8%, 0.8%, 0.4%.
  - Exp E1: the R/κ/M/κ table. I also recomputed it by hand.
  - Exp E2: 0.90, 64, and the range 1.6–4.64.
  - Exp G: 7.17, 7.30, 1.41/1.37/1.34, 0.42.
  - Exp F: the full table, from `mlp_summary.json`.
- One provenance gap. The E3 slopes quoted in the paper (0.567 sim / 0.577 df, "8 unsaturated points") are **not** the slopes stored in `crossover_results.json`, which are 0.569 / 0.549 over all 9 points. I reproduced 0.567 / 0.577 by excluding the row with ρ_u* = 240, but no script performs that filtering. This is minor, but it breaks the appendix claim that "every number is read from a JSON file".

I found no fabricated or unbacked numbers. However, one headline accuracy claim is selectively reported (see W3).

## 3. Reference spot-check (web search)

| Reference | What the paper says it shows | Verified? |
|---|---|---|
| Aghajanyan et al. 2023 (ICML) | >250 models, 7 modalities, additive interaction term, "competition barrier", 30B speech–text model crosses it | Yes |
| Liang et al. 2025, Mixture-of-Transformers (TMLR) | Matches dense Chameleon-7B at 55.8% of FLOPs; modality-decoupled non-embedding parameters | Yes |
| Shukor et al. 2025 (ICCV) | 457 models; early ≈ late fusion; MoE learns modality-specific weights and helps | Yes. The paper omits Shukor et al.'s finding that modality-*agnostic* routing beats modality-*aware* routing, which is relevant to P6 (see W7). |
| Huang et al. 2022 (ICML, PMLR 162:9226–9259) | Modality competition proved in GD dynamics of late-fusion networks | Yes |
| Chang et al. 2024 (EMNLP) | Curse of multilinguality across 250 languages; depends on language similarity | Yes |
| Wu, Zhang & Ré 2020 (ICLR) | Misalignment causes negative transfer in shared-module models | Yes. But the paper's statement that Wu et al. "do not study how the effect scales with capacity" is inaccurate: Wu et al. explicitly analyse the shared module's capacity (output dimension) and its effect on transfer. |
| Reed et al. 2022, Gato (TMLR) | 604 tasks, single network | Yes |

All seven references I checked exist and broadly say what is claimed. None is fabricated.

## 4. Strengths

1. **Clean, honest and fully reproducible.** Scripts are deterministic and outputs reproduce exactly. Limitations are stated candidly, including the df model's failures at low SNR and the ReLU competition below k_s. The paper says clearly that it does not claim transformers are linear shared-core models.
2. **Correct elementary theory.** I checked Lemma 1, Prop. 1 (all four cases), Cor. 1, Lemma 2, Cor. 2's formula and its d ≫ Mq limit, Cor. 3, Cor. 4's closed form, the full proof of Prop. 3 including κ and M/κ, and the BBP scaling in Remark 4. All are correct as stated, apart from a mis-phrased condition in Cor. 4 (minor issue m1).
3. **Useful conceptual decomposition.** The paper separates a capacity limit, a statistical limit and an alignment limit. The scale-invariant capacity tax under power-law spectra is a crisp statement. The observation that the alignment tolerance tightens with data (θ_c ∝ n^{-1/2}) is a nice, falsifiable and somewhat counter-intuitive prediction.
4. **Good experimental hygiene within the toy model.** The theory is checked against Monte Carlo simulation and also tested in a regime where it fails. The paper ships a pre-registered protocol with explicit pass/fail criteria.

## 5. Weaknesses

**W1. The research question is answered only inside a toy model, and the link to real unified models is untested.** The question is where the limits of unified multimodal (vision/audio/video/action) models lie. Every quantitative result concerns a linear rank-r regression with isotropic Gaussian inputs, orthogonal design and oracle rank selection. The only bridge to real networks is a ReLU teacher–student study: one hidden layer, d = 32, 3 seeds. Real multimodal data are not used anywhere, not even the cheapest checks (for example, estimating α and φ from pretrained unimodal encoders, or a small two-modality transformer sweep). The pre-registered protocol is not executed. As a result, "the limits of unified models" in the title and Discussion are limits of the linear model. The Discussion's architectural implications (MoE/MoT/MoMa "act on exactly the two penalty terms") are interpretations, not findings.

**W2. The ReLU experiment contradicts Corollary 2, and the paper's explanation points the wrong way.** Cor. 2 gives Γ, whose infimum over q/d is reached at d ≫ Mq: Γ → 1 − φ(1 − 1/M). This is the *most* synergy the df model allows. With φ = 0.5 that is 0.75 (M = 2), 0.625 (M = 4) and 0.5625 (M = 8). For the actual MLP configuration (d = 32, q = 8, k_s = k_p = 6), Γ is 0.80 (M = 2) and 0.76 (M = 4). For M = 8 it is undefined, because ρ_u = 54 > d. The ReLU networks reach 0.61, 0.51 and 0.49, which is below even the d ≫ Mq bound. The paper compares these to "the linear Γ (about 0.94)", but that is the Γ of the *Exp A* setup, not the MLP setup. It then says the gap is "consistent with Corollary 2's dependence on q/d". Smaller q/d moves Γ toward the bound but cannot get below it. So the nonlinear study shows that the "bounded synergy" claim, presented in the abstract and Discussion as a structural limit, does not carry over to the one nonlinear model tested. This needs to be reported as a failure of the prediction, not rationalised.

**W3. The headline accuracy claim is selective.** The abstract says Monte Carlo matches "the theory's loss ratios within 2.3% (median 1–2%) for n ≥ 1000". That holds for Exp A only. From the shipped JSON:
- Exp B at n = 1000: 3.7% max error.
- Exp C at n = 1000: 6.6% max.
- Exp E2: max relative errors of 5.8%, 6.2%, 6.2% and **10.2%** at n = 2000, 8000, 32,000 and 128,000. At n = 128,000 the plateau ratio is 0.89 in simulation against 0.80 in theory.

The E2 discussion reports only n = 500 and n = 10^7, which skips exactly the intermediate-n curves where the df model is visibly off. The abstract should state the error range across all experiments.

**W4. The finite-sample "n" is not a sample size in the low-data regime.** The exact reduction assumes X_mᵀX_m = nI, which requires n ≥ d. Exp D (n = 50 with d = 64) and much of Exp G (n = 8–48 with d = 64; the shared-feature detection point n = 48 is quoted as evidence) fall in the region n < d, where no such design exists and OLS is not defined. Here "n" is only an inverse noise variance. This matters most for the claims that depend on low-SNR behaviour, chiefly the threshold shifts in Remark 4 and Exp G. Those claims should be restated as an SNR model or redone with random Gaussian designs. Random designs are also listed as unexamined in Section 8.

**W5. Weak empirical support for the threshold-shift remark.** Exp G identifies "onset" on a coarse grid {…, 32, 48, 64, 96, 128}. The observed onsets (48 / 96 / 128) are about 3× above the BBP predictions (16 / 32 / 64). The "≈2× = √M" shift for shared features is the ratio of two adjacent grid points, and the private shift (1.33×) misses √M = 2. Such a test cannot distinguish √M from any factor between roughly 1.3 and 3. Either use a fine n-grid with an overlap-based detection statistic, or downgrade the claim.

**W6. Several "validations" are tautological or weak.**
- E1 checks the continuum approximation of Prop. 3 against the discrete sum of the *same* spectral model. It verifies the integral, not the model.
- In E2, the finite-feature population ratio rises monotonically from 1.6 to 4.64 over r = 2–64. The paper calls this "close to R = 2.25, as Proposition 3 predicts", but it is not flat and spans a factor of about 3.
- The 140/140 sign agreement in Exp C is dominated by cells with |log-ratio| ≫ 0. Only the handful of cells near the boundary are informative, so the accuracy of the *boundary location* should be reported.
- The crossover slope (0.567 against an asymptotic 0.5) is given without a confidence interval: 3 repetitions, a step-2 r-grid and 8 points.

**W7. Novelty is moderate, and related work is incomplete.**
- Prop. 1 follows directly from Ky Fan / Eckart–Young once the model is written down.
- The finite-sample analysis is the textbook reduced-rank-regression df count applied to two matrix shapes.
- Remark 4 is a direct application of Benaych-Georges–Nadakuditi.
- The scale invariance in Prop. 3 is essentially built into the self-similarity of a power-law spectrum. Still, R and κ are a tidy and, as far as I can find, new formulation.

Missing or mis-characterised prior work:
- Wu et al. (2020) *do* study shared-module capacity. The paper's characterisation of them needs correcting.
- Ildiz, Zhao & Oymak (AISTATS 2024): task competition and inverse scaling in multitask representation learning, via random-matrix theory. This is closely related.
- Solvable power-law scaling models: Bahri et al. 2021; Maloney, Roberts & Sully 2022; Paquette et al. 2024. These are the natural comparators for Prop. 3 and the n^{1/α} crossover.
- Multilingual scaling-law work that explicitly models capacity-limited interference: Fernandes et al. 2023 on multilingual NMT scaling; He et al. 2024/25; ATLAS 2025.
- Shukor et al. (2025) found that modality-*agnostic* routing beats modality-*aware* routing. This bears on P6 and on the claim that "modality-specific routing" removes the head-interference term. The paper cites Shukor et al. in support but omits this finding.

**W8. Statistical reporting is thin in places.**
- MLP: 3 seeds and one learning rate shared by all arms; no per-arm tuning; early stopping on a validation set with only n/4 + 64 points.
- Crossover: 3 repetitions. E2: 4 repetitions.
- No CIs on slopes or crossover widths.
- The "first width with ratio < 1 after peak" statistic is fragile. At n = 256, M = 2, the ratio dips to 0.89 at h = 8 and then returns to 1.0–1.05.
- In the ReLU study, the M = 8 teacher has 54 feature directions in d = 32, so the assumption of mutually orthogonal shared and private subspaces is violated. The comparison with k_s + Mk_p = 54 is therefore not on the theory's footing.

## 6. Minor issues

- **m1.** Cor. 4's condition "k_s+Mk_p ≤ r < k_s+Mk_p+(M−1)k_s, so the core cannot also represent the gap directions" is misphrased. With r = 30 the core *can* represent 10 of the 12 gap directions. It is the df cost, not capacity, that makes them unprofitable at small θ. Restate the result as a first-order result under that condition.
- **m2.** Prop. 2 is labelled a proposition but is a heuristic, as the text admits. Call it a "df approximation" or "Heuristic 1".
- **m3.** The E3 slopes are not in the JSON (see Section 2). Add the filtering to `exp_crossover.py`.
- **m4.** `exp_linear.py` uses the mean of ratios, while `exp_gap.py` uses the ratio of means. Make the definitions consistent or state them.
- **m5.** "Population window 5–19": the code stores (5, 20) as (first r with ratio > 1, first r with ratio < 1). The table's "5–19" is correct, but the convention should be stated in the caption.
- **m6.** Exp A at n = 200 and Exp C at r = 4 show finite-sample synergy *below* k_s (ratios 0.83–0.92). This is interesting, and it is the regime of Remark 4. It deserves a sentence, since the abstract states that the interaction is "zero below k_s".
- **m7.** In Exp F, competition at h ≤ k_s reaches about 2.0 at h = 6 for n = 4096, M = 4, not just 1.2–1.4. The text reports only h ≤ 4.

## 7. Questions to the authors

1. What does the df model predict for the actual MLP configuration (d = 32, q = 8, M = 2/4), and how do you reconcile observed ratios below 1 − φ(1 − 1/M) with Cor. 2?
2. Can you redo Exp D/G with random Gaussian designs (n ≥ d) and a fine n-grid, and report BBP-shift estimates with CIs?
3. Can you run even the smallest instance of the Section 7 protocol (for example, two modalities with ~10M–100M-parameter decoder transformers at 3 sizes and 2 data budgets) to test P1/P2?
4. How sensitive are the conclusions to oracle rank selection? What happens with a validation-selected rank?
5. How do you reconcile P6 and the routing claims with Shukor et al.'s finding that modality-agnostic routing outperforms modality-aware routing?

## 8. Assessment

This is a careful, honest and exactly reproducible toy-model paper with correct, if elementary, derivations and a crisp conceptual framing of capacity, statistical and alignment limits. However:

- its contact with real multimodal systems is nil;
- its one nonlinear test contradicts a headline corollary, and the paper explains this away incorrectly;
- its abstract overstates theory–simulation agreement;
- its low-n results rest on a design assumption that cannot hold there;
- its novelty over standard reduced-rank-regression and multitask theory is moderate.

At a top venue this is borderline to weak reject. It would make a good workshop paper. With real-model evidence (even small scale) and an honest treatment of W2–W4, it could become a solid main-track submission.

## 9. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 5 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision (weak reject in its current form).**

No fatal issues: the rerun outputs match all reported numbers, and every spot-checked reference exists and supports its claims.
