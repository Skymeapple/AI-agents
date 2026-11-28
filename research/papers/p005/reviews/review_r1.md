# Review (Round 1): "Interpolation Is Relative to a Representation: Vacuous Input Hulls, a Readout Interpolation Certificate, and the Order in Which Grokking Networks Generalize and Then Interpolate"

Reviewer role: area chair / senior reviewer. Target venue: ICLR 2027 main track.

## 1. Summary

The paper takes up the question "do networks really generalize, or only interpolate?" and argues that it cannot be answered until one names the representation in which interpolation is measured. It has three formal pieces.

- **Proposition 1.** Inputs in convex position (one-hot or multi-hot codes, points on a sphere) never lie in the hull of a training set, so hull-based "interpolation" is vacuous for discrete inputs.
- **Proposition 2.** With one-hot modular addition, input-space 1-NN and nearest-class-hull classifiers are exactly 0% accurate on held-out pairs.
- **Proposition 3.** A "readout interpolation certificate": if φ(x) lies within δ of a point h in the same-label training hull, and the margin at h is larger than ‖w_y − w_c‖δ, then the prediction is y.

The experiments use small numpy MLPs.

- **Modular addition (p = 53, 6 runs).** Nearest-class-hull accuracy at the last hidden layer tracks test accuracy. Layer 1 and the input are not interpolative. The certified fraction C crosses 50% 2,500–4,000 steps after test accuracy does ("certification lag").
- **UCI 8×8 digits (5 seeds).** Pixel-space nearest-class-hull accuracy (98.4%) beats the MLP (97.1%). Certified predictions are 99.75% accurate, against 78.6% for uncertified ones. Certification collapses under 1–2 px shifts, and a "hull contrast" score detects errors better than max-softmax under shift.
- **Diagnostic.** δ shrinks and C rises with more training data per class.

The paper concludes that generalization, on these tasks, amounts to building a representation in which test points become certifiable interpolations, and that this comes after an earlier phase of uncertified generalization.

## 2. Reproduction and verification

**Code rerun.** I copied `code/` to a scratch directory and reran the full pipeline unchanged: `exp_modadd.py`, `exp_digits.py`, `exp_diag.py` and `make_figures.py`, with OMP/OPENBLAS threads set to 1. Section 9 compares the regenerated JSON with the shipped `results/` and the paper.

**Numbers in the paper vs the shipped JSON.** I checked every table entry against `results/summary.json`, `results/digits.json` and `results/diag.json`.

- Tables 1–4, the lag table, the pooled certified counts (5,245/5,985), the pooled AUROCs and the LP membership fractions all match the shipped outputs to the stated precision.
- There are small wording inconsistencies, but no fabricated numbers:
  - "78.5%" (mean over seeds) and "78.6%" (pooled) are used interchangeably.
  - Section 6.1, point 3, says I₂ crosses 0.5 "at the same checkpoint as test accuracy in all four grokking runs" and then lists 3,500 vs 3,000 for fraction 0.6.

**Figures.** Fig. 1 and Fig. 2 are consistent with the JSON. Fig. 2 also plots the non-grokked fraction 0.4 run, although the caption speaks of "real-label runs". That is acceptable, but should be stated.

**References.** I spot-checked 9 references by web search. All exist and say roughly what is claimed:

- Balestriero et al. 2021 (arXiv 2110.09485; the affine-subspace dimension statement is correct)
- Yousefzadeh 2021 (2101.09849; NeurIPS 2020 DL-IG workshop)
- Yousefzadeh 2022 (2202.04052; test points are outside the hull but close to it in feature space)
- Bonnasse-Gahot 2022 (2207.08648; most test points are inside the hull in an autoencoder latent space of the last layer, and membership is not the key criterion)
- Hwang & Shin 2024 (2405.16082; "To-hull Uncertainty" for test selection)
- Gromov 2023 (2301.02679)
- Wu et al. NAACL 2024, pp. 1819–1862 (DOI correct)
- Power et al. 2022
- Nanda et al. 2023

I found no fabricated or misattributed citation. One closely relevant paper is missing: Yousefzadeh (2022), "Deep Learning Generalization, Extrapolation, and Over-parameterization" (arXiv 2203.10366), which argues directly that interpolation is inadequate for understanding generalization.

## 3. Strengths

1. **Framing.** The paper names the representation explicitly and separates hull *membership*, *relative* proximity and a decision-linked criterion. This clarifies a debate that is often muddled.
2. **Proposition 2.** It is correct (I checked it line by line; see Section 5), cleanly stated, and a nice pedagogical example: in a natural task, input-space proximity is not merely uninformative but exactly wrong.
3. **Honesty.** The paper is careful about scope. Section 8 is frank: small MLPs, few seeds, the certificate is sufficient but not necessary, I₂ does not lag, and there are no OOD baselines. The superseded preprocessing run is kept and disclosed.
4. **Reproducibility.** The code is small, self-contained and deterministic given the seeds. The paper points every number to a specific JSON field.
5. **Controls.** The random-label control and the data-size diagnostic (Table 4) are sensible.

## 4. Weaknesses

### 4.1 Major

**M1. The "certificate" certifies nothing that is not already observable, and the "interpolation of training decisions" reading is a relabelling.**

The conclusion of Proposition 3 is that "the network predicts y on x". That is read directly off the logits. In the label-free use (Remark 2), the certificate is computed for ŷ = argmax f(x), so its conclusion is tautological. What the condition actually measures is a geometric ratio. Is φ(x) within δ of the same-label hull, with δ smaller than h's distance to the pairwise decision boundaries? This is a robustness-radius or "distance to hull vs margin" statistic.

Calling a decision "provably a convex interpolation of training decisions" when this ratio is below 1, and "not because of convex interpolation" when it is above 1, is a definitional choice. It is not a theorem about mechanism. The class regions of a linear readout are convex and contain the same-label training hull (when training is fit), so every correctly classified test point is already "decided like" some hull point in a weaker sense. The certificate adds a particular, conservative threshold.

The paper's strongest interpretive claims rest on this choice:

- Section 6.1, "The network is right, but not *because* of convex interpolation"
- Section 7.1(c), "generalization that is not interpolation"
- the Abstract, "rule-like, uncertifiable generalization comes first"

Section 8 partially retracts this ("what lags is *certifiable* interpolation under our bound"), but the abstract and title do not.

**M2. The certification lag is not shown to be more than margin and compression growth, which are already known to continue after grokking.**

Fig. 2 shows that C stays near 0 until test accuracy is above about 0.95, and then shoots up. That is the signature of a statistic that needs large normalized margins and small residuals. Both are known to improve during the "cleanup" phase under weight decay: Nanda et al. 2023 describe cleanup, and Papyan et al. 2020 describe neural collapse.

The paper offers no control that separates "certified interpolation" from simpler monotone quantities, such as:

- the fraction of test points whose logit margin exceeds the median training margin, or any fixed threshold;
- test loss;
- within-class variability of φ (the NC1 statistic);
- weight norm.

If a plain margin threshold shows the same lag, the "generalize first, interpolate later" finding reduces to "margins and collapse keep improving after accuracy saturates", which is well known. This control is cheap. Without it, the novelty of C4 is unestablished.

**M3. Statistical support for the dynamics claims is thin.**

- The lag is reported from 4 grokking runs: 3 seeds at one fraction, 1 seed at fraction 0.6. There are no confidence intervals.
- The checkpoint resolution is 500 steps, so each lag value has ±500-step quantization.
- C is computed on a fixed 250-pair subsample, while test accuracy uses the full held-out set.
- The claim that "the lag should shrink as the training fraction grows (4,000 → 2,500)" (P1) rests on one seed at 0.6 versus a 3,000–4,000 range at 0.5.
- No hyperparameters were swept (lr, weight decay, width, depth). It is unknown whether the ordering survives, say, lower weight decay, where cleanup is slower or absent.
- The single non-grokking run (fraction 0.4) gives no information about the lag.

**M4. The Experiment 2 conclusions depend on a weak network and a single synthetic shift.**

- The MLP is trained for 1,500 full-batch steps on 600 images and is *beaten* by pixel-space nearest-class-hull and roughly matched by kNN-type baselines. The claim that "the interpolation depth of this task is 0" is then partly an artefact of an under-trained or under-regularized network. A stronger model (CNN, longer training, augmentation) could change the profile.
- The shift probe (right-shift with zero fill) is one synthetic perturbation. At 2 px, network accuracy is below chance, so AUROC comparisons there rest on 565 correct predictions.
- The AUROC differences in distribution (0.956 vs 0.960 vs 0.975, with 171 errors pooled over 5 seeds) carry no confidence intervals or per-seed spread. They may well be within noise.
- Pooling across seeds mixes score scales across different networks.
- Baselines for error detection omit standard ones: energy, Mahalanobis, kNN in feature space with tuned k, temperature-scaled softmax. The paper acknowledges this.

**M5. Novelty is limited, and the theory is close to folklore.**

- Proposition 1 is a one-line consequence of the definition of extreme points. The vacuity of hull membership for one-hot inputs is immediate and implicit in Balestriero et al.
- Proposition 3 is Cauchy–Schwarz on an affine margin, the same bound used in standard margin- and Lipschitz-robustness certificates.
- The empirical observations already exist in prior work:
  - Representation-space hull proximity tracks accuracy at the last layer (Yousefzadeh 2022; Bonnasse-Gahot 2022).
  - Hull distance can be used as an uncertainty score (Hwang & Shin 2024).
  - Last-layer features compress during late training (neural collapse; grokking cleanup).
- What is new is (i) the specific exactly-zero example of Proposition 2, (ii) the certificate *as a named statistic*, and (iii) its timing in a small grokking MLP. That is an incremental, workshop-level contribution unless M2 is resolved in the paper's favour and the result is shown to hold beyond one toy architecture.

**M6. The headline answer to the research question is nearly tautological for a linear readout.**

"Generalization consists in building a representation in which test points become certifiable interpolations" holds, for any linear-readout network that is accurate with large margins and has collapsed within-class features, almost by construction. The paper should either show a case where a network generalizes well while C stays low at convergence (which would make C informative), or tone down the claim that this answers the title question. Notably, the paper's own evidence shows I₂ (relative interpolation) tracking accuracy *exactly*, without lag. That is equally consistent with "the last layer is linearly separable and roughly clustered because the network generalizes" (correlation, not mechanism).

### 4.2 Minor

- m1. Section 4.1, "every language model at its input layer": true for one-hot tokens, but the first *learned* layer (embeddings of sequences) is not generally in convex position. Say explicitly that the claim is about the one-hot input only.
- m2. I₀ = 0 "at every one of 25 checkpoints" is padding. Input-space classifiers do not depend on the network, so I₀ is constant over training by construction.
- m3. "Visible fraction" of ‖r‖² (Table 4) needs a null baseline. A random residual would have visible fraction about rank/d: 52/128 ≈ 0.41 for modular addition and 9/64 ≈ 0.14 for digits. Without that baseline, 0.53–0.94 cannot be interpreted.
- m4. Table 4 reports medians only. Pattern 4 ("in every setting the margin at φ(x) is smaller") is a statement about medians and should say so.
- m5. AdamW weight decay is applied to biases too. This is harmless but should be stated.
- m6. Inconsistent figures: 78.5% vs 78.6%, and the I₂ crossing "same checkpoint in all four runs" vs 3,500 vs 3,000 for fraction 0.6.
- m7. The normalization r_ℓ (RMS radius) is a convention, and layer comparisons of d_ℓ across widths 106/128/128 are not obviously meaningful. Consider a scale-free alternative (e.g. rank-based).
- m8. Prop. 2: the probability that the condition fails is better given with the exact hypergeometric expression than as "≈ 2^{−104}". This is cosmetic.
- m9. The Chinese abstract and the English abstract are consistent. Good.

## 5. Proof check

- **Prop. 1.** Correct and trivial.
- **Prop. 2.**
  - Step 1 is correct: same-label points share neither coordinate.
  - Step 2 is correct: h has zeros at a and p + b. Each block is a probability vector over at most n_c coordinates (the a′ values are distinct within a class), so ‖h₁‖² ≥ 1/n_c. Disjoint supports give ‖e − h‖² = 2 + ‖h₁‖² + ‖h₂‖² ≥ 2 + 2/n_c.
  - Step 3 is correct: the neighbour (a, b′) is at squared distance 2.
  - The conclusion holds for both 1-NN and nearest-class-hull, and there are no ties. Correct.
- **Prop. 3.** Correct (affine margin plus Cauchy–Schwarz). The implementation is sound. The NNLS solution is renormalized onto the simplex, so h is a valid hull point and the bias terms are included. A sub-optimal NNLS solution only makes the certificate conservative, as claimed.

## 6. Questions to the authors

1. Does a plain margin-threshold statistic (fraction of test points with margin > τ, for τ calibrated on training margins) show the same lag as C? Does NC1 or test loss?
2. Can you exhibit a network that generalizes well with *low* C at convergence? Without such a case, how is C informative beyond accuracy plus margin?
3. How does the lag depend on weight decay (e.g. 0.1, 0.3, 1.0, 3.0), on width, and on a transformer? Even one transformer run (P1) would greatly strengthen the paper.
4. What are the per-seed AUROCs and bootstrap CIs for Table 3?
5. Does the "interpolation depth 0" on digits survive a stronger network (CNN or longer training) that beats pixel nearest-class-hull?

## 7. Assessment against the research question

The sub-question is well posed: "In which representation, and when during training, do test points become interpolations of training points, in a decision-relevant sense?" The negative half of the answer is convincing but close to obvious: input-space hull interpolation is vacuous or misleading. The positive half is not established at the strength claimed: generalization is "built-then-certified interpolation" in a learned space. The key mechanistic claim ("generalization that is not interpolation", then "consolidation into interpolation") rests on a conservative threshold statistic with no control against simpler margin dynamics, on 4 runs of one toy architecture.

## 8. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 4 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 8 |
| **Overall** | **5** (weak reject; workshop-level in current form) |

**Decision: major revision (weak reject for the main track).**

The numbers are reproducible and the citations are genuine; I found no fatal issue. But the central claims are over-interpreted (M1, M6), the key dynamic finding lacks an essential control (M2), the statistics are thin (M3), and the theoretical contribution is close to folklore (M5).

## 9. Reproduction log

REPRO_PLACEHOLDER
