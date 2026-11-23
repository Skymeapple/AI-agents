# Response to Round-1 Review (p005)

We thank the reviewer for a careful report, including the full reproduction. We ran the experiments the reviewer asked for. They largely **confirmed** M1, M2 and M6, and we have rewritten the paper's central claim accordingly.

The round-1 headline was that grokking networks "generalize first (uncertifiably) and interpolate later". It is withdrawn. A single fixed logit-margin threshold reproduces the certification lag in every grokked run. The paper now reports this as a margin phenomenon.

What survives:

- the exact anti-interpolation result (Prop. 2);
- the layer-wise profile;
- a new invariance result for the certificate (Prop. 3′/4);
- per-point evidence, with per-seed CIs and standard baselines, that hull geometry adds modest but significant information beyond the logit margin.

**Provenance.** All new numbers come from code written and run in this environment:

| scripts | outputs |
|---|---|
| `code/exp_controls.py`, `exp_transformer.py`, `analyze_controls.py` | `results/controls_*.json`, `controls_summary.json` |
| `code/exp_digits2.py`, `analyze_digits2.py` | `results/digits2.json`, `digits2_summary.json` |
| `make_figures2.py` | `fig4_controls.png`, `fig5_digits_auroc.png` |

Shared helper: `code/ckpt_metrics.py`.

**Citations.** All new references were verified by web search:

- Yousefzadeh 2022b (arXiv:2203.10366)
- Hendrycks & Gimpel 2017 (1610.02136)
- Liu et al. 2020 (2010.03759)
- Lee et al. 2018 (1807.03888)
- Sun et al. 2022 (2204.06507)
- Lyu & Li 2020 (1906.05890)
- Lyu et al. 2024 (2311.18817)

Chollet (2019) was dropped together with the sentence that cited it.

---

## Major issues

### M1. The certificate's conclusion is observable or tautological; "interpolation of training decisions" is a relabelling

**We agree.** Changes:

- **Prop. 3 is recast as a standard bound.** It is now labelled "hull-margin bound; standard" (Sec. 4.3). A new paragraph, "What (2) does and does not say", states that the conclusion can be read off the logits and is tautological with ŷ. Its content is a geometric ratio (distance to the same-label hull against the hull point's normalized margin). It is a *definition* of decision-robust hull proximity, not a theorem about mechanism.
- **Round-1 wording removed.** "The network is right, but not *because* of convex interpolation" and "generalization that is not interpolation" are gone from the abstract, Sec. 6, Sec. 7.1(c) and the conclusion. The title no longer contains "The Order in Which Grokking Networks Generalize and Then Interpolate".
- **New Prop. 3′ (readout-subspace certificate C_vis).** The certificate stays sound after projecting onto span{w_c − w_1}, and it is sharper there (δ_vis ≤ δ). Exact C_vis membership is *equivalent* to the test logit-difference vector being a convex combination of same-label training logit-difference vectors (proof: an injective affine map preserves hull membership). This gives a literal, non-tautological meaning to "interpolation of training outputs" for points that are actually in the hull. We state that this is still a definition.
- **New Prop. 4 plus an experiment: C is not invariant to readout-invisible directions.** We appended 256 zero-readout ReLU units to the digits MLP. The maximum logit change was < 10⁻¹³. C fell from 0.876 to 0.855, 0.559 and 0.001 as the padding scale grew (α = 1, 4, 16; 5 seeds), while C_vis stayed exactly 0.915 (Table 3). This sharpens the reviewer's point: full-space C partly measures directions the decision ignores. We now recommend C_vis.

### M2. The lag is not separated from margin, test loss, NC1 or weight norm

**We ran the control. The reviewer's suspicion is confirmed.**

Experiment 4 (new; Sec. 5 and 6.2; Table 2; Fig. 4) has 17 MLP runs and checkpoints every 250 steps. It tracks:

- C and C_vis on 700 held-out pairs;
- fixed-threshold margin fractions M_abs(τ) for τ = 2, 5, 10;
- training-quantile margin fractions M_q10 and M_q50;
- test and train loss, NC1 (train and test) and weight norm;
- per-point margins and certification every 500 steps.

Results:

- **A fixed logit-margin threshold reproduces the lag.** One threshold, τ = 8.0, was fitted jointly on all MLP runs. Its 50% crossing equals C's, or precedes it by exactly one 500-step checkpoint, in **all 14 grokked MLP runs** (lag correlation r = 0.97; mean absolute difference 286 steps). In the 3 grokked transformers, τ = 6.25 matches all three crossings exactly. Every M_abs(τ) lags accuracy, increasingly with τ (650, 1,800 and 4,625 steps for τ = 2, 5 and 10 at fraction 0.5).
- **Per point, C is close to but not identical to a margin threshold.** Margin predicts certification with AUROC 0.86–0.91, and the best single-threshold rule agrees on 80–87% of points.
- **Training-relative margin thresholds never reach 0.5.** Training margins stay larger than test margins.
- **NC1 is not the explanation in our setting.** Test NC1 does not fall after grokking: its half-decline occurs during memorization, 5,500–7,000 steps before test accuracy crosses 50%.
- **Test loss** falls below 0.1 between the accuracy crossing and the certificate crossing.
- **Weight norm** is recorded in the per-run JSON. It is not monotone: it rises to a peak during memorization (about 65–76 at steps 1,250–1,750 for weight decay ≤ 1) and then declines slowly, to about 49–59 at the end. It is a smooth global quantity with no feature at the certificate crossing, so we make no claim that it explains the lag.

Changes to the paper:

- Sec. 6.2 is titled "The certification lag is a margin phenomenon".
- C4 and the abstract now say exactly this.
- The "progress measure" use case is removed (Sec. 7.2: "Not recommended").
- Related work now cites late-phase margin maximisation (Lyu & Li 2020; Lyu et al. 2024).

### M3. Thin statistics

Changes:

- **Seeds.** Fraction 0.5 now has 5 seeds at weight decay 1 (round 1 had 3). Fractions 0.6 and 0.7 have 3 seeds each (round 1: 1 and 0).
- **Weight-decay sweep.** We added weight decay 3 (3 seeds) and 0.3 (3 seeds, 24,000 steps).
- **Transformer.** We added a one-layer transformer following Nanda et al. (2023): 3 seeds at fraction 0.5, plus 3 at fraction 0.3, which did not grok within 6,000 steps and are reported as such.
- **Resolution and sample size.** Checkpoints are every 250 steps, and C uses 700 rather than 250 pairs.
- **Crossing definition.** Crossings are now *sustained* (the statistic stays above the threshold until the end of training), so early transients cannot produce spurious crossings.
- **Confidence intervals.** All lags carry 95% t-CIs over seeds (Table 2).
- **The fraction trend is now supported by 3–5 seeds per condition.** C's lag is 3,450 [2,940, 3,960] at fraction 0.5, 2,250 [1,629, 2,871] at 0.6 and 1,583 [1,225, 1,942] at 0.7. The intervals for 0.5 and 0.7 do not overlap. The margin thresholds show the same trend, which is consistent with M2.
- **Weight decay.** At weight decay 3, the lag is 2,833 [837, 4,830], so we say only that it "tends to be shorter". At weight decay 0.3, two of three runs grok only after about 20,000 steps and end at 80–85% accuracy with C ≤ 0.004.
- **Remaining limits** are stated in Sec. 8: 3 seeds give wide CIs, and the weight-decay-0.3 runs did not converge.

### M4. Weak digits MLP, single synthetic shift, no CIs, missing baselines

Changes (Experiment 6, Tables 4–5, Fig. 5):

- **Stronger network.** We added a small CNN, which reaches 0.977 ± 0.005. It still does not beat pixel-space nearest-class-hull (0.984).
- **Rebuttal on "depth 0".** Because I₀ = 0.984 and no classifier exceeds 1.0, depth 0 under ε = 0.02 holds for *any* network on this split. So it is not an artefact of the weak MLP, but it is also nearly content-free. We now say so explicitly. The substantive statement is only that pixel-space relative interpolation already nearly solves this easy task.
- **Correction.** The round-1 remark that "the learned representation extrapolates worse than input interpolation" holds for the MLP but not for the CNN (right 1 px: 0.597 vs I₀ 0.559). We corrected it.
- **Six perturbations instead of one:** none, right 1, right 2, down 1, Gaussian noise (σ = 0.25 of the range) and 15° rotation.
- **Per-seed statistics.** AUROCs are now reported as per-seed mean ± s.d. with 95% t-CIs. Pooling across seeds is no longer used for inference.
- **The in-distribution differences are significant per seed.** For hull contrast minus logit margin, the paired difference is +0.020 ± 0.006 (p = 0.0015) for the MLP and +0.021 ± 0.005 (p < 0.001) for the CNN.
- **Baselines added:** logit margin; energy (Liu et al. 2020); Mahalanobis with shared covariance (Lee et al. 2018); deep-kNN on L2-normalized features with k = 1 and k = 10 (Sun et al. 2022); and max-softmax (Hendrycks & Gimpel 2017). Hull contrast has the highest AUROC among non-certificate scores in 11/12 model×perturbation conditions. The exception is MLP / down 1, where kNN (k = 1) is 0.759 against 0.756.
- **New incremental test.** We fit a leave-one-seed-out logistic regression on the logit margin, with and without hull contrast. Adding hull contrast improves held-out-seed AUROC in all 12 conditions (gain +0.019 to +0.205; p < 0.05 in 12/12). Adding the C_vis slack helps less (significant in 10/12).

### M5. Limited novelty; theory close to folklore; missing Yousefzadeh 2022 (2203.10366)

We agree in substance. Changes:

- **Prop. 1** is labelled "elementary" and described as implicit in Balestriero et al. (2021).
- **Prop. 3** is labelled "standard". We claim no novelty for the inequality.
- **Prior empirical findings are credited explicitly.** A sentence in Related Work disclaims hull proximity tracking accuracy (Yousefzadeh 2022a; Bonnasse-Gahot 2022), hull-distance uncertainty (Hwang & Shin 2024) and late-phase compression or margin growth.
- **Yousefzadeh (2022b)**, arXiv:2203.10366 (verified), is cited in the introduction and in Related Work.
- **The remaining contributions are stated plainly:**
  - Prop. 2 (exact anti-interpolation), with an exact failure probability;
  - Prop. 3′/4 (the readout-subspace certificate, its logit-hull interpretation, and the non-invariance of full-space C);
  - the controls reducing the lag to the margin, which is a negative result that we think is useful to report;
  - the per-point incremental-information evidence against standard baselines.

  We do not call any of this a breakthrough.

### M6. The headline is near-tautological for a linear readout; show a well-generalizing network with low C

We addressed this in three ways and report the outcome honestly.

1. **Natural low-C networks exist, but they have small margins.** With weight decay 0.3, two runs reach 80% and 85% test accuracy with C ≤ 0.004 and C_vis ≤ 0.011. Their median test margins are 3.7 and 4.5, so the margin explains the low C. We found *no* natural network with high accuracy, large margins and low C_vis. We say this explicitly in Sec. 6.4.
2. **A function-identical network with C ≈ 0 exists** (the padding construction, Prop. 4). This shows that C is not a function of accuracy plus margin. It does so for a reason that counts *against* C, and is why we introduce C_vis.
3. **Per-point information beyond the margin** (Table 4, leave-one-seed-out). This is the evidence that hull geometry is not merely a restatement of the margin. The effect is modest in distribution (+0.02 AUROC) and larger under shift.

The headline answer to the research question is accordingly toned down (abstract, Sec. 7.1, Sec. 9):

- input-space interpolation is vacuous or exactly misleading for discrete tasks;
- last-layer relative interpolation tracks accuracy, which is correlational, as the reviewer notes;
- certified interpolation is, to first order, logit-margin growth;
- hull geometry adds modest per-point information beyond the margin.

The I₂-tracks-accuracy observation is now explicitly described as "correlational … not a mechanism" (Sec. 6.1, Sec. 7.1(b)).

---

## Minor issues

| # | Issue | Change |
|---|---|---|
| m1 | "Every language model at its input layer" | Prop. 1 is now stated to concern the *one-hot input* only. "Learned embedding layers are generally not in convex position, so it says nothing about them" (Sec. 4.1, C1). |
| m2 | I₀ = 0 at 25 checkpoints is padding | Now stated as "constant over training by construction" (Sec. 4.2, Sec. 6.1). |
| m3 | Visible fraction needs a null baseline | Table 6 now gives the analytic null rank/d (0.41 for modular addition, 0.14 for digits). Observed fractions are 2–2.3× and 3.8–5.4× the null. |
| m4 | Table 4 reports medians only | The table (now Table 6) header says "medians over test points", and the pattern is stated as "a statement about medians". Experiments 4 and 6 report means ± s.d. over seeds. |
| m5 | Weight decay is also applied to biases | Stated in Sec. 5. |
| m6 | 78.5 vs 78.6; I₂ crossing "same checkpoint in all four" | 0.785 is now labelled the mean over seeds and 0.786 the pooled value. The I₂ crossing now reads "the same checkpoint in the three fraction-0.5 runs; one checkpoint later for fraction 0.6 (3,500 vs 3,000)". |
| m7 | r_ℓ is a convention | Sec. 3 and Sec. 8 say that layer comparisons rest on the scale-free I_ℓ (argmin is invariant to r_ℓ), and that raw d values across widths should not be compared quantitatively. |
| m8 | Exact hypergeometric probability | Sec. 4.2 now gives C(p² − 1 − 2(p − 1), n_tr)/C(p² − 1, n_tr): 6.8·10⁻³³ at a 50% split (≈ 2^−106.9, not ≈ 2^−104 as round 1 said), 2.3·10⁻²⁴ at 40% and 2.0·10⁻⁴³ at 60%. The expected number of violating held-out pairs at 50% is 9.6·10⁻³⁰. Computed with `math.comb`. |
| — | Fig. 2 includes the non-grokked fraction-0.4 run | The figure list now says so explicitly. |

## Reviewer's six questions

1. **Does a margin threshold show the same lag? Does NC1 or test loss?** Margin: yes (M2). NC1: no, it does not decline after grokking. Test loss: it falls between the two crossings.
2. **Can you exhibit a well-generalizing network with low C?** Yes, but only with small margins (weight decay 0.3), or artificially by padding. See M6.
3. **How does the lag depend on weight decay, width and architecture?** Weight decay: the lag tends to be shorter at weight decay 3, and there is no certification within 24k steps at 0.3. Transformer: a short lag (917 steps on average), also matched by a margin threshold. **Width was not swept**; this is stated in the limitations.
4. **Per-seed AUROCs and CIs?** Provided in Table 4 and `digits2_summary.json`.
5. **Does depth 0 survive a stronger network?** Yes, trivially, because I₀ = 0.984. See M4.

## Not done, and stated as limitations

- No width sweep.
- No deeper transformers or language models.
- Lags at fraction 0.3 for the transformer are unknown (not grokked within budget).
- The weight-decay-0.3 runs did not converge.
- The margin threshold τ was fitted post hoc. It is one shared parameter, but it was evaluated on the same runs.
