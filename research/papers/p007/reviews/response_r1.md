# Response to Reviewer (round 1)

Thank you for a careful review. You re-ran every script and checked the numbers and the proofs, and that work exposed real problems. Below we answer each point: what we changed, where, and with which new evidence. All new numbers come from scripts we ran in this environment, with outputs in `results/`. New scripts: `prep_tokens.py`, `exp_transformer.py`, `analyze_transformer.py`, `exp_random_design.py`, `exp_threshold_fine.py`, `exp_mlp2.py`, `analyze_mlp2.py`, `analyze_r2.py`. Modified: `exp_crossover.py`.

Summary of the main changes:
- **Real-data evidence.** Small decoder-only transformers trained on real image (MNIST), audio (Free Spoken Digit Dataset, VQ tokens) and text (Tiny Shakespeare) token streams, at 4 sizes and 2 per-modality budgets with unimodal baselines, plus a modality-addition sweep (new Sec. 5.8).
- **The ReLU contradiction.** Resolved by identifying a convention mismatch. We concede that our round-1 explanation was wrong. ReLU study re-run.
- **Error reporting.** Errors are now reported across all experiments, including the 5–11% errors at intermediate $n$ under power-law spectra, with an explanation.
- **$n<d$.** Restated as an SNR model, plus new random-design experiments.
- **Threshold claim.** Redone on a fine grid against asymptotic spiked-matrix predictions, with CIs and a finite-size check.
- **Novelty and related work.** Rewritten, with corrections and 7 new verified references.

---

## Major issues

### M1. The question is answered only inside a linear toy model; no real data; Section 7 not executed.

**Agreed, and partly addressed.**
1. *New real-data experiment (Sec. 5.8, `exp_transformer.py`).* Decoder-only transformers (2 layers, $d_{\rm model}\in\{16,32,64,128\}$, 12k–442k parameters) trained on four real token streams that share one 144-token vocabulary: MNIST image tokens, FSDD audio tokens (log band energies, 64-code k-means VQ), Tiny Shakespeare characters, and Fashion-MNIST. The grid covers 4 sizes × 2 per-modality budgets ($D$ = 25k and 100k tokens), with a unified image+audio model and both unimodal specialists in each cell. A P4 sweep adds text and fashion at fixed size. Unified models and specialists see identical per-modality tokens and steps. We report test cross-entropy at each modality's own best-validation checkpoint, which removes the confound of modalities overfitting at different rates, and also at the joint checkpoint. Results and which predictions they support or contradict: [[R-TF]]
2. *Scope claims reduced.* The title now says "Model … with Small-Scale Tests". The abstract, the Introduction ("Scope and claim") and Section 6 say explicitly that the derived limits are limits of the model. Section 6 states which predictions have been tested at which level of realism.
3. *What we could not do.* We cannot run the Section 7 protocol at 20M–1B parameters on CPU. We did not measure $\alpha$, $\phi$ or the irreducible loss for the transformer data, so $R$ and $\kappa$ are **not** compared with any measured quantity, and P2, P3 and P5 remain untested on real data. Section 8 says this explicitly. The protocol now also requires estimating the irreducible loss, so that excess-loss ratios can be formed, and reporting both fixed-size and best-size comparisons.

### M2. The ReLU experiment contradicts Corollary 2; the explanation points the wrong way.

**You are right that our explanation was wrong, and we withdraw it.** Investigating it turned up the actual cause: a **convention mismatch** that the round-1 paper did not notice.
- Corollary 2 bounds the ratio at the **oracle width** (each model uses its best width $\rho\le r$). The round-1 ReLU table reported **fixed-width** ratios $L_u(h)/L_s(h)$. At large $h$ a width-$h$ specialist for a 12-feature teacher degrades, and the unified network degrades less.
- Re-analysing the *round-1 data* under the oracle-width convention (`analyze_r2.py` → `analysis_r2.json`, key `corollary2_vs_mlp_r1data`), the ratios at the largest width are 0.84 / 1.13 / 1.68 ($M=2/4/8$) at $n=1024$ and 1.03 / 1.03 / 1.49 at $n=4096$. **None is below the bound $1-\phi(1-1/M)$ = 0.75 / 0.625 / 0.56, so there is no contradiction with Cor. 2.** The ReLU networks in fact show *less* oracle-width synergy than the df value. We report that as a quantitative miss of the df model, not as support.
- The low fixed-width ratios also have a model-based explanation. New **Remark 2** gives the fixed-rank version of the df heuristic. For the Exp. A configuration at $n=1000$ it gives 0.53 ($r=20$) and 0.61 ($r=24$), against an oracle-width $\Gamma=0.94$. That is the same order as the ReLU fixed-width ratios (0.49–0.61). Fixed-width synergy is a separate mechanism: shared width wastes less on noise directions.
- Your $\Gamma$ values for the MLP configuration (0.80, 0.76) come from Cor. 2's formula. We note that the linear analogue is only formal there, because $q=8<k=12$. The comparison with Exp. A's $\Gamma=0.94$ was wrong and has been removed.
- Re-run (`exp_mlp2.py`): $d=64$ for all $M$ (so 54 ≤ 64), exactly orthonormal teacher directions, [[R-MLP2-SEEDS]] seeds, two learning rates selected per model, width and modality, and both conventions reported with bootstrap CIs. Result: [[R-MLP2]]
- Section 6, item 2 now says that the ReLU networks respect the oracle-width bound, and that fixed-size comparisons, which is how published comparisons are made, include an additional synergy mechanism.

### M3. The abstract's "within 2.3% for $n\ge1000$" holds only for Exp. A.

**Agreed; fixed.** A new summary table in Section 5 lists max and median errors for every experiment and every $n$ (computed by `analyze_r2.py`, metric $|{\rm sim}-{\rm th}|/{\rm th}$ throughout, which is why some values differ slightly from yours). At $n=1000$: A 2.3%, B 3.9%, C 7.0%. E2: 5.5%, 5.5%, 5.9%, 6.6% and 11.3% at $n$ = 500, 2000, 8000, 32,000 and 128,000, and 0.65% at $10^7$. The E2 text now reports all six $n$, including the plateau mismatch at $n=1.28\cdot10^5$ (0.89 simulated against 0.80 predicted). We checked that the mismatch is not an artefact of E2's coarse rank grid: `exp_crossover.py` uses the full rank grid in the same configuration and gets a plateau of 0.891. Our explanation, now in the paper: a power-law spectrum always has retained directions just above the noise edge, where first-order perturbation overstates recovery. The abstract now quotes these ranges.

### M4. $X^\top X=nI$ requires $n\ge d$; Exp. D and Exp. G use $n<d$.

**Agreed; fixed in two ways.**
1. Section 4.2 now states that no orthogonal design exists for $n<d$. In that range the Gaussian-perturbation model is an **SNR model** with $n_{\rm eff}$ = inverse noise variance. Exp. D and the threshold study are labelled accordingly.
2. *New random-design experiment* (Sec. 5.4, `exp_random_design.py`). Gaussian $X_m$ with real $n\in\{100,\dots,3000\}$ ($n>d+1$); specialists by exact reduced-rank regression, the unified model by alternating least squares. With the substitution $n\to n-d-1$ (the mean OLS noise covariance), the window, the peak at $r=8$ and $\Gamma$ survive for $n\ge500$, with errors of at most 4.6%. Without the correction the prediction is clearly worse (3.02 against a simulated 2.74 at $n=500$, $r=8$). At $n=200$ the low-SNR failure reappears (1.03 against 0.94). At $n=100$ specialists recover nothing and the unified model wins (0.86), as Remark 4 predicts.

### M5. Weak support for the threshold shift (coarse grid, onsets 3× above BBP, $\sqrt M$ = ratio of adjacent grid points, private shift misses).

**Agreed; experiment replaced** (`exp_threshold_fine.py`, Sec. 5.6). Changes:
- 40 log-spaced SNR values and 40 repetitions. The statistic is the SNR at which a fixed fraction of the signal is recovered, log-interpolated, with bootstrap CIs.
- Predictions now come from the asymptotic spiked-matrix formulas for oracle hard truncation [Benaych-Georges & Nadakuditi 2012; Gavish & Donoho 2014, reference verified]. These predict a shift of 2.27–2.32 for shared features and 1.72–1.76 for private ones at these aspect ratios, not exactly $\sqrt M$. Remark 4 now distinguishes detection (which shifts by $\sqrt M$) from useful recovery.
- Results: at $d=64$, 2.06 [2.03, 2.09] to 2.16 for shared and 1.37 [1.35, 1.38] for private. At $d=256$ (finite-size check), 2.23–2.27 against 2.27–2.32, and 1.64–1.68 against 1.72–1.76. The private shortfall at $d=64$ is a finite-size effect: 32 spikes fill half of a $64\times64$ matrix. It shrinks as $d$ grows.
- The "3× above BBP" observation is explained: useful recovery under hard truncation needs more SNR than detection. Asymptotically, the specialist's loss falls to 90% of the signal at $n_{\rm eff}=118$, against the detection threshold of 32. The simulated value is 96.
- The old "2× = √M" sentence is removed.

### M6. Tautological or weak validations (E1, E2 "close to R", 140/140, crossover slope without CI).

**Agreed; each fixed.**
- **E1** is now labelled "numerical check of the continuum step only … it verifies the integral approximation, not the model."
- **E2**: the sentence "close to $R=2.25$" is deleted. The paper now says that the finite-feature population ratio rises monotonically from 1.6 to 4.64 and is not flat, because flatness needs many more features.
- **140/140**: we now report that only 15 cells are near the boundary ($|\log{\rm ratio}|<0.1$) and that the sign agrees in all 15. For each of the 14 capacities, the first grid angle with competition is the same in simulation and theory. We also note that the 10° grid limits this, so the fine $\theta_c$ sweep is what locates the boundary (within 0.1° for $n\ge3000$).
- **Crossover slope**: `exp_crossover.py` now uses 10 repetitions (was 3), filters capped rows in the script, and reports a bootstrap CI and a grid-resolution band. For $\alpha=2$: 0.567, CI [0.567, 0.571], grid band 0.567–0.618; df 0.577; asymptotic 0.5. We say explicitly that the step-2 grid, not the repetition noise, dominates the uncertainty, and that the slope is 13–20% above $1/\alpha$ (pre-asymptotic).

### M7. Novelty moderate; related work mis-characterised or missing.

**Agreed on both counts.**
- A new paragraph, "Novelty, stated plainly" (Sec. 1), says that Prop. 1 is Eckart–Young, that the finite-data analysis is the textbook RRR df count, that Remark 4 is a BBP / Gavish–Donoho application, and that Prop. 3's scale invariance comes from power-law self-similarity. It also says what we believe is new: the closed-form tax $R$, $\kappa$; $\sin^2\theta_c\propto1/n$; head interference as a separate cost; and the multi-level tests. Prop. 2 is renamed **Heuristic 1** (see m2).
- **Wu et al. 2020**: corrected. They *do* analyse shared-module capacity, and the paper now says so and states that round 1 was wrong.
- **Added, each verified by web search**: Ildiz, Zhao & Oymak (AISTATS 2024, PMLR 238:4726–4734); Bahri et al. (PNAS 2024, arXiv:2102.06701); Maloney, Roberts & Sully (arXiv:2210.16859); Paquette et al. (NeurIPS 2024, arXiv:2405.15074); Fernandes et al. (ICML 2023, PMLR 202:10053–10071); ATLAS, Longpre et al. (arXiv:2510.22037, ICLR 2026); Gavish & Donoho (IEEE TIT 2014). We point out that Fernandes et al.'s finding, that mixture weights change only the multiplicative constant of the scaling law, is an empirical analogue of Prop. 3's scale-free ratio. We did not add "He et al. 2024/25", because we could not pin down the exact reference.
- **Shukor et al. (learned beats modality-aware routing)**: verified and now reported in the Introduction and Section 6. P6 is restated in terms of modality-specific *capacity*, however it is allocated. We state that the model is silent on learned versus hand-assigned routing (it has no within-modality heterogeneity), and we no longer imply that hand-assigned routing is preferable.

### M8. Thin statistics in the nonlinear study; M=8 teacher violates orthogonality.

**Addressed** by the round-2 ReLU study (`exp_mlp2.py`, `analyze_mlp2.py`):
- $d=64$ for all $M$, with orthonormal teacher directions from one QR, so the $M=8$ teacher (54 directions) satisfies the model's assumption.
- [[R-MLP2-SEEDS]] seeds instead of 3, and two learning rates selected per arm on validation.
- The fragile "first width below 1" statistic is replaced by "the smallest width after the peak from which the ratio stays below 1 at every larger width", with bootstrap CIs over seeds.
- Both conventions are reported.
- Transformer study: 2 seeds per cell (fewer than we would like, because of CPU time). Seed-to-seed standard errors are reported where both seeds finished.

---

## Minor issues

- **m1 (Cor. 4 wording).** Restated as a first-order result under the stated assumption. The text now says the core *can* hold some gap directions (10 of 12 at $r=30$), and that the df cost, not capacity, makes them unprofitable at small $\theta$.
- **m2 (Prop. 2 is a heuristic).** Renamed **Heuristic 1 (df approximation; not a theorem)** throughout. Corollaries 2–4 are explicitly conditional on it.
- **m3 (E3 slopes not in JSON).** Fixed. The cap filter is now in `exp_crossover.py`, and the JSON stores `n_points_used`, the slopes, the CI and $r_\times/\rho_u^*$. The re-run with 10 repetitions happens to give 0.567 / 0.577 again for $\alpha=2$; for $\alpha=1.5$ it gives 0.750 / 0.782 (round 1: 0.74 / 0.78).
- **m4 (ratio definitions).** Both definitions are now stated in Section 5. `analyze_r2.py` shows that they differ by at most 0.09% in Exp. A.
- **m5 (window convention).** Stated in the Section 5 preamble: code (a, b+1) ↔ table "a–b".
- **m6 (finite-sample synergy below $k_s$).** New **Remark 1**: the df heuristic predicts it, and the predictions match (Exp. A, $n=200$: 0.92 / 0.88 / 0.83 simulated against 0.93 / 0.89 / 0.84). The abstract now says "at the population level" when it states that the interaction is zero below $k_s$.
- **m7 (competition at $h\le k_s$ reaches ~2.0).** The round-2 table reports the ratio at $h=k_s$ explicitly: [[R-M7]]

## Answers to the reviewer's questions

1. *df prediction for the MLP configuration and reconciliation with Cor. 2.* See M2: the observed ratios were fixed-width, and under the oracle-width convention they are above the bound.
2. *Random designs and a fine $n$-grid with CIs.* Done (M4, M5).
3. *Smallest instance of the Section 7 protocol.* Done at a much smaller scale than the protocol specifies (M1). P1 and P4 are tested; P2 is not, because irreducible losses were not estimated.
4. *Sensitivity to oracle rank selection.* Remark 2 and the fixed-width ReLU/transformer comparisons show that it matters a great deal, in both directions. The conclusions now name the convention they rely on. We did not implement validation-selected rank in the linear model. The transformer and ReLU studies use validation-based early stopping, but they compare fixed widths.
5. *P6 vs Shukor et al.* See M7.
