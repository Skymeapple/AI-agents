# Review (Round 2): "Is Pretraining Scaling Slowing Down? Mechanical Deceleration, Noise-Limited Detectability, and How to Tell a Data Wall from Other Breaks"

Reviewer role: independent area-chair-level reviewer. Research question: 预训练的扩展定律是否正在放缓？下一个增长点在哪里？

## 1. Summary

The revision repositions the paper as a measurement paper with four contributions.

1. **Definitions.** It separates mechanical deceleration, which is built into $L^*=E+GC^{-\gamma}$, from a genuine slowdown, $\delta>0$ in a nested break model. The paper now states that these definitional facts are textbook.
2. **Calibrated re-analysis of the Epoch reconstruction of the Chinchilla runs (240 points).** The break test uses a wide breakpoint grid, parametric and wild-cluster calibration, and cluster bootstraps. A two-break model and a noise decomposition are added. The constant-exponent law is rejected ($p=1/301$), but the departure is non-monotone and sits at the low-compute end. For the top segment ($\ge 2.8\times10^{20}$ FLOP), the exponent change is $\hat\delta_2=-0.009$, 95% CI $[-0.075, 0.015]$. The v0 claim "no slowdown up to $10^{22}$" is withdrawn.
3. **A noise-resolved detectability map for the exact real-data procedure (Exp. 8).**
   - With independent noise $\sigma_w\le3\times10^{-4}$, $\delta=0.02$ is detected with power 1.00 even for a window ending at $10^{28}$ FLOP.
   - At $\sigma_w=10^{-3}$ the power is 0.22.
   - A per-budget shared offset of $10^{-3}$ cuts the power to 0.10–0.13.
   - The v0 claim that "loss curves cannot settle it" is withdrawn.
4. **Mechanisms and the data wall.**
   - Three stylised break mechanisms are compared: a unique-data cap, a decline in marginal-token quality, and a non-data exponent break. Matched data-doubling gaps separate them.
   - Self-similarity is generalised to $\alpha\neq\beta$ (Prop. 3′, $\tau=(\alpha+\beta)/\alpha$).
   - The data-wall milestones are recomputed under the 2026 Lovelace et al. laws. The 50%-efficiency point spans 28× to 7,822× $C_{\rm wall}$ across functional forms.

## 2. Reproduction and verification

**Rerun.** I copied `code/` and `data/` to a scratch directory and reran Exps. 1, 2, 4, 5, 9, 9b, 10, 11 and 12 with the documented single-thread BLAS settings. I also reran the first cells of Exp. 8, which are seeded per cell, so a cell-by-cell comparison is valid.

RERUN_RESULTS_PLACEHOLDER

**Derivations checked.**

- **Prop. 1 and Lemma 1.** Correct, and now correctly labelled as standard.
- **Prop. 2.** The identity $\ln L_\delta-\ln L_0=\ln(1+\rho[(C/C_b)^\delta-1])$ is exact. The SNR scaling is properly called a heuristic.
- **Prop. 3′.** I checked the proof line by line. Substituting $N=U^{\beta/\alpha}n$ and $D=Ud$ gives $c\,U_D^{\beta/\alpha}=U^{\beta/\alpha}c\min(d,1)^{\beta/\alpha}$, so $R_N$ depends only on $(n,d)$. Hence $N'=U^{\beta/\alpha}\phi_N$ and $AN'^{-\alpha}+BD'^{-\beta}=U^{-\beta}[\cdot]$. With $C=6U^{\tau}nd$, the result follows. The constant $c=(\alpha A/\beta B)^{1/\alpha}$ is the correct Chinchilla optimality relation $N=cD^{\beta/\alpha}$. The proof is correct.
- **A small inconsistency.** The text says that for $\alpha=\beta$ the consistent rule "reduces to" the linear $k_N$ rule. The code, however, hard-codes $k_N=0.051$, whereas $c=(521/1488)^{1/0.35}\approx0.050$. That is why Table 4 / Exp. 4 gives 75× (Exp. 10 gives 74×) while the "consistent" row of Table 7 gives 71× for the same law. The difference is harmless, but it should be explained.
- **Corollary 1.** Correct.

**References.** I checked 13 references by web search. All exist and support the claims attributed to them:

- Lovelace et al. 2026 (arXiv:2605.01640): additive overfitting penalty; 1-, 2- and 4-parameter ladder; strong weight decay ($\lambda=1.0$) cuts the overfitting coefficient by about 70%; code at github.com/justinlovelace/data-constrained-scaling.
- Bryant & Liu 2026 (arXiv:2605.09189).
- Qin, Hamidieh & Alvarez-Melis 2026 (arXiv:2607.25271): token-effectiveness function; depends on model size and tokens per parameter; NeurIPS 2026 oral.
- Lu 2026 (arXiv:2603.28507): v1 and v2 titles as stated.
- Kim et al. 2025 (arXiv:2509.14786): ICLR 2026; weight decay about 30× standard; ensembling.
- Choshen et al. 2025 (ICML, PMLR 267): "nearly 500 models"; estimates sensitive to single badly behaved runs.
- van der Wal et al. 2025 (PolyPythias, ICLR 2025): 14M–410M, largely stable across seeds.
- Epoch "models over 1e25 FLOP": Grok-3 at 4.6e26 FLOP, the first model above 1e26 in their dataset.
- Besiroglu et al. 2024, Muennighoff et al. 2023, Villalobos et al. 2024, Sevilla et al. 2024 and Epoch 4–5×/yr were verified in round 1, and the attributions are unchanged.

I found no unverifiable citation.

**Missing related work (new).** Two 2026 papers bear directly on §5.5 and are not cited:

- Xu et al. 2026, *Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws* (arXiv:2606.06888; ICML/NeurIPS 2026). It argues that additive Chinchilla-style laws *do not fit* the data-constrained, compute-rich regime, and proposes a coupled "SoftQ" law. That is exactly the regime of Table 7, and it directly challenges the additive Lovelace forms that produce the 28–33× figures.
- arXiv:2610.37842, *Improving Data-Constrained Scaling with Hyperparameter Transfer and Ensembling*. It notes that existing data-constrained prescriptions hold hyperparameters fixed across scales.

Both should be discussed. Xu et al. would ideally be added as another row of Table 7.

## 3. Status of the round-1 major issues

**W1 (theoretical novelty; strawman test).**

- *Resolved as a framing issue.* The theory is now labelled "standard facts". The curvature test is no longer a baseline or a contribution. §3.2 states that no published paper uses it.
- *Not resolvable by revision.* The intellectual contribution stays modest. Prop. 3′ is a dimensional-analysis argument. The rest consists of a re-analysis of one public digitised dataset, simulations of a stylised law, and an audit of other people's functional forms. See Weakness 1.

**W2 ("no slowdown up to 1e22" overstated; grid edge; iid bootstrap).** *Resolved.*

- The claim is withdrawn.
- The effective support is reported (23 runs at or above $10^{21}$, 4 at or above $3\times10^{21}$).
- The grid is widened, and the optimum moves to the lower edge.
- Calibrated and cluster-robust tests replace the percentile bounds.
- The 245-run sensitivity is reported.

The new statement, that the top-segment change lies in $[-0.075,0.015]$ and nothing beyond about $3\times10^{21}$ FLOP is informative, follows from the evidence. Residual concerns are minor:

- The two-break breakpoints (10th and 70th percentiles) were chosen after seeing the one-break profile, and the CIs do not account for that selection.
- The wild-cluster null keeps the smooth misfit, which is 73% of the residual variance. The rejection is therefore a rejection of the five-parameter form in general, not specifically of a constant exponent. The authors say this, but the abstract's "rejects $\delta=0$" could still be read as evidence about exponents.

**W3 (detectability conclusions rest on digitisation-level noise).** *Resolved, and the authors confirmed the reversal with the exact procedure.* I reproduced the relevant cells. Two new concerns arise with the replacement claim (Weaknesses 2 and 3):

- The "correlated noise is binding" conclusion rests on an unanchored, assumed $\sigma_b$.
- The detection design is economically unrealistic at frontier scale.

**W4 ("next growth point" built in).** *Partially resolved.* Three mechanisms are now compared, and the conclusion is made conditional. However:

- (a) The discrimination result is largely true by construction.
- (b) The mechanisms are not matched in magnitude, so the power comparison is confounded.
- (c) RL, inference-time compute and algorithmic-efficiency drift remain unmodelled.

The second half of the research question ("where is the next growth point?") therefore receives a conditional, largely qualitative answer. See Weakness 4.

**W5 (data-wall constants rest on one extrapolated form; missing 2025–26 work).** *Largely resolved.*

- "Universal" is gone.
- The Lovelace forms are implemented from the authors' released fits, and I reproduced the Table 7 numbers.
- The more-than-two-orders-of-magnitude spread is reported honestly.

Two gaps remain:

- Xu et al. 2026 and arXiv:2610.37842 are missing (above).
- Every Table 7 row at $U\ge10^{10}$ extrapolates fits made on 50M–6B unique tokens by 2–6 decades. The table says so, but the text still reads the settled "hard-wall" values (28–33×) as properties of the additive forms at scale, rather than as artefacts of extrapolating a penalty that is linear in $N/U_D$.

**W6 (self-similarity fragile when $\alpha\neq\beta$; P1 transfer; linear $k_N$).** *Resolved.*

- Prop. 3′ is correct.
- The consistent rule gives identical multiples across $U=10^8$–$10^{15}$; I reproduced 69.18 at every $U$.
- The linear-rule drift (124 to 36) is reported.
- The $\tau$ uncertainty is quantified (0.32–5.2× over 6 decades).
- The "<1.4×" bound is withdrawn.
- P1 now requires $\tau$ to be estimated in-sweep.

## 4. Strengths

1. **Exemplary responsiveness and honesty.** Two headline claims were withdrawn. A numerical artefact in the authors' own v0 optimiser was disclosed, with the old outputs archived. Every claim is tagged [computed] or [cited].
2. **Reproducibility is excellent.** Seeds are fixed. The outputs I reran match the shipped files exactly, and the run order and timings are documented.
3. **The statistical treatment of the real data is now sound for what it claims.** It uses a profiled test calibrated under three noise models, cluster bootstraps, a report of the effective support, power on the actual design, and a sensitivity check on the exclusion rule.
4. **The detectability map is useful to practitioners.** It shows quantitatively that independent evaluation noise is not the binding constraint, and that the correlated, per-budget component is the quantity to measure.
5. **The data-wall audit is a genuine service.** Showing that published 2026 forms disagree by more than 100× on the 50%-efficiency milestone, while fitting the same runs, is a clear and citable negative result.

## 5. Weaknesses

1. **Limited novelty and significance for a NeurIPS main-track paper.** Every ingredient is acknowledged to be known. The empirical core is a re-analysis of one digitised public dataset, whose informative range ends about $3\times10^{21}$ FLOP, five orders of magnitude below the frontier. The main real-data finding is that a five-parameter law is misspecified on a digitised figure, with an untested attribution to small-scale effects or digitisation, and that the data cannot bound a top-end slowdown below about 8% of $\gamma$. This is careful but incremental, and it does not answer the research question. The question is answered as "public data cannot tell" plus conditional statements.

2. **The new "correlated noise is binding" conclusion rests on an assumed $\sigma_b$.** The per-budget offset of $10^{-3}$ is described as "small by the standards of separately tuned runs", but no citation or measurement supports that magnitude. The conclusion is also close to definitional. Model (2) makes the break a function of $C$ alone. A per-budget offset is also a function of $C$ alone, so it is perfectly confounded with the break signal, and the test effectively has $K=12$ observations. The paper should:
   - (a) say this explicitly;
   - (b) anchor $\sigma_b$ empirically, for example from published re-tuning studies (Porian et al. 2024 report loss differences from tuning), PolyPythias seed spreads, or the size of the Chinchilla per-budget residual structure itself;
   - (c) show power as a function of $K$ at fixed total runs. More budgets is the obvious remedy.

3. **The detection design is not one any lab would run at the frontier.** A 240-run sweep covering $10^{25}$–$10^{28}$ FLOP, with 20 runs per budget at $10^{28}$, costs on the order of $10^{29}$–$10^{30}$ FLOP, which is more than the frontier run it is meant to diagnose. The realistic protocol is to fit a small-scale sweep and test the forecast residuals of one or a few frontier runs. That is a different test, with different power and much stronger dependence on the extrapolation of $E$; Exp. 6a shows that $\hat E$ absorbs breaks. Without this design, P5 and the claim that labs "can detect small breaks" do not carry over to practice. The 48-run small design does not fix this, because it still places 8 runs at each of 6 budgets up to $10^{28}$.

4. **The mechanism comparison (Exp. 11) is largely by construction, and its power comparison is confounded.**
   - M3 is defined as a data-independent break, so its matched data-doubling gap is identically zero by definition. M1 and M2 differ only beyond about $30\,C_x$, by about $3$–$10\times10^{-4}$ in log-loss, which by the paper's own noise analysis needs very low difference noise.
   - The mechanisms share an onset, but they are not matched in size. At $3\,C_x$, $\eta=0.73$ for M3 against 0.97 for M1 and 0.96 for M2; M3 with $\delta=0.05$ is 29% of $\gamma$. "The test detects M3 best" therefore partly reflects M3 being a much larger break, besides it matching model (2). Power should be compared at matched $\eta$, or at matched log-loss excess.
   - $\theta=0.5$ and $\delta=0.05$ are illustrative, so Table 8's levers (∞, 1.5×, 1×) are restatements of the assumptions.
   - Matched data-doubling (or data-scarcity) experiments are exactly what Muennighoff et al. 2023, Lovelace et al. 2026 and Kim et al. 2025 already run. The recommendation is not new.

5. **"Next growth point" remains mostly outside the analysis.** RL scaling, inference-time compute, synthetic data and algorithmic-efficiency drift, which are the levers practitioners actually debate, are excluded by scope. Algorithmic-efficiency drift is dismissed as "observationally equivalent to rescaled $C$". That is true for the loss curve, but it is the central quantity for "where does growth come from". The title and research question promise more than §5.6 delivers.

6. **The Table 7 extrapolations and the calendar.** The calendar compares dates across laws whose $C_{\rm wall}$ differ ("a rough comparison"), and it uses a growth rate of 3–5×/yr while Epoch now reports Grok 4 at about $5\times10^{26}$, roughly flat since Grok-3. If included at all, the calendar should be in an appendix with stronger caveats.

## 6. Minor issues

- Explain the 71× (consistent rule) versus 74–75× (linear $k_N=0.051$) difference for the same $\alpha=\beta$ law.
- The Huber threshold $\delta_H=10^{-3}$ is fixed while $\sigma_w$ ranges from $10^{-4}$ to $7.5\times10^{-3}$. At high noise the estimator is effectively L1, which lowers efficiency (and power) in those cells. Scale $\delta_H$ with $\sigma$, or justify keeping it fixed.
- The fitted-null size check gives 0/20 rejections in both cells. That hints at a conservative fitted-null procedure. The real data use fitted-null calibration while Table 3 uses oracle calibration, so the oracle powers may overstate real-procedure power. Report fitted-null power in at least one cell.
- The abstract's "rejects $\delta=0$" should read "rejects the five-parameter constant-exponent form", given the misspecification interpretation.
- `code/README.md` still carries the old title ("Deceleration Without a Break"), and its Exp. 1 entry is not in `timing.txt`; Exp. 1 was not part of the r1 rerun, although I confirmed that it reproduces. Exp. 12 is also missing from `run_all_r1.sh`.
- The two-break model's breakpoints are post hoc (see W2 above); state this.

## 7. Questions for the authors

1. What empirical basis supports $\sigma_b\approx10^{-3}$? How does the power change with the number of budgets $K$ at fixed total runs?
2. What is the power of a forecast-residual test: a small sweep up to $10^{23}$ plus 1–3 runs at $10^{26}$?
3. At matched $\eta$ (or log-loss excess), how do the M1, M2 and M3 detection powers compare?
4. How do the Table 7 milestones change under the SoftQ law of Xu et al. 2026?

## 8. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 7 |
| Correctness | 8 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5.5 → 6 (borderline)** |

## 9. Decision

**Major revision** for the NeurIPS main track; it is closer to a solid TMLR paper.

The revision fixes every correctness problem raised in round 1, and the paper is now honest and reproducible. What remains is that the contribution is incremental, and the two new headline claims are not yet established. The first is "correlated noise is binding": it rests on an unanchored $\sigma_b$ under a design no lab would run. The second is "matched doubling discriminates mechanisms": it is true by construction and confounded in its power comparison. The single most valuable change would be a realistic frontier-detection design (small sweep plus a few frontier runs, forecast-residual test) with an empirically anchored correlated-noise level. That would turn the detectability section from a conditional map into an actionable result.
