# Response to Review Round 1 (p003)

We thank the reviewer for an unusually careful review. In particular, they reran every script and tested the offline-audit claim on our own configurations. We accept every major point. In two places the new experiments go further than the reviewer's own critique:
- The 100-step on-policy window mostly observes the outcome: 93% of races are decided by step 100.
- Part of the round-1 small-λ "lag" in Part A was the reviewer's (G−1)/G bias, not slow equilibration.

Several headline claims are withdrawn or narrowed. All new numbers come from code in `code/` run in this environment, with outputs in `results/`:
- `exp2b.json`, `exp8.json` – `exp12.json`;
- figures `fig8`, `fig9` and `fig11`.

No result was obtained from an LLM we trained. E11 analyses published judgment files.

The new and modified scripts are:
- `exp2b_partA_bias.py`;
- `exp8_offline_audits.py`;
- `exp9_meanfield_boundary.py`;
- `exp10_robustness.py`;
- `exp11_real_judges.py`;
- `exp12_metastability.py`;
- `figs_revision.py`.

`run_all.sh` and `README.md` are updated.

---

## Major issues

### M1. Missing related work and overstated novelty (W1)

**Verification.** We verified each suggested source by web search: titles, venues and abstracts.
- Egashira et al. 2026, arXiv:2605.02909, COLM 2026.
- Moya, Thornley & Lin, arXiv:2609.35677.
- "Sparse Reward Auditing for RLVR under Policy-Dependent Verification Error", NeurIPS 2026 virtual page 164475. The author list was not retrievable, so we cite it by title and venue.
- Kwa, Thomas & Garriga-Alonso, "Catastrophic Goodhart", arXiv:2407.14503, NeurIPS 2024.
- "Checking the Checkers", arXiv:2610.17824. The author list was not retrievable, so we cite it by title and arXiv number.

**Changes.**
- **Section 1** has been rewritten. It no longer frames response-dependent error as "the case theory leaves open". It now says explicitly that recent work (Rad et al.; Egashira et al.; Moya et al.; Sparse Reward Auditing) has established that error structure matters. Our contribution is stated as quantitative and in four parts:
  - the rate/fate split and the three-mode closed form;
  - a race boundary in (J, h0) for practical PG;
  - a matched-budget audit comparison;
  - a real-judge measurement of J by domain.
- **Section 2** has a new paragraph, "Verifier error in RLVR (closest work)", with one item per paper stating what we add relative to it:
  - *Egashira et al.:* we parametrise which patterns collapse via J*(h0).
  - *Moya et al.:* we contrast the tilt with softmax PG, quantify the race, and use audits as forecasts rather than corrections. Our E8 agrees with their non-identifiability result.
  - *Sparse Reward Auditing:* our audit budgets are comparable, but we use them diagnostically.
  - *Kwa et al.:* our Corollary 2 is a bounded-reward, binary-label instance of their tail principle.
  - *The survey:* it reaches the same "structure over rate" conclusion.
- **Section 3.** Theorem 1 is now called "elementary bookkeeping". Corollary 2 is identified as "the textbook concentration of a Gibbs measure on the argmax of f" and linked to Kwa et al.

### M2. The "on-policy beats offline" audit claim rested on a weak baseline (W2)

**Changes.** We added experiment E8 (`exp8_offline_audits.py`, `exp8.json`, Fig. 8). It covers the 60 round-1 configurations, with identical training seeds so the labels reproduce, plus 60 new ones, for 120 in total (52 collapsed). Every AUC now carries a 2,000-sample bootstrap 95% CI.

The audits compared are:
- *Aggregate offline* at 256 samples (round 1) and at a matched 2,816 samples.
- *Tail-aware offline, feature-blind,* at 2,816 judge calls: 352 base samples, each judged 8 times. The statistic is ĥ_tail/Ĵ, where ĥ_tail is the share of wrong samples accepted more often than the average correct sample.
- *One-step look-ahead offline,* at 2,816 samples. It uses only base samples and the policy's score function.
- *On-policy* audits at 20, 50 and 100 steps.
- *The reviewer's oracle statistics:* the x₀ acceptance gap, h0/J, and the J*(h0) − J margin.
- *Reference:* the exact exploit mass at the window end.

We also recorded each run's decision time.

**Results (all 120 runs).**

| audit | AUC [95% CI] |
|---|---|
| aggregate, 256 samples | 0.766 [0.675, 0.846] |
| aggregate, 2,816 samples | 0.877 [0.814, 0.932] |
| look-ahead | 0.891 [0.827, 0.941] |
| tail-aware | 0.914 [0.864, 0.959] |
| on-policy, steps 0–20 | 0.771 [0.684, 0.848] |
| on-policy, steps 0–50 | 0.957 [0.922, 0.984] |
| on-policy, steps 0–100 | 0.987 [0.970, 0.998] |
| x₀ gap (oracle) | 0.988 [0.973, 0.998] |
| h0/J (oracle) | 0.997 [0.990, 1.000] |

On the original 60 runs we reproduce the reviewer's values: x₀ gap 0.988, h0/J 1.000, on-policy at 100 steps 0.984.

**Timing.**
- Decision times run from 40 to 130 steps (median 60). No race is decided by step 20, 44% are decided by step 50, and 93% by step 100.
- So the round-1 statistic largely observes the outcome. This confirms the reviewer's concern about the window overlapping the decisive phase.
- Among the 67 runs still undecided at step 50, the on-policy slope reaches 0.89 [0.81, 0.96], tail-aware 0.80 [0.68, 0.89] and aggregate 0.69.

**Claim changes.**
- Withdrawn: "Audit verifiers on-policy, not offline" and "on-policy 0.98 vs offline 0.77".
- New claim:
  - (i) aggregate audits are weak;
  - (ii) tail-aware offline audits are the best feature-blind forecasts before training;
  - (iii) exploit-aware offline statistics are near-decisive;
  - (iv) the on-policy slope is a strong early *monitor* once training is under way, not a forecast.
- Abstract, contributions, Section 7 and Section 9 (practical implications) are updated to match.
- Pre-registered prediction P4 is rewritten to test tail-aware against aggregate offline audits at equal budget.

### M3. The evidence is far from the research question (W3)

We could not train an LLM: there is no GPU, no model weights are reachable, and there is no API. Instead we added a **real-verifier anchor** from public data, E11 (`exp11_real_judges.py`, `exp11.json`, Fig. 11).

**What E11 does.**
- JudgeBench (Tan et al., ICLR 2025; citation verified) publishes judgment files for 33 judges, at github.com/ScalerLab/JudgeBench, commit e2c52c28.
- The judges are 5 scalar reward models and 28 pairwise LLM judges or fine-tuned judges.
- The files cover 350 pairs. Each pair holds a correct and an incorrect GPT-4o answer to the same question, with gold labels from MMLU-Pro, LiveBench and LiveCodeBench.
- We computed J_pair = 2·P(prefer correct) − 1 by domain, with bootstrap CIs. We prove in Section 8 that J_pair = α − β exactly for a class-conditional binary verifier, so it is the J of our theory.
- For the reward models we also computed tail proxies: upper-tail inversion shares and confident inversions.

**Results.**
- Median J_pair is 0.12 on humanities/social/biology knowledge, 0.28 on STEM knowledge, 0.19 on reasoning, 0.49 on math and 0.02 on code.
- All 33 judges discriminate better on math than on humanities/social/biology (sign test p = 2.3×10⁻¹⁰).
- Confident inversions by reward models occur in 6.3–11.2% of non-math knowledge pairs, against 1.5–6.0% of math pairs.
- On the derived toy boundary (E9), J ≈ 0.12 tolerates h0 ≈ 0.27%, against about 1.6% at J ≈ 0.49.

**Limits, stated in the paper.**
- The pairs are hard and come from one strong model, so they do not represent an RL base policy.
- Inversion shares are not h0.
- Open-ended domains are not covered.
- The placement on the toy boundary is labelled "illustration only".

**Claim changes.** The conclusion is restated: the transferable claim is *ordinal* (non-math judge rewards have markedly lower J, and lower J means slower SGD learning and a smaller exploit budget in every model studied). The quantitative constants are not claimed to transfer.

We did not use the master-key FPRs of Zhao et al. as the anchor. The full text and tables were unreachable from this environment, which blocks arxiv.org and mirrors, and we would not cite numbers we could not read.

### M4. The "power law" is a fit to a curved, saturating boundary (W4)

**Changes.**
- We added E9 (`exp9_meanfield_boundary.py`, `exp9.json`, Fig. 9).
- The boundary is now *derived* from the population (mean-field) dynamics. We take exact expected std-normalised updates over all 1,024 strings, with the same learning rate and horizon as before, and bisect on J to 10⁻⁴. The grid has 18 values of h0 from 3.6×10⁻⁴ to 0.064.
- We do not obtain a closed form; the boundary is computed numerically.

**Results.**
- The mean-field boundary matches the round-1 finite-sample bisection points to a mean absolute difference of 0.007 (factorised) and 0.008 (AR1), within the round-1 resolution of about 0.015.
- Local log–log slopes fall smoothly from 1.10 to 0.27 (factorised) and from 1.14 to 0.30 (AR1). This confirms the reviewer's curvature (their 1.07 → 0.41 on the coarse grid).
- The factorised J* saturates at η − β = 0.8 for h0 ≥ 0.05.
- At small h0 the boundary is near-linear (J*/h0 ≈ 43–46).
- The reviewer's initial-drift estimate J ≈ 15.4·h0 is confirmed as the sign change of the initial x₀ drift. It underestimates J* by about 3× at small h0 (about 1.3× near saturation), because the race is decided along the trajectory. We say this explicitly, and we list the absence of a closed form as a limitation.

**Claim changes.**
- The power law (12.6·h0^0.80, 8.9·h0^0.76) is withdrawn from the abstract, contributions and conclusion.
- P2 is rewritten. The old version read "slope > 1". The new version reads: "monotone, log–log slope ≤ about 1.1 and decreasing in h0".

---

## Minor issues

**m1. Proposition 2 was mis-stated.** Accepted.
- The proposition is restated with 0 < FPR < TPR < 1, A0 ∈ (0,1), and a no-large-atom condition on π0 restricted to wrong responses.
- The construction now chooses δ′ ∈ (0, FPR(1−A0)) explicitly and gives the value of f on the remaining wrong mass.
- The unused parameter δ is removed.
- The text notes why each condition is needed. In particular, at TPR = 1 we get A → A0/(A0+δ′), as the reviewer showed.

**m2. Few seeds and no CIs.**
- Wilson 95% intervals are now reported on collapse fractions, and bootstrap CIs on AUCs, median t90 and the t90-vs-J slope.
- The new experiments use 16 seeds for the optimiser and learning-speed cells and 8 for the encoding and lr/G cells.
- The round-1 E4 cells (3 seeds) are re-run with 8 seeds in E10a. The base noisy-judge factorised cell is 5/8 collapse [0.31, 0.86], where round 1 reported 3/3. The paper reports the 8-seed figure.

**m3. t90·J is not a confirmation of Corollary 1.** Accepted.
- The text now says that the 1/J scaling of softmax PG is an *empirical* observation, and that Corollary 1 concerns the tilt.
- We explain it with the reviewer's J/σ argument. The initial reward SD σ0 varies only from 0.366 to 0.408 across J (`exp10.json`), so J/σ ≈ J/0.39.
- With 16 seeds the slope of log t90 against log J is −0.97 [−0.99, −0.95].
- The J = 0.1 point is flagged as resting on 4 non-collapsing seeds.

**m4. Part A (G−1)/G bias.** Accepted, and it mattered more than either of us expected. We added E2b (`exp2b.json`). With the fixed point at t = (G−1)/(Gλ):
- the centred-advantage predictions match GRPO within 0.0042 at all six λ;
- uncorrected, the error reached 0.032 at λ = 0.1.

The round-1 text attributed the small-λ discrepancy to non-equilibration, and that is corrected. For std-normalised advantages the correction does not explain the small-λ gap, so non-equilibration remains the explanation there. The table now shows both predictions.

**m5. The E5 J = 0.8 cell set α = η.** Accepted.
- The cell is now described as degenerate and uninformative.
- We added a J = 0.75 column (α = 0.90 < η; E10e, 8 seeds, factorised and AR1). It shows 0/8 collapse at all three h0, including h0 = 0.025, consistent with the derived J*(0.025) = 0.64.

**m6. The E4 sharing result depends on one encoding.** Accepted. E10a tests four encodings in three policy classes, under two verifiers, with 8 seeds each: the base encoding, a near miss (x₀..x₃=1, x₄=0, x₉=1), an anti-correlated exploit (x₁=x₂=0, x₈=x₉=1), and heterogeneous η ~ U[0.6, 1].
- The *direction* holds in every case: tabular policies never collapse (0/64), while shared ones can.
- The *size* depends strongly on the encoding:
  - under the noisy judge, near miss 8/8 (factorised and AR1), base 5/8, heterogeneous η 1/8, anti-correlated 0/8;
  - under the math-like verifier, the near miss still collapses 5/8 (factorised).
- The anti-correlated exploit also has 2.5× lower h0, and we say this confound plainly.
- The limitations section now states that only directions transfer.
- E10d adds lr ∈ {0.25, 0.5, 1} × G ∈ {8, 16, 32}: the factorised policy collapses in 4–7 of 8 runs in every cell, so the result is not a learning-rate or group-size artefact. This also answers Question 6.

**m7. Entropy bonus and natural gradient were never tested (P5).** E10b tests them, using the factorised policy, 5 (h0, J) cells and 16 seeds.
- Entropy bonuses raise collapse: 0 → 10/16 (c = 0.01) and 0 → 16/16 (c = 0.05) at (0.0026, 0.2), and 0 → 16/16 at (0.009, 0.45) with c = 0.05.
- Adam raises collapse in four cells, for example 0 → 7/16 at (0.009, 0.45).
- The diagonal natural gradient learns about 2.5× faster and raises collapse only in the lowest-J cell (0 → 3/16).
- The paper reports P5 as supported in the toy for entropy and Adam, and only weakly for our natural-gradient variant.

**m8. Citing Lv et al. as consistent with "4× more steps" was misleading.** Accepted.
- The Discussion no longer cites Lv et al. as consistent with the 1/J slowdown. It notes the tension: they report rapid convergence at 40% flips.
- We tested one candidate explanation. Under Adam, the toy's t90 depends on J only as J^−0.38 [−0.41, −0.36], against J^−0.97 under SGD, so adaptive optimisers largely cancel the J-dependent signal scale.
- This is presented as a reconciliation we have not tested at LLM scale. P1 now predicts an exponent smaller in magnitude under Adam.

**m9. arXiv:2512.16912 disputes the clipping account.** Accepted and verified: Chen, Li, Li, Yin, Chen & Lin, "Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward", ICLR 2026. It is cited in Related Work and Limitations as disputing that clipping bias alone supplies a learning signal.

**Other points from W7.**
- *Rate claim and E5.* The Discussion's "Rate" item now separates Corollary 1 (tilt) from the empirical softmax-PG scaling.
- *Fig. 6, "cap".* The text now says the tilt is the KL-optimal policy, not a ceiling on accuracy. GRPO exceeds the tilt's accuracy because it has not reached the optimum.
- *Fig. 6, metastability.* E12 (`exp12.json`) measures exploit mass over 32,000 steps.
  - Tabular, λ = 0.3: exploit mass rises from 0.0030 to 0.0077, at about 0.02 decades per 1,000 steps. The tilt's exploit mass is 0.354, so about 80,000 more steps would be needed at that rate.
  - Tabular, λ = 0.5: the run converges to the tilt (A = 0.43 vs 0.430).
- *A further error found during the revision.* The tilt lies outside the factorised policy class, so the factorised columns in E6 are not tests of the tilt. This is now stated.

## Questions

1. **Difference from Egashira et al., Moya et al., Sparse Reward Auditing and Kwa et al.** See M1. The Section 2 paragraph states the difference from each.
2. **Tail-aware offline audits at matched budgets with CIs.** See M2 and E8.
3. **Deriving the boundary from the mean-field ODE.** Yes, numerically (M4, E9). It explains the curvature: the slope falls from 1.10 to 0.27 and saturates at η − β. A closed form is still open.
4. **J and tail proxies for a real judge.** We measured them for 33 published judges by domain (M3, E11). We could not measure them on Su et al.'s data or along an RL trajectory, and the paper says so.
5. **Entropy bonus and natural gradient.** See m7 and E10b.
6. **Sensitivity to learning rate, group size and encoding.** See m6, E10a and E10d.

## Citation integrity

- Every new reference was verified by web search in this session: Egashira et al.; Moya et al.; Sparse Reward Auditing; Kwa et al.; Checking the Checkers; Chen et al. 2025; JudgeBench.
- Two references give no author list, because we could not retrieve one: Sparse Reward Auditing and Checking the Checkers. We cite them by title and venue or arXiv number rather than guess authors.
- No round-1 reference was removed. The reviewer had confirmed them all.
