# Response to Round-1 Review (p004)

We thank the reviewer for a careful review and a full reproduction. We agree with all six major points. Two of them (M3, M4) led us to **withdraw a claim** after new experiments contradicted it. Below, "new" marks code and results produced in this revision. Every new number comes from scripts we ran in this environment, with outputs saved under `results/`. Everything else is cited and was checked by web search.

New code and outputs:
- `code/e5_tinylm_retention.py`, `code/e5_analyze.py`: tiny-transformer experiment, 78 runs. Outputs `results/e5_tinylm.json`, `e5_analysis.json`, `fig5_tinylm.png`.
- `code/e6_verifier_design.py`, `code/e6_analyze.py`: decorrelated verifier grid and verifier-cost experiment, 170 runs. Outputs `results/e6_verifier_design.json`, `e6_analysis.json`, `e6_summary.txt`, `fig6_verifier_design.png`.
- `code/e7_threshold_storage.py`: threshold-storage analysis. Output `results/e7_threshold.json`.
- `code/e2_predictors.py`: re-analysis of E2 predictors. Output `results/e2_predictors.txt`.
- `code/e4_horizon.py`, rewritten with labelled anchors.

Renamed files: the urn script is now `e3_urn_scaling.py` and the horizon script is now `e4_horizon.py`.

The title is now "Exchange Rates for Synthetic Data: Coverage, Learnability and Verification Limits on Stretching the Human Data Stock".

---

## Major issues

### M1. The theory is mostly standard results in new notation, and Prop. 2's invariance restates the assumption.

**Agreed. Changes:**
- **Relabelled the standard results.** Prop. 1 is now "Fact 1 (garbling; standard)". Thm. 2 is now "Lemma 3 (accounting identity; elementary)". Prop. 3 is now "Remark 1 (plurality; strong law)". Prop. 4 is now "Prop. 5 (application of known results)". The introduction has a new paragraph, "What is standard and what is new", which removes these from the contribution list. The Limitations section has a "Theory depth" item that says the same.
- **Thm. 2's first step is an equality, as the reviewer said.** Lemma 3 now gives the equality with its one-line proof: $I(\theta;V\mid\cdot)=I(\theta,C;V\mid\cdot)=I(C;V\mid\cdot)$. It writes the per-proposal term explicitly as an expectation, and it obtains the marginal-$q$ form as an upper bound by Jensen. This also covers m2.
- **Prop. 2 is restated as a consequence of the memoryless assumption.** The text now says plainly that its invariance "is a restatement of the memoryless assumption".
- **New result (Prop. 2b).** Under threshold storage (a fact is stored iff at least $m\ge2$ storage events occur), the per-fact multiplier diverges as $\lambda^{-(1-1/m)}$ for rare facts and tends to the memoryless value for frequent facts. So the corpus-level $M$ depends on $(\alpha,N)$ and can exceed $1/s$. The proof uses compound-Poisson asymptotics. New script E7 computes it exactly with Panjer's recursion. At $s=0.1$, $r=7$, $m=2$, $M$ ranges from 5.87 to 8.71 across $\alpha$ and $N$, while the memoryless value is 5.70 everywhere. The per-fact $M$ is 32.4 at $m=3$, $\lambda=0.1$. This answers Question 5: no, invariance does not survive.
- **New result (Prop. 2c).** The coverage ceiling holds for every storage law. It is the one robust knowledge-side statement, and E5 finds that it binds exactly.
- **New result (Prop. 4, verifier description length).** For any number of proposals, $I(\theta;\text{all})\le I(\theta;D_h)+L$ for a verifier of $L$ bits. With Fano, learning $\eta K'$ unseen $k$-ary facts then needs $L\gtrsim\eta K'\log_2k$. This turns the earlier verbal "knowledge verifiers must contain the facts" into a statement. It is a short data-processing argument, and we present it that way.

We do not claim these additions raise the theory to a deep level. The paper now states its theoretical depth honestly and puts more weight on the experiments.

### M2. Prop. 4 is the randomly reinforced urn, and the RRU literature is not cited.

**Agreed. Changes:**
- Muliere, Paganoni & Secchi (2006, JSPI 136:1853–1874) and May & Flournoy (2009, Ann. Statist. 37(2):1058–1078) are now cited. We verified both by web search.
- Related Work has a new paragraph on randomly reinforced urns.
- The result is renamed "Prop. 5 (verified self-training is a randomly reinforced urn; application of known results)". The phrase "new scaling law" is gone.
- We state that the rigorous convergence rates are May & Flournoy's. We did not re-derive their theorem, and we use it only for the correspondence "verifier $\leftrightarrow$ reinforcement means". That correspondence gives the exponent $1-b/a$.

We did not cite Durham, Flournoy & Li (1998), because we could not verify its details in this session.

### M3. E2 does not test the knowledge-vs-skill asymmetry; the verifier is a free oracle on both tasks.

**Agreed. New experiment E6 Part B (verifier cost).** Each verifier is now given an explicit description length.
- The knowledge verifier is a lookup table storing a fraction $\kappa$ of the facts, at $3\kappa\cdot400$ bits.
- The skill verifier is the teacher quantized to $\beta$ bits per parameter, at $168\beta$ bits. Its errors are systematic.

Results (5 seeds each):
- At 504–672 bits, the program verifier raised skill accuracy by +0.286 and +0.353.
- At 600 bits, the lookup verifier raised knowledge accuracy by +0.050. It needed 1,200 bits (all facts) to reach +0.182.
- The skill learner converged to the verifier's own agreement with the truth (0.834 vs 0.836; 0.901 vs 0.921), an instance of Yi et al.'s "knowledge center".

Together with Prop. 4, this gives the asymmetry a measured basis and a formal statement. The E2 section now says explicitly that its $M=35.9$ for knowledge shows only that an oracle storing all 400 facts (1,200 bits) transmits them.

**Caveat, stated in the paper.** The human budgets differ between tasks, and lookup versus quantized program is a stylised pair of verifier designs.

### M4. "Gain tracks verifier information" is confounded by design; the p-values assume independence; E2 bits use marginal q.

**Agreed, and the claim is withdrawn.**
- **Re-analysis of E2 (`e2_predictors.txt`).** This reproduces the reviewer's table. Pooled Spearman for $-b/a$ is 0.868 (skill) and 0.857 (knowledge), against 0.894 and 0.830 for information. Within budgets, $-b/a$ averages 0.978 and 0.970, and information averages 0.938 and 0.970.
- **New decorrelated grid (E6 Part A).** We crossed $a\in\{0.25,0.5,1\}$ with $b/a\in\{0,0.2,0.5,0.8\}$, with 5 seeds, on both tasks. We computed information both with marginal $q$ and with per-proposal $q_j$ (the learner's probability of its sampled label, a proxy for Lemma 3's conditional term). Spearman correlations with gain:

  | predictor | skill | knowledge |
  |---|---|---|
  | information | 0.78 | 0.81–0.85 |
  | $-b/a$ | 0.97 | 0.97 |
  | precision | 0.99 | 0.96 |

  - The partial rank correlation of gain with $b/a$ given information is 0.99 (skill) and 0.94 (knowledge).
  - In 2,000 seed-bootstrap resamples, information never beat $b/a$.
  - At fixed $b/a=0$, an 8× increase in information (from raising $a$ fourfold) raised the gain only from 0.359 to 0.389.
- **What we now say.** Information (Lemma 3, Prop. 4) is a ceiling, not a predictor. Gain follows the purity $b/a$ of the accepted stream, which is what the urn rate predicts: $a$ only rescales time. P4 is withdrawn and replaced by P4′ ("purity predicts gain"). The abstract, findings and conclusion were rewritten to match.
- **p-values.** They have been removed from the paper. The text says they treated non-independent cells as independent.
- **Marginal versus conditional q.** E2 bit counts are now labelled "marginal $q$" and described as an upper bound by Jensen. E6 reports both versions. The conditional version is smaller, as theory predicts, for example 7.62 vs 12.33 kbits at $(1,0)$ on the skill task.

### M5. The "≈3 years" headline is ln10/ln2 applied to an unmeasured M; there is no anchoring to WRAP or Kang et al.

**Agreed. Changes:**
- The "M=10 buys about 3 years" headline has been removed.
- `e4_horizon.py` now uses multipliers labelled by provenance:
  - WRAP, about 3× pre-training speed-up (Maini et al., 2024; external).
  - Kang et al. (EMNLP 2025, verified): a one-third rephrased mix reaches the same validation loss 5–10× faster (external).
  - Muennighoff et al.: 4 epochs (external).
  - Our E5 tiny-LM multipliers, 2.92 and 3.75 (this paper).
  - Model-only hypotheticals, labelled as such.
- The external figures are flagged as training speed-ups, not strict human-data multipliers.
- The answer to the research question is now: measured rephrasing gains of about 3–10× correspond to about 1.4–3.3 years of horizon delay under the stated 2.0–2.24×/yr dataset-growth assumption. With our tiny-LM values the range is 1.3–3.3 years. We call this arithmetic on external inputs, not a forecast.
- On estimating $s$, $a$ and $b$: E5 estimates $s$ for a tiny transformer, and E6 measures empirical $(a,b)$ for quantized-program verifiers. Neither is an LLM estimate. A pre-registered LLM protocol is given in §6.1.

### M6. Only toy learners; E1 is circular; no neural or LM experiment; no baselines.

**Partly addressed. Changes:**
- **New neural experiment E5.** A 0.67M-parameter, 2-layer causal transformer is trained with next-token loss on 2,000 planted facts in 8 templates. There are 78 runs, all on CPU. It implements the reviewer's highest-leverage suggestion: estimate $s$ from recall versus exposure count, then test Prop. 2 and P1 against real learning curves. Results:
  - **Coverage ceiling confirmed exactly.** Paraphrase recall saturates at the coverage ceiling: 0.813 vs 0.812, and 0.934 vs 0.934.
  - **One paraphrase is worth about one fresh occurrence.** $M(r=1)=2.09$ (90% CI 1.98–2.19).
  - **Paraphrase beats repetition.** 2.97 vs 2.66 at $r=3$; 1.80 vs 1.47 at $N_0=8000$.
  - **Prop. 2's invariance is falsified for this learner.** $M(r=3)$ is 1.80, 2.97 and 3.75 across $(\alpha,N_0)$.
  - **$s$ is not a learner constant.** The geometric $s$ rises from 0.029 to 0.205 as $N$ grows eightfold.
  - **Storage law.** Storage is close to memoryless, with at most an $m=2$ threshold. Thresholds of $m\ge3$ are clearly rejected.
  - **Predictions.** The memoryless model predicts $M$ well at $r=1$ and overpredicts it at $r\ge3$.
- **E1 is now framed as a code check.** The paper states that E1 tests the closed form against a simulation of the same model, and that the N-equivalent check is guaranteed algebraically.
- **Not done.** We ran no STaR/ReST or Feng et al.-style baseline on neural models, and no LLM-scale experiment. There is no GPU or LLM API in this environment. The Limitations section says so, and §6.1 gives a pre-registered protocol. We think this remains the main gap for an ICLR main-track paper.

---

## Minor issues

- **m1 (fidelity and Prop. 3 at φ=0.5).** Fixed. The fidelity threshold is now stated as holding only as $r\to\infty$. The original exposure is always correct, and at $\phi=1/2$ the limit is undetermined. E1's decline at $\phi=0.5$ is attributed to tie-breaking and finite-$r$ dilution, "not from Remark 1".
- **m2 (Thm. 2 expectation; one bit trivial).** Fixed. Lemma 3 writes the term as an expectation over $\mathcal F_j$. The trivial "at most one bit" sentence has been removed.
- **m3 (Villalobos intervals).** Fixed. 2026–2032 is called an 80% interval, the 100T–1000T stock a 90% CI, and the 1000T row is "upper end of 90% CI".
- **m4 (Huang et al. quotation).** Fixed. The quotation marks are gone, and we now paraphrase their view as extracting knowledge already latent in the model.
- **m5 (section/script numbering).** Fixed. The scripts are renamed `e3_urn_scaling.py` → `results/e3_urn.json`, `fig3_urn.png`, and `e4_horizon.py` → `results/e4_horizon.json`, `fig4_horizon.png`. Experiment labels in the paper match script names.
- **m6 (MC spread).** Fixed. The paper now reports 1.34 population s.d., about 1.9 standard errors using the $n-1$ s.d.
- **m6b (relative error).** Fixed. It is reported as $2\times10^{-11}$, recomputed from `e1_knowledge.json`.
- **m7 (dead code; inf cells).** Fixed. The dead expression in `e2_analyze.py` is replaced by `float("inf")`. Re-running gives an `e2_table.txt` byte-identical to before. The paper now says that "inf" cells are excluded from $M$ summaries and that this biases aggregates downward.
- **m8 (blocked urn simulation).** Fixed. This is stated in the paper (§5.5) and in the script docstring.
- **m9 (README timing).** Fixed. The README now gives E2 as 18–45 min and machine-dependent. It also gives timings for the new scripts, noting that they were measured on shared cores.
- **m10 (missing related work).** Fixed. Added Gan & Liu (ICLR 2025, arXiv:2410.01720), Kang et al. (EMNLP 2025, arXiv:2510.01631, with its 5–10× result used in E4), Allen-Zhu & Li Part 3.1 (ICML 2024, PMLR 235), and the urn papers. All were verified by web search. Gunasekar et al. was removed because it was no longer used.

## Questions

1. **RRU positioning.** See M2. What we add is the mapping from verifier to urn, the simulation check, and E6's evidence that the urn's prediction (purity $b/a$ governs; $a$ rescales time) carries over to a learner that generalizes.
2. **Decorrelated verifier design.** Done (E6 A). The result goes against our earlier claim: $b/a$ and precision beat information.
3. **Verifier cost and coverage.** Done (E6 B, Prop. 4).
4. **Estimate $s$ for a small LM.** Done for a 0.67M-parameter transformer (E5). $s$ ranges from 0.03 to 0.28 depending on corpus size. This turns the horizon figure into a range anchored by measurements for this model, and by cited speed-ups for real pipelines. It is not an LLM measurement.
5. **Correlated or thresholded storage.** Done (Prop. 2b, E7, E5). Invariance fails, both in theory and in the tiny LM.

## Remaining weaknesses we acknowledge

- No experiments at LLM scale.
- No neural verified-self-training baseline.
- Theory consists of short arguments.
- External speed-ups are not strict multipliers.

The paper's claims are now limited to what the evidence supports, and it reports the two places where our own earlier claims were falsified.
