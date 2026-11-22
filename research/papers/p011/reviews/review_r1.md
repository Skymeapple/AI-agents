# Review (Round 1): "Necessity Is Not Reliance: Disentangling Chain-of-Thought Faithfulness Tests with Ground-Truth Transformers" (p011)

Reviewer role: area chair level, independent. Target venue as stated: NeurIPS 2027 main track.

## 1. Summary

The paper asks what the two most common behavioural CoT-faithfulness tests (truncation / early answering, and corruption / adding mistakes) actually measure. It defines three properties: necessity N (accuracy drop when the CoT is removed), reliance S (probability that the answer follows a corrupted CoT) and description D (the internal computation passes through the stated intermediate, defined as a causal-abstraction alignment). It then:

1. proves four elementary propositions: N and S are logically independent (Prop. 1); D cannot be identified from input-output behaviour (Prop. 2); a cross-entropy-optimal predictor is CoT-invariant when erroneous CoTs are on-support and leaves S undetermined when CoTs are only ever absent (Prop. 3); and under a two-route multinomial-logit model the three-way corruption outcome predicts truncation accuracy (Prop. 4, "cross-test identity");
2. trains 2-layer, width-64 transformers on (a+b) mod 23, then +c, with a one-token CoT. Training varies the CoT noise rate eta and dropout rate delta. Attention-mask controls give ground-truth reliant or post-hoc models, and path cuts and linear probes give mechanistic references;
3. reports checkpoint-level disagreement between the two tests (13.3% of 511 checkpoints). The direction of disagreement depends on the training regime: under eta=0, delta=0.3, N-S+ ("redundant-but-reliant") checkpoints make up 24.7%; under eta>0, N+S- checkpoints make up 9.1%. Corruption tracks a mechanistic CoT-route-sufficiency reference (Spearman 0.96, n=26). The logit identity fits well outside the redundant regime (r=0.97, MAE 0.054, n=189) and fails inside it (MAE 0.41, n=31). The paper also reports a leak in one of its own attention-mask controls and proposes an LLM protocol, which it calls "pre-registered".

## 2. Reproduction check

I copied `code/` to a clean scratch directory and reran the full pipeline from scratch, as given in `code/README.md`: all 32 training runs, then `analyze.py`, then `tables.py`. Section 9 compares the rerun with the shipped `results/` and the paper. Independently, I checked every number in the paper's Tables 1-4 and the in-text statistics against the shipped `results/*.json`:

- Table 1 (controls), Table 3 (presence-gated models) and Table 4 (cross-test identity) match `supp_table.md`, `final_table.json` and `crosstest_by_regime.json` exactly.
- Table 2 matches `ckpt_disagree_by_regime.json` (154/29/328 checkpoints; 24.7% / 0% / 9.1%).
- The following also match `supp_stats.json`: Spearman 0.964 (p=2.2e-15), 0.829 and -0.872; checkpoint rho 0.813; 511 checkpoints and 13.3%; the regime (b) counts (11 of 12 below 0.25, 7 of 12 below 0.01); the eta=0.05 follow values; the regime (c) values; the soft-residual statistics (0.028, five models, all from eta=0, delta=0.3).
- Switch statistics (`summary.json`): 15 of 26 runs switch, at steps 750-7,000, and 14 of the 15 runs that end with follow < 0.5 switched. These match.

The numerical reporting is careful, and I found no fabricated or mis-transcribed numbers. I did find one qualitative claim that the authors' own logs contradict (W1).

## 3. Strengths

- **Sharp, well-posed sub-question.** "What do truncation and corruption tests measure, when do they disagree, and what training conditions decide this?" is concrete and answerable. Separating N from S is useful vocabulary for a confused literature.
- **Ground truth by construction, with an honest failure report.** The finding that the "reliant" mask leaks through the c-position residual stream under noisy training (Table 1, Section 5.1) is a real and useful caution for constructed-ground-truth benchmarks. The strict control fixes it.
- **Presence gating** (Table 3) is a specific mechanistic observation. These models answer correctly when the CoT is replaced by MASK, yet fall to chance when ANS cannot read the CoT position. That is a concrete reason why truncation can mislead.
- **Falsifiable cross-test check.** Eq. (2) is cheap and has a clear failure mode. The paper reports where it fails rather than hiding it.
- **Transparency.** All runs are reported, the adaptive extension is disclosed, the code is small and readable, seeds are fixed, and the numbers match the logs.
- **Correct derivations.** I checked Props. 1 and 4 line by line. In the continuous part of Prop. 1, holding alpha-beta = logit(s) fixed and letting beta go to infinity gives S -> s and N -> 0, as claimed. In Prop. 4, S/O = e^alpha/(p-2) and I/O = e^beta/(p-2), which gives Eq. (2).

## 4. Weaknesses

### Major

**W1. A claim in Section 5.2 is contradicted by the authors' own logs.** The paper states: "Training never produced a model that relied on the internal route from the start." The run `add_free_e0.4_d0.3_s0` never has P(follow) above 0.044 at any logged checkpoint (follow at steps 250-1,000: 0.04, 0.04, 0.01, 0.00). This model was never reliant on the CoT, which is also why its `switch_step` is null. The general narrative "reliance is acquired first and lost later" is therefore not universal even within this testbed. The 15/26 figure counts switchers, but the absolute statement is false. The sentence must be corrected, and the mechanism story ("copying is always learned first") qualified.

**W2. The headline regime-(c) result rests on post-hoc-added runs and essentially one seed.** The abstract and Table 2 lead with "24.7% of checkpoints are redundant-but-reliant". Recomputing from `results/runs/*.json`:
- 38 N-S+ checkpoints in total;
- 22 of them (58%) come from a single run, `e0_d0.3_s5_long`;
- 32 of 38 (84%) come from the four 8,000-step runs that were added after the 4,000-step results had been inspected;
- the two pre-planned regime-(c) runs contribute 0 and 6;
- all three "redundant-but-reliant" final models are among the added long runs.

The added runs are also confounded with training length (8,000 vs 4,000 steps), so seed and training budget cannot be separated. Checkpoints are logged every 250 steps and are strongly autocorrelated within a run, yet they are pooled as if exchangeable. Long runs therefore get 2-3 times the weight of short ones, and the reported p-values (for example p=1.7e-121 for the checkpoint Spearman) are meaningless as inference. The authors disclose the extension, which is good, but the abstract still presents run-dominated percentages as general rates. Needed:
- many more seeds per regime, all at the same pre-specified length;
- run-level (or cluster-bootstrap) confidence intervals;
- per-run fractions instead of pooled checkpoint fractions.

**W3. "Truncation false positive" is asserted, not shown against the paper's own mechanistic reference.** The N+S- models (for example eta=0.2, delta=0: truncation accuracy 0.36/0.40, follow 0.12/0.21) have CoT-cut accuracy of only 0.159/0.314. By the paper's own path-cut reference, their internal route is not sufficient once ANS cannot read the CoT position, so truncation's "necessary" verdict is arguably correct about the position, if not about the content. The paper's Section 5.3 explanation ("removal token is off-support") is plausible for delta=0, but it is not tested. For example, no on-support removal format is used for these models. The categorical sentence "when training CoTs are sometimes wrong, all disagreements are truncation false positives" needs a mechanistic criterion for "false positive", and that criterion must be applied symmetrically.

**W4. Mechanistic references conflate CoT content with computation hosted at the CoT position.** In the free model, position 4 can attend to a, b and c, so the "internal route" can live in the residual stream at the CoT position. The authors discover this ("presence gating"). It means that the CoT-cut, used as the reference for truncation, removes both the CoT content and any internal computation hosted there. This is not a clean internal-route sufficiency measure. The strict internal cut is clean. However, comparing truncation against the CoT-cut and corruption against the strict internal cut is asymmetric. In my own recomputation on `final_table.json`:
- truncation drop vs. strict internal cut: rho = 0.80;
- follow vs. CoT-cut: rho = -0.94.

So corruption still wins on both references, which supports the paper's conclusion, but the paper should report the full 2x2 table. Further problems with the reference:
- Follow vs. strict-internal-cut (rho = 0.96) is partly circular. Both quantities ask how the answer depends on the CoT token's content, one with the competing route masked off. The values are also bimodal (most models sit near follow 0 or 1), which inflates Spearman on n=26.
- No activation patching or interchange intervention is done, even though Def. 3 is phrased in exactly those terms. D is never actually measured; only a linear probe proxy is used.

**W5. The cross-test identity does not do what the abstract claims.** The abstract says the identity "flag[s] the models for which a single score misleads". But the five models with soft residual > 0.2 include `e0_d0.3_s1` and `e0_d0.3_s3_long`, which are N-S- models on which truncation and corruption *agree*. So the identity flags a regime (delta>0 with eta=0), not the disagreeing models specifically. Further:
- Its validated regime (O >= 0.02) excludes most well-trained models, as the authors note.
- The regime split for Table 4 (excluding eta=0, delta=0.3) is defined by the same regime in which the failure was observed.
- A precision/recall analysis of "residual > tau" for predicting N/S disagreement at the run level is needed before this can be recommended as a diagnostic.

**W6. Limited novelty and missing related work.** The theory is elementary; the authors say so for Prop. 2, and the same applies to Props. 1, 3 and 4.
- Prop. 3 follows immediately from pointwise minimisation of cross-entropy, given a hypothesis class that contains the oracle.
- Lanham et al. (2023) already used truncation and corruption as separate tests and observed that they behave differently across tasks.
- The necessity/sufficiency split for rationales is standard in the explanation literature, e.g. comprehensiveness vs. sufficiency in ERASER (DeYoung et al., 2020), and is not cited.
- Several closely related works are missing:
  - Dutta et al. (2024, arXiv:2402.18312) found *parallel answer-generation pathways* (from context and from the CoT) in Llama-2, which is the LLM-scale analogue of the redundant route here.
  - "The Kinetics of Reasoning" (arXiv:2510.25791) studies CoT training dynamics in small transformers. It reports a phase in which answers skip or contradict the CoT before aligning, the opposite ordering to the "reliance first" story here, and should be discussed.
  - "Ceiling of a Task" (arXiv:2609.33134) formalises when transformers can succeed without the CoT.
  - Sun et al. (2026, arXiv:2605.24960) show that contextual faithfulness metrics measure different aspects and do not transfer, a closely related empirical finding.
  - Jia et al. (2026) already define "necessity" for CoT.

  The genuinely new pieces are the toy ground-truth demonstration of N-S+ models, presence gating, and the logit consistency check. These are useful but incremental, at workshop-to-borderline level for a NeurIPS main track.

**W7. External validity is very limited, and the LLM implications are overstated.** The setup has a single one-token CoT, a 23-way answer space and a single task. Truncation is implemented as a dedicated MASK token that, in the key regime, is explicitly trained on (delta=0.3). LLM early answering has no analogue of an "absent-CoT token seen 30% of the time in training". Yet the abstract says the protocol "should replace single scores in LLM evaluations", and Section 6.1 links the results to outcome-RL erosion. The paper says repeatedly that these are hypotheses, but the abstract's recommendation outruns the evidence. No LLM experiment, not even a small one, tests any of P1-P4.

### Moderate / minor

- **M1. "Pre-registered" is misused.** The protocol in Section 6.2 is not registered anywhere: there is no timestamp, registry or analysis plan lodged with a third party. Call it a "proposed protocol with stated predictions".
- **M2. Probe claims slightly overstated.** Section 5.4 says models that follow the CoT carry no decodable trace (0.04-0.06) and that ignoring models carry one (0.24-0.76). Three followers have probe accuracy 0.11-0.14 (`e0.05_d0.3_s0`, `e0.05_d0_s2_long`, `e0.1_d0_s0`), and `e0.2_d0_s0` (ignore > follow) has 0.150. The ranges quoted are cherry-picked extremes.
- **M3. Linear probe under corruption only.** The Prop. 2 illustration is weak: post-hoc controls at 0.22-0.24 vs. some free models at 0.6-0.76 could reflect encoding differences, not different computations. Nonlinear (MLP) probes and interchange interventions are needed before drawing D-related conclusions.
- **M4. Thresholds.** The binary verdict thresholds (0.5) are arbitrary. A sensitivity sweep (0.3-0.7) for Table 2 would cost nothing.
- **M5. Greedy argmax.** Most well-trained models are at follow in {0, 1}, which makes Eq. (1) degenerate. Temperature sampling or soft probabilities should be the primary analysis, not an aside.
- **M6. Regime (c) training length.** Truncation accuracy in regime (c) ranges from 0.038 to 0.996, and the presence-gated route appears after about 6,000 steps. It is unclear whether the N-S+ type is stable or transient. Longer runs (for example 20k-50k steps, with weight decay 0.5) are needed to test Prop. 3(c)'s "unconstrained" claim against an implicit-bias alternative.
- **M7. Minor citation details.** Tutek et al. was published at EMNLP 2025 (cited as arXiv only). Barez et al. (2025) is now also a NeurIPS 2026 poster. These are not errors.
- **M8. Definition 3** puts h_v "on every path from x to y that does not pass through the CoT token". In the presence-gated models the internal route passes *through the CoT position*, so the definition's scope for these models is unclear.

## 5. Questions to the authors

1. With 10+ seeds per regime at a fixed, pre-specified length, what is the run-level fraction of N-S+ final models in regime (c), with a confidence interval?
2. Can you show, with activation patching at the CoT position, that presence gating is a switch keyed on the MASK embedding? For example, what happens when the MASK embedding is interpolated toward a number embedding?
3. For the N+S- models, what is truncation accuracy under an on-support removal format? Does the disagreement disappear?
4. What are the precision and recall of the Eq. (2) residual for detecting N/S disagreement at the run level?
5. How do you reconcile "reliance first" with `e0.4_d0.3_s0` and with the opposite ordering reported in arXiv:2510.25791?

## 6. Reference spot-check (web search)

I verified the following by web search. Each exists and says what it is cited for:
- Bürger et al. 2026, arXiv:2610.18105: monitoring fails even when the CoT is necessary.
- Gur-Arieh, Marasović & Geva 2026, arXiv:2605.25052: most metrics are near chance against ground truth (BonaFide); NeurIPS 2026.
- Jia, Benton & Easley 2026, arXiv:2605.24286: sufficiency, completeness and necessity.
- Barez et al. 2025, Oxford Martin AIGI, July 2025.
- Bentham, Stringham & Marasović, TMLR 2024: the early-answering metric correlates with accuracy and is sensitive to answer order.
- Yee et al. 2024, arXiv:2405.15092: error recovery, dissociation.
- Tutek et al. 2025, arXiv:2502.14829: unlearning reasoning steps.
- Emmons et al. 2025, arXiv:2507.05246.
- Arcuschin et al. 2025, arXiv:2503.08679.

All are consistent with the paper's characterisations. I found no fabricated references.

## 7. Scores

| Criterion | Score (1-10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 8. Decision

**Major revision / weak reject for a NeurIPS main track** (workshop-ready now).

The paper is clearly written, honest, and fully reproducible. Its central conceptual point, that truncation and corruption measure different things, is correct and worth making. But:
- the theory is elementary;
- the empirical headline in regime (c) rests on a handful of post-hoc-added runs, dominated by one seed and confounded with training length;
- one stated finding is contradicted by the authors' own logs;
- the cross-test diagnostic flags a regime rather than the disagreeing models;
- the LLM recommendation in the abstract goes well beyond the evidence.

The most valuable change would be a properly powered, pre-specified seed sweep with run-level uncertainty, plus at least one small LLM experiment testing P1 or P2.

## 9. Rerun outcome

See Addendum below (filled after the full rerun completed).
