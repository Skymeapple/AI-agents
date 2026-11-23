# Review (Round 1): "Necessity Is Not Reliance: Disentangling Chain-of-Thought Faithfulness Tests with Ground-Truth Transformers" (p011)

Reviewer role: independent area-chair-level reviewer. Venue as stated by the authors: NeurIPS 2027 main track.
Research question: whether a model's chain of thought (CoT) faithfully reflects its actual reasoning process.

## 1. Summary

The paper argues that the two standard behavioural CoT-faithfulness tests measure different properties:

- **Truncation / early answering** measures **necessity N**: the accuracy drop when the CoT is removed.
- **Corruption / adding mistakes** measures **reliance S**: the probability that the answer follows a corrupted CoT.
- Neither test measures **description D**: whether the internal computation passes through the stated intermediate, defined as a causal-abstraction alignment.

**Theory.** There are four propositions:

1. N and S are logically independent.
2. D is not identifiable from input-output behaviour.
3. A cross-entropy-optimal predictor is CoT-invariant when erroneous CoTs are on-support, and S is left unconstrained when CoTs are only ever absent (never wrong).
4. Under a two-route multinomial-logit model, the three-way corruption outcome predicts truncation accuracy (Eq. 2).

**Testbed.** 2-layer, width-64 transformers are trained on s=(a+b) mod 23, y=(s+c) mod 23, with a one-token CoT. Training varies η (the rate of random CoT tokens) and δ (the rate of a MASK "absent CoT" token). Attention-mask controls provide positive and negative ground truth. Path cuts and linear probes provide mechanistic references.

**Main claims.**

- The two tests disagree on 13.3% of 511 checkpoints, and the direction of disagreement is set by the training regime:
  - In regime (c), η=0 and δ=0.3, 24.7% of checkpoints are N−S+ ("redundant-but-reliant").
  - With η>0, 9.1% of checkpoints are N+S−.
- Corruption tracks a mechanistic CoT-route reference with ρ=0.96 (n=26).
- Eq. 2 fits outside regime (c) (r=0.97, MAE 0.054, n=189) and fails inside it (MAE 0.41, n=31).
- One attention-mask control leaked.
- The paper also proposes a "pre-registered" protocol for LLMs.

## 2. Reproduction

**Static check of the shipped outputs.** Every number I checked in Tables 1–4, the abstract and Sections 5.2–5.5 matches the shipped `results/*.json` and `*.md` files. This covers:

- the Table 1 controls, and the leak figures 0.998 (non-strict cut) → 0.177 (strict cut);
- Table 2: 154/29/328 checkpoints, and 24.7%, 0% and 9.1%;
- Table 3, the three presence-gated models;
- Table 4: 0.83/0.104, 0.97/0.054/0.5%, and 0.89/0.409/58.1%;
- the Spearman correlations 0.964 (p=2.2e-15), 0.829 and −0.872, and the checkpoint-level ρ=0.813;
- 11/12 runs with follow below 0.25 and 7/12 below 0.01;
- the regime (c) values;
- the soft-residual statistics: 0.028, and five large-residual models, all in regime (c);
- the switch statistics: 15 of 26 runs switch, at steps 750–7,000.

**Full rerun.** I copied `code/` to a clean scratch directory and reran the complete pipeline as given in the README: 32 training runs, then `analyze.py`, then `tables.py`. Section 9 reports the outcome.

## 3. Strengths

- **A clear, useful conceptual separation.** The distinction between "is there another route when the CoT is absent?" (N) and "does the answer follow the CoT when it is present?" (S) is correct and practically relevant. The four-quadrant construction in Prop. 1 is a clean way to make the point.
- **Ground truth by construction, reported honestly.** The leak in the "reliant" mask (Table 1) is a good cautionary finding for constructed-faithfulness benchmarks. The authors located it with the strict cut and closed it.
- **Presence gating** (Table 3) is a concrete mechanism behind N−S+. These models answer correctly when the CoT is replaced by MASK, yet fall to chance when ANS cannot read the CoT position. It shows how truncation can probe a computation that never runs when a CoT is present.
- **Robust to the verdict threshold.** I swept the binary-verdict thresholds from 0.3 to 0.7 on the shipped histories. The direction asymmetry in Table 2 holds at every threshold: regime (c) has 22–27% N−S+ and 0% N+S−, while η>0 has 0% N−S+ and 9–13% N+S−.
- **Transparent and reproducible.** The code is small and readable, seeds are fixed, all runs are reported, and the adaptive extension is disclosed.

## 4. Weaknesses

### Major

**W1. Closely related prior work is not cited, and it substantially reduces the novelty of the training-dynamics results.** Wang, Alazali & Zhong (2026, arXiv:2602.01017), "How Does Unfaithful Reasoning Emerge from Autoregressive Training? A Study of Synthetic Experiments", does the following:

- trains small transformers from scratch on step-by-step modular arithmetic with noisy training data (numbers replaced by random values);
- measures faithfulness by counterfactual intervention on the reasoning steps;
- reports that faithful stepwise reasoning emerges only below a critical noise threshold;
- reports that training moves from faithful stepwise reasoning, through a mixed phase, to unfaithful skip-step reasoning, which it attributes to simplicity bias.

This is essentially the same finding as Section 5.2 of the submission: reliance is acquired first and lost later, and convergence slows as η shrinks. The setting is also nearly identical. The paper must cite and position against this work. After doing so, the genuinely new pieces are:

- the N/S separation itself, which is conceptually simple;
- the δ (absent-CoT) knob, with the N−S+ / presence-gating observation;
- the Eq. 2 consistency check.

Other omissions:

- the comprehensiveness/sufficiency distinction from the rationale-evaluation literature (DeYoung et al., 2020, ERASER), which is the direct ancestor of N vs. S;
- Dutta et al. (2024), who found parallel answer pathways in LLM CoT;
- Jia et al. (2026), who already define necessity for CoT. This one is cited, but the overlap is understated.

**W2. A stated finding is contradicted by the authors' own logs.** Section 5.2 says: "Training never produced a model that relied on the internal route from the start." Run `add_free_e0.4_d0.3_s0` has P(follow) ≤ 0.044 at every logged checkpoint: 0.04, 0.04, 0.01 and 0.00 over the first four, final 0.006. It was never reliant, which is why its switch step is null and why the paper's own count is "14 of the 15" runs. Unless this sentence is corrected, it is a claim not supported by the run outputs. The "reliance always first" mechanism story must be qualified.

**W3. The headline regime-(c) statistic rests on post-hoc-added runs and essentially one seed, and checkpoints are treated as independent.** Recomputing from `results/runs/*.json`:

- The 38 N−S+ checkpoints split as 22 from `e0_d0.3_s5_long`, 6 from `e0_d0.3_s1`, and 4, 3 and 3 from the other long runs.
- So 84% come from the four 8,000-step runs that were added after the 4,000-step results had been inspected, and 58% from one run.
- All three "redundant-but-reliant" final models are among the added runs.
- Seed and training length are confounded: the added runs are 8,000 steps, the planned ones 4,000.

Checkpoints logged every 250 steps are strongly autocorrelated, yet they are pooled as exchangeable units. The p=1.7e−121 for the checkpoint-level Spearman correlation is therefore meaningless as inference, and percentages like "24.7%" in the abstract are not rates of anything well defined. Needed:

- at least 10 seeds per regime at a pre-specified length;
- run-level fractions with confidence intervals, or a cluster bootstrap over runs;
- training length treated as a separate factor.

**W4. "Truncation false positive" (N+S−) is asserted, not demonstrated, and the paper's own mechanistic reference disagrees.** The abstract says that with η>0, "all disagreements are truncation false positives". The final N+S− models (η=0.2, δ=0, both seeds) have:

| Measure | Seed 0 | Seed 1 |
|---|---|---|
| Truncation accuracy | 0.36 | 0.40 |
| Follow | 0.12 | 0.21 |
| CoT-cut accuracy | 0.16 | 0.31 |
| Strict internal-cut accuracy | 0.32 | 0.25 |

Neither route alone suffices. These models combine the CoT position and the internal route, so truncation's verdict that the CoT position is needed is mechanistically correct. What fails is the corruption test's notion of following the CoT's content. The "off-support MASK" explanation is plausible but untested: no on-support removal format is evaluated for δ=0 models. A mechanistic criterion for a false positive must be stated and applied symmetrically to both tests.

**W5. The validation of corruption against mechanism is partly circular and statistically inflated.** The ρ=0.96 compares two quantities:

- P(follow), the answer tracking a corrupted CoT token;
- strict internal-cut accuracy, the answer computed correctly when a and b are hidden and only the CoT token and c remain.

Both measure essentially the same property: whether the answer is a function of the CoT token and c. The comparison is not an independent ground truth. The 26 values are also strongly bimodal (follow ≈ 0 or ≈ 1), which inflates rank correlation; the p≈2e−15 should not be quoted. The comparison is also asymmetric: truncation is compared against the CoT-cut, which removes the CoT content *and* any internal computation hosted at the CoT position. The authors themselves show that presence gating lives exactly there. The full 2×2 of tests vs. references should be reported. Def. 3 is phrased in terms of interchange interventions, but none is performed: D is never measured, only a linear-probe proxy.

**W6. The cross-test identity (Eq. 2) adds little beyond a trivial baseline, and it flags a regime rather than misleading models.**

- **Trivial baseline.** I computed the naive predictor P̂_trunc = P(ignore) on the same 189 checkpoints. It gives Pearson r=0.968, against r=0.971 for Eq. 2; the MAE is 0.079 vs. 0.054. In regime (c) it gives r=0.82 and MAE 0.48, against 0.89 and 0.41. Almost all of the reported agreement therefore comes from "truncation accuracy ≈ ignore rate", which needs no logit model. The paper must report this baseline.
- **What the residual flags.** The five large-residual models include `e0_d0.3_s1` and `e0_d0.3_s3_long`. Both are N−S− models on which the two tests *agree*. The residual flags "δ>0, η=0" as a regime, not the models "for which a single score misleads", as the abstract claims.
- **Coverage.** The check needs O ≥ 0.02, which excludes most well-trained models.
- **Circular split.** The split in Table 4 is defined by the same regime in which the failure was found.

Precision and recall of the residual for detecting N/S disagreement, at the run level, are needed before the check can be recommended.

**W7. Prop. 3 is elementary, and its application to the experiment is loose.**

- Part (a) is the textbook fact that the unique minimiser of cross-entropy on deterministic labels is the point mass on the support. Parts (b)–(d) follow by inspection.
- Two technical points:
  - A softmax network never attains the point mass, so "minimiser over a class that contains δ_{y*}" describes an infimum the transformer class does not achieve.
  - Evaluation is on held-out triples (a 20% split). For those x, the pairs (x, z′) are off the training support even when η>0. The proposition's "full support of x" assumption is therefore violated in the experiment, and the link to Section 5 relies on generalisation, not on the proposition.
- Props. 1, 2 and 4 are correct; I checked them line by line. For Prop. 1, with α−β=logit(s) fixed and β→∞, S→s and P_clean, P_trunc→1. For Prop. 4, S/O=e^α/(p−2) and I/O=e^β/(p−2). But all four are short observations, not results that carry a NeurIPS main-track paper.

**W8. External validity is minimal, and the recommendation to the LLM community outruns the evidence.** The setup has:

- a one-token CoT and a 23-way answer space;
- a single task;
- a dedicated MASK token for "absent CoT" that, in the key regime, is seen in 30% of training examples.

LLM early answering has no such trained absence token. The abstract still states that the protocol "should replace single scores in LLM evaluations", and no experiment, not even a small open-weights LLM, tests any of P1–P4.

### Minor

- **M1. "Pre-registered" is misused.** Nothing is registered with a third party. Call it a "proposed protocol with stated predictions".
- **M2. Probe ranges are cherry-picked.** Section 5.4 says followers have probe accuracy of 0.04–0.06 and ignorers 0.24–0.76. But:
  - followers `e0.05_d0.3_s0`, `e0.05_d0_s2_long` and `e0.1_d0_s0` have 0.11–0.14;
  - `e0.2_d0_s0` (ignore > follow) has 0.150.
- **M3. "Their CoT route alone suffices" is overstated.** The abstract says this of the three N−S+ models, but seed 2's strict cut is 0.652.
- **M4. Greedy argmax makes Eq. 1 degenerate for most final models.** The soft-probability analysis should be primary, not an aside.
- **M5. Regime (c) may be transient.** The presence-gated route appears after about 6,000 steps. It is unknown whether N−S+ survives longer training under weight decay 0.5. Prop. 3(c)'s "unconstrained" claim should be tested against implicit-regularisation alternatives with runs of 20k–50k steps.
- **M6. Definition 3's scope is unclear.** It requires h_v "on every path not passing through the CoT token". In presence-gated models the internal route runs through the CoT *position*, so it is unclear how the definition applies to them.
- **M7. Citation details.** Tutek et al. appeared at EMNLP 2025. Barez et al. also appears as a NeurIPS 2026 poster.

## 5. Questions to the authors

1. With 10 or more seeds per regime at a fixed length, what is the run-level fraction of N−S+ final models in regime (c), with a confidence interval?
2. How does Eq. 2 compare with the naive predictor P̂_trunc = P(ignore) across all regimes? What do you gain from the logit model?
3. For the η>0, δ=0 N+S− models, what does truncation give under an on-support removal format? Does the disagreement disappear?
4. How do your Section 5.2 dynamics differ from Wang, Alazali & Zhong (2026)'s stepwise → skip-step transition under noise?
5. Can you show with activation patching or interchange interventions, rather than attention cuts, that presence gating is a switch keyed on the MASK embedding?

## 6. Reference spot-check (web search)

I verified the following by web search. Each exists, and the paper's characterisation is consistent with the source:

- Gur-Arieh, Marasović & Geva (2026), arXiv:2605.25052. BonaFide benchmark; most metrics near chance. NeurIPS 2026 poster.
- Jia, Benton & Easley (2026), arXiv:2605.24286. Sufficiency, completeness and necessity; prompt→CoT→answer information flow.
- Bürger et al. (2026), arXiv:2610.18105. Monitoring fails even when CoT is necessary; hierarchy of necessity levels.
- Bentham, Stringham & Marasović (2024), TMLR, arXiv:2402.14897. Early-answering metric correlated with accuracy; sensitive to answer order.
- Tutek et al. (2025), arXiv:2502.14829 (EMNLP 2025). Unlearning reasoning steps.
- Emmons et al. (2025), arXiv:2507.05246.
- Barez et al. (2025). Oxford Martin AIGI, July 2025.
- Yee et al. (2024), arXiv:2405.15092.

I found no fabricated or mischaracterised references. The problem is missing references (W1), not wrong ones.

## 7. Novelty assessment

- Behavioural truncation and corruption tests are from Lanham et al. (2023).
- The necessity/sufficiency split mirrors comprehensiveness/sufficiency from rationale evaluation.
- Toy-transformer studies of how noisy CoT training produces unfaithful (skip-step) reasoning already exist (Wang et al., 2026).
- Meta-evaluation of faithfulness metrics against ground truth exists for LLMs (Gur-Arieh et al., 2026).

What remains new:

- an explicit absent-vs-wrong (δ vs. η) manipulation;
- the N−S+ presence-gated models;
- the observation that the direction of test disagreement depends on regime;
- a consistency statistic whose added value over a trivial baseline is small (W6).

This is a useful, incremental contribution at workshop or borderline level.

## 8. Scores and decision

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 4 |
| Correctness | 5 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision (weak reject at a NeurIPS main track; suitable for a workshop now).**

The paper is clear, honest and reproducible, and its central conceptual point is right. But:

- it omits the closest prior work, which already establishes the training-dynamics story;
- one empirical statement is contradicted by its own logs;
- the regime-(c) headline is driven by a few post-hoc runs, one of which dominates, with checkpoint pseudo-replication;
- the "false positive" label and the mechanistic validation are partly circular;
- the cross-test identity barely beats P̂_trunc = P(ignore);
- there is no LLM evidence for a recommendation aimed at LLM evaluation.

**Single most valuable change:** a pre-specified, properly powered seed sweep with run-level uncertainty, plus at least one small open-weights LLM experiment testing P1/P2, positioned explicitly against Wang et al. (2026).

## 9. Rerun outcome

I ran the full pipeline from a clean copy, with the commands from `code/README.md`: 32 runs at 4 in parallel, then `analyze.py`, then `tables.py`. It took about 45 minutes on 4 cores.

- Every regenerated output is **byte-identical** to the shipped outputs: `supp_stats.json`, `summary.json`, `ckpt_disagree_by_regime.json`, `crosstest_by_regime.json`, `final_table.md` and `supp_table.md`.
- `final_table.json` has a maximum absolute difference of 0 over all float fields.
- All figures were regenerated by the same code.

**The paper's numbers are therefore backed by code that runs and is deterministic.** No numerical mismatch was found.

The rerun histories also confirm a point that strengthens M5 and W3. In `e0_d0.3_s4_long`, truncation accuracy stays at 0.03–0.05 until step about 5,750, then rises to 0.61 by step 7,750. At the original 4,000-step budget this run would have been classified N+S+, not N−S+. The N−S+ classification is therefore a function of the post-hoc choice of an 8,000-step budget.

Two claims are contradicted by or unsupported in the run outputs, as qualitative statements rather than numbers:

- the "never relied on the internal route from the start" sentence (W2);
- "all η>0 disagreements are truncation false positives", which the authors' own path cuts do not support (W4).
