# Response to Review r1 (p011, "Necessity Is Not Reliance")

We thank the reviewer for a careful review and for rerunning the pipeline. We agree with both fatal issues and with most of the major ones.

**Summary of changes.**

- We ran a new pre-specified seed sweep: 20 new runs and 44 saved checkpoints.
- We added new mechanistic analyses: interchange interventions for D, presence-gating tests, alternative removal formats, and a symmetric test-vs-reference comparison.
- We replaced Prop. 3 with a quantitative, ε-optimal version and checked it empirically.
- We corrected or withdrew every claim the reviewer showed to be unsupported.
- We added the missing related work.
- We did **not** run an LLM experiment. The reasons are under W8.

The round-1 paper is preserved as `reviews/paper_r1_snapshot.md`.

All new numbers come from code that was written and run in this environment:

- code: `code/run_r2.py`, `code/r2_analyze.py`, `code/r2_trainsupport.py`, `code/r2_bound.py`, `code/r2_stats.py`;
- outputs: `results/runs_r2/`, `results/r2_models_{old,new}.json`, `results/r2_stats.json`, `results/r2_trainsupport.json`, `results/r2_bound.json`, `results/fig5_runlevel.png`, `results/fig6_gating.png`.

---

## Fatal issues

### F1. "Training never produced a model that relied on the internal route from the start" (Section 5.2) is contradicted by run `add_free_e0.4_d0.3_s0`

**Agreed. Corrected.** That run's follow rate never exceeds 0.044 (`r2_stats.json: round1.never_reliant_runs`).

New Section 5.4 states the count across both rounds: P(follow) exceeds 0.5 at some checkpoint in 45 of 46 free runs (25/26 in round 1 and 20/20 in round 2). The exception is the η=0.4 run. The corrected claim reads: "in all runs with η ≤ 0.2, reliance was acquired before any loss of it; at η=0.4 one of four runs never acquired it." The abstract states 45 of 46.

### F2. "When training CoTs are sometimes wrong, all disagreements (9.1%) are truncation false positives" is not supported by the path cuts

**Agreed. The claim is withdrawn and replaced by what the evidence supports** (new Section 5.6).

1. **Symmetric mechanistic criterion.** We now classify every model by a mechanistic N (CoT-position cut costs more than 0.5 accuracy) and a mechanistic S (strict internal cut gives more than 0.5), using the same threshold as the behavioural verdicts. The two round-1 N+S− models (η=0.2, δ=0) are mechanistically N+S−:

   | Model | CoT-cut | Strict cut |
   |---|---|---|
   | Seed 0 | 0.159 | 0.322 |
   | Seed 1 | 0.314 | 0.250 |

   Truncation's verdict is therefore mechanistically correct. These are combined-route models, not false positives.

2. **The off-support explanation.** We now evaluate truncation under four removal formats: MASK, SEP, mean number embedding and zero embedding. For δ=0 models no on-support removal exists, and we say so explicitly. For the two N+S− models, truncation accuracy ranges over 0.21–0.57 across formats. The result is format-dependent, and that is all we claim.

3. **Replication.** The pre-specified sweep at (η=0.2, δ=0) found **0/10** N+S− final models at both 4k and 8k steps (Clopper–Pearson 95% CI [0, 0.31]). N+S− appears only transiently during training: in all 10 runs, with a mean per-run checkpoint fraction of 0.16 (95% CI over runs 0.09–0.25).

The abstract now reports this as a correction.

---

## Major issues

### W1. Missing closest prior work (Wang, Alazali & Zhong 2026); ERASER; Dutta et al. 2024

**Agreed. Added and positioned.** We verified all three by web search:

- arXiv:2602.01017, submitted 1 Feb 2026;
- ACL 2020, pp. 4443–4458;
- arXiv:2402.18312.

What changed in the paper:

- **Introduction ("Closest prior work") and Section 5.4.** We state that our "reliance acquired first and lost later, slower at small η" observation is consistent with, and largely anticipated by, Wang et al.'s stepwise→skip-step transition under noise. We credit that qualitative finding to them. Our characterisation of their paper is limited to what its abstract states.
- **Our distinct contributions are now stated narrowly:**
  1. the absent-vs-wrong (δ vs η) manipulation;
  2. the mechanistic account of the removal/editing disagreement (presence gating);
  3. a new finding from interchange interventions, which also sharpens the relation to Wang et al. Losing reliance on the CoT *token* is not the same as becoming skip-step. 10 of 16 behaviourally post-hoc models (follow ≈ 0, truncation ≈ 1) still route the answer through the internal intermediate computed at the CoT-generation position (IIA > 0.5; median 0.77). The post-hoc mask controls are mostly skip-step (IIA 0.14).
- **ERASER (Section 2).** Comprehensiveness is the analogue of N. ERASER sufficiency is the analogue of our strict-cut CoT-route sufficiency, not of S. S (direction of the answer under conflict) is asked by neither.
- **Dutta et al. (Sections 1, 2, 6.1).** Cited as LLM evidence of parallel answer pathways.
- **Jia et al.** Our claimed difference from them is narrowed to separating N from content-reliance under conflict, plus the mechanistic characterisation.

### W2/W3. The regime-(c) headline is driven by post-hoc 8,000-step runs; seed is confounded with length; checkpoints are pooled; p-values are meaningless

**Agreed. Addressed with a pre-specified sweep** (Section 4.3 and new Section 5.2, Table 2, Fig. 5).

**Sweep design.**

- Before launching, we wrote an analysis plan to `code/sweep_r2_spec.json`. It contains:
  - new seeds 100–109;
  - length as a factor (4k and 8k);
  - endpoints E1–E3;
  - the rule "the run is the unit; never pool checkpoints".
- We recorded its hash and time in `results/sweep_r2_spec.sha256`.
- **Disclosed amendment.** Measured throughput (about 1.8 effective cores) made the planned 40-run design take about 2 h. About 10 minutes after launch, before any checkpoint was evaluated, we cut two regimes ((0,0) and (0.2,0.3)). The amendment is recorded in the spec file. We call this an "author-specified plan", not a registration.

**Results** (run-level, 95% Clopper–Pearson):

| Regime | 4,000 steps | 8,000 steps | 16,000 steps |
|---|---|---|---|
| η=0, δ=0.3: N−S+ | 4/10 [0.12, 0.74] | 7/10 [0.35, 0.93] | 2/4 |
| η=0.2, δ=0: N+S− | 0/10 | 0/10 | — |

- The qualitative claim (N−S+ arises when CoTs are absent-but-never-wrong) **replicates on new seeds at the originally planned length.** The rate depends on training length: three seeds move N+S+→N−S+ between 4k and 8k.
- We confirm the reviewer's rerun observation that round-1 seed 4 would have been N+S+ at 4k (`round1.regc_long_quadrant_at_4000`).

**Pooling and p-values.**

- All checkpoint-pooled p-values are removed, including 1.7e−121 and 2.2e−15.
- Checkpoint-level quantities are now summarised per run:
  - regime (c): mean per-run N−S+ checkpoint fraction 0.40, 95% CI over runs 0.22–0.60, with 8/10 runs having any;
  - round-1 ρ(drop, follow) = 0.81, with cluster-bootstrap CI [0.62, 0.94].
- The 22-of-38 concentration in round 1 is reported (`round1.regc_NminusSplus_ckpts_by_run`).

### W4. "Truncation false positive" is asserted, not demonstrated

**Agreed.** See F2. The label is withdrawn, the symmetric criterion is stated and applied to both tests (Table 4), and the off-support explanation is reported as untestable in its strong form for δ=0 models.

### W5. Corruption's mechanistic validation is partly circular and inflated by bimodality; truncation is compared asymmetrically; D is never measured by interchange interventions

**Agreed on all points. Four changes** (Sections 5.3 and 5.5):

1. **Circularity acknowledged.** The text now says that follow and strict-cut accuracy both ask whether the answer is a function of the CoT token and c. Bimodality is quantified: only 27% of round-1 final models have follow in (0.1, 0.9). The p-value is removed. The remaining claim is narrower: the corruption verdict, measured with both routes open, matches the white-box verdict with the internal route silenced (agreement 0.92 in round 1 and 0.95 in round 2).
2. **Full 2×2 of tests against references** (Table 4). We report verdict agreement and Spearman correlation for:
   - truncation vs CoT-cut;
   - corruption vs strict cut;
   - truncation vs strict cut;
   - corruption vs CoT-cut.

   Truncation agrees with the positional CoT-cut for only 0.81 (round 1) and 0.65 (round 2) of models. Mismatches occur in both directions. In N−S+ models, truncation says "not necessary" while cutting the position destroys the answer. In two η=0.2 models at 4k, truncation accuracy is 0.80 and 1.00 while CoT-cut accuracy is 0.09 and 0.42.
3. **Presence gating tested directly** (also the reviewer's Q5).
   - **Embedding interpolation** from a corrupted token to MASK. In all 16 N−S+ models the answer moves from following the CoT to the correct answer, crossing ignore 0.5 at λ = 0.6–0.9.
   - **Activation patching.** Patching the layer-1 residual at the CoT position from the MASK run does not transfer the switch: ignore ≤ 0.09 in 15 of 16 models.
   - **Conclusion.** The gate is read directly from the CoT-position embedding by layer-1 attention elsewhere. We did not localise it to heads; this is stated as a limitation.
4. **D measured by interchange interventions**, as Def. 3 calls for. Def. 3 is rewritten to specify the site constraint and the CoT condition, which also answers M6.
   - **Method.** We copy the layer-1 residual at the c-free sites {0,1,3} from a source with different (s, c). The outcome s_src + c_base counts as success (IIA).
   - **Results.** Among 16 behaviourally identical internal-route models, IIA ranges 0.016–1.000. A random-subspace control gives ≤ 0.085, and the post-hoc controls give 0.14.

### W6. Eq. 2 barely beats the trivial predictor P(ignore); the residual flags a regime, not misleading models; coverage; circular split

**Agreed. Eq. 2 is demoted to a weak diagnostic** (Section 5.7, Table 5).

- **Baseline confirmed.** We reproduce the reviewer's numbers exactly. On 189 round-1 checkpoints, Eq. 2 gives r = 0.971 and MAE 0.054; the trivial predictor gives r = 0.968 and MAE 0.079.
- **New sweep.** On the round-2 η=0.2 checkpoints (n = 169, 10 runs), Eq. 2 gives 0.989/0.034 and the trivial predictor 0.988/0.048.
- **Run-level detection of behavioural N/S disagreement** by the soft residual:

  | Set | Precision | Recall |
  |---|---|---|
  | Round 2, 8k | 0.78 | 1.0 (7/7) |
  | Round 1 | 0.60 | 0.60 |

  False flags are regime-(c) N−S− models, so the residual flags the regime, as the reviewer says.
- **Abstract.** The claim that the residual identifies "the models for which a single score misleads" is removed. The abstract now says the identity "barely improves on the trivial predictor".
- **Circular split.** The split by regime is now presented only descriptively, alongside the baseline.

### W7. Prop. 3 is elementary; the softmax infimum is not attained; full-support is violated by held-out evaluation; Props. 1, 2 and 4 are short

**Agreed. Three changes** (Section 3.3):

1. **Relabelled.** Props. 1 and 2 are now "Observations" and are explicitly called elementary. Prop. 4 is kept and labelled as a one-line identity.
2. **Prop. 3 replaced by a quantitative statement for any predictor, not a minimiser.** With ε the answer cross-entropy on the training distribution:
   - (a) S_tr ≤ pε/((p−1)η);
   - (b) N_tr ≤ ε/δ;
   - (c) for η=0, every reliance value is reachable at arbitrarily small ε, so the softmax-infimum issue is handled explicitly.

   We computed the exact ε for all 38 η>0 models and checked the bound (`results/r2_bound.json`). It holds in every case and is loose by about 7–70×. We say so.
3. **Training support.** Prop. 3 is now stated for training-support inputs. We evaluated every model on training triples (`results/r2_trainsupport.json`). Training and held-out follow and truncation differ by at most 0.034 across 66 models and checkpoints, so the held-out evaluation is not what drives the comparison with Prop. 3.

We do not claim the theory is deep. It now does what it can: it gives a quantitative reason for the 1/η-slower erosion and a precise statement of what is left undetermined in regime (c).

### W8. External validity is minimal; the recommendation to replace single scores in LLM evaluations outruns the evidence; no LLM test of P1–P4

**Partly addressed. The claims are reduced. The experiment could not be run.**

- **Why no LLM experiment.** We attempted to obtain a small open-weights model. The environment has no GPU, and the network proxy blocks the model hub (`huggingface.co` returns 403). We therefore could not run any LLM experiment, and we do not describe one.
- **Abstract.** The sentence "should replace single scores in LLM evaluations" is removed. The abstract now says: "We ran no LLM experiments. The LLM protocol is a proposal with stated predictions, and nothing here establishes that the effects transfer."
- **Protocol (Section 6.2).** It is retitled "A proposed LLM protocol with stated predictions (not run)".
  - P1 is rewritten around removal-format dependence. This is the property our new Section 5.6 shows the N−S+ verdict depends on: with SEP, mean or zero removal, the regime-(c) N−S+ models fall to chance.
  - P4 now predicts the S≠D dissociation.
- **Limitations.** The trained MASK token, which has no LLM analogue, is listed as a specific threat to transfer.

---

## Minor issues

| # | Issue | Change |
|---|---|---|
| M1 | "Pre-registered" misused | Replaced everywhere by "proposed protocol with stated predictions" (Section 6.2) and "author-specified analysis plan, not lodged with an external registry" (Section 4.3). |
| M2 | Probe ranges cherry-picked | The Section 5.4 probe ranges were removed. The full sorted lists are in `r2_stats.json: probes_r1`: followers 0.041–0.138, ignorers 0.150–0.756. The reviewer's values are confirmed. The D discussion now relies on interchange interventions rather than probes. |
| M3 | "CoT route alone suffices" overstated (seed 2 strict cut 0.652) | Claim removed from the abstract. Table 3 reports 0.652. |
| M4 | Greedy argmax makes Eq. 1 degenerate | Section 5.7 reports the soft-probability residual as the model-level detector (primary). The checkpoint table is kept for comparison with round 1. |
| M5 | Is N−S+ stable under longer training? | Seeds 100–103 were run to 16k. Of the three that were N−S+ at 8k, two stayed N−S+; seed 102 lost reliance (follow 0.626→0.202). Section 5.2 now says N−S+ is not shown to be stable. 20k–50k runs were not done (compute); this is stated in the Limitations. |
| M6 | Def. 3 scope for presence-gated models | Def. 3 now names the site constraint (sites carrying (a,b)- but not c-information) and is evaluated under a stated CoT condition (corrupted or absent). For presence-gated models it applies under the absent condition. |
| M7 | Citation venues | Tutek et al. → EMNLP 2025, pp. 9935–9960. Barez et al. → also NeurIPS 2026 poster. Both verified by web search. |
| Positive note | Threshold robustness | The 0.3–0.7 threshold sweep is now reported (Section 5.2; `round1.threshold_sweep`): regime (c) N−S+ 22–27% and N+S− 0%; η>0 N−S+ 0% and N+S− 9.1–12.8%. |

## Questions to the authors

1. **Run-level N−S+ fraction with 10 seeds:** 4/10 [0.12, 0.74] at 4k and 7/10 [0.35, 0.93] at 8k.
2. **Eq. 2 vs P(ignore):** see W6. The gain is about a 30% MAE reduction and no gain in correlation.
3. **On-support removal for η>0, δ=0:** none exists in this testbed. Alternative formats change truncation accuracy by up to 0.35 for the N+S− models, and the disagreement did not replicate in final models (F2).
4. **Relation to Wang et al.:** see W1. The dynamics are theirs. The S≠D dissociation (losing token reliance without becoming skip-step) and the absent-vs-wrong manipulation are ours.
5. **Presence gating with patching:** see W5(3). Interpolation of the token embedding shows the switch. Patching the layer-1 residual at the CoT position does not transfer it, which localises the gate to direct reads of the embedding.

## Claims removed or weakened

- "Training never produced a model that relied on the internal route from the start" → removed (45/46).
- "All η>0 disagreements are truncation false positives" → withdrawn. The models are combined-route models, and the effect does not replicate in final models.
- "24.7% of checkpoints" as a headline → replaced by run-level rates with CIs.
- "Reliance by corruption tracks mechanism (ρ = 0.96, p ≈ 2e−15)" → reported as partly definitional agreement, without a p-value.
- "The cross-test identity flags the models for which a single score misleads" → removed. It is now a weak diagnostic next to its trivial baseline.
- "Should replace single scores in LLM evaluations" → removed.
- "Pre-registered" → removed.
- "Their CoT route alone suffices" → removed.
