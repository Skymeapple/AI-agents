# Review of p009, Round 2: "How Much State Does In-Context Recall Need? A Superposition-Capacity Budget for Fixed-State Sequence Models, Checked on Trained Mixers and on Real Token Streams"

Reviewer: independent area-chair-level review, round 2. I did not write this paper.
Research question: can new architectures (SSMs, hybrids, ...) replace the Transformer?
Materials read: `paper.md` (Revision 1), `reviews/review_r1.md`, `reviews/response_r1.md`, all of `code/`, and `results/`.

---

## 1. Summary

The paper builds a state budget for one capability, single-hop, previous-token-keyed associative recall, and tests each link of it at CPU scale. The four parts:

- **A. Capacity of additive outer-product memory with an argmax read-out over V codes.** A Gaussian integral (Eq. 1) and a closed form, N\* ≈ d_k/(2 ln V (1/d_v + 1/V)) (Eq. 2). The revision adds:
  - a fan-out term (Corollary 1), for keys that recur with several values;
  - a frame-potential "ceiling" on how much learned codes can beat random ones (Corollary 2);
  - Monte Carlo CIs and a check at V = 41,509.
- **B. Trained single-layer mixers (linear, GLA, DeltaNet, Gated DeltaNet).** 3 seeds on a pre-bound, write-masked MQAR task. They reach 1.00–1.07× the simulated ideal additive memory. A new overwrite task (B′) shows the delta rule winning by about 3×.
- **C. Real-text demand (N_ctx, ρ).** Measured on 18 Gutenberg books (word-level and BPE) and the Go standard library. Held-out context-specificity criteria D0–D4 and a document-level cluster bootstrap.
- **D′. Replay (new).** Real windows are written into *ideal random-code* memories with corpus-sized vocabularies. The load-aware Eq. (1) tracks measured recall (median MAE 2.3 points). The naive budget S_req = 2 ln V · N_ctx is optimistic by 1.3–14× for oracle writing, because of key fan-out. A d_v = 64 value path never reaches 50% recall for T ≥ 4k.

The paper now frames "fixed vs. growing memory decides replacement" as a pre-registered hypothesis (P1–P5). It is untested, because no pretrained or trained LM is used.

## 2. Status of round-1 issues

| Round-1 issue | Status |
|---|---|
| F1: "8 of 18 books" vs `n_docs_eligible` = 12 | **Resolved.** The text now says 12. I confirmed 12 eligible and 12 sampled at T = 32,768 in `demand_v2.json`. |
| F2: abstract attributed Eq. (1)'s accuracy to Eq. (2) | **Resolved.** Table 1 separates Eq. (1), rule 2, rule 2 with s²/d_v kept, and rule 2-med; the abstract is corrected. I checked the decomposition numerically (√(2 ln 512) = 3.53 vs exact median 3.00; ratio² = 1.39). |
| F3: "at least linear" contradicted for code | **Resolved.** Exponents are reported per corpus and criterion with bootstrap CIs, and match the JSON. |
| W1: no pretrained-model evidence; untested extrapolations | **Partly addressed.** The replay tests the ln V and demand links, but only for ideal random-code memories. The η ≈ 1 link is still tested only on 1-layer pre-bound tasks. Claims are now properly scoped. The decisive gap remains (see M1). |
| W2: weak "not top-1" criterion; leakage | **Resolved well.** Two-fold held-out statistics, D0–D4, cluster bootstrap. Demand drops by 29–42% (N_ctx) and about 40–48% (ρ), as I suspected. |
| W3: 4,000× KV ratio presented as a finding | **Mostly resolved.** The 4,000× figure is gone. A 400–1000× figure stays in the abstract, with a "not like-for-like" caveat (see m4). |
| W4: thin statistics | **Largely resolved for A, B and C.** 200-trial CIs, 3 seeds with ANOVA, and a dense evaluation grid. **New gap:** the replay (D′) uses 5 windows per cell with no uncertainty, and the overwrite baselines use 30 trials (see M3). |
| W5: "regardless of update rule" built into the task | **Resolved.** The claim is scoped, and the overwrite task is a good addition showing where the update rule matters. |
| W6: missing associative-memory literature | **Resolved for the works I named.** But closely related 2026 work is still missing (see M2). |
| Minor: Welch bound → frame potential, s²/d_v decomposition, Eq. (1) grid inconsistency, Nemotron denominators, gate stats, BPE/no-KJV | All addressed. |

The authors' response is honest and thorough. Several claims were weakened rather than defended, which I credit.

## 3. Reproduction (this round)

I copied `code/`, `data/` and `results/` to a scratch directory and reran from there, so the authors' outputs were not overwritten. The machine was heavily shared, so runs were slow.

1. **`train_mqar.py`, revision runs.** Two runs, at the paper's settings:
   - `delta --dk 16 --seed 2 --dense_eval 1` (retain-everything) gives an accuracy vector **bitwise identical** to `results/train_seeds/delta_dk16_dv64_s2.json`, with β = 0.1150 (paper: 0.115).
   - `delta --dk 16 --seed 1 --task overwrite` gives a **bitwise identical** accuracy vector, 0.902 at P = 16, β = 0.339 (paper: 0.90, 0.34).
2. **`recall_demand_v2.py`** (Exp. C, all four corpora). Reproduces `demand_v2.json`. Examples:
   - N_ctx D3 at 32k: 2,190 (prose), 2,117 (BPE), 2,120 (no KJV), 1,821 (code).
   - Exponents and CIs identical, e.g. code D3 global 1.011 [0.989, 1.034].
3. **`capacity_ci.py`** (Exp. A, revision): all 8 V = 512 cells reproduce `sim_capacity_ci.json` **exactly** (identical N\* and bootstrap CIs, e.g. (16, 64): 87.944 [87.23, 88.85]); 3 of the 4 large-V cells also reproduce exactly (52.10, 52.84, 105.65) — I stopped the last cell (d_k = 32, Zipf) for time on the shared machine.
4. **`replay_memory.py`** (Exp. D′): full rerun (both corpora, 3 T, 5 windows, all S, both shapes, D0/D3). Every record and summary field is **identical** to `replay_memory.json` apart from wall-clock timing.
5. **Recomputing every table from the JSON outputs** (`analysis_v2.json`, `sim_capacity_ci.json`, `demand_v2.json`, `replay_memory.json`, `train/`):
   - **Tables 1, 2, 2b, 3 and 5** match to the reported precision, including N\* per seed, η_sim ranges, ANOVA p = 0.95 / 0.65 / 0.001, slopes 1.07 / 1.06 / 1.03 / 1.07, overwrite accuracies and gates, D0/D3/D4 T\* values and the † markings.
   - **Table 4** matches, with two trivial rounding slips: code T = 1k oracle S_half = 1.35·10⁴ is printed as 1.4·10⁴, and ratio 13.95 is printed as 14.0.
   - **Derived statements.** The "≈ 11k (prose) / ≈ 6k (code)" log-interpolation for a square 2^16 state, the KV ratios (1.34·10⁸ / 1.3·10⁵ ≈ 1,030; / 3.2·10⁵ ≈ 420), and Prop. 2's 4.6 S bound with 0.086 S ≈ 2% are all correct.
   - **Small text–output slip.** "D0 → D3 removes 41–48% of ρ": prose at T = 2,048 is 40%. Trivial.
6. **Figure** `fig_revision.png` panels (a)–(c) are consistent with the tables.

I found **no unbacked or mismatched results**. Reproducibility remains exemplary: seeds are fixed, reruns are bit-exact, and every number is traceable.

Code-level oddities (not affecting the paper's claims):

- **Inverted T\* CI ranges.** `analysis_v2.json` contains inverted "range_ci" for extrapolated T\* (for example prose D3 at 2^18: [153,363, 152,285]). One D0 range at 2^20 does not contain its estimate. The paper does not quote these ranges, but the bootstrap-on-extrapolation code is wrong and should be fixed or deleted.
- **Wrong output names in the docstring.** `analyze_v2.py` says it writes `fig_seeds.png`, `fig_overwrite.png` and `fig_replay.png`; it actually writes `fig_revision.png`.
- **Mixed interpolation grids in the width-control η values.** The single-seed width-control runs (d_v = 32, d = 512) use N\* interpolated on the old factor-2 grid, which the paper itself shows is biased up by about 2%. But their η_sim is computed against the dense-grid simulation. So η = 1.17 (d_v = 128) and 1.09 (DeltaNet, d_v = 32) are likely about 2% high. Correcting this would bring DeltaNet at d_v = 32 to about 1.07, roughly at its stated ceiling of 1.06.

## 4. Reference check (web-verified this round, prioritising references added in revision)

| Reference | Verified | Notes |
|---|---|---|
| Giorlandino, Goldt & Maillard 2026, arXiv:2605.10795 / ICLR 2026 | Yes | arXiv title "Factual recall in linear associative memories: sharp asymptotics and mechanistic insights". ICLR 2026 version "Sharp storage capacity of a simplified model of linear associative memory". Threshold p log p / d² = 1/2 for the decoupled model. Described correctly. |
| Barnfield, Kim, Nichani, Lee & Lu 2026, arXiv:2605.05189 | Yes | The top-1 threshold d² ~ 2 n log n is conjectural in v1 and stated as sharp in the later version, whose subtitle is "From Winner-Take-All to Listwise Retrieval", as the paper notes. Described correctly. |
| Okpekpe & Orvieto 2025, arXiv:2508.19029 | Yes | Learning-rate sensitivity of recurrent models, width helps most, a tuned single-layer Mamba solves recall. Described correctly. (v1 title: "When recalling in-context, Transformers are not SSMs".) |
| Nichani, Lee & Bietti, ICLR 2025, arXiv:2412.06538 | Yes | Storage capacity linear in parameter count; linear-attention analysis. Correct. |
| Cabannes, Dohmatob & Bietti, ICLR 2024, arXiv:2310.02984 | Yes | Scaling laws for outer-product associative memories with heavy-tailed data. Correct. |
| Benedetto & Fickus 2003, Adv. Comput. Math. 18, 357–385 | Yes | Finite normalized tight frames; frame-potential minimisers. Correct use of the bound Σ⟨k_i,k_j⟩² ≥ K²/d. |
| McEliece et al. 1987; Amit et al. 1985; Kohonen 1972; Merrill et al. 2024 | Consistent with my knowledge | Standard; bibliographic details look correct. |
| Round-1 references (Lufkin, Qiao, Wang, Guo, Kimi, Nemotron-H) | Verified last round | Nemotron denominators now fixed. |

I found no fabricated or misdescribed references.

## 5. Novelty and missing prior work

**Theory.** The authors now position Prop. 1 correctly as a specialisation of the classical analysis. I note that, for square states, Eq. (2) *is* the Barnfield et al. / Giorlandino et al. sharp threshold, n log n ≈ d²/2. What is genuinely added:

- the explicit 1/V term and the lag-resolved decay horizon;
- the fan-out corollary, which is simple but useful and is borne out by the replay;
- a second-moment frame-potential ceiling.

All three are incremental.

**Missing, closely related 2026 work.** These appeared before this revision and should be discussed:

- **Koren, Ben-Kish, Giryes, Wolf & Zimerman, "On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing"** (arXiv:2609.07681; NeurIPS 2026). It derives recall scaling laws that predict, from a model's state size, context length and embedding dimensions, the largest vocabulary that can be perfectly recalled. It reports that *trained nonlinear Mamba* follows the linear model's scaling. This overlaps directly with contribution 1 and with the paper's "budget" framing. It also partially answers the paper's main open question (do trained models reach the law?) for a production architecture.
- **Boesch & Wee, "Anatomy of Associative Recall in Fixed-State Recurrences"** (arXiv:2609.16183). It gives matched-state comparisons of delta-rule vs. diagonal transitions with an "interference wall". This is relevant to Sections 5.2–5.3 and to the claim that no update rule beats additive storage at matched state.
- Work showing that trained linear-attention states are low-rank (for example arXiv:2602.04852) bears on the paper's nominal-vs-effective-state point.

**What is novel.** The empirical bridge from measured real-text demand to a state budget, and the replay showing that key fan-out (not total association count) dominates, are the paper's most original contributions. They are interesting. But they are validated only for an idealised memory with one random key per previous-token type, which no trained model uses (M1).

Overall: incremental theory, a useful but confirmatory trained-model study (now with a nice overwrite contrast), and a novel but idealised demand-to-budget analysis.

## 6. Proof and derivation check (line by line)

- **Prop. 1 moments.** Correct-score variance F/d_v + G/d_k·[(1−1/V)/d_v + 1/V] ≈ F/d_v + G(1/(d_k d_v) + 1/(V d_k)). Competitor variance adds s²/d_v. Both check. The 1/V term arises from P(v_j = w) = 1/V contributing c_j² with unit overlap. Treating the V−1 competitors as independent is an approximation, since competitors share the same r. The paper states this.
- **Eq. (2).** Follows from setting s = √(2 ln V)·σ_w with s²/d_v dropped. Checks.
- **Decomposition.** √(2 ln 512) = 3.532; exact median of the max of 511 normals is 2.999; (3.53/3.00)² = 1.39. V = 41,509: 4.61 vs 4.15. All correct.
- **Corollary 1.** F < d_v/(2 ln V) ≈ 64/21.3 ≈ 3. Correct. "A larger d_k does not help" is correct for the F term.
- **Corollary 2.**
  - The frame-potential algebra is correct: mean squared off-diagonal overlap ≥ (K/d − 1)/(K − 1); key-side gain (1 − 1/K)/(1 − d_k/K); for values, the competitor variance becomes exactly G/(d_k d_v).
  - However, the derivation drops s²/d_v, which a tight frame also reduces (by a factor (V/d_v − 1)/(V − 1) · d_v = 0.877 at d_v = 64). Keeping it, the analytic second-moment gain under rule 2-med is about 1.15 rather than 1.125.
  - I checked this by simulation (`d_k` = 16, `d_v` = 64, V = 512, a unit-norm tight frame built by alternating projection, 60 trials per N). The N\* gain of tight-frame codes over random codes was **1.11**. So 1 + d_v/V is a reasonable empirical guide, but it is neither a bound nor exactly the second-moment prediction.
  - The abstract's wording, "a frame-potential bound ... (η ≲ 1 + d_v/V)", overstates it. Also, trained models could have exploited structure other than tight frames: LayerNorm, a learned bias, non-Gaussian score distributions. "Below the ceiling" is therefore weak evidence for the mechanism.
  - Measuring the frame potential of the learned W_o rows and value projections directly would turn this into an actual test. It is cheap and missing.
- **Decay horizon** τ\* = ln(1/(√(2 ln V) σ))/ln(1/γ). Algebra correct.
- **Prop. 2 (Fano).** Σ_i I(V_i; Z) ≤ I(V_1..V_N; Z) needs the values to be independent, which holds. Average-error Fano is used via concavity. N((1−ε) log₂V − 1) ≤ Sb holds. Numbers check (4.57 S).

I found no errors that invalidate a conclusion. The only quantitative slip is the Corollary 2 ceiling, which is mis-stated as a "bound".

## 7. Major issues (remaining)

**M1. The research question is still not engaged by any trained language model, and the new replay evidence is about an idealised memory whose key design drives its main result.**

- The replay writes each previous-token type with one random key. So for keys like "the" or Go's ".", the bigram query is inherently ambiguous. The evaluation forgives the ambiguity in the argmax, but the formula charges for it through the own-key load F.
- The headline correction ("naive budget optimistic by 1.3–14×", "d_v = 64 caps recall below 50% for T ≥ 4k, whatever d_k") is therefore largely a statement about the paper's own choice of demand unit. Real recall heads key on richer, multi-token, positional and content features that split the fan-out. The authors concede this in Limitations: multi-token keys "would make a fixed state look better".
- The result is that the paper's quantitative budget is bracketed between a naive estimate it now calls optimistic and a corrected estimate that is plausibly pessimistic. The width of that bracket (1.3–14×) is the same order as the effects the budget is meant to predict.
- Requested:
  - A replay with 2-token (or hashed-context) keys, to show how much of the fan-out correction survives.
  - At least one *trained* multi-token-key model.
- I accept that pretrained checkpoints could not be downloaded. But a small CPU-trainable experiment would directly test P1 at toy scale and would change the paper's evidential status. For example: a 2-layer linear-attention / GLA / DeltaNet LM versus a softmax-attention LM of matched width, trained on the Gutenberg or Go data for an hour or two, with recall-token loss measured against T.
- Without it, the answer to "can new architectures replace the Transformer?" is a set of untested predictions.

**M2. Missing directly overlapping 2026 work.** See Section 5. Koren et al. (arXiv:2609.07681) in particular proposes recall scaling laws in state size, context length and embedding dimension, and validates them on trained Mamba. This weakens the novelty of contributions 1 and 5. It also provides exactly the trained-model evidence this paper lacks, so it needs to be discussed and compared quantitatively.

**M3. The new experiments lack uncertainty, at a level of precision the paper relies on.**

- **Replay (Table 4).** Only 5 windows per (corpus, T) cell, and no CIs on S_half, the ratios, or "max accuracy".
  - The claim "never reaches 50% for T ≥ 4,096" rests on prose oracle max accuracies of 0.475 (D0) and 0.476 (D3), five windows, within noise of 0.5.
  - The non-monotone code oracle ratios (8.7, 5.4, 14.2 for T = 1k, 4k, 16k) contradict the text's "more so at larger T and for code" and indicate sizeable window variance.
  - "Predicts recall to within 0.6–7 points" (abstract) reports a per-condition *mean* absolute error over S as if it were a bound. Report maximum errors and window-level CIs.
- **Overwrite baselines.** The ideal memories in Table 2b use 30 trials × P queries (120 queries at P = 4). "DeltaNet beats the ideal β = 0.5 rule" at P = 16 (0.90–0.91 vs 0.87, about 480 queries, SE ≈ 0.015) is about 2 SE. The explanation that "an intermediate β overwrites the same key while sparing other keys" is asserted, not tested: no ideal β ≈ 0.33 baseline was run. The overwrite task also has 2 seeds and one d_k.
- **Single-seed width control.** The value-width control (d_v = 32 and the d = 512 runs) is single-seed and evaluated on the biased factor-2 grid (Section 3). Yet it is the only evidence for the d_v-dependence of η and of the "ceiling".

**M4. Significance relative to the research question.** The paper is careful, but what it can conclude is narrow:

- an outer-product memory of a given shape can hold about S/(2 ln V) distinct-key associations (classical);
- trained 1-layer models reach that on a task designed so they should;
- delta rules help when values are superseded (known qualitatively from the DeltaNet / Gated DeltaNet literature);
- real text has many recurring context-specific bigrams whose keys fan out.

The bridge to "replacing the Transformer" goes through P1–P5, which are pre-registered but untested. That is a legitimate workshop contribution. For a main-track ICLR paper on this question, at least one prediction needs testing.

## 8. Minor issues

- **m1.** Corollary 2 is called a "bound" in the abstract. It is a heuristic, and the stated value ignores the s²/d_v reduction (Section 6). Rephrase as "a second-moment estimate", and measure the learned code geometry.
- **m2.** The abstract is about 450 words with dozens of numbers. It should be cut to the 4–5 claims that matter, with the details moved to the body.
- **m3.** "41–48% of ρ" should be 40–48%. Table 4 has 1.4·10⁴ where the output gives 1.35·10⁴.
- **m4.** The KV comparison uses 2dT with d = 4096 (full multi-head KV). Modern models use GQA/MQA; for example 8 KV heads × 128 gives 2,048 floats per token per layer, which would cut the ratio about 4×. The 400–1000× figure in the abstract, even with its caveat, invites misreading. I recommend moving it out of the abstract or using a GQA-realistic cache.
- **m5.** Width-control η is computed on the factor-2 grid against the dense simulation (Section 3). Recompute on the dense grid, or flag it.
- **m6.** Extrapolated T\* bootstrap ranges in `analysis_v2.json` are inverted or exclude the estimate. Fix or delete them.
- **m7.** The held-out D3 statistic for KJV windows comes from the other fold, which lacks biblical language. So D3 likely over-counts context-specific bigrams for KJV. The no-KJV control mitigates this, but it should be mentioned.
- **m8.** The replay's random keys are per token type with d_k up to 1,024. For the square shape at S = 2^20, d_k = 1,024 exceeds the number of distinct key types in a 1k window, so the memory is nearly exact. This is fine, but it explains why accuracy saturates there and should be noted.
- **m9.** README revision section: the figure names do not match `analyze_v2.py`'s docstring.
- **m10.** At V = 41,509 the most careful closed form (rule 2-med) is −15% and Eq. (1) is −17 to −19%. So "the closed form is within 9%" at large V is again partly a cancellation of errors, as at V = 512. The Gaussian theory degrades with V, yet the replay relies on Eq. (1) at V up to 149k. Its small accuracy-space MAE partly reflects how flat accuracy is in S, not N\*-level precision. Please report the replay agreement in S_half terms (predicted vs. measured S_half).

## 9. Strengths

1. **Exemplary reproducibility.** Bit-exact reruns, fixed seeds, and every number traceable to a JSON field.
2. **A responsive, honest revision.** Claims were narrowed where the evidence was weaker, and a self-correcting new result (the fan-out correction) was reported prominently, even though it undercuts the first version's budget.
3. **Good controls added:**
   - held-out demand criteria with cluster bootstrap;
   - BPE and no-KJV variants;
   - multi-seed ANOVA;
   - the overwrite task, which cleanly shows when the update rule matters.
4. **The load-aware Eq. (1) matching ideal-memory recall on real streams with V up to 149k** is a nice validation of the theory's applicability beyond toy codes, within its idealisation.

## 10. Assessment

All three round-1 fatal issues are fixed, and my reruns reproduce the revision bit for bit. No numbers are unbacked and no references are fabricated. Most major statistical and scoping concerns are addressed.

What remains is the paper's core evidential gap: no trained language model, of any size, tests any link between the budget and actual sequence-model behaviour on text. The new replay shows that the budget's quantitative output depends strongly on an idealised previous-token key design. Closely related 2026 work (Koren et al.) already offers recall scaling laws in state size validated on trained Mamba, and is not cited.

The contribution is a careful, reproducible, but incremental and idealised analysis. It sharpens the replacement question without answering it. I rate it borderline: a good workshop paper, below the bar for the ICLR main track as it stands.

## 11. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 6 |
| Reproducibility | 9 |
| **Overall** | **5.5 → 5** (weak reject; borderline/workshop) |

**Decision: major revision.** The path to acceptance:

1. Run one small trained-LM test of P1/P2 on CPU (fixed-state vs. attention, recall-token loss vs. T).
2. Run a replay with multi-token or hashed keys, to bound the fan-out correction.
3. Add CIs to the replay and stronger baselines to the overwrite task.
4. Discuss and compare with Koren et al. (2026) and Boesch & Wee (2026).
5. Recast Corollary 2 as an estimate and test it against the learned code geometry.
