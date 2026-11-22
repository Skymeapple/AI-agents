# Review of p009: "How Much State Does In-Context Recall Need? A Superposition-Capacity Law for Fixed-State Sequence Models, Checked on Trained Mixers and Calibrated on Real Text"

Reviewer: independent area-chair-level review, round 1
Research question: Can new architectures (SSMs, hybrids, ...) replace the Transformer?

---

## 1. Summary

The paper turns the replacement question into a budget for one capability, single-hop associative recall. It has four parts.

- **(A) Theory and simulation.** A Gaussian signal-versus-crosstalk analysis of additive outer-product memory with an argmax read-out over V random codes. It gives an integral approximation (Eq. 1) and a rule of thumb, N\* ≈ d_k / (2 ln V (1/d_v + 1/V)) ≈ S/(2 ln V) (Eq. 2). A Fano-type bound (Prop. 2) shows that additive memory uses about 2% of the bits a 16-bit state could hold.
- **(B) Trained models.** Single-layer linear attention, GLA, DeltaNet and Gated DeltaNet are trained on a "capacity-isolating" MQAR task: pairs pre-bound into single tokens, writes masked at queries, K = 4096 keys, V = 512 values. All families reach about 1.05× the capacity of a simulated ideal additive memory. Learned β and decay settle in an additive regime. A width control shows that the effective value rank, not nominal d_v, sets capacity.
- **(C) Demand.** Counts of recurring, "context-specific" bigrams (N_ctx) and the recall-opportunity rate ρ(T), measured on 18 Gutenberg books and the Go standard library.
- **(D) Combination.** Critical context lengths T\*(S) under three write policies, and a comparison with the KV cache (about 4,000× less memory needed under the oracle policy at 32k tokens). Predictions P1–P5 are pre-registered for pretrained models.

The paper's answer: fixed versus growing memory, not attention versus recurrence, is what decides the question. Selective writing is the main lever.

## 2. Reproduction

I copied the package to a scratch directory and reran it there, so the authors' outputs were not overwritten.

- `capacity_sim.py` (Exp. A): reproduces `sim_capacity.json` **exactly**. Same N\* values for additive, delta and the d_v sweep. Decay mean accuracies 0.065 / 0.154 / 0.273 / 0.332.
- `recall_demand.py` (Exp. C): reproduces `demand.json` exactly (prose rows checked; for example N_ctx(32768) = 3345.5 and ρ = 0.301).
- `train_mqar.py` (Exp. B), three runs at the paper's settings:
  - `linear d_k=16`: N\* = 95.6 (paper 95.6).
  - `gdelta d_k=32`: accuracies identical to the file, β = 0.053, decay 0.9999.
  - `delta d_k=8`: bitwise-identical accuracy vector.
- Seed check I added: seed 1 gives N\* = 44.8 for `linear d_k=8` (seed 0: 45.4) and N\* = 44.0 for `delta d_k=8` (seed 0: 45.3). So the seed-to-seed spread is about 1–3%. That supports the "all families coincide" claim, though it also shows that the reported η differences (1.01 vs 1.13) are partly noise.
- I recomputed Tables 1–4 from the JSON outputs. They match to the reported precision. One exception: the Eq. (1) N\* for d_k = 64 is 347.8 in Table 1 but 339.1 in Table 2 (different N grids). The paper should explain this or unify the two.

**Mismatches between text and outputs:**

1. **Section 4.3:** "At T = 32,768 this leaves 8 of the 18 books (`n_docs_eligible` in `demand.json`)". The file, and my rerun, say **12**. Book lengths: 12 of 18 books are at least 32,768 words.
2. **Abstract and contribution 1:** they say "the law [N\* ≈ d_k/(2 ln V(1/d_v+1/V))] predicts the 50%-accuracy capacity to within 3–11% for d_v ∈ {64,128} (larger errors at d_v = 16 and 256)". The 3–11% figures belong to the integral, Eq. (1). The closed-form law, Eq. (2), is off by 21% (d_v = 64) and 16% (d_v = 128), and it is actually *most* accurate at d_v = 256 (5%) and d_v = 16 (12%). From `sim_capacity.json`, d_k = 16:

   | d_v | Rule (2) | Simulated |
   |---|---|---|
   | 16 | 19.9 | 22.7 |
   | 128 | 131.3 | 155.9 |
   | 256 | 218.9 | 229.6 |

   The abstract therefore attributes one approximation's accuracy to another and gets the error pattern backwards.
3. **Section 5.3(b):** "N_ctx grows *at least* linearly over this range". For code the fitted exponent is 0.95 ± 0.02 and the local exponent is 0.84, both below 1. The claim holds only for prose.
4. **Minor slips.** Section 1 points to "Section 8" for limitations (they are in Section 7). Section 4.2 repeats the sentence defining η. Section 5.1 reports errors to 0.1% (for example "2.3% and 9.5%") while the two independent simulations of the same cell (d_k = 16, d_v = 64) differ by 3% (92.3 vs 89.8). The precision is spurious.

Each is easy to fix. But under this venue's standard, claims that the code outputs contradict must be fixed before acceptance.

## 3. Reference check (web-verified)

| Reference | Verified? | Notes |
|---|---|---|
| Lufkin et al. 2026, Hybrid Associative Memories, arXiv:2603.22325 | Yes | Zyphra; RNN state plus data-dependent growing KV cache, as described |
| Qiao 2025, arXiv:2508.08289 | Yes | Later version titled "Attention as Conditioning…". It states Θ(d_k) for reconstruction versus near-quadratic capacity (up to a log factor) for selection from a vocabulary. v1 states O(√d_k); the paper cites the later version's claim, acceptable with the version noted |
| Wang et al. 2025, arXiv:2507.06457 | Yes | 72 models; recall improves particularly below 3:1; recommends HGRN-2 / GatedDeltaNet at 3:1–6:1. All as claimed |
| Guo et al. 2026, Log-linear attention, ICLR 2026 | Yes | Authors, venue and content as stated |
| Kimi Linear, arXiv:2510.26692 | Yes | 3:1 KDA:MLA, as claimed |
| Nemotron-H, arXiv:2504.03624 | Yes | "Roughly 8% of the total number of layers" are attention. Note: the denominator includes FFN layers. As a share of *sequence-mixing* layers it is 4/28 ≈ 14% (8B). Comparing it with Kimi's 25% (a share of mixer layers) mixes denominators; please fix |
| Arora et al. 2024a/b, Jelassi et al. 2024, Wen et al. 2025, Schlag et al. 2021, Yang et al. 2024a/b/2025 | Consistent with my knowledge | Not separately searched; descriptions are standard and accurate |

I found no fabricated references.

## 4. Novelty and prior work

**Theory (A).** This is the weakest part on novelty. Capacity of correlation-matrix / outer-product memories with argmax decoding, scaling as (number of parameters)/(2 log ·), is classical (Kohonen; Hopfield-style signal-to-noise analyses; the McEliece et al. 1987 d/(2 ln d)-type thresholds). It has recently been treated sharply and in LM-relevant settings. The paper cites none of the following:

- Cabannes, Dohmatob & Bietti, "Scaling Laws for Associative Memories" (arXiv:2310.02984): outer-product memories with random embeddings, explicit statistical rates.
- Nichani, Lee & Bietti, "Understanding Factual Recall in Transformers via Associative Memories" (ICLR 2025): storage capacity of linear associative memories scales linearly with parameter count, up to logs, including a linear self-attention analysis.
- Recent sharp-threshold work: Giorlandino, Goldt & Maillard (arXiv:2605.10795), threshold p log p / d² = 1/2, which is essentially the same S/(2 log) scaling; "Sharp Capacity Thresholds in Linear Associative Memory: From Top-1 Retrieval to Tail-Average Learning" (arXiv:2605.05189); and "Sharp storage capacity of a simplified model of linear associative memory" (ICLR 2026).
- On the empirical side: Okpekpe & Orvieto, "Revisiting Associative Recall in Modern Recurrent Models" (arXiv:2508.19029), and the Zoology/Based state-size sweeps already show recall tracking state size.

Given this literature, Prop. 1 is a textbook-level derivation. Its contribution is the explicit (d_v, V) form with the 1/V correction and the lag-resolved version, which is incremental. Prop. 2 is, as the authors admit, the standard counting bound.

**Trained-model check (B).** Showing that gates and delta rules collapse to additive storage when every item must be retained is a clean, mildly interesting observation. But the outcome is close to predetermined by the task design: no overwriting, uniform queries, writes masked at queries.

**Demand measurement (C) and the combined budget (D).** This is the most original part. I am not aware of prior work that turns bigram recurrence statistics into a state budget for fixed-state layers. But the operationalisation is crude (see below).

Overall: incremental theory, a confirmatory toy experiment, and a novel but weakly validated demand-to-budget bridge.

## 5. Strengths

1. **Exemplary reproducibility.** Everything I reran reproduced, often bit-for-bit. Seeds are fixed and the README is honest about the machine-sharing cut.
2. **Clear, falsifiable framing.** P1–P5 come with explicit thresholds and a concrete protocol on the public Waleffe et al. checkpoints.
3. **The width control (d = 512) is a good experiment.** It isolates the nominal-versus-effective state issue and corrects what would otherwise have been a misleading d_v plateau.
4. **Honest limitations section.** It explicitly scopes the work to single-hop bigram recall and non-causal oracle writing.
5. **Mechanistic explanation of why gated/delta models do not beat additive memory here.** The learned β is about 0.03–0.14 and the decay about 0.9999. Both are recorded, and I reproduced them.

## 6. Weaknesses

### Major

**W1. The answer to the research question rests on extrapolations the experiments do not test.** The headline numbers (T\*, S_req, the 4,000× gap) multiply three quantities. Each is validated only in a regime far from real LMs:

- **η ≈ 1** comes from a 1-layer, pre-bound, write-masked task with S ≤ 2,048 and V = 512.
- **ln V** is replaced by ln of the corpus vocabulary (41k / 149k). The law was validated only at V = 512 with random codes, and the paper itself shows Eq. (1) breaks down when d_v/V is not small.
- **D(T)** is a bigram heuristic (W2).

No pretrained-model measurement anchors any of this. P1–P5 are all untested. As a result, the conclusion that "fixed vs growing memory decides replacement" is a reasonable hypothesis, but this paper does not show it.

**W2. N_ctx / ρ overstates "context-only" recall.** A bigram counts as context-specific if its successor is not the corpus-wide top-1 successor. That filter is very weak. I measured it with the authors' code and data: about **45% of the prose N_ctx bigrams at T = 1k–4k are among the top-10 corpus successors of their head word** (for example "of a" when the top-1 successor is "the"). Such continuations are largely predictable from parametric statistics and do not need in-context storage. Other problems:

- ρ counts every repeat of such generic bigrams as a "recall opportunity".
- The top-1 table is estimated on the same corpus that contains the evaluated windows (mild leakage).
- At least one alternative is needed: a filter based on conditional probability (for example p(b|a) < τ) or on a held-out LM's surprisal, and a sensitivity analysis of the T\* and S_req numbers to that choice.

The 30% / 48% recall-opportunity figures in the abstract should not be reported without this.

**W3. The KV-cache comparison is not apples to apples.** The "about 4,000× smaller" figure compares:

- a hypothetical memory holding *only oracle-selected bigram associations* at random-code capacity, against
- a d = 4096 KV cache that stores full representations serving every function of attention: multi-token keys, multi-hop composition, positional retrieval, copying of long spans.

The every-token comparison (2 ln V versus 2d floats per token) has the same problem. The paper should either drop the headline ratio or frame it explicitly as a lower bound on a narrow sub-task. It is in the abstract and conclusion as though it were a finding.

**W4. The statistics are thin for the precision claimed.**

- Exp. B uses one seed per configuration. My seed-1 runs move N\* by 1–3%, which is comparable to the reported η differences.
- N\* is log-interpolated between N values a factor of 2 apart, with no uncertainty reported.
- Exp. A's Monte Carlo noise is about 3%, yet errors are reported to 0.1%.
- Exp. C: the exponent "± 0.03" is a residual-based SE over 7 correlated T points (the code comment admits that bootstrapping over T is meaningless). It ignores window-sampling noise and the overlap of windows within the few long documents. At T = 32k there are only 30 windows from 12 books and 53 Go packages; no confidence intervals are given.
- d_k = 64 is missing for 3 of 4 families, and linear attention at d_k = 64 never crosses 50%. So the "linear in d_k" slope rests on 3 points per family.

**W5. The "regardless of update rule" result is largely built into the task.** With uniform queries over all stored items, no overwriting and writes masked at queries, additive storage is optimal (the authors' own Section 3.2). So the finding that gradient descent drives β → small and decay → 1 confirms the setup, not a property of the architectures. Phrases like "Trained models hit the law, regardless of update rule" should be scoped in the abstract to retain-everything workloads. The regime that matters for the replacement question is the one where delta and gating could help (recency-weighted, overwrite-heavy, or multi-layer settings), and it is untested.

**W6. Missing related work (Section 4).** Cabannes et al. 2023, Nichani et al. 2025, the 2026 sharp-threshold papers and Okpekpe & Orvieto 2025 need to be cited, and Prop. 1 positioned against them. As written, "we derive a capacity law" over-claims originality.

### Minor

- **Welch-bound argument (Section 3.2(c)).** The Welch bound constrains the *maximum* cross-correlation. Recall error is governed by the *summed squared* crosstalk. The relevant tool is the frame-potential bound Σ_{i,j}⟨k_i,k_j⟩² ≥ K²/d_k. It gives the same conclusion (mean squared crosstalk ≳ 1/d_k), but the argument as stated does not support it.
- **Dropping s²/d_v in Eq. (2).** At d_v = 64, 2 ln V / d_v ≈ 0.19, which is not small. The ~20% underestimate is a mix of two opposing approximations and should be decomposed rather than attributed only to the extreme-value median.
- **Mixed denominators** between Nemotron-H's 8% and Kimi's 25% (see Section 3).
- **Table 1 vs Table 2:** Eq. (1) N\* at d_k = 64 is 347.8 vs 339.1.
- **Missing gate statistics.** Gates were not recorded for the d_v = 64 DeltaNet runs (older code version). I reran `delta d_k=8` and got β = 0.046 (seed 0) and 0.092 (seed 1). Please regenerate and report them, including seed variation.
- **Corpus scope.** The prose corpus (18 old books, one of which is the very repetitive KJV Bible at ~790k words, so it dominates long-window sampling) and a single code library are too narrow for exponents that drive extrapolation to 1M tokens (Table 4, †). I checked that restricting T = 1k–4k windows to the 12 long books changes N_ctx by only about 2%, so composition bias is small at those T. Wider corpora (web text, multi-file repositories, a subword tokenizer) are still needed.

## 7. Questions for the authors

1. How do S_req and T\* change if N_ctx uses a probability threshold (for example p(b|a) < 0.01) or a held-out-LM-surprisal criterion instead of "not top-1"?
2. Can you run even one step of the pre-registered protocol (for example ρ-token loss versus context length on the released Mamba-2 / Transformer 8B checkpoints, or on small open SSMs such as Mamba-130M and Pythia-160M)? Even a small-model knee test of P1 would change the paper's evidential status.
3. What is η in a 2-layer model where the key–value binding must be formed (no pre-bound tokens) and writes are not masked?
4. Why should ln V of the full corpus vocabulary be the right V for real-text recall, given that Eq. (1) was validated only at V = 512 and d_v/V ≪ 1?
5. Please report multi-seed means and CIs for N\* and η, and bootstrap CIs over documents for N_ctx(T) and ρ(T).

## 8. Assessment

The submission is careful, honest and highly reproducible. The demand-to-budget framing is a useful way to sharpen the replacement question. But:

- the theory is classical and under-cites close recent work;
- the trained-model result is mostly determined by the task design;
- the real-text demand measure is crude and inflates "context-only" recall;
- the headline conclusions (T\*, the 4,000× gap, "fixed vs growing memory decides") rest on untested extrapolations;
- several statements in the text contradict the outputs (n_docs 8 vs 12; the abstract's attribution of the 3–11% error; "at least linear" for code).

With pretrained-model evidence on even one prediction, a stronger demand definition and proper positioning against the associative-memory capacity literature, this could become a solid paper. In its current form it is at workshop level.

## 9. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision (weak reject in current form).**
