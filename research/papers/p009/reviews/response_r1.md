# Response to Review Round 1 (p009)

We thank the reviewer for a careful report, which included full reruns. We agree with all three fatal issues and with most of the major ones. The revision fixes the text–output mismatches and adds four new experiments:

- Monte Carlo CIs and a large-V check of the formula.
- 3 seeds for all trained families, plus a new overwrite task.
- Held-out demand criteria with document-level bootstrap, BPE and no-KJV variants.
- A replay of real token streams through ideal fixed-state memories.

It also repositions the theory against the associative-memory literature. Several headline claims are now weaker, because the new evidence does not support them: the 4,000× KV ratio is gone from the abstract, the naive budget turned out to be optimistic by up to 14×, and "fixed vs growing memory decides replacement" is now stated as a hypothesis. Every new number comes from code in `code/` with outputs in `results/` (listed in the README).

New or changed files:

- `code/capacity_ci.py` → `results/sim_capacity_ci.json`
- `code/train_mqar.py` (new flags), `code/run_train_seeds.sh` → `results/train_seeds/`, `results/train_overwrite/`
- `code/recall_demand_v2.py` → `results/demand_v2.json`
- `code/replay_memory.py` → `results/replay_memory.json`
- `code/analyze_v2.py` → `results/analysis_v2.json`, `results/fig_revision.png`

---

## Fatal issues

**F1. "8 of the 18 books eligible at T = 32,768" contradicts `demand.json` (12).**
*Agreed; it was a transcription error.* Section 4/5.4 (note under Table 3) now says 12 of 18 books are eligible and all 12 were sampled in the revised run. The corresponding code figures are 53 eligible and 38 sampled Go packages. All counts come from `n_docs_eligible` / `n_docs_sampled` in `results/demand_v2.json`.

**F2. The abstract attributed the 3–11% error to the closed-form law; the error pattern was backwards.**
*Agreed.* We reran Experiment A with 200 trials per cell, a 2^{1/8} grid and bootstrap CIs (`capacity_ci.py`). Table 1 now reports, separately for every cell:

- the simulated N\* with 95% CI;
- Eq. (1);
- rule (2);
- rule (2) keeping s²/d_v;
- rule (2) with s²/d_v kept and the exact extreme-value median.

The abstract and Section 5.1 now state that Eq. (1) is within 3–9% at d_v ∈ {64, 128}, and that the closed form always underestimates, by 13–21% at d_v ∈ {32, …, 128} and 5–7% at d_v ∈ {16, 256}. Its apparent accuracy at the extremes is a cancellation of two opposing errors. Section 3.2 now decomposes them (this also answers Minor 2). Contribution 1 no longer quotes the old figure.

**F3. "N_ctx grows at least linearly" is contradicted for code.**
*Agreed.* Section 5.4(b) now reports bootstrap CIs over documents and states the claim per corpus.

- Prose is superlinear: 1.23 [1.19, 1.27] globally under D3, 1.22 under D0.
- Code under the original criterion D0 is sublinear: 0.96 [0.94, 0.98] globally and 0.87 [0.78, 0.94] locally.
- Code under the held-out criterion D3 is consistent with linear growth: 1.01 [0.99, 1.03] globally and 0.95 [0.86, 1.04] locally.

The phrase "at least linearly" has been removed throughout.

## Major issues

**M1 (W1). The headline rests on three untested extrapolations (η, ln V at corpus V, bigram demand); there is no pretrained-model evidence.**
*Partly addressed, and the claims are scoped down.* We could not test P1/P2 on pretrained models. The environment has no GPU, and the Hugging Face hosts are blocked by the network proxy (`403 … no rule allows this host`), so no checkpoints can be downloaded. We say so in Section 6. Training small LMs on CPU until induction-style recall emerges is beyond the ~20-minute run budget. We tested the two links that can be tested without a pretrained model:

1. **ln V at corpus vocabulary.** (a) At V = 41,509, simulated capacity matches rule (2) within 8–9% (Eq. 1: −17 to −19%). This holds with uniform and with Zipf-distributed values (Section 5.1). (b) In the new **replay experiment** (Section 5.5), real prose and Go windows are written into ideal fixed-state memories with the *whole* corpus vocabulary as competitors (V = 41,509 / 149,292). A load-aware version of Eq. (1) predicts measured recall within 0.6–7.3 points (median 2.3) over 48 conditions.
2. **Demand → budget.** The same experiment shows that the naive budget of Eq. (4) is **optimistic by 1.3–14× for oracle writing**, and more so at larger T and for code. The cause is key fan-out, now formalised as Corollary 1: frequent keys recur with many values, and that load scales with 1/d_v, not with S. With one head's value width (d_v = 64), recall of the context-specific set stays below 50% for T ≥ 4k at any d_k. This is a substantive correction to the first version's budget, and it came out of the reviewer's question.

As a result:

- the abstract no longer presents T\* or the KV ratio as findings about real LMs;
- Table 5 is labelled as upper bounds on T\*, with the replay correction stated next to it;
- "fixed vs growing memory decides replacement" is presented in Section 6 as a hypothesis, explicitly not established;
- P1 now uses the fan-out-corrected T\*;
- the protocol uses a surprisal-based recall-token definition;
- Limitations opens with "No pretrained-model evidence".

η ≈ 1 remains validated only on single-layer models. We say so.

**M2 (W2). The N_ctx/ρ filter (not top-1) is too weak; ~45% of N_ctx bigrams are in the top-10; top-1 is estimated on a corpus containing the evaluated windows.**
*Agreed.* `recall_demand_v2.py` changes the measurement in three ways:

- Parametric statistics come from the *other half* of a two-fold document split, which removes the leakage.
- Five criteria are compared: D0 (original), D1 (held-out top-1), D2 (held-out top-10), D3 (held-out smoothed bigram p(b|a) < 0.01, now primary) and D4 (p < 0.001).
- Document-level cluster-bootstrap CIs are reported.

Results:

- *Top-10 filter.* Held-out top-10 removes 31–34% of prose N_ctx at T = 1k–4k. This confirms the direction of the reviewer's measurement; our figure is a little smaller because our top-10 table is held-out.
- *Effect of D3.* Moving from D0 to D3 removes 29–42% of N_ctx and 41–48% of ρ.
- *Abstract figures.* The 30%/48% recall-opportunity figures are replaced by 16% [14, 19] (prose) and 28% [26, 31] (code) under D3. D0 and D4 are given as a sensitivity range in Table 3.
- *Budget.* Table 5 carries D0, D3 and D4.

We note in Limitations that a held-out bigram model is still weaker than an LM's parametric knowledge, so D3 still over-estimates the context-only demand. The pre-registered protocol now uses surprisal under a context-free reference model.

**M3 (W3). The KV-cache comparison is not like for like; the 4,000× figure is presented as a finding.**
*Agreed.* The 4,000× figure is removed from the abstract and conclusion. Section 5.6 now compares measured S_half from the replay (oracle, D3, square state) at T = 16k with one d = 4096 KV layer. The ratio is ≈ 1,000× (prose) and ≈ 400× (code). The section states explicitly that the comparison is not like for like: the fixed-state side holds only oracle-selected, previous-token-keyed associations at 50% recall, while the KV cache also serves multi-token keys, multi-hop composition, positional retrieval and copying. The ratio is labelled a lower bound for a narrow sub-task. The "2 ln V vs 2d floats per token" sentence is removed.

**M4 (W4). Thin statistics for the precision claimed.**
*Addressed point by point.*

- *Exp. B seeds.* 3 seeds (0, 1, 2) for linear, gla, delta and gdelta at d_k ∈ {8, 16, 32} (36 runs). Table 2 gives N\* per seed, mean ± sd, η_sim per seed, and a one-way ANOVA of η_sim across families at each d_k (p = 0.95 at d_k = 8 and 0.65 at d_k = 16, so there are no family differences beyond seed noise; at d_k = 32, p = 0.001, entirely because DeltaNet is ≈ 5% lower as its learned β grows to 0.13–0.17). The first-version η differences (1.01 vs 1.13) were mostly the error of Eq. (1) plus seed noise. The paper now reports η only against the simulated ideal memory and no longer interprets differences between families.
- *Interpolation.* Evaluation now uses a 2^{1/4}-spaced N grid (16,384 queries per point). Seed 0 retrains the identical first-version models. Its dense-grid N\* differs from the factor-2-grid values by +1.1% to +3.3% (mean +2.2%), which quantifies the old interpolation error.
- *Exp. A noise.* 200 trials per cell with bootstrap CIs (about ±1%). We now quote errors to whole percent. The first version's two estimates of the (16, 64) cell, 92.3 and 89.8, were 2–5% above the new 87.9 [87.2, 88.9].
- *Exp. C.* Cluster bootstrap over documents for every per-T mean and for both global and local exponents, as the reviewer asked. There are now 100 windows per T up to 8k and 60 beyond (previously 60/30). At T = 32k, 12 books and 38 Go packages contribute; this is stated in the paper.
- *d_k = 64.* Still missing for three families. The linear-in-d_k claim is now restricted to d_k = 8–32, with per-seed slopes and their sd (1.07 ± 0.002 linear, 1.06 ± 0.005 gla, 1.03 ± 0.02 delta, 1.07 ± 0.0002 gdelta). The simulated ideal memory covers d_k = 8–64 (slope 1.04).

**M5 (W5). "Regardless of update rule" is built into the task.**
*Agreed, and tested.* The claim is now scoped to retain-everything workloads in the abstract, contributions, Section 5.2 and the conclusion. The text also says that additive storage is optimal there by construction. We added **Experiment B′ (overwrite task)**: each of P live keys is written 3 times with new values, and the target is the latest value. Results (Table 2b, 2 seeds). Linear attention tracks the ideal additive memory (≈ 0.30 at P = 16). DeltaNet learns β ≈ 0.33, five times its retain-everything value, and reaches 0.90–0.91 at P = 16, also beating the ideal β = 1 (0.77) and β = 0.5 (0.87) rules. Gated DeltaNet learns β ≈ 0.25 (0.88–0.89). GLA learns decay ≈ 0.985 (0.59–0.61). This is the regime the reviewer described, and in it the update rule does matter.

**M6 (W6). Novelty overstated; close prior work missing.**
*Agreed.* Every reference below was checked by web search.

- Section 2 has a new paragraph on the capacity of associative memories. It cites Cabannes, Dohmatob & Bietti (ICLR 2024; arXiv:2310.02984), Nichani, Lee & Bietti (ICLR 2025; arXiv:2412.06538), Giorlandino, Goldt & Maillard (ICLR 2026; arXiv:2605.10795; threshold p log p/d² = 1/2), Barnfield, Kim, Nichani, Lee & Lu (arXiv:2605.05189, sharp top-1 thresholds), McEliece et al. (1987) and Okpekpe & Orvieto (arXiv:2508.19029).
- Okpekpe & Orvieto's point about optimisation confounds is discussed, together with how our simulated-ideal baseline and multi-seed agreement address it.
- Proposition 1 is now titled "a specialisation of the classical analysis". The text says the S/(2 log) scaling is not new and states what is added: the (d_v, V) form, the fan-out corollary, the frame-potential ceiling, and the link to measured demand.
- Proposition 2 is titled "the standard counting argument".

## Minor issues

1. **Welch bound vs summed squared crosstalk.** *Agreed.* We replaced it with the frame-potential bound [Benedetto & Fickus, 2003], now Corollary 2. It gives a key-side gain ≤ 1 + d_k/K (< 1% here) and a value-side gain ≤ 1 + d_v/V: a tight value frame removes the 1/V crosstalk term exactly. This predicts trained η_sim slightly above 1 and increasing in d_v/V, below the ceilings 1.06 (d_v = 32), 1.125 (d_v = 64) and 1.25 (d_v = 128). Against the revised simulation, the observed values are 1.03/1.09 (d_v = 32, single seed), 1.00–1.07 for family means and 0.98–1.09 for individual runs (d_v = 64, 3 seeds), and 1.17 (d_v = 128, d = 512, single seed). All are below the ceiling except single-seed DeltaNet at d_v = 32, which is 3% above it. At d_v = 32 the crosstalk is far from Gaussian, and the paper says the ceiling is a second-moment heuristic, not a rigorous bound. The bound also explains why the first version's "learned keys spread more evenly" account was wrong: the key-side gain is negligible at K/d_k ≥ 128.
2. **Dropping s²/d_v.** *Agreed.* Section 3.2 now decomposes the error. The extreme-value approximation alone would make rule (2) underestimate by ≈ 1.39× at V = 512, and dropping s²/d_v pushes the other way. Table 1 shows both corrections separately. Keeping s²/d_v and using the exact median is within 0–6% at d_v = 64.
3. **Eq. (1) N\* = 347.8 vs 339.1.** These came from different N grids (factor-2 grid in Table 1, a 60-point log grid in Table 2). Both tables now use one quadrature on a 400-point log grid: 338.8 at d_k = 64.
4. **Mixed denominators (Nemotron-H 8% vs Kimi 25%).** *Agreed.* Section 1 now gives Nemotron-H-8B as 4 of 28 sequence-mixing layers (≈ 14% of mixers) and explains that NVIDIA's "roughly 8%" counts FFN layers. Kimi Linear is 25% of mixers. The incorrect sentence in Section 6 about "8–25% attention layers" is removed.
5. **"Section 8" pointer; duplicated η sentence.** Both fixed. Section 1 points to Section 7, and the methods text was rewritten without the duplicate.
6. **Gate statistics for the d_v = 64 DeltaNet runs.** Recorded in all 36 seed runs. DeltaNet at d_v = 64: β = 0.046/0.092/0.088 (d_k = 8), 0.064/0.103/0.115 (d_k = 16), 0.127/0.154/0.168 (d_k = 32) for seeds 0/1/2 . Gated DeltaNet: β = 0.025–0.026, 0.035–0.037 and 0.053–0.055, decay ≥ 0.9999. These agree with the reviewer's reruns (0.046 and 0.092 at d_k = 8, seeds 0 and 1).
7. **Narrow corpus base; no subword tokenizer.** *Partly addressed.* We added a byte-level BPE (vocabulary 8,192) version of the prose corpus and a version without the KJV Bible (37% of prose tokens). N_ctx(32k) changes by ≤ 3% without KJV and stays within the CIs under BPE; exponents change by ≤ 0.02. We had no web or multi-repository corpus available offline. This remains in Limitations, and extrapolations beyond 32k stay marked †.

## Reviewer questions

1. *S_req and T\* under a probability threshold?* Table 3 and Table 5 give D0, D3 and D4. At S = 2^14, oracle T\* is 8.4k / 12.8k / 23k for prose and 7.2k / 11.4k / 18k for code. The replay correction (M1) matters more than the choice of threshold.
2. *One step of the protocol on pretrained models?* Not possible in this environment (no GPU, model hosts blocked). The replay experiment is the closest substitute we could run, and it is labelled as such.
3. *η in a 2-layer unbound task?* Not run, for compute reasons. Listed as a limitation.
4. *Why ln V of the full vocabulary?* Tested directly: simulation at V = 41,509 (Section 5.1), and the replay with full-vocabulary competitors (Section 5.5), where the load-aware formula with ln V of the full vocabulary predicts recall within a median 2.3 points. The caveat is that this holds for random codes. A trained LM's effective competitor set may be smaller.
5. *Multi-seed CIs and bootstrap over documents?* Done (M4).

## Summary of claim changes

| v1 claim | v2 |
|---|---|
| Law accurate to 3–11% (attributed to closed form) | Eq. (1): 3–9% at d_v ∈ {64, 128}; closed form: −13 to −21%, decomposed |
| Trained models hit the law "regardless of update rule" | Only for retain-everything; the delta rule wins on the overwrite task |
| η = 1.01–1.13 (single seed, vs Eq. 1) | η_sim = 1.00–1.07 (3 seeds, vs simulation), within the frame-potential ceiling |
| ρ(32k) = 30% / 48% | 16% / 28% (held-out D3), with CIs; 27% / 48% under D0 |
| N_ctx grows "at least linearly" | Superlinear for prose, ≈ linear (D3) or sublinear (D0) for code |
| S_req(32k) ≈ 6–7×10⁴, 4,000× < KV cache | Naive budget optimistic by 1.3–14×; measured S_half(16k) 1.3×10⁵–3.2×10⁵; ≈ 400–1000× < one KV layer, stated as a narrow-sub-task lower bound |
| Fixed vs growing memory decides replacement | Hypothesis, pre-registered; not established |
