# Review (Round 1): "Long Context Is Not Long-Term Memory: A Recency–Retention Bound for Softmax Readout under Knowledge Revision" (p006)

Reviewer role: area chair–level independent reviewer. Target venue: NeurIPS main track.

## 1. Summary

The paper asks whether a long context window works as a long-term memory. It answers no, on the grounds that memory must support *revision*. The setting is retrieval from a stream of key–value writes. For a single softmax head with (A1) a query that does not depend on the history and (A2) logits bounded in [−B, B], with arbitrary causal key representations, Theorem 1 proves (⌊A/Δ⌋+1)·L_ε ≤ 2B − log(T−1), where L_ε = log((1−ε)/ε). Proposition 2 gives an exact-match plus linear-recency head that comes close to this bound. Proposition 3 restates that a delta-rule memory with orthonormal keys returns the latest value exactly. The numerical experiments (E1) compare the bound with the construction, trace a hot/cold "frontier", and measure delta-rule capacity. The trained-model experiments (E2) use 1/2/4-layer RoPE Transformers and a 1-layer delta-rule model on a 16-key/16-value revision task, with 2–3 seeds. The paper also pre-registers a protocol for frontier models but does not run it.

## 2. Verification performed

**Code reruns.**
- `code/theory_sim.py` **does not run as shipped**. Running it gives `IndentationError: unexpected indent (line 51)`. Line 53 also has a trailing comment that swallows the comma between two dict arguments (`... else -1  # Thm1: A/Delta < floor(R_ub),`), which is a second syntax error. The script that the README says produces every E1 number, and Fig. 1, cannot be executed. Both errors are in the committed snapshot.
- I fixed the two syntax errors in a scratch copy and reran it. The E1 bound-vs-construction rows match `results/theory_sim.json` and the Section 5.1 table exactly (e.g., T=10³, B=12: bound 5, achieved 3.875; T=10⁷, B=20: bound 8, achieved 6.0). The T=10⁷ part runs much longer than the advertised 6 minutes, and I did not wait for the frontier and delta-capacity sections to finish. Those numbers in the JSON are therefore checked only against the paper text, not regenerated.
- `results/theory_sim.json` contains a `"note": "... recomputed from R_ub"` field in every E1 row. The code never writes such a field, so the JSON was post-edited by hand or by another script. The values agree with my rerun, but this contradicts the README's claim that "every number comes from" the shipped code.
- E2: I reloaded the saved checkpoints `tf1_s0`, `tf4_s0` and `delta_s0` and reran the paper's evaluation protocol with the same eval RNG for N ∈ {32, 128}. In-distribution accuracy and every grid cell match the JSONs exactly (max difference 0.0). I started a from-scratch retraining of `tf1_s0`, but the shared CPU was too slow, so training reproducibility is unverified. The tables in the paper match `results/train_toy_*_table.md`.
- All numbers I checked in Sections 6.1–6.2 match the result files: frontier values, delta capacity 0.88/0.82/0.80 at M=d and 0.28/0.27/0.28 at M=4d, Tables 2–3, logit spreads 11.3/8.3/6.4, and in-distribution accuracies.

**Proof check.** Theorem 1 is correct as stated, and I checked it line by line. The chain step uses causal consistency correctly: σ_i is identical across H_j for j ≥ i. The retention step (1−ε ≤ e^{σ₁}/(e^{σ₁}+(T−1)e^{−B})) gives σ₁ ≥ log(T−1) − B + L_ε, and combining the two steps gives the claim. The derived bound A* < Δ·⌊(2B−log(T−1))/L_ε⌋ follows. Proposition 2's algebra is correct: the condition g(c) < ρ is equivalent to c > log(1+1/ρ), and the factor at ε=0.05 is 0.983. One caveat: "letting c ↓ log(1+1/ρ)" makes the additive term log(1/(ρ−g(c))) diverge, so the "within factor 0.98 up to an additive term" statement hides a real trade-off. Corollary 1 and Proposition 3 are correct, and Proposition 3 is a textbook property of the delta rule (Schlag et al., 2021).

**Citations spot-checked (8).** PI-LLM / "Unable to Forget" (arXiv 2506.08184, log-linear decline, retrieval of overwritten values): exists and matches. Veličković et al. (2410.01104, dispersion): exists and matches, though it now appears at ICML 2025 under the title "...Sharp Size Generalisation". Nakanishi (2501.19399, SSMax, s·log n scaling, "attention fading"): matches. LongMemEval (2410.10813, ~30% drop, knowledge updates among five abilities, ICLR 2025): matches. Liu et al. 2023, flip-flop / attention glitches (NeurIPS 2023): matches. Barbero et al. 2024, "Transformers need glasses" (NeurIPS 2024): matches. RULER and BABILong are well-known papers with correctly characterized claims. I found no fabricated references.

**Additional diagnostics I ran.**
1. *The frontier "corner" is an artifact of the one-parameter family.* On the paper's exact Fig. 1b task (T=4096, B=10, hot spacing 32±8, success α ≥ 0.5), I tried a two-level step head with logit B for ages < 24 and −1.6 otherwise. It gets **hot 0.75 and cold 1.00**. The paper reports that no λ exceeds 0.15 on both, with best compromise (0.43, 0.15), and calls the frontier "essentially a corner". That holds only for linear-recency heads. Moreover, with the α ≥ 0.5 criterion, L_ε = 0 and Theorem 1 says nothing. Fig. 1b is therefore not evidence for the theorem and is presented misleadingly.
2. *Train/eval distribution shift in E2.* Training keys are uniform over 16 keys, so the count of the queried key is roughly Binomial(N, 1/16). P(R ≥ 16 | N = 32) ≈ 1.2·10⁻¹¹, while P(R ≥ 16 | N = 128) ≈ 6·10⁻³, and P(R ≥ 64 | N = 128) ≈ 10⁻⁴¹.

## 3. Strengths

- The question is sharp and timely: revision semantics, not storage, is the right lens on "context = memory". The store vs memory distinction is well articulated.
- Theorem 1 is clean, short and correct. The causal-consistency chain argument is elegant, and the "each version boundary costs L_ε nats, shared with retention" framing is memorable. Corollary 1 (log-scaled logits buy only logarithmically many revisions) is a nice consequence.
- Proposition 2 shows that the form of the bound is not vacuous.
- The paper is unusually candid about its scope: the multi-layer gap, toy models, RoPE extrapolation, the edge of the delta-rule regime and evaluation shift are all listed.
- The trained-model evaluation decouples R from recency, and the saved checkpoints reproduce the reported evaluation numbers exactly.

## 4. Weaknesses

### Fatal / near-fatal

**F1. The main E1 script does not execute.** `theory_sim.py` has two syntax errors, so neither Fig. 1 nor the Section 5.1 table can be regenerated from the submitted code without modification. The JSON also carries hand-added fields that the code does not produce. The venue's standard here is that every result must be backed by code that runs. The fix is trivial, and after fixing it the E1 bound-vs-construction numbers do reproduce. As submitted, though, this is a reproducibility failure.

**F2. The headline trained-model result (Table 2, latest version at age 0) is not a prediction of Theorem 1, yet it is presented as one.** In Theorem 1, the revision count in the chain is R = ⌊A/Δ⌋+1, where A is the age of the *latest* version. At A = 0 this gives R = 1, and the bound reduces to the dilution condition, with no revision cost at all. A single fixed-query head with a steep recency slope retrieves an age-0 latest write perfectly for any number of prior versions. So "accuracy falls with R at fixed age 0" is *not* what the theory predicts. Observation (i) says the decline "cannot be explained by recency or position. It is the signature of stale versions competing with the current one", and the abstract says the models show "the pattern the theory predicts". Both mislabel a phenomenon that the theory says is avoidable. The decline must come from something the theorem does not model: the learned solution, training distribution shift (point 2 below), or multi-layer effects.

### Major

**M1. Observation (ii), "revision spacing matters as predicted", is confounded with distribution shift.** At fixed R = 16, going from N = 32 to N = 128 changes spacing, but it also changes how far the test cell is from the training distribution: P ≈ 10⁻¹¹ versus 6·10⁻³ under training. The paper acknowledges the shift in its Limitations but still draws the Δ-dependence conclusion in the abstract and in Contribution 4. A control is needed, for example training with R drawn uniformly, or matching the training key-count distribution, before Δ-dependence can be claimed.

**M2. The success criterion makes the "arbitrary causal representation" generality partly hollow.** Theorem 1 bounds the weight on the specific token t\*, not retrieval of the correct value. With arbitrary causal z_t, the most recent position z_T may encode the current value of every key; a delta-rule state is itself such a causal function. A head that attends to position T then retrieves the right answer while violating the paper's notion of faithfulness. So the theorem is about *pointer location in an append-only store read by one fixed-query head*, not about the value-retrieval capability of models with rich causal representations. The abstract and Section 8 ("the answer to the title question ... is no") overstate what is proved. The statement should be reframed, or the result strengthened to a value-retrieval criterion under an explicit restriction on z, for example value-carrying tokens without cross-position mixing.

**M3. Fig. 1b's frontier is shown for a one-parameter family under an ε for which the theorem is vacuous.** As shown above, a simple step-function head reaches (hot 0.75, cold 1.00) where the paper reports the best achievable as (0.43, 0.15). The "corner" claim should be removed or replaced with a frontier optimized over general monotone score functions, and with ε < 1/2 so that Theorem 1 actually applies.

**M4. Limited novelty and significance of the theory.** The core ingredients are known: bounded-logit dispersion (Veličković et al.; Nakanishi), latest-write recall failures (flip-flop, Liu et al. 2023), empirical proactive interference (PI-LLM), and delta-rule overwrite (Schlag et al.). The new piece is a short chain argument combining recency and dilution for a single fixed-query head. It is neat, but it is a two-paragraph lemma, and the paper's own discussion admits it does not cover any layer beyond the first, which is where real models resolve recency. Without at least a partial multi-layer result, the theory's bearing on "long context vs long-term memory" in actual LLMs is speculative. I found no prior paper stating exactly this bound, so the contribution is genuine but incremental.

**M5. Weak statistics.** There are 2 seeds for most models and 300 streams per cell. Means ± s.d. over 2 seeds are not meaningful uncertainty estimates. There are no confidence intervals (a binomial CI on 300 draws is about ±0.05) and no significance tests for the "monotone" and "flat" claims. The "monotone decline at every depth and length" claim is not literally true: for L=1 at N=512, age 0, the sequence is 0.55 → 0.49 → 0.51 → 0.47 → 0.47. At N=512 even R=1/age 0 is only 0.55 for L=1, so length-extrapolation failure confounds every N=512 claim, including the "never-revised fact at chance" claim in the abstract.

**M6. No connection between measured logit spreads and the bound.** The paper measures spreads (11.3/8.3/6.4) but then says quantitative use is not justified. For L = 1, where the theorem does apply to the single attention layer (modulo 4 heads plus an MLP), the authors could test the quantitative prediction directly: the cold-retention versus hot-accuracy trade-off as a function of the measured spread and N. This is the one experiment that would actually test Theorem 1, and it is missing.

### Minor

- Proposition 2: the "within factor 0.98" statement should state the trade-off with the diverging additive term explicitly.
- Multi-head: four heads with value mixing and an MLP readout already break the single-head assumption, even for the "1-layer" model.
- "Decreases with context length" holds only under fixed B. Scalable-Softmax-style rescaling is discussed in Corollary 1, but the abstract should say "at fixed logit budget".
- Corollary 1's "o(1)" comes from log T − log(T−1), which is fine but should be stated.
- Veličković et al. should be cited as ICML 2025 with its current title.
- The delta-rule model gets an extra skip connection to the query embedding and an MLP readout, and its per-head dimension of 16 equals K. The comparison is fair but sits exactly at the edge of the Proposition 3 regime, as the authors note.
- The README advertises a 6-minute E1 runtime. The T=10⁷ sweep takes far longer on a 4-core CPU.

## 5. Questions for the authors

1. How do you reconcile Table 2 (age 0) with Theorem 1, which gives no revision cost at A = 0? What mechanism do you propose for that decline?
2. Does the R-decline at age 0 persist when training covers large R, for example with the queried key's count drawn uniformly in {1, …, N/2}?
3. Can Theorem 1 be stated for value retrieval rather than token weight, under an explicit restriction on z_t?
4. For the 1-layer model, does the cold/hot trade-off across N quantitatively follow (spread − log N)/L_ε?
5. What does the frontier look like when optimized over all monotone age-score functions, with ε = 0.05?

## 6. Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 5 |
| Rigor | 4 |
| Correctness | 5 (theorem correct; empirical attribution to the theorem incorrect) |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 4 (E1 script does not run as shipped; E2 checkpoints evaluate exactly) |

**Overall: 4 / 10. Decision: Reject (major revision needed before resubmission).**

The theorem is correct and nicely presented. However, the empirical section's main evidence (the age-0 R-decline and the spacing effect) is not what the theorem predicts and is confounded by distribution shift. The frontier figure overstates the trade-off because it uses a restricted head family. The theory's reach, a single fixed-query head with a token-weight criterion, is much narrower than the title and abstract claim. The primary E1 script also fails to execute.

## 7. Most valuable change

Align the experiments with the theorem. Test the theorem's actual prediction, which is the joint hot-revision / cold-retention trade-off at latest age A > 0 as a function of N and of the measured logit spread, in a 1-layer single-head model trained on a distribution covering the test R. Drop or reattribute the age-0 R-decline claims, and fix `theory_sim.py` so it runs.
