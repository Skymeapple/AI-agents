# Response to Round-1 Review (p006)

We thank the reviewer for a careful and constructive report. In particular, the reviewer re-ran our code, checked the proof line by line, and built a counter-example to Fig. 1b. We agree with both fatal points and with most major points. The revision changes the paper's claims, not only its presentation:

* The title and abstract now state the scope: a fixed-query (first-layer) softmax read, and a margin bound rather than an accuracy bound.
* We withdraw the attribution of the age-0 R-decline and the spacing claim to Theorem 1.
* We replace Fig. 1b with an exact frontier over all age-score heads, computed by LP at ε = 0.05.
* We add the experiment the reviewer asked for: single-head models with a known logit budget, trained on a distribution that covers the test structure, evaluated on the proof's own chain at A > 0.
* We add new theory: a staircase construction that matches the bound up to log(1+ρ) per revision, an age-0 result for token-and-age heads, a value-retrieval corollary, and a remark showing that A1 cannot be dropped.

Every new number comes from code in `code/` run in this environment, with outputs in `results/`.

---

## Fatal issues

### F1. `theory_sim.py` does not run, and the JSON has hand-added fields

**Agreed.** Changes:

* `code/theory_sim.py` was rewritten and runs as shipped: `python3 theory_sim.py` finishes in about 8 s. The worst-case weight of the linear head is now computed in closed form with geometric sums, so the T = 10⁷ sweep is fast.
* The script writes every field of `results/theory_sim.json` itself. The `note` fields are gone. The legacy one-parameter frontier code was removed, and the frontier is now computed by `code/frontier_lp.py` (about 40 s).
* **Reproduction check:** all 21 E1a rows (bound and best linear head) are identical to the submitted values. We compared the new JSON with the old one programmatically, and the old JSON is kept in our scratch space.
* The delta-capacity section now uses its own RNG stream and reports standard errors. Its numbers changed in the second decimal (for example M = d: 0.89/0.83/0.83, previously 0.88/0.82/0.80), and the paper quotes the new values.
* The README runtime claims were corrected. Every analysis script writes its own outputs: `e3_*`, `e2_r1_*`, `frontier_lp.json`, and `e3_lp_reference.json`.

### F2. The age-0 R-decline (Table 2) is not a prediction of Theorem 1

**Agreed. This was a substantive error.** Changes:

* §4.1 now says explicitly: "Theorem 1 predicts no revision cost for a latest version at age 0". It also says that the theorem is a *joint* statement that does not say which history fails. We restated Theorem 1 in the sharper form that the proof actually uses: a cold fact at age A, together with hot facts whose latest age is ≤ A − Δ. The original family form is a special case.
* Observation (i) and the abstract's phrase "the pattern the theory predicts" were removed. §6.4 lists them under **"What we withdraw"**.
* **New theory for the age-0 cell (Proposition 4).** For token-and-age heads, which include every first-layer head, retaining never-revised facts at ages g, …, Mg caps the weight on a latest version at age 0. The cap is o_h ≤ e^{2B}/[(T−1)Σ_j o_c(jg) + T−1−M], which implies a cost of log(R−1) nats. This is the correct, weaker age-0 statement. It does not cover heads with history-carrying key features, and therefore does not cover later layers.
* **New test (E3b).** In trained single heads with known B, the Proposition-4 ceiling is never exceeded. Pointer-trained heads with B = 4 sit on it: measured 0.49/0.25/0.13/0.07/0.03 against ceilings of 0.51/0.27/0.14/0.07/0.04 for R = 2…32. Heads with looser budgets lie far below it, and for those heads the paper attributes the decline to their learned shallow recency profile, not to the bound.
* **New control answering Question 2.** We retrained the 1-layer Transformer on an R-covering distribution (R ~ U{1..N/2}, 2 seeds). At N = 128 and R = 64, age-0 accuracy rises from 0.36 [0.32, 0.39] to 0.77 [0.73, 0.80], and the logistic slope on log₂R halves (−0.51 → −0.25). Most of the original decline was therefore distribution shift, as the reviewer suspected. The same control loses retention: a never-revised fact at age N/4 falls from 0.38 to 0.13. We report this trade, but we do not attribute it quantitatively to the theorem, because a 4-head model with an MLP readout is outside its scope.

---

## Major issues

### M1. The spacing (Δ) claim is confounded with distribution shift

**Agreed.** Observation (ii) and the Δ-dependence claims in the abstract and in Contribution 4 have been removed. §6.4 explains why: under training, P(R ≥ 16 | N = 32) ≈ 10⁻¹¹. The spacing dependence is now tested only where it is identified: in the LP frontier (E1b) and in E3. In E3, the chain spacing Δ = 16 and all test lengths lie inside a training distribution that covers R ≤ 8 and gaps 8–40.

### M2. The token-weight criterion makes the "arbitrary causal z" generality hollow, and the conclusion overstates

**Partly agreed. We restricted the claims and strengthened the result where that was possible.**

* **Corollary 2 (new).** Under an explicit token-local value assumption (A3), which holds in a single attention layer on token embeddings with positional information only in queries and keys, Theorem 1 holds for *value* retrieval (total weight on the correct value ≥ 1−ε). The proof chooses distinct values in the chain. This answers Question 3 under the restriction the reviewer suggested.
* **We agree that without A3 the theorem says nothing about value retrieval.** The paper now states this, using the reviewer's example of z_T carrying a delta-rule state. It describes Theorem 1 as a statement about *locating the current version in an append-only store by a fixed-query read*.
* **Remark 3 (new): A1 cannot be dropped.** A two-layer counting construction (write index n_t as a key feature, total count c in the query) escapes the bound for every R and every age, at the cost of head dimension and counting precision. This makes clear how little the theorem says about later layers.
* **Margin, not accuracy (new, and confirmed by E3).** L_ε → 0 as ε → 1/2, so argmax decoding over distinct values is limited only by dilution. E3 shows this directly: value-trained heads decode 100% correctly with only 0.34–0.61 weight on the current version.
* **Title and conclusion.** The title is now "Revision Costs Logit Range: A Recency–Retention Bound for Fixed-Query Softmax Readout of an Append-Only Context". The claim "the answer is no" was replaced by a scoped answer: for fixed-query softmax readout, window size is the wrong measure, and whether depth escapes the constraint is left open.

### M3. The Fig. 1b corner is an artifact of the one-parameter family and of α ≥ 0.5

**Agreed. Fig. 1b was replaced.**

* **New exact frontier (`code/frontier_lp.py`).** Over *all* age-dependent score functions f(a) ∈ [−B, B], every faithfulness constraint is linear in e^{f}. The worst case over hot histories is encoded exactly with dynamic-programming variables, so "hot horizon A_h and cold horizon C" is an LP feasibility problem. Every solution is re-verified by exact worst-case evaluation. The setting is ε = 0.05, T = 4096, Δ = 32 and B ∈ {6, …, 10}.
* **Results.**
  * The maximal hot horizon equals the Theorem-1 limit for every B (31, 31, 63, 95, 95), so the bound is tight at Δ resolution for token-and-age heads.
  * Retention collapses as the hot horizon grows. At B = 10, C = 4095, 575 and 95 for A_h ≤ 31, A_h ∈ [32, 63] and A_h ∈ [64, 95].
  * Linear-recency heads lie far inside the frontier. At B = 10 none retains cold facts beyond age 87 while keeping a non-empty hot horizon.
* **The reviewer's counter-example, reproduced.** The step head (B for age < 24, −1.6 otherwise) gives hot horizon 23 and cold horizon 4095 under α ≥ 1/2 (`frontier_lp.json: reviewer_step_head_eps_half`). The paper now says that the old "corner" was wrong and why.

### M4. Novelty is incremental, the result covers only one fixed-query head, and real models resolve recency in later layers

**Agreed that the core lemma is short and builds on known ingredients. The paper now says so** (Contributions and Limitations). We strengthened what surrounds the lemma:

* **Proposition 2′ (staircase).** It achieves R·L_ε + (R−1)·log(1+ρ) ≤ 2B − log(T−1), so the gap to Theorem 1 is (R−1)·log(1+ρ) nats, 0.051 per revision at ε = 0.05, with no diverging additive term. It matches the Theorem-1 horizon in 14 of the 15 feasible cells of E1a, verified by exact dynamic programming for T ≤ 10⁵.
* **The exact LP frontier (E1b)** and **Proposition 4** for the age-0 cell.
* **Corollary 2 and Remark 3**, which mark the boundary of what the theorem covers.

On "later layers", we agree, and Remark 3 makes the point formally. The paper no longer claims to explain PI-LLM-style failures of deep models. §7 now proposes a protocol that tests first-layer selection margins and asks where depth helps. We do not claim a multi-layer theorem.

### M5. Weak statistics; "monotone" claim false at L=1, N=512; N=512 confounded by length extrapolation

**Agreed.**

* **New analysis script (`code/analyze_toy_r1.py`).** It pools each seed's streams (600–900 per cell) and reports Wilson 95% intervals plus a logistic trend test on log₂R for every row (`results/e2_r1_tables.md`).
* **Monotonicity claim.** The blanket claim was removed. The analysis confirms that the means are not strictly decreasing for L = 1 at N = 512 (0.55, 0.49, 0.51, …), and also not for the delta-rule model at N = 512. In addition, the delta-rule model's R-decline is statistically clear (slope −0.58, p ≈ 3·10⁻²⁴ at N = 128), so "almost flat" was an overstatement. The paper now says "much shallower".
* **N = 512.** All N = 512 claims, including "never-revised fact at chance" in the abstract, were withdrawn because of the length-extrapolation confound.
* **E3 statistics.** E3 uses 500 instances per cell, test lengths inside the training range, and 1–3 seeds per configuration, with seed counts listed and min–max over seeds reported. Seed-to-seed spread of the key quantity m is ≤ 0.02 in every cell. Because the CPU was shared with other jobs, we could not run more seeds, and single-seed configurations are labelled as such.

### M6. The one experiment that tests Theorem 1 quantitatively is missing

**Done (new §6.3, E3).** Design:

* **Model.** A single head with exactly the theorem's assumptions: a query from the query token only; a factorized token-local embedding; content logits plus a learned bias for every age; logits clamped by B·tanh(·/B) with *known* B ∈ {4, 6}, or unclamped with the spread measured; and a token-local value path with a linear readout.
* **Training.** The training distribution covers the test structure: R ≤ 8, latest age uniform, test lengths 64/128/256 inside the training lengths 48–256.
* **Objectives.** Value cross-entropy, and a pointer objective that trains selection margin.
* **Evaluation.** The proof's own chain H_1..H_R at A ∈ {0, …, 192} with Δ = 16. For every instance we record the worst-chain weight m, the measured spread S, and the ceiling m ≤ 1/(1+exp(−(S − log(N−1))/R)) implied by the proof.

Findings:

1. (★) holds in all 135,000 instances. This is a check of the setup, not evidence for the theory, since (★) is a theorem.
2. At R = 1, trained heads sit *on* the dilution ceiling and follow its log N dependence. For example, at B = 4 the measured weights are 0.925/0.866/0.765 against ceilings of 0.930/0.872/0.772 at N = 64/128/256.
3. At R ≥ 2, heads sit well inside the Theorem-1 ceiling. An LP reference (`code/e3_lp_reference.py`) shows that much of the gap comes from the training requirement to retain cold facts at *all* ages. With that requirement, the best achievable worst-case weight for B = 6 is 0.87/0.68/0.53 at R_A = 2/4/7, against trained values of 0.60/0.44/0.36.
4. Changing the training mixture moves heads along the hot/cold trade-off. With the pointer objective at B = 6, raising p_single from 0.125 to 0.5 moves the cold weight from 0.91 to 0.99 and the hot weight from 0.44 to 0.28.
5. The age-0 decline tracks the Proposition-4 ceiling when the budget is tight (see F2).

---

## Minor issues

* **Prop. 2 additive term.** The text now states that log(1/(ρ−g(c))) diverges as c ↓ log(1+1/ρ), so the factor 0.98 and the additive term cannot both be attained. Proposition 2′ supersedes Proposition 2.
* **The "1-layer" Transformer has 4 heads and an MLP.** Agreed. The E2 models are now explicitly outside the theory (§6.4 and Limitations), and the quantitative test uses true single-head models (E3).
* **"Decreases with context length" holds only at fixed B.** The abstract and §4.1 now say "at fixed B". Corollary 1 makes the o(1) term explicit as log(T/(T−1)).
* **Veličković et al. citation.** Verified by web search. It is now cited as ICML 2025, PMLR 267:61190–61211, under the title "Softmax is not Enough (for Sharp Size Generalisation)".
* **README runtime.** Corrected: about 10 s for `theory_sim.py` and about 40 s for `frontier_lp.py`. E3 runtimes are also stated.
* **Delta-rule model dimension and skip connection.** These are now stated in the Limitations, and no new claims rest on that model.

## Citations

No citations were added. The one updated entry (Veličković et al.) was verified online. All other references are unchanged from the submitted version, which the reviewer spot-checked.

## Questions

1. *How do you reconcile Table 2 (age 0) with Theorem 1?* We cannot, and we withdrew the attribution (F2). Proposition 4 gives the correct age-0 statement for first-layer heads. For the multi-layer models, the control shows the decline was mostly distribution shift.
2. *Does the decline persist with R-covering training?* It is reduced substantially but not eliminated (slope −0.25, p ≈ 5·10⁻²⁵), and retention drops at the same time (F2).
3. *Can the theorem be stated for value retrieval?* Yes, under a token-local value path (Corollary 2). Without that assumption, no.
4. *Does the 1-layer cold/hot trade-off follow (spread − log N)/L_ε?* For never-revised facts, yes, to within 0.01 in clamped heads. For R ≥ 2, trained heads are below the ceiling, mostly because they must also retain cold facts at all ages (LP reference). The training mixture moves them along the trade-off (E3).
5. *What does the frontier look like over all monotone age-score functions at ε = 0.05?* It is now computed exactly over *all* age-score functions, monotone or not (Fig. 1b and §6.2).

## Remaining limitations, stated in the paper

* The theory covers only fixed-query reads, and Remark 3 shows deeper layers can escape it.
* The bound constrains margin, not argmax accuracy with distinct values.
* E3 is small: d = 64, one task, 1–3 seeds per configuration.
* The multi-layer results are exploratory.
* The frontier-model protocol has not been run.
