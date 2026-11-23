# Review (Round 1): "Observational Twins: An Information-Theoretic Test for Separating Causal Inference from Correlational Shortcuts in Learned Reasoners"

**Reviewer role:** independent senior reviewer / area chair
**Research question as posed:** 模型能否进行真正的因果推理，而不只是相关性推断？(Can models do genuine causal reasoning rather than correlational inference?)

---

## 1. Summary

The paper proposes a *twin test*: pairs of SCMs (M, M′) with the same observational distribution but opposite answers to an ancestral query. It argues three things:
1. Any reasoner that sees only observational samples plus side information shared within the pair gets pair-averaged accuracy exactly 1/2 (Prop. 1).
2. With extra evidence, accuracy is at most (1+TV)/2, and the likelihood-ratio test reaches that bound (Prop. 2, Le Cam).
3. For linear-Gaussian SCMs, reversing a covered edge and refitting by population regression gives an exact twin (Prop. 3). Repeated reversals reach a twin for every query that is unidentified within the Markov equivalence class (MEC). This is checked exhaustively on all 543 four-node DAGs.

The experiments train small DeepSets "causal-query" networks on CPU under 8 regimes. A label-order shortcut lifts i.i.d. accuracy from 75% to 92% while Gaussian twin accuracy stays at 50%. Models trained with Laplace noise or interventional samples earn positive surplus, reported as "ceiling capture" κ = Δ/Δ_LR. An LLM protocol is specified but not run.

## 2. Reproduction and verification

- **Theory checks.** I reran `code/verify_theory.py` (about 9 s). It reproduces `results/theory_checks.json` exactly: 543 DAGs, 185 MECs, 4,008 unidentified (DAG, pair) combinations, 0 without a twin, reversal histogram 2280/1248/432/48, 84.59% of SCMs with a covered edge, and a maximum covariance difference of 7.1e-14. The counts 543 and 185 match the known numbers of labelled DAGs and MECs on 4 nodes.
- **Training runs.** I reran `twins.py` from a clean copy of the code using the same configurations as `run_all.sh`. The four Gaussian observational seed-0 runs (obs_p0, obs_p05, obs_p09, obs_p1) are **bit-identical** to the shipped JSONs: every numeric field matches with difference 0. Results for the remaining runs are reported in the addendum at the end of this review. The machine was shared and heavily loaded, so the full 12-run sweep took far longer than the stated 30–35 min.
- **Paper vs. outputs.** Every number I checked in Tables 1–2, Section 5.4, Section 6 and the abstract matches `results/summary_table.md` and the per-run JSONs. Examples:
  - obs_p1 twin accuracy 49.97% with CI [49.63, 50.30];
  - κ per seed 0.868/0.852, 0.705/0.527 and 0.849;
  - the Section 6 decomposition 26.1×0.55 ≈ 14.4 and 7.5×0.45 ≈ 3.4.

  The figures are produced by `summarize.py` from these JSONs. I found no fabricated or unbacked numbers.
- **Minor reproducibility notes.**
  - `summarize.py` globs `*_s*.json`. In a results directory containing any other file matching that pattern (e.g. `supp_stats.json`) it would crash on `d["name"]`. It should match `*_s[0-9].json`.
  - `summarize.py` references a regime `int8_p0` that is never run.
  - The README says the four obs seed-0 runs came from an "earlier pass". My bit-exact match confirms this does not matter.

## 3. Reference spot-check (web-verified)

| Reference | Exists? | Supports the claim? |
|---|---|---|
| Yu & Zhou (2026), *Caliper*, arXiv:2606.04915 | Yes (June 2026, Zhenyu Yu & Shuigeng Zhou) | Yes: placeholder variable names open an accuracy gap on CLadder-style benchmarks |
| Kasetty et al. (2024), arXiv:2404.05545 | Yes; NeurIPS 2024 CaLM workshop | Yes |
| Richens & Everitt (2024), ICLR | Yes (ICLR 2024 oral) | Yes: a regret bound under distribution shifts implies an approximate causal model |
| Bareinboim, Correa, Ibeling, Icard (2022), ACM chapter 27, doi 10.1145/3501714.3501743 | Yes | Yes (Causal Hierarchy Theorem) |
| Reisach, Seiler, Weichwald (2021), NeurIPS | Yes | Yes, with a caveat below |
| Zečević et al. (2023), *Causal Parrots*, TMLR | Yes | Yes |
| Chickering (1995), UAI, pp. 87–98 | Yes | Yes: covered-edge definition; reversal of a covered edge preserves equivalence; equivalent DAGs are connected by covered reversals |

No unverifiable citations were found.

The Reisach caveat: the paper states that per-dataset standardization "removes marginal-variance (varsortability) cues". That is true for marginal variance. However, Reisach et al. themselves note that high-varsortability generating processes leave a covariance pattern that can still be exploited after standardization (later formalised as R²-sortability). This does not affect the twin floor, but the i.i.d. regimes are not free of order leakage through the data.

## 4. Strengths

1. **The framing is clean and correct.** Using exact non-identifiability as an evaluation instrument, with a floor that holds for every reasoner and a computable LR ceiling, is a crisp way to state "does the answer depend on rung-2-relevant evidence?" The Remark on what Δ > 0 does and does not certify is unusually honest.
2. **The propositions are correct.**
   - Prop. 1 is a one-line invariance argument.
   - Prop. 2 is Le Cam's two-point bound; the proof is right, including randomized rules and the LR rule attaining the bound.
   - Prop. 3 is right: a covered reversal gives a Markov-equivalent G′, and the population regression on G′-parents reproduces Σ exactly.
   - The interventional-ceiling argument (only do(u) or do(v) separates a covered-edge twin, so the ceiling is about 0.75) is also correct. Interventions elsewhere leave every factor identical, because regression on unchanged parent sets recovers the original coefficients.
3. **Engineering is exemplary.** Small, deterministic code reproduces bit-exactly. The exhaustive n=4 check is a nice touch. The paper is careful to label the LLM section as a protocol rather than a result.
4. **The graph-text regime is a useful miniature** of the "text overrides data" concern. The conflict probe is well designed, though not yet diagnostic, as the authors admit.

## 5. Weaknesses

### W1. The headline empirical result is a theorem, not a finding (major)
Table 1's twin column (50.0% for every Gaussian regime) is guaranteed by Prop. 1 for *any* function of the input, including an untrained network. Exact twin generation was already verified (covariance error 1e-14). So the "dissociation" in Figure 1 and the abstract's "twin accuracy is 50.0% (CI 49.6–50.3)" carry essentially no empirical information about learned reasoners. They are a sanity check of the generator.

The only empirical content in Section 5.2 is that a label-order shortcut raises i.i.d. accuracy. That is expected by construction and mirrors known results (varsortability; Caliper). Presenting the bootstrap CI around a quantity that is 1/2 by theorem is misleading about what was learned.

### W2. The "observational limit" oracle is not a limit, and the key quantitative claim is mis-calibrated (major)
The abstract, Figure 1 ("MEC-uniform oracle (observational limit)"), Section 5.2 item 1 and Section 8 claim i.i.d. accuracy rises "past what any reasoner could achieve from observational data without side information (the MEC-uniform oracle)" at 80–81%. This is wrong.

The MEC-uniform oracle takes a majority over member DAGs with uniform weight and scores ties as 0.5. The training prior is not uniform within MECs: P(DAG) is proportional to its number of linear extensions, and queries are sampled balanced between positives and negatives. I computed the exact Bayes-optimal rule that knows only the population MEC and the query, under the paper's actual prior (code: `random_scm` with p_order=0 plus `pick_query`). It achieves **86.4%**, not 80.6%.

It gets worse. A Gaussian reasoner sees Σ, not just the MEC, and the parameter prior (|w| ∈ [0.5, 2], s ∈ [0.5, 1.5]) is informative within an MEC. I checked 16,896 covered-edge twins under the prior: **88.3% of the refit twins M′ have at least one coefficient or noise scale outside the prior's support**. A Bayes-optimal observational reasoner can therefore exploit the parameter prior and exceed 86.4%, possibly by a lot. The true observational limit is unknown.

The qualitative claim for p ≥ 0.9 may still hold (89.5% and 92.5% vs ≥ 86.4%), but the margin is a few points at most and unproven. The statement that "the excess can only come from the label-order cue" is not established. The paper needs a Bayes-optimal observational baseline under its own prior, for example a finite-sample posterior over (DAG, θ) or a strong network trained with labels removed. The ordinal claims should be redone against it.

### W3. Out-of-prior twins confound the interpretation of surplus (major)
Because most refit twins M′ lie outside the training prior, the twin test in the Laplace and interventional regimes is evaluated on a covariate-shifted distribution for one member of each pair. With Laplace noise, P_M ≠ P_M′, so a network can earn "surplus" from any feature separating in-prior from out-of-prior parameterisations, not just from LiNGAM-style direction cues. Prop. 2 allows this ("by any mechanism"). But then the claim in Section 5.3 that "the network has learned to exploit [LiNGAM identifiability]" is not supported without an ablation, for example twins whose M′ is resampled or rejected to lie in-prior, or a model trained on a prior closed under covered reversal.

Relatedly, in the Laplace and interventional regimes the pairs are no longer twins at all. "Twin accuracy" there is just accuracy on a label-balanced subset of adjacent, covered-edge queries, and κ is accuracy normalised by an LR oracle. That is useful but not novel as a metric.

### W4. Novelty is limited and prior work is under-cited (major)
- **Prop. 1** restates non-identifiability (the Causal Hierarchy Theorem / Markov equivalence) as a symmetry argument.
- **Prop. 2** is textbook Le Cam.
- **Prop. 3 and completeness** follow directly from Chickering (1995), which already proves that covered reversals preserve equivalence and connect each class. The exhaustive n=4 check confirms a known theorem.
- **Corr2Cause** (Jin et al., 2024) already builds its labels from MEC reasoning: a hypothesis is valid only if it holds across the MEC. That is closely related to scoring only identified queries.
- **Missing: Montagna et al. (2024/2025), "Demystifying amortized causal discovery with transformers"** (arXiv:2405.16924). It analyses CSIvA through identifiability theory and shows that amortized models fail on noise distributions unseen in training, with training on mixtures as the remedy. This substantially overlaps with the paper's Section 5.4 / Section 8 claim that "causal ability is regime-specific" (a Gaussian-trained model fails on Laplace twins). It must be cited and the contribution repositioned.
- **Missing:** identifiability-aware evaluation in amortized causal discovery (e.g., AVICI reporting MEC-aware metrics) and abstention benchmarks with non-identifiable queries.

The genuinely new pieces are the packaging (twin pairs, surplus, κ) and the protocol. That is a modest contribution for a main-track venue. It might suit the D&B track if an actual benchmark or LLM results were released.

### W5. The research question is not answered (major)
The posed question is whether *models* (in context, LLMs) reason causally. The paper evaluates no LLM. It also tests no strong amortized causal discovery model (AVICI/CSIvA), only a bespoke 4-variable DeepSets network. The empirical conclusions are either theorems (W1) or results about this small network:
- it learns a label shortcut when given one;
- it learns to use Laplace or interventional signal when trained on it;
- it does not transfer across noise types.

None of this bears on the question for LLMs. The "pre-registered protocol" has not been registered anywhere (no OSF or AsPredicted link or timestamp), so the term is a misnomer.

### W6. Test coverage is narrow (moderate)
- Twin pairs come only from single covered reversals, so every twin query is an *adjacent* pair (the reversed edge). Non-adjacent unidentified ancestral queries and deeper twins (43% of unidentified pairs) are never tested. A model could in principle behave differently on adjacent queries, and the test cannot detect this.
- n = 4, linear, no confounding.
- The interventional regime uses a single random target per dataset, which caps the ceiling at about 0.75 and makes κ noisy.

### W7. Statistics (moderate)
- Between-seed variation is estimated from at most 2 seeds; 4 of 8 regimes have 1 seed. κ for int32 differs by 0.18 between its two seeds, so the 32→128 trend (0.62→0.85) rests on one 128 run and is not reliable.
- No CIs are given for κ: it is a ratio, and an LR ceiling near 0.75 amplifies noise.
- There are no ablations of architecture or features. For example, the product features hand the network second-order statistics but no explicit higher moments, which matters for the Laplace and cross-regime conclusions ("captures none of the information").
- The obs_p0 model (74.6%) is well below even the weak 80.6% oracle and the 86.4% Bayes reference, so the trained reasoners are far from optimal. This weakens any statement about what such models "can" do.

### W8. Smaller issues
- The abstract says the label-order model is "above the 81% ceiling of an oracle that knows the true MEC". That oracle is suboptimal (W2), so "ceiling" is the wrong word.
- The Section 6 decomposition ignores that frac_identified differs slightly across regimes (43.9 vs 45.7). This is fine as an approximation but should be noted.
- The "MEC + label-order oracle" falls below the MEC-uniform oracle at p=0 (79.0 vs 80.6) because it trusts uninformative labels. It is a reference point, not an oracle.
- Section 8: "An LLM that has memorized that smoking causes cancer will score at chance on twin pairs about smoking and cancer". That is true only if the twin with the reversed arrow is presented under the same names, which collides with the LLM's world-knowledge prior. The protocol uses abstract names, so this example is not actually tested.
- `summarize.py` glob fragility and the dead `int8_p0` entry (Section 2).

## 6. Questions for the authors

1. What is the Bayes-optimal accuracy of an observational reasoner under your exact prior, given (a) the population MEC (I get 86.4%) and (b) Σ or the finite sample, using the parameter prior? Do p=0.9 and p=1 still exceed it?
2. How does the Laplace and interventional surplus change if twins M′ are constrained to lie in the training prior's support, or the training prior is made closed under covered reversal?
3. How do you position the work relative to Montagna et al. (2024) and to Corr2Cause's MEC-based labelling?
4. Can you run at least one real LLM (an open-weights model on CPU is feasible at n=3–4 with short tables) on formats F1/F3 of your protocol? Without this the paper does not address its own research question.
5. Why restrict to single-reversal (adjacent) twins when your BFS already yields deeper twins?

## 7. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 6 (theory correct; the "observational limit" claim is wrong) |
| Significance | 4 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** (weak reject) |

## 8. Decision

**Major revision / weak reject.** The code is excellent and reproduces bit-exactly, the propositions are correct, and the citations are real. But:
- the headline empirical finding is a consequence of Prop. 1 rather than evidence;
- the "observational limit" against which the shortcut is measured is mis-calibrated (86.4%+ rather than 80.6%);
- the surplus results are confounded by out-of-prior twins;
- closely related prior work (Montagna et al.) is missing;
- the actual research question (do models, i.e. LLMs, reason causally) is not empirically addressed.

**Most valuable change:** run the protocol on real LLMs and a strong amortized discovery model, and measure the shortcut excess against a Bayes-optimal observational baseline under the paper's own prior.

---

## Addendum: rerun results

I reran all 8 seed-0 training runs: obs_p0, obs_p05, obs_p09, obs_p1, laplace_p0, graph_p0, int32_p0 and int128_p0. Every numeric field in each output JSON is **bit-identical** to the shipped `results/*_s0.json`.

The 4 seed-1 runs were still running on a heavily loaded shared machine when this review was finalised, so they were not compared. Given that the code is deterministic (fixed numpy and torch seeds, one thread), I have no reason to expect a mismatch.

`verify_theory.py` also reproduces `theory_checks.json` exactly.
