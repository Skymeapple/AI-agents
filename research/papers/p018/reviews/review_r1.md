# Review (Round 1): "How Much Does Exhaustive Small-Case Verification Certify? A Survival Analysis of Machine-Accepted Graph Conjectures"

**Research question it answers:** 数学研究：AI能否独立证明重要的新定理？ (Can AI independently prove important new theorems?)
**Reviewer role:** independent area chair
**Recommendation:** Major revision (weak reject as it stands). **Overall: 5/10**

---

## 1. Summary

The paper asks how much evidence a Graffiti-style "true and sharp on every connected graph with ≤ k vertices" gate provides. It enumerates 12,922 inequality templates s·(qX − pW) ≤ c over 14 exact graph invariants. For each k = 4..10 it takes the sharpest constant E_k that holds on all connected graphs of order 2..k, then checks those statements against every connected graph of order ≤ 10 (11,989,763 graphs), a deterministic 4/256 `geng` slice of order 11 (12.97M graphs), and 15,860 random and structured graphs of orders 11–16. The headline findings are:
1. About 80% of accepted statements are refuted, whether k = 4 or k = 10.
2. Most failures are "drift": the per-order maximum keeps growing. Proposition 2 gives a sufficient condition for this, and trajectory statistics predict refutation with AUC ≈ 0.99.
3. For "stable" statements (record age ≥ 2), the observed false-acceptance rate falls from 29.5% to 4.9%, and the one-step hazard falls with record age.
4. The 134 stable statements that fail at orders 11–16 are caught by "structured families" and hardly ever by G(n,p).
5. A simplified Dalmatian significance filter does not improve precision.

The paper ends with a reporting standard and a pre-registered protocol for LLM conjecturers. That protocol was not run.

## 2. Reproduction and verification

- **Full rerun.** I reran the full pipeline in an isolated copy: built geng and the C engine, regenerated every invariant file, then ran `gen_large_sample.py`, `analysis.py`, `analysis2.py` and `analysis3.py`. The invariant binaries for n = 2..10, the order-11 slice and the large sample are **byte-identical** to the submitted `data/`. `summary.json`, `summary2.json`, `summary3.json`, `late_counterexamples_stable.json`, `conjectures_by_k.csv`, `templates.csv` and `per_order_maxima.npy` are also **byte-identical / exactly equal** to the submitted `results/`. The brute-force validation reports 0 mismatches on 1,055 graphs.
- **Paper vs output.** I checked every number in the abstract, in Tables 1–3 and in Sections 5.2–5.5 against the JSON outputs. They match, including: 80.4%/79.4%; the attained/not-attained splits; relaxed-by-one 77.6%; AUCs 0.974/0.988, 0.962/0.992, 0.978/0.995; the confusion matrix 9,573/157/664/2,528; the drift census 8,530/1,061/1,327/2,004 and 99.9%/9.5%/0.15%; the hazards 0.016, 0.015, 0.017, 0.044; the 134 late stable failures by order and age; 7 caught by G(n,p); 127 caught only by structured families; 78% and 5% coverage; the gap rule 30 flagged, 17 TP, 117 FN; tiny-tight 0.25% of 1,591 vs 11.3%; the clean filter 114/1,127; CV AUCs 0.971/0.971/0.989 and their calibration; Dalmatian 211 significant, 75.4% vs 79.9%; 2,660 survivors at k = 10.
- **χ ≤ ω + 1.** I checked this directly from `data/inv_n*.bin`. max(χ − ω) is 1 on orders 5–9 and 2 at order 10, attained by exactly one graph with invariant vector (10, 35, 7, 7, 2, 4, 6, 2, 5, 2, 2, 2, 5, 50). A 7-regular graph on 10 vertices with α = 2 and ω = 4 has a 2-regular, triangle-free complement with independence number 4, and that forces the complement to be 2C₅. So the graph is indeed C₅ ∨ C₅. Caveat: `results/check_chi_omega.txt` is **not produced by any script in `code/`**. This is a minor reproducibility gap. Add the script.
- **Template counts.** 14·13·7 = 1,274; 14·C(13,2)·3·2 = 6,552; 14·C(14,2)·2·2 = 5,096, where the second factor counts multisets of size 2 from the other 13 invariants. Total 12,922. Correct.

Conclusion: **no numerical mismatches.** Reproducibility is exemplary.

## 3. References (spot-checked by web search)

| Reference | Verified? | Notes |
|---|---|---|
| Pastorek 2026, AutoGraphForge, arXiv:2609.03478 | Yes | Exists. "≈348,000 graphs" (348,207: HoG + 273,192 census + special families), Graffiti3 generator, Lean 4, and 6,522 survivors are all confirmed. |
| Feng et al. 2026, arXiv:2602.10177 | Yes | "Towards Autonomous Mathematics Research", Aletheia agent, Erdős problems. |
| Google DeepMind / Feng et al. 2026, arXiv:2601.22401 | Yes | Reports 50 "technically valid but mathematically vacuous" solutions. The paper's paraphrase is accurate. |
| Brewster, Dinneen & Faber 1995, Discrete Math 147:35–55 | Yes | >200 conjectures checked on graphs with ≤ 10 vertices. "Six of these eleven conjectures failed for the same graph: K₂" is confirmed. |
| Vito & Stefanus 2023, arXiv:2306.07956 | Yes | Adaptive Monte Carlo search for conjecture refutation. |
| Jooken 2025, arXiv:2508.20825 | Yes | "Computer-assisted graph theory: a survey". |
| Larson & Van Cleemput 2016, AI 231:17–38 | Yes | Dalmatian heuristic revisited. |
| Aouchiche, Caporossi & Hansen 2013, EURO J. Comput. Optim. | Yes | Vol. 1, pp. 181–199. Add volume and pages. |
| Hubert et al. 2025, Nature, doi:10.1038/s41586-025-09833-y | Yes | AlphaProof. |
| Georgiev, Gómez-Serrano, Tao & Wagner 2025, arXiv:2511.02864 | Yes | AlphaEvolve at scale. |
| Davila et al. 2025, arXiv:2507.17780 | Yes (title) | Title appears in search results. Author list not independently verified. |

No fabricated or misattributed references were found. Minor issue: Fajtlowicz (1988) lacks a volume and page numbers (it is Discrete Math. 72:113–118).

## 4. Strengths

1. **Exact, reproducible measurement.** The universe is exhaustive up to order 10, the C engine is independently validated, and the pipeline reproduces bit for bit. That is rare and valuable.
2. **The gate is separated from the generator.** The paper characterises the gate's error profile over a whole hypothesis class, not over one system's output. This is the right experimental idea.
3. **Honest scoping.** The paper says plainly that it neither finds new theorems nor answers the headline question, and it separates exhaustive from sampled labels (lower bounds) carefully.
4. **Useful, concrete artefacts.** The χ ≤ ω + 1 / C₅ ∨ C₅ example, the "tiny-graph tightness is a safety signal" observation, and staircase drift (γ(Pₙ) = ⌈n/3⌉) are instructive. The reporting standard (certified order, record age, drift class, falsifier provenance) is sensible and cheap.
5. **Pre-registered predictions P1–P4.** They are falsifiable and well posed.

## 5. Weaknesses

### Major

**M1. The headline "flat 80%" mostly reflects the template grid, not the gate.**
About 66% of templates (8,530) have per-order maxima that strictly increase on orders 8–10. For essentially all of them (e.g. m − n, t − c·m, n − α), sup_G e(G) = ∞, so *no* constant makes the statement true and C_k(e) is false for every k by construction. The fraction refuted therefore converges to roughly the fraction of templates with unbounded supremum, a number set by the authors' choice of invariants and coefficient grid. With a different grid (e.g. only ratios on dimensionally homogeneous pairs, or an LP-fitted slope as in TxGraffiti) the "80%" could be 30% or 95%. The claim that "a larger database does not change this" is likewise near-tautological for unbounded templates. The paper should:
- (a) report the headline separately for templates whose supremum is provably finite and for those provably infinite (Proposition 2 lets the authors certify the latter in bulk, e.g. via dominating-vertex or pendant-path operators);
- (b) show how sensitive the headline rate is to the template grid;
- (c) stop calling the 80% a property of "the gate".

**M2. The drift results are close to tautological.**
"Strictly increasing on orders 8–10 ⇒ 99.9% refuted beyond 10" and "drift statistics predict refutation by order 10 with AUC ≈ 0.99" restate the definition: a sequence that is still rising at the top of the window will, with high probability, exceed its current maximum one step later. The AUC is dominated by the 66% of unbounded templates, which any reasonable heuristic separates. The informative test is prediction *within the stable stratum*. There the paper's own numbers are modest: AUC 0.78–0.83, and the gap rule misses 117 of 134. The abstract should lead with these numbers, not with 0.988–0.995. The abstract's "AUC 0.988 to 0.995" range is also k = 6..8 only; the code reports 0.969 (k = 4) and 0.982 (k = 5) without disclosing them.

**M3. Table 3's trend confounds universe size with horizon and detection power.**
At k = 4, refutation is judged over 6 exhaustive orders plus samples; at k = 9 over 1 exhaustive order; at k = 10 over samples only. The decline from 29.5% to 4.9% therefore mixes (i) better certification, (ii) a shorter exhaustive horizon, and (iii) weaker sampled detection. The paper acknowledges (iii) but presents the factor of "1.1–1.5 per order" as an effect of the universe. A horizon-matched analysis is needed. I computed one from the released data: P(refuted within h exhaustive orders | age ≥ 2).

| k | h = 1 | h = 2 |
|---|---|---|
| 4 | 11.6% | 18.7% |
| 5 | 8.2% | 12.0% |
| 6 | 5.0% | 9.6% |
| 7 | 5.2% | 6.7% |
| 8 | 2.3% | 4.5% |
| 9 | 3.2% | — |

The qualitative claim survives with clear parity effects, but this horizon-matched analysis, not Table 3, is the evidence that should carry the claim.

**M4. The "structured vs random falsifier" result is confounded by density and by how families are labelled.**
The "structured" set (2,360 graphs) includes 300 *random labelled trees* per order (1,800 graphs) and 20 random 3- and 4-regular graphs. The G(n,p) sample draws p ~ U(0.08, 0.92), which at n = 11–16 essentially never produces trees, paths or K_n. Of the 134 stable failures, 2 are refuted *only* by random trees and 41 involve trees (my recomputation). The finding is really "dense Erdős–Rényi graphs miss sparse and extreme graphs", which is well known. It is not evidence that "structure" beats randomness. Fixes:
- (a) relabel random trees and random regular graphs as random;
- (b) add a density-matched random baseline (e.g. G(n, m) with m near n − 1, random graphs conditioned on a degree sequence);
- (c) report coverage per graph checked.

The 390-vs-2,250-per-order comparison is not size-matched either.

**M5. Weak connection to the research question, and no AI system is evaluated.**
The question concerns whether AI can independently prove important theorems. The paper evaluates no AI system: no LLM was run, and Section 7's protocol is unexecuted. The template-space "conjecturer" is a brute-force enumerator. The conclusions about LLM pipelines (e.g. that a system's yield should be "discounted by a factor 1.05 to 5") are extrapolations without empirical support from any LLM. Running even a small version of the Section 7 protocol, with one open-weights or API model on k ∈ {7, 8, 9}, would turn this from a gate calibration into evidence about AI conjecturing. Testing P1 and P2 would also give the paper a result that bears directly on the research question.

**M6. Novelty is incremental.**
Brewster et al. (1995) already established that finite small-order verification of Graffiti conjectures leaves many false, often small-graph-tight, statements. AutoGraphForge and the Graffiti literature explicitly note that finite datasets do not certify universal statements, and the Monte Carlo refutation literature (Roucairol & Cazenave; Vito & Stefanus) motivates structured or adaptive falsifiers. The new elements are the nested-universe survival framing, the hazard-vs-record-age curves, and the dataset. These are useful, but for a top venue the contribution is a careful measurement on a synthetic hypothesis class, with modest theory: Propositions 1 and 3 are elementary, and Proposition 2 is a one-line induction.

### Minor

- m1. **Proposition 3's null is a straw man.** It assumes exchangeable maxima with no ties. Integer invariants produce ties constantly, and per-order maxima of most templates are monotone by construction because they are closed under vertex addition. That real templates beat 1/k is therefore unsurprising. The phrase "no statement ... would be more than a coin flip" is also inaccurate: under the null, P(survive to K) = (k−1)/(K−1) → 0.
- m2. **The record-age effect is largely a cohort effect.** Most old records (age 6–8) are templates whose maximum was set at K₂ or P₃, with slack that grows with n. The same 202-template cohort appears at ages 3, 4 and 5 for k = 6, 7 and 8 with zero hazard. Report hazard by age after excluding tiny-tight templates.
- m3. **Proposition 2 is invoked as the explanation for the 8,530 increasing templates, but never checked.** Show what fraction of drifting templates admit one of the stated operators (dominating vertex, pendant vertex, path extension).
- m4. **No uncertainty quantification.** Templates are strongly correlated: the 134 failures involve only 90 distinct (X, Y, Z) triples and 11 target invariants. Use cluster bootstrap CIs over (X, W) families, or deduplicate templates by logical implication. Hazard cells with n = 18–90 especially need intervals.
- m5. **Inconsistent "stable" definitions.** Age ≥ 1 is used for the CV model and age ≥ 2 elsewhere. Report the CV results at age ≥ 2 as well.
- m6. **"This rate falls with k" (Table 1, last column) is not monotone:** 6.3% → 7.9%, 2.9% → 4.6%. Rephrase.
- m7. **The order-11 slice is non-uniform** (maximum δ = 7, maximum ω = 9). It catches nothing beyond the other falsifiers ("stable_only_n11_slice = 0"), which suggests it adds little. Say so.
- m8. **The Dalmatian comparison** rests on 5–211 significant statements, with no test of significance, and uses a simplified, order-free filter. The negative conclusion is weak. Either implement the actual Dalmatian procedure or soften the claim.
- m9. **Code hygiene.** In `analysis.py`, `tiny_only = jfirst <= 3 & (~attained_top)` has an operator-precedence bug. The variable is unused, so no reported number is affected, but remove or fix it. `check_chi_omega.txt` has no generating script.
- m10. **Venue fit.** The Datasets & Benchmarks framing would be stronger with a released, documented benchmark: a split, a metric and baselines for "predict refutation from in-universe statistics".

## 6. Questions for the authors

1. What fraction of the 12,922 templates have provably infinite supremum over connected graphs? What is the false-acceptance rate restricted to templates with finite supremum, for k = 4..10?
2. With density-matched random falsifiers and random trees counted as random, does the "structured ≫ random" asymmetry survive?
3. Can you run P1/P2 with at least one LLM at k = 8? Without this, how do you justify the "discount factor 1.05–5" for real systems?
4. How do the stable-stratum results change when tiny-tight (K₂/P₃) templates are excluded?
5. How sensitive are the results to the coefficient grid, e.g. LP-fitted slopes as in TxGraffiti?

## 7. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 8 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 8. Decision

**Major revision (weak reject in current form).** The engineering and reproducibility are excellent, and every number I checked matches the code. The two headline claims, however, are largely artefacts of the hypothesis-class design: flat 80% precision and AUC ≈ 0.99 drift prediction. The falsifier-asymmetry claim is confounded by graph density and by labelling random trees as "structured", and the stated research question is not engaged empirically because no AI system is evaluated. Two changes would move the paper to borderline accept: re-centring it on the bounded-supremum / stable stratum with horizon-matched rates and CIs, and running the Section 7 protocol on at least one LLM.
