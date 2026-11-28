# Response to Review Round 1 (p002)

We thank the reviewer for a careful report, which included a full re-run of our pipeline. We accepted nearly every point. Several of the requested analyses changed our conclusions, most importantly the general claim that parametric forecasts are biased toward "more compute will keep helping". We report those changes as they came out, not only the ones that favour the paper.

**Summary of changes.**
1. New Section 5: a two-sided finite-task minimax analysis (Proposition 4). A Chernoff-certified linear estimator gives the upper bound and a Pratt-integrated Le Cam bound on a Hellinger modulus gives the lower bound. The bracket is tight to a factor of 1.46–3.81 over 330 cells (`code/exp_minimax.py`, `results/minimax_*.csv`, `results/fig_minimax.png`).
2. Corrected erfc-fit statement and disclosure of the LP-tolerance sandwich cell.
3. Theorems 1–3 re-labelled as a known lemma and corollaries of known results, with credit.
4. Stronger parametric baselines on WalkSAT and synthetic laws: zero-inflated BB, 2- and 3-component Beta mixtures, ZI mixture, BIC choice, and a budget-matched dynamic-sampling design (`code/exp_baselines.py`).
5. New Section 6.3: analysis of public Large Language Monkeys data (MATH, 7 Pythia models, 128 problems × 10,000 samples), including the protocol we had only pre-registered (`code/exp_llm.py`).
6. An efficiency report and a split coverage count. Cost analysis of the capped first-success design.
7. All minor points addressed (details below).

All new numbers come from code run in this environment, with outputs in `results/`. The LLM counts are public data whose provenance is documented in the paper (Sec. 6.3) and in the README.

---

## Fatal issue

> **Mismatch between paper and code output (erfc fit). The paper says 2E_n is within 0.035 of erfc(n/√k); the correct number is 0.069; relative deviations up to 59%; `sandwich_holds=false` not disclosed.**

**Accepted and fixed.** The 0.0346 in the code was max|E_n − ½erfc|, so for the width the correct figure is 0.069 (n = 128, k = 453,259). `exp_theory_check.py` now writes both quantities explicitly (`max_abs_dev_width_2E_vs_erfc_n_ge_16` = 0.0691 and `max_abs_dev_E_vs_half_erfc_n_ge_16` = 0.0346). Section 4 now states:
- The ratio 2E_n/erfc(n/√k) lies in [0.63, 1.09] for n ≥ 16, and in [0.94, 1.09] wherever 2E_n ≥ 10⁻³. The 59% relative deviation is the cell n = 16, k = 33, where E_n = 2.6×10⁻⁵.
- The maximum absolute deviation is 0.069. The paper says explicitly that the first version's 0.035 was a factor of 2 too small.
- The erfc rule is called "a numerical approximation, not a theorem". The "erfc-rule horizons" (77 / 1,234 / 19,754) are removed. The horizon paragraph now gives the rigorous bound k ≤ n²/ln(4/δ) and the exact LP grid horizons (k/n² ∈ [0.15, 0.28] at width 0.01).
- The sandwich check is disclosed. E_n/ρ ∈ [0.521, 1.00004]. One cell exceeds 1 by 4×10⁻⁵ (relative), within the LP solver's 10⁻⁴ tolerance. `theory_checks.json` now reports `sandwich_holds_exactly = false`, `n_cells_ratio_above_1 = 1` and `sandwich_holds_within_lp_tol = true`, and the paper states this.
- Abstract and contributions now say "equals erfc(n/√k) times a factor between 0.63 and 1.09".

---

## Major issues

### M1. The headline finite-task claim is not supported by the lower bounds

The reviewer was right on all five sub-points. Instead of only weakening the claim, we replaced the analysis, following the reviewer's "highest-leverage change".

**What we did.** Section 5 now characterizes the minimax 95% half-width R*(n, k, M) through a single object, the Hellinger modulus ω(η) = sup{θ(G) − θ(F) : 1 − BC(π_F, π_G) ≤ η}, with **both laws free**. The old analysis used three fixed reference laws.

- **Upper bound (Prop. 4a).** A Chernoff-certified linear estimator gives intervals valid for every F and for fixed tasks. By convex duality (Sion plus Cauchy–Schwarz), its optimal length is exactly ω(1 − (ε/2)^{1/M}). Numerically, we evaluate the certificate rigorously on a 280k-point grid with Lipschitz slack, using the weak-duality form, which is valid for any dual variable. The achieved R_up is within a median factor of 1.06 of the theoretical optimum for this estimator class.
- **Lower bound (Prop. 4b).** Pratt's identity for expected CI length is integrated along the segment between the two modulus-optimal laws, with TV ≤ √(1 − BC^{2M}). It is computed by a well-scaled SOCP over pairs of grid laws. Grid laws are genuine laws, so the bound is rigorous. It is a median 6.8× larger than the two-point bound from the same pair.
- **Result.** R_up/R_lo ∈ [1.46, 3.81] (median 2.1) over n ∈ {16, 32, 64}, 22 values of k ∈ [n, 4n²] and M ∈ {240, 10³, 10⁴, 10⁵, 10⁶}. The first version's gap was 5–35×.

**Point by point:**
- **(a) "The 43/95/210 horizons come from one estimator."** Now every horizon is reported as a bracket: "achieved" (from R_up) and "impossible beyond" (from R_lo, which binds any procedure). At δ = 0.02 and M = 10⁶ these are 61/85, 127/184 and 227/379 for n = 16, 32 and 64.
- **(b) "R_lo at (16,116), (32,283), (64,689) is below δ."** That was correct for the old χ² bound around fixed reference laws. The new lower bounds at these points are **0.037 at (16,116), 0.039 at (32,256) and 0.056 at (32,323), the grid points around 283, and 0.058 at (64,689)**, all above δ = 0.02. The statement "these grid points cannot be certified" is therefore now a theorem-backed statement about *every* procedure, not an artefact of one estimator.
- **(c) Width vs half-width mismatch.** Fixed. All finite-task quantities are now half-widths, and the population comparator is E_n (half-width) at the same δ = 0.02. The population horizons on this basis are 121, 483 and 1,924. The gap to the finite-M horizons is moderate at n = 16 (85 vs 121) and large at n = 64 (379 vs 1,924). The paper reports both and no longer says "0.3n²" for this comparison. A conventions paragraph has been added to Section 3.
- **(d) 8-point k grid.** Now 22 log-spaced k per n, with log-interpolated crossing points alongside the grid values (`minimax_horizon.csv`). "2.7n–3.3n" is replaced by "achievable 3.5–4n, impossible beyond 5.3–5.9n at M = 10⁶, δ = 0.02, for n = 16–64", stated as holding within our range only.
- **(e) "41×" is nearly vacuous.** Agreed and removed. We now report what the reviewer suggested. At fixed M, R_lo is ∝ M^{−1/2} at k = n (pure sampling error) and grows about 120-fold from k = 64 to k = 689 at n = 64, M = 10⁶, while E_n stays at 2.8×10⁻⁴.

We also state honestly that the achievable side barely moved. The first version's Hoeffding estimator is only 1.0–1.4× the new Chernoff R_up. The improvement is almost entirely on the lower-bound side.

### M2. Limited novelty

**Accepted.** The paper now presents:
- Lemma 1 as "the classical moment-matching / best-approximation duality [Wu & Yang 2016, 2019]", included with its proof for completeness;
- Corollary 2 (formerly Theorem 2) as an application of Newman–Rivlin and Sachdeva–Vishnoi via y = z²;
- Proposition 3 (formerly Theorem 3) as Chebyshev's best approximation of 1/(x−c) after an affine change of variables;
- arXiv:2609.09245 with its authors (Singh & Singh, 2026) and credit for identification;
- the concurrent arXiv:2611.21266 (Ye, 2026), discussed in the introduction and related work. It gives simultaneous, variance-adaptive regions for all bounded functionals, which differs from our per-functional minimax bracket. We say that we have not compared widths with it.

The contributions list now states plainly what is new: the TTC-specific rates and the predictability–ceiling trade-off; the finite-task minimax bracket (an instance of Juditsky–Nemirovski / Polyanskiy–Wu modulus theory, which we cite, with constants we can compute); and the empirical checks. We do not claim a breakthrough.

### M3. The WalkSAT parametric comparison is close to a strawman

**Accepted. The new results confirm the reviewer's suspicion, and we changed the conclusions accordingly.** `exp_baselines.py` fits BB, ZIBB, BMix2, BMix3 and ZIBMix2, plus the BIC choice, under a fixed design and a budget-matched dynamic-sampling design (a simplified variant in the spirit of Kazdan et al.; we say it is not their exact algorithm). It runs on WalkSAT (n ∈ {16, 64}, 10 subsets) and on the four synthetic laws (n = 16, M = 5,000, 20 replications).
- **WalkSAT.** The zero-inflated BB removes most of the bias: pass@1000 bias +0.055/+0.002/+0.016 vs +0.287/+0.135/+0.111 for BB at n = 16. The paper now states this, and that the test bed is a stress test for models without a zero atom, not a neutral benchmark. A residual issue remains: BIC picks plain BB in 14 of 30 n = 16 fits, so the BIC-selected forecast still overestimates by up to 0.13.
- **Synthetic.** BIC picks ZIBB under both the 30%-unsolvable law (bias +0.002, correct) and the 30%-hidden-solvable law (bias −0.167 at pass@1024, −0.262 on the ceiling). No model in the family is right on both nearly indistinguishable laws, which is the identification point of Lemma 1.
- The Abstract and Discussion **no longer** say that forecasts are "biased toward more compute will keep helping". Section 7 states that the direction of the error is not universal and withdraws the earlier claim. The LLM data (M4) confirm this.

### M4. No LLM data

**Accepted.** Hugging Face is blocked from our environment. However, the public code release of Schaeffer et al. (2025) contains the per-problem correct-sample counts (out of 10,000) for the Brown et al. MATH/Pythia subsets (7 models × 128 problems). It is derived from `ScalingIntelligence/monkey_business`. We extracted these counts, with provenance and commit hash, and checked them against the stored pass@9000 values (agreement within 0.0019). Section 6.3 runs the pre-registered protocol:
- **P3 (the informative, falsifiable prediction) failed.** Beta-binomial forecasts at k = 4n² were above the truth in only 57 of 140 cells (41%). From n = 16, the error at k = 10⁴ ranges from −0.143 to +0.315 across models. ZIBB underestimates by up to 0.464, because problems with zero successes at n = 16 are often solved within 10⁴ samples. The power law underestimates by 0.24–0.48. We report this prominently, in the Abstract and Sections 6.3 and 7.
- **P2**, a theorem check as the reviewer noted, held at n = 16. At n ≥ 32 it could not be checked numerically. The exact-equality LPs are reported infeasible, and an L1 slack of 10⁻⁸ inflates the width beyond 2E_n at n = 64 and 128. We report this as a numerical manifestation of the ill-conditioning of Section 5 and restrict LLM oracle intervals to n ≤ 32.
- The box interval covered in all 1,120 cells. The oracle interval from n = 16 for Pythia 12B pass@10⁴ is [0.582, 1.000], i.e. uninformative, consistent with Singh & Singh.
- Limitations: one benchmark, one model family, M = 128; other subsets (Llama-3, Gemma; CodeContests, MiniF2F) were not reachable.

### M5. The finite-task procedures are very conservative; efficiency not reported

**Accepted.** Changes:
- **Coverage is split.** 720/720 two-sided cells, plus 120/120 pass@∞ cells for which only the lower end is meaningful. The phrase "840/840" no longer appears as a headline.
- **Efficiency vs CLT where k ≤ n.** The box half-width is a median 1.80× (range 1.63–2.74×) the CLT half-width around the unbiased estimator (0.113 vs 0.062).
- **Efficiency vs oracle where k > n.** The box exceeds the oracle half-width by 0.13–0.26.
- **The first version's minimax-Hoeffding estimator** has 99.97% coverage against a nominal 95%, and its median half-width is 11× the population identified half-width. The paper calls this "heavily conservative".
- **What is achievable at all.** The new minimax bracket shows that at M = 240 no procedure can reach ±0.035 even at k = n = 16, and the best achievable at k = n is 0.087. Part of the box's width is therefore unavoidable at benchmark scale. We state what is unavoidable and what is excess.
- **First-success design.** The capped first-success design gets a simultaneous DKW–Massart band of ±0.088 at M = 240 for all k ≤ D (Section 7).

---

## Minor issues

| Issue | Change |
|---|---|
| Gap stated as 5–35× in Sec. 5 but "up to an order of magnitude" in Limitations | Both replaced by the new bracket, 1.46–3.81, stated consistently in the Abstract, Contributions, Sec. 5, Limitations and Conclusion. |
| "Sharp/exact" intervals exclude the truth (Beta(0.3,3) pass@32; WalkSAT C pass@32) | **Explained and relabelled.** Synthetic: with the exact beta-binomial count law, the interval [0.536819, 0.536829] contains the exact truth 0.536824. The miss came from 2×10⁵-draw MC count laws vs 2×10⁶-draw MC truth. WalkSAT: the oracle interval is sharp for the *plug-in* law, and the plug-in value 0.76308 lies in [0.76307, 0.76310]; the reported truth is the unbiased estimator (0.76322). Both explanations are now in the paper, and the intervals are called "sharp up to Monte Carlo and LP error ≈ 10⁻³". |
| Bootstrap coverage 84–88% at M = 5,000 hidden by averaging | Now reported by M: 84–88% (M = 5,000) and 94–96% (M = 500). |
| Majority-vote collapse rests on 5 values of n | Now called "an approximate numerical regularity, not a law", with the values (0.072 / 0.058 / 0.063 at n²/k = 0.94 / 0.98 / 0.82, spread ±11%). |
| Cite 2609.09245's authors; discuss 2611.21266 | Done: Singh, P. & Singh, P. (2026); Ye, K. (2026). Both verified by web search. |
| `chi2_lower` docstring | Now describes the χ² *sum* constraint actually used, and notes that the function is superseded by `exp_minimax.py`. |
| Width vs half-width conventions; table header "population E_n" | A conventions paragraph in Sec. 3. Population: width 2E_n, half-width E_n. All Sec. 5 tables are half-widths, and the header now reads "population E_n" deliberately, meaning half-width. "Width ≤ 4e^{−n²/k}" and "E_n ≤ 2e^{−n²/k}" are now labelled consistently. |
| Unnecessary δ₀ padding in Lemma 1's proof | Removed: σ⁺ and σ⁻ each have mass exactly 1. |
| The capped first-success design deserves a cost comparison | Added to Sec. 7. Expected cost Σ_{j<D}(1 − pass@j), worst case D. On WalkSAT with D = 1,000 it costs 162–451 rollouts per task (2.5–7× n = 64) and gives a simultaneous band of ±√(ln(2/ε)/(2M)) for all k ≤ D (0.0136 at M = 10⁴). Fixed n = 64 at M = 10⁴ cannot beat ±0.13 at k = 897. We now recommend this design when a target horizon D is known, and note that it does not help with pass@∞. |

## Questions for the authors

1. **Tight finite-M rate?** Partly answered. We have no closed form, but R* is characterized by the Hellinger modulus to within a factor of 1.46–3.81 across the computed range (Prop. 4 and Table, Sec. 5). This is the Wu–Yang / Polyanskiy–Wu / Juditsky–Nemirovski machinery, made explicit and computable.
2. **ZIBB / Beta mixtures?** Done (M3). ZIBB fixes WalkSAT but fails on the hidden-solvable synthetic law and on LLM data.
3. **Brown et al. release?** Done for the MATH/Pythia subsets (M4).
4. **Relation to arXiv:2611.21266?** Discussed in Sec. 2. It gives simultaneous variance-adaptive regions; we give per-functional minimax rates and brackets. No width comparison yet (listed in Limitations).
5. **First-success design?** Yes. For a known target horizon it is the practical recommendation (Sec. 7), with costs reported.

## Files added or changed
- `paper.md` (rewritten sections 1–9); previous version kept at `reviews/paper_r0_backup.md`.
- New: `code/exp_minimax.py`, `code/exp_baselines.py`, `code/exp_efficiency.py`, `code/exp_llm.py`.
- Updated: `code/exp_theory_check.py`, `code/exp_design.py` (docstring).
- New results: `minimax_bounds.csv`, `minimax_horizon.csv`, `fig_minimax.png`, `baselines_{raw,summary,bic_selection}.csv`, `capped_design_cost.csv`, `efficiency_{walksat.csv,summary.json}`, `llm_monkeys_math_pythia_counts.csv`, `llm_{raw,summary}.csv`, `llm_prereg.json`, `fig_llm.png`, and `theory_checks.json` (updated).
