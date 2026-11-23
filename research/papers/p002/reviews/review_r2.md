# Review, round 2: "How Far Can a Fixed-Rollout Evaluation See? Polynomial-Approximation and Minimax Limits on Forecasting Test-Time-Compute Gains"

Reviewer role: area chair / senior reviewer. Recommendation: **major revision** (borderline). Overall score: **6/10**.

## 1. Summary

The paper asks how far a fixed-n, M-task evaluation can forecast pass@k (and related test-time-compute functionals) when nothing is assumed about the law F of per-task success probabilities. It has three parts.

1. **Population level.** The worst-case identified-set width equals 2E_n(g), via the classical moment-matching / best-approximation duality. Known approximation results then give an n² horizon for pass@k (Newman–Rivlin plus Sachdeva–Vishnoi), a closed form r^n for the resample-until-accept ceiling (Chebyshev's 1/(x−c)), and non-identification of pass@∞.
2. **Finite-task level (new in r1).** A Hellinger-modulus bracket on the minimax 95% half-width R*(n,k,M). The upper side comes from a Chernoff-certified linear estimator whose optimal length is ω(1−(ε/2)^{1/M}). The lower side is a Pratt-integrated Le Cam bound. Numerically, R_up/R_lo ∈ [1.46, 3.81] over 330 cells.
3. **Empirics.** The paper tests parametric forecasts against stronger baselines (ZIBB, Beta mixtures, BIC, a budget-matched dynamic design) on WalkSAT with known ceilings and on synthetic laws. It also analyses public Large Language Monkeys MATH/Pythia counts (7 models × 128 problems × 10⁴ samples), where the forecast error has no consistent sign and the pre-registered overshoot prediction P3 fails.

The revision is substantial and honest. It withdraws the previous headline claim that forecasts are biased toward "more compute keeps helping", and it reports a failed pre-registered prediction. It is clearly better than r0.

## 2. Reproduction

I copied the package to a scratch directory and reran the following:

- `exp_theory.py` and `exp_theory_check.py`. All numbers match `theory_checks.json`: ratio E_n/ρ ∈ [0.5213, 1.00004]; 1 cell above 1; max |2E_n − erfc| = 0.06913 at (128, 453,259); ceiling closed form within 0.33%. From `theory_passk_with_rho.csv` I recomputed 2E_n/erfc ∈ [0.628, 1.091] (n ≥ 16, E_n > 10⁻⁵), and [0.946, 1.091] where 2E_n ≥ 10⁻³. The horizon tables also match.
- `exp_minimax.py` at cells (16,16), (16,95), (16,116), (32,203) and (64,689), for all five M. R_lo and R_up agree with `minimax_bounds.csv` to all printed digits.
- I independently checked the lower-bound pair at (16, 95, M = 240). Both laws are genuine probability vectors. With independently computed binomial count laws, 1−BC = 0.00209 ≤ η = 1/240 and Δθ = 0.356. A Monte Carlo estimate of the product TV(mid^M, G^M) is 0.38, below the Hellinger bound of 0.46 that the code uses. The lower bound therefore appears genuine. The worst-case pair is "54% mass at p≈0" against "84% mass at p≈0.015", i.e. the unsolvable versus hidden-solvable dichotomy.
- `exp_efficiency.py`. Output is byte-identical to `efficiency_summary.json`.
- `exp_llm.py`. I deleted the cached `llm_fits.pkl`, so the parametric fits were recomputed from scratch, and reran the script. See Section 6 for the result.
- Against the stored CSVs I checked these claims: the Section 5 table and the horizons table (all 21 entries); the WalkSAT intervals and bias numbers in Sec. 6.2; the baseline biases (BB/ZIBB/BMix/BIC on WalkSAT and synthetic laws); the LLM bias numbers at k = 10⁴ (n = 16 and n = 128); the box half-widths; the DKW numbers; and the capped-design costs. They match, with the exceptions listed under Weaknesses W1.

## 3. Status of the round-1 issues

| Round-1 issue | Status |
|---|---|
| FATAL: erfc fit off by 2×; sandwich_holds=false undisclosed | **Resolved.** 0.069 is now stated, along with the ratio range 0.63–1.09 and the LP-tolerance cell. The rule is labelled a numerical approximation. Verified by rerun. |
| MAJOR: finite-task horizon claim not supported by lower bounds | **Largely resolved.** The new Hellinger-modulus bracket has both laws free. R_lo now exceeds δ = 0.02 at the cited points (0.037 at (16,116); 0.039/0.056 at k = 256/323 for n = 32; 0.058 at (64,689)), and half-width conventions are unified. The new k grid has 22 points. The "41×" ratio has been removed. Remaining caveat: these are *global worst-case* statements, and the paper's wording ("no 95% procedure can forecast…") still invites a reading in which they hold for every benchmark (W3). |
| MAJOR: limited novelty | **Acknowledged and relabelled honestly, but not resolved.** The finite-task bracket is now the claimed main technical contribution. It is a direct instance of Donoho (1994) / Donoho–Liu modulus theory, and the paper does not cite that work. It also competes with Ye (2026), who claims first-order worst-case-optimal widths for this same model, and no comparison is made (W2). |
| MAJOR: WalkSAT strawman | **Resolved.** ZIBB, mixtures, BIC and dynamic sampling are evaluated. The paper reports that ZIBB fixes WalkSAT, labels the test bed a stress test, and withdraws the general bias-direction claim. |
| MAJOR: no LLM data | **Partly resolved.** Large Language Monkeys MATH/Pythia counts are analysed with documented provenance, and P3 is run and reported as failed. Scope is narrow: one benchmark, one weak model family, M = 128. There are also numerical inconsistencies in the P2 and oracle reporting (W1, W4). |
| MAJOR: conservativeness / efficiency unreported | **Resolved.** Coverage is now split 720/120. The paper reports box/CLT = 1.80× (1.63–2.74), box-minus-oracle excess 0.13–0.26, and certified/population = 11×, and it calls the methods heavily conservative. |

## 4. Strengths

- **S1.** Proposition 4 is correct as far as I can verify line by line. On (a): Chernoff gives the one-sided tails. The sup over F includes point masses, so the bound holds for fixed tasks. The perspective α·ln E e^{h/α} is jointly convex in (h, α) and the objective is concave in (F, G), so Sion applies. Cauchy–Schwarz gives inf_h = 2 ln BC, attained at h = (α/2)ln(π_G/π_F). The α-minimisation then yields the threshold BC ≥ (ε/2)^{1/M}. On (b): Pratt's identity holds by Fubini. P_mid(θ′ ∈ CI) ≥ 1−ε−TV. TV ≤ √(1−BC^{2M}) by tensorisation. Joint convexity gives 1−BC(mid, F_λ) ≤ λη/2. The closed form follows by bounding the non-increasing integrand by its value at λ = 1. The numerical implementation is careful: genuine grid laws, exact recomputation of 1−BC with repair, a right-endpoint Riemann sum, and a weak-duality evaluation of the upper certificate with Lipschitz slack.
- **S2.** The bracket is genuinely two-sided and reasonably tight (median 2.1×). It turns the r0 claim into one backed by lower bounds.
- **S3.** The empirical reporting is unusually honest: a failed pre-registered prediction, a withdrawn headline claim, explicit statements that ZIBB fixes WalkSAT but fails on LLM data, and disclosure of LP-tolerance artefacts.
- **S4.** The predictability–ceiling trade-off (Corollary 3, √(t/q) depth law) is a clean and practically quotable consequence.
- **S5.** Reproducibility is good. Code runs in minutes per cell, outputs reproduce exactly, and data provenance is documented with a commit hash.

## 5. Weaknesses

### W1. Remaining mismatches between text and stored outputs (each small, but under this venue's standard every one must be fixed)

1. **P2 at n = 16, Sec. 6.3.** The paper says P2 "held in all 7 cells at n = 16 (widths ≤ 3×10⁻⁴ vs 2E_n = 4.4×10⁻³)". In `llm_prereg.json`, Pythia 12B at n = 16, k = 64 has `oracle_width = NaN`: the LP failed and `oracle_lo` is NaN in `llm_summary.csv`. Only 6 of 7 cells were checked.
2. **P2 at n = 32 and the "report only n ≤ 32" decision.** The paper says P2 "could not be checked at n ≥ 32" and that oracle intervals are therefore reported "only at n ≤ 32". In `llm_prereg.json`, n = 32 gives two computed cells *exceeding* the theorem bound: 6.9B has 0.00781 and 12B has 0.00859, against 2E_n = 0.00468. Two further cells are NaN. In `llm_summary.csv`, Pythia 12B, n = 32, k = 256 has oracle width 0.0086 > 2E_n = 0.0047. The n = 32 oracle intervals are therefore affected by the same slack artefact the paper uses to exclude n ≥ 64. Either exclude n = 32 too, or say that the n = 32 oracle intervals are slack-inflated.
3. **P3 breakdown.** The paper says P3 "held in 60–70% of cells for 70M, 160M and 410M and in 0–40% for 1B–12B". `llm_prereg.json` gives Pythia 12B at n = 16 frac_above = 0.7. The statement is false for 12B.
4. **Lower-bound improvement, Sec. 5.** The paper says the lower bound "rose by a factor of 2–6" relative to r0. Merging `design_bounds.csv` with `minimax_bounds.csv` gives factors from 1.58 (n = 64, k = 7420) to 11.3 (k = n cells) over the 27 shared cells.
5. **Synthetic BIC counts.** "100/100 fits" and "95/100 fits" count 20 replications × 5 targets. These are 20 and 19 distinct fits, so the stated counts inflate the apparent sample size. The WalkSAT "14 of 30" is counted correctly, so the units are inconsistent within the paper.

None of these changes a conclusion. Items 1–3, however, concern exactly the pre-registered checks whose credibility the paper leans on.

### W2. Novelty and positioning of the "main new technical result"

- **Missing prior work.** Proposition 4 says that the minimax expected length of fixed-confidence intervals for a linear functional over a convex class is governed, up to a constant, by a (Hellinger) modulus, and that it is attained by an affine/linear procedure. This is the core of Donoho (1994, *Ann. Statist.* 22:238–270; optimal-recovery CIs within 19% of minimax among all procedures in the Gaussian case) and Donoho & Liu (1991, "Geometrizing rates of convergence" II/III, Hellinger modulus for i.i.d. data). It also overlaps with Cai & Low (2004) on adaptive CI length. The paper cites Juditsky–Nemirovski and Polyanskiy–Wu but not these. The contribution is an instantiation with computable constants, which is useful, and it should be framed that way.
- **Concurrent work not compared.** Ye (arXiv:2611.21266, confirmed to exist; K. Ye, UC Irvine) treats this same fixed-benchmark binomial-mixture model and claims *first-order worst-case-optimal width*. That is directly competitive with Proposition 4. "We have not compared widths" is not acceptable for the paper's main claimed contribution. At least one numerical comparison of Ye's region against R_lo and R_up at matching (n, k, M) is needed.
- **Section 4 is a careful synthesis of known results,** as the paper now says. Singh & Singh (arXiv:2609.09245) already give identification and LLM-data ambiguity. The genuinely new pieces are therefore the TTC-specific rates (mostly one-line substitutions), Corollary 3, the computed bracket, and the empirical study.

### W3. Worst-case framing is over-read in the headline

The abstract says: "no 95% procedure can forecast pass@k beyond k ≈ 85, 184 and 379". What is proved is weaker: no procedure has *worst-case-over-F* expected half-width ≤ 0.02. The extremal pair I extracted puts more than 50% of the mass at p≈0 against mass at p≈1/k, so it is the hidden-solvable construction. For a specific benchmark, an adaptive procedure such as the box interval can be far tighter. The paper's own WalkSAT oracle intervals at n = 16 are within ±0.003 at k = 100 ≈ 6n. Please (i) qualify the statement everywhere ("uniformly over F"), and (ii) ideally add a local or instance-dependent version, such as the modulus restricted to a Hellinger neighbourhood of the observed π̂. That would tell a practitioner what *their* data can support. As written, Discussion §7 ("±0.02 is impossible beyond about 5–6n") reads as a statement about every evaluation.

### W4. LLM evidence is thin relative to the research question

The research question concerns LLM test-time compute. The only LLM evidence is MATH with Pythia 70M–12B and M = 128, where most models solve under 30% of problems at k = 1. None of the modern reasoning models that motivate the question appear. The conclusion "the error has no consistent sign" rests on 7 models from one family with 10 subsets each, and the subsets share the same 128 problems. No uncertainty is reported for the cross-model sign pattern; with M = 128, problem-level bootstrap intervals on the bias at k = 10⁴ are needed. The oracle analysis is unusable beyond n = 16 (W1.2). The Llama-3/Gemma and CodeContests/MiniF2F subsets were "not reachable", but Singh & Singh report Llama-3-70B results on GSM8K and CodeContests from the same release, so the data are obtainable.

### W5. Smaller technical points

- R_lo can be strengthened for free by max(R_lo, (1−2ε)E_n). Any valid CI must cover both members of a population-indistinguishable pair. At (16, 1024, M = 10⁶), the reported R_lo = 0.226 is below (1−2ε)E_n = 0.231. The SOCP grid also produces visible non-monotone dips in R_lo (Fig. 2, n = 32 near k ≈ 1600 and n = 64 near k ≈ 1500). These do not affect validity, but they show the lower-bound optimisation is not reaching the modulus there.
- "R_up within a median 1.06 of W(2η_up)/2 … essentially optimal": the maximum ratio is 1.56, and in several n = 64 cells the selected direction is the r0 Hoeffding polynomial, not the modulus direction. "Essentially optimal" should be qualified.
- **Capped first-success design.** The comparison with "fixed n = 64" is not budget-matched: the capped design spends 162–451 rollouts per task. Compare it with fixed n ≈ 400. Even then the DKW band (±0.0136 at M = 10⁴) probably wins against R_up ≈ 0.03–0.04 at k ≈ 2.4n, so the recommendation may survive, but the comparison should be made fairly.
- **Horizon numbers.** They are log-interpolated between grid points, and the interpolation assumes monotone R_lo, which fails at some large k (see above). This is harmless at δ = 0.02, but the assumption should be stated.
- **Majority-vote "collapse".** The paper now correctly calls it an approximate regularity. It is only 5 points and could be dropped.
- **erfc-rule range.** The range 0.63–1.09 excludes cells with E_n < 10⁻⁵. Say so in the abstract or figure caption.

## 6. Reproduction of the LLM section

I reran `exp_llm.py` from scratch after deleting the cached fits (`llm_fits.pkl`), on 4 processes, in about 10 minutes. All 112 rows of `llm_summary.csv` reproduce. The largest numeric difference is 1×10⁻⁴, in the BMix2/BMix3 biases (optimizer noise). P3 = 57/140 (0.407). Box coverage is 1,120/1,120. BIC picked BB in 279 fits and ZIBB in 1. The rerun also reproduces the problems in W1.1–W1.2: Pythia 12B at n = 16 gives oracle_width NaN, and 6.9B and 12B at n = 32 give widths of 0.0078 and 0.0086, above 2E_n = 0.0047. The data are public counts from a third-party processed file. I could not reach Hugging Face to re-derive them independently. The paper's internal consistency check (pass@9000 agreement within 0.0019) is reasonable.

## 7. Reference spot-check (web search)

| Reference | Result |
|---|---|
| Singh & Singh 2026, arXiv:2609.09245 | Exists, with the stated title. Its content (identified sets, k = 1000 ambiguity, reporting guidance) is consistent with how the paper describes it. |
| Ye 2026, arXiv:2611.21266 | Exists (K. Ye, stat.ME, v2 17 Nov 2026), with the stated title. It claims first-order worst-case-optimal width, which bears on novelty (W2). |
| Kazdan et al. 2025, arXiv:2510.05197 | Exists, with the stated authors. Beta-binomial model plus dynamic sampling, as described. |
| Saibaba 2021, arXiv:2101.06818 | Exists. Gives the error of approximating monomials with a probabilistic (random-walk) interpretation, consistent with the paper's use of it. |
| Polyanskiy & Wu, arXiv:1902.05616 | Exists. Gives tightness of Le Cam's bound for affine functionals over convex classes, as described. |
| Kraus, Vassilevski & Zikatanov 2012, CMAM 12:448–468, arXiv:1002.1859 | Exists, as cited (best uniform approximation to 1/x). |
| Stroebl, Kapoor & Narayanan, arXiv:2411.17501 | Exists. Later retitled "The Limits of Inference Scaling Through Resampling", as noted. Imperfect-verifier ceiling, as described. |

No fabricated or misdescribed references were found. Missing references: Donoho (1994), Donoho & Liu (1991), Cai & Low (2004).

## 8. Questions for the authors

1. How do Ye's (2026) regions compare in width with your R_up and R_lo, and with your box interval, at (n, k, M) ∈ {16, 32} × {4n, n²} × {128, 10⁴}?
2. Can you give an instance-dependent version of Prop. 4: the modulus restricted to a neighbourhood of the observed count law? For a given benchmark that answers "how far can *my* evaluation see".
3. Why does the oracle LP fail (NaN) for Pythia 12B at n = 16, k = 64, and can the n = 32 oracle intervals be computed without slack inflation (e.g. with exact rational arithmetic or a Chebyshev-basis reformulation)?
4. Do the sign-indeterminacy conclusions hold on the Llama-3 / CodeContests subsets of the same release?

## 9. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **6** |

**Decision: major revision.** All round-1 issues are addressed in substance, and the core numbers reproduce exactly. Two things remain. First, under the stated standard, several text/output mismatches in the pre-registered LLM checks must be corrected (W1); they are easy to fix. Second, the main new contribution needs to be positioned against Donoho-style modulus theory and compared quantitatively with Ye (2026). As it stands this is a careful, honest, workshop-to-borderline paper; it is not yet a NeurIPS main-track accept.

**Single most valuable change.** Add the missing positioning (Donoho/Donoho–Liu/Cai–Low) and a quantitative width comparison with Ye (2026). Also replace the global worst-case headline with an instance-dependent (local-modulus) horizon computed on real LLM data, ideally also on the Llama-3 / CodeContests subsets.
