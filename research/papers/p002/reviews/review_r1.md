# Review (Round 1): "How Far Can a Fixed-Rollout Evaluation See? Polynomial-Approximation Limits on Forecasting Test-Time-Compute Gains"

**Reviewer role:** independent area-chair-level reviewer
**Research question:** 推理时计算（test-time compute）还能带来多大提升？
**Recommendation:** Major revision (overall 5/10, weak reject in current form)

---

## 1. Summary

The paper asks how far a fixed-n rollout evaluation can extrapolate test-time-compute (TTC) functionals θ = E_F[g(p)] when the task-difficulty law F is unrestricted. It reduces the question to best uniform polynomial approximation:

- **Theorem 1.** The worst-case identified-set width is 2E_n(g).
- **Theorem 2.** For pass@k, a Newman–Rivlin sandwich gives a width that is O(e^{-n²/k}) and numerically close to erfc(n/√k). This is the "n² horizon".
- **Theorem 3.** The resample-until-accept verifier ceiling has the closed form r^n, with r = (√t−√q)/(√t+√q).
- **Corollary 1.** pass@∞ is not identified, even at finite sample size.

The paper also gives a finite-task analysis: a Hoeffding-certified linear estimator, a Bonferroni Clopper–Pearson "box" interval, and χ²/Le Cam lower bounds computed by convex programs. Two empirical studies follow. One is synthetic, with four mixing laws. The other is a WalkSAT/random-3-SAT test bed (3 configurations × 240 instances × 10,000 runs) whose pass@∞ is known exactly from a complete solver.

## 2. Reproduction (what I re-ran)

I re-ran the pipeline in a clean scratch copy, without reusing cached results:

| Script | Outcome |
|---|---|
| `exp_theory.py` + `exp_theory_check.py` | All CSV/JSON outputs and `fig_theory.png` are **byte-identical** to `results/`. |
| `run_walksat.py` (config A regenerated from seed) | Run matrix and SAT labels identical to the stored `walksat_A_*.npz`. |
| `exp_walksat.py` (on stored A/B/C matrices) | `walksat_summary.csv`, `walksat_raw.csv` and `fig_walksat.png` byte-identical. |
| `exp_design.py` (from scratch, no cache) | `design_bounds.csv` identical in every column (relative difference 0); `design_horizon.csv` identical. |
| `exp_synthetic.py` | Started; it is slow (about 4 min per law at M = 5,000). I checked every synthetic claim against the stored `synthetic_summary.csv` / `synthetic_raw.csv.gz` instead (see below). |

I cross-checked every number in the paper against the outputs. Nearly all match: the duality check, Theorem 3's 0.33% error, the horizon table, best-of-k and majority values, all entries of the Section 5 table, the 2.7n–3.3n horizons, the WalkSAT biases (+0.281/+0.131/+0.105; +0.213/+0.086/+0.068; +0.228 at n = 64), the oracle intervals, the 840/840 box coverage, 2/144 failed LPs, synthetic coverage (7,198/7,200; 5,400/5,400), bootstrap coverage (91/32/16/9%), the biases, identified intervals and verifier half-widths. The engineering and reproducibility are well above average.

**One mismatch.** Section 4 and contribution 2 state: "2E_n is within 0.035 (absolute) of erfc(n/√k) for n ≥ 16". The code (`exp_theory_check.py`, key `max_abs_dev_from_half_erfc_for_n_ge_16`) computes |E_n − ½erfc|. So the value 0.0346 bounds the deviation of E_n, not of 2E_n. Recomputing from `theory_passk_with_rho.csv`, max |2E_n − erfc(n/√k)| = **0.069** (n = 128, k = 453,259). The stated fit accuracy is off by a factor of 2. The same file also shows relative deviations up to 59% for n²/k ≥ 0.25, which is the small-width regime that drives the "erfc rule" horizons. Under the venue rules this is a mismatch between paper and code output. It is easy to fix but must be fixed.

**A smaller point.** The stored `theory_checks.json` reports `"sandwich_holds": false`, because the ratio E/ρ reaches 1.00004 (LP tolerance). The paper reports "[0.52, 1.00]" without saying so. This should be disclosed.

## 3. Reference check

I spot-checked these references by web search:

| Reference | Finding |
|---|---|
| arXiv:2609.09245 | Exists. Authors Singh & Singh (2026), per a citing paper. The abstract matches the paper's description: n free moments, Hausdorff principal representations, n = 16 → k = 1000 ambiguity on the Brown et al. release, a one-coordinate finite-task certificate. The missing author list should be filled in. |
| Kazdan et al., arXiv:2510.05197 | Exists. Beta-binomial forecasting plus dynamic sampling, as described. |
| Newman & Rivlin (1976), *Aequationes Math.* 14:451–455 | Exists. |
| Saibaba, arXiv:2101.06818 | Exists. It restates Newman–Rivlin Thm 2 and the coin-toss interpretation. |
| Stroebl, Kapoor & Narayanan, arXiv:2411.17501 | Exists. Now titled "The Limits of Inference Scaling Through Resampling". The false-positive ceiling is as described. |
| Kraus, Vassilevski & Zikatanov (2012), CMAM 12(4):448–468, arXiv:1002.1859 | Exists. |
| Brown et al. 2024; Snell et al. 2024 | Consistent with the claims (15.9% → 56% on SWE-bench Lite; 14× larger model). |

No fabricated or misattributed citations found.

## 4. Novelty assessment

- **Theorem 1 is a known duality.** That the sup of ∫g d(μ−ν) over moment-matched pairs equals 2E_n(g) is the classical Chebyshev/Markov–Krein moment-matching duality. It is used explicitly in Wu & Yang (2016, 2019), which the paper cites, but only for a different point. The Bernstein-basis translation (count law ↔ n moments) is also in arXiv:2609.09245. A concurrent preprint, arXiv:2611.21266 ("Variance-adaptive confidence regions for fixed-benchmark binomial mixtures, with application to LLM evaluation", Nov 2026), states that its identification results "use the duality between moment matching and best polynomial approximation of Wu and Yang". Theorem 1 should be presented as a known lemma applied here, with credit, not as a contribution.
- **Theorem 2** follows from Newman–Rivlin and Sachdeva–Vishnoi via the y = z² reduction. The reduction is correct and the application is neat, but it is a corollary.
- **Theorem 3** is Chebyshev/Achieser's classical E_n(1/(x−c)) after an affine change of variables. I checked the algebra line by line and it is correct (c = (t+q)/(t−q); c − √(c²−1) = (√t−√q)/(√t+√q)). The interpretation, the √(t/q) depth law and the "better verifiers are harder to certify" trade-off are the most original insights in the paper.
- **The finite-task material** (Hoeffding linear estimator, Bonferroni CP box, two-point χ² Le Cam bounds) uses standard tools. arXiv:2611.21266 appears to address finite-sample, simultaneous, variance-adaptive regions for the same model and should be discussed.

Overall, the contribution is a clean, well-executed synthesis: approximation-theory rates on top of the identification result of 2609.09245. The mathematics is incremental. The test bed with an exactly known ceiling is a genuinely useful idea.

## 5. Correctness of proofs and derivations

- **Lemma 0 / Theorem 1.** Correct. Detail: with ‖σ‖_TV = 2 and σ(1) = 0, σ⁺ and σ⁻ each have mass exactly 1, so the δ₀ padding is unnecessary.
- **Theorem 2.** The even-symmetrization argument is correct. The SV bound with s = 2k, d = 2n gives 2e^{−n²/k}. The NR constant 1/(4e) is cited through Saibaba, and I could not verify it independently from the snippet I retrieved. The de Moivre–Laplace identification ρ_{2k,2n} ≈ erfc(n/√k) is correct.
- **Corollary 2.** Numbers verified, e.g. 2·0.157/(4e) ≈ 0.029 at k = n².
- **Corollary 3.** ln(1/r) = 2 artanh√(q/t). The values n = 7/22/70 are verified.
- **Corollary 1.** Correct.
- **Section 5.3 Le Cam bound.** The constants are right: H² ≤ χ², subadditive; TV ≤ 0.354; 0.95 + 0.596 − 1 = 0.546; half-width 0.273Δ. The mixture repair is valid. The reference laws are 100k-sample empirical laws, so θ₀ is exact for the law actually used. Fine.
- **"First success capped at D" design (Section 7).** Correct, including the cost formula. Note that this simple adaptive design largely dissolves the problem the paper studies. It deserves more prominence and an honest comparison of cost against fixed-n depth.

## 6. Major issues

1. **Claim/output mismatch for the erfc law (see Section 2).** "2E_n within 0.035 of erfc" is actually 0.069. The relative error in the tail regime reaches 59%, so the "empirical erfc rule" horizons (77 / 1,234 / 19,754) are rough approximations, not a law.

2. **The headline finite-task claim is not supported by the lower bounds.** The abstract and Section 5 say "rigorous Le Cam lower bounds show the usable horizon is far shorter than the population n² law" and "at benchmark scale the usable horizon is a small multiple of n". However:
   - (a) The "small multiple of n" (43/95/210) comes from **R_up**, the guarantee of one specific non-adaptive Hoeffding/range estimator. It is not a limit on what can be learned. Upper and lower bounds differ by 5–35×.
   - (b) The lower bounds do not exclude much longer horizons at the stated tolerance. At δ = 0.02 and M = 10⁶, R_lo = 0.011 at (16, 116), 0.011 at (32, 283) and 0.0117 at (64, 689). All are below δ, so the lower bound permits exactly the grid points the paper calls "not certified".
   - (c) The comparison with the "population law admits k ≈ 0.3n² (77, 308, 1,234)" mixes conventions. Those numbers are the erfc rule for **width** 0.01, while the certified numbers are for **half-width** 0.02. On the same grid and the same δ, `design_horizon.csv` gives population horizons of 116/283/689, only one grid step above the certified 43/95/210.
   - (d) The k-grid has only 8 log-spaced points, so "2.7n–3.3n" is resolution-limited.
   - (e) "R_lo is 41× the population half-width" is close to vacuous. At k = n (exact identification, population width about 1e-14), R_lo is still about 5e-5 at M = 10⁶, an infinite ratio, simply because of sampling error. The informative quantity is the amplification of R_lo with k at fixed M (about 240× from k = 64 to 689 at n = 64). That does show ill-conditioning, but the paper does not frame it that way.

   The finite-task section needs reframing. It should state what is proven (a lower bound at δ ≈ 0.01) and what is only an achievability artefact.

3. **The WalkSAT parametric comparison is close to a strawman.** The test bed has 15–45% of mass exactly at p = 0, and both baselines (single Beta, pure power law) assign zero mass there. Overestimation is therefore guaranteed by construction. The paper's own synthetic study shows the sign reverses for other laws, with logit-normal bias −0.11. The natural, widely used fixes are not evaluated: a zero-inflated beta-binomial, a Beta mixture, or Kazdan et al.'s dynamic sampling. The abstract line "parametric extrapolations overestimate by 0.07–0.28" and the Discussion's "biased toward more compute will keep helping" generalize further than the evidence supports.

4. **No LLM data.** The paper's question concerns LLM test-time compute, yet no LLM data are analysed, and the protocol is only "pre-registered". The Brown et al. 10,000-sample release is public. Prediction P2 in particular is a theorem check with little information content. Without at least one real-model analysis, the significance for the stated question is limited. The predecessor 2609.09245 already reports LLM-data ambiguity numbers.

5. **The finite-task procedures are not competitive even where the target is identified.** At M = 240 and n = 16, the box interval for pass@1 to pass@16 (exactly estimable with SE ≈ 0.03) has half-width about 0.15 (Fig. 3). Similarly, certified-estimator coverage of 99.97% against a nominal 95% indicates heavy conservatism. "840/840 coverage" therefore says little. It also includes 120 pass@∞ cells whose upper end is trivially 1. Report widths relative to an oracle, or to a CLT interval at k ≤ n, and drop the trivial cells from the coverage count.

## 7. Minor issues

- Limitations says the upper/lower gap is "up to about an order of magnitude", while Section 5 says 5–35×. Reconcile the two.
- Population and oracle identified intervals are computed from Monte Carlo count laws (200k draws) and an L1 slack of 1e-5/1e-6. Several do not contain the truth:
  - Beta(0.3,3) pass@32: truth 0.5370 vs [0.5362, 0.5363];
  - logit-normal pass@32 at n = 16 and pass@128 at n = 64;
  - WalkSAT C pass@32 at n = 16/32/64 (truth 0.76322 vs [0.76307, 0.76310]).

  These are harmless numerical artefacts, but the paper calls these intervals "sharp" and "exact".
- The beta-binomial bootstrap covers only 84–88% at M = 5,000 even when well specified. The "91%" averages over M and hides this.
- The majority-vote "collapse" rests on 5 values of n and loose agreement (E = 0.072/0.058/0.063 at n²/k ≈ 0.8–1.0). Call it "approximate" or quantify the spread.
- Give arXiv:2609.09245's authors. Cite the concurrent arXiv:2611.21266.
- The docstring of `chi2_lower` describes a per-cell box constraint, but the code uses the χ² sum constraint. Align them.
- Figure 1's left panel plots the bound 4e^{−x}. Corollary 2 uses ln(4/δ), which corresponds to width ≤ 4e^{−n²/k}. These are consistent, but the text alternates between E_n ≤ 2e^{−n²/k} and width ≤ 4e^{−n²/k}. Say "width" or "half-width" consistently throughout, including the table header "population E_n".
- The Theorem 1 proof's padding step is unnecessary (see Section 5).
- The README says the CNF instances were deleted. That is fine, since generation is seeded and I verified it reproduces exactly.

## 8. Questions for the authors

1. Can you state a finite-M minimax rate, even an order-of-magnitude one, that is tight in k, instead of an upper/lower gap of up to 35×? Does the Wu–Yang machinery give R(n, k, M) ≍ something explicit?
2. How do the conclusions change against a zero-inflated beta-binomial or a 2–3-component Beta mixture on WalkSAT and on the synthetic laws?
3. What does the box/oracle analysis give on the Brown et al. release? Even one configuration would anchor the significance claim.
4. How does your contribution relate to arXiv:2611.21266's variance-adaptive regions for the same binomial-mixture model?
5. Is the first-success-capped design (Section 7) cheaper than a fixed-n design reaching the same certified horizon on realistic difficulty laws? If so, isn't that the paper's main practical recommendation?

## 9. Strengths

- A precise, quantitative sub-question with a clean answer at the population level.
- Theorem 3 and its "better verifier, harder to certify" trade-off are elegant and new to this literature. They are exact, and confirmed numerically to 0.33%.
- Exceptional reproducibility: every number traces to code, and my reruns were byte-identical.
- An honest Limitations section and conservative validity guarantees.
- A test bed with an exactly known ceiling, which the community can reuse.

## 10. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 6 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **5** |

**Decision: major revision.** At a top venue I would currently lean toward a weak reject. The population theory is correct but largely repackages known approximation-theory facts. The most practically important claim, that the finite-task horizon is a small multiple of n, is not established by the lower bounds. One quantitative claim (the erfc fit accuracy) does not match the code output. The empirical evidence uses no LLM data and baselines that are misspecified by construction.

**Most valuable single change:** Replace the finite-task section's achievability-based headline with a tight (or explicitly characterized) finite-M minimax rate in (n, k, M), and validate it, together with stronger parametric baselines, on the public Brown et al. 10,000-sample LLM release.
