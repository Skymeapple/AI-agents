# Response to Reviewer (Round 1)

We thank the reviewer for a careful report, including the full re-run of the code. We agree with every major point. The paper has been repositioned and retitled. The previous title was "Knowing What You Don't Know Costs Bits: A Capacity Theory of Irreducible Hallucination under Penalized Grading". The new title is "The Bit Price of Abstention: Capacity Rates for Hallucination on Arbitrary Facts under Penalized Grading". We added a rigorous any-model theorem (Theorem 3, Corollary 2, Proposition 4) and a new large-universe experiment (Experiment C: 144 runs, 6 seeds, held-out selection, bootstrap CIs, measured open-world hallucination). Where the new evidence went against us (capacity invariance at s = 1.5, λ-slopes), we report that.

All new numbers come from code run in this environment:
- `code/theory_relax.py` → `results/relax_tables.txt`
- `code/theory_closed_form.py` → `results/closed_form.txt`
- `code/relax_resolution_check.py` → `results/relax_resolution.txt`
- `code/exp_c_large_universe.py` → `results/exp_c.json` and `exp_c.log` (1,151 s on 4 CPU processes)
- `code/analyze_rev.py` → `results/analysis_rev.txt`, `exp_c_summary.json` and `fig_exp_c.png`
- `code/corollary2.py` → `results/corollary2.txt`

---

## Major issues

### M1. Missing prior work: Guo & Li (ICML 2026) and Wang, Xu & Guo (2026)

**Agreed.** We verified both papers by web search:
- Guo & Li, arXiv:2602.00906, PMLR 306:38012–38051.
- Wang, Xu & Guo, arXiv:2609.12111 (Sept 2026).

**Changes:**
- Both papers are cited in the Abstract, Section 1, and Section 2. Section 2 has a new first paragraph, "Membership testing and hallucination (closest work)".
- The paper now says plainly that the qualitative impossibility (our Corollary 1 and Theorem 2b) is classical and was established for hallucination by Guo & Li. Corollary 1 is labelled "classical", with the explicit sentence "we do not claim it as a contribution".
- The contributions are narrowed to four:
  - (i) the per-fact KL relaxation under penalized grading with Zipf weights, value correctness and a fresh-key open world (Theorem 3, Corollary 2). We note that this is a close relative of Guo & Li's KL characterization.
  - (ii) rates in C and λ for arbitrary models (Proposition 4) and for store-and-filter policies (Proposition 3).
  - (iii) Experiment C, which tests these rates.
  - (iv) certified capacity measurement.
- Section 2 lists four concrete differences from Guo & Li.
- The Limitations section opens with "Novelty is narrower than in our first version".

### M2. The headline impossibility is near-tautological, and the rates were only shown for uniform filters

**Agreed on both parts.**

*Tautology.* Corollary 1 now carries the reviewer's example, about 2^−990 at B/g = 1000, and the statement that "the floor is practically empty". Theorem 2(b) is described as holding "within 𝒫_C by construction (f < ∞)" and is "no longer present[ed] as a result". The title and abstract no longer lead with "irreducible". The abstract now gives the practical caveat (h from 0.15 to 1.6×10^−13 for about 2.6 points of knowledge) and says the results are compatible with the "statistically negligible" view.

*Rates beyond uniform filters (the reviewer's Question 3).* New Section 4.2:
- **Theorem 3** holds for any model with H(M) ≤ B, trained on any data including all facts: Σ_i KL(P_i ‖ Q(ε)) ≤ I(M; D) ≤ B. P_i is the reject/wrong/correct outcome distribution on fact i, and Q(ε) is the outcome distribution on a key independent of M. The proof is exact: data processing plus superadditivity of mutual information over independent facts. Each fact can have its own acceptance probability, so this class includes per-fact (non-uniform) filters and negative lists.
- The resulting convex program (7) upper-bounds the penalized score of every C-bit model.
- **Proposition 4** gives its closed form in the continuum limit: L*(C) = κ·C^−(s−1), with ε* = argmin G(ε) independent of C. Its derivation is given in full, and it matches the exact discrete program to 4–5 significant digits (`closed_form.txt`). A 4× finer fact binning changes the discrete loss by ≤ 0.14% (`relax_resolution.txt`).
- Answer to Question 3: **non-uniform filters change the constant, not the exponent.**
  - The any-model bound's ε* drifts by at most 1.10× over C = 10^6–10^10.
  - Its loss slope equals −(s−1) to three decimals at all 24 settings.
  - Store-and-filter comes within a factor 1.003–1.53 of the bound across 168 grid points.
  - Weak duality gives an any-model hallucination floor at the store-and-filter knowledge level. It sits 2.6–7.5× below store-and-filter and has slope −0.19 / −0.49 (predicted −0.2 / −0.5) (Table 2).
- The optimal ε* for arbitrary models is 1.1–23× looser than the uniform-filter ε*. Per-fact shaping lets the background filter relax.

### M3. The extension of inequality (2) to "any B-bit model" was heuristic

**Agreed. It has been replaced by a rigorous statement.** **Corollary 2** (new) follows from Theorem 3 with the binary-KL data-processing inequality and a counting step. For every model with H(M) ≤ B and background false-accept rate ε,

h ≥ (q/2)·U(⌊B/δ⌋), with q = ε(1−1/V) and δ = q(1−ln2)/(2ln2).

Under Zipf this gives h = Ω(ε^s·B^−(s−1)). This is the "I(M; tail) ≤ B" argument the reviewer asked for, made exact. The old sentence ("Tail facts the model does not encode have keys independent of the model, so (2) bounds … any B-bit model") has been removed. We also evaluate Corollary 2 at the measured Experiment C operating points with B = 32P (`corollary2.txt`). The floor is nonzero but loose: networks sit 13–70× above it at s = 1.2 and 300–2,100× above it at s = 1.5. Section 6.2 and the Limitations say so.

### M4. The zero-false-accept experiment was misinterpreted

**Agreed. The claim is withdrawn and the result is reinterpreted with a diagnosis.** `analyze_rev.py` (1b) examines the strictest threshold in each A1 run, where only stored keys at or above the network's most confident stored key are accepted:
- The median false-accept rate there is 7.4×10^−4 (range 0 to 7.6×10^−2).
- Typically about a hundred fresh keys out of 200k outscore the best memorized fact.
- Theorem 1 needs only 1–9 bits (92 in one case) to permit such pairs.
- 7 of 60 runs saturate at conf = 1.0 in float32.

Section 6.1 now says this reflects poor confidence ranking and saturation, i.e. that "SGD-trained MLPs … are poor filters at the extreme", and that it is not evidence for the information bound. The observation has been removed from the abstract and is no longer cited in support of the theory. The coarse quantile grid (steps of about k/40 stored keys) is now discussed explicitly.

### M5. The empirical tests did not discriminate, the novel predictions were untested, and Experiment B was closed-world

**Agreed. This is now the empirical centrepiece (new Experiment C, Section 6.2).**
- **Setup.** A 2^20-fact Zipf world plus an infinite untrained tail. Universe size N_U·H ≈ 4×10^6 bits, against at most about 5×10^4 bits of network capacity. 4 widths (P = 1.1k–23k), s ∈ {1.2, 1.5}, 6 seeds. Open-world hallucination (unseen in-universe facts plus the tail) is measured directly.
- **C^−(s−1) exponent.** The h-slopes against P are −0.199/−0.202/−0.213/−0.202 at s = 1.2 (predicted −0.20) and −0.53/−0.58/−0.41/−0.51 at s = 1.5 (predicted −0.50). All CIs contain the prediction.
- **Discriminating test (dose-response in s).** Any account in which hallucination falls with scale at a rate unrelated to the query tail predicts equal exponents for both s. The measured difference is −0.33, −0.38, −0.20 and −0.30 (predicted −0.30). All four 95% CIs exclude 0, so that alternative is rejected.
- **Capacity invariance of ε*.** Confirmed at s = 1.2: no CI excludes 0. This bounds an alternative "bigger models become more selective" effect to γ ≤ 0.11–0.16. **Not confirmed at s = 1.5**: ε*-slopes are −0.16 to −0.31, and the CIs exclude 0 at λ = 1, 3, 9. We report this as a partial contradiction in the abstract, results, discussion and limitations. We offer the inefficient-small-filter explanation (consistent with Experiment A) only as a hypothesis.
- **λ-slope.** We now explain analytically when it discriminates. Under Chow's rule with confidence density ∝ (1−q)^(a−1) near 1, h* ∝ λ^−(1+a), so a smooth (uniform) confidence distribution predicts −2. The observed slopes reject a = 1 but do not single out our model among models with a ≈ 0. The λ-slope is therefore no longer a main test. In Experiment C the networks' λ-slopes (−0.70 to −0.95) are also *shallower* than store-and-filter theory (−1.12 to −1.15), with some CIs excluding the theory value. We report this disagreement.
- **The A2 exchange-rate test** remains confounded, and we state that.

### M6. Weak statistics: 2 seeds, no CIs, oracle selection, and C = 2.03P as a lower bound

**Addressed.**
- Experiment C uses 6 seeds, 2-fold cross-fitting over seeds, and two independent fresh-key samples (A for selection, B for evaluation). 95% CIs come from 200 seed-bootstrap resamples, with the full cross-fit re-run in each.
- For Experiment B we now report per-seed λ-slopes (−1.36/−1.17, −1.09/−0.96, −1.11/−1.10). Experiment B is demoted to a descriptive secondary result.
- On C = 2.03P: we agree it is a certified lower bound, and the first version's "one bit-budget behind theory" wording depended on it. Experiment C compares only constant-free quantities. Log-log slopes do not depend on a constant factor in C, and ε* does not depend on C. Section 5 and the Limitations state this.

---

## Minor issues

1. **Contribution 4 contradicted Section 6.1.** Fixed. The sentence has been removed, and Contribution 4 is now about certified capacity. Section 6.1 reports that one run had g = 1 at zero observed false accepts, within the reinterpreted discussion.
2. **Union bounds for data-chosen (g, a) and thresholds.** Added as a Remark after Theorem 1:
   - a union bound over all (g, a) costs log2((k+1)(n+1)) ≤ 44 bits;
   - t = 7 is used;
   - a Bonferroni-corrected Clopper–Pearson upper bound on ε is taken at level 0.05/43.
   The corrected certified maxima are 1.85 / 2.18 bits/param (medians 1.17 / 1.38), against 1.87 / 2.19 uncorrected.
3. **Morris et al. citation.** Updated to Morris, Sitawarin, Guo, Kokhlikyan, Suh, Rush, Chaudhuri & Mahloujifar (2026), "How much can language models memorize?", ICML 2026, PMLR 306, 90392–90411. The preprint title and arXiv ID are noted. Verified by web search.
4. **"Irreducible" overstated given T5.** Fixed in the title, abstract (practical caveat now included), Section 4.4 ("both true") and Conclusion.
5. **Coarse threshold grid and float32 saturation.** Now discussed in Section 6.1, with counts (7 of 60 runs saturated, grid steps of about k/40). Experiment C uses a finer grid: 61 stored-confidence quantiles plus 50 fresh-key quantiles down to 2×10^−5.
6. **Table 3 label.** The column is now "mean-over-seeds of per-run max certified bits/param", with an explanatory caption.

## Answers to the reviewer's questions

1. *Differences from Guo & Li / Wang et al.* See M1 and Section 2. The new parts are penalized grading, Zipf rates, value correctness with an open world, the per-fact KL relaxation adapted to these, and tests of the rates on trained networks.
2. *A version of (2) for arbitrary B-bit models.* Yes: Theorem 3 and Corollary 2.
3. *Non-uniform filters.* Yes, ε* stays capacity-invariant for arbitrary models in the continuum bound (Proposition 4). The rate stays C^−(s−1), and only constants change.
4. *Large-universe Experiment B with open-world hallucination.* Done as Experiment C. We used N_U = 2^20, about 1.05×10^6, matching the suggested ~10^6, with P ≤ 2.3×10^4.
5. *Seed variance, CIs and held-out selection.* Done (M6).
6. *A baseline predicting a λ-slope other than −1.* Smooth calibrated confidence predicts −(1+a), e.g. −2 for a uniform distribution. See M5.

## What did not improve

- At s = 1.5 the networks do not show capacity invariance.
- Their λ-slopes are shallower than theory.
- Corollary 2 is far from tight for real networks.
- All experiments remain small MLPs on random keys, and there are no language models.

These points are stated in the abstract, Section 6.2 and Section 8.
