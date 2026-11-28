# Response to Review Round 2 (p002)

We thank the reviewer for a second careful report, including a full rerun of the LLM pipeline. The report's verdict, "careful, honest, workshop-to-borderline", correctly identified novelty and significance as the binding constraints, and patching would not have fixed them. This revision therefore has a **new main result**. We also fixed every mismatch the reviewer found and ran each analysis the reviewer requested whose data we could reach.

## Diagnosis and the new contribution

The r1 paper said: "here is a numerically computed bracket, and here are its horizons for n ≤ 64." An expert would reply that the bracket is Donoho-style modulus theory, that the population rates come from known approximation theory, and that the horizons are numbers without a law. Nothing in the paper would make an expert say "I did not know that, and it matters."

The new contribution is a **rate theorem with a practical consequence**, plus **frontier-model evidence**.

1. **Theorem 5 (new, Section 5.2): k*_δ(n, M) = Θ(n·min{log M, n}).** The farthest budget at which any 95% procedure valid for all F can certify pass@k grows linearly in rollouts per task and only logarithmically in the number of tasks, until it saturates at the population n² horizon.
   - *Achievability:* a closed-form **block estimator** (`code/block.py`). It combines the unbiased statistics U_m(X) = C(n−X,m)/C(n,m) through the best degree-L approximation of w^q, w = (1−p)^b. It comes with a rigorous bias bound and an exact range.
   - *Converse:* a moment-matching pair supported on [0, μL²/k], with an explicit TV bound and a Markov-inequality lower bound on E_L((1−λt/k)^k).
   - *Interpretation:* this is the evaluation counterpart of the Good–Toulmin / Orlitsky–Suresh–Wu "log n" limit for unseen species. To our knowledge the connection has not been made in the LLM-evaluation literature; web searches found no prior mapping.
   - *Verification:* all ingredients are checked numerically in `exp_theorem5.py` (`results/theorem5_checks.json`, `results/theorem5_converse_horizon.csv`).
2. **Sharp constants and two regularities (Section 5.3).** A new 222-cell grid k = s·n (n = 8–128, M up to 10⁸; `exp_horizon_law.py`) and the block estimator up to n = 1,024 and M = 10¹² (`exp_block_law.py`) show two things.
   - (i) *Scale collapse:* at M = 10⁴ the lower bound at fixed s varies by at most 8% across n = 16–128.
   - (ii) *Log M growth:* each tenfold increase in M adds only 1.2–2.4 multiples of n to the ±0.02 horizon. n = 8 saturates at its population horizon, which is the "min{log M, n}" of the theorem.
   - *Practical rule:* deepen, don't widen.
3. **Frontier-model data with ground truth (Section 6.2).** Best-of-N jailbreaking logs [Hughes et al., 2024] for nine models (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Claude 3 Opus, Gemini 1.5 Flash/Pro, Llama-3-8B-Instruct and two defended models), 159 prompts, are rebuilt from public raw logs by `build_bon.py`. These runs are a *capped first-success design*, the design our paper recommends, so pass@k ground truth exists up to 10⁴ attempts.
   - Among prompts with no success in 60 attempts, 49–100% were broken later (5% for Cygnet).
   - Beta-binomial forecasts from n = 60 understate the attack success rate at the cap for all 9 models, and the prompt-level bootstrap interval excludes 0 for 7 of 9. The power law understates for all 9, all intervals below 0, by up to 0.68.
4. **Instance-dependent lower bounds on real data (Section 6.4, the reviewer's Q2).** These are local Pratt/Le Cam bounds at the NPMLE of the observed counts, in the spirit of Cai & Low's between-class modulus.
   - On MATH at M = 128 they depend mainly on s = k/n, and are ≈ ±0.17 at k = 10⁴.
   - On BoN they are ±0.13 at the cap (median over models).

We do not call this a breakthrough. Theorem 5's explicit constants are loose by one to two orders of magnitude (stated in Limitations), and the sharp constants are numerical.

---

## W1. Text/output mismatches (all fixed)

1. **P2 at n = 16 (12B NaN).** Accepted. The paper now says P2 was checked in **6 of 7** cells at n = 16, and that the 12B cell failed in the LP (`llm_prereg.json`).
2. **P2 at n = 32 and the "n ≤ 32" decision.** Accepted. At n = 32 two cells exceed 2E_n (0.0078, 0.0086 vs 0.0047) and two failed. Oracle intervals are now reported **only at n = 16**, and the n = 32 values are described as slack-inflated. We tried a Chebyshev-moment reformulation with exactly feasible grids (`code/oracle_cheb.py`), as Q3 suggested. In the one failing cell we tested (12B, n = 16, k = 64), HiGHS still reported infeasibility, so we report this attempt and do not use it.
3. **P3 breakdown.** Accepted. The paper now gives the full table: 70M 60/60%, 160M 60/70%, 410M 60/70%, 1B 20/20%, 2.8B 40/0%, 6.9B 20/20%, 12B 70/0% (n = 16 / n = 32).
4. **"Lower bound rose by 2–6×".** Accepted. Over the 27 shared cells the factor is 1.6–11.3 (median 3.6). This sentence is removed from the main text, because the r1→r2 comparison is no longer central. The numbers are given here for the record.
5. **BIC counts.** Accepted. The paper now says "all 20 fits" and "19 of 20 fits" for the synthetic laws. The WalkSAT "14 of 30" was already in fits.

## W2. Novelty and positioning

- **Donoho (1994), Donoho & Liu (1991), Cai & Low (2004).** Added and verified. Proposition 4 is now explicitly "an instance of Donoho–Liu / Donoho / Juditsky–Nemirovski modulus theory; we claim only the instantiation and the computation". It is no longer called the main result. The main result is Theorem 5, which is a rate in (n, k, M), not a modulus characterisation. It also uses a different tool: moment matching plus Markov's inequality for the converse, and a constructive block estimator for achievability. The section on local bounds follows Cai & Low's between-class modulus and cites it.
- **Ye (2026).** We confirmed the paper exists, its title, and its claim that for k ≫ n only lower bounds are supported. arXiv was not reachable from our environment, and our search tools returned only abstract-level information, so we could not implement Ye's regions. We do not fabricate a comparison. What we can state rigorously, and now state in Section 5.3, is this: R_lo binds every procedure whose coverage holds for all F, including Ye's simultaneous regions when they are used for pass@k. So no such region can beat R_lo, and Theorem 5's horizon applies to it. We also note the estimand difference (fixed-benchmark vs random-task, an O(M^{−1/2}) difference). Limitations states that no numerical width comparison was made.
- **Section 4 is a synthesis.** Agreed. Section 4 is shortened, labelled "known tools applied to TTC functionals", and the approximate majority-vote "collapse" is dropped (W5).

## W3. Worst-case framing

- **(i) Qualification.** Every horizon statement in the Abstract, Introduction and Section 5.3 now says "uniformly over F" or "worst case over F". The Introduction states explicitly that Section 6.4 gives instance-dependent versions.
- **(ii) Instance-dependent version.** Done (Section 6.4, `exp_local.py`, `exp_realdata.py`). For each observed evaluation we fit the grid NPMLE F̂ and compute a rigorous lower bound on the expected half-width, under F̂, of any procedure valid for all F. The construction is a SOCP over G in a Hellinger neighbourhood of F̂, followed by an exact recomputation of 1 − BC and Pratt integration.
  - These bounds are smaller than the worst-case R_lo, as they should be: 0.072 vs 0.118 at n = 16, s = 4, M = 128.
  - They remain large. On MATH the median at k = 10⁴ is 0.16–0.18 for every n ≤ 128. On BoN at the cap the median over models is 0.13. The box interval is within a median factor of 3.0 of the local limit.
  - The reviewer's example (WalkSAT oracle intervals ±0.003 at k = 6n) is a *population* (M = ∞, plug-in law) interval. It is consistent with Lemma 1 and does not contradict a finite-M worst-case bound. The Discussion no longer says "±0.02 is impossible beyond 5–6n" without the qualifier.

## W4. LLM evidence

- **Modern models.** Added: nine frontier and defended models on BoN jailbreaking with first-success ground truth (Section 6.2). These data partly fill the gap. They are a different task (jailbreaking, graded by an automatic classifier), and no long-chain-of-thought reasoning models are included; Limitations says so.
- **Bootstrap.** Done.
  - *MATH (`exp_llm_boot.py`).* 200 problem-level replicates, all 10 subsets refitted, same fitter as the main analysis. The BB error at k = 10⁴ from n = 16 has a 95% interval excluding 0 only for Pythia 160M. **We therefore weakened the MATH conclusion:** the cross-model sign pattern is suggestive but not statistically established at M = 128. The power-law underestimation is established for all 7 models.
  - *BoN (`exp_bon_boot.py`).* The BB underestimation is significant for 7 of 9 models and the power-law underestimation for 9 of 9.
  - Our first bootstrap attempt used a different, numerically unstable beta-binomial fitter and produced overflowing values; we discarded it and reran with the main fitter.
- **Llama-3 / CodeContests subsets.** The Schaeffer et al. release we can reach contains processed per-problem counts only for Pythia (MATH and GSM8K), and Hugging Face is blocked in our environment. We could not obtain the Llama-3 subsets and say so in Limitations. The BoN data are our substitute for modern-model evidence.
- **Oracle beyond n = 16.** See W1.2.

## W5. Smaller technical points

- **R_lo ← max(R_lo, (1−2ε)E_n).** Adopted throughout. On the r1 grid it binds in one cell, (16, 1024, 10⁶): 0.226 → 0.231. The non-monotone dips remain possible. Horizons are interpolated with the monotonicity assumption stated, and grid values are reported alongside.
- **"Essentially optimal" R_up.** Removed. R_up is now the minimum of the Chernoff search and the block estimator. The paper reports where each wins: the block estimator wins in 9% of the extended-grid cells, all at large M, by up to 1.6×. It also reports the full ratio range, [1.46, 3.81] on the r1 grid and [1.14, 9.65] on the extended grid, with the large values at M = 10⁸.
- **Capped design, budget-matched.** Done (`exp_capped_fair.py`). With fixed n equal to the capped design's per-task cost (413/451/162), the block-estimator half-width for pass@1000 is 0.20/0.19/0.36 at M = 240 and 0.040/0.034/0.169 at M = 10⁴. These are 2.1–12.4× the DKW band, which covers all k ≤ D at once. The recommendation survives, with the caveat that the block estimator is an upper bound on R*.
- **Horizon interpolation.** The assumption is stated in the Table 2 caption and in Limitations.
- **Majority-vote collapse.** Dropped.
- **erfc-rule range.** Section 4 and the Figure 1 caption now state "for n ≥ 16 and E_n > 10⁻⁵".

## Questions

1. **Ye widths.** See W2. We could not access the method details, so we give the structural statement only.
2. **Instance-dependent version.** Done (Section 6.4).
3. **Oracle LP failure.** The cause is ill-conditioning of exact moment constraints at n ≥ 16 for k ≫ n. The Chebyshev-basis reformulation did not cure it in the cell we tested. We report oracle intervals only at n = 16.
4. **Llama-3 / CodeContests.** Not reachable. The BoN frontier data are added instead. On BoN the sign *is* consistent: forecasts understate the gain for all 9 models.

## Reproducibility

New scripts: `block.py`, `exp_theorem5.py` (rewritten), `exp_horizon_law.py`, `exp_block_law.py`, `build_bon.py`, `exp_realdata.py`, `exp_bon_boot.py`, `exp_llm_boot.py` (rewritten), `exp_capped_fair.py`, `oracle_cheb.py`, `analyze_r2.py`. All outputs are in `results/` (`r2_summary.json` collects the headline numbers). The BoN table is rebuilt from raw logs with assertions on the batch/stopping design. One file left by an interrupted earlier session, `data/bon_jailbreak_text_attempts.csv`, had no build script. It was replaced by the scripted rebuild, which matches it exactly on the 7 models it contained, and then deleted.
