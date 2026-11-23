# Review of "Replication Is Not Validation: Shared-Bias Limits on AI-Verified Scientific Hypotheses" (p019, round 1)

## Summary

The paper asks when a verification step can change the posterior probability that an AI-generated, original hypothesis is true. Each measurement is modelled as true effect plus a modality-specific systematic bias plus sampling noise, and claims are accepted by fixed-level one-sided tests. Three results follow:

- **Theorem 1 (bias floor).** As n grows, same-modality replication and self-verification reach an FDR floor of π0/(π0+2π1F_b(μ)).
- **Theorem 2 (K\* rule / "one-bit ceiling").** Under conditional independence, an observational modality has a likelihood ratio Λ ≤ 2 in the large-n limit, so K\* ≥ log2 of the required odds.
- **Proposition 3 (shared-bias reversal).** If some false claims carry a bias that all modalities share, adding modalities drives FDR to 1.

The results are illustrated in 300 synthetic linear-Gaussian SCM "worlds". A bivariate-regression generator proposes 40 "original" causal claims per world, which are checked by self-verification, fresh replication, environment triangulation, and randomized interventions. Self-verification and replication give FDR 0.787. Four environments give 0.698. Two interventions give 0.007. A class-stratified pass-rate model predicts held-out triangulation FDR to within 0.011. In a depth-of-search experiment, the K\* rule predicts that 2 interventions suffice in every rank band.

## Reproduction

- **E1.** I copied `code/` to a scratch directory and reran `exp1_theory_check.py`. The regenerated `exp1_theory.json` is byte-identical to the stored file. Every Table 1 entry matches the output: minima 0.450/0.144/0.033/0.049 at n = 100/39/100/251, n = 10⁶ values 0.669/0.668/0.341/0.092, limits 0.672/0.672/0.349/0.093. The largest Monte Carlo–analytic gap is 0.007, at n = 30 for replication, as claimed. Figure 1 matches.
- **E2.** I reran `exp2_sem_world.py 300`. The regenerated `exp2_rows.json` (the main 12,000-candidate run) is byte-identical to the stored file. Running `analyze_exp2.py` on the stored raw outputs reproduces `exp2_summary.txt` exactly.
- **Paper numbers.** Every number I checked in Tables 2–5, §6.3, §6.4 and the novelty-composition file matches the script output. I also recomputed several by hand: the Theorem 1 limits, the Table 4 naive and decomposition predictions, the Table 5 predicted FDRs, the n× ≈ 271 crossover, the log2 36 ≈ 5.2 example, and ln(0.254/0.114) ≈ 0.8. All agree.
- **Full rerun.** The full `exp2_sem_world.py 300` rerun also reproduced the sweep and depth outputs byte-identically: `exp2_sweep_noiseless.json`, `exp2_sweep_noisy.json` and `exp2_depth_rows.json`. Rerunning `analyze_exp2.py` on the fresh outputs reproduces `exp2_summary.txt` exactly.
- **Runtime.** The claimed runtime of "about 5 CPU-minutes" is wrong. The full run took about 23 wall-clock minutes on 4 cores (1381 s), dominated by the two n = 30,000 sweep points. This is minor, but it should be corrected.
- **Verdict.** I found no mismatch between the paper and the code outputs.

## Citation spot-check (web)

I verified the following references:

- Fa & Culjak 2026, arXiv:2604.22080. Exists; a falsification-first position paper.
- Mitchener et al. 2025, Kosmos, arXiv:2511.02824. Exists.
- Huang et al. 2025, POPPER, ICML / PMLR 267. Exists; uses e-values with Type-I error control.
- Si, Hashimoto & Yang, arXiv:2506.20803. Exists; reports the ideation–execution gap and an ICLR 2026 poster.
- Munafò & Davey Smith 2018, *Nature* 553:399. Exists; "replication is not enough", triangulation.
- Eberhardt, Glymour & Scheines, UAI 2005, arXiv:1207.1389. Exists.

One imprecision: the paper says Eberhardt et al. bound the number of *single-variable* experiments. The UAI 2005 paper actually gives log2(N)+1 for experiments that intervene on many variables at once. The N−1 single-intervention bound is in the companion report. This is minor, not fabricated.

I found no non-existent or misrepresented references. The remaining references are standard, well-known works.

## Strengths

1. The question is sharp, and the paper frames it well: verification is only informative to the extent that it randomizes the bias sources behind the generator's errors. This is a useful corrective to "the agent replicated it / ran N falsification tests" claims in the AI-scientist literature.
2. The work is fully reproducible and honest. Seeds are fixed, outputs are byte-identical on rerun, and there is a world-level bootstrap. Limitations are clearly stated, including a candid "non-replication" of the non-monotone FDR shape in the SEM worlds and the use of ground-truth bias labels in E3.
3. The "anatomy of false claims" table (Table 3) is the most informative empirical result. It shows that environment variation removes latent-confounding errors at roughly the independent-bias rate, but leaves reverse-causation and common-cause errors almost untouched. The near/far-from-literature composition result is a genuinely interesting, unexpected observation.
4. The writing is clear, and the proposed "bias-source ledger" and test protocol are concrete.

## Weaknesses

### W1. The theory is correct but close to elementary, and its novelty is overstated

- **Theorem 1** says that as n → ∞ a fixed-α test on a biased estimator becomes a sign test of θ + b. Hence A0 → P(b > 0) = 1/2. This is the standard fact that sampling error vanishes while bias does not; compare Meng's (2018) "big data paradox" and data defect correlation, and the bias term in Ioannidis (2005). The paper does not cite Meng or the multiple-bias / bias-analysis literature (Greenland; Lash et al.). The proof is a two-line bounded-convergence argument.
- **Theorem 2** is the textbook PPV/FDR calculation for K conditionally independent tests with a product likelihood ratio, followed by taking logarithms.
- **Proposition 3** is a one-line observation.
- I checked all three proofs line by line and they are correct, including the intervention limit π0α^K/(π0α^K + 2π1F_b(μ)), which correctly retains the generator's 1/2 vs F_b(μ) factor.
- The contribution is packaging well-known meta-science logic (Ioannidis; Munafò & Davey Smith; Lawlor et al. 2016 on triangulation) for the AI-scientist setting. That is useful but incremental. It does not justify presenting these as three main theorems at a top ML venue.

### W2. The "one-bit ceiling" is an artefact of binarized decisions, not an information limit

- Λ ≤ 2 holds only because each modality is reduced to a fixed-α pass/fail bit in the large-n limit. With the continuous estimate X_{h,m} → θ + b_{h,m} and known F_b, one modality's likelihood ratio φ((x−μ)/σ_b)/φ(x/σ_b) is unbounded. For μ/σ_b = 2, a single observation can carry far more than one bit.
- So "each such modality contributes at most one bit of evidence" (abstract, §4, §7) is false as a statement about evidence. It is a statement about a particular, deliberately wasteful decision rule.
- The Limitations section mentions effect-size thresholds, but the headline claim is not qualified accordingly. The "≥ 6 observational modalities" figure is therefore a property of the test, not of the epistemic situation.

### W3. The synthetic generator is a strawman, which drives the headline empirical numbers

- The generator ranks pairs by the |t| of a *bivariate* regression on a fully observed 30-variable SCM. It adjusts for nothing and cannot orient edges.
- 52% of proposals are false through reverse causation or common causes that are *observed*. Any minimally competent observational pipeline would remove a large fraction of these, for example by adjusting for observed covariates, running PC/GES/LiNGAM-style orientation, or applying ICP across the environments the paper already simulates (§2 cites Peters et al. 2016).
- The headline contrast (0.787 → 0.698 → 0.007) is therefore largely a consequence of the generator and verifier being designed to share a bias that the available data could partly remove.
- The paper acknowledges that the environment verifier is "weak by design". But the conclusion that "the answer depends almost entirely on the evidence channel ... not on how clever its reasoning is" (§7) is exactly what this design cannot test, because reasoning quality was held at its minimum.
- A fair test needs at least one stronger observational generator or verifier.

### W4. The E3 "bias-source decomposition" success does not support the shared-bias mechanism it is credited to

- In `analyze_exp2.py` the "class-specific rates" model is `piS*fS**K + piI*fI**K`. This model treats the "shared" class as passing each environment *independently*, with rate 0.939. It is a stratified independent-bias model: a mixture of heterogeneous pass rates. It is not a shared-bias model.
- The model that actually encodes Proposition 3's shared bias ("perfectly correlated") overshoots, giving 0.724 at K = 4. That is outside the held-out 95% CI [0.683, 0.710].
- The empirical lesson is therefore "heterogeneity of per-class pass rates matters; a pooled rate is wrong". That is a Jensen/mixture effect. It is not "shared bias across modalities makes triangulation fail".
- The abstract's phrasing ("bias-source decomposition ... predicts held-out triangulation FDR to within 0.011") invites the wrong reading.
- The calibration also uses ground-truth class labels. The comparison with the naive model is therefore between an oracle-stratified and an unstratified predictor, so its practical predictive value is unclear. A held-out-label or estimated-label version is needed.

### W5. Several "predictive checks" are near-tautological or uninformative

- **E5 / Table 5.** The intervention verifiers are independent fresh randomized experiments with zero bias, by construction. Theorem 2's independence assumption therefore holds exactly, and the FDR predictions are almost guaranteed.
- **K\* agreement.** K\* = 2 in every band, so "correctly predicts that two interventions are needed in every band" is a single, non-discriminating outcome out of K ∈ {1, 2, 3}. The test has essentially no power to falsify the rule.
- **Wording.** The "pre-registered-style" label (Contribution 4) is not justified: nothing was pre-registered.

### W6. The key theoretical shape prediction did not replicate in the testbed

- The non-monotone FDR curve (Theorem 1(b)) does not appear in the SEM worlds (§6.4). The authors acknowledge this.
- Combined with the observed optimum in E1 lying at n ≈ 39, far below the stated crossover n× ≈ 271, the n× heuristic is only loosely connected to where FDR actually turns.
- In the SEM worlds every proposal passes self-verification at every n. The "beyond n×" regime is asserted rather than demonstrated by any variation.

### W7. The link to AI is thin

- No AI system, LLM or agent is evaluated. The "AI scientist" is a correlation ranker. The paper's claims about deployed systems (Kosmos, AI Scientist, co-scientist) amount to analogies: for example, that "expert-judged accuracy should be read as replication rates".
- The proposed real-system protocol is not run.
- As a contribution to the research question "Can AI propose and validate original hypotheses?", the evidence speaks to statistical verification design in general, not to AI capability.

### W8. Minor issues

- "Observed common cause" means any common ancestor among the observed variables. Such confounders could in principle be adjusted for, which reinforces W3.
- The second half of the abstract is crowded with numbers. The abstract also does not disclose that no AI or LLM was used, which the reader needs to know early.
- The Eberhardt et al. citation is imprecise: the cited UAI 2005 paper is about multi-variable interventions.
- The claimed runtime understates the actual cost.
- §6.1 says the minimum self-verification FDR is 0.450, but at α = 0.05 the small-n behaviour is driven by low power. A sentence explaining why self-verification is *non-monotone* even without replication would help.
- The conclusion's "70–79% of accepted original claims false" is specific to one toy generator. It should not be read as a general rate.

## Questions to authors

1. How do Table 2 and Table 3 change with a generator that adjusts for observed covariates, or that uses a standard causal-discovery algorithm (PC/GES/LiNGAM) or ICP over the four environments?
2. Can you state the information-theoretic version of Theorem 2 using the continuous estimates (for example, the Bayes-optimal combination over modalities with known F_b)? How many modalities are then needed for q = 0.1?
3. Can you rerun E3 with class shares *estimated* from a small validation subsample, rather than with oracle labels? Please also report the correlated-shared-class model's error relative to the CI honestly in the abstract.
4. Is there a configuration (σ_b, noise, n) of the SEM worlds where the rising FDR limb of Theorem 1(b) actually appears? If not, why should practitioners believe it?
5. Can you vary π1 enough in E5 that K\* changes across bands, so that the K\* rule is actually tested?

## Scores

| criterion | score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |

**Overall: 5 / 10 (weak reject).**

## Decision

**Major revision / weak reject.** The paper is honest and fully reproducible, and its accounting rule ("count independent bias sources, not replications") is a sensible message. However:

- the theorems are elementary restatements of known meta-science facts;
- the "one-bit" headline is an artefact of binarized testing;
- the empirical contrast is driven by a deliberately naive generator;
- the E3 "decomposition" success does not support the shared-bias mechanism it is credited to;
- no AI system is involved.

The paper would fit a workshop on AI for science or meta-science more naturally than a top-venue main track, unless it adds a non-strawman generator or verifier and a continuous-evidence version of Theorem 2.
