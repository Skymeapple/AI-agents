# Response to Round-1 Review (p031)

We thank the reviewer for a careful report and for re-running the code. The central criticism (W1) is correct, and we have rewritten the paper around the corrected statement.

Every new number below comes from code run in this environment, with outputs in `results/`:
- `revision_results.json`, from `run_revision.py`;
- `theory_results.json`, from `run_theory.py`, regenerated;
- `toy_models.json`, from `toy_models.py`, 3 seeds.

We re-ran all scripts from scratch before writing this response. `theory_results.json` and the toy outputs reproduced bit-for-bit, apart from wall-clock timings. One new validation check was added to `run_revision.py`. Because it draws from the shared RNG, the random configurations in the later property tests changed: the old-claim violation count moved from 302 to 304, and the new floors still have 0 violations.

## Major issues

### W1. Theorem 1 was wrongly extended to ensembles. **Accepted, fixed.**

The reviewer is right. Fréchet–Hoeffding gives a *lower* bound on ensemble detection, and Theorem 1 needs an *upper* bound. Our Table 2 contradicted the round-0 claim: anchored oversight at ρ=0 gives 0.993, against the claimed bound of 0.913.

Changes:
- **Theorem 1 restated (§4.1).** β is now defined as the ensemble's *common-mode miss probability* on an aligned lineage: the probability that *all* honest overseers miss. The bound reads P(S) ≤ exp(−λβG − n c₀(β)), with c₀ generalised accordingly. The proof is unchanged in structure. We note explicitly, agreeing with the reviewer, that the per-step argument is elementary.
- **Three corollaries replace the false claim:**
  - (1a) chain or comonotone: β ≥ 1−d₀, which is the old floor, now correctly scoped;
  - (1b) ρ-mixture: β ≥ ρ(1−d₀), a schedule-independent ensemble floor;
  - (1c) unknown dependence: no non-trivial floor.
- **New Proposition 2.** The ρ(1−d₀) rate is asymptotically *attained* by anchored oversight (μ₀=0, n→∞), with a proof sketch. Numerically, at G=4000 with μ₀=10⁻⁴, the optimised anchored loss rate divided by λ is 0.0107, 0.0266, 0.0466, 0.0741 and 0.1058, against a floor of 0.010, 0.025, 0.044, 0.070 and 0.100 for ρ = 0.1…1 (Figure 7, new). This makes the corrected floor a substantive statement about ensembles and not just a tautology.
- **Explicit correction paragraph in §4.1.** It quantifies the error. In 1,500 random anchored configurations solved by exact DP, the round-0 claim is violated in **304**. The corrected floor (realised β) and its ρ-form are violated in **0**. A further 300 brute-force configurations with κ<1 also give 0 violations.
- **Prop 3 (now Prop 4) reworded.** Ensemble detection lies in [max d_j, min(1, Σd_j)]. The lower end is what an ensemble guarantees; the upper end is what a floor needs.
- **Messages rewritten:**
  - C2, the abstract, the Interpretation paragraph, Discussion lever 2 and the Conclusion now say that the controlling quantity is the common-mode miss probability.
  - For chains this equals 1−d₀ (peer-level detection). For ρ-mixture ensembles it is ρ(1−d₀).
  - "d₀ is the single quantity that bounds bootstrapping" has been removed.
- **New consistency check in §5.3, item 4.** Table 2 now lies below the corrected floors: at ρ=0.44, 0.939 ≤ 0.950 (realised β) and ≤ 0.961 (ρ-form).

### W2. Limited novelty and depth; missing related work. **Partly accepted. Depth increased, claims toned down, references added.**

We agree that each per-step argument is elementary. The paper now says this in the introduction ("Scope of the formal contribution") and in the Limitations. We no longer present Theorem 3 (exact law) as more than the NSO product with G factored out. Its one new statement is the G-independence of Δ*.

New analysis added in response to the reviewer's suggestions ("optimal R and abort policies, or γ(Δ) scaling"):
- **Theorem 6 (new, with full proof).** For any persistence profile γ(Δ), lim_{G→∞} sup_n P(S) = sup_Δ π(Δ). The limit is positive if and only if misalignment is not absorbing. Corollaries cover constant γ (which recovers round-0 Prop 5), capability-scaled recovery (interior optimal step, π(0+)=0) and the relaxation length. See W3.
- **Proposition 2 (new):** asymptotic tightness of the ρ-floor.
- **Optimal retry budgets (new, §5.3).** We maximise S − c·F over R ∈ {1..30, ∞} for c ∈ {1, 3, 10, 30, 100}:
  - for ρ=0.44, R_opt falls from 6 to 3 as c rises;
  - for chains, R=∞ is optimal for c ≤ 10 and R=1 for c ≥ 30.

  We also compute the per-flag likelihood ratio of a compromised vs an aligned lineage: 15.4, 5.7 and 4.0 for ρ = 1, 0.44 and 0. This quantifies why resamples are better used as incriminating evidence.
- **Exact any-ρ anchored DP (new).** The paper no longer relies on Monte Carlo for any ρ<1 result.

**References.** We added both suggested papers after verifying them by web search:
- Dung and Mai (2025), "AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures?", arXiv:2510.11235. It is discussed in §1, §2 (Correlated failures) and §6. Our common-mode floor is a quantitative version of their defence-in-depth argument for one architecture.
- Xiao and Phuong (2026), "Bootstrapped Monitoring: Leveraging Transparent Reasoning to Oversee Stronger AI Agents", arXiv:2606.11998. It is discussed in §1 and §2.

We do not claim the paper is now a deep theory contribution. We have changed the framing to say that the value lies in identifying the controlling quantities (β, γ(Δ), the flag likelihood ratio) and in exact computation of each.

### W3. Prop 5 depends on parameterising recovery per step. **Accepted, analysed, restated.**

- **Round-0 Prop 5 replaced by Theorem 6 (§4.5).** It allows any γ(Δ) and proves that the G→∞ limit of the *optimised* success probability is sup_Δ π(Δ), where π = α/(α+ε) and α = (1−μ)(1−γ(Δ)).
- **Corollaries:**
  - (i) For constant γ, π is decreasing in Δ, so the old limit a/(a+ε₀) = π(0+) is the special case in which more steps always help.
  - (ii) If 1−γ(Δ)→0 as Δ→0, then π(0+)=0. The n→∞ limit at fixed G is 0 and the optimal step is interior, exactly as the reviewer anticipated.
  - (iii) The gap must exceed the relaxation length L = Δ/(α+ε) before the gap-independent regime applies.
- **New Table 3 and Figure 6.** Five persistence profiles: constant 1−γ ∈ {0.01, 0.001}; per-capability 1−γ = 1−e^{−bΔ}, matched to 0.01 or 0.001 at Δ=45.4; and a saturating profile.
  - Capability-scaled recovery lowers the asymptote from 0.904 to 0.643 (1−γ=0.01 at Δ_ref) and from 0.486 to 0.153 (1−γ=0.001).
  - The optimal step becomes finite (≈42 Elo).
  - The relaxation lengths are 3,000–7,000 Elo.
  - The closed form matches the DP to 5×10⁻¹⁵.
- **The policy message is restated as the absorbing vs ergodic dichotomy**, as the reviewer suggested. The existence of a gap-independent asymptote is robust. Its level, and the gap needed to reach it, depend on how recovery scales with step size. "Persistence, not gap, decides" has been removed.
- **The γ=0.999 tension in the old Table 3** (limit 0.495 vs best finite n 0.954 at G=400) is now explained as G ≪ L.

### W4. The toy ρ̂ measures a different quantity. **Accepted. New model-level experiment; old toy demoted.**

- **New `toy_models.py`, a model-level toy (§5.5).** It follows the reviewer's Question 4. Overseers are meta-classifiers that judge *trained networks*: tanh MLPs, either clean or backdoored, with the backdoor counted only if its attack success rate is ≥ 0.5. Each overseer has its own query set and its own pool of 250+250 shadow models, and is thresholded at a 5% target false-positive rate. The unit of observation is one draw of a misaligned model, which is BOS's ρ. There are two ensembles (a capability ladder q=4…256, and peers with independent queries and shadow pools) and two conditions (natural, and best-of-5 against a stronger auditor), each over 3 seeds.
- **Result.** ρ̂ = 0.15–0.32 across the 12 seed-conditions. Every 95% bootstrap CI excludes 0 and 1. Adversarial selection lowers detection by 40–47% but does not raise ρ̂.
- **Mixture adequacy (Question 5).** We fit one ρ to the all-miss rates of all 11 subsets of size ≥ 2. The maximum absolute residual is ≤ 0.026 and the relative RMSE ≤ 1.8%, so the one-parameter form is adequate in this toy. Individual pairwise values are noisy (−0.01 to 0.35).
- **Round-0 input-level toy.** We now say it measures a different quantity. Its mean pairwise ρ̂ (0.39–0.53 per condition) is reported next to the 4-way ρ̂ (0.42–0.47), and the discrepancy is acknowledged as suggesting misspecification there. It is no longer plugged into the model. The abstract no longer calls it a measurement of ρ, and the Figure 4 caption labels the 0.44 line as the input-level value.
- **Caveats stated plainly.** The model-level overseers are weak (detection 7–22%), so they sit far from the d₀=0.9 regime. Plugging ρ≈0.2 into BOS is labelled an illustration, not a prediction. The model-level estimate is well below the round-0 value of 0.44, and we say that the round-0 number overstated dependence.

### W5. The ρ<1 Monte Carlo was never validated, and "R=∞" was R=50. **Accepted, fixed.**

- **R=∞ is now truly unlimited.** `simulate` loops until every replication accepts, and raises an error at a safety cap instead of counting aborts. All R=∞ entries now show A=0.000.
- **The any-ρ anchored DP is exact.** It matches brute-force enumeration to 2.2×10⁻¹⁶ in 60 random configurations. All Table 2 and ρ-sweep numbers now come from it.
- **Brute-force enumeration validates Monte Carlo.** `bos_model.brute_force` enumerates all alignment histories and honesty patterns. Over 40 random configurations with every parameter randomised, including κ<1 and γ<1, chain and anchored, Monte Carlo matches it with maximum absolute error 0.0026 and maximum |z| of 3.4 over 120 probabilities. Monte Carlo also matches the exact Table 2 to within 0.0007.
- **Values that changed:**
  - ρ=0.44 at R=∞: P(F) 0.059 → 0.061, A 0.002 → 0;
  - ρ=0 at R=∞: P(F) 0.006 → 0.007, A 0.001 → 0;
  - ρ-sweep at ρ=0.2: P(F) 0.021 → 0.028. The round-0 Monte Carlo value was off by more than sampling error because of the retry cap.

## Minor issues

| Issue | Change |
|---|---|
| §5.4 vs §5.5 cross-references | All references updated; the toy measurement is §5.5 throughout. |
| "best n = 150" on the search cap | The search now extends to n=3000 and reports an `at_cap` flag. For constant γ<1 the best n lies at the cap because P(S) is monotone in n. The paper now says so and explains it through Theorem 6, Corollary (i). The κ=0.5 comparison is now reported at fixed n=150 (0.985 → 0.995); the round-0 value of 0.991 came from code that has since been corrected to fix overseer honesty within a step. |
| "First order" closed form | §4.2 now states that each factor is linearised separately and their product is kept, which retains the second-order cross term λd₁Δ² and drops μ₀λΔ, μ₀² and the O(μ²) denominator. |
| Theorem 2(a) wording | Now Theorem 3(a): "For every n, P(S) ≤ exp(−G r*). The bound is achieved by an integer n exactly when G/Δ* is an integer." |
| Thm 1 property test covers only chain with κ=1 | Two new tests: 1,500 anchored configurations with any ρ (exact DP) and 300 brute-force configurations with κ<1. Both give 0 violations (§5.1). |
| Sensitivity covers only r*/λ | New 81-setting sensitivity of the Table 2 conclusions (d₀, k, μ₀, f). The structural claims hold in all 81. "About half the benefit is lost at ρ=0.44" holds (median 0.51, range 0.45–0.78). "R=5 is a reasonable operating point" does **not** hold robustly: its success cost reaches 0.85. We withdrew it and replaced it with the S − cF optimisation. |
| Ctrl-Z authors | Now lists all eight authors in order (Bhatt, Rushing, Kaufman, Tracy, Georgiev, Matolcsi, Khan, Shlegeris), verified by web search. |
| Prop 4 limit depends on f | Now Prop 5: the chain limit is stated with s and q computed at the same f. |
| ρ̂ range in abstract | The abstract now reports the model-level range over individual seed-conditions (0.15–0.32). The input-level per-seed range (0.36–0.48) appears in §5.5. |

## Answers to the reviewer's questions

1. **Reconciling Table 2 with the claim.** It cannot be reconciled; the claim was wrong. It has been replaced by the common-mode floor, which Table 2 satisfies (§4.1, §5.3 item 4).
2. **1−γ scaling with Δ.** Theorem 6 and Table 3 cover this. The n→∞ limit at fixed G goes to 0, the optimal step is interior, and the G→∞ asymptote sup π is still gap-independent but lower.
3. **Exact validation of the ρ<1 simulator.** Done via brute-force enumeration and a new exact DP (W5).
4. **Toy at the level of trained models.** Done with backdoored MLPs and meta-classifier overseers (W4).
5. **Pairwise ρ̂.** For the input-level toy they differ from the 4-way value, which suggests misspecification there. In the model-level toy, a single ρ fits all 11 subsets with ≤ 0.026 absolute residual.

## Files changed

- `paper.md`: rewritten §§Abstract, 1, 2, 3 (computation), 4.1, 4.3, 4.4, 4.5, 5.1, 5.3, 5.4, 5.5, 6, 7, 8 and References.
- `code/bos_model.py`:
  - `anchored_exact` (any ρ);
  - `brute_force`;
  - `chain_two_state` with γ(Δ);
  - true R=∞ in `simulate`;
  - honesty fixed within a step in `chain_exact`.
- `code/run_theory.py`: Table 2 and the ρ sweep are computed exactly, with Monte Carlo kept as a cross-check; the γ/κ search is extended with an `at_cap` flag.
- New `code/run_revision.py`: validation, ensemble-floor tests, anchored rate vs ρ-floor, optimal R, sensitivity, γ(Δ). Outputs: `results/revision_results.json`, `fig6_gamma_delta.png`, `fig7_rho_floor.png`.
- New `code/toy_models.py`, with outputs `results/toy_models_seed{0,1,2}.json` and `results/toy_models.json`.
- `README.md`: updated commands.
