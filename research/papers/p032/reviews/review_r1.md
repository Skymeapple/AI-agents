# Review (Round 1): "Beyond the Chain: Redundancy, Correlated Errors, and Collusion in Nested Scalable Oversight"

Reviewer role: independent area-chair-level reviewer. Target venue: NeurIPS 2027 main track.

## Summary

The paper generalises the nested scalable oversight (NSO) model of Engels et al. (2025). In that model, a chain of Elo-rated Guard/Houdini games succeeds with probability prod_j p_j. The submission replaces the chain with "oversight graphs": a window-k topology, an optional human anchor, and the full topology. It adds two parameters. The first is within-level error correlation rho, modelled with a one-factor Gaussian copula. The second is a collusion parameter gamma: a compromised overseer keeps a fraction gamma of its detection ability. The paper separates *strict* success (no level ever escapes) from *target* success (the final level is trusted).

Contributions:
- an exact dynamic program over the trust states of the last k levels, validated against Monte Carlo;
- Prop. 2: OR-detection is monotone in rho, via Slepian's lemma;
- Prop. 3: under full collusion, the hazard of irrecoverable compromise is prod q_m^m at rho=0 and prod q_m at rho=1 (proof sketch, checked numerically);
- Prop. 4: with gamma>0, chain compromise reaches a stationary level;
- an 81-setting budget-constrained design search in three stylised regimes;
- a measurement protocol and three pre-registered predictions.

The paper runs no experiments with real models.

## Reproduction

I copied `code/` to a scratch directory and reran `python3 experiments.py`. It took 56 s, matching the paper's runtime claim. Every JSON/CSV output (e1–e8 and `e6_budget_table.csv`) is byte-identical to the shipped `results/`. Only `summary.json` differs, and only in its timing fields.

I checked the following numbers in the paper against the outputs. All match:
- E1: max |z| 2.30; max absolute error 0.0023 (target) and 0.0021 (strict).
- E2: n* = 2/4/7 with success 0.410/0.168/0.028; 0.00056 for n=1.
- E3: the n=30 and n=60 failure probabilities.
- Table 2: every hazard, prediction and ratio.
- E4: the two-overseer detection curve and the system table.
- E4: the cross-level Monte Carlo results.
- E5: the gamma table and the stationary check.
- E6: all selected rows.
- E7 and E8.

I also checked two claims directly from the files:
- "At rho=0.9 the strict gain is at most 0.0033": `e6_budget.json` gives 0.00333 (symmetric, Delta=800, B=32). The rounded CSV would suggest 0.004, so the claim is right only when computed from the JSON.
- "Best target design never worse than chain across 81 settings": confirmed.

**No fabricated or unsupported numbers were found.** This part of the submission is exemplary.

## Reference spot-check

| Reference | Result |
|---|---|
| Engels et al. 2025 (arXiv:2504.18530) | Exists. The figures 13.5% (Mafia), 51.7% (Debate), 10.0% (Backdoor Code) and 9.4% (Wargames) at a 400-Elo gap are confirmed. |
| Goel et al. 2025 (arXiv:2502.04313) | Exists (ICML 2025). CAPA, the finding that errors become more similar with capability, and the judge-similarity bias are confirmed. |
| Kenton et al. 2024 (NeurIPS 37, arXiv:2407.04622) | Exists. The "mixed results across asymmetries" characterisation is fair. |
| Brown-Cohen et al. 2025, prover–estimator debate (arXiv:2506.13609) | Exists and addresses obfuscated arguments. However, the arXiv listing I found names only Brown-Cohen, Irving and Piliouras. The extra authors "Chen, L., Li, J., & Xun, Z." could not be verified; please check the author list. |
| Ladha 1992, AJPS 36:617–634 | Exists. The bibliographic details match. |
| Slepian 1962, BSTJ 41:463–501 | Exists. It is the standard citation for Slepian's lemma, and the use in Prop. 2 is correct. |
| Greenblatt et al. 2024; Irving et al. 2018; Burns et al. 2023 | Well-known, with correct arXiv IDs. |

No reference is fabricated. One author list is probably wrong (minor).

## Strengths

1. **A sharp, well-posed question.** Does the pessimism of the chain-NSO model come from weak per-step games, or from the composition rule? The strict/target distinction is a useful conceptual contribution. The observation that the chain's product form is entirely a propagation assumption (Prop. 1 with gamma=1) is clean and instructive.
2. **Fully reproducible and honest.** All numbers regenerate exactly. Scope limits are stated clearly: the parameters are stylised and no real-model experiments were run.
3. **Correct core mathematics.**
   - Prop. 2's use of Slepian's lemma is correct, including the endpoint D(p;1) = max p.
   - Prop. 4 is a correct two-state Markov-chain computation.
   - I checked the combinatorics of the Prop. 3 sketch line by line. The exponent bookkeeping prod_{i=0}^{k-1} prod_{m=i+1}^{k} q_m = prod q_m^m is right. The rho=1 path factor min_i q = q_{i+1} is right given monotone q_m. The entry condition through (T, C^{k-1}) is right under gamma=0.
   - Table 2 confirms Prop. 3 to within 2.5% in the small-q regime.
4. **Good internal validation.** The DP and Monte Carlo agree on 24 random configurations. The paper says where its asymptotic formula breaks down (ratios 0.77–1.34) instead of hiding it.
5. **Separating per-level redundancy from propagation blocking** is a genuinely useful qualitative insight: correlation kills the first but not the second. It yields a concrete design heuristic: keep several generations of overseers in the window, and keep intermediate models non-deployed.

## Weaknesses

### Major

**M1. All quantitative results rest on three hand-picked stylised regimes, even though fitted parameters exist.** Engels et al. publish fitted Guard/Houdini slopes and intercepts for all four games. Plugging them in costs almost nothing and would ground the design study. As written, statements such as "0.168 → 0.772" depend entirely on beta and the slopes, which the authors chose. The "debate-like" and "backdoor-like" labels invite readers to transfer the conclusions even though the values are invented. At a minimum, add a sensitivity analysis over (m_G, m_H, beta), or use the Engels fits.

**M2. The headline gain is driven largely by modelling assumptions that favour graphs, and the paper underplays this.**
- (a) Detection is perfect correction: a detected Houdini "becomes T". Self-healing is entirely a consequence of this. Each detection of a successor by a still-trusted older overseer produces a fresh trusted level.
- (b) OR aggregation with no false-alarm cost in the objective. E8 shows a 54–79% chance of at least one false alarm for the recommended designs at phi=0.05. Yet the optimisation in E6 ignores this, so the designs it picks may be infeasible.
- (c) Independent stealth across levels in the theory.

Each of these is listed in the limitations, but none is treated as a robustness axis in the main design study. A cost-weighted objective, or a k-of-s threshold aggregation, should be optimised in E6. Imperfect correction (probability c that detection yields a trusted model) is a one-line change to the DP and should be swept.

**M3. The baseline is unfair in the "equal budget" comparison.** The chain is said to be unable to convert budget into success. But the model gives the chain no way to spend budget except by adding depth. A natural baseline gives each level r redundant checks by same-generation overseers: repeated games, or several independent instances or families at level j−1, with their own correlation. Without it, the paper cannot tell whether the gains come from cross-*generation* redundancy (the paper's thesis) or simply from more checks per level. Under propagation, same-generation copies of a compromised level are presumably also compromised, which is exactly the comparison that would support the thesis. Please add it.

**M4. Several general claims are not supported by the paper's own tables.**
- The abstract and Section 8 say "redundancy cannot rescue a game whose Houdini slope exceeds its Guard slope". But in the houdini-steeper regime at Delta=800, B=32, target success rises from 0.100 to 0.733, a 7× gain, which is larger than many symmetric-regime gains. At Delta=400 it rises from 0.500 to 0.998. The failure appears only at Delta=1600. So the failure is a large-gap / tiny-p phenomenon, not a slope-sign phenomenon.
- Regularity 1 says graphs dominate "whenever the per-step game is not Houdini-favoured". This is again contradicted by the houdini-steeper rows.
- Regularity 4 says anchoring "helps mainly at small gaps or Guard-favoured slopes". But `e6_budget.json` shows the anchor is chosen in 25 of the 27 houdini-steeper settings, including all 9 at Delta=1600.

Rewrite these claims to match the evidence.

**M5. Related work is missing or stale, and the novelty claim is overstated.** The paper says a "late 2026" search found no quantitative treatment. I found closely related 2025–2026 work on correlated monitor failures and ensemble monitoring:
- "Correlated Failures of AI Approval Monitors" (arXiv:2610.29178): fail-prone sets in place of independent error rates, with an attacker on the monitors;
- "Ensemble Monitoring for AI Control" (arXiv:2605.15377);
- "Decorrelation Is Not Complementarity" (arXiv:2608.16190);
- Ctrl-Z (arXiv:2504.10374), which ensembles trusted and untrusted monitors under collusion;
- "Calibrating Conservatism for Scalable Oversight" (arXiv:2605.28807), with multi-overseer panels.

None of these covers multi-generation NSO topologies, so the propagation angle may still be new. But the within-level correlation part (Prop. 2) is well known: equicorrelated Gaussian orthant monotonicity is textbook, and correlated-jury results already cover it. The novelty is therefore mainly the propagation analysis (Prop. 3) and the strict/target separation. That is a moderate, incremental extension of Engels et al.

**M6. The theory is thin for a "theory + simulation" main-track paper.**
- Prop. 2 is a direct application of Slepian's lemma.
- Prop. 4 is a two-state Markov chain.
- Prop. 3, the only non-trivial result, is a proof sketch. Its quasi-stationary argument ("cheapest path carries the leading-order mass") and the claimed 1+O(q_k) remainder are not proved, and the rho=1 case is not shown to have uniform remainder terms.

Either give a rigorous proof of Prop. 3 (for example via a perturbation bound on the quasi-stationary distribution of the 2^k-state substochastic matrix) or call it a conjecture or heuristic supported by numerics.

### Minor

- m1. The prover–estimator debate reference lists authors (Chen, Li, Xun) whom I could not verify on arXiv:2506.13609. Please check.
- m2. The E3 figure's "leading-order" curve exceeds 1 (it reaches 14.4 for the chain at n=60). Clip it, or omit it outside its range of validity.
- m3. The decomposition P(target compromised) ≈ per-level + (n−n0)·r_k is stated without error control. In Table 2 (small-q, k=2, rho=0) it gives about 2.4e-3 against an exact 3.2e-3.
- m4. The one-factor copula forces equal pairwise correlation among all overseers of a level. Distance-dependent correlation (adjacent generations are more similar) is the realistic case and could reverse some rankings. Please discuss.
- m5. Section 7 states that rho "equals the tetrachoric correlation". This holds only under the one-factor model and only for attacks by the same Houdini. Identifiability under model misspecification is not discussed.
- m6. The prediction P2 says chain compromise "grows roughly linearly in depth". Under gamma=0 it saturates exponentially toward 1 (E3), so it is not linear.
- m7. The bilingual abstract is fine, but the venue line ("Target venue: NeurIPS 2027 … submission draft") should be removed for anonymised review.
- m8. Report results for the full topology alongside window-k in E6. The full topology appears only in E7.

## Questions for the authors

1. How do the E6 conclusions change with the Engels et al. fitted slopes and intercepts for Debate and Backdoor Code?
2. What happens when detection corrects a level only with probability c < 1, or when a corrected level stays compromised with some probability?
3. Against a chain baseline with r same-generation checks per level, under matched cost and the same collusion semantics, does the window graph still win on target success?
4. Can Prop. 3 be made rigorous, with an explicit remainder bound?
5. Once a false-alarm cost is included, does the optimal design still prefer k ≥ 3 at phi = 0.05?

## Assessment of conclusions

The structural conclusions follow from the model:
- the chain's product form is a propagation assumption;
- correlation erodes per-level redundancy but not propagation blocking;
- partial collusion causes saturation.

The quantitative and design conclusions do not transfer, for three reasons: the parameters are invented, the baselines are restricted, and the objective omits false alarms. Some summary claims are contradicted by the paper's own tables (M4). The sub-question is sharp. The evidence supports a careful model paper of workshop or borderline quality, not a main-track contribution that would change practice.

## Scores (1–10)

| Criterion | Score |
|---|---|
| Novelty | 5 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** (weak reject) |

**Decision: major revision.** At a main-track venue in its current form, this is a weak reject.

**Single most valuable change:** Ground the design study in the Engels et al. fitted game parameters. Add a matched-cost same-generation-redundancy baseline and an imperfect-correction plus false-alarm-cost robustness sweep. Then restate the regularities so that they match the tables.
