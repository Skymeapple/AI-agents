# Response to Review (Round 1)

We thank the reviewer for a careful report and for reproducing every number. We accepted every major point. Two of them, M3 (the matched-cost baseline) and M4 (claims contradicted by our own tables), changed the paper's central claim. The thesis is now narrower and, we believe, correct: cross-generation oversight buys *recoverability after an escape* (propagation blocking), not *more detection*. It beats same-generation redundancy at matched cost only under conditions that we now state precisely.

New code:
- `code/experiments_r1.py`: experiments R0–R7, about 4 minutes on 4 cores;
- `code/analyze_r1.py`;
- extensions to `code/nso.py`.

New outputs are in `results/r1_*.json` and in `fig_r1_*.png`. The round-1 outputs regenerate byte-identically. The only exceptions are the E3 figure, which was fixed (m2), and the timing fields of `summary.json`.

---

## Major issues

### M1. Results rest on hand-picked stylised regimes; the Engels et al. fits are not used; no sensitivity analysis.

**What we did.**

1. **Calibration to published values (R1).** We tried to retrieve the numerical Guard/Houdini fit parameters from arXiv:2504.18530. Network access to arXiv was blocked in our environment, and web-search snippets did not contain the fit table. We did not want to guess numbers, so we calibrated to what we could verify:
   - the four published NSO success rates at Δ = 400 (51.7%, 13.5%, 10.0%, 9.4%);
   - the published qualitative slope descriptions: Debate has a positive Guard slope and a near-zero Houdini slope; Backdoor Code has a Houdini slope steeper than the Guard slope.

   For each game we take three slope pairs consistent with the description. For Mafia and Wargames, for which we use no slope information, we take symmetric and both asymmetric pairs. For each pair we solve for β so that our optimal chain reproduces the published number exactly. That gives 12 calibrated games, each evaluated over Δ ∈ {400, 800, 1600}, ρ ∈ {0, 0.5, 0.9}, ρ_s ∈ {ρ, (1+ρ)/2} and B ∈ {16, 32}: 432 design-study rows over 143 designs. Results are in Section 6.3 and Table 3 (`r1_r1_calibrated.json`).

   We state plainly in Section 5 and in the Limitations that these are calibrations to one number per game, not the Engels et al. fits. If the reviewer can confirm the fitted slopes, plugging them in is a one-line change (`SLOPES` and `PUBLISHED` in `experiments_r1.py`).
2. **Sensitivity sweep (R2).** We swept m_G ∈ {0.25, 0.5, 0.75, 1}, m_H ∈ {0, 0.5, 1, 1.25}, β ∈ {−500, −300, −100}, Δ ∈ {800, 1600} and ρ ∈ {0, 0.5}: 192 settings (`r1_r2_sensitivity.json`, `r1_r2_summary.json`, `fig_r1_sensitivity.png`). Findings:
   - graphs are never worse than chains;
   - they gain more than 0.1 in 61–64% of settings;
   - the decisive variable is the direct-oversight probability p(0 → target). The median gain is 0.0007 when it is below 10⁻³ and 0.60 when it is between 0.05 and 0.2.
3. The headline example "0.168 → 0.772" has been removed from the abstract. The abstract now reports only aggregate, calibration-based statements.

### M2. Gains depend on assumptions favouring graphs: perfect correction, OR aggregation, no false-alarm cost.

**What we did (R3, Section 6.5).** We added three robustness axes to the exact DP and re-optimised the full design space under each:
- **correction probability c** ∈ {1, 0.9, 0.7, 0.5}: a flagged level becomes trusted only with probability c, and otherwise stays compromised and colludes;
- **threshold aggregation** t ∈ {1, 2};
- **false alarms:** an objective S_target − κ·E[#falsely flagged levels] with κ ∈ {0.02, 0.1} and φ ∈ {0.01, 0.05}, plus a hard constraint P(any false flag) ≤ 0.25 at φ = 0.05.

The four games are symmetric, calibrated Debate, Backdoor Code and Mafia, with Δ ∈ {800, 1600} and ρ ∈ {0, 0.5}. That is 128 settings, of which 82 are non-degenerate. All extensions are validated against Monte Carlo (R0: 24 random configurations, max |z| = 2.37).

Findings, as now reported in the paper:
- **Imperfect correction** lowers all families but does not reverse the ranking. Cross-generation designs are best in 14/24, 15/22, 12/19 and 9/17 non-degenerate settings at c = 1, 0.9, 0.7 and 0.5.
- **Threshold aggregation** hurts graphs most, because distant, weak overseers rarely supply the second detection. With t = 2, cross-generation designs are best in 12/32 settings, against 38/50 with OR.
- **False alarms are binding.** With a cost of 0.1 per false flag at φ = 0.05, k ≥ 3 designs remain optimal in 26/65 settings. Under the hard constraint with OR, k ≥ 3 is never optimal (0/41), so the round-1 recommended designs are infeasible under such a budget. This directly answers the reviewer's question 5: no, not under a false-alarm budget.

Independent stealth across levels was already tested by Monte Carlo (E4). We now also test distance-dependent correlation (see m4).

### M3. Unfair equal-budget baseline: no same-generation-redundancy baseline.

We agree, and this was the most consequential point. We added two new design families. *Chain-r* has r ∈ {2, 3, 4} same-generation copies per level. Copies share their generation's trust status: copies of a compromised generation are compromised, which is the collusion semantics the reviewer suggested. *Hybrids* combine windows with copies. Copies have their own correlation ρ_s ≥ ρ (nested copula). All families are compared at equal cost (R1, R7; Section 6.3).

Results (calibrated games, 36 settings per cell):
- **Copies as diverse as generations (ρ_s = ρ).** At ρ = 0, chain-r is the best family in 29/36 settings at B = 16. At B = 32, cross-generation designs are at least as good as chain-r in 26/36, mostly as hybrids. At ρ = 0.9, the pure graph is best in 30/36.
- **Copies more alike than generations (ρ_s = (1+ρ)/2).** Cross-generation designs are at least as good as chain-r in 31–32/36.
- **Strict success.** Same-generation copies win in essentially every setting: 36/36 in all cells but one, which has 33/36.
- **Crossover in ρ_s (R7, stylised regimes).** In the symmetric regime at Δ = 1600 and ρ = 0, the crossover lies at ρ_s between 0.2 and 0.4. In the guard-steeper regime, and in the houdini-steeper regime at Δ = 800, the graph wins even at ρ_s = ρ = 0. At ρ = 0.5 it wins at ρ_s = ρ.

**How the claims changed.** The thesis is now that cross-generation redundancy's distinctive value is propagation blocking. It helps target success when errors are correlated or copies are alike, and it never helps strict success. We added a pre-registered prediction (P4) for this crossover in terms of the measurable difference ρ̂_s − ρ̂. In the related-work section we note that the empirical sign of ρ_s − ρ is unsettled: Koran et al. (2026) find that diverse monitors beat identical ones, and Jha (2026) finds that lineage diversity did not help at matched skill.

### M4. Summary claims contradicted by the paper's own tables.

The reviewer is right on all three counts. We audited the round-1 E6 table programmatically (R6, `r1_r6_audit.json`) and corrected the text (Section 6.4).
- *"Redundancy cannot rescue a game whose Houdini slope exceeds its Guard slope"* is withdrawn. The paper now says explicitly that houdini-steeper success rose from 0.500 to 0.998 at Δ = 400 and from 0.100 to 0.733 at Δ = 800. The failure is a large-gap, tiny-p phenomenon. R2 confirms this: the gain exceeds 0.1 in 42–46% of Houdini-steeper settings, and the median gain is about 0 only when p(0 → target) < 10⁻³.
- *Regularity 1* ("whenever not Houdini-favoured") is withdrawn and replaced by the p(0 → target) characterisation.
- *Regularity 4 (anchoring)* is corrected. The anchor is chosen in 25/27 houdini-steeper settings (all 9 at Δ = 1600), in 24/27 guard-steeper settings, and in 7/27 symmetric settings (all at Δ = 400). In the calibrated games: Debate 26/27, Mafia and Wargames 13/27 each, and Backdoor Code 0/27, where the full topology wins all 27. The paper now ties anchor use to the level-0 overseer being relatively strong against the target, not to Guard-favoured slopes.
- The abstract and conclusion were rewritten to match.

### M5. Missing related work; overstated novelty.

**What we did.** We verified all five suggested papers via web search (titles, IDs and authors) and now cite and discuss them in a new related-work paragraph, "Multi-overseer monitoring and correlated failure":
- Zanolini (2026), arXiv:2610.29178;
- Koran et al. (2026), arXiv:2605.15377;
- Jha (2026), arXiv:2608.16190;
- Bhatt et al. (2025), Ctrl-Z, arXiv:2504.10374;
- Overman & Bayati (2026), arXiv:2605.28807 (ICML 2026).

We position all of them as single-step, many-monitor work. Prop. 2 is now explicitly labelled a textbook consequence of Slepian's inequality and is no longer listed as a contribution. The round-1 claim that our search "found no quantitative treatment" is removed. The contributions list now puts the propagation analysis and the matched-cost test first, and calls the overall contribution "an incremental but clarifying extension of Engels et al." The DP and simulator are described as supporting tools.

### M6. Thin theory; Prop. 3 is only a sketch.

**What we did.** We replaced the proof sketch with a rigorous theorem and a clearly labelled conjecture (Section 4).
- **Theorem 3(i): exact identity.** h = q_k·ν(TC^{k−1}), where ν is the quasi-stationary distribution. Proof: sum the eigen-equation over states.
- **Theorem 3(ii): universal two-sided bounds, valid for every ρ.** (1 − k·a·λ^{−(k−2)})·R_k(ρ) ≤ h ≤ λ^{−(k−1)}·∏qₘ, with a = O(q₁) explicit. The proof combines (k−1)-step path counting under the QSD, monotonicity of all-miss probabilities in the set of overseers, and a projection argument that bounds the QSD mass of compromised bits.
- **Theorem 3(iii): sharp bounds for k = 2,** R₂(1 − O(q)) ≤ h ≤ R₂/(λ − q₁(1−q₂)/λ).
- **Corollaries.** h = ∏qₘ(1+O(kq₁)) at ρ = 1, which is now rigorous, including the uniformity the reviewer asked about. h = R₂(1+O(q₁)) for k = 2 at every ρ. h ≥ R_k(1−O(kq₁)) for k ≥ 3 at every ρ.
- **Conjecture 3′.** The matching upper bound h ≤ R_k(1+O(q)) for k ≥ 3 and ρ < 1 is stated as a conjecture. We explain why our upper-bound argument is too generous there.
- **Evidence (R5).** We computed the QSD hazard in 60-digit arithmetic for k = 2–5 at ρ ∈ {0, 1}, with q scaled down by 50×. The ratio h/R_k − 1 behaves as +q₁ at ρ = 0 and −q₁ at ρ = 1, and reaches 1.0020 at ε = 0.002 for k = 3–5. On 259 random configurations the bounds hold in every case, and the lower bound is non-vacuous in 247. The identity matches 1−λ to a relative error of 1.8·10⁻⁷ wherever h > 10⁻⁸ (`r1_r5_theorem3.json`, `fig_r1_theorem3.png`).

Prop. 4 is labelled an elementary computation. We do not claim the theory is deep. It is now correct as stated, and its unproved part is flagged.

---

## Minor issues

- **m1 (prover–estimator authors).** Fixed. arXiv:2506.13609 is by Brown-Cohen, Irving and Piliouras, verified via search. The title is now "Avoiding obfuscation with prover-estimator debate".
- **m2 (E3 leading-order curve > 1).** Fixed. The curve is drawn only where it is ≤ 0.5 and is labelled a heuristic (`experiments.py`, `fig_e3_accumulation.png`).
- **m3 (decomposition without error control).** The decomposition P ≈ per-level + (n−n₀)r_k has been removed as a quantitative claim. The paper notes its 2.4·10⁻³ versus 3.2·10⁻³ discrepancy and uses only the exact DP for numbers.
- **m4 (equicorrelation).** We added R4 (Section 6.6), a Monte Carlo study with distance-decaying correlation ρ_adj·η^{|i−i′|−1} for four (ρ_adj, η) pairs. Plugging the mean pairwise correlation into the one-factor model errs by at most 0.014 in target success and 0.012 in strict success. The ranking over k is unchanged, and the one-factor model is slightly optimistic for k = 4.
- **m5 (tetrachoric identification).** Section 7 now states that identification holds only under the one-factor Gaussian model with non-adaptive attacks. It explains that tail dependence or topology-aware attackers would make the measured ρ underestimate the deployment ρ, and points to R4 for the one form of misspecification we tested.
- **m6 (P2 "linear").** Corrected. Chain compromise now "approaches 1 geometrically" at rate q̂₁. The window-2 prediction is given as the bracket [π̂₀q̂₂, q̂₁q̂₂] from Theorem 3.
- **m7 (venue line).** Removed.
- **m8 (full topology in E6).** The full topology is now part of the design space in R1, R3 and R7. It is chosen, for example, in all 27 calibrated Backdoor Code settings.
- **"At most 0.0033" claim.** The claim is no longer in the paper, which now focuses on the matched-cost comparison. The audit JSON records the exact value, 0.003328 (`r1_r6_audit.json`).

## Answers to the reviewer's questions

1. **Engels et al. parameters.** We could not retrieve the fits (see M1). With games calibrated to their published success rates, Debate-like games reach target success of 0.79–1.00 with graphs at ρ = 0.9 (0.79–0.91 at Δ = 400). Backdoor-like games never exceed 0.42, 0.071 and 0.0018 at Δ = 400, 800 and 1600.
2. **Imperfect correction.** See M2. All families degrade, and cross-generation designs remain best in roughly half or more of the non-degenerate settings down to c = 0.5.
3. **Chain with r same-generation checks.** See M3. The window graph still wins on target success when ρ is high or ρ_s > ρ. It does not reliably win when ρ_s = ρ ≈ 0, and it never wins on strict success.
4. **Rigorous Prop. 3.** See M6. It is rigorous for ρ = 1 and for k = 2, and gives a rigorous lower bound in general. The general-k, ρ < 1 upper bound is a conjecture with numerical support.
5. **k ≥ 3 at φ = 0.05 with a false-alarm cost?** k ≥ 3 survives a soft cost of 0.1 per false flag in 26/65 settings. It never survives a hard budget P(any false flag) ≤ 0.25 with OR aggregation.

## Citation integrity

Every added reference was checked by web search for existence, ID, title and authors: arXiv:2610.29178, 2605.15377, 2608.16190, 2504.10374, 2605.28807 and 2506.13609. One corrected author list is the only change to existing references. No number in the paper comes from a source other than our own code outputs, except the four Engels et al. success rates, which are cited.
