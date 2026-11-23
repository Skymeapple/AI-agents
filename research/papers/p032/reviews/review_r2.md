# Review (Round 2): "Beyond the Chain: Propagation Blocking, Correlated Errors, and Collusion in Nested Scalable Oversight"

Reviewer role: independent area-chair-level reviewer. I read the round-1 review and the author response, but the judgement below is my own and is based on the current paper, code and outputs.

## 1. Summary

The paper generalises the chain model of nested scalable oversight (NSO) from Engels et al. (2025) to "oversight graphs". In these graphs, level j can be overseen by:
- several earlier generations (window-k, with or without a level-0 anchor, or the full topology);
- r same-generation copies of its predecessor (chain-r);
- both at once (hybrids).

The model has these parameters:
- ρ, the error correlation across generations;
- ρ_s, the error correlation among copies of one generation;
- γ, how much detection ability a compromised overseer keeps (collusion);
- c, the probability that a flagged level is actually corrected;
- t, the number of detections needed to flag (threshold aggregation);
- φ, the false-alarm rate.

The paper separates *strict* success (no level ever escapes) from *target* success (the final level is trusted).

The revision makes four main claims.
1. **Theorem 3.** Under full collusion, an exact identity and two-sided bounds hold for the quasi-stationary hazard h of irrecoverable compromise in a window-k graph. h ≈ ∏q_m is proved at ρ = 1, and h ≈ R_2 at k = 2. For k ≥ 3 and ρ < 1 only a lower bound is proved; the matching upper bound is stated as Conjecture 3′.
2. **Matched-cost test.** Cross-generation redundancy beats same-generation copies on target success only when errors are correlated or copies are more alike than generations. It never wins on strict success.
3. **Grounded and stress-tested design study.** The games are calibrated to the four published NSO success rates. The paper adds a 192-setting slope/intercept sweep and a robustness study over c, t and false-alarm costs or constraints.
4. **Measurement protocol** with four pre-registered predictions.

The thesis is now narrower and, in my view, more defensible: cross-generation oversight buys *recoverability* (propagation blocking), not *more detection*.

## 2. Reproduction

I copied `code/` and `results/` to a scratch directory, deleted the outputs and reran everything:
- `experiments.py`: 55 s;
- `experiments_r1.py`: R0, R5, R6, R4, R1, R3 and R7 took 456 s on 4 cores. I ran R2 separately, which took 120 s, because my first invocation hit a tool time limit;
- `analyze_r1.py`.

**Every output file regenerates byte-identically.** This covers all E1–E8 JSON/CSV files, every `r1_*.json` (including `r1_tables.json`), and all figures (`fig_e*.png`, `fig_r1_theorem3.png`, `fig_r1_sensitivity.png`). Only the timing fields of `summary.json` differ. Wall time on my machine was about 2.6× the stated 218 s. That is a hardware difference and not a concern, but the README could state the reference hardware.

I checked these numbers in the paper against the outputs. All match:
- Abstract and §6.1: 24 + 24 validation configurations; R0 max |z| = 2.374; max errors 0.0024 (target) and 0.0015 (strict).
- §6.2: chain n* = 2/4/7 with success 0.410/0.168/0.028. Accumulation at n = 60: 1.000/0.879/0.314/0.111. Table 2 is unchanged.
- §4 evidence (R5): 259/259 configurations satisfy the bounds, and the lower bound is non-vacuous in 247. Identity relative error is 1.79e-7. The scaling ratios match (k = 3: 1.119, 1.010, 1.0020; k = 3 at ρ = 0.5: 1.0023).
- Table 3: every cell. I recomputed all 72 ranges from `r1_ranges_B32`.
- §6.3 winner counts: 29/36, 26/36 (17 hybrid), 27/36, 30/36 and 27/36, 31/31/32, and strict 36/36 except 33+3.
- R7 crossover numbers: 0.873/0.698/0.510 vs 0.633; 0.989 vs 0.952; 0.949 vs 0.941; 0.733 vs 0.655; 0.451 vs 0.419; 0.838 vs 0.684; 0.985 vs 0.974; 0.873 vs 0.784.
- §6.4 (R2):
  - graph never worse than chain (minimum gain 2.3e-6);
  - gain > 0.1 in 63.5% and 61.5% of settings;
  - Houdini-steeper settings: 46% and 42%; m_H ≤ m_G: 81%;
  - median gains of 0.0007 (p_direct < 10⁻³) and 0.60 (0.05–0.2);
  - graph ≥ chain-r in 49% and 83% of settings;
  - the example (0.25, 0.5, −500) at Δ = 1600 gives 0.230 → 0.928.
- R6 audit: anchor chosen in 25/27, 24/27 and 7/27 settings; calibrated anchor/full counts 26, 13, 13 and 0/27 (full 27); Backdoor maxima 0.415, 0.071 and 0.0018.
- §6.5 (`r3_agg`):
  - 82/50/32/43, 65/33/32/26 and 73/18/53/12;
  - t = 2: 12/32 vs 38/50;
  - c-sweep: 14/24, 15/22, 12/19, 9/17;
  - 0/41 for k ≥ 3 under the hard constraint with OR.
- R4: maximum one-factor errors 0.0143 (target) and 0.0124 (strict).
- E5: 0.012/0.388/0.760.

**No fabricated or mismatched number was found.** One caveat: the "x/36" winner counts in §6.3 include 8–13 degenerate settings per cell in which every design scores below 0.05, so the winner is decided by differences of order 10⁻³. `analyze_r1.py` computes a `nontrivial` count but the paper does not report it. I recomputed the counts on non-trivial settings only. The conclusions survive and become slightly stronger:
- ρ = 0.9, B = 32: graph best in 23/24;
- ρ = 0, eq, B = 32: cross-generation ≥ chain-r in 24/28;
- ρ = 0, eq, B = 16: chain-r best in 20/27.

Please report the non-trivial denominators.

## 3. A summary claim contradicted by the paper's own outputs (major)

The abstract (English and Chinese), §8 and §10 make the following claims.
- Abstract: "All designs fail when direct oversight succeeds with probability below about 10⁻³, whatever the slope pattern."
- §8: "When direct oversight across the full gap succeeds with probability below about 10⁻³, no design within budget 32 helps (Section 6.4)."
- §10: "nothing rescues a per-step game with a vanishing direct-oversight probability."

`r1_r2_summary.json` contradicts this. In the p_direct ∈ [0, 10⁻³) bin, `max_gain_vs_chain` is 0.605 and `max_graph` is 0.633 at ρ = 0. Listing the 40 such settings in `r1_r2_sensitivity.json`:

| (m_G, m_H, β), Δ = 1600, ρ | p_direct | chain | chain-r | graph |
|---|---|---|---|---|
| (1, 1, −300), ρ = 0 | 5.6e-4 | 0.028 | **0.873** | **0.633** |
| (1, 1, −300), ρ = 0.5 | 5.6e-4 | 0.028 | 0.419 | 0.451 |
| (1, 1.25, −500), ρ = 0 | 1.8e-4 | 0.012 | 0.654 | 0.351 |
| (1, 1.25, −500), ρ = 0.5 | 1.8e-4 | 0.012 | 0.257 | 0.222 |

The first row is the paper's own **symmetric regime at Δ = 1600** (`e2_chain.json`: p_n1 = 0.00056). The paper itself reports graph = 0.633 and chain-r = 0.873 for this regime in §6.3/R7. So the paper contradicts itself.

What the data support is a *median* statement: the median gain is about 0 below 10⁻³. That is not "all designs fail" or "nothing rescues". The direct-oversight probability is a useful predictor, but it is not a hard threshold. This is the same kind of over-generalisation that round-1 M4 flagged (the slope-sign claim). The authors fixed that one but introduced a new one of the same type. It must be corrected in the abstract, Discussion and Conclusion. For example: "gains are typically negligible (median 0.0007) when p(0 → target) < 10⁻³, although a few settings with steep symmetric curves still reach 0.35–0.87."

## 4. Status of round-1 issues

| Round-1 issue | Status |
|---|---|
| M1: hand-picked regimes; Engels et al. fits unused | **Partially resolved.** Calibration to the four published rates plus a 192-setting sweep is a real improvement, and it is honestly labelled. The Engels et al. fitted double-ReLU parameters are public in their paper and still have not been used; the explanation is that arXiv was blocked in the authors' environment. The slope pairs remain the authors' choice. (0.1, 1) for Backdoor Code is extreme, and m_H = 0 for Debate gives the acknowledged artefact that success tends to 1. A camera-ready must use the actual fits. |
| M2: perfect correction, OR aggregation, no false-alarm cost | **Resolved.** The c, t, κ and P(FA) constraint axes were added, validated against Monte Carlo and re-optimised. The negative findings are reported prominently: t = 2 and the hard FA constraint remove the deep-graph advantage, and k ≥ 3 is optimal in 0/41 settings. This is commendable. |
| M3: unfair baseline | **Resolved.** The chain-r and hybrid families with nested ρ_s are exactly the needed comparison, and they changed the thesis. |
| M4: claims contradicted by tables | **Resolved for the three listed claims, but a new instance was introduced** (Section 3 above). |
| M5: related work and novelty | **Resolved.** The five works are cited and positioned correctly. Prop. 2 is demoted. The novelty claim is now appropriately modest. One gap remains (Section 6). |
| M6: Prop. 3 only a sketch | **Resolved.** The rigorous Theorem 3 is checked below, and the unproved part is honestly labelled as Conjecture 3′. |
| m1–m8 | All addressed. The prover–estimator author list is now correct. The E3 curve is clipped. P2 is restated as geometric. The full topology is now in the design space. |

## 5. Proof check (Theorem 3)

I checked the proof line by line. **It is correct.**

- **Irreducibility and QSD.** From T^k, any non-absorbed s is reached by emitting its bits, and intermediate windows keep a T prefix. From any state, k detections reach T^k, and a trusted overseer is always present. Correct.
- **(i) Identity.** The only transition into C^k is from TC^{k−1} via a miss by the distance-k overseer. Summing νM = λν gives h = q_k ν(TC^{k−1}). Correct.
- **(ii, lower) Path term.** After i misses from T^k, the trusted overseers sit at distances i+1…k, so the path probability is ∏_{i≤k−2} π_i = R_k/q_k. Correct, and the factor λ^{−(k−1)} ≥ 1 may be dropped.
- **(ii, lower) Projection bound.** The bound a′ ≤ a holds:
  - from newest-T states, a miss is at most q_1 and is never absorbed for k ≥ 2;
  - from other non-absorbing states except TC^{k−1}, the nearest trusted overseer is at distance ≤ k−1, so a miss is at most q_{k−1} by monotonicity;
  - misses from TC^{k−1} are absorbed and correctly excluded.

  The superset-of-overseers argument (all-miss ≤ any single miss) holds under any copula.
- **(ii, lower) Position-m bound.** ν(bit m = C) ≤ λ^{−(m−2)}a′, by the age argument plus "not absorbed at step 1". The union bound is fine.
- **(ii, upper).** A path into TC^{k−1} needs s's newest bit to be T. That bit is a trusted overseer at distance j for the j-th new level, which bounds each step's miss probability by q_j. Correct.
- **(iii), k = 2.** λν(CT) = (1−q_2)ν(TC) and λν(TC) = ν(TT)π_0 + q_1ν(CT) are right. The algebra giving ν(TC)D_2 ≤ π_0 and the lower bound are right.
- **Corollaries.** (a): R_k(1) = ∏q_m uses monotone q, as assumed. (b) and (c) follow.

Comments:
- The theorem covers only the stationary, anchor-free case with m_G = m_H, γ = 0, c = 1 and t = 1. The design study's asymmetric games (m_G ≠ m_H) are non-stationary and fall outside it. The paper says so.
- Mathematically, this is a standard quasi-stationary perturbation argument. It is careful and correct, not deep.
- The ρ = 0 structure (absorption after k consecutive escapes) closely resembles **consecutive-k-out-of-n:F reliability systems** and Markov-chain-imbedding results (e.g. Chiang & Niu 1981; Fu & Koutras 1994), here with state-dependent failure probabilities. That literature should be cited. The "q^k" scaling and product-of-powers leading order are the kind of results it contains, which further limits the theoretical novelty.

## 6. Novelty and prior work

I searched for 2025–2026 work extending Engels et al. to multi-overseer, multi-generation topologies, or modelling failure propagation in bootstrapped oversight, and found nothing that does what this paper does. The within-step multi-monitor literature is correctly cited and distinguished: Koran et al., Jha, Zanolini, Overman & Bayati, and Ctrl-Z.

The genuinely new pieces are:
- the separation of propagation blocking from per-level redundancy;
- the matched-cost test against same-generation copies;
- the strict/target distinction.

These are clarifying but incremental. Two of the headline results are close to structural consequences of the model:
- **The strict-success finding.** "Cross-generation never wins on strict success" is almost structural at ρ_s = ρ. Replacing a distant overseer with a copy of the nearest, strongest one can only raise per-level detection.
- **The never-worse finding.** "Graphs are never worse than the chain" is guaranteed: with γ = 0, adding overseers never lowers detection, and budget 32 always admits a window-2 version of the optimal chain.

The paper should say plainly that these are consequences of the model, not empirical findings. Overall novelty is moderate.

## 7. Statistics, baselines and ablations

- The DP is exact, so sampling uncertainty applies only to the Monte Carlo validation, which is adequately reported.
- The relevant uncertainty is *model and grid* uncertainty. All "x/36", "61–64%" and "49%/83%" figures depend on arbitrary grids (slope pairs, β ∈ {−500, −300, −100}, Δ values). Report them as descriptive, and ideally show how they move under a second grid.
- The baselines are now appropriate: chain, chain-r, graph and hybrid at matched cost.
- The ablations over c, t, κ, φ, ρ_s and distance-dependent correlation are good.

Missing ablations:
- an adaptive, topology-aware Houdini (acknowledged);
- a larger hybrid space: r = 2 only, k ∈ {2, 3}, no anchor;
- a cost model in which anchor (human) checks cost more than model checks. This matters because the anchor is selected in most guard-steeper and houdini-steeper settings.

## 8. Reference spot-check (web search)

| Reference | Result |
|---|---|
| Koran et al. 2026, arXiv:2605.15377 | **Exists.** Authors (Koran, Yun, Tetef, Arnav, Bernabeu-Pérez), 12 GPT-4.1-mini-based monitors, and the diverse-beats-homogeneous finding are all confirmed (also an ICML 2026 listing). Description accurate. |
| Jha 2026, arXiv:2608.16190 | **Exists.** Title and single author confirmed. The skill-not-lineage claim and the null result at matched capability are confirmed. Description accurate. |
| Overman & Bayati 2026, arXiv:2605.28807, ICML 2026 PMLR 306 | **Exists.** PMLR v306 (overman26a) is confirmed. The description (aggregating auxiliary overseers into a calibrated conservatism penalty, conformal guarantees) is accurate. |
| Bhatt et al. 2025, Ctrl-Z, arXiv:2504.10374 | **Exists.** The eight-author list matches exactly. Description accurate. |
| Zanolini 2026, arXiv:2610.29178 | **The paper exists** and the title matches. Its introduction confirms fail-prone/correlated-failure modelling of approval monitors and the liveness analysis. **I could not confirm the author attribution:** arXiv itself was not reachable from my environment, and search snippets showed Zanolini only in the paper's bibliography. Please double-check. |
| Engels et al. 2025, arXiv:2504.18530 (NeurIPS 2025) | **Exists.** The authors and the four success rates are confirmed. The Debate description (Guard/judge Elo rises, Houdini/debater Elo is roughly flat) is confirmed from the paper's own wording. |
| Brown-Cohen, Irving & Piliouras 2025, arXiv:2506.13609 | Corrected author list consistent with round-1 verification. |

No fabricated or misdescribed reference was found. One author attribution is unconfirmed (minor).

## 9. Other issues (minor)

1. Report the non-trivial denominators in §6.3 (Section 2 above).
2. The §8 claim that hybrids were "optimal in about half of the low-correlation calibrated settings" holds at B = 32 (17/36) but not at B = 16 (0/36). Qualify it.
3. In Table 3 the "graph = 1.00" entries for Debate come from the m_H = 0 linear-extrapolation artefact. Flag them in the caption.
4. §6.5's table omits the "chain best" column, so 18 + 53 ≠ 73 (chain is best in 2 settings).
5. §5 and the Reproducibility section state 218 s; on my 4-core machine it took about 9.5 min in total. State the hardware.
6. The phrase "it answers reviewer question 3/5" (§6.3, §6.5) should be removed from a camera-ready paper.
7. Conjecture 3′ states O(q_k), while the evidence suggests O(q_1). Either is consistent, but say which you believe.
8. Cite the consecutive-k-out-of-n and Markov-chain-imbedding reliability literature (Section 5).

## 10. Assessment

**Strengths.**
- Exemplary reproducibility: everything is byte-identical.
- A correct, honestly scoped theorem.
- A revision that took the strongest objection (same-generation redundancy) seriously and let it change the thesis.
- Prominent reporting of negative robustness results (threshold aggregation, false-alarm constraints).
- Clear separation of what is proved, conjectured and simulated.

**Weaknesses.**
- Model-only, with games calibrated to one number each rather than to the available fits.
- Theoretical depth is modest, and close to known reliability results.
- Several headline findings are near-structural consequences of the model.
- A new over-generalised summary claim ("all designs fail below 10⁻³") is contradicted by the paper's own symmetric regime.
- Practical significance is limited: the recommended deep windows do not survive a realistic false-alarm budget with OR aggregation, and the decisive quantities (ρ_s − ρ, γ) are unmeasured.

This is a careful, honest modelling paper that clarifies the NSO chain assumption. At a top main-track venue it is borderline. It would be a strong workshop paper, or an acceptable paper at a venue that values careful model analyses.

## Scores (1–10)

| Criterion | Score |
|---|---|
| Novelty | 5 |
| Rigor | 6 |
| Correctness | 6 |
| Significance | 5 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **6** (borderline) |

**Decision: minor revision.** These items must be fixed:
1. the "all designs fail below 10⁻³" claim in the abstract, §8 and §10;
2. the non-trivial denominators;
3. the reliability-theory citation.

**Path to a higher score:**
1. Plug in the Engels et al. fitted double-ReLU parameters, including the plateaus.
2. Add a topology-aware adaptive adversary, and a cost model in which human anchor checks are expensive.
3. Most importantly, run even a small real-model ladder (4–6 models on a checkable task) to measure ρ, ρ_s and γ and test P1/P4.
