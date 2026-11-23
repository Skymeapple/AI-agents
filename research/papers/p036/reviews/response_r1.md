# Response to Review Round 1 (p036)

We thank the reviewer for a careful review, which included a full rerun of the code. The review is right about the main problems: novelty is thin, the most relevant related work was missing, and the experiments were framed as tests when their outcomes were fixed in advance. We changed the paper's center of gravity accordingly.

- The Neyman–Pearson results (T1–T3) are now presented as a classical framework.
- The new technical contributions are two partial-mimicry theorems (T4, T5) and a graded-response proposition (P6).
- A new experiment, E2, tests something that could have failed: whether learned policies achieve the T5 frontier.

Every new number comes from code run in this environment, with outputs under `results/`. Every new reference was verified by web search.

## Major issues

### M1. Limited novelty (T1 = TV, T2 = NP lemma, T3 nearly definitional)

**We agree.** What changed:

- §4 is retitled "Framework results (classical)". It states explicitly that T3 is close to definitional once kernel independence and exact mimicry are assumed.
- **New Theorem 4 (mimicry-gap stability).** For any family of attack content distributions, let δ be the TV distance from the convex hull of that family to L_c. Every policy then has worst-case ASR ≥ f_K(U) − δ, and the s-only policy attains f_K(U). So content-reading defenses beat the unforgeable channel by at most δ, or √(KL/2) by Pinsker. The bound is tight; §5 gives an explicit tight instance. This answers "how fragile is T3?"
- **New Theorem 5 (exact ε-contamination frontier).** This is the result the reviewer asked for. The attacker's content is (1−ε)A₀ + εR with R free. We give the exact minimax frontier ASR*_ε(U) = min_a [(1−ε)V(U,a) + εa] and the structure of the optimal policy: NP tests on s with content-dependent power, under a per-content acceptance cap. We prove four properties:
  - the endpoints are T2 (static) and T3 (full mimicry);
  - the frontier is concave and nondecreasing in ε;
  - it is bracketed by ε f_K + (1−ε) ASR*_0 ≤ ASR*_ε ≤ f_K;
  - there is a closed form when no unforgeable signal exists.
  We relate it explicitly to Huber's (1965) censored likelihood-ratio test, verified as Ann. Math. Stat. 36(6):1753–1758. We also say plainly that Theorem 5 applies robust-testing ideas and that a special case may exist in that literature.
- **New Proposition 6 (graded responses).** See M5.
- **Verification.** Everything is checked by exact LP (`code/verify_partial.py` → `results/partial_theory_checks.json`):
  - T4: 200 instances, minimum slack −1.1·10⁻¹⁶;
  - T5: 1,000 (instance, ε) pairs, formula computed by an independent greedy algorithm, maximum error 4.2·10⁻⁹, with 0 concavity or bracket violations;
  - P6: 200 instances, maximum error 1.7·10⁻¹⁵.
- The contribution list in §1 now ranks T4, T5 and P6 as the new content.

### M2. Missing closest prior work

**Fixed.** All three works were verified by web search (ICML virtual poster page, arXiv listings):

- **Hermon, Gupta, Ruan, Sabir & Wang (2026)**, "Security–Fidelity Tradeoffs: No Universal Defense Against Prompt Injection", ICML 2026 Spotlight, arXiv:2606.30783. It is now discussed as the closest work, in §1, §2 and §9. We state that their overlap result and our T1/C1 express the same phenomenon, and we credit them. We name the differences: our framework is minimax against adaptive attackers (they evaluate fixed probes and leave adaptive attacks to future work), we separate content from unforgeable signals, and we treat partial overlap exactly (T4/T5). We note that we read their abstract and project page, not the full proofs.
- **Bhattarai & Vu (2026)**, "Trustworthy Agentic AI Requires Deterministic Architectural Boundaries", arXiv:2602.09947. Added in §1–2.
- **Gupta (2025)**, "Can AI Keep a Secret? Contextual Integrity Verification", arXiv:2508.09288, on cryptographic per-token provenance labels. Added in §1–2.

### M3. Experiments cannot falsify the theory

**We agree that the lower bounds cannot be falsified by any correct evaluation, and that E1/E3 were overclaimed.** What changed:

- **New §6.1, "What can and cannot be falsified".** It states that the bounds are mathematical. The empirical questions are achievability (does training reach the frontier?), whether static training fails in the predicted direction, and whether real pipelines satisfy the assumptions. The last question needs real systems, and §8.3 gives the protocol.
- **New E2, now the main experiment.** We train 144 learned policies (36 STATIC, 108 ROBUST) and evaluate them exactly by enumerating all 432 views. The evaluator is therefore an exhaustive worst-case attacker, while the training adversary is only a greedy search. This outcome could have failed: the T5-optimal policy is a capped mixture that gradient training need not find. Results (`results/e2_summary.json`, `results/e2_cap.json`, Fig. 3):
  - ROBUST policies reach the T5 frontier in every cell (median gap ≤ 0.0005, max 0.037).
  - All 33 ROBUST policies with U < q show the predicted censoring structure: per-content acceptance capped at 1 − q, i.e. content can veto but not override the flag.
  - STATIC policies exceed the frontier by a median of 0.058 (ε = 0.25) and 0.191 (ε = 0.5) at q = 0.8.
  - At ε = 0.5, 50% of STATIC policies are worse than ignoring content.
  - All 6 STATIC policies with U < q override the flag on some content.
- **E1 is reframed** as an achievability/sanity check (§7.2). We state that it amounts to saturating one flag value.
- **E3 is reframed as illustration** (§7.3). Each outcome is labeled:
  - GATED G2 = 0: by construction (conceded).
  - INBAND G2 → 1: guaranteed by C1 once the pattern is found (conceded). The only informative parts are that a 2-token search finds it and that static held-out evaluation misses it.
  - **Partial rebuttal on OOB G2 = 0: it is not by construction.** OOB is a learned model that reads every content token, and the attacker can write every structural token. UNDEF has exactly the same unforgeable role channel and is broken at 0.85–0.97 adaptive ASR. Whether a learned model relies on the unforgeable bit is an empirical outcome.
- Throughout, wording changed from "test/confirm the theory" to "achievability check" and "illustration".

### M4. Mimicry and kernel-independence assumptions unexamined; no partial-mimicry result

- **Partial mimicry:** addressed by T4 and T5 (see M1) and by E2.
- **Assumption (K)** is now stated explicitly in §3, with when it holds (labels derived from the arrival channel) and when it fails (trust labels assigned by a content-reading component). The Limitations section says that for content-dependent kernels we give no frontier, only the per-content inequality (step 1 of T4's proof).
- **Real-world estimation** of δ or ε is named as open (§9), and the protocol in §8.3 now includes building ε-contaminated attack sets by mixing verbatim-legitimate and native injections.

### M5. Jailbreak corollary overreaching

**We agree.** What changed:

- The abstract and §4 no longer claim that the one-for-one trade-off "cannot be changed" for jailbreaks.
- **New Proposition 6.** With a response menu where each response r has benign value b(r) and harm h(r), the minimax harm at helpfulness H under exact mimicry with no signal is the lower convex envelope Γ(H) of the menu. This is LP-verified.
  - One-for-one is only the binary special case h = b.
  - Graded or safety-framed responses can lie far below the diagonal.
  - What classifiers that read the request cannot do is beat Γ.
- Base rates are discussed: expected harm per request is π_mal × Harm, a linear rescaling that leaves the frontier unchanged. Harm weights enter through h.
- The jailbreak statement is now prediction **P5**, explicitly labeled an untested hypothesis. Measuring b and h for real domains is named as the obstacle.

### M6. No real-LLM experiment

**Not done. We explain why and did the most we could.** This environment has no GPU and no model API. ISE and ASIDE are multi-billion-parameter checkpoints, and adaptive attacks with 10¹–10⁴ queries per episode against them are not feasible on 4 CPU cores in the available budget. We do not describe any experiment we did not run. Instead:

- §8.3's pre-registered protocol now contains the specific P2 role-noise sweep the reviewer suggested: ε_role ∈ {0, 0.05, 0.1, 0.2} on an ISE or ASIDE checkpoint plus one content-only defense, with the primary outcome and the falsification criterion stated.
- It also contains the new ε-contamination test (P4).
- The abstract and Limitations state plainly that no production LLM was run.

We understand this keeps the paper below the bar for a venue that expects real-model evidence. We did not want to replace that evidence with anything synthetic that might be mistaken for it.

## Minor issues

1. **ddof = 0 s.d., no CIs.** Fixed. `code/stats_r1.py` reports sample s.d. (ddof = 1), a t-interval over seeds and a pooled-episode Wilson 95% interval. Table 2 was split into Table 3 (G2 and G1) in the format "mean ± s.d. [Wilson CI]". Figure 2 was redrawn with ddof = 1 bars and Wilson bands. We note that Wilson intervals ignore between-seed variance.
2. **INBAND more G1-vulnerable than UNDEF as a headline.** Removed from the abstract and from the key findings. It is now stated in §7.3 as "an observation to be tested, not a finding", with per-seed differences (INBAND higher in 3/3 seeds at k ≥ 2, by 0.07–0.60) and its caveats (3 seeds, one task, weak attacker).
3. **Ad hoc tolerance and SE floor in E1.** Replaced with Agresti–Coull standard errors for both U and ASR, using the delta method on the frontier slope. The TV check also uses per-model SEs. Results: minimum z = −1.24 against the frontier and −0.80 against the TV bound, with 0 models below −2. We also report that 12 of 36 models sit at degenerate operating points.
4. **OOB G1 "through the value computation" asserted without ablation.** The claim is removed. §7.3 now reports only that 95% of the stored spans contain a forged structural token, and says that no ablation was run and no mechanistic claim is made.
5. **Author lists.** Pant, D., Lohani, S., & Kumar, A. (2026) for arXiv:2606.27567, verified. Nasr et al. now lists all 14 authors, including Tramèr, verified against the arXiv listing.
6. **C3 qualifier.** C3 now begins "*under exact mimicry*" and states that under partial mimicry content defenses can help, by at most the T4/T5 amounts.
7. **Title overpromises.** The new title is "Only Unforgeable Bits Count: Minimax Limits of Content-Based Prompt-Injection Defenses under Full and Partial Mimicry". The introduction has an explicit "Scope" paragraph.
8. **Forgery statistics use the first 5 successes per cell.** Now stated in §7.3 and in the Limitations. The stored data contain only these spans, so extending the count would require rerunning E3 with full logging, which we did not do.
9. **Theorem 2's +∞ convention** (raised in the review body). It is now stated in the theorem: the singular part of P_L is accepted first at zero ASR cost.
10. **U measures only in-context instruction following** (raised in the review body). §3 now says that overall utility (CLEAN) lies outside the trade-off and is reported separately.

## Files changed or added

- `paper.md`: rewritten abstract (EN/ZH), §1–2, §4 reframed, new §5 (T4, T5, P6), new §6.1–6.2 and §7.1 (E2), revised §7.2–7.3, §8–9, references.
- New code:
  - `code/verify_partial.py`
  - `code/exp2_partial.py`
  - `code/stats_r1.py`
  - `code/analyze_r1.py`
  - `code/cap_analysis.py`
  - `code/README.md` (updated)
- New results:
  - `results/partial_theory_checks.json`
  - `results/e2_partial.json`
  - `results/e2_summary.json`
  - `results/e2_cap.json`
  - `results/stats_r1.json`
  - `results/figs/fig3_partial.png`
  - `results/figs/fig2_adaptive.png` (redrawn)
- Unchanged runs: E1 and E3 raw outputs (`e1_tradeoff.json`, `e3_adaptive.json`) and `theory_checks.json`.
