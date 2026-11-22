# Review (Round 1) — p038: "Odds, Not Scores: A Branching-Process Theory Linking Component Replication Evaluations to the Population-Level Risk of Autonomous AI Self-Replication and Self-Improvement"

Reviewer role: independent area-chair-level reviewer. Research question: "AI自主复制、自我改进的风险有多大？"

## 1. Summary

The paper models escaped AI replicas as a Galton–Watson (GW) process. In each epoch a replica makes up to *a* attempts. Each attempt succeeds with probability P. A success gets the parent detected with probability h_s, and a failure with probability h_f. The main claims are:

1. A closed-form mean offspring number. Its limit for unlimited retries is R∞ = Pσ/(P h_s + (1−P) h_f). The mean is monotone in *a*, which gives a "quiet-failure" criterion h_f < P/(1−P).
2. For checkpointed staged pipelines, a bottleneck bound R_ck ≤ σ min_k O_k with stage odds O_k = p_k/((1−p_k)h_k), plus elasticities and a "scores are not sufficient" proposition.
3. A valley-crossing bound for GW processes in a varying environment, ln P_surv ≈ −(ln R₀)²/(2g).
4. Escape amplification by 1/(1−R) under heritable promotion (an Iwasa–Michor–Nowak model).
5. Synchronised sweeps beat uniform patrols at equal mean detection (Jensen inequality plus Smith–Wilkinson).
6. A Clopper–Pearson certification sample size, n ≈ (σ/h) ln(K/δ).

Experiments E1–E8 are exact computations and Monte Carlo runs of the model itself. No real system is measured, and no LLM was run.

## 2. Reproducibility check (performed)

- I copied `code/` to a scratch directory and ran `bash run_all.sh`. It took 54 s on one core.
- All five JSON outputs (`e1_validation`, `e2a_bangbang`, `e2b_keynumbers`, `e3_e4_aggregation`, `e5_e8_dynamics`) were **byte-identical** to those in `results/`. The console log differs only in wall-clock timings.
- I checked every number in Sections 5.1–5.7 and the abstract against the log and JSON. All of them match, including:
  - E1: max |z| = 1.12, mean |z| = 0.49, max mean-gap 8.9e-16.
  - E2: R∞ values, and P = 0.067 with critical h = 0.072.
  - E3: all of Table 1, the counterexamples A–D, 28.5% vs 7.0% supercritical, and 0 bound violations.
  - E4: all of Table 2, and the allocation results (0.087 vs 0.40, ratio 4.6).
  - E5: slope 1.014, intercept 2.05, range 3.4e-65 to 0.33, the three Monte Carlo checks, and the defender-race values 0.105 / 0.040 / 0.0079.
  - E6: per-founder escape probabilities, N₀ = 100 values, the 3.3× linearisation error, and 1.65 SE.
  - E7: Lyapunov exponents and survival.
  - E8: n = 11 … 4,385; h thresholds 0.55 / 0.157 / 0.045; power 0 at n ≤ 300 and 0.98 at n = 1000.
- I hand-checked several values: R_ck(h = 0.01) = 0.782 with elasticity −0.887; n(h = 0.01) = ln 80 / ln 1.01 ≈ 440.4, giving 441; and the n = 10 threshold of 0.5499.

**Verdict: the numbers match the code outputs. I found no fabricated or unsupported numerical results.**

## 3. Proof / derivation check

- **Theorem 1.** Correct. The parent is alive before attempt j with probability r^{j−1}, and linearity of expectation gives the rest.
- **Proposition 2.** The difference m(a+1) − m(a) = r^a[Pσ − (1−δ)(1−r)] is correct, and so is the sign argument. Caveat: "an optimal replicator retries without limit or not at all" is optimal for the *mean*. The paper does not show that survival probability, which depends on the whole offspring law, is also monotone in *a*. Either state this restriction or prove it.
- **Corollary (quiet failure).** Correct.
- **Proposition 3.** Correct and elementary. With π_k = p_k/(p_k + (1−p_k)h_k), the count of completions is geometric. Since Π ≤ π_k and x/(1−x) is increasing, Π/(1−Π) ≤ O_k.
- **Proposition 4.** Re-derived; both elasticity formulas are correct.
- **Proposition 5.** True but nearly tautological: R depends on h, so no function of p alone can determine R. The phrase "R_ck → 0 as min_k h_k → 1 with some p_k → 0" is loosely stated, because the limit is driven by p_k → 0, not by h.
- **Theorem 6.** Upper bound (Markov) correct. Lower bound correct: I re-derived the second-moment recursion Var Z_{n+1}/M_{n+1}² = Var Z_n/M_n² + σ_n²/(m_n M_{n+1}), and Paley–Zygmund plus monotone limits completes it.
- **Scaling corollary.** Heuristic (Laplace) and acceptable as stated. The 24-point regression is a sanity check, not independent confirmation, because the bounds already pin down the exponent.
- **Proposition 7.** The linearisation is correct. However, the text "a population held at R = 0.99 buys 100 times as many tickets ... as one held at R = 0.5" does not match the stated mechanism. The factor 1/(1−R) gives 100 vs 2, a ratio of 50. The ≈100 ratio comes from R/(1−R) (99 vs 1). The exact E6 ratio is about 42. Fix the wording.
- **Proposition 8.** Correct: Jensen on ln(1−x), then Smith–Wilkinson. This is the classical geometric-mean-fitness / environmental-variance effect (Lewontin & Cohen 1969, and the bet-hedging literature), which should be cited.
- **Proposition 9.** Correct. The zero-success Clopper–Pearson bound is 1 − (δ/K)^{1/n}.

No proof errors. But almost all results are one-to-three-line applications of textbook branching-process facts, as the authors admit in Section 2.

## 4. Reference spot-checks (web search)

- **Exist and are consistent with the use in the paper:**
  - [1] RepliBench, arXiv 2504.18565: 20 families / 86 tasks; best model (Claude 3.7 Sonnet) above 50% pass@10 on 15/20 families.
  - [4] METR Rogue Replication Threat Model, Clymer / Wijk / Barnes, 12 Nov 2024. It notes disagreement over compute acquisition and names competition with human-directed AI as a key factor.
  - [11] Palisade Research, arXiv 2605.06760 (May 2026): end-to-end replication of weights and harness onto deliberately vulnerable hosts.
  - [12] Burtsev, "Recursive Criticality of AI Self-Improvement", arXiv 2609.00137. It defines R_AI for R&D feedback loops.
  - [13] "Reachability, Not Observation", arXiv 2609.19720. It treats spawning as GW with per-call approval rate p and threshold 1/b, as the paper says.
  - [15] Jagers 1974, JAP 11(1):174–178.
  - [16] Agresti 1975, JAP 12(1):39–46.
  - [18] Smith & Wilkinson 1969, AMS 40(3):814–827.
  - [19] Iwasa, Michor & Nowak 2003, PRSB 270(1533):2573–2578.
- **Not individually checked** (search budget used up; all are well-known items with plausible metadata): [2], [3], [5]–[10], [14], [17], [20]–[22].
- **Miscitation (minor):** Section 6 cites "ControlArena-style settings [22]". But [22] is the AISI RepliBench blog post, not ControlArena. Add the right reference.

No nonexistent references found.

## 5. Novelty and prior work

The modelling toolkit is classical: GW processes, varying and random environments, and multi-type escape. Modelling self-propagating software with a reproduction number is decades old: the malware-epidemic literature (e.g. Kephart & White 1991; Staniford, Paxson & Weaver 2002), which [13] itself mentions. None of it is cited.

Within AI safety, the closest works are:
- [13] arXiv 2609.19720, which already maps agent spawning to GW with a critical approval rate;
- a companion preprint, "Provisional Reachability" (arXiv 2609.21957), not cited;
- EVOMAL (arXiv 2608.25776), which models the spread of agent-authored skills as GW, not cited;
- "Dive into the Agent Matrix" (arXiv 2509.25302), which measures self-replication *risk* rather than success across 21 models, not cited and directly relevant to the "scores ≠ risk" thesis.

The genuinely new pieces are:
- separating success from per-failure detection, giving the odds form;
- the checkpointed-pipeline bottleneck bound and its link to "inability" safety cases;
- the certification sample-size calculation applied to replication benchmarks.

These are useful framings. Mathematically, they are incremental.

## 6. Strengths

- Clear, honest, carefully hedged writing. Illustrative parameters are labelled; the limitations section is unusually candid.
- Fully reproducible: deterministic outputs, a 1-minute runtime, and outputs byte-identical on rerun.
- One policy-relevant message is sharp and actionable: per-failure detectability *h* is unmeasured and enters R as a ratio with capability. The certification calculation (10 zero-success trials certify only if h ≥ 0.55) is a concrete, checkable critique of pass@10-style inability arguments.
- Exact computations are cross-checked against independent attempt-level Monte Carlo.

## 7. Weaknesses

1. **No empirical grounding; the research question is not answered quantitatively.** The question asks how large the risk is. The paper's answer is "it depends on h, which nobody has measured." That is defensible, but nothing is estimated: not h, not σ, not g, not μ, not the retry structure. The only real-world input is a pass@10 → P conversion that assumes independent attempts, which the authors concede pass@k does not guarantee. The pre-registered protocol (P1–P4) is a proposal; it has not been run or actually registered. For a main-track venue, at least one small empirical instantiation is expected. For example: measure ĥ under a simple monitor in an open sandbox such as [11]'s environment or RepliBench tasks.
2. **Mathematical contribution is thin.** Theorem 1 and Propositions 2–5 and 9 are short exercises. Theorem 6 and Propositions 7–8 are restatements of known results (varying-environment second moments, Iwasa–Michor–Nowak, Smith–Wilkinson / geometric-mean growth). The paper says so itself, but that leaves the contribution resting on framing alone.
3. **Modelling choices that drive the headline results are not stress-tested.**
   - (a) Unlimited retries are compressed into one epoch. This ignores the time cost of ~1/P attempts and defender reaction time. A continuous-time or rate-based model would replace R with a Malthusian growth rate that competes with response latency, and the "quiet failure" criterion could change materially.
   - (b) Detection is independent across replicas. Signature-based detection is common-mode, which would make the process a random-environment one and interact directly with Proposition 8.
   - (c) h is constant across tries. Agents learn within episodes, and failures may become *more* detectable as attempts accumulate (adaptive defenders).
   - (d) No density dependence.

   These are listed as limitations, but none is explored, even numerically, as a robustness check.
4. **The checkpointed model is never validated by simulation.** E1 validates only the single-stage atomic model. All E3/E4 results, including the "~100×" atomic-vs-checkpoint claim, are evaluations of the closed form. An agent-level staged simulation would be cheap.
5. **E3 depends on the sampling distribution.** Table 1 holds for one arbitrary distribution (Beta(0.6, 1.2), common h = 0.05). The conclusions about which aggregate tracks risk ("mean pass@10 is reasonable, mean pass@1 is poor") are not shown to be robust to other priors or to heterogeneous h.
6. **Certification assumes h is known**, yet estimating h is the crux of the paper. A joint (p, h) confidence analysis is needed before the certification claim can guide practice.
7. **Some policy claims overreach the model.**
   - "Reinforcing the hardest stage is 4.6× more effective" ignores costs, which the authors concede.
   - "Synchronised sweeps beat uniform patrols" needs sweep strengths up to 0.97 per epoch, and it inverts if sweeps are imperfect or staggered across providers.
   - "Margin matters" (Proposition 7) is qualitatively right, but μ is unmeasured.

## 8. Questions for the authors

1. Is survival probability (not only the mean) monotone in the attempt budget *a*? If not, does the bang-bang claim survive?
2. How do the conclusions change in a continuous-time version where each attempt takes time τ and defenders respond with latency?
3. Can you validate the checkpointed and staged models with an attempt-level simulation, and report E3 under several sampling distributions and heterogeneous h_k?
4. Can you give even a rough empirical estimate of h for one stage in an existing open sandbox?
5. How does the work relate to [13], arXiv 2609.21957, EVOMAL, "Dive into the Agent Matrix", and the classical worm-epidemic literature?
6. Please fix the "100× tickets" statement (Proposition 7 text) and the [22] / ControlArena citation.

## 9. Scores (1–10)

| Criterion | Score |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 8 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 10. Decision

**Major revision (weak reject at the current stage for a NeurIPS main track).** The work is honest, correct and fully reproducible, and the "measure h" message is valuable. But the mathematics is textbook, the modelling assumptions behind the headline claims are not stress-tested, related GW / epidemic work is missing, and nothing is measured on a real system. As it stands, it is a solid workshop paper. It would become competitive with an empirical estimate of per-failure detectability in an open replication sandbox, plus robustness analyses (continuous time, correlated detection, staged simulation).
