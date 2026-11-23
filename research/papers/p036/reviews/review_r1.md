# Review (Round 1): "Only Unforgeable Bits Count: A Minimax Characterization of When Prompt Injection and Jailbreaks Can Be Solved"

Reviewer role: area chair / senior reviewer. Research question: "Can jailbreaks and prompt injection be fundamentally solved?"

## 1. Summary

The paper models any defended LLM system as a [0,1]-valued policy over a view v = (c, s), where c is attacker-writable content and s is an "unforgeable" signal (out-of-band role ids, authentication, system prompt). It proves three results:

- T1: ASR >= U - TV(P_L, P_A).
- T2: the achievable (U, ASR) set is the Neyman-Pearson region, and likelihood-ratio tests attain it. Closed forms are given for a binary flag and a Gaussian score.
- T3: if the attacker is content-capable (L_c in Q) and the s-kernels do not depend on c, the minimax ASR at utility U equals the NP frontier of (K_L, K_A) alone. A policy that reads only s attains it.

Corollaries: with no unforgeable signal ASR >= U; exact solvability iff K_L and K_A are mutually singular; in-band markers are inert; s-only policies are budget-independent.

The experiments are CPU-scale toy experiments with 2-layer transformers. E1 has 36 models on a task where legitimate and injected content are identical and only a noisy role flag differs; the models land on the binary-flag frontier. E3 compares four systems (UNDEF, INBAND, OOB, GATED) under static attacks and white-box greedy coordinate attacks with k in {1, 2, 4, 8}. The in-band marker defense goes from 0% static ASR to about 100% adaptive ASR on forged delegations. The out-of-band and code-gated systems stay at 0. On "never-legitimate" hijacks (G1), every learned system degrades as budget grows.

## 2. Verification performed

**Code rerun.** I copied `code/` to a scratch directory and reran the whole pipeline: `verify_theory.py`, `exp1_tradeoff.py`, `exp3_adaptive.py`, `forgery_stats.py` and `analyze.py`.

- `verify_theory.py` reproduces `theory_checks.json` exactly: T1 min slack 0.01337, T2 max error 3.16e-14, T3 max error 3.16e-15, 300 instances.
- E1 (36 models, about 17.5 min on this machine) reproduces `e1_tradeoff.json` bit-for-bit. The maximum |ΔU|, |ΔASR| over all 36 models is 0.0. All E1 summary statistics are identical: median |gap| 0.00065, min z −1.79, max gap +0.0805, 0 models below 2 SE.
- E3 (4 systems × 3 seeds) reproduces `e3_adaptive.json` exactly. The maximum difference over all scalar fields is 0, and `e3_forgery_stats.json` is identical.
- Tables 1 and 2, the E1 bullets and the forgery statistics (56/90 = 62%, 0/90, 0/42, 95% structural for OOB) all match the stored `results/summary.json` and `results/e3_forgery_stats.json`.

**Derivations checked line by line.**

- T1 is the standard variational characterization of TV and is correct.
- T2 follows directly from the NP lemma. The size/power inversion and the convexity-by-mixing argument are fine.
- Closed form (a): acting on flag = 1 first gives U = q t and ASR = (1-q) t, so ASR = (1-q)U/q for U <= q. For U > q, ASR = (1-q) + q(U-q)/(1-q). Correct.
- Closed form (b): threshold tau gives U = 1 - Phi(tau - d'), ASR = 1 - Phi(tau), so ASR = 1 - Phi(Phi^{-1}(1-U) + d'). Correct.
- T3 proof (both directions) is correct under the stated assumptions. C2 is correct, including the U < 1 case.
- C4 is trivially true: a max over a larger set is nondecreasing.

**Reference spot-check (web search).** I checked eight references, and all exist and broadly say what is claimed:

- Bhatt et al. 2026, arXiv:2604.06436, Defense Trilemma.
- arXiv:2606.27567, "On the Inseparability of Instructions and Data...". Its authors are Pant, Lohani and Kumar, so the authors could have found the author list.
- Abdelnabi & Bagdasarian 2026, arXiv:2605.17634, a NeurIPS 2026 poster.
- Nasr et al. 2025, arXiv:2510.09023: 12 defenses, most above 90% adaptive ASR. Tramèr is a co-author.
- Costa et al. 2025, FIDES, arXiv:2505.23643.
- Zverev et al. ASIDE, ICLR 2026.
- Hughes et al., Best-of-N, NeurIPS 2025, with power-law scaling.
- The Andriushchenko and Wolf references are standard.

I found no fabricated citation.

## 3. Strengths

1. **Clean, correct formalization.** Splitting the input into attacker-writable content and unforgeable signal, and reducing the mimicry slice to a two-sample test, is crisp. It gives practitioners a useful vocabulary: "list the bits the attacker cannot write". The reporting recommendation (a (U, ASR) curve plus stated trust assumptions) is sensible.
2. **Honest scoping.** The paper separates the mimicry regime (information-theoretic) from the never-legitimate regime (budget-dependent robustness). It flags its toy experiments, its 3 seeds, and the fact that §7 is prediction only.
3. **Reproducible artifacts.** The code is small, readable and deterministic, and the stored outputs match the paper.
4. **E3 is a nice pedagogical demonstration.** A static benchmark rates INBAND and OOB as equally good. A 2-token adaptive search separates them and rediscovers the SIG forgery.

## 4. Weaknesses

### 4.1 Novelty is limited, and closest prior work is missing (major)

- The mathematical content is the Neyman-Pearson lemma, plus the observation that under exact mimicry the content coordinate carries no information. The authors acknowledge this; they claim the reduction as the contribution. But once L_c is in Q and the kernels do not depend on c, T3 is nearly definitional: the likelihood ratio is K_L/K_A by construction. The "minimax" content adds little beyond T2 applied to the s-marginal.
- The paper does not cite "Security–Fidelity Tradeoffs: No Universal Defense Against Prompt Injection" (Hermon et al., ICML 2026 Spotlight; arXiv:2606.30783). That work gives a decision-theoretic framework showing that no universal defense exists when benign and adversarial inputs overlap. This is essentially T1/C1 in spirit, with an empirical benchmark on real models.
- The paper also misses closely related 2026 arguments for deterministic architectural boundaries and provenance (e.g., arXiv:2602.09947; cryptographic provenance tags, arXiv:2508.09288).
- The qualitative thesis, that separation needs an unforgeable or provenance channel and that in-band delimiters are forgeable, is already widely held (CaMeL, FIDES, ASIDE, the 2606.27567 inseparability paper).
- The quantitative extension to noisy provenance (the ROC frontier) is the genuinely new piece, but it is a modest one.

### 4.2 The empirical tests are close to tautological (major)

- **E1 is not a real test of the theory.** Content is identical by construction, so no policy can fall below the frontier except through sampling noise. T1/T2 are mathematical facts about the evaluation distribution. The only empirical question is whether the network saturates one flag value: E[phi | s=0] = 0 or E[phi | s=1] = 1. That is an easy optimization question for a 2-layer transformer on 17 tokens. Presenting "all 36 models land on the predicted frontier" as confirmation overstates its evidential value.
- **OOB/GATED G2 = 0 holds by construction.** The attacker vocabulary excludes role ids. GATED is hard-coded. OOB only needs to learn "act iff role = 2", which it was trained on with q = 1.
- **INBAND G2 → 1.0 is likewise guaranteed.** The legitimate pattern is a 2-token string inside the attacker vocabulary. The static forged-SIG test already gives 1.000.
- None of these experiments could have falsified anything. They illustrate the theorems; they do not test them. The paper's framing ("We test the theory", "confirm") should be toned down accordingly.

### 4.3 The mimicry assumption carries the whole result, and its realism is not examined (major)

- Content-capability (L_c in Q) means the attacker can reproduce the full legitimate content distribution, including context such as thread history, sender names and plausibility relative to the user's task. In real agents, legitimate delegations are correlated with context the attacker may not control. Those contextual features are then either part of s (by the paper's definition) or are forgeable only partially.
- So the paper's sharp dichotomy collapses into the empirical question of how large the effective s really is, and the paper does not address that question.
- The kernel-independence assumption (K does not depend on c) is likewise strong. When it fails, only a lower bound is given.

### 4.4 The jailbreak corollary is overreaching (major)

§4 and the abstract state that, for anonymous dual-use requests, "better refusal training, better classifiers or longer system prompts cannot change" the one-for-one trade-off. Three things break this:

- **The action space is not binary.** The defender chooses how much to answer, and a harmful user who mimics a benign phrasing receives the benign-level answer. "Success" for a jailbreak is the harmful uplift, not compliance with an identical string. The binary act/don't-act model does not capture partial compliance or graded harm, and §8 admits as much.
- **Base rates and harm weights are ignored.** The "one-for-one" statement concerns rates conditional on the scenario, not expected harm.
- **Nothing is tested.** The mapping is "conceptual" (§8), yet the abstract states it as a result.

This should be reframed as a hypothesis.

### 4.5 Statistical reporting (minor to moderate)

- "± s.d." in Table 2 uses `np.std` with ddof = 0 over n = 3 seeds. This understates the sample s.d. by about 18%, and no confidence intervals are given.
- The claim that INBAND is more G1-vulnerable than UNDEF rests on 3 seeds, one task, and a white-box greedy attacker with at most 609 queries. The authors flag it, but it still appears among the key findings.
- The E1 "TV-bound violation" count in `analyze.py` uses an ad hoc tolerance (2.5 × 0.0065) instead of per-point SEs.
- The z-scores in E1 evaluate the frontier at the estimated U. That is fine, but the SE floor `max(p(1-p), 1/n)` is ad hoc.
- The forgery statistics use only the first 5 successes per cell (n = 42–90). This is small, though not obviously biased.
- The G1 interpretation for OOB, that attacks act "through the value computation rather than the trust decision", is asserted without a mechanistic test such as ablation or attention analysis.

### 4.6 No experiment on a real model (moderate)

The paper targets ICML main track yet runs nothing on an LLM. Running even the §7.3 protocol on one small open model with a role-embedding defense (e.g., ISE or ASIDE checkpoints) and a content-only defense would turn P1/P2 from predictions into evidence. P2 in particular (role-label noise ε moves the system to the q = 1 − ε frontier) is cheap to run.

### 4.7 Minor

- The author list for arXiv:2606.27567 is retrievable (Pant, Lohani, Kumar).
- Nasr et al. should list Tramèr among the named authors.
- Theorem 2 needs care with the "+∞ on the singular part" convention when P_A puts mass where P_L has none. This is standard, but it should be stated.
- C3's claim that "learned detectors that read c" are inert holds only under exact mimicry. Say so in the corollary itself, not only in §8.
- U only measures in-context instruction following. Overall utility (CLEAN) is reported but never enters the trade-off; clarify this in the abstract.
- The title and abstract say "when ... can be solved", but the paper answers only the mimicry slice under idealized assumptions. Consider narrowing the title.

## 5. Questions for the authors

1. What does T3 add beyond applying the NP lemma to the s-marginal, given that L_c in Q and c-independent kernels make c uninformative by construction? Is there a non-trivial case, for example partial mimicry with a bounded divergence D(Q || L_c) <= ε, where you can give the frontier? That would be a genuinely new result.
2. How does your framework relate to Hermon et al. (ICML 2026, security–fidelity trade-offs), whose decision-theoretic result appears to overlap with T1/C1?
3. Can you run P2 (role-noise sweep) on a real role-embedding model (ISE or ASIDE) to show real systems land on the q = 1 − ε frontier?
4. For jailbreaks, how would the framework handle graded responses, where the defender chooses answer depth?
5. For the G1 OOB results, can you show by ablation that success comes from corrupting the value computation?

## 6. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 9 |
| **Overall** | **5** |

## 7. Decision

**Weak reject (major revision).**

The paper is correct, clearly written and fully reproducible, and its framing is useful for practitioners. But at a top venue:

- The core theorems are a thin layer on the Neyman-Pearson lemma, and they become nearly definitional under the mimicry assumption that drives everything.
- The experiments cannot falsify the theory: their outcomes are fixed by construction.
- Closely related 2026 decision-theoretic work is missing.
- The jailbreak implications are overstated.

Two changes would most raise the paper's value: a non-trivial partial-mimicry result (bounded attacker divergence), and a real-model test of P1/P2.
