# Review (Round 1): "Knowing What You Don't Know Costs Bits: A Capacity Theory of Irreducible Hallucination under Penalized Grading"

**Reviewer role:** independent area-chair-level reviewer
**Research question:** 幻觉能否从根本上消除？ (Can hallucination be fundamentally eliminated?)
**Recommendation:** Major revision (overall 5/10, weak reject in current form)

---

## 1. Summary

The paper treats abstention on arbitrary (incompressible) facts as an approximate-membership problem. It makes four claims:

1. **Theorem 1.** A counting lower bound on the description length of any model that answers g of k random facts correctly while accepting at most a keys: B ≥ gH + log C(n,g) − log C(a,g) − log C(k,g) − 1 − t. In the main regime this is about k(H + log 1/ε).
2. **Corollary 1.** A positive false-accept floor, which gives h ≥ (1−1/V)·ε·U_ind on facts whose keys are independent of the model.
3. **Theorem 2.** Within the store-and-filter policy class under +1/0/−λ grading with Zipf queries, the optimum is either refuse-all or has h* > 0 at every finite C.
4. **Proposition 3.** A continuum first-order condition: the optimal ε* is capacity-invariant, so h* ∝ C^−(s−1), and the filter takes 17–77% of capacity.

Experiments: 90 small MLPs on random key-to-value memorization with an abstain class (Exp. A, used for "certified bits per parameter") and 36 MLPs under Zipf queries with penalty-optimal operating-point selection (Exp. B). The paper's answer to the title question is a qualified no. Hallucination on an open-world tail can be suppressed exponentially in filter bits but not removed at finite capacity without an external membership oracle.

## 2. Reproduction

I copied `code/` to a scratch directory and reran `theory.py`, `exp_a_frontier.py` (90 runs), `exp_b_penalty.py` (36 runs) and `analyze.py` on 4 CPU processes (torch 2.15). Total time was about 15 minutes, as the paper states.

- `theory_tables.txt` is **byte-identical** to the submitted file.
- `analysis_tables.txt` is **byte-identical** to the submitted file. Fixed seeds give bit-exact results.
- I checked every number in Tables 1–4, the T5 "price" numbers (0.850/0.840/0.833/0.824 and h = 0.150, 1.6e-4, 1.6e-7, 1.6e-13), the T6 range (0.17–0.77), the T3 slope agreement (max deviation 0.004 ≤ 0.005), the T4 range (−1.03 to −1.09), the certified maxima (1.87/2.19; CP medians 1.19/1.39), the B slopes (−1.25/−1.01/−1.10 vs −1.16/−1.18/−1.19), the knowledge drops (15/12/8 points), the knowledge gap (3.7–11.5 points, quoted as "4–12") and the ε* ratios (1.08–5.74×, quoted as "1.1–5.7×") against the output tables. All match.

**Numbers match the code outputs.** This part of the submission is exemplary.

One internal inconsistency in the text: Contribution 4 (Section 1) says "No network answered even one stored fact correctly at a threshold with zero observed false accepts". Section 6.1 and the abstract say one network did (g = 1, h = 32, k = 4096). The output supports Section 6.1, so Contribution 4 is wrong as written.

## 3. Reference check

I spot-checked these references by web search. All exist and broadly say what is claimed:

- Morris et al. 2025, arXiv:2505.24832, ~3.6 bits/param for GPT-style models. Confirmed. It has since appeared at ICML 2026 (PMLR 306) under the title "How much can language models memorize?", so the citation should be updated.
- Suzuki et al. 2025, arXiv:2502.12187. Confirmed: the probability of hallucination can be made negligible with enough data and good algorithms.
- Banerjee, Agarwal & Singla 2024, arXiv:2409.05746. Confirmed: computability and Gödel-style "structural hallucination".
- Kalavasis, Mehrotra & Velegkas, STOC 2025, arXiv:2411.09642. Confirmed: a consistency-versus-breadth trade-off.
- Dietzfelbinger & Pagh, ICALP 2008, LNCS 5125, 385–396, arXiv:0803.3693. Confirmed, including the retrieval-to-approximate-membership reduction that comes close to the lower bound.
- Kalai, Nachum, Vempala & Zhang 2025, arXiv:2509.04664. Confirmed. Grading that rewards guessing over abstention is the paper's central argument. The confidence-target remedy is consistent with the paper, though my search tools only showed secondary summaries of the exact scoring formula.

From standard knowledge, the bibliographic details of Bloom 1970, Carter et al. 1978, Chow 1970, Pagh–Pagh–Rao 2005, Kandpal et al. 2023, Kleinberg & Mullainathan 2024, Xu et al. 2024 and Wei et al. 2024 also look correct. I found no fabricated or misattributed citations.

## 4. Novelty and missing prior work (serious)

The paper's central framing is that hallucination on random facts is an approximate-membership (Bloom-filter) problem whose zero-false-positive limit needs unbounded memory. That framing has been published already, and the paper does not cite it:

- **Guo & Li, "Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing", arXiv:2602.00906 (Jan 2026), ICML 2026 (PMLR 306).** It models memorization of random facts as membership testing and links Bloom-filter error to LLM log-loss. It proves a rate-distortion theorem that recovers and sharpens Bloom-filter space lower bounds. It concludes that the space-optimal strategy at limited capacity is to give high confidence to some non-facts (hallucinate) rather than abstain, and that zero FPR is compatible only with refuse-all or unbounded memory. That is essentially the paper's Corollary 1 and its "qualified no", reached ten months earlier at a top venue.
- **Wang, Xu & Guo, "The Cost of Compression: A Rate-Distortion Limit on Factual Hallucination", arXiv:2609.12111 (2026).** It gives a B-bit closed-book learner bound that separates compression distortion from missing coverage, with simulations. It is also directly relevant and uncited.

Given these papers, the original part of the submission is limited to:
- the joint retrieval-plus-membership counting bound with partial recall (g < k), used to certify trained networks. This is a modest but useful extension of a classical counting argument.
- the penalized-grading optimization in the store-and-filter class (Theorem 2, Proposition 3): ε* is capacity-invariant, h* ∝ C^−(s−1), and the filter's share of capacity is quantified.
- the small-scale MLP experiments.

The paper must position itself against Guo & Li in particular. As written, the related-work section suggests a novelty that is not there.

## 5. Theory: line-by-line check

**Theorem 1.** The counting argument is correct. Covered datasets are at most C(a,g)·C(n−g,k−g)·V^(k−g). The identity C(n,k)C(k,g) = C(n,g)C(n−g,k−g) is used correctly. There are at most 2^(B+1)−1 descriptions, and the union bound is valid. Inequality (1) follows by setting the probability to 2^−t.
- *Gap (minor):* the theorem fixes (g, a) in advance, but the application uses a = k + ε̂(n−k), which depends on the data. A union bound over the possible (g, a) adds about log2(nk) ≈ 45 bits. This is negligible next to thousands of certified bits, but it should be stated. The maximum over about 43 thresholds is also a selection; it costs about 6 bits and should likewise be stated.
- *Measurement:* a uses the point estimate ε̂. The Clopper–Pearson check deals with this adequately.

**Corollary 1.** The algebra is correct: B ≥ gH + g·log2(ng/(e²ak)) − 1 − t gives ε ≥ (g/(e²k))·2^(H−(B+1+t)/g) − k/n. However:
- The bound is positive only because n → ∞. For realistic B/g the floor is astronomically small; for example, with B/g = 1000 and H = 10 it is about 2^−990. So "strictly positive hallucination" is true but carries no practical weight. In effect it restates that a finite Bloom filter over an infinite universe has nonzero FPR. The paper partly admits this ("suppressible at a quantifiable price"), but the title, abstract and Section 7 lean on "irreducible".
- The extension of (2) to "any B-bit model" in Section 4.2 relies on the notion of "tail facts the model does not encode, whose keys are independent of the model". This is not well defined for an arbitrary model trained on all facts, which is exactly the infinite-data setting of Theorem 2. A model's bits can depend weakly on many keys. A rigorous version needs a mutual-information or counting argument over the tail, for example: I(M; tail keys) ≤ B, so at most O(B) facts can have their acceptance probability materially changed. As written, the step from "any B-bit model" to h > 0 under Zipf queries is heuristic.

**Theorem 2.** The proofs of (a), (b) and (c) are correct; in (c), k = ⌊√C⌋ and f = √C − H is feasible and h* ≤ (1−1/V)(1−J*)/β → 0. But inside 𝒫_C, part (b) is close to a tautology. Every policy has ε = 2^−f with finite f, so h > 0 once k ≥ 1. Calling this "irreducibility" oversells it. The informative content is (c) and the rates.

**Proposition 3.** I re-derived it. Differentiating J(k) = 1 − U − βU·2^(H−C/k) gives m(1+βε) − βUε·ln2·C/k². With km = (s−1)U this gives (4) exactly. The derivation is correct as a continuum first-order condition, and the authors honestly say it has no error bound. Numerically it is strongly confirmed (median log-error 0.010; slopes within 0.004). Note that the capacity invariance of ε* is a property of the **uniform-filter** restriction. A model allowed a non-uniform filter (stronger rejection for higher-probability unstored facts) could do better, and the C^−(s−1) rate might change. The authors acknowledge this in Section 8, but the abstract presents the rate as a property of hallucination itself.

## 6. Experiments: rigor and interpretation

1. **The "no zero-hallucination operating point" observation does not test Corollary 1, and it is presented misleadingly (major).** For the h = 64 networks (about 16k certified bits), Corollary 1 allows zero false accepts among 200k keys at g of order hundreds: the bound at g = 100 is far below 10^−5. I inspected the strictest thresholds in `exp_a.json`. When the network accepts only its single most confident stored key, it still accepts about 10^−4 to 10^−2 of the fresh keys. Hundreds to thousands of random keys score higher than its best memorized fact. That is a failure of the networks' confidence ranking, with float32 saturation at conf = 1.0 visible in the w = 0.03 runs. It is not an information-theoretic floor. Theorem 1 at g = 1 needs only tens of bits. The abstract cites this observation as supporting evidence; it should be reframed as evidence that SGD-trained nets are poor filters, or dropped.
2. **The λ-slope test does not discriminate.** A slope of about −1 between penalty-optimal hallucination and λ arises for many threshold classifiers. By Chow's rule the threshold is set at P(correct) = λ/(1+λ), and the slope depends only on the tail of the wrong-answer confidence distribution. The theory slopes (−1.16 to −1.19) and the NN slopes (−1.01 to −1.25) are close, but the paper gives no alternative model that would predict a different slope. With 2 seeds and hallucination masses around 10^−4, the NN slopes also have wide uncertainty that is not reported.
3. **The distinctive predictions are untested.** The authors say so honestly: capacity invariance of ε* (Proposition 3) is not testable in Exp. B (finite universe), and the exchange-rate test (A2) is confounded. So the paper's most novel claims (P1–P3, capacity invariance, the C^−(s−1) rate) rest only on numerics of the model's own optimization problem.
4. **Statistics.** There are only 2 seeds per configuration, no confidence intervals on any NN quantity, and oracle (w, τ) selection on the evaluation set in Exp. B. The theory comparison uses C = 2.03·P, a *lower bound* on capacity, so "networks are about one bit-budget behind theory" is partly a matter of definition.
5. **Exp. B measures closed-world hallucination only.** All 16,384 facts appear in training, and ε is measured on fresh keys that are never queried. The open-world U_ind term that drives Corollary 1 and the paper's answer is not measured.
6. **Scale and external validity.** The networks are 2.7k–23k-parameter MLPs on 32-bit random keys, with no transformers and no language. The link to LLM hallucination (Section 7, consequence 1, the SimpleQA/Kandpal remarks) is speculative, which the authors note.

## 7. Is the sub-question sharp, and do the conclusions follow?

The reformulation is precise: "can a finite-capacity model reach h = 0 with positive knowledge, and at what rate does h fall in C and λ?" But the yes/no part is answered by an argument that is nearly trivial. With n → ∞, a finite description cannot encode exact membership. The rate part is the interesting contribution, and it is proven only for a restricted policy class and tested only on the model's own numerics. The paper's own finding that h can be pushed to 10^−13 for about 2.6 points of knowledge (T5) arguably supports Suzuki et al.'s "statistically negligible" view more than the "irreducible" framing. The conclusion "cannot be fundamentally eliminated" is literally true in the model but should be stated together with this practical caveat, more prominently in the abstract.

## 8. Strengths

- Fully reproducible, bit-exact. Every number traces to code output. The paper reports its own null and inconclusive results (A2, capacity invariance) and its limitations candidly.
- Theorem 1 with partial recall is correct and gives a neat certified-capacity measurement for trained networks (up to 2.19 bits/param, consistent with Allen-Zhu & Li).
- Proposition 3's derivation is correct and the numerical confirmation is convincing. The quantification of the abstention capacity share (17–77%) and the "bigger models become more knowledgeable at fixed selectivity" insight are crisp and interesting.
- The policy implication (penalized benchmarks measure a different capability and cost accuracy at fixed size) is useful.

## 9. Weaknesses (summary)

- Closely related prior work is missing, in particular Guo & Li (ICML 2026), which already establishes the membership-testing view of hallucination and the zero-FPR impossibility.
- The headline impossibility (Corollary 1, Theorem 2b) is close to a tautology; the meaningful rate results are restricted to uniform filters.
- The step from store-and-filter to arbitrary models in Section 4.2 is heuristic.
- The zero-FP experiment is misinterpreted, the λ-slope test does not discriminate, and the novel predictions are untested empirically.
- 2 seeds, no uncertainty estimates, oracle selection.
- Contribution 4 contradicts Section 6.1.

## 10. Questions for the authors

1. How does your work differ from Guo & Li (arXiv:2602.00906, ICML 2026) and Wang et al. (arXiv:2609.12111)? Which results are new in their light?
2. Can you prove a version of (2) for arbitrary B-bit models under the Zipf query model, for example via I(M; tail) ≤ B?
3. Does ε* stay capacity-invariant when non-uniform filters are allowed (per-fact f_i, possibly a negative list for the head of the unstored tail)? What is the optimal rate then?
4. Can you run Exp. B with a universe much larger than capacity (for example N_U ≈ 10^6 with P ≤ 2×10^4) to test capacity invariance directly, and measure open-world hallucination on unseen-but-queried facts?
5. Please report seed-level variance or bootstrap CIs for the λ slopes, and use held-out selection of (w, τ).
6. Can you show a baseline confidence model that predicts a λ-slope different from −1, so that the slope agreement is a real test?

## 11. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 4 |
| Rigor | 5 |
| Correctness | 7 |
| Significance | 5 |
| Clarity | 8 |
| Reproducibility | 10 |
| **Overall** | **5** |

**Decision: Major revision (weak reject as submitted).**

The derivations are correct and the reproducibility is exemplary. But the central "membership-testing makes hallucination irreducible" thesis was already published at ICML 2026 and is not cited, and the genuinely new rate results lack empirical tests that discriminate between hypotheses. The most valuable change would be to position the work explicitly against Guo & Li and to make the capacity-invariance / C^−(s−1) prediction the empirical centrepiece, testing it in a large-universe Exp. B with multiple seeds and CIs.
