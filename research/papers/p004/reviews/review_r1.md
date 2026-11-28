# Review (Round 1): "Information, Learnability, Verification: Three Ceilings on How Far Synthetic Data Can Stretch the Human Data Stock"

Reviewer role: area chair, independent. Target venue: ICLR 2027 main track.
Research question: 高质量人类数据快用完了，合成数据能否接续？ (High-quality human data is running out. Can synthetic data take over?)

---

## 1. Summary

The paper puts forward a "data multiplier" $M$: the factor by which a synthetic pipeline raises the human-equivalent sample size of a given learner. It separates three ceilings on $M$:

1. **Information ceiling (Prop. 1).** If $\theta\to D_h\to D_s$ is a Markov chain, then $M_{\text{Bayes}}=1$. This is the data processing inequality plus a randomization argument.
2. **Learnability ceiling (Prop. 2).** In a Zipfian Poisson exposure model where each exposure is stored independently with probability $s$, $r$-fold paraphrasing gives exactly $M(r)=(1-(1-s)^{1+r})/s\le 1/s$. The generator-retention variant gives $1+s_g(1-s)/s$.
3. **Verification ceiling (Thm. 2, Props. 3–4).** Verified self-generation adds at most $\sum_j I(C_j;V_j)$ bits. A plurality condition $aq>b\,w_{\max}$ holds for a fixed generator. In the accumulate regime a mean-field urn argument gives error $\asymp t^{-(1-b/a)}$.

The experiments are as follows. E1 checks Prop. 2 by Monte Carlo. E2 runs 144 self-training runs of multinomial logistic regression on a linear-teacher "skill" task and a one-hot random-label "knowledge" task under a grid of $(a,b)$ verifiers. E3 (script `e4_urn_scaling.py`) fits urn exponents. E4 (script `e3_horizon.py`) converts $M$ into years of horizon delay via $\Delta t=\ln M/\ln g_D$. The paper concludes that synthetic data stretches the human stock by a bounded factor. For knowledge that factor is set by learner inefficiency, and for skills by verifier information. $M=10$ buys about 3 years.

## 2. Reproduction

I copied `code/` and `results/` to a scratch directory and reran every script there, so the authors' files were not overwritten. I then compared the new outputs with the shipped ones and with the paper.

- **E1** (`e1_knowledge_amplification.py`, ~6 s). The output is identical to the shipped `e1_knowledge.json` because the seed is fixed. All numbers quoted in §5.1 match:
  - largest closed-form vs. MC gap is 0.7818 vs 0.7836 at $s=0.05, r=0$;
  - $M=19.29$ at $s=0.05, r=64$, and $M=5.00$ at $s=0.2$;
  - invariance value 8.332;
  - plain accuracy 0.818 vs. coverage ceiling 0.923;
  - generator-retention ceilings 1.90, 3.70, 6.40, 10.0;
  - fidelity values 1.76→10.05 ($\phi=0.9$), 1.39→9.74 ($\phi=0.7$), 0.819→0.688 with $M=0.10$ ($\phi=0.5$), and 0.019 ($\phi=0.4$).

  Two small inaccuracies. The paper says the gap is "within 1.4 MC standard deviations"; the run gives 1.34 SD, using the population s.d. of 3 replicates rather than the standard error. Against the standard error it is about 2.3 SE. The paper also gives the N-equivalent relative error as "about $10^{-9}$"; the run gives $\sim2\times10^{-11}$. Neither matters.
- **E3 urn** (`e4_urn_scaling.py`). Identical to the shipped file. Fitted exponents are 1.001, 0.751, 0.498, 0.216, 0.0003 and 0.496, all as reported.
- **E4 horizon** (`e3_horizon.py`). Identical. I also recomputed every row of the §5.4 table by hand ($\ln M/\ln 2$ and $\ln M/\ln\sqrt5$), and all are correct.
- **E2** (`e2_verified_selftraining.py` + `e2_analyze.py` + `e2_spearman.py`, ~18 min on 4 cores). See §2.1.

### 2.1 E2 rerun

The full 144-run grid finished in 2,611 s on this machine. That is about 44 min, not the ~18 min the README states, because of CPU contention.

- The regenerated `e2_selftrain.json` is bit-identical to the shipped one: the largest absolute difference in any per-round accuracy is 0.0.
- `e2_table.txt` and `e2_spearman.txt` are identical to the shipped files.
- Every row of Table 1 matches the table, including acc$_0$, acc$_8$, s.d., precision, kbits, $N_{\text{eff}}$ and $M$.
- The values in Findings 1–5 match: 19.7/9.6/4.6/2.5/1.4/1.07/1.00; 29.1/19.7/9.5; 0.892→0.750 and 0.804; replace vs accumulate 0.898 vs 0.816 and 0.702 vs 0.608; 0.932 vs 0.925 and 0.981 vs 0.949.
- The Spearman values (0.894, $p=3.9\times10^{-9}$; 0.830, $p=7\times10^{-5}$; within-$n_0$ 0.92–0.97) match.
- The job count is 5 budgets × 8 verifiers × 3 seeds (accumulate) plus 2 × 4 × 3 (replace), which is 144, as claimed.

**Conclusion: every number in the paper is backed by code that I reran. I found no mismatches.** The problems with this paper are about interpretation and significance, not fidelity to the code.

## 3. Strengths

1. **The question is framed in a useful way.** Recasting "can synthetic data take over?" as an exchange rate in human-data units is useful to planners. The three-channel decomposition (information / learnability / verification) is clean, and it organizes a scattered literature on collapse, accumulation, verification and rephrasing.
2. **Claims are honest and carefully scoped.** The paper calls Prop. 4 mean-field and Prop. 1 elementary. It discloses that E2 uses logistic regression only, and it lists no real estimates of $s, s_g, a, b$ as a limitation. The horizon table is labelled as arithmetic, not a forecast.
3. **Everything reproduces.** All scripts run on CPU with fixed seeds, and the shipped outputs regenerate.
4. **The E2 findings are mildly interesting.** Unverified self-training *lowers* accuracy under accumulation on the knowledge task, and replace beats accumulate when verifier precision rises over rounds. Both are small qualifications of Gerstgrasser et al. (2024) and Kazdan et al. (2025), stated with the right hedging.
5. **The predictions are falsifiable.** §6.1 gives concrete protocols, P1–P4.

## 4. Weaknesses

### 4.1 Major

**M1. The technical contributions are mostly standard results in new notation, and the paper's "exact identity" is a consequence of its own modelling assumptions.**
- Prop. 1 is the data processing inequality / Blackwell garbling.
- Thm. 2 is the chain rule plus DPI plus the mutual information of a binary asymmetric channel. Since $C_j$ is a deterministic function of $\theta$ given the proposal, the first inequality is actually an equality, and the result is a one-line observation.
- Prop. 3 is the strong law of large numbers applied to accepted counts.
- Prop. 2's "exact amplification identity for every frequency distribution" holds because the model *defines* each paraphrase as an extra independent Bernoulli($s$) storage trial attached to a Poisson exposure. Thinning then makes $M$ a per-exposure constant by construction. The identity says nothing about real learners beyond what was assumed. Its "for every $\alpha$ and $N$" invariance is the assumption restated, not a finding.
- The paper calls these a "theory" contribution, but none of the four results needs a non-trivial argument. For an ICLR main-track theory-plus-experiments paper, the theoretical depth is well below the bar.

**M2. Prop. 4 restates a known rigorous result, and the prior work is not cited.** Self-training accumulated into a pool with acceptance probabilities $a>b$ is a two-colour *randomly reinforced urn* (Muliere, Paganoni & Secchi, 2006; Durham, Flournoy & Li, 1998). For this urn, May & Flournoy (2009, *Ann. Statist.* 37(2):1058–1078) give the almost-sure and distributional asymptotics. In particular, the inferior colour's count grows like $n^{m_2/m_1}$, which gives the wrong share $\asymp n^{-(1-b/a)}$. The paper presents this as a new "scaling law for verified self-training" and says it was derived only in mean-field form, "not a rigorous almost-sure statement". The rigorous version is already in the literature. This must be cited, and the contribution reframed as an application.

**M3. E2 does not test the paper's central claim about knowledge versus skills.** The headline claim (§4.2 corollary (a), §6, §8) is that knowledge is bounded because a low-$b$ verifier for facts is expensive or must itself contain the facts. In E2 the verifier is a free, exact oracle on *both* tasks: it compares to the true label. Under a perfect verifier, the knowledge task actually gets the *largest* multiplier in the paper, $M=35.9$ at $n_0=200$, larger than any skill cell at comparable budgets. The paper admits this (Finding 4), but the result is that the experiment supporting the knowledge/skill asymmetry is the theoretical argument in §4.2(a), which is a verbal claim rather than a theorem. No experiment models verifier *cost* or verifier *coverage*. The skill/knowledge distinction in E2 therefore comes down to "structured vs. unstructured labels", which is a statement about generalization, not about verification.

**M4. The evidence that "gain tracks verifier information" (P4, Finding 2) is confounded by design and does not single out information.** Within each $n_0$, 6 of the 8 verifier settings have $a=1$ with increasing $b$. Bits, precision, $-b/a$ and $a-b$ are therefore all monotone in the same variable, and a high rank correlation follows from the design. I recomputed Spearman correlations of accuracy gain against alternative predictors on the shipped `e2_table.json` (accumulate cells):

| predictor | skill (24 cells) | knowledge (16 cells) |
|---|---|---|
| $\sum I(C;V)$ (paper) | 0.894 | 0.830 |
| $-b/a$ | 0.868 | 0.857 |
| $a-b$ | 0.847 | 0.857 |
| precision | 0.761 | 0.839 |

On the knowledge task the crude $-b/a$ ratio predicts *better* than the information quantity. Only two settings, $(0.7,0.1)$ and $(0.5,0.5)$, break the collinearity. The claim that information is the governing quantity is therefore not supported over simpler alternatives. A design that varies $a$ and $b$ independently at matched information, or matched $b/a$ with different information, is needed.

The p-values ($3.9\times10^{-9}$) also treat cells as independent. They are not: cells share seeds, tasks and learning curves, and the pooled analysis mixes budgets with different gain scales. The $p$-values overstate the evidence.

In addition, the "bits" computed in E2 use each round's *marginal* $q$, not the conditional $q_j$ in Thm. 2. By concavity of $I$ in the input prior, this overstates the theorem's term. It is still a valid upper bound, but it is not the quantity stated.

**M5. The quantitative conclusion about the motivating question rests on unmeasured parameters.** The abstract and conclusion state that "a multiplier of 10 delays the horizon by only about 3 years." That is the identity $\ln 10/\ln 2$. Whether $M$ is 2, 10 or 1000 for real pipelines depends on $s$, $s_g$, $a$ and $b$, and the paper estimates none of them (§7 admits this). The arithmetic is also blind to known real-world effective-token measurements: WRAP reports about 3× (Maini et al.); recent rephrasing work reports multiple-× effective-token gains over repetition; Kang et al. (EMNLP 2025) give scaling-law comparisons of synthetic mixtures. These are not used to anchor $s$ or $M$, even roughly. As written, the answer to the research question is a conditional ("if $M$ is X then Y years"), so the concrete "about 3 years" headline is not earned by the evidence.

**M6. The experiments are on toy learners and toy tasks, and there are no external baselines.**
- E1 checks a closed form against a simulation of the *same* stochastic model. The N-equivalent check inverts the closed form, so it is algebraically guaranteed to agree. Neither checks the model against any learning system.
- E2 uses convex logistic regression with 20 features or 400 one-hot items and 3 seeds. There is no neural network or LM, and no comparison to prior methods (STaR/ReST-style filtering, Feng et al.'s verifier-based selection, Gerstgrasser-style accumulation with real-data mixing ratios).
- The paper is honest that nothing transfers, but at ICLR a paper whose motivating question is about frontier LLM data is expected to include at least one small-LM experiment, for example measuring $s$ from planted-fact recall as in P1.

### 4.2 Minor

- m1. The fidelity paragraph (§4.1) says the threshold is a special case of Prop. 3 with $w_{\max}=1-\phi$. In E1, though, the original human exposure always carries the correct label, so the correspondence is only asymptotic in $r$. At $\phi=0.5$, Prop. 3 is silent (equality), yet the paper calls the observed collapse "the threshold predicted by Prop. 3". Please phrase this more carefully.
- m2. Thm. 2's second inequality writes a sum of a random expression in $q_j$. It should be an expectation over $\mathcal F_j$. Also, "Each proposal adds at most one bit" follows trivially from $V_j$ being binary.
- m3. "Villalobos et al. ... 2026 and 2032": Epoch describes this window as an 80% interval. The 90% interval belongs to the 100T–1000T stock. In the horizon table, 1000T is called the "95th pct"; that is consistent with a 90% CI but should be said explicitly.
- m4. Huang et al. (2025) do not say "cannot create new information" verbatim. They raise this as the standard objection and argue that self-improvement *extracts hidden knowledge*. The quotation marks in §3 should be removed or the wording paraphrased.
- m5. Section and script numbering are swapped: §5.3 "E3" runs `e4_urn_scaling.py`, and §5.4 "E4" runs `e3_horizon.py`. This is confusing.
- m6. The E1 MC standard deviation is the population s.d. over 3 replicates. Report the standard error, or use more replicates.
- m7. In `e2_analyze.py`, `n_equiv` contains the dead expression `float(">%d" % 0 or 0) if False else float("inf")`. It is harmless but sloppy. "inf" cells are omitted from $M$ summaries, which slightly biases any aggregate.
- m8. The urn simulation freezes the pool within blocks of size $n/200$, so it is an approximate simulation. That is fine for exponents, but it should be stated.
- m9. Related work misses: Gan & Liu (ICLR 2025, "reverse-bottleneck", an information-gain view of synthetic data in post-training); Kang et al. (EMNLP 2025, scaling laws for synthetic pre-training data); Allen-Zhu & Li Part 3.1 (knowledge augmentation/paraphrasing is needed for extraction, directly relevant to the learnability channel); and the randomly reinforced urn literature (M2).

## 5. Novelty assessment

- The conceptual framing ("exchange rate", three channels) is a reasonable synthesis but incremental.
- The information-only-through-verification idea is already explicit in Feng et al. (2025), Yi et al. (2025/ICLR 2026, convergence to the verifier's "knowledge centre"), Song et al. (2025, generation–verification gap) and Huang et al. (2025).
- The information-theoretic view of synthetic data appears in Gan & Liu (2025).
- The "effective data" notion is Muennighoff et al. (2023), and recent rephrasing work already reports effective-token multipliers.
- The urn law is classical (May & Flournoy, 2009).

What is new is the specific Poisson-thinning model of paraphrase and the packaging. Overall the contribution is incremental and closer to a position or workshop paper with toy validation.

## 6. Citation spot-check

I checked the following by web search.

| Reference | Exists? | Supports claim? |
|---|---|---|
| Villalobos et al. 2024 (arXiv:2211.04325) | Yes | Yes: ~300T tokens, 90% CI 100T–1000T, 2026–2032 (an 80% interval per Epoch; see m3) |
| Yi, Liu, Cheng, Xu 2025 (arXiv:2510.16657, ICLR 2026) | Yes | Yes: verifier-filtered retraining in linear regression/VAE, near-term gains that plateau |
| Huang et al. 2025 (arXiv:2412.01951, ICLR 2025) | Yes | Mostly; the paraphrase of "cannot create new information" is loose (m4) |
| Song et al. 2025 (arXiv:2412.02674, ICLR 2025) | Yes | Yes: generation–verification gap |
| Kazdan et al. 2025 (arXiv:2410.16713, ICML 2025, PMLR 267) | Yes | Yes: accumulate vs replace; synthetic data helps when real data is scarce |
| Maini et al. 2024 (WRAP, ACL 2024) | Yes | Yes: ~3× faster pre-training |
| Sevilla & Roldán 2024 (Epoch AI blog) | Yes (known) | Yes: 4–5×/yr |
| Feng et al. 2025, Allen-Zhu & Li 2025, Muennighoff et al. 2023, Shumailov et al. 2024 | Yes (well known) | Yes |

No fabricated references. One mis-characterization (m4) and one interval mislabel (m3), both minor.

## 7. Questions for the authors

1. Can you cite and position Prop. 4 against the randomly reinforced urn literature (Muliere et al. 2006; May & Flournoy 2009)? What, if anything, does your version add?
2. Can you design an E2 variant where verifier information and $b/a$ (or precision) are decorrelated, and show that information is the better predictor? The current table shows $-b/a$ predicting as well or better on the knowledge task.
3. Can you add an experiment that models *verifier cost or coverage* for knowledge, for example a verifier that knows only a fraction of facts? Without one, the knowledge/skill asymmetry rests on an untested verbal argument.
4. Can you estimate $s$ for even one small LM, as your protocol P1 proposes? With $\hat s$ in hand, the "3 years" headline would become a measurement.
5. Does Prop. 2 survive a correlated-paraphrase model, for example storage probability decaying with near-duplicate exposures? If $M$ then depends on $\alpha$ and $N$, the "property of the learner, not the data" claim fails.

## 8. Scores

| Criterion | Score (1–10) |
|---|---|
| Novelty | 3 |
| Rigor | 4 |
| Correctness | 7 |
| Significance | 4 |
| Clarity | 7 |
| Reproducibility | 9 |
| **Overall** | **4.5 (weak reject / clear reject boundary)** |

## 9. Decision

**Reject in current form (major revision needed for any resubmission).**

The paper is clearly written, honest and fully reproducible. Its numbers match its code. Its derivations are correct, but they are standard: DPI, binary channel capacity, SLLN, Poisson thinning under an assumption that makes the "exact identity" automatic. The one dynamical result is a known theorem about randomly reinforced urns, presented as new and stated only in mean-field form. The experiments do not test the paper's central knowledge-vs-skill asymmetry, since the verifier is a free oracle in both tasks. The "gain tracks information" evidence does not single out information over simpler predictors. The concrete answer to the motivating question ("about 3 years") is arithmetic on an unmeasured $M$.

The most valuable change would be one small-LM experiment that estimates $s$ (and ideally $a,b$ for a code verifier) and tests Prop. 2 and P1 against real learning curves, together with proper positioning against the urn and information-theoretic synthetic-data literature.
