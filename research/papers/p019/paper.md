# Replication Is Not Validation: Shared-Bias Limits on AI-Verified Scientific Hypotheses

**Target venue:** NeurIPS 2027, main track (theory and empirical methodology for AI-driven scientific discovery)

---

## Abstract

AI systems now generate scientific hypotheses at scale and increasingly "verify" them themselves: by re-analysing the data that suggested the hypothesis, by re-running the analysis on fresh data of the same kind, or by agentic falsification tests over existing databases. We ask a narrow question within the broader question of whether AI can propose and validate original hypotheses: **when can a verification step change how likely an AI-generated hypothesis is to be true, and how many independent lines of evidence does an original claim need?** We model each measurement as true effect plus a modality-specific systematic bias plus sampling noise. We prove three results. (i) *Bias floor*: under fixed-level significance testing, the false discovery rate (FDR) of a generator plus any number of same-modality replications converges, as sample size grows, to π0/(π0 + 2π1F_b(μ)). This limit is at least π0/(π0+2π1), however small the bias. More data therefore *raises* FDR once n passes roughly (z_α σ/σ_b)². (ii) *One-bit ceiling*: in the large-sample limit, an observational modality whose bias is symmetric and independent of the others multiplies the posterior odds by at most a factor of 2. Reaching FDR ≤ q therefore takes at least log2((1−q)π0/(qπ1)) such modalities, and each e-fold drop in prior odds (more original candidates) adds about 1/ln Λ modalities. (iii) *Shared-bias reversal*: if a fixed fraction of false claims carries a bias that every available modality shares, then adding modalities drives FDR toward 1. We test these predictions in 300 synthetic "scientific worlds", which are linear structural causal models with latent confounders and a simulated literature. In them, an observational hypothesis generator proposes 40 original causal claims per world. Self-verification and fresh-sample replication both accept everything, with FDR 0.787 (95% CI 0.779–0.795). Triangulating across four environments with resampled confounding only lowers FDR to 0.698, because 52% of the proposed claims are false through reverse causation or observed common causes, and those biases are shared. A bias-source decomposition calibrated on half of the worlds predicts held-out triangulation FDR to within 0.011. The naive independence model is off by up to 0.107. Two randomized interventions per claim give FDR 0.007. When the generator is pushed deeper for more claims, the prior truth rate falls from 0.203 to 0.102. Theorem 2 then predicts single-intervention FDR of 0.171/0.233/0.360 against an observed 0.166/0.241/0.343, and it correctly predicts that two interventions are needed in every band. The upshot is a concrete accounting rule for AI scientists: count how many sources of bias are independent, not how many replications were run.

## 中文摘要

AI 系统已能大规模生成科学假说，并越来越多地自行"验证"：在产生假说的同一数据上复核、在同类新数据上重复分析，或在现有数据库上进行智能体式证伪检验。本文聚焦"AI 能否提出并验证原创假说"中的一个可处理子问题：**验证步骤在什么条件下才能真正改变 AI 生成假说为真的概率？一个原创性假说需要多少条相互独立的证据？** 我们把每次测量建模为"真实效应 + 模态特有的系统偏差 + 抽样噪声"，并证明三条结果：(i) **偏差下限**：在固定显著性水平检验下，生成器加任意次同模态重复验证的错误发现率（FDR）随样本量增大收敛到 π0/(π0+2π1F_b(μ))；无论偏差多小，该值都不低于 π0/(π0+2π1)，因此当样本量超过约 (z_α σ/σ_b)² 后，更多数据反而提高 FDR。(ii) **一比特上限**：大样本极限下，偏差对称且彼此独立的观测模态，每增加一种至多使后验赔率提升 2 倍；要把 FDR 压到 q 以下，至少需要 log2((1−q)π0/(qπ1)) 种模态，而候选假说越原创（先验赔率越低），所需模态数按 1/ln Λ 的速率对数增长。(iii) **共享偏差反转**：若一部分错误假说的偏差被所有可用模态共享，则不断增加模态会使 FDR 趋于 1。我们在 300 个合成"科学世界"中检验这些预测。这些世界是带潜在混杂和模拟文献的线性结构因果模型，观测式假说生成器在每个世界提出 40 条原创因果假说。自我验证与新样本重复都全部接受，FDR 为 0.787（95% 置信区间 0.779–0.795）。跨四个混杂结构重新抽样的环境做三角验证，FDR 只降到 0.698，原因是 52% 的候选假说因反向因果或可观测共同原因而为假，这些偏差在环境间共享。用一半世界校准的偏差来源分解模型对留出世界的三角验证 FDR 预测误差不超过 0.011，而朴素独立模型的误差最高达 0.107。每条假说做两次随机化干预后，FDR 降到 0.007。让生成器挖得更深、产出更多假说时，先验真实率从 0.203 降到 0.102；定理 2 预测的单次干预 FDR 为 0.171/0.233/0.360，实测为 0.166/0.241/0.343，并正确预测每个区间都需要两次干预。由此得到一条可操作的记账规则：评估 AI 科学家时，应数独立的偏差来源，而不是重复次数。

---

## 1 Introduction

Whether AI can do science, not just help with it, now depends less on generating hypotheses than on validating them. LLM-based systems already produce research ideas that expert reviewers rate as more novel than expert-written ones [Si et al., 2025a]. End-to-end "AI scientist" pipelines generate ideas, run experiments, and write papers [Lu et al., 2024; Yamada et al., 2025]. Multi-agent "co-scientists" generate and rank biomedical hypotheses, some of which were later tested in wet labs [Gottweis et al., 2025]. Data-driven discovery agents run many analysis cycles over a dataset and report statements that domain experts then grade [Mitchener et al., 2025]. Other evidence points the other way. LLM ideas that look novel lose more ground than human ideas once they are actually executed [Si et al., 2025b]. Data-driven discovery benchmarks show low success rates [Majumder et al., 2025]. LLMs cannot reliably correct their own reasoning without external feedback [Huang et al., 2024]. Position papers argue that agentic analysis multiplies plausible explanations faster than it can test them [Fa and Culjak, 2026].

The most rigorous recent response is statistical. POPPER [Huang et al., 2025] has agents design falsification tests for a hypothesis's implications and combines them with e-values, which controls Type-I error *given that each test is valid*. That condition is the gap we study. Type-I error control assumes that a test's null distribution is correct. In observational science, the main threat is usually *systematic bias* instead: confounding, reverse causation, selection, or measurement artefacts. These make a null hypothesis look non-null in every dataset collected the same way. Meta-science has long warned that replication alone cannot remove such biases and called for *triangulation* across methods with different biases [Munafò and Davey Smith, 2018]. Meta-science also explains low replication rates through low prior odds and bias [Ioannidis, 2005; Open Science Collaboration, 2015]. As far as we know, nobody has turned these warnings into quantitative rules for *autonomous* hypothesis generators. Those generators differ from human science in three ways that matter. They can search very large hypothesis spaces, so prior odds are low. They are rewarded for originality, so they move further from what is already established. And they usually verify with the same kind of data that suggested the claim.

**Question.** We study one tractable piece of "Can AI propose and validate original hypotheses?": *Which verification procedures available to an AI scientist change the posterior probability that its original claims are true, by how much, and how many independent lines of evidence are required as originality grows?*

**Contributions.**

1. **A measurement model for AI verification with shared bias** (§3). Each verification is a measurement in a *modality* carrying hypothesis-specific systematic bias. Self-verification, fresh-sample replication, triangulation across environments, and randomized intervention become special cases that differ only in how their bias is correlated with the generator's.
2. **Three proved results** (§4). Theorem 1 gives the bias floor for same-modality replication, along with the non-monotone FDR curve it implies and the crossover sample size. Theorem 2 gives the "one-bit ceiling" and the logarithmic K* rule linking originality (prior odds) to the number of independent modalities needed. Proposition 3 shows that adding modalities which share a bias component eventually *increases* FDR.
3. **A synthetic-world testbed with ground truth** (§5). The testbed consists of linear-Gaussian structural causal models with latent confounders, a simulated literature, an observational hypothesis generator restricted to *original* claims, and five verifiers. It is cheap (about 5 CPU-minutes) and fully reproducible.
4. **Pre-registered-style predictive checks** (§6). Rates calibrated on half of the worlds predict held-out FDR for triangulation and for intervention-based verification. A bias-source decomposition predicts triangulation FDR to within 0.011, while the naive independent-bias model errs by up to 0.107. The K* rule correctly predicts how many interventions each originality band needs.
5. **An accounting rule and a test protocol for real AI-scientist systems** (§7). We propose that claims be reported with their *bias-source ledger*, and we give a concrete protocol for testing our predictions on deployed systems. We did not run that protocol.

We do not claim that AI cannot validate hypotheses. With randomized interventions, the simulated AI scientist validates original claims at FDR 0.007 (§6.1). Our claim is narrower: the *kind* of verification determines whether validation happens at all, and replications cannot substitute for independent bias sources.

## 2 Related Work

**Autonomous hypothesis generation.** Robot Scientist Adam generated functional-genomics hypotheses in yeast and tested them with laboratory automation [King et al., 2009]. Its verifier was a physical experiment. Recent LLM systems span idea generation [Si et al., 2025a], end-to-end ML research [Lu et al., 2024; Yamada et al., 2025], biomedical hypothesis generation with external wet-lab follow-up [Gottweis et al., 2025], and data-driven discovery over supplied datasets [Mitchener et al., 2025]. Wang et al. [2023] review AI across the scientific process. Benchmarks such as DiscoveryBench [Majumder et al., 2025] measure whether agents recover known data-driven findings. The ideation–execution study of Si et al. [2025b] shows that judgements of novelty and promise made before execution overestimate how execution turns out. Our model offers one mechanism for this gap: evaluating claims without independent evidence.

**Automated validation.** POPPER [Huang et al., 2025] combines agent-designed falsification experiments using e-values with Type-I error control under valid tests. Fa and Culjak [2026] argue that agentic science should adopt a falsification-first standard and that only experiments, controls and replications narrow the space of explanations. We complement both with a quantitative account of *which* replications narrow it. In our model, replications with shared bias narrow nothing once samples are large. LLM self-correction without external feedback is unreliable [Huang et al., 2024], which is a qualitative analogue of our self-verification result.

**Meta-science and multiple testing.** Ioannidis [2005] modelled the positive predictive value of findings as a function of prior odds, power and bias. The replication project of the Open Science Collaboration [2015] measured low replication rates empirically. Gelman and Loken [2014] described the "garden of forking paths". Munafò and Davey Smith [2018] argued for triangulation over replication. Our Theorem 1 sharpens the bias part of the Ioannidis model into an asymptotic floor that does not depend on the bias magnitude. Theorem 2 turns triangulation into a counting rule, and Proposition 3 identifies when triangulation backfires. FDR control [Benjamini and Hochberg, 1995] and its online versions [Javanmard and Montanari, 2018; Ramdas et al., 2017] are the natural tools for streams of AI-generated hypotheses. We use BH, and we note that online procedures are a deployment choice for future work. All of these procedures inherit the validity of the per-test p-values that biased modalities violate.

**Causal discovery with interventions and environments.** Causal effects are defined by interventions [Pearl, 2009]. Eberhardt et al. [2005] bound the number of single-variable experiments needed to identify causal structure. Invariant causal prediction exploits heterogeneity across environments [Peters et al., 2016]. Our "environments" verifier is a deliberately simple version of this idea, used to measure how much of an AI generator's error such variation removes.

## 3 Model

**Hypotheses and truth.** A generator proposes hypotheses h. Each has a true standardized effect θ_h ∈ {0, μ}, with μ > 0, and is true iff θ_h = μ. Among the *proposed* candidates the fraction of true hypotheses is π1, with π0 = 1 − π1. We call π1/π0 the *prior odds*. Originality enters only through π1: candidates further from established knowledge, or found by deeper search, have lower prior odds (§6.3 measures this).

**Modalities.** A *modality* m is a way of collecting evidence, such as observational data from one population, observational data from another population, or a randomized experiment. A dataset of n units from modality m gives an estimate

  X_{h,m} = θ_h + b_{h,m} + ε,  ε ~ N(0, σ²/n),

where b_{h,m} is the *systematic bias* of modality m for hypothesis h. Systematic bias is fixed for a given (h, m), so it repeats in every dataset from that modality, while ε is fresh in every dataset. We take b_{h,m} to have a continuous distribution F_b, symmetric about 0. In the Gaussian case F_b = N(0, σ_b²). Randomized interventions have b ≡ 0.

**Tests.** A verifier accepts if a one-sided level-α test rejects θ + b ≤ 0, i.e. if X > z_α σ/√n. The generator itself uses this test on its own modality-A dataset, so being proposed implies having passed the generator's test.

**Verification pipelines.** All of them accept only claims that both the generator and every verifier pass:

- *Self-verification*: re-test on the generator's own data. This adds nothing, because the verifier's estimate is identical to the generator's.
- *Same-modality replication* (K fresh datasets): each has new ε but the same b_{h,A}.
- *Orthogonal triangulation* (K modalities): new ε and new, independent biases b_{h,1},…,b_{h,K}.
- *Intervention* (K randomized experiments): b = 0.

The quantity of interest is the FDR among accepted claims, FDR = π0 A0 / (π0 A0 + π1 A1), where A_θ is the acceptance probability given θ. We also track power A1. The noise-only correlation between generator and replicator errors is ρ_n = σ_b² / (σ_b² + σ²/n), which tends to 1 as n grows. This one fact drives most of what follows: a replication becomes uninformative about bias at exactly the rate it becomes informative about noise.

## 4 Theory

Throughout, g_n(θ, b) = Φ̄(z_α − (θ + b)√n/σ) denotes the probability that one dataset passes, given θ and bias b.

**Theorem 1 (bias floor for same-modality replication).** Under the model of §3 with α, π1, μ and F_b fixed, the generator followed by K ≥ 0 same-modality replications satisfies, as n → ∞,

 A0 → 1/2,  A1 → F_b(μ),  FDR → FDR_∞^rep = π0 / (π0 + 2π1F_b(μ)) ≥ π0 / (π0 + 2π1).

The same limit holds for self-verification. With K independent-bias modalities in addition to the generator, FDR → π0 2^{−(K+1)} / (π0 2^{−(K+1)} + π1 F_b(μ)^{K+1}). With K randomized interventions, FDR → π0 α^K / (π0 α^K + 2π1 F_b(μ)).

*Proof.* Given b, the K+1 same-modality datasets have independent noise, so A_θ = E_b[g_n(θ, b)^{K+1}]. For θ + b ≠ 0, g_n(θ, b) → 1{θ + b > 0}, and the indicator is unchanged by being raised to the power K+1. Since F_b is continuous, P(θ + b = 0) = 0, so bounded convergence gives A_θ → P(b > −θ). This equals 1/2 for θ = 0 by symmetry and F_b(μ) for θ = μ. For independent modalities, A_θ factorizes into K+1 identical terms with the same limits. For an intervention, b = 0 and the pass probability under θ = 0 equals α for every n, while under θ = μ it tends to 1. The generator factor contributes 1/2 or F_b(μ). The lower bound uses F_b(μ) ≤ 1. ∎

Three consequences matter for AI scientists.

*(a) The floor does not depend on the size of the bias.* An arbitrarily small σ_b > 0 still produces the floor. Only the sample size at which the floor is reached changes. In the Gaussian case, g_n(0, b) moves from about α to about 1{b>0} once |b|√n/σ exceeds z_α, so the null pass rate starts to climb toward 1/2 around **n× ≈ (z_α σ/σ_b)²**. A generator that mines large observational datasets therefore operates *beyond* n× for most biases, and verification on more of the same data cannot help.

*(b) FDR is non-monotone in n.* At small n, replication filters out noise-driven false positives and FDR falls. Past n×, the shared bias takes over and FDR rises toward the floor. Figure 1 shows this.

*(c) Self-verification and replication become identical in the limit.* Both reach FDR_∞^rep, so a pipeline that "replicates" on fresh data of the same kind cannot be distinguished, asymptotically, from one that does not verify at all.

**Theorem 2 (K* rule; one-bit ceiling).** Suppose each of K verification modalities passes false candidates independently with probability f and true candidates independently with probability p > f, conditionally on truth, and that acceptance requires all K to pass. Write Λ = p/f. Then FDR ≤ q if and only if

 K ≥ K* = ⌈ ln((1 − q)π0 / (q π1)) / ln Λ ⌉.

In the large-n limit with symmetric bias, an observational modality has f = 1/2 and p = F_b(μ) ≤ 1, so **Λ ≤ 2**. Each such modality therefore contributes at most one bit of evidence, and K* ≥ log2((1 − q)π0/(q π1)). A randomized intervention has Λ = power/α, for example ≈ 19 at α = 0.05 and power 0.95.

*Proof.* FDR = π0 f^K / (π0 f^K + π1 p^K) ≤ q ⇔ (p/f)^K ≥ (1 − q)π0/(qπ1). Taking logarithms gives the first claim. The ceiling follows from Theorem 1. ∎

*Originality corollary.* Since K* is linear in −ln(π1/π0), every e-fold drop in prior odds costs 1/ln Λ additional independent modalities. That is about 1.44 bias-limited observational modalities, or about 0.34 interventions. Take prior odds 1/4 (π1 = 0.2) and q = 0.1. Then at least log2 36 ≈ 5.2, i.e. **6, observational modalities with independent biases** are needed, against 2 interventions. Figure 2 plots K* against prior odds for several signal-to-bias ratios μ/σ_b.

**Proposition 3 (shared-bias reversal).** Split the false candidates into a class S, of mass π_S, and a class I, of mass π_I. Suppose class S passes *all* available modalities jointly with probability f_S > 0 because its bias is shared by all of them. Class I passes each modality independently with probability f_I, and true candidates pass each independently with probability p < 1. Then

 FDR_K = (π_S f_S + π_I f_I^K) / (π_S f_S + π_I f_I^K + π1 p^K) → 1 as K → ∞.

*Proof.* The numerator is at least π_S f_S > 0, while π1 p^K → 0. ∎

Proposition 3 formalizes a practical trap. "Triangulating" across modalities that share a bias source does not merely stop helping. Because each extra modality costs some power on true claims while leaving the shared false claims untouched, it eventually makes the accepted set worse. In causal discovery, *reverse causation* and *observed common causes* are bias sources shared by every observational modality, whatever the population (§6.2).

## 5 Experimental Setup

All numbers in §6 come from code in `code/` that we ran in this environment (CPU only). The outputs are stored in `results/`. Section 5.1 checks the theory directly, and §5.2 builds a structurally different world in which the theorems' assumptions hold only approximately.

### 5.1 E1: analytic model versus Monte Carlo

We use π1 = 0.2, μ = 0.2, σ = 1, σ_b = 0.1 (so μ/σ_b = 2) and α = 0.05, with n from 10 to 10⁶. Analytic acceptance probabilities are computed by quadrature over b on a 200,001-point grid. We first tried Gauss–Hermite quadrature, but it was inaccurate at large n, where the integrand becomes step-like. Monte Carlo uses 200,000 hypotheses per point (`exp1_theory_check.py`).

### 5.2 E2–E5: synthetic scientific worlds

**World.** Each world is a linear-Gaussian structural causal model over D = 30 observed variables. Its DAG has edge probability 0.08 under a random order, with edge weights ±U(0.3, 1.0). There are L = 6 latent confounders, each loading on 6 random variables with weights N(0, 0.8²). The ground truth for "X_i → X_j with sign s" is that the total causal effect T_ij = [(I − B)⁻¹ − I]_ij is non-zero and has sign s.

**Literature and originality.** The simulated literature contains the 40% of true causal pairs with the largest |T_ij|, standing in for the strong effects human science found first. A candidate is *original* iff neither (i, j) nor (j, i) is in the literature. *Distance from the literature* is the shortest-path distance between i and j in the undirected literature graph.

**Generator.** On an observational dataset (n = 5000), the generator ranks original ordered pairs by the |t|-statistic of the simple regression of X_j on X_i and proposes the top M = 40, with the sign of the slope as the predicted direction. The generator is intentionally simple, with no LLM. It stands in for any proposer that mines associations, and it shares the defining weakness of observational mining: it cannot tell causation from confounding or reverse causation. The `self` flag records whether a proposal is significant on its own data.

**Verifiers.** All use one-sided t-tests at α = 0.05 in the proposed direction:
- *replicate*: a fresh observational sample from the same environment, so the confounding is unchanged;
- *env-K*: K fresh observational samples from K environments whose latent-confounder weights are resampled, while the causal mechanism B stays fixed;
- *intervene*: a randomized experiment that assigns X_i exogenously (incoming edges cut) in n_int = 300 units. Three independent experiments are stored per claim;
- *BH*: Benjamini–Hochberg at q = 0.1 over the interventional p-values of each world's proposals.

**Runs.** The main run uses 300 worlds (12,000 candidates). E3 calibrates on worlds 1–150 and predicts on worlds 151–300. E4 is a sample-size sweep with n ∈ {100, …, 30,000} on 150 worlds per n, run in two regimes: as above (noiseless) and with added measurement noise of SD 4 on every recorded variable (noisy). E5 is a depth-of-search run on 150 worlds with M = 200 proposals per world. Uncertainty is a 95% world-level bootstrap (2,000 resamples). The full pipeline takes about 5 minutes on 4 CPU cores.

## 6 Results

### 6.1 E1: the theory and its non-monotone FDR curve

The analytic curves match Monte Carlo at every n tested. The largest absolute difference is 0.007, at n = 30 for replication. Values are in `results/exp1_theory.json`.

**Table 1.** E1, π1 = 0.2, μ/σ_b = 2, α = 0.05. Analytic FDR; the Theorem 1 limit is in the last column.

| pipeline | min FDR (at n) | FDR at n = 10⁶ | Theorem 1 limit |
|---|---|---|---|
| self-verification | 0.450 (n = 100) | 0.669 | 0.672 |
| generator + 2 same-modality replications | 0.144 (n = 39) | 0.668 | 0.672 |
| generator + 2 independent-bias modalities | 0.033 (n = 100) | 0.341 | 0.349 |
| generator + 1 randomized intervention | 0.049 (n = 251) | 0.092 | 0.093 |

Replication behaves as Theorem 1 predicts (Figure 1). Its FDR falls to 0.144 at n ≈ 40 and then rises steadily to 0.668, far above the nominal α and almost back to the self-verification level (0.669). The crossover sample size (z_α σ/σ_b)² ≈ 271 lies on the rising limb, as consequence (a) predicts. Two independent-bias observational modalities help but stay bias-limited at the floor of 0.349. A single intervention ends near π0α/(π0α + 2π1F_b(μ)) = 0.093. Here the residual FDR comes from the level α itself, which BH or a second intervention can address (§6.3), and not from bias.

### 6.2 E2/E3: in synthetic worlds, triangulation across environments barely helps, and we can predict why

**Table 2.** E2, 300 worlds, 40 original proposals per world, π1 = 0.213 among proposals. FDR with 95% world-bootstrap CI; discoveries and true discoveries per world. Source: `results/exp2_summary.txt`.

| verification pipeline | FDR [95% CI] | accepted / world | true / world |
|---|---|---|---|
| self-verification | 0.787 [0.779, 0.795] | 40.00 | 8.50 |
| fresh replication, same environment | 0.787 [0.780, 0.795] | 40.00 | 8.50 |
| triangulate across 1 environment | 0.744 [0.736, 0.753] | 32.34 | 8.26 |
| triangulate across 2 environments | 0.720 [0.711, 0.729] | 28.63 | 8.02 |
| triangulate across 4 environments | 0.698 [0.688, 0.708] | 25.25 | 7.64 |
| 1 intervention, α = 0.05 | 0.153 [0.141, 0.167] | 9.39 | 7.95 |
| 2 interventions, α = 0.05 each | **0.007** [0.004, 0.011] | 7.72 | 7.66 |
| 1 intervention + BH (q = 0.1) per world | 0.070 [0.060, 0.081] | 8.38 | 7.79 |
| 2-environment screen → intervention + BH | 0.070 [0.060, 0.080] | 8.03 | 7.46 |

Self-verification and replication accept every proposal; on the generator's own data and on fresh data of the same kind, 100% of proposals are significant. Four environments with resampled confounding remove only about 9 FDR points. Interventions remove almost all false claims while keeping 90% of the true discoveries (7.66 of 8.50 per world with two interventions). The screened pipeline (last row) reaches the same FDR as BH alone with 28.6 instead of 40 interventions per world, i.e. 28% fewer experiments, at a cost of 0.33 true discoveries per world.

**Anatomy of false claims.** Table 3 classifies false proposals by the source of their bias.

**Table 3.** False-claim anatomy (12,000 proposals). Pass rates for K = 1 and K = 4 environments, replication, and one intervention.

| class | share of proposals | env K=1 | env K=4 | replicate | 1 intervention |
|---|---|---|---|---|---|
| true | 0.213 | 0.972 | 0.898 | 1.000 | 0.935 |
| reverse causation / wrong sign | 0.215 | 0.964 | 0.890 | 1.000 | 0.045 |
| observed common cause | 0.304 | 0.919 | 0.784 | 1.000 | 0.048 |
| latent confounding only | 0.224 | 0.421 | 0.033 | 1.000 | 0.041 |
| other (indirect) | 0.044 | 0.460 | 0.057 | 1.000 | 0.056 |

Environment variation works as intended on what it varies. Claims biased only by latent confounding pass all four environments 3.3% of the time, close to the Theorem 2 prediction for independent biases (0.421⁴ = 0.031). However, 52% of proposals are false through reverse causation or observed common causes, and these biases are *invariant* across environments because the causal mechanism is unchanged. They pass four environments almost as often as true claims do (0.890 and 0.784 against 0.898). This is Proposition 3's shared class in natural form. Interventions remove all classes at about the nominal α.

**E3: predicting held-out triangulation FDR.** From the calibration worlds we estimate single-environment pass rates and use them to predict FDR for K = 1…4 on held-out worlds:

**Table 4.** E3. Calibration: π1 = 0.212, p = 0.972, pooled false pass f = 0.766; shared class π_S = 0.519 with f_S = 0.939; independent class π_I = 0.268 with f_I = 0.432.

| K | held-out observed [95% CI] | naive independent (pooled f) | decomposition, shared class perfectly correlated | decomposition, class-specific rates |
|---|---|---|---|---|
| 1 | 0.744 [0.732, 0.755] | 0.745 | 0.745 | 0.745 |
| 2 | 0.717 [0.704, 0.731] | 0.697 | 0.728 | 0.717 |
| 3 | 0.705 [0.692, 0.718] | 0.645 | 0.723 | 0.698 |
| 4 | 0.696 [0.683, 0.710] | 0.589 | 0.724 | 0.685 |

The naive model treats every false claim as having an independent bias and over-promises at K = 4 by 0.107, more than ten times the CI half-width. Splitting false claims by bias source fixes most of this error. The class-specific-rates variant stays within 0.011 of the observed value at every K and inside the held-out 95% CI at every K (it sits 0.007 and 0.011 below the point estimate at K = 3 and 4). The perfectly-correlated variant errs in the opposite direction, by up to 0.028, and brackets the truth from above. That direction is expected: shared-class claims are not *perfectly* shared, because their pass rates still decay through sampling noise and variation in effect size. The point for practice is that **predicting how much verification helps requires knowing which bias sources the verifier randomizes**. A single pooled "false-positive rate" for a verifier is not enough.

### 6.3 E5 and the novelty analysis: originality lowers prior odds, and the K* rule tracks it

**Originality through depth of search.** When the generator must produce 200 original claims per world instead of 40, the extra claims have lower prior odds (Table 5). Using π1, power and false-pass rate from the calibration worlds of each band, Theorem 2 predicts FDR for K = 1, 2, 3 interventions on held-out worlds:

**Table 5.** E5, 150 worlds × 200 proposals; held-out half shown (calibration estimates in parentheses).

| proposal rank band | π1 (cal) | power / false-pass (cal) | Λ | predicted FDR K=1 / 2 / 3 | observed FDR K=1 / 2 / 3 | K* predicted | smallest K with observed FDR ≤ 0.1 |
|---|---|---|---|---|---|---|---|
| 1–40 | 0.203 | 0.941 / 0.049 | 19.1 | 0.171 / 0.011 / 0.001 | 0.166 / 0.007 / 0.002 | 2 | 2 |
| 41–100 | 0.142 | 0.901 / 0.045 | 19.9 | 0.233 / 0.015 / 0.001 | 0.241 / 0.013 / 0.000 | 2 | 2 |
| 101–200 | 0.102 | 0.786 / 0.050 | 15.7 | 0.360 / 0.035 / 0.002 | 0.343 / 0.032 / 0.000 | 2 | 2 |

Cutting the prior odds of the candidates by a factor of about 2.2 roughly doubles the single-experiment FDR (0.166 → 0.343). The quantitative predictions are within 0.017 at K = 1 and 0.004 at K = 2. Because one intervention carries Λ ≈ 16–20, the extra ln-odds cost of the deeper bands (ln(0.254/0.114) ≈ 0.8) is smaller than one intervention, so K* stays at 2. In this bias-limited observational regime, Theorem 2 instead predicts that the same originality increase would require about 1.2 additional independent-bias observational modalities (0.8/ln 2), on top of a baseline of at least 6. We did not observe this second prediction directly, because the environments verifier here does not have independent biases for the majority classes (§6.2).

**Originality as distance from the literature.** Binning proposals by their distance from the literature graph gives a result we did not expect. The prior truth rate is nearly flat (0.197, 0.217, 0.221 for distance 2, 3, ≥4), so in this world closeness to the literature does not raise prior odds. The *composition* of errors does change, though. Among false claims close to the literature (distance 2), 96.7% are false through shared biases (reverse causation or an observed common cause), compared with 73.9% at distance 3 and 48.0% at distance ≥4 (`results/novelty_bias_composition.txt`). Accordingly, four-environment triangulation barely helps for near claims (FDR 0.795 [0.781, 0.810] against 0.803 for self-verification) but helps more for far claims (0.605 [0.595, 0.616] against 0.779). One reading is that "incremental" claims close to known causal structure are exactly the ones whose errors come from that known structure, through common ancestors and reverse paths, and those errors survive any change of population. Interventional verification is insensitive to this: with two interventions, FDR ≤ 0.014 in every band.

### 6.4 E4: sample-size sweeps in synthetic worlds

The theorems treat all of a modality's systematic error as one Gaussian bias. In the SEM worlds the biases are large relative to noise. In the noiseless regime, every top-40 proposal is significant on its own data already at n = 100, and replication accepts 97–100% of proposals for all n (FDR 0.808 at n = 100 and 0.795 at n = 30,000). The worlds therefore sit beyond n× at every n we tried, which is consequence (a) of Theorem 1, and from n = 300 on replication is exactly as uninformative as self-verification (identical FDR). In the noisy regime (measurement SD 4), replication FDR falls from 0.897 at n = 100 to 0.853 at n = 3,000 and then *plateaus* at the self-verification value (0.854–0.856 for n ≥ 10,000). Meanwhile its acceptance rate climbs from 5.3 to 40 claims per world, so verification reduces to rubber-stamping exactly as Theorem 1 predicts. We did not observe the *rise* in FDR at large n (consequence (b)) in the SEM worlds. Their mixture of bias magnitudes and much lower π1 among candidates at small n (0.06–0.14 noisy) give a different trajectory from the stylized model, and we report this as a non-replication of the shape of that prediction. Measurement noise also exposes a limit of the intervention arm. With n_int = 300 and measurement SD 4, interventional power falls to about 0.10 and FDR to 0.72–0.90, because the verifier is underpowered. Theorem 2 accounts for this through a small Λ, and it shows that interventions are not automatically decisive.

## 7 Discussion

**What this says about "Can AI validate original hypotheses?"** In our model, the answer depends almost entirely on the evidence channel the AI has access to, not on how clever its reasoning is. An AI limited to observational data of one kind, however large, cannot push FDR below π0/(π0+2π1F_b(μ)), and checking its own work adds nothing. An AI that can obtain evidence from modalities with independent biases gains at most one bit per modality in the large-sample limit, so it needs about log2 of the prior odds against its claims in such modalities. An AI that can intervene, as a robot scientist does [King et al., 2009], or that hands claims to wet-lab collaborators [Gottweis et al., 2025] gains about log2 19 ≈ 4.2 bits per experiment. It validated original claims at FDR 0.007 in our worlds. Originality, measured as lower prior odds, increases the evidence required logarithmically, not catastrophically. The required evidence must still be of the *right kind*.

**A bias-source ledger.** We propose that AI-scientist systems, and benchmarks that evaluate them, report for each accepted claim (i) the modalities of evidence used, (ii) for each candidate bias source (confounding, reverse causation, selection, measurement, analytic flexibility [Gelman and Loken, 2014]), which modalities randomize it, and (iii) the implied effective K and Λ. Under Theorem 2 and Proposition 3, a claim's credibility is governed by the *minimum over bias sources* of the evidence that randomizes that source. Large counts of replications, tool calls or falsification tests on one database do not count toward K if they share the dominant bias. This ledger complements e-value-based validation [Huang et al., 2025], which controls error under valid tests, by asking whether each test's null is valid. It also gives a concrete form to the falsification-first standard [Fa and Culjak, 2026].

**Interpreting reported validation rates.** Systems that report expert-judged accuracy of their statements [Mitchener et al., 2025] or reviewer scores [Lu et al., 2024; Yamada et al., 2025] are measuring agreement with judges who mostly see the same modality of evidence as the system. Our results suggest such numbers be read as replication rates, which are upper bounds on validation rates when bias is shared. The ideation–execution gap [Si et al., 2025b] fits this reading: judging an idea's promise without execution is a zero-independent-modality evaluation.

**Pre-registered protocol for real systems (proposed; not run).** The predictions can be tested on deployed systems without our simulator:
1. *Corpus.* Take an AI data-driven discovery agent and a set of public observational datasets for which randomized or quasi-experimental ground truth exists. Examples are gene-knockout screens paired with expression data, or policy evaluations with RCT follow-ups.
2. *Arms.* For each accepted claim, record (a) self-assessed validity, (b) replication on a held-out split, (c) replication on a different population, and (d) the experimental ground truth.
3. *Predictions.* (P1) Arms (a) and (b) give the same precision against (d) once the splits are large (Theorem 1). (P2) The gain from arm (c) is predicted by the fraction of false claims whose dominant bias is population-invariant, such as reverse causation (Proposition 3, Table 4). (P3) The precision of claims sorted by generation depth falls as the prior-odds model predicts (Table 5).
4. *Falsification criteria.* P1 is falsified if held-out replication improves precision against ground truth by more than the bootstrap CI at large split sizes. P2 is falsified if the decomposition predicts no better than the pooled model.

## 8 Limitations

- **Synthetic worlds only.** All empirical results come from linear-Gaussian SCMs with a hand-designed generator, literature and environments. No LLM was run; there was no GPU or API access. We make no claims about specific deployed systems beyond what their papers report. The protocol in §7 is proposed, not executed.
- **Stylized bias model.** Theorems 1–2 assume one additive bias per hypothesis–modality pair, symmetric and continuous, and conditional independence between modalities with independent biases. Real biases may be asymmetric (for example, publication-driven) or partly shared. Proposition 3 handles only the extreme case of perfect sharing. Asymmetric bias changes the constant 1/2 in Theorem 1 to P(b > 0), and so changes the one-bit ceiling to 1/P(b>0).
- **Fixed-α testing.** The floor relies on testing at a fixed level as n grows. A verifier that instead requires a minimum effect size τ has null pass rate F̄_b(τ) and can trade power for a lower floor. Theorem 2 still applies with f = F̄_b(τ). We did not study effect-size thresholds empirically.
- **Non-replication of one shape prediction.** The rise in replication FDR at large n (Figure 1) did not appear in the SEM sweeps (§6.4). The plateau at the self-verification value did appear. The stylized model should therefore not be read as a quantitative description of mixtures of heterogeneous biases.
- **Originality operationalized narrowly.** We measure originality as prior odds (through depth of search) and as graph distance from a simulated literature. The latter did not lower prior odds in our worlds, so the "originality tax" was demonstrated only through depth of search. Expert-judged novelty, as in Si et al. [2025a], is a different construct.
- **The environments verifier is weak by design.** It resamples only latent confounding. Richer environment designs, such as soft interventions on subsets of variables or invariance-based tests [Peters et al., 2016], would randomize more bias sources and change Table 2.
- **E3 decomposition uses ground-truth bias labels.** The class-specific prediction needs the share of each bias class, which we read from the simulator. In real systems these shares must themselves be estimated, for example from a sample of claims sent for experimental follow-up, and this adds uncertainty that we did not model.

## 9 Conclusion

An AI scientist that proposes original hypotheses and then verifies them with more data of the same kind is, in the large-sample limit, not verifying them. We proved that same-modality replication has an FDR floor that does not depend on bias magnitude, that bias-limited observational modalities give at most one bit of evidence each, and that modalities sharing a bias source eventually make triangulation counterproductive. In 300 synthetic causal worlds, these results explain why self-verification, replication and environment triangulation leave 70–79% of accepted original claims false, while two randomized interventions leave 0.7%. Calibrated bias-source accounting predicted the held-out FDR of each strategy to within about 0.01–0.02. The resulting rule is to count independent bias sources, not replications, and to scale that count with the logarithm of the prior odds against the claims. Whether deployed AI scientists can meet the rule depends on whether they have experimental access, not only on how capable their models are.

## Reproducibility

Code: `code/` (see `code/README.md`). The commands are `python3 exp1_theory_check.py`, `python3 exp2_sem_world.py 300` and `python3 analyze_exp2.py`. Together they take about 5 CPU-minutes, and all seeds are fixed. Outputs: `results/exp1_theory.json`, `results/exp2_summary.json`, `results/exp2_summary.txt`, `results/novelty_bias_composition.txt`, and figures `results/fig1_fdr_vs_n.png` (Theorem 1 curves with Monte Carlo), `fig2_kstar.png` (K* rule), `fig3_sem_sweep.png` (E4 sweeps), `fig4_e3_prediction.png` (E3), `fig5_novelty.png` (FDR by distance from the literature). Every numerical result in §6 is read from these files. The only numbers not produced by our code are the descriptive statements about other work in §1, §2 and §7, which come from the cited sources.

## References

- Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Eberhardt, F., Glymour, C., and Scheines, R. (2005). On the number of experiments sufficient and in the worst case necessary to identify all causal relations among N variables. In *Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence (UAI)*. arXiv:1207.1389. (Companion report: "N−1 experiments suffice to determine the causal relations among N variables", Carnegie Mellon University, 2004.)
- Fa, D., and Culjak, M. (2026). Sound agentic science requires adversarial experiments. arXiv:2604.22080.
- Gelman, A., and Loken, E. (2014). The statistical crisis in science. *American Scientist*, 102(6), 460. doi:10.1511/2014.111.460.
- Gottweis, J., et al. (2025). Towards an AI co-scientist. arXiv:2502.18864.
- Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. (2024). Large language models cannot self-correct reasoning yet. In *International Conference on Learning Representations (ICLR)*. arXiv:2310.01798.
- Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., and Leskovec, J. (2025). Automated hypothesis validation with agentic sequential falsifications. In *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267. arXiv:2502.09858.
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine*, 2(8), e124. doi:10.1371/journal.pmed.0020124.
- Javanmard, A., and Montanari, A. (2018). Online rules for control of false discovery rate and false discovery exceedance. *Annals of Statistics*, 46(2), 526–554. doi:10.1214/17-AOS1559.
- King, R. D., Rowland, J., Oliver, S. G., Young, M., et al. (2009). The automation of science. *Science*, 324(5923), 85–89. doi:10.1126/science.1165620.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292.
- Majumder, B. P., Surana, H., Agarwal, D., Dalvi Mishra, B., et al. (2025). DiscoveryBench: Towards data-driven discovery with large language models. In *International Conference on Learning Representations (ICLR)*. arXiv:2407.01725.
- Mitchener, L., et al. (2025). Kosmos: An AI scientist for autonomous discovery. arXiv:2511.02824.
- Munafò, M. R., and Davey Smith, G. (2018). Robust research needs many lines of evidence. *Nature*, 553, 399–401. doi:10.1038/d41586-018-01023-3.
- Open Science Collaboration (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251), aac4716. doi:10.1126/science.aac4716.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Peters, J., Bühlmann, P., and Meinshausen, N. (2016). Causal inference by using invariant prediction: identification and confidence intervals. *Journal of the Royal Statistical Society: Series B*, 78(5), 947–1012. doi:10.1111/rssb.12167.
- Ramdas, A., Yang, F., Wainwright, M. J., and Jordan, M. I. (2017). Online control of the false discovery rate with decaying memory. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*. arXiv:1710.00499.
- Si, C., Yang, D., and Hashimoto, T. (2025a). Can LLMs generate novel research ideas? A large-scale human study with 100+ NLP researchers. In *International Conference on Learning Representations (ICLR)*. arXiv:2409.04109.
- Si, C., Hashimoto, T., and Yang, D. (2025b). The ideation–execution gap: Execution outcomes of LLM-generated versus human research ideas. arXiv:2506.20803 (ICLR 2026).
- Wang, H., Fu, T., Du, Y., et al. (2023). Scientific discovery in the age of artificial intelligence. *Nature*, 620, 47–60. doi:10.1038/s41586-023-06221-2.
- Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv:2504.08066.
