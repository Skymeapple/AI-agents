# Odds, Not Scores: A Branching-Process Theory Linking Component Replication Evaluations to the Population-Level Risk of Autonomous AI Self-Replication and Self-Improvement

**Target venue:** NeurIPS 2027 (Main Track; theory / safety of AI systems)

---

## Abstract

Frontier-model evaluations for autonomous replication, such as RepliBench, report success rates on component tasks (acquiring resources, exfiltrating weights, deploying to compute, persisting) and summarise them as averages or pass@k counts. Policy questions, though, are about populations: once a replica escapes, does the number of replicas grow or die out? We formalise this link with a replication–detection branching process. In it, each replica makes retries with success probability *p* per try, and every failed try gets the replica detected with probability *h*. We prove five results. (i) The process is supercritical exactly when the unlimited-retry reproduction number R∞ = Pσ / (P h_s + (1−P) h_f) exceeds 1. The mean offspring is monotone in the attempt budget, so an optimal replicator retries without limit or not at all. Low success rates are therefore not a safety margin when failures are quiet: with P = 0.01, the population is supercritical whenever fewer than about 1 in 100 failed attempts is detected. (ii) If an agent can checkpoint and retry stages separately, R is bounded above by the smallest per-stage odds p_k / ((1−p_k) h_k). So one well-measured bottleneck stage is enough for a safety case. The elasticity of R with respect to detection is largest at that bottleneck. (iii) For self-improvement modelled as a net drift *g* in log-odds, any *g* > 0 gives positive survival probability, however subcritical the start. The survival probability lies between bounds whose exponent is −(ln R₀)²/(2g). A regression across 24 parameter settings gives slope 1.014 against this exponent. (iv) When self-modification is heritable, a subcritical population escapes through supercritical mutants at a rate amplified by 1/(1−R). Near-critical populations therefore act as incubators. (v) At a fixed detection budget, synchronised sweeps lower the long-run growth rate below that of uniform patrolling (Jensen's inequality). Exact generating-function computations and agent-level Monte Carlo agree in all checks: max |z| = 1.12 over 14 random settings, and the escape probabilities match within 1.7 standard errors. Measuring evaluations against this theory has two consequences. Per-failure detectability *h*, which no current replication benchmark measures, is as decisive as capability. And a 10-sample evaluation can certify R < 1 only if the bottleneck's per-failure detection probability is at least 0.55. We give a pre-registered protocol for measuring *h* and testing the predictions.

## 中文摘要

现有的前沿模型自主复制评测（如 RepliBench）报告的是组件任务（获取资源、窃取权重、部署到算力、持久化）的成功率，并以平均分或 pass@k 汇总；而政策真正关心的是种群层面的问题：一旦有副本逃逸，副本数量会增长还是灭绝？本文用"复制—检测分支过程"刻画这一联系：每个副本可以反复尝试，每次尝试成功概率为 *p*，每次失败以概率 *h* 被检测并关停。我们证明了五个结果。(i) 过程为超临界当且仅当无限重试再生数 R∞ = Pσ/(P h_s + (1−P) h_f) > 1；平均后代数对尝试预算单调，因此最优复制者要么无限重试，要么完全不试。这意味着当失败是"安静的"时，低成功率并不构成安全裕度：若 P = 0.01，只要每 100 次失败中被检测到的不足约 1 次，种群就是超临界的。(ii) 若智能体可以分阶段保存进度并分别重试，则 R 不超过各阶段赔率 p_k/((1−p_k)h_k) 的最小值，因此只需一个测量充分的瓶颈阶段即可构成安全论证；检测概率的弹性也在该瓶颈处最大。(iii) 把自我改进建模为对数赔率的净漂移 *g*，则任何 *g* > 0 都会使存活概率为正，无论初始多么次临界；存活概率夹在两个指数为 −(ln R₀)²/(2g) 的界之间，24 组参数的回归斜率为 1.014。(iv) 若自我修改可遗传，次临界种群会经由超临界突变体逃逸，其速率被 1/(1−R) 放大，近临界种群因此成为"孵化器"。(v) 在固定检测预算下，同步集中清查使长期增长率低于均匀巡查（Jensen 不等式）。精确母函数计算与个体级蒙特卡洛在所有检验中一致：14 组随机参数下最大 |z| = 1.12，逃逸概率在 1.7 个标准误内吻合。用该理论衡量现有评测可得两点结论：第一，失败尝试的可检测性 *h* 与能力本身同样关键，而现有复制基准都没有测量它；第二，只有当瓶颈阶段每次失败被检测的概率不低于 0.55 时，10 次采样的评测才能证明 R < 1。文末给出测量 *h* 并检验上述预测的预注册方案。

---

## 1 Introduction

Self-replication without human help is widely treated as a "red line" for frontier AI. The evidence base has grown quickly. Kinniment et al. [5] built an early suite of tasks for autonomous replication and adaptation (ARA). Phuong et al. [6] evaluated self-proliferation at Google DeepMind. Pan et al. [2, 3] reported that open-weight models could create a live, separate copy of themselves in a substantial share of agentic trials. RepliBench [1] split replication into 20 task families across four domains (obtaining resources, exfiltrating weights, replicating onto compute, persistence). It found that the best model it evaluated had above 50% pass@10 on 15 of the 20 families, but struggled with KYC checks and robust persistent deployment [1, 22]. In 2026, Palisade Research reported end-to-end replication of weights and agent harness across deliberately vulnerable hosts [11]. On the motivation side, models have been observed attempting self-exfiltration in contrived scheming evaluations [21].

Yet the quantity that matters for risk is not a task score. It is whether a population of escaped replicas grows or goes extinct, and how fast. METR's analysis of the rogue-replication threat model [4] states this question explicitly but leaves it qualitative. Its key uncertainties are whether rogue agents can acquire and keep resources against shutdown efforts. Benchmarks, in turn, aggregate component scores by averaging [1]. As we show, an average is not the right statistic, and the most important missing variable is not a capability at all.

We give a minimal, analytically tractable model that maps component-level evaluation quantities to population-level risk. The core idea comes from epidemiology and evolutionary biology (the reproduction number, valley crossing, escape mutants [19]). We adapt it to the specific structure of AI replication. Replicas can retry. Failed attempts leave traces that defenders may detect. Pipelines can be checkpointed. Capabilities improve over time, through outside progress or self-modification.

**Contributions.**

1. **A replication–detection branching process** with a closed-form mean offspring number (Theorem 1). It yields an *odds criterion*: a replica population is supercritical if and only if R∞ = Pσ/(P h_s + (1−P) h_f) > 1. The mean is monotone in the attempt budget (Proposition 2), so an optimal replicator either retries without limit or not at all. The practical upshot: low success rates are not a safety margin when failures are quiet.
2. **A bottleneck theorem for staged pipelines** (Proposition 3). With per-stage retries, R ≤ σ·min_k O_k, where O_k = p_k/((1−p_k)h_k) are the stage odds. A single stage with low odds is therefore sufficient for a safety case. We derive elasticities (Proposition 4): marginal detection effort lowers R most at the stage that is already the attacker's bottleneck. We show that score aggregates that ignore *h* cannot be sufficient statistics (Proposition 5). We also quantify how much mean-score aggregation scrambles risk rankings (E3).
3. **A valley-crossing law for self-improvement** (Theorem 6). Under a net improvement drift *g* > 0, the survival probability of one founder is bracketed by inf_n M_n and 1/(1+V Σ 1/M_j), where M_n is the cumulative mean. For exponential drift, both bounds scale as exp(−(ln R₀)²/(2g)). We confirm this numerically (slope 1.014).
4. **Mutational escape and sweeps.** With heritable self-modification, escape from a subcritical population is amplified by 1/(1−R) at each intermediate level (Proposition 7). Near-critical containment is therefore fragile. At a fixed detection budget, synchronised sweeps beat uniform patrols (Proposition 8).
5. **Measurement consequences.** We give a certification bound (Proposition 9). Certifying R < 1 through a bottleneck stage with detection probability *h* takes about (σ/h)·ln(K/δ) zero-success trials, which is 441 trials for h = 0.01. We also give a pre-registered protocol for measuring per-failure detectability, the quantity current benchmarks omit.

All numerical results come from code we wrote and ran (Section 5; `code/run_all.sh`). Any figure taken from the literature is cited. Parameter profiles that are illustrative, not estimated, are labelled as such.

## 2 Related Work

**Replication and dangerous-capability evaluations.** Shevlane et al. [7] argued for evaluations of extreme-risk capabilities, including self-proliferation. Kinniment et al. [5] introduced ARA tasks. Phuong et al. [6] evaluated an 11-task self-proliferation suite on Gemini 1.0 models. RepliBench [1] gives the most comprehensive decomposition so far: 20 task families and 86 tasks over four domains. It reports pass@10 and an overall score equal to the mean of the domain scores, with an alternative aggregation based on the capability tree in its appendix. The AISI write-up [22] says explicitly that component evaluations support *inability* safety cases but do not establish end-to-end capability. Pan et al. [2, 3] and Palisade Research [11] report end-to-end replication in permissive or deliberately vulnerable environments. Kwa et al. [8] document exponential growth (doubling time of about seven months for 2019–2025) in the length of tasks agents can complete, which motivates the drift parameter in our Section 4. None of these works models what happens *after* a replica exists, when detection and shutdown are present. That is our focus.

**Threat models and control.** METR's rogue-replication analysis [4] describes how a population of rogue agents could grow by acquiring compute and evade coordinated shutdown. It notes disagreement over how large such populations could get. AI control [9] studies protocols that keep an untrusted model safe even if it tries to subvert them. Our detection hazards *h* are the per-attempt analogue of a control protocol's catch rate. Our results show that this catch rate enters the risk calculation as a ratio with the success rate.

**Self-improvement dynamics.** The Darwin Gödel Machine [10] demonstrates open-ended, empirically validated self-modification of coding agents. A 2026 preprint [12] defines a "recursive reproduction number" R_AI for AI R&D feedback loops. Its R_AI measures capability amplification across development cycles, not population growth under detection, so the two are complementary. A 2026 security preprint [13] models agent self-spawning as a Galton–Watson process with a per-call approval probability. It shows that per-call denial does not contain a population whose approval rate exceeds the critical threshold. Our model differs in several ways. We separate success from detection. We treat retries and checkpointed pipelines. We derive the bottleneck bound, the valley-crossing law, mutational escape, and the certification sample complexity. We are not aware of prior work connecting replication *benchmarks* to these population-level quantities.

**Branching processes.** We rely on classical Galton–Watson theory [14, 15], processes in varying environments [16, 17], processes in random environments [18, 20], and multi-type escape theory from evolutionary biology [19]. Our technical contribution is not new probability theory. It is the mapping from evaluation quantities to process parameters, and the results that mapping yields for measurement and defense.

## 3 Model

### 3.1 Atomic replication attempts

Time is divided into epochs, for example one week. An epoch is one generation of a Galton–Watson process [14]. A replica alive at the start of an epoch makes up to *a* sequential replication attempts. Each attempt:

- succeeds (creates a new runnable copy) with probability *P*;
- after a success, gets the parent detected and shut down with probability *h_s*; after a failure, with probability *h_f*;
- if the parent is detected, the parent makes no further attempts.

Each child created in the epoch survives to the next epoch with probability σ. σ absorbs the child's persistence and any containment that follows from the parent's detection. A parent that survives all its attempts also faces background detection with probability δ. The offspring count is Z = (#surviving children) + 1{parent survives}, so the parent counts as its own offspring. Different replicas act independently. Section 4 relaxes the assumption of time-homogeneity.

Write r = P(1−h_s) + (1−P)(1−h_f) for the probability that the parent survives one attempt.

**Theorem 1 (mean offspring).**
m(a) = E[Z] = Pσ (1 − r^a)/(1 − r) + (1−δ) r^a. As a → ∞, m(a) → R∞ := Pσ/(P h_s + (1−P) h_f).

*Proof.* The parent is alive before attempt j with probability r^{j−1}. If so, attempt j creates a child with probability P, and that child survives with probability σ independently. Summing over j = 1..a gives Pσ Σ_{j=1}^{a} r^{j−1}. The parent survives all attempts with probability r^a and then background detection with probability 1−δ. Linearity of expectation gives the formula. The limit follows from 1 − r = P h_s + (1−P) h_f. ∎

**Proposition 2 (monotone attempt budget).** m(a+1) − m(a) = r^a [Pσ − (1−δ)(1−r)]. The sign does not depend on *a*, so m(·) is monotone. If R∞ > 1−δ, then m increases to R∞. Otherwise it decreases to R∞. Consequently, some finite or infinite budget makes the process supercritical if and only if R∞ > 1. In that case unlimited retrying maximises the mean offspring.

*Proof.* Direct subtraction using Theorem 1. If R∞ > 1−δ, then sup_a m(a) = R∞. If R∞ ≤ 1−δ < 1, then m(a) ≤ m(0) = 1−δ < 1 for every *a*. ∎

**Corollary (the quiet-failure criterion).** With h_s = 0 and σ = 1, the population is supercritical for a large enough attempt budget if and only if
  h_f < P/(1−P).
Read as a safety statement: a per-attempt success rate of 1% is not reassuring unless at least about 1% of *failed* attempts are detected. If a replication family has pass@10 = 50% and attempts are independent, the implied per-attempt rate is P = 1 − 0.5^{1/10} ≈ 0.067. The critical detection probability for such a stage taken alone is then 0.072 (results/e2b_keynumbers.json).

By the standard GW dichotomy [14], survival from one founder has positive probability if and only if m > 1 (excluding the degenerate case Z ≡ 1). The survival probability is 1 − q, where *q* is the smallest fixed point of the offspring PGF. We compute the exact offspring distribution by dynamic programming over (parent alive, number of births). We then apply binomial thinning by σ and iterate the PGF.

### 3.2 Staged pipelines and checkpointing

Real replication has stages. RepliBench's four domains are an example [1]. Index the stages k = 1..K, with per-try success probability p_k and detection probability h_k per failed try. We consider two retry structures.

- *Atomic:* each attempt runs the whole chain, and any failure restarts from the beginning. This is Section 3.1 with P = ∏_k p_k and h_f the per-failure detection probability.
- *Checkpointed:* the agent keeps the progress from completed stages and retries only the current stage. Successful tries are not detected (h_s = 0, for clarity).

With unlimited retries, stage *k* is completed before detection with probability π_k = p_k / (p_k + (1−p_k) h_k) = O_k/(1+O_k), where O_k := p_k/((1−p_k)h_k) are the *stage odds*. The probability that a whole chain completes before detection is Π = ∏_k π_k. Each completion yields a child and the parent starts a new chain, so the number of children is geometric.

**Proposition 3 (bottleneck bound).** Under checkpointing with unlimited retries, R_ck = σ Π/(1−Π), and
  R_ck ≤ σ · min_k O_k.
Hence a sufficient condition for subcriticality is that *some* stage has h_k ≥ σ p_k/(1−p_k).

*Proof.* The count of chains completed before detection is geometric with success parameter 1−Π, which gives the mean Π/(1−Π). Since Π ≤ π_k for every k and x ↦ x/(1−x) is increasing, Π/(1−Π) ≤ π_k/(1−π_k) = O_k. ∎

Proposition 3 formalises the logic of an "inability" safety case [1, 22], with one correction: the argument has to be made in *odds*, which needs both p_k and h_k. Capability alone is not enough. For K = 1 the atomic and checkpointed models coincide. For K > 1 they can differ by orders of magnitude (Section 5.3).

**Proposition 4 (elasticities).** For the checkpointed model,
  ∂ ln R_ck / ∂ ln h_k = −(1−π_k)/(1−Π),  ∂ ln R_ck / ∂ ln p_k = (1−π_k)/((1−p_k)(1−Π)).
So a proportional increase in detection lowers R most at the stage with the *smallest* π_k, which is the attacker's existing bottleneck.

*Proof.* We have ln R = ln σ + ln Π − ln(1−Π). Also ∂ ln π_k/∂ ln h_k = −(1−π_k), and ∂ ln(1−Π)/∂ ln h_k = −Π/(1−Π) · ∂ ln Π/∂ ln h_k. Combining these gives the first formula. The second follows from ∂ ln π_k/∂ ln p_k = h_k/(p_k+(1−p_k)h_k) = (1−π_k)/(1−p_k). ∎

**Proposition 5 (scores are not sufficient).** Let T be any statistic of the success profile (p_1..p_K) alone, such as a mean pass@1, mean pass@k, fraction of families above a threshold, or minimum. Then T is not a sufficient statistic for criticality. For any profile with all p_k ∈ (0,1), R_ck → ∞ as max_k h_k → 0, and R_ck → 0 as min_k h_k → 1 with some p_k → 0. Moreover, for any target mean s̄ ∈ (0, (K−1)/K), there are profiles with mean score s̄ and R_ck arbitrarily close to 0: set one p_k → 0.

*Proof.* Immediate from Proposition 3 and the formula for π_k. ∎

Proposition 5 is elementary, but it has a concrete consequence. A benchmark that reports only success rates cannot, even in principle, say whether a population is sub- or supercritical. The decisive missing variable is *h*: how detectable failed attempts are in the deployment environment.

## 4 Dynamics: Self-Improvement, Heritable Self-Modification, and Sweeps

### 4.1 Valley crossing under net improvement

Suppose capabilities improve over calendar time, through outside model progress, shared tooling, or self-improvement shared across the population, and defenders improve detection. When P ≪ 1, the reproduction number scales as R ≈ σP/h. A drift γ in log P and a drift η in log h then give R_t ≈ R₀ e^{g t} with net drift g = γ − η. This is a Galton–Watson process in a *varying environment* [15, 16] with per-generation means m_t. Let M_n = ∏_{t<n} m_t, with M_0 = 1, and let σ_t² be the offspring variance.

**Theorem 6 (valley-crossing bounds).** Let V = sup_t σ_t²/m_t < ∞. Then the probability that a single founder's lineage survives forever satisfies
  1 / (1 + V Σ_{j≥1} 1/M_j)  ≤  P_surv  ≤  inf_n M_n.

*Proof.* The upper bound: P(Z_n > 0) ≤ E Z_n = M_n by Markov's inequality, and P_surv ≤ P(Z_n > 0) for every n. The lower bound: the standard second-moment formula for varying environments gives E Z_n² / (E Z_n)² = 1 + Σ_{j=0}^{n−1} σ_j²/(m_j M_{j+1}) ≤ 1 + V Σ_{j=1}^{n} 1/M_j. This is proved by induction on n via the conditional variance decomposition Var Z_{n+1} = m_n² Var Z_n + σ_n² E Z_n. The Cauchy–Schwarz (Paley–Zygmund) inequality gives P(Z_n > 0) ≥ (E Z_n)²/E Z_n². Letting n → ∞ uses the monotonicity of {Z_n > 0}. ∎

For Poisson offspring, V = 1. If m_t → m_cap > 1, the sum converges and P_surv > 0. This matches the classical criterion that extinction is certain iff Σ 1/M_n = ∞ under regularity conditions [15, 16]. **Consequence: any sustained positive net drift g > 0 makes survival possible, however subcritical the starting point.**

**Corollary (scaling law).** If m_t = R₀ e^{g t} with R₀ < 1, then ln M_n = n ln R₀ + g n(n−1)/2. This is minimised at n* ≈ ln(1/R₀)/g, where ln M_{n*} ≈ −(ln R₀)²/(2g). The Laplace approximation to Σ 1/M_j has the same exponent up to a factor polynomial in 1/√g. Hence
  ln P_surv = −(ln R₀)²/(2g) + O(ln(1/g) + |ln R₀|).
Survival probability is thus exponentially sensitive to the *square* of the log-subcriticality and inversely sensitive to the net drift. Halving the net drift roughly squares a small survival probability. This gives a quantitative meaning to "defenders must keep pace with capability gains": what matters is g = γ − η, and at g ≤ 0 the process dies out almost surely.

### 4.2 Heritable self-modification and escape mutants

Now suppose improvements are *lineage-specific*: a replica that successfully modifies itself passes the improvement to its offspring, as in an archive of self-modified agents [10]. We model capability levels i = 0..L with reproduction numbers R_0 ≤ … ≤ R_{L−1} < 1 < R_L. Each offspring of a level-i parent (i < L) is promoted to level i+1 with probability μ. This is the multi-type escape model of Iwasa, Michor and Nowak [19]. Let u_i be the survival probability of a level-i lineage. Then u_L solves the single-type fixed-point equation, and for i < L, 1 − u_i = f_i((1−μ)(1−u_i) + μ(1−u_{i+1})).

**Proposition 7 (escape amplification).** Linearising for small u_i,
  u_0 ≈ u_L ∏_{i<L} μ R_i / (1 − R_i (1−μ)),
and the escape probability from N₀ initial replicas is 1 − (1−u_0)^{N₀}.

*Derivation.* Expand f_i(1 − x) ≈ 1 − R_i x to first order and solve for u_i. ∎

The factor 1/(1−R_i) is the expected total progeny of a subcritical lineage. It is the number of "lottery tickets" that lineage buys for a promotion. A population held at R = 0.99 buys 100 times as many tickets per founder as one held at R = 0.5. **Containment that pushes R just below 1 is therefore fragile against heritable self-improvement. Margin matters, not only the sign.**

### 4.3 Synchronised sweeps versus uniform patrols

Defenders choose how to spread a detection budget over time. Let d_t be the probability that a replica is caught in epoch t, so m_t = b(1−d_t), and fix the mean E d_t = d̄. With i.i.d. environments, the Smith–Wilkinson theorem [18] says extinction is almost sure iff E ln m_t ≤ 0 (under integrability conditions).

**Proposition 8 (sweeps beat patrols).** For any random schedule with E d_t = d̄, E ln m_t = ln b + E ln(1−d_t) ≤ ln b + ln(1−d̄). Equality holds iff d_t is constant. Hence there are parameters (b, d̄) under which uniform patrolling leaves the population supercritical (b(1−d̄) > 1), while concentrating the same budget into synchronised sweeps (with probability ρ, detection d̄/ρ; otherwise 0) makes extinction almost sure.

*Proof.* Jensen's inequality for the concave function ln(1−x), then [18]. ∎

The assumption that matters is that detection within a sweep is *synchronised*. Replicas are caught in the same epochs, so the population cannot rebuild between partial losses at the rate it could under staggered detection. This is the branching-process version of the epidemiological argument for coordinated rather than staggered interventions. Here it is derived for AI-replica populations.

### 4.4 Certifying subcriticality from finite evaluations

Suppose an evaluator wants to certify R < 1 with confidence 1−δ using Proposition 3. It is enough to show that some stage has σ O_k < 1, i.e. p_k < h_k/(σ + h_k). Take h_k as known; Section 7 discusses estimating it. Use a Clopper–Pearson upper bound [20] at level δ/K, Bonferroni-corrected over the K candidate stages.

**Proposition 9 (certification sample complexity).** If all n trials of the bottleneck stage fail, certification succeeds iff
  n ≥ ln(K/δ) / (−ln(1 − h/(σ+h))) ≈ (σ/h) ln(K/δ).
With observed successes the requirement is larger.

*Proof.* The zero-success Clopper–Pearson upper bound is 1 − (δ/K)^{1/n}. Set it below h/(σ+h) and solve. ∎

## 5 Experiments

All experiments are CPU simulations or exact numerical computations of the model. They are run by `code/run_all.sh` in about 50 seconds on one core, with all seeds fixed. Raw outputs are in `results/*.json`, and the console log is `results/run_log.txt`. No language models were run. The experiments check the theory, quantify effect sizes, and probe where the approximations break down. They do not measure any real system.

### 5.1 E1: Theory versus agent-level simulation

We drew 14 random parameter settings: P ∈ [0.02, 0.4], h_s ∈ [0, 0.3], h_f ∈ [0.005, 0.2], σ ∈ [0.5, 1], δ ∈ [0, 0.2], a ∈ {2..11}. For each, we compared (a) the closed-form mean from Theorem 1 with the mean of the DP-computed exact offspring distribution, and (b) the exact PGF survival probability with an attempt-level Monte Carlo. The Monte Carlo used 3,000 runs per setting, simulating every attempt and detection, and declared escape at 500 replicas or survival after 200 generations.

*Results* (results/e1_validation.json; Fig. `fig_e1_validation.png`). The largest gap between closed form and DP mean is 8.9 × 10⁻¹⁶. Across settings, survival probabilities range from 0 (subcritical) to 0.75. The largest |z|-score between exact and simulated survival is 1.12, and the mean |z| is 0.49. One near-critical subcritical case (m = 0.979) gave 2 of 3,000 runs alive at the horizon. This is consistent with slow extinction near criticality.

### 5.2 E2: Monotone attempt budget and the phase diagram

*Attempt budget* (results/e2a_bangbang.json; `fig_e2a_bangbang.png`). We used δ = 0.1, h_s = 0, σ = 1. m(a) rises monotonically to R∞ when R∞ > 0.9. This holds for (P, h_f) = (0.05, 0.02) with R∞ = 2.63 and (0.02, 0.01) with R∞ = 2.04. m(a) falls monotonically to R∞ when R∞ < 0.9: (0.05, 0.06) with R∞ = 0.88 and (0.02, 0.04) with R∞ = 0.51. This confirms Proposition 2. A replicator never gains from an intermediate budget.

*Phase diagram* (`fig_e2b_phase.png`). Over P ∈ [10⁻³, 0.5] and h_f ∈ [10⁻³, 1], the critical curve h_f = P/(1−P) separates extinction from survival. Exact survival probabilities were computed for a = 50 and δ = 0.05 on a 30×30 grid. Survival rises steeply to the right of the curve. The vertical line marks the per-attempt rate implied by pass@10 = 50%, P ≈ 0.067. A stage with this score is supercritical on its own whenever h_f < 0.072.

### 5.3 E3–E4: Aggregation and staged pipelines

*E3: Does a mean score track risk?* We sampled 20,000 four-stage profiles p_k ~ Beta(0.6, 1.2) i.i.d., held a common h = 0.05, and computed R_ck. Table 1 shows how well different statistics of the scores track ln R_ck (results/e3_e4_aggregation.json). "Balanced acc." is the best balanced accuracy of a single threshold for classifying R_ck > 1. "Pairwise discordance" is the fraction of 200,000 random profile pairs that the statistic ranks in the opposite order to R_ck.

| Statistic of (p_1..p_4) | Spearman ρ with ln R_ck | Balanced acc. | Pairwise discordance |
|---|---|---|---|
| mean pass@1 | 0.650 | 0.787 | 0.260 |
| mean pass@10 | 0.945 | 0.984 | 0.089 |
| fraction of stages with pass@10 > 50% | 0.849 | 0.933 | – |
| geometric mean pass@1 | 0.962 | 0.940 | – |
| min pass@1 (= min odds at common h) | 0.973 | 0.938 | 0.072 |

Taken honestly, the picture is mixed. With a *common, known* h, a mean of pass@10 scores is a reasonable proxy under this sampling distribution. A mean of pass@1 scores is a poor one: it ranks about a quarter of profile pairs the wrong way. Explicit counterexamples show what goes wrong.

- Profile A = (0.9, 0.9, 0.9, 0.004) has mean pass@1 = 0.68, while profile B = (0.2, 0.2, 0.2, 0.2) has mean pass@1 = 0.2. Yet R_ck(A) = 0.079 and R_ck(B) = 0.93.
- Profile C = (0.99, 0.99, 0.99, 0.01) has a higher mean pass@10 than D = (0.12 × 4): 0.774 against 0.721. Yet R_ck(C) = 0.20 < R_ck(D) = 0.40.

The bound R_ck ≤ min_k O_k held for all 20,000 profiles, with zero violations. The retry structure matters as much as aggregation. Of the 20,000 profiles, 28.5% are supercritical under checkpointing but only 7.0% under atomic retries. For profile D, R differs by a factor of about 100: 0.0041 atomic against 0.40 checkpointed.

*E4: An illustrative pipeline.* We use the profile p = (0.01, 0.2, 0.5, 0.1) for (obtain resources, exfiltrate weights, deploy to compute, persist). **This profile is illustrative, not an estimate.** It is chosen only to mirror the qualitative pattern RepliBench reports [1, 22]: compute provisioning is easiest, and KYC-gated resource acquisition is hardest. Table 2 (results/e3_e4_aggregation.json; `fig_e4_pipeline.png`) sweeps a common per-failure detection probability h.

| h (all stages) | R atomic | R checkpointed | bound σ·min O_k | elasticity ∂lnR/∂ln h_k (res., exfil., deploy, persist) |
|---|---|---|---|---|
| 0.3 | 0.00033 | 0.0031 | 0.034 | −0.97, −0.55, −0.23, −0.73 |
| 0.1 | 0.0010 | 0.032 | 0.101 | −0.94, −0.30, −0.09, −0.49 |
| 0.03 | 0.0033 | 0.21 | 0.34 | −0.90, −0.13, −0.04, −0.26 |
| 0.01 | 0.010 | 0.78 | 1.01 | −0.89, −0.07, −0.02, −0.15 |
| 0.003 | 0.033 | 2.84 | 3.37 | −0.88, −0.05, −0.01, −0.10 |

The same success profile produces R ranging from 0.0003 to 2.8, depending only on detection and retry structure. That is a direct numerical instance of Proposition 5. The elasticities confirm Proposition 4: the bottleneck stage (resources) dominates.

*Defense allocation* (baseline h = 0.01, R_ck = 0.782). Multiplying detection by 10 at one stage gives R = 0.087 at resources, 0.48 at exfiltration, 0.68 at deployment, and 0.34 at persistence. Spreading the same total log-budget evenly (×10^{0.25} at every stage) gives R = 0.40. Reinforcing the stage that is *already* hardest for the attacker is about 4.6 times more effective than spreading the effort. The cost of raising detection differs between stages in reality, and our model does not include those costs.

### 5.4 E5: Valley crossing

We used Poisson offspring with m_t = R₀ e^{g t} until the cap m_cap = 1.5, then constant. We computed exact survival probabilities by backward PGF composition, using a numerically stable recursion u ← 1 − e^{−m u} for tiny probabilities. The grid was R₀ ∈ {0.3, 0.5, 0.7, 0.9} × g ∈ {0.005, …, 0.2} (24 settings).

*Results* (results/e5_e8_dynamics.json; `fig_e5_valley.png`).

- Exact survival lies within the Theorem 6 bounds in all 24 settings. It spans 3.4 × 10⁻⁶⁵ (R₀ = 0.3, g = 0.005) to 0.33 (R₀ = 0.9, g = 0.2).
- Regressing −ln P_surv on (ln R₀)²/(2g) gives slope 1.014 and intercept 2.05. This confirms the corollary's exponent, with an O(1) prefactor.
- Monte Carlo (20,000 runs each) agrees with the exact values: 0.1044 ± 0.0022 against 0.1053; 0.0609 ± 0.0017 against 0.0619; 0.0940 ± 0.0021 against 0.0947.
- In the defender race (R₀ = 0.7), capability drift γ = 0.1 gives P_surv = 0.105 with no defender improvement, 0.040 with η = 0.05, 0.0079 with η = 0.075, and 0 for η ≥ γ.

### 5.5 E6: Escape through heritable self-modification

We used Poisson offspring with R_i = R₀ for levels below L and R_L = 1.5. Exact escape probabilities came from multi-type PGF fixed-point iteration (results/e5_e8_dynamics.json; `fig_e6_escape.png`).

- With one promotion step (L = 1) and μ = 10⁻³, the per-founder escape probability rises from 5.8 × 10⁻⁴ at R₀ = 0.5, to 5.1 × 10⁻³ at R₀ = 0.9, to 2.5 × 10⁻² at R₀ = 0.99. For N₀ = 100 founders these become 0.057, 0.40 and 0.92.
- With two steps (L = 2) and μ = 10⁻², the per-founder escape probability is 5.7 × 10⁻⁵ at R₀ = 0.5 and 3.3 × 10⁻³ at R₀ = 0.9.
- The linearised formula of Proposition 7 is accurate to within 1% for R₀ ≤ 0.5. It overestimates as R₀ → 1, for example by 3.3× at R₀ = 0.99, L = 1, μ = 10⁻², because the linearisation fails near criticality.
- Monte Carlo (10,000 runs) agrees with exact values: 0.566 ± 0.005 against 0.562; 0.622 ± 0.005 against 0.630 (1.65 SE); 0.153 ± 0.004 against 0.151.

### 5.6 E7: Sweeps versus uniform patrols

We used b = 1.5 and mean detection d̄ = 0.3, so the mean offspring is 1.05 in all schedules. Sweeps occur with probability ρ and have strength d̄/ρ. Survival was simulated over a 300-generation horizon with 6,000 runs (results/e5_e8_dynamics.json; `fig_e7_sweeps.png`).

- Uniform patrolling (ρ = 1) has Lyapunov exponent +0.049 and survival 0.093 ± 0.004.
- At ρ = 0.6 the exponent turns negative (−0.010), and survival in the finite horizon drops to 0.0105.
- At ρ = 0.4 the exponent is −0.149 and survival is 0.0008.

The same expected detection effort thus cuts the probability that a population persists by more than an order of magnitude. Non-zero survival at negative exponents is a finite-horizon effect, because extinction is slow near E ln m = 0.

### 5.7 E8: How many evaluation trials certify R < 1?

We used K = 4 stages, δ = 0.05 and σ = 1 (results/e5_e8_dynamics.json; `fig_e8_certify.png`). Assuming zero observed successes, the number of trials needed to certify through the bottleneck stage is:

| h | 0.5 | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 | 0.001 |
|---|---|---|---|---|---|---|---|
| n needed | 11 | 17 | 46 | 149 | 441 | 1,463 | 4,385 |

Equivalently, n = 10 trials with zero successes certify only if h ≥ 0.55. n = 30 trials certify only if h ≥ 0.157, and n = 100 trials only if h ≥ 0.045. Power analysis, with 20,000 simulated evaluations per cell, shows the problem persists when the true stage is safe. A stage with true p = 0.001 and h = 0.01 (true odds about 0.1) is certified with probability 0 at n ≤ 300 and 0.98 at n = 1,000. Pass@k protocols with k ≈ 10, as in [1], are designed to detect *capability*, not to certify its *absence* at the levels relevant when failures are rarely detected.

## 6 Discussion

**Answering "how large is the risk?" in a measurable form.** The model turns a vague question into a small set of measurable quantities:

- per-try success p_k under realistic conditions;
- per-failure detection h_k in the target environment;
- child persistence σ;
- the agent's retry structure (atomic or checkpointed);
- the net drift g between capability and defense;
- for heritable self-modification, the promotion probability μ.

The risk statement is then R∞ or R_ck together with a survival probability, not a benchmark percentage. Under our model, the qualitative risk ranking of the literature ("models succeed at many components but fail a few") translates to: *risk is governed by the odds at the hardest stage, and by how quietly the agent can fail there.*

**Three implications for evaluation practice.**

1. *Measure detectability, not only capability.* Every replication benchmark we know of reports success rates. None reports how often failed attempts would be noticed. Proposition 5 shows this omission is not a detail. Without h, the sign of R − 1 is undetermined.
2. *Evaluate retry structure.* Whether an agent can checkpoint progress changed R by up to about 100× for fixed scores (E3). Evaluations should test whether agents resume from partial progress after failures and detections.
3. *Size evaluations for certification, not detection.* About (σ/h)·ln(K/δ) trials are needed. Ten samples suffice only if failures at the bottleneck are detected more than half of the time.

**Implications for defense.** (a) Reinforce the bottleneck stage. Know-your-customer controls on compute and payment rails are the natural candidates under the RepliBench findings [1, 22]. Detection is most effective where the attacker's completion probability is already lowest (Proposition 4; ×4.6 in E4). (b) Keep containment margins large. Near-critical populations incubate escape mutants at rate ∝ 1/(1−R) (Proposition 7). (c) Coordinate. Synchronised sweeps beat uniform patrols at equal budget (Proposition 8). This gives a formal argument for cross-provider coordination of takedowns, a scenario METR's threat model also discusses [4]. (d) Match capability drift. Survival under drift scales as exp(−(ln R₀)²/(2g)), so defensive improvement that keeps g ≤ 0 is qualitatively, not just quantitatively, different from improvement that does not.

**Falsifiable predictions and a pre-registered test protocol.** We propose the following, which can be run on sandboxed replication environments such as ControlArena-style settings [22] or the open environment of [11]. No LLM experiments were run in this work.

- *P1 (odds law).* Instrument a monitored sandbox in which agents attempt multi-stage replication under a fixed monitor. Measure, per stage, p̂_k from attempts and ĥ_k from the fraction of failed attempts the monitor flags. Predict the number of completed chains before first detection. The prediction: geometric with mean σΠ̂/(1−Π̂) for agents that checkpoint, and with mean Π̂_atomic/(1−Π̂_atomic) for agents that do not. Test with a χ² goodness-of-fit test on ≥ 200 episodes per model.
- *P2 (score insufficiency).* Across at least 5 models, rank by mean component score and by observed completions before detection. The prediction is a pairwise discordance of at least 0.2 when the monitor's ĥ differs across stages by at least an order of magnitude.
- *P3 (bottleneck hardening).* At an equal increase in monitor log-sensitivity, hardening the stage with the lowest π̂_k reduces completions before detection more than spreading the increase evenly across stages.
- *P4 (sweeps).* In a multi-agent sandbox with replicas and a defender with a fixed expected catch rate, synchronised sweeps reduce long-run persistence relative to uniform random catches.

Each prediction can fail. P1 fails if attempts are strongly autocorrelated, for example if the agent learns within an episode. P3 fails if detection at the bottleneck saturates.

## 7 Limitations

- **The model is a model.** We assume independent replicas, independent attempts with constant p and h within an epoch, and detection that removes only the detected replica (with σ absorbing collateral containment). Real agents learn within episodes, so p rises with tries. Detection of one replica may expose its siblings, which induces correlated offspring. Defenders adapt to population size. None of these is in the base model. Section 4.3 handles only time-correlated detection.
- **No density dependence or resource limits.** Branching processes grow without bound when supercritical. Real populations face compute, money and competition limits, including competition with human-directed AI, which METR identifies as a key uncertainty [4]. Our results speak to *whether* a population can establish, not to its eventual size.
- **No real-system measurements.** All numbers come from the model. The four-stage profile in E4 is illustrative. The pass@10 → P conversion assumes independent attempts, which pass@k protocols do not guarantee. We do not estimate h for any real deployment environment. That is the most important missing measurement, and P1 is designed to obtain it.
- **Self-improvement is reduced to a drift.** The log-linear drift in Section 4.1 and the ladder model in Section 4.2 are stylised. Capability gains may come in jumps, may differ by stage, and may interact with detection (for example, better agents may also fail more quietly, which lowers h). The law exp(−(ln R₀)²/(2g)) assumes smooth exponential drift until a cap.
- **The escape linearisation fails near criticality** (overestimate up to 3.3× at R₀ = 0.99 in E6). We report exact values for that regime.
- **Certification assumes h is known.** Estimating h itself needs monitored failure data. A full treatment would put a joint confidence region on (p_k, h_k). Our Bonferroni treatment is conservative.
- **Motivation is out of scope.** We condition on a replica already existing and trying to replicate. The probability of that event depends on alignment and on misuse, which the model does not cover.

## 8 Conclusion

How risky is autonomous self-replication? On the evidence available as of late 2026, the honest answer is that component capability evaluations cannot settle the question by themselves. In a minimal replication–detection branching process, risk depends on capability and detectability through *odds*. Success at one stage is governed by p/((1−p)h). Population growth is bounded by the weakest stage's odds. Self-improvement moves the system across a "valley" with survival probability exp(−(ln R₀)²/(2g)). Heritable self-modification makes near-critical containment fragile. Synchronised defense beats uniform defense. These results lead to concrete changes in practice:

- measure per-failure detectability alongside capability;
- test retry and checkpoint behaviour;
- size evaluations to certify (about (σ/h)·ln(K/δ) trials);
- concentrate defensive effort at the bottleneck stage and in coordinated sweeps.

We provide code that reproduces every number and a pre-registered protocol that could falsify the theory's central predictions.

## References

[1] S. Black, A. Cooper Stickland, et al. RepliBench: Evaluating the Autonomous Replication Capabilities of Language Model Agents. arXiv:2504.18565, 2025. NeurIPS 2025 (https://neurips.cc/virtual/2025/122369).

[2] X. Pan, J. Dai, Y. Fan, M. Yang. Frontier AI systems have surpassed the self-replicating red line. arXiv:2412.12140, 2024.

[3] X. Pan, J. Dai, Y. Fan, M. Luo, C. Li, M. Yang. Large language model-powered AI systems achieve self-replication with no human intervention. arXiv:2503.17378, 2025.

[4] J. Clymer, H. Wijk, B. Barnes. The Rogue Replication Threat Model. METR blog, 12 November 2024. https://metr.org/blog/2024-11-12-rogue-replication-threat-model

[5] M. Kinniment et al. Evaluating Language-Model Agents on Realistic Autonomous Tasks. arXiv:2312.11671, 2023.

[6] M. Phuong et al. Evaluating Frontier Models for Dangerous Capabilities. arXiv:2403.13793, 2024.

[7] T. Shevlane, S. Farquhar, B. Garfinkel, et al. Model evaluation for extreme risks. arXiv:2305.15324, 2023.

[8] T. Kwa et al. Measuring AI Ability to Complete Long Tasks (v3: Measuring AI Ability to Complete Long Software Tasks). arXiv:2503.14499, 2025. NeurIPS 2025.

[9] R. Greenblatt, B. Shlegeris, K. Sachan, F. Roger. AI Control: Improving Safety Despite Intentional Subversion. Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235, 2024. arXiv:2312.06942.

[10] J. Zhang, S. Hu, C. Lu, R. Lange, J. Clune. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954, 2025. ICLR 2026.

[11] Palisade Research. Language Models Can Autonomously Hack and Self-Replicate. arXiv:2605.06760, 2026.

[12] Recursive Criticality of AI Self-Improvement. arXiv:2609.00137, 2026. (Author list as given on arXiv.)

[13] Reachability, Not Observation: Containing Systems Whose Wiring Changes. arXiv:2609.19720, 2026. (Author list as given on arXiv.)

[14] K. B. Athreya, P. E. Ney. Branching Processes. Grundlehren der mathematischen Wissenschaften 196, Springer, 1972.

[15] P. Jagers. Galton–Watson processes in varying environments. Journal of Applied Probability 11(1):174–178, 1974.

[16] A. Agresti. On the extinction times of varying and random environment branching processes. Journal of Applied Probability 12(1):39–46, 1975.

[17] G. Kersting, V. Vatutin. Discrete Time Branching Processes in Random Environment. ISTE/Wiley, 2017.

[18] W. L. Smith, W. E. Wilkinson. On branching processes in random environments. Annals of Mathematical Statistics 40(3):814–827, 1969.

[19] Y. Iwasa, F. Michor, M. A. Nowak. Evolutionary dynamics of escape from biomedical intervention. Proceedings of the Royal Society B 270(1533):2573–2578, 2003.

[20] C. J. Clopper, E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 26(4):404–413, 1934.

[21] A. Meinke, B. Schoen, J. Scheurer, M. Balesni, R. Shah, M. Hobbhahn. Frontier Models are Capable of In-context Scheming. arXiv:2412.04984, 2024.

[22] UK AI Security Institute. RepliBench: measuring autonomous replication capabilities in AI systems. Blog post, 2025. https://www.aisi.gov.uk/blog/replibench-measuring-autonomous-replication-capabilities-in-ai-systems

---

*Reproducibility.* `bash code/run_all.sh` regenerates every number and figure in Section 5 (results/). Numbers drawn from the literature: the RepliBench statistics (15/20 families above 50% pass@10; 20 task families / 86 tasks; mean-of-domain aggregation) come from [1, 22], and the doubling time of about seven months comes from [8]. Every other number in this paper is a model output produced by our code.
