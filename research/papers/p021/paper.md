# Reliability Ceilings in Agent Evaluations: Lapse Rates, Task-Level Failure Correlation, and What Time-Horizon Data Can Certify

**Target venue:** NeurIPS 2027, Datasets & Benchmarks Track (evaluation methodology)

---

## Abstract

Deploying an autonomous agent needs evidence that it fails rarely, on the order of 1% or 0.1% of the time, on the tasks it will be given. The best-known capability measurement, the *time horizon* (the human task length at which an agent succeeds 50% of the time), is estimated with a two-parameter logistic curve. That curve cannot express a floor of failures that does not go away as tasks get shorter. We make three changes. First, we import the *lapse rate* from psychophysics: an upper asymptote 1−λ on success probability. If λ > 1−q, no task length reaches reliability q, so the q-horizon does not exist. A two-parameter logistic fit cannot detect this case (Proposition 2). Second, we show that the correlation of failures within a task (ICC) sets three things: whether retries help, how fast pass^k decays, and how many *distinct tasks* a reliability certificate needs (Propositions 3–4). Third, we apply these tools to METR's public Time Horizon 1.1 run data: 20 agents, 228 tasks, 23,235 runs. Three results stand out. (i) The two most recent frontier agents in the data have positive lapse rates: GPT-5.3-Codex with λ̂ = 3.5% (profile 95% CI 2.1–4.6%) and Claude Opus 4.6 with λ̂ = 2.3% (0.5–4.2%). For both, a 99% horizon is undefined with high bootstrap probability, while the two-parameter fit reports one of 0.2–0.7 minutes. (ii) On tasks shorter than 15 human-minutes, failures of frontier agents are mostly task-level (median ICC 0.76). The 50% horizon doubled every 4.1 months across these agents, while the short-task failure rate halved only every 8.8 months, and it showed no detectable trend among the 10 agents released since April 2025. Two successive releases from one developer show a statistically detectable short-task *regression* despite a longer 50% horizon. (iii) With the measured ICC, certifying a 1% failure rate needs 240–300 distinct short tasks even if no failures are observed, and extra runs per task barely reduce that number. That is 2–2.5× the 119 sub-15-minute tasks in the suite, and a 0.1% certificate needs 2,400–3,000 tasks. A simulation calibrated to the data shows that the current design cannot reliably detect a 1–2% ceiling, and that adding runs does not fix this while adding tasks does. We release code and propose a reporting protocol for deployment-grade reliability claims.

## 中文摘要

部署自主智能体，需要有证据表明它在实际任务上的失败率足够低（约 1% 或 0.1%）。目前最受关注的能力度量是"时间视界"，即智能体成功率为 50% 时对应的人类任务时长。它用两参数 logistic 曲线估计，而这种曲线无法表示"任务再短也消除不了的失败下限"。本文借用心理物理学中的"失误率（lapse rate）"概念，给成功概率设定上渐近线 1−λ。我们证明：若 λ > 1−q，则任何任务长度都达不到可靠度 q，即 q-视界不存在；两参数模型在结构上无法发现这种情况（命题 2）。我们还说明，任务内失败相关性（ICC）同时决定三件事：重试是否有效、pass^k 衰减多快，以及可靠性认证需要多少个*不同的任务*（命题 3–4）。在 METR 公开的 Time Horizon 1.1 数据（20 个智能体、228 个任务、23,235 次运行）上，我们得到三点发现。（i）数据中最新的两个前沿智能体存在显著的失误下限：GPT-5.3-Codex 的 λ̂ = 3.5%（95% 剖面置信区间 2.1–4.6%），Claude Opus 4.6 的 λ̂ = 2.3%（0.5–4.2%）。两者的 99% 视界在多数自助样本中不存在，而两参数模型仍给出 0.2–0.7 分钟的数值。（ii）在人类耗时 15 分钟以内的任务上，前沿智能体的失败主要发生在任务层面（ICC 中位数 0.76）。50% 视界每 4.1 个月翻一番，短任务失败率则每 8.8 个月才减半；2025 年 4 月以后发布的 10 个智能体之间，短任务失败率没有可检测的下降趋势。同一开发者的两代连续模型在 50% 视界提升的同时，出现了统计上可检测的短任务可靠性倒退。（iii）按实测 ICC 计算，即使零失败，认证 1% 失败率也需 240–300 个不同的短任务（现有 119 个），认证 0.1% 需 2,400–3,000 个；增加每个任务的运行次数几乎无济于事。最后，我们提出了部署级可靠性声明的报告规范。

---

## 1 Introduction

Whether an AI agent can be deployed is decided less by what it can sometimes do than by how often it fails on what it is asked to do. A customer-service agent, a coding agent merging to a production branch, or an operations agent acting on infrastructure is plausibly held to failure rates of 1% or lower on routine work. Throughout, we use 1% and 0.1% as illustrative targets, not as sourced industry thresholds. Current agent evaluation, however, mostly reports *capability* summaries. Headline success rates, pass@k [Chen et al. 2021], and METR's *50% time horizon* [Kwa et al. 2025] all describe the frontier of what an agent can do, not the floor of how reliably it does easy things.

The time-horizon methodology is a natural place to study this gap. It is longitudinal, it uses human-calibrated task lengths, and its run-level data are public. It fits, for each agent, a two-parameter logistic regression (2PL) of success on log human task length, and reads off the length at which predicted success is 50%. Higher-reliability horizons (80%, and in derived work 99% [Ord 2025]) are read off the same curve. METR itself cautions that the 2PL "doesn't fit the top and bottom of the success curve separately." It also warns that broken or mislabelled tasks could make a benchmark "saturate at 98%," in which case the 99% horizon would be zero, and that measuring 99% horizons would need roughly 300 diverse tasks per time bucket [METR 2026]. These cautions are qualitative. This paper turns them into a model, theorems, measurements, and a sample-size calculus.

**Sub-question.** We ask a narrow, answerable version of "how does agent reliability reach a deployable level?": *Given repeated-run evaluation data indexed by task length, is there evidence of a length-independent failure floor (a reliability ceiling) for current frontier agents? Is that floor shrinking as capability grows? And how much evaluation data would it take to certify failure rates of 1% or 0.1%?*

**Contributions.**

1. **A lapse-augmented time-horizon model and its consequences (§3).** We add a lapse parameter λ (upper asymptote 1−λ, as in psychometric-function fitting [Wichmann & Hill 2001]). We prove that the q-horizon does not exist when λ > 1−q (Proposition 1). We show that the standard 2PL, by construction, reports a finite positive q-horizon for every q < 1 (Proposition 2), so it cannot reveal a ceiling.
2. **A decomposition of failures that bears on deployment (§3.3).** Using repeated runs per task, we separate within-task (stochastic) from between-task (systematic) failure variance with unbiased U-statistics. We show that this intra-task correlation (ICC) determines how much retries help, how pass^k decays, and the identified upper bound on the mass of never-solved tasks (Proposition 3).
3. **A certification bound (§3.4).** If failures are perfectly task-level, no number of runs per task can substitute for distinct tasks. Any valid (1−δ) upper confidence bound after zero failures on N tasks is at least 1−δ^{1/N} ≈ ln(1/δ)/N (Proposition 4). For intermediate ICC we use the Kish design effect.
4. **An empirical audit of METR Time Horizon 1.1 (§5).** On 20 agents and 23,235 runs we find: statistically detectable lapse rates of 2–4% in the two newest frontier agents; a 50% horizon that doubles roughly twice as fast as the short-task failure rate halves; no detectable short-task reliability improvement among the ten agents released since April 2025; one statistically detectable within-developer short-task regression; and frontier short-task failures that are mostly task-level (median ICC 0.76) and concentrated on a handful of tasks.
5. **A calibrated simulation and a reporting protocol (§5.5, §6).** A simulation matched to the data quantifies how well the current design can detect a 1–2% ceiling, how far 2PL extrapolation misleads, and how badly run-level binomial intervals under-cover. We end with a concrete protocol for deployment-grade reliability claims.

All numbers in this paper come either from code we ran on the public METR data (labelled *[ours]*, files under `results/`) or from cited sources. We ran no new agent experiments.

## 2 Related Work

**Time horizons and their limits.** Kwa et al. [2025] introduced the 50% task-completion time horizon, fit by logistic regression on log human time, using HCAST [Rein et al. 2025], RE-Bench, and short SWAA tasks. They reported doubling roughly every seven months since 2019. Ord [2025] observed that a constant-hazard (exponential) model fits the same data and implies T80 ≈ T50/3 and T99 ≈ T50/70. METR [2026] lists limitations including the 2PL's inability to fit the tails separately, label noise that would cap measurable reliability, and the number of tasks needed for 99% horizons. We formalize and measure these concerns. We also show that the logistic and exponential shapes disagree on T99 by a factor of 1.1–54 at the same T50: our 2PL fits give T50/T99 between 78 and 3,746 across agents, against 70 under constant hazard. The 99% horizon is therefore set by the assumed shape, not by the data.

**Agent reliability and consistency.** τ-bench [Yao et al. 2024] introduced pass^k, the probability that all k i.i.d. trials of a task succeed, and showed it falls steeply with k for strong function-calling agents. τ²-bench [Barres et al. 2025] extends this to dual-control settings. Rabanser et al. [2026] decompose reliability into consistency, robustness, predictability, and safety, with twelve metrics over 14–15 models. They find that reliability gains lag capability gains, which is consistent with our trend result, though our evidence is of a different kind (a parametric ceiling and short-task failure rates in a length-indexed suite). A study of OSWorld agents [*On the Reliability of Computer Use Agents*, 2026] attributes run-to-run unreliability to execution stochasticity, task ambiguity, and behavioural variability. ReliabilityBench [Gupta 2026] adds perturbation and fault-injection stress. Kapoor et al. [2024] argue for cost-aware, reproducible agent evaluation. Our contribution is complementary. We give an identification-aware account of *what a length-indexed repeated-run suite can establish about the high-reliability regime*, and a sample-size calculus for certifying it.

**Statistics of evaluations.** Miller [2024] treats evaluation items as samples from a super-population and recommends clustered standard errors. Bowyer et al. [2025] show that CLT-based intervals under-cover on small benchmarks. For repeated sampling, Kazdan et al. [2025] fit beta-binomial models to extrapolate pass@k. A 2026 analysis [*What Fixed-Rollout pass@k Evaluations Can Identify*] proves that n rollouts per task identify only n moments of the latent success distribution, so pass@k for k > n is partially identified. We use the identified quantities (pass^k and E[(1−p)^n] for k, n at most the number of runs) and do not extrapolate beyond them.

**Lapses and rare-event certification elsewhere.** The lapse rate is standard in psychometric-function fitting, where ignoring stimulus-independent errors biases slope and threshold estimates [Wichmann & Hill 2001]. Testing the boundary null λ = 0 needs the chi-bar-square reference distribution [Self & Liang 1987]. The "rule of three" [Hanley & Lippman-Hand 1983] and the autonomous-vehicle argument that one cannot "drive to safety" [Kalra & Paddock 2016] motivate our certification analysis. The design effect for clustered samples is due to Kish [1965].

## 3 Model and Theory

### 3.1 Setup

An agent is evaluated on tasks i = 1,…,N. Task i has a human completion time t_i, with x_i = log₂ t_i, and receives n_i independent runs with binary outcomes Y_ij. Conditional on the task, runs are i.i.d. Bernoulli(p_i). This is the conditional-binomial model used by τ-bench and pass@k analyses. Tasks are drawn from a population, so p_i is random with a distribution that depends on x. Define the length-conditional reliability p̄(x) = E[p_i | x_i = x] and the **q-horizon**

  H_q = sup{ t : p̄(log₂ t) ≥ q }, with H_q undefined (written ∅) if the set is empty.

The standard 2PL takes p̄(x) = σ(a − b x) with b > 0, so H_q = 2^{(a − logit q)/b}. The **lapse model** (3PL) takes

  p̄(x) = (1 − λ) σ(a − b x),  0 ≤ λ < 1/2,

where λ is the probability of failure on arbitrarily short tasks. It can arise from run-level slips (formatting errors, tool or harness faults, premature submission), from task-level defects (ambiguous or mis-scored tasks, mis-estimated human times), or from task-level capability gaps that do not scale with length (e.g., character counting).

### 3.2 Ceilings and the 2PL

**Proposition 1 (reliability ceiling).** Under the lapse model, H_q exists iff q < 1−λ, and then H_q = 2^{(a − logit(q/(1−λ)))/b}. More generally, for any p̄ that is non-increasing in x, H_q = ∅ whenever q > lim_{x→−∞} p̄(x).

*Proof.* sup_x (1−λ)σ(a−bx) = 1−λ, approached only as x → −∞ and never attained, so {x : p̄(x) ≥ q} is non-empty iff q < 1−λ. On that set, solving (1−λ)σ(a−bx) = q gives the expression. The general statement follows because p̄ never exceeds its supremum, which a non-increasing p̄ approaches as x → −∞. ∎

**Proposition 2 (the 2PL cannot express a ceiling).** For every (a, b) with b > 0 and every q ∈ (0,1), the 2PL horizon H_q is finite and strictly positive. Hence, whatever the data, a 2PL fit reports a positive q-horizon for every q < 1, including when the data-generating process has a lapse λ > 1−q and the true H_q is ∅.

*Proof.* logit q is finite for q ∈ (0,1) and b > 0, so H_q = 2^{(a − logit q)/b} ∈ (0, ∞). ∎

Proposition 2 is elementary, but it matters in practice. Any statement of the form "the agent's 99% horizon is T" that comes from a 2PL is an *extrapolation conditional on λ = 0*. The data must be checked for a ceiling separately, and the 2PL itself cannot do that check. A second effect, which we measure rather than prove, is that a lapse forces the 2PL's slope b to flatten so that it can track short-task failures. This propagates to the 50% horizon (§5.1).

**Shape dependence of high-q horizons.** Even with λ = 0, H_q/H_50 is set by the assumed link. Under the 2PL it is 2^{−logit(q)/b}, which is 1/200 at q = 0.99 for a typical b = 0.6. Under the constant-hazard model of Ord [2025] it is ln q / ln 0.5, which is 1/69 at q = 0.99. Both models fit the bulk of the data. The 99% horizon is therefore mostly a statement about the link function.

### 3.3 Decomposing failures: within-task vs between-task

Let μ = E[p] over a task set (e.g., tasks shorter than T minutes). The run-level failure variance splits as

  μ(1−μ) = E[p(1−p)] + Var(p),

where the first term is within-task (stochastic) and the second is between-task (systematic). The **intra-task correlation** ρ = Var(p)/(μ(1−μ)) is the correlation between two runs' outcomes on the same task. With n_i ≥ 2 runs, E[p²] and E[p(1−p)] have unbiased estimators s_i(s_i−1)/(n_i(n_i−1)) and s_i(n_i−s_i)/(n_i(n_i−1)), where s_i is the number of successes. ρ is therefore identified with as few as two runs per task. With one run per task it is not.

**Proposition 3 (what ρ and the count law imply for deployment).** Let π₀ = P(p = 0) be the mass of never-solved tasks.
(a) *Retries with a perfect verifier.* Success with r retries is E[1 − (1−p)^r], which increases to 1 − π₀ as r → ∞. With n runs per task, E[(1−p)^n] is identified (unbiasedly estimated by C(n−s, n)/C(n, n) averaged over tasks, generalising to C(n−s, m)/C(n, m) for m ≤ n) and satisfies E[(1−p)^n] ≥ π₀. So pass@n is an identified *lower* bound on the retry ceiling 1 − π₀.
(b) *Consistency.* pass^k = E[p^k] ≥ μ^k (Jensen). For fixed μ, pass^k increases with Var(p) and hence with ρ. At ρ = 1 (p ∈ {0,1}), pass^k = μ for every k. At ρ = 0, pass^k = μ^k.
(c) *Detectability.* In every case, quantities beyond the n-th moment (pass^k or pass@k for k > n, and π₀ itself) are only partially identified. This follows from the moment argument of [*What Fixed-Rollout pass@k Evaluations Can Identify*, 2026].

*Proof.* (a) (1−p)^r decreases in r and converges to 1{p=0}, so monotone convergence gives the limit. E[(1−p)^n] ≥ E[1{p=0}] = π₀ because (1−p)^n ≥ 1{p=0} pointwise. Unbiasedness of the hypergeometric estimator is the standard argument behind the pass@k estimator [Chen et al. 2021]. (b) Jensen gives the inequality. For p ∈ {0,1}, p^k = p. For degenerate p ≡ μ, E[p^k] = μ^k. Monotonicity in Var(p) at fixed mean holds for k = 2, since E[p²] = μ² + Var(p). For larger k we state it only for the two-point extremes. ∎

The deployment reading is direct. When ρ is high, failures sit on particular tasks. Retries and consistency look good, but the failing tasks stay failed, and every one of them is a deployment incident unless a human catches it. When ρ is low, retries with a verifier can buy reliability, but pass^k decays geometrically.

### 3.4 Certification needs distinct tasks

**Proposition 4 (tasks, not runs).** Suppose p_i ∈ {0,1} for all tasks (ρ = 1). Let U be any upper confidence bound for μ_fail = 1 − μ with coverage at least 1−δ for every μ_fail. If zero failures are observed on N tasks, with any number of runs per task, then U ≥ 1 − δ^{1/N} ≈ ln(1/δ)/N.

*Proof.* When p_i ∈ {0,1}, all runs on a task agree, so the data are equivalent to N i.i.d. Bernoulli(μ) task outcomes, and P_{μ_fail}(zero failures) = (1−μ_fail)^N. Suppose U(zero failures) = u < 1 − δ^{1/N}. Choose μ_fail ∈ (u, 1−δ^{1/N}). Then U fails to cover μ_fail whenever zero failures occur, which has probability (1−μ_fail)^N > δ, contradicting coverage. ∎

At the other extreme, p_i ≡ μ (ρ = 0), the bound improves to 1 − δ^{1/(N m)} with m runs per task. In between, we use the Kish [1965] design effect: n_eff = N m / (1 + (m−1)ρ). The marginal value of the m-th run per task tends to zero as ρ → 1. With δ = 0.05, ln 20 ≈ 3, so this reproduces the rule of three [Hanley & Lippman-Hand 1983] *in units of tasks*.

## 4 Data and Estimation

**Data [public].** METR `eval-analysis-public`, report `time-horizon-1-1`, file `runs.jsonl` (commit `52cb829`, March 2026). We use all non-human runs: 20 agents from GPT-4 (March 2023) to Claude Opus 4.6 and GPT-5.3-Codex (February 2026), 228 tasks from HCAST, RE-Bench and SWAA, and 23,235 runs. Most agent–task pairs have 4–8 runs. Human task lengths range from 1.2 seconds to 30 hours (median 12.6 minutes). There are 119 tasks shorter than 15 minutes and 82 shorter than 4 minutes. Release dates come from the repository's `release_dates.yaml`. We use the binarized score, as METR does. Runs ending in a "usageLimits" fatal error (997 runs, about 4%) count as failures, as in the source data.

**Estimation [ours].** For each agent we fit the 2PL and the lapse model by maximum likelihood on per-task binomial counts (unweighted). We use multi-start Nelder–Mead with λ = 0.5·sigmoid(θ₃). The null λ = 0 lies on the boundary, so the likelihood-ratio test uses the ½χ²₀ + ½χ²₁ reference [Self & Liang 1987]. The 95% interval for λ is a profile-likelihood interval over an 81-point grid on [0, 0.25]. Horizon uncertainty comes from a 300-replicate *task-family cluster bootstrap*. Within each replicate, an undefined horizon is counted as 0 minutes, and we report the fraction of replicates in which H_q exists. As a robustness check we refit with METR's equal-task weights. Our 2PL fits are unregularized and unweighted, so they approximate METR's pipeline rather than reproduce it. All 2PL horizons we quote are *our* estimates, not METR's published values.

## 5 Results

### 5.1 Lapse rates and the (non-)existence of high-reliability horizons

Table 1 reports the fits *[ours; results/fits.csv]*. For the 10 agents released before April 2025, the lapse MLE is essentially zero (≤ 1.1%) and the LRT is not significant. Their failure curves are steep enough, and their short-task failures rare enough relative to the slope, that the 2PL fits the short-length regime well.

**Table 1.** Selected agents. H in human minutes. λ̂ with 95% profile CI. p is the boundary-corrected LRT. "P(H99 exists)" is the bootstrap fraction of replicates with a defined 99% horizon under the lapse model.

| Agent (release) | H50 2PL | H50 lapse | λ̂ [95% CI] | LRT p | H99 2PL [boot 95%] | P(H99 exists) |
|---|---|---|---|---|---|---|
| GPT-4o (2024-05) | 7.3 | 7.3 | 0.000 [0, 0.006] | 0.50 | 0.04 [0.02, 0.10] | 0.99 |
| Claude 3.7 Sonnet (2025-02) | 60.1 | 63.1 | 0.011 [0, 0.031] | 0.088 | 0.18 [0.02, 1.5] | 0.51 |
| o3 (2025-04) | 131 | 133 | 0.003 [0, 0.014] | 0.13 | 1.55 [0.39, 5.3] | 0.79 |
| GPT-5 (2025-08) | 205 | 205 | 0.000 [0, 0.006] | 1.0 | 0.85 [0.19, 3.6] | 0.94 |
| Gemini 3 Pro (2025-11) | 231 | 230 | 0.004 [0, 0.013] | 0.047 | 1.59 [0.16, 12] | 0.91 |
| GPT-5.1-Codex-Max (2025-11) | 241 | 241 | 0.000 [0, 0.004] | 1.0 | 1.87 [0.30, 8.1] | 0.99 |
| Claude Opus 4.5 (2025-11) | 315 | 315 | 0.000 [0, 0.013] | 0.50 | 0.71 [0.13, 6.3] | 0.81 |
| GPT-5.2 (2025-12) | 401 | 389 | 0.006 [0, 0.021] | 0.079 | 1.84 [0.26, 9.8] | 0.71 |
| Claude Opus 4.6 (2026-02) | 824 | 622 | **0.023 [0.005, 0.042]** | **0.0075** | 0.22 [0.01, 3.2] | **0.36** |
| GPT-5.3-Codex (2026-02) | 396 | 371 | **0.035 [0.021, 0.046]** | **1.3×10⁻⁵** | 0.71 [0.08, 7.9] | **0.29** |

Four findings follow.

1. **Two frontier agents have detectable ceilings.** For GPT-5.3-Codex the lapse model improves AIC by 15.6, and λ̂ = 3.5% excludes 2% at 95%. A 99% horizon is undefined in 71% of bootstrap replicates, and a 95% horizon is undefined in 15%. Claude Opus 4.6 has λ̂ = 2.3% (AIC improvement 3.9, LRT p = 0.0075), and its 99% horizon is undefined in 64% of replicates. Across 20 tests, only GPT-5.3-Codex survives a Bonferroni correction (threshold 0.0025). Opus 4.6 is significant at the per-test 1% level but not after correction. Gemini 3 Pro (p = 0.047) is not significant after correction. The equal-task-weighted fits give similar lapse estimates: 3.8% and 2.5%.
2. **The 2PL reports 99% horizons of 0.2–0.7 minutes for both agents anyway** (Proposition 2). These numbers carry no information about whether 99% reliability is attainable on any task length in this suite.
3. **A ceiling also biases the headline 50% horizon.** To fit the short-task failures, the 2PL flattens. Claude Opus 4.6 has the shallowest slope of any frontier agent (b = 0.39 vs. 0.50–0.72). Its 2PL H50 is 824 minutes, against 622 minutes under the lapse model, a 32% inflation. For GPT-5.3-Codex the inflation is 7%. For the other frontier agents it is under 3%. So the lapse is not only a high-q issue: it changes the headline number.
4. **For most agents a ceiling of practical size cannot be ruled out.** The profile upper bound on λ is at least 1% for 10 of 20 agents, including o3, Claude 4/4.1 Opus, Gemini 3 Pro, Claude Opus 4.5 and GPT-5.2 (upper bounds 1.0–2.1%). Only GPT-5 and GPT-5.1-Codex-Max have upper bounds below 0.6%. For most frontier agents, then, the data are *consistent with* a 99% ceiling, and they do not *establish* a 99% horizon.

Figure 1 (results/fig1_curves.png) shows binned failure rates on a log scale with both fits. For GPT-5.3-Codex the observed failure rate is flat at about 5–10% from 1 to 16 minutes. The 2PL cannot follow this pattern and the lapse model can.

### 5.2 Anatomy of short-task failures

We restrict to tasks shorter than 15 human-minutes (119 tasks), the regime where deployment-grade reliability would first be expected *[ours; results/decomp_T15.csv, fig2_decomp.png]*. We classify each failed run as *intermittent* (the same agent solved that task in another run) or *consistent* (all runs failed). We also flag tasks as *consensus-hard* when the pooled success of the other frontier agents (released ≥ April 2025) is below 50%.

- **Frontier short-task failure rates sit at 2–6% of runs.** The rates are 2.0% (Gemini 3 Pro, 95% CP 1.0–3.4%), 2.1% (GPT-5.1-Codex-Max), 2.6% (o3, GPT-5), 2.7% (GPT-5.2), 3.0% (Claude Opus 4.5), 4.1% (Claude 4.1 Opus), 4.5% (Claude Opus 4.6), 4.9% (GPT-5.3-Codex) and 6.0% (Claude 4 Opus). No frontier agent's 95% upper bound on the run-level failure rate is below 3.4%. Run-level binomial intervals are also too narrow when failures cluster (§5.5).
- **Failures are mostly task-level.** The unbiased ICC on short tasks ranges from 0.66 to 0.82 across frontier agents (median 0.76). Two consensus-hard tasks, `blackbox/apron` and `blackbox/beach` (rated 10 human-minutes), were failed in all 46 frontier runs by all 10 frontier agents. They account for 18–80% of each frontier agent's sub-15-minute failures (median 47%). These tasks award partial continuous scores (e.g., 0.5) that binarize to failure. This pattern fits either a mis-estimated human time or a scoring threshold, rather than a capability frontier at 10 minutes. We cannot tell these apart from the run data. Without the two tasks, frontier short-task failure falls to 0.4–5.0% (GPT-5.1-Codex-Max 0.4%, Gemini 3 Pro 0.7%, GPT-5.3-Codex 3.4%, Claude 4 Opus 5.0%) *[ours]*.
- **The remaining failures are concentrated too.** Twelve further sub-15-minute tasks have pooled frontier success below 90% (results/consensus_tasks.csv). Among them are counting words in a passage (`count_words/*`, 76–85%), reversing hashes (77–85%), and small data-cleaning tasks (68–86%). These are short tasks with known model weaknesses, not broken tasks.
- **Below 4 minutes, frontier failures are entirely intermittent.** For every frontier agent, every failure on a sub-4-minute task was on a task the agent solved in another run, so pass@4 ≈ 1.000 for o3, GPT-5, Gemini 3 Pro, Opus 4.5 and GPT-5.2. In the very short regime, then, frontier unreliability takes the form of run-level slips that a perfect verifier with retries would remove. In the 4–15-minute regime it is task-level.
- **pass^k.** On sub-15-minute tasks, the unbiased pass^4 of frontier agents is 0.91–0.97 *[ours]*, against a pass^1 of 0.94–0.98. Meanwhile pass@4 exceeds pass^1 by only 0.4–2.3 pp, and is capped at 117/119 = 0.983 by the two consensus-hard tasks. That is a modest decay, as Proposition 3(b) predicts for high ICC.

### 5.3 Does short-task reliability track capability?

Across all 20 agents, log₂ H50 (2PL) rises linearly in release date. The fitted doubling time is **4.1 months** (95% CI 3.6–4.7), estimated on this 2023–2026 agent set *[ours; results/trends.json]*. This is shorter than the roughly 7-month long-run figure of Kwa et al. [2025], which covers 2019 onwards. The run-level failure rate on short tasks also falls, but more slowly: it **halves every 8.8 months** for tasks under 15 minutes and every 8.6 months for tasks under 4 minutes. Excluding the two consensus-hard tasks, the halving time is 6.5 months. Restricted to the 10 agents released since April 2025, the slope of log₂ short-task failure on date is −0.10 ± 0.66 per year (p = 0.89) for tasks under 15 minutes, and −0.53 ± 1.40 (p = 0.71) for tasks under 4 minutes. Over the same agents, log₂ H50 rises by +2.97 per year. Over the most recent ten months of releases, the 50% horizon grew at a fitted rate of roughly 8× per year, while short-task reliability shows no detectable change (Figure 4, results/fig4_trends.png). The CIs are wide, so we do not claim that reliability has stalled. We claim only that the data give no evidence that it improved over this window.

**Within-developer paired comparisons** *[ours; results/paired_tests.csv]*. On common tasks, with a task-family cluster bootstrap (5,000 replicates):

| Older → newer | H50 2PL (min) | fail, tasks < 4 min | fail, tasks < 15 min |
|---|---|---|---|
| GPT-5.1-Codex-Max → GPT-5.3-Codex | 241 → 396 | 0.20% → 2.34% (+2.1 pp, 95% CI [+0.2, +7.6], p ≈ 0.017) | 2.1% → 5.3% (+3.2 pp, [+0.2, +7.9], p ≈ 0.007) |
| Claude Opus 4.5 → Opus 4.6 | 315 → 824 | 0.41% → 2.24% (+1.8 pp, [0.0, +5.6], p ≈ 0.06) | 3.6% → 4.5% (+0.8 pp, [−0.7, +3.0], p ≈ 0.36) |
| GPT-5 → GPT-5.2 | 205 → 400 | 1.5% → 1.2% (n.s.) | 3.4% → 2.7% (n.s.) |
| o3 → GPT-5 | 131 → 205 | 1.1% → 1.5% (n.s.) | 3.3% → 3.4% (n.s.) |
| Claude 4.1 Opus → Opus 4.5 | 101 → 315 | 2.4% → 0.4% (n.s.) | 5.3% → 3.6% (n.s.) |

(Failure rates here are task-averaged, which is why they differ slightly from §5.2's run-weighted figures.) GPT-5.3-Codex, the successor with the higher 50% horizon, is *less* reliable on short tasks than its predecessor. The difference is detectable at both thresholds, and the <15-minute comparison survives a Bonferroni correction over the five pairs at the 5% level (0.007 < 0.01) but not over all ten tests. The 5.3-Codex failures cluster on `count_words`, `data_cleaning_arjun` and `interpret_building_data`, tasks its predecessor solved in all, or all but one, of its runs (`count_words`: 0 predecessor failures). These are 4–6 runs per task from a single scaffold, so the regression could reflect elicitation or harness differences rather than the model weights. Either way, it shows that a better 50% horizon does not guarantee better short-task reliability for a deployed configuration.

### 5.4 What it would take to certify deployment-grade reliability

Table 2 applies Proposition 4 and the design-effect approximation with the measured median frontier ICC ρ = 0.76 *[ours; results/certification.csv]*. Targets are one-sided 95% upper bounds below ε, planned for the expected number of failures.

**Table 2.** Distinct tasks needed to certify a short-task failure rate below ε.

| ε | true rate | 1 run/task | 8 runs/task, ρ = 0 | 8 runs/task, ρ = 0.76 | 8 runs/task, ρ = 1 |
|---|---|---|---|---|---|
| 5% | 0 | 60 | 8 | 48 | 60 |
| 1% | 0 | 301 | 38 | 238 | 301 |
| 1% | 0.25% | 474 | 60 | 375 | 474 |
| 1% | 0.5% | 1,065 | 134 | 842 | 1,065 |
| 0.1% | 0 | 3,053 | 382 | 2,413 | 3,053 |
| 0.1% | 0.05% | 10,607 | 1,326 | 8,383 | 10,607 |

At the measured ICC, eight runs per task are worth about 1.26 independent tasks. Certifying a 1% failure rate on short tasks needs 238–1,065 distinct tasks, depending on the true rate. That is 2–9× the 119 sub-15-minute tasks in the suite, and it assumes every task is correctly specified, since a single broken task contributes a full unit of failure (cf. §5.2). A 0.1% certificate needs thousands of tasks. These numbers turn METR's informal estimate (~300 tasks per bucket for 99%) into a calculation. They also show that spending budget on extra runs instead of new tasks is mostly wasted when ICC is high.

### 5.5 Simulation: detectability, extrapolation, and interval coverage

To check what the *current design* can detect, we simulate from a model calibrated to Claude Opus 4.5 *[ours; results/sim_S1.csv, sim_S2.csv, sigma_calibration.json]*. The model has a random task intercept u_i ~ N(0, σ²), fitted by marginal maximum likelihood (σ = 7.7 on the logit scale, which reproduces the strong task-level clustering), plus either a run-level lapse λ_R or a task-level lapse (a fraction π_T of tasks with p = 0). We keep the actual task lengths and run counts, and use 200 replicates per cell.

**Table 3.** Power of the boundary-corrected lapse LRT (α = 0.05), and the median [10th, 90th percentile] 2PL-extrapolated 99% horizon in *seconds*, under the current design. Power under two augmented designs with the same budget increase (4× more short tasks vs. 4× more runs per task) is also shown. "Ceiling" is the true asymptotic failure probability. When it is at least 1%, the true H99 does not exist.

| Lapse configuration | ceiling | power, current | power, 4× short tasks | power, 4× runs | 2PL H99 (s), current |
|---|---|---|---|---|---|
| none | 0 | 0.02 | 0.01 | 0.03 | 36 [9, 85] |
| run-level 1% | 1% | 0.34 | 0.67 | 0.65 | 11 [4, 33] |
| run-level 2% | 2% | 0.56 | 0.93 | 0.91 | 3.7 [1.3, 13] |
| task-level 1% | 1% | 0.31 | 0.58 | 0.39 | 11 [1.6, 50] |
| task-level 2% | 2% | 0.43 | 0.79 | 0.59 | 4.6 [0.5, 33] |
| task 1% + run 1% | ≈2% | 0.52 | 0.84 | 0.75 | 5.0 [0.7, 20] |

Three conclusions follow.
(i) **The current design misses realistic ceilings about half the time or more.** Power to detect a 1% ceiling is 0.31–0.34, and power for a 2% ceiling is 0.43–0.56. The non-detections in Table 1 are therefore weak evidence that no ceiling exists. Under this heterogeneous model the test is conservative (size 0.01–0.03), because the random task effects push the fit toward λ̂ = 0.
(ii) **When the ceiling is task-level, more tasks beat more runs.** Quadrupling short tasks raises power to 0.58–0.79, while quadrupling runs on the same tasks raises it only to 0.39–0.59. For run-level lapses, the two augmentations are about equally effective (0.65–0.93), as expected. This is Proposition 4 in operation.
(iii) **The 2PL always reports a 99% horizon.** In every configuration with a ceiling of 1% or more, where the true H99 does not exist, the 2PL returns a median 99% horizon of 4–11 seconds. The returned horizon gets *shorter* as the ceiling grows, which can look like a meaningful number but is not one.

**Interval coverage (S2).** For the short-task (<15 min) failure rate, the run-level Clopper–Pearson 95% interval, which treats runs as independent, covers the true rate in only 73–85% of replicates. The task-family cluster bootstrap covers it in 93–96%, with intervals about twice as wide (mean width 4.9–6.9 pp vs. 2.3–3.2 pp) *[ours; results/sim_S2.csv]*. The run-level intervals reported in §5.2 should therefore be read as optimistic. Figure 5 (results/fig5_sim.png) summarizes S1.

## 6 Discussion: a reporting protocol for deployment-grade reliability claims

The results suggest a protocol for anyone claiming that an agent is reliable enough to deploy on a class of tasks. It is falsifiable and modest in cost.

1. **Fit and report a ceiling.** Report λ̂ with a profile CI and a boundary-corrected test alongside any time horizon. State "H_q does not exist" when the upper confidence bound on 1−λ is below q. Never report a 2PL-extrapolated H_q with q above the empirical success rate on the shortest bucket.
2. **Report the ICC and the failure anatomy.** With at least two runs per task, report the within/between decomposition, pass^k and pass@k for k ≤ n, and the identified bound E[(1−p)^n] ≥ π₀. List the tasks that make up the failure mass. Our data show that two tasks can carry most of it.
3. **Budget by distinct tasks.** Size the reliability test set with Proposition 4 and the design effect: about 3/ε *tasks* at ρ ≈ 1. Audit every failing task in the high-reliability regime for specification and scoring errors, because one broken task sets a floor of 1/N.
4. **Use cluster-level uncertainty.** Use task- or family-cluster bootstrap intervals, not run-level binomial intervals (§5.5).
5. **Test reliability regressions between releases.** Use a task-paired cluster bootstrap on the short-task bucket, as in §5.3, and treat a significant increase as a regression even if the 50% horizon improved.

**Falsifiable predictions.** (P1) If the `blackbox/apron` and `blackbox/beach` failures come from scoring or time-estimation artefacts, then re-scoring them with a lower threshold or re-baselining their human times will lower the fitted λ for every frontier agent. The fitted λ for GPT-5.3-Codex would stay above 1% only if its `count_words`/`data_cleaning` failures are reproducible. (P2) With independent repeated runs under a fixed harness, GPT-5.3-Codex's sub-4-minute failure rate will exceed GPT-5.1-Codex-Max's in a pre-registered replication on new short tasks of the same families. (P3) Because frontier short-task failures are task-level, best-of-r retry with a perfect verifier (r ≥ 4) will raise sub-15-minute success by less than 2.5 percentage points for every frontier agent here. Our data show pass@4 − pass@1 of 0.4–2.3 pp, and pass@4 is an identified lower bound on the retry ceiling, so this prediction can fail only if success keeps rising for r > 4. (P4) For any new suite with ICC ≈ 0.76 and 6 runs per task, doubling runs per task will narrow a cluster-robust CI on short-task failure by only about 1–2% (design-effect calculation: √[(1+11ρ)/12 ÷ (1+5ρ)/6] = 0.987), while doubling tasks will narrow it by about 29% (1/√2).

**Implications for the opening question.** "Deployable" reliability is a property of the agent–task-distribution pair, and it is limited by a failure floor that does not shrink with task length. On public data, this floor is 2–6% of runs for every frontier agent on sub-15-minute software tasks. Most of it sits on identifiable tasks, and its trend over the past year is not distinguishable from flat. Reaching deployability therefore needs work on two fronts: reducing task-level systematic failures (or fixing the tasks), and building test suites with hundreds to thousands of distinct, audited short tasks. Neither is measured by the 50% horizon.

## 7 Limitations

- **One suite, one domain.** All data are software, ML-engineering and cybersecurity tasks from a single organization's suite, run with specific scaffolds. Both the lapse estimates and the ICC are properties of agent, scaffold and suite together, and may not transfer to customer service, web or computer-use agents.
- **Few tasks in the regime that matters.** The lapse estimates rest on 119 sub-15-minute tasks, and a handful of tasks drive the results. This is the central caveat. Our own Table 2 implies that the suite cannot certify 1% reliability. Our positive findings (λ > 0 for two agents, a regression for one pair) are *detections*. Our negative findings (no recent trend) are *non-detections* with wide intervals.
- **We cannot audit task validity.** We infer possible task defects (`blackbox/*`) from cross-model consensus and partial scores, not from inspecting the tasks, most of which are private. A task failed by every agent could also be genuinely hard and under-timed.
- **Usage-limit failures.** About 4% of runs end in a "usageLimits" error, and we count them as failures, following the data. This is a deployment-relevant failure mode, but it may reflect evaluation budget settings rather than the agent. Of the short-task failures we inspected for the lapse agents, none were usage-limit failures, but we did not re-run every analysis without them.
- **Model approximations.** Our 2PL is unregularized and unweighted, so it does not reproduce METR's pipeline. The lapse model assumes a single constant λ across lengths. The design-effect calculation is a first-order approximation, exact only at ρ ∈ {0, 1}. Multiple comparisons across 20 agents and 10 paired tests are corrected only where stated.
- **No causal attribution.** We cannot separate model-weight effects from scaffold, elicitation, or infrastructure effects, or from run-order and date effects. Release dates are a proxy for vintage.
- **No new agent runs.** We had no GPU or API access. The predictions in §6 are a pre-registration and have not been tested.

## 8 Conclusion

Reading high-reliability horizons off a two-parameter logistic curve assumes away the quantity that matters for deployment. Adding a lapse parameter costs one extra parameter. On METR's public data, the lapse model shows 2–4% failure floors in the two newest frontier agents (one of them surviving a correction for 20 tests), and it shows that the 2PL reports 99% horizons where none can exist. Frontier failures on short tasks are mostly task-level. That explains why retries help little, why pass^k decays slowly, and why certification must be budgeted in distinct tasks: roughly 3/ε of them, or about 300 for a 1% claim and 3,000 for 0.1%. Capability, measured by the 50% horizon, has been doubling about every four months, while short-task reliability has shown no detectable gain over the last ten months of releases. Measuring deployability is a different problem from measuring capability. It needs larger audited suites of short tasks, cluster-aware statistics, and explicit ceiling tests.

## Reproducibility

All code is in `code/` (`run_all.sh` runs a1–a5 in about 15–20 CPU-minutes). Outputs and figures are in `results/`. The data source is pinned to a commit. Each number marked *[ours]* maps to a file in `results/`.

## References

- Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. (2025). τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.
- Bowyer, S., et al. (2025). Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints. arXiv:2503.01747 (ICML 2025 position track).
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Gupta, A. (2026). ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions. arXiv:2601.06112.
- Hanley, J. A., & Lippman-Hand, A. (1983). If nothing goes wrong, is everything all right? Interpreting zero numerators. *JAMA*, 249(13), 1743–1745.
- Kalra, N., & Paddock, S. M. (2016). Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability? *Transportation Research Part A: Policy and Practice*, 94, 182–193.
- Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N., & Narayanan, A. (2024). AI Agents That Matter. arXiv:2407.01502.
- Kazdan, J., Schaeffer, R., Allouah, Y., Sullivan, C., Yu, K., Levi, N., & Koyejo, S. (2025). Efficient Prediction of Pass@k Scaling in Large Language Models. arXiv:2510.05197.
- Kish, L. (1965). *Survey Sampling*. New York: Wiley.
- Kwa, T., West, B., Becker, J., et al. (2025). Measuring AI Ability to Complete Long Software Tasks (originally "Measuring AI Ability to Complete Long Tasks"). arXiv:2503.14499.
- METR (2026, January 22). Clarifying limitations of time horizon. METR Notes. https://metr.org/notes/2026-01-22-time-horizon-limitations/
- METR. eval-analysis-public (code and run data for the time-horizon reports). https://github.com/METR/eval-analysis-public, commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2.
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640.
- *On the Reliability of Computer Use Agents* (2026). arXiv:2604.17849.
- Ord, T. (2025). Is there a half-life for the success rates of AI agents? arXiv:2505.05115.
- Rabanser, S., Kapoor, S., Kirgis, P., Liu, K., Utpala, S., & Narayanan, A. (2026). Towards a Science of AI Agent Reliability. arXiv:2602.16666; *Proceedings of ICML 2026*, PMLR 306.
- Rein, D., et al. (2025). HCAST: Human-Calibrated Autonomy Software Tasks. arXiv:2503.17354.
- Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association*, 82(398), 605–610.
- *What Fixed-Rollout pass@k Evaluations Can Identify* (2026). arXiv:2609.09245.
- Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. *Perception & Psychophysics*, 63(8), 1293–1313.
- Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.
