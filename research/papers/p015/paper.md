# Frailty, Not Fatigue: Replicate Runs Show That Between-Task Heterogeneity, Not Declining Hazard, Shapes AI Agents' Success on Long Tasks

**Target venue:** ICLR 2027 (main track, evaluation methodology / datasets and benchmarks area)

**Code and data:** `code/` (run `bash code/run_all.sh`); all outputs in `results/`. Every number in this paper comes either from that code, run on METR's public run logs, or from a cited source. Which one is stated at each use.

---

## Abstract

Can AI agents complete tasks that take a human days or weeks? The main empirical tool for this question is METR's *time horizon*: a curve of agent success against the time a human needs for the task, fitted to one pool of tasks. Ord (2025) read these curves as survival curves with a constant hazard, i.e. a fixed chance of failing per minute of task. Later pooled fits found a Weibull shape *k* ≈ 0.6, which has been read as a hazard that *declines* as a task goes on. That reading leads to fairly optimistic extrapolations to multi-day tasks. We point out that a pooled success curve cannot separate a hazard that changes *within* tasks from *heterogeneity between* tasks of the same length. In survival analysis this second effect is called frailty. We then show that the replicate runs in METR's public data do separate them. Our contributions are four. (i) **Theory.** Every pooled curve with a decreasing population hazard, Weibull *k* ≤ 1 included, can be produced exactly by tasks that each have a constant hazard (Bernstein's theorem). Any frailty pushes the pooled Weibull shape strictly below the task-level shape. With two or more runs per task, a nonparametric heterogeneity index *H* is identified, and *H* fixes how much retries help. (ii) **Measurement.** We apply this to 23,235 runs by 20 agents on 228 tasks (METR Time Horizon 1.1). Within narrow length strata, between-task heterogeneity accounts for a median *H* = 0.61 of outcome variance (IQR 0.49–0.75). Differences in task length within a stratum explain under 3% of it. (iii) **Decomposition.** Frailty GLMMs fitted to the replicate runs give a task-conditional Weibull shape with median *k* = 1.46, against a pooled *k* = 0.63. The conditional shape exceeds the pooled one for all 20 agents, and is above 1 for 20/20 agents under Gaussian frailty and 15/20 under a three-class frailty. For a task of fixed relative difficulty, then, cumulative hazard grows *at least* linearly with length. The apparent "Lindy" decline is a property of the task population. (iv) **Consequences.** On held-out tasks the frailty model predicts pass@k accurately (mean absolute error 0.008–0.013 for k = 1–4). The homogeneous model overstates retry gains by a factor of 2.3–2.6. On tasks of 4 hours or more, pass@4 is 0.30 observed, 0.30 predicted by the frailty model and 0.47 predicted by the homogeneous model. Extrapolated to a 40-hour "work-week" task, the best agents' expected success on a random task from this population ranges from 0.01 to 0.76 depending on which family is assumed, which current data do not pin down. In every frailty fit that mass comes from a minority of tasks that are near-reliably solved, while most week-long tasks are near-hopeless. We conclude that the multi-day question is mainly about the distribution of difficulty among long tasks, and we propose a replicate-run protocol to measure it.

## 中文摘要

AI 智能体能否完成人类需要数天乃至数周的任务？目前回答这一问题的主要实证工具是 METR 的"时间跨度"（time horizon）：在一个任务池上拟合智能体成功率与人类完成时长之间的关系曲线。Ord（2025）将其解读为恒定风险率（每分钟失败概率固定）的生存曲线；此后的合并拟合得到 Weibull 形状参数 *k*≈0.6，被解读为"风险率随任务进行而下降"，并据此对多日任务给出较乐观的外推。本文指出：仅凭合并成功率曲线，无法区分"任务内部风险率随时间变化"与"同等时长任务之间的难度异质性"（生存分析中称为 frailty，即脆弱性/异质性）；而 METR 公开数据中的重复运行恰好可以识别二者。贡献包括：（i）理论：任何总体风险率递减的合并曲线（包括 *k*≤1 的 Weibull）都可以完全由"每个任务风险率恒定"的混合生成（Bernstein 定理）；任何异质性都会使合并 Weibull 形状严格小于任务层面的形状；当每个任务有两次及以上运行时，非参数异质性指数 *H* 可识别，并决定重试的收益。（ii）测量：在 METR Time Horizon 1.1 的 20 个智能体、228 个任务、23,235 次运行上，按时长分层后，任务间异质性占结果方差的中位数比例为 *H*=0.61（四分位距 0.49–0.75），层内时长差异只解释其中不到 3%。（iii）分解：基于重复运行的 frailty 广义线性混合模型给出任务条件 Weibull 形状中位数 *k*=1.46，而合并拟合为 0.63；20 个智能体的条件形状全部大于合并形状，在高斯异质性假设下 20/20 大于 1，在三类潜变量异质性假设下 15/20 大于 1。也就是说，对难度固定的任务，累积风险至少随时长线性增长；所谓"越做越稳"是任务总体的属性，而非单个任务的属性。（iv）推论：在留出任务上，frailty 模型对 pass@k 的预测平均绝对误差为 0.008–0.013（k=1–4），而同质模型将重试收益高估 2.3–2.6 倍；在 4 小时以上的任务上，pass@4 观测值为 0.30，frailty 模型预测 0.30，同质模型预测 0.47。外推到 40 小时（一个工作周）的任务时，最强智能体在该任务总体中随机抽取任务的成功率随所选分布族不同而在 0.01 至 0.76 之间变化，现有数据无法确定；在所有 frailty 拟合中，这部分成功率来自少数几乎总能完成的任务，而多数周级任务几乎无望完成。因此，"能否完成多日任务"本质上是关于长任务难度分布的问题。我们提出一种基于重复运行的测量协议来回答它。

---

## 1 Introduction

Planning and execution over long horizons is widely seen as the capability that separates today's AI agents from autonomous workers. Short-horizon planning benchmarks such as PlanBench (Valmeekam et al., 2023) show that language models struggle even on classical planning problems when they act without external verifiers (Kambhampati et al., 2024). For long *execution*, the most widely used quantitative evidence is METR's time-horizon methodology (Kwa et al., 2025; METR, 2026). Agents attempt software, ML-engineering and cybersecurity tasks whose length is calibrated by human completion time (Rein et al., 2025; Wijk et al., 2024). A logistic curve in log human time is fitted per agent, and the *50% time horizon* is the length at which predicted success is one half. METR reports that this horizon has grown exponentially for six years. Under Time Horizon 1.1 it doubles roughly every four to seven months depending on the period (METR, 2026). Extrapolating the trend suggests agents that can "independently complete a large fraction of software tasks that currently take humans days or weeks" within the decade (Kwa et al., 2025).

How far that extrapolation can be trusted depends on the *shape* of the success-versus-length curve, not only its location. Ord (2025) noted that a constant per-minute failure rate gives exponential survival, *P*(*T*) = 2^(−*T*/*h*₅₀), so the 50% horizon is a half-life. On this model the 80% horizon is about a third of it and the 99% horizon about 1/70. A later analysis by Gus Hamilton, reported by Ord (2026), fitted a Weibull survival function and found shape *k* ≈ 0.6 < 1 for every agent. Ord read this as hazard rates that "decline as a task goes on", a Lindy-like effect under which an agent that has survived the early part of a task becomes less likely to fail. The reading matters because a hazard that declines within tasks makes very long tasks much easier than constant hazard would.

This paper starts from a classical point in survival analysis (Vaupel, Manton and Stallard, 1979; Proschan, 1963). **A falling population hazard does not imply a falling individual hazard.** If items differ in frailty, the frail ones fail early and the survivors are increasingly the robust ones. The population hazard then falls even when every item's own hazard is constant or rising. For AI agents the "items" are (agent, task) pairs. Tasks of the same human length plainly differ in how hard they are for a given agent. This is documented informally by item-response analyses of METR's data (BRIDGE, Liu et al., 2026; Nguyen, 2026; and two LessWrong analyses cited below). Pooled curves therefore cannot tell *fatigue-like* within-task dynamics from *frailty*. The two explanations predict very different things about week-long tasks and about whether retries, checkpoints and verification help.

**The sharp, tractable question** we answer is this. *Using only public data, how much of the observed shape of AI success-versus-length curves comes from heterogeneity between tasks, and what is the task-level hazard shape once heterogeneity is removed?* The key observation is that METR's public logs contain **replicate runs**: a median of 4 independent attempts per agent-task pair, with at least 2 runs for >99.9% of pairs. Replicates identify the *second moment* of task-level success probabilities, and pooled single-run data cannot.

**Contributions.**

1. **Identification theory (Sec. 3).** (a) Proposition 1: the pooled curves that a population of constant-hazard tasks can produce are exactly the completely monotone survival functions. Weibull *k* ≤ 1 lies in this class, so pooled data cannot separate within-task declining hazard from frailty. (b) Proposition 2: for any non-degenerate frailty, the pooled local Weibull shape is strictly below the task-level shape *k*. (c) Proposition 3: with ≥ 2 runs per task, a nonparametric heterogeneity index *H*(*T*) ∈ [0, 1] is identified. (d) Proposition 4: *H* fixes retry gains exactly for *k* = 2 and bounds them for all *k*.
2. **A measurement method (Sec. 4).** We give a stratified unbiased estimator of *H*, frailty GLMMs with nested family/task random effects, and a held-out-task test of retry predictions. A simulation on METR's actual design shows that the method recovers the truth in cases where pooled fits cannot tell them apart (Sec. 5.6).
3. **Empirical findings on METR TH1.1 (Sec. 5).** Heterogeneity is large: median *H* = 0.61 within strata. The task-conditional Weibull shape is ≥ 1, with median 1.46 and range 1.18–1.80 across frailty specifications, against a pooled shape of 0.63. The "declining hazard" is a population effect. The frailty model predicts held-out pass@k to within about 0.01, while the homogeneous model overstates retry gains by a factor of about 2.3–2.6.
4. **Implications for multi-day tasks (Sec. 5.7, 6).** The expected success on a random 40-hour task is not identified by current data: plausible models give 0.01–0.76 for the strongest agent. What *is* robust is that success at multi-day lengths is concentrated in a minority of tasks. So "can agents do week-long tasks?" has to be answered task-class by task-class, and the critical unknown is the lower tail of the difficulty distribution among long tasks. We give a concrete protocol for measuring it.

We do not run new agents. All empirical results are re-analyses of METR's public run logs.

## 2 Related Work

**Time horizons.** Kwa et al. (2025) introduced the 50% time horizon on a suite combining HCAST (Rein et al., 2025), RE-Bench (Wijk et al., 2024) and short SWAA tasks. They fitted per-agent logistic regressions in log₂ human minutes and reported exponential growth with a ~7-month doubling. Time Horizon 1.1 expanded the suite to 228 tasks with more long tasks (METR, 2026). METR has also measured how horizons vary across domains (METR, 2025). BRIDGE (Liu et al., 2026) fits a two-parameter IRT model across benchmarks and finds that latent task difficulty is roughly linear in log human time. Nguyen (2026) replaces the linear-in-log-time difficulty map with a shared spline and IRT and reports better cross-validated scores. Two LessWrong analyses fit Bayesian IRT models with task-difficulty spread. One argues that "typical-task" and "marginal" horizons differ by roughly an order of magnitude at 80% success. The other finds that tasks of equal length are solved by the frontier months apart. Our work is complementary to all of these. We frame the problem in survival terms (hazard shape versus frailty), prove the identification results, provide an IRT-free nonparametric heterogeneity index based on replicate runs, and test retry predictions on held-out tasks.

**Survival views of agent reliability.** Ord (2025) proposed the constant-hazard ("half-life") model. Ord (2026) reports Hamilton's Weibull fits (*k* ≈ 0.6 for all agents) and notes that the Weibull and log-logistic fits differ by a factor of about 20 at the 99% horizon. We show that this ambiguity is the pooled-data face of a deeper non-identification, and that replicates resolve part of it.

**Long-horizon execution.** Sinha et al. (2025) isolate execution from planning with tasks whose per-step difficulty is constant. They show that small gains in per-step accuracy compound into large gains in achievable length, and identify *self-conditioning*: models become more error-prone once their own earlier errors are in context. That is an *increasing* within-task hazard, which matches our conditional estimate *k* > 1. Vending-Bench (Backlund and Petersson, 2025) documents agents derailing in long-running business simulations, with high variance across runs. TheAgentCompany (Xu et al., 2024) finds that long, multi-step workplace tasks remain mostly unsolved. Kambhampati et al. (2024) argue for external verifiers ("LLM-Modulo") in planning, an intervention whose value we show depends on *H*.

**Frailty and marginal versus conditional models.** Vaupel et al. (1979) showed that ignoring heterogeneity in frailty distorts inferred individual ageing rates. Proschan (1963) showed that mixtures of exponentials have decreasing failure rate. Zeger, Liang and Albert (1988) formalised how population-averaged (marginal) and subject-specific (conditional) regression coefficients differ in non-linear models with random effects. Our Proposition 2 is the survival-shape version of that attenuation. The unbiased pass@k estimator we use is from Chen et al. (2021).

## 3 Theory

**Setup.** A fixed agent attempts tasks *i* with human length *T*ᵢ. Conditional on the task, runs are i.i.d. Bernoulli with success probability *p*ᵢ = exp(−Λᵢ(*T*ᵢ)), where Λᵢ is the task's cumulative hazard. Λᵢ is a summary of how failure risk accumulates over the work the task demands. Write the *pooled curve* as *S̄*(*T*) = E[*p*ᵢ | *T*ᵢ = *T*]. This is what a logistic, Weibull or exponential fit to pooled data estimates. The *frailty* specification is Λᵢ(*T*) = *Z*ᵢ Λ₀(*T*), with *Z*ᵢ independent of *T*ᵢ and Λ₀(*T*) = *cT^k*. Here *k* is the task-level Weibull shape: *k* = 1 is constant hazard, *k* < 1 declining hazard, *k* > 1 increasing hazard.

**Proposition 1 (pooled curves cannot reveal within-task hazard dynamics).** Let every task have a constant hazard, Λᵢ(*T*) = λᵢ*T*, with λᵢ ~ *F* independent of *T*ᵢ. Then *S̄*(*T*) = ∫ e^(−λ*T*) d*F*(λ) is completely monotone (CM), and the pooled hazard −d log *S̄*/d*T* is non-increasing. Conversely, every CM survival function with *S̄*(0⁺) = 1 arises this way for a unique *F*. In particular, for every 0 < *k* ≤ 1 the pooled Weibull curve exp(−(*T*/τ)^*k*) is produced exactly by a population of constant-hazard tasks.

*Proof.* *S̄* is a Laplace transform of a probability measure and hence CM. The converse is Bernstein's theorem (Feller, 1971, Ch. XIII). The pooled hazard of a mixture of exponentials is non-increasing (Proschan, 1963). It equals E[λ | survival to *T*], whose derivative is −Var[λ | survival] ≤ 0. For the Weibull case, *x* ↦ e^(−*x*) is CM and *x* ↦ *x^k* is a Bernstein function for 0 < *k* ≤ 1, and a CM function composed with a Bernstein function is CM (Feller, 1971, Ch. XIII.4). ∎

*Consequence.* Hamilton's pooled finding *k* ≈ 0.6 (Ord, 2026) is equally consistent with (a) homogeneous tasks whose hazard declines over the course of each task and (b) heterogeneous tasks each with constant hazard. Single-run pooled data cannot separate them. A pooled log-logistic curve 1/(1 + (*T*/*h*)^β) with β > 1 has a hazard that first increases, so it is *not* CM and cannot come from constant-hazard frailty alone. In our fits β exceeds 1 for only 2 of 20 agents, and only barely (Sec. 5.1).

**Proposition 2 (frailty attenuates the pooled shape).** Under the frailty specification with *Z* non-degenerate, the local pooled shape κ(*T*) := d log(−log *S̄*(*T*)) / d log *T* satisfies κ(*T*) < *k* for all *T* > 0.

*Proof.* Let *x* = *cT^k* and *g*(*x*) = −log E[e^(−*Zx*)], so −log *S̄* = *g*(*x*). Then −*g*(−*u*) is the cumulant generating function of *Z*, which is convex, so *g* is concave. It is strictly concave when *Z* is non-degenerate, and *g*(0) = 0. For a strictly concave *g* with *g*(0) = 0 and *x* > 0, *g*(*x*) > *x g*′(*x*). Hence κ(*T*) = *k* · *x g*′(*x*)/*g*(*x*) < *k*. ∎

So a pooled Weibull fit *underestimates* the task-level shape, by an amount that grows with the spread of *Z*. This is the survival analogue of the attenuation of marginal coefficients in random-effects models (Zeger et al., 1988). It is the reason the pooled *k* ≈ 0.6 says little about within-task dynamics.

**Proposition 3 (replicates identify heterogeneity).** Suppose each task has *n*ᵢ ≥ 2 conditionally i.i.d. runs with *s*ᵢ successes. Then *s*ᵢ(*s*ᵢ − 1)/(*n*ᵢ(*n*ᵢ − 1)) is unbiased for *p*ᵢ². Among tasks of (approximately) equal length *T*, the index

  *H*(*T*) = Var(*p*ᵢ | *T*) / [*S̄*(*T*)(1 − *S̄*(*T*))] ∈ [0, 1]

is therefore identified. *H* = 0 if and only if all tasks of length *T* share one success probability, as in any homogeneous model whatever its within-task hazard shape. *H* = 1 if and only if every task is solved always or never.

*Proof.* E[*s*(*s* − 1)] = *n*(*n* − 1)*p*² for a binomial, which gives unbiasedness. The bounds follow from 0 ≤ Var(*p*) ≤ E[*p*] − E[*p*]², with equality on the right exactly when *p* ∈ {0, 1} almost surely. ∎

*H* is the intraclass correlation of run outcomes within tasks. Pooled single-run data identify only E[*p* | *T*]. This is why Propositions 1 and 2 do not bind once replicates are used.

**Proposition 4 (heterogeneity limits retries).** Let pass@*k*(*T*) = 1 − E[(1 − *p*ᵢ)^*k* | *T*] be the success rate of best-of-*k* with a perfect verifier. Then

  pass@2 = [1 − (1 − *S̄*)²] − *H* · *S̄*(1 − *S̄*),

so the gain from a second attempt equals (1 − *H*) times its value under homogeneity. For all *k*, pass@*k* ≤ 1 − (1 − *S̄*)^*k*, and pass@*k* → P(*p*ᵢ > 0 | *T*) as *k* → ∞.

*Proof.* Expand E[1 − (1 − *p*)²] = 2E[*p*] − E[*p*²] and substitute E[*p*²] = *S̄*² + *H S̄*(1 − *S̄*). The inequality is Jensen's, since (1 − *p*)^*k* is convex in *p*. The limit follows by monotone convergence. ∎

**Remark (the tail problem at multi-day lengths).** Under frailty, *S̄*(*T*) = E[exp(−*Z cT^k*)]. As *T* grows, this is controlled by the mass of *Z* near zero, i.e. by the fraction of tasks that are "easy for their length". Data on *T* ≤ *T*_max constrain the distribution of *Z* only through the values *Z cT^k* takes over that range. Two frailty laws that agree closely for *Z* ≳ (*c T*_max^*k*)^(−1) can therefore differ arbitrarily in *S̄*(*T*) for *T* ≫ *T*_max. Sec. 5.7 shows that this is not hypothetical for current data.

## 4 Method

**Data.** We use METR's public Time Horizon 1.1 run logs (`reports/time-horizon-1-1/data/raw/runs.jsonl` at commit `52cb829` of `METR/eval-analysis-public`), excluding the human-baseline rows. This leaves 23,235 runs by 20 agents, from GPT-4 0314 to Claude Opus 4.6 and GPT-5.3-Codex, on 228 tasks in 79 families. Human task lengths range from 0.02 to 1,800 minutes. 55 tasks are ≥ 4 h and only 5 are ≥ 16 h. 162 tasks have lengths from human baselines and the rest from estimates. The outcome is METR's binarised score. Agent-task pairs have a median of 4 runs, ≥ 2 runs for 99.96% of pairs and ≥ 4 for 98.3%. 4.3% of runs ended with a usage-limit error. We keep them, as METR does, and check robustness to dropping them. For robustness we also use TH1.0 (33 agents). [All counts: computed by our code.]

**Pooled families (Sec. 5.1).** Per agent, we fit five two-parameter (or one-parameter) survival families to run-level outcomes with equal task weights (each task's runs sum to weight 1): log-logistic (METR's form), Weibull, exponential (Ord's constant hazard), Lomax (gamma frailty over constant hazards) and log-normal. We also fit the full CM class by nonparametric maximum likelihood over mixtures of exponentials (EM on an 81-point hazard grid). This is the "pure constant-hazard frailty" model of Proposition 1. We compare in-sample NLL, 5-fold task-grouped cross-validated NLL, and an extrapolation test: train on tasks < 2 h, test on tasks ≥ 4 h, for the 17 agents with any long-task success.

**Heterogeneity index (Sec. 5.2).** Per agent we stratify tasks by length into <1 m, 1–4 m, 4–15 m, 15–60 m, 1–4 h, 4–16 h and >16 h. We keep strata with ≥ 8 tasks and pooled success in (0.03, 0.97), which leaves 89 agent-strata. In each stratum we estimate *H* with the unbiased second-moment estimator of Proposition 3, using cross-task products for *S̄*², and compute 95% task-bootstrap intervals (500 resamples). To check that within-stratum *length* variation does not inflate *H*, we compute the share of variance that the agent's pooled logistic curve attributes to length differences within the stratum (*H*_len).

**Frailty GLMMs (Sec. 5.3).** For each agent we fit by maximum likelihood

  P(success of a run on task *i* in family *f*) = exp(−exp(μ + *k* log(*T*ᵢ/60) + σ_f *v*_f + σ_t *u*ᵢ)),  *v*_f, *u*ᵢ ~ N(0, 1),

a complementary log-log model whose task-conditional cumulative hazard is ∝ *T^k*. Integration uses nested Gauss–Hermite quadrature (15 × 25 nodes). We compare *k* with the pooled *k*_marg obtained when σ_f = σ_t = 0. Wald standard errors come from a numerical Hessian. The logit-link analogue gives the conditional and marginal METR slopes.

**Robustness (Sec. 5.4).** We re-estimate *k* with task-level random effects only, under: Gaussian frailty; Gaussian frailty whose log-s.d. is linear in log *T*; a three-point latent-class frailty (a nonparametric-style alternative to Gaussian); dropping usage-limit runs; baseline-timed tasks only; and TH1.0.

**Retry test (Sec. 5.5).** Five-fold task-grouped CV per agent. We fit the frailty model and the homogeneous pooled Weibull on 4/5 of tasks. On held-out tasks with ≥ 4 runs we predict pass@*k* (*k* = 1–4) from task length alone and compare with the unbiased empirical pass@*k* of Chen et al. (2021).

**Identification simulation (Sec. 5.6).** On the exact design of one agent (Claude 4 Opus: 228 tasks, 1,368 runs) we simulate three truths with matched pooled 50% horizons. A: *k* = 1, σ = 2.5. B: *k* = 0.63, σ = 0. C: *k* = 1.5, σ = 3.2, close to our empirical medians. We run 30 replicates each and apply every estimator.

## 5 Results

### 5.1 Pooled curves cannot choose a hazard story

[Source: `results/marginal_summary.txt`, `results/marginal_fits.csv`, `results/extrapolation.csv`.]

**Table 1.** Pooled families. Mean per-task NLL over 20 agents (equal task weights). "CV wins" counts agents for which the family has the lowest CV NLL. Extrapolation: train < 2 h, test ≥ 4 h, 17 agents.

| Family | In-sample NLL | 5-fold CV NLL | CV wins | Extrap. test NLL (mean) | Extrap. mean signed error in success rate |
|---|---|---|---|---|---|
| CM mixture (constant-hazard frailty, NPMLE) | 0.316 | **0.325** | **11** | 0.599 | +0.180 |
| log-logistic (METR) | 0.324 | 0.331 | 3 | 0.596 | +0.203 |
| log-normal | 0.324 | 0.332 | 0 | 0.638 | +0.236 |
| Weibull | 0.326 | 0.333 | 3 | **0.569** | +0.152 |
| Lomax (gamma frailty) | 0.326 | 0.334 | 3 | 0.777 | +0.326 |
| exponential (constant hazard, no frailty) | 0.374 | 0.382 | 0 | 0.704 | −0.131 |

Three findings follow. First, the single-hazard exponential model is clearly worst (CV NLL 0.382 against 0.325–0.334). Second, the CM class, i.e. tasks with constant hazards that differ from task to task, has the best cross-validated fit for 11 of 20 agents. By Proposition 1 this means pooled data are fully consistent with *no* within-task hazard dynamics at all. Third, the two-parameter families are close in CV NLL (0.331–0.334) yet differ widely in their tails. The median ratio *h*₉₉/*h*₅₀ is 0.0010 (Weibull), 0.0038 (log-logistic), 0.0029 (CM), 0.0066 (log-normal) and 0.0145 (exponential), a 15-fold range. This reproduces the concern in Ord (2026). Pooled Weibull shapes range from 0.46 to 0.78 across agents (median 0.61 in this weighting). The log-logistic β exceeds 1 only for GPT-4 Turbo (1.006) and o3 (1.04), so the CM condition of Proposition 1 is essentially never violated. In the short-to-long extrapolation test, every family except the exponential *over*-predicts long-task success, by 0.15–0.33. The exponential under-predicts by 0.13.

### 5.2 Replicate runs reveal large between-task heterogeneity

[Source: `results/heterogeneity_summary.txt`, `results/fig_heterogeneity.png`.]

Across the 89 eligible agent-strata, the median *H* is **0.61** (IQR 0.49–0.75). The 95% bootstrap interval excludes 0 for 81% of them. Medians by stratum are: 1–4 m 0.67, 4–15 m 0.72, 15–60 m 0.47, 1–4 h 0.68, 4–16 h 0.67. Strata < 1 m and > 16 h had too few eligible tasks. The share of variance explained by length differences *within* a stratum, *H*_len, is only 0.008–0.027 on average, so *H* is not an artefact of the stratum width. Roughly 60% of run-to-run outcome variance at a given task length is therefore *which task* it is, and about 40% is run-level stochasticity. There is no sign that *H* falls for long tasks (median 0.67 at 4–16 h). The 4–16 h stratum is exactly where a compounding-error account in which all long tasks are alike would predict *H* near 0.

Proposition 4 then predicts muted retry gains. Summed over agent-strata, the observed gain of pass@2 over pass@1 is 5.71 success-rate units against 14.17 predicted by homogeneity, a ratio of **0.40** = 1 − *H̄* (Fig. `fig_heterogeneity.png`b).

### 5.3 Conditional versus pooled hazard shape

[Source: `results/frailty_summary.txt`, `results/kcond_with_se.csv`, `results/fig_main.png`a.]

**Table 2.** Weibull shape from the frailty GLMM (nested family + task effects, cloglog link) against the pooled shape (same likelihood, no random effects). Selected agents. All 20 are in `results/frailty_glmm.csv`.

| Agent | pooled *k*_marg | conditional *k* (± 1.96 s.e.) | σ_task | σ_family | LR stat. (RE vs none) |
|---|---|---|---|---|---|
| GPT-4 0314 | 0.65 | 1.38 ± 0.37 | 1.39 | 1.41 | 225 |
| GPT-4o | 0.51 | 0.97 ± 0.25 | 1.32 | 1.30 | 147 |
| o1-preview | 0.61 | 1.04 ± 0.21 | 1.21 | 0.99 | 243 |
| Claude 3.7 Sonnet | 0.61 | 1.64 ± 0.60 | 2.62 | 2.54 | 251 |
| o3 | 0.79 | 1.69 ± 0.58 | 2.12 | 2.25 | 158 |
| GPT-5 | 0.67 | 2.02 ± 0.68 | 3.03 | 2.21 | 222 |
| Gemini 3 Pro | 0.73 | 2.05 ± 0.85 | 2.02 | 3.60 | 212 |
| GPT-5.2 | 0.73 | 1.98 ± 0.82 | 3.70 | 3.54 | 260 |
| Claude Opus 4.6 | 0.51 | 1.66 ± 0.48 | 4.20 | 3.70 | 480 |
| GPT-5.3-Codex | 0.64 | 1.38 ± 0.30 | 3.83 | 3.18 | 540 |
| **Median (20 agents)** | **0.63** | **1.46** | — | — | **223** |

Random effects improve the likelihood enormously for every agent. The LR statistic is at least 143 on 2 extra parameters. The conditional shape exceeds the pooled shape for **20/20** agents, as Proposition 2 requires if frailty is real. Its median is **1.46**. The Wald interval lies entirely above 1 for 12/20 agents and includes 1 for the other 8. No agent's interval lies below 1. On the logit scale the conditional METR slope (median 2.13 per natural-log unit) is a median 2.5 times the pooled slope (median 0.85). For a *typical* task, then, success falls off with length far more steeply than the pooled curve suggests. The pooled curve is flattened by a mix of easy and hard tasks at every length.

Interpreted, the task-level evidence points to constant or *increasing* hazard (*k* ≥ 1), not a declining one. Doubling a typical task's length multiplies its cumulative hazard by 2^1.46 ≈ 2.75 rather than 2^0.63 ≈ 1.55. This agrees with the self-conditioning mechanism of Sinha et al. (2025), in which accumulated errors raise later error rates.

**Capability trend.** [Source: `results/trend_and_sensitivity.txt`, `results/fig_trend.png`.] Across agents, log pooled *h*₅₀ correlates with conditional *k* (*r* = 0.68) and with log total frailty s.d. (*r* = 0.87, n = 20). More capable agents show *sharper* task-level transitions and a *wider* spread of difficulty in log-hazard units. Their success is increasingly bimodal across tasks, so pooled curves flatten even as typical-task curves steepen. This is descriptive and confounded with release date and scaffold changes.

### 5.4 Robustness

[Source: `results/robustness_summary.txt`.]

**Table 3.** Conditional Weibull shape *k* under alternative specifications (task-level random effect only). Medians over agents.

| Variant | agents | pooled *k*_marg | conditional *k* (IQR) | share *k* > 1 | share *k* > *k*_marg |
|---|---|---|---|---|---|
| Gaussian frailty (base) | 20 | 0.63 | 1.56 (1.34–1.85) | 1.00 | 1.00 |
| log-s.d. linear in log *T* | 20 | 0.63 | 1.80 (1.27–2.08) | 1.00 | 1.00 |
| 3-class latent frailty | 20 | 0.63 | 1.18 (1.01–1.32) | 0.75 | 0.95 |
| drop usage-limit runs | 20 | 0.62 | 1.50 (1.37–1.85) | 1.00 | 1.00 |
| baseline-timed tasks only | 20 | 0.71 | 1.38 (1.21–1.62) | 1.00 | 1.00 |
| TH1.0 data | 33 | 0.70 | 1.52 (1.31–2.12) | 0.94 | 1.00 |

The qualitative conclusion, conditional *k* ≥ 1 and well above pooled *k*, holds under every specification. The size of the effect is sensitive to the shape of the frailty distribution. A discrete three-class frailty, which can put mass on "always solvable" and "never solvable" classes, gives *k* ≈ 1.2. The heteroscedastic variant finds no systematic change of frailty spread with length (median slope of log-s.d. on log *T* = −0.04; median LR = 1.9 for one extra parameter). The three-class model fits *worse* than the Gaussian (median LR −3.7), so we treat 1.2–1.8 as the plausible range and 1.46–1.56 as the central estimate.

### 5.5 Retry predictions on held-out tasks

[Source: `results/passk_heldout_summary.txt`; in-sample version `results/passk_validation.csv`.]

**Table 4.** Held-out-task pass@*k*, mean over 20 agents of agent-level means. Predictions use only the held-out task's human length.

| *k* | observed | frailty model | homogeneous model | observed gain over *k* = 1 | frailty gain | homogeneous gain |
|---|---|---|---|---|---|---|
| 1 | 0.652 | 0.654 | 0.656 | — | — | — |
| 2 | 0.692 | 0.693 | 0.760 | 0.040 | 0.039 | 0.103 |
| 3 | 0.712 | 0.710 | 0.805 | 0.061 | 0.055 | 0.148 |
| 4 | 0.726 | 0.719 | 0.831 | 0.074 | 0.065 | 0.175 |

The mean absolute agent-level error is 0.008–0.013 for the frailty model and 0.068–0.105 for the homogeneous model at *k* = 2–4. At *k* = 1 the homogeneous pooled curve is slightly better (0.006 against 0.013). The homogeneous model overstates the gain from retries by 2.6× (*k* = 2), 2.4× (*k* = 3) and 2.3× (*k* = 4). The gap is largest where it matters for long-horizon work. On held-out tasks ≥ 4 h, pass@4 is **0.30** observed, **0.30** under frailty and **0.47** under homogeneity, with pass@1 at 0.22 observed.

### 5.6 The method recovers the truth where pooled fits cannot

[Source: `results/simulation_summary.txt`, `results/fig_identification.png`.]

**Table 5.** Simulation on METR's design (30 replicates; mean ± s.d.).

| Truth | pooled *k*_marg | *H* | conditional *k* | σ̂ |
|---|---|---|---|---|
| A: *k* = 1, σ = 2.5 | 0.46 ± 0.04 | 0.58 ± 0.05 | 1.00 ± 0.11 | 2.49 ± 0.32 |
| B: *k* = 0.63, σ = 0 | 0.64 ± 0.03 | 0.02 ± 0.03 | 0.65 ± 0.03 | 0.08 ± 0.10 |
| C: *k* = 1.5, σ = 3.2 | 0.55 ± 0.05 | 0.65 ± 0.05 | 1.47 ± 0.18 | 3.07 ± 0.38 |

Scenarios B and C give similar pooled shapes (0.64 and 0.55), and both are in the range seen in real data. The underlying processes are opposite: a declining hazard in homogeneous tasks against an increasing hazard in heterogeneous tasks. *H* and the GLMM separate them sharply and recover *k* and σ without visible bias. The empirical values (pooled 0.63, *H* ≈ 0.61, conditional 1.46, σ ≈ 3.2) match scenario C, not B.

### 5.7 What this implies for day-, week- and month-long tasks

[Source: `results/projections.csv`, `results/projection_sensitivity.csv`. All values below are *extrapolations* beyond the 30-hour maximum task length and assume the TH1.1 task population.]

**Table 6.** Projected success on a random task of length *T* drawn from a METR-like population, for frontier agents, under different models. Gaussian and 3-class are frailty GLMMs; the rest are pooled fits.

| Agent | *T* | Gaussian frailty | 3-class frailty | pooled log-logistic | pooled Weibull | pooled CM | pooled exponential |
|---|---|---|---|---|---|---|---|
| GPT-5.3-Codex | 8 h | 0.59 | 0.55 | 0.48 | 0.47 | 0.45 | 0.39 |
| GPT-5.3-Codex | 40 h | 0.41 | 0.27 | 0.24 | 0.14 | 0.12 | 0.01 |
| GPT-5.3-Codex | 160 h | 0.28 | 0.01 | 0.11 | 0.01 | 0.09 | 0.00 |
| Claude Opus 4.6 | 8 h | 0.53 | 0.79 | 0.58 | 0.57 | 0.54 | 0.48 |
| Claude Opus 4.6 | 40 h | 0.35 | 0.76 | 0.36 | 0.28 | 0.30 | 0.03 |
| Claude Opus 4.6 | 160 h | 0.22 | 0.72 | 0.20 | 0.08 | 0.24 | 0.00 |
| GPT-5.2 | 40 h | 0.25 | 0.33 | 0.19 | 0.09 | 0.21 | 0.01 |

Three conclusions follow.

(a) **The population-average week-long success rate is not identified by current data.** For the strongest agents, plausible models give 0.01–0.76 at 40 h. The spread is driven by the lower tail of the difficulty distribution (Remark, Sec. 3). The three-class model for Claude Opus 4.6 even places 62% of tasks in a class whose success hardly decays with length, which is an obvious extrapolation artefact. The suite contains only five tasks ≥ 16 h, so it cannot discipline this tail. This agrees with METR's own caution that horizon measurements above 16 hours are unreliable on the current suite (METR, 2026).

(b) **The *typical* week-long task is out of reach. Whatever success exists sits in a minority of tasks.** Under Gaussian frailty, the success probability of the median-difficulty 40-hour task is ≤ 0.005 for 19 of 20 agents. GPT-5.3-Codex is the exception at 0.18. At the same time, 29% (Claude Opus 4.6) and 34% (GPT-5.3-Codex) of 40-hour tasks are predicted to be solved with *p*ᵢ ≥ 0.8, and 54% and 46% to have *p*ᵢ ≤ 0.05 (Fig. `fig_main.png`c). Pooled curves average over these two groups.

(c) **Retries and best-of-*k* help far less than homogeneous models imply.** For Claude Opus 4.6 at 40 h, the projected pass@8 is 0.45 under frailty against 0.93 under the homogeneous pooled Weibull. For GPT-5.2 it is 0.34 against 0.51.

## 6 Discussion

**What has been learned about "can models do days-to-weeks tasks?"** The current curve-fitting debate asks which pooled shape is right. Our results suggest that this framing hides the decisive variable. At the level of a task of fixed relative difficulty, risk accumulates at least linearly with length (*k* ≥ 1). Agents do not "settle in". If anything, typical tasks get disproportionately harder to finish as they get longer, as self-conditioning would predict (Sinha et al., 2025). The pooled curve is flatter, and its implied hazard declines, because a population that mixes tasks of different difficulty contains some long tasks that are easy for the agent. Whether a given *kind* of week-long task is feasible therefore depends on where that kind sits in the difficulty distribution. The fraction of week-long tasks within reach is set by the lower tail of that distribution, which current suites barely sample.

**Implications for horizon reporting.** (1) Report *H* or a frailty s.d. next to the horizon. Two agents with equal *h*₅₀ can differ greatly in how predictable their failures are. (2) Report typical-task and population-average horizons separately, as the LessWrong IRT analysis also urges. (3) Do not translate a pooled *k* or a pooled curve into per-task or per-minute reliability. Proposition 2 guarantees that this understates how sharply a given task's reliability falls with its length. (4) Use pass@*k* or best-of-*k* scaffold results only with a heterogeneity-aware model.

**Implications for agent design.** Proposition 4 sets out the economics. When *H* is large, re-running the same agent on the same long task buys little, since most failures are tied to the task and not to chance. Interventions that change the agent-task interaction should be worth more. Decomposition into verified sub-goals changes the effective Λᵢ. External verification (Kambhampati et al., 2024) and memory or handoff mechanisms for multi-session work change the effective task too. When *H* is small, sampling and voting are effective. Run-level and task-level failure modes call for different engineering, and *H* says which one dominates.

**Pre-registered test protocol for multi-day horizons.** The tail problem can be solved with data. We propose the following, with predictions stated before the data exist. (P1) Build a set of ≥ 40 tasks of 16–80 human hours in ≥ 10 families, sampled to *span* rather than cherry-pick difficulty. (P2) Run each frontier agent ≥ 4 times per task. (P3) Estimate *H* and the frailty GLMM jointly with the existing suite. **Predictions** from our TH1.1 fits: (i) *H* in the 16–80 h stratum will be ≥ 0.5 for any agent whose stratum success lies in (0.1, 0.9); (ii) the conditional *k* re-estimated with the long tasks included will remain ≥ 1; (iii) observed pass@4 − pass@1 on these tasks will be at most half of the homogeneous prediction 1 − (1 − *S̄*)⁴ − *S̄*; (iv) the across-task distribution of per-task success will be bimodal, with more than a third of tasks never solved in 4 runs whenever the stratum mean is below 0.5. Any of (i)–(iii) failing would show that long-task failures are dominated by run-level stochasticity, as a homogeneous compounding model assumes. That would favour the pooled extrapolations. A within-run complement would also help: logging time-stamped irrecoverable-error events in agent trajectories would allow a direct estimate of within-run hazard, which no outcome-only analysis, ours included, can give.

## 7 Limitations

- **Cross-sectional "conditional *k*".** Our *k* compares tasks of different lengths at the same frailty quantile. It is not a within-run hazard measured over elapsed time. It equals the within-task Weibull shape only if long tasks are, in distribution, "stretched" versions of short ones with frailty independent of length. The heteroscedastic check finds no length trend in frailty spread, but frailty *location* could still co-vary with length through suite construction (e.g., RE-Bench tasks with thresholded continuous scores). That would bias *k* in an unknown direction.
- **Distributional assumptions.** The magnitude of *k* depends on the frailty family (1.18 under three latent classes against 1.46–1.80 under Gaussian variants). We have not fitted a fully nonparametric frailty jointly with *k*. *H* is nonparametric but uses wide length strata. Within-stratum length variation explains under 3% of variance by our check, but that check uses a fitted curve.
- **Errors in human time.** 66 of 228 task lengths are estimates rather than baselines, and baselines themselves are noisy. Errors in *T* bias slope parameters toward zero in both pooled and conditional fits. Restricting to baselined tasks gives *k* = 1.38, still well above 1.
- **Replicate independence.** We assume runs are conditionally independent given the task. Shared infrastructure failures, the 4.3% usage-limit terminations, or scaffold changes within an agent's runs could create spurious within-task correlation. Dropping usage-limit runs does not change the results.
- **Extrapolation.** All multi-day numbers in Sec. 5.7 extrapolate beyond the longest task (30 h), and only 5 tasks are ≥ 16 h. Section 5.7 is a statement about *non-identification* and about the robust bimodality. It is not a forecast.
- **Scope.** The data cover software, ML-engineering and cybersecurity tasks with automatic scoring, chosen by METR. Real multi-day work involves interruptions, coordination and changing specifications, none of which is represented. The capability correlations in Sec. 5.3 are descriptive, with n = 20 and confounded with release date and scaffold.
- **No new agent runs.** We had no GPU or model access. Every empirical claim is a re-analysis of METR's public logs.

## 8 Conclusion

Single-run pooled success-versus-length curves cannot tell whether AI agents' failure risk declines within a task or whether tasks of equal length simply differ in difficulty. Replicate runs can. On METR's Time Horizon 1.1 data, between-task heterogeneity is large (*H* ≈ 0.6). Once it is accounted for, the task-level hazard shape is constant-to-increasing (*k* ≈ 1.2–1.8) rather than declining (*k* ≈ 0.6). A frailty model predicts the limited value of retries on held-out tasks, and a homogeneous model does not. For the question of tasks spanning days and weeks, this means two things. A typical long task gets disproportionately harder with length. And the share of long tasks agents can do, set by the easy tail of the difficulty distribution, is what current suites cannot yet measure. Replicate-run measurement of long-task heterogeneity, as in the protocol of Sec. 6, is a cheap next step for anyone extrapolating time horizons to multi-day work.

## References

All references were checked to exist via web search in November 2026. Where authorship could not be verified, the work is cited by title and URL only.

- Backlund, A. and Petersson, L. (2025). Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents. arXiv:2502.15840.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Feller, W. (1971). *An Introduction to Probability Theory and Its Applications, Vol. II* (2nd ed.). Wiley. (Ch. XIII: completely monotone functions, Bernstein's theorem.)
- Kambhampati, S., Valmeekam, K., Guan, L., Verma, M., Stechly, K., Bhambri, S., Saldyt, L. and Murthy, A. (2024). Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. *Proceedings of ICML 2024*, PMLR 235:22895–22907. arXiv:2402.01817.
- Kwa, T., West, B., Becker, J., Deng, A., Garcia, K., Hasin, M., Jawhar, S., Kinniment, M., Rush, N., Von Arx, S., et al. (2025). Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499 (NeurIPS 2025).
- Liu, F., Gala, J., Nilaksh, Bahdanau, D., Reddy, S. and Larochelle, H. (2026). BRIDGE: Predicting Human Task Completion Time From Model Performance. arXiv:2602.07267.
- METR (2025). How Does Time Horizon Vary Across Domains? Blog post, 14 July 2025. https://metr.org/blog/2025-07-14-how-does-time-horizon-vary-across-domains/
- METR (2026). Time Horizon 1.1. Blog post, 29 January 2026. https://metr.org/blog/2026-01-29-time-horizon-1-1/ ; data and code: https://github.com/METR/eval-analysis-public (commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2).
- Nguyen, D. (2026). On the estimation and validity of AI time horizons—a statistical look at the METR plot. arXiv:2610.12466. (First author as announced by the author; full author list per arXiv.)
- Ord, T. (2025). Is there a half-life for the success rates of AI agents? arXiv:2505.05115.
- Ord, T. (2026). Hazard Rates for AI Agents Decline as a Task Goes On. Blog post, 4 February 2026, reporting analysis by Gus Hamilton. https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline
- Proschan, F. (1963). Theoretical explanation of observed decreasing failure rate. *Technometrics*, 5, 375–383.
- Rein, D., Becker, J., Deng, A., Nix, S., Canal, C., O'Connell, D., Arnott, P., Bloom, R., Broadley, T., Garcia, K., et al. (2025). HCAST: Human-Calibrated Autonomy Software Tasks. arXiv:2503.17354.
- Sinha, A., Arun, A., Goel, S., Staab, S. and Geiping, J. (2025). The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs. arXiv:2509.09677 (ICLR 2026).
- Valmeekam, K., Marquez, M., Olmo, A., Sreedharan, S. and Kambhampati, S. (2023). PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2206.10498.
- Vaupel, J. W., Manton, K. G. and Stallard, E. (1979). The impact of heterogeneity in individual frailty on the dynamics of mortality. *Demography*, 16(3), 439–454.
- Wijk, H., et al. (2024). RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv:2411.15114 (ICML 2025, PMLR 267).
- Xu, F. F., et al. (2024). TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. arXiv:2412.14161.
- Zeger, S. L., Liang, K.-Y. and Albert, P. S. (1988). Models for longitudinal data: a generalized estimating equation approach. *Biometrics*, 44(4), 1049–1060.
- "A(nother) look at METR's time horizon: tasks of the same duration get solved months apart." LessWrong post (author not verified). https://www.lesswrong.com/posts/qHztdHtoXixCWAr7z/
- "(Updated) METR's data can't distinguish between trajectories (and 80% horizons are an order of magnitude off)." LessWrong post (author not verified). https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T/
