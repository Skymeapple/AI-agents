# Replicates Buy Hazard, Not Length: What Repeated Runs Can and Cannot Tell Us About AI Agents on Multi-Day Tasks

**Target venue:** ICLR 2027 (main track, evaluation methodology / datasets and benchmarks area)

**Code and data:** `code/` (run `bash code/run_all.sh`); all outputs are in `results/`. Every number in this paper comes either from that code, run on METR's public run logs, or from a cited source. Each use says which.

---

## Abstract

Can AI agents complete tasks that take a human days or weeks? The usual evidence is METR's *time horizon*: a curve of agent success against human task length, fitted to a pool of tasks and extrapolated. Whether that extrapolation is safe depends on how failure risk grows with length. Pooled curves cannot answer this, because a flattening curve can come from a hazard that declines within tasks or from tasks of equal length that differ in difficulty ("frailty"). Others have made this second point before us. We ask a sharper question: what do the *replicate runs* in public agent logs identify, and is that enough to forecast success on longer tasks?

**Theory.** (i) Outcome data cannot identify how risk accumulates *within* a task, even with infinitely many replicates. They identify only the distribution of per-task success at each length. Any claim about hazard shape, ours included, therefore rests on an invariance assumption: that the law of frailty does not change with length. (ii) Under that assumption we prove a **replicate–length duality**. The probability of succeeding in all *j* runs of tasks of length *T* (τ-bench's pass^*j*) equals single-run success on tasks whose cumulative hazard is *j* times larger. Mathematically this is the diagonal of the Archimedean copula that a shared frailty induces (Oakes, 1989). It has three uses. It identifies the hazard-length map without any assumption about the frailty distribution. It turns heterogeneity models into falsifiable forecasts with likelihood-ratio bounds. And it shows that replicates extend what is identified in *hazard*, but not in *length*. (iii) A closed-form, sharp bracket on best-of-*m* success, given the heterogeneity index *H*.

**Evidence.** We analyse 23,235 runs by 20 agents on 228 tasks (METR Time Horizon 1.1).

- Heterogeneity is large and mostly task-intrinsic. Median *H* = 0.61 within length strata. About 80% of between-task log-hazard variance is shared across agents.
- The hazard-length elasticity is above 1 with the frailty law left unrestricted. Its median is 1.77, and 17 of 20 agents have a profile CI above 1.
- **The decisive test fails, and it fails at a specific place.** Models calibrated on tasks under 1 h, or under 2 h, overpredict success on longer tasks. This holds for every model, the nonparametric frailty model included: +0.15 on 1–4 h tasks, with family-bootstrap 95% CI [+0.05, +0.26]. The observed rate lies outside the model's 95% bounds for 6 of 20 agents. A quarter to a half of the bias comes from never-solved tasks whose lengths are mostly estimated. With those removed, the bias is +0.11 on 1–4 h tasks (CI [+0.00, +0.23]) and +0.07 on all tasks ≥ 1 h (CI [+0.01, +0.13]). Calibrated on tasks under 4 h, the same models predict 4–30 h tasks without bias: +0.01, CI [−0.06, +0.08]. Across all 20 agents, a piecewise hazard map is preferred, with a steep segment between 1 and 4 h (joint p = 2×10⁻⁶). In simulation, the forecast test returns errors of −0.09 to +0.02 when invariance holds. A kink in the hazard-length map reproduces the observed +0.12.
- On TH1.1, then, the binding obstacle to extrapolation is not between-task heterogeneity, which replicates measure, but a change in how difficulty scales with length, which only data at the target lengths can reveal.
- Where invariance does hold (4–30 h), 95% bounds on the four strongest agents' success on a random 40-hour task lie within [0.02, 0.42]. At 160 hours every lower bound is ≤ 0.007.

## 中文摘要

AI 智能体能否完成人类需要数天乃至数周的任务？目前的主要证据是 METR 的"时间跨度"（time horizon）：在一个任务池上拟合智能体成功率随人类完成时长变化的曲线，再向更长任务外推。外推是否可靠，取决于失败风险如何随任务时长增长。仅凭合并曲线无法回答这个问题：曲线变平，既可能是因为单个任务内部的风险率在下降，也可能是因为同样时长的任务难度不同（生存分析中称为 frailty，即异质性）。后一点前人已经指出。本文提出一个更尖锐的问题：公开日志中的**重复运行**到底能识别什么？这些信息是否足以预测更长任务上的成功率？

**理论。**（i）只用结果数据，即使重复次数无限，也无法识别单个任务*内部*的风险累积方式，能识别的只是各时长下"每个任务成功概率"的分布。因此，任何关于风险形状的结论（包括本文的结论）都依赖一个不变性假设：异质性的分布不随时长改变。（ii）在这一假设下，我们证明了**"重复—时长对偶"**：长度为 *T* 的任务上 *j* 次运行全部成功的概率（即 τ-bench 的 pass^*j*），等于累积风险为其 *j* 倍的任务上的单次成功率。数学上，这是共享 frailty 所诱导的阿基米德 copula 的对角线（Oakes，1989）。它有三个用途：不对异质性分布作任何假设即可识别"风险—时长映射"；把异质性模型变成可证伪的预测，并给出似然比界；并且表明重复运行扩展的是可识别的*风险*范围，而不是*时长*范围。（iii）给定异质性指数 *H*，我们给出 best-of-*m* 成功率的闭式锐界。

**证据。** 我们分析了 METR Time Horizon 1.1 中 20 个智能体在 228 个任务上的 23,235 次运行。

- 异质性很大，且主要是任务本身的属性：按时长分层后 *H* 的中位数为 0.61；任务间对数风险方差中约 80% 为各智能体共有。
- 在不限定异质性分布的情况下，风险—时长弹性大于 1：中位数 1.77，20 个智能体中 17 个的轮廓置信区间完全在 1 以上。
- **决定性检验没有通过，而且失败在一个明确的位置。** 用 1 小时（或 2 小时）以内任务校准的所有模型，包括非参数 frailty 模型，都高估更长任务的成功率：在 1–4 小时任务上高估 0.15（任务族 bootstrap 95% 区间 [0.05, 0.26]），20 个智能体中有 6 个的观测值落在模型 95% 界之外。四分之一到一半的偏差来自从未被解决、且时长多为估计值的任务；去掉这些任务后，1–4 小时任务上的偏差为 0.11（区间 [0.00, 0.23]），全部 1 小时以上任务上为 0.07（区间 [0.01, 0.13]）。用 4 小时以内任务校准时，对 4–30 小时任务的预测没有偏差：+0.01，区间 [−0.06, 0.08]。对全部 20 个智能体而言，分段风险映射更受支持，其中 1–4 小时之间有一段陡升（联合 p = 2×10⁻⁶）。模拟显示：当不变性成立时，该检验的预测误差在 −0.09 到 +0.02 之间；风险—时长映射中的一个拐折则能复现观测到的 +0.12。
- 因此，在 TH1.1 上，阻碍外推的主要不是任务间异质性（重复运行可以测量它），而是难度随时长的标度方式发生了变化，这只有在目标时长上的数据才能揭示。
- 在不变性成立的区间（4–30 小时）内，最强的四个智能体在随机 40 小时任务上成功率的 95% 界都落在 [0.02, 0.42] 之内；到 160 小时，所有智能体的下界都不超过 0.007。

---

## 1 Introduction

METR's time-horizon methodology is the most widely used quantitative evidence on whether AI agents can do long work (Kwa et al., 2025; METR, 2026). Agents attempt software, ML-engineering and cybersecurity tasks whose length is calibrated by human completion time (Rein et al., 2025; Wijk et al., 2024). A logistic curve in log human time is fitted per agent, and the *50% time horizon* is where predicted success is one half. The horizon has grown exponentially for six years. Extrapolating the trend is how one gets to agents that "independently complete a large fraction of software tasks that currently take humans days or weeks" (Kwa et al., 2025). METR itself warns that horizons above 16 h are unreliable on the current suite (METR, 2026), and that reasonable modelling alternatives move recent horizons substantially (METR, 2026b).

How far an extrapolation can be trusted depends on the *shape* of the success-versus-length relation. Ord (2025) proposed a constant hazard: a fixed chance of failing per minute, so that the 50% horizon is a half-life. A Weibull analysis by Gus Hamilton, reported by Ord (2026), found shape *k* ≈ 0.6 < 1 for every agent. It was read as a hazard that "declines as a task goes on". That reading is optimistic for multi-day tasks. Several public analyses have since pointed out the classical objection (Vaupel, Manton and Stallard, 1979; Proschan, 1963). A pooled curve also flattens if tasks of equal length differ in difficulty, because the hard ones drop out early. A LessWrong analysis simulated per-task constant hazards with spread difficulty and reproduced a pooled shape near 0.57 (LessWrong, 2026a). Item-response analyses showed that tasks of the same length are solved by the frontier months apart (LessWrong, 2026a; Nguyen and Fithian, 2026). An essay argued that the time-horizon curve "cannot tell a half-life from recovery" (Reasoning Under Uncertainty, 2026). A preprint frames rising, flat and cliff-shaped hazards as all compatible with one pooled curve (Mahendrakar, 2026). The round-1 version of this paper re-derived the frailty explanation and measured it with replicate runs. We now treat the frailty critique of the Lindy reading as **known**, and do not claim it.

This paper asks the next question. METR's public logs contain **replicate runs**: a median of 4 independent attempts per agent-task pair, and at least 2 for 99.96% of pairs. *Exactly what do replicates identify, and is it enough to forecast success on tasks longer than any in the suite?* The question decides whether "days or weeks" can be answered from today's data, or only from new data at those lengths.

**Contributions.**

1. **What is and is not identified (Sec. 3).**
   - *Theorem 1.* Even with infinite replicates, outcome data identify only the law of per-task success *p* given length *T*. Within-task hazard shape is identified only under an *invariance* assumption (frailty law independent of *T*). Under a location drift of frailty with log *T*, only the sum of elasticity and drift is identified. This sets an honest ceiling on what we, or anyone, can claim from outcome logs.
   - *Theorem 2 (replicate–length duality).* Under invariance, pass^*j* at length *T* equals pass@1 at the length whose cumulative hazard is *j* times larger. As a consequence, the hazard-length map is identified with the frailty law left completely unrestricted, a single elasticity must fit every *j* (an overidentifying restriction), and *n* replicates extend the identified range by a factor *n* in hazard. Converting that to length requires the hazard-length map beyond the data, which replicates cannot supply. The identity is the diagonal section of the Archimedean copula of a shared-frailty model (Oakes, 1989). We claim its use for identification and forecasting in agent evaluation, not the identity itself.
   - *Proposition 3.* A closed-form sharp bracket for best-of-*m* success given *S̄* and *H*, for every *m*. This answers a gap in the round-1 version.
2. **A falsifiable forecasting method (Sec. 4).** It combines a nonparametric-maximum-likelihood (NPMLE) frailty model with a profiled elasticity, likelihood-ratio bounds on long-task success computed as convex programs, and an out-of-window test that trains on short tasks and predicts longer ones.
3. **Findings on METR TH1.1 (Sec. 5).**
   - (a) Heterogeneity is large (*H* ≈ 0.61) and mostly shared across agents.
   - (b) The elasticity is > 1 with an unrestricted frailty law, with a median of 1.77.
   - (c) **Invariance fails between about 1 and 4 hours.** Every model trained below 1–2 h overpredicts longer tasks by 0.12–0.31. The nonparametric model is the least biased of them, yet its bounds are violated for 6–7 of 20 agents. The over-prediction shrinks, but persists, when never-solved tasks are removed. Simulations show that the test is unbiased under invariance and reproduces the observed bias under a kinked hazard map. Trained below 4 h, the models are unbiased on 4–30 h tasks.
   - (d) Under invariance beyond 4 h, 40-hour success is bounded to intervals about 0.3 wide for frontier agents, and 160-hour success is essentially unidentified.
4. **Design rule (Sec. 6).** To identify success at length *T*\* from tasks no longer than *T*_max under invariance, one needs *n* ≥ (*T*\*/*T*_max)^*k* replicates. With *k* ≈ 1.8, that is about 5 runs of 16-hour tasks for 40-hour claims. But invariance must itself be checked at the target lengths, because Sec. 5.4 shows that it can break.

All empirical results are re-analyses of METR's public run logs. We ran no agents.

## 2 Related Work

**Time horizons and their statistics.** Kwa et al. (2025) introduced the 50% time horizon on HCAST (Rein et al., 2025), RE-Bench (Wijk et al., 2024) and SWAA tasks. Time Horizon 1.1 expanded the suite to 228 tasks (METR, 2026). METR has also measured horizons across domains (METR, 2025) and has studied the sensitivity of horizons to modelling assumptions (METR, 2026b). BRIDGE (Liu et al., 2026) fits a two-parameter IRT model across benchmarks and finds latent difficulty roughly linear in log human time. Nguyen and Fithian (2026) relax linearity with splines and IRT, report better cross-validated scores, and recommend diagnostic plots, especially as suites grow to include longer tasks. Our finding of a steep segment between 1 and 4 h (Sec. 5.4) is the kind of non-linearity their spline model allows. We add a *forecasting* test of it, and a decomposition into heterogeneity versus the hazard-length map.

**Frailty critiques of pooled hazard shapes.** These were made publicly before this paper.

- *LessWrong (2026a)* simulates per-task constant hazards with spread difficulty and reproduces a pooled shape near METR's. It also documents that same-length tasks are solved months apart.
- *LessWrong (2026b)* contrasts typical-task and marginal horizons under Bayesian IRT.
- *Reasoning Under Uncertainty (2026)* argues that pooled curves cannot separate decay from recovery.
- *Mahendrakar (2026)* frames three hazard shapes under one horizon.

We could read the two LessWrong posts in full. For the essay and the preprint we had only search-engine abstracts, because their hosts were unreachable from our environment, so we cite them only for the framing visible there. None of these uses replicate runs to identify anything, and none tests a forecast out of window. That is the scope of our contribution.

**Repeated sampling and reliability.** τ-bench (Yao et al., 2024) introduced pass^*k*, the probability that all *k* trials succeed, as a reliability metric. Chen et al. (2021) gave the unbiased pass@*k* estimator. Schaeffer et al. (2025) showed that power-law scaling of pass@*k* in the number of attempts arises from a heavy left tail of per-problem success probabilities. Our duality links pass^*j* to task length, and our Prop. 3 bounds pass@*m* through the second moment, which is the quantity replicates identify.

**Long-horizon execution.** Sinha et al. (2025) isolate execution and find *self-conditioning*: models err more once their own errors are in context. That is an increasing within-run hazard, which Theorem 1 says outcome logs cannot confirm. Vending-Bench (Backlund and Petersson, 2025) and TheAgentCompany (Xu et al., 2024) document derailment and low success on long, multi-step work. PlanBench (Valmeekam et al., 2023) and LLM-Modulo (Kambhampati et al., 2024) concern planning with and without verifiers.

**Frailty, copulas and marginal versus conditional models.** Vaupel et al. (1979) and Proschan (1963) are the classical sources on frailty-induced hazard decline. Oakes (1989) showed that shared frailties induce Archimedean copulas whose generator is the frailty Laplace transform. Our Theorem 2 is the diagonal of that structure, read along the length axis. Zeger, Liang and Albert (1988) formalised marginal-versus-conditional attenuation. The Kiefer–Wolfowitz NPMLE of a mixing distribution (Kiefer and Wolfowitz, 1956) underlies our frailty estimator.

## 3 Theory

**Setup.** Fix an agent. Task *i* has human length *T*ᵢ and success probability *p*ᵢ ∈ [0, 1]. Runs are conditionally i.i.d. Bernoulli(*p*ᵢ) given the task. Write *p*ᵢ = exp(−Λᵢ), where Λᵢ ≥ 0 is the task's cumulative hazard. Define the pooled curve *S̄*(*T*) = E[*p* | *T*] and the replicate moments *M*ⱼ(*T*) = E[*p*^*j* | *T*]. *M*ⱼ is the population value of τ-bench's pass^*j*. A **multiplicative frailty model** is Λᵢ = *Z*ᵢ Λ₀(*T*ᵢ), with Λ₀ continuous, strictly increasing and Λ₀(0) = 0. **Invariance** means *Z* ⟂ *T* with law *G*. The Weibull case is Λ₀(*T*) = *cT^k*. We call *k* the *hazard-length elasticity*: the growth of log cumulative hazard per unit of log length at fixed frailty.

**Background facts (classical, stated for reference).**

- (F1) Pooled curves generated by tasks that each have a constant hazard are exactly the completely monotone survival functions. Every Weibull curve with *k* ≤ 1 is one of them (Bernstein; Feller, 1971, Ch. XIII; Proschan, 1963).
- (F2) Under invariance with non-degenerate *G*, the local pooled shape d log(−log *S̄*)/d log *T* is strictly below *k*. The reason is that *g*(*x*) = −log E e^(−*Zx*) is strictly concave with *g*(0) = 0, so *x g*′(*x*) < *g*(*x*). This is the survival form of the attenuation in Zeger et al. (1988).
- (F3) With ≥ 2 replicates, *s*(*s* − 1)/(*n*(*n* − 1)) is unbiased for *p*², so *H*(*T*) = Var(*p* | *T*)/[*S̄*(1 − *S̄*)] (the intraclass correlation of runs within tasks) is identified. Ratio estimators of it are consistent, not unbiased.

F1 and F2 are why the pooled *k* ≈ 0.6 says nothing about within-task dynamics. That point is not new (Sec. 2).

**Theorem 1 (what outcome data identify).**

- (a) The distribution of all run outcomes, for any number of replicates per task, depends on the task population only through the joint law of (*T*, *p*).
- (b) For any joint law of (*T*, *p*) with *p* ∈ (0, 1], and *any* strictly increasing Λ₀ (constant, declining or increasing hazard), the dependent frailty *Z* := −log *p* / Λ₀(*T*) reproduces it exactly.
- (c) If log *Z* | *T* = δ log *T* + ε, with ε ⟂ *T*, and Λ₀ = *cT^k*, then the law of the data depends on (*k*, δ) only through *k* + δ.

*Proof.* (a) Given *p*, runs are i.i.d. Bernoulli(*p*), so the joint law of any number of runs is a mixture over the law of (*T*, *p*). (b) By construction *Z*Λ₀(*T*) = −log *p*. (c) log Λ = log *c* + (*k* + δ) log *T* + ε. ∎

So "within-task hazard" claims, Lindy or fatigue alike, are claims about Λ₀ that outcome data reach only through invariance. Without invariance, *k* is the *across-suite* elasticity of difficulty with length. That quantity is still what matters for extrapolating to longer tasks drawn from the same kind of suite, but it is not a within-run hazard. We use *k* in that sense from now on. Theorem 1(c) is also why the drift scenario in our misspecification simulation inflates the estimated *k* (Sec. 5.6).

**Theorem 2 (replicate–length duality).** Under invariance, for every *T* and every integer *j* ≥ 1,

  *M*ⱼ(*T*) = *S̄*( Λ₀⁻¹( *j* Λ₀(*T*) ) ).

Its consequences are:

- (i) *G-free identification of the hazard-length map.* If *S̄* is strictly decreasing, the map φⱼ(*T*) := *S̄*⁻¹(*M*ⱼ(*T*)) = Λ₀⁻¹(*j*Λ₀(*T*)) is identified from (*S̄*, *M*ⱼ) alone, with *G* arbitrary. In the Weibull case φⱼ(*T*) = *j*^(1/*k*) *T*, so *k* = log *j* / log(φⱼ(*T*)/*T*) at every *T* and *j*. Constancy across *T* and *j* is an overidentifying restriction.
- (ii) *Replicates extend identification in hazard, not in length.* With *n* replicates on tasks up to *T*_max, the law of *p* is identified at every cumulative-hazard level up to *n*Λ₀(*T*_max). Which *length* that level corresponds to depends on Λ₀ beyond *T*_max. Data on tasks ≤ *T*_max do not identify that unless Λ₀ is assumed to keep its form. In the Weibull case, pooled success is identified up to *n*^(1/*k*) *T*_max.
- (iii) *Falsifiability.* Calibrated on tasks shorter than *C*, the model predicts the whole law of *p* on longer tasks with no free parameters left. Under invariance and a correct Λ₀ form, success on longer tasks must lie within the likelihood-ratio bounds of Sec. 4. A violation means that invariance or the Λ₀ form fails beyond *C*.

*Proof.* *M*ⱼ(*T*) = E[e^(−*jZ*Λ₀(*T*))] = 𝓛_G(*j*Λ₀(*T*)), where 𝓛_G is the Laplace transform of *G*. Likewise *S̄*(*T*′) = 𝓛_G(Λ₀(*T*′)). Put *T*′ = Λ₀⁻¹(*j*Λ₀(*T*)). For (i), 𝓛_G is strictly decreasing when *G* ≠ δ₀, so *S̄* is invertible on its range. (ii) follows because 𝓛_G is identified on [0, *n*Λ₀(*T*_max)] by (*M*₁, …, *M*ₙ) at *T* ≤ *T*_max. ∎

*Remarks.*

- With *k* = 1 (Ord's constant hazard at the task level), "pass^4 on 2-hour tasks equals pass@1 on 8-hour tasks". With *k* ≈ 1.8 (Sec. 5.3), it equals pass@1 on about 4.3-hour tasks.
- The identity is the diagonal of the Archimedean copula *C*(*u*, *v*) = 𝓛_G(𝓛_G⁻¹(*u*) + 𝓛_G⁻¹(*v*)) induced by a shared frailty (Oakes, 1989). Its use here is new to our knowledge: replicate runs at one length are mapped onto single runs at another length.
- Because 𝓛_G is analytic, exact knowledge of it on any interval would determine it everywhere. Analytic continuation is severely ill-posed, though, and with finite data the bounds of Sec. 4 are the operational content of (ii).

**Proposition 3 (sharp retry bracket).** Let *S̄* = E *p*, *Q* = 1 − *S̄* and *H* as in F3, all at fixed *T*. For every *m* ≥ 1,

  (*S̄*/*r*) [1 − (1 − *r*)^*m*]  ≤  pass@*m* := 1 − E(1 − *p*)^*m*  ≤  1 − *Q* (*Q* + *H S̄*)^(*m*−1),  with *r* = *S̄* + *H Q*.

Both bounds are attained, by two-point laws of *p* on {0, *r*} and on {1 − *Q* − *H S̄*, 1} respectively. For *m* = 2 they coincide: pass@2 = 1 − *Q*² − *H S̄ Q*.

*Proof.* Let *q* = 1 − *p* ∈ [0, 1] and *f*(*q*) = *q^m*, so that *f*‴ ≥ 0. The mean *Q* and the second moment E*q*² = *Q*² + *H S̄ Q* are fixed.

- *Lower bound on E f*, which gives the upper bound on pass@*m*. Take *b* = E*q*²/*Q* = *Q* + *H S̄*. Let *h* be the quadratic that interpolates *f* at 0 and has a double contact at *b*. Then *f*(*q*) − *h*(*q*) = *f*‴(ξ) *q* (*q* − *b*)²/6 ≥ 0. Hence E *f* ≥ E *h*. E *h* depends only on the first two moments and equals E *f* under the two-point law on {0, *b*}, which gives *Q b*^(*m*−1).
- *Upper bound on E f.* Use the double contact at *a* and the simple contact at 1: *f* − *h* = *f*‴(ξ)(*q* − *a*)²(*q* − 1)/6 ≤ 0. The extremal law puts *p* on {0, *r*}, and pass@*m* = (*S̄*/*r*)(1 − (1 − *r*)^*m*). ∎

Under homogeneity (*H* = 0) the upper bound is 1 − *Q^m*. Large *H* compresses both bounds toward the gain of a single retry, multiplied by (1 − *H*).

## 4 Method

**Data.** We use METR's public Time Horizon 1.1 run logs (`reports/time-horizon-1-1/data/raw/runs.jsonl` at commit `52cb829` of `METR/eval-analysis-public`), excluding human-baseline rows. This gives:

- 23,235 runs by 20 agents, from GPT-4 0314 to Claude Opus 4.6 and GPT-5.3-Codex, on 228 tasks in 79 families;
- lengths from 0.02 to 1,800 min, with 55 tasks ≥ 4 h and 5 tasks ≥ 16 h;
- 162 tasks timed by human baselines, the rest estimated;
- a median of 4 runs per pair (≥ 2 for 99.96%, ≥ 4 for 98.3%).

Usage-limit terminations (4.3% of runs) are kept, as METR does. Robustness to dropping them was checked in round 1, and the results did not change. [Counts: our code.]

**NPMLE frailty model.** For each agent, P(success | task *i*) = exp(−exp(*k* log(*T*ᵢ/60) + *u*ᵢ)), with *u*ᵢ ~ *G* and *G* unrestricted. We approximate *G* by free weights on a 105-point grid of *u* ∈ [−16, 10] (step 0.25), estimated by EM (Kiefer–Wolfowitz NPMLE). We profile *k* over [0.3, 3.0] in steps of 0.05 and refine. 95% CIs are profile-likelihood intervals (deviance ≤ 3.84). By Theorem 2(i), *k* is identified without any form for *G*.

**Model-free moment matching.** For *j* = 2, 3, 4 we smooth the U-statistics of *M*ⱼ and of *S̄* in log *T* (local-linear, Gaussian kernel, bandwidth 0.7). We then choose *k*ⱼ so that *M*ⱼ(*T*) ≈ *S̄*(*j*^(1/*k*)*T*) over *T* ∈ [4 min, 16 h/*j*^(1/*k*)]. Agreement of *k*₂, *k*₃ and *k*₄ is the overidentification check of Theorem 2(i).

**Likelihood-ratio bounds.** For fixed *k*, success at *T*\* is linear in the weights of *G*, and the log-likelihood is concave in them. The minimum and maximum of *S̄*(*T*\*) over {*G* : ℓ(*k*, *G*) ≥ ℓ̂ − 1.92} are therefore convex programs, which we solve with Clarabel via CVXPY. We take the union over *k* in its profile CI on a 0.1 grid. These are 95% profile-LR bounds for a scalar functional, conservative in *k*.

**Out-of-window test.** Per agent, all models are fitted on tasks with *T* < *C*, for *C* ∈ {1 h, 2 h, 4 h}. They predict the equal-task-weight mean success on tasks in [*C*, 4*C*) and on all tasks ≥ *C*. The models are:

- pooled log-logistic (METR's form);
- pooled Weibull;
- the completely monotone mixture (constant-hazard frailty, NPMLE);
- Gaussian frailty GLMM;
- NPMLE frailty;
- NPMLE frailty with *k* fixed at 1.

Uncertainty accounts for the fact that all agents share the target tasks. We use a family-cluster bootstrap (2,000 draws) of the mean-over-agents error, plus leave-one-family-out.

**Hazard-length map.** To locate the failure, we replace *k* log *T* by a continuous piecewise-linear function of log *T*, with knots at 1 h and 4 h and slopes *k*₁, *k*₂, *k*₃, keeping the NPMLE *G*. We test against a single *k* by LR (2 df per agent), and summarise over agents by summing the LR statistics. As an alternative identification, we estimate a *within-family* elasticity. This uses a cloglog model with family fixed effects (ridge 0.01) and Gaussian task frailty, on the 155 tasks in the 28 families that contain tasks differing in length by at least a factor 2.

**Crossed model, task audit, retry baselines.** These address review r1.

- A joint cloglog model with crossed effects: η_ai = μ_a + *k*_a log *T*ᵢ + λ_a *u*ᵢ + τ_a *w*_ai. Here *u*ᵢ is shared task difficulty, either Gaussian or discrete with free weights, and *w*_ai is the agent-specific task effect. An optional latent class of "invalid" tasks has *p* = 0 for all agents.
- A task-validity audit of tasks never solved by any agent.
- Held-out pass@*m* against beta-binomial, IRT-style and latent-class baselines, with folds shared across agents.
- Profile-likelihood CIs for the Gaussian model.
- A misspecification simulation, now extended to the NPMLE estimator and to the out-of-window test (Sec. 5.6).

## 5 Results

### 5.1 Heterogeneity is large, and mostly a property of the task

[Source: `results/heterogeneity_summary.txt`, `results/crossed_summary.txt`.]

Across the 89 eligible agent × length strata, the median *H* is **0.61** (IQR 0.49–0.75), and the bootstrap CI excludes 0 for 81% of strata. The stratum medians are 0.67 (1–4 min), 0.72 (4–15 min), 0.47 (15–60 min), 0.68 (1–4 h) and 0.67 (4–16 h). Length differences within a stratum explain at most 0.027 of variance on average.

The crossed agent × task model separates heterogeneity that is shared across agents (scale λ_a) from heterogeneity that is agent-specific (τ_a). The median agent-specific share τ²/(τ² + λ²) is **0.19** (Gaussian *u*), and 0.16 with a discrete *u*. Roughly four-fifths of the between-task spread in log hazard at fixed length is therefore task-intrinsic, ordering tasks the same way for all 20 agents. The model's latent "invalid task" class receives weight 0.011 (Gaussian) or 0.006 (discrete), so the data give little support to a block of broken tasks.

**Task audit.** 17 tasks are never solved by any agent, all HCAST and 13 of them ≥ 4 h, and 16 of the 17 have *estimated* lengths. That is far above the base rate of 66/228. In 6 of them some run reached a continuous score ≥ 0.8, so they are near the binarisation threshold. In 5 every run scored ≤ 0.02. These tasks are candidates for mis-timing or mis-scoring, and every result below is reported with and without them. 59 tasks, all SWAA and under 1 min, are solved in every run by every agent.

### 5.2 Retries: what *H* buys and what it does not

[Source: `results/passk_baselines_summary.txt`, `results/passk_H_brackets.csv`.]

Proposition 3's bracket contains the observed pass@3 in 98.9% of the 89 agent-strata and pass@4 in 96.6% (tolerance ±0.02). The mean bracket widths are 0.033 and 0.065. The homogeneous prediction lies a mean 0.13 above the upper bracket.

On held-out tasks (5 task folds shared by all agents), pass@*m* is predicted to within a mean absolute agent-level error of 0.004–0.009 at *m* = 2–4 by every overdispersed model:

- beta-binomial with logit mean linear in log *T*: 0.0059 / 0.0040 / 0.0037;
- Gaussian frailty: 0.0090 / 0.0066 / 0.0078;
- logit-normal frailty: 0.0072 / 0.0054 / 0.0066.

The homogeneous pooled curve misses by 0.067–0.104. On held-out tasks ≥ 4 h, pass@4 is 0.30 observed, 0.31 under Gaussian frailty, 0.32 under the beta-binomial and 0.47 under homogeneity.

We draw a narrower conclusion than in round 1. *Retry gains are governed by overdispersion (H), which any overdispersed model captures.* They are not evidence for a particular survival structure. The frailty model underpredicts the gain at *m* = 4 by about 12% (0.065 against 0.074).

### 5.3 The hazard-length elasticity without assumptions on the frailty law

[Source: `results/duality_summary.txt` (A, B), `results/profile_ci_summary.txt`, `results/crossed_summary.txt`, `results/fig_duality.png`a.]

**Table 1.** Elasticity *k* under increasingly weak assumptions about heterogeneity. Medians over 20 agents. "CI > 1" counts agents whose 95% profile CI lies entirely above 1. Rows marked * use outcome-selected filters and are diagnostics only.

| Estimator | *k* median (IQR) | *k* > 1 | CI > 1 | CI < 1 |
|---|---|---|---|---|
| pooled Weibull, no heterogeneity | 0.63 | 0/20 | — | — |
| Gaussian frailty, per agent | 1.56 (1.34–1.85) | 20 | 15 | 0 |
| crossed agent × task, Gaussian shared effect | 1.78 (1.54–2.06) | 20 | 19 (Wald) | 0 |
| crossed, discrete shared effect + invalid class | 1.69 (1.47–2.00) | 20 | — | — |
| **NPMLE frailty, G unrestricted** | **1.77 (1.45–2.09)** | **20** | **17** | **0** |
| NPMLE, baseline-timed tasks only | 1.79 (1.53–1.89) | 20 | 18 | 0 |
| NPMLE, RE-Bench removed | 1.78 (1.45–2.10) | 20 | 17 | 0 |
| NPMLE, never-solved tasks removed* | 1.60 (1.35–1.87) | 20 | 15 | 0 |
| NPMLE, never- and always-solved removed* | 1.49 (1.10–1.86) | 17 | 10 | 1 |
| crossed Gaussian, never-solved removed* | 1.72 (1.53–1.96) | 20 | 19 (Wald) | 0 |

Removing the 17 never-solved tasks lowers the NPMLE median from 1.77 to 1.60. This answers the reviewer's concern that the frontier estimates rest on those tasks: Claude Opus 4.6 goes from 1.92 to 1.85, GPT-5.2 from 2.00 to 1.81, and GPT-5.3-Codex from 2.29 to 2.00. The large drops the reviewer found came from the single-agent Gaussian model, whose tail is forced. The NPMLE absorbs never-solved tasks with atoms instead of a forced Gaussian tail. Also removing the 59 always-solved sub-minute tasks takes away most short-length leverage, and leaves a median of 1.49. A common *k* in the crossed model is 1.42 (profile CI 1.39–1.56), but the free-*k* model fits better (LR 49.6 on 19 df).

**Overidentification.** The model-free moment estimates *k*₂, *k*₃ and *k*₄ agree within agents: the median within-agent range is 0.08, and the medians are 2.10, 2.19 and 2.17. This is the restriction that Theorem 2(i) imposes along the *j* direction, and it holds.

The *level* of the kernel estimator is imprecise. It hits the grid edge of 3.0 for 4 agents, and falls below 1 for the five weakest agents. For those agents, success is near zero beyond 1 h, so the matching is dominated by sub-hour tasks. Its ranking of agents agrees with the NPMLE (Spearman ρ = 0.82, p = 9×10⁻⁶), so we use it only as a check.

Fig. `fig_duality.png`a shows the NPMLE reproducing pass@1, pass^2 and pass^4 jointly for two agents.

**Within-family elasticity.** [Source: `results/hazard_map_summary.txt` (b).] Using only length contrasts within the same family, the median *k* is 1.42. It is above 1 for 14 of 20 agents, with the CI above 1 for 5 and below 1 for none. The intervals are wide (155 tasks in 28 families), but they do not suggest a declining hazard even when family composition is held fixed.

*Reading.* At fixed frailty, the cumulative hazard of the suite's tasks grows faster than proportionally with length. Doubling length multiplies it by 2^1.77 ≈ 3.4, against 2^0.63 ≈ 1.55 for the pooled fit. By Theorem 1 this is an across-suite elasticity, and a within-run "fatigue" only under invariance. It agrees with the self-conditioning mechanism of Sinha et al. (2025), but it does not establish it.

### 5.4 The decisive test: short-to-long forecasts fail at one to four hours

[Source: `results/duality_summary.txt` (C), `results/forecast_bootstrap.txt`, `results/hazard_map_summary.txt` (a), `results/fig_duality.png`b.]

**Table 2.** Out-of-window forecasts of mean success on longer tasks. Mean over 20 agents of (predicted − observed). The family-bootstrap CI and the coverage of the NPMLE's 95% LR bounds are given for the target band [*C*, 4*C*).

| Train on | Target | Observed | log-logistic (METR) | Weibull | CM mixture | Gaussian frailty | **NPMLE frailty** [95% family-bootstrap CI] | NPMLE, *k* = 1 | Observed inside NPMLE LR bounds |
|---|---|---|---|---|---|---|---|---|---|
| < 1 h | [1 h, 4 h), 18 tasks | 0.337 | +0.197 | +0.177 | +0.279 | +0.148 | **+0.151** [+0.050, +0.261] | +0.211 | 14/20 |
| < 1 h | ≥ 1 h, 73 tasks | 0.239 | +0.210 | +0.176 | +0.314 | +0.152 | **+0.120** [+0.063, +0.179] | +0.226 | 18/20 |
| < 2 h | [2 h, 8 h), 31 tasks | 0.241 | +0.188 | +0.152 | +0.175 | +0.158 | **+0.142** [+0.038, +0.237] | +0.213 | 13/20 |
| < 4 h | [4 h, 16 h), 50 tasks | 0.221 | +0.010 | −0.076 | −0.101 | +0.029 | **+0.008** [−0.061, +0.079] | +0.167 | 16/20 |
| < 4 h | ≥ 4 h, 55 tasks | 0.206 | +0.017 | −0.068 | −0.095 | +0.036 | **+0.021** [−0.053, +0.091] | +0.178 | 17/20 |

Point errors are means over agents of agent-level means, from `duality_summary.txt`. The CIs come from `forecast_bootstrap.txt`, which averages per task over agents. The two differ by at most 0.002, because one agent lacks some tasks.

Four facts stand out.

- **(1) Calibrated below one or two hours, every model overpredicts longer tasks, the nonparametric frailty model included.** The NPMLE is the least biased (+0.12 to +0.15), yet its bias is large and its family-bootstrap CIs exclude 0. Leave-one-family-out ranges stay between +0.11 and +0.19, so no single family drives it. Eighteen of 20 agents err in the same direction. Because the agents share tasks, we rely on the task-level bootstrap and not on that count. The observed value falls outside the 95% LR bounds for 6 of 20 agents in the 1–4 h band and 7 of 20 in the 2–8 h band. Under invariance with a power-law Λ₀ the nominal expectation is about 1. Our simulation shows that profile intervals undercover (Sec. 5.6), so this count is suggestive, and the bootstrap carries the inference.
- **(2) The failure is not about heterogeneity.** Modelling heterogeneity reduces the bias (+0.20 for the logistic, +0.15 for the NPMLE), but cannot remove it. The NPMLE leaves *G* free, and in simulation non-Gaussian frailty does not bias its forecasts upward (Sec. 5.6). By Theorem 2(iii), what fails is invariance or the form of Λ₀ beyond 1–2 h.
- **(3) Calibrated below four hours, forecasts of 4–30 h tasks are unbiased** for the logistic, the Gaussian and the NPMLE models (+0.01 to +0.04, CIs covering 0). Pooled Weibull and CM curves now *under*predict. Coverage of the LR bounds rises to 16–17 of 20.
- **(4) Fixing *k* = 1 (task-level constant hazard) overpredicts by 0.17–0.23 at every cut.** Long tasks are harder than a constant per-minute hazard allows, even after heterogeneity is accounted for.

**Where the map bends.** A piecewise hazard-length map with knots at 1 h and 4 h beats a single *k* jointly across agents: summed LR = 94.8 on 40 df, p = 2.4×10⁻⁶. Individually, 6 of 20 agents reach p < 0.05. The median local elasticities are *k*₁ = 1.41 below 1 h, ***k*₂ = 2.91 between 1 and 4 h**, and *k*₃ = 1.80 above 4 h. The segment estimates are noisy. For the weakest agents, whose success is near zero beyond 1 h, *k*₂ and *k*₃ are unidentified, and the fits place them at extreme values (Table in `hazard_map_summary.txt`). The medians are robust to this.

So in the TH1.1 suite, failure risk at fixed frailty rises steeply between about one and four human-hours. Below and above that range it rises more slowly. This is visible as a shoulder in the pass^*j* curves of Fig. `fig_duality.png`a. It shows up as a regime change that short-task calibration cannot anticipate.

Two caveats limit how far this can be read.

- The [1 h, 4 h) band contains only 18 tasks in 14 families. The [*C*, 4*C*) bootstrap intervals are wide for this reason.
- By Theorem 1(c), "steeper Λ₀" and "harder tasks in this length range" cannot be told apart. 6 of these 18 lengths are estimates, and 2 of the 18 tasks are never solved. Suspect target tasks account for part of the effect [Source: `results/forecast_subsets.txt`]. Removing the never-solved target tasks is an outcome-selected filter, so it biases the comparison *against* finding over-prediction. Even so, the NPMLE bias in the 1–4 h band falls from +0.151 to +0.114 (CI [+0.004, +0.227]), and on all tasks ≥ 1 h from +0.122 to +0.065 (CI [+0.008, +0.126]). Restricting targets to baseline-timed tasks gives +0.090 (CI [−0.018, +0.212]) and +0.076 (CI [−0.002, +0.151]). In every variant, calibrating below 4 h gives no significant bias. So the over-prediction survives the audit. About 40–50% of its size in the ≥ 1 h bands is attributable to never-solved tasks with mostly estimated lengths, and on baselined tasks alone it is borderline.

METR (2026b) notes that the logistic fit is highly sensitive to performance on very easy tasks, which flattens the fitted slope. That explains part of the logistic's bias here. It cannot explain the NPMLE's, which leaves the frailty law free and so has no such sensitivity.

Either way the forecasting conclusion is the same. *Short-task data, with or without replicates and with or without a heterogeneity model, did not predict success on the next factor of four in length on this suite.*

### 5.5 What the data allow at day, week and month lengths

[Source: `results/duality_bounds.csv`, `results/fig_duality.png`c.]

These numbers assume invariance and a power-law Λ₀ beyond the data. Sec. 5.4(3) supports that assumption within 4–30 h, and nothing supports it beyond.

**Table 3.** 95% LR bounds (union over *k* in its CI) on success on a random task of length *T*\* drawn from a TH1.1-like population, for frontier agents. The point value is the NPMLE fit.

| Agent | 8 h: point [bounds] | 40 h: point [bounds] | 160 h: point [bounds] |
|---|---|---|---|
| Claude Opus 4.6 | 0.56 [0.46, 0.66] | 0.24 [0.09, 0.42] | 0.19 [0.007, 0.40] |
| Claude Opus 4.5 | 0.44 [0.33, 0.55] | 0.17 [0.04, 0.35] | 0.00 [0.00, 0.24] |
| GPT-5.2 | 0.47 [0.36, 0.58] | 0.19 [0.06, 0.38] | 0.00 [0.00, 0.25] |
| GPT-5.3-Codex | 0.46 [0.35, 0.56] | 0.11 [0.02, 0.31] | 0.04 [0.00, 0.29] |
| GPT-5.1-Codex-Max | 0.35 [0.25, 0.46] | 0.11 [0.00, 0.28] | 0.10 [0.00, 0.28] |
| Gemini 3 Pro | 0.33 [0.23, 0.44] | 0.06 [0.03, 0.21] | 0.00 [0.00, 0.19] |
| GPT-5 | 0.34 [0.24, 0.45] | 0.13 [0.00, 0.30] | 0.11 [0.00, 0.29] |

What these bounds show:

- **At 8 h, inside the data, the answer is identified to about ±0.1.**
- **At 40 h, the bounds narrow the round-1 range but do not close it.** The earlier range of model-dependent projections, 0.01–0.76 for Claude Opus 4.6, came from different parametric frailty families. The exponential (0.03) and the three-class model (0.76) both fall outside the bounds [0.09, 0.42]. Those were artefacts of parametric tails, as round 1 suspected.
- **At 160 h, the lower bound is ≤ 0.007 for every agent.** Week-to-month success is not identified even under invariance. That is Theorem 2(ii) in practice: with *n* ≤ 8 replicates and *k* ≈ 1.8, the identified range ends near 8^(1/1.8) × 30 h ≈ 95 h, and only for the few tasks with 8 runs.
- **Every frailty fit makes success at multi-day lengths bimodal.** For Claude Opus 4.6 at 40 h, the NPMLE puts 20% of tasks at *p* ≥ 0.8 and 48% at *p* ≤ 0.05. Whatever success exists sits in a minority of tasks that are easy for their length.

For 12 of the 20 agents, all of them released before Claude Opus 4.5, the upper bound at 40 h is below 0.12.

### 5.6 Simulations: what the estimators recover and what the forecast test detects

[Sources: `results/misspec_summary.txt`, `results/npmle_sim_summary.txt`. Both use Claude 4 Opus's TH1.1 design: 228 tasks, 1,368 runs.]

*Gaussian GLMM under misspecification* (20 replicates per scenario). With a declining true hazard (*k* = 0.63) and non-Gaussian frailty, the Gaussian GLMM reports:

- 0.83 for two-point frailty, with 10% of replicates above 1;
- 0.63 for *t*₃ frailty;
- 0.57 for skewed three-point frailty;
- 0.61 with 10% impossible tasks;
- 0.74 with 30% impossible long tasks.

So it does not manufacture *k* > 1 from a declining hazard under these forms. Label noise of s.d. 0.7 in log *T* barely moves it. The exception is the one Theorem 1(c) predicts. A frailty location drift of +0.8 per log *T* with true *k* = 0.63 yields *k̂* = 1.44, above 1 in every replicate. Drift is unidentifiable, and we do not claim to exclude it. The mechanical-correlation check (true σ and *k* fixed, capability varied) gives corr(log *ĥ*₅₀, log σ̂) = 0.05. So the round-1 correlation of 0.87 between capability and frailty spread across agents is not a fitting artefact. It remains descriptive and confounded with release date.

*NPMLE and the forecast test* (10 replicates per scenario). The NPMLE recovers a declining hazard under non-Gaussian frailty: *k̂* = 0.65 (homogeneous), 0.65 (two-point), 0.72 (skewed three-point) and 0.67 (10% impossible tasks), all for true *k* = 0.63. In a Gaussian truth with *k* = 1.5 it gives 1.66 ± 0.24, a small upward bias. Its profile CIs undercover: they contain the truth in 6–10 of 10 replicates depending on scenario. This is one reason we report agent-level CIs only as descriptive and rely on task-level bootstraps for the forecast test. With location drift (M7), *k̂* = 1.51, as Theorem 1(c) says it must be.

The out-of-window test (train < 1 h, predict the true mean *p* on 1–4 h tasks) separates the scenarios:

- When invariance and a power-law Λ₀ hold, its mean error is −0.03 to −0.09, and +0.02 in one scenario. It never shows the large positive bias seen in the data.
- *Smooth drift* (M7) also gives a negative error (−0.06), because a window fit absorbs drift into *k*.
- An *invariant frailty with a kinked hazard-length map* (*k* = 1, then 3 on 1–4 h, then 1) gives **+0.12 ± 0.11**, the magnitude observed in the data.

The test therefore detects exactly the departure that Sec. 5.4 reports: a change of regime in length, not misspecified heterogeneity and not smooth drift.

## 6 Discussion

**What can be said today about tasks spanning days and weeks.**

- *On TH1.1-type tasks, heterogeneity is not the main obstacle to extrapolation.* Replicate runs measure it, nonparametrically. Whatever success agents have on long tasks is concentrated in a minority of tasks.
- *The obstacle is the hazard-length map.* Between 1 and 4 h it steepens in a way that no model trained on shorter tasks anticipated. Nothing in outcome data can rule out a similar break between 30 h and 300 h.
- *Within the range where the map is stable (4–30 h), it supports useful bounds.* For frontier agents, success on a random 40-hour task is between about 0.05 and 0.4. Month-scale success is not identified.

A statement such as "the 50% horizon will reach a work-week by year *X*" is therefore an extrapolation of an *across-length* relation that has already bent once inside the observed range. It should be read as such.

**Why the duality matters beyond this suite.** Theorem 2 converts a cheap measurement, more runs of existing tasks, into information about higher cumulative hazard. It also makes precise what that measurement cannot buy: length. That yields a *design rule*. To identify, under invariance, success at length *T*\* from tasks no longer than *T*_max, one needs *n* ≥ (*T*\*/*T*_max)^*k* replicates per task. With *k* ≈ 1.8, this is 5.2 runs of 16-hour tasks, or 18 runs of 8-hour tasks, to reach 40 h. Sec. 5.4 shows that invariance itself needs checking at the target length, so the economical design is a *few* tasks at the target length to test invariance, plus *many* replicates on somewhat shorter tasks to pin down the frailty tail.

**Recommendations for horizon reporting.**

- (1) Report the out-of-window forecast error, i.e. train on tasks below *C* and predict tasks above it, as a routine diagnostic. This complements the diagnostic plots recommended by Nguyen and Fithian (2026).
- (2) Report *H*, or pass^*j* by length, next to the horizon.
- (3) Do not translate a pooled *k* into per-minute reliability (F2), or a conditional *k* into within-run fatigue (Theorem 1).
- (4) Report retry gains through *H*, which bounds them (Prop. 3).

**Implications for agent design.** When *H* is large, re-running the same agent on the same long task buys little. Prop. 3 bounds the gain, and about four-fifths of the heterogeneity is task-intrinsic. Interventions that change the agent-task interaction should be worth more than resampling: decomposition into verified sub-goals, external verifiers (Kambhampati et al., 2024), and memory or hand-off across sessions. If the 1–4 h steepening reflects within-run error accumulation, as self-conditioning would predict (Sinha et al., 2025), then checkpointing and context resets around that scale are where to look. Testing this requires time-stamped trajectory data. Outcome logs cannot test it (Theorem 1).

**A protocol, with predictions stated in advance.** This is not a registered protocol. It is a proposal with predictions written before any data exist.

- (P1) Build ≥ 40 tasks of 16–80 human hours in ≥ 10 families, sampled to span difficulty.
- (P2) Run each frontier agent ≥ 6 times per task.
- (P3) Fit the NPMLE frailty model to the existing suite, *without* the new tasks, and publish its LR bounds for the new tasks before scoring them.

Our predictions from TH1.1:

- (i) *H* in the new stratum ≥ 0.5 whenever stratum success lies in (0.1, 0.9);
- (ii) observed success inside the published 95% bounds for at least 15 of 20 agents if invariance holds from 4 h to 80 h, and systematically below them if a second steepening exists;
- (iii) observed pass@4 − pass@1 within Prop. 3's bracket computed from the new stratum's *H*;
- (iv) a bimodal distribution of per-task success, with more than a third of tasks never solved in 6 runs whenever stratum success is below 0.5.

## 7 Limitations

- **Invariance is assumed, and partly rejected.** All statements about *k* as a within-task hazard, and all bounds at 40 and 160 h, assume invariance and a power-law Λ₀ beyond the data. Sec. 5.4 shows that this fails across 1–4 h on this suite. Our support for invariance beyond 4 h rests on 55 tasks (37 families) and only 5 tasks ≥ 16 h.
- **Drift is unidentifiable (Theorem 1c).** The elasticity estimates cannot separate "longer tasks accumulate hazard faster" from "the suite's longer tasks are intrinsically harder". The within-family estimate (median 1.42) is the closest we can get, and it is imprecise.
- **Few long tasks; estimated lengths.** 40 of the 55 tasks ≥ 4 h, and 16 of the 17 never-solved tasks, have estimated rather than baselined lengths. Restricting to baselined tasks leaves *k* unchanged (1.79), but it also removes most long tasks.
- **The 1–4 h band is small.** Its 18 tasks in 14 families make the band-specific CI wide. The piecewise slopes are unidentified for weak agents, and the fits put them at extreme values. The knots at 1 h and 4 h were chosen once and not tuned, but other knots could locate the bend differently.
- **Shared tasks.** All agents share the tasks, so agent counts are not independent evidence. We report task-level (family-cluster) uncertainty for the forecast test. Per-agent profile CIs in Table 1 do not account for this dependence.
- **Likelihood bounds.** The LR bounds are 95% profile-likelihood bounds on a 105-point frailty grid with a union over a *k* grid. They are not exact finite-sample confidence sets. The grid also caps *k* at 3.0, and for a few agents the CI reaches that cap.
- **Moment-matching estimator.** It is a crude kernel method. We use it only for its internal consistency across *j* and its rank agreement with the NPMLE.
- **Replicate independence.** Shared infrastructure failures (4.3% usage-limit runs) or scaffold changes could inflate within-task correlation. Dropping usage-limit runs did not change the round-1 results.
- **Scope.** The data are METR's automatically scored software, ML and cybersecurity tasks. Real multi-day work involves interruptions, coordination and changing specifications. We ran no agents, and every empirical claim is a re-analysis of public logs.
- **Prior work we could not read in full.** Two related pieces (Reasoning Under Uncertainty, 2026; Mahendrakar, 2026) were reachable only through search snippets. They may contain points we have not credited.

## 8 Conclusion

The flattening of pooled success-versus-length curves has been read as agents "settling in" to long tasks. That reading was already known to be confounded with heterogeneity. We showed what replicate runs add. Even with infinite replicates they cannot reveal within-task hazard dynamics without an invariance assumption. Under that assumption, through the replicate–length duality, they identify the hazard-length map without any model of heterogeneity, and they turn heterogeneity models into falsifiable forecasts.

On METR's Time Horizon 1.1 data, the forecasts fail in a specific place. Tasks between one and four human-hours are harder than any model calibrated on shorter tasks predicts. Beyond four hours the models recover, and they bound frontier agents' success on a random 40-hour task to roughly 0.05–0.4. They leave month-long success unidentified.

For the question this paper started from, the answer is that today's public data cannot say whether agents will complete tasks spanning weeks. The suite has already shown one change of regime with length, and only replicate-rich measurements at the target lengths can rule out another.

## References

All references were checked to exist via web search in November 2026. For the two items whose hosts were unreachable from our environment, the content was confirmed only through search snippets, and we cite them only for what those show.

- Backlund, A. and Petersson, L. (2025). Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents. arXiv:2502.15840.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Feller, W. (1971). *An Introduction to Probability Theory and Its Applications, Vol. II* (2nd ed.). Wiley.
- Kambhampati, S., Valmeekam, K., Guan, L., Verma, M., Stechly, K., Bhambri, S., Saldyt, L. and Murthy, A. (2024). Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. *ICML 2024*, PMLR 235:22895–22907. arXiv:2402.01817.
- Kiefer, J. and Wolfowitz, J. (1956). Consistency of the maximum likelihood estimator in the presence of infinitely many incidental parameters. *Annals of Mathematical Statistics*, 27(4), 887–906.
- Kwa, T., West, B., Becker, J., Deng, A., Garcia, K., Hasin, M., Jawhar, S., Kinniment, M., Rush, N., Von Arx, S., et al. (2025). Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499 (NeurIPS 2025).
- LessWrong (2026a). "A(nother) look at METR's time horizon: tasks of the same duration get solved months apart." https://www.lesswrong.com/posts/qHztdHtoXixCWAr7z/ (author not shown in our sources).
- LessWrong (2026b). "(Updated) METR's data can't distinguish between trajectories (and 80% horizons are an order of magnitude off)." https://www.lesswrong.com/posts/sBEzomgnYJmYHki9T/ (author not shown in our sources).
- Liu, F., Gala, J., Nilaksh, Bahdanau, D., Reddy, S. and Larochelle, H. (2026). BRIDGE: Predicting Human Task Completion Time From Model Performance. arXiv:2602.07267.
- Mahendrakar, P. (2026). Three Hazards Under One Horizon. Preprint, 8 October 2026, Zenodo DOI 10.5281/zenodo.23240102. (Content known to us only from search snippets.)
- METR (2025). How Does Time Horizon Vary Across Domains? Blog post, 14 July 2025. https://metr.org/blog/2025-07-14-how-does-time-horizon-vary-across-domains/
- METR (2026). Time Horizon 1.1. Blog post, 29 January 2026. https://metr.org/blog/2026-01-29-time-horizon-1-1/ ; data and code: https://github.com/METR/eval-analysis-public (commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2).
- METR (2026b). Impact of modelling assumptions on time horizon results. Note, 20 March 2026. https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/
- Nguyen, D. T. and Fithian, W. (2026). On the estimation and validity of AI time horizons—a statistical look at the METR plot. arXiv:2610.12466 (NeurIPS 2026).
- Oakes, D. (1989). Bivariate survival models induced by frailties. *Journal of the American Statistical Association*, 84(406), 487–493.
- Ord, T. (2025). Is there a half-life for the success rates of AI agents? arXiv:2505.05115.
- Ord, T. (2026). Hazard Rates for AI Agents Decline as a Task Goes On. Blog post, 4 February 2026, reporting analysis by Gus Hamilton. https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline
- Proschan, F. (1963). Theoretical explanation of observed decreasing failure rate. *Technometrics*, 5(3), 375–383.
- Reasoning Under Uncertainty (2026). "The time-horizon curve cannot tell a half-life from recovery." Substack essay. https://reasoningunderuncertainty.substack.com/p/the-time-horizon-curve-cannot-tell (author and date not verified; content known to us only from search snippets).
- Rein, D., Becker, J., Deng, A., Nix, S., Canal, C., O'Connell, D., Arnott, P., Bloom, R., Broadley, T., Garcia, K., et al. (2025). HCAST: Human-Calibrated Autonomy Software Tasks. arXiv:2503.17354.
- Schaeffer, R., et al. (2025). How Do Large Language Monkeys Get Their Power (Laws)? *ICML 2025*, PMLR 267. arXiv:2502.17578.
- Sinha, A., Arun, A., Goel, S., Staab, S. and Geiping, J. (2025). The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs. arXiv:2509.09677 (ICLR 2026).
- Valmeekam, K., Marquez, M., Olmo, A., Sreedharan, S. and Kambhampati, S. (2023). PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2206.10498.
- Vaupel, J. W., Manton, K. G. and Stallard, E. (1979). The impact of heterogeneity in individual frailty on the dynamics of mortality. *Demography*, 16(3), 439–454.
- Wijk, H., et al. (2024). RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv:2411.15114 (ICML 2025).
- Xu, F. F., et al. (2024). TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. arXiv:2412.14161.
- Yao, S., Shinn, N., Razavi, P. and Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045 (ICLR 2025).
- Zeger, S. L., Liang, K.-Y. and Albert, P. S. (1988). Models for longitudinal data: a generalized estimating equation approach. *Biometrics*, 44(4), 1049–1060.
