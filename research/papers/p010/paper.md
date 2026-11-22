# The Definition Is the Forecast: How Reliability Thresholds and Cross-Domain Aggregation Determine Measured Distance to AGI

*Target venue: NeurIPS 2027, Datasets & Benchmarks Track (evaluation methodology)*

---

## Abstract

"How far are we from AGI?" is usually answered by extrapolating one scalar: a percentage score on a battery of cognitive domains, or a task-completion time horizon. Each scalar silently fixes three choices. These are the set of domains, the success rate that counts as "can do it" (the reliability threshold q), and the rule that combines domains into one number (the aggregator). We formalise a definition of AGI progress as such a tuple and prove four results about it. (i) Arrival times are ordered by the aggregator's power-mean exponent p. (ii) For exponentially growing per-domain capabilities, the long-run growth rate of a power-mean aggregate equals the rate of the *fastest* domain when p>0, the *average* domain when p=0 and the *slowest* domain when p<0. The long-range forecast therefore jumps when p crosses 0. (iii) Under the logistic success model used for time horizons, raising the reliability threshold from 50% to q costs a constant delay of logit(q)/|β| doublings, where β is the logistic slope, and leaves the doubling time unchanged if and only if β is stationary. (iv) Bottleneck aggregators estimated from noisy domain scores are biased toward "farther from AGI", with the bias growing like σ√(2 ln n) in the number of domains n. We then re-analyse 24,008 public run-level records from METR's Time Horizon 1.1 suite (20 frontier systems, March 2023 to February 2026) and replicate on Time Horizon 1.0 (41,629 runs). We use a penalised hierarchical logistic model whose shrinkage is chosen by cross-validation over tasks. We cross 3 reliability thresholds with 8 aggregators over task families. Across these 24 definitions, extrapolated arrival dates of a one-work-month (167 h) horizon range from 2027.1 to 2032.6 (point estimates), and doubling times range from 3.7 to 6.8 months. Definition choice accounts for 43% of the total variance in arrival dates, and the remainder is hierarchical-bootstrap sampling noise. The closed-form reliability delay matches direct fits (e.g. 0.85 vs 0.95 years for q=0.8). The logistic slope shows no detectable trend. Cross-family dispersion grows over time, but the growth lives in the lower tail and is not detectable when saturated and floored families are excluded. Under per-domain linear extrapolation, the published CHC-based scores (GPT-4 27%, GPT-5 57%) never reach a 90% AGI threshold under any power mean, because three domains did not improve between the two snapshots. We end with pre-registered, falsifiable predictions and a reporting protocol: any claim about distance to AGI should state (Ω, q, aggregator, target) and report how its forecast moves when each is varied.

## 中文摘要

"我们离AGI还有多远？"通常的回答方式是外推某个单一标量：认知领域测验的百分比得分，或任务完成"时间跨度"（time horizon）。每个标量都隐含了三项选择：领域集合、算作"能完成"的成功率阈值q、以及把多个领域合成一个数的聚合规则。本文把"AGI进展的定义"形式化为这样一个元组，并证明了四个结论。（i）到达时间按聚合器的幂平均指数p单调排序。（ii）若各领域能力呈指数增长，幂平均聚合量的长期增长率在p>0时等于最快领域的增长率，在p=0时等于平均增长率，在p<0时等于最慢领域的增长率；因此p跨过0时长期预测会出现跳变。（iii）在时间跨度所用的logistic成功模型下，把可靠性阈值从50%提高到q，会带来恒定的logit(q)/|β|个倍增期的延迟（β为logistic斜率）；当且仅当β不随时间变化时，倍增时间保持不变。（iv）用带噪声的领域分数估计"瓶颈型"聚合量，会系统性地偏向"离AGI更远"，偏差随领域数n按σ√(2 ln n)增长。随后我们重新分析了METR公开的Time Horizon 1.1逐次运行数据（24,008条记录，20个前沿系统，2023年3月至2026年2月），并在Time Horizon 1.0（41,629条）上复现。方法是带惩罚的分层logistic模型，收缩强度由按任务的交叉验证选定。我们把3个可靠性阈值与8种跨任务族的聚合方式组合成24种定义。在这些定义下，外推得到的"一个工作月（167小时）时间跨度"到达日期点估计从2027.1年到2032.6年不等，倍增时间从3.7个月到6.8个月不等。到达日期总方差中有43%来自定义的选择，其余为分层bootstrap抽样噪声。可靠性延迟的闭式解与直接拟合吻合（如q=0.8时为0.85年对0.95年）。logistic斜率没有可检测到的趋势。跨任务族的离散度随时间上升，但上升集中在下尾；排除已饱和或完全失败的任务族后，检测不到这种上升。对已发表的CHC领域分数（GPT-4为27%，GPT-5为57%）按领域做线性外推，在任何幂平均下都达不到90%的AGI阈值，因为有三个领域在两次测量之间没有进步。最后我们给出预注册的可证伪预测和一套报告规范：任何关于"离AGI多远"的论断都应写明（领域集合Ω、阈值q、聚合器、目标），并报告逐一改变这些选择时预测如何变化。

---

## 1 Introduction

Two quantitative answers to "how far are we from AGI?" were widely discussed in 2025–2026. The first is psychometric. Hendrycks et al. (2025) define AGI as matching "the cognitive versatility and proficiency of a well-educated adult" and operationalise it with ten domains from Cattell–Horn–Carroll (CHC) theory (McGrew, 2009), each weighted 10%. Under this framework GPT-4 scores 27% and GPT-5 57%. A Google DeepMind team proposed a related ten-faculty cognitive taxonomy and evaluation protocol (Burnell et al., 2026). The second answer is behavioural. METR measures the length of software tasks, in human-expert minutes, that an agent completes with 50% probability. This "time horizon" has doubled roughly every seven months since 2019, and faster recently (Kwa et al., 2025). Extrapolating it gives dates at which agents could complete month-long tasks.

Both answers reduce a high-dimensional, jagged capability profile (Dell'Acqua et al., 2023) to one number and extrapolate that number. The reduction involves choices that the headline figure does not show. Fourati (2025) showed that the CHC percentage is sensitive to how domains are combined. Replacing the arithmetic mean by a generalised mean with exponent p<1 sharply lowers GPT-5's score. Ord (2025) and the original METR paper discuss the 80% threshold as an alternative to 50%. METR's follow-up (METR, 2025) finds similar doubling rates across nine benchmark domains but absolute horizons that differ by orders of magnitude. What is missing is a joint treatment that (a) asks how these choices affect the *forecast*, meaning time-to-threshold and doubling time, rather than the static score, (b) derives which choices change the *rate* and which only add a *constant delay*, and (c) measures on real run-level data how much forecast uncertainty comes from definition choice compared with sampling noise.

**Sub-question.** We do not propose a new definition of AGI. We ask a narrower and tractable question: *for a fixed body of evaluation evidence, how much does the answer to "when will we get there?" depend on the definitional choices made when that evidence is reduced to a scalar, and which choices matter most?*

**Contributions.**

1. **Formal framework (§3).** We define an AGI-progress definition as a tuple D = (Ω, q, A_p, τ): domains, reliability threshold, power-mean aggregator and target. We prove: (Prop. 1) arrival times are monotone in p; (Prop. 2) the asymptotic growth rate of a p-aggregate of exponentially growing domains is the softmax-weighted average of domain rates with inverse temperature p·t, which converges to the max, mean or min rate depending on the sign of p; (Prop. 3) in the logistic time-horizon model a reliability threshold q adds a constant delay of logit(q)/|β| doublings and changes doubling times only through drift in the logistic slope β; (Prop. 4) bottleneck aggregators of noisy estimates are biased toward pessimism, with an explicit bound.
2. **Measurement method (§4).** We give a penalised hierarchical logistic estimator of per-family horizons. Its shrinkage is chosen by task-level cross-validation. Hierarchical bootstrap intervals are attached to every definition's doubling time and arrival date.
3. **Empirical re-analysis (§5)** of all public METR run-level data (TH1.1 main, TH1.0 replication). Across 24 definitions, 167-hour arrival dates span 5.5 years of point estimates, and 43% of the total variance in arrival dates comes from the choice of definition. We also test Prop. 3's stationarity condition (not rejected), characterise the reliability and structure of family-level "jaggedness", and show that bottleneck metrics depend heavily on the shrinkage strength.
4. **Re-analysis of CHC scores (§5.8)** showing that two-snapshot linear extrapolation does not identify time-to-AGI on a bounded scale: stagnant domains cap the reachable score below 0.8 under every power mean.
5. **Pre-registered predictions and a reporting protocol (§6)** that later model releases can falsify.

All numbers in this paper are either (a) produced by code in `code/` and saved in `results/` (marked [ours]), or (b) taken from a cited source (marked with the citation).

## 2 Related Work

**Defining AGI and general intelligence.** Legg and Hutter (2007) define universal intelligence as expected performance across all computable environments, weighted by simplicity. This is a principled aggregator, but it cannot be computed. Chollet (2019) defines intelligence as skill-acquisition efficiency over a scope of tasks, given priors and experience, and introduced ARC. Morris et al. (2023) propose "Levels of AGI" along performance and generality axes, where a level requires a given percentile of skilled humans on a *wide range* of tasks. That is implicitly a quantile aggregator. Hendrycks et al. (2025) operationalise AGI with ten equally weighted CHC domains and an arithmetic mean. Burnell et al. (2026) propose a ten-faculty cognitive taxonomy with held-out tasks that yields cognitive profiles rather than a single score. Our work is orthogonal: we take any such profile as input and study how the reduction to a scalar shapes the forecast.

**Aggregation and compensability.** Generalised (power) means and their ordering are classical (Hardy, Littlewood & Pólya, 1934/1952). Fourati (2025) introduced compensability-indexed generalised means of the CHC domain scores and an area-under-the-curve summary across p. Applied to GPT-4 and GPT-5, it shows that the geometric mean collapses toward zero because of zero-scored domains. Burnell et al. (2023) argue that aggregate metrics hide instance-level structure and call for item-level reporting. Fourati's analysis is static, comparing scores at one time. We add the *dynamic* consequence: which aggregators change the forecast's rate (Prop. 2) and which add a constant delay (Prop. 3), together with an empirical decomposition on run-level data.

**Time horizons and their thresholds.** Kwa et al. (2025) introduced the 50%-task-completion horizon and reported similar doubling times at 50% and 80% (212 vs 213 days, as quoted by Ord, 2025). Ord (2025) models agent failure as a constant hazard per human-minute. This implies exponential decay of success with task length and motivates attention to high-reliability horizons. METR (2025) found similar doubling rates across benchmark domains but absolute levels that differ by orders of magnitude. METR's public page now states that measurements above 16 hours are unreliable with the current task suite (METR, 2026). Prop. 3 explains *when* the threshold changes the rate (slope drift) and when it adds only a constant delay. We test the stationarity condition directly.

**Metric choice and apparent discontinuities.** Schaeffer et al. (2023) showed that apparent "emergent abilities" can be artefacts of nonlinear metrics. Our Prop. 2 is a time-domain analogue: the *sign* of the aggregation exponent switches which domain governs the long-run forecast. Ruan et al. (2024) show that benchmark performance lies in a low-dimensional capability space. This bears on whether "jaggedness" is shared across models, which we measure (§5.5).

**Expert forecasts.** Surveys such as Grace et al. (2025; survey fielded October 2023) report wide disagreement among researchers about when "high-level machine intelligence" will arrive. They also report very different answers for the "all tasks" and "all occupations" framings: the 50% year is 2047 for the first and 2116 for the second. That is itself evidence that definitional framing moves forecasts by decades. Our work quantifies a similar effect on measured rather than elicited quantities.

## 3 Framework and Theory

### 3.1 Definitions as tuples

Let Ω = {1,…,n} be a set of domains (cognitive domains, task families, benchmarks). Let x_i(t) > 0 be system capability on domain i at time t. This is either a bounded score in [0,1] or an unbounded quantity such as a time horizon H_i(t) in minutes. For weights w_i > 0 with Σ w_i = 1, the power-mean aggregator is

  A_p(x) = (Σ_i w_i x_i^p)^{1/p}, A_0(x) = Π_i x_i^{w_i}, A_{+∞} = max_i x_i, A_{−∞} = min_i x_i.

The exponent p indexes *compensability*. p=1 lets strength in one domain fully offset weakness in another. p → −∞ is a pure bottleneck: "general" means general in every domain. Quantile aggregators such as "the 10th-percentile domain" sit between p=0 and p=−∞. A **definition** is D = (Ω, q, A, τ), where q is the success probability at which a domain counts as mastered (for time horizons) and τ is the target level. The **distance to AGI** under D is either the gap τ − A(x(t)) or the **arrival time** T_D = inf{t : A(x(t)) ≥ τ}. The second is what forecasts are about.

### 3.2 Results

**Proposition 1 (ordering of arrival times).** If each x_i(t) is continuous in t, then for p < p′ we have T_{(Ω,q,A_{p′},τ)} ≤ T_{(Ω,q,A_p,τ)}.

*Proof.* The power-mean inequality (Hardy et al., 1952, ch. 2) gives A_p(x) ≤ A_{p′}(x) for every x. If A_p(x(t)) ≥ τ, then A_{p′}(x(t)) ≥ τ, so the set of crossing times for p′ contains that for p, and its infimum is no larger. ∎

Prop. 1 is elementary. It means that disagreements about whether AGI has arrived can be ordered: a compensatory definition always declares arrival no later than a bottleneck one. The next result shows that the *size* of the gap is governed by the spread of domain growth rates.

**Proposition 2 (growth-rate selection).** Let log x_i(t) = a_i + s_i t with distinct rates s_i. Then

  d/dt log A_p(x(t)) = Σ_i π_i(t) s_i, where π_i(t) = w_i e^{p(a_i + s_i t)} / Σ_j w_j e^{p(a_j + s_j t)} (p ≠ 0),

and d/dt log A_0 = Σ_i w_i s_i for all t. Consequently, as t → ∞, the growth rate of A_p tends to max_i s_i when p > 0 and to min_i s_i when p < 0, and it equals the weighted mean rate when p = 0.

*Proof.* For p ≠ 0, log A_p = p^{−1} log Σ_i w_i e^{p(a_i+s_i t)}. Differentiating gives Σ_i π_i(t) s_i, a softmax-weighted average of the s_i with logits p(a_i + s_i t). As t → ∞ the logit differences p(s_i − s_j)t diverge. The weights therefore concentrate on argmax_i s_i when p > 0 and on argmin_i s_i when p < 0. For p = 0, log A_0 = Σ w_i(a_i + s_i t) is linear. ∎

*Implications.* (a) The long-range forecast is discontinuous in p at p = 0. Two definitions with p = +0.1 and p = −0.1 eventually grow at the rates of different domains. (b) The transition time scale is 1/(|p|·Δs), where Δs is the gap between the extreme rates, so heterogeneity in *rates*, not only in *levels*, decides how much definitions diverge. (c) For a distant target τ, T_p(τ) ≈ log τ / s_*(p). The ratio of forecasts between a bottleneck definition and a compensatory one therefore tends to s_max/s_min. Figure 1 (left) shows Prop. 2 numerically for ten domains [ours, `results/fig_theory.png`].

**Proposition 3 (reliability threshold: constant delay unless the slope drifts).** Suppose success on a task of human length h follows P(success | h, t) = σ(α(t) + β(t) log₂h) with β(t) < 0. Then the q-horizon satisfies log₂H_q(t) = (logit q − α(t))/β(t), and:

 (i) log₂H_q(t) − log₂H_{1/2}(t) = logit(q)/β(t). At fixed β, a definition at reliability q lags the 50% definition by exactly logit(q)/|β| doublings. If the 50% horizon doubles every DT months, the delay is logit(q)·DT/|β| months.

 (ii) d/dt log₂H_q = −α′/β − log₂H_q · β′/β. Hence two thresholds q₁ ≠ q₂ share a doubling time for all t if and only if β′ ≡ 0. If |β| shrinks over time (flattening), higher-q horizons grow more slowly.

*Proof.* Solve σ(α + β log₂H) = q for log₂H, then differentiate. ∎

Prop. 3 turns a definitional choice (50% vs 80% vs 99%) into two estimable quantities: the slope level |β| sets the delay, and the slope drift β′ sets any change of rate. Kwa et al.'s observation of near-identical 50% and 80% doubling times is, by (ii), evidence that β was approximately stationary. We test this on newer data in §5.2. The proposition says nothing about whether the logistic tail is correct at extreme q. Ord's (2025) constant-hazard model and its later refinements imply different tails, so we treat q ≥ 0.99 as illustrative only.

**Proposition 4 (bottleneck estimates are biased toward "farther").** Let x̂_i = x_i + ε_i with E ε_i = 0, on a log scale. Then E[min_i x̂_i] ≤ min_i x_i. If the ε_i are i.i.d. N(0, σ²), then min_i x_i − E[min_i x̂_i] ≤ σ√(2 ln n), with equality to first order when all x_i are tied and n is large.

*Proof.* Let j = argmin_i x_i. Then min_i x̂_i ≤ x̂_j pointwise, so E[min x̂] ≤ x_j. For the bound, min_i x̂_i ≥ min_i x_i + min_i ε_i, and E[max of n i.i.d. N(0,σ²)] ≤ σ√(2 ln n) (Boucheron, Lugosi & Massart, 2013, ch. 2). Apply this to −ε. ∎

The bias shrinks when domains are well separated (with 20 domains spread over 3σ, the simulated bias is −1.0σ, against −1.87σ when tied) [ours, `theory_chc.json`]. Bottleneck definitions therefore need many tasks per domain or explicit shrinkage. Shrinkage biases the other way (toward the mean), so the estimate of a bottleneck definition depends on a tuning choice. §5.6 shows this is large in practice.

## 4 Data and Estimation

**Data.** We use the public run-level files of METR's time-horizon analyses (repository `METR/eval-analysis-public`), downloaded unchanged. TH1.1 contains 24,008 agent runs (excluding human baselines) of 20 systems on 228 tasks in 79 task families from HCAST, RE-Bench and SWAA. Release dates run from GPT-4 0314 (2023-03-14) to Claude Opus 4.6 and GPT-5.3-Codex (2026-02-05). TH1.0 contains 41,629 runs of 33 systems on 170 tasks in 50 families. We restrict TH1.0 to systems released from 2023 onward (31 systems) so that its window matches TH1.1. Task families are small (median 2 tasks per family in both versions), which is why shrinkage is needed. Every run carries METR's diversity weight (`invsqrt_task_weight`), which we use after rescaling to sum to the number of runs per system.

**Model.** For system m, run r on a task of family f(r) with human time h_r:

  logit P(y_r = 1) = a_m + d_{m,f(r)} + b_m log₂h_r, with penalty λ Σ_f d_{m,f}².

Setting d ≡ 0 recovers the pooled METR-style fit. The q-horizon of family f is log₂H_{m,f}(q) = (logit q − a_m − d_{m,f})/b_m. We fit by penalised IRLS (`code/core.py`). The 24 definitions are the cross product of q ∈ {0.5, 0.8, 0.95} with eight aggregators: pooled (no family offsets); power means of family horizons with p ∈ {+1, 0, −1}; the family median, 25th and 10th percentiles; and the minimum family.

**Choosing λ.** We choose λ by 5-fold cross-validation over *tasks* (2 repetitions). A held-out task from a family absent in training receives offset 0. The weighted held-out log-loss per run is minimised at λ = 1 (0.3542; λ=2: 0.3550; λ=0.25: 0.3691; pooled, λ=∞: 0.3852) [ours, `cv_lambda_1-1.json`]. Family offsets therefore carry real out-of-sample information beyond task length: the CV loss improves by 8% over the pooled model. We had fixed λ = 1 as the main setting before running CV, and CV agrees with it. λ ∈ {0.25, 4} are reported as sensitivity analyses.

**Trends and arrival dates.** For each definition we take the running-maximum frontier in release order. We regress log₂ horizon on release date (years) by OLS to obtain a doubling time DT = 12/slope months. Extrapolating linearly gives the date at which the fitted frontier reaches a target: 40 h (one work-week) or 167 h (one work-month, METR's convention). The 167-hour target lies far beyond the task suite's longest tasks (≤ 30 h) and beyond the 16-hour reliability limit that METR states for the suite (METR, 2026). These dates are therefore *comparisons between definitions under a common extrapolation*, not predictions of real-world automation.

**Uncertainty.** We use a hierarchical bootstrap that resamples task families and then tasks within families, keeping all runs, and refits every system. Duplicated families are relabelled as distinct. TH1.1 uses B = 500 and TH1.0 uses B = 300. Intervals are 5th–95th percentiles. Total CPU time is about 45 minutes.

**Reproduction check.** Our pooled TH1.1 50% horizons are 4.0 min for GPT-4 0314, 203 min for GPT-5 and 719 min for Claude Opus 4.6 [ours, `metr_1-1_lam1_point.csv`]. On TH1.0, GPT-5 is 131 min. These are close to, though not identical with, METR's published figures; METR's regularisation and weighting choices differ slightly from ours. Our pooled 50% doubling time on the 2023–2026 TH1.1 frontier is 4.2 months [3.4, 5.1]. This is consistent with METR's observation that the trend accelerated after 2024 relative to the 7-month long-run figure (Kwa et al., 2025).

## 5 Results

### 5.1 Same data, 24 definitions

Table 1 reports the doubling time and the extrapolated 167-hour arrival date for every definition on TH1.1 [ours, `metr_1-1_lam1_summary.csv`]. Figures 2–3 visualise them.

**Table 1.** TH1.1, λ = 1. DT = doubling time (months). "Arrival" = extrapolated date at which the frontier reaches 167 h. Brackets: 90% hierarchical-bootstrap intervals. "Latest" = level for Claude Opus 4.6 (minutes).

| Aggregator | q | DT (mo) | Arrival 167 h | Latest (min) |
|---|---|---|---|---|
| pooled (METR-style) | 0.5 | 4.2 [3.4, 5.1] | 2027.7 [2027.2, 2028.3] | 719 |
| power mean p=+1 | 0.5 | 3.7 [3.0, 4.8] | 2027.1 [2026.5, 2027.9] | 3363 |
| geometric (p=0) | 0.5 | 4.2 [3.5, 5.2] | 2027.7 [2027.1, 2028.3] | 935 |
| power mean p=−1 | 0.5 | 4.9 [4.1, 6.2] | 2028.5 [2027.9, 2029.3] | 106 |
| median family | 0.5 | 3.9 [3.1, 4.9] | 2027.5 [2026.8, 2028.2] | 1973 |
| 25th-pct family | 0.5 | 5.7 [4.0, 7.0] | 2029.2 [2027.8, 2030.0] | 95 |
| 10th-pct family | 0.5 | 5.1 [4.2, 7.5] | 2029.0 [2028.3, 2030.7] | 32 |
| worst family | 0.5 | 4.9 [4.0, 7.4] | 2029.6 [2028.5, 2031.9] | 9 |
| pooled | 0.8 | 4.3 [3.6, 5.9] | 2028.7 [2028.0, 2029.8] | 70 |
| power mean p=+1 | 0.8 | 4.1 [3.5, 5.0] | 2028.1 [2027.6, 2028.7] | 354 |
| geometric (p=0) | 0.8 | 4.9 [3.9, 6.0] | 2028.9 [2028.2, 2029.6] | 98 |
| power mean p=−1 | 0.8 | 5.4 [4.0, 7.2] | 2029.7 [2028.5, 2031.1] | 11 |
| median family | 0.8 | 4.4 [3.6, 5.5] | 2028.5 [2027.8, 2029.3] | 208 |
| 25th-pct family | 0.8 | 6.3 [4.2, 8.3] | 2030.6 [2028.6, 2032.3] | 10 |
| 10th-pct family | 0.8 | 5.3 [4.0, 8.6] | 2030.1 [2028.7, 2032.6] | 3.4 |
| worst family | 0.8 | 6.3 [3.8, 9.1] | 2031.7 [2028.9, 2035.1] | 1.0 |
| pooled | 0.95 | 4.8 [3.4, 7.2] | 2030.0 [2028.5, 2032.2] | 5.1 |
| power mean p=+1 | 0.95 | 4.9 [3.9, 6.0] | 2029.6 [2028.6, 2030.4] | 28 |
| geometric (p=0) | 0.95 | 5.2 [3.9, 7.3] | 2030.2 [2028.7, 2032.0] | 7.8 |
| power mean p=−1 | 0.95 | 6.0 [3.8, 8.7] | 2031.2 [2028.8, 2033.7] | 0.9 |
| median family | 0.95 | 5.5 [3.8, 6.7] | 2030.5 [2028.7, 2031.5] | 16.5 |
| 25th-pct family | 0.95 | 6.8 [4.1, 9.7] | 2032.1 [2029.2, 2035.2] | 0.8 |
| 10th-pct family | 0.95 | 5.4 [3.8, 9.9] | 2030.9 [2029.1, 2035.5] | 0.3 |
| worst family | 0.95 | 6.3 [3.5, 11.2] | 2032.6 [2029.1, 2038.2] | 0.1 |

Four patterns stand out.

1. **The headline is the most optimistic corner.** The METR-style definition (pooled, q = 0.5) and its compensatory relatives (p = +1, median family) give the earliest arrival dates (2027.1–2027.7). Every move toward reliability (higher q) or toward breadth (lower p, lower quantile) delays arrival, as Prop. 1 requires for the aggregator dimension. The latest point estimate, the worst family at 95% reliability, is 2032.6, a 5.5-year spread on the same data.
2. **Aggregators change rates, not only levels.** Doubling times rise from 3.7–4.2 months for compensatory aggregators to 4.9–5.7 months for bottleneck-leaning ones at q = 0.5. This is the signature Prop. 2 predicts when family-level growth rates are heterogeneous: as p decreases, the aggregate tracks slower families.
3. **The current gap between "typical" and "worst-decile" ability is large.** For Claude Opus 4.6, the pooled 50% horizon is 719 min (about 12 h), but the 10th-percentile family horizon is 32 min and the worst family is 9 min [ours]. On this benchmark the frontier is 22 times further along in a "typical" sense than in a "nine-in-ten families" sense.
4. **Arithmetic means of unbounded quantities are not meaningful.** The p = +1 aggregate (3,363 min for Opus 4.6) is dominated by a few saturated families whose horizons are extrapolated well beyond their task lengths. We report it for completeness but recommend against p > 0 aggregation of time horizons.

### 5.2 The reliability threshold: a constant delay, as Prop. 3 predicts

The mean logistic slope of frontier systems on TH1.1 is b̄ = −0.573. With DT₅₀ = 4.23 months, Prop. 3(i) predicts the delays of the q-definitions relative to q = 0.5 shown in Table 2 [ours, `reliability_tax.json`].

**Table 2.** Closed-form "reliability tax" vs direct per-q trend fits (pooled aggregator, 167-h target).

| q | doublings logit(q)/|b̄| | predicted delay (yr) | direct-fit delay (yr) | direct DT (mo) |
|---|---|---|---|---|
| 0.8 | 2.42 | 0.85 | 0.95 | 4.3 |
| 0.9 | 3.84 | 1.35 | 1.41 | 4.3 |
| 0.95 | 5.14 | 1.81 | 2.32 | 4.8 |
| 0.99 | 8.02 | 2.83 | 3.34 | 4.8 |
| 0.999 | 12.06 | 4.25 | 4.50 | 4.7 |

The closed form tracks the direct fits within 0.1 years at q ≤ 0.9. At higher q the direct fits drift later because frontier selection and the sparse high-q region add noise. The q ≥ 0.99 rows extrapolate the logistic tail and are illustrative only (§3.2).

Prop. 3(ii) says that doubling times differ across q only if the slope drifts. The estimated trend in the pooled slope b is +0.012 per year (90% CI [−0.077, 0.096]; 57% of bootstrap replicates positive). The ratio of 80% to 50% doubling times is 1.02 [0.90, 1.34] [ours, `metr_1-1_lam1_slopes.json`]. Both are consistent with a stationary β and with Kwa et al.'s finding on the earlier data. The single flattest slope is that of the most recent system (Opus 4.6, b = −0.41). If later systems continue this, Prop. 3 predicts that 80% horizons will grow measurably slower than 50% horizons (§6).

### 5.3 How much of the forecast uncertainty is definitional?

For each bootstrap replicate b and definition d we compute the 167-hour arrival date A_{b,d}. We split the variance into a between-definition part, Var_d(mean_b A), and a within-definition part, mean_d Var_b(A) [ours, `variance_decomposition.json`]. Across the 24 definitions the between-definition variance is 1.77 yr² and the mean within-definition (sampling) variance is 2.39 yr². **Definition choice therefore accounts for 43% of total variance.** The within-definition variance itself grows with q and as p decreases, because bottleneck and high-reliability metrics are noisier (Prop. 4). Restricting to the aggregator alone at q = 0.5, the definitional share is 61%. Restricting to the threshold alone (pooled aggregator, three q's), it is 52%. For comparison, the 90% bootstrap interval of the headline definition (pooled, q = 0.5) is [2027.2, 2028.3], about one year wide. The point estimates across definitions span 5.5 years. Precise sampling-error bars on a single definition therefore give a misleading sense of how well "distance to AGI" is known.

### 5.4 Is the "jaggedness" real, stable and shared?

Bottleneck definitions are meaningful only if family-level differences are signal. Four measurements bear on this [ours, `profiles_1-1.json`, `cv_lambda_1-1.json`].

- **Out-of-sample signal.** As reported in §4, family offsets reduce held-out-task log-loss by 8% relative to the pooled model. Families differ in difficulty beyond what task length explains.
- **Run-level reliability.** We split each task's runs at random into halves and refit. Family-offset profiles correlate at r = 0.80 between halves (mean over 20 systems and 10 splits), giving a Spearman–Brown full-data reliability of 0.88. Reliability is higher for recent systems (0.92–0.94 for GPT-5.3-Codex and Opus 4.6). This measures run-sampling noise only, not task-sampling noise. The latter is what the hierarchical bootstrap captures, and it dominates.
- **Shared structure.** The mean pairwise correlation of family-offset profiles across the 20 systems is 0.42, and the first principal component explains 35% of profile variance. Jaggedness is partly shared, since some families are hard for every system, but far from one-dimensional. The mean profile of the five earliest systems correlates only 0.25 with that of the five latest. *Which* families are the bottleneck changes over time.
- **Identity of bottlenecks.** For the five latest systems, the three lowest-horizon families are all HCAST families. They include `automatic_jailbreak` (in all five), `local_research`, `spn_cryptanalysis` and, for two systems, short pattern or counting tasks (`count_words`, `continue_pattern`). The low ranking of these short tasks shows that bottleneck metrics pick up idiosyncratic failures, not only "hard" long tasks.

**Is the profile becoming more or less jagged?** The standard deviation of family offsets (logit units) rises by 0.16 per year (90% CI [0.075, 0.23]; 100% of bootstrap replicates positive). In log₂-horizon units the rise is 0.35 doublings per year [0.04, 0.65] (Figure 4). This would favour bottleneck definitions lagging ever further behind, which is the Prop. 2 mechanism. A ceiling check qualifies the finding [ours, `robust_dispersion_1-1.json`]. The number of saturated families (all runs succeed) rises by about 15.5 per year, and the number of floored families falls by about 13.6 per year. When dispersion is computed only over families a system neither saturates nor floors, the trend is 0.04 per year (model-resampling 90% CI [−0.02, 0.11]), which is not distinguishable from zero. The lower semi-deviation (spread of the below-median half) does rise robustly (0.35 per year, [0.22, 0.54]). Our reading is that the widening is concentrated in a lagging lower tail. We cannot fully separate it from censoring at the ceiling, and we therefore do not claim that capability profiles are intrinsically becoming more jagged.

### 5.5 Bottleneck metrics hinge on shrinkage

Prop. 4 predicts that unshrunk bottleneck estimates are biased low and that the remedy, shrinkage, introduces a tuning-dependence. The data show this strongly [ours, `profiles_1-1.json`]. For Claude Opus 4.6, the minimum-family 50% horizon is 0.17 min at λ = 0.01, 3.8 min at λ = 0.25, 9.0 min at λ = 1, 65 min at λ = 4 and 287 min at λ = 16. That is a factor of more than 1,700 across settings that differ in CV loss by only a few percent around the optimum. For o3, the range is 0.26 to 82 min. Any "weakest-domain" definition of AGI therefore needs a stated, validated shrinkage procedure. Without one, its level is close to arbitrary.

LAMBDA_SENS_PLACEHOLDER

### 5.6 Replication on Time Horizon 1.0

V10_PLACEHOLDER

### 5.7 The bounded-score case: CHC domain scores

We apply the same lens to the published CHC profiles: GPT-4 (K 8, RW 6, M 4, R 0, WM 2, MS 0, MR 4, V 0, A 0, S 3; total 27%) and GPT-5 (9, 10, 10, 7, 4, 0, 4, 4, 6, 3; total 57%), each domain out of 10 (Hendrycks et al., 2025). Note that S is 3 for both. Under p = 1 the scores are 0.27 and 0.57. Under p = 0.5 they are 0.15 and 0.49. Under p ≤ 0 both are exactly 0, because long-term memory storage (MS) is 0 for both systems. With an ε-floor on zeros the geometric mean is still decided by the floor, which is arbitrary. GPT-5 scores 0.36 at ε = 0.005 and 0.49 at ε = 0.1 [ours, `theory_chc.json`, Figure 5]. These static facts agree with Fourati (2025).

The dynamic question is new. We extrapolate each domain linearly in time between the two release dates (2023-03-14 and 2025-08-07), capped at 1. Three domains do not improve (MS stays at 0, MR at 0.4, S at 0.3). Under p = 1 the aggregate reaches 0.7 in 2027.7 and then stalls at 0.77. It never reaches 0.9 or 1.0. Under p ≤ 0.5 it never reaches 0.7. Read this way, the CHC numbers do not say that we are "57% of the way" in any temporal sense. With two snapshots, a bounded scale and stagnant domains, the time to a high threshold is *unidentified* (infinite under the linear model). The answer depends entirely on when currently stagnant capabilities, above all continual learning and memory storage, start to move. That onset is not a trend that can be extrapolated, which is the bounded-scale counterpart of Prop. 2's p < 0 regime: the slowest domain sets the forecast.

## 6 Pre-registered Predictions and Test Protocol

The following predictions are fixed by the fits above (TH1.1, λ = 1) and can be checked against run-level data for systems released after 2026-02-05, if such data are published. They can also be checked on any successor suite with the same structure.

- **P1 (stationary slope).** For the next five frontier systems on a comparable suite, the OLS trend in the pooled logistic slope b over the full window remains within [−0.08, +0.10] per year, and the 80%/50% doubling-time ratio stays within [0.90, 1.35]. *Falsified* if both fall outside these ranges in the same direction (flattening plus a slower 80% trend). That would mean Prop. 3(ii)'s rate effect has switched on and the 80% definition's arrival date moves later relative to Table 1.
- **P2 (reliability delay).** For new frontier systems, log₂H₅₀ − log₂H₈₀ = logit(0.8)/|b| holds per system by construction, and the frontier average of this gap stays within 2.4 ± 0.8 doublings (H₅₀/H₈₀ between about 3× and 9×). *Falsified* if the gap leaves this band for three consecutive frontier systems.
- **P3 (lower-tail lag).** The ratio of the pooled 50% horizon to the 10th-percentile-family 50% horizon for frontier systems does not shrink. Our point trend implies that it widens by roughly 0.5 doublings per year (difference of the two slopes in Table 1). *Falsified* if the ratio for new frontier systems falls below its 2025–2026 frontier median.
- **P4 (bottleneck turnover).** The identity of the three lowest-horizon families changes between successive frontier systems more often than it persists. Based on the profile correlation of 0.25 between early and late systems, we predict that fewer than half of a new system's bottom-3 families coincide with the bottom-3 of the frontier system two releases earlier. *Falsified* by persistent overlap of 2 or more of 3 in a majority of new systems.

**Reporting protocol.** Any claim of the form "system S is X% of the way to AGI" or "AGI-level horizon by year Y" should state (1) the domain set Ω and its weights; (2) the reliability threshold q; (3) the aggregator (p or quantile); (4) the target τ; and (5) the shrinkage or denoising used for domain estimates. It should also report (6) the arrival date or score under at least q ∈ {0.5, 0.8} and p ∈ {pooled, 0, a lower-quantile bottleneck}. That is a six-cell sensitivity table like Table 1, computable from run-level data with the code released here.

## 7 Discussion

**What the question "how far from AGI?" is actually asking.** Our results support a deflationary but constructive reading. Distance to AGI is not a property of a model alone; it is a property of a model *and* a definition. On a fixed, public, run-level dataset, reasonable definitions disagree by more than five years about when one fixed target is reached. Much of that disagreement comes from two interpretable dials: how reliable a capability must be (q) and how evenly it must be spread (p). Prop. 3 shows that the reliability dial mostly adds a constant delay, which is easy to communicate ("high-reliability AGI comes about logit(q)/|β| doublings later"). Prop. 2 shows that the breadth dial can change the *rate*, which is harder: under a bottleneck definition the forecast is held back by whichever domain improves slowest.

**Which definition should be used?** We do not claim to settle this, but the analysis narrows the options. (i) For unbounded quantities like time horizons, p > 0 aggregation should be avoided, because it is dominated by extrapolated ceilings (§5.1). (ii) Pure minima should be avoided unless a validated shrinkage procedure is stated, because their level depends strongly on tuning (§5.5). (iii) Low quantiles (10th–25th percentile) with cross-validated shrinkage are a reasonable middle ground. They encode "general" as "competent in nearly all domains", which matches Morris et al.'s (2023) generality axis, and remain estimable. (iv) For safety- or deployment-relevant claims, q should be at least 0.8, and the reliability delay should be reported.

**Relation to the "jaggedness" debate.** Capability profiles are often described as jagged (Dell'Acqua et al., 2023; Hendrycks et al., 2025). We find evidence that family-level structure is real (CV gain, high split-half reliability) and partly shared across systems, but also that the bottleneck identities rotate. If bottlenecks rotate, a bottleneck forecast may be more optimistic than Prop. 2's worst case: no single domain stays slowest for long. If the widening of the lower tail is real rather than censoring, the forecast is more pessimistic. Telling these apart needs larger task families and suites with more headroom, which is a concrete request to benchmark builders.

## 8 Limitations

- **Narrow domain coverage.** METR's suite consists of software, ML-research and cybersecurity tasks. "Task families" are not cognitive domains, and our aggregator results speak to breadth *within* this suite, not to AGI breadth. We use the suite because it is the only public source with run-level data, human timings and a multi-year span. The CHC analysis covers broader domains but has only two data points.
- **Extrapolation far beyond the data.** The 167-hour target is about 6× the longest tasks and about 10× METR's stated reliability limit of 16 hours (METR, 2026). The arrival dates are best read as a common yardstick for comparing definitions. They are not forecasts, and trend breaks of any kind would invalidate them.
- **Small families and censoring.** Median family size is 2 tasks. Family horizons for saturated or floored families are determined mostly by the prior (ridge) and by the shared logistic slope. This affects bottleneck and p = +1 aggregates most, and the dispersion-trend result is not robust to excluding such families (§5.4).
- **Model form.** We assume a common slope across families within a system and a logistic link. Ord's (2025) hazard-based models, or family-specific slopes, would change high-q tails. Rows with q ≥ 0.99 are illustrative.
- **Frontier selection and linear trends.** Running-maximum frontiers and OLS on log horizons follow METR's practice but ignore uncertainty in which systems are frontier and the possibility of curvature. Release dates come from METR's file.
- **Not an exact reproduction.** Our pooled estimates are close to but not identical with METR's published values, because weighting and regularisation details differ. Conclusions about *relative* effects of definitions do not depend on matching METR's exact levels.
- **No new model evaluations.** We ran no models. All capability data are third-party, and errors or scaffold differences in those data carry through to our results.

## 9 Conclusion

Asking "how far are we from AGI?" means choosing a definition, and on today's best public run-level data that choice moves the answer by years. We formalised definitions as (domains, reliability, aggregator, target) tuples. We proved that reliability thresholds add a constant, computable delay unless the logistic slope drifts, that the sign of the aggregation exponent decides which domain's growth rate governs the long-run forecast, and that bottleneck estimates are biased toward pessimism. On METR data, these choices account for 43% of the variance in extrapolated arrival dates, comparable to all sampling uncertainty. The practical recommendation is modest and checkable. Report forecasts as a small sensitivity table over q and aggregator, use cross-validated shrinkage for any bottleneck measure, and treat bounded-scale percentages as levels rather than as fractions of a journey with a known end.

## References

Boucheron, S., Lugosi, G., & Massart, P. (2013). *Concentration Inequalities: A Nonasymptotic Theory of Independence.* Oxford University Press.

Burnell, R., Schellaert, W., Burden, J., Ullman, T. D., Martínez-Plumed, F., Tenenbaum, J. B., Rutar, D., Cheke, L. G., Sohl-Dickstein, J., Mitchell, M., Kiela, D., Shanahan, M., Voorhees, E. M., Cohn, A. G., Leibo, J. Z., & Hernández-Orallo, J. (2023). Rethink reporting of evaluation results in AI. *Science*, 380(6641), 136–138.

Burnell, R., Yamamori, Y., Firat, O., Olszewska, K., Hughes-Fitt, S., Kelly, O., et al. (2026). Measuring progress toward AGI: A cognitive framework. arXiv:2605.28405.

Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547.

Dell'Acqua, F., McFowland III, E., Mollick, E., Lifshitz-Assaf, H., Kellogg, K. C., Rajendran, S., Krayer, L., Candelon, F., & Lakhani, K. R. (2023). Navigating the jagged technological frontier: Field experimental evidence of the effects of AI on knowledge worker productivity and quality. Harvard Business School Working Paper No. 24-013.

Fourati, F. (2025). A coherence-based measure of AGI. arXiv:2510.20784 (FAST@AAAI 2026).

Grace, K., Sandkühler, J. F., Stewart, H., Weinstein-Raun, B., Thomas, S., Stein-Perlman, Z., Salvatier, J., Brauner, J., Korzekwa, R. C., et al. (2025). Thousands of AI authors on the future of AI. *Journal of Artificial Intelligence Research* (published 6 Oct 2025); arXiv:2401.02843 (2024).

Hardy, G. H., Littlewood, J. E., & Pólya, G. (1952). *Inequalities* (2nd ed.; 1st ed. 1934). Cambridge University Press.

Hendrycks, D., Song, D., Szegedy, C., Lee, H., Gal, Y., Brynjolfsson, E., Li, S., Zou, A., Levine, L., Han, B., Fu, J., Liu, Z., Shin, J., Lee, K., Mazeika, M., Phan, L., Ingebretsen, G., Khoja, A., Xie, C., Salaudeen, O., Hein, M., Zhao, K., Pan, A., Duvenaud, D., Li, B., Omohundro, S., Alfour, G., Tegmark, M., McGrew, K., Marcus, G., Tallinn, J., Schmidt, E., & Bengio, Y. (2025). A definition of AGI. arXiv:2510.18212.

Kwa, T., West, B., Becker, J., Deng, A., Garcia, K., Hasin, M., Jawhar, S., Kinniment, M., Rush, N., Von Arx, S., et al. (2025). Measuring AI ability to complete long tasks. arXiv:2503.14499.

Legg, S., & Hutter, M. (2007). Universal intelligence: A definition of machine intelligence. *Minds and Machines*, 17(4), 391–444.

McGrew, K. S. (2009). CHC theory and the human cognitive abilities project: Standing on the shoulders of the giants of psychometric intelligence research. *Intelligence*, 37(1), 1–10.

METR (2025). How does time horizon vary across domains? METR blog, 14 July 2025. https://metr.org/blog/2025-07-14-how-does-time-horizon-vary-across-domains

METR (2026). Task-completion time horizons of frontier AI models (web page, accessed November 2026). https://metr.org/time-horizons. Run-level data: https://github.com/METR/eval-analysis-public

Morris, M. R., et al. (2023). Levels of AGI for operationalizing progress on the path to AGI. arXiv:2311.02462 (ICML 2024 position paper).

Ord, T. (2025). Is there a half-life for the success rates of AI agents? arXiv:2505.05115.

Ruan, Y., Maddison, C. J., & Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024).

Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *Advances in Neural Information Processing Systems 36* (NeurIPS 2023).

---

**Figures** (in `results/`): Fig. 1 `fig_theory.png` (Props. 2 and 4, numerical); Fig. 2 `fig_trajectories.png` (five definitions on TH1.1); Fig. 3 `fig_arrival_forest.png` (all 24 definitions with 90% intervals); Fig. 4 `fig_dispersion_slope.png` (family-offset dispersion and logistic slope over time); Fig. 5 `fig_chc.png` (CHC scores under power means).
