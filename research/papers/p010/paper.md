# How Much Does the Definition Move the Forecast? Reliability Thresholds, Cross-Family Aggregation and METR Time-Horizon Extrapolations

*Target venue: NeurIPS 2027, Datasets & Benchmarks Track (evaluation methodology). Revised version (round 2).*

---

## Abstract

"How far are we from AGI?" is often answered by extrapolating one scalar, such as METR's 50% task-completion time horizon. Any such scalar fixes three choices: the domains, the success rate that counts as "can do it" (the reliability threshold q), and the rule that combines domains into one number (the aggregator). This paper is a methodology study of how much these choices move extrapolated arrival dates on the one public source with run-level data, human task timings and a multi-year span: METR's time-horizon suites. It is not a new definition of AGI, and it does not forecast AGI. We use 23,235 agent runs of 20 systems on 228 tasks in 79 task families (Time Horizon 1.1, March 2023 to February 2026). We replicate on 37,539 agent runs of 30 systems (Time Horizon 1.0). We collect four standard facts in one framework: power-mean ordering, the softmax form of a power mean's growth rate (including a new first-order finite-horizon version), the logistic "reliability delay" of logit(q)/|β| doublings, and the downward bias of noisy minima. We then cross 3 thresholds with 8 aggregators over task families. Across all 24 definitions, the extrapolated date at which the frontier reaches a 167-hour horizon ranges from 2027.1 to 2032.6. Dropping the two aggregators we argue against narrows this to 2027.5–2032.1. Only one definitional dial is well identified on these data: raising q from 0.5 to 0.8 under the pooled model delays the date by 0.95 years (closed form 0.85), and the logistic slope shows no trend. Family-level aggregates are not well identified. For 20 of the 21 family-level definitions, 62–100% of the frontier family horizons that drive them lie more than two doublings outside the lengths of their family's tasks, and 29–100% come from families with all-success or all-failure outcomes. Their dates also move by up to 5.3 years with the shrinkage rule. Under task resampling alone, definition choice accounts for about half of arrival-date variance on TH1.1 (nested-bootstrap 90% CI 0.40–0.68 for all 24 definitions, 0.35–0.68 for the 50%-vs-80% pair). The share is similar for every menu we tried. Adding human-time error, leave-one-system-out, release-date jitter, trend-window choice and a robust trend estimator reduces it to 0.30–0.43, and the window choice matters most. On TH1.0 the share is smaller (0.12–0.32). Bottleneck aggregators show slower doubling in a paired bootstrap on TH1.1 (e.g. 25th-percentile family vs pooled: −0.72 doublings/yr, 90% CI [−1.03, −0.11]). That effect does not survive trend-specification uncertainty and does not replicate on TH1.0. Family growth rates are heterogeneous (permutation p < 0.001), but initially weak families catch up rather than lag. We end with prespecified predictions and a reporting protocol. Read in AGI terms, the evidence supports a modest claim: on today's best run-level data, "how reliable" moves extrapolated dates by about a year per step from 50% to 80%, while "how broad" cannot yet be measured well enough to say.

## 中文摘要

"我们离AGI还有多远？"常见的回答是外推某个单一标量，例如METR的50%任务完成时间跨度（time horizon）。任何此类标量都隐含三项选择：领域集合、算作"能完成"的成功率阈值q、以及把多个领域合成一个数的聚合规则。本文是一项方法学研究，考察这些选择在多大程度上改变外推的"到达日期"。数据来自目前唯一同时具备逐次运行记录、人类任务计时和多年跨度的公开来源，即METR的时间跨度任务集。本文既不提出新的AGI定义，也不预测AGI。主分析使用Time Horizon 1.1中20个系统在79个任务族、228个任务上的23,235次智能体运行（2023年3月至2026年2月），并在Time Horizon 1.0的30个系统、37,539次运行上复现。我们把四条标准结论放进同一框架：幂平均的序关系；幂平均增长率的softmax形式（含一个新的有限时间一阶展开）；logistic模型下logit(q)/|β|个倍增期的"可靠性延迟"；以及带噪声最小值的向下偏差。随后把3个阈值与8种跨任务族聚合方式组合成24种定义。全部24种定义下，前沿达到167小时时间跨度的外推日期从2027.1年到2032.6年不等；去掉本文不推荐的两种聚合方式后为2027.5–2032.1年。在这些数据上只有一个定义维度是可靠识别的：在合并模型下把q从0.5提高到0.8，日期推迟0.95年（闭式解0.85年），且logistic斜率没有趋势。任务族层面的聚合量识别不足：在21种任务族层面的定义中，有20种的驱动性前沿任务族时间跨度有62%到100%落在该族任务时长范围之外两个倍增以上，另有29%到100%来自全部成功或全部失败的任务族；随收缩规则不同，其日期最多相差5.3年。仅考虑任务重抽样时，TH1.1上约一半的到达日期方差来自定义选择（全部24种定义的嵌套bootstrap 90%区间为0.40–0.68；50%对80%这一对为0.35–0.68），且在我们尝试过的各种定义菜单下都相近。若再加入人类计时误差、逐一剔除系统、发布日期扰动、趋势窗口选择和稳健趋势估计，这一比例降到0.30–0.43，其中窗口选择影响最大；在TH1.0上更小（0.12–0.32）。在TH1.1的配对bootstrap中，瓶颈型聚合方式的倍增更慢（例如第25百分位任务族相对合并模型为每年−0.72个倍增，90%区间[−1.03, −0.11]），但该效应在考虑趋势设定不确定性后不再显著，且未在TH1.0上复现。各任务族的增长率存在异质性（置换检验p < 0.001），但起初较弱的任务族是在追赶而非落后。最后我们给出事先设定的预测和一套报告规范。就AGI问题而言，证据只支持一个有限的结论：在现有最好的逐次运行数据上，"要求多可靠"每从50%提高一档到80%，外推日期约推迟一年；而"要求多全面"目前还无法测得足够准确来下结论。

---

## 1 Introduction

Two quantitative answers to "how far are we from AGI?" were widely discussed in 2025–2026. The first is psychometric. Hendrycks et al. (2025) define AGI as matching "the cognitive versatility and proficiency of a well-educated adult" and score ten Cattell–Horn–Carroll (CHC) domains (McGrew, 2009), each weighted 10%. Under this scheme GPT-4 scores 27% and GPT-5 57%. A Google DeepMind team proposed a related ten-faculty taxonomy (Burnell et al., 2026). The second answer is behavioural. METR measures the length of software tasks, in human-expert minutes, that an agent completes with 50% probability. This "time horizon" has doubled roughly every seven months since 2019, and faster recently (Kwa et al., 2025). Extrapolating it gives dates at which agents might complete month-long tasks.

Both answers reduce a jagged capability profile to one number and extrapolate it. Fourati (2025) showed that the CHC percentage depends strongly on the aggregator. Kwa et al. (2025) and Ord (2025) discuss 80% as well as 50% horizons. METR (2025) reports similar doubling rates but very different absolute levels across nine benchmark domains. METR's own analyses include hierarchical bootstraps and sensitivity checks. Our question is narrower:

**Question.** *For one fixed body of run-level evidence, how much do the reliability threshold and the cross-family aggregator move the extrapolated arrival date and doubling time? Which of these effects are statistically identified, compared with the non-definitional uncertainty that any such extrapolation carries?*

**What is new relative to prior work.** Most of the theory here is standard, and we say so in §3. The contributions are empirical and methodological:

1. **A definition-by-uncertainty decomposition on run-level data (§5.3).** We report the definitional share of arrival-date variance for several definition menus. The non-definitional side includes task resampling, human-time measurement error, leave-one-system-out, release-date jitter, trend-window choice and trend estimator. Confidence intervals come from a nested bootstrap.
2. **Identification diagnostics for family-level definitions (§5.4–5.5).** We measure task support, censoring and shrinkage dependence (CV, per-system empirical Bayes, run-count-scaled ridge). Together these show which definitional dials the METR suite can and cannot inform.
3. **Direct tests of the "aggregators change rates" mechanism (§5.2).** We use a paired bootstrap of slope differences and family-specific growth rates with a permutation test, on both suite versions.
4. **A reliability-delay check (§5.1).** The closed-form logit(q)/|β| delay is compared with direct fits, and its stationarity condition is tested.
5. **Prespecified predictions and a reporting protocol (§6),** with released code.

We use "AGI" only in the motivating sense above. Task families in METR's software/ML/cybersecurity suite are not cognitive domains, and §7 states exactly what our results do and do not imply about the AGI question.

All numbers are either produced by code in `code/` and saved in `results/` (marked [ours]) or taken from a cited source.

## 2 Related Work

**Defining AGI.** Legg and Hutter (2007) define universal intelligence as simplicity-weighted performance over all computable environments. Chollet (2019) defines intelligence as skill-acquisition efficiency. Morris et al. (2023) propose "Levels of AGI" along performance and generality axes, which implicitly uses a quantile-type aggregator. Hendrycks et al. (2025) use ten equally weighted CHC domains and an arithmetic mean. Burnell et al. (2026) propose cognitive profiles rather than a single score. We take any such profile as input and study how its reduction to a scalar shapes the extrapolation.

**Aggregation.** Power means and their ordering are classical (Hardy, Littlewood & Pólya, 1952). Fourati (2025) applies compensability-indexed generalised means to the CHC scores and summarises them across p. His analysis is static. Burnell et al. (2023) call for instance-level reporting. Our dynamic statements about rates follow from the gradient of log-sum-exp and are elementary. What we add is their empirical testing on run-level data, together with identification diagnostics.

**Time horizons and thresholds.** Kwa et al. (2025) introduced the 50% horizon and reported similar 50% and 80% doubling times (212 vs 213 days, as quoted by Ord, 2025). Ord (2025) models a constant hazard per human-minute. METR (2025) compares domains. METR's time-horizons page states that measurements above 16 hours are unreliable with the current suite (METR, 2026). That the H80/H50 ratio is fixed by the logistic slope is implicit in all of these. Our Prop. 3 only states this explicitly and adds the stationarity test.

**Metric artefacts and forecasts.** Schaeffer et al. (2023) show that apparent emergence can come from nonlinear metrics. Ruan et al. (2024) find low-dimensional capability structure. Grace et al. (2025) report that elicited forecasts differ by decades between the "all tasks" framing (50% year 2047) and the "all occupations" framing (2116). Our work quantifies framing effects on measured rather than elicited quantities.

## 3 Framework: four standard facts

Let Ω = {1,…,n} be domains (here, task families) with capability x_i(t) > 0 (a time horizon in minutes) and weights w_i > 0, Σw_i = 1. The power mean is A_p(x) = (Σ w_i x_i^p)^{1/p}, with A_0 = Π x_i^{w_i} and A_{±∞} = max/min. The α-quantile aggregator Q_α(x) is the α-quantile of {x_i}. A **definition** is D = (Ω, q, A, τ): domains, reliability threshold, aggregator and target. The **arrival time** is T_D = inf{t : A(x(t)) ≥ τ}. None of the following results is deep. We state them because they organise the empirical tests.

**Proposition 1 (ordering).** (a) For p < p′, T under A_{p′} ≤ T under A_p. (b) For α < α′, T under Q_{α′} ≤ T under Q_α. (c) Quantiles are *not* ordered relative to power means in general. Only min = A_{−∞} ≤ Q_α ≤ A_{+∞} = max holds.

*Proof.* (a) A_p ≤ A_{p′} pointwise (power-mean inequality; Hardy et al., 1952, ch. 2), so every time at which A_p ≥ τ also has A_{p′} ≥ τ. No continuity is needed. (b) Q_α ≤ Q_{α′} pointwise. (c) Counterexample: x = (ε, 1, 1) has median 1 > A_0 = ε^{1/3} for ε < 1. Empirically, the median-family horizon of Claude Opus 4.6 (1,973 min) exceeds its geometric mean (935 min) (Table 1). ∎

The round-1 version of this paper wrongly placed quantiles "between p = 0 and p = −∞". Our checks of empirical ordering therefore use only the power-mean chain (p = +1, 0, −1, −∞) and, separately, the quantile chain (50th, 25th, 10th percentile, min).

**Proposition 2 (growth rate of a power mean).** Let ln x_i(t) = a_i + s_i t. For p ≠ 0,
 r_p(t) := d/dt ln A_p(x(t)) = Σ_i π_i(t) s_i, with π_i(t) ∝ w_i exp(p(a_i + s_i t)),
and r_0 = s̄ := Σ w_i s_i. (i) As t → ∞, r_p → max_i s_i for p > 0 and → min_i s_i for p < 0. (ii) *Finite horizon.* r_p(t) is smooth in p for every fixed t, and
 r_p(t) = s̄ + p·[Cov_w(a, s) + t·Var_w(s)] + O(p²).

*Proof.* ln A_p = p⁻¹ ln Σ w_i e^{p(a_i+s_i t)}. Its time derivative is the π-weighted mean of the s_i, and (i) follows because the logits diverge. For (ii), ∂_p Σ_i π_i s_i = Cov_π(a + s t, s), which at p = 0 equals Cov_w(a, s) + t Var_w(s). ∎

Part (ii) corrects an overstatement in round 1. The rate is *continuous* in p at every finite horizon. The "jump at p = 0" exists only in the t → ∞ limit, and the time needed to approach it grows like 1/(|p|·Δs). For the empirical question, (ii) says the rate gap between geometric and bottleneck aggregation grows with the *cross-family variance of growth rates* and with the *covariance between level and rate*. A negative covariance means weak families catch up, and it shrinks or reverses the gap. Both quantities can be estimated (§5.2).

**Proposition 3 (reliability delay).** If P(success | h, t) = σ(α(t) + β(t) log₂h) with β < 0, then log₂H_q − log₂H_{1/2} = logit(q)/β(t). At fixed β the q-definition therefore lags by logit(q)/|β| doublings. Moreover, d/dt log₂H_q = −α′/β − log₂H_q·β′/β, so two thresholds share a doubling time for all t if and only if β′ ≡ 0.

*Proof.* Solve σ(α + β log₂H) = q for log₂H, then differentiate. ∎

This algebra is implicit in Kwa et al. (2025) and Ord (2025). It is useful here because it turns the threshold choice into two estimable quantities, |β| and β′.

**Proposition 4 (noisy minima).** If x̂_i = x_i + ε_i with E ε_i = 0 (log scale), then E min_i x̂_i ≤ min_i x_i. If the ε_i are i.i.d. N(0, σ²), the bias is at most σ√(2 ln n) (Boucheron, Lugosi & Massart, 2013, ch. 2). This is a textbook extreme-value bound. In our estimator, ridge shrinkage biases offsets toward zero and family noise is heteroscedastic, so the proposition gives only qualitative guidance for §5.5. The numerical illustration (Fig. 1) shows a bias of −1.0σ for 20 well-separated domains and −1.87σ when they are tied [ours, `theory_chc.json`].

## 4 Data and Estimation

**Data.** We use METR's public run-level files (`METR/eval-analysis-public`, commit `52cb829`, 2026-03-05, the repository's latest commit when we checked on 2026-11-22). Our copies are byte-identical to that commit (SHA-256 hashes in `README.md`). That commit adds GPT-5.3-Codex and Claude Opus 4.6 and includes METR's "analysis updates" of March 2026. The repository contains no run-level data for systems released after 2026-02-05, so later METR updates cannot be analysed here. The TH1.1 file has 24,008 rows, of which 773 are human baselines. We analyse the **23,235 agent runs** of 20 systems on 228 tasks in 79 families (HCAST, RE-Bench, SWAA), from GPT-4 0314 (2023-03-14) to Claude Opus 4.6 and GPT-5.3-Codex (2026-02-05). The TH1.0 file has 41,629 rows, of which 793 are human baselines. It has 40,836 agent runs, of which we analyse the **37,539 runs** of the 30 systems released from 2023 onward (170 tasks, 50 families). Median family size is 2 tasks in both versions. In TH1.1, task lengths come from human baselines for 162 tasks and from METR estimates for 66. Only 5 of the 31 tasks of 8 hours or more have baseline-measured lengths. Each run carries METR's diversity weight (`invsqrt_task_weight`), which we rescale to sum to the number of runs per system.

**Model.** For system m and run r on a task of family f(r) with human length h_r,
 logit P(y_r = 1) = a_m + d_{m,f(r)} + b_m log₂h_r, with ridge penalty (λ/2) Σ_f d_{m,f}².
d ≡ 0 is the pooled METR-style fit. The family q-horizon is log₂H_{m,f}(q) = (logit q − a_m − d_{m,f})/b_m. The slope is common across families within a system, which is a restriction (§8). We fit by penalised IRLS on run data collapsed to (family, task-length) cells. This is exact for the weighted Bernoulli likelihood, and it reproduces the round-1 estimates to 10⁻¹⁴. The 24 definitions cross q ∈ {0.5, 0.8, 0.95} with eight aggregators: pooled; power means p ∈ {+1, 0, −1}; family 50th, 25th and 10th percentiles; and the minimum family.

**Shrinkage.** λ = 1 was fixed before analysis and then checked by 5-fold cross-validation over tasks (2 repetitions). A held-out task from a family absent in training gets offset 0. The weighted held-out log-loss is 0.3542 at λ = 1, 0.3550 at λ = 2, 0.3691 at λ = 0.25 and 0.3852 for the pooled model [ours, `cv_lambda_1-1.json`]. §5.5 adds per-system empirical-Bayes and run-count-scaled alternatives.

**Trends and arrival dates.** For each definition we take the running-maximum frontier in release order and regress log₂ horizon on release date by OLS. The doubling time is DT = 12/slope months. We extrapolate to 167 h (one work-month). That target is about 6× the longest task and about 10× the 16-h reliability limit METR states (METR, 2026). Arrival dates are therefore *a common yardstick for comparing definitions*, not forecasts.

**Uncertainty.** We use five layers of uncertainty, all implemented in `ext_bootstrap.py` and `decompose.py`.
- (U1) *Task resampling.* A hierarchical bootstrap over families, then tasks within families. All runs are kept and every system is refitted (B = 500 for the main tables, B = 1,000 for the decomposition).
- (U2) *Human-time error.* Per-task log₂-length noise with SD 0.5 for baselined tasks and 1.0 for estimated tasks, plus one shared shift with SD 0.5 for all estimated tasks. These magnitudes are **assumptions**, not estimates, chosen to represent roughly 1.4× and 2× per-task error and a systematic mis-estimation of long tasks.
- (U3) *Frontier composition.* One randomly chosen system is dropped (leave-one-system-out).
- (U4) *Release-date jitter.* Uniform ±45 days per system.
- (U5) *Trend specification.* The window starts in 2023 or in 2024 (equal probability), and the trend estimator is OLS or Theil–Sen (equal probability).
U3–U5 are drawn once per replicate and shared by all definitions, so they enter only the non-definitional side. For the variance share S = V_between/(V_between + V_within), we report a **nested bootstrap** CI (40 outer × 100 inner replicates for TH1.1, 30 × 80 for TH1.0, task resampling only; `nested_share.py`). Arrival dates are roughly 1/slope and heavy-tailed, so we also report a robust share that replaces variances with squared MAD scales. Total CPU time for all analyses is about 45 minutes on 3–4 cores.

**Reproduction check.** Our pooled TH1.1 50% horizons are 4.0 min for GPT-4 0314, 203 min for GPT-5 and 719 min for Claude Opus 4.6 [ours]. These are close to, though not identical with, METR's published values; METR's regularisation and weighting choices differ slightly from ours. The pooled 50% doubling time on the 2023–2026 frontier is 4.2 months [3.4, 5.1].

## 5 Results

**Table 1.** TH1.1, λ = 1, B = 500. DT = doubling time (months). Arrival = date at which the extrapolated frontier reaches 167 h. Brackets: 90% task-bootstrap intervals. "Latest" = level for Claude Opus 4.6 (minutes). "Off-support" = share of frontier points whose determining family horizon lies more than 2 doublings outside that family's task lengths. For power means this is the mean share over families; pooled horizons are compared with the suite-wide range. "Cens." = share of frontier points whose determining family has all-success or all-failure outcomes for that system [ours, `metr_1-1_lam1_summary.csv`, `decomp_1-1.json`].

| Aggregator | q | DT (mo) | Arrival 167 h | Latest (min) | Off-support | Cens. |
|---|---|---|---|---|---|---|
| pooled (METR-style) | 0.5 | 4.2 [3.4, 5.1] | 2027.7 [2027.2, 2028.3] | 719 | 0.00 | – |
| power mean p=+1 | 0.5 | 3.7 [3.0, 4.8] | 2027.1 [2026.5, 2027.9] | 3,363 | 0.72 | 0.64 |
| geometric (p=0) | 0.5 | 4.2 [3.5, 5.2] | 2027.7 [2027.1, 2028.3] | 935 | 0.71 | 0.65 |
| power mean p=−1 | 0.5 | 4.9 [4.1, 6.2] | 2028.5 [2027.9, 2029.3] | 105 | 0.69 | 0.65 |
| median family | 0.5 | 3.9 [3.1, 4.9] | 2027.5 [2026.8, 2028.2] | 1,973 | 1.00 | 0.77 |
| 25th-pct family | 0.5 | 5.7 [4.0, 7.0] | 2029.2 [2027.8, 2030.0] | 95 | 0.91 | 0.73 |
| 10th-pct family | 0.5 | 5.1 [4.2, 7.5] | 2029.0 [2028.3, 2030.7] | 32 | 0.70 | 0.60 |
| worst family | 0.5 | 4.9 [4.0, 7.4] | 2029.6 [2028.5, 2031.9] | 9.0 | 0.33 | 0.42 |
| pooled | 0.8 | 4.3 [3.6, 5.9] | 2028.7 [2028.0, 2029.8] | 70 | 0.00 | – |
| power mean p=+1 | 0.8 | 4.1 [3.5, 5.0] | 2028.1 [2027.6, 2028.7] | 354 | 0.67 | 0.65 |
| geometric (p=0) | 0.8 | 4.9 [3.9, 6.0] | 2028.9 [2028.2, 2029.6] | 98 | 0.65 | 0.65 |
| power mean p=−1 | 0.8 | 5.4 [4.0, 7.2] | 2029.7 [2028.5, 2031.1] | 11 | 0.65 | 0.63 |
| median family | 0.8 | 4.4 [3.6, 5.5] | 2028.5 [2027.8, 2029.3] | 207 | 0.90 | 0.60 |
| 25th-pct family | 0.8 | 6.3 [4.2, 8.3] | 2030.6 [2028.6, 2032.3] | 10 | 1.00 | 1.00 |
| 10th-pct family | 0.8 | 5.3 [4.0, 8.6] | 2030.1 [2028.7, 2032.6] | 3.4 | 1.00 | 0.62 |
| worst family | 0.8 | 6.3 [3.8, 9.1] | 2031.7 [2028.9, 2035.1] | 1.0 | 0.62 | 0.38 |
| pooled | 0.95 | 4.8 [3.4, 7.2] | 2030.0 [2028.5, 2032.2] | 5.1 | 0.00 | – |
| power mean p=+1 | 0.95 | 4.9 [3.9, 6.0] | 2029.6 [2028.6, 2030.4] | 28 | 0.74 | 0.63 |
| geometric (p=0) | 0.95 | 5.2 [3.9, 7.3] | 2030.2 [2028.7, 2032.0] | 7.8 | 0.74 | 0.63 |
| power mean p=−1 | 0.95 | 6.0 [3.8, 8.7] | 2031.2 [2028.8, 2033.7] | 0.9 | 0.73 | 0.64 |
| median family | 0.95 | 5.5 [3.8, 6.7] | 2030.5 [2028.7, 2031.5] | 16.5 | 0.67 | 0.67 |
| 25th-pct family | 0.95 | 6.8 [4.1, 9.7] | 2032.1 [2029.2, 2035.2] | 0.8 | 1.00 | 0.83 |
| 10th-pct family | 0.95 | 5.4 [3.8, 9.9] | 2030.9 [2029.1, 2035.5] | 0.3 | 0.83 | 0.67 |
| worst family | 0.95 | 6.3 [3.5, 11.2] | 2032.6 [2029.1, 2038.2] | 0.1 | 0.86 | 0.29 |

The point estimates span 2027.1–2032.6 across all 24 rows. They span 2027.5–2032.1 without p = +1 and the minimum, and 2027.7–2030.6 for a "recommended" menu of pooled, geometric, 25th- and 10th-percentile aggregators at q ∈ {0.5, 0.8} [ours, `decomp_1-1.json`]. Arrival is non-decreasing along the power-mean chain in 9 of 9 adjacent pairs and rises with q for 8 of 8 aggregators. Both hold at λ ∈ {0.25, 1, 4} [ours, `robustness.json`]. The last two columns qualify every family-level row; see §5.4.

### 5.1 The reliability dial is identified and adds a near-constant delay

The mean pooled logistic slope of frontier systems is b̄ = −0.573. With DT₅₀ = 4.23 months, Prop. 3 predicts the delays in Table 2 [ours, `reliability_tax.json`].

**Table 2.** Closed-form reliability delay vs direct per-q trend fits (pooled aggregator, 167-h target).

| q | doublings logit(q)/|b̄| | predicted delay (yr) | direct-fit delay (yr) | direct DT (mo) |
|---|---|---|---|---|
| 0.8 | 2.42 | 0.85 | 0.95 | 4.3 |
| 0.9 | 3.84 | 1.35 | 1.41 | 4.3 |
| 0.95 | 5.14 | 1.81 | 2.32 | 4.8 |
| 0.99 | 8.02 | 2.83 | 3.34 | 4.8 |
| 0.999 | 12.06 | 4.25 | 4.50 | 4.7 |

The closed form tracks the direct fits within 0.1 years at q ≤ 0.9. The q ≥ 0.95 rows extrapolate the logistic tail, where hazard-type models (Ord, 2025) imply different tails, so they are illustrative. The pooled-slope trend is +0.012 per year (90% CI [−0.077, 0.096]), and the ratio of 80% to 50% doubling times is 1.02 [0.90, 1.34] [ours, `metr_1-1_lam1_slopes.json`]. Both are consistent with a stationary β (Prop. 3). Pooled horizons at q ≤ 0.8 lie inside the suite's task-length range for every frontier system (Table 1). The 50%→80% step (0.95 yr on TH1.1, 0.91 yr on TH1.0) is therefore the one definitional effect that is both in support and replicated.

### 5.2 Do aggregators change rates? A paired test

Round 1 claimed that doubling times rise as p decreases, citing heavily overlapping intervals. We now test slope differences (doublings per year) *within* each bootstrap replicate [ours, `decomp_{v}.json`, `paired_slope_tests`].

**Table 3.** Slope of aggregator minus slope of pooled definition, q = 0.5 (negative = slower growth). 90% intervals; "P(<0)" = share of replicates with a negative difference.

| | TH1.1, task resampling | TH1.1, all uncertainty (U1–U5) | TH1.0, task resampling |
|---|---|---|---|
| geometric − pooled | −0.00 [−0.23, 0.14], P=0.61 | −0.00 [−0.22, 0.40], P=0.34 | — |
| p=−1 − pooled | −0.38 [−0.93, −0.06], P=0.98 | −0.38 [−1.27, 0.25], P=0.86 | +0.10 [−0.60, 0.33], P=0.51 |
| 25th-pct − pooled | −0.72 [−1.03, −0.11], P=0.98 | −0.72 [−1.26, 0.28], P=0.88 | −0.05 [−0.42, 0.40], P=0.53 |
| 10th-pct − pooled | −0.48 [−1.21, 0.00], P=0.95 | −0.48 [−1.61, 0.29], P=0.87 | +0.20 [−0.42, 0.53], P=0.39 |
| min − pooled | −0.40 [−1.42, 0.29], P=0.86 | −0.40 [−1.74, 0.73], P=0.76 | −0.63 [−1.38, 0.70], P=0.68 |
| median − pooled | +0.26 [−0.06, 0.52], P=0.11 | +0.26 [0.03, 1.30], P=0.04 | — |

On TH1.1, under task resampling alone, the p = −1 and 25th-percentile aggregates grow significantly more slowly than the pooled one. The difference is about 0.4–0.7 doublings per year on a base of about 2.8. Once trend-window and estimator uncertainty are added, every interval includes zero. On TH1.0 the bottleneck–pooled differences are centred near zero. **We therefore withdraw the round-1 claim that aggregators change rates on these data.** What survives is a TH1.1-specific, specification-sensitive tendency.

**Family-specific rates.** As a direct test of the Prop. 2 inputs, we regress each family's log₂-horizon offset (−d_{m,f}/b_m) on release date across systems [ours, `family_rates_1-1.csv`]. The rates are clearly heterogeneous. The SD of relative family rates is 0.86 doublings/yr (task-bootstrap 90% CI [0.68, 1.09]). Permuting release dates across systems gives 0.35 (95th percentile 0.49; p < 0.001). TH1.0 gives 0.70 vs 0.28 (p < 0.001). However, the covariance has the "catch-up" sign. A family's level among the five earliest systems correlates −0.30 (TH1.0: −0.38) with its growth rate estimated from the *later* systems only, so regression to the mean from shared data does not drive this. By Prop. 2(ii) a negative Cov(a, s) works against bottleneck lag. The plug-in prediction for the p = −1 vs geometric rate gap, from current family levels and rates, is −0.13 doublings/yr, against an observed −0.38. On TH1.0 it is −0.31 against an observed −0.03. The model of persistent slow families does not quantitatively explain either suite. Some of the catch-up is probably censoring. The number of saturated families grows by 15.5 per year and the number of floored families falls by 13.6 per year, which mechanically pulls extreme offsets inward.

### 5.3 How much of the uncertainty is definitional?

For a menu of D definitions and B replicates of arrival dates A_{b,d}, V_between = Var_d(mean_b A_{b,d}) and V_within = mean_d Var_b(A_{b,d}).

**Table 4.** Definitional share S of 167-h arrival-date variance [ours, `decomp_{v}_table.csv`, `nested_share_{v}.json`]. Task resampling uses B = 1,000 (TH1.1) or 500 (TH1.0); brackets give nested-bootstrap 90% CIs. "All" adds U2–U5. The robust share (MAD scale) is in parentheses.

| Menu | TH1.1 task resampling | TH1.1 + time error | TH1.1 + window | TH1.1 all (U1–U5) | TH1.0 task resampling | TH1.0 all |
|---|---|---|---|---|---|---|
| all 24 | 0.52 [0.40, 0.68] (0.56) | 0.49 | 0.30 | 0.32 (0.36) | 0.32 [0.07, 0.54] (0.27) | 0.26 |
| drop p=+1 and min (18) | 0.53 [0.38, 0.65] (0.55) | 0.48 | 0.27 | 0.30 (0.32) | 0.15 [0.04, 0.41] (0.14) | 0.12 |
| q ≤ 0.8, defensible (12) | 0.58 [0.44, 0.69] (0.59) | 0.54 | 0.32 | 0.36 (0.36) | 0.19 [0.05, 0.47] | 0.16 |
| recommended (8) | 0.55 [0.42, 0.68] (0.57) | 0.51 | 0.29 | 0.33 (0.33) | 0.15 [0.03, 0.41] | 0.13 |
| aggregator only, q=0.5 (6) | 0.60 [0.45, 0.71] | 0.60 | 0.37 | 0.43 | 0.21 [0.07, 0.48] | 0.20 |
| pooled, q=0.5 vs 0.8 | 0.58 [0.35, 0.68] | 0.60 | 0.36 | 0.42 (0.44) | 0.19 [0.01, 0.42] | 0.20 |

Five points follow.

1. **The menu matters less than round 1 implied on TH1.1, but more on TH1.0.** On TH1.1 the share is 0.52–0.60 for every menu under task resampling. Removing the definitions we disavow, or the logistic-tail rows, does not reduce it. On TH1.0, removing p = +1 and the minimum halves it (0.32 → 0.15), because there the minimum-family row (DT 8.3 months, arrival 2032.8) carries most of the between-definition spread.
2. **The share has a wide CI.** Even with task resampling only, the nested 90% CI is roughly ±0.14 on TH1.1 and spans 0.01–0.54 on TH1.0. The single full-data figure is also sensitive to Monte Carlo draws: 0.43 with the B = 500 main bootstrap and 0.52 with the B = 1,000 extended bootstrap, for the same specification. The cause is the heavy right tail of 1/slope. The robust share is steadier (0.56).
3. **Adding realistic non-definitional uncertainty lowers the share to about a third.** With U1–U5, S = 0.30–0.43 on TH1.1 and 0.12–0.26 on TH1.0. The largest single source is the trend window. A 2024+ window moves every TH1.1 arrival date earlier and compresses the 24 dates to 2026.8–2029.1 (pooled q = 0.5: 2027.2 vs 2027.7) [ours, `point_arrival_window2024`]. Theil–Sen gives 2027.1–2032.0. Leave-one-system-out moves the pooled 50% date within 2027.4–2027.8. Release-date jitter has negligible effect, and the assumed human-time error raises V_within by 28–75% depending on the menu.
4. **The 50%-vs-80% comparison alone** gives S = 0.58 [0.35, 0.68] with task resampling and 0.42 with all sources on TH1.1. On TH1.0 it gives 0.19 and 0.20, where the pooled 50% interval is twice as wide.
5. Round 1 wrote that definitional variance is "comparable in size to all sampling uncertainty". That holds for TH1.1 under task resampling only. With all five non-definitional layers, definitional variance is roughly half the size of the non-definitional variance on TH1.1 and between a seventh and a third of it on TH1.0.

### 5.4 Family-level definitions are poorly identified on this suite

Family horizons are mostly extrapolations. Families have a median of two tasks, often of similar length. For each frontier system we find the family that determines a quantile or minimum aggregate and check whether its horizon lies within two doublings of that family's task lengths (Table 1). For the 25th- and 10th-percentile aggregates, 60–100% of frontier points lie *below* that range, and up to a further 17% lie above it. Some are far below: Opus 4.6's 10th-percentile family horizon at q = 0.95 is 0.3 min, and the minimum is 0.1 min [ours]. For power means, 65–74% of family horizons lie outside the range and 63–65% of families are saturated or floored. For quantile aggregates, 60–100% of the determining families are saturated or floored for the system in question. Their horizons are set mainly by the ridge prior and the common slope b_m. Only the pooled rows are inside the task support. This directly answers the reviewer's question about extrapolation below the support. It is the main reason we do not interpret the aggregator dimension of Table 1 as a measured property of the systems.

**Reliability and structure of the profiles** [ours, `profiles_1-1.json`]. Splitting each task's runs into halves gives family-offset profiles that correlate at r = 0.80 (Spearman–Brown 0.88). For GPT-5.3-Codex and Opus 4.6 the split-half r is 0.92 and 0.94 (Spearman–Brown 0.96 and 0.97). This measures run noise only. Task-sampling noise, which the bootstrap captures, dominates. Profiles are partly shared: the mean pairwise correlation is 0.42 and the first principal component explains 35%. They also change over time. The mean profile of the five earliest systems correlates 0.25 with that of the five latest. Profiles of systems two releases apart correlate 0.40–0.82, so turnover is gradual. The family `automatic_jailbreak` is among the three lowest-horizon families for all five latest systems.

**Dispersion over time, now under one bootstrap.** Round 1 compared a task-bootstrap interval for the overall dispersion trend with a model-resampling interval for the robustness variants. All three are now computed with the same task-hierarchical bootstrap [ours, `decomp_1-1.json`]. The results are: SD of all family offsets +0.16 logit/yr [0.07, 0.24]; SD over unsaturated, unfloored families +0.04 [−0.06, 0.16]; lower semideviation +0.35 [0.21, 0.47]. TH1.0 gives +0.16 [0.04, 0.27], +0.10 [−0.13, 0.22] and +0.29 [0.11, 0.42]. The conclusion is unchanged but now rests on comparable intervals. The widening is in the lower tail and cannot be separated from ceiling and floor censoring.

### 5.5 Shrinkage dependence

For Opus 4.6 the minimum-family 50% horizon is 1.8 min at λ = 0.1, 3.8 at λ = 0.25, 9.0 at λ = 1, 65 at λ = 4 and 287 at λ = 16 [ours, `profiles_1-1.json`]. That is a factor of 162 across the CV grid. Between λ = 0.25 and 4, where CV loss is within about 4% of its minimum, the factor is 17. (Round 1 quoted λ = 0.01, which lies outside the validated grid; at that value the minimum is 0.17 min.)

**Is a single global ridge the problem?** The reviewer notes that rescaling weights to the per-system run count makes λ = 1 mean different effective shrinkage for systems with 961–1,661 runs. We tested three alternatives [ours, `eb_lambda_1-1.json`]:
- (a) *Run-count-scaled ridge*, λ_m = n_m/n̄. The 24 arrival dates move by at most 0.53 years (Spearman 0.99 with the global fit). The specific concern is real but small.
- (b) *Per-system empirical Bayes.* We treat d_{m,f} ~ N(0, 1/λ_m) as random effects and maximise the Laplace-approximated marginal (weighted pseudo-)likelihood, a GLMM-style variance-component estimate. λ_m^EB has median 0.15 (range 0.05–0.38) and is uncorrelated with run count (r = −0.03). The 24 dates then span 2026.3–2038.0 (11.7 years).
- (c) *The same empirical Bayes with a design-effect correction* that counts each task once, since runs of one task are not independent. Here λ_m has median 4.8 (range 1.0–234), and the dates span 2027.3–2031.2 (3.9 years).
Task-level CV prefers λ ≈ 1. Its loss is 0.388 at λ = 0.1 and 0.361 at λ = 4, which favours (c) over (b), but none of the three is decisive. The pooled rows do not depend on λ. With global λ ∈ {0.25, 1, 4} (each B = 500), the 24-date spread is 7.3, 5.5 and 3.8 years, and the rank correlation with λ = 1 is 0.91 and 0.93 [ours, `robustness.json`]. **The level and spread of family-aggregate arrival dates are therefore determined substantially by a modelling choice that the data constrain only weakly.** That is why the abstract reports a range for them rather than a single number.

### 5.6 Replication on TH1.0

On TH1.0 (30 systems, latest Claude Opus 4.5, λ = 1, B = 500 [ours, `metr_1-0_lam1_*`]), the pooled 50% doubling time is 5.8 months [4.7, 7.1]. Point arrival dates span 2028.0–2032.8 across the 24 definitions and 2028.2–2030.0 without p = +1 and the minimum. The rank correlation with TH1.1 is 0.82. The power-mean ordering holds in 9 of 9 adjacent pairs, and monotonicity in q holds for 6 of 8 aggregators. The 80%-vs-50% delay is 0.91 years. The pooled-slope trend is +0.005 per year [−0.19, 0.14], and the 80%/50% doubling-time ratio is 0.99 [0.80, 1.31]. What replicates: the reliability delay, slope stationarity, the power-mean ordering of levels, rate heterogeneity with catch-up, and the lower-tail dispersion trend. What does not: slower growth of bottleneck aggregates (§5.2), and a large definitional variance share once p = +1 and the minimum are removed (§5.3).

### 5.7 Bounded scores: the CHC profiles (illustration only)

For the published CHC profiles of GPT-4 (27%) and GPT-5 (57%) (Hendrycks et al., 2025), the scores are 0.27/0.57 under p = 1 and 0.15/0.49 under p = 0.5. They are exactly 0 under p ≤ 0, because memory storage is scored 0 for both systems. With an ε-floor the geometric mean is set by the floor: GPT-5 scores 0.36 at ε = 0.005 and 0.49 at ε = 0.1 [ours, `theory_chc.json`]. These static facts are Fourati's (2025). A per-domain linear extrapolation between the two release dates, capped at 1, never reaches 0.9 under any p, because three domains did not change. This is close to a tautology: two integer-scored snapshots with flat domains cannot identify a time to threshold. We keep it only to make the point that a bounded-scale percentage is a level, not a fraction of a journey of known length. We draw no forecasting conclusion from it.

## 6 Prespecified Predictions and Reporting Protocol

The following predictions were fixed by the TH1.1 fits (λ = 1) in this revision. They are stated in this version-controlled revision of the paper but are **not registered with an independent registry**. We intend to file them on OSF before any successor data are released. They can be checked when run-level data for at least five frontier systems released after 2026-02-05 become available, re-estimating over the extended 2023+ window with this code.

- **P1 (stationary slope).** Falsified if *either* the pooled-slope trend leaves [−0.08, +0.10] per year *or* the 80%/50% doubling-time ratio leaves [0.90, 1.35]. (Round 1 required both, which was too lenient.)
- **P2 (reliability delay).** The frontier mean of log₂H₅₀ − log₂H₈₀ stays within its current 90% bootstrap band of 2.0–2.9 doublings (H₅₀/H₈₀ between 4.1× and 7.6×; point estimate 2.47) [ours]. Falsified if the frontier mean over the new systems leaves this band. (Round 1 used 2.4 ± 0.8.)
- **P3 (lower-tail lag, pooled vs 10th-percentile family, λ = 1).** Falsified if the ratio of the pooled to the 10th-percentile 50% horizon, for the new frontier systems, falls below its 2025–2026 frontier median. Because the 10th-percentile horizon is poorly identified (§5.4), we treat P3 as a test of the *estimator*, not of capabilities.
- **P4 (gradual profile turnover; replaces round 1's bottom-3 overlap rule, which `automatic_jailbreak` satisfied trivially).** The correlation between a new frontier system's family-offset profile and that of the system two releases earlier lies in [0.40, 0.85], the observed range of 0.40–0.82. Falsified if 3 or more of the next 5 systems fall outside.

**Reporting protocol.** A claim of the form "AGI-level horizon by year Y" should state (1) the domain set and weights, (2) q, (3) the aggregator, (4) the target and (5) the shrinkage or denoising rule. It should report (6) results under at least q ∈ {0.5, 0.8} and (7) the trend-window sensitivity. For every family-level number it should give (8) the share of determining horizons inside the task support. Item (7) is new in this revision because the window is the largest non-definitional source (§5.3). Item (8) is new because it is the main obstacle to identifying breadth.

## 7 Discussion: what this does and does not say about "distance to AGI"

**What the data support.** On the one public dataset with run-level outcomes and human timings, extrapolated "arrival" depends on the definition. The well-identified part of that dependence is the reliability dial. It follows a closed form (logit(q)/|β| doublings, about 0.9–1 year from 50% to 80% at current doubling rates), and the logistic slope has been stable. A statement such as "a 167-h horizon by 2027.7" therefore becomes "about a year later at 80% reliability", other things equal. This is easy to communicate and is supported on both suite versions.

**What the data do not support.** The breadth dial (aggregation over task families) cannot be measured well on this suite. The family horizons that drive bottleneck definitions sit mostly outside the task lengths of their families or come from censored families, and their levels shift with defensible shrinkage choices by factors of 17–160. Prop. 2's prediction, that a bottleneck definition is held back by the slowest domain, did not hold up: families differ in growth rate, but weak ones catch up, and the slower growth of bottleneck aggregates on TH1.1 is fragile and absent on TH1.0. We therefore make no claim about how much "breadth" delays AGI. The TH1.1 aggregator rows of Table 1 should be read as outputs of a model, not as measurements.

**Why this is still relevant to the AGI question.** Many public "distance to AGI" statements are, in effect, a METR-style pooled 50% extrapolation (one corner of Table 1) or a bounded-scale percentage (§5.7). Our results bound how much of their uncertainty is definitional. With task resampling only, a single definition's error bar covers about half of the total arrival-date variance on TH1.1. With realistic non-definitional uncertainty, definitional variance remains about a third of the total, and the trend window matters as much as any definitional dial. Making breadth-based definitions measurable needs suites with many tasks per domain spanning a range of lengths. METR's families have a median of two tasks. Such suites would also need domains that are cognitive rather than software-specific. Absent those, the honest answer to "how far, under a breadth-requiring definition?" is that the data do not tell us.

**Which definition to report.** (i) Avoid p > 0 aggregation of unbounded horizons, which is dominated by extrapolated ceilings. (ii) Report any family-level aggregate only together with its support and censoring diagnostics and its shrinkage sensitivity. (iii) Report q ∈ {0.5, 0.8} side by side, together with the reliability delay. (iv) Report trend-window sensitivity.

## 8 Limitations

- **One benchmark family.** Both suite versions are METR's software, ML-research and cybersecurity tasks, and they share many tasks. Task families are not cognitive domains. We did not analyse other suites (e.g. the nine domains in METR, 2025), because run-level outcomes with human timings are not public for them. Our conclusions about breadth concern families within this suite.
- **Extrapolation.** The 167-h target is about 10× METR's stated 16-h reliability limit. Arrival dates are a comparison device and would be invalidated by any trend break. A 2024+ window alone shifts them by 0.3–3.5 years.
- **Assumed human-time error.** The U2 magnitudes are assumptions, not estimates. METR does not publish per-task timing uncertainty, and 26 of TH1.1's 31 tasks of 8+ hours have estimated lengths.
- **Model form.** Each system has one logistic slope across families. Family-specific slopes would change high-q tails, and the data are too thin to fit them (median 2 tasks per family). The rows with q ≥ 0.95 extrapolate the logistic tail.
- **Shrinkage is weakly identified.** CV and two empirical-Bayes variants disagree by an order of magnitude in λ. The marginal likelihood is a weighted pseudo-likelihood that ignores dependence among runs of a task beyond the design-effect correction.
- **Share CIs.** The nested bootstrap covers only task resampling (40×100 replicates on TH1.1, 30×80 on TH1.0). CIs for the U1–U5 shares were not computed by nested bootstrap, because that would need refits under human-time noise at both levels. The single-level intervals we computed for them are very conservative, with a lower bound near 0 (`decomp_*_table.csv`).
- **Predictions not externally registered** (see §6). No post-February-2026 run-level data were available to test them.
- **Not an exact reproduction** of METR's published levels; relative comparisons do not depend on matching them.
- **No new model evaluations.** We ran no models. All capability data are third-party.

## 9 Conclusion

The question "how far are we from AGI?" hides definitional choices. On the best public run-level data they move extrapolated arrival dates by years, but only some of that movement is identified. The reliability threshold adds a near-constant, closed-form delay (about one year from 50% to 80%) that replicates across suite versions. The cross-family aggregator produces large differences in Table 1, but these rest mostly on extrapolated, censored or shrinkage-determined family horizons. Its apparent effect on growth rates does not survive specification uncertainty or replication. Definitional variance is about half the total under task resampling alone and about a third once trend-window, frontier, timing and estimator uncertainty are included. The practical recommendation is to report q-sensitivity and window sensitivity routinely. Breadth-based claims should come with support diagnostics, and benchmarks need more tasks per domain before breadth can be measured.

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

METR (2026). Task-completion time horizons of frontier AI models (web page, accessed November 2026). https://metr.org/time-horizons. Run-level data: https://github.com/METR/eval-analysis-public (commit 52cb829, 2026-03-05).

Morris, M. R., et al. (2023). Levels of AGI for operationalizing progress on the path to AGI. arXiv:2311.02462 (ICML 2024 position paper).

Ord, T. (2025). Is there a half-life for the success rates of AI agents? arXiv:2505.05115.

Ruan, Y., Maddison, C. J., & Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024).

Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *Advances in Neural Information Processing Systems 36* (NeurIPS 2023).

---

**Figures** (in `results/`): Fig. 1 `fig_theory.png` (Props. 2 and 4, numerical); Fig. 2 `fig_trajectories.png` (five definitions on TH1.1); Fig. 3 `fig_arrival_forest.png` (all 24 definitions with 90% intervals); Fig. 4 `fig_dispersion_slope.png` (family-offset dispersion and logistic slope over time); Fig. 5 `fig_chc.png` (CHC scores under power means); Fig. 6 `fig_decomposition.png` (definitional variance share by menu and uncertainty set, TH1.1 and TH1.0).
