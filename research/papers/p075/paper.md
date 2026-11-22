# Who Can Close the Window? Substitution, Superadditivity, and the Coordination Gap in Open-Weight AI Release

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES 2027)

---

## Abstract

Debates about open-weight foundation models are usually framed at the level of a single release: does publishing *this* model's weights add risk or benefit at the margin, relative to closed models and existing technology? We argue that this framing systematically mis-sizes the stakes, because open-weight releases from different developers are close substitutes on the capability frontier. We formalise the open-weight frontier as the running maximum over developers' releases and show that any access metric of the form "integral of a monotone function of the open frontier" is a monotone submodular set function of the set of releasing developers. Three consequences follow. (i) The effect of a coalition withholding its releases is at least the sum of its members' unilateral effects, and can be much larger (a *coordination gap*). (ii) In a stylised model with *k* near-frontier developers, the expected ratio of summed unilateral effects to the joint effect is exactly 1/*k*. (iii) A release rule adopted by a subset of developers can never set the open frontier back by more than removing those developers entirely. We then measure these quantities on the Epoch Capabilities Index (ECI; 236 models, 113 open-weight, data to 2026-11-17), propagating the published 90% credible intervals by Monte Carlo. From January 2025 to November 2026 the open-weight frontier lagged the overall frontier by 4.4 months (90% interval 3.7–5.2) and 8.1 ECI points, with no detectable trend since 2024 (+0.24 months/year, interval −1.26 to +1.60). Chinese developers held the open frontier on 100% of days in that window (interval 97–100%). Removing any single developer sets the open frontier back by at most 0.57 months (DeepSeek); removing all Chinese developers jointly sets it back by 4.1 months (3.0–5.8), almost four times the 1.1-month sum of unilateral effects (ratio 0.26, interval 0.18–0.37; superadditive in every draw). Removing all US developers' open releases has no measurable effect in this window, and a "release only after the closed frontier has led by Δ months" rule adopted only by US (or all non-Chinese) developers leaves the open lag unchanged for every Δ up to 12 months, whereas universal adoption raises the lag roughly to Δ. The result is a narrow but policy-relevant claim: for frontier-level open access, both the harms and the benefits attributed to an individual release are small relative to what is decided at the level of coalitions of developers, and in 2025–2026 that coalition was located in a single jurisdiction. We state falsifiable predictions and release code that recomputes every number from future ECI vintages.

## 中文摘要

关于开放权重基础模型的争论，通常聚焦于单次发布：公开*某一个*模型的权重，相对于闭源模型和已有技术，究竟在边际上增加了多少风险或收益？本文认为，这一框架会系统性地低估或错置真正的利害，因为不同开发者的开放权重发布在能力前沿上是高度可替代的。我们将开放权重前沿形式化为各开发者发布能力的累积最大值，并证明：任何形如"对开放前沿的单调函数求时间积分"的可及性指标，都是开发者集合上的单调次模（submodular）集合函数。由此得到三点推论：（1）一个开发者联盟共同不发布的效果，不小于其成员单方面不发布效果之和，且可能大得多（"协调缺口"）；（2）在含 *k* 个近前沿开发者的程式化模型中，单方面效果之和与联合效果之比的期望恰为 1/*k*；（3）仅由部分开发者采用的发布规则，对开放前沿的推迟不可能超过将这些开发者完全移除的效果。随后我们在 Epoch 能力指数（ECI；236 个模型，其中 113 个为开放权重，数据截至 2026-11-17）上测量上述量，并通过蒙特卡洛传播其公布的 90% 可信区间。2025 年 1 月至 2026 年 11 月，开放权重前沿落后整体前沿 4.4 个月（90% 区间 3.7–5.2），即 8.1 个 ECI 点，且自 2024 年以来没有可检测的趋势。在此期间，开放前沿 100% 的天数由中国开发者保持。移除任何单一开发者最多使开放前沿推迟 0.57 个月（DeepSeek）；而同时移除所有中国开发者则推迟 4.1 个月（3.0–5.8），约为单方面效果之和（1.1 个月）的四倍（比值 0.26，在每一次抽样中均呈超可加性）。移除所有美国开发者的开放发布在该时段没有可测效果；"闭源前沿领先 Δ 个月后才允许开放发布"的规则若仅由美国（或所有非中国）开发者采用，对任意 Δ≤12 个月都不改变开放滞后，而普遍采用则使滞后大致升至 Δ。本文的结论是有限但与政策相关的：就前沿级开放可及性而言，归因于单次发布的风险与收益都远小于在开发者联盟层面所决定的部分，而在 2025–2026 年，这一联盟集中于单一司法辖区。我们给出可证伪的预测，并公开可在未来 ECI 版本上重算全部数字的代码。

---

## 1 Introduction

Whether highly capable foundation models should be released with open weights is one of the most contested questions in AI governance. Proponents point to competition, scrutiny, local deployment, and the distribution of power [Bommasani et al. 2024; Kapoor et al. 2024]. Critics point to the irreversibility of release and the ease with which safety fine-tuning is removed once weights are public [Seger et al. 2023; Gade et al. 2023; Qi et al. 2024]. The most influential analytical move in this debate has been the *marginal risk* framing: the relevant question is not whether an open model can be misused, but how much its open availability adds relative to closed models and pre-existing technologies [Kapoor et al. 2024; NTIA 2024].

The marginal-risk framing is usually applied to one release, or to "open models" as a category. This paper points out that between those two levels sits the level at which most of the action happens: the *set of developers* who release at or near the open-weight frontier. If several developers release models of similar capability within a few months of each other, then the counterfactual "this model is not released" changes little — someone else's model fills the gap — while the counterfactual "none of these developers release" changes a great deal. The marginal impact of individual releases is then a poor guide to the stakes of the open-versus-closed choice, for harms and benefits alike. This is a familiar structure in economics (perfect substitutes, public goods, weakest-link and best-shot games), but to our knowledge it has not been formalised or measured for open-weight AI.

We ask a narrow, answerable question: **On the general-capability frontier, how much does a single developer, a national group of developers, or a release rule adopted by a subset of developers change when frontier-level capability becomes available as open weights?** The answer bears directly on the "trade-off" question in the title. If the open-weight frontier is supplied redundantly, a single developer's decision to open or close its weights has small consequences either way, and debates about "the risks and benefits of open weights" are, in practice, debates about coordination among developers and jurisdictions.

**Contributions.**

1. **A formal model of substitution among open-weight releases** (Section 3). We define the open frontier as a running maximum over developers and show that a broad class of access metrics — anything of the form ∫ G(open frontier) dt with G nondecreasing, which includes "months behind", "capability-months of access", and "months during which a hazardous threshold is openly available" — is monotone submodular in the set of releasing developers (Proposition 1). Consequences: coalition effects are superadditive in unilateral effects and Shapley values dominate leave-one-out effects (Corollary 1); a stylised model gives an exact 1/*k* law for the ratio of summed unilateral to joint effects (Proposition 3); and a release rule adopted by a subset of developers is bounded by that subset's removal effect (Proposition 4).
2. **A measurement methodology** (Section 4) for the open–closed lag and for leave-one-developer-out and leave-coalition-out counterfactual frontiers on an aggregate capability index, with Monte Carlo propagation of per-model score uncertainty and explicit handling of right-censoring.
3. **Empirical estimates on the Epoch Capabilities Index** (Section 5), data through 2026-11-17: the open lag (4.4 months since 2025), the identity of open-frontier holders, the frontier setback from removing each developer and each national coalition, the coordination gap (sum of unilateral effects is 26% of the joint effect for Chinese developers), Shapley attributions, replacement times for every open-frontier record since 2024, and the effect of staleness-based release rules under different adopters.
4. **Falsifiable predictions and a re-run protocol** (Section 6.3) that can be checked against future ECI vintages using the released code.

We do not claim to settle whether open weights are good or bad on balance. Our contribution is to show, formally and with data, *at which level of aggregation* the trade-off is actually decided for frontier-level general capability, and to quantify how large the gap between levels currently is.

## 2 Related Work

**Release norms and the open/closed spectrum.** Solaiman [2023] describes a gradient of release options from fully closed to fully open. Shevlane [2022] proposes *structured access* through APIs as a way to share capabilities without releasing weights. Seger et al. [2023] evaluate risks and benefits of open-sourcing highly capable models and argue that for some future models the risks may outweigh the benefits, recommending alternatives such as staged release. Bommasani et al. [2024] and Kapoor et al. [2024] argue for evaluating open models by their *marginal* risk relative to closed models and existing technology, identifying five distinguishing properties of open models (broader access, customisability, local inference, irrevocability, weaker monitoring). The NTIA [2024] report adopts the marginal-risk framing and recommends monitoring rather than immediate restrictions. Our work accepts the marginal-risk framing but points out that "marginal relative to what" has a set-valued answer: marginal to a single release, to a developer, or to a coalition, and these differ by large factors when releases are substitutes.

**Safeguard removal.** Work showing that safety fine-tuning is removed cheaply once weights are available [Gade et al. 2023; Qi et al. 2024] motivates the assumption, standard in this literature and used here, that frontier-level open weights imply frontier-level capability without provider-side safeguards. Our metric is therefore *time-to-unsafeguarded-access* at a given general capability level.

**Measuring the open–closed gap.** Epoch AI's open-models report found that open models lagged closed ones by about a year in benchmark performance and about 15 months in training compute [Cottier et al. 2024]. Later Epoch analyses on the Epoch Capabilities Index (ECI) found an average lag of about three months over January 2023–October 2025 [Epoch AI 2025], and about four months (8 ECI points, 90% interval 7–11) since January 2026 [Edwards & Emberson 2026]. The UK AI Security Institute reports that leading open-weight models trail the closed frontier on cyber tasks by 4–7 months, narrower than the 6–10 months it measured through most of 2025 [UK AISI 2026]. The ECI itself is an item-response-theory aggregation of many benchmarks [Ho et al. 2025], since November 2026 fitted with a Bayesian model that reports 90% credible intervals [Epoch AI 2026b]. These works measure *how far behind* open models are. We measure *who* keeps them where they are, and how much the lag would change under counterfactual removal of developers or under release rules.

**Formal models of openness regulation.** Qiu et al. [2025] model a generalist model producer and a downstream specialist under openness regulation, characterising equilibria as functions of regulatory penalties and open-source thresholds. Their model has one upstream producer; ours has many substitutable producers and no strategic behaviour, and is designed to be measured rather than solved.

**Submodularity and attribution.** Our main formal tool is the elementary fact that the maximum of a set is submodular, so integrals of monotone transforms of a running maximum are submodular [Nemhauser et al. 1978]; and we use the Shapley value [Shapley 1953] to apportion joint effects among developers. The application to substitutable open-weight releases, and the resulting bounds on unilateral versus coalitional restraint, are, as far as we can find, new.

## 3 Model

### 3.1 Objects

Let *N* be a finite set of developers. Each developer *k* makes a set of open-weight releases; release *i* has date *t_i* and capability score *x_i* on a common scale (here, ECI). For *S* ⊆ *N*, the **open frontier of S** is

  O_S(t) = max{ x_i : i is an open release by a developer in S, t_i ≤ t },

with O_∅(t) = −∞. Write O = O_N. Let C(t) and A(t) be the running maxima over closed-weight releases and over all releases. For a nondecreasing step function Y, its **first-reach time** at level *c* is T_Y(c) = inf{ t : Y(t) ≥ c }.

Two derived quantities organise the analysis.

- **Horizontal lag** of the open frontier: L(t) = T_O(O(t)) − T_A(O(t)), i.e., how long after the overall frontier first reached the open frontier's current level the open frontier reached it. This is a "catch-up time" that does not grow while the open frontier is flat, and is closely related to the lag estimated by Epoch [Edwards & Emberson 2026].
- **Frontier setback** from removing a set *R* of developers, over a window *W*:

  σ_R = (1/|W|) ∫_W [ T_O(O(t)) − T_O(O_{N∖R}(t)) ] dt.

  At each date, T_O(O_{N∖R}(t)) is the date on which the actual open frontier first reached the level that the counterfactual frontier has at *t*; the difference is how many months behind the actual open frontier the counterfactual one is. σ_R = 0 if and only if *R* never matters on *W*.

### 3.2 Access values

Let G be a nondecreasing function from capability levels to the reals (with a finite value at −∞). Define the **access value** of a developer set

  v_G(S) = ∫_W G(O_S(t)) dt.

Three choices of G matter for policy:

- G(x) = T_O(x): then v_G(N) − v_G(N∖R) = |W| σ_R, the frontier setback.
- G(x) = (x − b)_+ for a baseline b: v_G measures **capability-months** of open access above b.
- G(x) = 1[x ≥ c*]: v_G is the number of **months during which capability level c\* is openly available**. If c* is read as a hazard threshold, this is the quantity a marginal-risk analysis of threshold-level misuse cares about; if c* is read as a usefulness threshold, it is the quantity a benefits analysis cares about.

The same function measures harm and benefit; the sign is a matter of interpretation. Our results concern its *set structure*, which is the same under both readings.

### 3.3 Results

**Proposition 1 (submodularity).** For every nondecreasing G, v_G is monotone and submodular on 2^N.

*Proof.* Fix *t*. Let o_k(t) be developer *k*'s own running maximum, so O_S(t) = max_{k∈S} o_k(t). Because G is nondecreasing, G(O_S(t)) = max_{k∈S} G(o_k(t)) (with the value G(−∞) for S = ∅). For numbers a_k, the function f(S) = max_{k∈S} a_k is monotone, and the marginal gain of adding *j* to S is (a_j − f(S))_+, which is nonincreasing in S because f(S) is nondecreasing in S; hence f is submodular. Integrals over *t* of monotone submodular functions are monotone submodular. ∎

**Corollary 1 (coordination gap; Shapley dominance).** Let Δ_R = v_G(N) − v_G(N∖R) be the effect of removing a set *R*, and Δ_k = Δ_{{k}}. Then for every R ⊆ N:

 (a) Δ_R ≥ Σ_{k∈R} Δ_k;
 (b) the Shapley value φ_k of v_G satisfies φ_k ≥ Δ_k for every *k*, and Σ_k φ_k = v_G(N) − v_G(∅).

*Proof.* (a) Order R = {k_1, …, k_m} and telescope: Δ_R = Σ_j [v(N∖{k_1..k_{j−1}}) − v(N∖{k_1..k_j})]. The j-th term is the marginal value of adding k_j to the set N∖{k_1..k_j}, which is a subset of N∖{k_j}; by submodularity it is at least the marginal value of adding k_j to N∖{k_j}, which is Δ_{k_j}. (b) φ_k is a weighted average of the marginal values of adding *k* to subsets of N∖{k}; each is at least Δ_k by submodularity. Efficiency is a standard Shapley property [Shapley 1953]. ∎

We call Δ_R − Σ_{k∈R} Δ_k the **coordination gap** of R and Σ_{k∈R} Δ_k / Δ_R ∈ [0, 1] its **unilateral share**. A small unilateral share means that analyses of individual releases or individual developers see little of what is at stake.

**Proposition 2 (what a unilateral effect measures).** Δ_k is nonzero only at dates where *k* is the *unique* holder of the open frontier, and there it equals G(o_k(t)) − G(max_{j≠k} o_j(t)): the leader's margin over the runner-up. The joint effect of a coalition R is G(O(t)) − G(max_{j∉R} o_j(t)): the coalition's margin over the best outsider.

*Proof.* Immediate from O_{N∖{k}}(t) = max_{j≠k} o_j(t). ∎

Proposition 2 explains why unilateral effects are small whenever there is a close runner-up inside the coalition, regardless of how far ahead the coalition is of everyone else.

**Proposition 3 (a 1/k law).** Suppose the closed frontier grows strictly and continuously, C(t) strictly increasing; coalition developers j = 1..k each have open frontier o_j(t) = C(t − d_j) with lags d_j i.i.d. Uniform(0, D); and the best outsider has open frontier C(t − D). Take G = T_O (frontier setback) and a window late enough that all frontiers are defined. Then E[Σ_j Δ_j] / E[Δ_R] = 1/k.

*Proof.* Let d_(1) < d_(2) be the two smallest lags. The actual open frontier is O(t) = C(t − d_(1)), so T_O(c) = T_C(c) + d_(1). Removing all of R leaves C(t − D), so T_O(O_{N∖R}(t)) = t − D + d_(1) and the setback is D − d_(1) at every *t*. Removing only the leader leaves C(t − d_(2)), with setback d_(2) − d_(1); removing any other member has zero effect. For k i.i.d. Uniform(0, D) variables, E[d_(1)] = D/(k+1) and E[d_(2)] = 2D/(k+1), so E[Σ Δ_j] = D/(k+1) and E[Δ_R] = kD/(k+1). ∎

The proposition is stylised: releases are continuous and lags are exchangeable. Section 5.6 checks by simulation that the 1/k ratio is approximately preserved when releases arrive as Poisson processes with rates between one per four months and four per month.

**Proposition 4 (partial-adoption bound for release rules).** Consider any rule that delays (or cancels) some releases of developers in a set J ⊆ N and leaves all other releases unchanged, producing an open frontier O^rule. Then O_{N∖J}(t) ≤ O^rule(t) ≤ O(t) for all t, and the frontier setback of the rule is at most σ_J. If J = N and the rule is "an open release of capability *x* may occur only Δ months after the closed frontier first reached *x*", then the horizontal lag of O^rule behind the closed frontier is at least Δ at every date.

*Proof.* The first claim holds because the rule leaves every release by N∖J in place and only removes or postpones releases, so the running maximum lies between those of N∖J and N; monotonicity of G = T_O gives the bound. For the second, every open release of level x under the rule occurs at or after T_C(x) + Δ, so T_{O^rule}(O^rule(t)) ≥ T_C(O^rule(t)) + Δ. ∎

Proposition 4 has a direct policy reading: a jurisdiction that imposes any release restriction on its own developers cannot delay open availability by more than the setback from those developers disappearing altogether, which is a quantity we can measure.

## 4 Data and Measurement

**Data.** We use Epoch AI's per-model ECI scores [Epoch AI 2026a], vintage of 2026-11-18, which contains 257 models released between 2023-02-24 and 2026-11-17, with point estimates, 90% credible intervals from the Bayesian ECI fit [Epoch AI 2026b], release dates, developer organisation, country, and accessibility. We keep the 236 models classified as "Open weights" (113; unrestricted, restricted-use and non-commercial licences) or "Closed weights" (123) and drop 21 with unknown accessibility. Co-authored or sub-unit labels are merged into one decision-making developer (e.g., "DeepSeek, Peking University" into DeepSeek), leaving 22 open-weight developers. Countries are Epoch's organisation-country field. Epoch's website was not reachable from our environment; the file was obtained unchanged from a public mirror that records its download from epoch.ai on 2026-11-18 (hashes and provenance in the repository). A second vintage (2026-11-11) is used as a robustness check. ECI is anchored so that Claude 3.5 Sonnet scores 130 and GPT-5 scores 150 [Epoch AI 2026b].

**Uncertainty.** For each Monte Carlo draw (1,000 draws), each model's ECI is perturbed by an independent normal error whose standard deviation is the 90% interval width divided by 2 × 1.645. All frontiers, holders, and counterfactuals are recomputed in every draw, so frontier *identity* is uncertain, not only frontier *level*. We report the median and the 5th–95th percentiles across draws as "90% intervals". Independence across models ignores posterior correlation in the ECI fit; Section 7 discusses this.

**Computation.** Frontiers are evaluated on a daily grid from 2023-01-01 to 2026-11-17. The main window is 2025-01-01 to 2026-11-17 (the "2025+" window); we also report 2024+, 2026+, and calendar years. Threshold-availability delays are right-censored when the counterfactual frontier never reaches the threshold before the data end; we report censored fractions and treat censored values as lower bounds. Shapley values are estimated with 2,000 random permutations of the 22 developers, on point estimates, using the capability-months value with baseline b = 131.71 (the open frontier on 2025-01-01) and the months-above-155 value. Release-rule counterfactuals use the first 200 Monte Carlo draws. All code is in `code/` and runs in about five minutes on one CPU core.

## 5 Results

All numbers in this section are produced by `code/analysis.py` and `code/toy_model.py` and stored in `results/`.

### 5.1 The open–closed lag

Figure 1 shows the closed, open, and counterfactual frontiers. The horizontal lag of the open frontier behind the overall frontier averages 4.44 months (90% interval 3.69–5.16) over 2025+, with a vertical gap of 8.05 ECI points (7.21–8.88). Over 2024+ the lag is 4.52 months (3.53–5.70); over 2026+ it is 5.54 months (4.47–6.79) with a vertical gap of 10.05 points (8.84–11.30). The lag behind the *closed* frontier is identical to the lag behind the overall frontier after 2025 because no open model was ever the overall frontier in that period.

These magnitudes are close to Epoch's own estimates of about four months and 8 points since January 2026 [Edwards & Emberson 2026]; the differences reflect a different estimator (theirs requires the open model to beat the closed one in at least 5% of posterior samples and notes that a stricter point-estimate criterion would give six months), a later data vintage that includes several closed releases from September–November 2026, and our use of a time-averaged catch-up lag. Fitting a linear trend to monthly mean lags from January 2024 to November 2026 gives a slope of +0.24 months per year (90% interval −1.26 to +1.60): no detectable widening or narrowing (Figure 2).

### 5.2 Who holds the open frontier

In the 2025+ window, Chinese developers held the open frontier on 100% of days (90% interval 97–100%). Over 2024+ the shares are China 70%, United States 20%, France 10% (point estimates), and in calendar 2024 they were United States 56%, France 30%, China 14%. Of the 21 open-frontier records set since January 2024, 16 were set by Chinese developers (DeepSeek, Moonshot, Alibaba, Zhipu), 4 by US developers (Meta and Microsoft, all in 2024), and 1 by Mistral. The last US-held open record in the data is Phi-4 (December 2024).

### 5.3 Releases are replaced quickly

For every open-frontier record since 2024 we compute the time until a model *from a different developer* reached the same ECI (`results/open_records.csv`). The median replacement time among the 20 uncensored records is 1.49 months; 17 of 21 records (81%) were matched by another developer within three months. The longest replacement times were for DeepSeek-R1 (5.7 months, January 2025), Llama 3.1-405B (4.7 months, July 2024) and Kimi K3 (3.1 months, July 2026). The most recent record (Kimi K3.1 Pro, 2026-10-23) is censored at data end. In the marginal-risk language, the counterfactual for most individual frontier-setting releases is the same capability, openly available, one to two months later.

### 5.4 Unilateral and coalition removal: the coordination gap

Table 1 reports the frontier setback σ_R for removing each developer whose removal matters and each national coalition (2025+ window, 90% intervals over score noise).

**Table 1. Frontier setback from removing developers' open releases, 2025-01-01 to 2026-11-17.**

| Removed | Open releases removed | Setback σ_R (months) | Vertical gap (ECI) | Counterfactual lag behind closed frontier (months) |
|---|---|---|---|---|
| none (actual) | 0 | 0 | 0 | 4.44 (3.64–5.16) |
| DeepSeek | 16 | 0.57 (0.22–1.04) | 1.13 | 4.78 (3.97–5.66) |
| Moonshot | 7 | 0.23 (0.10–0.48) | 0.67 | 4.95 (4.09–5.69) |
| Alibaba | 23 | 0.18 (0.00–0.42) | 0.19 | 4.48 (3.63–5.28) |
| Zhipu (Z.ai) | 6 | 0.11 (0.01–0.20) | 0.09 | 4.45 (3.63–5.25) |
| any other single developer (18) | — | 0.00 | 0.00 | unchanged |
| all US developers | 44 | 0.00 (0.00–0.01) | 0.00 | 4.45 (3.65–5.18) |
| all non-Chinese developers | 57 | 0.00 (0.00–0.01) | 0.00 | 4.45 (3.65–5.18) |
| **all Chinese developers** | 56 | **4.13 (3.01–5.76)** | **6.86 (5.78–7.91)** | **7.72 (6.39–8.80)** |

*Source: `results/removal_counterfactuals.csv`. "Counterfactual lag" is the horizontal lag of the counterfactual open frontier behind the closed frontier (median and 90% interval over 1,000 draws). Removing US developers changes the lag only through the few draws in which a US model holds the open frontier at the start of the window.*

The sum of the four unilateral setbacks of Chinese developers is 1.09 months (0.76–1.58), against a joint setback of 4.13 months: a unilateral share of 0.26 (0.18–0.37). The joint effect exceeded the sum of unilateral effects in 100% of draws, as Corollary 1 requires (Figure 3). By calendar window (point estimates), the joint setback of Chinese developers versus the sum of unilateral setbacks was 2.79 vs 1.21 months in 2025 and 4.97 vs 0.62 months in 2026 (unilateral share 0.12): the coordination gap grew as more Chinese developers released near-frontier models. In 2024, when US and French developers held most of the open frontier, the corresponding numbers for non-Chinese developers were 2.31 vs 1.59 months.

In threshold terms, removing all Chinese developers would have delayed open availability of ECI 145 by 6.9 months, ECI 150 by 3.3 months, and ECI 155 by 3.0 months (21% of draws censored, i.e., not reached by the data end); ECI 158 would not have been reached by non-Chinese open models by the data end in 95% of draws (delay at least 4.1 months). No single developer's removal delays any of these thresholds by more than one month (Moonshot: 0.9 months at 155 and 158).

### 5.5 Shapley attribution

With capability-months above the January-2025 open frontier as value, the total open-access value in 2025+ is 344.7 ECI-months; the 22 leave-one-out marginals sum to only 42.2 ECI-months (12%), so 88% of the value is invisible to any single-developer counterfactual. Shapley values apportion the total as China 273.7 (79%), United States 61.1 (18%), France 9.9 (3%). By developer: DeepSeek 88.0, Moonshot 72.7, Alibaba 62.6, Zhipu 40.6, OpenAI 23.3, Google DeepMind 12.0, Mistral 9.9, MiniMax 9.9, Nvidia 9.1, Thinking Machines 8.9 (Figure 5). US developers have positive Shapley value (their models would matter if Chinese developers did not release) but zero leave-one-out marginal value. With the months-above-ECI-155 value, the total is 4.1 months, leave-one-out marginals sum to 0.9 months, and the Chinese share of Shapley value is 3.8 months.

The Shapley value is not offered as a normative allocation of responsibility; it is a reminder (Corollary 1b) that leave-one-out reasoning attributes almost nothing to anyone when the frontier is supplied redundantly.

### 5.6 The 1/k law in simulation and data

`code/toy_model.py` simulates Proposition 3 with Poisson release arrivals (rates 0.25, 0.5, 1 and 4 per month; D = 12 months; 150 replications per cell). The ratio of mean summed unilateral setback to mean joint setback is 0.50 / 0.46 / 0.44 / 0.52 for k = 2, 0.26 / 0.23 / 0.21 / 0.22 for k = 4, and 0.13 / 0.12 / 0.10 / 0.09 for k = 8 across the four release rates (`results/toy_model.json`), close to 1/k. Empirically, four Chinese developers held the open frontier at some point in 2025+ and the measured unilateral share is 0.26. We report this agreement as a consistency check, not a test: the number of developers was not fixed in advance, and Chinese developers' lags are not exchangeable.

### 5.7 Release rules under partial adoption

We simulate a staleness rule: an open release of capability x is postponed until Δ months after the closed frontier first reached x (Figure 4; 200 draws). If all developers adopt it, the open lag behind the closed frontier in 2025+ rises from 4.33 months (Δ = 0) to 4.80 (Δ = 3), 6.59 (Δ = 6; interval 6.29–7.01), 9.20 (Δ = 9) and 12.17 (Δ = 12), approaching Δ as Proposition 4 requires. If only US developers adopt it, or only all non-Chinese developers, the lag is 4.33–4.34 months for every Δ up to 12, i.e., unchanged; the frontier setback relative to the actual open frontier is 0.00 (interval upper bound 0.03 months). If only Chinese developers adopt it, the lag rises to 6.30 months at Δ = 6 and 8.23 months (6.27–9.39) at Δ = 12; at large Δ it is capped by non-Chinese open models, consistent with the bound σ_rule ≤ σ_J in Proposition 4.

### 5.8 Robustness

- *Data vintage.* On the 2026-11-11 vintage the 2025+ lag point estimate is 4.50 months (main vintage 4.55) and the Chinese-coalition setback is 3.82 months (main vintage 3.81; both point estimates).
- *Licence definition.* Counting only "Open weights (unrestricted)" as open raises the lag to 5.18 months and the Chinese-coalition setback to 4.43 months (point estimates).
- *Interval interpretation.* Treating the published intervals as 95% rather than 90% (smaller noise) gives a 2025+ lag of 4.47 months (4.03–5.15).
- *Window.* Over 2024+ the Chinese-coalition setback is 2.56 months against a unilateral sum of 0.68 (point estimates); the qualitative pattern holds in every window from 2025 on.

## 6 Discussion

### 6.1 What this says about the open-versus-closed trade-off

The trade-off over open weights is usually posed as risk versus benefit for a given model. Our results suggest that, for general capability at the frontier, the decisive variable is instead the *number and location of developers willing to release near the frontier*. Three implications follow.

*First, release-level marginal analysis undercounts in both directions.* A developer considering whether to open a frontier-level model can truthfully say that its release changes open availability by one or two months (Section 5.3). Critics who argue that a particular release is harmful, and advocates who argue that a particular release is a public good, are both describing a quantity that is small whenever close substitutes exist. Corollary 1 guarantees that summing such release-level or developer-level assessments understates the joint stakes. A regime that approves each release because its marginal risk is small can approve a set of releases whose joint effect is large; in our data the gap is a factor of about four for Chinese developers in 2025–2026 and a factor of eight in 2026 alone. This is the familiar logic of collective action applied to the marginal-risk framework of Kapoor et al. [2024] and NTIA [2024], and it argues for evaluating open-weight policy at the level of the coalition that can actually change the open frontier.

*Second, unilateral national restrictions on frontier open weights currently have little effect on frontier access.* In 2025–2026 the open frontier was held by Chinese developers throughout, so a restriction on US developers' open releases — however strict — would not have moved it (Table 1; Section 5.7; Proposition 4 makes this a bound, not just an observation). This is not an argument that such restrictions are useless: they may affect non-frontier models, fine-tuning ecosystems, domain-specific capabilities, or the norms and leverage a country brings to international negotiations. But they cannot be justified by their effect on *frontier-level* open access in the present configuration. Conversely, the benefits US policy-makers attribute to promoting US open models (for example, offering alternatives to Chinese open models) are, on this metric, about the *identity* of the open frontier holder, not its *level*: removing all US open releases changes the frontier level not at all.

*Third, a meaningful change to open-frontier timing requires the participation of the developers who hold it.* Universal adoption of a Δ-month staleness rule raises the open lag roughly to Δ; adoption by any coalition excluding Chinese developers leaves it unchanged. The structure is that of a "best-shot" public good: the frontier is set by whoever releases the best model, so effective restraint requires that *all* close substitutes participate. This pushes the open-weights question into the domain of international coordination, and it gives a concrete metric — the coalition setback σ_R — for what a proposed agreement would actually buy.

### 6.2 What the measurements do not say

The four-to-five-month lag is a lag in an aggregate index of general capability. Domain-specific lags differ: for example, the UK AISI reports 4–7 months for cyber tasks [UK AISI 2026]. Risks concentrated in narrow domains may depend on whether particular capabilities are suppressed or included in specific releases, not on aggregate scores. Our counterfactuals also hold every other release fixed. If open developers learn from each other's releases (through published methods, distillation, or released weights used as a base), then removing a coalition would also slow outsiders, and our joint setbacks are underestimates; if outsiders would accelerate to fill a vacated niche, our joint setbacks are overestimates. Neither consideration changes Proposition 1, which holds for whatever release schedule is realised, but both change the numbers.

### 6.3 Falsifiable predictions and a re-run protocol

We state predictions that can be checked by re-running `code/analysis.py` on future ECI vintages, with the 2025+ window extended to the new data end and the same Monte Carlo settings.

- **P1 (small unilateral effects).** As long as at least three developers have an open release within three months (horizontal lag) of the open frontier, no single developer's frontier setback over any rolling 12-month window will exceed 1.0 month.
- **P2 (1/k scaling).** Across rolling 12-month windows, the unilateral share of the coalition holding the open frontier will lie within a factor of two of 1/k_eff, where k_eff is the number of coalition developers that hold the open frontier at least once in the window.
- **P3 (stable lag).** Through 2027, the annual mean horizontal open lag on ECI will stay between 3 and 7 months; the fitted trend over 2024–2027 will have a 90% interval that includes zero.
- **P4 (jurisdictional irrelevance).** Unless a US developer sets an open-frontier record, the setback from removing all US developers' open releases over any 12-month window will be below 0.5 months.

P1 and P4 are the most falsifiable: a single large, unreplicated open release (for instance a US developer opening a model well above the existing open frontier) would violate them, and that is exactly the event in which unilateral decisions matter.

## 7 Limitations

- **Static counterfactuals.** All counterfactuals hold other releases fixed. Strategic responses, knowledge spillovers, and distillation are not modelled. Section 6.2 explains the direction of each bias; we cannot estimate their size from these data.
- **One aggregate metric.** ECI is a single IRT-based index; it does not capture domain-specific hazards (biology, cyber), agentic reliability, or properties such as tamper resistance. A model can be far from the ECI frontier and still be the most hazardous open model in a narrow domain.
- **Coverage and selection.** ECI includes only models Epoch has evaluated or collected scores for. Missing open models would make the open frontier, and hence substitution, *higher* than measured; missing closed models would make the lag larger. We drop 21 models with unknown accessibility, some of which (hosted Qwen variants) are near the frontier.
- **Uncertainty model.** We perturb scores independently using normal approximations to published 90% credible intervals. The posterior of the ECI fit is correlated across models (shared benchmark difficulties), so our intervals for *differences* between models may be too wide or too narrow. Release dates are treated as exact.
- **Weights are not the only kind of access.** We equate "open frontier" with the best open-weight model by ECI. Actual uplift also depends on inference cost, hardware, licences (restricted-use licences are counted as open in the main analysis; Section 5.8 shows the effect of excluding them), and the ease of removing safeguards.
- **Developer and country attribution.** We treat developers as decision units and use Epoch's organisation-country field. Joint ventures, subsidiaries and multinational teams are simplified.
- **Small number of frontier holders.** The empirical coordination gap rests on a few developers and about 20 record-setting releases. The agreement with the 1/k law is post hoc.
- **Data provenance.** epoch.ai was not reachable from our environment; the ECI file came from a third-party mirror that documents its download from Epoch. We publish hashes so that readers can verify it against Epoch's own file.
- **Normative scope.** We measure how much access changes, not whether the change is good. Converting setbacks into welfare requires hazard and benefit functions that we do not estimate.

## 8 Conclusion

The open-weight frontier is a running maximum over developers, and running maxima are submodular. This simple fact implies that the effect of any one developer's release decision is bounded by its margin over the runner-up, while the effect of a coalition's decision is its margin over the best outsider. On the Epoch Capabilities Index through November 2026, the runner-up has usually been close and the best outsider far: individual developers' open releases are replaced within one to two months, the sum of individual effects is about a quarter of the joint effect of the Chinese developers who held the open frontier throughout 2025–2026, and release restrictions adopted by any group excluding those developers would not have changed the timing of frontier-level open access. The question "should open weights be released?" is, for frontier general capability, decided at the level of coalitions and jurisdictions rather than individual releases. Evaluations of marginal risk and marginal benefit should be carried out at that level, and the coalition setback offers a measurable quantity to do so.

---

## References

- Bommasani, R., Kapoor, S., Klyman, K., Longpre, S., Ramaswami, A., Zhang, D., Schaake, M., Ho, D. E., Narayanan, A., & Liang, P. (2024). Considerations for governing open foundation models. *Science*, 386(6718), 151–153. https://doi.org/10.1126/science.adp1848
- Cottier, B., et al. (2024). How far behind are open models? (Open models report). Epoch AI. https://epoch.ai/publications/open-models-report
- Edwards, J., & Emberson, L. (2026, May 29). Open models lag state-of-the-art closed models by 4 months. Epoch AI Data Insight. https://epoch.ai/data-insights/open-closed-eci-gap
- Epoch AI (2025, October 30). Open-weight models lag state-of-the-art by around 3 months on average. Epoch AI Data Insight (earlier version of the analysis at https://epoch.ai/data-insights/open-closed-eci-gap).
- Epoch AI (2026a). Epoch Capabilities Index (ECI) scores, `eci_scores.csv`, accessed via mirror of the 2026-11-18 download. CC BY 4.0. https://epoch.ai/benchmarks/eci
- Epoch AI (2026b). Updating the ECI methodology to use a Bayesian model. https://epoch.ai/publications/eci-bayesian-methodology
- Gade, P., Lermen, S., Rogers-Smith, C., & Ladish, J. (2023). BadLlama: cheaply removing safety fine-tuning from Llama 2-Chat 13B. arXiv:2311.00117.
- Ho, A., Denain, J.-S., Atanasov, D., Albanie, S., & Shah, R. (2025). A Rosetta Stone for AI Benchmarks. arXiv:2512.00193.
- Kapoor, S., Bommasani, R., et al. (2024). Position: On the societal impact of open foundation models. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2403.07918.
- National Telecommunications and Information Administration (NTIA) (2024, July 30). *Dual-Use Foundation Models with Widely Available Model Weights*. U.S. Department of Commerce. https://www.ntia.gov/issues/artificial-intelligence/open-model-weights-report
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions—I. *Mathematical Programming*, 14, 265–294. https://doi.org/10.1007/BF01588971
- Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., & Henderson, P. (2024). Fine-tuning aligned language models compromises safety, even when users do not intend to! In *International Conference on Learning Representations (ICLR)*. arXiv:2310.03693.
- Qiu, T., Laufer, B., Kleinberg, J., & Heidari, H. (2025). Modeling the economic impacts of AI openness regulation. In *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2507.14193.
- Seger, E., et al. (2023). Open-sourcing highly capable foundation models: An evaluation of risks, benefits, and alternative methods for pursuing open-source objectives. Centre for the Governance of AI. arXiv:2311.09227.
- Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games II* (Annals of Mathematics Studies 28, pp. 307–317). Princeton University Press.
- Shevlane, T. (2022). Structured access: an emerging paradigm for safe AI deployment. arXiv:2201.05159.
- Solaiman, I. (2023). The gradient of generative AI release: Methods and considerations. In *Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT)*, 111–122. arXiv:2302.04844.
- UK AI Security Institute (2026). How far behind the frontier are leading open weight models on cyber? https://aisi.gov.uk/blog/how-far-behind-the-frontier-are-leading-open-weight-models-on-cyber

---

### Appendix A: Reproducibility

```
cd code
python3 analysis.py 1000 > ../results/run_log.txt
python3 figures.py
python3 toy_model.py
```

Figures: `results/fig1_frontiers.png` (frontiers and leave-coalition-out counterfactuals), `results/fig2_lag.png` (monthly lag), `results/fig3_coordination_gap.png` (unilateral vs joint setback), `results/fig4_release_rule.png` (staleness rule by adopter), `results/fig5_shapley.png` (Shapley vs leave-one-out).

### Appendix B: Provenance of numbers

Every quantitative statement in Sections 5 and 6 is computed by the code above from the ECI file, except the following, which are cited: Epoch's lag estimates of about three months (2023–Oct 2025) [Epoch AI 2025] and four months / 8 points (since Jan 2026) [Edwards & Emberson 2026]; the one-year lag in the 2024 open-models report [Cottier et al. 2024]; the cyber lag of 4–7 months [UK AISI 2026]; and the ECI anchoring and credible-interval definitions [Epoch AI 2026b].
