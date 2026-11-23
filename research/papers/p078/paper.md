# Who Is at the Frontier? Measuring AI Power Concentration under Definitional Ambiguity and Disclosure Censoring

Target venue: ACM Conference on Fairness, Accountability, and Transparency (FAccT 2027)

---

## Abstract

Policy discussion of the concentration of frontier AI in a few companies leans on one empirical premise: that we can tell how many actors are at the frontier and whether that number is going up or down. We test that premise. We use the public Epoch AI *Notable AI Models* database (1,092 models, data through 28 October 2026). On it we formalize a family of frontier-concentration measures indexed by a compute tolerance δ, a time window W and an actor-attribution rule. We then quantify two threats to inference. The first is definitional fragility. Across 27 reasonable definitions, the change in the effective number of frontier actors (1/HHI) from end-2022 to October 2026 is negative for 16 definitions, positive for 6 and zero for 5. The second is disclosure censoring, which is non-random. For closed language and multimodal models, the share lacking any training-compute estimate rose from 16% (2021) to 93% (2026). For open-weight models it stayed at 27–40% (2025–26 odds ratio 13.7, Fisher p≈10⁻¹³). Disclosed data put the number of actors within 1 order of magnitude (OOM) of the compute frontier at 11 at end-2024 and 1 in October 2026. Partial-identification bounds for October 2026, however, are [1, 24]. A binomial disclosure model shows that observing one disclosed frontier actor is consistent (at 5%) with up to 15 true ones. The apparent 2025–26 "re-concentration" is therefore not identified from public compute data. We pair the measurement results with a minimal model of frontier membership under exponentially rising training costs. We estimate the cost growth from the same data: 3.44×/yr for frontier models, 2018–2026. The model gives five propositions. (i) Tenure at the frontier is logarithmic in budget. (ii) Pooling resources, as in a "CERN for AI", buys only ln 2/γ ≈ 0.56 years per doubling of a consortium's budget. (iii) Open-weight diffusion with lag Λ bounds every actor's capability deficit at ĝΛ ≈ 0.96 OOM regardless of budget. (iv) Binding compute caps deconcentrate on a timescale set by hardware price-performance (implied doubling 2.17 years). (v) The identification result above. We close with falsifiable predictions and a test protocol. The governance implication is direct: mandatory, standardized disclosure of training compute is a prerequisite for any concentration-based policy trigger, not an optional transparency add-on.

## 中文摘要

关于"少数公司掌握前沿AI"的政策讨论，隐含一个经验前提：我们能够测量前沿上有多少主体，并判断其数量在上升还是下降。本文检验这一前提。基于Epoch AI公开的"Notable AI Models"数据库（1,092个模型，截至2026年10月28日），我们把前沿集中度形式化为一族由算力容差δ、时间窗W和主体归属规则索引的度量，并量化两种推断威胁。（1）定义脆弱性：在27个合理定义下，2022年底至2026年10月前沿有效主体数（1/HHI）的变化有16个为负、6个为正、5个不变。（2）非随机的披露审查：闭源语言/多模态模型中缺少训练算力估计的比例从2021年的16%升至2026年的93%，开源权重模型则保持在27–40%（2025–26年优势比13.7，Fisher检验p≈10⁻¹³）。若只看已披露数据，距算力前沿1个数量级以内的主体数从2024年底的11个降至2026年10月的1个；但部分识别界为[1, 24]。二项披露模型表明，"仅观察到1个已披露前沿主体"在5%水平上与多达15个真实主体相容。因此，2025–26年表面上的"再集中"无法从公开算力数据中识别。我们还给出一个在训练成本指数增长下的前沿成员资格模型，成本增长率由同一数据估计（2018–2026年前沿模型为每年3.44倍），得到五个命题：（i）停留在前沿的时长对预算呈对数关系；（ii）资源合并（"AI版CERN"）每使联盟预算翻倍仅延长约0.56年；（iii）滞后Λ的开源权重扩散把任何主体的能力差距上限压到ĝΛ≈0.96个数量级，与预算无关；（iv）有约束力的算力上限以硬件性价比的速度（隐含倍增期2.17年）实现去集中；（v）上述识别结果。最后给出可证伪预测与检验方案。政策含义是：标准化、强制性的训练算力披露，是任何以集中度为触发条件的治理工具的前提，而不是可有可无的透明度附加项。

---

## 1 Introduction

The worry that "a handful of companies control frontier AI" now drives a large share of AI governance. Examples include competition-authority reviews of foundation-model markets [CMA 2023], proposals for public compute (the US National AI Research Resource [NAIRR Task Force 2023]), calls for a "CERN for AI", open-weight advocacy and its critics [Seger et al. 2023; Widder, West & Whittaker 2023], and compute-threshold regulation such as the EU AI Act's 10²⁵ FLOP presumption for general-purpose models with systemic risk [EU 2024, Art. 51]. Economic analysis predicts that fixed training costs that are high and rising, together with low marginal deployment costs, push the market for the most capable models toward natural oligopoly [Vipra & Korinek 2023; Korinek & Vipra 2025]. This is an AI instance of Sutton's endogenous-sunk-cost logic [Sutton 1991].

Each of these interventions is justified by, and would be evaluated against, a *measured* level and trend of concentration. A measurement question therefore comes before the policy question, and it has received surprisingly little formal attention. **How many actors are at the AI frontier, and can public data tell us whether that number is rising or falling?** We argue, and show empirically, that the answer depends on (a) arbitrary definitional choices and (b) a disclosure process that has itself become strongly selective. In late 2026 these two problems are large enough that the sign of the recent concentration trend is not identified from public compute data.

We narrow the broad question "how should power concentration in frontier AI be addressed?" to a tractable sub-question that we can answer: *What can and cannot be inferred about frontier concentration from the best public data, and, under a minimal calibrated model, how strongly do the main proposed remedies (public pooling, open-weight diffusion, compute caps) move frontier membership?*

**Contributions.**

1. **A parameterized family of frontier-concentration measures** (Section 3.1). The family is indexed by compute tolerance δ (OOM below the running compute envelope), window W and actor attribution (lab vs. capital bloc). We document how sensitive levels and trends are to these choices on real data. Across 27 definitions, the sign of the 2022→2026 change in effective actor count is not robust (16 negative, 6 positive, 5 zero).
2. **Evidence that compute disclosure is censored non-randomly, and partial-identification bounds** (Sections 3.2, 5.2). The share of closed models without a compute estimate rose steeply (logit slope 0.43/yr, SE 0.06) while the open-weight share stayed flat. We give Manski-style worst-case and incumbent-imputation bounds on the number of frontier actors. We also prove a simple binomial identification result: with 2026 disclosure rates, one observed actor is consistent with up to 15 true ones.
3. **A minimal frontier-membership model with five propositions** (Section 4). It is calibrated to growth rates we estimate from the same data: compute 4.75×/yr, cost 3.44×/yr, implied price-performance doubling 2.17 years. The results are quantitative: pooling is logarithmically weak, open-weight diffusion bounds capability gaps uniformly, and caps deconcentrate at the hardware-progress rate.
4. **An empirical check of the diffusion bound** (Section 5.4). The measured open-weight compute gap tracks ĝ·Λ(t) (correlation 0.62; mean absolute error 0.31 OOM, 2020–2026).
5. **Falsifiable predictions and a test protocol** (Section 6), together with a reproducible pipeline. Every number in this paper is produced by code in `code/` from the public data, with outputs in `results/`.

We do not claim to resolve the normative question of how much concentration is acceptable, and we do not measure capabilities directly. Our claim is narrower: concentration-based governance currently rests on measurements that cannot bear the weight placed on them, and we quantify by how much.

## 2 Related Work

**Compute trends and costs.** Sevilla et al. [2022] identified a "large-scale era" of training compute. Epoch AI later estimated that frontier training compute grows by roughly 4–5× per year [Sevilla & Roldán 2024]. Cottier et al. [2024] estimated that amortized hardware-and-energy cost of frontier training runs grew about 2.4× per year since 2016 and projected billion-dollar runs by 2027. Hobbhahn & Besiroglu [2022] estimated that GPU price-performance (FLOP/s per $) doubles roughly every 2.5 years (2.07 years for GPUs typically used in ML). We re-estimate compute and cost growth on the 2018–2026 frontier subset of the current database, and we use these rates to calibrate our model rather than as contributions in themselves.

**Compute divide and concentration.** Ahmed & Wahed [2020] documented a "compute divide" that crowds mid-tier universities out of AI research. Besiroglu et al. [2024] showed the growing share of industry in compute-intensive ML. Vipra & Myers West [2023] analyzed concentration across the compute supply chain. Sastry et al. [2024] argued that compute is a governance lever because it is detectable, excludable and quantifiable, and warned of centralization risks. Vipra & Korinek [2023] and Korinek & Vipra [2025] analyzed market structure in foundation models and predicted strong concentration at the frontier. The CMA [2023] review drew similar conclusions about key inputs. All these works assume that concentration can be read off public data. We test that assumption.

**Openness.** Cottier et al. [2024b] (Epoch AI) found that the largest open-weight models lagged the largest closed models in training compute by about 15 months (90% CI 6–22 months). A later Epoch data insight [Emberson 2025] found an average lag of about 3 months on the Epoch Capabilities Index. Seger et al. [2023] weighed the risks and benefits of open-sourcing highly capable models. Widder, West & Whittaker [2023] argued that "open" AI does not by itself reduce concentrated power. Our Proposition 3 formalizes the specific sense in which open-weight diffusion *does* bound capability deficits, and Section 5.4 tests it.

**Transparency.** The Foundation Model Transparency Index [Bommasani et al. 2023] found low developer transparency. Its 2025 edition reports a decline in average score (58 to 40) and identifies training compute among the least-disclosed topics [Bommasani et al. 2025, arXiv:2512.10169]. We quantify the *consequence* of this opacity for concentration measurement: compute disclosure falls far faster for closed models than for open ones, and this bias specifically destroys identification of the number of frontier actors.

**Industrial organization and identification.** We borrow the endogenous-sunk-cost view of concentration [Sutton 1991] and the entry-threshold logic of Bresnahan & Reiss [1991]. We use the Herfindahl–Hirschman index [Rhoades 1993] and its numbers-equivalent. For missing data we follow the partial-identification approach of Manski [2003]: report bounds instead of imposing untestable missing-at-random assumptions.

**Gap addressed.** We know of no prior work that (i) treats "the frontier" as a parameterized object and reports the sensitivity of concentration conclusions to its parameters, (ii) documents differential disclosure censoring by access regime and bounds concentration accordingly, or (iii) ties a calibrated membership model to the specific remedies on the table, with closed-form, quantitative comparisons.

## 3 Data and Measurement Framework

### 3.1 Data

We use Epoch AI's *Notable AI Models* database [Epoch AI 2026] (CC-BY 4.0), retrieved on 22 November 2026. It covers 1,092 models published up to 28 October 2026. The database's CSV endpoint was not reachable directly from our computing environment, so we retrieved it through a web-extraction service. That service replaced line breaks with spaces, and we restored rows using the known column count and date anchors (`code/parse_epoch.py`). We also retrieved Epoch's separate *Frontier AI Models* file independently. As a validation check, training-compute values agree exactly for all 114 models present in both files (`results/parse_validation.json`). From 2010 onward the database contains 966 models, of which 481 carry a training-compute estimate (FLOP). Fields used: model, organization, publication date, training compute, training compute cost (2023 USD, Epoch's estimate), model accessibility, country, organization category, domain, and Epoch's "frontier model" flag (top-10 by compute at release).

Accessibility classes: *open* = open weights (unrestricted, restricted-use, or non-commercial); *closed* = API access, hosted access, limited access or unreleased.

### 3.2 Frontier sets and concentration measures

Let each model *i* have publication time *tᵢ* and log-compute *Lᵢ* = log₁₀ Cᵢ. The **compute envelope** is the running maximum M(τ) = max{Lᵢ : tᵢ ≤ τ}. For tolerance δ ≥ 0 and window W > 0, the **(δ, W)-frontier set** at time τ is

  S_{δ,W}(τ) = { i : τ − W < tᵢ ≤ τ, Lᵢ ≥ M(τ) − δ }.

An **attribution rule** *a(·)* maps a model to an actor. *Lab* attribution takes the first listed organization and merges corporate aliases (e.g., Google, Google Brain, DeepMind and Google DeepMind become "Alphabet"; Meta AI and FAIR become "Meta"). *Capital-bloc* attribution additionally maps labs to their largest publicly reported strategic investor (OpenAI→Microsoft, Anthropic→Amazon, Inflection→Microsoft). This mapping is deliberately coarse and serves only as a sensitivity check. With *sₐ* the share of models in S attributed to actor *a*, we report:

- the actor count n(τ) = |{a(i) : i ∈ S}|;
- the effective number N_eff(τ) = 1 / Σₐ sₐ², i.e., the numbers-equivalent of the HHI;
- the top-actor share.

We evaluate on a monthly grid from January 2016 to October 2026, with δ ∈ {0.3, 0.5, 1.0, 1.5} OOM, W ∈ {0.5, 1, 2} years, and both attribution rules. As an external alternative definition we also use Epoch's own top-10 frontier flag.

### 3.3 Disclosure censoring and partial identification

A model with no compute estimate cannot be placed relative to M(τ). This censoring is harmless only if it is unrelated to the outcome, and Section 5.2 shows it is not. Following Manski [2003], we bound the actor count for language/multimodal industry models in the window that lack compute ("unknown models"):

- **Lower bound (disclosed only):** n_LB = n(τ) computed on models with compute.
- **Upper bound (worst case):** n_UB adds every actor with at least one unknown model in the window, i.e., it assumes every such model could be δ-frontier.
- **Incumbent imputation (point estimate under an explicit assumption):** add an actor's unknown models only if that actor had a disclosed δ-frontier model, measured relative to the envelope at that model's own release, within the previous 24 months.

## 4 A Minimal Model of Frontier Membership

**Setup.** Time is continuous. The cost of a training run that matches the compute envelope at time *t* is C(t) = C₀e^{γt}. The envelope compute itself grows at rate *g* (nats/yr), so compute per dollar improves at rate *h* = *g* − γ. Reaching the δ-frontier requires compute M(t)·10^{−δ}, which at time-*t* prices costs C(t)·10^{−δ}. Actor *i* can spend Bᵢ(t) = bᵢe^{μt} per year on a single frontier-scale run. Actor *i* is **δ-capable** at *t* if Bᵢ(t) ≥ C(t)10^{−δ}. We express budgets relative to current frontier-run cost, xᵢ = bᵢ/C₀, so no firm-level budget data are needed.

**Proposition 1 (Tenure is logarithmic in budget).** If γ > μ, actor *i* remains δ-capable exactly until

  Tᵢ(δ) = [ ln xᵢ + δ ln 10 ]⁺ / (γ − μ).

Hence (a) the number of δ-capable actors N_δ(t) is non-increasing in *t*; (b) raising δ by Δ shifts every tenure by Δ ln10/(γ − μ), so N_{δ+Δ}(t) = N_δ(t − Δ ln10/(γ−μ)); (c) multiplying a budget by a factor *k* extends tenure by ln k/(γ − μ).

*Proof.* Bᵢ(t) ≥ C(t)10^{−δ} ⇔ ln bᵢ + μt ≥ ln C₀ + γt − δ ln10 ⇔ t ≤ (ln xᵢ + δ ln 10)/(γ−μ). Each actor's capable set is an initial interval, so the count is non-increasing. Parts (b) and (c) follow by substitution. ∎

Part (b) has a direct measurement implication. *In the model*, the choice of δ shifts the concentration series in time but cannot reverse the sign of its trend. Section 5.1 shows that on real data the sign *does* flip across definitions. The definitional fragility therefore comes from sparse, lumpy, censored data, not from a substantive disagreement among definitions.

**Proposition 2 (Pooling is logarithmically weak).** A coalition K that pools budgets has tenure T_K = [ln Σ_{i∈K} xᵢ + δ ln10]⁺/(γ − μ). Its gain over the best member is

  T_K − max_{i∈K} Tᵢ = ln( Σᵢ xᵢ / maxᵢ xᵢ ) / (γ − μ) ∈ [0, ln|K| /(γ − μ)],

with the upper bound attained for equal members. The coalition remains capable forever if and only if its budget grows at least as fast as frontier cost (μ_K ≥ γ).

*Proof.* Apply Proposition 1 to the summed budget. The ratio Σxᵢ / max xᵢ lies in [1, |K|]. ∎

**Proposition 3 (Diffusion bounds capability deficits uniformly).** Define actor *i*'s accessible compute Aᵢ(t) as the maximum of what it can train and the largest open-weight model O(t). Define its exclusivity deficit as Eᵢ(t) = log₁₀M(t) − log₁₀Aᵢ(t). Suppose the open frontier follows the envelope with lag Λ, i.e., O(t) = M(t − Λ), and the envelope grows at ĝ OOM/yr. Then Eᵢ(t) ≤ ĝΛ for every actor and time, independent of budget. Without open release, for an actor that has exited (t > Tᵢ), Eᵢ(t) = δ + (γ − μ)(t − Tᵢ)/ln10, which grows without bound.

*Proof.* Aᵢ(t) ≥ O(t) = M(t−Λ) = M(t)·10^{−ĝΛ}. For the second claim, after exit the actor's affordable compute is Bᵢ(t)/C(t)·M(t) = M(t)·10^{−δ}·e^{−(γ−μ)(t−Tᵢ)}. ∎

**Proposition 4 (Caps deconcentrate at the hardware rate).** Suppose a binding cap K̄ on training compute takes effect at t_c ≥ 0, so the envelope is frozen at K̄. An actor whose affordable compute at t_c is Aᵢ(t_c) < K̄10^{−δ} becomes δ-capable at

  t_c + ln( K̄10^{−δ}/Aᵢ(t_c) ) / (μ + h).

Every actor with μ + h > 0 eventually becomes capable, and the deconcentration timescale is 1/(μ + h).

*Proof.* Affordable compute grows at μ + h once the envelope stops moving. Solve for the crossing time. ∎

**Proposition 5 (Identification under selective disclosure).** Suppose N actors are truly at the frontier in a window. Each has *m* frontier models, and each model's compute is disclosed independently with probability *q*. The number of actors with at least one disclosed model is then Binomial(N, p), with p = 1 − (1 − q)^m. Given an observed count n_obs, the set of N consistent at level α is {N : P(Bin(N, p) ≤ n_obs) ≥ α}. If disclosure is positively correlated within actors (labs either disclose or do not), the effective p falls toward *q* and the consistent set widens.

*Proof.* Actors' indicators of "at least one disclosed model" are independent Bernoulli(p). Under perfect within-actor correlation, each actor is observed with probability *q* ≤ p, and the binomial CDF at n_obs is decreasing in p. ∎

**Calibration.** We estimate *g* and γ by OLS of log₁₀ compute and log₁₀ cost on publication date for Epoch-flagged frontier models, 2018–2026, with 2,000-sample bootstrap CIs (Section 5.3). *h* is then implied. We report tenure for budget ratios X ∈ [0.1, 30], δ ∈ {0, 0.5, 1}, and μ ∈ {0, ln 1.25, ln 1.5} (flat budgets, +25%/yr, +50%/yr).

## 5 Results

All numbers in this section are produced by `code/analysis.py` and `code/model.py` from the Epoch data. Outputs are in `results/empirical.json` and `results/model.json`.

### 5.1 Concentration levels and trends depend on the definition

**Table 1.** Frontier actors (n) and effective number (N_eff = 1/HHI) at year-end, window W = 1 year, lab attribution. "2026" means the 12 months ending October 2026.

| Definition | 2022: n / N_eff | 2024: n / N_eff | 2026: n / N_eff |
|---|---|---|---|
| δ = 0.3 OOM | 2 / 1.47 | 4 / 4.00 | 1 / 1.00 |
| δ = 0.5 OOM | 3 / 2.00 | 6 / 6.00 | 1 / 1.00 |
| δ = 1.0 OOM | 7 / 3.09 | 11 / 9.80 | 1 / 1.00 |
| δ = 1.5 OOM | 9 / 2.91 | 16 / 10.27 | 2 / 2.00 |
| Epoch top-10 flag | 3 / 1.82 | 7 / 5.14 | 1 / 1.00 |

Taken at face value, Table 1 and Figure 2 (`results/fig2_definition_sweep.png`) tell a striking story. The frontier "deconcentrated" through 2024, with up to 11 actors within 1 OOM of the envelope and N_eff ≈ 9.8. It then "re-concentrated" to a single disclosed actor (OpenAI, whose GPT-6 Astra at 9.4×10²⁶ FLOP set the envelope on 3 September 2026). However:

- **Levels vary by an order of magnitude across definitions.** At end-2024, n ranges from 4 (δ = 0.3) to 16 (δ = 1.5).
- **Trend signs are not robust.** Over 27 definitions (24 combinations of δ, W and attribution, plus the Epoch flag at three window lengths), the sign of the change in N_eff from end-2022 to October 2026 is negative for 16, positive for 6 and zero for 5. Two-year windows produce *increases* for δ ∈ {0.3, 0.5, 1.5}, because the window then still includes the 2024–25 cohort.
- **Attribution matters less than expected at year-ends.** Capital-bloc attribution changes the year-end values only in 2023 (δ = 1: 7 → 5 actors; N_eff 6.40 → 4.00). The reason is that few bloc-linked labs appear simultaneously in the disclosed frontier sets.

Proposition 1(b) says that, in the idealized model, δ shifts the series in time and preserves the sign of its trend. The observed sign reversals therefore indicate that discreteness and the composition of the disclosed sample, not substantive definitional disagreement, drive measured trends. The next subsection shows that composition is not random.

### 5.2 Compute disclosure collapses for closed models only

**Table 2.** Share of language/multimodal notable models with no training-compute estimate, by year and access class.

| Year | Closed: n (missing share) | Open weights: n (missing share) |
|---|---|---|
| 2021 | 19 (15.8%) | 19 (15.8%) |
| 2022 | 29 (24.1%) | 18 (27.8%) |
| 2023 | 36 (61.1%) | 40 (27.5%) |
| 2024 | 48 (66.7%) | 30 (26.7%) |
| 2025 | 51 (80.4%) | 41 (26.8%) |
| 2026 (to Oct.) | 55 (92.7%) | 30 (40.0%) |

In 2025–26, 92 of 106 closed models lack compute estimates, compared with 23 of 71 open-weight models (odds ratio 13.7; Fisher exact p = 1.3×10⁻¹³). A logistic regression of missingness on publication year for closed models (n = 289, 2018–2026) gives a slope of 0.426/yr (SE 0.061), i.e., the odds of missingness multiply by 1.53 each year. Before 2022 the two access classes were disclosed at similar rates. Since then they have diverged sharply.

The 2025 Foundation Model Transparency Index reports declining developer transparency, with training compute among the least-disclosed topics [Bommasani et al. 2025]. Our numbers are consistent with that. They also add something the index does not show: the censoring is *differential by access regime*. Closed frontier labs are exactly the actors whose presence at the frontier is in question, so the censoring is informative about the quantity being measured.

**Partial identification.** Figure 3 (`results/fig3_partial_id.png`) shows the bounds. In October 2026 (δ = 1, W = 1), the disclosed-only lower bound is 1 actor (OpenAI). Incumbent imputation gives 3 (adding Meta and xAI, which have unknown-compute models in the window and disclosed δ-frontier models within the prior 24 months). The worst-case upper bound is 24, because 82 industry language/multimodal models released in the last 12 months lack compute. For end-2024 the corresponding bounds are 11 / 12 / 26. The lower bound collapses from 11 to 1 over 2025–26, while the upper bound barely moves (26 to 24). The measured "re-concentration" is therefore a movement of the lower bound inside an interval that has not shrunk.

Incumbent imputation does not add Alphabet or Anthropic. Their most recent *disclosed* δ = 1 frontier models are older than 24 months, or lie just over 1 OOM below the envelope at release (Claude 3.7 Sonnet, 3.35×10²⁵ FLOP, released one week after the 3.5×10²⁶ FLOP Grok 3). This illustrates the problem rather than resolving it: assessing two of the most prominent frontier developers requires assumptions that public compute data cannot check.

**Identification under the binomial model (Proposition 5).** Among the 70 closed language/multimodal models released in the 12 months to October 2026, only 5.7% have compute estimates (q = 0.057). The eight most prolific closed actors released a median of m = 5.5 notable models each: OpenAI 21, Anthropic 10, Alphabet 7, Alibaba 6, xAI 5, Meta 5, Character.ai 3, ByteDance 2. This gives p = 1 − (1 − q)^m = 0.276. With N = 5 true frontier actors, the probability of observing at most one disclosed actor is 0.58. With N = 10 it is 0.19. The largest N consistent with n_obs = 1 at the 5% level is **15** (Figure 5, right). At a 2021-like disclosure rate (q = 0.8), the same observation would identify N = 1. If disclosure is correlated within labs, the consistent set is wider still. These numbers apply q from all closed notable models to frontier models, which is an assumption. Epoch may preferentially estimate frontier-scale models, which would raise the effective q. Section 7 discusses this caveat.

### 5.3 Growth rates used for calibration

**Table 3.** OLS on Epoch-flagged frontier models. 95% bootstrap CIs in brackets. OOM/yr = log₁₀ units per year.

| Period | n | Compute growth (OOM/yr) | ×/yr | Cost n | Cost growth (OOM/yr) | ×/yr |
|---|---|---|---|---|---|---|
| 2018–2026 | 45 | 0.676 [0.623, 0.730] | 4.75 | 38 | 0.537 [0.456, 0.615] | 3.44 |
| 2018–2022 | 23 | 0.653 [0.525, 0.805] | 4.49 | 21 | 0.392 [0.216, 0.637] | 2.46 |
| 2022–2026 | 22 | 0.641 [0.523, 0.763] | 4.37 | 17 | 0.557 [0.363, 0.809] | 3.61 |

Our compute growth of 4.75×/yr agrees with Epoch's published 4–5×/yr [Sevilla & Roldán 2024]. Our 2018–2022 cost growth of 2.46×/yr matches Cottier et al.'s [2024] 2.4×/yr. The later period is faster (3.61×/yr), but the CI is wide. The implied compute-per-dollar improvement is h = g − γ = 0.32 nats/yr (doubling time 2.17 years). This falls between Hobbhahn & Besiroglu's [2022] estimates of 2.46 years (all GPUs) and 2.07 years (ML GPUs), which provides an internal consistency check that the model does not impose. The most expensive runs with cost estimates are Grok 4 ($388M, 2023 USD), GPT-4.5 ($366M) and Grok 3 ($218M). Cost estimates are missing for most recent closed models, so these are lower envelopes of the true frontier cost.

### 5.4 Remedies under the calibrated model

**Pooling (Proposition 2).** With flat budgets (μ = 0) and γ = 1.236 nats/yr (3.44×/yr), each doubling of a consortium's budget buys **0.56 years** of frontier tenure (95% range 0.49–0.66). Pooling *k* equal members buys 0.56 (k = 2), 1.30 (k = 5), 1.86 (k = 10), 2.67 (k = 27) and 3.72 years (k = 100). A consortium with an annual budget ten times the current frontier-run cost (X = 10) stays at the exact frontier (δ = 0) for 1.86 years (1.63–2.19). If it settles for δ = 1, it stays 3.72 years, and 5.54 years if its budget also grows 50% per year. Permanent membership requires budget growth of at least 3.44× per year (Figure 5, left; full grid in `results/model.json`). Even large pooling initiatives, sized as multiples of the current frontier run, therefore buy a few years of frontier participation, not durable parity, unless they are funded to grow with frontier costs. They may still serve other goals well, such as access for researchers below the frontier, evaluation capacity or sovereignty in deployment.

**Diffusion (Proposition 3) — empirical check.** Figure 4 (`results/fig4_open_gap.png`) plots the measured gap G(t) = log₁₀M(t) − log₁₀O(t) between the compute envelope and the largest open-weight model, against ĝΛ(t), where Λ(t) is the time since the envelope first reached O(t). Over 2020–2026 the mean gap is 0.90 OOM and the mean lag 1.42 years. G tracks ĝΛ with correlation 0.62 and mean absolute error 0.31 OOM. The deviations come from the stepwise envelope, which the proposition's exponential-envelope assumption smooths over. The year-end gap fell to 0.12 OOM in 2024, when Llama 3.1-405B (3.8×10²⁵ FLOP) nearly matched the then-envelope. It has since widened to 1.39 OOM (lag 2.91 years) in October 2026, because no open-weight model with a compute estimate has exceeded Llama 3.1-405B. The mean lag gives a diffusion bound of ĝΛ̄ ≈ 0.96 OOM.

The compute lag diverges substantially from capability-based lags. Epoch's 2024 analysis reported a compute lag of about 15 months [Cottier et al. 2024b], consistent with our 1.42-year mean. Their 2025 capability-index analysis, however, reports an average lag of only about 3 months [Emberson 2025]. Combining our compute lag with their capability lag gives a back-of-envelope compute-efficiency offset of ĝ(1.42 − 0.25) ≈ 0.79 OOM. On this reading, compute-based measures *overstate* the capability exclusivity of the closed frontier by almost an order of magnitude. This number mixes two sources and two periods, and we treat it as a hypothesis (Prediction P4), not a finding.

**Caps (Proposition 4).** With h = 0.32 nats/yr, an actor that can afford 10% of a cap-level run when the cap binds catches up in 7.19 years with flat budgets and 3.17 years with budgets growing 50% per year. At 1% the catch-up takes 14.39 and 6.35 years respectively. Caps deconcentrate in the long run, but only on hardware-progress timescales, and only if the cap actually binds the leaders.

**Comparison.** For an actor priced out of the frontier, the remedies scale very differently. Pooling adds tenure proportional to ln k. Caps close the gap at rate μ + h. Diffusion bounds the deficit at ĝΛ immediately and for every actor, including those with no training budget at all. Diffusion is the only one of the three whose benefit does not depend on the recipient's budget. It also transfers the risks of open release that Seger et al. [2023] analyze, and it does not address concentration in deployment, distribution or compute infrastructure [Widder et al. 2023; Vipra & Myers West 2023].

### 5.5 Geography

Among δ = 1, W = 1 frontier models with disclosed compute, the end-2024 set contained 9 US, 4 Chinese and 1 other model, from 7 US, 3 Chinese and 1 other actor. The end-2025 and October-2026 sets contain only US models (5 and 1). Given Section 5.2, we do not interpret this as evidence that Chinese developers left the frontier. Thirty-two language/multimodal models from Chinese organizations released since January 2025 lack compute estimates (including Qwen, GLM, Kimi, MiniMax and Tencent Hunyuan releases).

## 6 Falsifiable Predictions and Test Protocol

We pre-specify four predictions, testable against future versions of the same public database. The pipeline in `code/` can be rerun unchanged on a new snapshot.

**P1 (Re-identification).** If Epoch back-fills compute estimates for 2025–26 closed releases, so that the closed-model disclosure share for 2026 rises above 50%, then the δ = 1, W = 1 actor count for the window ending October 2026 will be ≥ 3. *Falsified if* disclosure rises above 50% and the count stays at 1.

**P2 (Diffusion-bound tracking).** On the 2027 snapshot, the gap G(t) will continue to lie within 0.5 OOM of ĝΛ(t) for at least 75% of 2027 months. Separately, unless an open-weight model with an estimated compute above 10²⁶ FLOP is released by end-2027, the end-2027 gap will exceed 1.5 OOM. *Falsified if* the gap shrinks below 1.5 OOM without such a release (that would indicate envelope stagnation or a data revision).

**P3 (Logarithmic pooling).** Consider any public or consortium frontier effort whose annual training budget, relative to the contemporaneous largest disclosed frontier-run cost, can be measured at two dates. Its δ-frontier status will be lost within the tenure predicted by Proposition 1, using the γ CI from Table 3, unless its budget grows at ≥ 3.4×/yr. *Falsified* by an effort that stays within δ for longer than the upper CI tenure without budget growth matching γ (that would imply a cost-efficiency advantage the model omits).

**P4 (Compute overstates capability exclusivity).** Matching compute lags and capability-index lags on the same set of models and the same period will give an offset ĝ(Λ_compute − Λ_capability) of at least 0.4 OOM. *Falsified if* the matched offset is below 0.4 OOM.

**Protocol.** (1) Freeze the code at this paper's version. (2) Rerun on the first snapshot after 1 January 2028. (3) Report all four outcomes, including failures. (4) For P1, also report the disclosure share by access regime, so that a change in the measured count can be attributed to disclosure versus entry.

## 7 Discussion

**What the evidence supports.** Three claims are robust to our sensitivity analyses. (a) Training-compute and cost growth (4.75×/yr and 3.44×/yr) mean that independent frontier participation is a tenure measured in years even for well-funded actors, and pooling extends it only logarithmically. This supports the endogenous-sunk-cost view of a concentrated frontier [Sutton 1991; Korinek & Vipra 2025]. (b) The public data cannot currently say whether the number of frontier developers is rising or falling. The lower bound collapsed, the upper bound did not, and the collapse coincides with a selective disclosure shift. (c) Open-weight diffusion has historically held the compute deficit of non-frontier actors near one OOM. This is the only lever whose benefit is independent of the recipient's budget.

**Governance implications.**

1. *Disclosure first.* Policies keyed to concentration thresholds (merger review, utility-style obligations [Vipra & Korinek 2023], or triggers for public-option investment) need an observable trigger. At 2026 disclosure rates the trigger is not observable. A standardized, mandatory report of training compute (and ideally cost) for models above a modest threshold would collapse the identification interval at essentially no capability cost. The EU AI Act's 10²⁵ FLOP presumption [EU 2024] already requires developers to estimate this quantity. Publishing it, even in coarse bins, would be enough to restore identification, because Proposition 5's p approaches 1 when q approaches 1.
2. *Size public options to their purpose.* Proposition 2 implies that "CERN for AI"-style efforts should not be justified by frontier parity unless they are funded to grow with frontier cost. Their stronger rationale is sub-frontier access, evaluation and science, for which Proposition 3 suggests that a pipeline of open-weight releases may deliver more capability per dollar.
3. *Measure capabilities, not only compute.* The divergence between compute lag and capability lag (Section 5.4) means compute-based concentration metrics may overstate exclusivity. A concentration index for governance should report capability-based and compute-based variants side by side, together with the δ-sweep, rather than a single number.

**Why the measurement problem is itself a power problem.** When the most-resourced actors stop disclosing the inputs that would reveal their market position, the measured structure of the market comes to be shaped by the strategic choices of the actors whose power is in question. This is a transparency issue of the kind FAccT is concerned with. It is not specific to AI, but AI shows it in an extreme form: between 2021 and 2026, the share of closed language/multimodal models with a compute estimate fell from 84% to 7%.

## 8 Limitations

- **Single data source.** All empirical results use Epoch AI's database. Its notability criteria, compute estimates (many labeled "Speculative" or "Likely") and back-fill timing all shape our results. Recent releases are especially affected: compute estimates are often added months after release, so part of the 2026 missingness may be reporting delay rather than permanent non-disclosure. P1 is designed to test this.
- **Retrieval path.** We obtained the CSV through a web-extraction service, not the official download, and reconstructed rows by parsing. Cross-file agreement (114/114 compute values) supports the reconstruction, but six rows required resynchronization and fields we did not validate (e.g., country, accessibility) could contain residual errors.
- **Compute is a proxy.** Training compute does not capture algorithmic efficiency, post-training, inference-time compute, data advantages or distribution power. The model treats a "frontier run" as one homogeneous good, and in Section 5.4 the capability lag is much shorter than the compute lag.
- **Attribution is coarse.** Mapping multi-organization models to their first listed organization, and labs to "capital blocs", is a simplification. Investor relationships (e.g., Anthropic with both Amazon and Alphabet) are not exclusive.
- **Model simplicity.** Budgets are exogenous and deterministic. Costs grow at a constant rate, with no strategic response, no revenue feedback, no capacity constraints on chips or energy, and no endogenous γ. Proposition 3 assumes a constant lag and an exponential envelope, whereas the real envelope is stepwise and the open lag fluctuates (0.1–2.9 years).
- **Identification parameters.** The binomial model applies the disclosure rate of all closed notable models to frontier models and assumes independence within actors. Both choices could bias the consistent set, in opposite directions.
- **Small samples.** Frontier subsets contain 22–45 models, and the cost regressions 17–38. The CIs in Table 3 reflect sampling variation but not estimation error in the underlying cost figures.
- **No direct capability data.** We cite, but did not re-analyze, capability-index lags. P4 is a hypothesis.

## 9 Conclusion

The question "how do we address concentration of frontier AI in a few companies?" assumes we can see that concentration. Using the most comprehensive public dataset, we find that the answer to "how many actors are at the frontier?" varies by an order of magnitude across reasonable definitions. We also find that, since 2023, it has been obscured by a disclosure collapse confined to closed models, to the point where October-2026 data are consistent with anywhere from 1 to 24 frontier actors. A calibrated model shows that, of the main remedies, pooling buys frontier tenure only logarithmically, caps deconcentrate at the slow pace of hardware progress, and open-weight diffusion bounds every actor's capability deficit at about one order of magnitude of compute. The most immediately actionable conclusion concerns measurement, not market structure: standardized public reporting of training compute is a precondition for governing concentration at all. We provide the code and pre-specified predictions so that future data can confirm or refute these claims.

## References

- Ahmed, N., & Wahed, M. (2020). The De-democratization of AI: Deep Learning and the Compute Divide in Artificial Intelligence Research. arXiv:2010.15581.
- Besiroglu, T., Bergerson, S. A., Michael, A., Heim, L., Luo, X., & Thompson, N. (2024). The Compute Divide in Machine Learning: A Threat to Academic Contribution and Scrutiny? arXiv:2401.02452.
- Bommasani, R., Klyman, K., Longpre, S., Kapoor, S., Maslej, N., Xiong, B., Zhang, D., & Liang, P. (2023). The Foundation Model Transparency Index. arXiv:2310.12941.
- Bommasani, R., et al. (2025). The 2025 Foundation Model Transparency Index. arXiv:2512.10169.
- Bresnahan, T. F., & Reiss, P. C. (1991). Entry and Competition in Concentrated Markets. *Journal of Political Economy*, 99(5), 977–1009.
- Competition and Markets Authority (CMA). (2023). *AI Foundation Models: Initial Report*. 18 September 2023. https://www.gov.uk/government/publications/ai-foundation-models-initial-report
- Cottier, B., Rahman, R., Fattorini, L., Maslej, N., Besiroglu, T., & Owen, D. (2024). The rising costs of training frontier AI models. arXiv:2405.21015.
- Cottier, B., You, J., Martemianova, N., & Owen, D. (2024b). How far behind are open models? Epoch AI, 4 November 2024. https://epoch.ai/publications/open-models-report
- Emberson, L. (2025). Open-weight models lag state-of-the-art by around 3 months on average. Epoch AI Data Insight, 30 October 2025. https://epoch.ai/data-insights/open-weights-vs-closed-weights-models
- Epoch AI. (2026). Data on Notable AI Models (and Frontier AI Models). CC-BY 4.0. https://epoch.ai/data/notable-ai-models (snapshot retrieved 22 November 2026).
- European Union. (2024). Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 51: Classification of general-purpose AI models as general-purpose AI models with systemic risk.
- Hobbhahn, M., & Besiroglu, T. (2022). Trends in GPU price-performance. Epoch AI, 27 June 2022. https://epoch.ai/blog/trends-in-gpu-price-performance
- Korinek, A., & Vipra, J. (2025). Concentrating intelligence: scaling and market structure in artificial intelligence. *Economic Policy*, 40(121), 225–256. (NBER Working Paper 33139, 2024.)
- Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer Series in Statistics. Springer. doi:10.1007/b97478.
- National Artificial Intelligence Research Resource Task Force. (2023). *Strengthening and Democratizing the U.S. Artificial Intelligence Innovation Ecosystem: An Implementation Plan for a National Artificial Intelligence Research Resource*. January 2023.
- Rhoades, S. A. (1993). The Herfindahl-Hirschman Index. *Federal Reserve Bulletin*, 79, 188–189.
- Sastry, G., Heim, L., et al. (2024). Computing Power and the Governance of Artificial Intelligence. arXiv:2402.08797.
- Seger, E., et al. (2023). Open-Sourcing Highly Capable Foundation Models: An evaluation of risks, benefits, and alternative methods for pursuing open-source objectives. arXiv:2311.09227.
- Sevilla, J., Heim, L., Ho, A., Besiroglu, T., Hobbhahn, M., & Villalobos, P. (2022). Compute Trends Across Three Eras of Machine Learning. *2022 International Joint Conference on Neural Networks (IJCNN)*. arXiv:2202.05924.
- Sevilla, J., & Roldán, E. (2024). Training compute of frontier AI models grows by 4–5x per year. Epoch AI. https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year
- Sutton, J. (1991). *Sunk Costs and Market Structure: Price Competition, Advertising, and the Evolution of Concentration*. MIT Press.
- Vipra, J., & Korinek, A. (2023). Market Concentration Implications of Foundation Models: The Invisible Hand of ChatGPT. Brookings Center on Regulation and Markets Working Paper; arXiv:2311.01550.
- Vipra, J., & Myers West, S. (2023). *Computational Power and AI*. AI Now Institute.
- Widder, D. G., West, S. M., & Whittaker, M. (2023). Open (For Business): Big Tech, Concentrated Power, and the Political Economy of Open AI. SSRN 4543807.

---

*Reproducibility.* `bash code/run_all.sh` regenerates every number and figure (CPU, < 1 minute). Figures: `results/fig1_envelope.png` (compute envelope, open vs. closed), `fig2_definition_sweep.png`, `fig3_partial_id.png`, `fig4_open_gap.png`, `fig5_model.png`. Numerical outputs: `results/empirical.json`, `results/model.json`, `results/parse_validation.json`. Data-provenance details: `README.md`.
