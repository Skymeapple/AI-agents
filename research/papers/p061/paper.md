# Aggregation, Not Exposure: How Task-to-Job Assumptions Determine Estimates of AI Job Displacement and Industry Rankings

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES 2027). The paper also fits a policy-oriented economics journal such as *Economic Policy* or *Research Policy*.

**Code and data:** `code/` and `results/` in this package; run instructions are in `README.md`.

---

## Abstract

Most widely cited answers to the question "how many jobs will AI replace, and in which industries?" have two steps. First, someone rates the AI exposure of individual work tasks. Second, those ratings are turned into a claim about jobs, usually by taking the share of an occupation's tasks that are exposed. Almost all methodological debate concerns the first step. We show that the second step, *aggregation*, matters much more. We combine the public task-level GPT-4 and human exposure labels of Eloundou et al. (19,265 O\*NET tasks), O\*NET 30.0 task-importance ratings and May-2024 BLS OEWS employment by occupation and NAICS sector, covering 96.8% of employment in detailed occupations. On these data we build a multiverse of 1,680 specifications: 7 exposure measures × 4 task weightings × 60 aggregation rules. The rules include the linear exposure share, whole-job thresholds, O-ring (all-core-tasks) complementarity, and a closed-form CES task model with demand response under automation, augmentation and label-based modes. Main-effect variance decomposition attributes 84% of the variance in the headline "net share of jobs displaced" to the aggregation rule, 4% to the choice of exposure measure, 0.1% to whether GPT-4 or humans did the rating, and under 0.001% to task weighting. With one exposure measure held fixed, the headline ranges from an 88% employment *gain* to an 80% *loss*. We prove three simple results that organise these findings: (i) an industry ordering holds for every aggregation rule that is monotone in the exposed share if and only if the industries' employment-weighted exposure distributions are ordered by first-order stochastic dominance; (ii) the linear exposure index is the exact CES outcome only on the knife-edge where the elasticity of substitution across tasks equals the demand elasticity; (iii) under pure automation, employment in an exposed occupation rises if and only if the demand elasticity exceeds ε\* ≥ max(1, σ). Empirically, only 27% of sector pairs under our preferred measure (and 15–54% across measures) satisfy dominance. Finance, management of companies, professional services and information rank as the most exposed sectors under most rules, but whether exposure means losing or gaining jobs flips with the demand elasticity. Changes in occupational employment between May 2021 and May 2024 (panel-disjoint OEWS estimates) cannot tell the rules apart. A Monte Carlo that uses the actual residual noise shows that the linear rule would be detected with only 22% power even if a fifth of its implied long-run displacement had already happened. We end with a pre-registrable test protocol and falsifiable predictions.

## 中文摘要

关于"AI会替代多少工作、哪些行业冲击最大"，最常被引用的答案通常分两步得出：先评估单个工作任务的AI暴露度，再把任务层面的暴露度换算成岗位层面的结论，最常见的做法是计算一个职业中暴露任务所占的份额。现有方法论讨论几乎都围绕第一步展开，本文则表明第二步"聚合"的影响要大得多。我们把Eloundou等人公开的任务级GPT-4与人工暴露标注（19,265项O\*NET任务）、O\*NET 30.0任务重要性评分以及2024年5月BLS OEWS分职业、分NAICS行业的就业数据合并，覆盖细分职业就业的96.8%，并在此基础上构建了一个包含1,680种设定的"多重宇宙"：7种暴露度量 × 4种任务权重 × 60种聚合规则。这些规则包括线性暴露份额、整岗阈值、O型环（所有核心任务都须可自动化）互补性，以及一个带需求响应的闭式CES任务模型，分为自动化、增强和按标签区分三种模式。主效应方差分解显示，"净替代就业比例"这一核心数字的方差中，84%来自聚合规则，4%来自暴露度量的选择，0.1%来自由GPT-4还是人工打分，任务权重的贡献不足0.001%。即使固定一种暴露度量，结论也可以从就业增加88%变到减少80%。我们证明了三个简单命题：(i) 行业排序对所有关于暴露份额单调的聚合规则都保持不变，当且仅当各行业按就业加权的暴露分布满足一阶随机占优；(ii) 线性暴露指数只有在任务间替代弹性恰好等于需求弹性这一"刀锋"条件下，才是CES模型的精确结果；(iii) 在纯自动化情形下，暴露职业的就业上升当且仅当需求弹性超过ε\* ≥ max(1, σ)。实证上，在我们的首选度量下只有27%的行业对满足占优（各度量下为15–54%）。在大多数规则下，金融保险、企业管理、专业技术服务和信息业是暴露度最高的行业，但暴露意味着失业还是就业增加，取决于需求弹性。2021年5月至2024年5月的职业就业变化（OEWS样本面板互不重叠）无法区分各种规则。基于真实残差噪声的蒙特卡洛模拟显示，即使线性规则所隐含的长期替代已经实现五分之一，检出概率也只有22%。最后，我们给出一个可预注册的检验方案和若干可证伪的预测。

---

## 1 Introduction

Public and policy discussion of generative AI and work keeps returning to two questions: *how many jobs will AI replace*, and *which industries will be hit hardest*? The quantitative answers in circulation almost all follow one template. A rubric scores each O\*NET work task for whether a large language model, or software built on one, could cut the time it takes substantially. The scorer is a human annotator, an LLM, or patent text matching. The task scores are then averaged within an occupation, and the occupational average is reported as the share of the occupation's work that is "exposed". Reported numbers then go one of two ways. Some report the employment-weighted mean of that share (in effect, "x% of work hours are exposed"). Others count the workers in occupations above a cutoff ("y% of workers have at least half of their tasks exposed") (Eloundou et al., 2024; Gmyrek, Berg and Bescond, 2023). Industry rankings come from averaging the occupational numbers with an industry's occupational mix as weights (Felten, Raj and Seamans, 2021).

Measurement research has focused on the first step: which rubric, which rater, which model. Recent work shows that LLM-generated exposure scores are unstable across raters. Effects estimated with them vary in size, and sometimes in sign, depending on the annotating model (Henseke et al., 2025; Yin, Vu and Persico, 2026). The second step, turning task scores into statements about *jobs*, is usually treated as bookkeeping, even though it embeds substantive economic assumptions. Averaging task exposure assumes that tasks are separable and that freed-up hours become lost jobs one for one. Thresholds assume that jobs are destroyed whole once enough of their tasks can be done by machine. Neither rule allows for demand to respond when costs fall, or for the remaining "bottleneck" tasks to become more valuable when complementary tasks are automated. Classic critiques made the analogous point for the earlier automation literature. Arntz, Gregory and Zierahn (2016) showed that moving from occupation-level to task-level assessment cut the share of "automatable" jobs far below the occupation-level estimates of Frey and Osborne (2017), to about 9% across 21 OECD countries. Theoretical work on task-based technical change (Acemoglu and Restrepo, 2018; Acemoglu, 2025), on O-ring complementarity in automation (Gans and Goldfarb, 2026), on expertise (Autor and Thompson, 2025) and on demand elasticity (Bessen, 2019) shows that the mapping from task exposure to employment is not mechanical. What has been missing is a quantitative accounting, on the actual data behind the headline numbers, of *how much* the aggregation choice matters compared with the measurement choices the literature argues about, together with criteria for which conclusions survive *any* reasonable aggregation.

**This paper.** We take the most widely used public task-level dataset, the GPT-4 and human exposure labels of Eloundou et al. (2024), which come with a separate automation-intensity label, and push it through a transparent multiverse (Steegen et al., 2016; Simonsohn, Simmons and Nelson, 2020) of exposure measures, task weightings and aggregation rules. We then ask three questions. How much do headline numbers move, and which choice moves them? Which industry rankings are robust, and is there a criterion that predicts robustness before any model is fitted? Can currently available occupational employment data tell the aggregation rules apart?

**Contributions.**

1. **A variance accounting of AI-displacement estimates (empirical).** Across 1,680 specifications built on real task labels, O\*NET weights and May-2024 OEWS employment, the aggregation rule explains 84% of the variance in the headline net displacement share, 78% of gross displacement and 72% of the share of employment in "majority-displaced" occupations. The exposure measure explains 4–16%. The GPT-4 versus human rater choice explains 0.1–0.4%, and task weighting explains essentially nothing (§5.1).
2. **A dominance criterion for robust industry rankings (theory plus test).** We state and verify numerically that a pairwise industry ordering is invariant to every aggregation rule that is non-decreasing in the exposed share if and only if the two industries' employment-weighted exposure distributions are ordered by first-order stochastic dominance (FOSD). The ordering reverses entirely under rules that are decreasing in exposure. Only 15–54% of the 190 sector pairs satisfy FOSD, depending on the measure (27% for our preferred measure). Rules that are not functions of the exposed share alone, such as O-ring or automation/augmentation splits, reverse at least one ordering in 42–99% of pairs (§3.1, §5.2).
3. **Two closed-form results on the economic content of exposure indices (theory).** In a CES task model with demand response, the linear exposure index equals the exact employment loss if and only if the elasticity of substitution across tasks equals the price elasticity of demand (σ = ε), whatever the size of the cost reduction. Under pure automation, employment in an exposed occupation rises if and only if ε exceeds an *employment-neutral elasticity* ε\*(s, σ, θ) ≥ max(1, σ), which we tabulate (§3.2–3.3, §5.3).
4. **An identification audit (simulation on real noise).** Using panel-disjoint OEWS estimates for May 2021 and May 2024, no aggregation rule explains more than 2% of the within-major-group variance in occupational employment growth. In a Monte Carlo that resamples the actual residuals, even a correctly specified linear rule is detected with only 22% size-adjusted power when 20% of its implied long-run displacement has already happened (§5.4).
5. **Falsifiable predictions and a pre-registrable test protocol** for discriminating aggregation rules with the microdata that are becoming available, such as payroll panels and task-level usage logs (§6).

We do not offer a new forecast of how many jobs AI will replace. Our claim is narrower and, we think, more useful. On current public data, the answer to that question depends mostly on assumptions about how tasks combine into jobs and how demand responds, not on how exposure is measured. Some industry-level conclusions are nevertheless robust, and we show how to tell which.

## 2 Related Work

**Exposure measures.** Task-based approaches to technological change start with Autor, Levy and Murnane (2003). Frey and Osborne (2017) estimated the probability of computerisation for 702 occupations, and Arntz, Gregory and Zierahn (2016) showed that task-level heterogeneity within occupations cuts that estimate sharply. Brynjolfsson, Mitchell and Rock (2018) built a task-level "suitability for machine learning" score and concluded that few occupations are fully automatable while most contain some suitable tasks. Webb (2020) measured exposure through overlap between patent text and task text. Felten, Raj and Seamans (2021) built occupation, industry and geographic AI exposure indices (AIOE/AIIE/AIGE) from ability-application links. For generative AI, Eloundou et al. (2024) introduced the E0/E1/E2 rubric. In the preprint version they reported that about 80% of US workers have at least 10% of their tasks exposed and about 19% have at least half exposed. The published *Science* version reports that about 1.8% of jobs have over half their tasks affected by LLMs with simple interfaces, and just over 46% once complementary software is counted. That spread is itself a construct (α vs γ) effect of the kind we quantify. Gmyrek, Berg and Bescond (2023) applied a GPT-4-based approach globally and concluded that augmentation is more likely than automation. Usage-based measures map real LLM conversations to O\*NET tasks (Handa et al., 2025; Tomlinson et al., 2025).

**Measurement uncertainty.** Henseke et al. (2025) carry disagreement among nine LLM raters into labour-market estimates. Yin, Vu and Persico (2026) replicate exposure scoring with four frontier models and find that estimated effects change in magnitude and even sign depending on the rater. Our work complements theirs. We hold the rater fixed (or vary it as one factor) and show that rater choice is second-order compared with aggregation.

**Theory of task aggregation.** Acemoglu and Restrepo (2018) formalise automation as capital taking over tasks, offset by the creation of new tasks. Acemoglu (2025) uses task-level cost savings and Hulten's theorem to bound the macroeconomic effects of AI. Kremer's (1993) O-ring production function is the canonical model of strong task complementarity. Gans and Goldfarb (2026) apply it to automation. They argue that linear exposure indices overstate displacement when tasks are complements, and that labour income can rise under partial automation because the remaining bottleneck tasks gain value. Autor and Thompson (2025) show that the employment and wage effects of automation depend on whether the automated tasks are the expert or the inexpert ones. Bessen (2019) documents that the employment response of automating industries depends on demand elasticity. Our CES block is deliberately minimal; it nests the linear index as a special case and makes the role of (σ, ε) explicit. Our contribution is to take these mechanisms to the actual exposure data and measure their quantitative weight next to measurement choices.

**Early outcome evidence.** Using ADP payroll data, Brynjolfsson, Chandar and Chen (2025) document relative employment declines for workers aged 22–25 in the most exposed occupations, concentrated where AI use is automative rather than augmentative, with no evidence of economy-wide displacement. Humlum and Vestergaard (2025) find precise null effects of AI chatbots on earnings and hours in Danish administrative data. The Yale Budget Lab (Gimbel et al., 2025) finds no relation between exposure measures and changes in occupational employment or unemployment so far. Our identification audit (§5.4) explains why aggregate occupational data are not yet informative about the *shape* of the task-to-job mapping, even if the effects are real.

**Robustness methodology.** We borrow the multiverse (Steegen et al., 2016) and specification-curve (Simonsohn, Simmons and Nelson, 2020) ideas and add a structural criterion, stochastic dominance (Shaked and Shanthikumar, 2007), that predicts which conclusions survive.

## 3 Model

Index occupations by j and tasks by i ∈ 𝒯_j. Each task has a weight w_ij ≥ 0 with Σ_i w_ij = 1, interpreted as its share of the occupation's labour cost or time, and an exposure score e_ij ∈ [0, 1]. The **exposed share** is s_j = Σ_i w_ij e_ij. An *aggregation rule* is a map from the task vector (w_j, e_j) to an occupational displacement share d_j ≤ 1, where d_j > 0 means employment falls and d_j < 0 means it rises. Industry k's displacement is D_k = Σ_j π_jk d_j, where π_jk is occupation j's share of industry k's employment. The economy-wide headline is H = Σ_j π_j d_j.

### 3.1 Rules that depend only on s, and the dominance criterion

Many rules used in practice are of the form d_j = f(s_j). The linear rule is f(s) = s. The threshold rule is f(s) = 1[s ≥ τ], the "share of workers with at least half their tasks exposed" when τ = 0.5. Let G_k be the employment-weighted distribution of s across industry k's occupations. Then D_k = ∫ f dG_k.

**Proposition 1 (dominance criterion).** For industries k and l, D_k ≥ D_l for *every* non-decreasing f if and only if G_k first-order stochastically dominates G_l. If neither distribution dominates the other, there exist thresholds τ, τ′ with 1[s ≥ τ] ranking k above l and 1[s ≥ τ′] ranking l above k. For every non-increasing f the ordering under dominance reverses.

*Proof.* This is the standard characterisation of the usual stochastic order (Shaked and Shanthikumar, 2007, Ch. 1). Sufficiency follows because any bounded non-decreasing f is a mixture of threshold indicators plus a constant, and ∫1[s ≥ τ]dG_k = 1 − G_k(τ⁻). For necessity, if G_k(τ) > G_l(τ) at some τ and G_k(τ′) < G_l(τ′) at some τ′, the two indicator rules give opposite orderings. The last claim follows by applying the argument to −f. ∎

The proposition gives an *ex-ante* robustness test that needs no assumption about f: compute the employment-weighted CDFs of s by industry and check dominance. It also implies that rankings by the *mean* exposure, which is the linear rule, carry no information about robustness on their own.

### 3.2 A CES task model with demand response

To give f economic content, consider an occupation that produces output Y with a CES task aggregator, elasticity of substitution σ > 0 across tasks, and unit labour cost normalised to one. A share θ ∈ (0, 1) of the cost of exposed tasks is saved. Exposed cost share s splits into an *automated* part a, where labour is replaced by AI at cost 1 − θ, and an *augmented* part g = s − a, where labour stays but becomes 1/(1 − θ) times as productive. Output demand has constant price elasticity ε, so Y ∝ P^(−ε). Standard CES algebra gives the unit cost

  P^(1−σ) = (1 − s) + s(1 − θ)^(1−σ)  (with P = (1 − θ)^s when σ = 1),

and the employment ratio

  **L′/L = [(1 − s) + g(1 − θ)^(1−σ)] · P^(σ−ε).**  (1)

Labour demand on unexposed tasks is (1 − s)P^σ Y. On augmented tasks the labour input per unit of task output is (1 − θ), which gives g(1 − θ)^(1−σ)P^σ Y. Multiplying by Y = P^(−ε) yields (1). Two polar cases are pure automation (g = 0), with L′/L = (1 − s)P^(σ−ε), and pure augmentation (a = 0), with L′/L = P^(1−ε). The implied displacement is d = 1 − L′/L. This is a partial-equilibrium comparative static within the occupation. It ignores reallocation across occupations, new tasks, and general-equilibrium income effects, all of which we discuss in §7.

**Proposition 2 (the linear index is a knife-edge).** Under pure automation, d = s for all s ∈ [0, 1) if and only if σ = ε, for any θ ∈ (0, 1). If σ > ε then d > s, and if σ < ε then d < s, for every s ∈ (0, 1).

*Proof.* With g = 0, 1 − d = (1 − s)P^(σ−ε). For s > 0 and θ > 0, P < 1, so P^(σ−ε) = 1 if and only if σ = ε, and P^(σ−ε) is less than 1 when σ > ε and greater than 1 when σ < ε. ∎

We also checked numerically that the code reproduces d = s to machine precision (maximum absolute error 5.6 × 10⁻¹⁷; `results/extra_stats.json`). The linear exposure share is therefore not a neutral descriptive statistic. It is the employment prediction of a specific economic model, the one in which the gains from substituting toward cheaper tasks exactly offset the output expansion induced by lower prices.

**Proposition 3 (employment-neutral demand elasticity).** Under pure automation with s ∈ (0, 1) and θ ∈ (0, 1), employment rises (L′ > L) if and only if ε > ε\*(s, σ, θ) ≡ σ + ln(1 − s)/ln P, and ε\* ≥ max(1, σ). As θ → 1, ε\* → 1 for σ < 1 and ε\* → σ for σ > 1.

*Proof.* ln(L′/L) = ln(1 − s) + (σ − ε) ln P is positive if and only if (ε − σ)(−ln P) > −ln(1 − s), that is, ε > σ + ln(1 − s)/ln P. Both logarithms are negative, so ε\* > σ. For σ < 1, ln P = ln Φ/(1 − σ) with Φ = (1 − s) + s(1 − θ)^(1−σ) ≥ 1 − s, so ln(1 − s)/ln P = (1 − σ)·ln(1 − s)/ln Φ ≥ 1 − σ and ε\* ≥ 1. For σ ≥ 1, ε\* > σ ≥ 1. For the limits: if σ < 1, Φ → 1 − s as θ → 1, so ε\* → σ + (1 − σ) = 1; if σ > 1, Φ → ∞ and ln P → −∞, so ε\* → σ. ∎

Under pure augmentation, (1) gives employment rising if and only if ε > 1. The *sign* of the employment effect of exposure is therefore an empirical question about demand elasticities, which exposure data say nothing about. This is Bessen's (2019) argument in closed form, applied to task exposure.

### 3.3 Rules that use the within-occupation distribution: O-ring complementarity

In an O-ring technology (Kremer, 1993), an occupation's job disappears only if *all* of its core tasks can be done without the worker. If e_i is the probability that core task i is automatable, P(all core automated) = Π e_i under independence. Without assumptions on dependence, the Fréchet bounds give max(0, Σ_i e_i − (n − 1)) ≤ P(all) ≤ min_i e_i. These rules depend on the whole vector e_j, not just on s_j, so Proposition 1 does not protect rankings under them. Gans and Goldfarb (2026) develop the economics of this case. Here we use its displacement implication only as a polar alternative to the linear rule.

## 4 Data and Multiverse Design

**Task exposure.** The public file `full_labelset.tsv` from Eloundou et al. (2024; github.com/openai/GPTs-are-GPTs) contains 19,265 task statements for 923 O\*NET-SOC occupations. Each task has a GPT-4 exposure label and an aggregated human exposure label (E0 none, E1 direct LLM exposure, E2 exposure via LLM-powered software), plus a GPT-4 automation-intensity label T0–T4. Following the original paper, we form α = 1[E1], β = 1[E1] + 0.5·1[E2] and γ = 1[E1 ∨ E2], separately for the GPT-4 and human labels (six measures), and add the T-scale mapped to {0, 0.25, …, 1} as a seventh measure ("gpt_auto"). For the label-based CES mode we treat an exposed task as *automated* if its T-label is T2 or above and as *augmented* otherwise.

**Task weights.** O\*NET 30.0 Task Ratings supply importance (IM, 1–5) and relevance (RT, % of incumbents). Our four weighting schemes are equal weights, core-tasks-only, IM, and IM × RT. IM ratings matched 92.9% of labelled tasks; the remainder take the occupation median. Weights are normalised within O\*NET occupations, and O\*NET occupations are equally weighted within the 6-digit SOC codes used by OEWS.

**Employment.** BLS OEWS national estimates for May 2021–May 2024, detailed occupations, and the May-2024 national file by 2-digit NAICS sector (20 sectors). Labelled tasks map to 779 of the 831 detailed OEWS occupations, covering 96.8% of their 154.2 million jobs (`results/prep_info.csv`). In the sector file, 7.5% of occupation-sector cells have suppressed employment and are dropped. bls.gov and onetcenter.org were not reachable from our compute environment, so all files were obtained from public GitHub mirrors (listed in `README.md`). The 2024 national file reports 154,187,380 total jobs.

**Rules.** We use 60 rules: linear (1); thresholds τ ∈ {0.3, 0.5, 0.7} (3); O-ring with independence and the Fréchet lower bound on core tasks (2); and the CES model (1) in three modes (pure automation, pure augmentation, label-based split) × σ ∈ {0.5, 1, 2} × ε ∈ {0.5, 1, 1.5} × θ ∈ {0.5, 0.9} (54). The grid for θ reflects the rubric's definition of exposure as at least halving task time. The grids for σ and ε bracket unity because Propositions 2–3 show that unity is where the qualitative behaviour changes. We do not claim the grid is a calibrated prior. §7 discusses what that implies for reading the variance shares.

**Preferred specification.** Where we report a single case, we use human β with IM weights. It is the human analogue of the measure most often quoted from Eloundou et al. Results with GPT-4 β are similar (Tables in `results/`).

All numbers below come from scripts in `code/`, with outputs in `results/`. Where a number comes from the literature, we cite it.

## 5 Results

### 5.1 What moves the headline number?

Figure 1 (`results/fig1_spec_curve.png`) shows the distribution of the employment-weighted net displacement share H across the 1,680 specifications. The median is 17.7%, the 5th–95th percentile range runs from −20.5% to 63.1%, and the extremes are −132% and +87%. 14.8% of specifications imply net employment *gains* in exposed occupations (`results/analysis_summary.json`). Very large negative values arise in augmentation specifications with ε > 1 and σ = 2. They are mechanical consequences of the model at large cost reductions and should be read as describing direction, not as forecasts.

**Table 1. Share of variance in headline estimates explained by each design factor (main-effect η²; full-factorial design, n = 1,680).** Source: `results/analysis_summary.json`.

| Headline | Aggregation rule | Exposure measure | (of which: construct α/β/γ/T) | (of which: GPT-4 vs human rater) | Task weighting | Interactions |
|---|---|---|---|---|---|---|
| Net displacement H_net | **0.843** | 0.044 | 0.043 | 0.001 | <0.00001 | 0.112 |
| Gross displacement H_gross | **0.783** | 0.141 | 0.136 | 0.004 | <0.0001 | 0.076 |
| Share of employment with d ≥ 0.5 | **0.721** | 0.160 | 0.155 | 0.004 | 0.0001 | 0.119 |

Within the 1,512 CES specifications, the automation/augmentation mode accounts for 32.8% of the variance in H_net, the demand elasticity ε for 22.5%, σ for 5.2%, the exposure measure for 3.4% and θ for 0.1%; interactions account for the remaining 35.9%. Almost all of the measure effect comes from the *construct* (α vs β vs γ, that is, whether LLM-powered software counts), not from who did the rating. GPT-4 and human labels give nearly the same aggregate numbers once the construct is fixed. Task weighting is irrelevant at this level: the four schemes move the linear headline by at most 0.7 percentage points for any measure, and any rule's headline by at most 5.7 points (`results/multiverse_headline.csv`).

**Table 2. Economy-wide net displacement for the preferred measure (human β, IM weights) under selected rules.** Source: `results/analysis_summary.json`.

| Rule | H_net | Share of employment with d ≥ 0.5 |
|---|---|---|
| Linear exposure share | 30.7% | 21.6% |
| Threshold τ = 0.3 / 0.5 / 0.7 | 53.2% / 21.6% / 0.9% | same |
| O-ring, independence / Fréchet lower bound | 0.35% / 0.05% | 0.07% / 0.07% |
| CES automation, σ = ε = 0.5 (knife-edge, Prop. 2) | 30.7% | 21.8% |
| CES automation, σ = 0.5, ε = 1.5, θ = 0.9 | −9.5% (gain) | 0% |
| CES augmentation, σ = 0.5, ε = 0.5, θ = 0.9 | 21.0% | 0.7% |
| CES augmentation, σ = 0.5, ε = 1.5, θ = 0.9 | −30.6% (gain) | 0% |
| CES label-based, σ = 0.5, ε = 1, θ = 0.5 | 18.7% | 4.5% |
| CES label-based, σ = 2, ε = 0.5, θ = 0.9 | 67.7% | 75.9% |

Holding the measure fixed, the 60 rules span −88% to +80% (`fixed_measure_range_H_net`). The two most-cited summary formats, the hours-weighted mean (30.7%) and the share of workers with at least half their tasks exposed (21.6%), are just two points on this range. They correspond to the linear rule and to τ = 0.5. Our 21.6% for human β and employment weights from May 2024 is close to, though not identical to, the "≈19% of workers" in the Eloundou et al. preprint; the data vintage and measure differ. The O-ring rule gives almost no full-job displacement: under human β, 95.6% of occupations have at least one core task with zero exposure (`results/extra_stats.json`). Under that rule, exposure is mainly a story about *task reallocation within jobs*, not job loss, in line with the qualitative conclusion of Brynjolfsson, Mitchell and Rock (2018).

### 5.2 Which industries are hit hardest, and how robust is the ranking?

**Table 3. Sector displacement shares and ranks (1 = largest loss) for the preferred measure.** Source: `results/sector_table_preferred.csv`.

| NAICS sector | Linear d (rank) | Threshold 0.5 (rank) | O-ring (rank) | CES automate σ = .5, ε = 1.5 (rank) | CES label σ = .5, ε = 1 (rank) |
|---|---|---|---|---|---|
| Finance and Insurance | 52.0% (1) | 55.1% (1) | 1.6% (1) | −14.3% (20) | 23.9% (1) |
| Management of Companies | 47.7% (2) | 51.0% (3) | 0.7% (3) | −13.6% (19) | 19.9% (2) |
| Professional, Scientific & Technical | 47.4% (3) | 46.6% (4) | 0.3% (9) | −13.4% (18) | 19.4% (4) |
| Information | 47.3% (4) | 51.8% (2) | 0.5% (6) | −13.1% (17) | 19.5% (3) |
| Wholesale Trade | 36.9% (5) | 38.7% (5) | 1.2% (2) | −10.2% (13) | 13.2% (6) |
| Educational Services | 35.7% (6) | 20.5% (11) | 0.1% (18) | −11.3% (16) | 11.2% (8) |
| Retail Trade | 32.0% (9) | 9.9% (18) | 0.4% (8) | −10.5% (14) | 10.9% (10) |
| Administrative & Support | 26.4% (14) | 24.1% (8) | 0.6% (5) | −7.8% (7) | 10.0% (11) |
| Health Care & Social Assistance | 28.4% (12) | 14.9% (14) | 0.1% (15) | −9.2% (11) | 8.1% (14) |
| Construction | 19.1% (18) | 11.5% (16) | 0.1% (16) | −6.2% (3) | 6.4% (17) |
| Accommodation & Food Services | 17.2% (19) | 2.3% (20) | 0.0% (20) | −6.1% (2) | 4.1% (19) |
| Agriculture, Forestry, Fishing | 11.9% (20) | 5.4% (19) | 0.1% (17) | −4.0% (1) | 2.9% (20) |

(The other 8 sectors are in the CSV. The CES columns use θ = 0.9.)

Three findings stand out.

*(a) The top of the ranking is fairly stable as long as the direction is fixed.* Among the 1,508 specifications in which sectors differ at all (172 degenerate specifications give every sector the same value, such as all zero), Finance and Insurance ranks in the top three in 80% of them. Professional, Scientific and Technical Services and Management of Companies each do so in 67–71%. Information has a median rank of 4. Accommodation and Food Services and Agriculture are in the bottom three in 69–73% (`results/sector_rank_ranges.csv`; Fig. 3). Concordance across specifications is high within the linear (Kendall's W = 0.91), threshold (0.79), CES automation (0.75) and label-based (0.73) families and lower for O-ring (0.65). It is essentially zero within the augmentation family (0.002), because that family splits evenly between rules that are increasing and decreasing in exposure.

*(b) The direction is not robust.* Under pure automation with ε = 1.5 and σ = 0.5 the ordering is almost exactly reversed: finance becomes the sector with the largest employment *gain*. Within the augmentation family, 50% of specifications order sectors negatively relative to the linear rule (Kendall τ < 0); the figures are 4% for automation and 5% for label-based (`share_specs_tau_negative_by_family`). By Proposition 3, which outcome applies depends on whether the relevant demand elasticities exceed ε\*. The exposure data cannot answer that.

*(c) The middle of the ranking is fragile, and dominance predicts where.* For the preferred measure, only 27.4% of the 190 sector pairs are FOSD-ordered (range across the 28 measure-weight combinations: 14.7–53.7%; `results/fosd_pairs.csv`). As Proposition 1 requires, the share of pairs whose ordering is invariant across a dense grid of 199 thresholds equals the FOSD share exactly in all 28 combinations, which is a direct numerical check of the proposition. The coarse rule set that practitioners actually use (linear plus three thresholds) leaves 72% of pairs unchanged. That understates the fragility, because three thresholds do not span the class of monotone rules. Examples of non-robust pairs include Educational Services, which ranks 6th under the linear rule and 11th at τ = 0.5, and Retail Trade, which falls from 9th to 18th: these sectors have many moderately exposed workers but few highly exposed ones. Rules that are not functions of s alone reverse at least one ordering in 96% of pairs for the preferred measure (range 42–99%). Under O-ring aggregation, Wholesale Trade rises to 2nd, Utilities to 4th and Administrative & Support to 5th. These sectors contain occupations such as brokerage clerks and customer service representatives in which *every* core task is exposed (`results/top_occ_oring_hum_beta.csv`). Among occupations, the Spearman correlation between linear and O-ring displacement is only 0.30 for human β (0.51 for GPT-4 β), and the top-20 lists share 9 (14) occupations.

### 5.3 The employment-neutral elasticity

Figure 4 and `results/eps_star_table.csv` give ε\* for pure automation. At θ = 0.9 and σ = 0.5, ε\* rises from 1.24 at s = 0.1 to 1.33 at s = 0.5. At θ = 0.5 it is 2.27–2.69, and at σ = 2 it is above 2 everywhere. Employment in an exposed occupation therefore rises only when demand for its output is quite elastic, and more elastic still when cost savings are modest. For occupations whose output is an intermediate input, such as clerical work, ε is plausibly below 1, so the displacement reading of exposure applies. For occupations producing goods whose demand has been rationed by cost, such as some professional services, ε > ε\* cannot be ruled out. The model says *which elasticity to estimate*. The exposure data do not estimate it.

### 5.4 Can recent employment data discriminate between rules?

**Design.** OEWS estimates pool six semi-annual panels collected over three years. The May-2024 estimates combine the November 2021 to May 2024 panels (BLS, 2024), so the May-2021 and May-2024 estimates use disjoint panels, and 2021→2024 is the longest panel-disjoint window available in our data. We regress Δ ln employment (2021→2024) for 779 occupations on standardised exposure statistics, weighting by 2021 employment, with SOC major-group fixed effects and HC1 standard errors. These are descriptive regressions without a valid pre-period, and the 2021 base still reflects the pandemic recovery.

**Results** (`results/regressions.csv`). With major-group fixed effects, a 1-SD increase in human-β exposure is associated with +1.5% employment growth (SE 1.1%); GPT-4 β gives +0.6% (SE 1.2%). The minimum detectable effect at 80% power is about 3.2% per SD (`results/mde.json`). The augmentation share g has a positive coefficient (+1.3%, t = 2.25), but a similar coefficient appears in the overlapping 2021→2022 window, which is mostly before ChatGPT's release (+0.8%, t = 2.38). That points to pre-existing trends rather than an AI effect. The O-ring statistic is negative and significant in the overlapping 2023→2024 window (t = −2.9 for human β, −4.6 for GPT-4 β) but not over 2021→2024. We ran 72 such coefficients, the windows overlap, and nominal HC1 tests over-reject here (below). We therefore treat this as a lead to follow up, not a finding. In a horse race, no rule explains more than 1.9% of the within-major-group variance in 2021→2024 growth (`results/rule_horse_race.csv`). These nulls are consistent with Gimbel et al. (2025) and Humlum and Vestergaard (2025).

**Identification audit.** Are the nulls informative? We simulate worlds in which a true rule r\* holds and a fraction φ of its long-run displacement has already happened: y_j = −φ d\*_j + u_j, where u_j are bootstrap draws of the *actual* fixed-effect residuals of 2021→2024 growth. We then ask (i) whether r\* best fits among six candidate rules, and (ii) whether a test on r\*'s own prediction rejects, using critical values calibrated by simulation under φ = 0. The calibration matters: nominal |t| > 1.96 tests on these employment-weighted data reject 31–42% of the time under the null (`results/power_sim.csv`). Figure 5 shows the results. With φ = 0.2, the true linear rule is selected 30% of the time (chance is 17%) and detected with 22% power. Reaching 89% power requires φ = 0.5. O-ring displacement is essentially undetectable (power ≤ 10% even at φ = 0.5), because it concentrates on a handful of small occupations. Only rules with sharp, concentrated predictions are recoverable at φ ≈ 0.1–0.2: a threshold at 0.5 (59% power at φ = 0.1) and automation with σ = 2. Even before estimation, the linear rule and pure augmentation with σ = ε = 0.5 make predictions correlated above 0.99 after fixed effects (`results/candidate_rule_corr_full.csv`), so they cannot be told apart.

The implication is simple. Under the preferred linear specification, φ = 0.2 would mean roughly 0.2 × 30.7% ≈ 6% of all employment displaced within three years, far more than anyone has observed. At realistic early magnitudes, occupational employment totals cannot reveal the shape of the task-to-job mapping, so the current "no effect yet" findings cannot tell us which headline number is right.

## 6 Falsifiable Predictions and a Test Protocol

The framework yields predictions that distinguish aggregation rules. Each can be tested with data that now exist or are emerging: payroll panels such as ADP (Brynjolfsson, Chandar and Chen, 2025), usage logs mapped to O\*NET tasks (Handa et al., 2025; Tomlinson et al., 2025), and future OEWS waves.

**P1 (bottleneck protection; O-ring vs linear).** Compare occupations with equal exposed share s. Under the O-ring rule, those with a larger share of *unexposed core tasks* should lose fewer jobs (and, per Gans and Goldfarb, 2026, may see wage gains). Under the linear rule, employment does not depend on that share once s is held fixed. Test: the coefficient on the bottleneck share (`bott` in our code), conditional on s, should be positive for employment growth under O-ring and zero under linear.

**P2 (dispersion).** Under threshold rules with worker-level heterogeneity in task mix, employment losses are convex in s near the threshold. Holding s fixed, occupations with more dispersed within-occupation exposure (`sd`) should show *larger* losses below the threshold and *smaller* losses above it. A linear rule predicts no dispersion effect.

**P3 (demand-elasticity sign switch; Prop. 3).** Among highly exposed occupations, employment growth should be increasing in the product-demand elasticity of the industries that employ them. Ranking those industries by estimated ε should reverse the sign of the exposure-employment slope around ε\*. A linear rule predicts a uniformly negative slope.

**P4 (automation vs augmentation).** Under (1), automated exposure a predicts declines when ε < ε\*, and augmented exposure g predicts increases when ε > 1. The two slopes should differ in sign in low-elasticity industries and can share a sign in high-elasticity ones.

**P5 (industry rankings).** Sector pairs that are FOSD-ordered (`results/fosd_pairs.csv`) should keep their realised order of employment change wherever the overall direction is the same. Non-dominated pairs should be no better than coin flips at matching their predicted order.

**Pre-registrable protocol.** (i) Freeze the task-level exposure labels, the weights and the 60-rule set before outcome data are opened. (ii) Primary outcome: log employment growth by occupation × industry cell from a payroll panel, at the earliest horizon at which simulated power (our `power_sim.py`, with the panel's own residuals) exceeds 80% for the linear rule at φ = 0.1. (iii) Primary tests: P1 and P3, with major-group × industry fixed effects, wild-cluster bootstrap by industry, and critical values calibrated under a placebo exposure permutation. (iv) Placebo: the same specification on a pre-2022 window. (v) Report the full specification curve, not a selected rule. (vi) Declare the decision rule in advance. The linear rule is rejected in favour of the O-ring rule if the P1 coefficient is positive with simulation-calibrated p < 0.05 and the placebo is null.

## 7 Discussion

**What this means for the headline question.** The question "how many jobs will AI replace?" is usually answered with a single number derived from exposure. On the data behind such numbers, that number mostly reflects the analyst's implicit model of how tasks combine into jobs and how demand responds, which is the aggregation step. It reflects the exposure measurement far less. Many commentaries treat a figure like "about 30% of work is exposed" as an upper bound on displacement. Proposition 2 shows it is not one: when σ > ε, displacement exceeds exposure. It is also not a central estimate unless σ = ε. Public statements should report exposure as exposure, a property of tasks, and present displacement only together with the aggregation assumptions it relies on.

**What survives.** Three conclusions hold across our multiverse. (1) If exposure lowers employment anywhere, it lowers it most in finance and insurance, management of companies, professional and technical services and information, and least in agriculture, accommodation and food services, construction and mining. These orderings are largely FOSD-protected for the γ-type constructs and strongly concordant across monotone rules. (2) The ranking of the middle sectors, including retail, education, health care and administrative support, depends on aggregation, and Proposition 1 identifies which pairs to treat as indeterminate. (3) The *sign* of employment effects is not identified by exposure data. Answering "which industries will lose the most jobs" requires demand elasticities, and the most-exposed industries are exactly the ones for which the sign question matters most.

**Relation to rater-instability work.** Henseke et al. (2025) and Yin, Vu and Persico (2026) show that the measurement step is fragile across LLM raters. Our variance accounting suggests that, for aggregate counts, this fragility is dominated by the aggregation step. GPT-4 versus human labels explain 0.1–0.4% of headline variance once the construct is fixed. The two concerns are complementary: rater instability matters most for regression coefficients and individual occupations, aggregation for counts and rankings.

**Interpreting variance shares.** The η² shares depend on the grid of rules. A narrower grid would give aggregation a smaller share. The grid is not arbitrary, though. Each rule appears in the literature or is a one-parameter neighbour of one that does, and the CES grid brackets the unity thresholds that Propositions 2–3 identify as qualitative switch points. More importantly, the *range* results (−88% to +80% with the measure fixed) and the ranking results do not depend on weighting the grid.

## 8 Limitations

- **Partial equilibrium within occupations.** The CES block holds wages fixed and ignores movement of workers across occupations, the creation of new tasks (Acemoglu and Restrepo, 2018), changes in expertise requirements (Autor and Thompson, 2025) and economy-wide income effects. The d_j are comparative statics, not forecasts, and have no time dimension. The very large gains under some augmentation specifications reflect this.
- **The parameter grid is not calibrated.** We do not estimate σ, ε or θ for any occupation. Propositions 2–3 state where these parameters matter. Estimating them is left to future work.
- **One exposure dataset, one vintage.** All measures derive from the Eloundou et al. labels (GPT-4-era capabilities, O\*NET tasks as of their release). We did not use Felten et al., Webb, or usage-based measures (Handa et al., 2025; Tomlinson et al., 2025). The task-ID match to O\*NET 30.0 importance ratings is 92.9%; unmatched tasks take occupation medians.
- **Data obtained through mirrors.** OEWS and O\*NET files came from public GitHub mirrors because the official hosts were unreachable from our environment. We checked internal consistency (for example, the 2024 national total) but could not byte-compare against bls.gov.
- **Industry aggregation is occupational.** Sector displacement averages occupation-level d with sector employment weights. It ignores the fact that the same occupation's task mix and demand elasticity differ by industry. Suppressed cells (7.5%) are dropped.
- **Independence in O-ring rules.** The independence version of the O-ring rule assumes automatability is independent across tasks. With binary labels the independence value, the Fréchet bounds and the minimum coincide, but with β-type scores they do not.
- **The empirical analysis is descriptive.** OEWS is not designed for time comparisons. The 2021 base is affected by the pandemic, there is no clean pre-period, and occupation-level totals cannot capture within-occupation age composition (where Brynjolfsson, Chandar and Chen, 2025, find effects). The simulation assumes that bootstrap residuals represent the noise, and it treats displacement as a fraction φ of a static long-run target.
- **Equal weighting of O\*NET occupations within SOC codes** is a simplification.

## 9 Conclusion

Estimates of how many jobs AI will replace, and which industries it will hit hardest, are built from task-level exposure scores and an aggregation rule. On the most widely used public data, the aggregation rule accounts for about 84% of the variation in the headline net displacement share. Measurement choices account for a few percent, and the GPT-4 versus human rater choice for a tenth of a percent. The linear exposure share is the employment prediction of a knife-edge economic model. The sign of employment effects in exposed occupations depends on whether demand elasticities exceed a computable threshold ε\* ≥ max(1, σ). Industry rankings are robust exactly where employment-weighted exposure distributions are ordered by stochastic dominance. That holds at the top (finance, management, professional services, information) and the bottom (agriculture, accommodation and food), but not in much of the middle. Current occupational employment data are statistically unable to discriminate between aggregation rules at plausible early effect sizes. Progress on the question therefore needs data and designs that target the aggregation step directly. Our predictions P1–P5 and the pre-registrable protocol are meant to provide them.

## References

Acemoglu, D. (2025). The simple macroeconomics of AI. *Economic Policy*, 40(121), 13–58. https://doi.org/10.1093/epolic/eiae042

Acemoglu, D., & Restrepo, P. (2018). The race between man and machine: Implications of technology for growth, factor shares, and employment. *American Economic Review*, 108(6), 1488–1542. https://doi.org/10.1257/aer.20160696

Arntz, M., Gregory, T., & Zierahn, U. (2016). The risk of automation for jobs in OECD countries: A comparative analysis. *OECD Social, Employment and Migration Working Papers*, No. 189. https://doi.org/10.1787/5jlz9h56dvq7-en

Autor, D. H., Levy, F., & Murnane, R. J. (2003). The skill content of recent technological change: An empirical exploration. *Quarterly Journal of Economics*, 118(4), 1279–1333. https://doi.org/10.1162/003355303322552801

Autor, D., & Thompson, N. (2025). Expertise. NBER Working Paper No. 33941. https://www.nber.org/papers/w33941

Bessen, J. (2019). Automation and jobs: When technology boosts employment. *Economic Policy*, 34(100), 589–626.

Brynjolfsson, E., Chandar, B., & Chen, R. (2025). Canaries in the coal mine? Six facts about the recent employment effects of artificial intelligence. Working paper, Stanford Digital Economy Lab (revised 2026). https://digitaleconomy.stanford.edu/publication/canaries-in-the-coal-mine-six-facts-about-the-recent-employment-effects-of-artificial-intelligence/

Brynjolfsson, E., Mitchell, T., & Rock, D. (2018). What can machines learn, and what does it mean for occupations and the economy? *AEA Papers and Proceedings*, 108, 43–47. https://doi.org/10.1257/pandp.20181019

Bureau of Labor Statistics (2024). Occupational Employment and Wage Statistics, May 2024: Technical notes. https://www.bls.gov/oes/2024/may/oes_tec.htm

Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. *Science*, 384(6702), 1306–1308. https://doi.org/10.1126/science.adj0998 (preprint: arXiv:2303.10130; data: https://github.com/openai/GPTs-are-GPTs)

Felten, E., Raj, M., & Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence: A novel dataset and its potential uses. *Strategic Management Journal*, 42(12), 2195–2217. https://doi.org/10.1002/smj.3286

Frey, C. B., & Osborne, M. A. (2017). The future of employment: How susceptible are jobs to computerisation? *Technological Forecasting and Social Change*, 114, 254–280. https://doi.org/10.1016/j.techfore.2016.08.019

Gans, J. S., & Goldfarb, A. (2026). O-ring automation. NBER Working Paper No. 34639. https://doi.org/10.3386/w34639

Gimbel, M., Kinder, M., Kendall, J., & Lee, M. (2025). Evaluating the impact of AI on the labor market: Current state of affairs. The Budget Lab at Yale, October 1, 2025. https://budgetlab.yale.edu/research/evaluating-impact-ai-labor-market-current-state-affairs

Gmyrek, P., Berg, J., & Bescond, D. (2023). Generative AI and jobs: A global analysis of potential effects on job quantity and quality. ILO Working Paper 96. https://doi.org/10.54394/FHEM8239

Handa, K., Tamkin, A., McCain, M., et al. (2025). Which economic tasks are performed with AI? Evidence from millions of Claude conversations. arXiv:2503.04761.

Henseke, G., Davies, R., Felstead, A., Gallie, D., Green, F., & Zhou, Y. (2025). Nine raters, one index: Carrying LLM disagreement into labour-market estimates. arXiv:2507.22748.

Humlum, A., & Vestergaard, E. (2025). Large language models, small labor market effects. NBER Working Paper No. 33777 (later retitled "Still waters, rapid currents: Early labor market transformation under generative AI"). https://www.nber.org/papers/w33777

Kremer, M. (1993). The O-ring theory of economic development. *Quarterly Journal of Economics*, 108(3), 551–575. https://doi.org/10.2307/2118400

National Center for O\*NET Development. O\*NET 30.0 Database: Task Ratings. https://www.onetcenter.org/database.html

Shaked, M., & Shanthikumar, J. G. (2007). *Stochastic Orders*. Springer Series in Statistics. New York: Springer. https://doi.org/10.1007/978-0-387-34675-5

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4, 1208–1214. https://doi.org/10.1038/s41562-020-0912-z

Steegen, S., Tuerlinckx, F., Gelman, A., & Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*, 11(5), 702–712. https://doi.org/10.1177/1745691616658637

Tomlinson, K., Jaffe, S., Wang, W., Counts, S., & Suri, S. (2025). Working with AI: Measuring the occupational implications of generative AI. arXiv:2507.07935.

Webb, M. (2020). The impact of artificial intelligence on the labor market. Working paper, Stanford University. SSRN 3482150. https://papers.ssrn.com/abstract=3482150

Yin, M., Vu, H., & Persico, C. (2026). How (un)stable are LLM occupational exposure scores? Evidence from multi-model replication. NBER Working Paper No. 35110. https://www.nber.org/papers/w35110

---

### Appendix A: Reproducibility map (number → file)

| Quantity | File |
|---|---|
| Coverage, match rates | `results/prep_info.csv` |
| 1,680 headline estimates | `results/multiverse_headline.csv` |
| η² decomposition, ranges, Kendall's W, preferred-measure table | `results/analysis_summary.json` |
| Sector ranks and ranges | `results/sector_table_preferred.csv`, `results/sector_rank_ranges.csv`, `results/multiverse_sector.csv` |
| FOSD test (Prop. 1) | `results/fosd_pairs.csv` |
| ε\* table (Prop. 3) | `results/eps_star_table.csv` |
| Knife-edge check, occupation correlations, top lists | `results/extra_stats.json`, `results/top_occ_*.csv` |
| Regressions, horse race | `results/regressions.csv`, `results/rule_horse_race.csv` |
| Power simulation, MDE | `results/power_sim.csv`, `results/mde.json`, `results/candidate_rule_corr*.csv` |
| Figures 1–5 | `results/fig1_spec_curve.png` … `results/fig5_power.png` |
