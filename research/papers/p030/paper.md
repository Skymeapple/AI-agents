# From Success Rates to Hours Saved: A Delegation-Surplus Curve for Evaluating Whether an Agent Is Useful

**Target venue:** NeurIPS, Evaluations & Datasets (Datasets and Benchmarks) track

## Abstract

Agent benchmarks report a success rate, but a person deciding whether to hand work to an agent cares about something else: how much of their own labour the agent removes once they have paid for checking its output and for the failures they miss. We formalise this as a three-action delegation problem (do the task yourself, delegate and verify, delegate and accept blindly) and define usefulness as the share of human hours saved under the user's best policy. We prove that, for every deployment context in this family (verification cost, harm from an undetected failure, operating overhead), usefulness is an explicit function of one curve, the value-weighted stop-loss transform π(k) of the user's belief about success; this curve is a rescaled, value-weighted decision curve. One agent is at least as useful as another in every context if and only if its curve is pointwise higher, which is the increasing convex order on beliefs. Accuracy is the single point π(0), so it can never certify usefulness on its own, and run-to-run inconsistency has a closed-form price. We then estimate these quantities on the public METR time-horizon data (20 models, 228 tasks, 23,235 agent runs after excluding human rows). For the most accurate model, 84.2% task success corresponds to 54.4% of human hours completed and, in a reference context where checking costs 20% of doing, to between 29% and 35% of hours saved depending on what the user can predict; with perfect foresight it would be 52%. Accuracy orders the models well (Kendall τ about 0.9 against usefulness when users have task history) but is a poor cardinal guide: the same accuracy gain is worth about five times more hours near the top of the range than in the middle. When verification is expensive, usefulness depends almost entirely on predictability of success rather than on its rate. We give a reporting protocol and state what the analysis cannot show.

## 中文摘要

智能体基准通常只报告成功率，但决定是否把工作交给智能体的人关心的是另一件事：在付出核查成本、并承担未被发现的失败之后，智能体究竟替自己省下了多少劳动。本文把这一问题形式化为三选一的委托决策（自己做、委托并核查、委托后直接采纳），并把“有用性”定义为用户在最优策略下节省的人工工时占比。我们证明：在该族的任意部署情境下（核查成本、未发现失败的损害、运行开销），有用性都是同一条曲线的显式函数，即用户对成功概率的信念的价值加权止损变换 π(k)；这条曲线等价于按价值加权并重新缩放的决策曲线。一个智能体在所有情境下都不劣于另一个，当且仅当它的曲线逐点更高，即信念分布满足递增凸序。准确率只是这条曲线上的一个点 π(0)，因此单凭准确率无法保证有用性；同一任务多次运行结果不一致的代价也有闭式表达。随后我们在公开的 METR 时间跨度数据上（20 个模型、228 个任务、剔除人类记录后共 23,235 次运行）估计这些量。对准确率最高的模型，84.2% 的任务成功率对应 54.4% 的人工工时被完成；在核查成本为自己完成成本 20% 的参考情境下，依用户可预测程度不同，节省的工时为 29% 至 35%，而完全预知时为 52%。准确率对模型的排序基本可靠（用户掌握任务历史时与有用性的 Kendall τ 约为 0.9），但作为基数指标很差：同样的准确率提升，在高端所值的工时约为中段的五倍。当核查昂贵时，有用性几乎完全取决于成功是否可预测，而非成功率本身。我们给出一套报告规范，并说明本分析不能证明的内容。

## 1 Introduction

An agent that resolves 80% of the tasks in a benchmark is not thereby known to save anyone 80% of anything. Three well-documented observations make the gap concrete. In a randomised trial with 16 experienced open-source developers working on 246 real issues, allowing early-2025 AI tools made them 19% slower, although they believed they had been sped up (Becker et al., 2025; sourced). When maintainers reviewed 296 agent-written pull requests on SWE-bench Verified repositories, their merge decisions ran about 24 percentage points below the automated grader's pass rate (METR, 2026; sourced). And on a benchmark built from paid remote-work projects, the best agent's automation rate was 2.5% (Mazeika et al., 2025; sourced), at a time when coding benchmarks reported much higher success rates.

The standard responses are to make benchmarks more realistic, to add cost as a second axis (Kapoor et al., 2025), to measure reliability separately (Yao et al., 2025; Rabanser et al., 2026), or to run field experiments. All of these help. None of them answers a narrower question that a benchmark owner can act on today: *given per-task results that a benchmark already produces, what is the least one must compute and report so that a reader can work out usefulness for their own deployment context?* Accuracy plainly does not suffice, but what does, and how wrong is accuracy in practice?

This paper answers that question inside one explicit model. A user faces tasks, each of which would take them a known amount of time. For each task they can do it themselves, delegate and check the result (redoing the task if it failed), or delegate and accept the result unchecked (bearing a harm if it failed). Usefulness is the share of human hours the user saves under their best policy. It depends on the agent, on the deployment context (how costly checking is, how bad an undetected failure is, how much overhead delegation carries) and on what the user can predict about the agent's success before delegating.

**Contributions.**

1. *A sufficiency result.* In the three-action model, usefulness in every context is an explicit function of the value-weighted stop-loss curve π(k) of the user's belief about success (Proposition 1). The curve equals (1−k) times a value-weighted decision-curve net benefit (Vickers and Elkin, 2006), which connects agent evaluation to an established tool.
2. *A dominance characterisation.* Agent A is at least as useful as agent B in every context if and only if π_A ≥ π_B pointwise, i.e. the belief distributions are ranked in the increasing convex order (Theorem 1). Accuracy is π(0); for any two accuracies there are agent–information pairs whose usefulness ranking is the reverse of their accuracy ranking (Corollary 1).
3. *Bounds and a price for inconsistency.* π lies between a no-information floor and a perfect-foresight ceiling, rises with the user's information, and the part of the gap that no ex-ante information can close has the closed form E[min{p(1−k), k(1−p)}] in the per-task success probability p (Proposition 2).
4. *An estimator and an empirical decomposition on real data.* We estimate π honestly, as the held-out value of a delegation policy, on 23,235 public agent runs from the METR time-horizon suite, under four information structures, and decompose the path from headline accuracy to hours saved for 20 models, with cluster-bootstrap intervals, a weighting sensitivity analysis and a simulation of estimator bias.
5. *A reporting protocol* that a benchmark can adopt with data it already has, plus falsifiable predictions for the one information structure we could not test (the agent's own stated confidence).

We do not claim the model captures everything that makes an agent useful, and we do not measure deployment contexts; Section 8 is specific about both.

## 2 Related Work

**Benchmark validity for agents.** SWE-bench (Jimenez et al., 2024) established test-based scoring of agent patches; later work showed that test-passing overstates acceptability to maintainers (METR, 2025; METR, 2026). Kapoor et al. (2025) argued that accuracy-only leaderboards mislead and proposed joint reporting of cost and accuracy; the Holistic Agent Leaderboard (Kapoor et al., 2026) built infrastructure for it. Our contribution is complementary: cost enters our model as one parameter, and we ask what else must be reported.

**Reliability.** τ-bench introduced pass^k, the probability that all of k trials succeed (Yao et al., 2025). Rabanser et al. (2026) proposed twelve reliability metrics across consistency, robustness, predictability and safety and found reliability improving more slowly than capability. These works argue that reliability matters; Proposition 2 below says how much, in hours, within a decision model, and separates the part of unreliability that a user can route around from the part they cannot.

**Economic value and field evidence.** GDPval (Patwardhan et al., 2025) and the Remote Labor Index (Mazeika et al., 2025) score agents on economically valuable deliverables. Field studies measure realised productivity: a 15% gain in issues resolved per hour for customer-support workers (Brynjolfsson et al., 2025; sourced) and a 19% slowdown for experienced developers (Becker et al., 2025). The METR time-horizon work (Kwa et al., 2025) attaches a human completion time to each task, which is what makes our hours-saved analysis possible. Liu et al. (2025) argue in a position paper that the barrier to agent usability is "agentic ROI", a ratio of quality and time gains to cost that counts human verification time; they do not derive what must be measured to compute it across contexts, nor estimate it from run-level data. Huang et al. (2026) study delegation and verification choices theoretically, focusing on the misalignment between a worker's incentives and an institution's; our user is a single decision maker and our aim is measurement.

**Decision-theoretic evaluation of predictors.** Decision curve analysis evaluates a risk model by its net benefit across threshold probabilities (Vickers and Elkin, 2006). Selective classification (Geifman and El-Yaniv, 2017) and learning to defer (Madras et al., 2018; Mozannar and Sontag, 2020) study when a model should abstain or hand over to a human. The comparison of information structures by their value in all decision problems goes back to Blackwell (1953), and the increasing convex order is treated in Shaked and Shanthikumar (2007). The mathematics we use is therefore standard. What we add is the reduction of the agent-delegation problem to these objects, the resulting statement of what a benchmark must report, and the measurement on agent data.

## 3 Model

**Tasks and the agent.** A task i has a human cost t_i > 0, the time a qualified person needs to do it. An agent attempt on task i succeeds with probability p_i; Y_i ∈ {0,1} denotes the outcome of one attempt. Tasks carry value weights w_i > 0. The default is w_i = t_i, so that weighted averages are shares of human hours; Section 6.5 varies this.

**Deployment context.** A context θ = (ρ, h, c) has three non-negative parameters, all expressed as multiples of t_i:

- ρ, the *verification cost ratio*: checking the agent's output costs ρ·t_i. We assume checking reveals the outcome.
- h ≥ 1, the *harm multiple*: an undetected failure costs h·t_i (the work must still be done, plus downstream damage).
- c ∈ [0,1), the *overhead*: specifying the task, waiting, and paying for compute cost c·t_i whenever the user delegates.

**Actions.** For each task the user chooses one of:

- D, do it yourself: cost t_i.
- V, delegate and verify, redoing the task on failure: cost (c + ρ)t_i + (1 − Y_i)t_i.
- B, delegate and accept blindly: cost c·t_i + (1 − Y_i)h·t_i.

**Information.** Before choosing, the user observes information I (possibly nothing) and forms the belief q_i = P(Y_i = 1 | I). Expected savings relative to D, as a fraction of t_i, are 0 for D, q − ρ − c for V, and 1 − c − (1 − q)h for B. The best achievable fractional saving is

s_θ(q) = max{0, q − ρ − c, 1 − c − (1 − q)h},

which is non-decreasing, convex and piecewise linear in q.

**Definition (usefulness).** U_θ(A, I) = E[w·s_θ(q)] / E[w]. With w = t this is the share of human hours that agent A saves a user with information I in context θ. It lies in [0, 1 − c].

**Definition (delegation-surplus curve).** π_{A,I}(k) = E[w·(q − k)_+] / E[w] for k ∈ [0,1]. This is the stop-loss transform of the value-weighted distribution of beliefs. π(0) is the value-weighted accuracy a_w = E[wY]/E[w], π is convex and non-increasing, and π(1) = 0.

Two remarks. First, the headline accuracy a = E[Y] with equal task weights differs from a_w by the identity a_w − a = Cov(w, p)/E[w]: when agents fail more on long tasks, hours completed fall short of tasks completed. Second, π(k) is the value of the policy "delegate when q > k" in a problem where delegation costs k per unit of value. Writing TP and FP for the weighted shares of delegated tasks that succeed and fail, π(k) = TP − k(TP + FP) = (1 − k)·[TP − FP·k/(1 − k)], which is (1 − k) times the net benefit of decision curve analysis at threshold k.

## 4 Theory

**Proposition 1 (sufficiency of the curve).** Fix an agent and information structure with curve π, and a context θ = (ρ, h, c). Let k₁ = ρ + c.

- If h = 1, then U_θ = π(c).
- If h > 1, let k₂ = 1 − ρ/(h − 1). If k₂ ≥ k₁ then U_θ = π(k₁) + (h − 1)·π(k₂). Otherwise U_θ = h·π(1 − (1 − c)/h).

*Proof.* If h = 1 the blind branch equals q − c ≥ q − ρ − c, so s_θ(q) = (q − c)_+. If h > 1, the verify line q − k₁ and the blind line 1 − c − (1 − q)h cross at q = k₂, with the blind line steeper. When k₂ ≥ k₁ the verify line becomes positive first and is overtaken at k₂, so s_θ(q) = (q − k₁)_+ + (h − 1)(q − k₂)_+; checking q > k₂ gives q − ρ − c + (h − 1)q − (h − 1) + ρ = 1 − c − (1 − q)h. When k₂ < k₁ the blind line is above the verify line wherever the latter is positive, so s_θ(q) = (1 − c − (1 − q)h)_+ = h·(q − (1 − (1 − c)/h))_+. Taking weighted expectations gives the claim. ∎

The proposition says that no context in this family needs more than the curve. It also gives a reading of the two kinks: k₁ is the belief above which delegation is worth the checking, and k₂ is the belief above which checking is no longer worth it. The identity holds exactly for the held-out policy values we estimate in Section 5, because it is an identity between payoffs action by action; on the data the largest absolute discrepancy across 4,320 model–information–context cells is 2.8×10⁻¹⁶ (computed; `results/usefulness.json`).

**Theorem 1 (usefulness dominance).** Let A and B be agents, each paired with an information structure, with curves π_A, π_B and value-weighted belief distributions F_A, F_B. The following are equivalent.

1. U_θ(A) ≥ U_θ(B) for every context θ with ρ ≥ 0, h ≥ 1, c ∈ [0,1).
2. π_A(k) ≥ π_B(k) for every k ∈ [0,1).
3. F_A dominates F_B in the increasing convex order.

*Proof.* (2)⇒(1): by Proposition 1, U_θ is a non-negative combination of values of π at points that are either in [0,1) or at which both curves vanish. (1)⇒(2): the context (ρ, h, c) = (0, 1, k) has U_θ = π(k). (2)⇔(3) is the characterisation of the increasing convex order by stop-loss transforms (Shaked and Shanthikumar, 2007, Section 4.A). ∎

**Corollary 1 (accuracy cannot certify usefulness).** Accuracy fixes only π(0). For any 0 < b < a < 1, let A have value-weighted accuracy a and a user with no information, and B have accuracy b and a user with perfect foresight. Then π_A(k) = (a − k)_+ and π_B(k) = b(1 − k), so π_B(k) > π_A(k) for all k > (a − b)/(1 − b). In every context with h = 1 and c above that threshold, and in every context with h > 1, k₂ ≥ k₁ and k₁ above it, the less accurate agent is strictly more useful.

The corollary uses extreme information structures to make the point cleanly; Section 6 asks how much of this possibility is realised among actual models.

**Proposition 2 (bounds, information, and the price of inconsistency).** For any information structure I and k ∈ [0,1]:

(a) (a_w − k)_+ ≤ π_I(k) ≤ a_w(1 − k). The lower bound is attained with no information and the upper with perfect foresight of each run's outcome.

(b) If I′ refines I then π_{I′}(k) ≥ π_I(k).

(c) If I is available before the run and the runs of a task are exchangeable given the task, then π_I(k) ≤ π_task(k) := E[w(p − k)_+]/E[w], and the shortfall from the ceiling is

a_w(1 − k) − π_task(k) = E[w·min{p(1 − k), k(1 − p)}] / E[w],

which is zero if and only if (weighted-almost) every task has p ∈ {0,1}, for k ∈ (0,1).

*Proof.* (a) and (b): x ↦ (x − k)_+ is convex, so Jensen's inequality applied to q = E[Y | I] and to q = E[q′ | I] gives the lower bound and monotonicity; the upper bound follows because a convex function on [0,1] lies below its chord, (q − k)_+ ≤ q(1 − k). (c): under exchangeability any pre-run belief is a conditional expectation of p, so (b) applies with I′ = task identity; and p(1 − k) − (p − k)_+ equals p(1 − k) when p ≤ k and k(1 − p) when p > k. ∎

Part (c) puts a price on what pass^k-style metrics detect. An agent that succeeds on a fixed 70% of tasks every time and one that succeeds on every task 70% of the time have the same accuracy; a user can learn to route around the first but must verify the second, and the difference is the displayed expectation. Only verification, an ex-post instrument paid for through ρ, recovers it.

## 5 Data and Estimation

**Data (public, analysed here).** We use the run-level file released with METR's time-horizon analysis, report `time-horizon-1-1`, at commit `52cb829` of `METR/eval-analysis-public` (Kwa et al., 2025). After dropping 773 human rows it contains 23,235 agent runs from 20 model configurations on 228 tasks in 79 task families; each run has a binary score and each task a human completion time. Tasks come from three sources: 157 HCAST tasks, 5 RE-Bench tasks and 66 short SWAA tasks. Each model has a median of 4 to 8 runs per task. Four models lack a few tasks (213 to 226 present). We use METR's binarised score as given, including runs it scored as failures after fatal errors. The file's generation-cost field is zero for 18 of the 20 models, so we treat overhead c as a context parameter and do not estimate dollar costs.

The value weighting is very concentrated (computed; `time_weight` in `results/usefulness.json`). The 228 tasks total 541 human hours with a median of 12.6 minutes; the 50 tasks longer than four hours carry 85.5% of the hours, the ten longest carry 31.4%, and the 66 SWAA tasks carry 0.02%. The effective number of tasks under time weighting, (Σt)²/Σt², is 52.6.

**Information structures.** We estimate four.

- *None.* The user knows only the model's value-weighted accuracy; q = a_w for every task. This is a user who has read a leaderboard.
- *Length.* The user knows how long the task would take them. q is a logistic function of log₂ t fitted per model by binomial regression and cross-fitted: tasks are split into five folds by task family, and each task's q comes from a fit that excluded its family.
- *Task history.* The user has previously seen the agent attempt this task. For each of 20 random half-splits of each task's runs, q is the success frequency in one half shrunk towards the length prediction with pseudo-count 2, and the policy is evaluated on the other half; both directions are averaged. Tasks with a single run fall back to the length belief.
- *Oracle.* The user knows each run's outcome in advance. This is the ceiling a_w(1 − c) of Proposition 2(a), and equals what verification would deliver if it were free.

**Estimator.** We never plug estimated beliefs into s_θ. Instead we evaluate the policy that acts optimally *given* the belief against held-out outcomes: with ȳ the held-out success frequency, the realised fractional saving is 0, ȳ − ρ − c or 1 − c − (1 − ȳ)h according to the action chosen, and π̂(k) = Σw·1[q > k](ȳ − k)/Σw. This is the value a user would actually obtain from that belief, so miscalibrated beliefs are penalised and estimates can be negative. The headline accuracy a is the equal-task-weight mean of per-task success frequencies (this differs from METR's own diversity weighting).

**Uncertainty.** We resample the 79 task families with replacement 300 times and recompute everything, including the cross-fits; intervals are 2.5th to 97.5th percentiles.

**Contexts.** The reference context is θ₀ = (ρ, h, c) = (0.2, 5, 0.05): checking costs a fifth of doing, a missed failure costs five times the task, overhead is 5%. These are assumptions chosen to be moderate, not measurements. For rank analyses we use the 72-context grid ρ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} × h ∈ {1, 2, 5, 20} × c ∈ {0, 0.05, 0.2}.

**Estimator bias (simulation; synthetic data).** `code/sim_estimator.py` draws 228 tasks with p ~ Beta(0.5, 0.5) and n runs each and compares estimators of π_task(k) with the truth over 400 replications. The naive plug-in E[(p̂ − k)_+] is biased upward by 0.014 to 0.016 with n = 4 and by 0.005 to 0.006 with n = 12; the split-half policy value is biased downward by 0.034 to 0.037 with n = 4, 0.015 to 0.019 with n = 6 and about 0.012 with n = 12 (`results/sim_estimator.json`). Our task-history figures are therefore conservative: they are the value of a user who has seen only two or three attempts.

## 6 Results

All numbers in this section are computed by `code/usefulness.py` from the public data and stored in `results/`.

### 6.1 From accuracy to hours saved

Table 1 gives the decomposition at θ₀ with time weighting for a subset of models; `results/table_main.csv` has all 20 with intervals.

**Table 1.** Headline accuracy a, time-weighted accuracy a_w, and usefulness U at θ₀ under four information structures. Brackets are 95% cluster-bootstrap intervals.

| Model | a | a_w | U none | U length | U task history | U oracle |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 0.842 [0.764, 0.900] | 0.544 [0.415, 0.690] | 0.294 | 0.291 [0.129, 0.440] | 0.350 [0.267, 0.457] | 0.517 |
| GPT-5.2 | 0.826 | 0.457 | 0.207 | 0.187 [0.043, 0.357] | 0.278 [0.202, 0.388] | 0.434 |
| GPT-5.3-Codex | 0.816 | 0.436 | 0.186 | 0.169 | 0.279 | 0.414 |
| Claude Opus 4.5 | 0.802 | 0.422 | 0.172 | 0.144 | 0.267 | 0.401 |
| GPT-5 | 0.776 | 0.358 | 0.108 | 0.109 | 0.216 | 0.340 |
| o3 | 0.737 | 0.225 | 0.000 | 0.027 | 0.121 [0.056, 0.210] | 0.214 |
| Claude 3.7 Sonnet | 0.658 | 0.138 | 0.000 | 0.012 | 0.077 | 0.131 |
| Claude 3.5 Sonnet (New) | 0.565 | 0.111 | 0.000 | 0.015 | 0.045 | 0.106 |
| GPT-4o | 0.459 | 0.049 | 0.000 | 0.003 | −0.006 | 0.047 |
| GPT-4 Turbo | 0.402 | 0.012 | 0.000 | 0.000 | 0.003 | 0.011 |

Three wedges separate the first column from the rest.

*The value-weighting wedge.* The most accurate model succeeds on 84.2% of tasks but completes 54.4% of the human hours. It succeeds on 95.5% of attempts at tasks under 15 minutes and 59.4% at tasks over four hours, and the latter carry most of the hours. For o3 the drop is from 73.7% to 22.5%; for GPT-4o from 45.9% to 4.9%. A model with a 40% success rate on this suite completes about 1% of its hours.

*The oversight wedge.* A user who knows only the leaderboard number and faces θ₀ must verify everything or delegate nothing. Verification costs 25% of each task's time including overhead, so usefulness is a_w − 0.25 when positive: 0.294 for the top model, and exactly zero for the 13 models whose a_w is below 0.25, among them o3 with 73.7% accuracy.

*The predictability bonus.* Knowing the task length recovers little or nothing at θ₀ (0.291 against 0.294 for the top model; small positive amounts for weaker models). Having seen the agent attempt the task before adds 0.056 for the top model (to 0.350) and turns o3's zero into 0.121. Perfect foresight would give 0.517. So in this context roughly a third of the hours the best model can complete are lost to the user's inability to know in advance which attempts will succeed.

The length result deserves a comment, since length is a strong predictor of success in the usual sense: across models the cross-fitted length belief has a mean Brier score of 0.077 against 0.175 for a constant. The reason it buys little is that Brier score weights tasks equally, while hours are concentrated in long tasks, which is exactly where the fitted success probability sits in the uninformative middle of its range.

### 6.2 Verification cost decides how much predictability is worth

Holding h = 5 and c = 0.05 and varying ρ for the top model (`rho_sweep` in `results/usefulness.json`): at ρ = 0 every information structure gives 0.494, because free checking makes foresight worthless. At ρ = 0.2 the values are 0.294 (none), 0.291 (length) and 0.350 (task history). At ρ = 0.5 they are 0.000, 0.013 and 0.241, against an oracle value of 0.517 throughout. When checking costs half as much as doing, a leaderboard-informed user gets nothing from the most accurate model in the dataset, while a user with a short history on each task keeps almost half of the oracle value. In such contexts usefulness is determined by how predictable success is, and the success rate matters mainly through that.

### 6.3 Accuracy is a fair ordinal proxy and a poor cardinal one

*Ordinal agreement.* At θ₀ the Kendall τ between accuracy and usefulness across the 20 models is 0.905 with task history (interval [0.800, 0.929]), 0.811 with length ([0.515, 0.947]) and 0.568 with no information ([0.429, 0.742]); the last is low because 13 models tie at zero. Across the 72 contexts, τ with task history has median 0.926 and minimum 0.884, and the most accurate model is the most useful in every context. With length information the median is 0.895 but the minimum is 0.105, and the top model differs in 6 of 72 contexts; all six have ρ ≥ 0.3 and a best usefulness below 0.035, so they are contexts in which no model is worth much. Restricting to the 60 contexts where some model reaches U ≥ 0.1, the minimum τ is 0.568 and the top model never differs. In short: on this suite, where models lie along a single capability trend, accuracy ranks models about as usefulness does whenever usefulness is non-negligible.

*Dominance.* Theorem 1 gives a sharper test: does the more accurate model of a pair have the higher curve everywhere? With task history, on a grid of 20 thresholds, 140 of 190 pairs are ordered by dominance in the direction of accuracy. Of the 50 pairs whose curves cross, 10 cross by more than 0.005 and 4 by more than 0.02; the five largest crossings all involve pairs with accuracy gaps below 0.025, such as GPT-5.2 against GPT-5.3-Codex (gap 0.011; the less accurate model's curve is higher by 0.027 at k = 0.75). With length information 182 pairs cross and 64 by more than 0.02, but these estimates rest on out-of-sample length predictions over an effective 53 tasks and we read them as showing that length-only beliefs are fragile rather than as reliable reversals.

*Cardinal distortion.* From GPT-4o to o3, accuracy rises by 0.278 and task-history usefulness at θ₀ by 0.127. From o3 to Claude Opus 4.6, accuracy rises by 0.105 and usefulness by 0.229. An accuracy point is worth about 0.46 points of hours saved in the first interval and about 2.2 in the second, a ratio near five. This convexity is what the model predicts: s_θ is convex, and gains on long tasks arrive late. A reader who extrapolates usefulness linearly from accuracy will overstate it for weak agents and understate how quickly it changes near the top.

### 6.4 Inconsistency

Across models, between 8.3% and 32.9% of tasks have mixed outcomes across a handful of runs. For the top model the figure is 12.7% of tasks but 28.5% of human hours: inconsistency is concentrated where the value is. This is the quantity Proposition 2(c) prices. Our split-half estimate of the task-history value (0.350) against the oracle (0.517) bounds the combined cost of irreducible run-to-run variation and of having seen only two or three attempts; the simulation in Section 5 suggests the second component accounts for a few hundredths.

### 6.5 Sensitivity to the value weighting

Time weighting is one choice. With weights t^γ (`results/table_gamma.csv`), the top model's task-history usefulness at θ₀ is 0.350 at γ = 1, 0.442 at γ = 0.5 and 0.704 at γ = 0 (every task equally valuable, costs still proportional to its length), against an accuracy of 0.842. The qualitative decomposition survives, but the magnitude of the value-weighting wedge is a property of the benchmark's task mix and of the weighting, not of the agent alone. The ordering can also move: at γ = 0 with length information GPT-5.2 (0.618) is ahead of Claude Opus 4.6 (0.584), the reverse of their accuracy order and of their order at γ = 1.

## 7 Discussion

**What "useful" means here, and what should be reported.** The analysis suggests that a benchmark can say much more about usefulness without new experiments, provided it has repeated runs and a per-task human cost. We propose reporting, per agent:

1. the headline success rate with an interval, as now;
2. the value-weighted success rate a_w under a stated weighting, with the concentration of that weighting (effective number of tasks);
3. the delegation-surplus curve π(k), estimated as a held-out policy value, under at least two information structures, a pre-run one available to any user (such as task length) and task history;
4. the perfect-foresight ceiling a_w(1 − k), so that the gap attributable to unpredictability is visible;
5. at least four runs per task, with the split-half bias noted.

With the curve in hand, a reader inserts their own (ρ, h, c) into Proposition 1. Claims that one agent is "better" can be graded: dominance in the sense of Theorem 1, superiority in a stated region of contexts, or merely higher accuracy.

**A design implication.** Sections 6.1 and 6.2 locate much of the lost value in the user's inability to predict success. That is something agents can supply: a calibrated statement, before or after the attempt, of whether the task was done. In the model this is an information structure like any other, and its worth is measured by how far it lifts π towards the ceiling.

**A protocol we could not run.** We have no model access here, so the agent's own confidence is untested. The protocol is: elicit from each agent, before each run and again on completion, a probability that the run succeeds; compute π under these two information structures by the estimator of Section 5. Predictions that would be falsified by the data: (P1) the pre-run self-report curve lies above the length curve for k between 0.2 and 0.6 for frontier models; (P2) the post-run self-report curve lies above the task-history curve at the same thresholds, because it can see run-specific evidence that history cannot; (P3) the share of the oracle gap closed by post-run self-reports rises more slowly across model generations than a_w does, in line with the finding of Rabanser et al. (2026) that reliability lags capability. If P2 fails, verification by the user remains the only way to recover inconsistency costs.

**Relation to field evidence.** The model offers one reading of a measured slowdown alongside high benchmark scores (Becker et al., 2025): if ρ + c is large for experts in familiar code, and if users misjudge q, then delegation that looks attractive at benchmark accuracy has negative surplus. We cannot test this reading with the data we have, and the field studies include effects the model omits.

## 8 Limitations

- **Contexts are assumed, not measured.** ρ, h and c are free parameters. We chose θ₀ and a grid for illustration. All levels in Section 6 depend on them; only the structural results and the comparisons across information structures do not.
- **Costs proportional to task length.** The model takes verification, harm and overhead to scale with t. In practice checking a long task can be much cheaper or dearer than a fixed fraction of doing it, and ρ may vary by task type. Proposition 1 does not cover task-varying parameters, though the curve can be computed within strata.
- **Verification is perfect and failures are binary.** Real review misses defects and real outputs are partially useful or repairable. The 24-point gap between test-passing and mergeable patches (METR, 2026) is itself a warning that the benchmark's success label is not the user's.
- **No learning, no interaction, no quality effects.** The user does not iterate with the agent, the agent does not ask questions, and effects on the user's own skill or on output quality are absent. Usefulness as defined is labour saving on a fixed task set and nothing more.
- **The benchmark is not a workload.** The task mix is METR's, built to span lengths for a capability measurement, mostly software and research engineering. The value-weighting wedge depends on it. With time weighting the effective sample is about 53 tasks, intervals are wide (for the top model's length-information usefulness, [0.129, 0.440]), and the 50 tasks over four hours dominate.
- **Human times are estimates** from baselines and expert judgement and enter both the weights and the cost scale.
- **The task-history structure is stylised.** Real users rarely repeat an identical task; history on similar tasks is weaker information. Our estimate is also biased downward by the small number of runs, and the shrinkage pseudo-count of 2 was fixed in advance, not tuned.
- **The "none" structure uses the in-sample a_w**, a single parameter. The source data use different agent scaffolds for different models (the file's scaffold field shows at least three families), so cross-model comparisons mix model and scaffold differences.
- **Ordinal agreement may not transfer.** The high rank correlation was found among general-purpose models along one capability trend. Corollary 1 shows the reversal is possible in principle, and agents that differ in calibration or consistency by design could realise it; we did not observe large reversals and do not claim they are common.
- **The mathematics is known.** The theorems are applications of standard results on stochastic orders and decision curves. The contribution is the reduction, the reporting consequence and the measurement.
- **Run log.** `results/run_log.txt` records the final run; the figure `results/fig_usefulness.png` is produced by the same script and is not needed for any number in the text.

## 9 Conclusion

Within a simple and explicit delegation model, whether an agent is useful is settled by one curve per agent and information structure, of which benchmark accuracy is the leftmost point. On public run-level data, the distance from that point to hours saved is large: for the most accurate model, 84% of tasks, 54% of hours, and about a third of hours saved in a moderate context. Accuracy ranked these models about as usefulness did, but it misstated magnitudes by a wide and non-constant margin, and it said nothing about predictability, which governs usefulness whenever checking is costly. Reporting the curve requires repeated runs and a human cost per task, both of which several benchmarks already collect.

## References

Becker, J., Rush, N., Barnes, E., and Rein, D. (2025). Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity. arXiv:2507.09089.

Blackwell, D. (1953). Equivalent Comparisons of Experiments. *Annals of Mathematical Statistics*, 24(2), 265–272.

Brynjolfsson, E., Li, D., and Raymond, L. (2025). Generative AI at Work. *Quarterly Journal of Economics*, 140(2), 889–942.

Geifman, Y., and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems 30*, 4878–4887.

Huang, L., Xiao, W., and Vishnoi, N. K. (2026). Delegation and Verification under AI. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306, 48039–48071. arXiv:2603.02961.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *International Conference on Learning Representations*. arXiv:2310.06770.

Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N., and Narayanan, A. (2025). AI Agents That Matter. *Transactions on Machine Learning Research*. arXiv:2407.01502.

Kapoor, S., et al. (2026). Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation. *International Conference on Learning Representations*. arXiv:2510.11977.

Kwa, T., West, B., Becker, J., Deng, A., et al. (2025). Measuring AI Ability to Complete Long Tasks (later versions titled "Measuring AI Ability to Complete Long Software Tasks"). *Advances in Neural Information Processing Systems 38*. arXiv:2503.14499. Data: github.com/METR/eval-analysis-public, commit 52cb829c.

Liu, W., Qin, J., Huang, X., Zeng, X., Xi, Y., et al. (2025). Position: The Real Barrier to LLM Agent Usability is Agentic ROI. arXiv:2505.17767.

Madras, D., Pitassi, T., and Zemel, R. (2018). Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer. *Advances in Neural Information Processing Systems 31*, 6147–6157.

Mazeika, M., Gatti, A., Menghini, C., Sehwag, U. M., Singhal, S., Orlovskiy, Y., et al. (2025). Remote Labor Index: Measuring AI Automation of Remote Work. arXiv:2510.26787.

METR (2025). Research Update: Algorithmic vs. Holistic Evaluation. METR blog, 12 August 2025. https://metr.org/blog/2025-08-12-research-update-towards-reconciling-slowdown-with-time-horizons

METR (2026). Many SWE-bench-Passing PRs Would Not Be Merged into Main. METR note, 10 March 2026. https://metr.org/notes/2026-03-10-many-swe-bench-passing-prs-would-not-be-merged-into-main/

Mozannar, H., and Sontag, D. (2020). Consistent Estimators for Learning to Defer to an Expert. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 7076–7087.

Patwardhan, T., Dias, R., Proehl, E., Kim, G., Wang, M., et al. (2025). GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks. arXiv:2510.04374.

Rabanser, S., Kapoor, S., Kirgis, P., Liu, K., Utpala, S., and Narayanan, A. (2026). Towards a Science of AI Agent Reliability. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306. arXiv:2602.16666.

Shaked, M., and Shanthikumar, J. G. (2007). *Stochastic Orders*. Springer Series in Statistics. Springer.

Vickers, A. J., and Elkin, E. B. (2006). Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Medical Decision Making*, 26(6), 565–574.

Yao, S., Shinn, N., Razavi, P., and Narasimhan, K. (2025). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. *International Conference on Learning Representations*. arXiv:2406.12045.
