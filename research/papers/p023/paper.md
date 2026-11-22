# Redundancy or Division of Labor? Correlated Errors Set the Hand-off Budget of Multi-Agent Teams

**Target venue:** NeurIPS (main track; theory plus empirical validation)

---

## Abstract

Multi-agent LLM systems handle the three classic problems of collaboration in different ways. **Redundancy** means many agents solve the same problem and their conflicts are settled by voting. **Division of labor** means specialists solve parts, and their outputs must be passed between them through lossy hand-offs. Practitioners pick between these designs by trial and error. We give a budgeted reliability model that makes the trade-off quantitative. In the model, per-item difficulty is latent and Beta-distributed, so agents' errors have intra-item correlation ρ. Each hand-off between specialists fails with probability ε. We prove four results. (i) Plurality voting saturates at the ceiling P(θ > 1/m), not at 1. (ii) For m = 2, the remaining gap closes only as f′(1/2)/(8n), not exponentially; this also gives a closed-form sign condition for when adding voters lowers accuracy. (iii) Tuple-level and coordinate-wise voting share the same infinite-team limit. (iv) As the budget grows, division of labor beats redundancy if and only if ε < ε\* = 1 − (∏C_G/∏C_S)^{1/(K−1)}, where C_G and C_S are the voting ceilings of generalists and specialists. A corollary: when generalists are better than chance and errors are independent (ρ → 0), ε\* → 0. Correlated errors are therefore what keeps division of labor worthwhile at large budgets. In exact computations (K = 4, p_G = 0.6, p_S = 0.7, 100 calls per subtask), the tolerable hand-off error rises from 2.9% at ρ = 0 to 23.6% at ρ = 0.1. We test the model in two settings. The first is 3,075 small learned agents (MLPs; 5 seeds × 3 data-diversity conditions × 205 networks), where training-data diversity varies ρ. A two-parameter fit from an 8-agent pilot predicts held-out vote curves with mean absolute error (MAE) 0.009–0.013, against 0.059–0.098 for the independent-error model, and predicts the break-even hand-off error with r = 0.83 (MAE 0.009). The second is the released answers of 46 LLMs on 11,429 MMLU-Pro items. There, the two bounding wrong-answer models (m = 2 and m = 10) bracket the observed voting curves, and routing each subject to its best models buys so little that any hand-off error above about 0.5% cancels it. We close with a pre-registered protocol for testing the predictions on LLM agent teams.

## 中文摘要

多智能体大模型系统处理协作中的三个经典问题——分工、通信与冲突——大致有两种设计。一种是**冗余**：多个智能体求解同一问题，用投票化解冲突。另一种是**分工**：专家各自求解子任务，结果要经过有损的交接（通信）才能拼合。目前这两种设计的取舍主要靠经验。本文建立一个带预算的可靠性模型，把这一取舍量化。模型中题目难度是服从 Beta 分布的潜变量，因而智能体的错误存在题内相关 ρ；专家之间每次交接以概率 ε 出错。我们证明了四个结果：(i) 多数（众数）投票的极限是 P(θ>1/m)，而不是 1；(ii) 当 m=2 时，与上限的差距只以 f′(1/2)/(8n) 的速度缩小，而不是指数速度，由此还得到"增加投票者反而降低准确率"的闭式符号条件；(iii) 整体元组投票与逐子任务投票的无限团队极限相同；(iv) 预算趋于无穷时，分工优于冗余当且仅当 ε < ε\* = 1 − (∏C_G/∏C_S)^{1/(K−1)}，其中 C_G、C_S 分别是通才与专家投票的准确率上限。推论：若通才优于随机猜测且错误相互独立（ρ→0），则 ε\*→0。也就是说，大预算下分工之所以仍有价值，恰恰是因为错误相关。精确计算显示（K=4，p_G=0.6，p_S=0.7，每个子任务调用 100 次），可容忍的交接错误率从 ρ=0 时的 2.9% 升至 ρ=0.1 时的 23.6%。我们在两种场景中检验模型。第一种是 3,075 个小型学习型智能体（MLP；5 个种子 × 3 种数据多样性条件 × 205 个网络），用训练数据的多样性调节 ρ：用 8 个智能体的试点数据拟合两个参数，预测留出集投票曲线的平均绝对误差（MAE）为 0.009–0.013，而独立错误模型为 0.059–0.098；对临界交接错误率的预测与观测相关系数 r=0.83（MAE 0.009）。第二种是 46 个公开大模型在 11,429 道 MMLU-Pro 题目上的作答：两种极端的错误答案模型（m=2 与 m=10）夹住了实际投票曲线；按学科把题目路由给该学科最强的模型几乎没有收益，只要交接错误率超过约 0.5%，这点收益就会被抵消。最后，我们给出在大模型智能体团队上检验这些预测的预注册方案。

---

## 1 Introduction

Ask how a team of LLM agents should collaborate and you get three linked questions. How should work be divided? How should agents communicate? How should disagreements be resolved? Deployed systems answer in two very different ways. Role-based frameworks such as MetaGPT [Hong et al., 2024], ChatDev [Qian et al., 2024] and AutoGen [Wu et al., 2023] divide labor. Specialists handle sub-problems and pass artifacts along a pipeline, so communication is a chain of hand-offs. Sampling-and-voting methods [Wang et al., 2023; Li et al., 2024] and multi-agent debate [Du et al., 2024] use redundancy instead. Many agents attack the same problem and conflicts are resolved by aggregation.

Empirically, neither design dominates. Failure analyses of deployed multi-agent systems attribute many failures to inter-agent misalignment and lost information at hand-offs [Cemri et al., 2025]. Controlled scaling studies find that multi-agent coordination helps on some tasks and hurts on others, especially sequential ones [Kim et al., 2025b]. On the redundancy side, voting gains diminish and can turn negative as calls are added [Chen et al., 2024]. Debate largely reduces to voting [Choi et al., 2025; Smit et al., 2024]. LLMs also make strongly correlated errors, both across samples and across vendors [Kim et al., 2025a]. Practitioners lack a model that says *when* to split a task, given a budget, an error correlation, and an imperfect communication channel.

**This paper.** We narrow the broad question to one we can answer: *Under a fixed call budget, when should a team divide a K-part task among specialists, paying hand-off errors, rather than have generalists solve the whole task redundantly and vote?* We answer it with a minimal model. Its three ingredients map onto the three parts of the question:

- **Conflict:** plurality voting over agents whose correctness on an item is Bernoulli(θ), where θ is a latent item difficulty with mean p and intra-item correlation ρ (beta-binomial).
- **Division of labor:** specialists have higher per-subtask accuracy p_S than whole-task generalists (p_G). This is the "returns to specialization" of Becker and Murphy [1992].
- **Communication:** each of the K − 1 hand-offs that stitch specialist outputs together fails with probability ε. Part of the budget can be spent on interface checks.

**Contributions.**

1. **A correlated-vote ceiling and a polynomial approach to it (Thm. 1, Prop. 2).** Infinite-team plurality accuracy equals P(θ > 1/m). For m = 2 and odd n, the gap to the ceiling is f′(1/2)/(8n) + o(1/n). This is polynomial, unlike the exponential Condorcet rate for independent errors. For Beta difficulty, its sign is sign(p − 1/2), which gives a closed-form condition for when additional voters *reduce* accuracy. The constant matches exact computation to four significant digits.
2. **Tuple versus coordinate voting (Lemma 3).** Voting over whole K-part answers has the same infinite-team limit as voting part by part. At finite n it is markedly worse; we quantify the gap by Monte Carlo.
3. **The division-of-labor crossover (Thm. 4, Cor. 5).** At large budgets, division beats redundancy if and only if ε < ε\* = 1 − (∏C_G/∏C_S)^{1/(K−1)}. With independent errors and p_G > 1/m, ε\* → 0. With correlated errors, ε\* is bounded away from 0 and peaks at intermediate ρ. Correlated errors therefore set the "hand-off budget" a pipeline can afford. We also map finite-budget phase diagrams and the optimal share of the budget spent on interface checks. That share is *not* monotone in ρ, a negative result we report.
4. **Two empirical validations.** (a) Learned agents: 3,075 small MLPs (5 seeds × 3 conditions × 205), with ρ manipulated by training-data diversity. An 8-agent pilot fit predicts held-out vote curves and the empirical break-even ε\*(n). (b) Real LLMs: released answers of 46 models on MMLU-Pro. The m = 2 and m = 10 versions of the model bracket the observed voting curves. Subject-level routing ("division of labor by routing") yields a break-even hand-off error below 1%.
5. **A pre-registered protocol** (Sec. 8) that turns the theory into falsifiable predictions for LLM agent teams. It needs only a pilot of about 8 samples × 100 items per agent type.

We do not claim the model captures everything about agent collaboration. Its value is that it is small enough to prove things about, its two difficulty parameters can be estimated cheaply, and it makes predictions that can fail.

## 2 Related Work

**Multi-agent LLM frameworks and their failures.** Role-specialized pipelines (MetaGPT [Hong et al., 2024], ChatDev [Qian et al., 2024], AutoGen [Wu et al., 2023], CAMEL [Li et al., 2023]) embody division of labor with structured communication. Surveys catalogue the design space [Guo et al., 2024; Tran et al., 2025]. Cemri et al. [2025] built MAST, a taxonomy of 14 failure modes in three categories (system design, inter-agent misalignment, task verification) from annotated traces. Our hand-off error ε is a one-number abstraction of their inter-agent misalignment category. Kim et al. [2025b] ran 180 controlled configurations. They report that multi-agent benefit depends on task structure and that gains shrink once the single agent is strong. Our Thm. 4 gives one mechanism consistent with this: when generalist and specialist ceilings are close, ε\* is small (Sec. 7).

**Redundancy, voting and debate.** Self-consistency [Wang et al., 2023] and "More Agents" [Li et al., 2024] show accuracy gains from sampling and voting. Chen et al. [2024] show the gains can be non-monotone in the number of calls: voting helps on easy queries and hurts on hard ones. Prop. 2 recovers this inside a parametric model and gives the asymptotic rate and the sign condition. Debate [Du et al., 2024; Liang et al., 2024] has been benchmarked against voting [Smit et al., 2024]. Choi et al. [2025] model debate as a belief update that forms a martingale, and they find majority voting accounts for most of debate's gains. This motivates our focus on voting as the conflict-resolution primitive. Mixture-of-Agents [Wang et al., 2025] aggregates outputs of different LLMs. Self-MoA [Li et al., 2026] finds that mixing weaker models often hurts through a quality-diversity trade-off. Our ceiling C(p, ρ) gives one formal version of that trade-off.

**Correlated errors.** Kim et al. [2025a] document across 350+ LLMs that models agree on wrong answers far more than chance, more so within a provider. Correlated votes are a classical topic in social choice. Ladha [1992] extends the Condorcet jury theorem to correlated voters, and Boland [1989] analyses majority systems with dependence. Ensemble learning studies diversity–accuracy relations [Krogh and Vedelsby, 1995; Kuncheva and Whitaker, 2003]. Hong and Page [2004] argue diversity can beat ability. We use the beta-binomial correlation model [Skellam, 1948] that is standard in this literature. What we add is the coupling of correlated voting with a division-of-labor alternative and lossy communication under a shared budget, which we have not found treated elsewhere.

**Economics of the division of labor.** Becker and Murphy [1992] argue that specialization is limited by coordination costs rather than by market size alone. Our model is a stochastic, reliability-theoretic version of that argument for agent teams. Its specific new feature is that the *alternative* to specialization, redundancy, has a return that is capped by error correlation.

## 3 Model

**Task.** A task has K subtasks. Subtask k has m candidate answers, one correct. A team has a budget of B *units*. One unit is one subtask-sized agent call.

**Latent difficulty and correlated errors.** Each subtask instance has a latent difficulty θ_k ∈ [0,1], drawn i.i.d. across k from Beta(a, b) with mean p. Given θ_k, agents are correct independently with probability θ_k. When wrong, an agent picks uniformly among the m − 1 wrong answers. Two agents' correctness indicators on the same item then have correlation ρ = 1/(a + b + 1) [Skellam, 1948]. We parametrize by (p, ρ): a = p(1−ρ)/ρ, b = (1−p)(1−ρ)/ρ. ρ = 0 is the independent (Condorcet) case. The uniform wrong-answer assumption is optimistic; m = 2 is the pessimistic case in which all wrong agents agree. Real LLMs fall in between (Sec. 6.3).

**Strategies.** Agents come in two types: generalists (accuracy p_G, correlation ρ_G) and specialists (p_S, ρ_S).

- **Redundancy (R).** n = ⌊B/K⌋ generalists each solve the whole task, at a cost of K units each. Answers are aggregated by *coordinate-wise plurality* (R-coord) or by plurality over whole K-tuples (R-tuple). R-tuple applies when outputs cannot be parsed into parts.
- **Division of labor (D).** n = ⌊B/K⌋ specialists per subtask, with per-subtask plurality. The K partial results are then integrated through K − 1 hand-offs, each of which independently corrupts the result with probability ε. The task succeeds only if every subtask vote is correct and no hand-off fails.
- **Division with interface checks (D+c).** c of the B units are spent on checks, spread evenly over the K − 1 interfaces. Each check independently catches and repairs a hand-off error with probability d, so the effective hand-off error is ε(1−d)^{c/(K−1)}, and n = ⌊(B−c)/K⌋.

Let A_n(p, ρ, m) be the plurality accuracy of n agents on one subtask (random tie-breaking). Under the model,

  Acc(R-coord) = A_n(p_G, ρ_G)^K,  Acc(D) = (1−ε)^{K−1} A_n(p_S, ρ_S)^K.

The independence of θ_k across subtasks, and of a generalist's errors across subtasks given θ, are assumptions. We test the second one in Sec. 6.2.

For m = 2, A_n has the closed form Σ_{j>n/2} BetaBin(j; n, a, b), plus half the mass at the tie when n is even. For m > 2 we use Monte Carlo. All code is in `code/theory.py`.

## 4 Theory

### 4.1 The correlated-vote ceiling

**Theorem 1 (ceiling).** Assume θ has no atom at 1/m. Then lim_{n→∞} A_n(p, ρ, m) = C_m(p, ρ) := P(θ > 1/m) = 1 − I_{1/m}(a, b), where I is the regularized incomplete beta function. For ρ = 0, C_m = 1[p > 1/m].

*Proof.* Condition on θ. By the strong law of large numbers, the vote share of the correct answer converges a.s. to θ, and the share of each wrong answer converges to (1−θ)/(m−1). The correct answer is eventually the strict plurality if θ > (1−θ)/(m−1), which is equivalent to θ > 1/m. It is eventually not the plurality if θ < 1/m. So the conditional accuracy converges to 1[θ > 1/m] for every θ ≠ 1/m. Dominated convergence (accuracies lie in [0,1]) and the no-atom assumption give the result. ∎

So redundancy cannot push accuracy past the mass of items on which agents are better than chance. For m = 2 the ceiling is P(θ > 1/2). With p = 0.65 it equals 0.984, 0.911, 0.829, 0.750 and 0.687 for ρ = 0.02, 0.05, 0.1, 0.2 and 0.4 (exact; `results/theory_results.json`, E1; Fig. 1a).

### 4.2 How fast the ceiling is approached

**Proposition 2 (polynomial approach; m = 2, odd n).** Let θ have a density f that is C² near 1/2. Then

  A_n − C_2 = − f′(1/2) / (8n) + o(1/n).

For Beta(a, b), sign f′(1/2) = sign(a − b) = sign(p − 1/2). So if p > 1/2, accuracy rises toward the ceiling from below. If p < 1/2, it *falls* toward the ceiling from above: adding voters lowers accuracy.

*Proof sketch.* Let g_n(θ) = P(Bin(n, θ) > n/2). For odd n, g_n(1−θ) = 1 − g_n(θ). Splitting the integral at 1/2 and substituting θ = 1/2 ± δ gives

  A_n − C_2 = ∫_0^{1/2} g_n(1/2 − δ) [f(1/2 − δ) − f(1/2 + δ)] dδ.

The bracket equals −2f′(1/2)δ + O(δ³). The non-uniform Berry–Esseen bound gives g_n(1/2 − δ) = Φ(−2δ√n) + O(n^{−1/2}(1 + |2δ√n|³)^{−1}). Substituting u = 2δ√n, the main term is −2f′(1/2) · (1/(4n)) ∫_0^∞ u Φ(−u) du = −2f′(1/2)/(16n), since ∫_0^∞ u Φ(−u) du = 1/4. The error terms contribute o(1/n). For Beta, f′(x)/f(x) = (a−1)/x − (b−1)/(1−x). At x = 1/2 this equals 2(a − b), and a − b = (2p − 1)(1 − ρ)/ρ. ∎

*Numerical check (computed; `results/rate_check.json`).* At n = 6401, n·(C_2 − A_n) = 1.9529 against the predicted f′(1/2)/8 = 1.9547 (p = 0.65, ρ = 0.05). The other pairs are 0.9893 vs 0.9900 (ρ = 0.1), 0.35462 vs 0.35476 (ρ = 0.2), 0.90892 vs 0.90942 (p = 0.7, ρ = 0.1) and 0.49756 vs 0.49806 (p = 0.55, ρ = 0.1). Log–log slopes of the gap for n ≥ 41 lie between −0.96 and −1.03 (Fig. 1b).

*Implications.* With independent errors, Hoeffding gives 1 − A_n ≤ exp(−2n(p − 1/2)²), so the number of agents needed to get within η of the limit grows like log(1/η). With correlated errors it grows like f′(1/2)/(8η). Prop. 2 also gives an analytic version of the non-monotone "more calls" curves of Chen et al. [2024]. In this model they occur when there is more density just below the threshold than just above it.

### 4.3 Tuple versus coordinate voting

**Lemma 3.** Suppose a generalist's subtask answers are conditionally independent across k given θ, and the θ_k are independent. Then R-tuple and R-coord have the same infinite-team accuracy, ∏_k C_m(p, ρ).

*Proof.* Given θ, the limiting share of a tuple (v_1, …, v_K) is ∏_k q_k(v_k), where q_k(correct) = θ_k and q_k(wrong) = (1 − θ_k)/(m − 1). A product of positive factors is maximized by maximizing each factor separately. So the all-correct tuple is the strict maximizer if and only if θ_k > 1/m for every k, which is exactly the event that coordinate voting is correct. ∎

At finite n, tuple voting is much worse because its votes are spread over m^K bins. Monte Carlo with 4,000 items (E2): for K = 3, m = 5, p = 0.45, ρ = 0.1, n = 9, R-tuple reaches 0.184 and R-coord 0.337; at n = 401 they reach 0.711 and 0.800, against a common limit of 0.851. For K = 3, m = 2, p = 0.7, ρ = 0.1, both approach 0.728. Here R-coord at n = 129 is 0.7115 (exact value 0.7115). Being able to parse outputs into parts is therefore worth a large amount of budget, a point that matters for protocol design.

### 4.4 When does division of labor pay?

**Theorem 4 (asymptotic crossover).** As B → ∞ with K fixed,

  Acc(R) → ∏_k C_m(p_G, ρ_G) =: Γ_G  and  Acc(D) → (1−ε)^{K−1} ∏_k C_m(p_S, ρ_S) =: (1−ε)^{K−1} Γ_S.

Hence, for large enough budgets, division of labor is strictly better if and only if

  ε < ε\* := 1 − (Γ_G/Γ_S)^{1/(K−1)}  (with identical subtasks: 1 − (C_G/C_S)^{K/(K−1)}).

*Proof.* Apply Thm. 1 subtask by subtask; Lemma 3 covers R-tuple. Then compare the two limits. ∎

**Corollary 5.** Assume identical subtasks and ρ_G = ρ_S = ρ.
(a) *Independent errors make division of labor fragile.* If p_S ≥ p_G > 1/m, then ε\* → 0 as ρ → 0: with a large enough budget, any positive hand-off error makes redundancy the better choice.
(b) *Hard tasks.* If p_G < 1/m < p_S, then ε\* → 1 as ρ → 0: redundancy converges to accuracy 0 and division wins for any ε < 1.
(c) *Perfect correlation.* As ρ → 1, ε\* → 1 − (p_G/p_S)^{K/(K−1)}, the single-agent (n = 1) break-even.
(d) *Asymmetric correlation.* ε\* increases with C_S and decreases with C_G. In the regimes we computed (p > 1/m), C is decreasing in ρ, so specialists that are *more* correlated than generalists shrink ε\*.

*Proof.* As ρ → 0, Beta(a, b) concentrates at p, so C_m → 1[p > 1/m]; this gives (a) and (b). As ρ → 1, Beta(a, b) converges weakly to Bernoulli(p) on {0, 1}, so C_m → p; this gives (c). (d) follows from the formula for ε\*. ∎

The central qualitative point is that **error correlation is what keeps division of labor valuable at scale.** If errors were independent, voting would eventually fix any per-subtask accuracy deficit, and the specialists' advantage would vanish while their hand-off cost would not. Correlation caps redundancy and so preserves the value of specialization. Between the two extremes, ε\* is non-monotone in ρ (Fig. 3). For K = 4, p_G = 0.6:

- p_S = 0.7: ε\* = 0.030 at ρ = 0.01, rises to a maximum of 0.236 at ρ ≈ 0.13, and tends to 0.186 as ρ → 1.
- p_S = 0.65: the peak is 0.148 at ρ ≈ 0.09.
- p_S = 0.9: ε\* keeps rising to 0.418 over the range plotted.

(All values computed; E3.)

### 4.5 Finite budgets and communication spending

For n = 1, ε\* = 1 − (p_G/p_S)^{K/(K−1)}, whatever ρ is. Fig. 4 maps the sign of Acc(D) − Acc(R-coord) for K = 4, p_G = 0.6, p_S = 0.7 (exact computations, E4). The boundary ε at which redundancy starts to win is:

| B (n per subtask) | ρ = 0 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| 4 (1) | 0.186 | 0.186 | 0.186 | 0.186 | 0.186 |
| 40 (10) | 0.240 | 0.239 | 0.233 | 0.221 | 0.203 |
| 400 (100) | **0.029** | 0.215 | **0.236** | 0.229 | 0.207 |

At moderate budgets the region where division wins is largest for independent errors, because specialists are the first to benefit from voting. At large budgets it collapses for independent errors, as Cor. 5(a) predicts. The *same* system parameters can therefore favor opposite designs depending on the budget and on ρ.

**Hard tasks (E5).** If generalists are below chance on each subtask (p_G = 0.45, m = 2), redundancy gets *worse* with more agents: at ρ = 0, accuracy goes from 0.041 at n = 1 to 0.0003 at n = 125. Division (p_S = 0.7, ε = 0.1) goes from 0.175 to 0.729. At ρ = 0.3 the corresponding values are 0.034 and 0.258.

**Interface checks (E6).** Take K = 4, p_S = 0.7, ε = 0.1, d = 0.5 and optimize c by grid search. Checks help a lot. At B = 200 and ρ = 0.1, accuracy rises from 0.442 to 0.600 with c\* = 20 units. At B = 1000 and ρ = 0, it rises from 0.729 to ≈1.000 with c\* = 100. However, the optimal *share* c\*/B is not monotone in ρ. At B = 1000 it is 0.10 at ρ = 0, 0.035 for ρ ∈ [0.02, 0.2], and 0.05 at ρ = 0.5. At B = 40 it rises from 0.10 to 0.50 as ρ goes from 0 to 0.5. The intuition is that at ρ = 0 voting saturates quickly and leaves spare budget for checks, while at large ρ extra votes are nearly worthless. We had expected "more correlation ⇒ more communication spending"; this did not hold in general, so we do not claim it.

## 5 Experimental Setup

All numbers below were produced by code in `code/` and are stored in `results/`. Section 4 numbers come from exact computation or Monte Carlo. Section 6 numbers come from experiments we ran on CPU. Total runtime is about 3 minutes on 4 cores.

**Learned agents (`learned_agents.py`).** We use a synthetic compositional task with K = 4 binary subtasks. Input x ∈ ℝ^32 is made of four 8-dimensional blocks. The label of subtask k depends only on block k, through a fixed random two-layer tanh teacher; training labels have 10% flips, test labels are clean. Each agent is a one-hidden-layer tanh MLP (32 hidden units, Adam, 400 full-batch epochs, BCE loss) trained on 400 examples.

- A **generalist** sees all 32 inputs and predicts all 4 labels.
- A **specialist** sees one block and predicts one label.

The specialist's accuracy advantage arises naturally because it does not have to learn to ignore 24 irrelevant inputs. We train 41 generalists and 4 × 41 specialists per condition, under three data-diversity conditions that manipulate ρ:

- *shared*: identical data, different random initialization;
- *bootstrap*: bootstrap resamples of a shared pool;
- *independent*: fresh i.i.d. data per agent.

This gives 205 trained networks per condition (615 per seed). We repeat with 5 seeds (different teachers, pools and initializations), for 3,075 networks in total. Evaluation uses 2,000 test items. Items are split into a calibration half and a held-out half. **Pilot:** (p, ρ) for each agent type and subtask are fit by method of moments from the first 8 agents on calibration items only. **Evaluation:** vote curves on held-out items, using random n-subsets (200 draws) of the other 33 agents.

**Real LLMs (`real_data_test.py`).** We use an item × model answer matrix built (`build_matrix.py`) from the model outputs released in the public MMLU-Pro repository [Wang et al., 2024]: 46 models and the 11,429 items present for all of them with at least 3 options. Two files were excluded because of data problems documented in the script. Each model has one answer per item, so here "agents" are distinct models rather than samples from one model. We split items 50/50 at random into calibration and held-out halves.

- **R1:** random n-subsets of a "peer" pool (13 models with calibration accuracy 0.60–0.80) and a "mid" pool (19 models, 0.35–0.60).
- **R2:** "division of labor by routing". We exclude the single dominant model (calibration accuracy 0.90), because it would otherwise be every category's specialist. Generalist teams are random n-subsets of the top-8 models overall. Specialist teams for each of the 14 subjects are random n-subsets of that subject's top-8 models; all selection is done on calibration items. A composite task takes one held-out item from each of K = 4 random subjects (300 subject sets). Because items are drawn independently, composite accuracy is the product of per-subject accuracies.

## 6 Results

### 6.1 Theory checks

The Monte Carlo estimates agree with the exact beta-binomial values within sampling error (E2). Proposition 2's constant is reproduced to 3–4 significant digits (Sec. 4.2).

### 6.2 Learned agents

Training-data diversity moves ρ as intended. All values below are means ± SD over 5 seeds (`results/learned_summary.json`).

| condition | p_G | p_S | ρ_G | ρ_S | vote-curve MAE, beta-binomial (8-agent pilot) | vote-curve MAE, i.i.d. model |
|---|---|---|---|---|---|---|
| shared | 0.798 ± 0.008 | 0.847 ± 0.007 | 0.326 ± 0.016 | 0.435 ± 0.020 | **0.0094 ± 0.0019** | 0.098 ± 0.006 |
| bootstrap | 0.795 ± 0.008 | 0.844 ± 0.006 | 0.313 ± 0.018 | 0.328 ± 0.011 | **0.0125 ± 0.0033** | 0.090 ± 0.005 |
| independent | 0.790 ± 0.002 | 0.838 ± 0.004 | 0.176 ± 0.004 | 0.162 ± 0.005 | **0.0097 ± 0.0022** | 0.059 ± 0.003 |

**(i) Two parameters from 8 agents predict held-out voting.** The independent-error model predicts that 33 agents would reach near-perfect accuracy. The observed curves instead plateau at the beta-binomial ceiling (Fig. 6). The pilot fit reduces vote-curve error 6–10× relative to the i.i.d. model.

**(ii) The cross-subtask independence assumption holds here.** The mean correlation of a generalist's errors across its own subtasks is −0.001, −0.002 and −0.003 in the three conditions.

**(iii) The break-even hand-off error tracks the theory (Fig. 7).** We compute the observed ε\*(n) from measured team accuracies as 1 − (Acc_R/Acc_D,ε=0)^{1/(K−1)}. Over n = 1…33 it falls from 0.075 to 0.032 (shared), from 0.075 to 0.056 (bootstrap) and from 0.077 to 0.044 (independent). The pilot-based predictions are 0.077→0.032, 0.083→0.066 and 0.074→0.055. Across all 105 (seed, condition, n) points, predicted vs observed ε\* have Pearson r = 0.83, MAE 0.0090 and mean bias +0.0039 (slight over-prediction).

The ordering of the declines matches Cor. 5. The decline is fastest in *shared*, where specialists are more correlated than generalists (ρ_S > ρ_G, Cor. 5d); n1→n33 drop 0.043 ± 0.005. It is intermediate in *independent*, where ρ is low (Cor. 5a); drop 0.034 ± 0.008. It is slowest in *bootstrap*, where ρ_S ≈ ρ_G ≈ 0.32 is high; drop 0.019 ± 0.006.

The model's main miss is in *independent*. There, ε\* keeps falling faster than the beta fit predicts (0.044 observed vs 0.055 predicted at n = 33). The fitted Beta apparently puts too much mass just below the threshold, a tail misfit the two-parameter family cannot correct.

At n = 33 and ε = 0, division beats redundancy in every condition: 0.804 vs 0.703 (independent), 0.659 vs 0.556 (bootstrap), 0.606 vs 0.549 (shared). Even so, a hand-off error of 3–6% is enough to reverse the ranking.

### 6.3 Real LLMs (MMLU-Pro)

**Errors are strongly correlated, and wrong answers are coordinated.** This reproduces the computation of `exp1_correlation.py`. Over the 1,035 model pairs, the mean pairwise correctness correlation is 0.344 (median 0.347). When both models in a pair are wrong, they pick the same wrong letter 38.7% of the time, against 12.5% expected under uniform wrong answers (3.1×). Same-family pairs are more correlated than cross-family pairs (ρ 0.410 vs 0.336), consistent with Kim et al. [2025a]. Plurality over all 46 models reaches 0.745, while 98.2% of items are answered correctly by at least one model.

**R1: the m = 2 and m = 10 versions of the model bracket the real voting curves (Fig. 8).** In the peer pool (p̂ = 0.695, ρ̂ = 0.518), observed plurality accuracy rises from 0.701 at n = 1 to 0.785 at n = 13. The m = 2 prediction (all wrong answers coincide) gives 0.695→0.717 (MAE 0.051). The m = 10 prediction (uniform wrong answers) gives 0.692→0.848 (MAE 0.034). The independent-error model predicts 0.9995 at n = 13 (MAE 0.162).

In the mid pool (p̂ = 0.448 < 1/2, ρ̂ = 0.393), the m = 2 model predicts that voting *hurts* (0.448→0.435), as Prop. 2 says it should when p < 1/2. The observed curve instead rises from 0.456 to 0.618 because wrong answers are only partly coordinated; the m = 10 model gives 0.456→0.691 (MAE 0.044). In both pools every observed point lies inside the [m = 2, m = 10] band (within 0.01). An "effective m" between 2 and 10 is therefore the missing third parameter for real LLMs.

**R2: routing-based division of labor among strong LLMs buys almost nothing.** Averaged over the 14 subjects, specialist teams selected per subject on calibration items have mean calibration accuracy p_S = 0.776, against p_G = 0.772 for the generalist top-8. Correlation is high (ρ ≈ 0.60). The observed break-even hand-off error for K = 4 is 0.0025, 0.0047, 0.0038 and 0.0007 at n = 1, 3, 5 and 7 (SD over subject sets 0.002–0.004). The bounding predictions are 0.007–0.008 (m = 2) and 0.007–0.012 (m = 10). The theory correctly predicts the order of magnitude, which is below 1%, but over-predicts by a few tenths of a percentage point. The practical conclusion is that among current frontier-class models, subject routing is worth less than any realistic inter-agent hand-off error rate. Division of labor would need a much larger specialization gain p_S − p_G, of the kind tool use or retrieval might provide, to pay for its communication.

## 7 Discussion

**A decision rule.** The theory reduces the design question to three measurable quantities. The first is each agent type's ceiling C(p, ρ), estimable from a small pilot. The second is the number of hand-offs K − 1. The third is the per-hand-off error ε, measurable by injecting known-good partial results. Division of labor is justified only when ε < 1 − (Γ_G/Γ_S)^{1/(K−1)}, adjusted for the finite budget (Fig. 4). This explains several empirical patterns without ad-hoc arguments:

- the sign of the multi-agent benefit depends on the task;
- sequential pipelines, which have many hand-offs, are the most fragile;
- gains shrink once single agents are strong, because Γ_G/Γ_S → 1 and so ε\* → 0.

The last point is consistent with the capability-saturation finding of Kim et al. [2025b]. We have not fit our model to their data.

**Diversity versus quality.** Cor. 5 and Thm. 1 make the quality–diversity trade-off observed for Mixture-of-Agents [Li et al., 2026] explicit. Adding a lower-p but less-correlated agent raises the ceiling only if it shifts mass of θ across 1/m. The real-data measurement that wrong answers coincide 3.1× more often than chance shows that the *structure* of wrong answers (effective m), not only correctness correlation, governs whether voting resolves conflicts.

**Communication is the binding constraint at scale.** Under correlation, redundancy stops paying after a modest team size: the gap closes only as O(1/n). Under independence, redundancy eventually beats specialization outright. Either way, at large budgets the system's reliability is set by the hand-off channel. That is the regime where investing in interface checks, structured artifacts [Hong et al., 2024] and parseable outputs (Lemma 3) has the highest return.

## 8 A Pre-registered Protocol for LLM Agent Teams

We did not run LLM agents; there is no GPU or API access in this environment. The following protocol would test the theory.

1. **Tasks.** Composite tasks with K ∈ {2, 4, 8} independently checkable parts, for example multi-part math word problems or multi-function coding tasks with per-function unit tests.
2. **Pilot.** For each agent type (whole-task generalist prompt; part-specific specialist prompt or tool), sample 8 completions on 100 calibration items. Estimate p, ρ (method of moments) and an effective m from the wrong-answer agreement rate.
3. **Hand-off error.** Measure ε directly: feed known-correct partial results through the integration step and count corruptions.
4. **Predictions to test**, each falsified if the 95% CI excludes the prediction:
   - **P1** — held-out vote accuracy follows the fitted beta-binomial within the [m = 2, m_eff] band and saturates below 1;
   - **P2** — the gap to the ceiling shrinks ∝ 1/n (log–log slope in [−1.3, −0.7] for n ∈ [8, 64]);
   - **P3** — the observed winner between division and redundancy, as a function of B and injected ε, matches the sign predicted by the finite-budget phase diagram in at least 80% of cells;
   - **P4** — increasing sampling diversity (temperature, prompt or model mixing) lowers ρ and, at large B, lowers the measured ε\*;
   - **P5** — R-tuple voting is worse than R-coord at n ≤ 16 and converges to it.

## 9 Limitations

- **Independence across subtasks and hand-offs.** We assume θ_k independent across subtasks and hand-off failures independent of subtask correctness. In real pipelines a hard subtask probably also produces hard-to-integrate output. This held in our MLP task (cross-subtask error correlation ≈ 0) but was not tested on LLM pipelines.
- **Beta difficulty and uniform wrong answers.** A two-parameter Beta misfits tails: ε\* was over-predicted by about 0.01 in the *independent* condition and by a few tenths of a percentage point on MMLU-Pro. Real wrong answers are coordinated (3.1× chance), so we bracketed predictions between m = 2 and m = 10 rather than fitting a principled effective-m model.
- **Exchangeability.** The theory treats agents of a type as exchangeable. The MMLU-Pro "agents" are heterogeneous models with one answer each, and the pools were chosen by accuracy band to reduce heterogeneity. Heterogeneous weighting (e.g., Dawid–Skene) is not covered.
- **Static aggregation only.** We model conflict resolution as plurality voting. Debate, critique and learned aggregators are not modeled, though Choi et al. [2025] suggest voting captures much of debate's effect.
- **Simple hand-off model.** Hand-offs are binary independent corruptions. Partial information loss, error detection by downstream agents, and adaptive routing are not modeled. The interface-check model (fixed per-check detection d) is the simplest possible.
- **Scale of the empirical evidence.** The learned agents are small MLPs on a synthetic task, and the LLM evidence is multiple-choice QA, a single-step task. No multi-step LLM agent pipelines were run. The MMLU-Pro routing experiment measures only one kind of specialization (subject routing among general models).
- **Proof rigor.** Proposition 2 is a proof sketch: the non-uniform Berry–Esseen step is standard but not written out in full, and the constant is checked numerically to 3–4 digits.

## 10 Conclusion

Division of labor, communication and conflict resolution can be analysed together in one budgeted model. Redundancy, which resolves conflicts by voting, is capped by error correlation at P(θ > 1/m) and approaches that cap only polynomially. Division of labor avoids the cap but pays a multiplicative toll at every hand-off. The break-even hand-off error ε\* has a closed form. It vanishes when errors are independent and stays positive when they are correlated. It can be predicted from an 8-agent pilot in learned-agent experiments, and it is below 1% for subject routing among today's strong LLMs on MMLU-Pro. We hope the protocol in Sec. 8 makes these claims easy to refute or confirm on real agent pipelines.

---

## Reproducibility

See `README.md`. Commands: `python3 code/run_theory.py`, `python3 code/check_rate.py`, `SEED=s python3 code/learned_agents.py` for s = 0…4, `python3 code/aggregate_learned.py`, `python3 code/real_data_test.py`. The MMLU-Pro matrix `results/answers.npz` is rebuilt by `code/run_all.sh <dir>` from the public repository (instructions in that script). Figures: `results/fig1_ceiling.png`, `fig3_eps_star.png`, `fig4_phase_finite.png`, `fig5_comm_share.png`, `fig6_learned_vote_seed*.png`, `fig7_learned_eps_star.png`, `fig8_real_vote.png`, `fig9_real_eps_star.png`.

**Data provenance labels.** Sec. 4 numbers come from exact computation or Monte Carlo in this environment. Sec. 6.2 numbers come from experiments run here. Sec. 6.3 numbers come from analyses run here of public data released by Wang et al. [2024]. All other quantitative statements about prior work are attributed to the cited papers.

## References

- Becker, G. S., and Murphy, K. M. (1992). The Division of Labor, Coordination Costs, and Knowledge. *Quarterly Journal of Economics*, 107(4), 1137–1160.
- Boland, P. J. (1989). Majority Systems and the Condorcet Jury Theorem. *Journal of the Royal Statistical Society, Series D (The Statistician)*, 38(3), 181–189.
- Cemri, M., Pan, M. Z., Yang, S., et al. (2025). Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657.
- Chen, L., Davis, J. Q., Hanin, B., Bailis, P., Stoica, I., Zaharia, M., and Zou, J. (2024). Are More LLM Calls All You Need? Towards the Scaling Properties of Compound AI Systems. *NeurIPS 2024*. arXiv:2403.02419.
- Choi, H. K., Zhu, X., and Li, S. (2025). Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? *NeurIPS 2025*. arXiv:2508.17536.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. (2024). Improving Factuality and Reasoning in Language Models through Multiagent Debate. *ICML 2024*, PMLR 235:11733–11763. arXiv:2305.14325.
- Guo, T., Chen, X., Wang, Y., et al. (2024). Large Language Model based Multi-Agents: A Survey of Progress and Challenges. *IJCAI 2024*. arXiv:2402.01680.
- Hong, L., and Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. *PNAS*, 101(46), 16385–16389.
- Hong, S., Zhuge, M., Chen, J., et al. (2024). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *ICLR 2024*. arXiv:2308.00352.
- Kim, E., Garg, A., Peng, K., and Garg, N. (2025a). Correlated Errors in Large Language Models. *ICML 2025*, PMLR 267:30038–30066. arXiv:2506.07962.
- Kim, Y., et al. (2025b). Towards a Science of Scaling Agent Systems. arXiv:2512.08296.
- Krogh, A., and Vedelsby, J. (1995). Neural Network Ensembles, Cross Validation, and Active Learning. *Advances in Neural Information Processing Systems 7* (NIPS 1994; MIT Press), 231–238.
- Kuncheva, L. I., and Whitaker, C. J. (2003). Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy. *Machine Learning*, 51(2), 181–207.
- Ladha, K. K. (1992). The Condorcet Jury Theorem, Free Speech, and Correlated Votes. *American Journal of Political Science*, 36(3), 617–634.
- Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and Ghanem, B. (2023). CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society. *NeurIPS 2023*. arXiv:2303.17760.
- Li, J., Zhang, Q., Yu, Y., Fu, Q., and Ye, D. (2024). More Agents Is All You Need. *Transactions on Machine Learning Research*. arXiv:2402.05120.
- Li, W., Lin, Y., Xia, M., and Jin, C. (2026). Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? *Transactions on Machine Learning Research*. arXiv:2502.00674.
- Liang, T., He, Z., Jiao, W., et al. (2024). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. *EMNLP 2024*. arXiv:2305.19118.
- Qian, C., Liu, W., Liu, H., et al. (2024). ChatDev: Communicative Agents for Software Development. *ACL 2024*. arXiv:2307.07924.
- Skellam, J. G. (1948). A Probability Distribution Derived from the Binomial Distribution by Regarding the Probability of Success as Variable between the Sets of Trials. *Journal of the Royal Statistical Society, Series B*, 10(2), 257–261.
- Smit, A., Duckworth, P., Grinsztajn, N., Barrett, T. D., and Pretorius, A. (2024). Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs. *ICML 2024*. arXiv:2311.17371.
- Tran, K.-T., Dao, D., Nguyen, M.-D., et al. (2025). Multi-Agent Collaboration Mechanisms: A Survey of LLMs. arXiv:2501.06322.
- Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., and Zou, J. (2025). Mixture-of-Agents Enhances Large Language Model Capabilities. *ICLR 2025*. arXiv:2406.04692.
- Wang, X., Wei, J., Schuurmans, D., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171.
- Wang, Y., Ma, X., Zhang, G., et al. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. *NeurIPS 2024 Datasets and Benchmarks*. arXiv:2406.01574.
- Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155.
