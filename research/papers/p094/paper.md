# How Many Attempts Certify Safety? Identification and Sample-Complexity Limits for Measuring Dangerous Capabilities Against Repeated-Attempt Adversaries

Target venue: NeurIPS 2027 (main track; evaluation methodology / learning theory)

## Abstract

Dangerous-capability evaluations usually report how often a model succeeds when the evaluator tries each task a few times (pass@1, or pass@n with n between 1 and about 10). The quantity that matters for risk is different. A misuse actor or an autonomous agent can retry a task many times, so the relevant estimand is *budget-B coverage* C(B): the share of threat-relevant tasks solved at least once in B attempts, where B may be in the hundreds or thousands. We study what an evaluator can soundly say about C(B) from n ≪ B attempts per task, and how many attempts any protocol needs to certify that C(B) is below a threshold. We prove three results. (i) **Identification.** Under exchangeable attempts, n attempts per task reveal only the first n moments of the task-difficulty distribution. The worst-case width of the identified set for C(B) is exactly 2E_n(B), twice the best uniform approximation error of x^B by degree-n polynomials on [0,1]. (ii) **Rate.** This width is at most 4·exp(−n²/B), and it stays above 0.2 unless n ≥ 0.44√B. So model-free extrapolation needs Θ(√B) attempts per task. Numerically, the width collapses onto n/√B: it is 0.51 at n = √B/2, 0.16 at n = √B and 0.005 at n = 2√B, for B from 64 to 4096. (iii) **Certification budget.** Any protocol, adaptive or not and over any number of tasks, that tells C(B) ≤ τ₀ apart from C(B) ≥ τ₀+Δ with both error rates at most δ must spend Ω((B(1−τ₀)/Δ)·log(1/2δ)) attempts in total. A simple "direct probe" (run each sampled task until its first success or until B attempts) matches the linear dependence on B and is within a factor of 7–25 of the bound in the settings we computed. In simulations over four task-difficulty populations (200 replications per cell), the naive pass@n upper bound covered C(B) in 0–76% of replications. Beta-binomial and power-law extrapolation, both at nominal 95%, fell to 29% and 70% coverage on a heavy-tailed population, and their coverage got *worse* as the number of tasks grew. A model-free linear-programming bound covered in every cell but was often close to vacuous. On public METR Time-Horizon 1.1 run data (20 frontier agents, 228 tasks), 4 runs per task leave the model-free 95% set for a 64-attempt adversary reaching 0.996–1.000 for every model. In a subsample backtest with mild extrapolation (2–4 runs extrapolated to 6–8), the naive bound missed the realised coverage in up to 100% of splits, while parametric and model-free bounds did not miss. Our conclusion is that "below threshold" claims against repeated-attempt adversaries need either an evaluation budget that grows linearly in B or explicitly stated distributional assumptions. We give a reporting protocol for both cases.

## 中文摘要

危险能力评估通常报告评估者对每个任务尝试少数几次（pass@1，或 n 约为 1 到 10 的 pass@n）时的成功率。但真正与风险相关的量不同：滥用者或自主智能体可以对同一任务反复尝试，所以应当测量的是*预算 B 覆盖率* C(B)，即在 B 次尝试内至少成功一次的威胁相关任务所占比例，B 可达数百乃至数千。本文研究两个问题：评估者只能对每个任务尝试 n ≪ B 次时，关于 C(B) 能做出哪些可靠的结论；以及任何评估协议需要多少次尝试才能证明 C(B) 低于某个阈值。我们证明了三个结果。（i）**可识别性**：在尝试可交换的假设下，每个任务 n 次尝试只揭示任务难度分布的前 n 阶矩；C(B) 识别集的最坏情况宽度恰好等于 2E_n(B)，即 [0,1] 上用 n 次多项式一致逼近 x^B 的最佳误差的两倍。（ii）**速率**：该宽度不超过 4·exp(−n²/B)，且除非 n ≥ 0.44√B，否则宽度不低于 0.2，因此无模型外推每个任务需要 Θ(√B) 次尝试。数值计算显示，对 B 从 64 到 4096，宽度只取决于 n/√B：n=√B/2 时为 0.51，n=√B 时为 0.16，n=2√B 时为 0.005。（iii）**认证预算**：任何协议（无论是否自适应、使用多少任务），若要以不超过 δ 的两类错误率区分 C(B) ≤ τ₀ 与 C(B) ≥ τ₀+Δ，总尝试次数至少为 Ω((B(1−τ₀)/Δ)·log(1/2δ))。一个简单的"直接探测"协议（对每个抽样任务一直尝试到首次成功或用满 B 次）在 B 上达到同样的线性依赖，在我们计算的设定中与下界相差 7 到 25 倍。在四种任务难度总体上的模拟中（每格 200 次重复），朴素的 pass@n 上界只在 0–76% 的重复中覆盖 C(B)；名义水平为 95% 的 beta-二项和幂律外推在重尾总体上的覆盖率分别降到 29% 和 70%，且任务数越多覆盖率越差；无模型线性规划界在所有格子中都覆盖了真值，但常常接近无信息。在公开的 METR Time-Horizon 1.1 运行数据上（20 个前沿智能体，228 个任务），每任务 4 次运行时，针对 64 次尝试攻击者的无模型 95% 集合对每个模型都延伸到 0.996–1.000。在外推幅度较小的子样本回测中（由 2–4 次运行外推到 6–8 次），朴素上界在最多 100% 的划分中低于实际覆盖率，参数化界和无模型界则没有出现低估。我们的结论是：针对可重复尝试的对手做出"低于阈值"的声明，要么需要随 B 线性增长的评估预算，要么必须明确写出分布假设。我们为这两种情形分别给出了报告规范。

## 1 Introduction

Frontier AI developers and government institutes now gate deployment decisions on dangerous-capability evaluations [Shevlane et al., 2023; Phuong et al., 2024; Anthropic RSP]. Most of these evaluations end in a threshold judgment of the form "the model does not reach capability level X". Such a judgment is only as good as the match between what was measured and what an adversary can do. One gap is easy to state and has large effects: **evaluators make few attempts per task, while adversaries can make many.** Repeated sampling raises the share of solved problems steeply. Brown et al. [2024] report SWE-bench Lite coverage going from 15.9% with one sample to 56% with 250 samples. Hughes et al. [2024] report best-of-N jailbreak success of 89% on GPT-4o and 78% on Claude 3.5 Sonnet at N = 10,000 augmented prompts. If an adversary has a verifier for success, which is true for many cyber and agentic tasks where success is checkable, the threat-relevant capability is the *coverage at the adversary's budget*, not pass@1.

Evaluators know this. Current practice handles it by reporting pass@n for a small n, or by fitting a scaling curve to pass@k for k ≤ n and extrapolating it to a large budget. Brown et al. [2024] and Hughes et al. [2024] observed power-law-like scaling of −log pass@k. Schaeffer et al. [2025] explained it as the result of a heavy-tailed distribution of per-task single-attempt success probabilities, and used that to forecast the exponent. These extrapolations are useful forecasts. Our question is whether they can support a *safety certificate*: a statement that holds with stated error probability whatever the unknown task-difficulty distribution.

This paper answers that question formally. We model an evaluation as T tasks, each with an unknown per-attempt success probability p drawn from a population distribution G, and n exchangeable attempts per task. The target is C(B) = E_G[1 − (1−p)^B]. The sub-question we solve is:

> *What is the smallest set of values of C(B) consistent with what n attempts per task can reveal, and how many total attempts does any evaluation protocol need to certify C(B) < τ?*

**Contributions.**

1. **An exact identification result (Theorem 1).** The histogram of per-task success counts depends on G only through its first n moments. The worst-case identified set for C(B) has width *exactly* 2E_n(B), where E_n(B) is the best uniform approximation error of x^B by degree-n polynomials on [0,1]. This turns a measurement question into a classical approximation-theory question, and the identified set can be computed by a linear program.
2. **A sharp rate (Theorem 2).** 2E_n(B) ≤ 4e^{−n²/B}, and 2E_n(B) ≤ 0.2 requires n ≥ 0.44√B. Model-free extrapolation to an adversary with B attempts therefore needs Θ(√B) attempts per task. Below that, any "extrapolated certificate" rests on distributional assumptions that the data cannot check. We verify the n/√B collapse numerically for B ∈ {16, …, 4096} (Fig. 1).
3. **An evaluator-budget lower bound with matching achievability in B (Theorem 3).** Certifying C(B) < τ₀ + Δ against C(B) ≤ τ₀ needs Ω((B(1−τ₀)/Δ) log(1/2δ)) total attempts under *any* protocol, including adaptive ones that know task features. The direct probe matches the linear B dependence (Fig. 2).
4. **Empirical evidence of where current estimators fail.** In simulation (Exp. 3), parametric extrapolators hold their nominal coverage when correctly specified and lose it badly (as low as 29%) on heavy-tailed populations, and more data makes this worse. In a real-data backtest on METR Time-Horizon 1.1 runs (Exp. 4), the naive pass@n bound under-covers even small extrapolations, and the model-free answer for a 64-attempt adversary is nearly vacuous for every one of 20 frontier agents.
5. **A reporting protocol** (Section 7) that separates the model-free identified set, assumption-dependent extrapolations, and direct-probe certificates, so a threshold decision states which of these it relies on.

We do not claim that extrapolation is useless. Our claim is narrower and testable: without either Θ(√B) attempts per task (for identification) or Θ(B) attempts per certified task (for testing), a "below threshold" verdict against a B-attempt adversary is a modelling assumption, not a measurement.

## 2 Related Work

**Dangerous-capability evaluation and thresholds.** Shevlane et al. [2023] argued that evaluations for extreme risks are central to AI governance. Phuong et al. [2024] piloted dangerous-capability evaluations (persuasion, cyber, self-proliferation, self-reasoning, CBRN) on Gemini 1.0 models. Developer frameworks such as Anthropic's Responsible Scaling Policy tie required safeguards to capability thresholds. Our work is about the statistical meaning of "below threshold" in these frameworks when the threat actor can retry.

**Elicitation and under-elicitation.** A large literature studies whether evaluations draw out a model's full capability. Greenblatt et al. [2024] stress-test elicitation with password-locked models. van der Weij et al. [2024] show that models can be prompted or fine-tuned to sandbag on dangerous-capability evaluations. Our setting covers a different, purely statistical, kind of under-elicitation: even an honest model is under-measured if the evaluator's attempt budget is much smaller than the adversary's. Our "elicitation floor" assumption (Section 5.3) is one way to make elicitation claims formal.

**Repeated sampling and pass@k.** Chen et al. [2021] introduced the unbiased pass@k estimator for k ≤ n, and noted that no unbiased estimator exists for k > n. Theorem 1 measures how much worse things get: beyond k = n, C(k) is not even identified, and we give the exact size of the identified set. Brown et al. [2024] documented coverage growth over four orders of magnitude of samples. Hughes et al. [2024] observed power-law scaling of best-of-N attack success. Schaeffer et al. [2025] showed that per-task exponential scaling combined with a heavy-tailed distribution of p produces aggregate power laws, and proposed distributional forecasting. That forecasting is a parametric extrapolation of the kind our theory shows must be assumption-dependent below n ≍ √B. Our simulations include a heavy-tailed population for this reason.

**Statistics of evaluations.** Miller [2024] argued for treating evaluations as experiments with standard errors, clustered errors and power analysis. We add a different source of uncertainty that does not shrink with more tasks: identification error coming from the attempt budget.

**Partial identification and moment problems.** Manski [2003] developed bounds-based inference for parameters the data identify only partially. Tian, Kong and Valiant [2017] studied learning a population of Bernoulli parameters from binomial counts. They noted that binomial observations with t trials carry no information beyond the first t moments, and that distributions matching t moments can differ in the (t+1)-th. Theorem 1 applies this moment-matching view to a different functional, 1 − E(1−p)^B, and gives its exact worst-case identified width. Our rate uses the monomial-approximation results in Sachdeva and Vishnoi [2014] and the Markov brothers' inequality.

**Time-horizon measurement.** Kwa et al. [2025] measure the length of tasks AI agents complete at 50% reliability, and METR's public repository releases per-run data for the Time-Horizon 1.1 suite [METR, 2026]. We reuse these runs only as a real-world source of repeated attempts. We do not re-analyse time horizons.

## 3 Setup

**Tasks and attempts.** A threat-relevant task distribution 𝒟 is fixed (for example, a cyber-offense suite). Task i ∼ 𝒟 has a per-attempt success probability p_i ∈ [0,1] under a fixed elicitation (scaffold, prompt, sampling temperature). Attempts on a task are i.i.d. Bernoulli(p_i) given p_i. We call this assumption **(A1) exchangeable attempts**. G denotes the law of p_i when i ∼ 𝒟.

**Estimand.** An adversary with budget B attempts per task (and a success verifier) solves task i with probability 1 − (1−p_i)^B. The budget-B coverage is

  C_G(B) = E_G[1 − (1−p)^B] = 1 − ∫ x^B dG̃(x),  x = 1 − p,

where G̃ is the law of the failure probability x. C_G(1) is pass@1. C_G(B) increases in B, and C_G(∞) = 1 − G({0}) is the share of tasks the model can *ever* solve under this elicitation.

**Evaluation data.** The evaluator runs n attempts on each of T sampled tasks and observes counts X_i ∼ Bin(n, p_i). The histogram h_j = #{i : X_i = j}, j = 0..n, is a sufficient statistic for G under (A1) when task identities carry no further information.

**Certification.** The decision problem is H₀: C_G(B) ≥ τ (dangerous) versus H₁: C_G(B) < τ (safe). A *false-safe* error is declaring H₁ when H₀ holds. A sound evaluator reports a (1−α) upper confidence bound U with P_G(U ≥ C_G(B)) ≥ 1−α *for every G*, and declares safe only if U < τ.

## 4 Theory

### 4.1 What n attempts reveal

Write q_j(G) = P_G(X = j) = ∫ C(n,j) p^j (1−p)^{n−j} dG(p). Each q_j is the integral of a polynomial of degree ≤ n. These polynomials span the space 𝒫_n of polynomials of degree ≤ n, so (q_0, …, q_n) and the moment vector (m_1, …, m_n), m_k = ∫ x^k dG̃, determine each other through an invertible linear map.

**Lemma 1.** Under (A1), two populations G and G′ give the same distribution of the histogram h for every T if and only if ∫ x^k dG̃ = ∫ x^k dG̃′ for k = 1, …, n.

*Proof.* The histogram is multinomial(T; q(G)), and q(G) is an invertible linear function of the first n moments. ∎

Define the identified set I_n(G) = {C_{G′}(B) : G′ matches the first n moments of G} and its width w_n(G) = sup I_n(G) − inf I_n(G). Let

  E_n(B) = min_{P ∈ 𝒫_n} max_{x ∈ [0,1]} |x^B − P(x)|

be the best uniform approximation error of the monomial x^B by polynomials of degree ≤ n on [0,1].

**Theorem 1 (exact worst-case identified width).** For 1 ≤ n < B, sup_G w_n(G) = 2E_n(B).

*Proof.* (≤) Let G, G′ match moments up to n and let P* be the best approximation. Then ∫ P* d(G̃ − G̃′) = 0, so |C_G(B) − C_{G′}(B)| = |∫ (x^B − P*) d(G̃ − G̃′)| ≤ E_n(B) · ‖G̃ − G̃′‖_TV,1 ≤ 2E_n(B).
(≥) By Chebyshev's equioscillation theorem, e(x) = x^B − P*(x) attains ±E_n(B) with alternating signs at n+2 points 0 ≤ x_0 < … < x_{n+1} ≤ 1. Take the divided-difference weights w_k = 1/∏_{j≠k}(x_k − x_j). Then Σ_k w_k Q(x_k) = 0 for every Q ∈ 𝒫_n, and sign(w_k) = (−1)^{n+1−k} alternates. Rescale so that Σ|w_k| = 2. Because constants are annihilated, Σ w_k = 0, so the positive and negative parts each have mass 1. Define G̃ = Σ_{w_k>0} w_k δ_{x_k} and G̃′ = Σ_{w_k<0} |w_k| δ_{x_k}. These are probability measures with equal moments up to order n. Choosing the global sign so that sign(w_k) = sign(e(x_k)), we get ∫ x^B d(G̃ − G̃′) = Σ_k w_k e(x_k) = E_n(B) Σ|w_k| = 2E_n(B). ∎

So the worst case is attained by two populations, each supported on at most n+2 difficulty levels, that no number of tasks can tell apart. For a *specific* G, I_n(G) is the solution of two linear programs over measures (Section 5.1). The worst case is not a pathology either: Fig. 1 and Table 1 show realistic populations with wide identified sets.

### 4.2 How fast the identified set shrinks

**Theorem 2 (rate).** For 1 ≤ n ≤ B:
(a) 2E_n(B) ≤ 4 exp(−n²/B).
(b) If 2E_n(B) ≤ 2ε with ε < (1 − e^{−1})/2, then n² ≥ B (1 − e^{−1} − 2ε) / (2(1+ε)). In particular, width ≤ 0.2 requires n ≥ 0.44 √B.

*Proof.* (a) Sachdeva and Vishnoi [2014, Ch. 3] construct, for integers s and d, the polynomial p_{s,d}(y) = E[T_{D_s}(y) · 1{|D_s| ≤ d}], where D_s is a sum of s independent Rademacher signs and T_k is the k-th Chebyshev polynomial. It satisfies sup_{y∈[−1,1]} |y^s − p_{s,d}(y)| ≤ 2 exp(−d²/(2s)). Take s = 2B and d = 2n. D_{2B} is even, and T_k is even for even k, so p_{2B,2n} is an even polynomial of degree ≤ 2n, i.e. p_{2B,2n}(y) = Q(y²) with deg Q ≤ n. Substituting x = y² ∈ [0,1] gives sup_{x∈[0,1]} |x^B − Q(x)| ≤ 2 exp(−4n²/(4B)) = 2e^{−n²/B}.
(b) Let |x^B − P| ≤ ε on [0,1]. Then max|P| ≤ 1+ε, and by the Markov brothers' inequality on an interval of length 1, max|P′| ≤ 2n²(1+ε). Also P(1) − P(1 − 1/B) ≥ 1 − (1 − 1/B)^B − 2ε ≥ 1 − e^{−1} − 2ε. By the mean-value theorem, 2n²(1+ε)/B ≥ 1 − e^{−1} − 2ε. With ε = 0.1 this gives n² ≥ 0.196B. ∎

**Interpretation.** For an adversary with B = 1000 attempts, model-free extrapolation within ±0.05 needs about 50–60 attempts per task (Table 1: the computed threshold for width ≤ 0.1 lies between n = 32 and n = 48 at B = 1024). Typical dangerous-capability suites run between 1 and about 10 attempts per task. At n = 8 and B = 1024 the worst-case identified width is 0.79.

### 4.3 How many attempts any certificate costs

Theorems 1–2 concern identification from a fixed design. A protocol could instead spend attempts adaptively, for example by concentrating them on tasks that keep failing. Theorem 3 bounds every protocol.

**Theorem 3 (evaluator budget).** Fix τ₀ ∈ [0,1), Δ ∈ (0, 1−τ₀), δ ∈ (0, 1/2), and let W = 1 − τ₀ and c = −ln(1 − Δ/W) < B. Consider any protocol that samples tasks from 𝒟 (it may see task features, choose tasks and allocate attempts adaptively, and use any number of tasks), makes at most N attempts in total, and decides between H₀: C(B) ≤ τ₀ and H₁: C(B) ≥ τ₀ + Δ with both error probabilities ≤ δ for every G. Then

  N ≥ ln(1/(2δ)) / (−ln(1 − c/B)) ≥ (B/c − 1) · ln(1/(2δ)) ≈ (B W / Δ) · ln(1/(2δ)).

*Proof.* Let tasks carry a hidden label: a share W of them are "hard", the rest "easy" (features may reveal the label). Under G₀, hard tasks have p = 0 and easy tasks p = 1, so C(B) = 1 − W = τ₀. Under G₁, hard tasks have p = c/B and easy tasks p = 1, so C(B) = τ₀ + W(1 − (1 − c/B)^B) ≥ τ₀ + W(1 − e^{−c}) = τ₀ + Δ. Couple the two worlds with the same task draws, the same protocol randomness, and attempt outcomes coupled so that a hard-task attempt fails in G₀ and fails in G₁ unless a Bernoulli(c/B) coin comes up heads. The transcripts differ only if some hard-task attempt succeeds in G₁, which has probability ≤ 1 − (1 − c/B)^N. So TV(P₀, P₁) ≤ 1 − (1 − c/B)^N. Any test has error sum ≥ 1 − TV, and requiring this to be ≤ 2δ gives the bound. The second inequality uses −ln(1−u) ≤ u/(1−u). ∎

**Achievability (direct probe).** Sample m tasks and run each until its first success or until B attempts. Each task's outcome is *exactly* Bernoulli(C(B)), with no extrapolation. An exact one-sided binomial test with m = O(log(1/δ)/Δ²) controls both errors, and the cost is at most mB. More precisely, under the construction above it is m(WB + (1 − W)). The dependence on B is therefore optimal. The remaining gap is in Δ: the lower bound scales as 1/Δ and the probe as 1/Δ². Exp. 2 computes the gap.

**Corollary (what a few-attempt protocol cannot do).** Any protocol that spends o(B W/Δ) total attempts can only "certify" C(B) < τ₀ + Δ by relying on an assumption that excludes populations like G₁: many tasks that are solvable but rare. This is exactly the heavy tail that Schaeffer et al. [2025] identify as the source of power-law scaling in real models.

## 5 Estimators Compared

All estimators output a 95% upper confidence bound (UCB) on C(B) from the histogram h (code: `code/core.py`).

1. **Naive pass@n.** A Clopper–Pearson UCB on the share of tasks with ≥1 success in n attempts. It ignores B and stands for the common practice of reporting the evaluator's own pass@n.
2. **Power-law extrapolation.** Fit log(−log pass@k) = a + b log k on k = 1..n, using the Chen et al. [2021] estimator for pass@k, and evaluate at k = B. The UCB is the 95th percentile over a task-level bootstrap. This follows the power-law practice of Brown et al. [2024] and Hughes et al. [2024].
3. **Beta-binomial extrapolation.** Fit G = Beta(a, b) by maximum likelihood on h and compute C(B) = 1 − Beta(a, b+B)/Beta(a, b). The UCB is the bootstrap 95th percentile. This is a simple version of distributional forecasting in the spirit of Schaeffer et al. [2025].
4. **Model-free LP (ours).** Let [ℓ_j, u_j] be Bonferroni-simultaneous Clopper–Pearson 95% intervals for the histogram cell probabilities q_j. The UCB is max Σ_g w_g (1 − (1 − p_g)^B) over probability weights w on a 400-point grid of p values (log-dense near 0, with p = 0 included), subject to ℓ_j ≤ Σ_g w_g Bin(j; n, p_g) ≤ u_j. Coverage holds whenever the simultaneous box contains q(G), which has probability ≥ 95%, up to the grid approximation. With q known exactly, the LP returns the identified set I_n(G) of Theorem 1.

### 5.3 An assumption that can restore identification

The **elicitation floor (A2)** says that every task is either impossible under this elicitation (p = 0) or has p ≥ p_min. Under (A2), C(B) ≥ C(∞) − (1 − p_min)^B, so for B ≫ 1/p_min the target reduces to the "ever solvable" share. We implement (A2) by restricting the LP grid to {0} ∪ [p_min, 1]. (A2) cannot be checked from n ≪ 1/p_min attempts, so it has to be argued from domain knowledge. Exp. 3 reports how much of each simulated population violates it.

## 6 Experiments and Results

All numbers below come from code in `code/` that we ran in this environment. Outputs are in `results/` (file names given). CPU time: Exp. 1 about 3 min, Exp. 2 under 1 min, Exp. 3 599 s (`results/exp3_log.txt`), Exp. 4 255 s (`results/exp4_log.txt`).

### 6.1 Exp. 1: identification width (Theorems 1–2)

We computed E_n(B) by a minimax linear program on a grid of about 3,600 points in [0,1] (Chebyshev extrema plus points clustered near x = 1), using a Chebyshev basis (`results/exp1a_approx_error.json`).

**Table 1.** Worst-case identified width 2E_n(B) (computed) by attempts per task n.

| n/√B | B=64 | B=256 | B=1024 | B=4096 | bound 4e^{−n²/B} |
|---|---|---|---|---|---|
| 1/2 | 0.512 (n=4) | 0.514 (n=8) | 0.514 (n=16) | 0.515 (n=32) | 1 (vacuous) |
| 1 | 0.162 (n=8) | 0.164 (n=16) | 0.164 (n=32) | 0.164 (n=64) | 1 (vacuous) |
| 3/2 | 0.034 (n=12) | 0.035 (n=24) | 0.035 (n=48) | 0.035 (n=96) | 0.42 |
| 2 | 0.0044 (n=16) | 0.0047 (n=32) | 0.0048 (n=64) | 0.0048 (n=128) | 0.073 |

The widths collapse onto a single curve in n/√B (Fig. 1, `results/fig1_identification_width.png`). This confirms the √B scaling of Theorem 2. The upper bound is valid but loose by roughly one order of magnitude at n = 2√B. For widths below about 10⁻⁹ the LP solver returns 0; this is a numerical floor and we do not interpret those values. The smallest grid n with width ≤ 0.1 was 6, 12, 24, 48, 96 for B = 16, 64, 256, 1024, 4096 (`results/exp1a_nmin.json`), i.e. 1.5√B in each case. Because n was evaluated on a grid, the true threshold lies in (√B, 1.5√B].

**Population-limit identified sets.** For four populations (40,000 atoms each; `code/exp1_theory.py: populations`) we solved the LP with the exact histogram probabilities q (slack 10⁻⁴). The populations are: *beta(0.2,2)* (smooth, mass near 0); *zero+beta* (35% impossible tasks, the rest Beta(0.8,0.6)); *hard-tail* (40% of tasks with p log-uniform on [10⁻⁴, 10⁻¹]); and *locked* (25% impossible, 25% at p = 3·10⁻³, the rest uniform). Table 2 (`results/exp1b_population_identified.json`) shows that even with infinitely many tasks, n = 16 attempts per task leave very wide intervals at B = 1000.

**Table 2.** True C(B) and identified interval with infinitely many tasks (computed).

| population | B | C(B) | n=4 | n=16 | n=64 |
|---|---|---|---|---|---|
| beta(0.2,2) | 100 | 0.561 | [0.349, 0.933] | [0.527, 0.604] | [0.561, 0.561] |
| beta(0.2,2) | 1000 | 0.722 | [0.349, 1.000] | [0.534, 0.993] | [0.648, 0.843] |
| zero+beta | 1000 | 0.643 | [0.571, 1.000] | [0.627, 0.891] | [0.638, 0.676] |
| hard-tail | 1000 | 0.892 | [0.557, 1.000] | [0.729, 0.999] | [0.822, 0.950] |
| locked | 1000 | 0.740 | [0.452, 1.000] | [0.538, 0.942] | [0.699, 0.857] |

### 6.2 Exp. 2: evaluator budget (Theorem 3)

With τ₀ = 0.5 and δ = 0.05, we computed the lower bound N_lower and the cost of the direct probe under the hard instance, N_probe = m(WB + (1−W)), with m the smallest sample size for an exact binomial test with both errors ≤ δ (`results/exp2_budget.json`, Fig. 2). For Δ = 0.1: at B = 100, N_lower = 1,031 and N_probe = 13,534 (m = 268). At B = 10,000, N_lower = 103,187 and N_probe = 1,340,134. Both grow linearly in B, and the ratio settles near 13.0. The ratio was about 25 for Δ = 0.05 (m = 1,092) and about 7.4 for Δ = 0.2 (m = 67), matching the 1/Δ gap predicted by theory. A Monte Carlo check of the total-variation step (4,000 replications) gave a detection probability of 0.1995 against a bound of 0.2000 (B = 1000, N = 2000), and 0.890 against 0.893 (B = 1000, N = 20,000). The bound is essentially tight for the hard construction.

Concretely: certifying that a model's 1000-attempt coverage of a cyber suite is below 60% when the null coverage is 50% requires *at least* about 10⁴ agent attempts under any protocol. The direct probe needs about 1.3·10⁵. An evaluation with 5 attempts on each of 200 tasks (10³ attempts) cannot certify this against a B = 1000 adversary without assumptions.

### 6.3 Exp. 3: coverage of UCBs in simulation

**Design.** We used the 4 populations of Section 6.1, n ∈ {4, 16}, T ∈ {200, 1000} and B ∈ {100, 1000}, with 200 replications per (population, n, T) cell. Bootstraps used 40 resamples. "Coverage" is the share of replications with UCB ≥ C(B). Nominal coverage is 95%. Data: `results/exp3_sim_summary.json`. Fig. 3: `results/fig3_sim_coverage.png`.

**Results** (32 population × n × T × B cells):

- **Naive pass@n** covered in 0% of replications in 23 of 32 cells, with a maximum of 76% (zero+beta, n = 16, T = 200, B = 100). Reporting the evaluator's own pass@n as the threat-relevant capability is unsafe whenever B > n.
- **Beta-binomial extrapolation** covered at about nominal level on beta(0.2,2), where it is correctly specified (91–95%). On hard-tail with B = 1000 coverage fell to 81% (n=4, T=200), 65% (n=4, T=1000), 65% (n=16, T=200) and **29%** (n=16, T=1000). On zero+beta and locked it was *conservative but biased*: its point estimate exceeded the truth by up to 0.23 (zero+beta, n=4, T=1000, B=1000: 0.875 versus 0.647).
- **Power-law extrapolation** behaved similarly. On hard-tail with B = 1000 and n = 16, coverage was 81% (T=200) and **70%** (T=1000). Elsewhere it was mostly ≥ 96%, often because it over-predicted.
- **More data hurts misspecified extrapolators.** On hard-tail with n = 16 and B = 1000, increasing T from 200 to 1000 lowered beta-binomial coverage from 65% to 29%, and power-law coverage from 81% to 70%. Misspecification bias does not shrink with T, while the bootstrap interval does.
- **Model-free LP** covered in 100% of replications in all 32 cells. Its mean UCB ranged from 0.75 (locked, n=16, T=1000, B=100; truth 0.560) to essentially 1.00 for every B = 1000 cell with n = 4. It is safe and, in the n ≪ √B regime, almost uninformative, as Theorems 1–2 predict.
- **Elicitation floor (A2).** Adding p_min ∈ {10⁻², 10⁻³} changed the LP UCB only slightly in these finite-T settings. The largest effect was locked, n=16, T=1000, B=1000, where the UCB dropped from 1.00 to 0.89 with p_min = 10⁻². Three of the four populations violate (A2) at p_min = 10⁻² for 25–48% of tasks (`floor_violation_mass`). At these (n, T), the limiting factor is the sampling uncertainty of the histogram more than the support assumption.

### 6.4 Exp. 4: real repeated-attempt data (METR Time-Horizon 1.1)

**Data.** We used `runs.jsonl` of the Time-Horizon 1.1 report in METR's public repository [METR, 2026]: 24,008 runs with binarised scores on 228 software and research tasks [Kwa et al., 2025]. We dropped human baselines. Many (agent, task) pairs have 4, 6 or 8 runs. These tasks are general autonomy tasks, not dangerous-capability tasks. We use them only as real data on how success varies across attempts.

**(a) Subsample backtest.** For each agent with at least 60 tasks having ≥ 8 runs (4 agents), or at least 140 tasks having ≥ 6 runs (6 agents), we kept n_full ∈ {8, 6} runs per task (seeded random choice). The target is the *realised* share of those tasks with ≥ 1 success in n_full runs. The evaluator sees k ∈ {2, 4} (n_full = 8) or k ∈ {2, 3} (n_full = 6) runs per task and extrapolates to B = n_full. There were 60 random subsets per configuration and 100 bootstrap resamples (`results/exp4_metr.json`, Fig. 4).

- **Naive pass@k** under-covered substantially. For o1-preview with k = 2, its UCB was below the realised coverage in 100% of splits for both n_full = 8 (target 0.810, mean UCB 0.762) and n_full = 6 (target 0.733). Coverage was 53% for GPT-4 0314 (k=2→6), 63% for GPT-5.3-Codex (k=2→6), 75% for Claude Opus 4.6 (k=2→6), and 58% for o1-preview (k=4→8).
- **Power-law and beta-binomial** UCBs covered in ≥ 95% of splits in all 20 configurations (minimum 95%: beta-binomial, o1-preview, k=2→8). Their point estimates were within about 0.025 of the realised target, except for one degenerate beta-binomial MLE fit (GPT-4 0314 with n_full = 8, where 98.5% of tasks were solved and the point estimate was off by −0.49, while its bootstrap UCB still covered). This is the *mild-extrapolation* regime: B/k ≤ 4 and k ≥ √B for k ∈ {3, 4}. The theory predicts identification is reasonable there, and the data cannot test large-B extrapolation.
- **Model-free LP** covered in 100% of splits in all configurations. Its excess over the target ranged from +0.015 (GPT-4 0314, n_full = 8, near ceiling) to +0.229 (GPT-4 0314, k=2→6), typically about +0.10 to +0.18.

**(b) What 4 runs say about a 64-attempt adversary.** For all 20 agents (210–228 tasks with ≥ 4 runs each), Fig. 5 shows the model-free 95% set for C(64) next to the parametric point predictions (`results/fig5_metr_identified.png`). The model-free upper end is 0.996–1.000 for every agent. The lower end equals the observed-ever-solved lower confidence limit (for example 0.428 for GPT-4 0314 and 0.822 for Claude Opus 4.6). The two parametric extrapolators gave different predictions on the same data, differing by up to 0.047 (Claude 3.5 Sonnet (Old): beta-binomial 0.779, power-law 0.826). For C(16), the model-free set is already wide: [0.428, 0.842] for GPT-4 0314 and [0.822, 0.997] for Claude Opus 4.6.

## 7 Discussion: a Reporting Protocol

The results suggest a simple rule: **a capability-threshold claim against a B-attempt adversary should state which of three regimes it is in.**

1. **Direct certificate (n ≥ B on probed tasks).** Run m sampled tasks to first success or B attempts and report the exact binomial UCB. No assumptions beyond (A1) and task sampling are needed. Cost ≤ mB. Theorem 3 says no protocol can do better than linear in B.
2. **Identified extrapolation (n ≳ 1.5√B per task).** Report the model-free LP interval. Table 1 shows its worst-case width is ≤ 0.035 at n = 1.5√B, plus finite-T sampling error, which the LP accounts for.
3. **Assumption-based extrapolation (n ≪ √B).** Report the model-free interval *alongside* the parametric forecast, and name the assumption (power law, beta mixture, elicitation floor). A threshold decision made here is a modelling judgment. Exp. 3 shows that its error can exceed the nominal level by a large margin when the tail of hard-but-solvable tasks is heavier than assumed.

**Budget arithmetic.** The rule turns into evaluation budgets. For B = 1000, regime 2 needs about 48 attempts per task (Table 1) on enough tasks to pin down the histogram. For regime 1 with Δ = 0.1, the direct probe needs about 268 tasks × up to 1000 attempts. Both are expensive for long agentic tasks. That cost is the real price of a model-free "below threshold" claim against a persistent adversary, and the theory says it cannot be avoided.

**Where effort helps most.** Theorem 3's hard instance is a mass of tasks with p ≈ 1/B. Effort is best spent finding out whether such tasks exist. One option is adaptive "persistence probes": continue attempts only on tasks with zero successes, which costs about B per such task. These tasks, not the easy ones, carry the false-safe risk. This also links statistical under-elicitation to the elicitation literature [Greenblatt et al., 2024; van der Weij et al., 2024]: a stronger elicitation (better scaffold, fine-tuning) moves mass out of the p ≈ 1/B region and makes (A2) more plausible, and that kind of improvement is visible in the data.

**Falsifiable predictions.** (P1) In any repeated-sampling dataset with N_full ≥ 4√B_test attempts per task, the coverage of nominal-95% parametric UCBs computed from n ≪ √B_test attempts will *decrease* as the number of tasks grows whenever the heavy tail is mis-modelled. (P2) The model-free LP width computed from real data will scale with n/√B as in Table 1. A test protocol: take public repeated-sampling datasets with thousands of samples per problem (such as the samples released by Brown et al. [2024]), subsample n ∈ {2, …, 64}, and compare UCBs to the realised C(B) for B up to 10⁴. We did not run this, because those datasets require downloads larger than our budget and our METR data has at most 8 runs per task.

## 8 Limitations

- **Exchangeability (A1) is strong.** Real adversaries adapt between attempts, for example by using feedback or changing prompts, so their attempts are not i.i.d. draws from the evaluator's elicitation. If adaptation raises later success probabilities, C(B) under (A1) *understates* the threat. Our impossibility results remain valid lower bounds on difficulty, but our achievability (the direct probe) certifies only i.i.d.-retry adversaries. Adaptive-adversary evaluation needs separate red-teaming.
- **Tasks are treated as i.i.d. draws from a fixed 𝒟.** Benchmark tasks come in families (METR tasks are clustered by family). This adds variance that our CP boxes and bootstrap ignore, so real-data coverage claims are optimistic in this respect. The external validity of 𝒟 for real threats is outside our scope.
- **Binary success with a verifier.** We assume the adversary can recognise success. Without a verifier, best-of-B coverage overstates what the adversary can actually achieve.
- **Theorem 3 is tight in B but loose by a factor of about 1/Δ.** The direct probe is 7–25× above the bound in our settings. Closing the gap (for example, a bound that also charges for task sampling, which scales as 1/Δ²) is open.
- **The rate constants in Theorem 2 are loose.** The 4e^{−n²/B} bound is about 15× above the computed width at n = 2√B, and for width ≤ 0.2 the 0.44√B lower bound is within a factor of about 2.3 of the computed sufficient value n = √B (width 0.16). The exact limiting profile of 2E_n(B) as a function of n/√B is computed here, not derived.
- **The real-data backtest covers only mild extrapolation** (to at most 8 attempts), because the data has at most 8 runs per task. It cannot confirm the large-B failures we see in simulation. The METR tasks are software and research tasks, not dangerous-capability tasks, and one parametric fit (beta-binomial, near-ceiling data) was numerically degenerate.
- **The simulated populations are illustrative.** They were chosen to cover smooth, zero-inflated, heavy-tailed and "locked" structure. They are not fitted to any dangerous-capability dataset. Bootstrap counts (40 in simulation, 100 on real data) are modest, and LP coverage depends on a finite grid of 400 p-values.
- **No large-model experiments.** We ran no LLMs. All model data comes from METR's public runs.

## 9 Conclusion

Measuring a model's risk-relevant capability against an adversary who can retry is a partial-identification problem with an exact, computable answer. With n attempts per task, the threat-relevant coverage C(B) is identified only up to a set of worst-case width 2E_n(B). That width stays large until n ≈ √B and becomes negligible by n ≈ 2√B. Certifying that C(B) is below a threshold costs Ω(B) attempts under any protocol, and a simple direct probe achieves this order. Common practice falls between these regimes: reporting pass@n under-covers, and parametric extrapolation can be badly overconfident on heavy tails. We recommend that threshold claims state their regime (direct, identified or assumption-based) and report the model-free interval next to any forecast.

## Reproducibility

`code/README.md` lists the commands. `exp1_theory.py`, `exp2_budget.py`, `exp3_sim.py` (run with REPS=200), `exp4_metr.py` and `make_figs.py` write all JSON results and figures 1–5 to `results/`. Exp. 4 requires a clone of github.com/METR/eval-analysis-public under `ext/metr`.

## References

- Anthropic. *Responsible Scaling Policy* (versions 2.x–3.x). https://www.anthropic.com/responsible-scaling-policy (accessed November 2026).
- Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. (2024). Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv:2407.21787.
- Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Greenblatt, R., Roger, F., Krasheninnikov, D., and Krueger, D. (2024). Stress-Testing Capability Elicitation With Password-Locked Models. NeurIPS 2024. arXiv:2405.19550.
- Hughes, J., Price, S., Lynch, A., et al. (2024). Best-of-N Jailbreaking. arXiv:2412.03556 (NeurIPS 2025).
- Kwa, T., West, B., Becker, J., et al. (2025). Measuring AI Ability to Complete Long Tasks (v3 title: Measuring AI Ability to Complete Long Software Tasks). arXiv:2503.14499.
- Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer Series in Statistics. doi:10.1007/b97478.
- Markov, A. A. (1889). On a question by D. I. Mendeleev (Markov brothers' inequality); see Encyclopedia of Mathematics, "Markov inequality", https://encyclopediaofmath.org/wiki/Markov_inequality.
- METR (2026). eval-analysis-public: analysis code and data for METR's time horizon methodology, report time-horizon-1-1. https://github.com/METR/eval-analysis-public ; announcement: https://metr.org/blog/2026-1-29-time-horizon-1-1.
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640.
- Phuong, M., Aitchison, M., Catt, E., et al. (2024). Evaluating Frontier Models for Dangerous Capabilities. arXiv:2403.13793.
- Sachdeva, S., and Vishnoi, N. K. (2014). Faster Algorithms via Approximation Theory. *Foundations and Trends in Theoretical Computer Science* 9(2):125–210. doi:10.1561/0400000065.
- Schaeffer, R., Kazdan, J., Hughes, J., Juravsky, J., Price, S., Lynch, A., Jones, E., Kirk, R., Mirhoseini, A., and Koyejo, S. (2025). How Do Large Language Monkeys Get Their Power (Laws)? ICML 2025, PMLR 267:53132–53176. arXiv:2502.17578.
- Shevlane, T., Farquhar, S., Garfinkel, B., et al. (2023). Model evaluation for extreme risks. arXiv:2305.15324.
- Tian, K., Kong, W., and Valiant, G. (2017). Learning Populations of Parameters. NeurIPS 2017, 5778–5787. arXiv:1709.02707.
- van der Weij, T., Hofstätter, F., Jaffe, O., Brown, S. F., and Ward, F. R. (2024). AI Sandbagging: Language Models can Strategically Underperform on Evaluations. arXiv:2406.07358 (ICLR 2025).
