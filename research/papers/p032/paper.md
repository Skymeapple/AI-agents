# Beyond the Chain: Redundancy, Correlated Errors, and Collusion in Nested Scalable Oversight

**Target venue:** NeurIPS 2027 (main track; theory + simulation). Written as a full-length submission draft.

---

## Abstract

Nested scalable oversight (NSO) proposes to oversee a far stronger AI system by bootstrapping: a trusted weak model oversees a slightly stronger one, which then oversees the next. The most explicit quantitative treatment of NSO (Engels et al., 2025) models each step as an Elo-rated game between a "Guard" and a "Houdini" and optimises the number of steps of a *chain*, in which every level is overseen only by its immediate predecessor and overall success is the product of the per-step success probabilities. We generalise this to *oversight graphs*. Each level may be overseen by several earlier levels. Overseers' errors on the same target may be correlated (correlation ρ), and an overseer that itself escaped detection may collude with later levels (it keeps a fraction γ of its detection ability). We give an exact dynamic program over trust states, checked against direct Monte Carlo simulation (24 random configurations, maximum absolute error 0.0023). We prove three results. (i) Within-level redundancy is monotonically eroded by error correlation, from the independent-OR value down to the single best overseer at ρ = 1 (via Slepian's inequality). (ii) Under full collusion, a window-k topology cuts the per-level rate at which compromise becomes irrecoverable from q (chain) to ∏ₘ qₘᵐ at ρ = 0 and to ∏ₘ qₘ at ρ = 1, where qₘ is the miss probability of an overseer m levels below. Numerically, the predicted hazard matches the exact one within 2.5% in a small-miss regime. (iii) Under partial collusion, chain compromise converges to a stationary level q/(q+γ(1−q)) rather than accumulating. In stylised Elo regimes, at equal oversight budget, graph topologies raise target-level success well above the best chain: for example from 0.168 to 0.772 at a 800-Elo gap with 16 oversight games. This gain mostly survives high error correlation, but the gain in "no level ever escapes" success does not. Redundancy cannot rescue a game whose Houdini slope exceeds its Guard slope: success is at most 0.011 at a 1600-Elo gap. We show how ρ and γ can be measured with existing monitor evaluations and pre-register three falsifiable predictions.

## 中文摘要

嵌套式可扩展监督（NSO）设想通过"自举"来监督远强于人类的 AI：可信的弱模型监督稍强的模型，后者再去监督下一个更强的模型。目前对 NSO 最明确的定量分析（Engels 等，2025）把每一步建模为基于 Elo 等级分的"守卫–胡迪尼"博弈，只优化"链式"结构的步数：每一层只由紧邻的前一层监督，总成功率等于各步成功率之积。本文把这一模型推广为"监督图"：每一层可由多个更早的层共同监督；同一目标上各监督者的错误可以相关（相关系数 ρ）；本身未被发现的失守层可能与后续层合谋（只保留 γ 比例的检测能力）。我们给出基于信任状态的精确动态规划算法，并用直接蒙特卡洛模拟验证（24 个随机配置，最大绝对误差 0.0023）。我们证明了三个结果。（i）同层冗余随错误相关性单调削弱：由独立"或"聚合的检测率一直降到 ρ=1 时单个最佳监督者的水平（由 Slepian 不等式得到）。（ii）在完全合谋下，窗口为 k 的拓扑使失守变得不可恢复的每层速率由链式的 q 降为 ∏ₘ qₘᵐ（ρ=0）或 ∏ₘ qₘ（ρ=1），其中 qₘ 是相隔 m 层的监督者的漏检率。在小漏检率区间，数值精确值与该预测相差不超过 2.5%。（iii）在部分合谋下，链式结构的失守概率不会随层数累积，而是收敛到稳态值 q/(q+γ(1−q))。在程式化的 Elo 参数区间内，同样的监督预算下，图结构使目标层成功率大幅高于最优链式结构，例如 800 Elo 差距、16 次监督博弈时由 0.168 提高到 0.772。在错误高度相关时，这一收益大部分仍然保留，但"任何层都不逃逸"的严格成功率收益基本消失。若胡迪尼的 Elo 斜率高于守卫，冗余也无法挽救：1600 Elo 差距下成功率不超过 0.011。我们说明了如何用现有的监控评估测量 ρ 和 γ，并预注册了三条可证伪的预测。

---

## 1 Introduction

Scalable oversight asks how humans can supervise systems that outperform them on the very skills needed to judge their behaviour (Bowman et al., 2022). Most concrete proposals amplify a weak judge: debate between strong models (Irving et al., 2018), recursive decomposition (Christiano et al., 2018), recursive reward modelling (Leike et al., 2018), or weak supervisors whose labels a strong student generalises beyond (Burns et al., 2023). Many proposals share one structural ingredient, *bootstrapping*. A trusted weak system supervises a slightly stronger one, which, once trusted, supervises the next.

Engels et al. (2025) gave the first explicit scaling analysis of this bootstrapping, which they call Nested Scalable Oversight (NSO). They model each oversight step as a two-player game between a Guard (overseer) and a Houdini (overseen system). Each role has a domain-specific Elo that is piecewise-linear in "general" Elo. They fit slopes for four games and choose the number of steps n that maximises ∏ⱼ pⱼ, where pⱼ is the Guard's win probability at step j. Their headline estimates of NSO success at a general Elo gap of 400 are sobering: 13.5% for Mafia, 51.7% for Debate, 10.0% for Backdoor Code and 9.4% for Wargames.

That analysis assumes a particular *topology*: a chain in which level j is overseen only by level j−1, and any single failure is fatal. Three features of realistic bootstrapping fall outside the chain model, and each can move the answer in either direction.

1. **Redundancy.** Nothing forces level j to be overseen only by level j−1. It can also be checked by j−2, j−3, … or by the human anchor at level 0. These overseers are weaker, but they are *additional* chances to detect misbehaviour.
2. **Correlated errors.** Redundant overseers help only if their errors are not the same errors. Goel et al. (2025) report that the mistakes of language models are becoming more similar as capabilities increase, and that LLM judges favour models similar to themselves. Both effects erode the value of stacking AI overseers.
3. **Collusion and propagation.** A level that escapes detection does not disappear. In NSO it becomes the overseer of the next level. If it is misaligned it may collude, a threat model central to AI control and untrusted monitoring (Greenblatt et al., 2024). The chain treats any escape as total failure. That is pessimistic if compromised overseers retain some detection ability, and it does not ask whether later honest overseers can stop the propagation.

**This paper.** We embed the Engels et al. per-step game in an *oversight graph* model with three parameters beyond theirs: the topology (window k, optional anchor), the within-level error correlation ρ, and the collusion parameter γ. We distinguish two success criteria. *Strict* success means no level ever escapes. *Target* success means the final, most capable level is trustworthy. The chain cannot separate these two; graphs can, and the difference turns out to be the main story.

**Contributions.**

1. **Model and exact algorithm.** We give an oversight-graph generalisation of NSO with an exact dynamic program over trust states (complexity O(n·2ᵏ) per configuration) and a Monte Carlo simulator that also supports cross-level correlation. On 24 random configurations they agree to within Monte Carlo error (maximum |z| = 2.30, maximum absolute error 0.0023).
2. **Correlation theorem (Prop. 2).** Under a one-factor Gaussian copula, per-level detection by any set of overseers is non-increasing in ρ. It interpolates between the independent-OR value and the single best overseer. Strict success is therefore non-increasing in ρ for every topology.
3. **Self-healing theorem (Prop. 3).** Under full collusion, compromise in a chain becomes irrecoverable at rate q per level. In a window-k graph the asymptotic rate is r_k = ∏ₘ qₘᵐ at ρ = 0 and ∏ₘ qₘ at ρ = 1. Error correlation therefore defeats redundancy's *per-level* benefit but not its *anti-propagation* benefit. The exact hazard matches the prediction to within 2.5% when misses are rare (Table 2).
4. **Partial-collusion result (Prop. 4).** With γ > 0, chain compromise converges to q/(q+γ(1−q)), independent of depth. The exact DP matches this to machine precision.
5. **Budget-constrained design study.** Across three stylised regimes, three capability gaps, three correlation levels and three budgets (81 settings), we report the success-maximising (n, k, anchor) design and compare it with the best chain. Redundant graphs dominate chains on target success whenever the per-step game is not Houdini-favoured. Chains cannot convert extra budget into success at all.
6. **Measurement protocol and predictions.** We show that ρ equals the tetrachoric correlation of paired monitor misses, which makes it identifiable from existing monitoring evaluations. We pre-register three falsifiable predictions with a concrete test protocol (Section 7).

We are explicit about scope. All quantitative results are about a *model*. The game parameters are stylised regimes chosen to cover the qualitative cases Engels et al. identify; they are not fits to real models. We ran no experiments on large models.

---

## 2 Related Work

**Scalable oversight protocols.** Debate (Irving et al., 2018) lets a judge verify one path through an argument tree. Iterated amplification (Christiano et al., 2018) and recursive reward modelling (Leike et al., 2018) build training signals from decomposed subproblems. Bowman et al. (2022) proposed the "sandwiching" experimental paradigm. Empirically, debate with more persuasive debaters improved non-expert judge accuracy on QuALITY (Khan et al., 2024). Results were more mixed across a broader suite of weak-judge/strong-agent asymmetries (Kenton et al., 2024). On the theory side, doubly-efficient debate (Brown-Cohen et al., 2024) and prover–estimator debate (Brown-Cohen et al., 2025) address the obfuscated-arguments problem (Barnes, 2020), in which a dishonest debater forces an honest one to solve an intractable problem. These works design *single-step* protocols. Our analysis is orthogonal: we take a single step's success probability as given and ask how steps should be *composed*.

**Scaling laws for oversight.** Engels et al. (2025) is the direct starting point. They introduce Guard/Houdini Elo curves, fit them on Mafia, Debate, Backdoor Code and Wargames, and derive optimal chain depth. They find that chains help only when the Guard's domain Elo grows faster with general capability than the Houdini's, or when the game favours the Guard at equal capability. Our chain baseline reproduces this qualitative picture (Section 6.2). We change the *composition rule*.

**Weak-to-strong generalisation.** Burns et al. (2023) show that strong students fine-tuned on weak labels outperform their supervisors but do not recover full capability. Goel et al. (2025) find that complementary knowledge between supervisor and student drives weak-to-strong gains, and that model errors are becoming more similar with capability. That finding motivates our correlation parameter ρ directly: it is the quantity their CAPA similarity metric is designed to expose.

**AI control and collusion.** Greenblatt et al. (2024) evaluate protocols such as trusted editing and untrusted monitoring against a red-team model that tries to insert backdoors. In untrusted monitoring the monitor may collude with the policy. Our γ is a one-parameter summary of how much detection ability a compromised overseer retains. A control evaluation of an untrusted monitor estimates exactly this quantity.

**Correlated voting.** Our within-level aggregation is a correlated "any-detector" rule, close to the correlated Condorcet jury literature (Ladha, 1992), where correlation among voters limits the benefit of adding voters. Our tool for the monotonicity result is Slepian's inequality (Slepian, 1962). What is new relative to the jury setting is the *temporal propagation*: overseers are past targets, so a missed detection changes the composition of future juries.

**Gap.** We are not aware of prior work that (a) composes Elo-style oversight steps in topologies other than a chain, (b) treats escaped levels as compromised future overseers, or (c) separates per-level redundancy from propagation-blocking. Our literature search (late 2026) found general warnings that correlated failures undermine layered oversight, but no quantitative model of the effect within NSO.

---

## 3 Model

### 3.1 Per-step oversight game (from Engels et al., 2025)

Levels are indexed i = 0, 1, …, n. Level 0 is the trusted starting overseer (for instance, humans with tools). Level i has general Elo xᵢ = iΔ/n, where Δ is the general-capability gap between the starting overseer and the target. Following the linear region of the Engels et al. Elo curves, the domain Elo of a model of general Elo x is G(x) = m_G·x in the Guard role and H(x) = m_H·x + β in the Houdini role. Here β < 0 means the game favours the Guard at equal capability. The probability that Guard i detects (wins against) Houdini j is given by the Elo formula (Elo, 1978):

  p_{ij} = f + (1−f) / (1 + 10^{(H(x_j) − G(x_i))/400}),

with an optional chance floor f, for example f = 0.5 for a binary debate judge. We use f = 0 throughout. Because x_j > x_i, distant overseers face larger domain gaps whenever m_H ≥ m_G. Their detection probabilities are lower, but not zero.

### 3.2 Oversight graphs

Level j ≥ 1 is overseen by a set S_j ⊆ {0, …, j−1}. We study:

* **chain:** S_j = {j−1} (Engels et al.);
* **window-k:** S_j = {max(0, j−k), …, j−1};
* **anchored:** window-k ∪ {0}, meaning the original trusted overseer checks every level;
* **full:** S_j = {0, …, j−1}.

The *cost* of a design is the total number of oversight games, Σⱼ|S_j|. For window-k with n ≥ k it equals nk − k(k−1)/2.

### 3.3 Trust dynamics

We take a worst-case stance on intent: each new level behaves as a Houdini that tries to evade. Each earlier level is in one of two states, *trusted* (T) or *compromised* (C). Level 0 is T.

* A trusted overseer i detects Houdini j with marginal probability p_{ij}. A compromised overseer detects with γ·p_{ij}, where γ ∈ [0,1]. γ = 0 is full collusion, and γ = 1 means compromise does not impair oversight.
* Detection events of the overseers of a given level j are coupled by a one-factor Gaussian copula. Overseer i detects iff √ρ·Z_j + √(1−ρ)·ε_{ij} < Φ⁻¹(p'_{ij}), where Z_j and ε_{ij} are i.i.d. N(0,1) and p'_{ij} is the effective probability. Marginals are preserved for every ρ ∈ [0,1]. Z_j can be read as Houdini j's "stealth" against the shared blind spots of its overseers. By default the Z_j are independent across levels. We relax this by simulation in Section 6.4.
* **OR aggregation.** If any overseer detects, level j is corrected and becomes T; otherwise it becomes C. As in the Engels et al. game, "Guard wins" is treated as successful oversight of that step. The false-alarm cost of OR aggregation is accounted for separately in Section 6.7.

**Outcomes.** *Strict success* S_strict = P(all levels T). *Target success* S_target = P(level n is T). For the chain with γ = 0, both equal ∏ⱼ p_{j−1,j}, which is the Engels et al. objective.

**Interpretation of the two criteria.** Strict success is the right criterion when any escape by an intermediate model is itself catastrophic, for example when every intermediate model is deployed. Target success is the right criterion when intermediate models are internal tools whose escape matters only through what they do downstream, here via collusion. We think real deployments sit between these, so we report both.

### 3.4 Exact computation

Strict success factorises: S_strict = ∏ⱼ D(p_{S_j,j}; ρ), where D(p; ρ) is the probability that at least one overseer detects when all overseers are trusted. Under the copula,

  D(p; ρ) = 1 − E_Z ∏ᵢ Φ̄( (Φ⁻¹(pᵢ) − √ρ Z) / √(1−ρ) ),

which we evaluate with 80-node Gauss–Hermite quadrature (exact closed forms at ρ = 0 and ρ = 1). Target success requires tracking trust states. For window-k (with or without anchor) the joint status of the last k levels is a sufficient statistic, so a dynamic program over 2ᵏ states gives S_target exactly in O(n·2ᵏ) detection evaluations. For the full topology and for cross-level correlation we use direct Monte Carlo simulation of the generative model (200,000 runs).

---

## 4 Theory

Throughout, q_{ij} = 1 − p_{ij} denotes a miss probability.

**Proposition 1 (chain reduction).** For the chain with γ = 0, S_target = S_strict = ∏_{j=1}^{n} p_{j−1,j}. For the chain with γ = 1, S_target = p_{n−1,n}.

*Proof.* With one overseer per level and γ = 0, level j is T iff level j−1 is T and it detects; induction gives the product. With γ = 1 the status of j−1 is irrelevant. ∎

The γ = 1 case shows that the chain's product form is entirely a *propagation* assumption: the product arises because a compromised overseer is assumed to be useless.

**Proposition 2 (correlation erodes within-level redundancy).** Fix overseers with marginal detection probabilities p₁, …, p_s. Under the one-factor Gaussian copula, D(p; ρ) is non-increasing in ρ ∈ [0,1], with D(p; 0) = 1 − ∏ᵢ(1−pᵢ) and D(p; 1) = maxᵢ pᵢ. Consequently, for any topology, S_strict is non-increasing in ρ. At ρ = 1 it equals the strict success of the design in which each level keeps only its single best overseer.

*Proof.* Let Yᵢ = √ρ Z + √(1−ρ) εᵢ, which is standard normal with Corr(Yᵢ, Y_l) = ρ for i ≠ l. The all-miss event is ∩ᵢ{Yᵢ ≥ tᵢ} with tᵢ = Φ⁻¹(pᵢ), which equals ∩ᵢ{−Yᵢ ≤ −tᵢ}. The vector (−Yᵢ) is centred Gaussian with unit variances and pairwise correlations ρ. Slepian's inequality (Slepian, 1962) states that, for centred Gaussian vectors with equal variances, P(∩ᵢ{Xᵢ ≤ cᵢ}) is non-decreasing in each pairwise correlation. Hence P(all miss) is non-decreasing in ρ, and D is non-increasing. At ρ = 0 the misses are independent. At ρ = 1, Yᵢ = Z for all i, so detection is {Z < maxᵢ tᵢ}, which has probability maxᵢ pᵢ. S_strict is a product of such D terms, each with all overseers trusted, so it inherits monotonicity. ∎

Proposition 2 makes precise a common intuition ("stacking similar monitors does not help") and gives its endpoints. At ρ = 1, adding weaker overseers to a level is worthless *for that level*. Proposition 3 shows it is not worthless for the system.

**Proposition 3 (propagation blocking under full collusion).** Consider a stationary window-k design (m_G = m_H, fixed step δ, so the miss probability of an overseer at distance m, qₘ, does not depend on j), with γ = 0 and no anchor. Assume q₁ ≤ q₂ ≤ … ≤ q_k, which holds automatically because the gap grows with distance. The all-compromised window state C^k is absorbing: once all k recent levels are compromised, no future level can be detected. Let h denote the per-level hazard of entering C^k from the quasi-stationary distribution on non-absorbed states. Then, as q_k → 0,

  h = r_k·(1 + O(q_k)),  with r_k = ∏_{m=1}^{k} qₘ^{m} at ρ = 0 and r_k = ∏_{m=1}^{k} qₘ at ρ = 1.

For the chain (k = 1) both reduce to h = q₁.

*Proof sketch.* Entry into C^k at level j+1 requires the window before that step to be (T, C, …, C), with the k−1 most recent levels compromised. The new level is then overseen by exactly one trusted overseer, at distance k, and it misses with probability q_k. Any other predecessor state has at least one trusted overseer among the k−1 most recent levels, and a single further miss leaves that trusted level inside the next window, so the move cannot reach C^k. To leading order, the quasi-stationary mass of (T, C^{k−1}) is the probability of the cheapest path from T^k, which carries mass 1 − O(q). That path compromises k−1 consecutive levels. The i-th of them (i = 0, …, k−2) has trusted overseers at distances i+1, …, k, all of which miss. At ρ = 0 the joint miss probability is ∏_{m=i+1}^{k} qₘ. At ρ = 1 it is the minimum of the trusted miss probabilities, q_{i+1} (Prop. 2). Multiplying the k−1 path factors by the final q_k gives ∏_{i=0}^{k−1} ∏_{m=i+1}^{k} qₘ = ∏ₘ qₘ^{m} for ρ = 0, and ∏_{i=0}^{k−1} q_{i+1} = ∏ₘ qₘ for ρ = 1. Every other path to (T, C^{k−1}) passes through additional transitions that each cost a factor O(q), giving the 1 + O(q_k) correction. ∎

We call this *self-healing*. In a window graph, a compromised level does not doom the rest of the ladder: it drops out of the window after k steps unless k−1 further consecutive levels are also compromised. The target-failure probability therefore decomposes approximately as

  P(level n compromised) ≈ (per-level compromise probability) + (n − n₀)·r_k,

with a per-level term of ∏ₘqₘ at ρ = 0 and q₁ at ρ = 1, and n₀ a burn-in of order k. Correlation raises the per-level term from ∏ₘqₘ to q₁; this is Proposition 2. It also raises the accumulation rate from ∏qₘᵐ to ∏qₘ, but this rate remains of order q^k rather than the chain's q. Section 6.3 verifies the hazard formula numerically and shows where the leading-order approximation breaks down.

**Proposition 4 (partial collusion stops accumulation in chains).** For the stationary chain with γ > 0 and per-step miss probability q, trust status is a two-state Markov chain with P(T→C) = q and P(C→T) = γ(1−q). For every initial condition, P(level n compromised) converges geometrically, at rate |1 − q − γ(1−q)|, to

  π_C = q / (q + γ(1−q)).

*Proof.* A trusted predecessor detects with probability 1−q. A compromised predecessor detects with probability γ(1−q). The stationary distribution of a two-state chain is π_C = P(T→C)/(P(T→C)+P(C→T)). ∎

Propositions 1 and 4 together show that the chain's notorious exponential decay in n is a statement about γ = 0 *and* the strict criterion. Under target success with any residual detection by compromised overseers, compromise saturates at π_C instead of accumulating. π_C can still be large: at q = 0.24 and γ = 0.1, π_C = 0.760 (Section 6.5).

---

## 5 Experimental Setup

All numbers below come from `code/experiments.py` (total runtime about 56 s on one CPU core). Outputs are saved under `results/` and labelled by experiment ID.

**Regimes (stylised, not fitted).** We use three parameter settings that cover the three qualitative cases identified by Engels et al. (2025):

| Regime | m_G | m_H | β | Guard win prob. at equal capability (x = 0) |
|---|---|---|---|---|
| symmetric | 1.0 | 1.0 | −300 | 0.849 |
| guard-steeper ("debate-like" in sign of slopes) | 1.0 | 0.4 | −100 | 0.640 |
| houdini-steeper ("backdoor-like" in sign of slopes) | 0.4 | 1.0 | −400 | 0.909 |

The labels refer only to the *sign pattern* of the slopes that Engels et al. report qualitatively (Debate: Guard slope positive, Houdini slope near zero; Backdoor Code: Houdini slope steeper). The numerical values are ours.

**Experiments.** E1 validates the DP and strict-success formulas against Monte Carlo. E2 is the chain baseline. E3/E3b test accumulation and Proposition 3. E4 sweeps ρ. E5 sweeps γ. E6 is the budget-constrained design search. E7 compares topologies, including the full graph, by Monte Carlo. E8 accounts for false alarms.

---

## 6 Results

### 6.1 Validation (E1)

On 24 random configurations (all three regimes; n ∈ [3,12], k ∈ [1,4], with and without anchor, ρ ∈ {0, .3, .6, .9}, γ ∈ {0, .25, .5}, Δ ∈ {400, 800, 1600}), the exact DP and the Monte Carlo simulator (200,000 runs each) agree. The maximum |z|-score of the difference in target success is 2.30, the maximum absolute error is 0.0023 for target success, and 0.0021 for strict success (`results/e1_validation.json`). A maximum |z| of 2.3 across 24 comparisons is consistent with sampling noise alone.

### 6.2 Chain baseline (E2)

For the chain, success as a function of n reproduces the qualitative shape reported by Engels et al. (Fig. `results/fig_e2_chain.png`; `e2_chain.json`). In the symmetric regime the optimal depth grows with the gap: n* = 2, 4, 7 for Δ = 400, 800, 1600, with optimal success 0.410, 0.168 and 0.028. Direct oversight (n = 1) at Δ = 1600 succeeds with probability 0.00056. In the guard-steeper regime at Δ = 1600, n* = 5 achieves 0.277, against 0.043 for n = 1. In the houdini-steeper regime, nesting barely helps: n* = 1 or 2, with 0.100 at Δ = 800 and 0.001 at Δ = 1600. A key structural fact is that **the optimal chain uses only n* games**. Additional budget cannot be converted into success, because extra steps multiply in extra failure chances.

### 6.3 Self-healing and the accumulation rate (E3, E3b)

**E3 (symmetric regime, fixed step δ = 100, ρ = 0, γ = 0).** We grow the ladder at a fixed per-step gap, so the per-distance miss probabilities are q = (0.240, 0.360, 0.500, 0.640) for m = 1..4. The probability that the target is compromised at depth n = 30 is 0.9997 for the chain, 0.672 for k = 2, 0.210 for k = 3 and 0.087 for k = 4. At n = 60 it is 1.000, 0.879, 0.314 and 0.111 (`e3_accumulation.json`, Fig. `fig_e3_accumulation.png`). The chain is essentially certain to be compromised by n ≈ 20. Window graphs accumulate failure slowly despite the much weaker distant overseers (q₄ = 0.64 means the overseer four levels down misses most of the time). For k ≥ 3 the curves are non-monotone over the first few levels. Early levels are overseen by the strong level-0 anchor through fewer, shorter edges, so failure dips before the stationary accumulation sets in. The dotted leading-order curve in the figure is a poor approximation here because the qₘ are not small. E3b checks the asymptotics where they apply.

**E3b (hazard of irrecoverable compromise).** We compute the exact per-level hazard of entering the absorbing all-compromised state at n = 80 and compare it with r_k from Proposition 3 (`e3b_hazard.json`).

*Table 2. Exact asymptotic absorption hazard vs. Proposition 3.*

| Regime (q₁..q₄) | k | ρ | exact hazard | predicted r_k | ratio | P(target compromised), n = 80 |
|---|---|---|---|---|---|---|
| small-q: β=−700, δ=50 (0.023, 0.031, 0.040, 0.053) | 1 | – | 2.32e-2 | 2.32e-2 | 1.000 | 0.847 |
| | 2 | 0 | 2.22e-5 | 2.18e-5 | 1.022 | 3.18e-3 |
| | 3 | 0 | 1.48e-9 | 1.44e-9 | 1.025 | 5.77e-5 |
| | 4 | 0 | 1.19e-14 | 1.16e-14 | 1.025 | 3.06e-6 |
| | 2 | 1 | 6.95e-4 | 7.10e-4 | 0.978 | 0.0749 |
| | 3 | 1 | 2.81e-5 | 2.87e-5 | 0.977 | 0.0255 |
| | 4 | 1 | 1.49e-6 | 1.53e-6 | 0.977 | 0.0235 |
| symmetric: β=−300, δ=100 (0.24, 0.36, 0.50, 0.64) | 2 | 0 | 3.27e-2 | 3.11e-2 | 1.050 | 0.938 |
| | 4 | 0 | 8.73e-4 | 6.53e-4 | 1.337 | 0.126 |
| | 2 | 1 | 7.42e-2 | 8.65e-2 | 0.858 | 0.998 |
| | 4 | 1 | 2.14e-2 | 2.77e-2 | 0.772 | 0.864 |

In the small-miss regime the formula is accurate to within 2.5% for every k and both correlation extremes. The approximation degrades, as expected, when misses are common (ratios 0.77 to 1.34 in the symmetric regime).

The ρ = 1 rows show the separation described in Section 4. With perfectly correlated overseers, the target-compromise probability for k ≥ 3 sits near q₁ = 0.023: per-level redundancy is gone, as Proposition 2 says. Yet the failure no longer *accumulates* (0.0235 at k = 4 vs. 0.847 for the chain at the same depth), because escaped levels are still flushed out of the window by later honest detections.

### 6.4 Correlation (E4)

Setting: symmetric regime, Δ = 1600, n = 16 (step 100), γ = 0 (`e4_rho.json`, Fig. `fig_e4_rho.png`).

*Per-level detection.* With two overseers at distances 1 and 2 (p₁ = 0.760, p₂ = 0.640), detection falls from 0.914 at ρ = 0 to 0.883, 0.850, 0.810 and 0.760 at ρ = 0.25, 0.5, 0.75 and 1. The decline is monotone, as Proposition 2 requires.

*System level.*

| k | strict ρ=0 | strict ρ=0.5 | strict ρ=1 | target ρ=0 | target ρ=0.5 | target ρ=1 |
|---|---|---|---|---|---|---|
| 1 (chain) | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 |
| 2 | 0.196 | 0.066 | 0.012 | 0.522 | 0.371 | 0.247 |
| 3 | 0.374 | 0.098 | 0.012 | 0.844 | 0.655 | 0.452 |
| 4 | 0.461 | 0.111 | 0.012 | 0.924 | 0.757 | 0.543 |

Strict success collapses to the chain value at ρ = 1, exactly as Proposition 2 predicts, and already loses most of its gain by ρ = 0.5. Target success keeps a large advantage even at ρ = 1: 0.543 vs. 0.012 for k = 4. This is the propagation-blocking effect of Proposition 3.

*Cross-level correlation (Monte Carlo, ρ = 0.3).* We add a stealth factor shared by all Houdinis, with cross-level correlation c. The chain's success *increases* with c: 0.0125, 0.0413 and 0.0787 for c = 0, 0.3, 0.6, because failures concentrate in the same runs. Window-3 target success decreases slightly: 0.733, 0.715, 0.710. Window-2 increases: 0.425, 0.472, 0.510. The qualitative ranking of topologies is unchanged in this range. The effect of cross-level correlation is not covered by our theory.

### 6.5 Collusion (E5)

Setting: symmetric regime, Δ = 1600, n = 16, ρ = 0.3 (`e5_gamma.json`, Fig. `fig_e5_gamma.png`).

| k | γ = 0 | γ = 0.2 | γ = 0.5 | γ = 1 |
|---|---|---|---|---|
| 1 | 0.012 | 0.388 | 0.613 | 0.760 |
| 2 | 0.427 | 0.756 | 0.835 | 0.877 |
| 3 | 0.734 | 0.866 | 0.891 | 0.910 |
| 4 | 0.833 | 0.897 | 0.909 | 0.923 |

The chain is extremely sensitive to the collusion parameter: a compromised overseer that keeps only 20% of its detection ability raises chain target success thirty-fold. Window graphs are much less sensitive. This matters for measurement: the chain's predictions hinge on a parameter (γ) that is currently unmeasured, whereas graph designs are robust to it. The stationary formula of Proposition 4 matches the exact DP at n = 200 to all printed digits: π_C = 0.7597, 0.5132 and 0.3451 for γ = 0.1, 0.3 and 0.6, with q = 0.240.

### 6.6 Budget-constrained design (E6)

For each regime, Δ ∈ {400, 800, 1600}, ρ ∈ {0, 0.5, 0.9} and budget B ∈ {8, 16, 32} games, we enumerated all designs with n ≤ B, k ≤ 6 and anchor on/off within budget, at γ = 0. We report the best design for target and for strict success alongside the best chain (`e6_budget_table.csv`, Fig. `fig_e6_heatmap.png`). Selected rows:

| Regime | Δ | ρ | B | best chain (target = strict) | best target (n;k;anchor) | best strict (n;k;anchor) |
|---|---|---|---|---|---|---|
| symmetric | 800 | 0 | 16 | 0.168 (n=4) | 0.772 (6;3;no) | 0.453 (6;3;no) |
| symmetric | 800 | 0.9 | 16 | 0.168 | 0.540 (6;3;no) | 0.171 (4;3;yes) |
| symmetric | 1600 | 0 | 32 | 0.028 (n=7) | 0.633 (11;3;no) | 0.242 (11;3;no) |
| symmetric | 1600 | 0.5 | 32 | 0.028 | 0.451 (11;3;no) | 0.078 (11;3;no) |
| symmetric | 1600 | 0 | 8 | 0.028 | 0.037 (4;2;no) | 0.028 (chain) |
| guard-steeper | 1600 | 0.5 | 32 | 0.277 (n=5) | 0.930 (11;2;yes) | 0.318 (9;4;no) |
| guard-steeper | 400 | 0.9 | 16 | 0.415 (n=1) | 0.790 (5;4;yes) | 0.415 (chain) |
| houdini-steeper | 800 | 0 | 32 | 0.100 (n=2) | 0.733 (7;6;yes) | 0.447 (7;6;yes) |
| houdini-steeper | 1600 | 0 | 32 | 0.001 (n=1) | 0.011 (7;6;yes) | 0.001 (chain) |

Four regularities hold across all 81 settings in the table file:

1. **Target success.** The best graph design is never worse than the best chain. It is substantially better whenever the budget allows k ≥ 2 at a depth near the chain optimum, and the gain persists at ρ = 0.9.
2. **Strict success.** Graph gains are real at ρ = 0 but shrink rapidly with ρ. At ρ = 0.9 the best graph design improves strict success over the best chain by at most 0.0033 in every one of the 27 settings at that correlation. This is Proposition 2 at system scale.
3. **Optimal designs are deeper than optimal chains.** With redundancy, extra levels are no longer pure liabilities, so optimal n roughly doubles (e.g. 4 → 6 to 9, and 7 → 11 in the symmetric regime).
4. **Anchoring helps mainly at small gaps or Guard-favoured slopes.** The human anchor is chosen at Δ = 400 in most regimes and in the guard-steeper regime. At large Δ in the symmetric regime the anchor's gap is too large for it to contribute.

The failure case is equally important. When the Houdini slope exceeds the Guard slope and the gap is large (houdini-steeper, Δ = 1600), no topology within budget 32 exceeds 0.011 target success. Redundancy multiplies chances to detect; it does not create detection ability that the per-step game lacks. Low budget is also binding: at B = 8 and Δ = 1600 (symmetric), the best design reaches only 0.037.

### 6.7 Topologies at fixed depth, and false alarms (E7, E8)

At n = 10 in the symmetric regime (`e7_topologies.json`, Monte Carlo), the target success of chain, anchor+chain, window-2, window-3 and full topologies is 0.134, 0.788, 0.849, 0.976 and 1.000 at Δ = 400 (ρ = 0; costs 10, 19, 19, 27, 55). At Δ = 1600 it is 0.025, 0.070, 0.342, 0.555 and 0.680. Anchoring to the human is nearly as good as window-2 at a small gap, but far worse at a large one: 0.070 vs. 0.342.

OR aggregation buys detection at the price of false alarms (`e8_false_alarms.json`). With an independent per-overseer false-positive rate φ = 0.05 on a benign model, the probability of at least one false alarm across the pipeline is 0.185 for the chain with 4 games, 0.537 for the (6;3) design with 15 games, and 0.785 for the 30-game designs. The expected numbers of false alarms are 0.20, 0.72 and 1.40. At φ = 0.01 the figures are 0.039, 0.140 and 0.260. Redundant oversight is therefore attractive only where a flag can be adjudicated more cheaply than an escape costs. That is plausible for audit-style flags reviewed by humans, but it is a real cost that our success metric does not include.

---

## 7 Measurement Protocol and Falsifiable Predictions

The model's two new parameters can be measured with experimental machinery that already exists.

**Measuring ρ.** Under the one-factor copula, ρ is the *tetrachoric correlation* of the miss indicators of two overseers on the same set of attacks. Given a pool of attacks (e.g. backdoored solutions in an APPS-style control setting; Greenblatt et al., 2024) and two monitors A and B with miss rates q_A and q_B, estimate the joint miss rate q_AB. ρ is then the unique value at which the bivariate normal orthant probability equals q_AB. This is closely related in spirit to the error-overlap metric CAPA of Goel et al. (2025), but tetrachoric correlation is the quantity that plugs directly into Propositions 2 and 3.

**Measuring γ.** Run an untrusted-monitoring control evaluation in which the monitor is a red-team-modified model instructed (or fine-tuned) to let a particular policy's attacks through while still flagging others. γ is the ratio of its detection rate on that policy's attacks to an honest copy's detection rate. Greenblatt et al. (2024) already estimate related quantities when they study collusion signals in untrusted monitoring.

**Pre-registered predictions** (for model families with at least four capability levels, e.g. a size ladder within one family):

* **P1 (redundancy decays with similarity).** Across pairs of monitors, the gain in detection from adding the second-best monitor will be predicted, within sampling error, by D(p; ρ̂) − max p, with ρ̂ the measured tetrachoric correlation. Pairs from the same family will show larger ρ̂ and smaller gains than cross-family pairs. *Falsified if* the gain is uncorrelated with ρ̂ or exceeds the ρ = 0 bound.
* **P2 (self-healing).** In a simulated multi-generation oversight ladder in which escaped models act as colluding monitors, the fraction of runs whose final model is compromised will grow roughly linearly in depth for the chain. For window-2 it will grow at a rate smaller than the chain's by a factor between about q₂² (if ρ̂ ≈ 0) and q₂ (if ρ̂ ≈ 1), where q₂ is the measured miss rate of the overseer two levels down (Prop. 3). *Falsified if* window-2 accumulation is comparable to the chain's.
* **P3 (chain fragility to γ).** Measured chain target success will vary by more than an order of magnitude across plausible γ ∈ [0, 0.2], while window-3 target success will vary by less than a factor of two. *Falsified if* the chain's sensitivity is comparable to the window's.

**Protocol sketch.** (1) Fix a task with checkable ground truth (backdoored code; incorrect proofs). (2) Choose a ladder of 4–6 models and fit Guard and Houdini Elo curves as in Engels et al. (2025). (3) Estimate pairwise ρ and γ as above on held-out attacks. (4) Use the DP in `code/nso.py` to make predictions. (5) Run the ladder with real models, in which each newly escaped model is placed in the monitor role with a collusion instruction, and compare. Costs are dominated by step (5). This protocol requires LLM access we did not have, so it is proposed, not executed.

---

## 8 Discussion

**What the chain model gets right and wrong.** The chain picture's pessimism has two separable sources: weak per-step games and the assumption that one escape is fatal. Our results suggest that the second is a design choice, not a law. Graph topologies, in which every level is also checked by a few levels further down, turn an exponentially decaying success probability into one that decays at a far smaller rate, of order q^{k(k+1)/2} for independent overseers. The first source is not fixable by topology. If the Houdini's domain Elo grows faster with capability than the Guard's, redundancy buys almost nothing (Section 6.6). Protocol design that changes the slopes, such as debate-like structures where oversight improves with capability (Engels et al., 2025), remains the first-order lever.

**Correlation is the price of homogeneous oversight.** If all overseers come from one training pipeline, ρ is plausibly high. Goel et al. (2025) find error similarity rising with capability. Proposition 2 says per-level redundancy then disappears, and Section 6.4 shows strict success collapsing to the chain value. What survives is propagation blocking, which needs only that *some* trusted overseer remains in the window when a level escapes. This suggests a concrete design heuristic: diversify the overseer pool, and when diversity is impossible, keep overseers from several *generations* in the loop so that an escaped generation can be flushed out.

**Which success criterion matters.** If every intermediate model is deployed and can act on the world, strict success is the relevant measure, and graphs help only to the extent that errors are diverse. If intermediate models are sandboxed tools used only to oversee the next model, target success is the relevant measure, and graphs help even with highly correlated overseers. Making intermediate oversight models *non-deployed* is itself a safety lever this model makes visible.

**Relation to AI control.** Our γ parameter connects the scalable-oversight literature, which usually assumes honest overseers, to the AI-control literature, which assumes untrusted monitors may collude (Greenblatt et al., 2024). The chain model's sensitivity to γ (Section 6.5) means that NSO success estimates are not meaningful until γ has been measured.

---

## 9 Limitations

* **Stylised parameters.** Our regimes are not fits to Engels et al.'s games or to any model family. Absolute success numbers should not be read as forecasts. Only the comparative and structural statements (Props. 2–4 and the topology rankings) are meant to transfer.
* **Elo abstraction.** Like Engels et al., we compress oversight into a single win probability per (Guard, Houdini) pair. Real oversight has many attack types, partial detections and adaptive attackers that choose *which* overseer's blind spot to exploit. An adversary aware of the topology could select attacks that maximise ρ among exactly the overseers of its level. Our copula holds ρ fixed and cannot model this.
* **Worst-case intent, binary trust.** Every level is treated as an attacker, and detection is treated as perfect correction. In practice detection may yield only a noisy signal, and correction (retraining) may fail or create new problems.
* **Independence across levels.** The theory assumes independent stealth across levels. Our Monte Carlo check with cross-level correlation up to 0.6 preserves the topology ranking but changes magnitudes (Section 6.4). We have no theory for strongly shared stealth.
* **OR aggregation and false alarms.** We optimise detection only. False alarms grow with the number of games (Section 6.7), and a full treatment would optimise a cost-weighted objective or use thresholded or majority aggregation.
* **Proposition 3 is asymptotic.** It is accurate when miss probabilities are small (within 2.5% in Table 2) and degrades to 23–34% error when misses are 0.24–0.64. The exact DP should be used for design; the formula is for intuition and scaling.
* **Cost model.** We count every oversight game equally. Overseeing a strong model with a weak one may cost differently from overseeing it with a near-peer, and the human anchor's time is far scarcer than model time.
* **No real-model experiments.** All results are from our model, our code and our simulations. The empirical test in Section 7 is proposed, not run.

---

## 10 Conclusion

Nested scalable oversight has so far been analysed as a chain whose success is a product of per-step probabilities. We showed that this product form rests on two assumptions, a chain topology and full collusion of escaped overseers, and that relaxing them changes the qualitative picture. Redundant oversight graphs block the propagation of escapes, so irrecoverable compromise accrues at rate ∏qₘᵐ (independent overseers) or ∏qₘ (perfectly correlated overseers) per level, rather than q. Error correlation removes redundancy's per-level benefit but not its anti-propagation benefit. Partial collusion makes chain compromise saturate rather than accumulate. None of this rescues a per-step game in which attackers gain more from capability than overseers do. The model's two new parameters, error correlation and collusion, can be measured with existing monitoring evaluations. Until they are, quantitative NSO forecasts, ours included, should be treated as conditional on them.

---

## Reproducibility

Code: `code/nso.py`, `code/experiments.py`; run `cd code && python3 experiments.py` (about 1 minute on CPU). All reported numbers are in `results/` (JSON/CSV) and figures in `results/fig_*.png`. Provenance: every number in Sections 4 (worked examples) and 6, and in the abstracts, comes from our code. The four Engels et al. success rates quoted in Section 1 are from their paper.

---

## References

Barnes, B. (2020). Debate update: Obfuscated arguments problem. *AI Alignment Forum*. https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem

Bowman, S. R., Hyun, J., Perez, E., et al. (2022). Measuring progress on scalable oversight for large language models. arXiv:2211.03540.

Brown-Cohen, J., Irving, G., & Piliouras, G. (2024). Scalable AI safety via doubly-efficient debate. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*.

Brown-Cohen, J., Irving, G., Piliouras, G., Chen, L., Li, J., & Xun, Z. (2025). Avoiding obfuscation with prover-estimator debate. arXiv:2506.13609.

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., Sutskever, I., & Wu, J. (2023). Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv:2312.09390.

Christiano, P., Shlegeris, B., & Amodei, D. (2018). Supervising strong learners by amplifying weak experts. arXiv:1810.08575.

Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. New York: Arco.

Engels, J., Baek, D. D., Kantamneni, S., & Tegmark, M. (2025). Scaling laws for scalable oversight. arXiv:2504.18530.

Goel, S., Struber, J., Auzina, I. A., Chandra, K. K., Kumaraguru, P., Kiela, D., Prabhu, A., Bethge, M., & Geiping, J. (2025). Great models think alike and this undermines AI oversight. arXiv:2502.04313.

Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2024). AI control: Improving safety despite intentional subversion. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2312.06942.

Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. arXiv:1805.00899.

Kenton, Z., Siegel, N. Y., Kramár, J., Brown-Cohen, J., Albanie, S., Bulian, J., Agarwal, R., Lindner, D., Tang, Y., Goodman, N. D., & Shah, R. (2024). On scalable oversight with weak LLMs judging strong LLMs. In *Advances in Neural Information Processing Systems 37 (NeurIPS)*. arXiv:2407.04622.

Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024). Debating with more persuasive LLMs leads to more truthful answers. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2402.06782.

Ladha, K. K. (1992). The Condorcet jury theorem, free speech, and correlated votes. *American Journal of Political Science*, 36(3), 617–634.

Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., & Legg, S. (2018). Scalable agent alignment via reward modeling: A research direction. arXiv:1811.07871.

Slepian, D. (1962). The one-sided barrier problem for Gaussian noise. *Bell System Technical Journal*, 41, 463–501.
