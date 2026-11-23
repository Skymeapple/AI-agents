# Beyond the Chain: Propagation Blocking, Correlated Errors, and Collusion in Nested Scalable Oversight

---

## Abstract

Nested scalable oversight (NSO) proposes to oversee a far stronger AI system by bootstrapping: a trusted weak model oversees a slightly stronger one, which then oversees the next. The most explicit quantitative treatment of NSO (Engels et al., 2025) models each step as an Elo-rated game between a "Guard" and a "Houdini" and composes the steps as a *chain*: every level is overseen only by its predecessor, and overall success is the product of per-step success probabilities. We ask how much of the chain's pessimism comes from the per-step games and how much from this composition rule. To do so we generalise the chain to *oversight graphs*. A level may be overseen by several earlier generations, by several same-generation copies, or both. Overseer errors may be correlated (ρ across generations, ρ_s among copies), an escaped level may collude with later ones (it keeps a fraction γ of its detection ability), detection may fail to correct (probability c), and flags may require t detections.

We give an exact dynamic program over trust states and validate it against Monte Carlo simulation on 48 random configurations (maximum |z| = 2.37). On the theory side, we prove an exact identity and two-sided bounds for the rate h at which a window-k graph becomes irrecoverably compromised under full collusion. These bounds give h = ∏ₘ qₘ·(1+O(kq₁)) rigorously at ρ = 1, and h = q₁q₂²·(1+O(q₁)) at k = 2 with independent overseers. For k ≥ 3 and ρ < 1 we prove the lower bound h ≥ ∏ₘ qₘᵐ·(1−O(kq₁)) and leave the matching upper bound as a conjecture, which 60-digit numerics support (h/∏qₘᵐ − 1 ≈ q₁). The chain's rate is q₁.

In a design study whose games are calibrated to the four NSO success rates published by Engels et al., and in a 192-setting sweep over Guard/Houdini slopes and intercepts, cross-generation graphs never do worse than the chain on target success. They improve it by more than 0.1 in 61–64% of settings. Against a matched-cost baseline that spends the same budget on same-generation copies, however, the picture is conditional:

- When copies are as decorrelated as different generations and errors are nearly independent, same-generation redundancy is as good or better.
- Cross-generation redundancy wins when errors are correlated (ρ = 0.9: best in 30/36 calibrated settings) or when copies are more alike than different generations.
- It never wins on strict ("no level ever escapes") success.

Its distinctive value is therefore *propagation blocking*, not more checks. Graph gains shrink substantially under 2-of-s threshold aggregation or a false-alarm budget. All designs fail when direct oversight succeeds with probability below about 10⁻³, whatever the slope pattern. All results are from a model; no experiments on real models were run.

## 中文摘要

嵌套式可扩展监督（NSO）设想通过"自举"来监督远强于人类的 AI：可信的弱模型监督稍强的模型，后者再去监督下一个。Engels 等（2025）把每一步建模为基于 Elo 的"守卫–胡迪尼"博弈，并以"链式"方式组合：每层只由前一层监督，总成功率为各步之积。我们追问：链式模型的悲观结论有多少来自单步博弈本身，又有多少来自这种组合规则？为此我们把链推广为"监督图"：每层可由多个更早的世代、多个同代副本或二者共同监督；跨代错误相关系数为 ρ，同代副本间为 ρ_s；逃逸层可能合谋（保留 γ 比例的检测能力）；检测后纠正可能失败（概率 c）；报警可要求 t 次检测。

我们给出精确动态规划，并在 48 个随机配置上用蒙特卡洛验证（最大 |z| = 2.37）。理论上，对完全合谋下窗口 k 图"不可恢复失守"的速率 h，我们证明了一个精确恒等式和双侧界。由此严格得到：ρ=1 时 h = ∏qₘ(1+O(kq₁))；k=2 时 h = q₁q₂²(1+O(q₁))。对 k≥3、ρ<1，我们证明了下界 h ≥ ∏qₘᵐ(1−O(kq₁))，匹配的上界则作为猜想，60 位高精度数值支持该猜想（h/∏qₘᵐ−1≈q₁）。链式结构的速率为 q₁。

我们做了两组设计研究：一组的博弈参数校准到 Engels 等公布的四个 NSO 成功率，另一组在守卫/胡迪尼斜率与截距上扫描 192 个设置。在目标成功率上，跨代监督图从不劣于链，并在 61–64% 的设置中提升超过 0.1。但若与等成本的"同代多副本"基线比较，结论是有条件的：
- 当副本与不同世代同样去相关、且错误近乎独立时，同代冗余同样好甚至更好。
- 当错误高度相关（ρ=0.9 时在 36 个校准设置中 30 个最优），或同代副本比不同世代更相似时，跨代冗余胜出。
- 在严格成功率（任何层都不逃逸）上，跨代冗余从不占优。

因此跨代冗余的独特价值是"阻断传播"，而不是"更多检查"。在"2-of-s"阈值聚合或误报预算下，图的优势大幅缩小。当直接监督成功率低于约 10⁻³ 时，不论斜率形态如何，所有设计都失败。所有结果都来自模型，我们没有在真实模型上做实验。

---

## 1 Introduction

Scalable oversight asks how humans can supervise systems that outperform them on the very skills needed to judge their behaviour (Bowman et al., 2022). Most concrete proposals amplify a weak judge: debate between strong models (Irving et al., 2018), recursive decomposition (Christiano et al., 2018), recursive reward modelling (Leike et al., 2018), or weak supervisors whose labels a strong student generalises beyond (Burns et al., 2023). Many proposals share one structural ingredient, *bootstrapping*: a trusted weak system supervises a slightly stronger one, which, once trusted, supervises the next.

Engels et al. (2025) gave the first explicit scaling analysis of this bootstrapping, which they call Nested Scalable Oversight (NSO). They model each oversight step as a game between a Guard (overseer) and a Houdini (overseen system), with domain Elo piecewise-linear in general Elo, fit the curves for four games, and choose the number of steps n that maximises ∏ⱼ pⱼ. At a general Elo gap of 400 their NSO success estimates are 13.5% (Mafia), 51.7% (Debate), 10.0% (Backdoor Code) and 9.4% (Wargames).

That analysis assumes a *chain*: level j is overseen only by level j−1, and any single escape is fatal. Three features of realistic bootstrapping fall outside it.

1. **Redundancy.** Level j can also be checked by j−2, j−3, … or by the human anchor at level 0. It can also be checked by several *copies* of level j−1.
2. **Correlated errors.** Redundant overseers help only if their errors differ. Goel et al. (2025) report that language-model errors become more similar as capability increases. Recent monitoring work finds that diversity among monitors, rather than more compute, drives ensemble gains (Koran et al., 2026), and that monitor skill rather than lineage governs agreement (Jha, 2026).
3. **Collusion and propagation.** In NSO an escaped level becomes the overseer of the next one and may collude, a threat model central to AI control (Greenblatt et al., 2024; Bhatt et al., 2025).

**This paper.** We embed the Engels et al. per-step game in an oversight-graph model and separate two quantities that the chain conflates. *Per-level redundancy* is the probability that some overseer catches a given level. *Propagation blocking* is whether an escaped level can later be flushed out by trusted overseers that are still in the window. We distinguish *strict* success (no level ever escapes) from *target* success (the final level is trusted).

**Contributions** (in order of what we consider new):

1. **Propagation analysis (Theorem 3).** Under full collusion, compromise in a chain becomes irrecoverable at rate q₁ per level. For a window-k graph we prove an exact identity for the quasi-stationary hazard h and two-sided bounds valid for every correlation level. The bounds pin down h to leading order at ρ = 1 (h ≈ ∏ₘ qₘ) and at k = 2 (h ≈ q₁q₂², or the correlated analogue). For k ≥ 3 and ρ < 1 they give the lower bound ∏ₘ qₘᵐ. The matching upper bound is stated as a conjecture with high-precision numerical support. Correlation erodes per-level redundancy (Prop. 2, a direct application of Slepian's inequality) but leaves propagation blocking of order q^k in place.
2. **Matched-cost test of the cross-generation thesis.** We compare chains, chains with r same-generation copies per level ("chain-r"), cross-generation graphs, and hybrids at equal budget. Cross-generation redundancy beats same-generation redundancy on target success when errors are correlated or when copies are more alike than generations. It does *not* when copies are as diverse as generations and errors are nearly independent, and it never does on strict success (Section 6.3).
3. **Grounded and stress-tested design study.** We calibrate the games to the published Engels et al. success rates, sweep slopes and intercepts, and stress the conclusions with imperfect correction, threshold aggregation, a false-alarm cost and a false-alarm constraint (Sections 6.2–6.5).
4. **Measurement protocol and pre-registered predictions** for ρ, ρ_s and γ, with a check of what happens when the one-factor model used to identify ρ is misspecified (Section 7).

The exact dynamic program and the simulator are supporting tools, validated in Section 6.1. Relative to Engels et al., our contribution is an incremental but, we think, clarifying extension: it identifies which part of the chain's pessimism is a composition assumption and when changing that composition helps. All quantitative results concern a model. The game parameters are calibrated to four published success rates, not to model ladders we ran, and we ran no experiments on large models.

---

## 2 Related Work

**Scalable oversight protocols.** Debate (Irving et al., 2018) lets a judge verify one path through an argument tree. Iterated amplification (Christiano et al., 2018) and recursive reward modelling (Leike et al., 2018) build training signals from decomposed subproblems. Bowman et al. (2022) proposed the "sandwiching" paradigm. Debate with more persuasive debaters improved non-expert judge accuracy on QuALITY (Khan et al., 2024), with mixed results across a broader suite of weak-judge/strong-agent asymmetries (Kenton et al., 2024). Doubly-efficient debate (Brown-Cohen et al., 2024) and prover–estimator debate (Brown-Cohen et al., 2025) address the obfuscated-arguments problem (Barnes, 2020). These works design *single-step* protocols. We take a step's success probability as given and ask how steps should be *composed*.

**Scaling laws for oversight.** Engels et al. (2025) is the direct starting point. They introduce Guard/Houdini Elo curves, fit them on four games, and derive optimal chain depth. They describe Debate as having a positive Guard slope and a near-zero Houdini slope, and Backdoor Code as having a Houdini slope steeper than the Guard's. We change the *composition rule* and calibrate our games to their published success rates (Section 5).

**Multi-overseer monitoring and correlated failure.** Several recent works study panels or ensembles of monitors within a single oversight step.
- Koran et al. (2026) build 12 monitors on coding tasks with adversarial inputs and find that a diverse three-monitor ensemble outperforms three identical monitors. They attribute the gain to signal diversity rather than compute.
- Jha (2026) finds, across 24 open-weight monitors from nine lineages, that monitor skill rather than pretraining lineage predicts agreement, and that correlation-weighted panel selection did not beat the best single monitor out of sample.
- Zanolini (2026) models approval monitors that can be fooled together using fail-prone systems from distributed computing. He characterises which approval rules are safe and live, and recommends counting independent sources rather than votes.
- Overman & Bayati (2026) aggregate several auxiliary overseers' concerns into a calibrated conservatism penalty with conformal guarantees.
- Ctrl-Z (Bhatt et al., 2025) combines trusted monitoring with resampling of suspicious actions in multi-step agent settings.

All of these concern *one* oversight step, possibly with many monitors. The within-step part of our model (Prop. 2) is a stylised version of their concern: equicorrelated Gaussian orthant monotonicity is textbook, and correlated-jury results (Ladha, 1992) cover the same ground. What these works do not address, and what we focus on, is the *multi-generation* structure of NSO, in which overseers are past targets and a missed detection changes the composition of future panels. Our ρ versus ρ_s distinction is the quantity that Koran et al. and Jha measure in their own settings (diverse versus identical monitors; cross- versus same-lineage pairs), and their mixed findings are one reason we treat ρ_s − ρ as an empirical unknown rather than assuming it is positive.

**Weak-to-strong generalisation.** Burns et al. (2023) show that strong students fine-tuned on weak labels outperform their supervisors but do not recover full capability. Goel et al. (2025) find that model errors are becoming more similar with capability, which motivates ρ directly.

**AI control and collusion.** Greenblatt et al. (2024) evaluate trusted editing and untrusted monitoring against a red-team policy. In untrusted monitoring the monitor may collude. Our γ summarises how much detection ability a compromised overseer retains.

---

## 3 Model

### 3.1 Per-step oversight game (from Engels et al., 2025)

Levels are indexed i = 0, …, n. Level 0 is the trusted starting overseer, and level i has general Elo xᵢ = iΔ/n. In the linear region of the Engels et al. curves, domain Elo is G(x) = m_G·x in the Guard role and H(x) = m_H·x + β in the Houdini role. The probability that Guard i detects Houdini j is

  p_{ij} = 1 / (1 + 10^{(H(x_j) − G(x_i))/400}).

We omit the saturation plateaus of the Engels et al. double-ReLU fits. Our Δ range is meant to lie within the linear region.

### 3.2 Oversight designs

Level j ≥ 1 is overseen by a set of *generations* S_j ⊆ {0, …, j−1}, and each generation fields r identical-generation *copies*:

* **chain:** S_j = {j−1}, r = 1 (Engels et al.);
* **chain-r:** S_j = {j−1}, r ≥ 2 copies of the predecessor. This is same-generation redundancy;
* **window-k graph:** S_j = {max(0, j−k), …, j−1}, r = 1, optionally with the **anchor** (level 0 checks every level). The **full** topology has S_j = {0, …, j−1};
* **hybrid:** window-k with r ≥ 2.

The cost of a design is the number of oversight games, r·Σⱼ|S_j|.

### 3.3 Trust dynamics

We take a worst-case stance on intent: every new level is a Houdini that tries to evade. Each earlier level is *trusted* (T) or *compromised* (C), and level 0 is T. All copies of a generation share its status: copies of a compromised generation are compromised.

* A trusted overseer instance detects Houdini j with probability p_{ij}. A compromised one detects with γ·p_{ij}.
* **Correlation.** Instance ℓ of generation i, overseeing level j, detects iff √ρ·Z_j + √(ρ_s−ρ)·V_{ij} + √(1−ρ_s)·ε_{ijℓ} < Φ⁻¹(p'_{ij}), with all latent variables i.i.d. N(0,1). Instances of different generations then have correlation ρ, and copies of the same generation have ρ_s ≥ ρ. Marginals are preserved. Z_j can be read as Houdini j's stealth against shared blind spots. The Z_j are independent across levels; Section 6.6 relaxes this.
* **Aggregation and correction.** Level j is *flagged* if at least t instances detect (t = 1 is OR). A flagged level is corrected and becomes T with probability c. Otherwise it is C. The original model is c = 1, t = 1.
* **False alarms.** On a benign model each instance raises a false flag independently with probability φ. We report the expected number of falsely flagged levels E[FA] and P(any false flag).

**Outcomes.** Strict success is S_strict = P(all levels T). Target success is S_target = P(level n is T). For the chain with γ = 0 both equal ∏ⱼ p_{j−1,j}, the Engels et al. objective.

**Interpretation.** Strict success is the right criterion when any escape by an intermediate model is itself catastrophic, for example when every intermediate model is deployed. Target success is the right criterion when intermediate models are non-deployed tools whose escape matters only through what they do downstream.

### 3.4 Exact computation

For window-k designs (any r, ρ, ρ_s, γ, c, t, with or without anchor), the joint status of the last k levels is a sufficient statistic. A dynamic program over 2ᵏ states gives S_target and S_strict exactly. Flag probabilities are evaluated by nested Gauss–Hermite quadrature: 48 nodes for Z and 24 for V, with a truncated convolution of per-generation count distributions when t ≥ 2. The full topology at n ≤ 8 is handled by the same DP with k = n. A direct Monte Carlo simulator of the generative model (200,000 runs) supports arbitrary correlation matrices across generations.

---

## 4 Theory

Throughout, q_{ij} = 1 − p_{ij}, and we use c = 1, t = 1, r = 1 unless stated otherwise.

**Proposition 1 (chain reduction).** For the chain with γ = 0, S_target = S_strict = ∏_{j=1}^{n} p_{j−1,j}. For the chain with γ = 1, S_target = p_{n−1,n}.

*Proof.* With γ = 0, level j is T iff level j−1 is T and it detects; induction gives the product. With γ = 1 the status of j−1 is irrelevant. ∎

The product form of the chain is therefore entirely a *propagation* assumption.

**Proposition 2 (correlation erodes within-level redundancy; Slepian).** Fix overseers with marginal detection probabilities p₁, …, p_s under the one-factor copula with correlation ρ. Then D(p; ρ) = P(at least one detects) is non-increasing in ρ, with D(p; 0) = 1 − ∏(1−pᵢ) and D(p; 1) = max pᵢ. Hence S_strict is non-increasing in ρ for every window topology, and at ρ = 1 it equals the strict success of keeping only each level's best overseer.

*Proof.* The all-miss event is ∩ᵢ{−Yᵢ ≤ −tᵢ} for an equicorrelated centred Gaussian vector (−Yᵢ). By Slepian's inequality (Slepian, 1962) its probability is non-decreasing in the common correlation. The endpoints are immediate. ∎

This is a textbook consequence of Slepian's inequality. We include it because Theorem 3 contrasts with it.

**Setting for Theorem 3.** Consider a stationary window-k design (no anchor, γ = 0, m_G = m_H, fixed step) with distance-m miss probabilities q₁ ≤ q₂ ≤ … ≤ q_k in (0,1). Window states are s ∈ {T,C}ᵏ, oldest bit first. The state C^k is absorbing: no trusted overseer remains. Let M be the substochastic transition matrix on the 2ᵏ−1 non-absorbed states. These states form an irreducible, aperiodic class: k consecutive detections lead from any state to T^k, and T^k has a self-loop. So M has a Perron root λ ∈ (0,1) and a unique quasi-stationary distribution (QSD) ν with νM = λν and Σν = 1. Define the per-level hazard h = 1 − λ. Let πᵢ(ρ) be the probability that trusted overseers at distances i+1, …, k all miss (i = 0, …, k−1), and let

  R_k(ρ) = ∏_{i=0}^{k−1} πᵢ(ρ),

so that R_k(0) = ∏ₘ qₘᵐ and R_k(1) = ∏ₘ qₘ. Write a = q₁/(λ + q₁ − q_{k−1}) for k ≥ 3, assuming q_{k−1} < λ + q₁ (true for small q), and a = q₁/(λ + q₁) for k = 2.

**Theorem 3 (propagation blocking under full collusion).** For every ρ ∈ [0,1]:

(i) *Identity.* h = q_k·ν(TC^{k−1}).

(ii) *Universal bounds.* (1 − k·a·λ^{−(k−2)})·R_k(ρ) ≤ h ≤ λ^{−(k−1)}·∏_{m=1}^{k} qₘ.

(iii) *k = 2.* With π₀ = P(both trusted overseers miss) and D₂ = λ − q₁(1−q₂)/λ > 0,
  R₂·(1 − (1+λ⁻¹)·π₀/D₂) ≤ h ≤ R₂/D₂.

*Corollaries.* As q → 0 (so λ → 1 and a = O(q₁)):
- (a) at ρ = 1, h = ∏ₘ qₘ·(1 + O(kq₁));
- (b) at k = 2, for every ρ, h = R₂(ρ)·(1 + O(q₁)); in particular h = q₁q₂²(1+O(q₁)) at ρ = 0;
- (c) for k ≥ 3 and every ρ, h ≥ R_k(ρ)·(1 − O(kq₁)).

For the chain (k = 1), h = q₁.

*Proof.* (i) Summing νM = λν over states gives λ = Σ_s ν(s)·(row sum of s). Every row sum is 1 except that of TC^{k−1}, the only state from which one miss leads to C^k. Its single trusted overseer is at distance k, so its row sum is 1 − q_k.

(ii, lower) From νM^{k−1} = λ^{k−1}ν we get ν(TC^{k−1}) ≥ λ^{−(k−1)}·ν(T^k)·M^{k−1}(T^k, TC^{k−1}). The only (k−1)-step path from T^k to TC^{k−1} is k−1 consecutive misses. At the i-th miss (i = 0, …, k−2) the trusted overseers are exactly those at distances i+1, …, k, so the path has probability ∏_{i=0}^{k−2} πᵢ = R_k/q_k. By (i), h ≥ ν(T^k)·R_k.

It remains to bound 1 − ν(T^k). Let a′ = ν(newest bit = C). A new level is missed with probability at most q_d, where d is the distance of its nearest trusted overseer: adding overseers cannot raise the all-miss probability under any copula. From states whose newest bit is T (mass 1 − a′) the miss probability is at most q₁. From the other non-absorbing states it is at most q_{k−1}, because misses from TC^{k−1} are absorbed and do not count. Projecting νM = λν onto "newest bit C" therefore gives λa′ ≤ q₁(1−a′) + q_{k−1}a′, i.e. a′ ≤ a. For k = 2 the second term is absent.

For a window position m ≥ 2, ν(bit m = C) = λ^{−(m−1)}·P_ν(the level added at the first step is C and is not absorbed for m−1 steps) ≤ λ^{−(m−1)}·λa′ = λ^{−(m−2)}a′. A union bound over the k positions gives 1 − ν(T^k) ≤ k·a·λ^{−(k−2)}.

(ii, upper) Again ν(TC^{k−1}) = λ^{−(k−1)}·Σ_s ν(s)·M^{k−1}(s, TC^{k−1}). A (k−1)-step path into TC^{k−1} requires the newest bit of s to be T. That level is then a trusted overseer, at distance j, of the j-th new level, and the j-th new level must be missed. So the path probability is at most ∏_{j=1}^{k−1} q_j, and multiplying by q_k gives the claim.

(iii) For k = 2 the non-absorbed states are TT, TC and CT. Here CT is entered only from TC, and only by a detection, so λν(CT) = (1−q₂)ν(TC). Also λν(TC) = ν(TT)π₀ + ν(CT)q₁, because a level whose only trusted overseer is at distance 1 is missed with probability q₁. Substituting gives ν(TC)·D₂ ≤ π₀, hence the upper bound h = q₂ν(TC) ≤ R₂/D₂. For the lower bound, ν(TT) = 1 − ν(TC) − ν(CT) ≥ 1 − (1+λ⁻¹)ν(TC), and h ≥ ν(TT)R₂ by (ii). ∎

**Conjecture 3′.** For k ≥ 3 and ρ ∈ [0,1), h = R_k(ρ)·(1 + O(q_k)).

The leading-order intuition is that the cheapest route into TC^{k−1} starts from T^k. The difficulty in proving it is that the upper-bound argument in (ii) charges each path only q_j per step, which is too generous when ρ < 1. A proof would need to account for the additional misses required to create the C bits in the starting state.

*Evidence.* We set qₘ = ε·aₘ with a = (1, 1.4, 1.8, 2.3, 2.8) and computed the exact QSD hazard in 60-digit arithmetic for ρ ∈ {0, 1}, and in double precision for ρ = 0.5 (`r1_r5_theorem3.json`, Fig. `fig_r1_theorem3.png`). At ρ = 0, h/R_k equals 1.119, 1.010 and 1.0020 at ε = 0.1, 0.01 and 0.002 for k = 3. For k = 4 and 5 it is 1.0020 at ε = 0.002. Empirically h/R_k − 1 ≈ +q₁ at ρ = 0 and ≈ −q₁ at ρ = 1, for k = 2 to 5. At ρ = 0.5, the ratio for k = 3 is 1.0023 at ε = 0.002. Separately, on 259 random (q, ρ, k ≤ 5) configurations, the bounds of Theorem 3 hold in every case, and the lower bound is non-vacuous in 247. The identity (i) agrees with 1−λ to a relative error of 1.8·10⁻⁷ wherever h > 10⁻⁸; below that, 1−λ is dominated by floating-point round-off.

**Interpretation.** The chain becomes irrecoverable at rate q₁ per level. A window-k graph becomes irrecoverable at a rate between ∏qₘᵐ and ∏qₘ, which is O(q^k) even when overseer errors are perfectly correlated. Correlation therefore removes the per-level benefit of redundancy (Prop. 2) but not the anti-propagation benefit. We call this *self-healing*: an escaped level drops out of the window after k steps unless the next k−1 levels also escape.

In round 1 we approximated the target-failure probability as a per-level term plus (n−n₀)r_k. That approximation has no error control (it gives 2.4·10⁻³ against an exact 3.2·10⁻³ in Table 2), so we no longer use it quantitatively. All design numbers come from the exact DP.

**Proposition 4 (partial collusion stops accumulation in chains).** For the stationary chain with γ > 0 and per-step miss probability q, trust status is a two-state Markov chain with P(T→C) = q and P(C→T) = γ(1−q). Hence P(level n compromised) → π_C = q/(q + γ(1−q)) geometrically. ∎ (This is an elementary computation.)

---

## 5 Experimental Setup

All numbers come from code we wrote and ran on CPU. `code/experiments.py` (round-1 experiments E1–E8, 50 s) and `code/experiments_r1.py` (revision experiments R0–R7, 218 s wall time on 4 cores) write to `results/`, and `code/analyze_r1.py` writes the summaries in `results/r1_tables.json`. The round-1 outputs regenerate byte-identically.

**Calibrated games (R1).** We could not retrieve the numerical fit parameters of Engels et al. in this environment; network access to arXiv was blocked. We therefore calibrate to what we could verify: their published NSO success rates at a 400-Elo gap (51.7%, 13.5%, 10.0%, 9.4%) and their qualitative slope descriptions. For each game we take three slope pairs (m_G, m_H) consistent with the description:
- Debate: (0.5, 0), (1, 0), (1, 0.25);
- Backdoor Code: (0.1, 1), (0.25, 1), (0.4, 0.8);
- Mafia and Wargames, for which we use no slope information: (1, 1), (0.75, 1), (1, 0.75).

We then solve for the intercept β such that our model's optimal-chain success at Δ = 400 equals the published value (`r1_r1_calibrated.json`; for example, β = −11.8 for Debate with m_H = 0 and β = −18.3 for Backdoor Code with (0.25, 1)). This yields 12 games that each reproduce one published number. Their behaviour at larger gaps depends on the slope pair, so we report ranges over the three pairs. These are *calibrations to one number per game, not the Engels et al. fits*, and the slope pairs are our choices.

**Stylised regimes** from round 1 (symmetric (1, 1, −300), guard-steeper (1, 0.4, −100), houdini-steeper (0.4, 1, −400)) are retained for E1–E8, R3 and R7.

**Design space.** We search budgets B ∈ {16, 32} games over 143 designs:
- chain, n ≤ 32;
- chain-r, r ∈ {2, 3, 4};
- window-k, k ≤ 5, with and without anchor, n ≤ 20;
- full topology, cost ≤ 32;
- hybrids with r = 2 and k ∈ {2, 3}.

γ = 0 throughout the design study (the worst case), and ρ_s = ρ ("eq") or ρ_s = (1+ρ)/2 ("mid").

---

## 6 Results

### 6.1 Validation (E1, R0)

The round-1 DP matches Monte Carlo on 24 configurations (max |z| = 2.30). The extended DP (copies, ρ_s, thresholds, correction, full topology) matches Monte Carlo (200,000 runs each) on 24 further random configurations: max |z| = 2.37, max absolute error 0.0024 for target success and 0.0015 for strict success (`r1_r0_validation.json`). Writing the revision also exposed and fixed a bug: with ρ_s = 1 and t ≥ 2, identical copies must contribute r detections at once. The validation runs after the fix.

### 6.2 Chain baseline and self-healing (E2, E3, E3b)

These results are unchanged from round 1 (`e2_chain.json`, `e3_accumulation.json`, `e3b_hazard.json`).
- **Chain (symmetric regime).** n* = 2, 4, 7 for Δ = 400, 800, 1600, with success 0.410, 0.168 and 0.028.
- **Accumulation at fixed step δ = 100 (ρ = γ = 0), n = 60.** The probability that the target is compromised is 1.000 for the chain, 0.879 for k = 2, 0.314 for k = 3 and 0.111 for k = 4. The chain's failure probability saturates geometrically, 1 − ∏(1−q), rather than growing linearly. In the figure, the leading-order curve is now drawn only where it is below 0.5.

*Table 2 (unchanged numbers). Exact hazard vs. leading order R_k (`e3b_hazard.json`).* Small-q regime: q = (0.023, 0.031, 0.040, 0.053).

| k | ρ | exact h | R_k | ratio | P(target compromised), n = 80 |
|---|---|---|---|---|---|
| 1 | – | 2.32e-2 | 2.32e-2 | 1.000 | 0.847 |
| 2 | 0 | 2.22e-5 | 2.18e-5 | 1.022 | 3.18e-3 |
| 3 | 0 | 1.48e-9 | 1.44e-9 | 1.025 | 5.77e-5 |
| 4 | 0 | 1.19e-14 | 1.16e-14 | 1.025 | 3.06e-6 |
| 2 | 1 | 6.95e-4 | 7.10e-4 | 0.978 | 0.0749 |
| 4 | 1 | 1.49e-6 | 1.53e-6 | 0.977 | 0.0235 |

The ratios are within 1 ± (q₁ + O(q²)), consistent with the corollaries of Theorem 3 and with Conjecture 3′. In the symmetric regime (q = 0.24 to 0.64), ratios range from 0.77 to 1.34 and the leading order is only qualitative.

### 6.3 Matched-cost comparison: is it cross-generation redundancy or just more checks? (R1, R7)

This experiment is the main new test, and it answers reviewer question 3. Table 3 shows the calibrated games at B = 32 with ρ_s = ρ, reporting the range over the three slope pairs per game (`r1_tables.json`, key `r1_ranges_B32`).

*Table 3. Best target success by design family (chain / chain-r / graph / hybrid), B = 32, ρ_s = ρ.*

| Game | Δ | ρ = 0 | ρ = 0.5 | ρ = 0.9 |
|---|---|---|---|---|
| Debate | 800 | 0.44–0.52 / 0.95–0.97 / 1.00 / 0.99 | 0.44–0.52 / 0.81 / 0.97–1.00 / 0.95–0.98 | 0.44–0.52 / 0.60–0.65 / 0.91–0.99 / 0.83–0.92 |
| Mafia | 800 | 0.02–0.03 / 0.19–0.51 / 0.18–0.48 / 0.33–0.65 | 0.02–0.03 / 0.07–0.19 / 0.12–0.36 / 0.16–0.41 | 0.02–0.03 / 0.03–0.07 / 0.08–0.28 / 0.07–0.22 |
| Mafia | 1600 | 0.00 / 0.00–0.42 / 0.00–0.20 / 0.00–0.34 | 0.00 / 0.00–0.12 / 0.00–0.15 / 0.00–0.18 | 0.00 / 0.00–0.02 / 0.00–0.11 / 0.00–0.08 |
| Wargames | 800 | 0.01–0.02 / 0.08–0.27 / 0.08–0.26 / 0.16–0.42 | 0.01–0.02 / 0.04–0.10 / 0.06–0.20 / 0.08–0.23 | 0.01–0.02 / 0.02–0.04 / 0.04–0.15 / 0.03–0.11 |
| Backdoor Code | 400 | 0.10 / 0.34 / 0.33–0.36 / 0.33–0.42 | 0.10 / 0.26 / 0.23–0.26 / 0.20–0.23 | 0.10 / 0.16 / 0.15–0.18 / 0.11–0.13 |
| Backdoor Code | 800 | 0.01–0.02 / 0.04–0.07 / 0.03–0.06 / 0.02–0.07 | 0.01–0.02 / 0.04–0.06 / 0.02–0.05 / 0.02–0.04 | 0.01–0.02 / 0.02–0.03 / 0.02–0.03 / 0.01–0.02 |

Counting winners across the 36 calibrated settings per cell (`r1_winners_target`):

* **ρ_s = ρ (copies as diverse as generations).**
  - At ρ = 0 and B = 16, chain-r is best in 29/36. At B = 32 a cross-generation design (graph or hybrid) is at least as good as chain-r in 26/36, and in 17 of those the hybrid is best.
  - At ρ = 0.5, cross-generation designs are at least as good as chain-r in 27/36 at B = 32.
  - At ρ = 0.9, the pure graph is best in 30/36 at B = 32 and 27/36 at B = 16.
* **ρ_s = (1+ρ)/2 (copies more alike than generations).** Cross-generation designs are at least as good as chain-r in 31/36 at ρ = 0, 31/36 at ρ = 0.5 and 32/36 at ρ = 0.9 (B = 32).
* **Strict success.** Chain-r is best in 36/36 in every cell, except ρ = 0 with ρ_s = mid at B = 32, where a hybrid wins in 3/36. Cross-generation redundancy buys no strict-success advantage: same-generation copies at distance 1 are the strongest overseers.

**Crossover in ρ_s (R7).** For the stylised regimes at B = 32 we swept ρ_s from ρ to 1 (`r1_r7_rho_s.json`).
- *Symmetric, Δ = 1600, ρ = 0.* Chain-r reaches 0.873, 0.698 and 0.510 at ρ_s = 0, 0.2 and 0.4, against 0.633 for the graph. The crossover lies between ρ_s = 0.2 and 0.4.
- *Symmetric, Δ = 800, ρ = 0.* Chain-r (0.974) beats the graph (0.952) only at ρ_s = 0.
- *Guard-steeper, and houdini-steeper at Δ = 800.* The graph beats chain-r even at ρ_s = ρ = 0: 0.989 vs 0.952, 0.949 vs 0.941 and 0.733 vs 0.655 in the three cases.
- *ρ = 0.5.* The graph beats chain-r already at ρ_s = ρ in every regime with non-negligible success, for example 0.451 vs 0.419 (symmetric, Δ = 1600) and 0.838 vs 0.684 (Δ = 800).
- *Hybrids* are often best at small ρ_s, for example 0.985 against 0.974 for chain-r (symmetric, Δ = 800, ρ_s = ρ = 0), but not always: at Δ = 1600 chain-r leads (0.873 vs 0.784).

**Conclusion of 6.3.** The round-1 comparison against a single chain conflated two effects. Measured against matched-cost same-generation redundancy, cross-generation redundancy helps target success through propagation blocking, and only that. It is decisive when overseer errors are correlated, because there per-level redundancy has collapsed (Prop. 2) while propagation blocking has not (Theorem 3). It is also decisive when copies of one model are more alike than models of different generations. It is not needed when independent same-generation checks are cheap and errors are nearly independent. Whether ρ_s exceeds ρ in practice is an open empirical question. Koran et al. (2026) find that identical monitors gain less than diverse ones, whereas Jha (2026) finds that lineage diversity did not help at matched skill.

### 6.4 Where redundancy fails: gap size, not slope sign (R2, R6)

The sensitivity sweep covers m_G ∈ {0.25, 0.5, 0.75, 1}, m_H ∈ {0, 0.5, 1, 1.25}, β ∈ {−500, −300, −100}, Δ ∈ {800, 1600} and ρ ∈ {0, 0.5}, at B = 32: 192 settings (`r1_r2_sensitivity.json`, `r1_r2_summary.json`, Fig. `fig_r1_sensitivity.png`).
- The best graph is never worse than the best chain on target success, and it improves on it by more than 0.1 in 64% (ρ = 0) and 61% (ρ = 0.5) of settings.
- Among Houdini-steeper settings (m_H > m_G), the gain exceeds 0.1 in 46% (ρ = 0) and 42% (ρ = 0.5); for m_H ≤ m_G, in 81%. For example, (m_G, m_H, β) = (0.25, 0.5, −500) at Δ = 1600 goes from 0.23 (chain) to 0.93 (graph).
- The decisive variable is the direct-oversight probability p(0 → target). When it is below 10⁻³, the median gain is 0.0007. Between 0.05 and 0.2, the median gain is 0.60.
- Against chain-r at ρ_s = ρ, the graph is at least as good in 49% of settings at ρ = 0 and 83% at ρ = 0.5. It is never better on strict success.

**Corrections to round-1 claims (R6, `r1_r6_audit.json`).** In round 1 we wrote that "redundancy cannot rescue a game whose Houdini slope exceeds its Guard slope" and that graphs dominate "whenever the per-step game is not Houdini-favoured". Both statements were wrong. In our own houdini-steeper regime, target success rose from 0.500 to 0.998 at Δ = 400 and from 0.100 to 0.733 at Δ = 800 (B = 32); it failed (≤ 0.011) only at Δ = 1600. The correct statement concerns large gaps with tiny per-step detection probabilities, not the sign of the slopes. We also wrote that anchoring helps "mainly at small gaps or Guard-favoured slopes". In fact the anchor was chosen in 25/27 houdini-steeper settings, including all 9 at Δ = 1600, and in 24/27 guard-steeper settings, but in only 7/27 symmetric settings (all at Δ = 400). In the calibrated games, the best graph uses the anchor in 26/27 Debate settings, 13/27 for Mafia and for Wargames, and 0/27 for Backdoor Code, where the full topology is chosen in all 27. The anchor helps whenever the level-0 overseer is relatively strong against the target, which happens when the Guard slope is small *or* the Houdini curve is shifted down. It is not tied to Guard-favoured slopes. In the calibrated Backdoor Code games, no design exceeds 0.42 target success at Δ = 400, 0.071 at Δ = 800 or 0.0018 at Δ = 1600.

### 6.5 Robustness: imperfect correction, threshold aggregation, false alarms (R3)

For four games (symmetric; calibrated Debate (1, 0), Backdoor Code (0.25, 1) and Mafia (1, 1)), Δ ∈ {800, 1600}, ρ ∈ {0, 0.5}, c ∈ {1, 0.9, 0.7, 0.5} and t ∈ {1, 2}, we re-optimised over the full design space under five objectives (`r1_r3_robustness.json`; aggregate in `r1_tables.json`, key `r3_agg`). We drop settings in which every design scores below 0.02.

| objective | settings | cross-generation best | chain-r best | of which k ≥ 3 or full |
|---|---|---|---|---|
| max S_target (φ irrelevant) | 82 | 50 | 32 | 43 |
| S_target − 0.1·E[FA], φ = 0.05 | 65 | 33 | 32 | 26 |
| S_target s.t. P(any FA) ≤ 0.25, φ = 0.05 | 73 | 18 | 53 | 12 |

* **Threshold aggregation hurts graphs most.** With t = 2, distant weak overseers rarely provide the second detection. Cross-generation designs are best in 12/32 nondegenerate settings, against 38/50 with OR. For example, in the symmetric regime at Δ = 800 and ρ = 0.5, the best value falls from 0.857 (hybrid, t = 1) to 0.453 (hybrid, t = 2). The pure graph falls from 0.838 to 0.235, and chain-r from 0.684 to 0.338.
* **Imperfect correction degrades all families, without reversing the ranking.** Cross-generation designs are best in 14/24, 15/22, 12/19 and 9/17 nondegenerate settings at c = 1, 0.9, 0.7 and 0.5. In the symmetric regime (Δ = 800, ρ = 0.5, t = 1), c = 0.5 gives 0.211 (graph), 0.110 (chain-r) and 0.032 (chain).
* **False alarms are the binding constraint for deep graphs.** This answers reviewer question 5. With a false-alarm cost of 0.1 per falsely flagged level at φ = 0.05, k ≥ 3 designs remain optimal in 26/65 settings. Under the hard constraint P(any false flag) ≤ 0.25 at φ = 0.05 with OR aggregation, k ≥ 3 is never optimal (0/41), and the optimum becomes a short chain-r or a (3; 2) graph. The round-1 recommended designs had P(any false flag) = 0.54–0.79 at φ = 0.05 (E8), so they are infeasible under such a budget. With t = 2, cross-generation designs (mostly thresholded hybrids) satisfy the constraint and are optimal in 12/32 of those settings.

### 6.6 Correlation structure: cross-level stealth and distance-dependent correlation (E4, R4)

These results are unchanged from round 1 (E4: symmetric, Δ = 1600, n = 16, γ = 0):
- At ρ = 1, strict success collapses to the chain value (0.012) for every k, whereas target success at k = 4 is 0.543.
- With a stealth factor shared across levels (cross-level correlation up to 0.6), the topology ranking is preserved.

The round-1 one-factor copula forces equal correlation among all overseers of a level. In R4 we instead simulate a distance-decaying truth (`r1_r4_distance_corr.json`): corr(overseer i, overseer i′) = ρ_adj·η^{|i−i′|−1}, with (ρ_adj, η) ∈ {(0.5, 0.6), (0.5, 0.9), (0.8, 0.85), (0.9, 0.95)}. We compare this against the one-factor model with ρ set to the window's mean pairwise correlation. Across k = 1 to 4, the one-factor prediction errs by at most 0.014 in target success and 0.012 in strict success, the ranking over k is unchanged, and the one-factor model is slightly optimistic (by 0.009–0.014 at k = 4). Within this family, the conclusions do not depend on the equicorrelation assumption.

### 6.7 Collusion (E5)

These results are unchanged. At ρ = 0.3, chain target success is 0.012, 0.388 and 0.760 at γ = 0, 0.2 and 1, whereas window-3 gives 0.734, 0.866 and 0.910. Prop. 4 matches the DP at n = 200 to four digits.

---

## 7 Measurement Protocol and Falsifiable Predictions

**Measuring ρ and ρ_s.** Under the one-factor model, the correlation between two overseers equals the *tetrachoric correlation* of their miss indicators on a common set of attacks from the same Houdini. ρ̂ is estimated on cross-generation pairs and ρ̂_s on same-generation copies (re-sampled or independently fine-tuned instances). This identification is model-dependent:
- It assumes Gaussian-copula dependence, so heavier tail dependence would make the tetrachoric estimate understate joint misses.
- It assumes non-adaptive attacks. A Houdini that chooses attacks to exploit exactly its overseers' shared blind spots raises the effective ρ at deployment above the value measured on a fixed attack pool.

R4 shows that plugging the mean pairwise ρ̂ into the one-factor model is accurate to about 0.014 when the truth is Gaussian with distance-decaying correlation. It does not cover non-Gaussian or adaptive dependence. The decision quantity of Section 6.3 is ρ̂_s − ρ̂: our model favours cross-generation oversight for target success when this difference is clearly positive or when both correlations are high.

**Measuring γ.** Run an untrusted-monitoring control evaluation in which the monitor is a red-team-modified model instructed to let a particular policy's attacks through while still flagging others. γ is the ratio of its detection rate on that policy's attacks to an honest copy's detection rate (Greenblatt et al., 2024).

**Pre-registered predictions** (for a model family with at least four capability levels):

* **P1 (redundancy decays with similarity).** The detection gain from adding a second monitor will be predicted, within sampling error, by D(p; ρ̂) − max p. *Falsified if* the gain is uncorrelated with ρ̂ or exceeds the ρ = 0 bound.
* **P2 (self-healing).** In a simulated ladder in which escaped models act as colluding monitors:
  - The fraction of runs whose final model is compromised will approach 1 geometrically with depth in the chain, at per-level rate q̂₁.
  - For window-2, the irrecoverable-compromise rate will lie between about π̂₀q̂₂ (the measured joint miss rate of the distance-1 and distance-2 monitors times q̂₂) and q̂₁q̂₂. This follows from Theorem 3(iii) and (ii).
  
  *Falsified if* window-2 accumulation is comparable to the chain's.
* **P3 (chain fragility to γ).** Chain target success will vary by more than an order of magnitude across γ ∈ [0, 0.2], while window-3 target success will vary by less than a factor of two.
* **P4 (matched-cost crossover).** At equal numbers of oversight calls, window designs will beat same-generation copies on final-model trust if and only if ρ̂_s − ρ̂ is clearly positive or ρ̂ is high. *Falsified if* copies win when ρ̂ ≥ 0.5 and ρ̂_s > ρ̂.

**Protocol sketch.** (1) Choose a task with checkable ground truth, such as backdoored code. (2) Fit Guard and Houdini Elo curves on a ladder of 4–6 models, as in Engels et al. (2025). (3) Estimate ρ, ρ_s and γ on held-out attacks. (4) Predict with `code/nso.py`. (5) Run the ladder with real models. This protocol requires LLM access we did not have, so it is proposed, not executed.

---

## 8 Discussion

**What the chain model gets right and wrong.** The chain's pessimism has two separable sources: weak per-step games, and the assumption that one escape is fatal and irreversible. The second is a design choice. A window of trusted earlier generations turns per-level irrecoverable compromise from rate q₁ into a rate of order q^k (Theorem 3). The first source is not fixable by topology. When direct oversight across the full gap succeeds with probability below about 10⁻³, no design within budget 32 helps (Section 6.4).

**Cross-generation versus same-generation redundancy.** The matched-cost comparison sharpens our thesis rather than confirming it in its round-1 form. Cross-generation oversight is not a better way to buy detection: on strict success, same-generation copies always win. It is a way to buy *recoverability* after an escape, and that matters most exactly when errors are correlated, which is when buying more detection stops working. In practice we would expect hybrids, a few copies of each of several generations, to be the robust choice. They were optimal in about half of the low-correlation calibrated settings.

**Which success criterion matters.** If intermediate models are deployed, strict success is the relevant criterion, and our results favour diverse same-generation monitors (the ensemble-monitoring setting of Koran et al., 2026). If intermediate models are sandboxed tools that only oversee the next model, target success is the relevant criterion, and multi-generation oversight retains a large advantage under correlation. Keeping intermediate oversight models non-deployed is itself a safety lever.

**False alarms bound the depth of graphs.** OR aggregation over many weak overseers raises false-flag rates to levels that a human adjudication budget may not absorb (Section 6.5). Thresholded hybrids are a better use of the same budget once false alarms are priced in.

---

## 9 Limitations

* **Calibration, not fitting.** Our games match one published number per game (NSO success at Δ = 400) and use slope pairs consistent with qualitative descriptions. They are not the Engels et al. fits, which we could not retrieve, and they omit saturation plateaus. For Debate with m_H = 0, our linear extrapolation makes target success approach 1 at large Δ, which is a model artefact. Absolute numbers should not be read as forecasts.
* **Theory gap.** Theorem 3 is rigorous. The matching upper bound for k ≥ 3, ρ < 1 (Conjecture 3′) is supported only numerically, and the universal upper bound ∏qₘ is loose there. The theory covers stationary windows without anchor, with γ = 0, c = 1 and t = 1.
* **Elo abstraction and adaptive adversaries.** We compress oversight into one win probability per pair. An adversary aware of the topology can choose attacks that maximise correlation among exactly its overseers. Our ρ is fixed.
* **Binary trust and correction semantics.** In our model, uncorrected flagged levels stay compromised and act as colluding overseers (c < 1). Other semantics, such as discarding a flagged model and retraining, would change costs and timing.
* **False-alarm model.** False flags are independent across instances at rate φ, and each flagged level costs the same. Correlated false positives would favour redundancy slightly. Escalation costs that grow with depth would disfavour it.
* **Cost model.** Every oversight game costs the same, whereas the anchor's (human) time is far scarcer than model time.
* **No real-model experiments.** All results come from our model, our code and our simulations. Section 7 is a proposal.

---

## 10 Conclusion

Nested scalable oversight has so far been analysed as a chain whose success is a product of per-step probabilities. That product form rests on a chain topology and on full collusion of escaped overseers.

With cross-generation oversight windows, irrecoverable compromise accrues at a rate of order ∏qₘᵐ (independent overseers, lower bound proved, leading order conjectured) to ∏qₘ (perfectly correlated overseers, proved), rather than q₁. This propagation blocking survives error correlation, whereas per-level redundancy does not.

At matched cost, same-generation copies are as good when errors are nearly independent and copies are diverse, and better for strict success. Cross-generation windows win on final-model trust when errors are correlated or copies are alike. Threshold aggregation and false-alarm budgets cut the advantage of deep graphs, and nothing rescues a per-step game with a vanishing direct-oversight probability. The quantities that decide which design to use (ρ, ρ_s, γ and the false-alarm rate) are measurable with existing monitoring evaluations. Until they are measured, quantitative NSO forecasts, ours included, are conditional on them.

---

## Reproducibility

Code: `code/nso.py` (model, DPs, simulators, Theorem 3 numerics), `code/experiments.py` (E1–E8, ~50 s), `code/experiments_r1.py` (R0–R7, ~4 min on 4 cores, ~12 CPU-minutes) and `code/analyze_r1.py`. Run `cd code && python3 experiments.py && python3 experiments_r1.py && python3 analyze_r1.py`. Seeds are fixed.

Provenance:
- Every number in Sections 4–6 and in the abstracts comes from these scripts' outputs in `results/`.
- The four Engels et al. success rates and their qualitative slope descriptions are taken from their paper.
- The descriptions of Koran et al., Jha, Zanolini, Overman & Bayati, and Bhatt et al. are taken from their papers' abstracts.

---

## References

Barnes, B. (2020). Debate update: Obfuscated arguments problem. *AI Alignment Forum*. https://www.alignmentforum.org/posts/PJLABqQ962hZEqhdB/debate-update-obfuscated-arguments-problem

Bhatt, A., Rushing, C., Kaufman, A., Tracy, T., Georgiev, V., Matolcsi, D., Khan, A., & Shlegeris, B. (2025). Ctrl-Z: Controlling AI agents via resampling. arXiv:2504.10374.

Bowman, S. R., Hyun, J., Perez, E., et al. (2022). Measuring progress on scalable oversight for large language models. arXiv:2211.03540.

Brown-Cohen, J., Irving, G., & Piliouras, G. (2024). Scalable AI safety via doubly-efficient debate. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*.

Brown-Cohen, J., Irving, G., & Piliouras, G. (2025). Avoiding obfuscation with prover-estimator debate. arXiv:2506.13609.

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., Sutskever, I., & Wu, J. (2023). Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv:2312.09390.

Christiano, P., Shlegeris, B., & Amodei, D. (2018). Supervising strong learners by amplifying weak experts. arXiv:1810.08575.

Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. New York: Arco.

Engels, J., Baek, D. D., Kantamneni, S., & Tegmark, M. (2025). Scaling laws for scalable oversight. arXiv:2504.18530 (NeurIPS 2025).

Goel, S., Struber, J., Auzina, I. A., Chandra, K. K., Kumaraguru, P., Kiela, D., Prabhu, A., Bethge, M., & Geiping, J. (2025). Great models think alike and this undermines AI oversight. arXiv:2502.04313.

Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2024). AI control: Improving safety despite intentional subversion. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2312.06942.

Irving, G., Christiano, P., & Amodei, D. (2018). AI safety via debate. arXiv:1805.00899.

Jha, A. (2026). Decorrelation is not complementarity: Skill, not lineage, governs trusted-monitor ensembles. arXiv:2608.16190.

Kenton, Z., Siegel, N. Y., Kramár, J., Brown-Cohen, J., Albanie, S., Bulian, J., Agarwal, R., Lindner, D., Tang, Y., Goodman, N. D., & Shah, R. (2024). On scalable oversight with weak LLMs judging strong LLMs. In *Advances in Neural Information Processing Systems 37 (NeurIPS)*. arXiv:2407.04622.

Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024). Debating with more persuasive LLMs leads to more truthful answers. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2402.06782.

Koran, E., Yun, Y., Tetef, S., Arnav, B., & Bernabeu-Pérez, P. (2026). Ensemble monitoring for AI control: Diverse signals outweigh more compute. arXiv:2605.15377.

Ladha, K. K. (1992). The Condorcet jury theorem, free speech, and correlated votes. *American Journal of Political Science*, 36(3), 617–634.

Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., & Legg, S. (2018). Scalable agent alignment via reward modeling: A research direction. arXiv:1811.07871.

Overman, W., & Bayati, M. (2026). Calibrating conservatism for scalable oversight. In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306. arXiv:2605.28807.

Slepian, D. (1962). The one-sided barrier problem for Gaussian noise. *Bell System Technical Journal*, 41, 463–501.

Zanolini, L. (2026). Correlated failures of AI approval monitors. arXiv:2610.29178.
