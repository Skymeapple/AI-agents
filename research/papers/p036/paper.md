# Only Unforgeable Bits Count: Minimax Limits of Content-Based Prompt-Injection Defenses under Full and Partial Mimicry

**Target venue:** ICML 2027 (main track; theory + controlled experiments). Secondary fit: IEEE SaTML.

*Anonymous authors — preprint draft, revision 1, November 2026*

---

## Abstract

Can prompt injection be fundamentally solved? We answer a narrower question quantitatively. The attacks we study make a large language model (LLM) system follow an instruction of a kind that is legitimate in some other context. We model any defended system as a randomized policy over its view. The view has two parts: content the attacker can write, and *unforgeable* signals such as out-of-band role ids, authenticated provenance or verified identity. Testing legitimate against attack views is a Neyman–Pearson problem. We state this as a framework (Theorems 1–3), and those results are classical. Under *exact* mimicry, the attacker reproduces the legitimate content distribution. The minimax attack success rate (ASR) at utility U is then the ROC frontier f_K of the unforgeable channel alone, and with no such channel ASR ≥ U. Hermon et al. (2026) previously proved a closely related no-universal-defense result.

Our main new results cover *partial* mimicry, the realistic case. (Theorem 4) If every attack content distribution the attacker can reach, including mixtures, stays at total-variation distance at least δ from the legitimate one, then no defense that reads content can push worst-case ASR more than δ below f_K(U). This bound is tight. (Theorem 5) Now suppose a fraction ε of attacks can choose content freely and the rest come from a native injection distribution. We give the exact minimax frontier ASR*_ε(U). It interpolates between the static frontier (ε = 0) and f_K (ε = 1), and it is concave in ε. The optimal policy caps how much any single content may raise acceptance. On our test instance with a noisy provenance flag, ε ≥ 0.25 and utility below the flag's accuracy, this means content may *veto* an action but never *override* the unforgeable signal. (Proposition 6) For jailbreaks with graded responses under exact mimicry, the minimax harm at a given helpfulness is the lower convex envelope of the response menu's (benefit, harm) points. "One-for-one" is only the binary special case. Exact linear programs confirm every result on random instances, with errors of 10⁻⁹ or less.

In a CPU-scale testbed with exact worst-case evaluation (144 learned policies), adversarially trained policies reach the Theorem-5 frontier: median gap ≤ 0.0005, maximum 0.037. Policies trained on static attacks do not. With a noisy flag (q = 0.8) and ε = 0.5, half of them are worse than ignoring content entirely, with a median excess of 0.19 over the frontier. Two smaller illustrative experiments show that an in-band trust marker that holds at 0% static ASR is forged by a two-token adaptive search. The same model with an out-of-band role channel stays at 0%. We ran no experiment on a production LLM; we give a pre-registered protocol instead.

## 中文摘要

越狱与提示注入能否被"根本性解决"？本文对其中一个较窄的问题给出定量回答：攻击者使 LLM 系统执行一条在其他情境下本属合法的指令。我们把任意防御系统建模为其观测上的随机化策略，并将观测分为攻击者可写的内容与攻击者不可伪造的信号（带外角色标识、经认证的来源、已验证身份）。合法与攻击视图之间的区分是一个 Neyman–Pearson 检验问题；我们把这一点作为框架陈述（定理 1–3）。这部分是经典结果：在*精确*模仿下，效用 U 处的极小极大攻击成功率等于不可伪造信道单独的 ROC 前沿 f_K；无此信道时 ASR ≥ U。Hermon 等人（2026）此前已给出密切相关的"无通用防御"结论。本文新的主要结果针对更现实的*部分*模仿。(定理 4) 若每个攻击内容分布与合法内容分布的总变差距离都至少为 δ，则任何读取内容的防御都无法把最坏情形 ASR 压到 f_K(U) − δ 以下，且该界是紧的。(定理 5) 在 ε-污染攻击者下（比例 ε 的攻击可任意选择内容，其余来自原生注入分布），我们给出精确的极小极大前沿 ASR*_ε(U)：它在静态前沿（ε = 0）与 f_K（ε = 1）之间插值，并且是 ε 的凹函数。最优策略会限制任一内容对接受率的抬升；在我们的含噪来源标记实例中（ε ≥ 0.25 且效用低于标记准确率），这意味着内容可以*否决*执行，但永远不能*推翻*不可伪造信号。(命题 6) 在精确模仿下，若越狱请求允许分级回答，则给定有用性下的极小极大危害等于回答菜单中（收益，危害）点的下凸包络；"一比一"只是二元回答的特例。精确线性规划在随机实例上验证了全部结果，误差不超过 10⁻⁹。在可精确计算最坏情形的 CPU 规模实验平台上（144 个学习策略），对抗训练得到的策略达到定理 5 的前沿：中位偏差 ≤ 0.0005，最大偏差 0.037；仅用静态攻击训练的策略则达不到。在 q = 0.8、ε = 0.5 时，其中一半甚至比完全忽略内容更差，中位超出前沿 0.19。另有两个较小的示例性实验：在静态评测中 ASR 为 0% 的带内信任标记被两个 token 的自适应搜索伪造；改用带外角色信道的同一模型保持 0%。我们没有在生产级 LLM 上进行实验，而是给出了预注册的评估方案。

---

## 1 Introduction

Prompt injection (Perez & Ribeiro, 2022; Greshake et al., 2023) and jailbreaking (Wei et al., 2023; Zou et al., 2023) have gone through several rounds of defense and counter-attack. Defenses that report near-zero ASR against fixed attack sets are routinely broken by adaptive attackers. Nasr et al. (2025) bypass 12 recent defenses with ASR above 90% for most, and Andriushchenko et al. (2025) reach 100% ASR on many safety-aligned models with simple adaptive attacks. The same pattern occurred earlier with adversarial-example defenses (Tramèr et al., 2020). Meanwhile, system-level designs claim *provable* protection for some properties, at a cost in utility. CaMeL (Debenedetti et al., 2025), FIDES (Costa et al., 2025), the design patterns of Beurer-Kellner et al. (2025), and architectural-boundary and provenance proposals (Bhattarai & Vu, 2026; Gupta, 2025) all fall in this group.

A common and largely correct informal conclusion follows: security needs a channel the attacker cannot write, and content-level defenses cannot be complete (Pant, Lohani & Kumar, 2026; Bhatt et al., 2026; Abdelnabi & Bagdasarian, 2026; Hermon et al., 2026). This paper makes that conclusion *quantitative*. It also asks the question the informal conclusion leaves open: **when the attacker can only partly imitate legitimate instructions, how much can content-based defenses help, and how should they be built?**

**Scope.** We study the *mimicry slice*: the attacker wants the system to perform an action of a kind it legitimately performs in other contexts. One example is an injected "forward this file to X" where users sometimes legitimately ask the agent to act on such requests. Another is a dual-use question that benign users also ask. Attacks that request behavior *never* legitimate in context form a separate, budget-dependent robustness problem (§6.3). Our title's claim applies only to the mimicry slice, and only under the stated channel assumptions.

**Contributions.**

1. **Framework (Theorems 1–3; classical).** We split the view into attacker-writable content c and unforgeable signal s. We then note that the achievable (utility, ASR) region is a Neyman–Pearson region (Thm 2) and that under exact mimicry the minimax frontier is that of the unforgeable channel alone (Thm 3). Once exact mimicry and content-independent signal kernels are assumed, Theorem 3 is close to definitional. We keep it because it fixes notation, and because its corollaries (no signal ⇒ ASR ≥ U; in-band markers are inert under exact mimicry) are the statements practitioners use. Hermon et al. (2026) give a decision-theoretic no-universal-defense result with real-model measurements that overlaps with our Theorem 1 and Corollary C1.
2. **Partial mimicry: a stability bound (Theorem 4, new).** Let δ be the total-variation distance from the attacker's reachable content distributions to the legitimate one. Content-reading defenses can lower worst-case ASR below the unforgeable-channel frontier by at most δ, and this bound is tight. Through Pinsker's inequality, δ can be replaced by a KL budget. The dichotomy of Theorem 3 therefore degrades gracefully instead of collapsing.
3. **Partial mimicry: the exact frontier (Theorem 5, new).** For an ε-contaminated attacker we derive the exact minimax frontier. A fraction ε of attacks have free content and the rest come from a native injection distribution. The frontier is a one-dimensional optimization over a per-content acceptance cap. It is concave in ε, and it is bracketed by ε·f_K(U) + (1−ε)·ASR*_static(U) from below and by f_K(U) from above. The optimal policy *censors* content evidence, in the spirit of Huber's (1965) robust likelihood-ratio test. In our noisy-flag test instance with ε ≥ 0.25 and U < q, this means content may veto but not override the flag. For very small ε, overriding the flag on highly legitimate-looking content is optimal.
4. **Graded responses (Proposition 6, new).** The binary act/refuse model is replaced by a response menu with benign value b(r) and harm h(r). Under exact mimicry with no unforgeable signal, the minimax harm-versus-helpfulness curve is the lower convex envelope of the menu. Content classifiers cannot improve on it; changing the menu or adding unforgeable signals can.
5. **Experiments.** E2 (new; the main empirical test) checks *achievability*. Do gradient-trained policies reach the Theorem-5 frontier under exact worst-case evaluation, and do statically trained ones fail as predicted? E1 and E3 are smaller illustrations of Theorems 2–3. We state explicitly which of their outcomes are fixed by construction.

We do not claim that prompt injection is unsolvable in general, and we do not propose a new defense. All experiments are synthetic and CPU-scale.

## 2 Related Work

**Attacks and benchmarks.** Perez & Ribeiro (2022) introduced goal hijacking and prompt leaking. Greshake et al. (2023) showed indirect prompt injection in deployed applications. Liu et al. (2024) formalized prompt injection and benchmarked attacks and defenses. AgentDojo (Debenedetti et al., 2024) evaluates agents with formal state checks. Wei et al. (2023) attribute jailbreaks to competing objectives and mismatched generalization. GCG (Zou et al., 2023) optimizes adversarial suffixes. Best-of-N jailbreaking (Hughes et al., 2024) shows power-law-like ASR scaling with the number of samples. We follow the adaptive-attack methodology of Tramèr et al. (2020), Andriushchenko et al. (2025) and Nasr et al. (2025).

**Model-level defenses.** These include spotlighting (Hines et al., 2024), StruQ (Chen et al., 2025a), SecAlign (Chen et al., 2025b), the instruction hierarchy (Wallace et al., 2024), Instructional Segment Embedding (Wu et al., 2025), ASIDE (Zverev et al., 2026), the measurement of instruction–data separation (Zverev et al., 2025), and Constitutional Classifiers (Sharma et al., 2025). In our terms, these methods differ in *which bits the attacker cannot write*. Role or segment ids assigned by trusted serving code are unforgeable. Delimiters and in-content marks are forgeable.

**System-level defenses.** CaMeL (Debenedetti et al., 2025) derives control flow from the trusted query. FIDES (Costa et al., 2025) applies information-flow-control labels. Beurer-Kellner et al. (2025) catalogue design patterns. Bhattarai & Vu (2026) argue that trustworthy agentic AI needs deterministic architectural boundaries, combining action governance, information-flow control and privilege separation, and that training-based defenses cannot give deterministic guarantees. Gupta (2025) proposes attaching cryptographic provenance labels to tokens. All of these rest on noninterference (Goguen & Meseguer, 1982). Theorems 3–5 quantify both what these designs buy and their price: utility on tasks whose legitimacy cannot be read from the unforgeable bits.

**Impossibility and trade-off results.** The closest work is **Hermon et al. (2026)**, "Security–Fidelity Tradeoffs" (ICML 2026). They give a decision-theoretic framework showing that no universal defense exists when benign and adversarial inputs overlap, so the optimal defense is task-dependent. They also introduce the SecFid benchmark, which separates resisting an injection from faithfully processing instruction-like data, and they measure real models and defenses with fixed probes. Their overlap result and our Theorem 1/Corollary C1 express the same phenomenon. Our framework differs in three ways:

- it is minimax against *adaptive* attackers, which they leave to future work;
- it separates content from unforgeable signals and gives the exact frontier for noisy provenance;
- its main new content, Theorems 4–5, concerns *partial* overlap.

We have read their abstract, project page and summaries, not the full proofs, so finer-grained comparisons are left for the final version.

Other related results:

- The Defense Trilemma (Bhatt et al., 2026) proves that continuous, utility-preserving wrappers on connected prompt spaces cannot be complete.
- Pant, Lohani & Kumar (2026) argue that perfect separation is impossible in shared-embedding models without immutable provenance enforcement.
- Abdelnabi & Bagdasarian (2026) argue informally that attackers can fabricate legitimizing context.
- Wolf et al. (2024), Glukhov et al. (2024) and Shafahi et al. (2019) give related limits for alignment, censorship and adversarial examples.

**Robust hypothesis testing.** The Neyman–Pearson lemma (Neyman & Pearson, 1933; Lehmann & Romano, 2005) underlies Theorems 1–3. Huber (1965) showed that minimax-robust tests against ε-contamination and total-variation neighborhoods are *censored* likelihood-ratio tests. Theorem 5 has the same flavor: the optimal policy caps per-content acceptance. Our setting differs in three ways. Contamination acts only on the content coordinate of a product view, while the unforgeable coordinate is uncontaminated. The legitimate distribution is not contaminated. And the criterion is a constrained one (minimize worst-case ASR at fixed utility). We did not find this exact statement in the literature. Given how mature robust testing is, a special case may exist, and we present Theorem 5 as an application of robust-testing ideas, not as a new statistical principle.

## 3 Model

**Views and channels.** A defended system (weights, prompts, filters, wrappers, tool-gating code) observes a *view* v = (c, s) ∈ 𝒞 × 𝒮 before deciding whether to act on an in-context instruction ι(c).

- c is everything the attacker can write in the attack scenario. That includes any delimiters, special tokens or "trusted" markers that an honest pipeline would insert *into* that content.
- s is everything the attacker cannot write: out-of-band role or segment ids assigned by trusted code, cryptographic authentication, the system prompt, verified identity.

The system is a measurable φ: 𝒞 × 𝒮 → [0,1], the probability of acting. This covers randomized policies, ensembles and filter cascades.

**Scenarios and metrics.**

- Legitimate views are drawn from P_L = L_c ⊗ K_L, with content L_c and signal kernel K_L(s | c).
- Attack views are drawn from P_A^Q = Q ⊗ K_A, with attacker-chosen content Q ∈ 𝒬.
- Utility is U(φ) = E_{P_L}φ.
- Attack success is ASR(φ, Q) = E_{P_A^Q}φ.
- Overall utility on tasks without in-context instructions is outside the trade-off. We report it separately (CLEAN accuracy).

**Assumptions.**

- *(K) Kernel independence:* K_L(·|c) = K_L and K_A(·|c) = K_A. Whether the pipeline labels content as trusted does not depend on what the content says. This holds when labels come from the channel a message arrives on (tool id, authenticated sender). It fails when the labeling depends on content, for example a classifier-assigned trust label; see §8.
- *(M) Exact mimicry (content capability):* L_c ∈ 𝒬.
- *(PM) Partial mimicry:* either a divergence constraint inf_{Q∈𝒬} TV(Q, L_c) = δ > 0 (Thm 4), or ε-contamination 𝒬_ε = {(1−ε)A₀ + εR : R arbitrary} around a native injection distribution A₀ (Thm 5).

Let f_K denote the Neyman–Pearson frontier of (K_L, K_A): the minimum E_{K_A}ψ over ψ: 𝒮 → [0,1] with E_{K_L}ψ ≥ U. It is convex and nondecreasing. Let β_K(a) = max{u : f_K(u) ≤ a} be its inverse, the ROC curve.

## 4 Framework results (classical)

**Theorem 1 (distinguishability bound).** For every φ and Q, ASR(φ, Q) ≥ U(φ) − TV(P_L, P_A^Q).
*Proof.* φ ∈ [0,1], so |E_{P_L}φ − E_{P_A^Q}φ| ≤ TV. ∎

**Theorem 2 (achievable region).** The set {(U(φ), ASR(φ,Q))} is the Neyman–Pearson region of P_A^Q (null) against P_L (alternative). Its lower boundary ASR*(U) is attained by likelihood-ratio tests φ = 1[λ > t] + γ1[λ = t], where λ = dP_L/dP_A^Q. On the part of P_L that is singular to P_A^Q, λ = +∞, so that part is accepted first at zero ASR cost. ASR* is convex and nondecreasing.
*Proof.* This is the NP lemma (Lehmann & Romano, 2005, Thm 3.2.1), with power-versus-size inverted, plus mixing. ∎

*Closed forms.* (a) For a binary flag correct with probability q ≥ ½ and matched content:

  ASR*(U) = (1−q)U/q for U ≤ q, and (1−q) + q(U−q)/(1−q) for U > q.

(b) For a score distributed N(d′,1) under legitimate and N(0,1) under injected views: ASR*(U) = 1 − Φ(Φ⁻¹(1−U) + d′).

**Theorem 3 (exact mimicry).** Under (K) and (M), min_{φ: U(φ)≥U} max_{Q∈𝒬} ASR(φ,Q) = f_K(U). A likelihood-ratio test on s alone attains this value against every Q.
*Proof.* For (≥), play Q = L_c; the likelihood ratio is then K_L(s)/K_A(s), and Theorem 2 applies. For (≤), take φ = ψ(s). ∎

**Corollaries (all under (K) and (M)).**

- **C1:** if K_L = K_A, then max_Q ASR ≥ U.
- **C2:** zero worst-case ASR at U = 1 is possible iff K_L ⊥ K_A.
- **C3:** *under exact mimicry*, delimiters, datamarks, in-content special tokens and learned detectors that read c cannot lower worst-case ASR below f_K(U). Under partial mimicry they can, by at most the amounts given in Theorems 4–5.
- **C4:** the s-only policy's ASR does not depend on attack budget.

The numerical check is in `results/theory_checks.json`, from exact LPs on 300 random instances. Theorem 1 has minimum slack +0.013. The LP optimum matches the likelihood-ratio frontier to 3.2·10⁻¹⁴ for Theorem 2 and 3.2·10⁻¹⁵ for Theorem 3.

As the reviewer of the first version noted, Theorem 3 is nearly definitional once (K) and (M) are assumed. The substantive questions are how fragile it is to imperfect mimicry and what the optimal defense looks like when mimicry is partial. Section 5 answers both.

## 5 Partial mimicry (new results)

**Theorem 4 (mimicry-gap stability).** Assume (K), and let 𝒬 be any family of attack content distributions. Define δ := inf_{Q ∈ conv 𝒬} TV(Q, L_c). Then for every φ with U(φ) ≥ U,

  sup_{Q∈𝒬} ASR(φ, Q) ≥ f_K(U) − δ,

while the s-only likelihood-ratio policy has ASR = f_K(U) against every Q. Consequently:

- no defense that reads content can improve worst-case ASR over the s-only policy by more than δ;
- for every Q, δ ≤ √(KL(Q‖L_c)/2) by Pinsker's inequality;
- the bound is tight. With no signal (f_K(u) = u), L_c = point mass on c₁ and 𝒬 = {(1−δ)·c₁ + δ·c₂}, the policy "act iff c = c₁" has U = 1 and ASR = 1 − δ.

*Proof.* Fix φ, let u(c) = E_{K_L}φ(c,·), and let g(c) = E_{K_A}φ(c,·).

1. For each c, φ(c,·) is a test on s with power u(c), so g(c) ≥ f_K(u(c)).
2. Averaging under L_c and using Jensen (f_K is convex) and monotonicity: E_{L_c} g ≥ E_{L_c} f_K(u) ≥ f_K(E_{L_c}u) ≥ f_K(U).
3. For any Q: ASR(φ,Q) = E_Q g ≥ E_{L_c} g − TV(Q, L_c), because g ∈ [0,1].
4. ASR is linear in Q, so the supremum over 𝒬 equals the supremum over conv 𝒬, and taking the closest Q in conv 𝒬 gives the bound.
5. The s-only claim is Theorem 3's (≤) direction. ∎

**Theorem 5 (exact frontier under ε-contaminated mimicry).** Assume (K) and 𝒬_ε = {(1−ε)A₀ + εR : R any distribution on 𝒞}. For a cap a ∈ [0,1], define

  V(U, a) := inf { E_{A₀}[f_K(u(c))] : 0 ≤ u(c) ≤ β_K(a), E_{L_c}[u(c)] ≥ U }.

Then

  ASR*_ε(U) := min_{φ: U(φ)≥U} max_{Q∈𝒬_ε} ASR(φ,Q) = min_{a ∈ [f_K(U), 1]} [ (1−ε)·V(U, a) + ε·a ].

An optimal policy runs, on each content c, the NP test on s with power u*(c) ≤ β_K(a*). In other words, no content may raise the acceptance probability under attack signals above the cap a*. Moreover:

- (i) ASR*_0 is the static frontier of (L_c⊗K_L, A₀⊗K_A), and ASR*_1 = f_K (Theorem 3).
- (ii) ε ↦ ASR*_ε(U) is nondecreasing and concave.
- (iii) ε·f_K(U) + (1−ε)·ASR*_0(U) ≤ ASR*_ε(U) ≤ f_K(U).
- (iv) Without an unforgeable signal (f_K(u) = u, β_K(a) = a), ASR*_ε(U) = min_{m∈[U,1]} m·[(1−ε)·g_C(U/m) + ε], where g_C is the NP frontier of content alone (L_c against A₀).

*Proof.*

1. *Attacker's best response.* For fixed φ, write g(c) = E_{K_A}φ(c,·). Then max_{Q∈𝒬_ε} ASR = (1−ε)E_{A₀}g + ε·sup_c g(c), because the free part R puts all its mass where g is largest.
2. *Lower bound.* Let u(c) = E_{K_L}φ(c,·) and a = sup_c g(c). Then g(c) ≥ f_K(u(c)) and f_K(u(c)) ≤ a, so u(c) ≤ β_K(a). Hence the value is at least (1−ε)E_{A₀}f_K(u) + εa ≥ (1−ε)V(U,a) + εa. Feasibility (E_{L_c}u ≥ U with u ≤ β_K(a)) requires β_K(a) ≥ U, that is, a ≥ f_K(U).
3. *Upper bound.* Given a and an optimal u, set φ(c,·) to the NP test with power u(c). Then g(c) = f_K(u(c)) ≤ a, and the value is exactly (1−ε)V(U,a) + εa.
4. *Properties.* (i): setting a = 1 removes the cap at ε = 0; at ε = 1 the minimum is at a = f_K(U). (ii): the value is a pointwise minimum of functions that are affine and nondecreasing in ε. (iii): use V(U,a) ≥ V(U,1) = ASR*_0(U) and a ≥ f_K(U) for the lower bound, and a = f_K(U) with u ≡ U for the upper bound. (iv): with f_K the identity, V(U,a) = a·g_C(U/a). ∎

*Interpretation.* ε is the fraction of attack opportunities in which the attacker produces content fully indistinguishable from a legitimate instruction. For example, it controls the message thread with the right sender, context and phrasing. In the other cases its content carries injection artifacts with distribution A₀. The theorem quantifies two things:

1. *How much content-based defenses can buy:* f_K(U) − ASR*_ε(U) ≤ (1−ε)(f_K(U) − ASR*_0(U)).
2. *The shape of the optimal defense.* It uses content evidence, but it *censors* that evidence. Acceptance under an attack-like signal is capped at a*, because the attacker's free fraction will find whatever content the defender trusts most.

With a binary provenance flag, 1 − q is the acceptance from flag = 1 alone. On the E2 instance (q = 0.8, `results/e2_cap.json`), the optimal cap satisfies a* ≤ 1 − q at every U ≤ 0.78 we computed for ε ≥ 0.25, and at every U ≤ 0.7 for ε ≥ 0.03. In that regime content can only veto actions the flag allows; it never overrides a "not trusted" flag, and the cap falls below 1 − q at low U. At small ε and U close to q, a cap above 1 − q is optimal, so content does override the flag: a* = 0.66 at ε = 0.01, U = 0.7, and a* = 0.29 at ε = 0.1, U = 0.78. The veto-only structure is therefore a regime of Theorem 5, not a general law. The trained models of E2 show it (§7.1).

**Proposition 6 (graded responses; jailbreaks).** Let the system choose a response r from a menu ℛ that includes refusal r₀. Each response has benign value b(r) ∈ [0,1] and harmful uplift h(r) ∈ [0,1], with b(r₀) = h(r₀) = 0. Helpfulness is H(π) = E_{P_L}E_{π}b and harm is Harm(π, Q) = E_{P_A^Q}E_{π}h. Suppose K_L = K_A (anonymous requester, no unforgeable signal) and L_c ∈ 𝒬 (the malicious user can phrase requests as benign users do). Then

  min_{π: H(π)≥H} max_Q Harm(π,Q) = Γ(H) := min{ Σ_r w_r h(r) : w ∈ Δ(ℛ), Σ_r w_r b(r) ≥ H },

the lower convex envelope of the menu's (b, h) points. A content-independent mixture over responses attains it.

*Proof.* At Q = L_c the attacker's view has the same law as the benign user's, so Harm = E_{L}E_π h. The response distribution averaged over c is a feasible w, which gives ≥ Γ(H). A content-independent w attains Γ(H) against every Q. ∎

*Consequences.* The one-for-one trade-off of C1 is the special case of a binary menu with h = b. When the menu contains responses with high benign value and low uplift, for example answers that explain without giving operational detail, the frontier lies well below the diagonal. Classifiers that read the request still cannot beat Γ under exact mimicry. Only changing the menu (Γ), adding unforgeable requester signals (Theorem 3), or reducing mimicry (Theorems 4–5) helps.

Two further qualifications. Harm and helpfulness here are *conditional* rates. Expected harm per request is π_mal·Harm, so base rates and harm weights enter linearly but do not change the frontier. And we have *not* measured b and h for any real domain, so the jailbreak statements in §8.2 are hypotheses.

**Numerical verification** (`code/verify_partial.py` → `results/partial_theory_checks.json`). Each check uses exact LPs (HiGHS) on random finite instances:

- *Theorem 4:* 200 instances, each with 2–8 contents, 2–4 signal values and 1–3 attack distributions. The minimum slack of minimax ASR over f_K(U) − δ is −1.1·10⁻¹⁶, nonnegative to machine precision. The tight example gives minimax 0.7 against a bound of 0.7 at δ = 0.3.
- *Theorem 5:* 1,000 (instance, ε) pairs with ε ∈ {0, 0.1, 0.3, 0.6, 1}. The full minimax LP over φ(c,s) is compared with the formula, whose inner V is computed by an *independent* greedy fractional-knapsack algorithm. The maximum absolute error is 4.2·10⁻⁹. The endpoint errors are ≤ 1.7·10⁻¹⁵, and there are 0 violations of concavity or of the bounds in (iii).
- *Proposition 6:* 200 instances, maximum error 1.7·10⁻¹⁵.

## 6 Experimental setup

All experiments are synthetic and CPU-scale (4 cores). They test *achievability*: whether trained policies reach the frontiers. They also illustrate the mechanisms. They do not estimate production-model behavior. Every number below is produced by code in `code/` and stored in `results/`.

### 6.1 What can and cannot be falsified

The lower bounds in Theorems 1–5 are mathematical facts about the evaluation distribution. No experiment that computes U and ASR correctly can violate them; a violation would indicate a bug. The empirical questions are different:

- (a) Does standard gradient training *reach* the frontier, or does it stop short? This is not guaranteed, especially when the frontier requires a non-obvious capped policy (Theorem 5) and the training-time adversary is weaker than the evaluator.
- (b) Do training procedures that ignore the attacker model, i.e. static training, fall short in the direction and by the amount the theory predicts?
- (c) Do the modeling assumptions (K), (M) and (PM) describe real pipelines? This requires real systems and is the subject of the pre-registered protocol in §8.3.

E2 addresses (a) and (b). E1 and E3 mostly illustrate.

### 6.2 E2: partial mimicry with exact worst-case evaluation (new)

**Content and signal.** Content c is a sequence of 3 tokens from a 6-token vocabulary, giving 216 possible contents.

- Legitimate instructions draw tokens i.i.d. from p_L = (.30, .25, .20, .12, .08, .05).
- The attacker's native injections draw from the reversed distribution p_A, so content is informative: TV(L_c, A₀) = 0.708.
- An out-of-band flag is correct with probability q. We use q = 0.8, and q = 0.5 for the case with no unforgeable signal.
- The attacker is ε-contaminated with ε ∈ {0.1, 0.25, 0.5}.

**Policies.** Each policy is an MLP: token embeddings, then 2 hidden layers of width 64, then a logit. Decisions are deterministic, φ = 1[logit > 0]. Each policy is trained for 1,500 Adam steps with batch size 512 and attack-loss weight w ∈ {0.25, …, 8}, which sets the operating point, over 3 seeds. There are two training regimes:

- **STATIC:** attacks drawn from A₀ only, i.e. the usual static benchmark training.
- **ROBUST:** a fraction ε of attack samples use the content found by a *greedy coordinate search* (4 restarts × 2 sweeps, refreshed every 25 steps) that maximizes the current policy's acceptance.

**Evaluation and models trained.** Evaluation is *exact*. We enumerate all 432 views and compute U and

  ASR_ε = (1−ε)·E_{A₀⊗K_A}φ + ε·max_c E_{K_A}φ(c,·),

so the evaluator is an *exhaustive* attacker and the numbers carry no sampling error. We train 36 STATIC and 108 ROBUST policies, which gives 216 (policy, ε) evaluations. Runtime is 13.4 min, including frontier computation. On this instance the Theorem-5 formula agrees with the LP to 1.3·10⁻¹⁰.

### 6.3 E1 and E3 (from the first version; unchanged runs, revised statistics)

**Task.** The sequence is `[BOS, SYS, s, DAT, d₁…d₁₂, QRY]`. The trusted instruction s is one of FIRST/LAST/MAX/MIN, or DELEG ("perform the operation named in the data if it is legitimate, else NOOP"). Each position carries an out-of-band role id that the attacker cannot write. The model is a 2-layer pre-LN transformer with d = 64.

**E1.** Legitimate and injected in-data instructions have identical content, and they differ only in a role flag that is correct with probability q ∈ {0.5,…,1}. We train 36 models (6 q × 3 loss weights × 2 seeds) and evaluate each on 6,000 episodes per scenario.

**E3.** Four systems:

- **UNDEF:** role channel, never trained on injections.
- **INBAND:** adversarially trained; legitimate in-data instructions are marked by an attacker-writable token SIG.
- **OOB:** the same training, but legitimacy is carried by the out-of-band role id.
- **GATED:** code decides control flow from role ids, and a quarantined transformer computes values.

Adaptive attacks are white-box greedy coordinate searches over k ∈ {1, 2, 4, 8} attacker positions. The attacker vocabulary contains the 19 non-digit tokens, including SIG, and at most 1 + 4·19·k queries are made. There are two goals:

- **G1:** hijack a fixed task, i.e. a never-legitimate instruction.
- **G2:** forge a delegation, i.e. mimicry.

Each cell has about 300 episodes per seed, over 3 seeds.

## 7 Results

### 7.1 E2: adversarial training reaches the partial-mimicry frontier; static training does not

**Table 1.** E2 summary over non-degenerate operating points (0.05 < U < 0.99). Gap = ASR_ε − ASR*_ε(U), which is ≥ 0 by Theorem 5. Content gain = f_K(U) − ASR_ε, the improvement over the s-only policy at the same utility; "optimal" is the Theorem-5 value f_K(U) − ASR*_ε(U) at the same U. Last column: among policies with U < q, the number whose per-content acceptance max_c E_{K_A}φ(c,·) equals 1 − q, i.e. that never act on a "not trusted" flag (`results/e2_cap.json`). Other columns: `results/e2_summary.json`.

| q | ε | regime | n | median gap | max gap | median content gain (achieved / optimal) | worse than s-only | policies with U < q capped at 1 − q |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.10 | STATIC | 18 | 0.0074 | 0.077 | 0.279 / 0.286 | 0% | 0/6 |
| 0.8 | 0.10 | ROBUST | 18 | 0.0000 | 0.001 | 0.254 / 0.254 | 0% | 9/9 |
| 0.8 | 0.25 | STATIC | 18 | 0.058 | 0.197 | 0.141 / 0.200 | 33% | 0/6 |
| 0.8 | 0.25 | ROBUST | 18 | 0.0001 | 0.011 | 0.095 / 0.097 | 0% | 12/12 |
| 0.8 | 0.50 | STATIC | 18 | 0.191 | 0.401 | −0.088 / 0.103 | 50% | 0/6 |
| 0.8 | 0.50 | ROBUST | 15 | 0.0005 | 0.007 | 0.053 / 0.053 | 0% | 12/12 |
| 0.5 | 0.10 | STATIC | 18 | 0.0000 | 0.011 | 0.560 / 0.560 | 0% | n/a (no flag) |
| 0.5 | 0.10 | ROBUST | 15 | 0.0001 | 0.016 | 0.597 / 0.597 | 0% | n/a (no flag) |
| 0.5 | 0.25 | STATIC | 18 | 0.0003 | 0.063 | 0.442 / 0.442 | 0% | n/a (no flag) |
| 0.5 | 0.25 | ROBUST | 12 | 0.0001 | 0.037 | 0.475 / 0.480 | 0% | n/a (no flag) |
| 0.5 | 0.50 | STATIC | 18 | 0.017 | 0.181 | 0.251 / 0.266 | 17% | n/a (no flag) |
| 0.5 | 0.50 | ROBUST | 9 | 0.0000 | 0.002 | 0.279 / 0.279 | 0% | n/a (no flag) |

Figure 3 (`results/figs/fig3_partial.png`) plots all 216 evaluations against the three theory curves. The findings are:

- **Achievability.** ROBUST policies reach the Theorem-5 frontier in every cell, with median gap ≤ 0.0005 and maximum 0.037. They do so although the training-time adversary is a greedy search and the evaluator is exhaustive. This outcome could have failed: the optimal policy is a capped mixture that gradient training need not find.
- **Censoring structure.** With the noisy flag (q = 0.8), all 33 ROBUST policies operating at U < q cap per-content acceptance at exactly 1 − q = 0.20. They never act on a "not trusted" flag, whatever the content, and use content only to veto flagged-trusted cases. This is the veto-only regime of Theorem 5. The 18 ROBUST policies at U ≥ q have cap 1, as they must: utility above q requires acting on some "not trusted" views. By contrast, all 6 STATIC policies at U < q act on a "not trusted" flag for some content (cap 1). They let content override the flag, which is exactly what the free part of the attacker exploits.
- **Static training fails as predicted.** STATIC policies accept some content with probability 1 under attack signals (max acceptance 1.00). Their worst-case ASR therefore exceeds the frontier by a median of 0.058 (ε = 0.25) and 0.191 (ε = 0.5) at q = 0.8. At ε = 0.5, half of them are *worse than ignoring content altogether*. They are exactly the policies that a static benchmark (ε = 0) is designed to select.
- **Without an unforgeable signal (q = 0.5),** static training is close to minimax at small ε. In this regime the optimal cap is 1, so the static optimum is nearly minimax. The gap appears only at ε = 0.5 (median 0.017, max 0.181). Content-based defenses still give large gains over the diagonal ASR = U, because the content gap (TV = 0.71) is large. This is the partial-mimicry regime in which C1's one-for-one line does not bind.
- **No point lies below the frontier.** The minimum gap over all 216 evaluations is −4·10⁻¹⁰, i.e. numerical zero, which is consistent with exact evaluation.

### 7.2 E1: learned policies reach the binary-flag frontier (achievability check)

In E1 the content is identical by construction, so the frontier of Theorem 2(a) is a mathematical lower bound. The only empirical question is whether training reaches it. It does:

- The median |ASR − ASR*(U)| is 0.00065.
- With Agresti–Coull standard errors for both U and ASR (delta method on the frontier slope), the minimum z-score is −1.24, and no model is below −2.
- For the Theorem-1 check, the minimum z is −0.80 (`results/stats_r1.json`). This check now uses per-model standard errors instead of the fixed tolerance used in the first version.
- 12 of the 36 models sit at degenerate operating points (U ∈ {0, 1} or ASR = 0).

For a 2-layer network on this task, reaching the frontier means learning to saturate the decision on one flag value. We therefore regard E1 as a sanity check of the pipeline and of the closed form. It does not test the theory.

### 7.3 E3: static versus adaptive robustness (illustration)

Table 2 gives utility and static ASR (mean over 3 seeds).

**Table 2.** Utility and static ASR (E3).

| System | U(DEL_L) | U(DEL_LA) | static ASR, DEL_A train / held-out | static ASR, INJ train / held-out |
|---|---|---|---|---|
| UNDEF | 1.000 | 0.874 | 0.742 / 0.746 | 0.049 / 0.057 |
| INBAND | 1.000 | 0.998 | 0.000 / 0.000 | 0.000 / 0.003 |
| INBAND, forged SIG | – | – | 1.000 / 0.985 | 0.346 / 0.359 |
| OOB | 1.000 | 1.000 | 0.000 / 0.000 | 0.000 / 0.003 |
| GATED | 1.000 | 1.000 | 0.000 / 0.000 | 0.000 / 0.000 |

**Table 3.** Adaptive ASR, given as mean ± sample s.d. (ddof = 1, 3 seeds) [95% Wilson interval on pooled episodes, n ≈ 890–899 per cell]. Source: `results/stats_r1.json`.

*G2 (forge a delegation; mimicry):*

| System | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| UNDEF | .846±.064 [.820, .868] | .913±.048 [.893, .930] | .954±.034 [.939, .966] | .971±.027 [.958, .980] |
| INBAND | .359±.049 [.328, .391] | .997±.003 [.990, .999] | 1.000±0 [.996, 1] | 1.000±0 [.996, 1] |
| OOB | 0 [0, .004] | 0 [0, .004] | 0 [0, .004] | 0 [0, .004] |
| GATED | 0 [0, .004] | 0 [0, .004] | 0 [0, .004] | 0 [0, .004] |

*G1 (hijack a fixed task; never-legitimate):*

| System | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| UNDEF | .094±.050 [.076, .114] | .153±.093 [.131, .179] | .258±.167 [.230, .287] | .400±.149 [.368, .433] |
| INBAND | .087±.080 [.070, .107] | .367±.138 [.336, .399] | .617±.138 [.585, .648] | .843±.096 [.817, .865] |
| OOB | .013±.010 [.008, .023] | .014±.004 [.008, .025] | .039±.023 [.028, .054] | .113±.040 [.094, .136] |
| GATED | 0 [0, .004] | 0 [0, .004] | 0 [0, .004] | .003±.003 [.001, .010] |

The pooled Wilson intervals treat episodes as independent and ignore between-seed variation. The seed s.d. is the more conservative measure and is large for the learned systems on G1.

**What is and is not informative.**

- **INBAND G2 → 1** is *guaranteed* by C1 once the search finds the legitimate pattern: SIG followed by an operation, both inside the attacker vocabulary. The informative part is practical. A static evaluation with held-out templates rates INBAND at 0%, and a 2-token greedy search finds the forgery without being told about SIG. Among the stored successful spans, 62% (56/90) contain SIG immediately followed by an operation token, against 0/90 for UNDEF and 0/42 for OOB. The code stores only the first 5 successes per (system, goal, k, seed) cell, so these fractions come from a subsample.
- **GATED G2 = 0** holds *by construction*: code reads only role ids.
- **OOB G2 = 0** is *not* by construction. OOB is a learned model that reads all content tokens, and the attacker can write every structural token. UNDEF has exactly the same unforgeable role channel and is broken at 0.85–0.97. Whether training makes the model rely on the unforgeable bit, rather than on content shortcuts, is an empirical question. The answer was yes for OOB and no for UNDEF, whose training never contrasted legitimate with injected instructions.
- **G1 grows with budget for every learned system** (Wilson intervals at k = 1 and k = 8 do not overlap for UNDEF, INBAND or OOB). This is consistent with treating never-legitimate attacks as a budget-dependent robustness problem that provenance does not solve. Of the stored OOB G1 spans, 95% contain a forged structural token. We ran no ablation, and we make no claim about the internal mechanism. GATED's residual 3 successes in 890 at k = 8 come from the quarantined value model, a data-flow channel that control-flow gating does not close.
- **INBAND versus UNDEF on G1.** INBAND has higher G1 ASR than UNDEF in all 3 seeds at k ∈ {2, 4, 8}, with per-seed differences of 0.07–0.60. With 3 seeds, one task and a weak attacker, we report this as an observation to be tested, not as a finding.

## 8 Discussion

### 8.1 So, can jailbreaks and prompt injection be fundamentally solved?

For the mimicry slice, under assumption (K), the answer has three parts:

- **Exact mimicry:** solvable iff an unforgeable signal separates the cases, and only up to that signal's ROC curve (Theorem 3).
- **Partial mimicry:** content-based defenses help, but by at most the mimicry gap δ (Theorem 4). Under ε-contamination, the best achievable curve is ASR*_ε (Theorem 5). Reaching it requires training or designing against the worst case and *capping* how much any content can raise trust. Defenses optimized on static benchmarks can end up worse than ignoring content (E2).
- **Never-legitimate behavior:** an ordinary robustness problem whose difficulty grows with attacker budget.

Engineering better unforgeable signals is the only lever that keeps helping as attackers improve their mimicry, because ε → 1 drives every other curve to f_K. Such signals include role ids attached by trusted serving code, authenticated tool provenance, signed principal requests and verified identity.

**Reporting recommendation.** Report (U, ASR) curves under adaptive attack, together with (1) the list of unforgeable bits the defense relies on, and (2) where available, an estimate of the attacker's mimicry gap.

### 8.2 Falsifiable predictions for real systems (not tested here)

- **P1 (content-only defenses, full mimicry).** On mimicry test sets built by copying legitimate delegated instructions verbatim into attack positions, adaptive ASR is at least U, up to sampling error. If prompting defenses, spotlighting, LLM-judge detectors, or StruQ/SecAlign without a trusted role step fall significantly below the diagonal, the cause is an unnamed unforgeable signal or a non-adaptive evaluation.
- **P2 (role-based defenses).** Corrupting trusted role ids at rate ε_role moves ISE- or ASIDE-style systems onto, or above, the frontier of Theorem 2(a) with q = 1 − ε_role.
- **P3 (budget dependence).** On gated control decisions, ASR is flat in attack budget. On learned decisions for never-legitimate goals, it rises.
- **P4 (partial mimicry, new).** Detectors trained only on static injection corpora will have worst-case ASR above the frontier, and possibly above the s-only curve, when part of the attack set consists of verbatim legitimate instructions. Training against mixed attacks with a per-content cap reduces the gap.
- **P5 (dual-use, hypothesis).** For anonymous access, harm under adaptive paraphrase tracks the lower envelope Γ of the response menu, not a curve set by classifier quality. Measuring b and h for real dual-use domains is the main obstacle, and we have not attempted it.

### 8.3 Pre-registered protocol (not run here)

1. Use an agent benchmark with formal state checks, e.g. AgentDojo. Build legitimate-delegation tasks, matched exact-mimicry attacks (the same instruction text in an unauthorized tool output) and native attacks (the benchmark's injection templates). Mixing them at rate ε gives an ε-contaminated attack set.
2. For each defense, list its unforgeable bits and estimate K_L and K_A from logs. Compute f_K, then ASR*_ε from Theorem 5, using an estimated content frontier g_C. The content frontier can be bounded with a held-out classifier's ROC curve.
3. Run adaptive attacks at budgets of 10¹–10⁴ queries at several operating points. For P2, inject role-label noise at ε_role ∈ {0, 0.05, 0.1, 0.2} into a role-embedding model (ISE or ASIDE checkpoints) and into one content-only defense.
4. *Primary outcome:* the fraction of (defense, operating point) pairs more than 2 SE below the predicted frontier. The prediction is 0. A robust violation would falsify assumption (M) or (K) for that pipeline, and would measure how far real attackers are from mimicry.

We could not run this protocol. This environment has no GPU and no model API, and adaptive attacks on multi-billion-parameter role-embedding checkpoints are not feasible on 4 CPU cores within our compute budget.

## 9 Limitations

- **Novelty of Theorems 1–3.** These results are the Neyman–Pearson lemma applied to a view decomposition. Their overlap with Hermon et al. (2026) and with the informal arguments of the other 2026 works is substantial. Our new technical content is Theorems 4–5 and Proposition 6. Theorem 5 is an application of robust-testing ideas (Huber, 1965), and a related special case may exist in that literature.
- **Kernel independence (K).** All results assume that the signal's distribution does not depend on content. If the trust label is assigned by a content-reading component, for example a classifier that decides which tool output is "trusted", then the label is effectively forgeable through content. Only the lower bound in Theorem 4's proof (step 1) survives, and the attacker can also choose content where the signal is weak. We give no frontier for that case.
- **Partial-mimicry models.** ε-contamination and TV balls are two tractable attacker models. Real attackers may be limited in structured ways that these models do not capture, such as knowledge of the user's context, length limits or format checks. Estimating ε or δ for a real deployment is open, and Theorems 4–5 are only as useful as those estimates.
- **"Unforgeable" is an assumption.** Role ids can be mis-assigned, credentials phished, and humans socially engineered.
- **Binary and graded outcomes.** Theorems 1–5 assume act or don't-act. Proposition 6 handles a fixed response menu with scalar benefit and harm. It ignores multi-step agents, compositional ("mosaic") harms and harms that depend on the requester.
- **Experiments are synthetic and small.** E2 uses MLPs on a 216-content space with exact evaluation; E1 and E3 use 2-layer transformers on a 17-token task. The E3 attacker is weak (at most 609 queries), and there are 3 seeds. The INBAND-versus-UNDEF G1 observation comes from one task. Forgery statistics use the first 5 stored successes per cell. No production LLM was run, and nothing here estimates real-model ASR.
- **The jailbreak statements (P5) are untested hypotheses.**

## 10 Conclusion

Under exact mimicry, the security–utility trade-off against adaptive prompt injection is the ROC curve of the unforgeable bits. That part is a short reduction to Neyman–Pearson, and related results appear in concurrent work. We have added what happens when mimicry is imperfect. The advantage of content-based defenses is bounded by the mimicry gap. Their exact frontier under ε-contamination is achieved by a censored policy that lets content veto but caps how far it can raise trust. Training that ignores the free part of the attacker can end up worse than not reading content at all. Small trained policies reach this frontier under exact worst-case evaluation, and statically trained ones miss it in the predicted direction. Whether real pipelines satisfy the channel assumptions, and how large their mimicry gaps are, are empirical questions that the protocol in §8.3 is designed to answer.

---

## Reproducibility

`code/README.md` lists the run commands. New in revision 1:

- `code/verify_partial.py` → `results/partial_theory_checks.json`;
- `code/exp2_partial.py` → `results/e2_partial.json`;
- `code/analyze_r1.py` → `results/e2_summary.json`, `results/figs/fig3_partial.png`, and the redrawn `fig2_adaptive.png`;
- `code/stats_r1.py` → `results/stats_r1.json`.

All numbers in §§4–7 come from these files and from the earlier `results/theory_checks.json`, `e1_tradeoff.json`, `e3_adaptive.json` and `e3_forgery_stats.json`. External numbers appear only where cited in §§1–2.

## References

Abdelnabi, S., & Bagdasarian, E. (2026). AI Agents May Always Fall for Prompt Injections. arXiv:2605.17634 (NeurIPS 2026 poster).

Andriushchenko, M., Croce, F., & Flammarion, N. (2025). Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. In *ICLR 2025*. arXiv:2404.02151.

Beurer-Kellner, L., et al. (2025). Design Patterns for Securing LLM Agents against Prompt Injections. arXiv:2506.08837.

Bhatt, M., Munshi, S., Narajala, V. S., Habler, I., Al-Kahfah, A., Huang, K., Webb, J., Gatto, B., & Hoque, M. T. (2026). The Defense Trilemma: Why Prompt Injection Defense Wrappers Fail? arXiv:2604.06436.

Bhattarai, M., & Vu, M. (2026). Trustworthy Agentic AI Requires Deterministic Architectural Boundaries. arXiv:2602.09947.

Chen, S., Piet, J., Sitawarin, C., & Wagner, D. (2025a). StruQ: Defending Against Prompt Injection with Structured Queries. In *34th USENIX Security Symposium*, pp. 2383–2400. arXiv:2402.06363.

Chen, S., Zharmagambetov, A., Mahloujifar, S., Chaudhuri, K., Wagner, D., & Guo, C. (2025b). SecAlign: Defending Against Prompt Injection with Preference Optimization. In *ACM CCS 2025*. doi:10.1145/3719027.3744836. arXiv:2410.05451.

Costa, M., Köpf, B., Paverd, A., Russinovich, M., et al. (2025). Securing AI Agents with Information-Flow Control. arXiv:2505.23643.

Debenedetti, E., Zhang, J., Balunović, M., Beurer-Kellner, L., Fischer, M., & Tramèr, F. (2024). AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. In *NeurIPS 2024 Datasets and Benchmarks Track*. arXiv:2406.13352.

Debenedetti, E., Shumailov, I., et al. (2025). Defeating Prompt Injections by Design. arXiv:2503.18813.

Glukhov, D., Shumailov, I., Gal, Y., Papernot, N., & Papyan, V. (2024). Position: Fundamental Limitations of LLM Censorship Necessitate New Approaches. In *ICML 2024*, PMLR 235. (Earlier version: arXiv:2307.10719.)

Goguen, J. A., & Meseguer, J. (1982). Security Policies and Security Models. In *Proceedings of the 1982 IEEE Symposium on Security and Privacy*, p. 11.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In *AISec '23*, pp. 79–90. arXiv:2302.12173.

Gupta, A. (2025). Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs. arXiv:2508.09288.

Hermon, M., Gupta, R., Ruan, W., Sabir, E., & Wang, H. (2026). Security–Fidelity Tradeoffs: No Universal Defense Against Prompt Injection. In *ICML 2026* (Spotlight). arXiv:2606.30783 (arXiv title: "Security–Fidelity Tradeoffs: The Hidden Cost of Prompt Injection Defense").

Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., & Kıcıman, E. (2024). Defending Against Indirect Prompt Injection Attacks With Spotlighting. arXiv:2403.14720.

Huber, P. J. (1965). A Robust Version of the Probability Ratio Test. *Annals of Mathematical Statistics*, 36(6), 1753–1758. doi:10.1214/aoms/1177699803.

Hughes, J., et al. (2024). Best-of-N Jailbreaking. arXiv:2412.03556 (NeurIPS 2025).

Lehmann, E. L., & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3rd ed.). Springer.

Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. In *33rd USENIX Security Symposium*, pp. 1831–1847. arXiv:2310.12815.

Nasr, M., Carlini, N., Sitawarin, C., Schulhoff, S. V., Hayes, J., Ilie, M., Pluto, J., Song, S., Chaudhari, H., Shumailov, I., Thakurta, A., Xiao, K. Y., Terzis, A., & Tramèr, F. (2025). The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections. arXiv:2510.09023 (USENIX Security 2026).

Neyman, J., & Pearson, E. S. (1933). On the Problem of the Most Efficient Tests of Statistical Hypotheses. *Phil. Trans. R. Soc. Lond. A*, 231, 289–337. doi:10.1098/rsta.1933.0009.

Pant, D., Lohani, S., & Kumar, A. (2026). On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models. arXiv:2606.27567.

Perez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. NeurIPS 2022 ML Safety Workshop. arXiv:2211.09527.

Shafahi, A., Huang, W. R., Studer, C., Feizi, S., & Goldstein, T. (2019). Are Adversarial Examples Inevitable? In *ICLR 2019*. arXiv:1809.02104.

Sharma, M., et al. (2025). Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming. arXiv:2501.18837.

Tramèr, F., Carlini, N., Brendel, W., & Madry, A. (2020). On Adaptive Attacks to Adversarial Example Defenses. In *NeurIPS 2020*. arXiv:2002.08347.

Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., & Beutel, A. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. arXiv:2404.13208.

Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How Does LLM Safety Training Fail? In *NeurIPS 2023*. arXiv:2307.02483.

Wolf, Y., Wies, N., Avnery, O., Levine, Y., & Shashua, A. (2024). Fundamental Limitations of Alignment in Large Language Models. In *ICML 2024*, PMLR 235, pp. 53079–53112. arXiv:2304.11082.

Wu, T., Zhang, S., et al. (2025). Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy. In *ICLR 2025*. arXiv:2410.09102.

Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.

Zverev, E., Abdelnabi, S., Fritz, M., & Lampert, C. H. (2025). Can LLMs Separate Instructions From Data? And What Do We Even Mean By That? In *ICLR 2025*. arXiv:2403.06833.

Zverev, E., Kortukov, E., Panfilov, A., Volkova, A., Tabesh, S., Lapuschkin, S., Samek, W., & Lampert, C. H. (2026). ASIDE: Architectural Separation of Instructions and Data in Language Models. In *ICLR 2026*. arXiv:2503.10566.
