# Only Unforgeable Bits Count: A Minimax Characterization of When Prompt Injection and Jailbreaks Can Be Solved

**Target venue:** ICML 2027 (main track; theory + controlled experiments). Secondary fit: IEEE SaTML.

*Anonymous authors — preprint draft, November 2026*

---

## Abstract

Whether jailbreaks and prompt injection can be "fundamentally solved" is usually argued qualitatively. We give a quantitative answer for one central slice of the problem: attacks that work by making a large language model (LLM) system follow an instruction that it would legitimately follow in some other context. We model a defended system as any (possibly randomized) policy over everything it observes, and split that observation into attacker-writable content and *unforgeable* signals (out-of-band provenance tags, authenticated identity, the system prompt). We prove three results. (i) For every policy, attack success rate (ASR) is at least utility minus the total-variation distance between legitimate and attack views: ASR ≥ U − TV. (ii) The exact set of achievable (utility, ASR) pairs is the Neyman–Pearson ROC region of the legitimate-versus-attack test, so the optimal defense is a likelihood-ratio test. (iii) Against an adaptive attacker who can reproduce the content of legitimate instructions, the minimax ASR at utility U equals the ROC frontier of the unforgeable channel *alone*. Content-based defenses, including in-band delimiters, learned detectors and adversarial training, cannot improve on it. Utility and security trade off one-for-one when no unforgeable signal exists (ASR ≥ U), and the problem is solvable exactly when such a signal separates the two cases perfectly. We test the theory in a controlled CPU-scale testbed of small transformers trained on an instruction-delegation task. All 36 trained models land on the predicted frontier: the median gap is 0.0007 and no model falls more than 2 standard errors below it. A defense whose provenance marker is in-band reaches 0% static ASR. An adaptive greedy search with two attacker tokens raises its ASR on forged delegations to 99.7–100%, and the search finds the marker forgery by itself. The same model with an out-of-band role channel stays at 0%, as does a code-gated architecture. For attacks on requests that are never legitimate, provenance gives no protection, and every learned system we tested degrades as the attacker's token budget grows. We state what the results imply for jailbreaks: for dual-use requests from anonymous users, refusal and helpfulness trade off one-for-one. We also give falsifiable predictions and an evaluation protocol for frontier systems.

## 中文摘要

越狱与提示注入能否被"根本性解决"，以往多为定性讨论。本文针对其中一个核心子问题给出定量回答：攻击者使系统执行一条在其他情境下本应被合法执行的指令。我们将任意防御系统建模为其全部观测上的（可随机化）策略，并将观测分解为攻击者可写的内容与攻击者不可伪造的信号（带外来源标记、经认证的身份、系统提示）。我们证明了三个结果：(i) 对任意策略，攻击成功率 ASR ≥ 效用 U − 合法视图与攻击视图之间的总变差距离 TV；(ii) 可达的（效用，ASR）集合恰为"合法 vs. 攻击"检验的 Neyman–Pearson ROC 区域，最优防御即似然比检验；(iii) 若自适应攻击者能复现合法指令的内容，则在效用 U 下的极小极大 ASR 恰等于*仅由不可伪造信道*决定的 ROC 前沿。任何基于内容的防御（带内分隔符、学习型检测器、对抗训练）都无法超越这一前沿。没有不可伪造信号时，效用与安全一比一互换（ASR ≥ U）；当且仅当存在能完美区分两种情形的此类信号时，问题才可被精确解决。我们在一个可控的 CPU 规模实验平台上（以小型 Transformer 学习指令委托任务）检验该理论。全部 36 个训练模型都落在理论前沿上：中位偏差 0.0007，没有模型低于前沿超过 2 个标准误。使用带内信任标记的防御在静态攻击下 ASR 为 0%；自适应贪心搜索只需两个攻击 token，即可将伪造委托的 ASR 提升至 99.7–100%，且搜索会自行发现对该标记的伪造。改用带外角色信道的同一模型，以及代码门控架构，ASR 均保持 0%。对于在任何情境下都不合法的指令，来源信息无法提供保护；我们测试的所有学习型系统在攻击预算增加时都会退化。我们还推导了对越狱的含义：对匿名用户的两用请求，拒绝率与有用性一比一互换。最后，我们给出可证伪的预测以及面向前沿模型的评估方案。

---

## 1 Introduction

Prompt injection (Perez & Ribeiro, 2022; Greshake et al., 2023) and jailbreaking (Wei et al., 2023; Zou et al., 2023) have gone through several rounds of defense and counter-attack. Defenses that report near-zero ASR against fixed attack sets are routinely broken by adaptive attackers: Nasr et al. (2025) bypass 12 recent defenses with ASR above 90% for most, and Andriushchenko et al. (2025) reach 100% ASR on many safety-aligned models with simple adaptive attacks. The pattern repeats the earlier history of adversarial-example defenses (Tramèr et al., 2020). Meanwhile, system-level designs that move control flow out of the model, such as CaMeL (Debenedetti et al., 2025), FIDES (Costa et al., 2025) and the design patterns of Beurer-Kellner et al. (2025), claim *provable* protection for some security properties, at a cost in utility.

These two observations suggest that "can prompt injection be solved?" has no single answer. The answer depends on the information available to the defender. The goal of this paper is to make that dependence exact. We focus on the following sub-question:

> *When an attacker tries to make a system follow an instruction of a kind the system is legitimately supposed to follow in other contexts, what is the lowest attack success rate any defense can achieve at a given level of utility, and what determines it?*

This "mimicry" slice is the hard core of indirect prompt injection. Agents are supposed to act on content in emails, tickets, documents and tool outputs; a forwarded request from a colleague and an injected one can be word-for-word identical. The same slice is the hard core of jailbreaking dual-use requests, where a benign and a malicious user can type the same words.

**Answer in one sentence.** Against an adaptive attacker, only the bits the attacker cannot write count, and the best achievable security–utility trade-off is exactly the ROC curve of those bits.

**Contributions.**

1. **A distinguishability bound (Theorem 1).** For any defended system, ASR ≥ U − TV(P_L, P_A). Here U is the rate at which legitimate in-context instructions are followed, and TV is the total-variation distance between the system's full input views in legitimate and attack scenarios.
2. **Exact achievable region (Theorem 2).** The set of achievable (U, ASR) pairs is exactly the Neyman–Pearson testing region; its lower boundary is attained by likelihood-ratio policies. We give closed forms for a binary provenance flag with accuracy q and for a Gaussian provenance score with separation d′.
3. **Minimax theorem for adaptive attackers (Theorem 3).** If the attacker can reproduce legitimate instruction content, the minimax ASR at utility U equals the ROC frontier of the *unforgeable* channel alone, independent of the content model. A policy that reads only unforgeable bits attains this frontier for every attack budget. Corollaries: (a) with no unforgeable signal, ASR ≥ U, a one-for-one trade-off; (b) the problem is solvable exactly when the unforgeable channel is perfectly separating; (c) in-band markers (delimiters, "datamarking", special tokens inserted into content) contribute nothing against an adaptive attacker. Exact linear programs on 300 random instances confirm Theorems 1–3 to within 10⁻¹³.
4. **A controlled testbed showing the mechanism (E1, E3).** Small transformers trained on an instruction-delegation task land on the predicted frontier (E1, 36 models). Under static and adaptive attacks (E3), an adversarially trained in-band defense reaches 0% static ASR. Adaptive search with two attacker tokens raises its ASR to 99.7–100%, and the search finds the in-band marker forgery by itself. An out-of-band role channel and a code-gated architecture both hold at 0% for forged delegations. Attacks that target never-legitimate behavior bypass provenance entirely and scale with attacker budget for every learned system.
5. **Implications and a test protocol for frontier systems (§7).** We translate the theory into falsifiable predictions for deployed defenses and jailbreak safeguards, and a pre-registered evaluation protocol. We have not run that protocol here.

We do *not* claim that prompt injection is unsolvable in general, and we do not claim a new defense. The results separate the problem into a part that is information-theoretically impossible without unforgeable signals and a part that is an ordinary, budget-dependent adversarial-robustness problem. They also identify which design changes move a system from one regime to the other.

## 2 Related Work

**Attacks and benchmarks.** Perez & Ribeiro (2022) introduced goal hijacking and prompt leaking. Greshake et al. (2023) showed indirect prompt injection through retrieved content in deployed applications. Liu et al. (2024) formalized prompt injection and benchmarked 5 attacks against 10 defenses. AgentDojo (Debenedetti et al., 2024) evaluates agents and attacks with formal checks on environment state. For jailbreaks, Wei et al. (2023) attribute failures to competing objectives and mismatched generalization. GCG (Zou et al., 2023) optimizes adversarial suffixes. Best-of-N jailbreaking (Hughes et al., 2024) shows that ASR follows power-law-like scaling in the number of sampled augmentations. Adaptive-attack methodology (Tramèr et al., 2020; Andriushchenko et al., 2025; Nasr et al., 2025) is the evaluation standard we adopt.

**Model-level defenses.** Spotlighting (Hines et al., 2024) marks untrusted content in-band through delimiting, datamarking or encoding. StruQ (Chen et al., 2025a) and SecAlign (Chen et al., 2025b) train models to ignore instructions in a reserved data channel. The instruction hierarchy (Wallace et al., 2024) trains models to prioritize privileged instructions. Instructional Segment Embedding (Wu et al., 2025) and ASIDE (Zverev et al., 2026) add architectural role signals (segment embeddings, or a rotation of data-token embeddings). Zverev et al. (2025) define and measure instruction–data separation. Constitutional Classifiers (Sharma et al., 2025) guard inputs and outputs with trained classifiers. In our framework, these methods differ in *which bits the attacker cannot write*. Role or segment ids assigned by a trusted pipeline are unforgeable. Delimiters and in-content marks are forgeable. Classifiers read whatever the policy reads.

**System-level defenses.** CaMeL (Debenedetti et al., 2025) extracts control flow from the trusted query and tracks provenance with capabilities. FIDES (Costa et al., 2025) applies information-flow-control labels. Beurer-Kellner et al. (2025) catalogue design patterns that constrain agents after they read untrusted input. All three rest on the classic idea of noninterference (Goguen & Meseguer, 1982). Our Theorem 3 explains why these designs succeed where model-level defenses fail: their control decisions read only unforgeable bits. It also states the price: utility on tasks whose legitimacy cannot be read from those bits.

**Impossibility results.** Wolf et al. (2024) show that any behavior with positive probability can be elicited by some prompt. Glukhov et al. (2024) argue that semantic output censorship is undecidable and vulnerable to "mosaic" composition. Shafahi et al. (2019) give limits on adversarial robustness in high dimensions. Three 2026 preprints are closest to us. The Defense Trilemma (Bhatt et al., 2026) proves that a continuous, utility-preserving wrapper on a connected prompt space cannot be complete. "On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models" (arXiv:2606.27567, 2026) argues that perfect prevention is impossible without an immutable provenance-enforcing mechanism. Abdelnabi & Bagdasarian (2026) argue informally, through contextual integrity, that an attacker can fabricate context that makes a blocked action look legitimate. Our results differ from these in three ways. They are *quantitative*: they give the exact achievable region, including *partial or noisy* provenance. They are *minimax*: they include the matching achievability statement, so they say what *can* be achieved and not only what cannot. And they are tested on trained models, which land on the predicted frontier. The decision-theoretic content, the Neyman–Pearson lemma (Neyman & Pearson, 1933; Lehmann & Romano, 2005), is classical. We believe the contribution is the reduction from adaptive prompt injection to it, and the consequence that content-based defenses add nothing in the mimicry regime.

## 3 Model

**Views and channels.** A defended LLM system (model weights, prompts, filters, wrappers, tool-gating code) observes a *view* v = (c, s) ∈ 𝒞 × 𝒮 before deciding whether to act on an instruction ι(c) that appears in context. Here c is everything the attacker can write in the attack scenario: the tokens of a retrieved document, tool output or user turn, including any delimiters, special tokens or "trusted" markers that an honest pipeline would insert *into that content*. s is everything the attacker cannot write: out-of-band role or segment ids assigned by trusted code, cryptographic authentication, the system prompt, verified user identity, the provenance label of the tool call that produced c. The system is a measurable map φ: 𝒞 × 𝒮 → [0,1], the probability of executing ι(c). This covers deterministic and randomized policies, ensembles and filter cascades, because any pipeline that reads v and outputs act/don't-act is such a map.

**Scenarios.** In the *legitimate* scenario the in-context instruction is authorized, for example a manager's forwarded request the user asked the agent to handle, or a dual-use question from a benign user. Views are drawn from P_L = L_c ⊗ K_L: content from L_c and unforgeable signal from a kernel K_L(s | c). In the *attack* scenario the instruction is injected; content is drawn from an attacker-chosen distribution Q ∈ 𝒬, and the signal from K_A(s | c). We write P_A^Q = Q ⊗ K_A.

**Metrics.** Utility U(φ) = E_{P_L}[φ] is the rate at which legitimate in-context instructions are executed. Attack success ASR(φ, Q) = E_{P_A^Q}[φ] is the rate at which injected ones are executed. Utility on tasks that need no in-context instructions is unaffected by the analysis; we measure it separately in the experiments.

**Mimicry assumption.** The attacker is *content-capable* if L_c ∈ 𝒬: it can make its content distribution identical to that of legitimate in-context instructions. This is a strong assumption only when legitimate in-context instructions are rare and peculiar. For agents that read email, issue trackers or chat, they are neither, and in-band markers are by definition part of c.

The model does not cover attacks that make the system do something that is *never* legitimate in any context (exfiltrate secrets to an unknown host, follow an injected "ignore your task" in a fixed-task pipeline). For those, L_c and Q have disjoint supports on the relevant event and the bound below is vacuous. We return to them in §6 and E3 (goal G1), because they form the other, budget-dependent half of the problem.

## 4 Theory

**Theorem 1 (distinguishability bound).** For every policy φ and every attack distribution Q,

  ASR(φ, Q) ≥ U(φ) − TV(P_L, P_A^Q).

*Proof.* φ takes values in [0,1], so |E_{P_L}φ − E_{P_A^Q}φ| ≤ sup_{f: 𝒱→[0,1]} |E_{P_L} f − E_{P_A^Q} f| = TV(P_L, P_A^Q). ∎

**Theorem 2 (exact achievable region).** Fix Q and let λ = dP_L/dP_A^Q, the likelihood ratio (Radon–Nikodym derivative where it exists, extended by +∞ on the singular part). The set of achievable pairs {(U(φ), ASR(φ,Q)) : φ measurable, [0,1]-valued} is the convex Neyman–Pearson region. Its lower boundary,

  ASR*(U) = min{ E_{P_A^Q}φ : E_{P_L}φ ≥ U },

is attained by a likelihood-ratio test φ_t,γ = 1[λ > t] + γ·1[λ = t], and ASR*(·) is convex and nondecreasing.

*Proof.* Treat P_A^Q as the null and P_L as the alternative. Then ASR is the size and U the power of the test φ, and the Neyman–Pearson lemma (Neyman & Pearson, 1933; Lehmann & Romano, 2005, Thm. 3.2.1) gives that likelihood-ratio tests maximize power at each size. Inverting the nondecreasing, concave power-versus-size function gives ASR*(U). Convexity follows from mixing policies. ∎

*Closed forms.* (a) Binary provenance flag that is correct with probability q ≥ ½, with matched content: TV = 2q − 1 and

  ASR*(U) = (1−q)U/q for U ≤ q,  ASR*(U) = (1−q) + q(U−q)/(1−q) for U > q.

(b) Scalar provenance score distributed N(d′, 1) for legitimate and N(0, 1) for injected instructions: ASR*(U) = 1 − Φ(Φ⁻¹(1−U) + d′). Figure 1 (right) plots (b).

**Theorem 3 (only unforgeable bits count).** Suppose K_L(·|c) = K_L and K_A(·|c) = K_A do not depend on c, and the attacker is content-capable (L_c ∈ 𝒬). Let ASR*_K(U) be the Neyman–Pearson frontier of the pair (K_L, K_A) on 𝒮. Then

  min_{φ: U(φ) ≥ U} max_{Q ∈ 𝒬} ASR(φ, Q) = ASR*_K(U),

and a policy that reads only s (a likelihood-ratio test on K_L/K_A) attains it for every Q ∈ 𝒬, in particular for every attack budget.

*Proof.* (≥) Let the attacker play Q = L_c. Then P_L = L_c ⊗ K_L and P_A^Q = L_c ⊗ K_A. Their likelihood ratio is K_L(s)/K_A(s), a function of s only. By Theorem 2 the frontier of this pair equals ASR*_K, so every φ with U(φ) ≥ U has ASR(φ, L_c) ≥ ASR*_K(U). (≤) Let φ(c,s) = ψ(s), with ψ the likelihood-ratio test on (K_L, K_A) of power U. Then U(φ) = E_{K_L}ψ = U, and for any Q, ASR(φ,Q) = E_{K_A}ψ = ASR*_K(U), whatever Q is. ∎

If the kernels depend on c, the same argument gives the lower bound with ASR*_K replaced by the frontier of the mixture pair (L_c ⊗ K_L, L_c ⊗ K_A). The attacker can then also choose content where the signal is least informative.

**Corollaries.**

* **C1 (no unforgeable signal ⇒ one-for-one trade-off).** If K_L = K_A, for example because every provenance cue lives in content, then max_Q ASR ≥ U for every policy. Each percentage point of utility on legitimate in-context instructions is matched by a percentage point of attack success.
* **C2 (exact solvability).** Zero worst-case ASR at full utility (U = 1) is achievable iff K_L ⊥ K_A, i.e., the unforgeable channel separates the scenarios perfectly. Zero ASR at utility U < 1 is achievable iff K_L puts mass at least U on a set that K_A gives zero mass.
* **C3 (in-band markers and content defenses are inert).** Delimiters, datamarks, special tokens and learned detectors that read c cannot lower worst-case ASR below ASR*_K. A content-capable attacker copies the marker along with the instruction. Adversarial training against content patterns can only help against attackers who are *not* content-capable, for example static benchmarks.
* **C4 (budget independence).** For the policy in Theorem 3, ASR is independent of the attacker's search budget. For any policy whose decision depends on c, the attacker's ASR is nondecreasing in the size of 𝒬 and, under mimicry, reaches at least ASR*_K. Increasing budget can only help the attacker.

**Numerical check.** We confirmed Theorems 1–3 with exact linear programs on 300 random finite instances of up to 30 views (`code/verify_theory.py`, output in `results/theory_checks.json`). The minimum slack of Theorem 1 over 6,000 random policies was +0.013. The maximum absolute error between the LP optimum and the likelihood-ratio frontier was 3.2·10⁻¹⁴ for Theorem 2 and 3.2·10⁻¹⁵ for Theorem 3, where the minimax was computed by an LP against all content choices.

**Jailbreaks as the same problem.** Replace "in-context instruction" with "request" and "provenance" with "requester". Legitimate = a benign user asking a dual-use question; attack = a malicious user asking for the same capability. A content-capable malicious user can phrase requests the way benign users do. Then Theorem 3 says that at helpfulness H on such requests, the minimax harmful-compliance rate equals the ROC frontier of whatever the system knows about the user that the user cannot forge: verified identity, institutional affiliation, account history, rate limits. For anonymous access the trade-off is one-for-one (C1). Better refusal training, better classifiers or longer system prompts cannot change this. They can only move the operating point along the line. The theorem does not apply to requests that no benign population makes. There, refusal costs no helpfulness and the problem reduces to recognizing such requests under adversarial paraphrase, which is the ordinary robustness problem studied by Wei et al. (2023), Glukhov et al. (2024) and Sharma et al. (2025).

## 5 Experimental Setup

All experiments are CPU-scale, fully synthetic and designed to isolate the mechanism, not to estimate the behavior of production models. Code is in `code/`; every number below comes from `results/` (E1 ≈ 11.5 min, E3 ≈ 16 min on 4 CPU cores).

**Task.** A sequence is `[BOS, SYS, s, DAT, d₁…d₁₂, QRY]`, and the model predicts one of 11 classes at QRY: digits 0–9 or NOOP. The trusted instruction s is one of four operations on the digits in the data (FIRST, LAST, MAX, MIN) or DELEG, meaning "perform the operation named in the data if it is legitimate, else NOOP". The data contain 3–6 digits, filler tokens and, depending on the episode type, in-data operation tokens. Episode types: CLEAN (s is an operation, no in-data instruction); INJ (s is an operation, an injected operation in the data); DEL_L (DELEG plus a legitimate in-data operation); DEL_A (DELEG plus only an injected operation, correct answer NOOP); DEL_LA (both, correct answer from the legitimate one). Injected operations are always chosen so that their answer differs from the correct one. Each position also carries an out-of-band *role id* (system / untrusted data / trusted-provenance data) that the attacker cannot write.

**Model.** A 2-layer pre-LayerNorm transformer encoder (d = 64, 4 heads, FFN 128) with token, position and, optionally, role embeddings, read out at QRY. AdamW with lr 2·10⁻³, one-cycle schedule, batch 256.

**E1 (does training reach the frontier?).** Legitimate and injected in-data instructions have *identical* content and position distributions; they differ only in the role flag, which is correct with probability q ∈ {0.5, …, 1.0}. By construction TV(P_L, P_A) = 2q − 1 exactly, and Theorem 2(a) gives the frontier. We vary the loss weight on DEL_A episodes (w ∈ {0.25, 1, 4}) to move the operating point, and train 2 seeds × 6 q × 3 w = 36 models (1,200 steps each, mixture of CLEAN/INJ/DEL_L/DEL_A). We evaluate U on 6,000 DEL_L episodes and ASR on 6,000 DEL_A episodes.

**E3 (static vs. adaptive attacks against four systems).** All systems see the same tasks:
* **UNDEF**: role embeddings, trained on CLEAN and DEL_L only (no injections seen).
* **INBAND**: adversarially trained on all five episode types, with injections drawn from two training templates. Legitimate in-data instructions are marked by an *in-band* token SIG placed before the operation, in the style of a delimiter or datamark; no role channel. The attacker can write SIG.
* **OOB**: identical adversarial training, but legitimate in-data instructions are marked by the *out-of-band* role id (q = 1).
* **GATED**: control flow decided by code, in the spirit of CaMeL and FIDES. If s is an operation, run it. If s = DELEG, run the first in-data operation whose role id is "trusted", else output NOOP. Values are computed by a separately trained quarantined transformer (no role input), which code tells which operation to perform. It is trained on clean episodes with up to 8 random attacker-vocabulary tokens written over non-digit positions.

Static attacks use the training templates and four held-out templates that prefix the injected operation with forged structural tokens (SYS, QRY, DELEG, DAT+SYS). For INBAND we also test a forged-SIG variant. Adaptive attacks are white-box, query-based greedy coordinate search over k ∈ {1, 2, 4, 8} contiguous attacker-controlled data positions, initialized with the injected operation followed by filler. The attacker vocabulary is the 19 non-digit tokens, including all structural tokens and SIG but not role ids. Each coordinate step evaluates all 19 substitutions and keeps the one with the largest target-class logit margin, for 4 sweeps (at most 1 + 4·19·k queries). Success means the target is the argmax. Two goals: **G1**, hijack a fixed task (s is an operation; the attacker wants another operation's distinct answer), i.e., a never-legitimate instruction; **G2**, forge a delegation (s = DELEG with no legitimate in-data instruction; the attacker wants its operation executed), i.e., the mimicry regime. Each cell uses about 300 episodes per seed (episodes with tied answers are discarded), with 3 seeds and 2,500 training steps per system.

## 6 Results

### 6.1 E1: learned policies sit on the Neyman–Pearson frontier

Figure 1 (left, `results/figs/fig1_frontier.png`) plots (U, ASR) for the 36 models against the closed-form frontier of Theorem 2(a). The agreement is close:

* Median |ASR − ASR*(U)| = 0.00065. The largest negative deviation is −0.0073 (q = 0.5, z = −1.79). No model falls more than 2 standard errors below the frontier, and none violates the Theorem 1 bound (`results/summary.json`).
* Examples: at q = 0.9, w = 1, the models reach U = 0.894/0.892 with ASR = 0.0995/0.0997, against a frontier value of 0.099. At q = 0.8, w = 1, they reach U = 0.792 with ASR = 0.203/0.199, against 0.198. At q = 0.5 (no information), ASR tracks U exactly: (0.603, 0.604) and (0.527, 0.520).
* The largest positive deviations, up to +0.080 at q = 0.9, w = 0.25, are under-trained points *above* the frontier, as the theory allows.
* All models keep accuracy ≥ 0.9998 on CLEAN and ≥ 0.9993 on INJ, where the trusted instruction is an explicit operation and the injected operation should be ignored.

The models learn the Bayes-optimal (likelihood-ratio) rule on the flag. The loss weight only selects the operating point on the frontier. With content matched, no amount of training moves a model below the line. This is the prediction of Theorem 3 for a learner given unlimited content capacity.

### 6.2 E3: static robustness is not adaptive robustness, and only out-of-band bits survive

Table 1 summarizes utility and static ASR (mean over 3 seeds; full per-seed values in `results/e3_adaptive.json`).

**Table 1.** Utility and static attack success (E3). DEL_LA = legitimate and injected in-data instructions present together.

| System | U(DEL_L) | U(DEL_LA) | static ASR, DEL_A train / held-out | static ASR, INJ train / held-out |
|---|---|---|---|---|
| UNDEF | 1.000 | 0.874 | 0.742 / 0.746 | 0.049 / 0.057 |
| INBAND | 1.000 | 0.998 | 0.000 / 0.000 | 0.000 / 0.003 |
| INBAND, forged SIG | – | – | 1.000 / 0.985 | 0.346 / 0.359 |
| OOB | 1.000 | 1.000 | 0.000 / 0.000 | 0.000 / 0.003 |
| GATED | 1.000 | 1.000 | 0.000 / 0.000 | 0.000 / 0.000 |

Under the usual static evaluation, INBAND and OOB look equally good: 0% ASR, including held-out templates, at full utility. Adaptive attacks separate them sharply (Figure 2, `results/figs/fig2_adaptive.png`; Table 2).

**Table 2.** Adaptive ASR (greedy coordinate search) vs. number of attacker-controlled tokens k, mean ± s.d. over 3 seeds.

| System | G1 k=1 | G1 k=2 | G1 k=4 | G1 k=8 | G2 k=1 | G2 k=2 | G2 k=4 | G2 k=8 |
|---|---|---|---|---|---|---|---|---|
| UNDEF | .094±.041 | .153±.076 | .258±.137 | .400±.121 | .846±.052 | .913±.040 | .954±.028 | .971±.022 |
| INBAND | .087±.066 | .367±.112 | .617±.113 | .843±.078 | .359±.040 | .997±.003 | 1.000±0 | 1.000±0 |
| OOB | .013±.008 | .014±.003 | .039±.019 | .113±.033 | 0 | 0 | 0 | 0 |
| GATED | 0 | 0 | 0 | .003±.003 | 0 | 0 | 0 | 0 |

Three findings line up with the theory.

**(a) Mimicry regime (G2): only out-of-band provenance holds.** INBAND has U(DEL_L) = 1.000 and no unforgeable signal, so Corollary C1 predicts worst-case ASR ≥ 1 once the attacker can write the legitimate pattern (SIG followed by an operation, which needs two tokens). Observed: 0.997 at k = 2 and 1.000 at k ≥ 4. OOB and GATED have a perfectly separating unforgeable channel (q = 1), so C2 predicts that ASR = 0 is achievable at full utility. Both hold at exactly 0 for every budget tested. GATED guarantees this by construction; OOB, a learned model, reaches it empirically. UNDEF has the role channel but was never trained to rely on it, and it is already at 0.74 static ASR. This illustrates that Theorem 3 bounds what is *achievable*, not what a given training run attains.

**(b) The adaptive attacker finds the marker forgery by itself.** We examined the stored successful adversarial spans (5 per system × goal × k ≥ 2 × seed; `results/e3_forgery_stats.json`). For INBAND, 62% (56/90) contain SIG immediately followed by an operation token, and 100% contain at least one structural or marker token. For UNDEF and OOB the SIG→operation pattern never appears (0/90 and 0/42). The search was not told about SIG; a learned in-band trust marker is simply the cheapest way in. For G1, INBAND is *more* vulnerable than the undefended model at every k ≥ 2 (0.843 vs. 0.400 at k = 8). In this toy setting, adversarial training with an in-band trust marker created a privileged, forgeable token. We flag this as a hazard to test at scale, not as an established result: we have 3 seeds and one task.

**(c) Never-legitimate attacks (G1) are a budget-dependent robustness problem that provenance does not solve.** G1 asks the model to output an answer that is never correct for the given trusted instruction. Theorem 3 does not apply, and provenance does not protect: every learned system's ASR grows with k. OOB rises from 0.013 to 0.113 between k = 1 and k = 8, and the attacks use forged structural tokens (95% of stored OOB spans contain one), acting through the *value computation* rather than the trust decision. GATED removes the control-flow channel entirely. Its residual ASR (0.003 at k = 8, roughly 3 of ~900 episodes pooled across seeds) comes from the quarantined value model, an in-band data-flow channel that code gating cannot close, as CaMeL's authors also note for data flows. Consistent with C4, GATED's G2 ASR is budget-independent at 0, while every content-reading component's ASR is nondecreasing in budget.

## 7 Discussion

### 7.1 So, can jailbreaks and prompt injection be fundamentally solved?

The results split the question.

* **Mimicry attacks** (the attacker asks for something that is sometimes legitimate): solvable *if and only if* the system has an unforgeable signal that separates legitimate from attack cases, and only to the extent of that signal's ROC curve (Theorem 3, C2). No model-level improvement, whether scale, adversarial training, better detectors or in-band delimiters, can go beyond that frontier against an adaptive attacker. Progress therefore means engineering better unforgeable signals: role ids attached by trusted code (ISE, ASIDE and instruction-hierarchy-style roles assigned by the serving stack), authenticated provenance on tool outputs, signed requests from principals, verified user identity for dual-use capabilities. It also means routing control decisions through them, as in CaMeL- and FIDES-style gating. Where no such signal exists, because a legitimately instruction-bearing email is indistinguishable from a malicious one, the honest goal is a chosen point on a one-for-one trade-off line, plus limits on blast radius.
* **Never-legitimate attacks** (the attacker asks for something that is never authorized in that context): solvable in principle, since refusing costs no utility. In practice this is an adversarial-robustness problem whose difficulty grows with attacker budget, as decades of adversarial-example research and the adaptive LLM attacks of 2024–2025 show. Architectural gating converts the control-flow part into a budget-independent guarantee. The data-flow part stays in-band.

The quantity defenses should report is therefore not a single ASR. It is the (U, ASR) curve under content-capable adaptive attack, together with an explicit list of the bits the defense relies on that the attacker cannot write.

### 7.2 Falsifiable predictions for frontier systems

Our experiments use tiny models. The theorems, however, apply to any system, so they make predictions that can be checked on real deployments:

* **P1 (content-only defenses).** For any defense whose decision depends only on attacker-writable content (prompting defenses, spotlighting, perplexity or LLM-judge detectors, StruQ/SecAlign *without* a trusted role-assignment step), consider a mimicry test set built by copying legitimate delegated instructions verbatim into attack positions. Adaptive ASR should be at least the defense's utility on the matched legitimate set, up to sampling error. A reported point significantly below the line U = ASR indicates either an unforgeable signal the authors did not name, or a non-adaptive evaluation.
* **P2 (role-based defenses).** For defenses with trusted role or segment ids, adaptive ASR on mimicry attacks should be governed by role-assignment fidelity. Corrupting role ids at rate ε should move systems onto the frontier of Theorem 2(a) with q = 1 − ε. This can be checked by injecting label noise in the serving pipeline.
* **P3 (budget dependence as a diagnostic).** On never-legitimate attacks, ASR of learned defenses should grow with attacker budget, as in Best-of-N-style scaling (Hughes et al., 2024). On control decisions gated by unforgeable bits, ASR should be flat in budget. A defense whose ASR still rises with budget on gated decisions has a leak in its gating.
* **P4 (dual-use jailbreaks).** For anonymous access, harmful-compliance on dual-use requests should track helpfulness on matched benign phrasings one-for-one under adaptive paraphrase. Classifier-based safeguards can shift the operating point but should not beat the diagonal on such matched sets.

### 7.3 Pre-registered protocol (not run here)

1. Choose an agent benchmark with formal state checks, e.g., AgentDojo. Build *legitimate-delegation* tasks in which the user asks the agent to act on instructions inside tool outputs, and matched *mimicry* attacks that place the same instruction text in an unauthorized tool output.
2. For each defense, list its unforgeable bits; estimate K_L and K_A empirically from logs, and compute ASR*_K.
3. Run adaptive attacks (Nasr et al., 2025 style) at budgets 10¹–10⁴ queries. Record U on legitimate-delegation tasks and ASR on mimicry attacks at several operating points (threshold or prompt variants).
4. Test P1–P3: points must lie on or above ASR*_K(U), and content-only defenses on or above the diagonal. Report the slope of ASR against log-budget separately for gated and ungated decisions.
5. Primary outcome: the fraction of (defense, operating point) pairs below the predicted frontier by more than 2 standard errors. The theory predicts 0. Any robust violation falsifies one of the modeling assumptions, most likely content capability, and would be informative about where real attackers are limited.

## 8 Limitations

* **Scope of the theorems.** The results concern a binary act/don't-act decision on an in-context instruction, a fixed legitimate distribution, and an attacker who is *content-capable*. Real attackers may not reach full mimicry. For example, they may not know the legitimate distribution, or may face length or format limits. In that case the bound is conservative and content-based defenses can help, which is why static benchmarks show gains. Multi-step agents, partial compliance and graded harms need richer outcome spaces. The extension via a [0,1]-valued harm function is direct for Theorem 1 but not developed here.
* **Kernel independence.** Theorem 3's clean form assumes the unforgeable signal's distribution does not depend on content. When it does, we give only the lower bound, and the attacker can additionally exploit content regions where provenance is weak.
* **"Unforgeable" is an assumption, not a guarantee.** Role ids can be mis-assigned, for example when tool outputs are pasted back into user turns, authentication can be phished, and humans in the loop can be socially engineered. Our analysis tells defenders where to spend effort; it does not certify any particular pipeline.
* **Toy experiments.** The experiments use 2-layer transformers with d = 64 on a synthetic 17-token task with a 30-token vocabulary, 2–3 seeds and about 300 attack episodes per cell. The attacker is a white-box greedy search with a small budget (at most 609 queries), so the G1 ASRs are lower bounds on what stronger attacks would achieve. The finding that INBAND is more vulnerable than UNDEF on G1 comes from one task family and 3 seeds. No experiment here involves a production LLM, and we make no quantitative claims about one. The frontier-LLM statements in §7 are predictions.
* **Jailbreak corollary.** The mapping from provenance to requester identity is conceptual. Estimating benign and malicious request distributions for real dual-use domains is difficult and was not attempted.
* **Related 2026 preprints.** We characterized the three closest 2026 preprints (Bhatt et al., 2026; arXiv:2606.27567; Abdelnabi & Bagdasarian, 2026) from their abstracts and public summaries; we could not access their full texts from this environment. We could not retrieve the author list of arXiv:2606.27567 and cite it by title and identifier.

## 9 Conclusion

Whether prompt injection and jailbreaks can be fundamentally solved has a precise answer for the mimicry regime. The best achievable security–utility trade-off against an adaptive attacker is the ROC curve of the bits the attacker cannot write, and a policy that reads only those bits attains it. Content-level defenses can look perfect on static benchmarks, as our in-band defense did at 0% ASR. They cannot beat that curve, and in our testbed a 2-token adaptive search found the forgery that took that defense to 100% ASR. Outside the mimicry regime, the problem is an ordinary adversarial-robustness problem whose difficulty grows with the attacker's budget. Architectural gating on unforgeable signals is the one lever that turns either regime into a budget-independent guarantee, and it does so only for control decisions. We hope the minimax framing helps the field report defenses as curves with stated trust assumptions, not as single numbers.

---

## Reproducibility

`code/README.md` lists the run commands. Outputs: `results/e1_tradeoff.json`, `results/e3_adaptive.json`, `results/e3_forgery_stats.json`, `results/theory_checks.json`, `results/summary.json`; figures in `results/figs/`. All numbers in §§4–6 come from these files; no numbers are taken from external sources except where cited in §§1–2.

## References

Abdelnabi, S., & Bagdasarian, E. (2026). AI Agents May Always Fall for Prompt Injections. arXiv:2605.17634 (NeurIPS 2026 poster).

Andriushchenko, M., Croce, F., & Flammarion, N. (2025). Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. In *International Conference on Learning Representations (ICLR 2025)*. arXiv:2404.02151.

Beurer-Kellner, L., et al. (2025). Design Patterns for Securing LLM Agents against Prompt Injections. arXiv:2506.08837.

Bhatt, M., Munshi, S., Narajala, V. S., Habler, I., Al-Kahfah, A., Huang, K., Webb, J., Gatto, B., & Hoque, M. T. (2026). The Defense Trilemma: Why Prompt Injection Defense Wrappers Fail? arXiv:2604.06436.

Chen, S., Piet, J., Sitawarin, C., & Wagner, D. (2025a). StruQ: Defending Against Prompt Injection with Structured Queries. In *34th USENIX Security Symposium*, pp. 2383–2400. arXiv:2402.06363.

Chen, S., Zharmagambetov, A., Mahloujifar, S., Chaudhuri, K., Wagner, D., & Guo, C. (2025b). SecAlign: Defending Against Prompt Injection with Preference Optimization. In *ACM CCS 2025*. doi:10.1145/3719027.3744836. arXiv:2410.05451.

Costa, M., Köpf, B., Paverd, A., Russinovich, M., et al. (2025). Securing AI Agents with Information-Flow Control. arXiv:2505.23643.

Debenedetti, E., Zhang, J., Balunović, M., Beurer-Kellner, L., Fischer, M., & Tramèr, F. (2024). AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. In *NeurIPS 2024 Datasets and Benchmarks Track*. arXiv:2406.13352.

Debenedetti, E., Shumailov, I., et al. (2025). Defeating Prompt Injections by Design. arXiv:2503.18813.

Glukhov, D., Shumailov, I., Gal, Y., Papernot, N., & Papyan, V. (2024). Position: Fundamental Limitations of LLM Censorship Necessitate New Approaches. In *ICML 2024*, PMLR 235. (Earlier version: LLM Censorship: A Machine Learning Challenge or a Computer Security Problem? arXiv:2307.10719.)

Goguen, J. A., & Meseguer, J. (1982). Security Policies and Security Models. In *Proceedings of the 1982 IEEE Symposium on Security and Privacy*, p. 11.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec '23)*, pp. 79–90. arXiv:2302.12173.

Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., & Kıcıman, E. (2024). Defending Against Indirect Prompt Injection Attacks With Spotlighting. arXiv:2403.14720.

Hughes, J., et al. (2024). Best-of-N Jailbreaking. arXiv:2412.03556 (NeurIPS 2025).

Lehmann, E. L., & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3rd ed.). Springer Texts in Statistics. Springer, New York.

Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024). Formalizing and Benchmarking Prompt Injection Attacks and Defenses. In *33rd USENIX Security Symposium*, pp. 1831–1847. arXiv:2310.12815.

Nasr, M., Carlini, N., Sitawarin, C., Schulhoff, S., et al. (2025). The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections. arXiv:2510.09023 (USENIX Security 2026).

Neyman, J., & Pearson, E. S. (1933). On the Problem of the Most Efficient Tests of Statistical Hypotheses. *Philosophical Transactions of the Royal Society of London, Series A*, 231, 289–337. doi:10.1098/rsta.1933.0009.

On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models (2026). arXiv:2606.27567. (Author list not retrieved; cited by title and identifier.)

Perez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. NeurIPS 2022 ML Safety Workshop. arXiv:2211.09527.

Shafahi, A., Huang, W. R., Studer, C., Feizi, S., & Goldstein, T. (2019). Are Adversarial Examples Inevitable? In *ICLR 2019*. arXiv:1809.02104.

Sharma, M., et al. (2025). Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming. arXiv:2501.18837.

Tramèr, F., Carlini, N., Brendel, W., & Madry, A. (2020). On Adaptive Attacks to Adversarial Example Defenses. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. arXiv:2002.08347.

Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., & Beutel, A. (2024). The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. arXiv:2404.13208.

Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How Does LLM Safety Training Fail? In *NeurIPS 2023*. arXiv:2307.02483.

Wolf, Y., Wies, N., Avnery, O., Levine, Y., & Shashua, A. (2024). Fundamental Limitations of Alignment in Large Language Models. In *ICML 2024*, PMLR 235, pp. 53079–53112. arXiv:2304.11082.

Wu, T., Zhang, S., et al. (2025). Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy. In *ICLR 2025*. arXiv:2410.09102.

Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.

Zverev, E., Abdelnabi, S., Fritz, M., & Lampert, C. H. (2025). Can LLMs Separate Instructions From Data? And What Do We Even Mean By That? In *ICLR 2025*. arXiv:2403.06833.

Zverev, E., Kortukov, E., Panfilov, A., Volkova, A., Tabesh, S., Lapuschkin, S., Samek, W., & Lampert, C. H. (2026). ASIDE: Architectural Separation of Instructions and Data in Language Models. In *ICLR 2026*. arXiv:2503.10566.
