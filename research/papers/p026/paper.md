# Identity Half-Life: Why Agent State Persistence Needs a Non-Contracting Channel

**Target venue:** NeurIPS 2027 (main track; theory + simulation). 

*Code, results and figures: `code/`, `results/` in this directory; `bash code/run_all.sh` reproduces every number (about 2 CPU-minutes).*

---

## Abstract

Language-model agents keep memory, persona and task state across sessions by repeatedly *consolidating* it: summarizing, rewriting, reflecting, evicting. Every such step is done by a generative model, so the agent's identity passes through a long chain of noisy channels. We model this chain and derive four results. (i) **Contraction.** If each consolidation step has Dobrushin coefficient δ<1 (for example, because with probability p it regenerates content from sources that do not depend on the current memory, giving δ≤1−p), then any decoder's ability to recover the agent's original identity decays geometrically. Bayes accuracy is at most 1/Q+(1−1/Q)δ^t, and fidelity reaches its prior-determined floor with half-life ⌈ln 2/ln(1/δ)⌉. A necessary condition for persistence over an unbounded horizon is Σ_t(1−δ_t)<∞. (ii) **Reversion to the consolidator's prior.** Under a keep-or-regenerate channel, the long-run probability that an identity fact is intact equals the probability that the *memoryless* model would produce it unprompted. Identities that match model defaults therefore look persistent and idiosyncratic ones do not. (iii) **A redundancy ceiling.** Keeping R self-reconciled copies of identity, as in self-consistency or multiple summaries, makes the half-life grow exponentially in R when errors are independent. If a fraction γ of consolidation errors is common-mode, the half-life is at most ⌈ln 2/(−ln(1−γ))⌉ *for every R*. (iv) **Verified anchors.** Reconciling against a verbatim, integrity-checked store gives fidelity that does not depend on the horizon, in closed form. An unverified "anchor" that the agent regenerates gives no benefit. We also show that, under recency-decayed salience, the eviction age grows only logarithmically in an identity item's salience. Exact computation and Monte Carlo simulation match every closed form. For example, the simulated half-lives at large R are 139/69/35/14 steps against ceilings of 138.3/69.0/34.3/13.5 for γ=0.005/0.01/0.02/0.05. Finally, we give a measurement method: a per-fact empirical-Bayes estimator predicts persistence at 1000 steps from 5-step probe chains with RMSE 0.045–0.066, while naive exponential fits are off by 0.10–0.55. All experiments are simulations. We pre-register a protocol for testing the predictions on LLM agents.

## 中文摘要

基于大语言模型的智能体通过反复"整合"（总结、重写、反思、淘汰）来跨会话保持记忆、人格与任务状态。每一步整合都由生成模型完成，因此智能体的身份信息要经过一条很长的噪声信道链。本文对这条信道链建模，得到四个结果。(i) 收缩定理：若每步整合的 Dobrushin 系数 δ<1（例如以概率 p 从与当前记忆无关的来源重新生成内容，此时 δ≤1−p），则任何解码器恢复原始身份的能力都按几何速度衰减。贝叶斯识别准确率不超过 1/Q+(1−1/Q)δ^t，保真度以半衰期 ⌈ln2/ln(1/δ)⌉ 衰减到由先验决定的下限。无限期持久化的必要条件是 Σ(1−δ_t)<∞。(ii) 向整合器先验回归：在"保留或重生成"信道下，某条身份事实的长期完好概率等于无记忆模型自发生成该事实的概率。因此与模型默认值一致的身份看起来很稳定，个性化的身份则会被侵蚀。(iii) 冗余上限：保存 R 份自我调和的副本（如自洽投票、多份摘要）时，若误差独立，半衰期随 R 指数增长；若有比例 γ 的误差是共模的，则对任意 R，半衰期都不超过 ⌈ln2/(−ln(1−γ))⌉。(iv) 经校验的锚点：与逐字存储、带完整性校验的锚点定期对账，可得到与时间跨度无关的保真度，并有闭式解；由智能体自己重新生成、未经校验的"锚点"则没有作用。本文还证明，在按时间衰减的显著性淘汰机制下，身份条目被淘汰的年龄只随其显著性对数增长。精确计算与蒙特卡洛模拟同所有闭式结果一致：例如当 γ=0.005/0.01/0.02/0.05 时，大 R 下的仿真半衰期为 139/69/35/14 步，理论上限为 138.3/69.0/34.3/13.5。最后，本文给出一种测量方法：逐事实经验贝叶斯估计器只用 5 步探测链即可预测 1000 步后的持久度，RMSE 为 0.045–0.066，而朴素指数拟合的误差为 0.10–0.55。所有实验均为仿真。我们预注册了一套在大模型智能体上检验上述预测的实验方案。

---

## 1 Introduction

Ask how an AI agent's memory, identity and state persist and you get two kinds of answer. The engineering answer lists mechanisms: tiered memory with paging between a context window and external storage [Packer et al., 2023]; observation streams consolidated into "reflections" [Park et al., 2023]; recursive summaries that fold each new session into the previous summary [Wang et al., 2023]; forgetting curves that decay memories with age [Zhong et al., 2024]; self-written lessons [Shinn et al., 2023]; extraction-and-update pipelines [Chhikara et al., 2025]; and self-organizing note graphs [Xu et al., 2025]. The empirical answer is a set of failure reports. Persona instructions drift within eight rounds of dialogue [Li et al., 2024]. Repeatedly rewriting an evolving context causes "context collapse" [Zhang et al., 2026]. Long-horizon memory benchmarks show large accuracy drops [Maharana et al., 2024; Wu et al., 2025]. Persona drift appears across 23 frontier models in long coding sessions, and in-session compaction does not reliably reset it [Ding et al., 2026].

What is missing is a *quantitative theory* of persistence: when identity survives, how fast it decays, and which architectural choices change the rate rather than just the constant. Without one, designs are compared on benchmarks that run for tens of sessions, while deployed agents run for thousands.

This paper supplies that theory for one central mechanism, **recursive consolidation**. The agent's persistent state M_t is rewritten by a generative process at every step (session end, context compaction, reflection), M_{t+1}=C(M_t, E_t), where E_t is new experience. The view connects three bodies of work that the agent-memory literature has mostly not used: contraction coefficients for Markov kernels [Dobrushin, 1956; Polyanskiy & Wu, 2017]; iterated learning, where chains of Bayesian learners converge to their prior [Griffiths & Kalish, 2007]; and reliable computation from unreliable components, where redundancy suppresses independent faults [von Neumann, 1956]. Recent evidence that LLM errors are strongly *correlated* across models [Kim et al., 2025] turns out to matter a great deal in this setting.

**Contributions.**

1. **A formal model of state persistence under recursive consolidation**, with an *identity half-life* defined for any consolidation kernel (§3).
2. **Theorem 1 (contraction).** Identity recoverability decays geometrically at the rate of the kernel's Dobrushin coefficient. This gives a half-life law and the necessary condition Σ(1−δ_t)<∞ for persistence over an unbounded horizon. **Proposition 1** solves the keep-or-regenerate channel with rehearsal in closed form and shows that, without rehearsal, long-run fidelity equals the memoryless model's prior probability of the fact (§4.1–4.2).
3. **Theorem 2 (redundancy ceiling).** With R self-reconciled replicas, the half-life is bounded by ⌈ln 2/(−ln(1−γ))⌉ for all R whenever a fraction γ of errors is common-mode. It grows exponentially in R when γ=0 (§4.3).
4. **Theorem 3 (verified anchors)** gives closed-form, horizon-independent fidelity for periodic reconciliation against an integrity-checked verbatim store. **Corollary 2** shows that an anchor which is itself regenerated cannot escape Theorem 1. **Proposition 4** shows that recency-decayed salience evicts identity at an age logarithmic in salience, and that recall-refresh makes the lifetime polynomial in salience (§4.4–4.5).
5. **A measurement method and its validation** (§5, §6.5): an estimator that predicts long-horizon persistence from short probe chains plus "cold-start" probes of the memoryless model. In three heterogeneity scenarios it has RMSE ≤0.066 at t=1000. Naive aggregate fits have RMSE 0.10–0.55.
6. **Simulation evidence** (§6) that confirms every closed form. Exact computations and Monte Carlo runs are in `results/`.
7. **A pre-registered protocol and five falsifiable predictions** for LLM agents (§7).

We do not claim to solve agent identity. The results concern a single, ubiquitous mechanism, and all of our experiments are simulations.

## 2 Related Work

**Agent memory architectures.** CoALA organizes language-agent memory into working, episodic, semantic and procedural stores [Sumers et al., 2024]. Surveys catalogue memory sources, forms and operations [Zhang et al., 2024] and their dynamics of formation, evolution and retrieval [Hu et al., 2025]. Systems vary in *where* consolidation happens. MemGPT pages content between context and external storage [Packer et al., 2023]. Generative Agents synthesize reflections from an observation stream [Park et al., 2023]. Recursive summarization rewrites a running memory from the previous memory plus new context [Wang et al., 2023]. MemoryBank decays and reinforces memories along an Ebbinghaus-style curve [Zhong et al., 2024]. Reflexion stores verbal self-feedback [Shinn et al., 2023]. Mem0 extracts and updates salient facts [Chhikara et al., 2025]. A-MEM lets new memories trigger the evolution of older ones [Xu et al., 2025]. ACE observes that repeatedly rewriting an accumulated context in full shrinks it ("context collapse") and proposes incremental delta updates [Zhang et al., 2026]. Our analysis explains *why* full rewriting is fragile: each rewrite is a contraction (§4.1). It also explains why incremental updates help: they leave most items outside the contracting channel.

**Drift and identity.** Li et al. [2024] measure instruction/persona drift in self-chats and attribute it partly to attention decay. Their split-softmax fix strengthens attention to the system prompt, which in our terms is the verbatim anchor. Ding et al. [2026] benchmark persona drift in very long coding sessions. Perrier & Bennett [2025] propose identity metrics including a persistence score across re-instantiations. Menon [2026] argues for identity distributed across multiple anchors (identity files plus memory logs). Lam et al. [2026] propose the SSGM governance framework, which separates memory evolution from governance and reconciles mutable memory against an immutable record of raw observations. These works are architectural or empirical. Ours supplies the quantitative laws: decay rates, ceilings and fixed points. It also separates anchors that help (verified, non-contracting) from anchors that do not (regenerated) (Theorem 3, Corollary 2).

**Iterated transmission and model collapse.** Transmission chains of LLMs drift toward attractor states, and open-ended instructions produce stronger attraction [Perez et al., 2024]. Training models recursively on generated data loses the tails of the distribution [Shumailov et al., 2024]. In iterated Bayesian learning with posterior sampling, the chain's stationary distribution is the prior [Griffiths & Kalish, 2007]. Proposition 1 is the agent-memory analogue: without fresh evidence, an identity reverts to the consolidator's prior.

**Redundancy, self-correction and correlated errors.** Von Neumann [1956] showed that redundancy plus majority voting yields reliable computation from unreliable components *if faults are independent*. Self-consistency majority-votes over sampled reasoning paths [Wang et al., 2023b]. Huang et al. [2024] find that intrinsic self-correction without external feedback often fails. Kim et al. [2025] evaluate more than 350 LLMs and report that when two models both err they agree on the same wrong answer 60% of the time on one leaderboard dataset. Theorem 2 turns this observation into a bound on persistence.

**Memory integrity.** AgentPoison [Chen et al., 2024] shows that a very small poisoning rate of an agent's memory or knowledge base can control its behaviour. Our verified-anchor analysis assumes an integrity check, the same primitive that defends against such attacks.

**Contraction coefficients.** We use standard facts about Dobrushin and KL contraction coefficients [Dobrushin, 1956; Polyanskiy & Wu, 2017]. Our contribution is to apply them to agent state and to derive design consequences, not new information-theoretic inequalities.

## 3 Model

**State and identity.** Let X be the agent's *identity*: a finite collection of persistent facts such as its name, role, commitments, user-specific preferences and standing instructions. We take X uniform over Q values where convenient. At time 0 the persistent store holds M_0=enc(X). A *consolidation step* produces

  M_{t+1} = C_t(M_t, E_t, U_t),

where E_t is new experience and U_t is the consolidator's internal randomness (sampling, model nondeterminism). We call a step **closed-world** if E_t is independent of X given M_t. That is, the environment does not re-state the agent's identity. Under closed-world consolidation, X→M_0→M_1→… is a Markov chain with kernels K_t(m,·)=P(M_{t+1}∈·|M_t=m).

**Dobrushin coefficient.** δ(K)=sup_{m,m'} TV(K(m,·),K(m',·)) ∈[0,1]. If K has a *regeneration component*, i.e. K(m,·) ≥ p·ν(·) for all m and some distribution ν, then δ(K) ≤ 1−p (Doeblin). Operationally, with probability at least p the step writes content whose distribution does not depend on the current memory. For LLM consolidation, this happens when a summary drops a fact and later regenerates the slot from the model's own prior: a default name, a typical preference, a generic policy.

**Per-fact keep-or-regenerate channel.** For concrete calculations, each identity fact with true value x∈[q] is stored as a value m_t. Each step it is kept with probability 1−p, or else replaced by a draw from the consolidator's *attractor* distribution π. Rehearsal is an exception to closed-world consolidation: with probability h the environment re-states the true value, for example when the user addresses the agent by name, and m_{t+1}:=x. We write π(x) for the probability that the memoryless model would produce the true value unprompted. This is the quantity measured by the "cold-start probe" of §5.

**Fidelity and half-life.** Fidelity is c_t=P(m_t=x). Its limit is c_∞. The **identity half-life** t_{1/2} is the first t at which the excess fidelity c_t−c_∞ falls to half its initial value 1−c_∞. For a whole-memory kernel we use the analogous quantity defined on Bayes recoverability (Theorem 1).

**Redundancy model.** The fact is stored as R replicas. In each step, with probability γ a *common-mode* error occurs: all replicas are overwritten by a single draw Y∼π. This models the same model making the same systematic substitution in every copy. Otherwise each replica is independently overwritten with probability p by an independent draw from π. A reconciliation rule f then sets every replica to f(replicas). We only require f to be *unanimity-preserving*: if all replicas equal y, the result is y. Plurality vote, with ties going to the previous consensus, is the instance we simulate.

**Verified anchor model.** A separate write-protected store A holds x verbatim together with an integrity check, such as a content hash. Every T steps the agent reads A. With probability 1−ε the read verifies and the memory is reset to x. Otherwise the read fails detectably and the memory is left unchanged. The contrasting *generative anchor* is a second copy that is itself rewritten by the consolidator at every step and is not verified.

## 4 Theory

### 4.1 Contraction and the half-life law

**Theorem 1.** Under closed-world consolidation with kernels K_0,…,K_{t−1} and δ_s=δ(K_s):
(a) for all identities x, x', TV(P_{M_t|X=x}, P_{M_t|X=x'}) ≤ ∏_{s<t} δ_s;
(b) if X is uniform on Q values, every decoder ψ satisfies P(ψ(M_t)=X) ≤ 1/Q + (1−1/Q)∏_{s<t} δ_s;
(c) I(X;M_t) ≤ (∏_{s<t} η_s) H(X) ≤ (∏_{s<t} δ_s) H(X), where η_s is the KL contraction coefficient of K_s.

*Proof.* (a) TV contracts under any kernel by its Dobrushin coefficient, TV(μK,νK)≤δ(K)TV(μ,ν) [Dobrushin, 1956]. Iterate this, starting from TV(P_{M_0|x},P_{M_0|x'})≤1. (b) The Bayes accuracy is (1/Q)Σ_m max_x P(m|x). Fix any x_0. Then max_x P(m|x) ≤ P(m|x_0)+Σ_{x≠x_0}(P(m|x)−P(m|x_0))_+. Summing over m gives Σ_m max_x P(m|x) ≤ 1+(Q−1)max_{x}TV(P_{·|x},P_{·|x_0}), and we apply (a). (c) This is the strong data-processing inequality applied along the chain X→M_s→M_{s+1}, together with η_KL≤δ [Polyanskiy & Wu, 2017]. ∎

The bound applies to the agent itself, since the agent is one particular decoder. It also applies to an evaluator with unlimited compute. In a time-homogeneous chain with δ<1, recoverability above chance halves every ⌈ln 2/ln(1/δ)⌉ steps. With a Doeblin regeneration probability p, this is at most ⌈ln 2/(−ln(1−p))⌉≈0.69/p steps.

**Corollary 1 (necessary condition for unbounded persistence).** Under closed-world consolidation, if Σ_t(1−δ_t)=∞ then I(X;M_t)→0 and Bayes accuracy→1/Q. Therefore any architecture that keeps identity information above chance indefinitely, without environmental rehearsal, must contain a channel whose contraction coefficients satisfy Σ(1−δ_t)<∞. The canonical example is a store that is never regenerated, with δ=1 on the identity component.

*Proof.* ∏δ_t→0 if and only if Σ(1−δ_t)=∞ (for δ_t∈[0,1]). Apply Theorem 1. ∎

Corollary 1 is the central design message. Mechanisms that **rewrite** identity at a non-vanishing rate (summarize, reflect, compact, "evolve") all have Σ(1−δ_t)=∞, however good the model is. Improving the consolidator lowers p, which lengthens the half-life linearly in 1/p, but it cannot make persistence unbounded.

### 4.2 Reversion to the consolidator's prior

**Proposition 1.** For the keep-or-regenerate channel with rehearsal probability h,
 c_{t+1} = h + (1−h)[(1−p)c_t + pπ(x)],
so c_t = c_∞ + (1−c_∞)λ^t with λ=(1−h)(1−p) and
 c_∞ = [h + (1−h)pπ(x)] / [1−(1−h)(1−p)].
In particular, with h=0, c_∞=π(x) and t_{1/2}=⌈ln 2/(−ln(1−p))⌉.

*Proof.* Condition on whether rehearsal occurs and then on keep versus regenerate. A regenerated value equals x with probability π(x) whatever the current value. The recursion is affine, and its fixed point and rate follow directly. ∎

Without rehearsal, the agent's long-run identity is *the consolidator's prior*. This parallels Griffiths & Kalish [2007]. It predicts a **survivorship illusion**. Facts that coincide with model defaults (π(x) large) look stable under drift metrics. Idiosyncratic facts (π(x) small) decay almost to zero. Aggregate persona-consistency scores therefore overstate persistence whenever an evaluation's identities are close to model defaults.

### 4.3 The redundancy ceiling

**Theorem 2.** In the redundancy model with any unanimity-preserving reconciliation rule:
(a) the consensus value is a Markov chain whose kernel satisfies K(s,·) ≥ γπ(·) for every s. Hence, for every R, c_t − c_∞ ≤ (1−γ)^t(1−c_∞) and t_{1/2} ≤ ⌈ln 2/(−ln(1−γ))⌉;
(b) the per-step loss probability from the correct state is at least γ(1−π(x)) for every R;
(c) if γ=0 and f is plurality vote with status-quo tie-breaking, the per-step loss probability is at most P(Bin(R, p(1−π(x))) ≥ R/2). For p(1−π(x))<1/2 this is ≤ exp(−R·D(½‖p(1−π(x)))), so the time until fidelity drops below ½ is at least ln 2/(−ln(1−that bound)). This grows exponentially in R.

*Proof.* (a) With probability γ all replicas equal Y∼π, and unanimity preservation makes the new consensus equal Y. So the consensus kernel dominates γπ, its Dobrushin coefficient is ≤1−γ, and |P_t(x)−s(x)| ≤ TV(δ_xK^t, s) ≤ (1−γ)^t(1−s(x)), where s is the stationary distribution. (b) A common-mode draw Y≠x occurs with probability γ(1−π(x)) and is adopted by unanimity. (c) If fewer than R/2 replicas move away from x, then x holds a strict majority. Each replica leaves x independently with probability p(1−π(x)). The Chernoff bound completes the proof. ∎

Theorem 2 says that **self-consistency and multiple-summary schemes cannot buy persistence beyond 1/γ**. Because γ measures how often the *same* model makes the *same* substitution, the relevant evidence is the error-agreement rate between samples or copies, such as the 60% wrong-answer agreement reported by Kim et al. [2025] across models. Same-model replicas are likely to be more correlated still, though this is untested here. Diversity of consolidators, meaning different models, prompts or modalities, raises the ceiling by lowering γ. Only a channel outside the generative loop removes it.

### 4.4 Verified anchors

**Theorem 3.** In the per-fact model with rehearsal h, consider reconciliation against a verified anchor every T steps with read-failure probability ε. Let λ=(1−h)(1−p) and let c^* be the Proposition 1 fixed point. In the periodic steady state the post-reconciliation fidelity is
 u = [1−ε+εc^*(1−λ^T)] / (1−ελ^T),
the time-averaged fidelity over a period is
 c̄ = c^* + (u−c^*)(1−λ^T)/(T(1−λ)),
and the minimum fidelity, just before reconciliation, is c^*+(u−c^*)λ^{T−1}. None of these depend on the horizon.

*Proof.* Between reconciliations, Proposition 1 gives c_{s}=c^*+(c_0−c^*)λ^s. A reconciliation maps c to (1−ε)+εc. Solving the fixed point u=(1−ε)+ε[c^*+(u−c^*)λ^T] gives u, and averaging the geometric segment gives c̄. ∎

**Corollary 2 (generative anchors do not help).** Suppose the "anchor" is itself rewritten by the consolidator, for example an identity summary that the agent regenerates, and is not verified. Then the joint system (memory, anchor) is closed-world consolidation with Σ(1−δ_t)=∞, so Theorem 1 applies and identity information decays to zero. Integrity verification is what places the anchor outside the contracting loop.

The storage cost of a verified anchor is that of the identity alone, O(k log q) bits for k facts. It does not grow with time. An append-only episodic log can also serve as a non-contracting store, but its size grows linearly in time, and recovering identity from it requires *retrieval*, which is a separate channel with its own errors (not modelled here).

### 4.5 Eviction under recency-decayed salience

Extractive memories do not corrupt facts. They *evict* them. Consider a budget of B slots. Each step n new items arrive with i.i.d. salience S∼F, and an item of age a has score S·λ^a with λ<1, an Ebbinghaus-style decay [Zhong et al., 2024]. The top B scores are kept, and the identity item has salience s.

**Proposition 4 (mean-field).** In steady state the retention threshold θ solves n Σ_{a≥0} P(S>θλ^{−a}) = B. The identity item is evicted at age
 a^* = ln(s/θ)/ln(1/λ),
which is *logarithmic* in salience. If the item is recalled with probability r per step and each recall resets its age, its expected lifetime is the expected waiting time for m=⌊a^*⌋+1 consecutive non-recalls, E[L]=((1−r)^{−m}−1)/r. This is approximately (s/θ)^{ln(1/(1−r))/ln(1/λ)}, i.e. polynomial in salience. Pinning (λ=1 for identity items) gives an unbounded lifetime.

*Derivation.* Scores decrease monotonically with age. So, at a constant threshold, an item is present exactly while its score exceeds θ. The expected number of present items is nΣ_aP(Sλ^a>θ), and setting it equal to B defines θ. The lifetime result is the standard waiting time for a run of m failures. The threshold step is a mean-field (law-of-large-numbers) approximation, which §6.4 tests. ∎

## 5 Measuring Identity Half-Life

The theory implies that a deployed agent's persistence is determined by per-fact parameters (p_i, π_i(x_i), h_i) and that these are heterogeneous across facts. We propose estimating them from cheap measurements and then *extrapolating* with the model, instead of running thousand-session evaluations.

**Data.** (1) *Probe chains:* for each identity fact i, run n_ch independent chains of K consolidation steps starting from a correct store, and after each step probe whether the fact is intact. (2) *Cold-start probes:* query the memoryless model m times for the fact. The fraction of times it outputs the true value estimates π_i(x_i).

**Estimators of population fidelity F(t)=mean_i c_i(t).**
- **A, naive aggregate:** fit c+(1−c)e^{−bt} to the aggregate curve for t≤K.
- **B, aggregate with measured floor:** the same fit with c fixed at the mean cold-start rate.
- **C, per-fact plug-in:** p̂_i=(k_i/n_i)/(1−π̂_i), where k_i counts correct→wrong transitions among the n_i correct-state transitions. Then F̂(t)=mean_i[π̂_i+(1−π̂_i)(1−p̂_i)^t].
- **D, per-fact empirical Bayes (proposed):** a nonparametric maximum-likelihood prior over p on a grid [Kiefer & Wolfowitz, 1956], fitted by EM to the binomial likelihoods k_i∼Bin(n_i, p_i(1−π̂_i)). F(t) is then predicted with per-fact posterior expectations of (1−p_i)^t.

The reasoning behind D is as follows. A mixture of geometric decays is not geometric. Aggregate fits therefore misjudge the long tail. Per-fact MLEs, in turn, set p̂=0 for slow facts that showed no failures in a short chain. Pooling across facts through the prior corrects both problems.

## 6 Results

All results below come from the scripts in `code/` (logs: `results/*_log.txt`). Default attractor in E2–E3: q=10 values, π(default)=0.5 and π=0.5/9 for each other value. The "idiosyncratic" fact has π(x)=0.056 and the "default-matching" fact has π(x)=0.5.

### 6.1 Theorem 1 is valid and tight for keep-or-regenerate (E1, Fig. 1)

We built four random consolidation kernels on 30–60 memory states, each the sum of a content-preserving part (identity mixed with a random stochastic matrix) and a Doeblin regeneration part with p∈{0.01,…,0.1}. We computed exact Bayes accuracy and I(X;M_t) for t≤300 by matrix powers. Across all 124 checkpoints, neither bound was violated (`e1_contraction.json`, `n_violations=0`). For the pure keep-or-regenerate kernel (S=30, |X|=4, p=0.01) the accuracy bound is attained at t=100 (exact 0.5245, bound 0.5245; MI 0.594 bits vs. bound 0.732 bits). When the content-preserving part also scrambles memory, the bound is conservative (e.g. exact accuracy 0.125 vs. bound 0.238 at t=100). This is expected, because δ only captures worst-case contraction.

### 6.2 Reversion to the prior and the value of rehearsal (E3, Fig. 3)

With p=0.02 and 4,000 Monte Carlo runs per cell, rewrite-only memory (P0) converges to π(x). Final fidelity at t=2000 is 0.054 for the idiosyncratic fact (Proposition 1: 0.056) and 0.491 for the default-matching fact (Proposition 1: 0.500). At t=100 the simulated values are 0.173 and 0.557, against closed forms of 0.181 and 0.566. Rehearsal at h=0.005 (P1) raises the floor to a tail average of 0.246 (closed form 0.245) and 0.597 (0.600). The **survivorship illusion** is visible: under identical dynamics, the default-matching fact looks about 9× more persistent than the idiosyncratic one.

### 6.3 The redundancy ceiling (E2, Fig. 2)

Table 1 reports half-lives computed from Monte Carlo transition matrices (400k samples per row, with the common-mode part added exactly), for p=0.05 per replica.

*Table 1. Identity half-life (steps) vs. replicas R and common-mode fraction γ (source: `results/e2_redundancy.json`).*

| γ \ R | 1 | 3 | 5 | 9 | 31 | Thm 2 ceiling |
|---|---|---|---|---|---|---|
| 0 | 14 | 328 | 4402 | >2·10^4 (lower bound 2.8·10^4) | lower bound 7.5·10^12 | ∞ |
| 0.005 | 13 | 98 | 135 | 139 | 139 | 138.3 |
| 0.01 | 12 | 57 | 68 | 69 | 69 | 69.0 |
| 0.02 | 10 | 32 | 35 | 35 | 35 | 34.3 |
| 0.05 | 7 | 13 | 14 | 14 | 14 | 13.5 |

With independent errors the half-life grows exponentially in R. With even γ=0.005, it saturates by R≈7 at the integer ceiling ⌈ln2/(−ln(1−γ))⌉ (139, 69, 35, 14). Per-step loss from the correct state converges to the floor γ(1−π(x)), e.g. 9.44·10^{-3} at γ=0.01 for every R≥15. In the policy simulation (E3, p=0.02), R=7 self-reconciled replicas with γ=0.01 (P2) delay decay but reach the same floor as rewrite-only (fidelity 0.056 at t=2000). The idealized γ=0 version (P3) keeps 0.999.

### 6.4 Verified versus generative anchors; eviction (E3, E4; Figs. 3–4)

*Table 2. Persistence policies, idiosyncratic fact, p=0.02 (source: `results/e3_policies.json`). Storage is in copies of the identity.*

| Policy | storage | fidelity t=2000 | time-avg (t>1000) | closed form |
|---|---|---|---|---|
| P0 rewrite only | 1 | 0.054 | 0.055 | c_∞=0.056 |
| P1 + rehearsal h=0.005 | 1 | 0.240 | 0.246 | 0.245 |
| P2 R=7 replicas, γ=0.01 | 7 | 0.056 | 0.055 | — (Thm 2: → floor) |
| P3 R=7 replicas, γ=0 | 7 | 0.999 | 0.999 | — |
| P4 verified anchor, T=20, ε=0.01 | 2 | 0.997* | 0.838 | 0.838 (min 0.697) |
| P5 verified anchor, T=1, ε=0.01 | 2 | 1.000 | 1.000 | 0.9998 |
| P6 generative anchor, T=20 | 2 | 0.054 | 0.054 | — (Cor. 2: → c_∞) |

*t=2000 is a reconciliation step for P4. Its time average is the relevant figure.

Theorem 3's closed form for the time-averaged fidelity under P4 (0.8377) matches simulation (0.838). Reconciling against a *generative* anchor (P6) is indistinguishable from doing nothing (0.054). This is Corollary 2 in action.

For eviction (E4, n=5 arrivals/step, B=100, S∼Exp(1), 20 runs per cell), the simulated eviction age matches a^*=ln(s/θ)/ln(1/λ) closely when s/θ≥2. At λ=0.99, for example: s/θ=10 gives 230.4±6.6 simulated vs. 229.1 predicted, and s/θ=1000 gives 689.6±4.6 vs. 687.3. Making the identity item 100× more salient (s/θ 10→1000) only triples its lifetime. With recall-refresh (λ=0.9), the mean lifetimes over 30 runs are within about 2 standard errors of the run-length prediction in 10 of 11 uncensored cells. Examples: r=0.2, s/θ=30 gives 6,962±1,324 vs. 7,884, and r=0.3, s/θ=10 gives 9,099±1,248 vs. 8,522. The exception is r=0.2, s/θ=10: 528±61 vs. 673. One cell (r=0.3, s/θ=30, prediction 431k steps) was censored at 60k steps in 26 of 30 runs.

### 6.5 Estimating long-horizon persistence from short chains (E5, Fig. 5)

Setup: M=300 facts, K=5 steps, 20 chains per fact, 50 cold-start probes per fact, and 40 replications. We used three heterogeneity scenarios. *Bimodal:* 50% fast facts with p∼Beta(2,20) and 50% slow facts with p∼Beta(1,300). *Lognormal:* median p=0.01 and σ=1.5. *Sub-resolution:* 60% of facts with p=2·10^{-4}, far below the per-fact resolution 1/(n_ch K)=0.01. In all scenarios 30% of facts are default-matching.

*Table 3. RMSE of predicted population fidelity at t=200 / t=1000 (bias in parentheses at t=1000). Source: `results/e5_estimator.json`.*

| Estimator | bimodal | lognormal | sub-resolution |
|---|---|---|---|
| A naive aggregate fit | 0.241 / 0.393 (+0.388) | 0.395 / 0.555 (+0.550) | 0.109 / 0.176 (+0.168) |
| B aggregate, cold-start floor | 0.250 / 0.096 (−0.095) | 0.209 / 0.052 (−0.052) | 0.468 / 0.398 (−0.398) |
| C per-fact plug-in MLE | 0.098 / 0.242 (+0.241) | 0.154 / 0.298 (+0.297) | 0.026 / 0.094 (+0.093) |
| **D per-fact empirical Bayes** | **0.030 / 0.066 (+0.033)** | **0.028 / 0.054 (+0.030)** | **0.013 / 0.045 (+0.005)** |

All four estimators fit the observed window equally well: RMSE at t=5 is ≤0.007. They diverge only when extrapolated. True mean fidelities at t=1000 are 0.284, 0.235 and 0.585. The naive fit overstates long-horizon fidelity by up to 0.55, because its fitted floor absorbs the slow facts. The plug-in MLE overstates it because it assigns p̂=0 to slow facts with no observed failures. The empirical-Bayes estimator has a small positive bias (+0.005 to +0.033). We regard short-window agreement combined with long-window disagreement as the main methodological warning for current drift benchmarks.

## 7 Discussion and Pre-registered Predictions

**What persists, and why.** Theorem 1 and Corollary 1 recast the question of how memory, identity and state persist. Persistence is not a property of a memory *format*, whether vector store, graph or summary. It is a property of the *channel* each identity-bearing bit passes through at every step. Any bit that is regenerated by a generative model at a non-vanishing rate has a finite half-life and reverts to the model's prior. Bits that persist indefinitely must travel through a non-contracting channel (verbatim, write-protected, verified storage) or be re-supplied by the environment (rehearsal). Agent designs can therefore be audited by tracing each identity component to its channel and asking whether Σ(1−δ_t) is finite.

**Design implications.** (1) Pin identity verbatim and verify it, and do not "evolve" it. Updates to identity should be explicit, logged and authorized writes to the anchor, not side effects of summarization. (2) Do not count same-model redundancy as protection. Estimate γ from the rate at which independent consolidations agree on the same wrong value, and expect a half-life of at most about 0.69/γ. (3) Salience-based retention cannot protect identity under decay, because lifetime is logarithmic in salience. Use pinning instead. (4) Benchmark designers should report persistence separately for default-matching and idiosyncratic facts, and should extrapolate with per-fact models rather than aggregate curves.

**Pre-registered protocol (not run here).** *Agents:* at least three open-weight instruction models of different families and sizes. Each is wrapped in (a) recursive-summary memory [Wang et al., 2023], (b) a fact-extraction memory in the style of Mem0, and (c) (a) plus a hash-verified identity file reconciled every T∈{1,5,20} sessions. *Identity:* 60 facts per agent (name, role, 20 user preferences, 20 standing instructions, 18 arbitrary codes), half chosen to be idiosyncratic. Idiosyncrasy is measured by cold-start probing (m=50 samples), and facts are stratified by π̂. *Dynamics:* simulated sessions of distractor dialogue with no re-statement (closed-world), plus a rehearsal arm in which each fact is re-stated with h=0.01. *Measurements:* K=5-step probe chains (20 per fact), long chains to t=500 as held-out ground truth, and same-model R∈{1,3,5,9} multi-summary reconciliation. *Analysis:* estimator D fitted on t≤5, evaluated at t∈{50,200,500}.

**Falsifiable predictions.**
- **P1 (prior reversion).** In closed-world arms, the long-run retention of each fact is predicted by its cold-start rate π̂_i (slope ≈1, intercept ≈0). This fails if retention of idiosyncratic facts stays well above π̂_i at t=500 without rehearsal.
- **P2 (survivorship illusion).** The half-life gap between default-matching and idiosyncratic facts persists after controlling for fact length and type.
- **P3 (redundancy ceiling).** Half-life versus R plateaus, and the plateau is within a factor of 2 of ln 2/(−ln(1−γ̂)), where γ̂ is the measured rate of common-mode substitution. Mixing consolidator models raises the plateau.
- **P4 (verified vs. generative anchors).** Arm (c) has horizon-independent fidelity that matches Theorem 3 given the measured p and ε. An unverified, agent-regenerated identity summary does not differ from arm (a) at t=500.
- **P5 (estimation).** Estimator D predicts held-out fidelity at t=500 more accurately than aggregate exponential fits.

## 8 Limitations

1. **No LLM experiments.** Every quantitative result is from simulations of the stylized channels in §3, or from exact computation on random kernels. Whether real consolidators are well described by a keep-or-regenerate channel with a stable p is an empirical question, addressed by the protocol in §7 but not answered here. LLM consolidation is probably context-dependent and non-stationary (p may depend on memory length and content).
2. **Independence across facts and steps.** The per-fact analyses assume facts are consolidated independently. Real summaries drop correlated groups of facts, and errors in one fact can induce errors in others, for example through a consistent but wrong persona. Theorem 1 holds for arbitrary whole-memory kernels, but Propositions 1 and 4 and Theorems 2 and 3 are per-fact.
3. **Closed-world assumption.** Real environments rehearse identity in structured ways, such as users who repeat preferences. We model only i.i.d. rehearsal. Adversarial "rehearsal" (memory poisoning [Chen et al., 2024]) is outside the model.
4. **Bounds can be loose.** The Dobrushin bound is worst-case and conservative when consolidation also scrambles content (§6.1). Proposition 4 relies on a mean-field threshold and is a validated approximation, not a theorem. Recall-refresh predictions deviate by up to about 22% (one cell lies outside 2 SE).
5. **Verified anchors are not free.** Theorem 3 assumes an integrity check that is itself reliable, and an agent that *uses* the anchor once it has been read. ContextEcho-style findings [Ding et al., 2026] and attention-decay results [Li et al., 2024] suggest that a correct anchor in storage does not guarantee anchored behaviour. Our "fidelity" is fidelity of *stored state*, not of *behaviour*.
6. **Identity is narrower here than in philosophy.** We treat identity as a finite set of facts. Values, style and skills that are not easily enumerated are not covered, although Theorem 1 applies to any encoded variable.
7. **Estimator scope.** Estimator D assumes the keep-or-regenerate form, and its accuracy beyond the probe window depends on pooling across facts. When slow facts are few and heterogeneous, its prior is poorly identified. The 40-replication results show residual bias of up to +0.033 at t=1000.

## 9 Conclusion

Recursive consolidation, the mechanism by which agents carry memory and identity forward, behaves like a sequence of contracting channels. Its consequences are quantitative and simple. Identity information has a half-life set by the regeneration rate. Without fresh evidence an identity reverts to the consolidator's prior. Same-model redundancy is capped by the common-mode error rate. Salience under decay buys only logarithmic lifetime. Only a non-contracting channel, a verified verbatim anchor or genuine rehearsal, gives persistence that does not depend on the horizon. We have checked these results in simulation, provided a short-chain estimator for measuring them, and stated the predictions that LLM-agent experiments would have to confirm or refute.

## References

- Chen, Z., Xiang, Z., Xiao, C., Song, D., & Li, B. (2024). AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2407.12784.
- Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
- Ding, X., Yu, Y., Liu, C., & Zhao, B. (2026). ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions. arXiv:2605.24279.
- Dobrushin, R. L. (1956). Central Limit Theorem for Nonstationary Markov Chains. I. *Theory of Probability and Its Applications*, 1(1), 65–80. doi:10.1137/1101006.
- Griffiths, T. L., & Kalish, M. L. (2007). Language Evolution by Iterated Learning With Bayesian Agents. *Cognitive Science*, 31(3), 441–480.
- Hu, Y., et al. (2025). Memory in the Age of AI Agents: A Survey (Forms, Functions and Dynamics). arXiv:2512.13564.
- Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *International Conference on Learning Representations (ICLR 2024)*. arXiv:2310.01798.
- Kiefer, J., & Wolfowitz, J. (1956). Consistency of the Maximum Likelihood Estimator in the Presence of Infinitely Many Incidental Parameters. *Annals of Mathematical Statistics*, 27(4), 887–906. doi:10.1214/aoms/1177728066.
- Kim, E. M., Garg, A., Peng, K., & Garg, N. (2025). Correlated Errors in Large Language Models. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, PMLR 267, 30038–30066. arXiv:2506.07962.
- Lam, C., Li, J., Zhang, L., & Zhao, K. (2026). Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework. arXiv:2603.11768.
- Li, K., Liu, T., Bashkansky, N., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2024). Measuring and Controlling Instruction (In)Stability in Language Model Dialogs. *Conference on Language Modeling (COLM 2024)*. arXiv:2402.10962.
- Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. *Proceedings of ACL 2024 (Volume 1: Long Papers)*, 13851–13870.
- Menon, P. G. (2026). Persistent Identity in AI Agents: A Multi-Anchor Architecture for Resilient Memory and Continuity. arXiv:2604.09588.
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *Proceedings of UIST 2023*. doi:10.1145/3586183.3606763.
- Perez, J., et al. (2024). When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions. arXiv:2407.04503.
- Perrier, E., & Bennett, M. T. (2025). Agent Identity Evals: Measuring Agentic Identity. arXiv:2507.17257.
- Polyanskiy, Y., & Wu, Y. (2017). Strong Data-Processing Inequalities for Channels and Bayesian Networks. In *Convexity and Concentration*, IMA Volumes in Mathematics and its Applications, vol. 161, pp. 211–249. Springer. arXiv:1508.06025.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2303.11366.
- Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755–759.
- Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive Architectures for Language Agents. *Transactions on Machine Learning Research*. arXiv:2309.02427.
- von Neumann, J. (1956). Probabilistic Logics and the Synthesis of Reliable Organisms from Unreliable Components. In C. E. Shannon & J. McCarthy (Eds.), *Automata Studies* (pp. 43–98). Princeton University Press.
- Wang, Q., et al. (2023). Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. arXiv:2308.15022.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023b). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. arXiv:2203.11171.
- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D. (2025). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. *ICLR 2025*. arXiv:2410.10813.
- Xu, W., Mei, K., Gao, H., Tan, J., Liang, Z., & Zhang, Y. (2025). A-MEM: Agentic Memory for LLM Agents. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2502.12110.
- Zhang, Q., et al. (2026). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. *ICLR 2026*. arXiv:2510.04618.
- Zhang, Z., et al. (2024). A Survey on the Memory Mechanism of Large Language Model based Agents. arXiv:2404.13501.
- Zhong, W., Guo, L., Gao, Q., Ye, H., & Wang, Y. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(17), 19724–19731.

---

*Provenance of numbers.* All numbers in §6 and in the abstract come from `code/*.py`, run in this environment (outputs in `results/`). Numbers attributed to other papers (e.g. 60% wrong-answer agreement, 23 models, eight rounds) are taken from the cited works' abstracts as retrieved by web search. Figures: `results/fig1_contraction.png` … `fig5_estimator.png`.
