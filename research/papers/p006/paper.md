# Long Context Is Not Long-Term Memory: A Recency–Retention Bound for Softmax Readout under Knowledge Revision

**Target venue:** NeurIPS (main track; theory + controlled experiments)

---

## Abstract

Context windows of language models have grown by orders of magnitude, and it is common to treat a long context as a long-term memory: if everything a user ever said fits in the window, the model "remembers" it. We argue that this equivalence fails for a specific and checkable reason. Long-term memory must handle *revision*: facts get updated, and a query must return the latest version while still retrieving old facts that were never revised. We formalize this as retrieval from a stream of key–value writes and prove a **recency–retention bound** for any single softmax readout head with a history-independent query, bounded logits (all logits in [−B, B]), arbitrary causal key representations and arbitrary positional encoding. Suppose the head must return the latest version of a key whose versions are Δ positions apart, in a context of T tokens, with attention weight at least 1−ε on the correct token. Then it can do so only for facts whose latest version has age A satisfying ⌊A/Δ⌋+1 ≤ (2B − log(T−1)) / log((1−ε)/ε). So the *revision-robust horizon* scales with the revision spacing Δ and the logit budget B, and **decreases** with context length. The window size does not appear in it at all. Beyond this horizon, the head must either lose old unrevised facts or confuse stale versions with current ones. An explicit construction comes within a constant factor of the bound. A delta-rule fast-weight memory escapes it entirely, but its capacity is limited by its state dimension. Numerical experiments (code included) confirm that the construction tracks the bound, trace the recency–retention frontier, and show the capacity limit of the delta rule. In small Transformers (1, 2 and 4 layers) and a delta-rule model, all trained on CPU on a revision-stream task, we find [E2 SUMMARY]. Together these results turn "is long context long-term memory?" into a quantitative statement. A context window is a memory store whose *update semantics* are bounded by logit range, not by window size. We give a falsifiable test protocol for frontier models.

## 中文摘要

随着语言模型上下文窗口扩大几个数量级，人们常把“长上下文”等同于“长期记忆”：只要用户说过的一切都能放进窗口，模型就“记得”。本文指出这一等价在一个具体、可检验的方面失效：长期记忆必须处理**修订**——事实会被更新，查询应返回最新版本，同时仍能检索从未修订过的旧事实。我们把该问题形式化为对键值写入流的检索，并对任意单个 softmax 读出头（查询与历史无关、logit 有界于 [−B, B]、键表示可以是任意因果表示、位置编码任意）证明了一个**新近性–保持性界**：若要在长度为 T 的上下文中，以至少 1−ε 的注意力权重取回一个各版本间隔为 Δ 的键的最新版本，则只有最新版本“年龄” A 满足 ⌊A/Δ⌋+1 ≤ (2B − log(T−1)) / log((1−ε)/ε) 的事实才能被可靠检索。换言之，**抗修订记忆视野**与修订间隔 Δ 和 logit 预算 B 成正比，且随上下文长度**下降**，与窗口大小本身无关；超出该视野时，读出头要么丢失旧的未修订事实，要么把过时版本误当成当前版本。我们给出一个在常数因子内达到该界的显式构造，并证明 delta 规则快权重记忆可以完全规避该界，但其容量受状态维度限制。数值实验（附代码）验证了构造与界的吻合、新近性–保持性前沿以及 delta 规则的容量极限。在 CPU 上训练的小型 Transformer（1/2/4 层）与 delta 规则模型的修订流任务中，我们发现[E2 中文摘要]。这些结果将“长上下文是否等于长期记忆”转化为一个定量命题：上下文窗口是一种其**更新语义**受 logit 范围而非窗口大小约束的记忆存储。我们还给出了针对前沿模型的可证伪检验方案。

---

## 1 Introduction

Context windows have grown from a few thousand tokens to hundreds of thousands and, in some deployed systems, millions. A natural reading of this trend is that the "memory problem" of language agents is being solved by brute force. If an assistant can keep a user's entire history in context, why would it need a separate memory system? Benchmarks have begun to question this reading. LongMemEval [Wu et al., 2025] reports that commercial chat assistants and long-context LLMs lose roughly 30% accuracy when information must be remembered across sustained interactions. "Knowledge updates" is one of its five core abilities. LoCoMo [Maharana et al., 2024] finds that LLMs struggle with temporal and causal structure over very long dialogues. Most directly, PI-LLM [Wang & Sun, 2025] streams semantically related key–value updates and queries only the final values. Retrieval accuracy declines log-linearly toward zero as the number of updates grows, even though the final values sit immediately before the query. The errors are retrievals of *overwritten* values.

These are empirical observations about specific models. They leave open whether the failure is incidental (fixable by more training or better prompts) or structural (a property of how softmax attention reads an append-only store). This paper gives a structural answer for the simplest setting where the question is sharp.

**The distinction we formalize.** A *store* keeps everything; a *memory* must also support **revision semantics**: after a fact is updated, reads should return the new value, while facts that were never updated stay retrievable however old they are. A context window is an append-only store. Old versions are never deleted, so "which version is current" must be decided at read time, by the attention scores. We ask: *what does it cost a softmax readout to decide recency, and how does that cost interact with retaining old facts?*

**Contributions.**

1. **A recency–retention impossibility bound (Theorem 1).** Consider any single softmax readout head with a history-independent query, logits bounded in [−B, B], arbitrary causal key representations and arbitrary positional scheme (absolute, relative, RoPE, ALiBi, learned timestamps, or none). If it ε-faithfully retrieves the latest version on all streams with revision spacing ≥ Δ and latest-version age ≤ A in a T-token context, then ⌊A/Δ⌋+1 ≤ (2B − log(T−1))/L_ε, where L_ε = log((1−ε)/ε). The revision-robust horizon is O(Δ·B), independent of window size and *decreasing* in T. Without revisions the bound reduces to the familiar attention-dilution condition.
2. **Near-tightness (Proposition 2).** An explicit exact-match plus linear-recency head achieves a horizon within a factor log(1/ρ)/log(1+1/ρ) (with ρ = ε/(1−ε)) of the bound, up to an additive term. Numerically, it reaches 55–80% of the bound across the logit budgets and context lengths where the bound is non-trivial (Fig. 1a).
3. **A clean contrast with overwrite memories (Proposition 3).** A delta-rule fast-weight memory [Schlag et al., 2021; Yang et al., 2024] returns the latest version exactly, for any number of revisions and any age, as long as the number of distinct keys is at most its key dimension. Beyond that, interference grows with M/d (Fig. 1c). Append-only context and overwrite memory therefore fail along *orthogonal* axes: revision-consistency versus capacity.
4. **Controlled CPU experiments with trained models.** We train 1-, 2- and 4-layer RoPE Transformers and a single-layer delta-rule model on a revision-stream task (3 seeds each). We evaluate them with the number of versions R decoupled from recency, at lengths within and 4× beyond training. [E2 contributions sentence]
5. **A falsifiable protocol for frontier models** (Section 7). It derives three signatures that distinguish "bounded logit budget" from other explanations of PI-LLM-style failures.

We do not claim that long-context models cannot be *trained* to behave like memories in practice. Multi-layer models can compute history-dependent queries, and our theorem does not cover them (Section 8). The claim is narrower and, we think, useful: *the window size of an append-only store is the wrong quantity for judging its usefulness as a revisable memory.* The relevant quantity is a logit budget per revision.

## 2 Related Work

**Long-context evaluation.** Needle-in-a-haystack style tests are nearly saturated by modern models, but harder probes reveal large gaps. Lost-in-the-middle effects show position-dependent use of context [Liu et al., 2024]. RULER shows that most models claiming ≥32K windows degrade well before their nominal length on tasks beyond single retrieval [Hsieh et al., 2024]. BABILong finds that popular LLMs effectively use only 10–20% of their context on reasoning-in-a-haystack tasks [Kuratov et al., 2024]. None of these isolates *revision*, which is our focus.

**Memory benchmarks and the update problem.** LongMemEval [Wu et al., 2025] and LoCoMo [Maharana et al., 2024] evaluate long-term conversational memory, including knowledge updates and temporal reasoning. PI-LLM [Wang & Sun, 2025] adapts the proactive-interference paradigm from cognitive psychology and documents a log-linear decline in final-value retrieval as updates accumulate. Our Theorem 1 gives a mechanism-level account of why such a decline is *expected* from bounded-logit softmax readout. It also predicts its dependence on update count and context length (Section 7).

**Limits of softmax attention.** Softmax cannot stay sharp as the number of items grows with bounded logits. Veličković et al. [2024] formalize this "dispersion". Nakanishi [2025] calls it attention fading and proposes Scalable-Softmax, which multiplies logits by s·log n. Barbero et al. [2024] study related over-squashing. Liu et al. [2023] show that Transformers make sporadic "attention glitches" on flip-flop languages, where the model must recall the latest write. That task is a single-register special case of ours. Our contribution is the *interaction* between dispersion and recency. Without revisions, a bounded-logit head is limited only by log T. With revisions, every version boundary consumes L_ε nats of logit range, and the budget is shared with retaining old facts.

**Position encodings.** ALiBi [Press et al., 2022] adds a linear recency penalty; RoPE [Su et al., 2021/2024] encodes relative position by rotation. Both fall under Theorem 1, whose score function may depend arbitrarily on age.

**Memory architectures.** Fast weights [Ba et al., 2016], linear Transformers as fast-weight programmers with the delta rule [Schlag et al., 2021], DeltaNet [Yang et al., 2024] and Gated DeltaNet [Yang et al., 2025] keep a fixed-size state that is overwritten rather than appended. Infini-attention [Munkhdalai et al., 2024] adds a compressive memory to local attention. Titans [Behrouz et al., 2025] frames attention as short-term memory and a test-time-trained neural module as long-term memory. Memorizing Transformers [Wu et al., 2022] add a kNN-retrieved external cache, and RAG [Lewis et al., 2020] and MemGPT [Packer et al., 2023] manage memory outside the model. Modern Hopfield networks connect attention to associative memory with exponential capacity [Ramsauer et al., 2021]. Zoology [Arora et al., 2024] and "Repeat After Me" [Jelassi et al., 2024] show that attention beats fixed-state models at in-context *recall*. Our results are complementary. On revision-consistent recall at long horizons the ordering flips, and the failure axes of the two families are orthogonal. This mirrors the complementary-learning-systems argument from neuroscience [McClelland et al., 1995]: fast episodic storage and consolidated updatable knowledge call for different mechanisms.

## 3 Problem Setup

**Streams.** A *history* is a sequence of T memory tokens x_1,…,x_T followed by a query token x_{T+1} = query(κ) for a key κ. Each memory token is a write x_t = (k_t, v_t) of a key and a value. The *age* of position t is a_t = T − t ∈ {0,…,T−1}. Let t*(h) be the position of the most recent write of κ. The correct answer is v_{t*}.

**Readout heads.** A softmax readout head assigns each memory position a logit s_t and weight α_t = e^{s_t} / Σ_{u=1}^{T} e^{s_u}. We allow the most general form compatible with a fixed query:

> s_t = S(q, z_t, a_t), with z_t = f_t(x_1,…,x_t) an arbitrary *causal* representation of the prefix, and q = g(x_{T+1}) depending only on the query token.

This covers content-based scores, any positional scheme (absolute, relative, rotary, linear biases, learned timestamps), and key representations produced by arbitrarily deep causal networks. For example, z_t may encode "this is the 3rd write of key k" or the absolute time. The restrictive assumption is **(A1) history-independent query**. It holds exactly for single-layer attention and for the first attention layer of any network. It fails for later layers, whose queries can depend on the history (Section 8). We further assume **(A2) bounded logits**: |s_t| ≤ B. Since softmax is shift-invariant, only the range matters: 2B is the maximal logit spread the head can produce. A2 holds for any finite-precision network with bounded activations, and it is the quantity that Scalable-Softmax-type methods increase with T.

**Faithful retrieval.** The head *ε-faithfully retrieves* on h if α_{t*(h)} ≥ 1 − ε. (For orthonormal value codes and a linear decoder, ε < 1/2 is sufficient for correct decoding. Our bounds use the weight criterion, which is the natural "the head knows which token is current" condition.) Write L_ε = log((1−ε)/ε).

**The revision family.** For integers T, A ≥ 0 and Δ ≥ 1, let 𝓗(T, A, Δ) be the set of histories with T memory tokens in which (i) the latest write of κ has age ≤ A and (ii) consecutive writes of κ are separated by at least Δ positions. All other positions hold writes of other keys. Small Δ means a "hot", frequently revised fact. A single write at large age is a "cold" fact that must be retained.

## 4 Theory

**Theorem 1 (recency–retention bound).** Let a readout head satisfy (A1)–(A2). If it ε-faithfully retrieves (ε < 1/2) on every history in 𝓗(T, A, Δ), then

  (⌊A/Δ⌋ + 1) · L_ε ≤ 2B − log(T − 1).

*Proof.* Let R = ⌊A/Δ⌋ + 1 and positions p_j = T − A + (j−1)Δ for j = 1,…,R. Their ages A − (j−1)Δ are all in [0, A], and consecutive positions are Δ apart. Fix a filler sequence: every position not in {p_j} holds a fixed write of some key ≠ κ. Fix values u_1,…,u_R. Define history H_j (j = 1,…,R) as the filler with κ written (value u_i) at p_1,…,p_j and with filler writes at p_{j+1},…,p_R. Then H_j ∈ 𝓗(T, A, Δ), and its latest write of κ is at p_j.

*Causal consistency.* For i ≤ min(j, j′), histories H_j and H_{j′} agree on positions 1,…,p_i. Hence z_{p_i} is identical in both. The query token and the ages are also identical (A1, same T). So σ_i := S(q, z_{p_i}, T − p_i) does not depend on which H_j (j ≥ i) we are in.

*Chain.* In H_j with j ≥ 2, faithfulness gives α_{p_j} ≥ 1−ε, so α_{p_{j−1}} ≤ ε. Then σ_j − σ_{j−1} = log(α_{p_j}/α_{p_{j−1}}) ≥ L_ε. Summing, σ_R ≥ σ_1 + (R−1)L_ε, and by (A2) σ_1 ≤ B − (R−1)L_ε.

*Retention.* In H_1, κ is written once, at age A. Each of the other T−1 positions has logit ≥ −B, so

  1 − ε ≤ α_{p_1} ≤ e^{σ_1} / (e^{σ_1} + (T−1)e^{−B}),

which gives σ_1 ≥ log(T−1) − B + L_ε. Combining both inequalities: log(T−1) − B + L_ε ≤ B − (R−1)L_ε, that is, R·L_ε ≤ 2B − log(T−1). ∎

**Reading the bound.** Define the *revision-robust horizon* A*(Δ) as the largest A for which faithful retrieval on 𝓗(T, A, Δ) is possible. Theorem 1 gives

  A*(Δ) < Δ · ⌊(2B − log(T−1)) / L_ε⌋.

Three consequences follow.

* **Window size is not the horizon.** A* does not grow with T; it *shrinks* by Δ·log(T)/L_ε. A model with a 10⁶-token window and the same logit budget as a model with a 10³-token window has a *shorter* revision-robust horizon.
* **Every revision costs L_ε nats.** Equivalently, a fact revised R times can be read faithfully, while old unrevised facts stay retrievable, only if R ≤ (2B − log(T−1))/L_ε. Beyond this point the head *must* fail on some history in the family. It either loses the cold fact (H_1) or returns a stale version (some H_j). This is the recency–retention dilemma.
* **Without revisions** (R = 1) the bound reduces to 2B ≥ log(T−1) + L_ε, the attention-dilution/fading condition of Veličković et al. [2024] and Nakanishi [2025].

**Corollary 1 (log-scaled logits).** If logits are rescaled with context size so that the budget is B(T) = B₀ + (c/2)·log T, as in Scalable-Softmax, then R ≤ (2B₀ + (c−1)·log T + o(1))/L_ε. The number of trackable revisions grows at most logarithmically in context length. Long-context scaling of this kind buys revisions only logarithmically.

**Proposition 2 (near-tightness).** Suppose keys are orthogonal and the head has logit B − λ·a on writes of κ at age a (clipped below at −B) and −B on all other positions. Set λ = c/Δ with c > log(1 + 1/ρ), where ρ = ε/(1−ε). This head ε-faithfully retrieves on 𝓗(T, A, Δ) whenever

  A ≤ (Δ/c) · [2B − log(T−1) − log(1/(ρ − g(c)))],  with g(c) = e^{−c}/(1 − e^{−c}).

*Proof sketch.* In the worst case the latest write has age A. Older versions sit at gaps of at least Δ, so their total weight relative to the latest is at most Σ_{j≥1} e^{−cj} = g(c). Every remaining position, including clipped old versions, has logit −B. These contribute at most (T−1)e^{−2B+λA}. Faithfulness requires g(c) + (T−1)e^{−2B+λA} ≤ ρ. ∎

As ε → 0, letting c ↓ log(1+1/ρ) gives a horizon within the factor L_ε/log(1+1/ρ) of Theorem 1, up to the additive log term. This factor is 0.98 at ε = 0.05. The construction therefore shows that the *form* of the bound, linear in Δ and in 2B − log T, is right.

**Proposition 3 (overwrite memory escapes the bound; capacity limits it instead).** Consider the delta-rule state S_t = S_{t−1} + β(v_t − S_{t−1}k_t)k_tᵀ with β = 1 and unit keys. If the M distinct keys are orthonormal (so M ≤ d), then S_T k_κ = v_{t*} for every history, every number of revisions and every age.

*Proof.* Writing (κ, v) sets S k_κ ← S k_κ + (v − S k_κ)(k_κᵀk_κ) = v. Writing any other key k′ ⊥ k_κ changes S k_κ by (·)(k′ᵀk_κ) = 0. ∎

With M > d, the keys cannot be orthogonal, and overwrites of one key perturb the others. The delta rule then trades the revision axis for a *capacity* axis, which we measure in Fig. 1c. The two memory types are thus limited by different resources. Append-only context is limited by logit range per revision (horizon ∝ ΔB). Overwrite memory is limited by state dimension (capacity ∝ d), independent of horizon. This is a formal version of the intuition that "attention is short-term memory and weight-like state is long-term memory" [Behrouz et al., 2025]. Here the separating quantity is explicit.

## 5 Experiments

All experiments run on a 4-core CPU. Code is in `code/` and raw outputs are in `results/`. Every number below comes from these files.

### 5.1 E1: bound, construction and frontier (numerical; `code/theory_sim.py`, `code/make_fig1.py`)

**Bound vs construction.** For ε = 0.05 (L_ε = 2.944), Δ = 16, T ∈ {10³, 10⁵, 10⁷} and B ∈ {4,…,20}, we compute the Theorem-1 bound on A/Δ and the largest A achieved by the Proposition-2 family. For each λ on a 400-point log grid, we evaluate the exact worst-case weight on the latest version and binary-search the largest feasible A; we then maximize over λ. [TABLE E1]

**Recency–retention frontier.** With T = 4096, B = 10 and orthogonal keys, we sweep λ and measure two quantities over random histories (1,500 per point):

* *hot*: the fact of interest is revised about every 32 positions (spacing jittered by ±8), with its latest version at a uniformly random age < 32;
* *cold*: the fact is written once, at a uniformly random age in [0, T).

Success is α_latest ≥ 0.5.

**Delta-rule capacity.** We use random unit keys with d = 64, M ∈ {8,…,256} distinct keys, R ∈ {1, 8, 32} revisions per key in random order, 64 random unit value codes, and nearest-code decoding. We average over 10–20 trials.

### 5.2 E2: trained models (`code/train_toy.py`, `code/analyze_toy.py`)

**Task.** We use K = 16 keys and V = 16 values. Each write is a single composite token (k, v), so the 1-layer model needs no induction circuit to bind values to keys. A stream of N writes is followed by a query token for a key; the target is the value of that key's most recent write. Training streams have N ~ U{16,…,128}, keys drawn uniformly, and the query key drawn from the keys present. With K = 16, a key has about N/16 versions on average, so revisions are common in training.

**Models** (d = 64, 4 heads):

* causal pre-LN Transformers with RoPE and L ∈ {1, 2, 4} layers;
* a single-layer, 4-head delta-rule fast-weight model (learned q/k/v projections, L2-normalized keys, learned per-token β = sigmoid(·)) with an MLP readout.

All models are trained with AdamW (lr 10⁻³, one-cycle schedule, batch 128, 4,000 steps) for 3 seeds each.

**Evaluation** uses 500 streams per cell. We control the number of versions R of the queried key and its *latest age* a. Separately from this:

* *fixed-age sweeps*: R ∈ {1,…,64} with the latest version at age 0 or at age N/4, and earlier versions placed uniformly before it;
* *age sweeps*: R ∈ {1, 8} with latest age ∈ {0, 0.5N, 0.95N}.

We evaluate at N = 32 and 128 (within the training range) and at N = 512 (4× beyond it). For the Transformers, we also record the spread (max − min) of final-layer attention logits at the query position, per head, on in-distribution streams. This gives an empirical 2B for comparison with Theorem 1.

[E2 RESULTS]

## 6 Results

[RESULTS]

## 7 A Falsifiable Protocol for Frontier Models

Theorem 1 makes three predictions that separate a "logit-budget" mechanism from alternatives such as insufficient instruction following or semantic confusion. We pre-register them as a protocol for anyone with model access. We did not run it: there is no LLM access in our environment.

**P1 (per-revision cost).** At fixed context length and fixed age of the latest version, final-value accuracy should fall with the number of prior versions R and vanish beyond R* ≈ (2B_eff − log T)/L_ε. Plotting logit-of-correct-version margins against R should show a roughly *constant per-revision decrement* in the margin available to older facts.

**P2 (context length hurts even with fixed content).** Pad the same update stream with irrelevant, *non-conflicting* filler. Holding R and the latest age fixed, the revision-robust capacity should fall roughly as log T. This differs from a pure "lost-in-the-middle" account, where filler only matters through position.

**P3 (retention is traded for recency).** Interventions that sharpen recency, such as prompting to "use only the latest value" or fine-tuning on update tasks, should *reduce* accuracy on cold, never-revised facts at large ages in the same context, unless they also increase the effective logit range. A matched intervention that adds an explicit overwrite store (e.g., a key→latest-value scratchpad, or a delta-rule/Gated-DeltaNet layer) should improve both.

**Measurement.** Use PI-LLM-style streams [Wang & Sun, 2025] with: (i) R ∈ {1, 2, 4, …, 256}; (ii) latest age fixed; (iii) cold probe facts at controlled ages; (iv) filler padding at T ∈ {4K, 32K, 128K, 1M}. For open-weight models, also record the attention-logit spread at retrieval heads, which identifies B_eff. The bound is falsified for a model class if accuracy on both hot and cold probes stays high for R·L_ε well beyond the measured logit spread minus log T at the identified retrieval heads.

## 8 Discussion

**What "long context = long-term memory" gets wrong.** A context window guarantees *availability*: the tokens are there. Memory also needs *currency* (reads reflect updates) and *retention* (old facts survive). Under an append-only store, currency must be computed at read time from scores. Theorem 1 shows that, for a single fixed-query softmax read, this computation draws on the same finite logit budget that retention needs, and that each version boundary consumes L_ε nats. Window size enters only through the log(T−1) dilution term, and that term works *against* the memory. So the answer to the title question, in the precise sense formalized here, is **no**. A long context is a large store whose revision-consistent horizon is set by Δ·B/L_ε, not by T.

**Why deeper models can partially escape, and what that costs.** Our theorem relies on a history-independent query. A two-layer model can, in principle, compute at the query position a representation that already reflects the history. For example, a first layer can locate the latest write and pass its position or content to the second layer's query. But this first-layer lookup is itself a fixed-query read, so it is subject to Theorem 1 *for its own task*. We conjecture that depth multiplies the budget rather than removing it: roughly, an L-layer stack has budget L·2B. [E2 DISCUSSION OF DEPTH]

**Design implications.**

* *Memory systems should expose overwrite semantics explicitly.* Delta-rule or gated-delta layers [Yang et al., 2024; 2025], key→latest-value stores in agent scaffolds (as in MemGPT-style tiered memory [Packer et al., 2023]), or "update-aware" indexing in RAG (as in LongMemEval's time-aware designs [Wu et al., 2025]) all move the version decision from read time to write time.
* *Evaluate memory with revision structure.* NIAH-style single-version tests measure only the R = 1 corner of the bound. Evaluations should report accuracy as a function of R, latest age and T jointly.
* *Hybrid architectures are not just an efficiency trick.* Attention and overwrite memory fail along orthogonal axes (Prop. 3, Fig. 1c). Hybrids therefore combine complementary *semantics*, not only complementary costs. This matches the complementary-learning-systems view [McClelland et al., 1995].

## 9 Limitations

* **Single-head, fixed-query theorem.** Theorem 1 is proved for one softmax readout with a history-independent query. Multi-head mixtures, multi-layer models with history-dependent queries, and non-softmax readouts (sparsemax, hard attention, top-k retrieval) are not covered. Our depth results are empirical and only on toy models.
* **The bounded-logit assumption is a modeling choice.** Real models can grow logit spreads through large query/key norms. The bound is informative only when B_eff is measured. Our measured spreads come from tiny models and say nothing quantitative about frontier LLMs.
* **ε-faithful weight criterion.** We bound the attention weight on the correct token, not end-task accuracy. A downstream MLP can sometimes decode correctly from a mixture, for example when stale values happen to coincide or when the mixture is dominated by one wrong value. Our trained-model accuracy measures the latter.
* **Toy task.** Single-token key–value writes with 16 keys and 16 values are far from natural conversational updates. Semantic similarity between versions, paraphrase, implicit updates and temporal reasoning are absent. The trained models are tiny (d = 64, ≤ 4 layers), trained briefly on CPU with 3 seeds, and their length generalization also depends on RoPE extrapolation. Some N = 512 effects may be positional out-of-distribution effects rather than the mechanism studied here.
* **No frontier-model experiments.** The protocol in Section 7 is pre-registered but unexecuted.
* **Delta-rule results assume idealized writes.** Proposition 3 assumes orthonormal keys and β = 1. Learned delta models need not discover such keys, and Fig. 1c uses random keys only.

## 10 Conclusion

Long context and long-term memory differ in their update semantics. We proved that a fixed-query softmax readout with logit budget B can keep facts both current and retained only up to a horizon of order Δ·(2B − log T)/log((1−ε)/ε). This horizon is independent of window size and degrades with context length. A near-matching construction shows the bound has the right form, and an overwrite (delta-rule) memory removes it at the price of a dimension-limited capacity. [E2 conclusion sentence] Taken together, the results suggest that progress on memory should be measured along revisions, not tokens.

## References

- Arora, S., Eyuboglu, S., et al. (2024). Zoology: Measuring and Improving Recall in Efficient Language Models. ICLR 2024. arXiv:2312.04927.
- Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., Ionescu, C. (2016). Using Fast Weights to Attend to the Recent Past. NeurIPS 29. arXiv:1610.06258.
- Barbero, F., Banino, A., Kapturowski, S., Kumaran, D., Araújo, J. G. M., Vitvitskyi, A., Pascanu, R., Veličković, P. (2024). Transformers Need Glasses! Information Over-squashing in Language Tasks. NeurIPS 2024. arXiv:2406.04267.
- Behrouz, A., Zhong, P., Mirrokni, V. (2025). Titans: Learning to Memorize at Test Time. arXiv:2501.00663.
- Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., Ginsburg, B. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? COLM 2024. arXiv:2404.06654.
- Jelassi, S., Brandfonbrener, D., Kakade, S. M., Malach, E. (2024). Repeat After Me: Transformers are Better than State Space Models at Copying. ICML 2024. arXiv:2402.01032.
- Kuratov, Y., Bulatov, A., Anokhin, P., Rodkin, I., Sorokin, D., Sorokin, A., Burtsev, M. (2024). BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack. NeurIPS 2024 Datasets and Benchmarks. arXiv:2406.10149.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 33.
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., Zhang, C. (2023). Exposing Attention Glitches with Flip-Flop Language Modeling. NeurIPS 2023. arXiv:2306.00946.
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the ACL 12. doi:10.1162/tacl_a_00638.
- Maharana, A., et al. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL 2024. arXiv:2402.17753.
- McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory. Psychological Review 102(3), 419–457.
- Munkhdalai, T., Faruqui, M., Gopal, S. (2024). Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention. arXiv:2404.07143.
- Nakanishi, K. M. (2025). Scalable-Softmax Is Superior for Attention. arXiv:2501.19399.
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
- Press, O., Smith, N. A., Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR 2022.
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., Holzleitner, M., Adler, T., Kreil, D., Kopp, M. K., Klambauer, G., Brandstetter, J., Hochreiter, S. (2021). Hopfield Networks is All You Need. ICLR 2021.
- Schlag, I., Irie, K., Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021, PMLR 139. arXiv:2102.11174.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864 (journal version: Neurocomputing, 2024).
- Veličković, P., Perivolaropoulos, C., Barbero, F., Pascanu, R. (2024). Softmax is not Enough (for Sharp Out-of-Distribution / Size Generalisation). arXiv:2410.01104.
- Wang, C., Sun, J. V. (2025). Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length. arXiv:2506.08184.
- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., Yu, D. (2025). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR 2025. arXiv:2410.10813.
- Wu, Y., Rabe, M. N., Hutchins, D., Szegedy, C. (2022). Memorizing Transformers. ICLR 2022. arXiv:2203.08913.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. arXiv:2406.06484.
- Yang, S., Kautz, J., Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
