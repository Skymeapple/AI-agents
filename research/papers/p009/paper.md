# How Much State Does In-Context Recall Need? A Superposition-Capacity Law for Fixed-State Sequence Models, Checked on Trained Mixers and Calibrated on Real Text

**Target venue:** International Conference on Learning Representations (ICLR 2027), main track

---

## Abstract

Whether state-space models (SSMs), linear-attention RNNs and their hybrids can replace the Transformer depends mostly on one capability: recalling arbitrary information from the context. Attention keeps a key–value (KV) cache that grows with the context. Fixed-state models have to compress the context into a state of constant size. Earlier work showed that this costs something: fixed-state models fail at copying and associative recall, and adding even a few attention layers repairs them. It did not say how much state a given recall workload needs, or how much recall real text actually requires. We work out that budget quantitatively. (i) For the additive outer-product memory that underlies linear attention and SSM-style recurrences, we derive a closed-form approximation to recall accuracy as a function of the number of stored associations *N*, key dimension *d_k*, value dimension *d_v* and output vocabulary *V*. It gives a capacity law, N\* ≈ d_k / (2 ln V · (1/d_v + 1/V)), which is about S/(2 ln V) for a state of S = d_k d_v floats, together with a lag-resolved version for decaying (gated) memories. In simulation the law predicts the 50%-accuracy capacity to within 3–11% for d_v ∈ {64, 128} (larger errors at d_v = 16 and 256). (ii) An information-theoretic bound shows that this superposition memory uses only about 2% of the bits a 16-bit state could hold, so the bound itself is far from binding. (iii) We train single-layer mixers of five families (softmax attention, linear attention, scalar-gated linear attention, DeltaNet, Gated DeltaNet) on a multi-query associative-recall task built so that the memory is the only bottleneck. Every fixed-state family reaches the same capacity: 1.01–1.13× the random-code prediction (1.04–1.07× a direct simulation of an ideal additive memory), with N\* growing linearly in d_k, and in d_v once the input carries enough value dimensions. A control shows that nominal state size can overstate effective state: a 64-dimensional value embedding caps a d_v = 128 state at the d_v = 64 capacity. Trained gates and delta-rule write strengths settle in a regime that behaves additively. The ideal delta rule with unit write strength is about 3× worse. (iv) We measure the recall demand of real text: the number of distinct context-specific bigram associations that recur within a window of length T, on 2.1M words of public-domain prose and 11.4M tokens of Go source code. Demand grows close to linearly in T. At T = 32,768, 30% (prose) and 48% (code) of tokens are context-only recall opportunities. (v) Putting the pieces together, a fixed-state layer with S floats saturates at a context length T\* that lies between ηS/(2 ln V) (write every token) and an oracle-selective bound about 10× larger. Holding the oracle-selected associations of a 32k-token window takes about 6–7×10^4 floats, roughly 4,000× less than one attention layer's KV cache at d = 4096. Our reading is that fixed versus growing memory, not attention versus recurrence, is what decides whether recall-complete replacement is possible. Selective writing is where most of the slack lies. We list falsifiable predictions and a pre-registered protocol for testing them on pretrained models.

## 中文摘要

状态空间模型（SSM）、线性注意力RNN及其混合架构能否取代Transformer，关键在于从上下文中召回任意信息的能力：注意力保留随上下文增长的键值（KV）缓存，而定长状态模型必须把上下文压缩进常数大小的状态。已有工作表明定长状态模型在复制和联想召回上失败，且少量注意力层即可修复，但并未定量回答"一个召回负载究竟需要多少状态"以及"真实文本究竟需要多少召回"。本文给出这一预算的定量分析。(i) 针对线性注意力与SSM类递归所共有的加性外积记忆，我们推导了召回准确率关于存储关联数N、键维d_k、值维d_v与输出词表V的闭式近似，得到容量律 N\* ≈ d_k/(2 ln V·(1/d_v+1/V))，对于S=d_k d_v个浮点数的状态约为S/(2 ln V)，并给出带衰减（门控）记忆的按滞后分解版本；在仿真中，当d_v∈{64,128}时其对50%准确率容量的预测误差为3–11%（d_v=16与256时误差更大）。(ii) 信息论上界表明，这种叠加式记忆仅使用了16位状态可容纳比特数的约2%，因此该上界远未成为约束。(iii) 我们在一个使记忆成为唯一瓶颈的多查询联想召回任务上训练了五类单层混合器（softmax注意力、线性注意力、标量门控线性注意力、DeltaNet、Gated DeltaNet）：所有定长状态模型都达到相同容量，为随机编码理论值的1.01–1.13倍（为理想加性记忆直接仿真值的1.04–1.07倍），N\*随d_k线性增长，且在输入提供足够值维度时随d_v增长；对照实验表明名义状态大小可能高估有效状态：64维的值嵌入会把d_v=128的状态容量限制在d_v=64的水平；训练得到的门控与delta写入强度停留在表现为加性的区间，而写入强度为1的理想delta规则约差3倍。(iv) 我们在210万词公版英文散文与1140万词元Go源代码上测量真实文本的召回需求，即窗口T内重复出现的、依赖上下文的不同二元关联数：需求随T近似线性增长，在T=32768时，30%（散文）与48%（代码）的词元是"仅能从上下文获得"的召回机会。(v) 综合上述结果，含S个浮点数的定长状态层在上下文长度T\*处饱和，T\*介于ηS/(2 ln V)（每个词元都写入）与约大10倍的"理想选择性写入"上界之间；容纳32k窗口中经理想选择的关联约需6–7×10^4个浮点数，约比d=4096时单个注意力层的KV缓存小4000倍。我们据此认为，决定能否实现"召回完备"替代的是定长记忆与可增长记忆之分，而非注意力与递归之分；大部分余量在于选择性写入。文末给出可证伪的预测与面向预训练模型的预注册检验方案。

---

## 1 Introduction

The Transformer [Vaswani et al., 2017] answers every query by attending over a KV cache whose size grows linearly with context length. A family of alternatives — structured and selective state-space models [Gu et al., 2022; Gu & Dao, 2023; Dao & Gu, 2024], linear attention and its gated and delta-rule variants [Katharopoulos et al., 2020; Schlag et al., 2021; Yang et al., 2024a; 2024b; 2025], and hybrids that interleave these with a few attention layers [Lieber et al., 2024; Waleffe et al., 2024; MiniMax, 2025; NVIDIA, 2025; Kimi Team, 2025] — replace the growing cache with a fixed-size recurrent state, trading recall for constant-memory decoding. The question in our title has a well-established qualitative answer: fixed-state models are provably limited on copying and associative recall [Jelassi et al., 2024; Wen et al., 2025; Arora et al., 2024a], they lag Transformers on recall-intensive tasks [Arora et al., 2024a; Waleffe et al., 2024], and a small fraction of attention layers restores recall [Wen et al., 2025; Waleffe et al., 2024; Wang et al., 2025]. Production hybrids use between ~8% (Nemotron-H) and 25% (Kimi Linear) attention layers, and a systematic study of 72 hybrid models recommends linear-to-full ratios between 3:1 and 6:1 [Wang et al., 2025].

What is missing is a *quantitative* account that connects three numbers: how many associations a fixed state of a given size can hold, how close trained models get to that, and how many associations real text actually asks a model to hold. Without it, statements like "SSMs cannot replace Transformers" or "a 3:1 hybrid suffices" cannot be extrapolated to other state sizes, context lengths, or data domains. This paper supplies such an account for one well-defined capability, single-hop associative recall, and uses it to sharpen the replacement question into a falsifiable one.

**Contributions.**

1. **A capacity law for additive superposition memory (Section 3).** For the outer-product state *M = Σ_j u_{v_j} k_jᵀ* shared by linear attention and (diagonal-free) SSM recurrences, with random unit keys and value codes, we derive a one-dimensional-integral approximation to recall accuracy and the rule of thumb *N\* ≈ d_k/(2 ln V (1/d_v + 1/V))* ≈ *S/(2 ln V)*. We extend it to decayed memories (lag-resolved accuracy). Simulation confirms the law within 3–11% for d_v ∈ {64,128} (Section 5.1).
2. **An information-theoretic efficiency gap (Proposition 2).** Any memory that stores N uniformly random values in S coordinates of b bits must satisfy N ≤ Sb/((1−ε)log₂V − 1). Additive superposition memory operates at ≈2% of this bound at b = 16, so fixed-state recall is not limited by information but by the linear read-out geometry.
3. **Trained models hit the law, regardless of update rule (Section 5.2).** On a capacity-isolating multi-query associative recall (MQAR) task, single-layer linear attention, gated linear attention, DeltaNet and Gated DeltaNet all reach 1.01–1.13× the random-code capacity, with N\* ∝ d_k (log–log slopes 1.04–1.07). Learned delta-rule write strengths settle at β ≈ 0.03–0.14 and learned decays at ≈ 0.9999, i.e. in an effectively additive regime. A width control shows that capacity also scales with d_v, but only up to the rank of the value information the layer receives, so *nominal* state size can overstate *effective* state size. The delta rule's well-known exactness for N ≤ d_k does not translate into more capacity at N ≫ d_k; the ideal unit-strength delta rule is ≈3× *worse* than additive storage.
4. **A measurement of recall demand in real text (Section 5.3).** We define the context-specific association count N_ctx(T) and the recall-opportunity rate ρ(T), and measure them on public-domain prose and on the Go standard library. Demand grows nearly linearly in T (local exponents 0.84–1.06 at T = 8k–32k), and ρ(32k) = 0.30 (prose), 0.48 (code).
5. **A memory budget and falsifiable predictions (Sections 5.4, 6).** Combining 1–4 yields bounds on the critical context length T\*(S) of a fixed-state layer under three write policies and the state S_req(T) needed to hold real-text demand, which is three to four orders of magnitude smaller than a KV cache. We state predictions P1–P5 with a pre-registered protocol for pretrained models.

The contribution is deliberately narrow. We do not claim that single-hop bigram recall is all that matters, and we do not run experiments on large pretrained models (Section 8).

## 2 Related Work

**Fixed-state sequence models.** S4 [Gu et al., 2022] made linear state-space layers practical; Mamba [Gu & Dao, 2023] added input-dependent ("selective") parameters; Mamba-2 [Dao & Gu, 2024] showed that SSMs with scalar-times-identity transitions are a form of gated linear attention (state-space duality). Linear attention [Katharopoulos et al., 2020] replaces the softmax kernel with a feature map so that the KV cache collapses to a d_k × d_v state; gated linear attention (GLA) adds data-dependent decay [Yang et al., 2024a]; DeltaNet [Schlag et al., 2021; Yang et al., 2024b] replaces the additive update with a delta rule; Gated DeltaNet combines both [Yang et al., 2025]. Log-linear attention [Guo et al., 2026] and Hybrid Associative Memories [Lufkin et al., 2026] let the memory grow sublinearly or selectively with context — exactly the regime our analysis singles out.

**Recall limits of fixed-state models.** Zoology [Arora et al., 2024b] introduced MQAR and showed that most of the quality gap between gated-convolution models and attention is explained by associative recall. Based [Arora et al., 2024a] made the state-size/recall trade-off explicit and proved lower bounds for recurrent models on recall. Jelassi et al. [2024] proved that a fixed-state model cannot copy strings longer than its state permits, while two-layer Transformers can copy exponentially long strings. Wen et al. [2025] proved that RNNs, even with chain-of-thought, cannot solve associative recall, and that one Transformer layer suffices. Merrill et al. [2024] showed that SSMs do not escape the TC⁰ limit on state tracking, a separate capability we do not study. Schlag et al. [2021] already observed that the additive fast-weight memory of a linear Transformer has a capacity limit tied to d_k, motivating the delta rule. Qiao [2025] argues, via a conditioning-theory lens, that a linear-attention layer that must *reconstruct* a value is limited to Θ(d_k) items whereas *selecting* from a known vocabulary allows much larger capacity; our Proposition 1 makes the selection case quantitative, including its dependence on d_v and log V.

**Hybrid design.** Jamba [Lieber et al., 2024], the NVIDIA 8B Mamba-2-Hybrid [Waleffe et al., 2024], Nemotron-H (~8% attention layers) [NVIDIA, 2025], MiniMax-01 (one softmax layer per seven lightning-attention layers) [MiniMax, 2025] and Kimi Linear (3:1) [Kimi Team, 2025] are production-scale hybrids. Wang et al. [2025] trained 72 hybrids and found that recall improves with the fraction of full-attention layers, particularly below a 3:1 linear-to-full ratio, while language-modelling loss is flat. These are empirical ratios; our goal is a model that predicts *when* a fixed state saturates.

**Associative-memory capacity.** Correlation-matrix memories [Kohonen, 1972] and Hopfield networks have classical capacity analyses — e.g. α_c ≈ 0.138 patterns per neuron for the Hopfield model [Amit et al., 1985]. Our Proposition 1 is a variant of the classical signal-to-crosstalk argument, specialised to the hetero-associative, argmax-over-vocabulary read-out used by language-model heads; the Welch bound [Welch, 1974] explains why learned keys cannot beat random keys by much when the key vocabulary greatly exceeds d_k. Induction heads [Olsson et al., 2022] motivate our choice of bigram associations as the unit of real-text recall demand. Heaps' law [Heaps, 1978] describes sublinear vocabulary growth; we measure the analogous growth of *recurring context-specific associations*.

## 3 Theory

### 3.1 Setting

A memory reads a stream of (key, value) pairs and must later answer queries "what value was paired with key k?". Keys are vectors k_j ∈ ℝ^{d_k} with ‖k_j‖ = 1. Each value is a symbol v_j ∈ [V], represented by a code u_{v} ∈ ℝ^{d_v}, ‖u_v‖ = 1. Read-out is a linear retrieval followed by an argmax over the V codes — the structure of a language-model head applied to the output of a linear-attention or SSM layer. We consider three update rules for the state M ∈ ℝ^{d_v × d_k}:

- additive (linear attention, fast weights): M_t = M_{t−1} + u_{v_t} k_tᵀ;
- decayed (GLA / Mamba-2-style scalar gate γ_t ∈ (0,1]): M_t = γ_t M_{t−1} + u_{v_t} k_tᵀ;
- delta rule with write strength β_t: M_t = M_{t−1} + β_t (u_{v_t} − M_{t−1}k_t) k_tᵀ (and its gated version).

A query with key k_i returns r = M k_i and predicts argmax_w ⟨u_w, r⟩. The state has S = d_k d_v floats.

### 3.2 Proposition 1 (capacity of additive superposition memory)

Let keys be i.i.d. uniform on the unit sphere in ℝ^{d_k}, codes i.i.d. uniform on the unit sphere in ℝ^{d_v}, and values i.i.d. uniform on [V]. Let the target item carry signal weight s (s = 1 for additive memory, s = γ^τ for an item written τ steps ago under constant decay γ), and let G = Σ_{j≠i} w_j² be the summed squared weights of the other stored items (G = N − 1 for additive memory). Write c_j = ⟨k_j, k_i⟩, which has mean 0 and variance 1/d_k. Then

- correct score: ⟨u_{v_i}, r⟩ = s + Σ_{j≠i} w_j c_j ⟨u_{v_j}, u_{v_i}⟩, with mean s and variance σ_c² ≈ G (1/(d_k d_v) + 1/(V d_k));
- wrong score for w ≠ v_i: ⟨u_w, r⟩ = s⟨u_w, u_{v_i}⟩ + Σ_j w_j c_j ⟨u_w, u_{v_j}⟩, with mean 0 and variance σ_w² ≈ s²/d_v + G (1/(d_k d_v) + 1/(V d_k)).

The 1/(V d_k) term comes from the ≈G/V stored items that share the competitor's (or the target's) value symbol. In the Gaussian approximation, treating the V − 1 competitor scores as independent,

  Acc ≈ ∫ φ(z) Φ((s + σ_c z)/σ_w)^{V−1} dz.  (1)

Since the maximum of V − 1 Gaussians with standard deviation σ_w is ≈ σ_w √(2 ln V), retrieval of an additive-memory item (s = 1) succeeds roughly while σ_w √(2 ln V) < 1, which gives

  **N\* ≈ d_k / (2 ln V · (1/d_v + 1/V))**, i.e. N\* ≈ S/(2 ln V) when d_v ≪ V.  (2)

*Derivation.* The means and variances follow from E[c_j] = 0, E[c_j²] = 1/d_k, E⟨u_a,u_b⟩² = 1/d_v for a ≠ b, independence of keys from codes, and P(v_j = w) = 1/V. Equation (1) treats the scores as jointly Gaussian and independent. Equation (2) uses the leading-order extreme-value approximation and drops the s²/d_v term, which is small whenever d_v ≫ 2 ln V. Both are approximations, not bounds. The competitor scores are positively correlated through the shared noise, which makes (1) slightly pessimistic, and (2) underestimates N\* by about 20% because the median of the maximum is below σ√(2 ln V) (Section 5.1). ∎

Three consequences. (a) Capacity is linear in the state size and only logarithmic in the vocabulary. (b) Key dimension and value dimension are interchangeable to first order as long as d_v ≪ V. This is why a single head with small d_k can still hold many more than d_k items when it has a wide value path. Section 5.2 confirms this on trained models and adds a caveat: the value path must actually carry that many independent dimensions. (c) Learned keys cannot do much better than random ones when the key vocabulary K ≫ d_k. The Welch bound [Welch, 1974] gives max_{i≠j} |⟨k_i,k_j⟩| ≥ √((K − d_k)/(d_k(K − 1))) ≈ 1/√d_k, the same crosstalk scale that random keys achieve on average.

**Decay.** For constant decay γ and N writes, an item at lag τ has s = γ^τ and G = Σ_{j=0}^{N−1} γ^{2j} − γ^{2τ} ≤ 1/(1 − γ²). Decay bounds the interference but makes the signal shrink geometrically. Eq. (1) then predicts a sharp recall horizon τ\*: recall succeeds while γ^τ ≳ √(2 ln V) σ, so τ\* ≈ ln(1/(√(2 ln V) σ))/ln(1/γ). When every stored item may be queried, the best constant decay is γ = 1. Decay only pays when old items are worth less than new ones.

### 3.3 Proposition 2 (information bound and efficiency gap)

*Any* (not necessarily linear) memory whose state after reading N pairs with distinct keys consists of S coordinates of b bits each, and which answers a query on a uniformly random stored key correctly with probability at least 1 − ε when values are i.i.d. uniform on [V], satisfies

  N ((1 − ε) log₂ V − 1) ≤ S b.  (3)

*Proof.* Condition on the keys. The values carry N log₂ V bits of entropy, and the state Z has H(Z) ≤ Sb. Let ε_i be the error probability on key i, so that the average of the ε_i is ε. By Fano's inequality, H(v_i | Z, keys) ≤ 1 + ε_i log₂ V. Chain-rule subadditivity gives H(v_1..v_N | Z, keys) ≤ Σ_i (1 + ε_i log₂ V) = N(1 + ε log₂ V). Therefore I(values; Z | keys) ≥ N log₂ V − N(1 + ε log₂ V), and this mutual information is at most H(Z) ≤ Sb. ∎

This is the standard counting argument behind recall lower bounds for recurrent models [Arora et al., 2024a; Jelassi et al., 2024]. Its use here is in the comparison. At V = 512, ε = 0.5 and b = 16, (3) allows N ≤ 4.6 S. Additive superposition memory achieves N\* ≈ 0.09 S at d_v = 64 (Section 5.1), about 2% of the bound. Fixed-state recall in this regime is therefore limited by the linear read/write geometry and not by the information content of the state. Non-linear read-outs, sparse codes or error correction could in principle recover a large factor. Section 6 returns to this.

### 3.4 From capacity to a critical context length

Let D(T) be the number of associations a model must hold at context length T, and let η be the capacity of a trained memory relative to Eq. (1) (η = 1 means it matches random-code theory). A fixed-state layer with S floats and an effective vocabulary V can hold D(T) only while

  D(T) ≤ η S/(2 ln V),  which defines  T\*(S) = D⁻¹(η S / (2 ln V)).  (4)

D depends on the write policy:

- **every-token**: one association is written per token (D = T). This is the regime of an un-selective recurrence.
- **dedup**: one write per distinct association (D = N_bi(T), the number of distinct bigrams).
- **oracle**: only context-specific associations that will be needed again are written (D = N_ctx(T), defined in Section 4.3). No causal model can do better than this without knowing the future.

Conversely, a memory that is to cover a context of length T needs S_req(T) = 2 ln V · D(T)/η floats. One softmax-attention layer stores 2dT floats.

## 4 Experimental Methods

All experiments run on CPU. Code and raw outputs are in `code/` and `results/`, and `README.md` gives the run commands. Every number below comes from these runs unless it is marked as cited.

### 4.1 Experiment A: ideal memories versus theory

We simulate additive, decayed and unit-strength delta-rule memories exactly as in Section 3.1, with V = 512 random codes, random unit keys, N ∈ {4, …, 1024}, d_k ∈ {8,16,32,64}, d_v ∈ {16,…,256}, and 20–40 trials per cell. N\* is the N at which accuracy crosses 0.5, interpolated log-linearly. Eq. (1) is evaluated by quadrature (`capacity_sim.py`).

### 4.2 Experiment B: trained single-layer mixers on capacity-isolating MQAR

The task is a variant of MQAR [Arora et al., 2024b] built to isolate the memory. A sequence holds N pair tokens with distinct keys drawn from a key vocabulary of K = 4096 (K ≫ d_k, so keys must live in superposition) and values drawn from V = 512, followed by Q = 32 query tokens whose keys are drawn from the stored set. A pair token is embedded as [E_k(key); E_v(value)] ∈ ℝ^{128}, and a query token as [E_k(key); 0]. Because pairs are pre-bound into single tokens, no token-shift or extra layer is needed to form the key–value binding. Writes are masked at query positions for all architectures, so the measured limit is storage capacity and not the model's ability to decide what to write.

All mixers use one layer, L2-normalised q and k projections of width d_k, and a value projection of width d_v. The read-out is LayerNorm followed by a linear map to V logits.

- **attn**: causal softmax attention over the pair positions, with a learned temperature. Its KV cache grows with N.
- **linear**: o_t = Σ_{s≤t} ⟨k_s, q_t⟩ v_s.
- **gla**: scalar data-dependent decay a_t = σ(w_aᵀx_t + b_a), forced to 1 at query positions.
- **delta**: DeltaNet with β_t = σ(w_bᵀx_t + b_b), computed in parallel by the triangular-solve (UT) form U = (I + diag(β) L)⁻¹ diag(β) V, with L the strictly lower-triangular key Gram matrix [Yang et al., 2024b].
- **gdelta**: Gated DeltaNet, the same form with decays folded into L and into the output mask.

Every fixed-state model has exactly d_k × d_v state per sequence. Training: AdamW with a one-cycle schedule peaking at 3·10⁻³, batch 64, 1,200 steps, N sampled uniformly from {4, 8, …, 256} for each batch, seed 0. Evaluation: 1,024 fresh sequences (32,768 queries) per N, seed 1234. We ran attn at d_k ∈ {8,16}; linear at d_k ∈ {8,16,32,64}; gla, delta and gdelta at d_k ∈ {8,16,32}; all at d_v = 64. We also ran a value-dimension sweep, linear and delta at d_k = 16 with d_v ∈ {32,128}. With the default token width d = 128 the value half of the embedding has only 64 dimensions, so v = W_v x carries value information of rank at most 64 whatever d_v is. We therefore repeated d_v = 128 (linear, delta) and d_v = 64 (linear, as a control) with d = 512, whose value half has 256 dimensions. We planned d_k = 64 for every family, but the machine was shared with unrelated jobs and we cut the gla, delta and gdelta runs at d_k = 64 (README). For delta and gdelta we also record the mean learned write strength β and decay a on pair tokens (`gates_on_pair_tokens` in the run JSON). The efficiency is η = N\*_trained / N\*_theory, where the theory value comes from Eq. (1) with the same (d_k, d_v, V). We also report η_sim = N\*_trained / N\*_sim against the simulated ideal additive memory of Experiment A, which removes the approximation error of Eq. (1). The efficiency is η = N\*_trained / N\*_theory, where the theory value comes from Eq. (1) with the same (d_k, d_v, V).

### 4.3 Experiment C: recall demand of real text

**Corpora.** *Prose*: the 18-book public-domain Project Gutenberg selection distributed with NLTK (2.14M lower-cased word tokens, vocabulary 41,509). *Code*: the Go 1.25.1 standard library, non-test .go files concatenated per package directory, keeping directories with at least 4,096 tokens (290 documents, 11.4M tokens, vocabulary 149,292). Tokens are identifiers/numbers and single punctuation characters.

**Measures.** For a random window of T tokens inside one document:

- N_bi(T) is the number of distinct bigrams.
- N_ctx(T) is the number of distinct bigrams (a, b) that occur at least twice in the window and whose successor b is not the corpus-wide most frequent successor of a. These are associations that recur, so storing them pays, and that cannot be guessed from parametric bigram statistics, so they must come from the context.
- ρ(T) is the fraction of positions whose bigram already occurred earlier in the window and is not the global top-1 continuation, i.e. the share of tokens on which an induction-style copy from context helps.

We use T ∈ {64, …, 32768}, with 60 windows per T up to 8,192 and 30 beyond (`recall_demand.py`). Windows of length T are drawn only from documents at least T tokens long. At T = 32,768 this leaves 8 of the 18 books (`n_docs_eligible` in `demand.json`) and 53 Go packages.

### 4.4 Experiment D: combining

`analyze.py` computes N\*, η and log–log slopes from Experiment B, and S_req(T) and T\*(S) from Eq. (4) for the three write policies, using the measured D(T). Here V is the corpus vocabulary, so ln V = 10.6 (prose) and 11.9 (code). For S beyond the measured range we extrapolate D(T) with the local exponent of the last three measured points, and we mark every such value as extrapolated.

## 5 Results

### 5.1 The capacity law matches ideal memories

Figure `results/fig_sim_capacity.png`(a) shows that Eq. (1) tracks simulated additive-memory accuracy across two orders of magnitude in N.

| d_k (d_v = 64) | N\* simulated | N\* Eq. (1) | N\* rule (2) | N\* ideal delta rule (β = 1) |
|---|---|---|---|---|
| 8  | 42.9 | 44.4 | 36.5 | 14.3 |
| 16 | 92.3 | 87.7 | 73.0 | 30.0 |
| 32 | 185.0 | 174.4 | 145.9 | 63.4 |
| 64 | 378.2 | 347.8 | 291.8 | 132.1 |

*Table 1 (source: `results/sim_capacity.json`).* For d_v = 64 the integral (1) is within 3.6–8.1% of simulation, and the rule (2) is a consistent ≈20% underestimate, as expected from the extreme-value approximation. Capacity is linear in d_k. The value-dimension sweep at d_k = 16 (panel b) confirms that capacity also grows with d_v: N\* = 22.7, 47.9, 89.8, 155.9, 229.6 for d_v = 16, 32, 64, 128, 256. The Gaussian integral is accurate at d_v = 64 and 128 (2.3% and 9.5% error). It underestimates at d_v ≤ 32 (by 18–45%), where the scores are far from Gaussian, and overestimates at d_v = 256 (by 28%), where d_v/V = 0.5 and competitor correlations matter. Using unit write strength, the ideal delta rule has about 3× *lower* capacity than additive storage at every d_k (≈ 2 d_k). With non-orthogonal keys, each write partially erases older items, so the delta rule's exactness for N ≤ d_k turns into recency-weighted forgetting once N > d_k.

For decayed memory (panel c; N = 256, d_k = 32), the lag-resolved form of Eq. (1) reproduces the sharp recall horizon: items within ≈15 (γ = 0.9), ≈40 (γ = 0.97) and ≈80 (γ = 0.99) writes are recalled, and older ones are lost. Mean accuracy over all 256 items is 0.06, 0.15, 0.27 and 0.33 for γ = 0.9, 0.97, 0.99 and 1.0, so in a retain-everything workload no decay is best, as predicted.

### 5.2 Trained mixers reach the law, whatever the update rule

| mixer | emb. d | d_k | d_v | S | N\* trained | N\* Eq. (1) | N\* simulated | η | η_sim |
|---|---|---|---|---|---|---|---|---|---|
| attn | 128 | 8 | 64 | (grows) | >256 (acc@256 = 1.00) | – | – | – | – |
| attn | 128 | 16 | 64 | (grows) | >256 (acc@256 = 1.00) | – | – | – | – |
| linear | 128 | 8 | 64 | 512 | 45.4 | 44.4 | 42.9 | 1.02 | 1.06 |
| linear | 128 | 16 | 64 | 1024 | 95.6 | 87.7 | 91.0 | 1.09 | 1.05 |
| linear | 128 | 32 | 64 | 2048 | 195.9 | 174.4 | 185.0 | 1.12 | 1.06 |
| linear | 128 | 64 | 64 | 4096 | >256 (acc@256 = 0.72) | 339.1 | 378.2 | – | – |
| gla | 128 | 8 | 64 | 512 | 45.4 | 44.4 | 42.9 | 1.02 | 1.06 |
| gla | 128 | 16 | 64 | 1024 | 95.2 | 87.7 | 91.0 | 1.09 | 1.05 |
| gla | 128 | 32 | 64 | 2048 | 194.0 | 174.4 | 185.0 | 1.11 | 1.05 |
| delta | 128 | 8 | 64 | 512 | 45.3 | 44.4 | 42.9 | 1.02 | 1.06 |
| delta | 128 | 16 | 64 | 1024 | 96.0 | 87.7 | 91.0 | 1.09 | 1.05 |
| delta | 128 | 32 | 64 | 2048 | 191.7 | 174.4 | 185.0 | 1.10 | 1.04 |
| gdelta | 128 | 8 | 64 | 512 | 45.0 | 44.4 | 42.9 | 1.01 | 1.05 |
| gdelta | 128 | 16 | 64 | 1024 | 95.7 | 87.7 | 91.0 | 1.09 | 1.05 |
| gdelta | 128 | 32 | 64 | 2048 | 197.6 | 174.4 | 185.0 | 1.13 | 1.07 |
| linear | 128 | 16 | 32 | 512 | 47.5 | 39.1 | 47.9 | 1.22 | 0.99 |
| delta | 128 | 16 | 32 | 512 | 50.5 | 39.1 | 47.9 | 1.29 | 1.05 |
| linear | 128 | 16 | 128 | 2048 | 97.0 | 170.8 | 155.9 | 0.57 | 0.62 |
| delta | 128 | 16 | 128 | 2048 | 98.5 | 170.8 | 155.9 | 0.58 | 0.63 |
| linear | 512 | 16 | 64 | 1024 | 96.2 | 87.7 | 91.0 | 1.10 | 1.06 |
| linear | 512 | 16 | 128 | 2048 | 177.2 | 170.8 | 155.9 | 1.04 | 1.14 |
| delta | 512 | 16 | 128 | 2048 | 177.5 | 170.8 | 155.9 | 1.04 | 1.14 |

*Table 2 (sources: `results/capacity_table.json`, `results/eta_vs_sim.json`; one seed per row).* N\* is the 50%-accuracy capacity interpolated log-linearly between the evaluated N ∈ {4, 8, …, 256}. "N\* simulated" is the ideal additive memory of Experiment A at the same (d_k, d_v) (mean of the two simulations at d_k = 16, d_v = 64). "emb. d" is the token width; its value half (d/2) bounds the rank of the value information that reaches the state.

Figure `results/fig_trained.png` shows the accuracy curves and `results/fig_dvsweep.png` the value-dimension sweep. Five findings stand out.

1. **Trained additive memory reaches random-code capacity, and slightly exceeds it.** Linear attention achieves η = 1.02, 1.09 and 1.12 at d_k = 8, 16 and 32 (η_sim = 1.05–1.06 against the simulated ideal memory, which is the cleaner comparison because it removes the 3–8% error of Eq. (1)). At d_k = 64 it is still at 0.72 accuracy at the largest N tested (256), consistent with the predicted N\* = 339. The small surplus over η = 1 is expected: learned keys can be spread somewhat more evenly than random ones, but the Welch bound limits the gain at K/d_k ≥ 64.
2. **Gating and the delta rule give no extra capacity here.** GLA, DeltaNet and Gated DeltaNet land on the same curve as linear attention, within 0.03 accuracy at every N (largest spread across the four families: 0.009, 0.020 and 0.030 at d_k = 8, 16 and 32), and η ranges over 1.01–1.13 for all of them. The learned gates explain why. Gated DeltaNet learns a mean write strength β = 0.026, 0.036 and 0.053 on pair tokens at d_k = 8, 16 and 32, and DeltaNet learns β = 0.025–0.14 in the value sweep. Gated DeltaNet's learned decay averages 0.9999 (minimum 0.9997). With β ≪ 1 the delta correction −β(Mk)kᵀ is a small perturbation of a rescaled additive write β u kᵀ, and the read-out LayerNorm removes the overall scale. With decay ≈ 1 GLA-style gating is switched off. Gradient descent thus moves both mechanisms to the additive regime that Experiment A identified as best for retain-everything workloads; the β = 1 delta rule, about 3× worse in simulation, is avoided. When every stored item may be queried, the capacity-maximising policy is additive storage (Section 3.2), and gradient descent finds it.
3. **Capacity is linear in state size.** The log–log slope of N\* against d_k at d_v = 64 is 1.05 (linear), 1.05 (gla), 1.04 (delta) and 1.07 (gdelta) over d_k = 8–32. In the value sweep at d_k = 16, capacity doubles from d_v = 32 to 64 (47.5 → 95.6 for linear, 50.5 → 96.0 for delta; η_sim = 0.99–1.05) and then stops: d_v = 128 gives 97.0 and 98.5 at token width d = 128, only 0.57–0.63 of the expected capacity.
4. **Nominal state is not effective state.** The d_v = 128 plateau is an input bottleneck, not a property of the memory. At d = 128 the value half of the embedding is 64-dimensional, so the value codes written into the state span at most 64 dimensions, and a 16 × 128 state behaves like a 16 × 64 one. Widening the token to d = 512 removes the plateau: N\* = 177.2 (linear) and 177.5 (delta) at d_v = 128, η = 1.04 against Eq. (1), while the d_v = 64 control is unchanged (96.2 vs 95.6). So key and value dimensions are interchangeable as Eq. (2) predicts, but only up to the rank of the value information the layer receives. For real models this means that the S in Eq. (4) should be the *effective* state, which can be smaller than d_k·d_v·(number of heads).
5. **Softmax attention is not capacity-limited at these sizes.** With d_k = 8 it is at 1.00 accuracy at N = 256, where every fixed-state model with the same d_k is at ≈0.05.

Taken together, Experiments A and B say that, for this task, the recall capacity of a fixed-state layer is set by its state size and the log of the vocabulary, and that the update rule does not change it. Single-layer training reaches the theoretical capacity (η ≈ 1).

### 5.3 Recall demand of real text grows nearly linearly with context

| T | prose N_bi | prose N_ctx | prose ρ | code N_bi | code N_ctx | code ρ |
|---|---|---|---|---|---|---|
| 256 | 238 | 9 | 0.045 | 163 | 27 | 0.207 |
| 1,024 | 886 | 62 | 0.094 | 497 | 120 | 0.325 |
| 4,096 | 3,271 | 327 | 0.145 | 1,676 | 426 | 0.388 |
| 16,384 | 11,173 | 1,591 | 0.239 | 5,092 | 1,453 | 0.460 |
| 32,768 | 20,009 | 3,345 | 0.301 | 8,516 | 2,522 | 0.481 |

*Table 3 (source: `results/demand.json`; means over 30–60 random windows).* Figure `results/fig_demand.png`.

Three observations. (a) The number of distinct bigrams grows sublinearly, in the manner of Heaps' law: N_bi ∝ T^{0.92} (prose) and T^{0.83} (code) over T = 256–16,384. (b) The demand that matters, N_ctx, grows *at least* linearly over this range. Its fitted exponents are 1.24 ± 0.03 (prose) and 0.95 ± 0.02 (code), and the local exponents at T = 8k–32k are 1.06 and 0.84. Superlinear growth cannot continue: N_ctx ≤ N_bi, and the curve is still filling in as more associations occur twice. Within the context lengths that matter today, though, a fixed-state model faces demand that rises in proportion to T. (c) The *value* of recall also grows with T. At 32k tokens, 30% of prose tokens and 48% of Go tokens continue a context-specific bigram already seen in the window, up from 4.5% and 21% at T = 256. Code is about twice as recall-heavy as prose at every T.

### 5.4 Memory budget: fixed versus growing state

Using η ≈ 1 (Section 5.2), Eq. (4) and the measured D(T):

| per-layer state S (floats) | prose T\* every-token | prose T\* dedup | prose T\* oracle | code T\* every-token | code T\* dedup | code T\* oracle |
|---|---|---|---|---|---|---|
| 2^14 ≈ 1.6·10^4 | 770 | 880 | 8.2k | 690 | 1.4k | 7.0k |
| 2^16 ≈ 6.6·10^4 | 3.1k | 3.9k | 30k | 2.8k | 7.7k | (36k)† |
| 2^18 ≈ 2.6·10^5 | 12k | 18k | (112k)† | 11k | (45k)† | (189k)† |
| 2^20 ≈ 1.0·10^6 | (49k)† | (92k)† | (413k)† | (44k)† | (269k)† | (987k)† |

*Table 4 (source: `results/tradeoff.json`, η = 1).* † marks extrapolation beyond the measured T ≤ 32,768. Every-token values follow directly from Eq. (4) with D = T and need no corpus data.

The selectivity dividend, the ratio of oracle to every-token T\*, is about 10× for prose and 13–17× for code at S between 2^14 and 2^16. Holding the oracle demand of a full 32k-token window takes S_req = 7.1·10^4 floats (prose) or 6.0·10^4 floats (code). One attention layer with d = 4096 stores 2dT = 2.7·10^8 floats for the same window, about 4,000× more. Even under the every-token policy a fixed-state layer needs only 2 ln V ≈ 21–24 floats per token of context, against 2d = 8,192 for a KV cache.

The budget therefore separates two questions that are often merged:

- *Can a fixed state ever be recall-complete?* No. D(T) grows roughly linearly, so for every S there is a T\* beyond which context-specific associations must be dropped. This is the quantitative form of the impossibility results of Jelassi et al. [2024] and Wen et al. [2025].
- *Is attention's way of being recall-complete efficient?* Also no, by a large factor. A memory that grows with T but stores associations in superposition needs orders of magnitude fewer floats than a KV cache to keep the measured demand. Growing compressed memories such as log-linear attention [Guo et al., 2026] or selectively growing caches [Lufkin et al., 2026] sit in this gap.

## 6 Discussion: what the budget predicts

The analysis turns "can new architectures replace the Transformer?" into a claim about three measurable quantities: the state per layer (S), how efficiently a trained model uses it (η), and the recall demand of the workload (D(T)). It yields the following falsifiable predictions. All are stated for single-hop, bigram-level recall, and all are pre-registered here.

**P1 (knee at T\*).** In a pure fixed-state language model, the loss on context-only recall tokens (the ρ-tokens of Section 4.3) stops improving with context beyond a knee T_knee. With per-layer state S, T_knee lies between the every-token and oracle T\* of Table 4. In a Transformer of matched size, the same loss keeps falling with T. *Falsified if* a pure SSM's recall-token loss keeps improving at T ≫ T\*_oracle(S), or saturates below T\*_every-token(S).

**P2 (state scaling).** Across a family of otherwise identical fixed-state models that differ only in state size, T_knee ∝ S^{1/β_loc} with β_loc ≈ 0.85–1.05, i.e. roughly proportional to S. *Falsified if* doubling S moves the knee by less than 1.5× or more than 2.5×.

**P3 (domain).** At equal T and S, the pure-SSM versus Transformer gap on recall tokens is larger for code than for prose, in proportion to ρ (≈1.6× at 32k).

**P4 (hybrid sufficiency).** One attention layer removes the knee, but the remaining loss gap then scales with the fraction of recall tokens whose information has to pass through fixed-state layers. Adding attention layers beyond the first should help mainly on multi-hop or multi-token-value recall, not on the single-hop demand measured here. This offers one mechanistic reading of why recall gains in Wang et al. [2025] concentrate at high attention fractions: those tasks need more than single-hop retrieval.

**P5 (efficiency headroom).** Since additive superposition uses ≈2% of the state's information capacity (Proposition 2), a fixed-state layer with a non-linear or sparse read-out should beat the S/(2 ln V) law on the capacity-isolating MQAR of Section 4.2. Our five families did not, which shows that delta rules and gates alone do not escape the linear-read-out regime.

**Protocol for pretrained models (not run here; no GPU).** (1) Take matched model families that share data and tokenizer and differ in mixer, e.g. the publicly released 8B Mamba-2, Mamba-2-Hybrid and Transformer checkpoints of Waleffe et al. [2024]. (2) On held-out long documents (books; repository-level code), mark ρ-tokens using top-1 successors estimated on a disjoint corpus with the model's tokenizer. (3) Compute per-token loss on ρ-tokens and on all other tokens as a function of the position of the previous occurrence and of the total context T ∈ {1k, …, 128k}. (4) Compute each model's per-layer state S from its configuration, and predict T\* from Eq. (4) with ln V of its tokenizer and with D(T) re-measured on its tokens. (5) Test P1–P3 with the falsification thresholds above, reporting bootstrap confidence intervals over documents.

**What this means for the replacement question.** Our results back neither a simple yes nor a simple no. A model with a fixed state cannot be a recall-complete replacement at arbitrarily long context, and its failure point is predictable from S, η and D(T). The Transformer's KV cache, however, is not the efficient alternative either. Bigram-level recall demand of real text could be held in a memory that grows at a few to a few tens of floats per token, against thousands for a KV cache. The architecture that would actually replace the Transformer on recall is one whose memory grows with context but at a much smaller per-token cost, with selective writing as the main lever (the 10–17× dividend above). Production hybrids with 8–25% attention layers are a coarse way to get growing memory. Our analysis suggests that the attention fraction in such hybrids should be chosen from the expected D(T) of the target workload, not fixed by rule.

## 7 Limitations

- **Single-hop bigram recall only.** N_ctx counts word- or token-level bigram associations. Real recall also involves multi-token keys (entities, variable names in scope), multi-token values, and multi-hop composition. Bigram demand is therefore probably a lower bound on what a model has to store, while ln V of a full tokenizer probably overstates the bits per association that context actually supplies. These two effects pull in opposite directions, and we have not measured their net effect.
- **"Oracle" is not causal.** N_ctx uses future recurrence to decide what is worth storing. The oracle T\* is an upper bound on what selective writing can achieve, not something we have shown to be achievable.
- **Random-code theory.** Eqs. (1) and (2) are Gaussian approximations with errors of 3–11% at d_v ∈ {64,128} and larger errors outside that range (Section 5.1). They describe linear read-outs. Multi-layer models, non-linear feature maps (e.g. Based's Taylor features), MLP-based read-outs and multi-head routing could achieve η > 1, and our single-layer experiments cannot rule that out.
- **Small, single-layer, single-seed experiments.** Experiment B trains 1-layer models with S ≤ 4,096 for 1,200 steps, with one seed per configuration, on a task where pairs are pre-bound and writes are masked at queries. η ≈ 1 was reached in this idealised setting. Whether η stays near 1 in multi-layer language models, where the state also serves other functions, is open, and it is the main uncertainty in transferring Table 4 to real models. Because of a shared machine we dropped d_k = 64 for the gated and delta families, and even linear attention at d_k = 64 never fell below 50% accuracy in the tested range (N ≤ 256), so its η is not measured. Agreement within 0.03 accuracy across families at d_k ≤ 32 makes a different outcome at 64 likely to be small, but we did not test it. The width control (d = 512) covers only d_k = 16. The gate statistics are averages over pair tokens of one evaluation batch; we did not study how they evolve during training or with more steps. Training curriculum, optimiser and the pre-bound task format could all bias the learned β towards small values; a task where old items should be overwritten would reward larger β and non-trivial decay, and our results say nothing about that regime.
- **Corpora.** Prose comes from 18 old public-domain books, and windows ≥ 16k come from a subset of them. Code is a single (Go) standard library. Exponents may differ for web text, dialogue, or multi-file repositories. T\* values beyond 32k are extrapolations.
- **No large-model evidence.** We ran no pretrained SSMs or Transformers. P1–P5 are predictions, not findings. Numbers cited from other papers (layer ratios of production hybrids) are taken from those papers and were not reproduced.

## 8 Conclusion

We derived and checked a superposition-capacity law for fixed-state sequence mixers, N\* ≈ S/(2 ln V). Single-layer linear attention, GLA, DeltaNet and Gated DeltaNet all reach it after training on a capacity-isolating recall task (η ≈ 1.0–1.1), and gates and delta rules give no extra capacity when everything must be retained. Measuring real-text recall demand shows that context-specific associations grow roughly linearly with context length, and that recall opportunities make up 30–48% of tokens at 32k. Together these numbers bound the context length at which a fixed state must start forgetting, and they show that a recall-complete memory needs far less than a KV cache. The useful question is therefore not whether SSMs can replace attention, but how cheaply a memory can grow with context. Section 6 lists predictions, with falsification thresholds, that pretrained models can test against this account.

## References

- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. *Physical Review Letters*, 55(14), 1530–1533.
- Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou, J., Rudra, A., & Ré, C. (2024a). Simple linear attention language models balance the recall-throughput tradeoff. *ICML 2024* (PMLR 235). arXiv:2402.18668.
- Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., & Ré, C. (2024b). Zoology: Measuring and improving recall in efficient language models. *ICLR 2024*. arXiv:2312.04927.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *ICML 2024* (PMLR 235). arXiv:2405.21060.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*. arXiv:2111.00396.
- Guo, H., Yang, S., Goel, T., Xing, E. P., Dao, T., & Kim, Y. (2026). Log-linear attention. *ICLR 2026*. arXiv:2506.04761.
- Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
- Jelassi, S., Brandfonbrener, D., Kakade, S. M., & Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. *ICML 2024* (PMLR 235). arXiv:2402.01032.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *ICML 2020*. arXiv:2006.16236.
- Kimi Team (2025). Kimi Linear: An expressive, efficient attention architecture. arXiv:2510.26692.
- Kohonen, T. (1972). Correlation matrix memories. *IEEE Transactions on Computers*, C-21(4), 353–359.
- Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. arXiv:2403.19887 (ICLR 2025 as "Jamba: Hybrid Transformer-Mamba Language Models").
- Lufkin, L., Figliolia, T., Millidge, B., & Krishnamurthy, K. (2026). Hybrid associative memories. arXiv:2603.22325.
- MiniMax (2025). MiniMax-01: Scaling foundation models with lightning attention. arXiv:2501.08313.
- Merrill, W., Petty, J., & Sabharwal, A. (2024). The illusion of state in state-space models. *ICML 2024* (PMLR 235). arXiv:2404.08819.
- NVIDIA (2025). Nemotron-H: A family of accurate and efficient hybrid Mamba-Transformer models. arXiv:2504.03624.
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., et al. (2022). In-context learning and induction heads. arXiv:2209.11895.
- Qiao, M. (2025). Understanding Transformers through the lens of Pavlovian conditioning. arXiv:2508.08289 (later version titled "Attention as Conditioning: What Classical Learning Theory Predicts About Linear Transformers").
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers are secretly fast weight programmers. *ICML 2021* (PMLR 139), 9355–9366. arXiv:2102.11174.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762.
- Waleffe, R., et al. (2024). An empirical study of Mamba-based language models. arXiv:2406.07887.
- Wang, D., Zhu, R.-J., et al. (2025). A systematic analysis of hybrid linear attention. arXiv:2507.06457.
- Welch, L. R. (1974). Lower bounds on the maximum cross correlation of signals. *IEEE Transactions on Information Theory*, 20(3), 397–399.
- Wen, K., Dang, X., & Lyu, K. (2025). RNNs are not Transformers (yet): The key bottleneck on in-context retrieval. *ICLR 2025*. arXiv:2402.18510.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024a). Gated linear attention Transformers with hardware-efficient training. *ICML 2024*. arXiv:2312.06635.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024b). Parallelizing linear Transformers with the delta rule over sequence length. *NeurIPS 2024*. arXiv:2406.06484.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with delta rule. *ICLR 2025*. arXiv:2412.06464.
