# How Much State Does In-Context Recall Need? A Superposition-Capacity Budget for Fixed-State Sequence Models, Checked on Trained Mixers and on Real Token Streams

**Target venue:** International Conference on Learning Representations (ICLR 2027), main track

*Revision 1. Changes from the submitted version are listed in `reviews/response_r1.md`.*

---

## Abstract

Whether state-space models (SSMs), linear-attention RNNs and their hybrids can replace the Transformer depends heavily on in-context recall. Attention keeps a key–value (KV) cache that grows with the context. A fixed-state model must compress the context into a constant-size state. Earlier work proved that this costs something on copying and associative recall, and showed that a few attention layers repair it. It did not say how much state a given recall workload needs. We build that budget for one capability, single-hop bigram-keyed recall, and test each link of it on CPU-scale experiments. No pretrained models are used. (i) For additive outer-product memory with an argmax read-out over V codes, we specialise the classical signal-to-crosstalk analysis (Kohonen; McEliece et al.; recently Cabannes et al., Nichani et al., Giorlandino et al.) to explicit (d_k, d_v, V) dependence: N\* ≈ d_k/(2 ln V (1/d_v + 1/V)) ≈ S/(2 ln V). We add a fan-out term for keys that recur with several values, and a frame-potential bound on how much learned codes can beat random ones (η ≲ 1 + d_v/V). The Gaussian integral (Eq. 1) is within 3–9% of simulation at d_v ∈ {64, 128}. The closed form (Eq. 2) underestimates by 13–21% there, as the net effect of two opposing approximations that we separate. At V = 41,509 the closed form is within 9%, also with Zipf-distributed values. (ii) Single-layer linear attention, GLA, DeltaNet and Gated DeltaNet trained on a capacity-isolating recall task (3 seeds) reach 1.00–1.07× the simulated ideal additive memory. Family differences are within seed noise at d_k = 8 and 16 (ANOVA p = 0.95, 0.65). At d_k = 32, DeltaNet is 5% lower (p = 0.001) as its learned β grows. This holds for a retain-everything task, where additive storage is optimal by construction, and gradient descent keeps the delta-rule write strength small (β ≈ 0.03–0.17) and the decay at 0.9993–0.9999. On an overwrite task the same families separate: DeltaNet learns β ≈ 0.33 and recalls 0.90–0.91 of latest values at 16 live keys, against 0.30–0.31 for linear attention (2 seeds). (iii) Real-text demand uses a held-out smoothed-bigram criterion for "context-specific" (p(b|a) < 0.01, statistics from a disjoint half of the corpus) and document-level bootstrap CIs. At T = 32,768, N_ctx = 2,190 [2,069, 2,349] (prose) and 1,821 [1,624, 2,057] (Go code). 16% [14, 19] and 28% [26, 31] of tokens are context-only recall opportunities. The original top-1 rule gave 27% and 48%. (iv) Replaying real windows through an ideal fixed-state memory with a corpus-sized vocabulary, the fan-out-generalised Eq. (1) predicts recall to within 0.6–7 percentage points (48 conditions). The naive budget S_req = 2 ln V · N_ctx underestimates the state an oracle-writing memory actually needs by 1.3–14×, increasingly with T, because frequent keys recur with many values. With one head's value width (d_v = 64), recall of the context-specific set never reaches 50% for T ≥ 4k, whatever d_k. The budget therefore bounds when a fixed state must forget, and shows that value width and multi-token keys matter as much as total state. Even corrected, the oracle-selected bigram associations of a 16k window fit in ~1.3×10⁵ (prose) to ~3×10⁵ (code) floats, 400–1000× less than one d = 4096 KV cache. This is a lower bound for a narrow sub-task, not a like-for-like comparison. We pre-register predictions for pretrained models.

## 中文摘要

状态空间模型（SSM）、线性注意力RNN及其混合架构能否取代Transformer，很大程度上取决于上下文召回能力：注意力保留随上下文增长的键值（KV）缓存，定长状态模型则必须把上下文压缩进常数大小的状态。已有工作证明了这一代价，并表明少量注意力层即可修复，但没有回答"一个召回负载需要多少状态"。本文针对单跳、以二元组为键的召回构建这一预算，并在CPU规模上逐环检验（未使用预训练模型）。(i) 对带V个码字argmax读出的加性外积记忆，我们把经典信号—串扰分析（Kohonen、McEliece等，及近期Cabannes等、Nichani等、Giorlandino等）具体化到(d_k,d_v,V)：N\*≈d_k/(2 ln V(1/d_v+1/V))≈S/(2 ln V)；并加入"同键多值"扇出项，以及学习码字至多优于随机码字约(1+d_v/V)倍的帧势界。高斯积分式(1)在d_v∈{64,128}时与仿真相差3–9%；闭式(2)低估13–21%，这是两个方向相反近似的净效应（文中分解）；在V=41,509（含Zipf分布值）时闭式误差在9%以内。(ii) 在隔离容量的召回任务上（3个随机种子），单层线性注意力、GLA、DeltaNet与Gated DeltaNet达到理想加性记忆仿真值的1.00–1.07倍；该结论限于"全部保留"任务（此时加性存储按构造最优），梯度下降使写入强度β≈0.03–0.17、衰减≈0.9993–0.9999；在"覆盖写"任务上各族明显分化：DeltaNet学到β≈0.33，在16个活跃键时召回0.90–0.91，而线性注意力仅0.30–0.31。(iii) 我们用留出语料上的平滑二元组概率（p(b|a)<0.01）定义"依赖上下文"，并给出文档级bootstrap置信区间：T=32,768时N_ctx为2,190[2,069,2,349]（散文）与1,821[1,624,2,057]（Go代码），仅能从上下文获得的召回机会占16%与28%（原top-1规则为27%与48%）。(iv) 将真实文本窗口回放进理想定长记忆（词表为整个语料词表），带扇出项的式(1)对召回率的预测误差为0.6–7个百分点（48种条件）；而朴素预算S_req=2 ln V·N_ctx把理想选择性写入记忆实际所需状态低估1.3–14倍，且随T增大，原因是高频键在上下文中对应多个值；当值宽为单头宽度(d_v=64)时，T≥4k时无论d_k多大都达不到50%召回。即便校正后，16k窗口中经理想选择的二元关联也只需约1.3×10⁵（散文）至3×10⁵（代码）个浮点数，比d=4096的单层KV缓存小400–1000倍——这只是一个窄子任务的下界，而非同口径比较。文末给出面向预训练模型的预注册预测。

---

## 1 Introduction

The Transformer [Vaswani et al., 2017] answers every query by attending over a KV cache whose size grows linearly with context length. Structured and selective state-space models [Gu et al., 2022; Gu & Dao, 2023; Dao & Gu, 2024], linear attention and its gated and delta-rule variants [Katharopoulos et al., 2020; Schlag et al., 2021; Yang et al., 2024a; 2024b; 2025], and hybrids that interleave them with a few attention layers [Lieber et al., 2024; Waleffe et al., 2024; MiniMax, 2025; NVIDIA, 2025; Kimi Team, 2025] replace the growing cache with a fixed-size recurrent state. The qualitative answer to our research question is established. Fixed-state models are provably limited on copying and associative recall [Jelassi et al., 2024; Wen et al., 2025; Arora et al., 2024a]. They lag Transformers on recall-intensive tasks [Arora et al., 2024a; Waleffe et al., 2024]. A small fraction of attention layers restores recall [Wen et al., 2025; Waleffe et al., 2024; Wang et al., 2025]. Nemotron-H-8B has 52 layers: 4 self-attention, 24 Mamba-2 and 24 FFN [NVIDIA, 2025]. Attention is therefore ≈ 8% of all layers but 4/28 ≈ 14% of sequence-mixing layers. Kimi Linear uses 25% of its mixing layers for attention (3:1). A study of 72 hybrids recommends linear-to-full ratios of 3:1 to 6:1 [Wang et al., 2025].

What is missing is a quantitative account that connects three numbers: how many associations a fixed state of a given size can hold, how close trained models get to that, and how many associations real text asks a model to hold. This paper gives such an account for one well-defined capability, single-hop associative recall keyed by the previous token. It then tests the link between the three numbers on real token streams. The capacity scaling itself is classical (Section 2). Our contribution is the budget that ties it to measured demand, and the tests of where that budget holds and where it fails.

**Contributions.**

1. **A capacity formula with explicit (d_k, d_v, V) dependence, a fan-out term and an efficiency bound (Section 3).** We specialise the classical signal-to-crosstalk analysis of outer-product memories to the hetero-associative, argmax-over-vocabulary read-out of a language-model head. The scaling N\* ∝ S/(2 log ·) is not new [McEliece et al., 1987; Cabannes et al., 2024; Nichani et al., 2025; Giorlandino et al., 2026]. We add three things. (a) The 1/V correction and the lag-resolved form for decayed memories. (b) A fan-out generalisation for keys that recur with several values, which real text needs (Section 5.5). (c) A frame-potential argument that learned codes can beat random ones by at most ≈ (1 + d_v/V)(1 + d_k/K). We quantify how accurate each approximation is, with Monte Carlo CIs (Section 5.1).
2. **A test of whether training reaches the formula, and of when the update rule matters (Sections 5.2–5.3).** On a retain-everything recall task, four fixed-state families reach 1.00–1.07× the simulated ideal additive memory (3 seeds). This is above 1 but below the frame-potential ceiling of 1.125. On an overwrite task the delta rule learns a large write strength and beats additive storage by about 3× at 16 live keys (0.90 vs 0.30 accuracy). So "the update rule does not matter" holds only when nothing should be forgotten.
3. **A measurement of real-text recall demand with held-out criteria and uncertainty (Section 5.4).** Prose (word-level and byte-level BPE) and Go code. Context-specific associations are defined by held-out smoothed-bigram probability, with a sensitivity analysis over five criteria and document-level bootstrap CIs.
4. **A replay test of the budget on real streams (Section 5.5).** The fan-out-generalised Eq. (1) predicts the recall of an ideal memory on real windows (V up to 149k) within 0.6–7 points. For oracle writing, the naive budget underestimates the needed state by 1.3–14×, and a 64-wide value path caps recall below 50% for T ≥ 4k.
5. **A corrected budget and pre-registered predictions (Sections 5.6, 6).** We state predictions P1–P5 for pretrained models. We have not tested them.

We do not claim that single-hop bigram recall is all that matters. We do not run experiments on pretrained models (Section 7).

## 2 Related Work

**Fixed-state sequence models.** S4 [Gu et al., 2022] made linear state-space layers practical. Mamba [Gu & Dao, 2023] added input-dependent ("selective") parameters. Mamba-2 [Dao & Gu, 2024] showed that SSMs with scalar-times-identity transitions are a form of gated linear attention. Linear attention [Katharopoulos et al., 2020] collapses the KV cache to a d_k × d_v state. GLA adds data-dependent decay [Yang et al., 2024a]. DeltaNet [Schlag et al., 2021; Yang et al., 2024b] uses a delta-rule update. Gated DeltaNet combines both [Yang et al., 2025]. Log-linear attention [Guo et al., 2026] and Hybrid Associative Memories [Lufkin et al., 2026] let the memory grow sublinearly or selectively with context.

**Recall limits of fixed-state models.** Zoology [Arora et al., 2024b] introduced multi-query associative recall (MQAR) and showed that it explains most of the gap between gated convolutions and attention. Based [Arora et al., 2024a] made the state-size/recall trade-off explicit and proved lower bounds for recurrent models. Jelassi et al. [2024] proved that fixed-state models cannot copy strings longer than their state permits. Wen et al. [2025] proved that RNNs cannot solve associative recall without attention. Okpekpe & Orvieto [2025] revisit associative recall in modern recurrent models. They report that recurrent models are much more sensitive to the learning rate than Transformers, benefit most from added width, and can solve recall in a single properly tuned Mamba layer. Optimisation effects can therefore confound capacity conclusions. We control for this by comparing against a simulated ideal memory, and by checking that four families coincide across seeds. Merrill et al. [2024] show that SSMs share the TC⁰ limit on state tracking, a capability we do not study. Qiao [2025] argues, through a conditioning-theory lens, that reconstruction is limited to Θ(d_k) items while selection from a vocabulary allows more.

**Capacity of associative memories.** Correlation-matrix memories [Kohonen, 1972] and Hopfield networks have classical capacity analyses: α_c ≈ 0.138 for the Hopfield model [Amit et al., 1985], and n/(2 log n) recoverable memories for outer-product storage [McEliece et al., 1987]. Cabannes et al. [2024] derive scaling laws for outer-product associative memories with random embeddings and Zipf-distributed data, as models of Transformer inner layers. Nichani et al. [2025] show that shallow Transformers store facts with parameter counts linear in the number of facts up to log factors, including through linear-attention value matrices. Giorlandino et al. [2026] characterise the storage capacity of a (decoupled) linear associative memory with statistical-physics methods. They report a sharp threshold at p log p / d² = 1/2, the same S/(2 log) form as our Eq. (2). Barnfield et al. [2026] derive sharp capacity thresholds for top-1 retrieval in linear associative memory. Our Proposition 1 is a variant of this signal-to-crosstalk argument. What we add is the (d_v, V) form for vocabulary read-outs, the fan-out term, the frame-potential ceiling for learned codes [Benedetto & Fickus, 2003], and above all the link to measured real-text demand. Induction heads [Olsson et al., 2022] motivate bigram associations as the unit of demand. Heaps' law [Heaps, 1978] describes sublinear vocabulary growth.

**Hybrid design.** Jamba [Lieber et al., 2024], the NVIDIA 8B Mamba-2-Hybrid [Waleffe et al., 2024], Nemotron-H [NVIDIA, 2025], MiniMax-01 [MiniMax, 2025] and Kimi Linear [Kimi Team, 2025] are production-scale hybrids. Wang et al. [2025] found that recall improves with the fraction of full-attention layers, particularly below a 3:1 linear-to-full ratio.

## 3 Theory

### 3.1 Setting

A memory reads (key, value) pairs and later answers "what value was paired with key k?". Keys k_j ∈ ℝ^{d_k} have unit norm. Value symbols v_j ∈ [V] have unit codes u_v ∈ ℝ^{d_v}. Read-out is r = M k, followed by argmax_w ⟨u_w, r⟩. This is the structure of an LM head applied to a linear-attention or SSM output. We use three update rules for M ∈ ℝ^{d_v × d_k} (S = d_k d_v floats):

- additive: M_t = M_{t−1} + u_{v_t} k_tᵀ;
- decayed (scalar gate γ_t): M_t = γ_t M_{t−1} + u_{v_t} k_tᵀ;
- delta rule with write strength β_t: M_t = M_{t−1} + β_t (u_{v_t} − M_{t−1}k_t) k_tᵀ (and its gated version).

### 3.2 Proposition 1 (additive superposition memory; a specialisation of the classical analysis)

Keys are i.i.d. uniform on the sphere in ℝ^{d_k}, codes are i.i.d. uniform on the sphere in ℝ^{d_v}, and values are i.i.d. uniform on [V]. The memory stores pairs (k_j, u_{v_j}) with weights w_j. The queried item i has weight s. We define two loads:

- **own-key load** F_i: the summed squared weights of the *other* values stored under the *same* key (F_i = 0 when keys are distinct);
- **cross-key load** G_i: the summed squared weights of all items stored under other keys.

Let c_j = ⟨k_j, k_i⟩, so E c_j = 0 and E c_j² = 1/d_k. To leading order:

- correct score ⟨u_{v_i}, r⟩ has mean s and variance σ_c² ≈ F_i/d_v + G_i (1/(d_k d_v) + 1/(V d_k));
- a competitor w ∉ {values stored under k_i} has mean 0 and variance σ_w² ≈ s²/d_v + F_i/d_v + G_i (1/(d_k d_v) + 1/(V d_k)).

Treating the V − 1 competitor scores as independent Gaussians,

  Acc ≈ ∫ φ(z) Φ((s + σ_c z)/σ_w)^{V−1} dz.  (1)

For distinct keys (F = 0), unit weights (s = 1, G = N − 1), and the leading extreme-value approximation max ≈ σ_w √(2 ln V), dropping s²/d_v:

  **N\* ≈ d_k / (2 ln V · (1/d_v + 1/V))**, i.e. N\* ≈ S/(2 ln V) when d_v ≪ V.  (2)

*Derivation.* The moments follow from E c_j² = 1/d_k, E⟨u_a,u_b⟩² = 1/d_v (a ≠ b), independence of keys and codes, and P(v_j = w) = 1/V. Items under the same key contribute without the 1/d_k attenuation, hence F/d_v. ∎

**What the approximations do (quantified in Section 5.1).** Eq. (2) makes two approximations that pull in opposite directions. (a) √(2 ln V) overstates the median of the maximum of V − 1 standard normals: 3.53 against 3.00 at V = 512, and 4.61 against 4.15 at V = 41,509. On its own this makes Eq. (2) *underestimate* N\* by a factor of about (3.53/3.00)² ≈ 1.39 at V = 512. (b) Dropping s²/d_v makes Eq. (2) *overestimate* N\*, by an amount that grows as d_v shrinks: at d_v = 64, 2 ln V/d_v = 0.19 is not small. Their net effect is the 13–21% underestimate seen at d_v ∈ {32, …, 128}. Keeping s²/d_v and using the exact median ("rule 2-med") is within 0–6% at d_v = 64.

**Corollary 1 (fan-out).** Suppose a key recurs with several values, as when the previous token "the" is followed by many different words in one context. The own-key load enters with weight 1/d_v instead of 1/(d_k d_v). Recall of such an item needs s² ≳ 2 ln V (F/d_v + G/(d_k d_v)), roughly F < d_v/(2 ln V) for unit weights. A larger d_k does not help. For d_v = 64 and V = 41,509, only keys with fewer than about 3 competing values are recallable. Real text has many such keys (Section 5.5). So the right budget variable is the load structure (F, G) of the write stream, not the count of associations alone.

**Corollary 2 (learned codes; frame-potential ceiling).** Can trained keys and codes beat random ones? For K unit vectors in ℝ^{d_k}, the frame potential satisfies Σ_{i,j} ⟨k_i,k_j⟩² ≥ K²/d_k [Benedetto & Fickus, 2003]. So the mean squared off-diagonal overlap is at least (K/d_k − 1)/(K − 1). Random keys give 1/d_k, so the key-side gain is at most a factor (1 − 1/K)/(1 − d_k/K) ≈ 1 + d_k/K, under 1% at K = 4096. For the V value codes, a tight frame lowers the mean squared overlap to (V/d_v − 1)/(V − 1). The competitor variance then becomes exactly G/(d_k d_v): the 1/V term disappears. The capacity gain is at most a factor 1 + d_v/V, which is 1.0625, 1.125 and 1.25 at d_v = 32, 64, 128 with V = 512. The argument is about second moments, the quantity that governs the Gaussian picture. It replaces the Welch-bound argument of our first version, which bounded the maximum and not the summed squared crosstalk. It predicts trained efficiencies slightly above 1 that increase with d_v/V. Section 5.2 tests this.

**Decay.** With constant decay γ, an item at lag τ has s = γ^τ and G ≤ 1/(1 − γ²). Eq. (1) predicts a sharp recall horizon τ\* ≈ ln(1/(√(2 ln V) σ))/ln(1/γ). When every stored item may be queried, γ = 1 is best. Decay and the delta rule pay off only when old items are worth less than new ones, or are superseded (Section 5.3).

### 3.3 Proposition 2 (information bound; the standard counting argument)

Consider any memory whose state after N pairs with distinct keys consists of S coordinates of b bits, and which answers a uniformly random stored key correctly with probability ≥ 1 − ε when values are uniform on [V]. It satisfies N((1 − ε) log₂ V − 1) ≤ S b (3). *Proof:* Fano's inequality per key, together with H(Z) ≤ Sb (as in Arora et al., 2024a; Jelassi et al., 2024). ∎ At V = 512, ε = 0.5 and b = 16, (3) allows N ≤ 4.6 S. Additive memory achieves ≈ 0.086 S at d_v = 64 (Table 1), about 2% of the bound. Even the frame-potential ceiling (Corollary 2) leaves this regime unchanged. The limit comes from the linear read/write geometry, not from the information content of the state.

### 3.4 From capacity to a budget

Let D(T) be the associations a model must hold at context length T, and η the trained memory's capacity relative to the ideal additive memory. The **naive budget** is

  D(T) ≤ η S/(2 ln V),  so  T\*(S) = D⁻¹(ηS/(2 ln V)),  S_req(T) = 2 ln V · D(T)/η.  (4)

D depends on the write policy. *every-token*: D = T. *oracle*: only context-specific associations that will recur are written, D = N_ctx(T). No causal model can do better without seeing the future. Eq. (4) assumes distinct keys. The **load-aware budget** instead evaluates Eq. (1) with each query's (s, F, G) taken from the actual write stream. Section 5.5 compares the two on real text.

## 4 Experimental Methods

All experiments run on CPU (4 cores, shared with unrelated jobs). Code and raw outputs are in `code/` and `results/`. Every number below comes from these runs unless it is marked as cited.

**Experiment A (ideal memories).** `capacity_sim.py` (original) and `capacity_ci.py` (revision). The revision uses an N grid spaced by 2^{1/8} around the threshold and 200 independent trials per cell (fresh keys, codes and values; all N items queried). It reports 95% bootstrap CIs for N\* (resampling trials) and compares Eq. (1), Eq. (2), Eq. (2) with s²/d_v kept, and Eq. (2) with the exact extreme-value median. A large-vocabulary check uses V = 41,509 (the prose vocabulary) with uniform values and with values drawn from the empirical prose unigram distribution (40 trials per cell).

**Experiment B (trained mixers, capacity-isolating MQAR).** As in the first version: N pre-bound pair tokens with distinct keys from K = 4096 and values from V = 512, then Q = 32 queries. Writes are masked at queries. There is one layer with L2-normalised q, k of width d_k and values of width d_v, and the read-out is LayerNorm followed by a linear map to V logits. Families: attn, linear, gla (scalar data-dependent decay), delta (DeltaNet, parallel UT form [Yang et al., 2024b]), gdelta (Gated DeltaNet). Training: AdamW, one-cycle schedule with peak 3·10⁻³, batch 64, 1,200 steps, N ∈ {4, …, 256}. **Revision:** seeds 0, 1, 2 for linear, gla, delta and gdelta at d_k ∈ {8, 16, 32}, d_v = 64 (`run_train_seeds.sh`). Each is evaluated on a 2^{1/4}-spaced N grid with 16,384 queries per point, which removes most of the factor-2 interpolation error of the first version. Seed 0 reproduces the first-version models bit for bit; only the evaluation grid changes. The value-width runs (d_v ∈ {32, 128}, token widths d ∈ {128, 512}) are single-seed, from the first version. d_k = 64 was run only for linear attention (Section 7).

**Experiment B′ (overwrite task, new).** Each sequence has P live keys. Each key is written 3 times with independent random values in random order, and queries ask for the latest value (`--task overwrite`). Retaining everything is now harmful. Families linear, gla, delta and gdelta at d_k = 16, d_v = 64, P ∈ {4, …, 128}, seeds 0 and 1, otherwise identical training. We compare against ideal memories (additive; delta with β ∈ {1, 0.5}; decay γ ∈ {0.9, 0.97}), simulated in `analyze_v2.py`.

**Experiment C (real-text demand).** `recall_demand.py` (original) and `recall_demand_v2.py` (revision). *Corpora:* the 18-book NLTK Gutenberg selection (2.14M lower-cased words, vocabulary 41,509); the same text without the KJV Bible (1.34M words); the same 18 books tokenised with a byte-level BPE of vocabulary 8,192 trained on the corpus with HuggingFace `tokenizers` (3.28M tokens); and the Go 1.25.1 standard library, non-test files concatenated per package (290 documents, 11.4M tokens, vocabulary 149,292). *Context-specific predicates* for a recurring bigram (a, b) use statistics from the *other half* of a two-fold document split (documents alternate by length), so a window never contributes to its own parametric statistics. The exception is D0, kept to reproduce the first version.

| def. | criterion |
|---|---|
| D0 | b ≠ in-sample corpus top-1 successor of a (first version) |
| D1 | b ≠ held-out top-1 successor of a |
| D2 | b ∉ held-out top-10 successors of a |
| **D3 (primary)** | held-out smoothed bigram p(b∣a) < 0.01, with p(b∣a) = (C(a,b) + p_uni(b))/(C(a) + 1) |
| D4 | p(b∣a) < 0.001 |

N_ctx[D](T) counts distinct bigrams that occur ≥ 2 times in a T-window and satisfy D. ρ[D](T) is the fraction of positions whose bigram occurred earlier in the window and satisfies D. There are 100 windows per T up to 8,192 and 60 beyond. 95% CIs come from a cluster bootstrap over documents (B = 1,000): documents are resampled with replacement, and every window from a resampled document enters with its multiplicity. The same procedure gives CIs for the global log–log exponent (T = 256–16,384) and the local exponent (T = 8,192–32,768).

**Experiment D (budget).** `analyze.py` (original) and `analyze_v2.py` (revision) evaluate Eq. (4) with ln V of the corpus vocabulary.

**Experiment D′ (replay on real streams, new).** `replay_memory.py`. Five random windows per T ∈ {1,024, 4,096, 16,384} from prose and code. We build ideal additive states M = Σ u_b k_aᵀ with a random unit key per token type, a random unit code per type, and the *whole* corpus vocabulary as competitors (V = 41,509 or 149,292). There are two policies: *every-token* (every bigram written, with multiplicity) and *oracle* (each context-specific recurring bigram written once, D0 or D3). There are two shapes: d_v = 64 (one head's value width; d_k = S/64) and square (d_k = d_v = √S). S ranges over 2^{10} to 2^{20}. Queries are the window's N_ctx set (up to 200 per window for prose and 100 for code), read at the end of the window. A query (a, b) succeeds if b outscores every code that is *not* an in-window successor of a. Legitimate ambiguity among several in-window successors is therefore not counted as an error, and an exact memory scores 1. For each condition we record the measured accuracy, the load-aware prediction of Eq. (1) (each query's s, F, G computed from the actual write multiset), and S_half, the state at which accuracy reaches 50%. We compare S_half with the naive S_req of Eq. (4).

## 5 Results

### 5.1 The formula against ideal memories, with uncertainty

| d_k, d_v (V = 512) | N\* simulated [95% CI] | Eq. (1) | rule (2) | rule (2) keeping s²/d_v | rule (2)-med (s²/d_v kept, exact median) | ideal delta, β = 1 (v1) |
|---|---|---|---|---|---|---|
| 8, 64 | 42.1 [41.6, 42.5] | 43.2 (+3%) | 36.5 (−13%) | 30.4 (−28%) | 44.5 (+6%) | 14.3 |
| 16, 64 | 87.9 [87.2, 88.9] | 85.5 (−3%) | 73.0 (−17%) | 59.7 (−32%) | 88.0 (0%) | 30.0 |
| 32, 64 | 181.3 [180.2, 182.3] | 169.9 (−6%) | 145.9 (−19%) | 118.5 (−35%) | 175.0 (−3%) | 63.4 |
| 64, 64 | 368.8 [367.3, 370.3] | 338.8 (−8%) | 291.8 (−21%) | 235.9 (−36%) | 349.0 (−5%) | 132.1 |
| 16, 16 | 21.3 [20.9, 21.8] | 12.5 (−41%) | 19.9 (−7%) | 5.4 (−75%) | 13.1 (−39%) | – |
| 16, 32 | 46.2 [45.6, 47.0] | 38.2 (−17%) | 38.6 (−16%) | 24.6 (−47%) | 39.5 (−14%) | – |
| 16, 128 | 151.4 [150.6, 152.3] | 165.8 (+9%) | 131.3 (−13%) | 119.5 (−21%) | 170.4 (+13%) | – |
| 16, 256 | 231.0 [229.2, 232.5] | 286.3 (+24%) | 218.9 (−5%) | 209.2 (−9%) | 294.0 (+27%) | – |

*Table 1 (source: `results/sim_capacity_ci.json`; the last column is from `results/sim_capacity.json`, first version).* Bootstrap CIs are about ±1%. The first version's two estimates of the (16, 64) cell (92.3 and 89.8; 20–40 trials, factor-2 grid) were 2–5% high, which is why we no longer quote errors to 0.1%. Eq. (1) is within 3–9% for d_v ∈ {64, 128}. It fails at d_v ≤ 32, where scores are far from Gaussian, and at d_v = 256 (d_v/V = 0.5). The closed form, Eq. (2), always underestimates, by 5–21%. **Correction to v1:** the first version attributed the 3–11% accuracy of Eq. (1) to the closed form. The closed form's apparent accuracy at d_v = 16 and 256 is a cancellation of its two opposing errors (Section 3.2), not evidence that the law is best there. Capacity is linear in d_k (log–log slope 1.04 over d_k = 8–64). The ideal β = 1 delta rule holds ≈ 3× fewer items than additive storage.

*Large vocabulary.* At V = 41,509, d_v = 64, the simulated N\* is 52.1 [50.7, 53.2] (d_k = 16) and 105.6 [104.4, 106.8] (d_k = 32) with uniform values. With Zipf-distributed values it is 52.8 [51.2, 54.3] and 104.7 [103.5, 106.0], so the value distribution does not matter here. Rule (2) gives 48.1 and 96.1 (−8%, −9%). Eq. (1) gives 43.4 and 85.8 (−17%, −19%). The ln V dependence therefore carries over from V = 512 to a corpus-sized vocabulary for distinct keys. It ignores fan-out, which Section 5.5 shows to be decisive on real text.

*Decay (first version, unchanged).* Recall horizons are ≈ 15, 40 and 80 writes for γ = 0.9, 0.97 and 0.99. Mean accuracy over 256 items is 0.06, 0.15, 0.27 and 0.33 for γ = 0.9, 0.97, 0.99 and 1.0.

### 5.2 Trained mixers on the retain-everything task (3 seeds)

| family | d_k | N\* per seed (0, 1, 2) | N\* mean ± sd | N\*_sim [95% CI] | η_sim mean (range) | learned β (seeds) | learned decay |
|---|---|---|---|---|---|---|---|
| linear | 8 | 44.1, 43.9, 43.8 | 43.9 ± 0.1 | 42.1 [41.6, 42.5] | 1.04 (1.04–1.05) | – | – |
| linear | 16 | 93.1, 94.6, 94.0 | 93.9 ± 0.8 | 87.9 [87.2, 88.9] | 1.07 (1.06–1.08) | – | – |
| linear | 32 | 193.3, 193.5, 192.7 | 193.2 ± 0.4 | 181.3 [180.2, 182.3] | 1.07 (1.06–1.07) | – | – |
| gla | 8 | 44.2, 44.1, 43.2 | 43.8 ± 0.6 | 42.1 | 1.04 (1.03–1.05) | – | 0.9993 |
| gla | 16 | 92.9, 94.3, 92.9 | 93.4 ± 0.8 | 87.9 | 1.06 (1.06–1.07) | – | 0.9995 |
| gla | 32 | 191.1, 192.7, 188.7 | 190.8 ± 2.0 | 181.3 | 1.05 (1.04–1.06) | – | 0.9996 |
| delta | 8 | 44.8, 42.7, 43.5 | 43.7 ± 1.1 | 42.1 | 1.04 (1.02–1.07) | 0.046, 0.092, 0.088 | – |
| delta | 16 | 94.1, 94.2, 91.2 | 93.2 ± 1.7 | 87.9 | 1.06 (1.04–1.07) | 0.064, 0.103, 0.115 | – |
| delta | 32 | 185.6, 183.2, 177.8 | 182.2 ± 4.0 | 181.3 | 1.01 (0.98–1.02) | 0.127, 0.154, 0.168 | – |
| gdelta | 8 | 44.0, 43.9, 43.8 | 43.9 ± 0.1 | 42.1 | 1.04 (1.04–1.05) | 0.025, 0.025, 0.026 | 0.9999 |
| gdelta | 16 | 93.6, 95.8, 93.6 | 94.3 ± 1.3 | 87.9 | 1.07 (1.06–1.09) | 0.036, 0.035, 0.037 | 0.9999 |
| gdelta | 32 | 193.0, 192.7, 192.1 | 192.6 ± 0.4 | 181.3 | 1.06 (1.06–1.07) | 0.053, 0.053, 0.055 | 0.9999 |

*Table 2 (source: `results/analysis_v2.json`, from `results/train_seeds/`).* N\* is interpolated on the 2^{1/4} grid. η_sim = N\*_trained / N\*_sim, where N\*_sim is from Table 1. The frame-potential ceiling of Corollary 2 is 1 + d_v/V = 1.125.

Four findings.

1. **Trained fixed-state memory reaches the ideal additive capacity, and slightly exceeds it, within the frame-potential ceiling.** Family means of η_sim are 1.00–1.07 (individual runs 0.98–1.09). That is above 1, as Corollary 2 allows through tighter-than-random value codes, and below the ceiling 1 + d_v/V = 1.125. The first version's η = 1.01–1.13 was measured against Eq. (1). It mixed Eq. (1)'s own 3–8% error with a ≈ 2% upward bias from interpolating on a factor-2 grid: seed-0 models are identical, and the old N\* values are 1.1–3.3% above the dense-grid ones (mean 2.2%).
2. **Families do not differ at d_k = 8 and 16; at d_k = 32 DeltaNet falls about 5% short.** In a one-way ANOVA of η_sim across the four families, p = 0.95 (d_k = 8) and 0.65 (d_k = 16). The sd between family means (0.003, 0.006) is below the sd between seeds (0.015, 0.014). At d_k = 32 the families differ (p = 0.001), entirely because of DeltaNet (η_sim 1.01 against 1.05–1.07). Its learned β grows with d_k (0.05–0.09 at d_k = 8, 0.13–0.17 at d_k = 32), and larger write strengths partly erase older items. The ideal β = 1 delta rule is 3× worse (Table 1). Gated DeltaNet keeps β ≈ 0.03–0.05 and matches linear attention. The learned decays of GLA and Gated DeltaNet are 0.9993–0.9999, so gating is effectively switched off. In this retain-everything task additive storage is optimal by construction (Section 3.2), and gradient descent finds it, or comes close to it.
3. **Capacity is linear in d_k over 8–32.** Per-seed log–log slopes are 1.07 ± 0.002 (linear), 1.06 ± 0.005 (gla), 1.03 ± 0.02 (delta) and 1.07 ± 0.0002 (gdelta). Linear attention at d_k = 64 (first version, single seed) was still at 0.72 accuracy at N = 256, consistent with N\*_sim = 369. Other families were not run at d_k = 64.
4. **Seed variation is 0.1–4 items (≤ 2.2% of N\*).** It is comparable to the between-family differences of the first version, which we no longer interpret, except for the DeltaNet effect at d_k = 32, which exceeds seed noise.

*Value-width control (first version, single seed; unchanged).* At token width d = 128, the value half of the embedding has 64 dimensions, so d_v = 128 gives no more capacity than d_v = 64 (N\* = 97.0 linear, 98.5 delta). At d = 512 the plateau disappears: N\* = 177.2 (linear) and 177.5 (delta), η_sim = 1.17 against the revised simulation (151.4), below the d_v = 128 ceiling of 1.25. At d_v = 32, η_sim = 1.03 (linear) and 1.09 (delta) against the revised simulation (46.2). The DeltaNet value is 3% above the second-moment ceiling of 1.06. It is a single seed, and at d_v = 32 the crosstalk is far from Gaussian (Eq. (1) is off by 17% there), so we read the ceiling as approximate in that regime. *Nominal* state d_k·d_v can therefore overstate *effective* state when the input does not carry enough independent value dimensions. Softmax attention (d_k ∈ {8, 16}) is at 1.00 accuracy at N = 256 (first version).

### 5.3 When the update rule matters: the overwrite task

| family (trained, 2 seeds) | P = 4 | 8 | 16 | 32 | 64 | 128 | learned gates |
|---|---|---|---|---|---|---|---|
| linear | 0.34 / 0.32 | 0.32 / 0.34 | 0.30 / 0.31 | 0.24 / 0.24 | 0.14 / 0.13 | 0.07 / 0.06 | – |
| gla | 0.50 / 0.48 | 0.53 / 0.56 | 0.59 / 0.61 | 0.54 / 0.57 | 0.39 / 0.39 | 0.24 / 0.24 | decay 0.985 / 0.983 |
| delta | 0.99 / 0.99 | 0.97 / 0.98 | 0.91 / 0.90 | 0.70 / 0.71 | 0.43 / 0.43 | 0.24 / 0.24 | β 0.33 / 0.34 |
| gdelta | 0.97 / 0.97 | 0.93 / 0.95 | 0.88 / 0.89 | 0.71 / 0.71 | 0.46 / 0.46 | 0.27 / 0.26 | β 0.24 / 0.26, decay 1.00 |
| *ideal additive* | 0.38 | 0.30 | 0.30 | 0.24 | 0.14 | 0.06 | |
| *ideal delta β = 1 / β = 0.5* | 0.98 / 1.00 | 0.93 / 0.98 | 0.77 / 0.87 | 0.45 / 0.64 | 0.23 / 0.37 | 0.11 / 0.20 | |
| *ideal decay γ = 0.9 / 0.97* | 0.90 / 0.60 | 0.80 / 0.68 | 0.59 / 0.72 | 0.36 / 0.56 | 0.19 / 0.35 | 0.10 / 0.19 | |

*Table 2b (figure: `results/fig_revision.png`(b); source: `results/train_overwrite/`, `results/analysis_v2.json`; accuracy for the latest value; seed 0 / seed 1; d_k = 16, d_v = 64; each of P live keys written 3 times).*

Here the update rule decides the outcome. Linear attention tracks the ideal additive memory: at P = 4 it is near 1/3, because each key's three values are superposed with equal weight. DeltaNet learns β ≈ 0.33, five times its retain-everything value at d_k = 16, and recalls 0.90–0.91 of latest values at P = 16, against 0.30–0.31 for linear attention. It also beats the ideal β = 1 delta rule (0.77) and the ideal β = 0.5 rule (0.87), because an intermediate β overwrites the same key while sparing other keys. GLA learns decay ≈ 0.985 and lands between the two. Gated DeltaNet learns β ≈ 0.25 and leaves its decay at 1. At large P all families converge towards the additive-capacity regime. So the retain-everything result of Section 5.2 is a statement about that workload. When stored values are superseded, delta-rule updates are worth up to a factor of 3 in recall at fixed state, and gradient descent finds the appropriate write strength.

### 5.4 Real-text recall demand with held-out criteria

| T | corpus | N_bi | N_ctx D0 (v1 rule) | N_ctx **D3** [95% CI] | N_ctx D4 | ρ D0 | ρ **D3** [95% CI] |
|---|---|---|---|---|---|---|---|
| 1,024 | prose | 896 | 59 | 34 [30, 39] | 18 | 0.085 | 0.050 [0.040, 0.061] |
| 4,096 | prose | 3,288 | 323 | 192 [171, 217] | 97 | 0.142 | 0.081 [0.065, 0.100] |
| 16,384 | prose | 11,277 | 1,586 | 1,010 [943, 1,087] | 495 | 0.238 | 0.129 [0.110, 0.154] |
| 32,768 | prose | 21,022 | 3,175 | 2,190 [2,069, 2,349] | 1,212 | 0.272 | 0.158 [0.136, 0.193] |
| 32,768 | prose, BPE-8k | 17,661 | 3,496 | 2,117 [2,020, 2,233] | 1,110 | 0.341 | 0.192 [0.168, 0.221] |
| 32,768 | prose w/o KJV | 20,950 | 3,212 | 2,120 [2,037, 2,228] | 1,043 | 0.273 | 0.144 [0.135, 0.157] |
| 1,024 | code | 507 | 119 | 69 [66, 73] | 39 | 0.321 | 0.174 [0.158, 0.190] |
| 4,096 | code | 1,662 | 412 | 254 [239, 268] | 144 | 0.389 | 0.204 [0.192, 0.217] |
| 16,384 | code | 5,255 | 1,492 | 1,003 [936, 1,072] | 632 | 0.454 | 0.253 [0.233, 0.275] |
| 32,768 | code | 8,618 | 2,576 | 1,821 [1,624, 2,057] | 1,241 | 0.478 | 0.280 [0.256, 0.309] |

*Table 3 (source: `results/demand_v2.json`).* Windows of length T come only from documents of at least T tokens. **Correction to v1:** at T = 32,768, 12 of the 18 books are eligible (all 12 were sampled), not 8 as the first version stated. The figures for code are 53 eligible and 38 sampled packages. The re-sampled D0 values differ from the first version's Table 3 by ≤ 5% (for example 3,175 against 3,345 at 32k), which is within the new CIs.

Four observations.

(a) *The first-version criterion inflated demand.* Replacing "not the in-sample top-1" with a held-out probability criterion removes 29–42% of N_ctx and 41–48% of ρ (D0 → D3). The held-out top-10 criterion (D2) removes 30–34% of prose N_ctx at T = 1k–4k (Table in `demand_v2.json`). This confirms the reviewer's estimate in direction; it is somewhat smaller than their in-sample 45%. The held-out model here is a smoothed bigram fitted on about 1M words. A real LM's parametric knowledge would remove more, so D3 is still an over-estimate of what *must* come from context. D4 (p < 0.001) roughly halves demand again. We carry D0, D3 and D4 through the budget as a sensitivity range.

(b) *Growth exponents (95% bootstrap CIs over documents).* Global fits over T = 256–16,384 for D3: prose 1.23 [1.19, 1.27], BPE prose 1.22 [1.17, 1.27], prose without KJV 1.25 [1.20, 1.30], code 1.01 [0.99, 1.03]. Local fits over T = 8k–32k: prose 1.11 [1.06, 1.17], BPE 1.09 [1.05, 1.14], code 0.95 [0.86, 1.04]. Under D0, code is 0.96 [0.94, 0.98] globally and 0.87 [0.78, 0.94] locally. **Correction to v1:** the first version said that N_ctx grows "at least linearly". That holds for prose (superlinear, because associations are still accumulating their second occurrence). It does not hold for code under D0, which is sublinear. Code under D3 is consistent with linear growth. The superlinear prose exponent cannot persist, because N_ctx ≤ N_bi ∝ T^{0.93}.

(c) *Corpus composition and tokenisation have little effect.* Dropping the KJV Bible changes N_ctx(32k) by −3% (D3) and the exponents by ≤ 0.02. Byte-level BPE gives the same N_ctx(32k) within CIs and a higher ρ (0.19 against 0.16), because subword continuations inside words recur.

(d) *Code remains about twice as recall-heavy as prose per token* (ρ_D3 0.28 against 0.16 at 32k), although N_ctx is similar.

### 5.5 Replay on real streams: the load-aware formula holds, the naive budget does not

| corpus | T | policy | N_ctx (D3) | max acc., d_v = 64 | S_half, square [naive S_req (4)] | ratio | MAE of Eq. (1) |
|---|---|---|---|---|---|---|---|
| prose | 1,024 | oracle | 37 | 0.86 | 1.6·10³ [7.9·10²] | 2.1 | 0.031 |
| prose | 4,096 | oracle | 191 | 0.48 | 1.2·10⁴ [4.1·10³] | 3.0 | 0.017 |
| prose | 16,384 | oracle | 993 | 0.26 | 1.3·10⁵ [2.1·10⁴] | 6.2 | 0.007 |
| prose | 1,024 | every-token | 37 | 0.48 | 1.9·10⁴ [2.2·10⁴] | 0.9 | 0.016 |
| prose | 4,096 | every-token | 191 | 0.26 | 3.2·10⁵ [8.7·10⁴] | 3.7 | 0.024 |
| prose | 16,384 | every-token | 993 | 0.09 | > 10⁶ (max acc. 0.20) [3.5·10⁵] | > 3 | 0.033 |
| code | 1,024 | oracle | 65 | 0.40 | 1.4·10⁴ [1.5·10³] | 8.7 | 0.022 |
| code | 4,096 | oracle | 237 | 0.33 | 3.0·10⁴ [5.6·10³] | 5.4 | 0.022 |
| code | 16,384 | oracle | 954 | 0.30 | 3.2·10⁵ [2.3·10⁴] | 14.2 | 0.023 |
| code | 1,024 | every-token | 65 | 0.25 | 3.4·10⁵ [2.4·10⁴] | 14.0 | 0.046 |
| code | 4,096 / 16,384 | every-token | 237 / 954 | 0.14 / 0.04 | > 10⁶ (max acc. 0.32 / 0.06) | – | 0.045 / 0.015 |

*Table 4 (figure: `results/fig_revision.png`(c); source: `results/replay_memory.json`; D3 queries; 5 windows per row; "MAE" is the mean absolute error between measured accuracy and the load-aware Eq. (1) over all S, for the square shape. Across all 48 conditions (2 corpora × 3 T × 2 policies × 2 definitions × 2 shapes) the MAE is 0.006–0.073, median 0.023).* D0 rows (in `replay_memory.json`) show the same pattern, with oracle ratios 1.3–13.7.

Three findings.

1. **The load-aware Eq. (1) predicts ideal-memory recall on real streams**, with Zipfian keys, repeated writes and V up to 149k. This is the strongest support we have for using ln V of a corpus vocabulary, the question raised in review. It holds for an ideal memory with random codes, not yet for a trained network.
2. **For oracle writing, the naive budget, Eq. (4), is optimistic by 1.3–14×, and more so at larger T and for code.** (For every-token writing the ratio ranges from 0.7 at T = 1k to > 3 at T ≥ 4k.) The cause is fan-out (Corollary 1). Frequent keys recur with many context-specific values, and their own-key load scales with d_v, not with S. At T = 16k, keys like "the" or Go's "." have dozens of stored successors.
3. **Value width caps recall.** With d_v = 64 (one typical head), recall of the context-specific set never reaches 50% for T ≥ 4,096 under either policy, even at S = 2^{20}. Maximum accuracy falls to 0.26–0.30 at 16k. For a single head, adding key dimensions cannot fix this. Only wider value paths (several heads storing the same association, or square states) or multi-token keys that split the fan-out can.

### 5.6 The corrected budget, and what it does and does not say about the KV cache

| S per layer | prose T\* every-token (Eq. 4) | prose T\* oracle, D0 / **D3** / D4 (Eq. 4) | code T\* every-token | code T\* oracle, D0 / **D3** / D4 |
|---|---|---|---|---|
| 2^14 | 770 | 8.4k / **12.8k** / 23k | 690 | 7.2k / **11.4k** / 18k |
| 2^16 | 3.1k | 32k / (45k)† / (72k)† | 2.8k | (35k)† / (51k)† / (70k)† |
| 2^18 | 12k | (121k)† / (155k)† / (233k)† | 11k | (175k)† / (219k)† / (258k)† |

*Table 5 (source: `results/analysis_v2.json`, η = 1).* † marks extrapolation beyond T = 32,768 with the local exponent. 95% bootstrap ranges for the interpolated D3 entries: prose 2^14: 11.8k–13.8k; code 2^14: 10.8k–12.0k. *These Eq. (4) values are upper bounds on T\* for an ideal memory.* Section 5.5 shows that the state actually needed is 2–6× (prose) and 5–14× (code) larger at T = 1k–16k. Read with that correction, a square 2^{16}-float state reaches 50% oracle recall of the context-specific set only up to T ≈ 11k (prose) and ≈ 6k (code), by log-interpolating the measured S_half, not the 45k and 51k of the naive D3 budget.

*Comparison with a KV cache.* At T = 16,384, measured S_half (oracle, square, D3) is 1.3·10⁵ floats (prose) and 3.2·10⁵ (code). One d = 4096 attention layer stores 2dT = 1.3·10⁸ floats, so the ratio is ≈ 1,000× (prose) and ≈ 400× (code). The first version reported 4,000× from the naive budget at 32k. We no longer state that figure. The comparison is deliberately *not* like for like. The fixed-state side holds only oracle-selected, previous-token-keyed associations at 50% recall. The KV cache also serves multi-token keys, multi-hop composition, positional retrieval and verbatim copying of spans, none of which we measure. The ratio is therefore a lower bound on what this narrow sub-task needs relative to one KV cache. It is not evidence that a compressed memory could replace attention in general.

## 6 Discussion and pre-registered predictions

The budget turns "can new architectures replace the Transformer?" into claims about measurable quantities: per-layer state and its shape (S, d_v), training efficiency (η), the load structure of the workload (D(T) together with the key fan-out), and the write policy. Within single-hop, previous-token-keyed recall, our experiments support three statements:

- *A fixed state must start forgetting context-specific associations at a context length predictable from these quantities.* The load-aware Eq. (1) predicts this for ideal memories on real streams. Demand grows roughly linearly with T, so there is no S at which a fixed state is recall-complete for all T. This is the quantitative counterpart of Jelassi et al. [2024] and Wen et al. [2025].
- *The update rule matters only when stored items lose value.* With retain-everything demand, gates and delta rules converge to additive storage. With superseded values, learned delta rules overwrite and win (Section 5.3).
- *Value width and key granularity matter as much as total state.* Real text has high-fan-out keys that a bigram-keyed memory with a narrow value path cannot resolve, at any d_k.

Our results do not establish that the recurrence-versus-attention question reduces to fixed-versus-growing memory. That is a hypothesis consistent with the above. Testing it requires pretrained models. We pre-register the following, all for single-hop recall tokens.

**P1 (knee).** In a pure fixed-state LM, loss on context-only recall tokens stops improving beyond a knee T_knee. With per-layer state S and head value width d_v, T_knee lies between the every-token T\* and the oracle T\* *after* the fan-out correction (measured by replay on that model's tokenizer). In a matched Transformer the same loss keeps falling. *Falsified if* the SSM's recall-token loss keeps improving beyond 4× the corrected oracle T\*, or saturates below the every-token T\*.

**P2 (state scaling).** Across otherwise identical fixed-state models differing only in state size, doubling S moves T_knee by a factor between 1.5 and 2.5. Doubling only d_k at fixed d_v moves it less than doubling d_v at fixed d_k (Corollary 1).

**P3 (domain).** At equal T and S, the SSM-versus-Transformer gap on recall tokens is larger for code than for prose, roughly in proportion to ρ_D3 (≈ 1.8× at 32k).

**P4 (hybrid sufficiency).** One attention layer removes the knee. Additional attention layers help mainly on multi-hop or multi-token-key recall.

**P5 (headroom).** A fixed-state layer with a non-linear or sparse read-out can beat the S/(2 ln V) law on capacity-isolating MQAR. Our four families do not exceed the frame-potential ceiling.

**Protocol (not run; no GPU or model downloads here).** (1) Matched families that share data and tokenizer, for example the released 8B Mamba-2, Mamba-2-Hybrid and Transformer checkpoints of Waleffe et al. [2024], and small open SSM/Transformer pairs. (2) On held-out long documents, mark recall tokens with a *surprisal-based* criterion: positions whose bigram occurred earlier in the context and whose continuation has probability < 0.01 under a context-free reference (a unigram-conditioned model trained on disjoint data, or the same LM given only the previous token). (3) Loss on recall and non-recall tokens as a function of distance to the previous occurrence and of total context T ∈ {1k, …, 128k}. (4) Predict T\* from each model's per-layer (d_k, d_v, heads) via the load-aware Eq. (1), with the write stream's (F, G) re-measured on its tokens. (5) Test P1–P3 with bootstrap CIs over documents.

## 7 Limitations

- **No pretrained-model evidence.** P1–P5 are untested. Every link of the budget is checked only at small scale: η on single-layer models with S ≤ 2,048 (d_k ≤ 32 with 3 seeds; linear attention at d_k = 64 never fell below 50% in the tested range); the ln V and fan-out dependence on *ideal* random-code memories replaying real text; demand on two small corpora. Whether trained multi-layer LMs, whose state serves many functions, reach η ≈ 1 on real text is the main open question.
- **Single-hop, previous-token keys.** N_ctx counts word- or token-level bigrams. Real recall involves multi-token keys, multi-token values and multi-hop composition. Multi-token keys would cut fan-out (making a fixed state look better than in Section 5.5), and multi-hop demand would add load (making it look worse). We measured neither.
- **Context-specificity is still a proxy.** D3 uses a smoothed bigram fitted on about 1M held-out words. A large LM would mark fewer tokens as context-only, so our demand is probably an over-estimate. D4 brackets this by roughly halving demand.
- **"Oracle" is not causal.** N_ctx uses future recurrence. Oracle T\* is an upper bound on what selective writing can achieve.
- **Task design biases Experiment B.** Pre-bound pairs and write masking isolate storage. The retain-everything result says nothing about multi-layer binding. We did not run a 2-layer task in which the key–value binding must be formed. The overwrite task covers one regime where gates matter, but not recency-weighted queries or interference with other functions of the state. The width control (d = 512) and the value sweep are single-seed. Learned gate values are averaged over pair tokens of one evaluation batch.
- **Approximations.** Eq. (1) is off by 17–19% at V = 41,509 for distinct keys, and by 17–41% at d_v ≤ 32. The frame-potential ceiling is a second-moment argument, not a rigorous bound on argmax accuracy.
- **Corpora.** 18 old books (the KJV Bible is ~37% of tokens; dropping it changes little) and one Go library. At T = 32k, 12 books and 38 packages contribute. There is no web text, dialogue or multi-file repository. T\* values beyond 32k are extrapolations.
- **Cited figures.** Layer ratios of production hybrids are taken from the cited papers and were not reproduced.

## 8 Conclusion

The S/(2 ln V) capacity scaling of additive superposition memory is classical. We specialised it to vocabulary read-outs, added a fan-out term and a learned-code ceiling, and tested the budget it implies. Trained single-layer fixed-state mixers reach the ideal additive capacity (1.00–1.07×, within the frame-potential ceiling) when everything must be retained, and the update rule matters once values are superseded. Real-text demand, measured with held-out criteria, grows roughly linearly with T, and 16–28% of tokens are context-only recall opportunities at 32k. On real token streams, the load-aware formula predicts ideal-memory recall closely, while the naive budget is optimistic by up to an order of magnitude because frequent keys fan out. The budget makes the replacement question concrete: what forgets, when, and which dimension of the state limits it. The decisive test on pretrained models is pre-registered here and remains to be done.

## References

- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. *Physical Review Letters*, 55(14), 1530–1533.
- Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou, J., Rudra, A., & Ré, C. (2024a). Simple linear attention language models balance the recall-throughput tradeoff. *ICML 2024*. arXiv:2402.18668.
- Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., & Ré, C. (2024b). Zoology: Measuring and improving recall in efficient language models. *ICLR 2024*. arXiv:2312.04927.
- Barnfield, N., Kim, J., Nichani, E., Lee, J. D., & Lu, Y. M. (2026). Sharp capacity thresholds in linear associative memory: From top-1 retrieval to tail-average learning. arXiv:2605.05189 (later version subtitled "From Winner-Take-All to Listwise Retrieval").
- Benedetto, J. J., & Fickus, M. (2003). Finite normalized tight frames. *Advances in Computational Mathematics*, 18, 357–385.
- Cabannes, V., Dohmatob, E., & Bietti, A. (2024). Scaling laws for associative memories. *ICLR 2024*. arXiv:2310.02984.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *ICML 2024*. arXiv:2405.21060.
- Giorlandino, A., Goldt, S., & Maillard, A. (2026). Sharp storage capacity of a simplified model of linear associative memory. *ICLR 2026*. Extended as "Factual recall in linear associative memories: sharp asymptotics and mechanistic insights", arXiv:2605.10795.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*. arXiv:2111.00396.
- Guo, H., Yang, S., Goel, T., Xing, E. P., Dao, T., & Kim, Y. (2026). Log-linear attention. *ICLR 2026*. arXiv:2506.04761.
- Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
- Jelassi, S., Brandfonbrener, D., Kakade, S. M., & Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. *ICML 2024*. arXiv:2402.01032.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *ICML 2020*. arXiv:2006.16236.
- Kimi Team (2025). Kimi Linear: An expressive, efficient attention architecture. arXiv:2510.26692.
- Kohonen, T. (1972). Correlation matrix memories. *IEEE Transactions on Computers*, C-21(4), 353–359.
- Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. arXiv:2403.19887.
- Lufkin, L., Figliolia, T., Millidge, B., & Krishnamurthy, K. (2026). Hybrid associative memories. arXiv:2603.22325.
- McEliece, R. J., Posner, E. C., Rodemich, E. R., & Venkatesh, S. S. (1987). The capacity of the Hopfield associative memory. *IEEE Transactions on Information Theory*, 33(4), 461–482.
- Merrill, W., Petty, J., & Sabharwal, A. (2024). The illusion of state in state-space models. *ICML 2024*. arXiv:2404.08819.
- MiniMax (2025). MiniMax-01: Scaling foundation models with lightning attention. arXiv:2501.08313.
- Nichani, E., Lee, J. D., & Bietti, A. (2025). Understanding factual recall in Transformers via associative memories. *ICLR 2025*. arXiv:2412.06538.
- NVIDIA (2025). Nemotron-H: A family of accurate and efficient hybrid Mamba-Transformer models. arXiv:2504.03624.
- Okpekpe, D., & Orvieto, A. (2025). Revisiting associative recall in modern recurrent models. arXiv:2508.19029.
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., et al. (2022). In-context learning and induction heads. arXiv:2209.11895.
- Qiao, M. (2025). Understanding Transformers through the lens of Pavlovian conditioning. arXiv:2508.08289 (later version titled "Attention as Conditioning: What Classical Learning Theory Predicts About Linear Transformers").
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers are secretly fast weight programmers. *ICML 2021*. arXiv:2102.11174.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762.
- Waleffe, R., et al. (2024). An empirical study of Mamba-based language models. arXiv:2406.07887.
- Wang, D., Zhu, R.-J., et al. (2025). A systematic analysis of hybrid linear attention. arXiv:2507.06457.
- Wen, K., Dang, X., & Lyu, K. (2025). RNNs are not Transformers (yet): The key bottleneck on in-context retrieval. *ICLR 2025*. arXiv:2402.18510.
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024a). Gated linear attention Transformers with hardware-efficient training. *ICML 2024*. arXiv:2312.06635.
- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024b). Parallelizing linear Transformers with the delta rule over sequence length. *NeurIPS 2024*. arXiv:2406.06484.
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with delta rule. *ICLR 2025*. arXiv:2412.06464.
