# What a Probe Gets for Free: Suffix Ceilings, Placement Floors, and When Decodable World States Are Evidence of a World Model

**Target venue:** NeurIPS 2027 (Main Track; Evaluations & Interpretability). Code and all raw outputs: `code/`, `results/` (round-2 tables in `results/summary_r2.md`).

## Abstract

A common argument that a sequence model "has a world model" is that a probe decodes the latent world state from its activations, and that the decoding beats an untrained-network baseline. We argue that the right comparison is with the *information an encoder gets for free from its architecture*, and we make this precise.

**Theory.** For any recurrent encoder with fading memory, and for any finite-receptive-field model, probe accuracy is bounded by the *k-suffix ceiling* Acc\*\_k: the Bayes accuracy attainable from the last k tokens, plus an explicit margin term (Theorem 1, Corollary 2). The ceiling is tight. An *untrained* contracting encoder with random tanh features attains Acc\*\_k exactly with a linear probe once it has one random feature unit per reachable k-suffix, plus a (k − 1)-unit delay line (Proposition 4). This suggests that what training buys is memory at sub-exponential width, not decodability as such.

**Gridworld with exact ceilings.** We test the theory in a torus gridworld where Acc\*\_k is computed exactly from the true finite-time prior.
- *Truncation test.* Theorem 1's prediction holds in all 324 cells, with a maximum excess of +0.002.
- *Untrained encoders.* Across 9 configurations of untrained d = 512 reservoirs, including orthogonal near-isometries that keep long memory, linear probes never exceed the 4-step ceiling. MLP probes reach at most 7 steps. (Two seeds per configuration and world are complete; the third seed finished one configuration only.)
- *Trained GRUs.* Across 40 independently trained GRUs, the trained models reach effective suffix lengths of 15–24 steps after 30 epochs.
- *Behaviour precedes state.* After 10 epochs the landmark-world GRUs predict the next observation with accuracy 0.959 ± 0.018, but their linear state probe reaches only 0.537 ± 0.196.
- *Interventions.* Across independent runs, the gain of probe-subspace interventions over random-subspace interventions tracks probe accuracy (r = 0.88, 95% CI [0.81, 0.93]).

**Othello.** For transformers the free information is different. Attention can read off an order-free, parity-split bag of the moves, that is, *where each player has placed discs*, without simulating a single flip.
- We trained a small Othello GPT on CPU. Its top-1 move is legal 92.4% of the time.
- Its linear board probe reaches 0.812 per square, against 0.698 for its own untrained initialisation, which is the conventional baseline.
- A decoder that sees only disc placement and the move index, and simulates no flips, already reaches 0.879. A nonlinear probe on the trained model reaches 0.835, so it stays below this floor as well.
- Stratifying by how long ago each square last changed shows that the probe's accuracy is roughly flat (0.61–0.63) over occupied squares that changed 1 to 16+ moves ago. In contrast, decoders limited to the last k moves fail precisely on old squares.
- These numbers describe one small model at one checkpoint (6,000 steps). A planned continuation run was interrupted and is not analysed, so we cannot say whether more training lifts the probe above the floor.

The untrained-twin baseline thus understated the free information by 18 points. A world-model claim needs to clear the floor appropriate to the architecture, stratified by how much history the answer requires.

## 中文摘要

判断序列模型"拥有世界模型"的常见论据是：探针能从激活中解码出潜在世界状态，而且解码准确率高于未训练网络这一基线。本文认为，正确的比较对象是*编码器凭其结构即可免费获得的信息*，并对此给出了精确刻画。

**理论。** 对任何具有衰减记忆的循环编码器，以及任何感受野有限的模型，探针准确率都不超过 *k-后缀上限* Acc\*\_k（仅凭最近 k 个词元可达到的贝叶斯准确率）加上一个显式的间隔项（定理 1、推论 2）。该上限是紧的：只要每个可达 k-后缀对应一个随机特征单元，再加上 k − 1 个延迟单元，带随机 tanh 特征的*未训练*压缩编码器配合线性探针即可恰好达到 Acc\*\_k（命题 4）。这提示：训练带来的是以次指数宽度实现的记忆，而不是可解码性本身。

**可精确计算上限的网格世界。** 我们在环面网格世界中检验该理论；在这里，Acc\*\_k 可依据真实的有限时间先验精确计算。
- *截断检验*：定理 1 的预测在全部 324 个单元中均成立，最大超出量为 +0.002。
- *未训练编码器*：在 9 种 d = 512 的未训练储备池配置中（包括保持长记忆的正交近等距映射），线性探针从未超过 4 步窗口的上限，MLP 探针最多达到 7 步。（每种配置、每个世界有两个随机种子完整跑完；第三个种子只完成了一种配置。）
- *训练过的 GRU*：在 40 个独立训练的 GRU 中，训练 30 轮后有效后缀长度达到 15–24 步。
- *行为先于状态*：训练 10 轮后，地标世界中的 GRU 预测下一观测的准确率为 0.959 ± 0.018，但线性状态探针仅为 0.537 ± 0.196。
- *干预*：在各独立运行之间，探针子空间干预相对随机子空间干预的增益与探针准确率高度相关（r = 0.88，95% 置信区间 [0.81, 0.93]）。

**黑白棋（Othello）。** 对 Transformer 而言，免费信息有所不同：注意力可以读出一个与顺序无关、按奇偶拆分的走子集合，即*双方各在哪些格子落过子*，而无需模拟任何一次翻转。
- 我们在 CPU 上训练了一个小型 Othello GPT，其首选走法合法的比例为 92.4%。
- 其线性棋盘探针的逐格准确率为 0.812，而作为常规基线的自身未训练初始化仅为 0.698。
- 一个只看落子位置与步数、完全不模拟翻转的解码器即可达到 0.879。训练模型上的非线性探针为 0.835，同样低于这一下限。
- 按格子距上次变化的时间分层可见：对于 1 步至 16 步以上之前发生变化的有子格，探针准确率大致持平（0.61–0.63）；而只看最近 k 步的解码器恰恰在"年老"的格子上失败。
- 以上数字只描述一个小模型的一个检查点（6,000 步）。原计划的续训被中断，未作分析，因此我们无法判断更多训练能否使探针超过该下限。

由此可见，未训练孪生模型这一基线把免费信息低估了 18 个百分点。世界模型的主张需要越过与模型结构相匹配的下限，并按答案所需历史的长短分层检验。

## 1 Introduction

Whether neural sequence models build an internal state of the world that generates their data, maintained and updated as the world evolves (cf. Ha & Schmidhuber, 2018), is a central open question. Much of the positive evidence comes from *probing*. Othello-GPT, a transformer trained only on move transcripts, carries board states that a probe can decode and that can be edited to change its moves (Li et al., 2023), and the board is *linearly* decodable relative to the player to move (Nanda, Lee & Wattenberg, 2023). Similar results exist for chess (Karvonen, 2024), space and time (Gurnee & Tegmark, 2024), entity states (Li, Nye & Andreas, 2021) and program states (Jin & Rinard, 2024). Skeptical work replies that next-token validity and probe success can coexist with incoherent implicit world models (Vafa et al., 2024) and with heuristic rather than law-like inductive biases (Vafa et al., 2025).

Probing has a well-known interpretive problem. A probe may learn the task itself (Hewitt & Liang, 2019), information may be encoded without being used (Ravichander, Belinkov & Hovy, 2021; Elazar et al., 2021), and untrained encoders can be strong feature extractors (Wieting & Kiela, 2019; Jaeger, 2001; Maass, Natschläger & Markram, 2002). The standard remedy is to compare with the same probe on a randomly initialised copy of the model, as Li et al. (2023) do.

This paper argues that the untrained-twin baseline answers the wrong question. It measures what one particular random network happens to expose, not what the *architecture* can expose without modelling the world. The second quantity can be much larger, and it can often be computed or estimated directly. We work it out for two encoder classes.

For **fading-memory encoders**, the free information is the recent input suffix. We bound probe accuracy by the Bayes accuracy of the last k tokens and show that the bound is attained by an untrained network of sufficient width. For **attention**, the free information is order-free. A transformer can aggregate *which* tokens occurred, split by position parity, without tracking any dynamics. In Othello this is the set of squares on which each player has placed a disc, and a decoder that sees only this sets a floor that a learned-board claim must clear.

**Contributions.**

1. **Suffix ceiling (Theorem 1, Corollary 2).** For any λ-contracting recurrent encoder and any linear probe, accuracy ≤ Acc\*\_k + P(margin ≤ L·δ\_t(k)) for every k. For finite-receptive-field models the bound holds with no margin term, for any probe.
2. **Tightness (Proposition 4, new).** Acc\*\_k is attained exactly by a linear probe on an *untrained* contracting tanh encoder with M random feature units and a (k − 1)-unit delay line, where M is the number of reachable k-suffixes. Proposition 3 gives free decodability of local states in a separability form.
3. **Exact ceilings and a stress test.** In a torus gridworld we compute Acc\*\_k from the true finite-time prior. We sweep nine untrained d = 512 reservoirs, including long-memory near-isometries, with linear and MLP probes, and compare them with 40 independently trained GRUs.
4. **Run-level intervention statistics.** Counterfactual rollout agreement (CRA) on discriminating steps, with random-subspace and full-swap controls, on 40 independent GRUs.
5. **A placement floor for attention, with age stratification, on Othello.** We train a small Othello GPT, probe it and its untrained twin, and compare both with learned floors, stratified by how long ago each square last changed. The untrained twin and the placement floor give opposite verdicts on the same model.
6. **Calibration of learned floors** against exact ceilings, which fixes the direction in which learned floors can be trusted.

We do not claim to settle whether large models have world models, and we make no claim about the original Othello-GPT. Our Othello model is small and partially trained, and its failure to clear the placement floor may reflect that (Section 9).

## 2 Related Work

**World representations in sequence models.** Li et al. (2023) introduced Othello-GPT and nonlinear probes with intervention-based evidence. Nanda et al. (2023) showed that a linear "mine/theirs" encoding exists and supports steering by vector arithmetic. Karvonen (2024) extended this to chess, including latent skill. Gurnee & Tegmark (2024) note that static spatial and temporal representations do not by themselves establish a dynamic world model. Jin & Rinard (2024) introduced an interventional baseline to separate what the LM represents from what the probe learns. Yuan & Søgaard (2025) revisit the Othello hypothesis across seven language-model architectures and report that the models learn highly similar board features, which they read as stronger evidence for the hypothesis. Their comparison is between models. Ours is between a model and what its architecture gets without modelling flips.

**Critiques of world-model evaluation.** Vafa et al. (2024) formalise world-model recovery for deterministic finite automata via the Myhill–Nerode theorem and show that models passing next-token and probe diagnostics can have incoherent implicit maps. Vafa et al. (2025) propose inductive-bias probes and find heuristic rather than law-like biases, including in Othello. They evaluate a model's *outputs* against the true automaton. We bound what *internal* probes can show. Brodjian, Hobley & Perona (2026) ask whether a generative model can be used as a simulator by editing its internal state, compare probe-based steering with a directly learned inverse map on Othello and a continuous environment, and find that which method works depends on the environment. Their editability criterion is close in spirit to our CRA.

**Floors for probe accuracy.** The closest work is Garg (2026), which appeared during our revision. It argues that a probe score is meaningful only relative to a floor computed from the input alone and a ceiling, and it builds floors from move summaries for the released Othello-GPT checkpoint, the richest of which excludes flips. Garg reports that on that checkpoint a linear probe in the mine/theirs encoding reaches 0.99 and recovers 92% of the headroom above the no-flip floor. We arrived at the placement floor independently, and we differ in three ways. Our floors are derived from the encoder class, with a theorem and a tightness result for the fading-memory case. They are computed exactly where the world allows, and the learned estimators are calibrated against the exact values. And we stratify by square age. Our Othello result also differs: our small model does not clear the floor. We read the two results together in Section 7.

**Probe methodology.** Control tasks (Hewitt & Liang, 2019) and MDL probing (Voita & Titov, 2020) address probe expressivity. Amnesic probing (Elazar et al., 2021) and Ravichander et al. (2021) address use versus encoding. Belinkov (2022) surveys these issues. Causal abstraction and interchange interventions (Geiger et al., 2021) give a framework for testing whether a representation plays the causal role of a high-level variable.

**Reservoir computing and fading memory.** Fading-memory operators can be approximated by finite-memory operators (Boyd & Chua, 1985). Echo-state networks with the echo-state property are universal approximators of fading-memory filters (Grigoryeva & Ortega, 2018). Theorem 1 is the probing-side counterpart, and Proposition 4 is a finite, exact version of the same universality for classification from a k-suffix. The capacity side relates to Cover's (1965) function-counting argument.

## 3 Theory

### 3.1 Setup

A world emits a token stream x\_1, x\_2, … and has a latent state s\_t ∈ 𝒮, a finite set, at each time t. An *encoder* maps each prefix x\_{1:t} to h\_t ∈ ℝ^d. A *linear probe* is f(h) = argmax\_{s∈𝒮} (w\_s·h + b\_s), with accuracy Acc(f) = P(f(h\_t) = s\_t) under the data distribution.

**Definition 1 (suffix ceiling).** Fix a time t. For 0 ≤ k ≤ t, a predictor is *k-suffix-measurable* if it is a function of x\_{t−k+1:t} (and of t) only. The k-suffix ceiling is

  Acc\*\_k = sup\_{g k-suffix-measurable} P(g(x\_{t−k+1:t}) = s\_t) = E[ max\_s P(s\_t = s | x\_{t−k+1:t}) ],

where the probability is under the *true* data distribution at time t, including any known start state. Acc\*\_k is non-decreasing in k. When the full history determines the state, Acc\*\_t = 1. Because the definition is at fixed t, predictors may depend on t, which covers absolute positional information. Numbers reported for a set of times are averages over those times.

**Definition 2 (fading-memory encoder).** A recurrent encoder h\_t = F(h\_{t−1}, x\_t) with fixed h\_0 is a *λ-contraction* on a set ℋ containing all reachable states if ‖F(h,x) − F(h′,x)‖ ≤ λ‖h − h′‖ for all x and all h, h′ ∈ ℋ, with λ < 1. Write D = sup\_t ‖h\_t − h\_0‖. The *restarted state* h\_t^{(k)} is the state obtained by running F from h\_0 on x\_{t−k+1:t} only.

An echo-state network h\_t = tanh(W h\_{t−1} + U x\_t + b) with spectral *norm* ‖W‖\_2 = λ is a λ-contraction, because tanh is 1-Lipschitz, and D ≤ √d when h\_0 = 0. The spectral *radius* does not give this guarantee, a distinction that matters in Section 5.1. Trained gated RNNs are generally not uniform contractions.

### 3.2 The suffix ceiling

**Lemma 1.** For a λ-contraction, δ\_t(k) := ‖h\_t − h\_t^{(k)}‖ ≤ λ^k ‖h\_{t−k} − h\_0‖ ≤ λ^k D.

*Proof.* Both trajectories receive x\_{t−k+1:t}. They start from h\_{t−k} and h\_0 respectively, and each step contracts their distance by λ. ∎

**Theorem 1 (suffix ceiling for fading-memory encoders).** Let f be a linear probe with weights (w\_s, b\_s). Let γ(h) be the gap between the largest and second-largest probe logits, and let L\_t = max\_s ‖w\_{f(h\_t)} − w\_s‖. Then for every k,

  Acc(f) ≤ Acc\*\_k + P( γ(h\_t) ≤ L\_t δ\_t(k) ) ≤ Acc\*\_k + P( γ(h\_t) ≤ L λ^k D ),

where L = max\_{s,s′} ‖w\_s − w\_{s′}‖.

*Proof.* Let ŝ = f(h\_t). For any s ≠ ŝ,
(w\_ŝ − w\_s)·h\_t^{(k)} + b\_ŝ − b\_s = [(w\_ŝ − w\_s)·h\_t + b\_ŝ − b\_s] − (w\_ŝ − w\_s)·(h\_t − h\_t^{(k)}) ≥ γ(h\_t) − L\_t δ\_t(k).
On the event E = {γ(h\_t) > L\_t δ\_t(k)} this is positive for all s ≠ ŝ, so f(h\_t^{(k)}) = f(h\_t). Hence P(f(h\_t) = s\_t) ≤ P(f(h\_t^{(k)}) = s\_t) + P(E^c). Since h\_t^{(k)} depends only on x\_{t−k+1:t}, the predictor f(h\_t^{(k)}) is k-suffix-measurable and its accuracy is at most Acc\*\_k. The second inequality follows from Lemma 1 and L\_t ≤ L. ∎

The first step of the proof does not use contraction: *for any recurrent encoder, the probe applied to the restarted state has accuracy at most Acc\*\_k*. Contraction is needed only to transfer this to the true state. The margin argument extends directly to Lipschitz probes.

**Corollary 2 (finite receptive field).** If h\_t depends on x\_{t−w+1:t} (and t) only, then Acc(f) ≤ Acc\*\_w for every probe f, linear or not. An n-layer transformer whose attention is masked to the last w′ tokens at every layer has receptive field n(w′ − 1) + 1 tokens, so the corollary applies with w = n(w′ − 1) + 1.

*Proof.* f(h\_t) is w-suffix-measurable. ∎

### 3.3 The ceiling is free: decodability without training

**Proposition 3 (free decodability of local states; separability form).** Suppose s\_t = φ(x\_{t−k+1:t}) almost surely, and let 𝒰 be the set of M reachable k-suffixes. Suppose the restarted features {h^{(k)}(u) : u ∈ 𝒰}, labelled by φ(u), are *multiclass linearly separable*: there are (w\_s, b\_s) with w\_{φ(u)}·h^{(k)}(u) + b\_{φ(u)} ≥ w\_s·h^{(k)}(u) + b\_s + γ\_0 for all u and all s ≠ φ(u), with γ\_0 > 0. Then this probe is perfect on restarted states. If the encoder is also a λ-contraction and γ\_0 > L λ^k D, then f(h\_t) = s\_t almost surely.

*Proof.* The first claim is the hypothesis. The second follows from the argument of Theorem 1 with the roles of h\_t and h\_t^{(k)} exchanged. ∎

*Sufficient condition.* If the M features are affinely independent, which requires d ≥ M − 1, the system w\_s·h^{(k)}(u) + b\_s = 1[φ(u) = s] is solvable and separability holds for every labelling φ. This condition is far from necessary. Round 1 invoked it where it did not apply. Section 5.1 now checks separability directly.

**Proposition 4 (tightness of the suffix ceiling).** Fix t ≥ k ≥ 1 and λ ∈ (0, 1). Let 𝒰 be the set of M reachable k-suffixes at time t and let m ≥ M. Consider the single-layer tanh encoder with state h = (z, r) ∈ ℝ^{(k−1)+m}:

- *Delay line.* z^1\_t = tanh(c(x\_t)) and z^j\_t = tanh(μ z^{j−1}\_{t−1}) for j = 2, …, k − 1, where c assigns distinct non-zero reals to the tokens and μ = λ/√2.
- *Random features.* r\_t = tanh(A z\_{t−1} + a·c(x\_t) + b), where the entries of A ∈ ℝ^{m×(k−1)}, a ∈ ℝ^m and b ∈ ℝ^m are drawn independently from continuous distributions, with those of A supported in an interval small enough that ‖A‖\_F ≤ λ/√2.

This encoder has the form h\_t = tanh(W h\_{t−1} + U x\_t + b̃) with ‖W‖\_2 ≤ λ, so it is a λ-contraction, and nothing in it is trained. With probability one over (A, a, b), there is a linear probe on h\_t with accuracy exactly Acc\*\_k. Combined with Theorem 1 (here δ\_t(k) = 0), Acc\*\_k is the exact maximum of linear-probe accuracy over this encoder family.

*Proof.* *Contraction.* The recurrent matrix is W = [[S, 0], [A, 0]], where S carries μ on the delay-line sub-diagonal. Then ‖W‖\_2² ≤ ‖S‖\_2² + ‖A‖\_2² ≤ μ² + ‖A‖\_F² ≤ λ².

*Finite memory.* With h\_0 = 0 and t ≥ k, z\_{t−1} is a function of x\_{t−k+1:t−1} and r\_t a function of x\_{t−k+1:t}. So h\_t = h\_t^{(k)}. The map u ↦ p(u) := (z\_{t−1}, c(x\_t)) ∈ ℝ^k is injective on k-suffixes, because c is injective and tanh and multiplication by μ are injective.

*Spanning.* Let u\_1, …, u\_M be the reachable suffixes and Φ the M × m matrix with Φ\_{ij} = tanh(α\_j·p(u\_i) + b\_j), where α\_j = (A\_j, a\_j) is the j-th row of weights. We show that Φ has rank M almost surely. It suffices to treat m = M and show det Φ ≠ 0. The determinant is real-analytic in (α, b), so it is either identically zero or non-zero outside a set of Lebesgue measure zero, hence non-zero almost surely under any continuous distribution on an open set. To exclude the first case, pick one direction α with the M values τ\_i = α·p(u\_i) pairwise distinct, which is possible because the p(u\_i) are distinct, and use it for every row. We claim the functions g\_i(b) = tanh(b + τ\_i) are linearly independent. Suppose Σ\_i β\_i g\_i ≡ 0 on ℝ. Differentiating gives Σ\_i β\_i sech²(b + τ\_i) ≡ 0, and by analytic continuation the identity holds as meromorphic functions of complex b. The function sech²(b + τ\_i) has a double pole at b = −τ\_i + iπ/2. Every pole of sech²(b + τ\_j) has real part −τ\_j ≠ −τ\_i, so no other term has a pole there. A function that is identically zero has no poles, so β\_i = 0 for every i. Linearly independent functions g\_1, …, g\_M admit points b\_1, …, b\_M with det[g\_i(b\_j)] ≠ 0. So det Φ is not identically zero.

*Probe.* Let g\* be a Bayes-optimal k-suffix decoder at time t. Since Φ has full row rank, for each s there is w\_s supported on the r-coordinates with Φ w\_s = (1[g\*(u\_i) = s])\_i. The resulting probe outputs a one-hot score at every reachable suffix and so equals g\*, with accuracy Acc\*\_k. ∎

*Remarks.* (i) The encoder is a structured reservoir, a delay line feeding random tanh features. We do not prove the statement for a dense i.i.d. recurrent matrix. (ii) The statement is at fixed t, because the encoder has no positional input. (iii) The proposition gives a *sufficient* width, M + k − 1, and M can grow exponentially in k (at most |V|^k). It does not prove that smaller random encoders must fail. Realising *every* labelling of M suffixes by a linear probe requires d ≥ M − 1, but for the specific Bayes labelling a lower bound is a conjecture, which the sweep in Section 5.2 is consistent with and does not establish. (iv) We did not instantiate the construction numerically.

The reading we draw: for fading-memory encoders, decodability up to Acc\*\_k is available without training, at a width cost that can be exponential in k. What a trained network can add is long memory at small width.

### 3.4 Diagnostics

**Truncation test (primary memory measure).** Evaluate the probe on the restarted state h\_t^{(k)} instead of h\_t. By the first step of Theorem 1, this accuracy must lie below Acc\*\_k for *any* encoder, which is the directly checkable prediction. For a contraction it also tracks full-state accuracy once the restart distance δ\_t(k) is small, so the k at which the two coincide measures the memory the probe relies on.

**Effective suffix length.** k\_eff(f) = min{k : Acc\*\_k ≥ Acc(f)}. With the exact ceiling of Definition 1, Acc\*\_t = 1, so k\_eff is always finite. Because k\_eff compares a possibly linear probe with a nonlinear Bayes ceiling, a small k\_eff is only a *lower bound* on the encoder's memory. We therefore report k\_eff for both linear and MLP probes where the distinction matters.

**Counterfactual rollout agreement (CRA).** Take a base prefix with state s at time t\_0 and a source prefix with state s′ ≠ s. Let P be the orthogonal projector onto the row space of the centred probe weights (rank ≤ |𝒮| − 1). Edit h ← h + P(h\_src − h), an interchange intervention (Geiger et al., 2021) restricted to the probe subspace. Then run the model *closed-loop* for R steps: at each step its predicted observation is fed back as input, followed by the base sequence's next action. CRA is the fraction of predictions that match the counterfactual world's observations, computed **only on discriminating steps**, where factual and counterfactual observations differ. Controls: no edit, an edit in a random subspace of equal rank, and a full-state swap (h ← h\_src), which bounds what any edit can achieve.

### 3.5 The free information of attention: placement floors

Theorem 1 says nothing about a full-context transformer, whose receptive field is the whole sequence. Its cheap information has a different shape. An attention head whose pattern depends on positions only can average the token embeddings of all earlier positions with the same parity as the query, and a second head those with the opposite parity. Together with the position embedding, which gives the count t, the result is an order-free, parity-split *bag* of the history.

In Othello each token is the square on which a disc is placed, and players alternate except for rare passes. The parity-split bag is then the **placement**: which squares each player has played. Placement determines exactly which squares are empty. It does not determine colours, because flips depend on move order.

**Definition 3 (floor).** For a summary σ of the history (the move index; the last k moves; the placement bag; combinations), the σ-floor is the accuracy of the best decoder of the state from σ(x\_{1:t}). With σ the last k tokens it is the suffix ceiling Acc\*\_k: the same quantity is a ceiling for a restricted encoder and a floor for a full-context model.

The argument above is a motivation and not a theorem about any particular network. It shows that placement is *available* to a shallow position-only attention computation, not that an untrained transformer exposes it. Ours does not (Section 6).

**Age stratification.** The age of a square at time t is the number of moves since it last changed by placement or flip. A decoder limited to the last k moves cannot know the colour of a square of age > k except through priors. A placement decoder knows who first played a square of any age, but not whether it was flipped since. Above-floor evidence for a maintained board must therefore appear on old squares that have been flipped.

## 4 Experimental Setup

All numbers come from code in `code/` run on CPU (4 shared cores, no GPU). Every number below traces to a file in `results/`. Nothing was regenerated when the manuscript was finalised. Incomplete runs are listed in Section 9.

**Gridworld.** A 5×5 torus (|𝒮| = 25). An agent starts at cell 0 and takes T = 24 uniformly random moves. Each sequence is the token stream obs\_0, m\_1, obs\_1, …, m\_T, obs\_T (49 tokens), where obs\_t is the colour of the cell occupied after move m\_t. The state probed at the position of token m\_t is the current cell. Three colourings, sampled per seed, control how local the state is. In **unique**, every cell has its own colour, so the last observation and move determine the state. In **two-colour**, a balanced random 2-colouring localises the agent gradually. In **3 landmarks**, three cells carry colour 1 and the rest colour 0, so the state is strongly non-local.

**Exact ceilings.** `world.suffix_bayes_exact` runs a forward filter over the last k (observation, move) pairs starting from the *exact finite-time marginal* of the walk from the known start cell. This is Definition 1's quantity, so Acc\*\_t = 1. Round 1 used the stationary uniform prior. All round-1 runs were re-scored without retraining (`recompute_exact.py`, `results/exact_ceiling.json`). Ceilings are averaged over t ∈ [12, 24]. For seed 0, Acc\*\_k at k = 1, 4, 8, 16 is 0.090, 0.474, 0.857, 0.996 in the two-colour world and 0.090, 0.222, 0.438, 0.892 in the landmark world.

**Encoders.** *Trained GRU*: one layer, 64 units, 32-dimensional embeddings, next-observation cross-entropy, Adam with a one-cycle schedule (peak 6·10⁻³), batch 128, 14,000 sequences, 10 or 30 epochs. *Untrained GRU*: the same architecture at initialisation. *Round-1 reservoirs*: tanh ESNs with spectral norm 0.9 (d = 64) and 0.99 (d = 512), unit input gain. *Sweep reservoirs* (`reservoir_sweep.py`): nine untrained d = 512 tanh reservoirs, namely a Gaussian matrix scaled to spectral *norm* 0.99 at input gain 1, Gaussian scaled to spectral *radius* 0.99 at gains 0.1 and 1, and λ·orthogonal matrices with λ ∈ {0.99, 1.0} at gains 0.1, 0.3 and 1. *Transformers*: two layers, d = 64, 4 heads, pre-LN, learned absolute positions, full causal context, 10 epochs (peak 2·10⁻³), and the same architecture untrained.

"Untrained encoder" always means an untrained encoder with *trained* linear readouts: the next-observation readout used in CRA rollouts and the state probe are both fitted on the probe-training split. Trained models use their own output head.

**Probes.** Multinomial logistic regression (L-BFGS, L2 = 10⁻⁴) on standardised hidden states, fitted on 6,000 sequences × 13 time steps and tested on 2,000 held-out sequences. A shuffled-label control stays at chance (0.038–0.042) for every round-1 encoder. In the sweep we also fit a one-hidden-layer MLP probe (512 ReLU units, 40 epochs).

**Truncation test.** For t ∈ {16, 20, 24} and k ∈ {1, 2, 3, 4, 6, 8, 10, 12, 16}, each recurrent encoder is restarted on the last 2k tokens and the *same* probe is applied.

**CRA.** t\_0 = 14, horizon R = 10, about 1,920 base/source pairs per run, probe-subspace rank 24. In the two-colour and landmark worlds a model that ignores the edit scores about 0.5 on discriminating steps. In the unique world it scores 0.

**Seeds.** Round-1 runs: seeds 0–2 for every world and recurrent encoder. Trained GRUs: 10 seeds × {two-colour, landmark} × {10, 30} epochs = 40 independent networks (`gru_seeds.py`), all with the same schedule. Sweep: seeds 0 and 1 are complete for all nine configurations in both non-local worlds. Seed 2 was terminated after the first configuration (norm-scaled, λ = 0.99) in both worlds. This gives 38 reservoirs instead of the planned 54. Transformers: seeds 0–2 in both non-local worlds.

**Othello.** We wrote an Othello engine (`code/othello/engine.py`) and generated games from the synthetic distribution of Li et al. (2023): each move is uniform over legal moves, and passes are not tokens. The vocabulary is the 60 non-centre squares. The model is a GPT with 4 layers, d = 128, 4 heads, pre-LN and learned positions, trained on 160,000 games for 6,000 steps (AdamW, one-cycle, peak 10⁻³, batch 128). The untrained twin is its exact initialisation. Each square is labelled empty, "mine" or "theirs" relative to the player who moves next (Nanda et al., 2023). We fit a linear probe at every residual layer and report the best layer, and an MLP probe (512 ReLU units) on the last layer. Probes are fitted on 5,000 labelled games and tested on the first 2,000 games of a separate 10,000-game set. Floors are decoders with a 512-unit embedding and two ReLU layers, trained with board supervision on 20,000 games and tested on all 10,000 test games. Their inputs are the move index t only; the last k moves in order plus t (k from 1 to 16); the parity-split placement bag plus t; and placement plus the last 8 moves plus t.

## 5 Results: Gridworld

### 5.1 Local states are free, and a correction to round 1

In the unique world (Acc\*\_1 = 1), every recurrent encoder decodes the state almost perfectly: probe accuracy is 1.000 for the trained GRU and both round-1 ESNs and 0.998–1.000 for the untrained GRU, with k\_eff = 1 throughout. The untrained encoders also pass the intervention test. A probe-subspace interchange makes their closed-loop rollouts follow the counterfactual on 0.926–0.950 (untrained GRU) and 1.000 (both ESNs) of discriminating steps, against 0.000–0.098 for a random subspace and 0.000 for no edit.

*Which proposition explains this.* There are M = 25 × 4 = 100 reachable 1-suffixes, so affine independence needs d ≥ 99, and the d = 64 encoders do not meet it. Proposition 3 in its separability form is what applies, and separability can be checked directly: the probe applied to restarted states reaches 1.000 for the d = 64 ESN at k = 1 in all three seeds, and 0.979, 0.982 and 0.990 for the untrained GRU at k = 2 (0.593–0.669 at k = 1). Only 25 classes have to be separated, not 100 arbitrary labels.

In a 1-local world a suffix cache and a world model are behaviourally identical, so neither probing nor intervention can distinguish them. The exception in round 1 was the untrained transformer, whose probe reached only 0.135 in the unique world (one seed): near-uniform attention at initialisation dilutes the last token. Free decodability depends on the architecture.

*Correction.* Round 1 reported that the ESN with "λ = 0.99" had an effective window of about 3 steps and attributed this to tanh saturation at unit input gain. That explanation was wrong for that reservoir. Its matrix was scaled to spectral *norm* 0.99. A Gaussian matrix scaled this way has a spectral radius of only about half its norm, so the reservoir contracts by roughly a factor of two per token regardless of saturation, and the round-1 phrase "nominal memory of 100 tokens" was a misreading. We did not archive the measured radii of the specific matrices, so we give no exact figure. The measured restart distance is consistent with this: δ(8) ≈ 1.2·10⁻⁶ for this reservoir. Saturation is real but separate. For the *orthogonal* reservoirs, whose radius equals their norm, δ(8) is 1.8–2.1 at input gain 0.1, about 0.5 at gain 0.3, and about 0.007 at gain 1. Unit input gain therefore does destroy long memory even in a near-isometry. Round 1's untrained encoders were all short-memory, for two different reasons, and its general claim about untrained encoders was not supported by them.

Re-scored against the exact ceiling, the round-1 encoders in the non-local worlds look as follows (three seeds each; probe accuracy and CRA unchanged from round 1, per-seed values in `results/exact_ceiling.json`). The untrained GRU and both norm-scaled ESNs have probe accuracy 0.110–0.168 (means) and k\_eff = 2 in every seed and both worlds. Their CRA is 0.508–0.528 under probe-subspace edits, 0.475–0.507 under random-subspace edits and 0.514–0.560 under a full swap, all close to the 0.5 of an edit-ignoring model. The 10-epoch trained GRUs have probe accuracy 0.804 ± 0.169 (two-colour) and 0.664 ± 0.307 (landmarks), with k\_eff of 20, 5, 8 and 5, 22, 15.

With the exact ceiling, round 1's "∞" category disappears. The trained GRUs that were labelled ∞ now score 20, 22 and 15, and the norm-0.99 ESN moves from 3 to 2.

*Truncation test against the exact ceiling.* Over all 324 (world, seed, encoder, k) cells of the round-1 runs, the largest excess of restarted-state probe accuracy over Acc\*\_k is +0.0022, and no cell exceeds 0.005. Under the stationary prior the same figure was +0.010, so most of the earlier excess was the approximation the reviewer identified. The a priori form of Theorem 1 was vacuous for these ESNs in round 1 and the data-dependent form loose. We did not recompute the bound infima against the exact ceiling and do not quote them.

### 5.2 Stress test: long-memory untrained reservoirs

The review's central objection was that a memory-tuned reservoir might hold many steps without training. Table 1 reports the sweep.

**Table 1.** Untrained d = 512 reservoirs. Each cell lists seed 0, seed 1. The three orthogonal λ = 1.0 configurations are omitted: they are within 0.011 in accuracy and one step in k\_eff of the λ = 0.99 rows at the same gain, with δ(8) of 2.1, 0.52–0.55 and 0.007–0.009 at gains 0.1, 0.3 and 1 (`results/sweep/`). δ(8) is the mean restart distance at k = 8. "norm" is Gaussian scaled to spectral norm, "radius" to spectral radius, "orth" is λ·orthogonal. Seed 2 exists only for the first row of each world (two-colour: linear 0.173, MLP 0.166, k\_eff 2 and 2; landmarks: 0.146, 0.139, k\_eff 2 and 2).

| World | W | λ | gain | linear probe | k\_eff lin | MLP probe | k\_eff MLP | δ(8) |
|---|---|---|---|---|---|---|---|---|
| two-colour | norm | 0.99 | 1.0 | 0.167, 0.164 | 2, 2 | 0.167, 0.159 | 2, 2 | 1.2·10⁻⁶ |
| two-colour | radius | 0.99 | 0.1 | 0.207, 0.139 | 3, 2 | 0.569, 0.495 | 5, 5 | 1.4 |
| two-colour | radius | 0.99 | 1.0 | 0.312, 0.284 | 4, 3 | 0.348, 0.304 | 4, 3 | 0.005 |
| two-colour | orth | 0.99 | 0.1 | 0.164, 0.115 | 2, 2 | 0.593, 0.514 | 5, 5 | 1.8 |
| two-colour | orth | 0.99 | 0.3 | 0.311, 0.232 | 4, 3 | 0.549, 0.495 | 5, 5 | 0.46–0.49 |
| two-colour | orth | 0.99 | 1.0 | 0.332, 0.299 | 4, 3 | 0.363, 0.327 | 4, 4 | 0.007–0.008 |
| landmarks | norm | 0.99 | 1.0 | 0.127, 0.171 | 2, 2 | 0.122, 0.161 | 2, 2 | 1.2·10⁻⁶ |
| landmarks | radius | 0.99 | 0.1 | 0.124, 0.159 | 2, 2 | 0.296, 0.359 | 6, 6 | 1.3 |
| landmarks | radius | 0.99 | 1.0 | 0.159, 0.241 | 3, 4 | 0.171, 0.243 | 3, 4 | 0.005–0.006 |
| landmarks | orth | 0.99 | 0.1 | 0.111, 0.132 | 2, 2 | 0.346, 0.416 | 7, 7 | 1.8 |
| landmarks | orth | 0.99 | 0.3 | 0.159, 0.227 | 3, 3 | 0.253, 0.351 | 5, 5 | 0.46–0.47 |
| landmarks | orth | 0.99 | 1.0 | 0.162, 0.242 | 3, 4 | 0.168, 0.248 | 3, 4 | 0.006–0.007 |

Four observations.

1. **Linear probes never exceed k\_eff = 4.** This holds for all 38 reservoirs, including those whose state still differs by δ(8) ≈ 2 from a restart eight steps back and so demonstrably carries older information.
2. **MLP probes on long-memory reservoirs extract noticeably more.** At input gain 0.1, MLP probes reach 0.495–0.593 (k\_eff = 5) in the two-colour world and 0.296–0.425 (k\_eff = 6–7) in the landmark world, while the linear probe on the same states stays at k\_eff = 2–3. The information is present and not linearly accessible. Proposition 4 leads one to expect this, since 512 units are far fewer than the reachable suffixes at k ≥ 5, but it does not prove that linear probes must fail there (Section 3.3, remark iii).
3. **Long memory and linear accessibility trade off.** Linear accuracy is highest at gain 1, where memory is short (δ(8) ≈ 0.007), and lowest at gain 0.1, where memory is long.
4. **No untrained reservoir is steerable.** Over all 38 reservoirs CRA(probe) is 0.495–0.572, CRA(random) 0.455–0.505, and even the full-state swap reaches only 0.504–0.595, close to the 0.5 that an edit-ignoring model obtains.

The restarted-state inequality holds throughout: over the 342 sweep cells the maximum excess over Acc\*\_k is +0.0039 and no cell exceeds 0.005. This includes reservoirs that are not strict contractions, as the first step of Theorem 1 predicts. For the long-memory reservoirs the restarted state is far from the true state and the fixed probe does poorly on it (for orthogonal λ = 1.0 at gain 0.1, accuracy 0.04–0.16 for every k up to 16). There, k\_eff under an MLP probe is the more informative memory measure.

*Does the criterion still separate trained from untrained encoders?* Against the 30-epoch GRUs of Section 5.3 (k\_eff 15–24), yes, by a wide margin and under both probe types. Against the 10-epoch GRUs the answer is qualified. Their linear-probe k\_eff ranges from 5 to 22, so the weakest of them (k\_eff 5–8) are above every *linear* reservoir result (≤ 4) but not above the best *MLP* reservoir results (5–7). Because one seed of the sweep is missing for eight of the nine configurations, the maxima of 4 and 7 are maxima over two seeds, not three.

### 5.3 Forty trained GRUs: behaviour, state and steerability

**Table 2.** Trained GRUs, 10 seeds per row, same one-cycle schedule. Mean ± s.d. over seeds.

| World | epochs | next-obs acc | probe acc | k\_eff exact (per seed) | seeds with k\_eff ≥ 12 | CRA probe | CRA random | CRA full swap |
|---|---|---|---|---|---|---|---|---|
| two-colour | 10 | 0.961 ± 0.042 | 0.840 ± 0.151 | 20, 5, 8, 12, 6, 11, 11, 14, 11, 8 | 3 | 0.650 ± 0.090 | 0.386 ± 0.047 | 0.908 ± 0.091 |
| two-colour | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 23, 20, 21, 23, 21, 24, 22, 23, 23, 24 | 10 | 0.847 ± 0.053 | 0.337 ± 0.050 | 1.000 ± 0.000 |
| landmarks | 10 | 0.959 ± 0.018 | 0.537 ± 0.196 | 5, 22, 15, 10, 7, 8, 9, 10, 11, 10 | 2 | 0.502 ± 0.088 | 0.411 ± 0.040 | 0.801 ± 0.106 |
| landmarks | 30 | 0.996 ± 0.006 | 0.960 ± 0.061 | 15, 24, 23, 17, 24, 23, 24, 17, 24, 23 | 10 | 0.770 ± 0.098 | 0.332 ± 0.072 | 0.978 ± 0.042 |

Seeds 0–2 at 10 epochs reproduce the round-1 networks: next-observation accuracy and the random-subspace, full-swap and no-edit CRA values are identical, and probe accuracy and CRA(probe) agree to within 0.0005 and 0.0013.

**Trained models exceed the ceiling of long windows.** After 30 epochs all 20 networks have k\_eff between 15 and 24. No untrained encoder in Section 5.2 approaches this. After 10 epochs only 5 of 20 reach k\_eff ≥ 12, and the per-seed spread is large.

**Behaviour precedes state.** In the landmark world, 10-epoch GRUs predict the next observation with accuracy 0.959 ± 0.018 while their state probe is at 0.537 ± 0.196. Next-observation accuracy is a weak measure here, because untrained reservoirs already reach 0.875–0.922 by predicting the majority colour. Landmark recall, the fraction of landmark hits predicted, is the more telling measure. The clearest single case is seed 5: next-observation accuracy 0.962 and landmark recall 0.815, with probe accuracy 0.394 and k\_eff = 8. Seed 0 has recall 0.607 with probe accuracy 0.250 and k\_eff = 5. Round 1 quoted recall 0.85 for this seed, which was the three-seed mean. Across the ten 10-epoch landmark seeds, recall ranges from 0.565 to 0.995. The 40-epoch training-dynamics runs of round 1 (constant learning rate; four runs), re-scored against the exact ceiling, show the same ordering within a run: at epoch 5, landmark seeds 0 and 2 have recall 0.861 and 0.856 with probe accuracy 0.338 and 0.362 (k\_eff = 7 for both). This matches the "task-specific heuristic" picture of Vafa et al. (2025), here located on the suffix curve.

**Interventions track decodability across independent runs.** Over the 40 networks, the intervention gain CRA(probe) − CRA(random) correlates with probe accuracy at Pearson r = 0.880 (95% bootstrap CI over runs [0.805, 0.928]; Spearman ρ = 0.902). This replaces round 1's r = 0.92 over 34 dependent checkpoints. All 29 runs with probe accuracy ≥ 0.75 have gain ≥ 0.17 (minimum 0.175). Of the 3 runs with probe accuracy below 0.4, two have gain ≤ 0 and the third has +0.012. A third run with negative gain (−0.021) has probe accuracy 0.530. So "probe edits are harmful before the state forms" is a tendency and not a rule.

**Probe-subspace edits underestimate causal use.** Over the runs with k\_eff ≥ 12, CRA(probe) ranges from 0.534 to 0.927 and CRA(random) from 0.207 to 0.452, while the full swap averages 1.000 (two-colour) and 0.978 (landmarks) in the 30-epoch runs. The probe's 24-dimensional row space captures most but not all of the causally used state. Round 1's abstract quoted the narrower 0.64–0.89, which held only for the training-dynamics checkpoints.

### 5.4 Transformers and the attention-window test

We trained transformers on three seeds per non-local world and applied the masked-attention truncation test, so that Corollary 2 is exercised (`tf_window.py`; per-cell values in `results/summary_r2.md`, Section 5). With full context, trained transformers reach probe accuracy 0.777, 0.598, 0.732 (two-colour; k\_eff 7, 5, 6) and 0.519, 0.549, 0.492 (landmarks; k\_eff 10, 9, 10), with next-observation accuracy 0.903–0.992. Untrained transformers have probe accuracy 0.055–0.067 and k\_eff = 1 in every run.

With attention restricted to the last w tokens in both layers, for w from 1 to 16, Corollary 2's bound holds in all 96 (world, seed, model, w) cells: the closest approach of windowed probe accuracy to Acc\*\_k, taking the better of the original and a refitted probe, is −0.041. But the bound is far from tight. Windowed accuracies are 0.03–0.13 at every window, for trained and untrained models alike, while Acc\*\_k rises to 0.84–1.00 at w = 16. Windowing a transformer trained with full context does not truncate its state computation. It breaks it. The attention-window test is therefore a valid check of Corollary 2 but not a usable memory measure for transformers, which is one reason we use placement floors for them. We still did not run CRA on transformers (Section 9).

## 6 Results: Othello

### 6.1 Calibrating learned floors

In Othello no ceiling or floor is available in closed form, so floors are estimated by training decoders. A learned decoder can only underestimate the true floor. To see by how much, we trained the same decoder family on the torus, where the exact ceiling is known (`calibrate_window_decoder.py`).

**Table 3.** Learned window decoder / exact Acc\*\_k on the torus.

| World | seed | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 |
|---|---|---|---|---|---|---|---|---|---|
| two-colour | 0 | 0.091/0.090 | 0.175/0.181 | 0.293/0.310 | 0.454/0.474 | 0.559/0.713 | 0.575/0.857 | 0.551/0.972 | 0.630/0.996 |
| two-colour | 1 | 0.092/0.090 | 0.180/0.180 | 0.294/0.307 | 0.459/0.482 | 0.538/0.743 | 0.513/0.884 | 0.453/0.982 | 0.496/0.998 |
| two-colour | 2 | 0.092/0.090 | 0.179/0.181 | 0.302/0.312 | 0.452/0.481 | 0.576/0.732 | 0.588/0.874 | 0.549/0.980 | 0.589/0.998 |
| landmarks | 0 | 0.091/0.090 | 0.137/0.138 | 0.175/0.176 | 0.208/0.222 | 0.264/0.323 | 0.290/0.438 | 0.350/0.704 | 0.487/0.892 |
| landmarks | 1 | 0.095/0.091 | 0.176/0.177 | 0.235/0.240 | 0.295/0.301 | 0.377/0.407 | 0.397/0.506 | 0.397/0.714 | 0.500/0.879 |
| landmarks | 2 | 0.088/0.088 | 0.147/0.149 | 0.191/0.197 | 0.238/0.246 | 0.305/0.336 | 0.322/0.427 | 0.345/0.642 | 0.493/0.842 |

For k ≤ 4 the decoders are within 0.029 of the exact ceiling (mean gap 0.006). For k ≥ 6 they fall short by 0.03 to 0.53. The mean gap over all cells is 0.143. Learned floors are therefore one-sided: a probe that fails to beat a learned floor also fails to beat the true one, but a probe that beats a learned floor may not beat the true one. We use the Othello floors only in the first direction. The size of the gap need not transfer to Othello. Its sign does.

### 6.2 The trained model against its untrained twin and against the floors

**Table 4.** Per-square accuracy (empty / mine / theirs), overall and by square age. Probes: 7.68M square-positions. Floors: 38.4M. Strata as fractions of the probe test set: empty 46.1%, age 0 5.1%, age 1–3 12.2%, age 4–7 11.7%, age 8–15 13.7%, age 16+ 11.2%.

| model / floor | all | empty | occupied | age 0 | age 1–3 | age 4–7 | age 8–15 | age 16+ |
|---|---|---|---|---|---|---|---|---|
| trained (6,000 steps), linear probe, layer 4 | 0.812 | 0.998 | 0.654 | 0.919 | 0.614 | 0.631 | 0.630 | 0.631 |
| trained, MLP probe, last layer | 0.835 | 0.999 | 0.694 | 0.921 | 0.664 | 0.677 | 0.674 | 0.666 |
| untrained twin, linear probe, layer 4 | 0.698 | 0.926 | 0.504 | 0.951 | 0.474 | 0.450 | 0.447 | 0.459 |
| untrained twin, MLP probe, last layer | 0.736 | 0.959 | 0.545 | 0.967 | 0.510 | 0.499 | 0.495 | 0.502 |
| floor: t only | 0.621 | 0.885 | 0.396 | 0.649 | 0.257 | 0.384 | 0.400 | 0.439 |
| floor: last 1 move + t | 0.651 | 0.885 | 0.451 | 0.932 | 0.405 | 0.382 | 0.393 | 0.429 |
| floor: last 4 moves + t | 0.719 | 0.890 | 0.573 | 0.870 | 0.843 | 0.427 | 0.432 | 0.469 |
| floor: last 8 moves + t | 0.767 | 0.905 | 0.650 | 0.845 | 0.800 | 0.808 | 0.443 | 0.482 |
| floor: last 16 moves + t | 0.818 | 0.934 | 0.719 | 0.833 | 0.786 | 0.779 | 0.765 | 0.475 |
| floor: placement + t | 0.879 | 1.000 | 0.776 | 0.790 | 0.733 | 0.764 | 0.786 | 0.818 |
| floor: placement + last 8 moves + t | 0.900 | 1.000 | 0.814 | 0.883 | 0.834 | 0.825 | 0.775 | 0.797 |

The trained model's top-1 next move is legal 92.4% of the time (first 500 test games), against 14.8% for the untrained twin. Linear probe accuracy rises from 0.641 at the embedding to 0.812 at layer 4 in the trained model, and from 0.589 to 0.698 in the twin.

**The conventional comparison supports a learned board.** The trained model's linear probe beats its untrained twin by 11 points overall (0.812 against 0.698) and by 15 points on occupied squares (0.654 against 0.504). The MLP probe shows a 10-point gap overall.

**The placement floor reverses the verdict.** A decoder that sees only which squares each player has played, and the move index, reaches 0.879. That is 7 points above the trained model's linear probe and 4 points above its MLP probe. On occupied squares the floor is 0.776 against 0.654 and 0.694. The untrained twin understates the free information by 18 points with a linear probe (0.879 against 0.698) and by 14 with an MLP probe. Adding the last 8 moves to placement raises the floor to 0.900.

**No stratum is above the floor.** The probe is above the placement floor only on squares changed by the current move (0.919 against 0.790), and there the last-move floor is higher still (0.932). On empty squares placement is exact (1.000 against 0.998). On every older stratum the best floor exceeds both probes by 11 to 23 points.

**The age profile shows what kind of information the probe uses.** In aggregate the linear probe (0.812) is close to a decoder that sees the last 16 moves (0.818). Their age profiles are opposite. The window decoder is accurate on squares inside its window (0.77–0.83) and falls to 0.475 on squares of age 16 or more. The probe is roughly flat at 0.61–0.63 from age 1 to age 16+. On old squares the probe is well above every window decoder (0.631 against 0.475), so the model does carry information older than 16 moves. But it stays below what placement alone gives on those squares (0.818). One reading is that this model has learned a partial, order-free summary of the history and has not learned to track flips. We did not test that reading directly.

**Attention windowing breaks the model.** With attention masked to the last w tokens in every layer, refitted probe accuracy rises from 0.638 (w = 1) to 0.749 (w = 16), and the legal-move rate from 0.354 to 0.744. At w = 16 the receptive field already covers the whole game (61 tokens), yet accuracy is 6 points below the unmasked model. As on the torus, Corollary 2 holds and the windowed model is not a faithful truncation.

**What this does and does not show.** For this model, the standard evidence (trained beats untrained twin) is accounted for by information that needs no flip simulation. This does not show that Othello-trained transformers lack board representations. The model has 6,000 steps of CPU training and a 92.4% legal-move rate, well below published Othello-GPT models, and Garg (2026) reports that the released checkpoint clears a no-flip floor comfortably. A continuation of training was started from the 6,000-step checkpoint with a planned 4,000 further steps. Its log ends at step 2,200, its schedule was shortened once during the run, and it was never analysed. We report nothing from it.

## 7 Discussion

**Three levels of evidence.** (a) *Decodable*: a probe recovers s\_t. This is free for fading-memory encoders up to the suffix ceiling of their window, and for attention a large share of the Othello board is free through placement. (b) *Decodable beyond the free level*: probe accuracy exceeds the ceiling or floor appropriate to the architecture. In the gridworld only trained GRUs reach k\_eff ≥ 12. In Othello our small model does not reach this level. (c) *Causally maintained*: editing the probed subspace changes multi-step closed-loop behaviour as the counterfactual world would, and beats a random subspace. In the gridworld, (b) and (c) rise together across 40 independent runs (r = 0.88), while behavioural competence comes earlier.

**The untrained twin is the wrong control.** An untrained network is one sample from the architecture, and it may expose much less than the architecture can. In the gridworld, the untrained GRU and the norm-scaled ESN have k\_eff = 2 where a tuned reservoir with an MLP probe reaches 5–7. In Othello, the untrained transformer's probes reach 0.70–0.74 where placement gives 0.88. In both cases a trained model can beat its twin by learning to expose free information better, without learning dynamics. A floor defined by the input summary the architecture can cheaply compute does not depend on the accident of initialisation.

**Reading our Othello result next to Garg (2026).** Garg reports that the released Othello-GPT clears a no-flip floor in the mine/theirs encoding, and we find that a small, briefly trained model does not. These are compatible. Together they suggest a training-time ordering like the one in the gridworld: legal-move competence and twin-beating probes first, above-floor board tracking later, if at all. With one checkpoint this is a hypothesis.

**Why stratify by age.** Aggregate accuracy hides the comparison that matters. Our probe and the 16-move window decoder agree to within 0.006 in aggregate and disagree by 16 points on old squares.

**Learned floors are one-sided.** Table 3 shows learned window decoders up to 0.53 below the exact ceiling. Reporting "the probe beats a window decoder" as positive evidence is therefore unsafe unless the decoder is known to be near-optimal. Our Othello conclusion is in the safe direction: the probe is below floors that are themselves underestimates.

## 8 Predictions and a Protocol for Larger Models

We could not run large models. The protocol below states predictions that could falsify our framing.

1. **Report the floor appropriate to the architecture.** For recurrent or finite-context models, estimate Acc\*\_k with the strongest affordable window decoder. For full-context transformers, also train decoders on order-free summaries such as the placement bag. *Prediction P1:* for partially trained Othello models with legal-move rates up to at least 92%, the mine/theirs probe stays below the placement floor, as in Table 4.
2. **Stratify by age.** *Prediction P2:* when a model does clear the placement floor, the excess is concentrated on old squares that were flipped after placement, because those are the squares placement gets wrong.
3. **Interventions with controls.** *Prediction P3:* on steps where factual and counterfactual legal-move sets differ, CRA(probe) lies strictly between CRA(random) and CRA(full). In our 40 GRUs with k\_eff ≥ 12 the ranges were 0.534–0.927 against 0.207–0.452, with the full swap near 1.0 after 30 epochs.
4. **Track training.** *Prediction P4:* along a checkpoint series, twin-beating probe accuracy and high legal-move accuracy appear before above-floor accuracy on old squares, and CRA gain rises with the latter.
5. **Calibrate the floor estimator** where the true value is known, and state the direction in which it can be trusted.

P1 describes our own regime and could fail with a different model size or optimiser. P2 and P4 are untested. A model that cleared the placement floor uniformly across ages, or before its legal-move rate was high, would contradict our reading.

## 9 Limitations

- **One Othello model, one checkpoint, small and partially trained.** The GPT has 4 layers and d = 128, was trained for 6,000 steps on CPU, and has a 92.4% legal-move rate. We make no claim about the original Othello-GPT, for which Garg (2026) reports the opposite outcome against a similar floor. The continuation run was interrupted and is not analysed, so we have no evidence on how the floor comparison changes with training. There is one training seed.
- **Incomplete reservoir sweep.** Seed 2 was terminated after the first of nine configurations in both worlds. The sweep covers 38 of 54 planned reservoirs, and for eight configurations the results rest on two seeds. The archived aggregate table in `results/summary_r2.md` (Section 3) lists only the one configuration that all three seeds share, under a "mean over 3 seeds" heading. Table 1 here is built from the per-seed files in `results/sweep/`.
- **The sweep is not exhaustive.** Nine configurations at one width (512), no leaky units, no input gains below 0.1, and one MLP probe architecture. A better-tuned reservoir or a stronger probe could reach longer windows, and the gap to 10-epoch GRUs is already small.
- **Proposition 4 covers a structured reservoir and gives sufficiency only.** It concerns a delay line with random features and not a dense random recurrent matrix. It was not instantiated numerically. It does not prove that linear probes on narrower random encoders fail, so "training buys memory at sub-exponential width" is an interpretation supported by Table 1 and not a theorem.
- **Unarchived quantities.** The spectral radii of the norm-scaled reservoir matrices were not saved, and the Theorem 1 bound infima were not recomputed against the exact ceiling.
- **Learned floors underestimate.** Table 3 shows gaps of up to 0.53 on the torus at k ≥ 6. Othello floors support only the claim that a probe is below the floor. The placement decoder uses move parity as a proxy for player, which passes make slightly inexact. This too can only lower the floor.
- **Mismatched test sets and layer selection in Othello.** Probes are tested on 2,000 games and floors on 10,000 games from the same generator. The reported linear-probe layer is the one with the best test accuracy among five, a small optimistic bias in the probe's favour. Legal-move rates are from 500 games. No confidence intervals are given for Table 4.
- **No CRA on transformers.** Editing one position's residual stream in a causal transformer does not propagate as a recurrent state edit does. A proper treatment needs edits at all later positions or in the key/value cache. The attention-window test, which we did run, turned out not to be a usable memory measure because windowing breaks the model.
- **Scale and world family.** Gridworld models are tiny (64-unit GRUs, 2-layer transformers) and the world has 25 states. Trained GRUs are highly variable across seeds at 10 epochs. The behaviour-before-state ordering is shown for GRUs in one world family.
- **Theorem scope.** Theorem 1 covers uniformly contracting recurrent encoders and linear (or Lipschitz) probes. For trained RNNs, radius-scaled or orthogonal reservoirs, and full-context transformers, k\_eff and the floors are empirical diagnostics and not guarantees. Only the restarted-state inequality is general.
- **No reruns at finalisation.** The final editing environment could not execute the code. Numbers were checked against the archived result files, not by re-execution.

## 10 Conclusion

A probe result is evidence of a world model only above what the encoder's architecture provides for free. For fading-memory encoders that level is the suffix ceiling, which we bounded, showed to be attainable without training at sufficient width, and checked against exact values: restarted-state accuracy never exceeded it by more than 0.004 in 666 cells. Untrained reservoirs stayed within a 4-step window under linear probes and a 7-step window under MLP probes in the 38 reservoirs we completed, while trained GRUs reached 15–24 steps after 30 epochs and became steerable in proportion to their decodability. For attention the free level includes order-free summaries. In Othello, disc placement alone predicts 0.879 of the board, above both the linear (0.812) and nonlinear (0.835) probes of a small trained model that beats its untrained twin by 11 points. The twin baseline and the placement floor disagree about the same model. These are small-scale results with one Othello checkpoint, and the published evidence on larger Othello models points the other way for fully trained networks. What we offer is the comparison itself: derive the floor from the architecture, estimate it conservatively, and stratify by how much history the answer needs.

## Reproducibility

`code/README.md` gives the run commands. Raw outputs are in `results/` (`raw_*.json`, `dyn_*.json`, `exact_ceiling.json`, `seeds/`, `sweep/`, `tf/`, `calib_window.json`, `othello/`, logs in `logs/`), with tables in `summary_r2.md` and `summary_othello.md` and figures `fig1`–`fig7`. The README lists analysis commands for the continuation checkpoint `gpt_long.pt`. Those were not run, and no `analysis_long.json` exists. The sweep logs end with "Terminated" during seed 2.

## References

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219. https://aclanthology.org/2022.cl-1.7/

Boyd, S., & Chua, L. O. (1985). Fading memory and the problem of approximating nonlinear operators with Volterra series. *IEEE Transactions on Circuits and Systems*, CAS-32(11), 1150–1161. doi:10.1109/TCS.1985.1085649

Brodjian, S., Hobley, M., & Perona, P. (2026). When generative models act as simulators. arXiv:2610.12847

Cover, T. M. (1965). Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. *IEEE Transactions on Electronic Computers*, EC-14(3), 326–334. doi:10.1109/PGEC.1965.264137

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics*, 9, 160–175. doi:10.1162/tacl_a_00359

Garg, P. (2026). How high is 0.6? Floors, ceilings, and headroom in interpretability probing. arXiv:2610.08544

Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. arXiv:2106.02997

Grigoryeva, L., & Ortega, J.-P. (2018). Echo state networks are universal. *Neural Networks*, 108, 495–508. doi:10.1016/j.neunet.2018.08.025

Gurnee, W., & Tegmark, M. (2024). Language models represent space and time. *International Conference on Learning Representations (ICLR 2024)*. arXiv:2310.02207

Ha, D., & Schmidhuber, J. (2018). World models. arXiv:1803.10122

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *Proceedings of EMNLP-IJCNLP 2019*, 2733–2743. https://aclanthology.org/D19-1275/

Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology.

Jin, C., & Rinard, M. (2024). Emergent representations of program semantics in language models trained on programs. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235, 22160–22184. arXiv:2305.11169

Karvonen, A. (2024). Emergent world models and latent variable estimation in chess-playing language models. *Conference on Language Modeling (COLM 2024)*. arXiv:2403.15498

Li, B. Z., Nye, M., & Andreas, J. (2021). Implicit representations of meaning in neural language models. *Proceedings of ACL-IJCNLP 2021*, 1813–1827. doi:10.18653/v1/2021.acl-long.143

Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent world representations: Exploring a sequence model trained on a synthetic task. *International Conference on Learning Representations (ICLR 2023)*. arXiv:2210.13382

Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states: A new framework for neural computation based on perturbations. *Neural Computation*, 14(11), 2531–2560.

Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. *Proceedings of the 6th BlackboxNLP Workshop*, 16–30. arXiv:2309.00941

Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? *Proceedings of EACL 2021*, 3363–3377. doi:10.18653/v1/2021.eacl-main.295

Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the world model implicit in a generative model. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2406.03689

Vafa, K., Chang, P. G., Rambachan, A., & Mullainathan, S. (2025). What has a foundation model found? Using inductive bias to probe for world models. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, PMLR 267, 60727–60747. arXiv:2507.06952

Voita, E., & Titov, I. (2020). Information-theoretic probing with minimum description length. *Proceedings of EMNLP 2020*, 183–196. doi:10.18653/v1/2020.emnlp-main.14

Wieting, J., & Kiela, D. (2019). No training required: Exploring random encoders for sentence classification. *International Conference on Learning Representations (ICLR 2019)*. arXiv:1901.10444

Yuan, Y., & Søgaard, A. (2025). Revisiting the Othello world model hypothesis. *ICLR 2025 Workshop on World Models*. arXiv:2503.04421
