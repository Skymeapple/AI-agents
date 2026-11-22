# Decodable Is Not Modelled: Suffix Ceilings and Counterfactual Rollouts for Testing World-Model Claims in Sequence Models

**Target venue:** NeurIPS 2027 (Main Track; Evaluations & Interpretability). Code and all raw outputs: `code/`, `results/`.

## Abstract

A common argument that a sequence model "has a world model" is that a probe can decode the latent world state from its hidden activations. We show that this argument has a precise blind spot and we give a way around it. First, we prove a *suffix ceiling*: for any recurrent encoder with fading memory (a uniform λ-contraction in its state), the accuracy of any linear probe is at most the Bayes accuracy attainable from the last k input tokens, Acc\*\_k, plus the probability that the probe's margin is smaller than an explicit, computable perturbation term that decays like λ^k. For finite-context models the ceiling holds with no correction term. Conversely, when the world state is a function of a short input suffix, any sufficiently wide encoder in general position, including an untrained one, decodes it perfectly. Probe accuracy is therefore evidence of a world model only insofar as it exceeds the suffix ceiling of the encoder's effective memory. From this we derive two quantities: the *effective suffix length* k\_eff, the shortest history window whose Bayes-optimal decoder matches the probe, and a *counterfactual rollout agreement* (CRA) score, which measures whether an interchange intervention restricted to the probe subspace makes the model's own closed-loop predictions follow the counterfactual world trajectory. We test them on a controlled torus gridworld in which we can compute Acc\*\_k exactly and can tune how local the latent state is. The encoders are trained GRUs, untrained GRUs, echo-state reservoirs with provable contraction, and small transformers. With three seeds per condition, all untrained and fading-memory encoders have k\_eff of 2–3 steps when the state is non-local (probe accuracy 0.11–0.17). Yet when the state is 1-local, the same untrained encoders reach probe accuracy ≥ 0.999 and pass the intervention test (CRA 0.94–1.00). Across 324 truncation cells, restarted-state probe accuracy never exceeds Acc\*\_k by more than 0.010. Trained GRUs exceed the ceiling of every history window only after a heuristic phase in which next-observation behaviour is already good (e.g., landmark recall 0.86 with probe accuracy 0.34). Before this transition, probe-subspace interventions do worse than random-subspace controls in 6 of 8 checkpoints. After it, they reach CRA 0.64–0.89 against 0.25–0.42 for random subspaces and 1.00 for full-state swaps. On CPU-scale models, the conclusion is that "the state is linearly decodable" and "the model maintains and updates a state" come apart, and that the regimes where they come apart can be predicted ahead of time from the world's suffix curve.

## 中文摘要

判断序列模型"拥有世界模型"的一个常见论据是：探针（probe）能够从隐藏激活中解码出潜在的世界状态。本文指出这一论据存在一个可精确刻画的盲点，并给出补救方法。第一，我们证明了"后缀上限"定理：对于任何具有衰减记忆（状态映射为一致 λ-压缩）的循环编码器，任意线性探针的准确率不超过仅凭最近 k 个输入词元所能达到的贝叶斯准确率 Acc\*\_k，再加上探针间隔小于一个随 λ^k 衰减、可显式计算的扰动项的概率；对有限上下文模型，该上限无需修正项即严格成立。反过来，当世界状态是短后缀的函数时，任何足够宽且处于一般位置的编码器（包括未经训练的随机网络）都能完美解码该状态。因此，探针准确率只有在超出编码器有效记忆所对应的后缀上限时，才构成世界模型的证据。据此我们提出两个量：有效后缀长度 k\_eff（其贝叶斯最优解码器与探针准确率相当的最短历史窗口），以及反事实展开一致性（CRA）：将交换干预限制在探针子空间内，检验模型自身的闭环预测是否随之沿反事实世界轨迹演化。我们在一个可精确计算 Acc\*\_k、且可调节潜在状态"局部性"的环面网格世界中，对训练过的 GRU、未训练 GRU、具有可证明压缩性的回声状态网络以及小型 Transformer 进行了检验。在每个条件三个随机种子的实验中：当状态非局部时，所有未训练及衰减记忆编码器的 k\_eff 仅为 2–3 步（探针准确率 0.11–0.17）；而当状态为 1-局部时，同样的未训练编码器探针准确率 ≥ 0.999，并能通过干预检验（CRA 0.94–1.00）。在全部 324 个截断单元中，重启状态上的探针准确率超出 Acc\*\_k 的幅度从未大于 0.010。训练后的 GRU 只有在经历一个"启发式阶段"之后才会超越所有历史窗口的上限；在该阶段，下一观测的预测已经很好（例如地标召回率 0.86，而探针准确率仅 0.34），探针子空间干预在 8 个检查点中有 6 个不如随机子空间对照；转变之后，CRA 达到 0.64–0.89，而随机子空间为 0.25–0.42，完整状态替换为 1.00。结论是：在 CPU 规模的模型上，"状态可被线性解码"与"模型维护并更新状态"是可以分离的，而且二者何时分离可以事先由世界的后缀曲线预测。

## 1 Introduction

Whether neural sequence models build internal representations of the world that generates their data, in the sense of a learned state that is maintained and updated as the world evolves (cf. the explicit world models of Ha & Schmidhuber, 2018), is one of the central open questions about modern AI. Much of the positive evidence comes from *probing*. Othello-GPT, a transformer trained only on move transcripts, carries board states that a probe can decode, and editing those internal states changes its move predictions (Li et al., 2023). The board turns out to be *linearly* decodable once it is written relative to the player to move (Nanda, Lee & Wattenberg, 2023). Chess models trained on PGN strings show similar results and also encode latent player skill (Karvonen, 2024). Language models carry linearly decodable spatial and temporal coordinates of entities (Gurnee & Tegmark, 2024), entity states in discourse (Li, Nye & Andreas, 2021), and program states in models trained on code (Jin & Rinard, 2024). Skeptical work replies that next-token validity and probe success can coexist with incoherent implicit world models (Vafa et al., 2024), and that foundation models' inductive biases often track task-specific heuristics rather than the generating law (Vafa et al., 2025).

Probing has a well-known interpretive problem. A probe may learn the task itself instead of reading it off (Hewitt & Liang, 2019). Information may be encoded without being used (Ravichander, Belinkov & Hovy, 2021; Elazar et al., 2021). And untrained, random encoders can be surprisingly strong feature extractors (Wieting & Kiela, 2019), which echoes the reservoir-computing finding that the transient dynamics of a fixed random network keep a rich trace of the recent input (Jaeger, 2001; Maass, Natschläger & Markram, 2002). Li et al. (2023) themselves report that probes on a randomly initialised Othello-GPT beat a constant baseline. What has been missing is a quantitative account of *how much* decodability an encoder gets for free from the structure of the input stream alone, stated in a form that can be subtracted from a probe result.

This paper supplies that account for a large and natural class of encoders, and turns it into a testing protocol. The key observation is simple. If the latent world state is (nearly) a function of the last few tokens, then any encoder that keeps a faithful trace of the last few tokens will "contain" the state, whether or not it models the world. Conversely, an encoder whose memory fades cannot contain more state information than its effective window allows. Both directions can be made exact.

**Contributions.**

1. **Suffix ceiling (Theorem 1).** For any recurrent encoder whose state update is a uniform λ-contraction, and any linear probe, probe accuracy ≤ Acc\*\_k + P(margin ≤ L·δ\_t(k)) for every k. Here Acc\*\_k is the Bayes accuracy of the best predictor of the state from the last k steps, δ\_t(k) is the distance between the true state and the state of the encoder restarted k steps back, and δ\_t(k) ≤ λ^k D. For finite-context models the bound holds exactly (Corollary 2).
2. **Free decodability (Proposition 3).** If the world state is a function of the last k tokens, any encoder whose k-suffix features are in affine general position with width d ≥ M−1 (M = number of reachable suffixes) admits a linear probe with perfect accuracy, regardless of training.
3. **Two diagnostic quantities.** The *effective suffix length* k\_eff is the smallest k with Acc\*\_k ≥ probe accuracy. *Counterfactual rollout agreement* (CRA) is the fraction of future steps, among those where factual and counterfactual worlds disagree, on which the model's closed-loop predictions follow the counterfactual after an interchange intervention restricted to the probe's row space. We pair CRA with random-subspace and full-swap controls.
4. **A controlled testbed with exact ceilings.** In a coloured 5×5 torus gridworld, the k-suffix posterior can be computed exactly by an HMM filter. Three colourings span "state is 1-local" to "state is strongly non-local". We evaluate trained GRUs, untrained GRUs, echo-state networks with spectral norm λ (for which Theorem 1 applies rigorously), and trained and untrained transformers, over three seeds, and we check the theorem's bounds numerically.
5. **A pre-registerable protocol** for applying the same diagnostics to Othello-, chess- and navigation-trained transformers (Section 7).

We do not claim to settle whether large models have world models. The contribution is a sharper criterion: the part of a decodability result that a world model is needed to explain is the part above the suffix ceiling, and the part of an intervention result that tests a world model is the part on counterfactually discriminating future steps.

## 2 Related Work

**World representations in sequence models.** Li et al. (2023) introduced Othello-GPT and nonlinear probes with intervention-based evidence. Nanda et al. (2023) showed that a linear "mine/theirs" encoding exists and supports steering by vector arithmetic. Karvonen (2024) extended this to chess, including latent skill. Gurnee & Tegmark (2024) probed Llama-2 models for space and time, and note that such representations do not by themselves establish a dynamic causal world model. Li, Nye & Andreas (2021) found linearly decodable, causally editable entity-state representations in BART and T5. Jin & Rinard (2024) found program-state representations in models trained on a grid-navigation DSL and introduced an interventional baseline to separate what the LM represents from what the probe learns. Our setting, a gridworld with move and observation tokens, is deliberately close to these navigation and board settings, but small enough that the Bayes ceiling is exact.

**Critiques of world-model evaluation.** Vafa et al. (2024) formalise world-model recovery for deterministic finite automata via the Myhill–Nerode theorem. They propose sequence-compression and sequence-distinction metrics and show that models passing next-token and probe diagnostics can have incoherent implicit maps (e.g., of Manhattan streets). Vafa et al. (2025) propose inductive-bias probes and find heuristic rather than law-like biases in orbital-mechanics, lattice and Othello settings. Our work is complementary. Vafa et al. evaluate the *outputs* of a generative model against the true automaton. We bound what *internal* probes can show, as a function of the encoder's memory and of the world's suffix structure, and we evaluate interventions only on futures where the true automaton distinguishes the factual and counterfactual states, which is the Myhill–Nerode idea applied to interventions.

**Probe methodology.** Control tasks (Hewitt & Liang, 2019) and MDL probing (Voita & Titov, 2020) address probe expressivity. Amnesic probing (Elazar et al., 2021) and Ravichander et al. (2021) address use versus encoding. Belinkov (2022) surveys these issues. Causal abstraction and interchange interventions (Geiger et al., 2021) give a principled framework for testing whether a representation plays the causal role of a high-level variable. Our CRA is an interchange intervention restricted to the probe subspace and evaluated over a multi-step closed-loop rollout. The suffix ceiling is, to our knowledge, the first baseline for probe accuracy that is (i) derived from the encoder's memory rather than from relabelled data and (ii) computable exactly when the world is known.

**Reservoir computing and fading memory.** Fading-memory operators can be approximated by finite-memory (moving-average) operators (Boyd & Chua, 1985). Echo-state networks with the echo-state property are universal approximators of fading-memory filters (Grigoryeva & Ortega, 2018). Liquid-state machines compute from transient perturbations (Maass et al., 2002). Theorem 1 can be read as the probing-side counterpart of these approximation results: what makes reservoirs universal for fading-memory filters is exactly what caps them at the suffix ceiling for non-local states. Proposition 3 rests on Cover's (1965) function-counting argument.

## 3 Setting and Theory

### 3.1 Setup

A world emits a token stream x\_1, x\_2, … and has a latent state s\_t ∈ 𝒮, a finite set, at each time t. An *encoder* maps each prefix x\_{1:t} to h\_t ∈ ℝ^d. A *linear probe* is f(h) = argmax\_{s∈𝒮} (w\_s·h + b\_s), with accuracy Acc(f) = P(f(h\_t) = s\_t) under the data distribution at a fixed time t (or averaged over a set of times).

**Definition 1 (suffix ceiling).** For k ≥ 0, a predictor is *k-suffix-measurable* if it is a function of x\_{t−k+1:t} only. The k-suffix ceiling is
  Acc\*\_k = sup\_{g k-suffix-measurable} P(g(x\_{t−k+1:t}) = s\_t) = E[ max\_s P(s\_t = s | x\_{t−k+1:t}) ].
Acc\*\_k is non-decreasing in k. For a deterministic world observed from a known start, Acc\*\_k → 1 as k covers the whole history. The *suffix curve* k ↦ Acc\*\_k describes how local the latent state is.

**Definition 2 (fading-memory encoder).** A recurrent encoder h\_t = F(h\_{t−1}, x\_t) with fixed h\_0 is a *λ-contraction* on a set ℋ containing all reachable states if ‖F(h,x) − F(h′,x)‖ ≤ λ‖h − h′‖ for all x and all h, h′ ∈ ℋ, with λ < 1. Write D = sup\_t ‖h\_t − h\_0‖. The *restarted state* h\_t^{(k)} is the state obtained by running F from h\_0 on x\_{t−k+1:t} only.

Examples: an echo-state network h\_t = tanh(W h\_{t−1} + U x\_t + b) with ‖W‖\_2 = λ is a λ-contraction (tanh is 1-Lipschitz), with D ≤ √d when h\_0 = 0. Trained gated RNNs are generally *not* uniform contractions. Precisely to integrate information over long horizons, they can learn marginally stable or multistable dynamics.

### 3.2 Results

**Lemma 1.** For a λ-contraction, δ\_t(k) := ‖h\_t − h\_t^{(k)}‖ ≤ λ^k ‖h\_{t−k} − h\_0‖ ≤ λ^k D.

*Proof.* Both trajectories receive x\_{t−k+1:t}. They start from h\_{t−k} and h\_0 respectively, and each step contracts their distance by λ. ∎

**Theorem 1 (suffix ceiling for fading-memory encoders).** Let f be a linear probe with weights (w\_s, b\_s). Let γ(h) be the gap between the largest and second-largest probe logits, and let L\_t = max\_s ‖w\_{f(h\_t)} − w\_s‖. Then for every k,
  Acc(f) ≤ Acc\*\_k + P( γ(h\_t) ≤ L\_t δ\_t(k) ) ≤ Acc\*\_k + P( γ(h\_t) ≤ L λ^k D ),
where L = max\_{s,s′} ‖w\_s − w\_{s′}‖. Consequently Acc(f) ≤ inf\_k [Acc\*\_k + P(γ(h\_t) ≤ L λ^k D)].

*Proof.* Let ŝ = f(h\_t). For any s ≠ ŝ,
(w\_ŝ − w\_s)·h\_t^{(k)} + b\_ŝ − b\_s = [(w\_ŝ − w\_s)·h\_t + b\_ŝ − b\_s] − (w\_ŝ − w\_s)·(h\_t − h\_t^{(k)}) ≥ γ(h\_t) − L\_t δ\_t(k).
On the event E = {γ(h\_t) > L\_t δ\_t(k)} this is positive for all s ≠ ŝ, so f(h\_t^{(k)}) = f(h\_t). Hence
P(f(h\_t) = s\_t) ≤ P(f(h\_t^{(k)}) = s\_t) + P(E^c).
Since h\_t^{(k)} depends only on x\_{t−k+1:t}, the predictor f(h\_t^{(k)}) is k-suffix-measurable and its accuracy is at most Acc\*\_k. The second inequality follows from Lemma 1 and L\_t ≤ L. ∎

The bound has two useful forms. The *data-dependent* form uses the measured δ\_t(k) per sample and needs only the ability to restart the encoder. The *a priori* form needs only λ, D and the probe's margins. For a probe whose margins are bounded below by γ\_0 on all but an ε fraction of inputs, the ceiling becomes Acc(f) ≤ Acc\*\_{k(γ\_0)} + ε with k(γ\_0) = ⌈log(L D / γ\_0) / log(1/λ)⌉. A fading-memory encoder can only ever show a probe-decodable "world state" that is recoverable from a window of logarithmic length in the margin.

**Corollary 2 (finite context).** If h\_t depends on x\_{t−w+1:t} only (e.g., a transformer with context window w; an n-layer transformer whose attention is masked to the last w′ tokens at every layer has w = n·w′), then Acc(f) ≤ Acc\*\_w for every probe f, linear or not.

*Proof.* f(h\_t) is w-suffix-measurable. ∎

Corollary 2 is trivial, but it bites in practice. A model with a fixed context window probed on a world whose suffix curve has not saturated at w cannot be showing a full world model, whatever its probe accuracy. And if the suffix curve *has* saturated at w, probe accuracy cannot tell a world model from a token-history cache.

**Proposition 3 (free decodability of local states).** Suppose s\_t = φ(x\_{t−k+1:t}) almost surely. Let 𝒰 be the set of M reachable k-suffixes, and suppose the restarted features {h^{(k)}(u) : u ∈ 𝒰} are affinely independent (which requires d ≥ M − 1). Then there is a linear probe f with f(h^{(k)}(u)) = φ(u) for all u. If the encoder is additionally a λ-contraction and that probe has minimum margin γ\_0 > L λ^k D, then f(h\_t) = s\_t almost surely.

*Proof.* With affinely independent points, the linear system w\_s·h^{(k)}(u) + b\_s = 1[φ(u) = s] has a solution for every s, because the augmented matrix [h^{(k)}(u)^T, 1] has full row rank M. The resulting probe outputs a one-hot score vector at every suffix feature, so it classifies correctly with margin 1 before rescaling. The second claim follows from the argument of Theorem 1 with the roles of h\_t and h\_t^{(k)} exchanged. ∎

Affine independence is a generic property. For any encoder family whose features are real-analytic in the random weights, such as tanh reservoirs, the set of weights for which a given M-point configuration is affinely dependent is either the whole space or of measure zero. So if one weight setting achieves independence, random initialisations achieve it almost surely. Cover's (1965) counting theorem gives the corresponding statement for random labellings below capacity. The practical content: for local states, *any* sufficiently wide encoder, trained or not, will pass a probing test.

### 3.3 Diagnostics derived from the theory

**Effective suffix length.** k\_eff(f) = min{k : Acc\*\_k ≥ Acc(f)}, with k\_eff = ∞ if no such k exists within the horizon. By Theorem 1, a λ-contracting encoder with margin-robust probes must have k\_eff ≲ log(LD/γ\_0)/log(1/λ). A large k\_eff, especially k\_eff = ∞ (probe accuracy above the ceiling of every window up to the whole history from an uninformative prior), is evidence that the encoder integrates over long horizons. A small k\_eff is not evidence of a world model even when probe accuracy is high.

**Truncation test.** Probe accuracy evaluated on h\_t^{(k)} instead of h\_t. This directly measures how much of the probe's success depends on history older than k. For a contraction, it must track Acc\*\_k from below.

**Counterfactual rollout agreement (CRA).** Take a base prefix with state s at time t\_0 and a source prefix with state s′ ≠ s. Let P be the orthogonal projector onto the row space of the centred probe weights (rank ≤ |𝒮| − 1). Edit h ← h + P(h\_src − h), which is an interchange intervention (Geiger et al., 2021) restricted to the probe subspace. Then run the model *closed-loop* for R steps: at each step it emits its predicted observation, which is fed back as input, followed by the base sequence's next action. CRA is the fraction of predictions that match the counterfactual world's observations (state s′ driven by the same actions), computed **only on discriminating steps**, where the factual and counterfactual observations differ. Steps where the two worlds agree are uninformative, and including them inflates both factual and counterfactual agreement. Controls: no edit, an edit in a random subspace of equal rank, and a full-state swap (h ← h\_src), which bounds what any edit can achieve.

## 4 Experimental Setup

All numbers in Sections 5–6 come from code in `code/` that we ran on CPU (4 cores, no GPU). Raw outputs are in `results/raw_*.json` and `results/dyn_*.json`, aggregated in `results/summary.md`.

**World.** A 5×5 torus (|𝒮| = 25 cells). An agent starts at cell 0 and takes T = 24 uniformly random moves (U/D/L/R). Each sequence is the interleaved token stream obs\_0, m\_1, obs\_1, …, m\_T, obs\_T (49 tokens), where obs\_t is the colour of the cell occupied after move m\_t. The latent state probed at the position of token m\_t is the current cell pos\_t, and the next token obs\_t is a deterministic function of it. Because the start cell is known, the full history determines pos\_t exactly, so a perfect world model attains 100% state and next-observation accuracy. Three colourings control how local the state is (colourings are sampled per seed):
- **unique**: every cell has its own colour. The last observation and the last move determine the state (Acc\*\_1 = 1).
- **two-colour**: a balanced random 2-colouring. Observation sequences localise the agent gradually (Acc\*\_k = 0.08, 0.44, 0.83, 0.98 for k = 1, 4, 8, 16; seed 0).
- **3 landmarks**: three cells carry colour 1, all others colour 0. Observations are rarely informative, so the state is strongly non-local (Acc\*\_k = 0.08, 0.19, 0.35, 0.61 for k = 1, 4, 8, 16; seed 0).

**Exact ceilings.** Acc\*\_k is computed by an exact forward filter on the last k (observation, move) pairs, starting from the uniform distribution over cells. The uniform distribution is the stationary distribution of the torus walk; we evaluate t ∈ [12, 24], where the walk from the fixed start is close to stationary. At k = t the filter ignores the known start, so Acc\*\_t < 1 even though full-history accuracy is 1. Probe accuracy above Acc\*\_t (k\_eff = "∞") therefore indicates use of information older than any window, including the start position.

**Encoders.** (i) *GRU (trained)*: a 1-layer GRU with 64 units and 32-dimensional embeddings, trained with next-observation cross-entropy (Adam with one-cycle learning rate, peak 6·10⁻³, batch 128, 10 epochs on 14,000 sequences). (ii) *GRU (random)*: the same architecture at its PyTorch initialisation, untrained. (iii) *ESN λ=0.9, d=64* and (iv) *ESN λ=0.99, d=512*: tanh echo-state networks with recurrent spectral norm exactly λ, so that Theorem 1 applies with D ≤ √d. (v) *Transformer (trained)*: 2 layers, d = 64, 4 heads, pre-LN, learned positional embeddings, full causal context, trained like the GRU (peak learning rate 2·10⁻³). (vi) *Transformer (random)*. Because of CPU cost, transformers were run on seed 0 only. For untrained encoders, the next-observation readout is a multinomial logistic regression fitted on the probe-training split.

**Probes.** Multinomial logistic regression (L-BFGS, L2 = 10⁻⁴) on standardised hidden states, with the standardisation folded back into the weights. Probes are fitted on 6,000 sequences × 13 time steps and tested on 2,000 held-out sequences. A shuffled-label control probe checks for memorisation; it stays at chance (≈0.04) for every encoder.

**Truncation test.** For t ∈ {16, 20, 24} and k ∈ {1, …, 16}, we restart each recurrent encoder from h\_0 = 0 on the last 2k tokens (obs\_{t−k}, …, m\_t), apply the *same* probe, and compute the data-dependent bound Acc\*\_k + P(γ ≤ L\_t δ\_t(k)). For the ESNs we also compute the a priori bound with δ ≤ λ^{2k}√d. Each step consists of two tokens, so the contraction acts 2k times.

**CRA.** t\_0 = 14, horizon R = 10 (11 predictions). Pairs are base/source sequences with different cells at t\_0, about 1,920 pairs per run. The probe-subspace rank is 24.

**Seeds.** Three seeds (0, 1, 2) for every world and recurrent encoder. Each seed changes the colouring, the data, the initialisation and the reservoir weights. We report mean ± standard deviation over seeds.

## 5 Results

All numbers below come from our own runs and are reproduced in `results/summary.md`. Means ± s.d. are over three seeds unless marked (n = 1 for transformers).

### 5.1 When the state is local, everything passes, including untrained networks

In the *unique* world (Acc\*\_1 = 1), every recurrent encoder decodes the state almost perfectly. Probe accuracy is 1.000 ± 0.000 for the trained GRU and both ESNs, and 0.999 ± 0.001 for the untrained GRU (Table 1, Fig. 1 left). k\_eff = 1 throughout. The untrained encoders also pass the *intervention* test. A probe-subspace interchange makes their closed-loop rollouts follow the counterfactual trajectory on 0.938 ± 0.010 (random GRU), 1.000 (ESN λ=0.9) and 1.000 (ESN λ=0.99) of discriminating steps. The random-subspace control gives 0.00–0.09, and no edit gives 0.000. This is Proposition 3 at work, and it is not a failure of CRA. In a 1-local world, the minimal automaton consistent with the data (its Myhill–Nerode quotient) *is* a cache of the last observation, so a world model and a suffix cache are behaviourally indistinguishable. Once the edited model emits the counterfactual colour, feeding that colour back re-anchors any suffix-based encoder. The lesson is that neither probing nor intervention can provide evidence for a world model beyond what the world's suffix curve already gives away.

The one exception is instructive. The *untrained transformer* decodes the state at only 0.135 in the unique world. At initialisation, attention is close to uniform over the context, so the previous token's identity, which is needed together with the current move, is diluted across about 2t positions. Proposition 3 requires the suffix features to be in general position, and uniform averaging violates that. Free decodability therefore depends on the architecture: it holds for recurrent reservoirs and fails for a randomly initialised causal transformer in our setting.

**Table 1.** Next-observation accuracy, linear state-probe accuracy, effective suffix length and counterfactual rollout agreement (CRA, discriminating steps only). GRUs trained 10 epochs. ∞ means the probe beats the ceiling of every window up to the full prefix with an uninformative prior.

| World | Encoder | next-obs acc | probe acc | k\_eff (per seed) | CRA probe-subspace | CRA random-subspace | CRA full swap |
|---|---|---|---|---|---|---|---|
| unique | GRU trained | 1.000 ± 0.000 | 1.000 ± 0.000 | 1, 1, 1 | 1.000 ± 0.000 | 0.114 ± 0.020 | 1.000 ± 0.000 |
| unique | GRU random | 0.999 ± 0.001 | 0.999 ± 0.001 | 1, 1, 1 | 0.938 ± 0.010 | 0.069 ± 0.001 | 0.996 ± 0.002 |
| unique | ESN λ=.9, d=64 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1, 1, 1 | 1.000 ± 0.000 | 0.090 ± 0.006 | 1.000 ± 0.000 |
| unique | ESN λ=.99, d=512 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1, 1, 1 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| unique | Transformer trained (n=1) | 1.000 | 1.000 | 1 | – | – | – |
| unique | Transformer random (n=1) | 0.136 | 0.135 | 1 | – | – | – |
| two-colour | GRU trained | 0.942 ± 0.052 | 0.804 ± 0.169 | ∞, 6, 8 | 0.669 ± 0.147 | 0.409 ± 0.022 | 0.872 ± 0.109 |
| two-colour | GRU random | 0.627 ± 0.017 | 0.140 ± 0.010 | 2, 2, 2 | 0.509 ± 0.009 | 0.497 ± 0.001 | 0.537 ± 0.009 |
| two-colour | ESN λ=.9, d=64 | 0.642 ± 0.009 | 0.139 ± 0.009 | 2, 2, 2 | 0.513 ± 0.004 | 0.507 ± 0.012 | 0.538 ± 0.015 |
| two-colour | ESN λ=.99, d=512 | 0.714 ± 0.019 | 0.168 ± 0.004 | 3, 3, 3 | 0.508 ± 0.012 | 0.475 ± 0.009 | 0.560 ± 0.011 |
| two-colour | Transformer trained (n=1) | 0.969 | 0.817 | 8 | – | – | – |
| two-colour | Transformer random (n=1) | 0.523 | 0.057 | 1 | – | – | – |
| landmarks | GRU trained | 0.972 ± 0.026 | 0.664 ± 0.307 | 6, ∞, ∞ | 0.523 ± 0.116 | 0.388 ± 0.062 | 0.876 ± 0.131 |
| landmarks | GRU random | 0.885 ± 0.011 | 0.125 ± 0.022 | 2, 2, 2 | 0.514 ± 0.007 | 0.491 ± 0.009 | 0.514 ± 0.013 |
| landmarks | ESN λ=.9, d=64 | 0.886 ± 0.012 | 0.110 ± 0.009 | 2, 2, 2 | 0.522 ± 0.014 | 0.505 ± 0.005 | 0.515 ± 0.014 |
| landmarks | ESN λ=.99, d=512 | 0.913 ± 0.006 | 0.148 ± 0.018 | 3, 3, 3 | 0.528 ± 0.011 | 0.486 ± 0.009 | 0.547 ± 0.012 |
| landmarks | Transformer trained (n=1) | 0.962 | 0.507 | 13 | – | – | – |
| landmarks | Transformer random (n=1) | 0.878 | 0.060 | 1 | – | – | – |

(Two-colour and landmark worlds have two observation colours, so CRA chance on discriminating steps is 0.5 for a model that ignores the edit but still emits one of the two colours. In the unique world, an unedited model always emits the factual colour and so scores 0.)

### 5.2 When the state is non-local, untrained and fading-memory encoders sit at the ceiling of a 2–3 step window

In the two-colour and landmark worlds, all untrained encoders have k\_eff = 2 or 3 in every seed. Their probe accuracies (0.11–0.17) are what a Bayes decoder of the last 2–3 steps achieves (Acc\*\_2 = 0.16 and Acc\*\_3 = 0.28 in the two-colour world). Even the ESN with nominal λ = 0.99 and d = 512, whose a priori memory scale is 1/(1−λ) = 100 tokens, behaves this way. The truncation test explains why. Its restarted-state distance δ\_t(k) is 0.002 at k = 4 and below 10⁻³ at k = 8. With unit input scaling the tanh units saturate, and the *effective* contraction is far stronger than the spectral norm suggests. The a priori bound of Theorem 1 (using λ^{2k}√d) is therefore vacuous in our runs (it equals 1 at every k). The data-dependent bound, which uses the measured δ\_t(k), is non-vacuous: its infimum over k is 0.52 (ESN λ=0.9) and 0.70 (ESN λ=0.99) in the two-colour world, and 0.32 and 0.34 in the landmark world (`results/bound_tightness.json`). These bounds hold in every run, but they are loose. The ceiling Acc\*\_k is achieved by a *nonlinear* Bayes filter, while a linear readout of a random reservoir recovers much less.

The sharper empirical check is the truncation test (Fig. 2). Theorem 1's argument implies that the probe applied to h\_t^{(k)} is a k-suffix-measurable predictor, so its accuracy must lie below Acc\*\_k. Across all 324 (world, seed, encoder, k) cells, the largest excess of truncated accuracy over Acc\*\_k is +0.010. This is within sampling error and the small mismatch between the stationary prior and the finite-time marginal. For the random encoders, truncated and full probe accuracies coincide from k ≈ 4 onwards: everything they "know" about the world state is in the last few steps.

### 5.3 Trained models exceed the ceiling, and next-token accuracy is a poor proxy for it

Trained GRUs exceed the ceiling of every window (k\_eff = ∞) in 1/3 seeds of the two-colour world and 2/3 seeds of the landmark world after 10 epochs, and in all four longer-training runs (constant learning rate, Section 5.4) from epoch 20 on. Their truncated-probe accuracy keeps rising with k (Fig. 2), as expected of an encoder that is *not* a uniform contraction and carries information from far back. The trained transformer, run on one seed, reaches k\_eff = 8 (two-colour) and 13 (landmark). These are well above any untrained encoder, but its probe accuracy (0.82 and 0.51) stays below the trained GRU's best seeds. With our CPU budget we could not tell whether this reflects architecture or under-training.

Next-observation accuracy does not track this distinction. In the landmark world, untrained encoders reach next-observation accuracy of 0.885–0.913 by predicting the majority colour, against a world-model ceiling of 1.0. The trained GRU of seed 0 reaches 0.937 and predicts landmark hits with recall 0.85 (Table 1, per-seed values in `results/summary.md`), yet its state probe is only 0.250 with k\_eff = 6. Its behaviour has improved by means that do not (yet) involve a decodable global position. This matches the "task-specific heuristic" picture of Vafa et al. (2025), but here it is *located* on the suffix curve.

### 5.4 Training dynamics: behaviour first, then a decodable state, then a causally used state

We retrained GRUs for 40 epochs and evaluated at epochs 1, 2, 5, 10, 20, 30 and 40 (three landmark seeds, one two-colour seed; Fig. 4, `results/summary.md`). Three regularities appear in all four runs.

1. **Behaviour precedes state.** At epoch 5 of landmark seed 0, landmark recall is already 0.861, but the probe is at 0.338 (k\_eff = 8). Seed 2 at epoch 5 is similar: recall 0.856, probe 0.362, k\_eff = 8.
2. **A sharp transition to k\_eff = ∞.** Between epochs 5 and 20, probe accuracy jumps from 0.34–0.79 to ≥ 0.997, and k\_eff jumps from 8–20 to ∞, together with next-observation accuracy reaching ≥ 0.9996.
3. **Probe-subspace interventions are harmful before the transition and helpful after it.** At epochs 2 and 5 (8 checkpoints), CRA(probe) is *below* the random-subspace control in 6 cases (e.g., landmark seed 0, epoch 5: 0.399 vs 0.469). The two exceptions are the checkpoints whose probe accuracy had already reached ≥ 0.76 (two-colour epoch 5: 0.706 vs 0.426; landmark seed 1 epoch 5: 0.563 vs 0.382). At every checkpoint with k\_eff = ∞, CRA(probe) lies in 0.64–0.89 and the random control in 0.25–0.42. Editing what the probe reads only steers the model once the probe reads the variable the model actually updates.

A consistent quantitative gap remains even in fully trained models. CRA(probe) plateaus at 0.70–0.89 while a full swap reaches 1.000. The linear probe's 24-dimensional row space captures most, but not all, of the causally used state. The rest is carried in directions the probe does not need for decoding, for example redundant or nonlinearly encoded copies. Intervention claims based on probe directions therefore *underestimate* causal use, and the full-swap control is needed to calibrate them.

**Figures** (in `results/`): `fig1_suffix_ceiling.png` shows Acc\*\_k curves with each encoder's probe accuracy (seed-mean). `fig2_truncation.png` shows probe accuracy on restarted states h\_t^{(k)} against Acc\*\_k. `fig3_rollout.png` shows CRA per encoder and edit type. `fig4_dynamics.png` shows probe accuracy and CRA(probe) − CRA(random) over training. `fig5_cra_vs_probe.png` plots intervention gain against probe accuracy for all trained-GRU evaluations.

## 6 Discussion

**What counts as evidence for a world model.** Our results support a three-level reading of "the model represents the world state".
(a) *Decodable*: a probe recovers s\_t. Proposition 3 and Section 5.1 show this is free whenever the world's suffix curve saturates early, even for untrained networks.
(b) *Decodable beyond the ceiling*: k\_eff exceeds what the encoder's restart-truncated features support, and in the strongest case k\_eff = ∞. Theorem 1 shows that no fading-memory encoder reaches this level with robust margins. In our runs only trained models did.
(c) *Causally maintained*: editing the probed subspace changes multi-step closed-loop behaviour in the way the counterfactual world would, on steps where the worlds differ, and beats a random subspace of the same rank.
Across all 34 trained-GRU evaluations in the non-local worlds (main runs plus training checkpoints; `results/cra_vs_probe.json`, `fig5_cra_vs_probe.png`), the intervention gain CRA(probe) − CRA(random) tracks probe accuracy closely (Pearson r = 0.92). All 10 evaluations with probe accuracy below 0.4 have gain ≤ 0, and all 21 with probe accuracy ≥ 0.75 have gain ≥ 0.17. Levels (b) and (c) therefore rose together in our runs, while behavioural competence (next-token accuracy) came *before* both, through a heuristic phase. That (b) and (c) co-vary here is an empirical finding about these small models, not a logical necessity, and the protocol in Section 7 is designed to test whether it holds at scale.

**Why the suffix curve should be reported alongside every probe result.** For Othello, chess or language, Acc\*\_k is not computable in closed form. It can be *estimated from below* by training the strongest affordable decoder (for example, a large MLP or a fresh transformer) on k-token windows to predict the state. Because such an estimate can only underestimate Acc\*\_k, it is conservative for *refuting* a world-model reading but not for supporting one. A probe that fails to beat the estimated ceiling also fails to beat the true one. A probe that does beat it should be checked against stronger window decoders before k\_eff is taken at face value. If a probe on the model under study does not beat what a dedicated decoder extracts from the last k tokens, for k comparable to the model's measured truncation depth, the probe result says little about world modelling. Othello is an instructive case. The board is a function of the full move history, but many squares are fixed by recent moves, and the "mine/theirs" relabelling (Nanda et al., 2023) depends on whose turn it is, which a transformer can read largely from positional information rather than from a model of the board. We predict (Section 7) that a substantial share of per-square probe accuracy in Othello-GPT will be recovered by short-window decoders, and that the informative squares are those whose state was last changed far in the past.

**Relation to Myhill–Nerode evaluation.** Vafa et al. (2024) score generative outputs on sequences that distinguish states. CRA brings the same idea inside the model: an intervention is only evaluated on future steps where the true automaton distinguishes the factual and counterfactual states. Without this restriction, CRA is inflated by steps that are uninformative (in our two-colour world, half of all steps).

**Relation to reservoir computing.** The universality of echo-state networks for fading-memory filters (Grigoryeva & Ortega, 2018) and Theorem 1 are two sides of one fact. A fading-memory encoder is, up to the margin term, a function of a bounded window. That is exactly what makes it universal *for fading-memory targets*, and exactly what limits it *for non-local world states*. The finding that nominal λ = 0.99 gave an effective window of about 3 steps (Section 5.2) is a reminder that spectral-norm arguments are conservative. The data-dependent δ\_t(k) is the quantity to measure.

## 7 A Pre-registerable Protocol for Large Models

We could not run large models here. The following protocol applies our diagnostics to published world-model claims, and states predictions that could falsify our framing.

1. **Suffix-ceiling estimate.** For each probed variable (an Othello square, a chess square, a navigation position), train a strong decoder from scratch on windows of the last k tokens, for k in {1, 2, 4, …, full}. Plot the estimated Acc\*\_k curve. *Prediction P1:* for Othello-GPT, more than half of the linear-probe accuracy above the constant-"empty" baseline will be matched by k ≤ 8.
2. **Truncation depth.** For transformers, compute probe accuracy when attention is masked to the last k′ tokens at every layer. This is the transformer analogue of restarting, and Corollary 2 applies to it exactly with window n\_layers·k′. *Prediction P2:* the truncation curve of a randomly initialised model of the same architecture lies near or below the trained model's estimated Acc\*\_k curve, and the trained model's curve keeps rising beyond it.
3. **Stratified probe accuracy.** Report probe accuracy separately for variables whose value was last determined more than k tokens ago (for example, Othello squares last flipped more than k moves ago). *Prediction P3:* the gap between trained and random-initialised models is concentrated in the long-ago stratum.
4. **CRA with controls.** Interchange interventions in the probe row space, a random subspace of the same rank, and a full-residual swap, evaluated over multi-step closed-loop rollouts and scored only on steps where the factual and counterfactual legal-move sets differ. *Prediction P4:* CRA(probe) is strictly between CRA(random) and CRA(full), as we found (0.70–0.89 vs 0.25–0.42 vs 1.0).
5. **Training-time tracking.** Checkpointed models should show the behaviour → decodable → causal ordering of Section 5.4. *Prediction P5:* intermediate checkpoints with high legal-move accuracy but k\_eff below the context length will have CRA(probe) at or below CRA(random).

Any of P1–P5 failing would be informative. For example, a large k\_eff together with CRA(probe) ≈ CRA(full) early in training would contradict the ordering we observed.

## 8 Limitations

- **Scale and architecture.** All models are tiny (64-unit GRUs, 2-layer d = 64 transformers) and the world has 25 states. The transformer results are from a single seed, and the transformer was likely under-trained (two-colour next-observation accuracy 0.969). We did not run CRA on transformers, because editing one position's residual stream in a causal transformer does not propagate to later positions the way a recurrent state edit does. A proper treatment needs edits at all subsequent positions or at the key/value cache.
- **Training budget.** With 10 epochs, the trained GRUs are bimodal across seeds (two-colour probe accuracy 1.000 / 0.587 / 0.825). Seed-level variance, not the mean, is the meaningful summary, and we report per-seed values. The longer-training runs use a different learning-rate schedule (constant 3·10⁻³) from the main runs (one-cycle).
- **Ceiling approximation.** Acc\*\_k uses the stationary (uniform) prior at the window start rather than the exact finite-time marginal from the known start cell. This slightly misstates the ceiling at small t. The largest observed excess of truncated accuracy over Acc\*\_k was +0.010.
- **Bound looseness.** The a priori form of Theorem 1 was vacuous for our ESNs, because the probe-weight scale L multiplied by √d is large. The data-dependent form was valid but loose (inf-bound minus accuracy 0.19–0.84 for untrained encoders), because Acc\*\_k is achieved by a nonlinear filter. Theorem 1 is a ceiling, not a prediction of probe accuracy.
- **Scope of the theorem.** Theorem 1 covers uniformly contracting recurrent encoders and (Corollary 2) finite-context models. Trained RNNs and full-context transformers are not covered. For them, the truncation test and k\_eff are empirical diagnostics, not guarantees. Proposition 3's generic-position assumption fails for some architectures, as the random transformer shows.
- **Linear probes only.** Nonlinear probes would raise the decodability of every encoder, including random ones. Corollary 2 holds for any probe, but Theorem 1's margin argument is specific to linear probes, though it extends directly to Lipschitz probes.
- **Single world family.** A torus gridworld with colour observations is a clean but narrow testbed. We have not shown that the regularities of Section 5.4 hold in board games or language.

## 9 Conclusion

Probe decodability of a latent world state is evidence for an internal world model only to the extent that it exceeds what the encoder's effective memory window gives for free. We made this precise with a suffix-ceiling theorem for fading-memory and finite-context encoders, and a free-decodability proposition for local states. We turned the theory into two diagnostics, the effective suffix length k\_eff and counterfactual rollout agreement on discriminating steps. In a controlled gridworld with exact ceilings, untrained and fading-memory encoders pass both probing and intervention tests when the state is local, and are pinned at a 2–3-step ceiling when it is not. Trained GRUs exceed every window's ceiling only after a heuristic phase in which behaviour is already good, and only then do probe-subspace interventions steer them. These are small-scale results, but the diagnostics are cheap, architecture-agnostic in their empirical form, and directly applicable to the Othello, chess and navigation models on which current world-model claims rest.

## Reproducibility

`code/README.md` gives the full run commands. Main experiment: `run_experiment.py` (3 worlds × 3 seeds; about 10 minutes per world on one CPU thread). Training dynamics: `training_dynamics.py` (4 runs). Aggregation and figures: `aggregate.py`. Raw outputs: `results/raw_*.json`, `results/dyn_*.json`, `results/bound_tightness.json`; tables: `results/summary.md`.

## References

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219. https://aclanthology.org/2022.cl-1.7/

Boyd, S., & Chua, L. O. (1985). Fading memory and the problem of approximating nonlinear operators with Volterra series. *IEEE Transactions on Circuits and Systems*, CAS-32(11), 1150–1161. doi:10.1109/TCS.1985.1085649

Cover, T. M. (1965). Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. *IEEE Transactions on Electronic Computers*, EC-14(3), 326–334. doi:10.1109/PGEC.1965.264137

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics*, 9, 160–175. doi:10.1162/tacl_a_00359

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
