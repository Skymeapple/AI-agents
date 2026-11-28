# The Rarity Tax: An Occurrence-Threshold Law for Feature Recovery and What It Implies for Scaling Sparse Dictionaries to Frontier Models

**Target venue:** ICLR (main conference track; mechanistic interpretability / representation learning)

## Abstract

Whether mechanistic interpretability scales to frontier models depends, for the currently dominant pipeline, on whether sparse dictionaries (sparse autoencoders and their relatives) can keep recovering a model's features as those features become rarer. Existing scaling analyses vary the dictionary width and treat training data as sufficient. We study the complementary constraint: how many training tokens a feature of a given frequency needs before a dictionary recovers it. In a controlled toy setting with known ground truth — 512 features in superposition in 128 dimensions, with Zipf-distributed frequencies, decomposed by a TopK sparse autoencoder — we run 29 training runs and measure recovery of every feature at 14 checkpoints. Three results follow. (i) Recovery is governed, to first order, by the expected number of occurrences m = pT of a feature and not by its frequency p alone: a one-parameter occurrence-threshold model predicts held-out-seed recovered fractions to within 0.014–0.016 mean absolute error, with a threshold of a few hundred occurrences (230–510 across Zipf exponents). (ii) The dependence is sub-linear under Adam: the time to first recovery scales as p^(-κ) with κ ≈ 0.58–0.61 at the baseline configuration, against κ ≈ 0.91–0.96 for momentum SGD, so the exponent is a property of the optimizer and not of the data; it is not pinned down tightly (two estimators disagree for some configurations, and we report both). (iii) Replaying a small buffer for many epochs recovers features as well as fresh data in this noise-free setting, so the constraint is on exposures and not on unique samples. We combine the occurrence law with a capacity limit into a two-resource coverage model and derive scaling exponents: with Zipf exponent α and rarity exponent β = 1/κ, the unrecovered activation mass falls as T^(-β(α-1)/α) with tokens, as M^(-(α-1)) with width, and as (MT)^(-(α-1)/(1+α/β)) with compute under the optimal allocation, which requires tokens to grow as M^(α/β). When α is close to 1, as the limited public evidence suggests, each tenfold reduction in unrecovered mass costs between a few tens and several orders of magnitude more compute, and a token budget that grows as M^0.6 (the published convergence rate) leaves a growing share of a wide dictionary's latents unable to learn the features of their rank. We do not run any experiment on a large model; we give a pre-registered protocol for testing the law on open models.

## 中文摘要

机制可解释性能否扩展到前沿规模的模型，就当前主流方法而言，取决于稀疏字典（稀疏自编码器及其变体）能否在特征越来越稀有时仍然把它们恢复出来。已有的扩展性分析只改变字典宽度，并默认训练数据充足。本文研究与之互补的约束：一个给定频率的特征需要多少训练词元才能被字典恢复。我们在真值已知的玩具设定中（512 个特征叠加在 128 维空间，频率服从 Zipf 分布，用 TopK 稀疏自编码器分解）完成了 29 次训练，并在 14 个检查点上测量每个特征的恢复情况。结果有三点。（一）恢复主要由特征的期望出现次数 m = pT 决定，而不是单由频率 p 决定：单参数的“出现次数阈值”模型对另一随机种子的恢复比例预测的平均绝对误差为 0.014–0.016，阈值为数百次出现（不同 Zipf 指数下为 230–510 次）。（二）在 Adam 下这种依赖是次线性的：首次恢复所需词元数按 p^(-κ) 增长，基线配置下 κ ≈ 0.58–0.61，而动量 SGD 下 κ ≈ 0.91–0.96，因此该指数是优化器的性质而非数据的性质；这一指数并未被精确确定（两种估计方法在部分配置下不一致，本文同时报告）。（三）在这个无噪声设定中，对小缓冲区反复重放与使用新数据的恢复效果相同，说明约束在于曝光次数而非独立样本数。我们把出现次数规律与容量限制结合成一个双资源覆盖模型，并推导出标度指数：设 Zipf 指数为 α、稀有性指数为 β = 1/κ，未恢复的激活质量随词元数按 T^(-β(α-1)/α) 下降，随宽度按 M^(-(α-1)) 下降，在最优分配下随算力按 (MT)^(-(α-1)/(1+α/β)) 下降，且最优分配要求词元数按 M^(α/β) 增长。当 α 接近 1（有限的公开证据指向这一情形）时，未恢复质量每降低十倍需要增加数十倍到若干个数量级的算力；而若词元预算只按 M^0.6（文献报告的收敛速率）增长，宽字典中将有越来越大比例的隐单元学不到与其排名对应的特征。本文没有在大模型上做任何实验，而是给出了一个可在开源模型上检验该规律的预注册方案。

## 1 Introduction

The question "can mechanistic interpretability scale to frontier-size models?" has several parts: whether the units of analysis can be found at scale, whether circuits built from those units stay faithful, and whether humans or automated tools can read the result. This paper addresses only the first part, and within it a single quantitative question about the method that most current frontier-scale work relies on.

That method is dictionary learning. A sparse autoencoder (SAE) or transcoder is trained on a model's activations and its latents are used as the features in which computation is described (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024; Gao et al., 2025; Lieberum et al., 2024; Ameisen et al., 2025). The largest published dictionaries have 16 million latents on GPT-4 (Gao et al., 2025) and 34 million on Claude 3 Sonnet (Templeton et al., 2024). The authors of the latter state that the model "contains many more features than we have found" and report that whether a concept has a dedicated feature is closely tied to how frequent the concept is in the training data. Rare features are not a side issue for the safety uses of interpretability: behaviours one most wants to audit are, by construction, rare in ordinary data.

A dictionary can miss a rare feature for two different reasons. It can be too narrow: its latents are spent on more frequent features. Or it can be under-trained with respect to that feature: the feature has not occurred often enough in the training tokens for a latent to align with it. The first limit has a recent theoretical treatment (Michaud et al., 2025), which models SAE training as allocating latents to features under a power-law frequency distribution and varies only the number of latents. The second limit has, to our knowledge, no quantitative treatment, although practitioners set token budgets by a scaling analysis of reconstruction loss (Gao et al., 2025; Templeton et al., 2024). Reconstruction loss is dominated by frequent features and is therefore a poor guide to whether rare features have been learned.

We ask: **how many training tokens does a feature of frequency p need before a sparse dictionary recovers it, and what does the answer imply for the joint width and data cost of covering a long-tailed feature set?**

We cannot answer this on a frontier model: there is no ground truth there, and we have no GPU. We answer it where it can be answered exactly — a synthetic superposition setting with known features — and then state precisely which parts of the answer are measured, which are derived, and which are extrapolated under assumptions.

**Contributions.**

1. **An occurrence-threshold law, measured.** Across three Zipf exponents and two seeds, per-feature recovery curves at different training lengths collapse when plotted against expected occurrences m = pT and do not collapse against p (Section 5.1). The fitted threshold is 230–510 occurrences. A model fitted on one seed predicts the recovered fraction and the unrecovered activation mass of another seed at every checkpoint with mean absolute error 0.014–0.016 and 0.002–0.003 respectively.
2. **A rarity exponent that depends on the optimizer.** The time to first recovery scales as p^(-κ). Under Adam at baseline settings κ ≈ 0.58–0.61 (rare features need fewer occurrences than frequent ones); under momentum SGD κ ≈ 0.91–0.96 (Section 5.2). A second estimator, a logistic fit on late checkpoints, gives a different value for some configurations; we report the disagreement and bound the exponent β = 1/κ to roughly the interval [1, 2] instead of quoting a point value.
3. **Exposures, not unique samples.** Training for 30 epochs on a buffer of 100,000 tokens recovers 99.6% of features, against 100% for 3 million fresh tokens, including features with as few as 21 unique occurrences in the buffer (Section 5.4).
4. **A two-resource coverage model with closed-form exponents** (Section 3) for unrecovered mass as a function of tokens, width and compute, and a mismatch result: a token budget growing as M^g with g below α/β leaves a fraction (M/M₀)^(gβ/α-1) of latents data-starved.
5. **A sensitivity analysis on sourced numbers** (Section 6) showing that the conclusion turns almost entirely on the Zipf exponent α of true feature frequencies, and a **pre-registered protocol** (Section 7) that would measure α, β and the threshold on an open model.

The contribution is a measurement in a toy setting plus a model built on it. It is not evidence about any particular frontier model.

## 2 Related Work

**Superposition and dictionary learning.** Elhage et al. (2022) showed in toy models that networks store more features than dimensions when features are sparse, which motivates decomposing activations with an overcomplete sparse dictionary. Cunningham et al. (2023) and Bricken et al. (2023) showed that SAEs find interpretable features in language models. Our data generator is in the family of the toy models of Elhage et al., with a Zipfian frequency profile.

**Scaling dictionaries.** Templeton et al. (2024) trained SAEs with about 1M, 4M and 34M latents on Claude 3 Sonnet; they report roughly 2%, 35% and 65% dead latents respectively, fewer than 300 active features per token, and at least 65% of variance explained. Their feature-completeness analysis found that the concept frequency at which a dictionary becomes more than 50% likely to contain a concept is "consistently slightly lower than the inverse of the number of alive features". Gao et al. (2025) introduced TopK SAEs with an auxiliary loss for dead latents, trained a 16M-latent SAE on GPT-4 for 40 billion tokens with 7% dead latents, and reported that the number of tokens to convergence grows as approximately n^0.6 for GPT-2 small and n^0.65 for GPT-4 in the number of latents n. Lieberum et al. (2024) released SAEs on all layers of Gemma 2 (JumpReLU architecture; Rajamanoharan et al., 2024). We use the TopK architecture and AuxK loss of Gao et al. and borrow their convergence exponent as a sourced input in Section 6.

**Theory of SAE scaling.** The closest work is Michaud et al. (2025), who adapt the capacity-allocation model of Brill (2024) to SAEs. They assume feature frequencies p_i ∝ i^(-(1+a)) and show that reconstruction loss falls as N^(-a) in the number of latents N unless feature manifolds absorb latents. In our notation α = 1 + a, and our width-limited exponent -(α-1) in Proposition 1 agrees with theirs. Their model varies only the number of latents and assumes the optimizer finds the optimal allocation; it does not model training tokens or optimizer effects. Our contribution is the data axis and its interaction with width. The lineage of both models is the "quantization" picture of neural scaling laws (Michaud et al., 2023) and Hutter's (2021) learning-curve theory, in which Zipf-distributed discrete units learned one by one give power-law learning curves. Sous and Winer (2026) analyse a random-feature model in which test loss is dominated by rare coordinates unseen in training; this is the same statistical phenomenon in a supervised setting.

**How feature frequencies are distributed.** Evidence is thin. Peter and Søgaard (2025) report that SAE latent activation frequencies follow an approximate power law in the central range with exponents between -1.00 and -1.20, with a sharper decline in the tail. Michaud et al. (2025) estimate a ≈ 0.5–0.7 (α ≈ 1.5–1.7) from Gemma Scope latent frequencies and caution that this assumes a one-to-one map between latents and features. The inverse relation between threshold frequency and alive latents in Templeton et al. (2024) corresponds to α ≈ 1 if their dictionaries were width-limited. All of these are frequencies of learned latents or of human-defined concepts, not of true features.

**Limits of SAEs unrelated to frequency.** SAE latents are not canonical: they split and absorb as width grows (Chanin et al., 2024; Leask et al., 2025); part of the SAE reconstruction error has structure that wider dictionaries do not readily remove (Engels et al., 2024); and gains on reconstruction-based proxy metrics do not reliably carry over to downstream evaluations (Karvonen et al., 2025). Sharkey et al. (2025) survey the open problems of the field, including the limitations of sparse dictionary learning. Our analysis is orthogonal to these: it grants the dictionary picture its best case (linear, discrete, noise-free features) and asks what the cost of the long tail is even then.

**Adaptive optimizers and rare events.** Adam (Kingma and Ba, 2015) normalizes each parameter's update by a running estimate of its gradient scale, and was motivated in part by sparse-gradient problems. That such normalization changes the relative learning speed of rare and frequent features is expected; we measure the size of the effect for dictionary learning.

## 3 A Two-Resource Coverage Model

### 3.1 Setting and assumptions

A model has N features with unit directions in R^d. Feature i is active on a token with probability p_i = C i^(-α), where i is the frequency rank, α ≥ 1 is the Zipf exponent and C is set by the mean number of active features per token, Σ p_i = L₀. A dictionary with M latents is trained on T tokens. Let R be the number of features it recovers.

**Assumption A1 (capacity).** A dictionary with M latents can hold at most ρM features, 0 < ρ ≤ 1, and holds the most frequent ones first.

**Assumption A2 (rarity law).** Given enough latents, a feature of frequency p is recovered after T tokens if and only if p·T^β ≥ c, for constants β > 0 and c > 0. Equivalently its recovery time is T*(p) = (c/p)^κ with κ = 1/β. When β = 1 this says that a feature needs m* = c occurrences.

A1 and A2 are tested in Section 5: A1 holds with ρ between 0.77 and 0.97; A2 holds as a first-order description with β between about 1 and 2 depending on the optimizer. Under A1 and A2 the recovered set is the top R(M, T) = min(ρM, R_T) features, where

R_T = (C T^β / c)^(1/α)

is the rank of the rarest feature that the data budget can teach.

We measure what is missed by the **unrecovered activation mass** U(R) = Σ_{i>R} p_i / Σ_{i≤N} p_i, the share of feature activations that belong to unrecovered features. With equal feature magnitudes it is proportional to the unexplained variance attributable to missing features. It is a generous metric for the dictionary: it weights features by frequency. The fraction of features missed, 1 - R/N, is far larger whenever the tail is long, and for auditing rare behaviours it is arguably the more relevant quantity. We return to this in Section 8.

### 3.2 Results

**Proposition 1 (single-resource exponents).** Let α > 1 and 1 ≪ R ≪ N. Then U(R) = R^(-(α-1)) / ((α-1) H_N) · (1 + O(1/R) + O((R/N)^(α-1))), where H_N = Σ_{i≤N} i^(-α). Consequently, in the width-limited regime (ρM < R_T) U ∝ M^(-(α-1)), and in the data-limited regime (R_T < ρM) U ∝ T^(-β(α-1)/α).

*Proof.* The tail sum is bounded by integrals: ∫_{R+1}^{N+1} x^(-α) dx ≤ Σ_{i=R+1}^{N} i^(-α) ≤ ∫_{R}^{N} x^(-α) dx. Both bounds equal (R^(1-α) - N^(1-α))/(α-1) up to a relative correction of order 1/R, and N^(1-α)/R^(1-α) = (R/N)^(α-1). Substituting R = ρM gives the first exponent and R = R_T ∝ T^(β/α) the second. ∎

**Proposition 2 (boundary case α = 1).** For α = 1, U(R) = 1 - H_R/H_N with H_n = ln n + γ + O(1/n), so U(R) ≈ ln(N/R) / (ln N + γ). Coverage grows only logarithmically in R: to leave a share U unrecovered the dictionary must recover R ≈ N^(1-U) e^(-γU) features, a number that grows with the total number of features N.

*Proof.* Immediate from the expansion of the harmonic number. ∎

The difference between Propositions 1 and 2 is the main reason the Zipf exponent matters so much. For α > 1 the number of features needed for a given mass coverage is asymptotically independent of N; for α = 1 it is a fixed power of N.

**Proposition 3 (optimal allocation under a compute budget).** Suppose training cost is K = M·T (each token is processed by each latent) and α > 1. Maximizing R(M, T) subject to MT = K gives ρM = R_T, hence

T_opt ∝ M^(α/β),  R_opt ∝ K^(β/(α+β)),  U_opt ∝ K^(-(α-1)/(1+α/β)).

*Proof.* R = min(ρM, aT^(β/α)) with a = (C/c)^(1/α). On the curve M = K/T the first argument decreases in T and the second increases, so the minimum is maximized where they are equal: ρK/T = aT^(β/α), giving T ∝ K^(α/(α+β)) and R ∝ K^(β/(α+β)). Proposition 1 gives U. ∎

Two corollaries are worth stating in words. First, tokens per latent at the optimum scale as M^(α/β-1): if α > β, wider dictionaries need more tokens per latent, not fewer. Second, the compute exponent is small whenever α is near 1: for α = 1.2 it is -0.091 at β = 1 and -0.125 at β = 2 (computed in `results/extrapolation.json`), so a tenfold reduction of U costs a factor of 10^(1/0.091) ≈ 10^11 or 10^(1/0.125) = 10^8 in compute in the asymptotic regime.

**Proposition 4 (budget mismatch).** Suppose the token budget follows T = T₀ (M/M₀)^g and the allocation is balanced at M₀ (ρM₀ = R_{T₀}). Then for M > M₀ the share of the ρM feature slots whose feature is data-learnable is min{1, (M/M₀)^(gβ/α-1)}. The full dictionary remains learnable at all widths if and only if g ≥ α/β.

*Proof.* R_T = ρM₀ (T/T₀)^(β/α) = ρM₀ (M/M₀)^(gβ/α); divide by ρM. ∎

Proposition 4 turns into a falsifiable statement once g is known. Gao et al. (2025) report g ≈ 0.6–0.65 for convergence of reconstruction loss. If the rarity law holds with α/β above that value, then dictionaries trained to reconstruction-loss convergence contain a growing share of latents that cannot have learned the feature of their rank; such latents must be dead, duplicated, or assigned to something else (for instance splitting of frequent features).

## 4 Experimental Setup

All numbers in Sections 5 come from code in `code/` that we ran on CPU; outputs are in `results/`.

**Data.** N = 512 unit-norm feature directions drawn at random in d = 128 dimensions (superposition ratio 4). Feature i is active independently with probability p_i = min(0.25, C i^(-α)), with C set so that the mean number of active features per token is L₀ = 4. Active features have magnitude uniform on [0.5, 1.5]. An activation vector is the sum of its active features times their magnitudes; there is no noise. For α = 1.2, 1.5 and 1.8 the rarest feature has frequency 7.4×10⁻⁴, 3.1×10⁻⁴ and 1.2×10⁻⁴; the range of frequencies spans about 2.5 to 3.3 decades (`results/extras.json`).

**Dictionary.** A TopK sparse autoencoder with k = 8 active latents (twice L₀), M = 768 latents by default, unit-norm decoder rows, tied initialization and the AuxK auxiliary loss for dead latents (coefficient 1/32), following Gao et al. (2025). Gradients are written by hand in numpy. The default optimizer is Adam with learning rate 10⁻³, β₁ = 0.9, β₂ = 0.999 (we write Adam's hyperparameters with subscripts to keep them distinct from the rarity exponent β), batch size 256, for 3 million tokens.

**Recovery metric.** At each of 14 geometrically spaced checkpoints (8,192 to 2,999,808 tokens) we record, for every true feature, the maximum cosine similarity to any decoder row. A feature is *recovered* if this exceeds 0.9. We also record the fraction of variance unexplained (FVU) on fresh data.

**Experiments.** Twenty-nine runs:

- **E1** (6 runs): α ∈ {1.2, 1.5, 1.8} × 2 seeds, baseline settings. Tests the occurrence law.
- **E2** (8 runs): α ∈ {1.2, 1.5} × M ∈ {64, 128, 256, 512}. Tests the capacity assumption.
- **E3** (8 runs): at α = 1.5, one change at a time: learning rate 3×10⁻⁴ or 3×10⁻³; batch 64 or 1024; d = 64 or 256; L₀ = 2 or 8 (with k = 2L₀).
- **E4** (3 runs): a finite buffer of 10⁵, 3×10⁵ or 10⁶ unique tokens replayed in shuffled epochs until 3 million tokens have been processed.
- **E5** (4 runs): Adam with β₂ = 0.99 or 0.9999; heavy-ball momentum SGD (momentum 0.9) with learning rate 0.1 or 0.3.

**Estimators.** (a) *Logistic model.* Over all features and all checkpoints with T ≥ 100,000 we fit P(recovered) = q·σ((log₁₀ p + β log₁₀ T - c)/s) by maximum likelihood, with β free, with β fixed to 1 (occurrence model, whose c is log₁₀ m*) and with β fixed to 0 (frequency-only model). Confidence intervals are from a bootstrap over features (40 replicates in Table 1, 20 in Table 2). (b) *First passage.* For each feature we take the geometric midpoint of the checkpoint interval in which it is first recovered, bin features in groups of 8 by rank, and regress the log of the bin-median recovery time on the log of the bin-median frequency, using all checkpoints; the slope is -κ. Bins whose median is censored are dropped. The two estimators weight the data differently: (a) uses only late checkpoints, where the transition is among rare features, and counts a feature as unrecovered if it is lost again; (b) uses the whole frequency range.

**Reproducibility.** Seeds are fixed. A complete second execution of the training grid in this environment reproduced the stored outputs of the first; the comparison is in `results/rerun_check.txt`.

## 5 Results

### 5.1 Recovery is governed by occurrences, to first order

Figure 1 (`results/fig1_collapse.png`) shows, for each α, the probability of recovery against expected occurrences m = pT (top row) and against frequency p (bottom row), one curve per checkpoint. Against m the curves from training lengths between 10⁵ and 3×10⁶ tokens nearly coincide; against p they are spread over more than a decade.

Table 1 quantifies this. The frequency-only model (β = 0) has a mean negative log-likelihood of 0.51–0.56 per observation; the occurrence model (β = 1) 0.22–0.25; freeing β improves it further to 0.21–0.23 with β between 1.27 and 1.47.

**Table 1.** E1: logistic fits (T ≥ 10⁵; source `results/summary.json`, key `E1`). NLL is the mean per feature-checkpoint observation; m* is the 50% point of the β = 1 model.

| α | seed | NLL, β = 0 | NLL, β = 1 | NLL, β free | β (95% CI, seed 0) | m* (occurrences) | final recovered |
|---|---|---|---|---|---|---|---|
| 1.2 | 0 | 0.515 | 0.223 | 0.215 | 1.32 (1.17–1.45) | 506 | 1.000 |
| 1.2 | 1 | 0.512 | 0.217 | 0.211 | 1.27 | 506 | 1.000 |
| 1.5 | 0 | 0.554 | 0.251 | 0.232 | 1.46 (1.36–1.59) | 344 | 1.000 |
| 1.5 | 1 | 0.547 | 0.241 | 0.227 | 1.39 | 340 | 0.998 |
| 1.8 | 0 | 0.555 | 0.249 | 0.220 | 1.47 (1.37–1.54) | 233 | 0.982 |
| 1.8 | 1 | 0.556 | 0.246 | 0.217 | 1.47 | 230 | 0.990 |

The threshold m* is stable across seeds (506 and 506; 344 and 340; 233 and 230) and varies by about a factor of two across α. The transition is sharp: the fitted logistic width is 0.13–0.15 decades, so going from 10% to 90% recovery probability takes about a factor of 4 in occurrences.

**Out-of-sample check.** We fitted the free-β model on seed 0 and predicted, for seed 1 (different feature directions, different data, different initialization), the recovered fraction and the unrecovered activation mass at each of the 8 checkpoints with T ≥ 10⁵. The mean absolute error of the recovered fraction is 0.014, 0.014 and 0.016 for α = 1.2, 1.5 and 1.8; for unrecovered mass it is 0.0023, 0.0030 and 0.0020. For example at α = 1.5 and T = 309,760 the prediction is 0.423 recovered against 0.461 observed, and at T = 1,209,600 it is 0.941 against 0.957.

### 5.2 The rarity exponent depends on the optimizer and is not sharply identified

If recovery depended on occurrences alone, β would equal 1 and recovery time would scale as 1/p. Table 2 gives both estimators for every training configuration at α = 1.5; Figure 4 (`results/fig4_beta.png`) shows the logistic estimates and thresholds.

**Table 2.** Rarity exponent by configuration (α = 1.5). β_logit: free-β logistic fit with 95% feature-bootstrap interval, and its range over a 3×3 grid of recovery thresholds {0.8, 0.9, 0.95} and transient cuts {3×10⁴, 10⁵, 3×10⁵} where computed (`results/summary.json`, `results/robustness.json`). κ: first-passage slope with 95% bin-bootstrap interval and 1/κ (`results/firstpassage.json`). m*: β = 1 threshold. "Final" is the recovered fraction after 3M tokens.

| configuration | β_logit (95% CI) | β_logit range over analysis grid | κ (95% CI) | 1/κ | m* | final |
|---|---|---|---|---|---|---|
| Adam, baseline | 1.46 (1.36–1.61) | 1.17–1.72 | 0.59 (0.52–0.71) | 1.68 | 344 | 1.000 |
| Adam, lr 3×10⁻⁴ | 1.77 (1.63–1.92) | 1.41–1.97 | 0.52 (0.46–0.63) | 1.92 | 920 | 0.951 |
| Adam, lr 3×10⁻³ | 0.81 (0.73–0.87) | 0.56–1.04 | 0.63 (0.54–0.80) | 1.59 | 164 | 0.973 |
| Adam, β₂ = 0.99 | 1.99 (1.83–2.10) | 1.63–2.31 | 0.48 (0.43–0.54) | 2.09 | 274 | 0.998 |
| Adam, β₂ = 0.9999 | 1.27 (1.18–1.38) | 1.01–1.57 | 0.63 (0.55–0.76) | 1.59 | 381 | 1.000 |
| batch 64 | 1.36 (1.16–1.46) | — | 0.59 (0.51–0.69) | 1.70 | 126 | 0.994 |
| batch 1024 | 1.41 (1.31–1.52) | — | 0.55 (0.46–0.74) | 1.82 | 901 | 0.867 |
| d = 64 | 1.32 (1.26–1.37) | — | 0.60 (0.51–0.75) | 1.67 | 805 | 0.941 |
| d = 256 | 1.06 (1.02–1.16) | — | 0.59 (0.50–0.76) | 1.71 | 196 | 1.000 |
| L₀ = 2, k = 4 | 1.33 (1.26–1.43) | — | 0.52 (0.45–0.64) | 1.92 | 94 | 1.000 |
| L₀ = 8, k = 16 | 1.00 (0.91–1.06) | — | 0.69 (0.58–0.85) | 1.46 | 1,997 | 0.920 |
| SGD, lr 0.1 | 0.77 (0.73–0.82) | 0.61–0.86 | 0.91 (0.78–1.12) | 1.10 | 1,973 | 0.506 |
| SGD, lr 0.3 | 0.53 (0.51–0.58) | 0.38–0.70 | 0.96 (0.80–1.23) | 1.04 | 997 | 0.658 |

What the table supports:

- **Adam is sub-linear in rarity; SGD is close to linear.** By first passage, every Adam configuration has κ between 0.48 and 0.69, and both SGD runs have κ of 0.91 and 0.96 with intervals that include 1. Across the six E1 runs κ is 0.58–0.61. Under baseline Adam the median number of occurrences at first recovery is 643 for the most frequent third of features and 307 for the rarest third, and about 10⁴ for the 8 most frequent features; under SGD with learning rate 0.3 the corresponding medians are 509 and 752, the latter over the rare features that were recovered at all (Figure 6, `results/fig6_firstpassage.png`; `results/extras.json`).
- **Adam's second-moment horizon moves the exponent in a consistent direction on both estimators:** β₂ = 0.99 gives the largest exponents (1.99 and 2.09), β₂ = 0.9999 smaller ones (1.27 and 1.59).
- **The two estimators disagree for some configurations.** For Adam with learning rate 3×10⁻³ the logistic fit gives β = 0.81 and first passage gives 1/κ = 1.59; for SGD the logistic fit gives 0.53–0.77 and first passage 1.04–1.10. The logistic estimate also moves by 0.5 or more across reasonable choices of recovery threshold and transient cut. The causes we can identify are that the logistic fit uses only late checkpoints, where the remaining transition is confined to less than one decade of frequency; that it counts features that were recovered and later lost as failures (1.2% of features at learning rate 3×10⁻³ and 2.7% for SGD at 0.3 were recovered at some checkpoint and not at the last; `results/extras.json`); and that the SGD runs did not finish (51% and 66% recovered), so their fits rest on the frequent half of the features. The relation between log recovery time and log frequency is also not exactly a straight line.

We therefore do not quote a point value. The defensible summary is: **β lies between about 1 and 2; β = 1 (a fixed number of occurrences) is the conservative choice in the sense that it implies the largest token requirement for rare features, and it is what plain SGD gives; Adam at standard settings is nearer β ≈ 1.4–1.7.** The sensitivity analysis in Section 6 carries β ∈ {1, 1.4, 2}.

We do not have a derivation of β for Adam. The direction of the effect is what one expects from per-parameter normalization, which enlarges updates for parameters with infrequent gradients, but a naive argument does not obviously predict the observed dependence on β₂, and we leave the mechanism open.

**What sets m*.** With β fixed to 1, the threshold varies by a factor of about 20 across configurations (Table 2): it falls with a larger learning rate (920, 344, 164 occurrences at 3×10⁻⁴, 10⁻³, 3×10⁻³), rises with batch size (126, 344, 901 at 64, 256, 1024 — roughly constant, 0.9 to 2.0, when expressed in units of batches), rises with the density of superposition (805 at d = 64, 196 at d = 256) and with the number of co-active features (94 at L₀ = 2, 1,997 at L₀ = 8). The threshold is therefore a property of the optimization problem and cannot be transferred to a real model as a number; only its order of magnitude in this setting (10² to 10³) and its dependencies are informative.

### 5.3 Width: a dictionary takes features in order of frequency

In E2 the token budget is ample for almost all features and width binds. Table 3 shows that the number of recovered features is 77–97% of the number of latents and that the recovered features are the frequent ones: the median rank of recovered features is close to half the number recovered (for example 26 for 51 recovered, 108.5 for 216 recovered), as expected if the top ranks are filled in order. Figure 2 (`results/fig2_width.png`) shows recovery against rank falling from near 1 to near 0 around rank ≈ M.

**Table 3.** E2, final checkpoint (`results/summary.json`, key `E2`). ρ is recovered features divided by M. M = 768 is the corresponding E1 run.

| α | M | recovered | ρ | recovered mass | FVU | median rank of recovered |
|---|---|---|---|---|---|---|
| 1.2 | 64 | 51 | 0.80 | 0.721 | 0.225 | 26 |
| 1.2 | 128 | 98 | 0.77 | 0.812 | 0.145 | 49.5 |
| 1.2 | 256 | 216 | 0.84 | 0.907 | 0.080 | 108.5 |
| 1.2 | 512 | 497 | 0.97 | 0.997 | 0.024 | 249 |
| 1.2 | 768 | 512 | (all) | 1.000 | 0.022 | 256.5 |
| 1.5 | 64 | 51 | 0.80 | 0.827 | 0.141 | 26 |
| 1.5 | 128 | 103 | 0.80 | 0.900 | 0.081 | 52 |
| 1.5 | 256 | 221 | 0.86 | 0.955 | 0.042 | 111 |
| 1.5 | 512 | 472 | 0.92 | 0.996 | 0.014 | 237.5 |
| 1.5 | 768 | 512 | (all) | 1.000 | 0.012 | 256.5 |

This supports A1 with ρ ≈ 0.8–0.97. In the M = 768 runs, which have 256 more latents than there are features, 240–251 latents are dead at the end of training: surplus latents with no feature to learn die, as Proposition 4 would have data-starved latents do.

Note also what FVU does and does not show. At α = 1.5, going from M = 256 to M = 512 more than doubles the number of recovered features (221 to 472) while FVU moves from 0.042 to 0.014. Reconstruction loss is insensitive to the tail by construction, which is why a token budget chosen by reconstruction-loss convergence need not be adequate for rare features.

### 5.4 Replay: exposures count, not unique samples

In E4 the dictionary sees 3 million tokens in total, drawn from a fixed buffer of 10⁵, 3×10⁵ or 10⁶ unique tokens (30, 10 or 3 epochs). Final recovered fractions are 0.996, 1.000 and 0.994, against 1.000 with fresh data, and the recovery curves over training are indistinguishable (Figure 3, `results/fig3_replay.png`; for instance, at 309,760 tokens the recovered fractions are 0.451, 0.445 and 0.414 for the three buffers and 0.432 for fresh data). In the 10⁵ buffer, 138 features have fewer than 50 unique occurrences (the minimum is 21), and 99.3% of those are recovered (`results/extras.json`). A logistic model in exposures p·T fits the pooled runs with NLL 0.241; the same model in unique occurrences fits with NLL 0.473.

So in this setting the rarity tax is paid in gradient steps on a feature, not in statistical information about it: a few dozen clean examples, repeated, suffice. We stress the qualifier. The toy features are fixed noise-free directions, so every occurrence is equally informative and nothing can be over-fitted. In real activations a feature's occurrences are embedded in varying context, and replaying few examples could produce a latent that memorizes those contexts. The result identifies a hypothesis worth testing on real models (Section 7), not a recommendation.

## 6 Extrapolation and Sensitivity

This section applies Section 3 to numbers taken from the literature. It involves no experiment on a large model. Every quantity is computed by `code/extrapolate.py` or `code/sourced_checks.py`; inputs from sources are marked as such.

### 6.1 How much does width alone buy?

Sourced input: the largest dictionary of Templeton et al. (2024) has about 34M latents of which roughly 65% are dead, so about 11.7M alive (our arithmetic on their figures). Table 4 gives the unrecovered mass if a dictionary captured exactly the top 1.2×10⁷ features, for several values of the unknown α and N.

**Table 4.** Unrecovered activation mass U if the top 1.2×10⁷ features are recovered, and the number of features needed for U = 10% and U = 1% (`results/extrapolation.json`, `tableA_capacity`).

| α | N | U at R = 1.2×10⁷ | R for U = 10% | R for U = 1% |
|---|---|---|---|---|
| 1.0 | 10⁹ | 0.208 | 1.2×10⁸ | 8.1×10⁸ |
| 1.0 | 10¹⁰ | 0.285 | 9.4×10⁸ | 7.9×10⁹ |
| 1.0 | 10¹² | 0.402 | 6.0×10¹⁰ | 7.5×10¹¹ |
| 1.1 | 10⁹ | 0.075 | 3.9×10⁶ | 4.9×10⁸ |
| 1.1 | 10¹⁰ | 0.100 | 1.2×10⁷ | 4.0×10⁹ |
| 1.1 | 10¹² | 0.133 | 7.7×10⁷ | 2.3×10¹¹ |
| 1.2 | 10⁹ | 0.020 | 3.1×10⁴ | 7.1×10⁷ |
| 1.2 | 10¹⁰ | 0.026 | 3.9×10⁴ | 2.4×10⁸ |
| 1.2 | 10¹² | 0.031 | 4.9×10⁴ | 1.3×10⁹ |
| 1.5 | 10¹⁰ | 0.0002 | 59 | 5.9×10³ |

Figure 5 (`results/fig5_extrapolation.png`; its legend writes γ for the budget exponent g) plots the same quantity against R and the learnable share of Table 6 against width. The table is dominated by α. At α = 1.5 a few thousand features carry 99% of the activation mass and coverage by mass is not a scaling problem at all. At α = 1.0 a dictionary with ten million features leaves 20–40% of the mass unrecovered and reaching 1% requires recovering most of the N features. Between 1.0 and 1.2 the answer changes by orders of magnitude. The public estimates of α quoted in Section 2 range from about 1.0 to 1.7 and none is a measurement of true feature frequencies. **The single most decision-relevant unknown for the scalability of dictionary-based interpretability is therefore the tail exponent of feature frequencies**, and it is measurable (Section 7).

### 6.2 The compute cost of the tail

Table 5 gives the multipliers for going from 10% to 1% unrecovered mass. They are ratios, so the unknown threshold constant c cancels; they depend on α, N and β only.

**Table 5.** Multipliers from U = 10% to U = 1% under balanced allocation (`results/extrapolation.json`, `tableB_multipliers`). Width multiplier is R(1%)/R(10%); token multiplier is the width multiplier to the power α/β; compute is their product.

| α | N | width × | tokens ×, β = 1 | tokens ×, β = 1.4 | tokens ×, β = 2 | compute ×, β = 1 | compute ×, β = 2 |
|---|---|---|---|---|---|---|---|
| 1.0 | 10¹⁰ | 8.4 | 8.4 | 4.6 | 2.9 | 70 | 24 |
| 1.0 | 10¹² | 12.7 | 12.7 | 6.1 | 3.6 | 160 | 45 |
| 1.1 | 10¹⁰ | 332 | 594 | 96 | 24 | 2.0×10⁵ | 8.1×10³ |
| 1.2 | 10¹⁰ | 6.2×10³ | 3.5×10⁴ | 1.8×10³ | 188 | 2.2×10⁸ | 1.2×10⁶ |
| 1.5 | 10¹⁰ | 99 | 988 | 138 | 31 | 9.8×10⁴ | 3.1×10³ |

Two readings. First, the α = 1.0 rows look cheap (a factor of 24–160 in compute) only because at α = 1 the starting point is already expensive: Table 4 shows that U = 10% itself requires about 10⁹ features for N = 10¹⁰. The α = 1.2 and 1.5 rows look expensive in ratio but start from tiny dictionaries (3.9×10⁴ and 59 features). Second, the optimizer exponent matters by one to two orders of magnitude in compute: moving from β = 1 to β = 2 reduces the α = 1.2 multiplier from 2.2×10⁸ to 1.2×10⁶. The rarity exponent is thus a lever, and it is one that practitioners control.

For absolute scale, under β = 1 with a threshold of m* occurrences, L₀ = 100 and N = 10¹⁰, the rank-R feature for U = 10% at α = 1.0 has frequency 4.5×10⁻⁹ and needs 2.2×10¹⁰ tokens at m* = 100 or 2.2×10¹² at m* = 10⁴; at α = 1.1 it needs 5.9×10⁸ to 5.9×10¹⁰ (`tableB2_absolute_beta1`). These are illustrations: m*, L₀ and N for a real model are not known, and Section 5.2 showed that m* varies by a factor of 20 even inside the toy. For comparison, the sourced budget of the largest OpenAI dictionary is 4×10¹⁰ tokens for 1.6×10⁷ latents, that is 2,500 tokens per latent (Gao et al., 2025); under β = 1 such a budget can teach features down to a frequency of m*/T, which is 2.5×10⁻⁹ for m* = 100 and 2.5×10⁻⁷ for m* = 10⁴ (`results/sourced_checks.json`).

### 6.3 Budget mismatch

Sourced input: tokens to convergence grow as n^g with g ≈ 0.60 (GPT-2 small) and 0.65 (GPT-4) (Gao et al., 2025). The rarity law requires T ∝ M^(α/β) (Proposition 3). Table 6 gives the share of latents whose feature is data-learnable when width is scaled up from a balanced reference point with the token budget following n^0.6.

**Table 6.** Learnable share of latents at 10×, 100× and 1000× the reference width, g = 0.60 (`results/extrapolation.json`, `tableC_gao_budget`).

| α | β | exponent gβ/α - 1 | 10× | 100× | 1000× |
|---|---|---|---|---|---|
| 1.0 | 1.0 | -0.40 | 0.40 | 0.16 | 0.06 |
| 1.0 | 1.4 | -0.16 | 0.69 | 0.48 | 0.33 |
| 1.0 | 2.0 | +0.20 | 1 | 1 | 1 |
| 1.2 | 1.0 | -0.50 | 0.32 | 0.10 | 0.03 |
| 1.2 | 1.4 | -0.30 | 0.50 | 0.25 | 0.13 |
| 1.2 | 2.0 | 0.00 | 1 | 1 | 1 |
| 1.5 | 1.0 | -0.60 | 0.25 | 0.06 | 0.02 |
| 1.5 | 1.4 | -0.44 | 0.36 | 0.13 | 0.05 |
| 1.5 | 2.0 | -0.20 | 0.63 | 0.40 | 0.25 |

Unless β is near 2 and α near 1, a budget that tracks reconstruction-loss convergence falls behind the tail, and the shortfall compounds with width. The model predicts that the number of *usefully distinct* latents grows as M^(gβ/α), sub-linearly.

One sourced observation has this shape. From the dead fractions reported by Templeton et al. (2024) (roughly 2%, 35%, 65% at about 1M, 4M and 34M latents), the number of alive latents is 1.03M, 2.73M and 11.7M, which grows as M^0.70 across both intervals (`results/sourced_checks.json`). We do not claim this as support. Dead latents have optimization causes unrelated to data starvation; the reported percentages are rounded; that paper does not state its token counts; and Gao et al. (2025) obtain only 7% dead latents at 16M with a different architecture and auxiliary loss, in which case data-starved latents, if any, would have to show up as duplicates or split features and not as dead ones. The observation shows only that sub-linear growth of usable latents with width has been seen at frontier scale and that the model offers one candidate account, which the protocol below can separate from the alternatives.

## 7 A Pre-Registered Test on Open Models

The model makes predictions that can be checked on an open model with public activations (for example Gemma 2 with the Gemma Scope training setup; Lieberum et al., 2024). We state them in advance. We have not run this protocol.

**Design.** Train TopK SAEs on one residual-stream site on a grid of at least 4 widths × 4 token budgets, spanning a factor of 64 in each (16 dictionaries), plus the two optimizer variants Adam β₂ ∈ {0.99, 0.999} at one grid point and a replay arm (the smallest token budget used as a buffer, replayed to match the largest).

**Probe set.** Since true features are unknown, use concept families whose corpus frequencies can be counted and span at least four decades, as in the feature-completeness analysis of Templeton et al. (2024) (chemical elements, cities, given names), and declare a concept represented if some latent passes a fixed specificity test fixed in advance. Complement this with synthetic features implanted into the activations at controlled frequencies (a known direction added at a known rate), which gives exact ground truth for the dictionary-learning step, though not for the model.

**Predictions.**

- P1 (occurrence law). In the cells where width is not binding, the 50% threshold frequency p₅₀ falls as T^(-β) with β between 1 and 2 and is independent of M. Falsified if p₅₀ does not move with T at fixed M, or if β is below 0.5.
- P2 (regime boundary). In the cells where data is not binding, p₅₀ depends on M and not on T. The grid therefore splits into two regions separated by a line T ∝ M^(α/β). Falsified if p₅₀ depends on both resources everywhere.
- P3 (optimizer). The β₂ = 0.99 arm has a lower p₅₀ at fixed T and M than the β₂ = 0.999 arm for the implanted features.
- P4 (mismatch). Along a path T ∝ M^0.6, the number of distinct concepts represented grows more slowly than M, with exponent near 0.6β/α using the α and β estimated from P1 and P2.
- P5 (replay). For implanted features, the replay arm matches the fresh-data arm (the toy result). For natural concepts we make no directional prediction; a gap there would measure how much real features depart from the noise-free idealization.

P1 and P2 jointly give α, which Section 6 identified as the pivotal unknown. The cost of the protocol is that of 16–20 SAE trainings at sizes already public for Gemma 2; it needs a GPU budget that we do not have.

## 8 Discussion

**What this says about the title question.** For the dictionary-learning route, "does it scale" becomes "how heavy is the tail, and how does the optimizer treat rare features". If true feature frequencies have α ≥ 1.5, mass coverage is cheap and the open problems are elsewhere (non-canonical latents, faithfulness of circuits, the reading bottleneck). If α is near 1, as the inverse-frequency threshold of Templeton et al. (2024) and the central-range exponents of Peter and Søgaard (2025) would suggest, covering the tail by brute force costs orders of magnitude per decade of unrecovered mass, and current token budgets fall behind the width they are paired with. In neither case is the answer a flat yes or no; it is a cost curve with an exponent that can be measured.

**Coverage by mass is the lenient metric.** All exponents above are for activation mass. The share of *features* recovered is R/N, which under the same assumptions is tiny whenever the tail is long: at α = 1.2 and N = 10¹⁰, 99% mass coverage corresponds to 2.4×10⁸ features, 2.4% of the total. If the purpose is to rule out a rare mechanism, mass coverage is the wrong assurance, and no realistic scaling of an untargeted dictionary provides the right one. This points to targeted methods — dictionaries trained on data enriched for the behaviour of interest, or supervised probes for specific features — as complements whose cost does not scale with the tail.

**Levers the law exposes.** (i) The optimizer: the rarity exponent changed from about 1 to about 2 across optimizer settings in the toy, which Table 5 values at one to two orders of magnitude in compute. Optimizers designed to equalize learning across activation frequencies are a concrete direction. (ii) Sampling: if the constraint is exposures (Section 5.4), then over-sampling contexts in which rare features are active, or replaying stored activations, substitutes for running the large model on more tokens, which is the dominant cost at frontier scale. Whether this survives contextual variability is P5. (iii) Stopping rules: reconstruction loss is the wrong convergence criterion for the tail (Section 5.3); a criterion based on recovery of implanted rare features is cheap and direct.

**Relation to capacity-allocation theory.** Michaud et al. (2025) show how width is allocated among features and manifolds at the optimum. Our results add that the optimum may not be reached for rare features within the token budget, and give the condition T ∝ M^(α/β) under which it is. The two analyses compose: their pathological regime reduces the effective ρ, ours limits R through T.

## 9 Limitations

- **Toy setting.** Features are fixed, linear, noise-free, independent directions; magnitudes are bounded; there are no feature manifolds, hierarchies or correlations. Real model features violate all of these, and SAE latents on real models split and absorb (Chanin et al., 2024; Leask et al., 2025). The occurrence law is established only for this setting.
- **Small scale.** N = 512, d = 128, at most 768 latents and 3 million tokens. Frequencies span about three decades and the rarest two thirds of the features lie within 0.6–0.9 decade of each other. We have not shown that the law holds over the eight or more decades at stake in Section 6; extrapolating a fitted exponent that far is an assumption.
- **The exponent is weakly identified.** Two estimators disagree for some configurations, the logistic estimate moves by 0.5 or more with analysis choices, the checkpoint grid is coarse (a factor of about 1.57 between checkpoints), and the SGD runs recovered only 51% and 66% of features. We bound β to roughly [1, 2]; we cannot say where in that interval a real SAE training run lies.
- **No theory of β.** The optimizer dependence is empirical; we do not explain the direction of the β₂ effect.
- **Few seeds and coarse intervals.** Two seeds for E1, one for all other configurations; the logistic-fit confidence intervals use only 20–40 bootstrap replicates and reflect feature sampling within a run, not run-to-run variation. Seed agreement in E1 is close, but the configuration effects in Table 2 rest on single runs.
- **One architecture.** TopK with AuxK only; no ReLU/L1, JumpReLU, transcoders or crosscoders.
- **Threshold not transferable.** m* varies by a factor of 20 within the toy; absolute token counts in Section 6.2 are illustrations.
- **Cost model.** K = M·T ignores the cost of running the language model to produce activations, which is proportional to T and dominates for small dictionaries; including it makes data relatively more expensive and strengthens the case for replay, but changes the exponents in Proposition 3.
- **Pivotal inputs are unknown.** α and N for true features of a frontier model are not known; Section 6 is a sensitivity analysis, not a forecast. The alive-latent exponent of 0.70 is arithmetic on rounded published percentages and has other explanations.
- **Scope.** We address feature discovery by untargeted dictionaries. We say nothing about circuit faithfulness, about whether discovered features are the right units, or about human or automated comprehension at scale. One source (the Templeton et al. paper) was read through its arXiv version because the original site was not reachable from our environment.

## 10 Conclusion

In a controlled superposition setting, whether a sparse dictionary recovers a feature is determined to first order by how many times the feature has occurred in training, with a threshold of a few hundred occurrences and an optimizer-dependent sub-linear correction: recovery time scales as p^(-κ) with κ near 0.6 for Adam and near 1 for SGD. Combined with a capacity limit this yields explicit exponents for coverage as a function of tokens, width and compute, and a condition, T ∝ M^(α/β), that token budgets must satisfy for wide dictionaries to use their latents on the tail. Applied to published numbers, the model says that the feasibility of covering a frontier model's features by scaling dictionaries is decided mainly by the Zipf exponent of feature frequencies, which has not been measured, and secondarily by how the optimizer treats rare features, which is under the practitioner's control. Both can be measured on open models with the protocol we give.

## References

Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N. L., Anil, C., Denison, C., Askell, A., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features

Brill, A. (2024). Neural Scaling Laws Rooted in the Data Distribution. arXiv:2412.07942.

Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., and Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507. (NeurIPS 2025.)

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., and Olah, C. (2022). Toy Models of Superposition. arXiv:2209.10652.

Engels, J., Riggs, L., and Tegmark, M. (2024). Decomposing The Dark Matter of Sparse Autoencoders. arXiv:2410.14670. (*Transactions on Machine Learning Research*.)

Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. (2025). Scaling and evaluating sparse autoencoders. *International Conference on Learning Representations (ICLR)*. arXiv:2406.04093.

Hutter, M. (2021). Learning Curve Theory. arXiv:2102.04074.

Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, Y.-T., Farrell, E., McDougall, C., Ayonrinde, K., Till, D., Wearden, M., Conmy, A., Marks, S., and Nanda, N. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *International Conference on Machine Learning (ICML)*. arXiv:2503.09532.

Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations (ICLR)*. arXiv:1412.6980.

Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., and Nanda, N. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. *International Conference on Learning Representations (ICLR)*. arXiv:2502.04878.

Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., and Nanda, N. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.

Michaud, E. J., Gorton, L., and McGrath, T. (2025). Understanding sparse autoencoder scaling in the presence of feature manifolds. arXiv:2509.02565.

Michaud, E. J., Liu, Z., Girit, U., and Tegmark, M. (2023). The Quantization Model of Neural Scaling. *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2303.13506.

Peter, H., and Søgaard, A. (2025). Empirical Evaluation of Progressive Coding for Sparse Autoencoders. arXiv:2505.00190.

Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramár, J., and Nanda, N. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.

Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J., Biderman, S., Garriga-Alonso, A., Conmy, A., Nanda, N., Rumbelow, J., Wattenberg, M., Schoots, N., Miller, J., Michaud, E. J., Casper, S., Tegmark, M., Saunders, W., Bau, D., Todd, E., Geiger, A., Geva, M., Hoogland, J., Murfet, D., and McGrath, T. (2025). Open Problems in Mechanistic Interpretability. arXiv:2501.16496.

Sous, J., and Winer, M. (2026). Asymmetric Scaling Laws from Sparse Features. arXiv:2605.23591.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Tamkin, A., Durmus, E., Hume, T., Mosconi, F., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*; arXiv:2605.29358.
