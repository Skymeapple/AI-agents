# Familiar Is Not Known: Exposure-Matched and Within-Stratum Tests of Machine Self-Knowledge Signals in Controlled Knowledge Worlds

**Target venue:** ICLR 2027 (main track; controlled empirical study with elementary theory). **Revision R1.** The changes made in response to review are listed in `reviews/response_r1.md`.

---

## Abstract

A model "knows that it does not know" when it can compute a signal that separates the queries it would answer wrongly from the queries it would answer correctly. We study two ways such signals fail and test them in controlled synthetic knowledge worlds, where entity familiarity, fact exposure, type structure and aleatoric ambiguity are set independently. We train small embedding-MLP ensembles on CPU in 7 regimes: 2 world families plus a compositional-name variant, and memorization ranging from perfect to strongly imperfect. There are 3–5 worlds per regime and 5 models per world. We draw three conclusions.

**(1) Ambiguity.** A detector that sees only a label-invariant statistic T(q) of the predictive distribution cannot beat AUROC 1 − α(1−δ_T)²/2. Here α is the aleatoric share of the queries the model knows, and δ_T is the total-variation distance between the laws of T(q) under ignorance and under known ambiguity. This is an elementary corollary of the Youden/TV inequality. The useful addition is that δ_T is invariant to any order-preserving injective recalibration, such as temperature scaling. We find that calibration is not what makes the bound bind; *shape matching* is. A perfectly calibrated Bayes-reference learner has δ_T = 1 in our original world. In a world built so that ignorance and known ambiguity produce the same sorted predictive distribution, the same learner has δ_T = 0, and the best trained q-only detector sits exactly at the ceiling (0.752 vs 0.75 at α = ½; 0.500 vs 0.5 at α = 1). No trained MLP reaches that regime: δ̂_T ranges from 0.51 to 0.97. Early stopping lowers both the calibration error and δ̂_T. Label smoothing lowers the calibration error but raises δ̂_T.

**(2) Familiarity.** The original ambiguity test was confounded by entity familiarity, as the reviewer noted: a −log entity-count baseline reaches AUROC 0.992. After matching on exposure, the count baseline falls to 0.50–0.56. Ensemble mutual information (MI) then stays at 0.96–0.98, but so does a fact-exposure count (1.000). In our worlds MI is a fact-exposure detector. A hidden-state correctness probe trained on natural data has high natural-cell AUROC (0.66–0.93 in 6 of 7 regimes) and labels most familiar-but-wrong answers "known" (false-"knows" rate 0.62–0.90 in 6 of 7 regimes). The striking below-chance dissociated AUROC (0.24) appears only under perfect memorization. Elsewhere it is about 0.50–0.57.

**(3) Within-stratum evaluation.** Within a familiarity stratum, the hidden-state probe has essentially no fact-specific signal in any regime: AUROC 0.45–0.52 on familiar entities' never-seen facts, and 0.50–0.57 on rare entities' seen facts. Output-distribution signals keep real signal. In imperfect-memorization regimes, maximum probability matches ensemble MI on seen-fact strata, with AUROC 0.78–0.98.

A Gaussian two-direction model of the probe (Proposition 2, an instance of the standard core/spurious-feature model) gets the direction right but fits magnitudes poorly (r = 0.36, MAE 0.15 over 69 runs). LDA fitted to the actual hidden-state covariance reproduces the probe almost exactly (r = 0.97, MAE 0.03). The shortcut is therefore linear-discriminant behaviour on a geometry that two cell-mean vectors do not summarize. All results come from CPU-scale synthetic experiments. We could not run an LLM pilot: model-weight hosts are unreachable from our environment. We give a protocol for one instead.

## 中文摘要

如果模型能计算出一个信号，把它会答错的问题与会答对的问题区分开，我们就说它"知道自己不知道"。本文研究这类信号失效的两种方式，并在受控的合成知识世界中检验；在这些世界里，实体熟悉度、事实曝光、类型结构与偶然歧义彼此独立设定。我们在 7 种设定下（两类世界加一个"组合式名字"变体，记忆程度从完美到明显不完美；每种设定 3–5 个世界，每个世界 5 个模型）用 CPU 训练小型嵌入 MLP 集成，得到三点结论。

**（1）歧义。** 只看预测分布某个标签无关统计量 T(q) 的检测器，AUROC 不超过 1 − α(1−δ_T)²/2。其中 α 是模型"知道"的问题中有歧义者所占的比例，δ_T 是"无知"与"已知歧义"两种情形下 T(q) 分布之间的全变差距离。这是 Youden/TV 不等式的直接推论；有用的补充在于：δ_T 在任何保序单射重校准（例如温度缩放）下不变。我们发现，让这个界起约束作用的不是校准，而是"形状匹配"。一个完美校准的贝叶斯参照学习器在原始世界里 δ_T = 1；在另一个世界里，我们特意让"无知"与"已知歧义"产生相同的排序后预测分布，同一学习器的 δ_T = 0，最优的已训练 q-only 检测器恰好落在上限上（α = ½ 时为 0.752，上限 0.75；α = 1 时为 0.500，上限 0.5）。训练出的 MLP 都没有进入这一区域：δ̂_T 在 0.51 到 0.97 之间。早停同时降低校准误差和 δ̂_T；标签平滑降低校准误差，却使 δ̂_T 升高。

**（2）熟悉度。** 正如审稿人指出的，原来的歧义实验受实体熟悉度混淆：−log 实体计数基线的 AUROC 达到 0.992。按曝光匹配后，计数基线降到 0.50–0.56；集成互信息（MI）仍保持在 0.96–0.98，但事实曝光计数同样如此（1.000）。在我们的世界里，MI 实际上是在检测事实曝光。用自然数据训练的隐藏状态正确性探针，在 7 种设定中有 6 种的自然单元 AUROC 较高（0.66–0.93），并把大多数"熟悉但答错"的回答判为"知道"（同样 6/7 种设定下误判率为 0.62–0.90）。分离单元上那个引人注目的低于随机水平的 AUROC（0.24）只在完美记忆时出现；其他设定下约为 0.50–0.57。

**（3）层内评测。** 在同一熟悉度层内，隐藏状态探针在任何设定下都几乎没有针对具体事实的信号：对熟悉实体的未见事实，AUROC 为 0.45–0.52；对稀有实体的已见事实，为 0.50–0.57。基于输出分布的信号仍保有真实信号；在记忆不完美的设定下，最大概率在已见事实层上与集成 MI 相当（AUROC 0.78–0.98）。

探针的高斯双方向模型（命题2，是标准"核心/伪特征"模型的一个实例）方向正确，但幅度拟合差（69 次运行上 r = 0.36，平均绝对误差 0.15）；而按隐藏状态实际协方差拟合的 LDA 几乎完全复现了探针（r = 0.97，平均绝对误差 0.03）。所以这一捷径是线性判别作用在一个无法用两个单元均值向量概括的几何结构上的结果。所有结果都来自 CPU 规模的合成实验。我们没能做大语言模型的小规模实验，因为本环境无法访问模型权重的托管站点；文中给出了相应的实验方案。

---

## 1 Introduction

Asking how to make a model reliably know that it does not know is, operationally, asking for a *signal* s(x) that the model can compute and that ranks queries by the probability that its answer is wrong. Such a signal is useful for abstention, retrieval triggering, or hedged answers. Two families of signals dominate current practice. The first reads the predictive distribution q(·|x): softmax response [Geifman & El-Yaniv 2017], sequence probability and calibration [Guo et al. 2017; Kadavath et al. 2022], and semantic entropy [Farquhar et al. 2024]. The second trains a probe on hidden activations to predict correctness: P(IK) heads [Kadavath et al. 2022], truthfulness and error probes [Azaria & Mitchell 2023; Orgad et al. 2025], fine-tuned uncertainty heads [Kapoor et al. 2024], and pre-generation estimators that read the subject representation [Gottesman & Geva 2024]. Mechanistic work finds "known entity" directions that causally gate refusal [Ferrando et al. 2025].

Two warnings recur in the LLM literature. Error probes do not transfer across datasets [Orgad et al. 2025]. Internal states mainly reflect *knowledge recall* rather than truthfulness, so hallucinations associated with a known subject look like correct recall [Cheang et al. 2026]. Separately, confidence and accuracy both rise with entity popularity, and overconfidence concentrates on popular knowledge [Ni et al. 2025; see also Mallen et al. 2023]. Cognitive psychology has a long-standing account of the same pattern in people: the initial feeling of knowing tracks familiarity with the question's terms rather than access to the answer [Reder & Ritter 1992; Koriat 1993].

These observations are made on LLMs, where the relevant variables cannot be controlled: pre-training exposure of each fact, entity familiarity, and whether an answer distribution is genuinely ambiguous. This paper takes the complementary route. We build synthetic knowledge worlds in which these variables are set independently, train small networks in several memorization regimes, and ask which self-knowledge signals survive *evaluation designs that remove the confounds*. Our contribution is mainly methodological and empirical. The theory is elementary, and we present it as such.

**Contributions.**

- **C1 (evaluation design).** Two controls that any self-knowledge evaluation can adopt. (a) *Exposure-matched* positives and negatives, with a −log entity-count baseline reported in every table. (b) *Within-familiarity-stratum* AUROC, for example correct vs wrong among never-seen facts about familiar entities, which removes the threshold artefacts of pooled "dissociated" cells. We show that each control changes conclusions that the uncontrolled design supported (Sections 6.3–6.4).
- **C2 (ambiguity, theory + test).** The Youden/TV ceiling for q-only detectors, stated for any label-invariant statistic T(q), with two additions. The ceiling is invariant to order-preserving injective recalibration (Corollary 1). And it binds under *shape matching*, not under calibration as such. We construct a world in which a calibrated reference learner attains the ceiling exactly, and we show that no trained MLP in 7 regimes reaches that regime (Section 6.3).
- **C3 (familiarity, test across regimes).** Hidden-state correctness probes trained on natural data inherit a familiarity shortcut in 6 of 7 regimes. Within familiarity strata they carry essentially no fact-specific signal in any regime. We also show where the Gaussian two-direction model of this shortcut (an instance of the standard spurious-feature model) fails: the problem is the two-vector summary of the geometry, not the linear-discriminant mechanism (Section 6.5).
- **C4 (scope conditions).** Several first-round findings hold only under perfect memorization and are now reported as such: the below-chance dissociated AUROC, the claim that "only ensemble MI is reliable", and the Good–Turing abstention budget.

## 2 Related Work

**Self-knowledge in language models.** Kadavath et al. [2022] found large LMs reasonably calibrated on formatted multiple-choice questions, with a P(IK) head that partially generalizes. Yin et al. [2023] built SelfAware and reported a substantial gap to human self-knowledge. Kapoor et al. [2024] found that fine-tuning on roughly a thousand graded examples yields useful uncertainty where prompting does not. Azaria & Mitchell [2023] trained truthfulness classifiers on hidden states. Orgad et al. [2025] found that error detectors fail to generalize across datasets. Gottesman & Geva [2024] (KEEN) estimate entity-level knowledge from subject representations before generation. Ferrando et al. [2025] found SAE entity-recognition directions that steer refusal and hallucination. Cheang et al. [2026] report that LLM hidden states mainly reflect knowledge recall. In their study, hallucinations associated with the subject cannot be separated from correct recall, while unassociated ones can. Ni et al. [2025] report that LLM accuracy and confidence both rise with knowledge popularity, and that overconfidence is tied to popularity. **These papers already establish the familiarity/popularity confound empirically in LLMs.** We do not claim to discover it. What we add is a controlled setting in which exposure, familiarity and ambiguity can be separated, together with evaluation designs (exposure matching, within-stratum AUROC) that test whether a signal measures knowledge beyond familiarity.

**Spurious correlations and shortcuts.** Proposition 2 below is an instance of the Gaussian "core vs spurious feature" model analysed in Sagawa et al. [2020], Nagarajan et al. [2021] and Rosenfeld et al. [2021]. We relabel the spurious feature as entity familiarity and the core feature as fact-specific knowledge. Its only new element is a closed form for the error on the familiar-unknown cell as a function of the training co-occurrence π. It instantiates shortcut learning [Geirhos et al. 2020].

**Why models hallucinate.** Kalai & Vempala [2024] relate the hallucination rate of calibrated LMs to the Good–Turing missing mass. Kalai et al. [2025] argue that binary grading rewards guessing. Long-tail studies link accuracy to popularity and pre-training counts [Mallen et al. 2023; Kandpal et al. 2023]. Fine-tuning on new knowledge increases hallucination [Gekhman et al. 2024]. Knowledge can be stored without being extractable [Allen-Zhu & Li 2024]. Our synthetic worlds follow this controlled-corpus methodology.

**Aleatoric vs epistemic uncertainty.** The qualitative point that q alone cannot separate ambiguity from ignorance is standard [Hüllermeier & Waegeman 2021]. Wimmer et al. [2023] criticize the entropy/MI decomposition. Bengs et al. [2022] show that second-order loss minimisation does not deliver faithful epistemic uncertainty. Deep ensembles [Lakshminarayanan et al. 2017] are the usual non-q-only signal. Proposition 1 makes the qualitative point quantitative by applying the textbook inequality TPR − FPR ≤ TV. Corollary 1 and the shape-matching experiment are what we add.

**Selective prediction.** Chow [1970] gave the optimal reject rule. Geifman & El-Yaniv [2017] gave risk-controlled selective classifiers. Guo et al. [2017] documented miscalibration.

## 3 Setting and Definitions

A query x = (e, r) asks for the value of relation r for entity e. The world has a conditional p*(·|x) over a finite value set V. This is a point mass for *deterministic* facts and spread out for *aleatoric* facts. A trained model outputs q(·|x).

**Knowing.** We use (a) *correctness* K(x) = 1[argmax q = argmax p*], and (b) *epistemic gap* G(x) = 1[TV(q, p*) > 0.5]. For an aleatoric fact the model has learned, G = 0 even though q is spread out. In the synthetic world we also know each fact's training exposure c(e, r), and we call a fact *seen* if c ≥ 1.

**Familiarity.** F(x) is a function of the entity's total training count n_e: familiar if n_e ≥ 50, rare if 1 ≤ n_e ≤ 10. The five deterministic *strata* are familiar-seen (FS), familiar-unseen-fact (FU), rare-seen (RS), rare-unseen-fact (RU) and unseen entity (UE). The *natural* pool is FS ∪ UE, and the *dissociated* pool is FU ∪ RS.

**Detectors.** A *q-only* detector is a measurable function of q. A *label-invariant* q-only detector factors through T(q) = the sorted probability vector. This covers maximum probability (MSP), entropy, temperature-scaled versions of these, top-k features, and in practice verbalized confidence and semantic entropy (see Remark 3). A *probe* is a function of a hidden representation fitted to predict K.

## 4 Theory

### 4.1 Proposition 1 (ambiguity ceiling): an elementary corollary

Let positives (G = 1) have q ~ P₀. Let negatives be a mixture: with probability 1−α a deterministic known item (q ~ P_det), and with probability α an aleatoric known item (q ~ P_alea). For a statistic T, let δ_T = TV(T#P₀, T#P_alea).

**Proposition 1.** For every detector s = h∘T,

  AUROC(s) ≤ (1−α)·A(s; P₀, P_det) + α·[1 − (1−δ_T)²/2] ≤ 1 − α(1−δ_T)²/2,

and for every α and δ there are laws attaining equality.

*Proof.* AUROC is linear in the negative law, so it splits into the two mixture components. Every threshold on s defines an event in the σ-algebra generated by T, so its (FPR f, TPR t) satisfies t − f ≤ δ_T, which is the Youden/TV inequality. The ROC curve therefore lies below min(1, f + δ_T), and ∫₀¹ min(1, f+δ) df = 1 − (1−δ)²/2. Tightness: take P_det disjoint from P₀. Take T#P_alea equal to T#P₀ except for a δ-mass of P₀ that is disjoint from it, and rank by likelihood ratio. ∎

This is a direct consequence of standard facts: the ROC/TV relation and the linearity of AUROC in the negative mixture. We state it because the *detector-specific* first form is what the experiments test. The dotted reference line in `fig_aleatoric.png` uses the measured A(s; P₀, P_det) of MSP (0.816). The detector-agnostic second form starts at 1 when α = 0. Table 2 reports the measured curves, not the bound.

**Corollary 1 (recalibration invariance).** Let g map the simplex into itself, let it be injective, and let it preserve the order of coordinates. Temperature scaling q ↦ softmax(log q / τ) for fixed τ > 0 is an example. Then sort(g(q)) is an injective function of sort(q), so δ_T is unchanged. No post-hoc, order-preserving recalibration can move the ceiling. *Proof.* TV is invariant under injective measurable maps. ∎

**Remarks.** (1) *Calibration is not sufficient for δ_T → 0.* δ_T measures whether ignorance and known ambiguity produce different *shapes* of sorted predictive distribution. A perfectly calibrated learner can output a type-prior shape under ignorance (e.g. 0.61 on one value, with the rest flat) and a (0.5, 0.3, 0.2) shape for a known coin-flip. These two shapes differ, so δ_T = 1. The ceiling binds when the two shapes coincide, which we call *shape matching*. Section 6.3 builds a world in which this holds. (2) The bound does not apply to signals that are not functions of q, such as ensemble disagreement or exposure counts. (3) Semantic entropy is computed from M samples of q. Its value is therefore a randomized function of q, and Proposition 1 applies to its expectation over sampling noise: the extra randomness cannot increase δ, by data processing. Whether it is label-invariant depends on whether the clustering uses answer identity. We do not test semantic entropy.

### 4.2 Proposition 2 (familiarity shortcut): an instance of the spurious-feature model

Whitened hidden state z = a·F·u + b·K·v + ε, with ε ~ N(0, σ²I) and u ⊥ v. Here F is the "spurious" feature (familiarity) and K the "core" feature (knowledge). This is the Gaussian model of Sagawa et al. [2020] and Nagarajan et al. [2021]. A probe is trained on data with P(K=1) = ½ and P(F = K) = π.

**Proposition 2.** Let κ(π) = (2π−1)σ²/(σ² + a²π(1−π)). The population LDA probe labels a familiar-unknown item "knows" with probability FK(π) = Φ((a²κ − b²)/(2σ√(a²κ² + b²))). (i) FK is increasing on π ∈ [½, 1). (ii) FK(½) = Φ(−b/2σ). (iii) As π → 1, FK → Φ((a²−b²)/(2σ√(a²+b²))), which exceeds ½ iff a > b. (iv) In-distribution accuracy stays at least π·Φ(√(a²+b²)/2σ). The proof is unchanged from R0: LDA with Σ_W = σ²I + a²π(1−π)uuᵀ, and a direct differentiation of g(κ).

**Domain (corrected).** The statements "monotone" and "cannot fall below Φ(−b/2σ)" hold on π ∈ [½, 1]. For π < ½ (anti-correlated training data), FK can fall below Φ(−b/2σ).

**Unbalanced classes.** `theory.lda_rates_general` drops the P(K=1) = ½ assumption. It uses the actual P(K=1) = p₁, P(F=1|K=1) = π₁ and P(F=0|K=0) = π₀, a pooled covariance, and the Bayes threshold log(p₀/p₁). We use it in Section 6.5 to test whether class imbalance explains the R0 misfit. It does not.

## 5 Controlled Experimental Setup

All numbers come from code in `code/` with outputs in `results/` (R0: `run_all.py`, `e1_qonly.py`; R1: `train_cache.py`, `analysis_r1.py`, `aggregate_r1.py` → `results/r1/`). Seeds are fixed.

**Original world** (`world.py`). 4,000 entities, 8 relations (6 deterministic, 2 aleatoric), 32 values, 8 latent types. A deterministic fact takes its type's default value with probability ρ = 0.6, otherwise a uniform value. Aleatoric facts put (0.5, 0.3, 0.2) on three values. 75% of entity–relation pairs are documented, 10% of entities never appear, popularity is Zipf(1), and there are 150,000 training samples.

**Shape-matched world** (new; `matched=True`). Each (type, relation) has three candidate values with prior (0.5, 0.3, 0.2). A deterministic fact's value is drawn once from this prior. An aleatoric fact's p* *is* this prior. A learner that knows the generative structure therefore outputs the same sorted q for "unseen deterministic fact" and "known aleatoric fact", so δ_T = 0.

**Compositional-name world** (new; `comp_names=True`). Each entity name is a pair of sub-tokens drawn from 2 × 300. The first sub-token is tied to the entity's type with probability 0.8. An unseen name is composed of *trained* sub-tokens and carries type information, in the way LLMs compose new names. This addresses the reviewer's point that random embeddings make unseen entities trivially out-of-distribution.

**Model and regimes** (`mlp.py`, `train_cache.py`). Embedding MLP: [entity (32); relation (32)] → ReLU(256) → softmax(32), trained with Adam and 5 ensemble members per world. Member 0 is "the model". Seven regimes:

| regime | world | training | memorization of seen facts (FS / RS accuracy) |
|---|---|---|---|
| base | original | 30 epochs | 1.000 / 1.000 |
| early | original | 3 epochs | 0.909 / 0.554 |
| small | original | width 32, emb 16, 10 epochs | 0.910 / 0.755 |
| comp | compositional names | 20 epochs | 0.997 / 0.622 |
| matched | shape-matched | 30 epochs | 1.000 / 1.000 |
| matched_early | shape-matched | 3 epochs | 0.985 / 0.689 |
| matched_ls | shape-matched | label smoothing 0.2, 10 epochs | 1.000 / 0.992 |

*Table 0. Measured (`results/r1/summary_r1.txt`, T-G). base uses 5 worlds; the others use 3.*

The *base* regime reproduces R0's networks exactly: same seeds. Per-world probe false-"knows" rates match R0 to within 0.002, the residual coming from float16 caching of hidden states. Worlds 3 and 4 are new. A **Bayes reference learner** (`analysis_r1.bayes_reference`) gives the exact predictive distribution of the generative model given the training counts, with oracle entity types. It is perfectly calibrated (deterministic ECE 0.002) and needs no training.

**Designs.**
- *E1 original* (R0): positives are deterministic items with G = 1, and negatives are a mix of deterministic G = 0 items and aleatoric items with c ≥ 20 and G = 0. About 27% of positives come from never-seen entities.
- *E1 exposure-matched* (new): everything is restricted to familiar entities (n_e ≥ 50). Positives are never-seen deterministic facts answered wrongly (FU-wrong). Negatives are seen-and-correct deterministic facts or known aleatoric facts (c ≥ 20, G = 0), resampled to match the positives' histogram over 7 entity-count bins. We report −log entity count (should be ≈ 0.5 if matching works) and −log fact count (an exposure oracle).
- *δ̂_T* (revised): a cross-fitted logistic classifier on T-features (top-5 sorted log-probabilities and entropy) separates positives from known-aleatoric items. Its threshold is fixed by Youden's J on the *training* fold and applied to the held-out fold, which addresses the optimism the reviewer pointed out. We give 90% bootstrap intervals over held-out items. The best trained q-only detector is the same logistic model trained on the full E1 task.
- *Probes* (E3): L2-logistic probes on hidden (256-d), hidden+output, or output-only features. They are trained on 1,500 items from held-out-entity halves, with a share π_d ∈ {0, 0.1, 0.5} from the dissociated pool, 5 resamples each. Ablations: L2 strength λ ∈ {0.01, 1, 100}, and threshold 0.5 vs a Youden threshold fixed on the training set.
- *Within-stratum AUROC* (new): AUROC for detecting wrong answers inside a single stratum (FU, RU, UE, and FS/RS where memorization is imperfect), on held-out entities.

**Statistics.** Values are means over worlds. ± values are now **95% t-intervals across worlds** (n = 5 for base, n = 3 otherwise). With n = 3 these intervals are wide (t = 4.30), and we interpret only differences that clear them.

**Compute.** R1 training took 43.2 CPU-minutes (19.7 min wall-clock on 4 cores, single-threaded BLAS per process; `results/rev_train_log.txt`), and R1 analyses took about 5 CPU-minutes. The reviewer measured about 3.2 h per world for R0's `run_all.py`, run concurrently with multithreaded BLAS on a 4-core machine. With `OMP_NUM_THREADS=1`, one base world (5 members) takes 216–269 s.

## 6 Results

### 6.1 The algebra matches simulation (unchanged from R0)

The Monte Carlo LDA probe matches Proposition 2's closed form at all 20 (a, b, π) settings, with maximum error 0.0034. A δ = 0 construction attains 1 − α/2 to within 0.0015 (`results/theory_check.json`, `fig_theory.png`). This checks the algebra only.

### 6.2 Memorization regimes and where errors live

| regime | FU acc / MSP | RU acc / MSP | UE acc / MSP | ECE (det.) | singleton frac. | missing mass | det. error rate | share of errors on c = 0 |
|---|---|---|---|---|---|---|---|---|
| base | 0.286 / 0.865 | 0.270 / 0.852 | 0.062 / 0.762 | 0.267 | 0.038 | 0.038 | 0.028 | 0.998 |
| early | 0.358 / 0.598 | 0.258 / 0.457 | 0.079 / 0.426 | 0.067 | 0.038 | 0.038 | 0.119 | 0.234 |
| small | 0.229 / 0.682 | 0.229 / 0.616 | 0.080 / 0.544 | 0.139 | 0.038 | 0.038 | 0.095 | 0.311 |
| comp | 0.353 / 0.709 | 0.301 / 0.687 | 0.285 / 0.691 | 0.158 | 0.038 | 0.038 | 0.070 | 0.383 |
| matched | 0.394 / 0.885 | 0.322 / 0.871 | 0.086 / 0.800 | 0.261 | 0.038 | 0.038 | 0.026 | 1.000 |
| matched_early | 0.388 / 0.709 | 0.305 / 0.564 | 0.087 / 0.517 | 0.093 | 0.038 | 0.038 | 0.072 | 0.366 |
| matched_ls | 0.378 / 0.426 | 0.378 / 0.437 | 0.083 / 0.291 | 0.117 | 0.038 | 0.038 | 0.025 | 0.949 |
| Bayes ref. (orig.) | 0.619 / 0.612 | 0.613 / 0.612 | 0.144 / 0.138 | 0.002 | — | — | — | — |

*Table 1. Measured (T-G in `summary_r1.txt`). Per-regime 95% intervals are in the summary file.*

The base regime is the degenerate one the reviewer described. Errors fall essentially only on never-seen facts (99.8%), and never-seen facts receive confident answers (MSP 0.86) that are right 27–29% of the time. A type-default predictor would be right about 62% of the time, and the Bayes reference is. The other regimes break this in different ways. With early stopping or small capacity, 23–31% of errors fall on never-seen facts and the rest on imperfectly memorized seen facts. In the compositional world, unseen entities are answered at 29% (vs 6%), because the name carries type information.

**Good–Turing (reviewer m5).** The R0 claim that the singleton fraction is "an accurate, label-free abstention budget" holds only under (near-)perfect memorization. In the four imperfect-memorization regimes (early, small, comp, matched_early), the deterministic error rate (0.070–0.119) is 1.8–3.1× the singleton fraction (0.038), and only 23–38% of errors fall on never-seen facts. We withdraw the general claim.

### 6.3 Ambiguity: the confound, exposure matching, and when Proposition 1 binds

**The R0 design was confounded (reviewer M2).** Table 2 adds the count baselines that R0 computed but omitted.

| detector (base, 5 worlds) | α = 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| MSP (1 − max q) | 0.816 | 0.652 | 0.494 | 0.331 | 0.169 |
| entropy of q | 0.814 | 0.657 | 0.504 | 0.347 | 0.191 |
| entropy of ensemble mean | 0.949 | 0.909 | 0.872 | 0.832 | 0.794 |
| ensemble MI | 0.950 | 0.954 | 0.960 | 0.965 | 0.970 |
| **−log entity count** | 0.748 | 0.804 | 0.869 | 0.929 | **0.992** |
| **−log fact count (exposure oracle)** | 0.924 | 0.941 | 0.963 | 0.980 | **1.000** |

*Table 2. Original E1 design, AUROC for G = 1, measured. 95% intervals ≤ ±0.012.*

At high α the R0 task is largely a familiarity task: aleatoric negatives with c ≥ 20 sit on heavily trained entities, while many positives sit on never-seen ones. R0's statement that MI is flat in α "because it separates ignorance from ambiguity" was not supported by that design.

**Exposure-matched E1.** Restricting to familiar entities and matching entity-count histograms brings the familiarity baseline down to 0.50–0.56 (Table 3).

| base regime, exposure-matched | α = 0 | α = 0.5 | α = 1 |
|---|---|---|---|
| MSP | 0.835 ± 0.022 | 0.478 ± 0.025 | 0.118 ± 0.031 |
| best trained q-only detector (T-features) | 0.833 ± 0.022 | 0.738 ± 0.036 | 0.947 ± 0.014 |
| ensemble MI | 0.984 ± 0.008 | 0.973 ± 0.012 | 0.962 ± 0.018 |
| entropy of ensemble mean | 0.982 ± 0.008 | 0.845 ± 0.024 | 0.708 ± 0.044 |
| −log entity count | 0.497 ± 0.002 | 0.525 ± 0.030 | 0.555 ± 0.057 |
| −log fact count | 1.000 | 1.000 | 1.000 |

*Table 3. Measured (T-B). About 330 positives, 1,630 deterministic negatives and 220 aleatoric negatives per world.*

MI keeps its separation once familiarity is controlled (0.96–0.98), so the R0 result was not purely a familiarity artefact. The fact-exposure oracle, however, is perfect. In these worlds "knowing" and "having seen the fact" nearly coincide, and MI is best read as a **fact-exposure detector**. The reviewer's M1 point stands: this world cannot distinguish "MI tracks knowledge" from "MI tracks exposure". The same pattern (MI ≥ 0.93, entity count ≤ 0.59, fact count 1.000) holds in all seven regimes (T-B).

**Is Proposition 1 ever binding? (reviewer M3.)** Table 4 gives δ̂_T (with the threshold fixed on the training fold), the best trained q-only detector at α = 1 in the exposure-matched design, and calibration error.

| learner / regime | ECE (det.) | δ̂_T [90% boot.] | q-only AUROC, pos vs aleatoric | best q-only, α = 1 | ceiling at α = 1 from δ̂_T |
|---|---|---|---|---|---|
| Bayes ref., original world | 0.002 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 |
| **Bayes ref., shape-matched world** | 0.003 | **0.000 [0.00, 0.00]** | **0.500** | **0.500** | **0.500** |
| MLP base | 0.267 | 0.785 [0.74, 0.83] | 0.951 | 0.947 | 0.977 |
| MLP early | 0.067 | 0.513 [0.44, 0.58] | 0.840 | 0.890 | 0.878 |
| MLP small | 0.139 | 0.603 [0.53, 0.67] | 0.881 | 0.924 | 0.921 |
| MLP comp | 0.158 | 0.630 [0.58, 0.68] | 0.898 | 0.874 | 0.931 |
| MLP matched | 0.261 | 0.906 [0.88, 0.93] | 0.980 | 0.977 | 0.996 |
| MLP matched_early | 0.093 | 0.704 [0.65, 0.76] | 0.934 | 0.918 | 0.956 |
| MLP matched_ls | 0.117 | 0.973 [0.96, 0.99] | 1.000 | 0.999 | 1.000 |

*Table 4. Measured (T-C). Bracketed: mean per-world 90% bootstrap interval over held-out items. For early and small, the best q-only value slightly exceeds the ceiling computed from δ̂_T, which is expected because δ̂_T is a classifier-based lower bound on δ_T; the two agree within their across-world intervals (±0.05–0.12).*

Four findings follow.

1. **Proposition 1 is tight on a calibrated learner under shape matching.** In the shape-matched world the Bayes reference has δ_T = 0, and the best q-only detector gives 1.000, 0.752 ± 0.019 and 0.500 at α = 0, ½, 1. The ceiling (1 − α)·1 + α/2 gives 1.0, 0.75 and 0.5 (`fig_r1_prop1_strata.png`, left). In the same world the fact-exposure signal stays at 1.000. Information about ignorance exists, but none of it is in q.
2. **Calibration alone does not make it bind.** The same perfectly calibrated learner in the original world has δ_T = 1, because ignorance gives a type-prior shape and known ambiguity gives a (0.5, 0.3, 0.2) shape. R0's conjecture that "calibration pushes δ → 0" is false as stated.
3. **Trained networks never reach the binding regime.** δ̂_T ranges from 0.51 to 0.97, and the best q-only detector recovers 0.87–1.00 at α = 1. Early stopping, which lowers ECE, also lowers δ̂_T (base 0.785 → early 0.513; matched 0.906 → matched_early 0.704). Label smoothing, which also lowers ECE (0.261 → 0.117), *raises* δ̂_T to 0.973, because it gives seen-aleatoric outputs a distinctive smoothed shape. Across the six MLP regimes without label smoothing, ECE and δ̂_T are rank-correlated (Spearman ρ = 0.77, n = 6). That is too few points to call it a law.
4. **Fixed monotone scores still fail.** MSP is below 0.5 at α = 1 in four of seven regimes (base 0.169, matched 0.087, small 0.474, comp 0.448; T-A). It inverts because these networks are more confident under ignorance than on known coin-flips. This is a property of a fixed score, not an information limit: a trained q-only detector avoids it.

### 6.4 Familiarity: pooled vs within-stratum evaluation of probes

| regime | hidden probe, natural | hidden probe, dissociated (pooled) | hidden probe, **within FU** | hidden probe, **within RS** | false-"knows" (FU-wrong) | output probe, within FU | MSP, within FU | ens. MI, within FU | type-default disagreement, within FU | −log entity count, dissociated |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.929 | **0.239** | 0.480 | n/a | 0.899 | 0.678 | 0.678 | 0.738 | 0.879 | 0.012 |
| early | 0.783 | 0.541 | 0.484 | 0.566 | 0.711 | 0.660 | 0.669 | 0.657 | 0.845 | 0.516 |
| small | 0.662 | 0.514 | 0.517 | 0.553 | 0.492 | 0.664 | 0.666 | 0.675 | 0.893 | 0.387 |
| comp | 0.516 | 0.502 | 0.516 | 0.504 | 0.632 | 0.666 | 0.663 | 0.681 | 0.836 | 0.511 |
| matched | 0.875 | 0.574 | 0.521 | n/a | 0.826 | 0.567 | 0.563 | 0.582 | 0.628 | 0.015 |
| matched_early | 0.847 | 0.508 | 0.451 | 0.555 | 0.838 | 0.548 | 0.568 | 0.569 | 0.639 | 0.487 |
| matched_ls | 0.703 | 0.550 | 0.496 | 0.539 | 0.625 | 0.556 | 0.570 | 0.546 | 0.647 | 0.075 |

*Table 5. AUROC for detecting wrong answers on held-out entities. The probe is trained on natural data (π_d = 0, λ = 1). Measured (T-D); 95% intervals across worlds are in `summary_r1.txt`. Within FU, the per-world n is about 230 items with about 150 wrong, so intervals are wide: ±0.02–0.17. "n/a": no wrong answers in that stratum (perfect memorization).*

The table supports three conclusions.

**(a) The pooled dissociated AUROC is largely an artefact of the thresholds, as the reviewer argued (M5).** The striking below-chance value (0.239) occurs only in the base regime. There the dissociated pool is "RS = 100% correct" plus "FU ≈ 70% wrong", and the cells are defined by entity-count thresholds. A familiarity score must then invert, as the count baseline does (0.012). Wherever seen facts are imperfectly memorized, the pooled dissociated AUROC of the hidden probe is about 0.50–0.57.

**(b) Within strata, the hidden-state probe has essentially no fact-specific signal in any regime.** Within FU it scores 0.45–0.52. Within RS, where it can be measured, it scores 0.50–0.57. Within FS it scores 0.65–0.72 (early, small, matched_early; T-D), against 0.94–0.98 for MSP and MI. Output-distribution signals keep real signal in every regime. Within FU, MSP scores 0.56–0.68. Within the seen-fact strata of imperfect regimes, MSP scores 0.78–0.98 and matches ensemble MI: early FS 0.956 vs 0.948, RS 0.828 vs 0.806; small RS 0.912 vs 0.921. The probe's high natural-cell AUROC therefore comes almost entirely from reading *which stratum* an item is in.

**(c) Correctness within never-seen facts is largely luck that the type prior can predict.** A single bit, whether the model's answer differs from the entity type's default value, reaches 0.84–0.89 within FU in the original-world regimes. No learned signal comes close. This confirms the reviewer's point that the correctness label mixes "knows" with "lucky guess". Any self-knowledge evaluation built on correctness labels should report how much of its AUROC a prior-agreement feature like this explains.

**The familiarity shortcut itself is robust.** In 6 of 7 regimes the natural-data probe labels most familiar-and-wrong items "known" (false-"knows" rate 0.62–0.90). The exception is small (0.49). Mixing dissociated items into probe training lowers this rate monotonically in 6 of 7 regimes (base 0.90 → 0.85 → 0.75 at π_d = 0, 0.1, 0.5). It does not in matched_ls (0.62 → 0.63 → 0.65). Mixing never raises the hidden-state probe's within-FU AUROC above about 0.53 (T-D, π_d = 0.5 rows). Adding output features does help (base hidden+output within FU: 0.557 at π_d = 0 vs 0.612 at π_d = 0.5).

**Comp is a partial exception for the hidden probe.** In the compositional world the hidden-state probe is near chance even on natural cells (0.516), while output features reach 0.80. With composed names, hidden features do not linearly separate unseen entities from seen ones. This is one way the base world's random-embedding shortcut does not transfer.

### 6.5 Where Proposition 2's misfit comes from (reviewer M3, m2)

For the hidden-state probe, we compare four quantities: the observed false-"knows" rate; LDA fitted to the whitened hidden states of the *same* training set; Proposition 2 evaluated on the two estimated vectors (a, b) with balanced classes (R0); and Proposition 2 with the actual class prior and conditional co-occurrences (`lda_rates_general`).

| regime (π_d = 0) | observed (logistic) | empirical LDA | Prop. 2, balanced | Prop. 2, class-prior corrected | ‖a‖ / ‖b‖ |
|---|---|---|---|---|---|
| base | 0.899 | 0.870 | 0.647 | 0.581 | 1.04 / 0.42 |
| early | 0.711 | 0.642 | 0.701 | 0.609 | 1.46 / 0.47 |
| small | 0.492 | 0.456 | 0.555 | 0.339 | 0.53 / 0.27 |
| comp | 0.632 | 0.648 | 0.553 | 0.786 | 0.81 / 0.46 |
| matched | 0.826 | 0.734 | 0.539 | 0.315 | 0.53 / 0.39 |
| matched_early | 0.838 | 0.805 | 0.773 | 0.729 | 1.68 / 0.43 |
| matched_ls | 0.625 | 0.601 | 0.545 | 0.358 | 0.63 / 0.46 |

*Table 6. Measured (T-E; `fig_r1_prop2_gap.png`). Pooled over 69 (regime, world, π_d) runs: the balanced closed form has r = 0.36 and MAE 0.15, biased low by 0.12. The class-prior-corrected form has r = 0.45 and MAE 0.18. Empirical LDA has r = 0.97 and MAE 0.03.*

The diagnosis is now specific. The *mechanism* in Proposition 2 explains the probe almost exactly: a linear discriminant trained where F and K co-occur. Empirical LDA reproduces the logistic probe to MAE 0.03 across regimes. What fails is the *two-vector Gaussian summary* of hidden-state geometry. Familiar and unfamiliar items do not differ by a single mean shift under shared isotropic noise. Correcting the class prior, as reviewer m2 suggested, does not close the gap; it widens it. We therefore drop the R0 description of Proposition 2 as a "lower-bound-style diagnostic". It is a qualitative model: it predicts the sign of the shortcut, which holds in 6 of 7 regimes, and the direction of the remedy, which holds in 6 of 7. It is not a quantitative forecaster.

### 6.6 Probe hyperparameters (reviewer M6)

Across λ ∈ {0.01, 1, 100} and both threshold rules, the base-regime natural-data probe keeps natural AUROC at 0.90–0.93, dissociated AUROC at 0.24–0.28, and a false-"knows" rate of 0.78–0.90. The only movement is FK@0.5 dropping to 0.775 at λ = 100. With λ ≤ 1, choosing the threshold by Youden's J on the training set instead of using 0.5 changes FK by at most 0.12 in every regime. At λ = 100 the two rules can differ by up to 0.24 (small, π_d = 0.5), because strong shrinkage pulls probe outputs toward the base rate and makes the fixed 0.5 threshold arbitrary (T-F). In the base regime, the shortcut is not an artefact of the fixed 0.5 threshold or of the regularization strength. Elsewhere, threshold-dependent rates should be read together with the threshold-free AUROCs in Table 5.

## 7 Protocol for Language Models (not run)

We did not run an LLM pilot. The reviewer's suggestion of a 7–8B model needs a GPU, and none is available. A CPU pilot with a sub-1B model also failed: model-weight hosts (huggingface.co and its CDN) are blocked by this environment's network policy. Only the PopQA question file was reachable. The protocol below incorporates the R1 controls.

**Items.** PopQA-style (subject, relation, object) questions [Mallen et al. 2023]. F is subject popularity, or the pre-training document count where available [Kandpal et al. 2023]. Fact exposure is proxied by subject–object co-occurrence counts.

**Controls (new in R1).** (i) Report the −log popularity and −log co-occurrence baselines in every table. (ii) For any ambiguity test, match positives and negatives on popularity histograms. (iii) Report *within-popularity-stratum* AUROC (e.g. correct vs wrong among popular subjects with low co-occurrence), not only pooled "dissociated" AUROC. (iv) Report how much AUROC a prior-agreement feature explains: whether the answer equals the most frequent object for that relation.

**Predictions** (falsifiable; derived from the synthetic results, not established for LLMs). H1: a natural-mix hidden-state probe has within-popular-stratum AUROC much closer to 0.5 than its natural-cell AUROC. *Falsified if* the within-stratum AUROC exceeds 0.75 while the natural-cell AUROC exceeds 0.85. H2: output-probability signals have higher within-stratum AUROC than hidden-state probes trained on the natural mix. H3: on genuinely open-ended questions (e.g. "name a city in X"), label-invariant q-only detectors lose AUROC as α grows, and the best one stays below 1 − α(1−δ̂_T)²/2. Cross-checkpoint disagreement does not lose AUROC. H4: the singleton co-occurrence fraction is *not* a reliable abstention budget unless measured error concentrates on zero-co-occurrence facts.

## 8 Discussion

**What the controls change.** Without exposure matching, R0's ambiguity test credited ensemble MI with separating ignorance from ambiguity, but a familiarity count did as well. With matching, MI still separates the two, and so does a fact-exposure count. Without within-stratum evaluation, R0 found a dramatic inversion of hidden-state probes. With it, the robust finding is weaker but more general: across seven regimes, hidden-state probes trained on natural data carry almost no fact-specific signal once familiarity is held fixed. Output-distribution signals carry some. We consider these two controls the main portable contribution.

**When the predictive distribution is enough.** R0 said q "fails in principle" to separate ignorance from ambiguity. That overreached, and E1b already showed it. The correct statement is conditional. A label-invariant q-only detector is limited exactly by δ_T, the shape difference between ignorance and known ambiguity. Post-hoc recalibration cannot change δ_T (Corollary 1). Calibration does not imply δ_T → 0. Under shape matching, though, even a perfectly calibrated learner is capped at 1 − α/2. Whether LLMs are near the shape-matched regime is an open empirical question. Their outputs on unknown facts are often confident single answers, and in our networks that confidence kept δ_T large. That makes q-only detection *easier* than the worst case, but only for detectors trained to use the shape, not for fixed monotone scores.

**Ensembles.** MI was the most robust signal we tested. In our worlds it detects fact exposure, and under imperfect memorization it was no better than MSP on seen-fact strata. We no longer claim that "only parameter disagreement remains reliable".

**Relation to LLM findings.** The synthetic results are consistent with Cheang et al. [2026] (internal states track recall, not truth) and Ni et al. [2025] (popularity-driven overconfidence). They do not test those claims; Section 7 states how to.

## 9 Limitations

- **No LLM evidence.** Every result is from small MLPs on synthetic worlds. We could not run even a small-model pilot (Section 7). The compositional-name and imperfect-memorization regimes remove two specific degeneracies of R0, but they are not language models: no attention, no text, no pre-training corpus.
- **K ≈ "fact seen" in all our worlds.** Within never-seen facts, correctness is mostly type-prior luck. Within seen facts, it is memorization quality. Our worlds therefore cannot separate "MI tracks knowledge" from "MI tracks exposure", and the fact-count oracle is perfect in every exposure-matched test.
- **Few replicates.** There are 5 worlds for base and 3 for each other regime, so 95% intervals across worlds are wide, and several within-FU differences (n ≈ 230 items per world) are not resolved. We report intervals rather than make claims at that resolution.
- **Proposition 2 is quantitatively poor** (MAE 0.15 across regimes). It earns its place only as a qualitative model. Its mechanism is validated through the empirical-LDA comparison, not through the closed form.
- **Proposition 1 binds only for a constructed learner.** We exhibit tightness for the Bayes reference in a world designed for shape matching. No trained network we built approaches δ_T = 0, and we have no evidence about where LLMs sit.
- **Arbitrary choices.** The familiar/rare thresholds (50/10), the τ = 0.5 epistemic-gap threshold, the matching bins, and c ≥ 20 for "known aleatoric" are fixed choices. We did not sweep them.
- **Theory novelty is low.** Proposition 1 is a corollary of standard ROC/TV facts. Proposition 2 is a relabelled instance of the spurious-feature model.

## 10 Conclusion

Whether a signal shows that a model "knows that it does not know" depends on what the evaluation holds fixed. In controlled worlds, two confounds change the picture. Entity familiarity explains much of what pooled evaluations credit to self-knowledge signals. Ambiguity-vs-ignorance is limited, for q-only detectors, by a shape difference that recalibration cannot change and calibration does not remove. Exposure matching and within-stratum AUROC change our own first-round conclusions, and both cost nothing to adopt. The strongest claim we can support is this: hidden-state correctness probes trained on natural data mostly read familiarity, and within a familiarity stratum they have almost no fact-specific signal, across every regime we built. Testing this on LLMs is the next step.

## Reproducibility

`README.md` lists the commands. R0: `theory_check.py`, `run_all.py`, `e1_qonly.py` and `make_figures.py`. These are unchanged, and their outputs are in `results/` (`summary.txt`, `fig_*.png`). R1: `train_cache.py <variant> <world>` caches ensemble predictions in `cache/`, which is not committed (about 600 MB). `analysis_r1.py <variant> <world>` writes `results/r1/<variant>_w<world>.json`. `aggregate_r1.py` writes `results/r1/summary_r1.txt`, `fig_r1_prop1_strata.png` and `fig_r1_prop2_gap.png`. Total compute is about 48 CPU-minutes, or about 22 min on 4 cores with `OMP_NUM_THREADS=1`. Seeds are fixed. Every number in Sections 5–6 is taken from `results/r1/summary_r1.txt`, except Section 6.1, which uses `results/theory_check.json`.

## References

- Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. *ICML 2024*, PMLR 235:1067–1077. arXiv:2309.14316.
- Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of EMNLP 2023*, 967–976. arXiv:2304.13734.
- Bengs, V., Hüllermeier, E., & Waegeman, W. (2022). Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation. *NeurIPS 2022*. arXiv:2203.06102.
- Cheang, C. S., Chan, H. P., Zhang, W., & Deng, Y. (2026). Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness. *Findings of ACL 2026*, 713–730. (arXiv:2510.09033; v1 titled "Large Language Models Do NOT Really Know What They Don't Know".)
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017), 625–630.
- Ferrando, J., Obeso, O., Rajamanoharan, S., & Nanda, N. (2025). Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models. *ICLR 2025*. arXiv:2411.14257.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *NIPS 2017*, 4878–4887. arXiv:1705.08500.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.
- Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? *EMNLP 2024*, 7765–7784. arXiv:2405.05904.
- Gottesman, D., & Geva, M. (2024). Estimating Knowledge in Large Language Models Without Generating a Single Token. *EMNLP 2024*. arXiv:2406.12673.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*, PMLR 70:1321–1330.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. *Machine Learning*, 110(3), 457–506.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. (2025). Why Language Models Hallucinate. arXiv:2509.04664.
- Kalai, A. T., & Vempala, S. S. (2024). Calibrated Language Models Must Hallucinate. *STOC 2024*. arXiv:2311.14648.
- Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *ICML 2023*, PMLR 202:15696–15707.
- Kapoor, S., Gruver, N., Roberts, M., et al. (2024). Large Language Models Must Be Taught to Know What They Don't Know. *NeurIPS 2024*. arXiv:2406.08391.
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling of knowing. *Psychological Review*, 100(4), 609–639.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NIPS 2017*, 6402–6413.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. *ACL 2023*, 9802–9822.
- Nagarajan, V., Andreassen, A., & Neyshabur, B. (2021). Understanding the Failure Modes of Out-of-Distribution Generalization. *ICLR 2021*. arXiv:2010.15775.
- Ni, S., Bi, K., Guo, J., & Cheng, X. (2025). How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception. arXiv:2505.17537. (A later arXiv version is titled "Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity".)
- Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., & Belinkov, Y. (2025). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. *ICLR 2025*. arXiv:2410.02707.
- Reder, L. M., & Ritter, F. E. (1992). What determines initial feeling of knowing? Familiarity with question terms, not with the answer. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 18, 435–451.
- Rosenfeld, E., Ravikumar, P., & Risteski, A. (2021). The Risks of Invariant Risk Minimization. *ICLR 2021*. arXiv:2010.05761.
- Sagawa, S., Raghunathan, A., Koh, P. W., & Liang, P. (2020). An Investigation of Why Overparameterization Exacerbates Spurious Correlations. *ICML 2020*, PMLR 119:8346–8356. arXiv:2005.04345.
- Wimmer, L., Sale, Y., Hofman, P., Bischl, B., & Hüllermeier, E. (2023). Quantifying aleatoric and epistemic uncertainty in machine learning: Are conditional entropy and mutual information appropriate measures? *UAI 2023*, PMLR 216:2282–2292.
- Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023). Do Large Language Models Know What They Don't Know? *Findings of ACL 2023*, 8653–8665.
