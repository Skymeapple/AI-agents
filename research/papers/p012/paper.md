# Familiar Is Not Known: Two Identifiability Failures in Machine Self-Knowledge, with Closed-Form Predictions and a Controlled Test

**Target venue:** ICLR 2027 (main track; theory + controlled empirical study)

---

## Abstract

A model "knows that it does not know" when it can compute a signal that separates the queries it would answer wrongly from the queries it would answer correctly. Two families of such signals are in common use: the model's own predictive distribution (maximum probability, entropy, verbalized confidence), and probes trained on hidden states to predict correctness. We formalize two structural reasons these signals can fail, and test both in a controlled setting. **(i) Ambiguity confound.** Suppose a fraction α of the queries the model knows are genuinely ambiguous (aleatoric), and their predictive distributions lie within total-variation distance δ of those produced by ignorance. Then every detector that is a function of the predictive distribution alone has AUROC at most 1 − α(1−δ)²/2, and the bound is tight (Proposition 1). **(ii) Familiarity shortcut.** Consider a Gaussian model of hidden states that encode both entity familiarity F and fact-specific knowledge K. A correctness probe trained on data where F and K agree with probability π mislabels familiar-but-unknown queries as "known" at a rate we derive in closed form. This rate increases in π and tends to a value above ½ exactly when familiarity is the more salient direction (Proposition 2). In-distribution accuracy stays high, so ordinary validation cannot detect the failure. Monte Carlo simulation matches both closed forms to within 0.004. We then train ensembles of small embedding MLPs on CPU in a synthetic knowledge world where familiarity, fact exposure, latent type structure and aleatoric ambiguity are controlled independently: 3 worlds with 5 models each. The measured results support the qualitative predictions, but not all the quantitative ones. A hidden-state correctness probe trained on natural data reaches AUROC 0.930 on natural items and 0.245 on dissociated items, labelling 89% of familiar-but-wrong answers as "known". Proposition 2, evaluated on the estimated hidden-state geometry, predicts the direction of this failure but underestimates its size (0.64 predicted vs 0.89 observed). Maximum-probability confidence falls from AUROC 0.816 to 0.169 as α goes from 0 to 1, because the networks are more confident when ignorant than on known coin-flip facts. A trained detector on the predictive distribution recovers most of this loss (0.954 at α = 1), so in our networks Proposition 1 is not binding (estimated δ ≈ 0.83). Only ensemble mutual information is reliable on every subset tested (AUROC 0.95–0.97). The Good–Turing singleton fraction (0.038) matches the mass of never-seen facts (0.038), which carry 99.9% of the errors. We give a pre-registered protocol for testing the same predictions on LLMs, and argue that self-knowledge evaluations should report results per familiarity × knowledge cell. We ran no LLM experiments.

## 中文摘要

如果模型能计算出一个信号，把它会答错的问题和会答对的问题区分开，我们就说模型"知道自己不知道"。常用的信号有两类：一是模型自身的预测分布（最大概率、熵、口头置信度），二是在隐藏状态上训练、用来预测回答是否正确的探针。本文形式化了这两类信号失效的两个结构性原因，并在可控环境中加以检验。**（一）歧义混淆：** 设模型"知道"的问题中有比例 α 本身就有歧义（偶然不确定性），且它们的预测分布与"无知"时的预测分布之间的全变差距离不超过 δ，那么任何只依赖预测分布的检测器，AUROC 都不超过 1 − α(1−δ)²/2（命题1，界是紧的）。**（二）熟悉度捷径：** 在一个隐藏状态同时编码实体熟悉度 F 与具体事实知识 K 的高斯模型中，若训练数据里 F 与 K 一致的概率为 π，则正确性探针会把"熟悉但不知道"的问题误判为"知道"，我们给出这一误判率的闭式解。它随 π 递增；当熟悉度方向更显著时，其极限超过 1/2（命题2）。而探针在训练分布上的准确率始终很高，因此常规验证发现不了这一失效。蒙特卡洛模拟与两个闭式解的误差在 0.004 以内。随后，我们在一个合成知识世界中用 CPU 训练了小型嵌入 MLP 集成（3 个世界 × 每个世界 5 个模型）；该世界中熟悉度、事实曝光、潜在类型结构与偶然歧义彼此独立可控。实测结果支持定性预测，但并非所有定量预测都成立。在"自然"数据上训练的隐藏状态探针，在自然样本上 AUROC 为 0.930，在分离样本上仅为 0.245，并把 89% 的"熟悉但答错"判为"知道"。用估计出的隐藏状态几何代入命题2，能预测失效的方向，但低估了幅度（预测 0.64，实测 0.89）。当 α 从 0 增至 1 时，最大概率置信度的 AUROC 从 0.816 降至 0.169，因为网络在无知时反而比面对已知的"抛硬币"式事实时更自信。在预测分布上训练的检测器能挽回大部分损失（α = 1 时为 0.954），因此在我们的网络中命题1并不起约束作用（估计 δ ≈ 0.83）。在所有检验过的子集上都可靠的信号只有集成互信息（AUROC 0.95–0.97）。Good–Turing 单例比例（0.038）与从未见过的事实所占比例（0.038）吻合，而 99.9% 的错误都落在这些事实上。我们给出在大语言模型上检验同一组预测的预注册方案，并主张自我知识评测应按"熟悉度 × 知识"单元分别报告结果。本文没有做任何大语言模型实验。

---

## 1 Introduction

Asking how to make a model reliably know that it does not know is, operationally, asking for a *signal* s(x) that the model can compute and that ranks queries by the probability that the model's answer is wrong. Such a signal is useful for abstention, retrieval triggering, or hedged answers. Two families of signals dominate current practice. The first reads the model's predictive distribution q(·|x): softmax response for selective classification [Geifman & El-Yaniv 2017], token- or sequence-level probability and calibration [Guo et al. 2017; Kadavath et al. 2022], and semantic entropy over sampled answers [Farquhar et al. 2024]. The second trains a probe on hidden activations to predict whether the model will be right. Examples are P(IK) heads [Kadavath et al. 2022], truthfulness and error probes [Azaria & Mitchell 2023; Orgad et al. 2025], fine-tuned uncertainty heads [Kapoor et al. 2024], and pre-generation knowledge estimators that read the subject representation [Gottesman & Geva 2024]. Mechanistic work finds "known entity" directions that causally gate refusal [Ferrando et al. 2025].

These methods work well in-distribution, but two empirical warnings recur. Error probes do not transfer across datasets [Orgad et al. 2025]. Hidden-state representations reportedly fail to separate hallucinations produced by spurious *associations* from correct recall [Cheang et al. 2025]. Cognitive psychology offers a precise, decades-old diagnosis of the same pattern in people: the initial feeling of knowing is driven by *familiarity with the question's terms*, not by access to the answer [Reder & Ritter 1992], and it is computed from whatever information is accessible, right or wrong [Koriat 1993]. Our thesis is that the machine case has the same structure. We make this structure precise enough to give quantitative, falsifiable predictions.

We isolate two **identifiability failures**:

1. **The ambiguity confound.** A model that knows a question is a coin flip outputs a spread distribution. So does a model that simply does not know a deterministic answer. Any detector that sees only q cannot tell these apart. This has long been argued qualitatively in the uncertainty-decomposition literature [Hüllermeier & Waegeman 2021; Wimmer et al. 2023]. We give the exact ceiling it implies for abstention AUROC.

2. **The familiarity shortcut.** Many queries about popular entities are answered correctly, and queries about never-seen entities are answered wrongly. Benchmarks for "does the model know" are therefore dominated by items where familiarity F and knowledge K agree. A probe trained on such data can reach high validation accuracy by reading F. It then fails exactly on the items that matter: facts never seen about familiar entities (confident hallucination), and facts seen once about obscure entities (needless abstention). This is a shortcut in the sense of Geirhos et al. [2020]. We derive its magnitude in closed form.

**Contributions.**

- **C1 (theory, ambiguity).** Proposition 1: a tight upper bound, 1 − α(1−δ)²/2, on the AUROC of any detector that is a function of the predictive distribution, where α is the aleatoric share of known items and δ is the TV distance between the predictive-distribution laws of aleatoric-known and unknown items. The proof uses a Youden-index bound on ROC curves.
- **C2 (theory, familiarity).** Proposition 2: for a whitened Gaussian hidden-state model with familiarity salience a and knowledge salience b, a closed form for the false-"knows" rate of the population LDA correctness probe on familiar-unknown items, as a function of the training co-occurrence π = P(F=K). The rate is monotone increasing in π. Its π→1 limit is Φ((a²−b²)/(2σ√(a²+b²))), which exceeds ½ iff a > b, while in-distribution accuracy remains high. A general (non-orthogonal) version is used for estimation.
- **C3 (controlled test).** A synthetic knowledge world in which familiarity, fact exposure, type-based inferability and aleatoric ambiguity are set independently. We train ensembles of small MLPs and measure four things. (a) Confidence and accuracy by familiarity cell. (b) The AUROC of q-only and ensemble signals as α varies, together with an estimate of δ and the best trained q-only detector, which tests whether Proposition 1 binds. (c) Correctness probes trained with varying shares of dissociated items, compared with Proposition 2 evaluated on the *estimated* hidden-state geometry. (d) A Good–Turing estimate of unknown-fact mass. We report where the predictions fail: Proposition 2 underestimates the shortcut by about 0.23, and Proposition 1 is not binding in our networks.
- **C4 (protocol).** A pre-registered protocol that turns the propositions into tests on LLMs. It uses entity popularity, fact-level exposure proxies, and a 2×2 familiarity × knowledge evaluation design that reports dissociated cells separately.

The contribution is diagnostic. We do not propose a new state-of-the-art detector. We show *why* two families of detectors must fail on specific, predictable subsets of queries, and we say which evaluation design exposes the failure.

## 2 Related Work

**Self-knowledge in language models.** Kadavath et al. [2022] showed that large LMs are reasonably calibrated on formatted multiple-choice questions, and that a P(IK) head partially generalizes across tasks but is poorly calibrated on new ones. Yin et al. [2023] built SelfAware, a benchmark of unanswerable questions, and reported a substantial gap to human self-knowledge. Kapoor et al. [2024] found that prompting alone does not yield good uncertainty, whereas fine-tuning on roughly a thousand graded examples does. Azaria & Mitchell [2023] trained truthfulness classifiers on hidden states. Orgad et al. [2025] found that error detectors fail to generalize across datasets and called truthfulness encoding "multifaceted". Gottesman & Geva [2024] (KEEN) estimate entity-level knowledge from subject representations before any generation. Ferrando et al. [2025] used sparse autoencoders to find entity-recognition directions that can steer models to refuse questions about known entities, or to hallucinate about unknown ones. Cheang et al. [2025] argue that hidden states do not separate associative hallucinations from factual associations. Our Proposition 2 gives a mechanism-agnostic account of why familiarity-type features dominate correctness probes, and predicts when they will.

**Why models hallucinate.** Kalai & Vempala [2024] proved that a calibrated LM must hallucinate at a rate close to the Good–Turing missing mass, the fraction of facts seen exactly once, for arbitrary facts. Kalai et al. [2025] argue that binary-graded evaluations reward guessing over abstaining. Long-tail studies show that accuracy depends strongly on entity popularity and pre-training document counts [Mallen et al. 2023; Kandpal et al. 2023]. Fine-tuning on new knowledge increases hallucination on previously known facts [Gekhman et al. 2024]. Synthetic-biography studies show that knowledge can be stored without being extractable [Allen-Zhu & Li 2024]. Our synthetic world borrows this controlled-corpus methodology. Our E4 uses the Good–Turing quantity as an abstention budget rather than a hallucination lower bound.

**Aleatoric vs epistemic uncertainty.** Hüllermeier & Waegeman [2021] survey the distinction. Wimmer et al. [2023] criticize the entropy and mutual-information decomposition. Bengs et al. [2022] show that second-order predictors trained by loss minimisation do not faithfully represent epistemic uncertainty. Deep ensembles [Lakshminarayanan et al. 2017] are the standard practical source of a non-q-only signal. Semantic entropy [Farquhar et al. 2024] targets "confabulations" by sampling, but it is still a function of the model's output distribution, so Proposition 1 applies to it. Proposition 1 turns the qualitative concern into a ceiling that holds for any q-only detector.

**Selective prediction.** Chow [1970] derived the optimal reject rule when posteriors are known. Geifman & El-Yaniv [2017] gave risk-controlled selective classifiers based on softmax response. Guo et al. [2017] documented miscalibration in modern networks. Our results say *which* items a softmax-response selector will misrank, and why.

**Feeling of knowing and shortcut learning.** Reder & Ritter [1992] showed experimentally that the initial feeling of knowing tracks exposure to the problem's *parts*, not knowledge of the answer. Koriat's [1993] accessibility model holds that feeling-of-knowing judgments are computed from accessible information, correct or not. Geirhos et al. [2020] describe shortcut learning as a general failure mode in which decision rules exploit features that are predictive in-distribution but do not transfer. Proposition 2 is, in effect, a quantitative cue-familiarity model for machine metacognition.

## 3 Setting and Definitions

A query x = (e, r) asks for the value of relation r for entity e. The world has a conditional p*(·|x) over a finite value set V. For a *deterministic* fact p* is a point mass. For an *aleatoric* fact p* is spread out (e.g., "which of three venues will host…"). A trained model outputs q(·|x).

**Knowing.** We use two operational targets. (a) *Correctness*: K(x) = 1[argmax q = argmax p*]. (b) *Epistemic gap*: G(x) = 1[TV(q, p*) > τ], with τ = 0.5. G separates "the model is wrong about the world" from "the world is random". For an aleatoric fact that the model has learned, G = 0 even though q is high-entropy.

**Familiarity.** F(x) is a function of the entity's exposure n_e (its total training count). It is distinct from the fact's exposure c(e,r). We call the four combinations of (familiar vs unfamiliar entity) × (fact known vs not) the *F×K cells*. Items with F = K are *natural*. Items with F ≠ K are *dissociated*: familiar-unknown (FU) and unfamiliar-known (UK).

**Detectors.** A *q-only* detector is any measurable s(q(·|x)). This covers maximum softmax probability, entropy, temperature-scaled versions of these, and verbalized confidence that is a deterministic function of q. A *probe* is a function of a hidden representation z(x) fitted to predict K on a labelled set.

## 4 Theory

### 4.1 Proposition 1 (ambiguity ceiling)

Let "positives" be queries the model does not know (G = 1), with q distributed as P₀. Let "negatives" be known queries (G = 0), a mixture: with probability 1−α a deterministic known item (q ~ P_det), and with probability α an aleatoric known item (q ~ P_alea). Let δ = TV(P₀, P_alea).

**Proposition 1.** For every q-only detector s,

  AUROC(s) ≤ (1−α) + α·[1 − (1−δ)²/2] = 1 − α(1−δ)²/2.

For any α and δ there exist P₀, P_alea and P_det for which equality holds.

*Proof.* AUROC is P(s(X⁺) > s(X⁻)) + ½P(tie), with X⁺ ~ P₀ and X⁻ from the negative mixture. This is linear in the negative law, so AUROC(s) = (1−α)·A(s; P₀, P_det) + α·A(s; P₀, P_alea), where A(s; P, Q) is the AUROC of s for positives P against negatives Q. The first term is at most 1. For the second, every ROC curve (FPR f, TPR t) generated by thresholding any statistic of q satisfies t − f ≤ sup_B [P₀(B) − P_alea(B)] = δ. (The difference of the two probabilities of any event is at most TV.) The ROC curve therefore lies below min(1, f + δ), and

  A ≤ ∫₀¹ min(1, f+δ) df = δ(1−δ) + (1−δ)²/2 + δ = 1 − (1−δ)²/2.

For tightness, take P_det disjoint from P₀. Take P₀ and P_alea on a common space with densities chosen so that the likelihood-ratio ROC is exactly min(1, f+δ): a δ-mass of P₀ disjoint from P_alea, and the remainder identical. Ranking by likelihood ratio attains both bounds at once. ∎

**Remarks.** (1) With δ = 0, which is what happens when a calibrated model outputs the same spread distribution for "I know it's 50/30/20" and "I have no idea", the ceiling is 1 − α/2. Because the result holds for every s, no recalibration, temperature, or entropy variant of q can escape it. (2) The bound does *not* apply to signals that are not functions of q, such as disagreement across ensemble members or samples from different parameter draws (mutual information), or exposure counts. Those signals can in principle reach AUROC 1. (3) Semantic entropy clusters samples from one model's q, so it is q-only in our sense, and it is subject to the bound once the aleatoric spread is semantic rather than lexical.

### 4.2 Proposition 2 (familiarity shortcut)

Model the (whitened) hidden representation of a query as

  z = a·F·u + b·K·v + ε,  ε ~ N(0, σ² I),  u ⊥ v unit vectors,

where F, K ∈ {0,1}. Here a is the salience of familiarity and b the salience of knowledge. A probe is trained on a distribution with P(K=1) = ½ and P(F = K) = π. We analyse the population LDA rule. It is Bayes-optimal among linear rules for the training distribution when class-conditional covariances are equal, and logistic regression approximates it in our regime.

**Proposition 2.** Let κ(π) = (2π−1)σ² / (σ² + a²π(1−π)). The probe's probability of labelling a familiar-unknown item (F=1, K=0) as "knows" is

  FK(π) = Φ( (a²κ(π) − b²) / (2σ √(a²κ(π)² + b²)) ).

(i) FK is strictly increasing in π on [½, 1). (ii) FK(½) = Φ(−b/(2σ)), the error of a probe that reads knowledge only. (iii) lim_{π→1} FK(π) = Φ((a² − b²)/(2σ√(a²+b²))), which exceeds ½ iff a > b. (iv) Over the same limit, in-distribution accuracy is at least π·Φ(√(a²+b²)/(2σ)). Validation on the training distribution therefore cannot detect the failure. By symmetry, the missed-knowledge rate on unfamiliar-known items is 1 − Φ((b² − a²κ)/(2σ√(a²κ²+b²))).

*Proof.* Within class K, F is Bernoulli with parameter π or 1−π, so the within-class covariance is Σ_W = σ²I + a²π(1−π)uuᵀ. The class means are μ₁ = aπu + bv and μ₀ = a(1−π)u. Hence μ₁ − μ₀ = a(2π−1)u + bv, and since u is an eigenvector of Σ_W, w = Σ_W⁻¹(μ₁−μ₀) = [a(2π−1)/(σ²+a²π(1−π))]u + (b/σ²)v = (1/σ²)(aκu + bv). The midpoint threshold is c = w·(μ₁+μ₀)/2 = w·(au + bv)/2. For z from the cell (F=1, K=0), w·z − c ~ N((a²κ − b²)/(2σ²), ‖w‖²σ²) with ‖w‖ = √(a²κ² + b²)/σ². Standardizing gives the formula. For (i): with g(κ) = (a²κ − b²)/√(a²κ²+b²), we have g′(κ) = a²b²(1+κ)/(a²κ²+b²)^{3/2} > 0 for κ > −1, and κ(π) is increasing on [½,1) because its numerator increases and its denominator decreases there. (ii) and (iii) follow by substituting κ(½) = 0 and κ(1) = 1. For (iv), the natural cells (weight π) are correctly classified with probability Φ(√(a²+b²)/(2σ)) at κ = 1. ∎

For estimation we use the general version, in which a and b are arbitrary vectors in whitened space (not necessarily orthogonal) and w, c and FK are computed numerically (`code/theory.py: lda_rates`).

**Predictions.** P1: probes trained on natural benchmarks (π near 1) inherit FK > ½ whenever familiarity is more linearly salient than fact-specific knowledge. P2: mixing dissociated items into probe training (lowering π) lowers FK monotonically, but cannot push it below Φ(−b/2σ). If knowledge is weakly encoded (small b), no linear probe will do well on dissociated cells. P3: q-only confidence rises with familiarity even for never-seen facts, whenever the network maps familiar-entity representations to sharp outputs.

## 5 Controlled Experimental Setup

All numbers below were produced by code in `code/` and saved in `results/` (seeds fixed; total runtime about 10 minutes on 4 CPU cores).

**World** (`world.py`). 4,000 entities, 8 relations (6 deterministic, 2 aleatoric), 32 values, 8 latent entity types. For each (type, relation) there is a default value. A fact takes the default with probability ρ = 0.6, and a uniformly random value otherwise. Aleatoric facts put probabilities (0.5, 0.3, 0.2) on three values; the mode follows the same default rule. Each entity–relation pair is documented with probability 0.75. Undocumented pairs never appear in training. 10% of entities never appear at all. Entity popularity follows Zipf(1.0) over the remaining entities. The training corpus has 150,000 (entity, relation, value) samples: an entity is drawn by popularity, then a documented relation uniformly, and the value is drawn from p*. Aleatoric facts are therefore observed with their true spread.

**Model** (`mlp.py`). An embedding MLP: [entity embedding (32) ; relation embedding (32)] → ReLU(256) → softmax over 32 values. It is trained with Adam (lr 2e-3, batch 512, 30 epochs) with decoupled weight decay 1e-4 on dense weights only. Embeddings of unseen entities keep their random initialization, a stand-in for an unfamiliar name. Each world gets a 5-member ensemble with different seeds. Member 0 is "the model"; the ensemble supplies mutual information (MI).

**Cells.** Familiar: n_e ≥ 50. Rare: 1 ≤ n_e ≤ 10. "Seen fact": c(e,r) ≥ 1. The five deterministic cells are familiar-seen, familiar-unseen-fact, rare-seen, rare-unseen-fact, and unseen-entity.

**E1 (ambiguity).** Positives: deterministic items with G = 1. Negatives: a pool of known items, a fraction α of which are aleatoric facts with c ≥ 20 and G = 0, the rest deterministic known items. We report AUROC averaged over 20 resamples per α and over 3 worlds.

**E1b (is Prop. 1 binding?).** We retrain member 0 with the same seed (`e1_qonly.py`; its unseen-entity accuracy matches Table 1) and estimate δ in two steps. First, we fit a cross-fitted logistic classifier on q-features (top-5 sorted log-probabilities and entropy) to separate unknown items from known-aleatoric items. Second, we take the Youden index of that classifier on held-out entities, which is a lower bound on TV. We also train the same q-only detector on the full E1 task at each α.

**E2 (cells).** Accuracy, mean MSP and ensemble MI per familiarity × fact-exposure cell, and as a function of entity training count.

**E3 (probes).** Labels: K = correctness. Entities are split in half; probes are trained on one half and evaluated on the other. The *natural* pool is familiar-seen ∪ unseen-entity. The *dissociated* pool is familiar-unseen-fact ∪ rare-seen. A probe is trained on 1,500 items, a share π_d of them drawn from the dissociated pool, π_d ∈ {0, .02, .05, .1, .25, .5}, with 5 resamples each. Features: hidden layer (256-d); hidden plus output features (top-5 sorted log-probabilities and entropy); output features alone. The probe is L2-regularized logistic regression. We record the effective co-occurrence π_eff = P(F=K) of each training set. We also estimate the familiarity and knowledge vectors (a, b) from cell means of whitened hidden states (pooled within-cell covariance with 30% shrinkage), and evaluate the general Proposition 2 at each π_eff.

**E4 (Good–Turing).** We draw 200,000 fresh queries from the training-generating distribution and compare the singleton fraction N₁/N with the actual mass of never-seen facts, and break down error by fact exposure.

## 6 Results

Every number in this section comes from `results/summary.txt`, `results/main_results.json`, `results/e1_qonly_w*.json` or `results/theory_check.json`, all produced by the code in `code/`. Values are means ± standard deviations over the 3 synthetic worlds unless stated otherwise. Three worlds is a small number of replicates; the ± values describe variation between worlds, not confidence intervals.

### 6.1 The theory matches simulation

For Proposition 2, the sample-based LDA probe (200,000 training points, 20-dimensional noise) agrees with the closed form at all 20 settings (a, b) ∈ {(2,1), (1,2), (1.5,1.5), (3,1)} × π ∈ {0.5, 0.7, 0.9, 0.97, 0.995}. The maximum absolute error is 0.0034. With a = 2, b = 1, the false-"knows" rate rises from 0.309 at π = 0.5 to 0.745 at π = 0.995. With a = 1, b = 2 it stays below 0.25. Both match predictions (i) and (iii). For Proposition 1, a one-dimensional construction with δ = 0 attains the ceiling 1 − α/2 to within 0.0015 at α ∈ {0, .25, .5, .75, 1} (Figure `fig_theory.png`). These checks confirm the algebra and nothing more. Whether the assumptions describe trained networks is tested next.

### 6.2 E2: what confidence looks like on never-seen facts

| cell (deterministic relations) | n per world | accuracy | mean MSP | MSP − acc | ensemble MI |
|---|---|---|---|---|---|
| familiar, fact seen | 1,629 | 1.000 ± 0.000 | 0.997 ± 0.000 | −0.003 | 0.004 |
| familiar, fact never seen (FU) | 467 | 0.295 ± 0.010 | 0.866 ± 0.007 | 0.570 | 0.827 |
| rare, fact seen (UK) | 5,622 | 1.000 ± 0.000 | 0.991 ± 0.001 | −0.009 | 0.011 |
| rare, fact never seen | 5,718 | 0.272 ± 0.006 | 0.853 ± 0.003 | 0.581 | 0.815 |
| entity never seen | 2,412 | 0.060 ± 0.003 | 0.757 ± 0.004 | 0.698 | 1.335 |

*Table 1. Measured in the synthetic world (code: `run_all.py`). n is the mean count per world.*

Three findings stand out. First, every documented fact the model saw, even once, is answered correctly (accuracy 1.000). The networks memorize their training set. Second, facts the model never saw receive a mean MSP of 0.85–0.87 while being right only 27–30% of the time. The overconfidence gap is about 0.57 whatever the entity's familiarity. Third, prediction P3 (confidence rises with familiarity for never-seen facts) is only **weakly** borne out. Across entity-count bins from [1,3) to [100,300), MSP on never-seen facts rises from 0.825 to 0.873, but accuracy also rises, from 0.190 to 0.286, so the gap does not grow (Figure `fig_familiarity.png`). In this world, the danger of familiarity is not that it inflates confidence relative to rare entities. The danger is that high confidence on never-seen facts is *already* the default, and familiarity does nothing to lower it. Notably, a predictor that output the default value of the entity's latent type would be right about 62% of the time on these cells (column `prior_acc` in `summary.txt`). The networks do not exploit type structure well for unseen facts: accuracy is 0.27–0.30.

Ensemble mutual information separates seen from never-seen facts by two orders of magnitude (≈0.01 vs ≈0.8) at every familiarity level. Unlike MSP, it does not shift with familiarity in a way that matters (Figure `fig_familiarity.png`, right).

### 6.3 E1: the ambiguity confound in a trained network

| α (aleatoric share of known items) | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| MSP (1 − max q) | 0.816 | 0.685 | 0.556 | 0.491 | 0.428 | 0.298 | 0.169 |
| entropy of q | 0.814 | 0.687 | 0.563 | 0.500 | 0.439 | 0.314 | 0.189 |
| entropy of ensemble mean | 0.947 | 0.917 | 0.888 | 0.870 | 0.856 | 0.823 | 0.791 |
| ensemble MI | 0.948 | 0.953 | 0.959 | 0.959 | 0.963 | 0.966 | 0.970 |
| Prop. 1 ceiling at δ = 0 | 1.000 | 0.900 | 0.800 | 0.750 | 0.700 | 0.600 | 0.500 |

*Table 2. AUROC for detecting the epistemic gap G = 1, averaged over 20 resamples and 3 worlds (measured; `run_all.py`). Standard deviations over worlds at α = 1 are at most 0.013.*

Single-model confidence loses discrimination linearly in α and falls **below chance** at α ≥ 0.5. This is worse than Proposition 1's δ = 0 ceiling, and the reason is instructive. Known aleatoric items have mean MSP 0.555 ± 0.011, while unknown deterministic items have mean MSP 0.806 ± 0.003. The model is *more* confident when it is ignorant than when it correctly represents a coin flip. Ranking by MSP therefore inverts the correct order on the aleatoric component. Ensemble MI is unaffected by α (0.948 → 0.970), consistent with Remark 2. Entropy of the ensemble mean, a total-uncertainty measure that mixes both sources, degrades in between (0.947 → 0.791).

Is Proposition 1 *binding* here? It bounds the best q-only detector, not MSP. E1b (`e1_qonly.py`) answers this by training a cross-fitted logistic detector on q-features (top-5 sorted log-probabilities and entropy).
The detector is trained per α on one half of the entities and tested on the other, with 10 resamples per α. Results (measured; `results/e1_qonly_summary.txt`):

| α | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| best trained q-only detector | 0.815 | 0.732 | 0.731 | 0.746 | 0.773 | 0.824 | 0.954 |
| MSP (same splits) | 0.819 | 0.683 | 0.555 | 0.493 | 0.428 | 0.302 | 0.173 |
| Prop. 1 ceiling at δ̂ | 1.000 | 0.997 | 0.994 | 0.993 | 0.991 | 0.988 | 0.985 |

*Table 2b. Means over 3 worlds; standard deviations ≤ 0.015.*

The cross-fitted Youden lower bound on TV(P₀, P_alea) is δ̂ = 0.829 ± 0.024. A q-only classifier separates unknown items from known-aleatoric items with AUROC 0.960. In this world, therefore, **Proposition 1 is not binding**. The model's outputs on aleatoric facts have a recognizable shape (mass near 0.5/0.3/0.2) that differs from its outputs under ignorance (a single overconfident peak). A *trained* q-only detector exploits this shape, recovering 0.954 at α = 1. It dips to about 0.73 at intermediate α, because one linear rule must reject two differently shaped negative populations at once. Two lessons follow. (a) The failure of MSP and entropy under ambiguity is not, in this world, an information-theoretic necessity. It comes from the *monotone* scoring rule, which assumes that ignorance looks spread out. (b) Proposition 1 becomes binding exactly when the model's output under ignorance resembles its output under genuine ambiguity (δ → 0). This is the regime a well-calibrated model approaches, since calibration pushes outputs on unknown facts toward spread distributions. Our toy networks are far from calibrated, so they do not test that regime. Section 7 states a test for it.

### 6.4 E3: correctness probes learn familiarity

The natural pool has co-occurrence π = P(F = K) = 0.964 ± 0.002. Estimated in whitened hidden space, familiarity is more salient than knowledge: ‖a‖ = 1.03, 1.05, 1.00 and ‖b‖ = 0.41, 0.39, 0.43 in the three worlds, with cosines between the two directions of 0.20, 0.24 and 0.03. Proposition 2 therefore predicts FK > ½ for a probe trained on natural data.

| probe features | π_d | AUROC natural | AUROC dissociated | false-"knows" (familiar & wrong) | missed-"knows" (rare & right) | Prop. 2 predicted FK |
|---|---|---|---|---|---|---|
| hidden | 0 | 0.930 | **0.245** | **0.889 ± 0.052** | 0.419 | 0.644 |
| hidden | 0.1 | 0.917 | 0.263 | 0.845 | 0.456 | 0.630 |
| hidden | 0.5 | 0.811 | 0.353 | 0.760 | 0.427 | 0.488 |
| hidden + output | 0 | 0.941 | 0.386 | 0.815 | 0.304 | 0.644 |
| hidden + output | 0.5 | 0.895 | 0.651 | 0.523 | 0.204 | 0.488 |
| output only | 0 | 0.913 | 0.786 | 0.298 | 0.197 | — |
| output only | 0.5 | 0.906 | 0.806 | 0.413 | 0.052 | — |
| *baseline:* MSP | — | 0.909 | 0.809 | — | — | — |
| *baseline:* ensemble MI | — | 0.960 | **0.970** | — | — | — |
| *baseline:* −log entity count | — | 0.955 | 0.014 | — | — | — |

*Table 3. Probe results on held-out entities (measured; `run_all.py`, 5 resamples × 3 worlds). AUROC is for detecting wrong answers.*

The hidden-state probe trained on natural data is the clearest instance of the familiarity shortcut. Its natural-cell AUROC is 0.930, comparable to strong published probes, yet its AUROC on dissociated cells is 0.245, well **below chance**. It labels 89% of familiar-and-wrong items as "knows". Pure familiarity (−log entity count) behaves the same way in caricature: 0.955 on natural cells, 0.014 on dissociated cells. In-distribution validation would rate both detectors as excellent.

The quantitative predictions of Proposition 2 are only partly right. **Direction:** correct. ‖a‖ > ‖b‖ and the observed FK exceeds ½ at every π. **Ordering:** the observed and predicted FK correlate at r = 0.57 across the 90 (world, π_d, resample) runs. **Magnitude:** wrong. The Gaussian-LDA model under-predicts FK by 0.23 on average (0.889 observed vs 0.644 predicted at π_d = 0). It also predicts a much steeper benefit from mixing in dissociated data than occurs (predicted 0.488 at π_d = 0.5, observed 0.760). We attribute the miss to two violated assumptions. Hidden states are ReLU features, far from Gaussian with shared covariance. And the "knowledge" signal b is estimated from cell means, while what distinguishes a correct from an incorrect never-seen fact is mostly the luck of a type-default guess, which is not linearly encoded. The proposition should be read as a lower-bound-style diagnostic for this regime, not as a calibrated forecaster.

Two remedies behave differently. Adding output features to the hidden state and mixing in dissociated training items do help: dissociated AUROC rises from 0.386 to 0.651 for hidden+output, consistent with P2. But neither reaches what the output distribution alone gives (0.786–0.806), and none reaches ensemble MI (0.970). In this world the richest representation is the worst place to look for self-knowledge, because familiarity is its dominant linear direction.

### 6.5 E4: the Good–Turing budget

On 200,000 fresh queries from the training distribution, the singleton fraction N₁/N is 0.038 ± 0.000 and the realized mass of never-seen facts is 0.038 ± 0.001. Of all errors on deterministic relations, 99.9% ± 0.1% fall on never-seen facts. The error rate is 0.741 on never-seen facts and 0.000 on facts seen once or more. The overall deterministic error rate is 0.028 ± 0.002. Here the Good–Turing quantity is an accurate, label-free estimate of how often a well-designed system should abstain. Note the contrast with Kalai & Vempala's [2024] lower bound: that bound concerns *calibrated* models and singleton facts. Our networks are *not* calibrated (Table 1) and memorize singletons perfectly, so their errors concentrate entirely on the missing mass instead.

## 7 Pre-registered Protocol for Language Models

We ran no LLM experiments. The following protocol states, before any such experiment is run, what our account predicts and what would falsify it.

**Items.** Build (subject, relation, object) questions from a knowledge base with popularity metadata (e.g., Wikidata subjects with page-view counts, as in PopQA [Mallen et al. 2023]). Familiarity F is subject popularity, or better, the subject's document count in the pre-training corpus where it is available [Kandpal et al. 2023]. Fact exposure is proxied by co-occurrence counts of subject and object in that corpus.

**Cells.** Report four cells separately. (i) Popular subject, high co-occurrence. (ii) Popular subject, low or zero co-occurrence: the familiar-unknown cell. Obscure relations of famous people are the canonical case. (iii) Obscure subject, high co-occurrence. (iv) Fabricated subjects, as in fictitious-entity benchmarks. Correctness is labelled per model by exact or judged match.

**Detectors.** MSP / sequence probability; verbalized confidence; semantic entropy [Farquhar et al. 2024]; a hidden-state correctness probe trained on a natural mix (cells i and iv); the same probe trained with a stated share of cells ii and iii; an entity-recognition direction [Ferrando et al. 2025] or KEEN [Gottesman & Geva 2024]; and, where affordable, disagreement across independently fine-tuned checkpoints.

**Predictions.**
- H1 (familiarity shortcut). Probes trained on the natural mix will have AUROC in cell (ii) significantly lower than on the natural cells, and will label most wrong answers in cell (ii) as known. This holds most strongly at layers where an entity-recognition direction is most linearly decodable. *Falsified if* cell-(ii) AUROC is within 0.05 of natural-cell AUROC for probes with natural-cell AUROC above 0.85.
- H2 (monotonicity). Increasing the share of dissociated training items monotonically reduces the cell-(ii) false-"knows" rate. Following Section 6.4, we expect the reduction to be smaller than a Gaussian-LDA fit predicts.
- H3 (ambiguity). On a set of questions whose answer distribution is genuinely spread (e.g., "name a city in X"), all q-only detectors, including semantic entropy, lose AUROC as the share of such questions in the "known" class grows. The best q-only AUROC stays below 1 − α(1−δ̂)²/2 with δ̂ estimated as in E1b. Cross-checkpoint disagreement does not lose AUROC.
- H4 (Good–Turing). The singleton fraction of (subject, object) co-occurrences in the corpus predicts the fraction of factual questions answered wrongly to within a factor of two on a uniformly sampled question set.

**Reporting.** Always report per-cell AUROC and per-cell false-"knows" / missed-"knows" rates, plus π = P(F = K) for every training and evaluation set. An aggregate AUROC alone is not interpretable: it can be dominated by the natural cells, as Table 3 shows.

## 8 Discussion

**What "knowing that you don't know" requires.** The two propositions point to two requirements. A signal must be able to separate "I lack the answer" from "the answer is random": that is the ambiguity confound. And it must reflect the fact-specific trace rather than the entity's familiarity: that is the familiarity shortcut. A model's own predictive distribution fails the first requirement in principle. Probes on representations fail the second whenever familiarity is more linearly salient than knowledge, and nothing in standard training makes it otherwise. The one signal that met both requirements in our setting is parameter disagreement (ensemble MI). This agrees with the uncertainty-decomposition literature's argument that epistemic uncertainty is a property of the learner's *state of information*, not of the predictive distribution [Hüllermeier & Waegeman 2021]. It sits in tension with the observation that second-order loss minimisation does not deliver faithful epistemic uncertainty [Bengs et al. 2022]: ensembles get disagreement from independent training runs, not from a loss that rewards it. For LLMs, full ensembles are expensive, so a practical question raised by our results is which cheap proxies (checkpoint diversity, LoRA ensembles, sampled-prompt disagreement) preserve the property that matters: agreement on seen facts, disagreement on unseen ones.

**Evaluation design is the immediate lever.** The results suggest a change to evaluation that costs nothing. Most self-knowledge benchmarks mix answerable questions about well-known entities with unanswerable questions about fabricated or obscure ones [Yin et al. 2023]. In our terms, π is near 1. An abstention method can score well on such a benchmark by reading familiarity. Reporting the dissociated cells separately, and stating π, would expose this. Our −log-entity-count baseline (AUROC 0.955 natural, 0.014 dissociated) is a useful sanity check to include: any self-knowledge method that cannot beat it on the dissociated cells has not shown that it measures knowledge.

**Connection to human metacognition.** The familiarity shortcut is the machine analogue of the cue-familiarity account of the feeling of knowing [Reder & Ritter 1992]. Our finding that output-distribution features generalize better than hidden-state features (Table 3) loosely echoes the accessibility account [Koriat 1993]: a signal computed from what is actually retrieved, here the answer distribution, carries some fact-specific information that the cue representation does not. We do not claim more than an analogy.

**Relation to concurrent findings on LLMs.** Cheang et al. [2025] report that LLM hidden states cannot separate hallucinations associated with a known subject from correct recall, but can separate unassociated ones. Orgad et al. [2025] report that error probes fail to transfer across datasets. Both observations are what Proposition 2 predicts if subject familiarity dominates the probed representation and the training sets differ in π. Our synthetic results are consistent with that reading but do not establish it for LLMs. Section 7 states how to test it.

## 9 Limitations

- **Toy models and toy world.** The networks are two-layer embedding MLPs on a synthetic relational world, not transformers on text. An unseen entity is represented by an untrained random embedding, which is a crude stand-in for an unfamiliar name: real models compose representations of new names from known sub-tokens. Our familiar/rare thresholds (≥50 and ≤10 training occurrences) are arbitrary.
- **Few replicates.** Three worlds, five ensemble members, one architecture, one training length (30 epochs, which gives perfect memorization of seen facts). We did not vary capacity, regularization or training time, and memorization regime probably matters a great deal (Section 6.5).
- **Proposition 2 is quantitatively inaccurate in our setting.** It predicts direction and ordering but under-predicts the false-"knows" rate by about 0.23 and over-predicts the benefit of mixing in dissociated data. Its Gaussian, shared-covariance, population-LDA assumptions are violated by ReLU features and finite logistic-regression probes.
- **Proposition 1 is not binding in our experiments.** In our world δ̂ ≈ 0.83, so the ceiling is about 0.99. The observed failure of MSP is a sign inversion caused by overconfidence on unknowns, and a trained q-only detector largely avoids it (Table 2b). We have not exhibited a trained network in the δ ≈ 0 regime where the ceiling bites. We argue only that calibration pushes toward that regime.
- **Correctness as the "knows" label.** Labelling by correctness counts lucky guesses on never-seen facts as "known". About 30% of familiar-unseen items are correct by chance through type defaults. This blurs the knowledge variable that Proposition 2 assumes is binary.
- **No LLM experiments.** All claims about LLMs are hypotheses stated in Section 7. The analogies to Cheang et al. [2025], Orgad et al. [2025] and Ferrando et al. [2025] are interpretive.
- **Ensembles are expensive.** The one detector that worked uniformly, ensemble MI, costs five training runs here and is impractical for frontier models without cheaper approximations, which we did not test.

## 10 Conclusion

"Knowing that you don't know" is a detection problem, and two structural confounds make it harder than in-distribution metrics suggest. A detector that sees only the predictive distribution cannot tell ignorance from genuine ambiguity, and we give the AUROC ceiling this implies. A correctness probe trained where familiarity and knowledge co-occur learns familiarity, and we give its false-"knows" rate in closed form for a Gaussian model. In a controlled synthetic world, both failures appear. The quantitative fit of the shortcut model is loose, but the qualitative predictions hold: probes that look excellent in-distribution invert on dissociated cells, single-model confidence inverts under ambiguity, and only parameter disagreement remains reliable. The most actionable consequence is for evaluation: report self-knowledge results per familiarity × knowledge cell, and state the co-occurrence π of every training and evaluation set.

## Reproducibility

`README.md` lists the commands. `theory_check.py` takes about 10 s. `run_all.py` took about 6 minutes per world on a 4-core CPU in our run. `e1_qonly.py` retrains member 0 of each ensemble (same seeds) for E1b; in our session the three worlds took about 35 minutes wall-clock in parallel on a loaded machine. `make_figures.py` regenerates `summary.txt` and all figures. Seeds are fixed.

## References

- Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:1067–1077. arXiv:2309.14316.
- Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. *Findings of the Association for Computational Linguistics: EMNLP 2023*, 967–976. arXiv:2304.13734.
- Bengs, V., Hüllermeier, E., & Waegeman, W. (2022). Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2203.06102.
- Cheang, C. S., Chan, H. P., Zhang, W., & Deng, Y. (2025). Large Language Models Do NOT Really Know What They Don't Know. arXiv:2510.09033. (Later version in *Findings of ACL 2026*.)
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017), 625–630.
- Ferrando, J., Obeso, O., Rajamanoharan, S., & Nanda, N. (2025). Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2411.14257.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 4878–4887. arXiv:1705.08500.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.
- Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., & Herzig, J. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? *Proceedings of EMNLP 2024*, 7765–7784. arXiv:2405.05904.
- Gottesman, D., & Geva, M. (2024). Estimating Knowledge in Large Language Models Without Generating a Single Token. *Proceedings of EMNLP 2024*. arXiv:2406.12673.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1321–1330.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. *Machine Learning*, 110(3), 457–506.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. (2025). Why Language Models Hallucinate. arXiv:2509.04664.
- Kalai, A. T., & Vempala, S. S. (2024). Calibrated Language Models Must Hallucinate. *Proceedings of the 56th Annual ACM Symposium on Theory of Computing (STOC 2024)*. arXiv:2311.14648.
- Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:15696–15707.
- Kapoor, S., Gruver, N., Roberts, M., et al. (2024). Large Language Models Must Be Taught to Know What They Don't Know. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2406.08391.
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling of knowing. *Psychological Review*, 100(4), 609–639.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 6402–6413.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. *Proceedings of ACL 2023 (Long Papers)*, 9802–9822.
- Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., & Belinkov, Y. (2025). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2410.02707.
- Reder, L. M., & Ritter, F. E. (1992). What determines initial feeling of knowing? Familiarity with question terms, not with the answer. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 18, 435–451.
- Wimmer, L., Sale, Y., Hofman, P., Bischl, B., & Hüllermeier, E. (2023). Quantifying aleatoric and epistemic uncertainty in machine learning: Are conditional entropy and mutual information appropriate measures? *Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 216:2282–2292.
- Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023). Do Large Language Models Know What They Don't Know? *Findings of the Association for Computational Linguistics: ACL 2023*, 8653–8665.

