# Familiar Is Not Known: Two Identifiability Failures in Machine Self-Knowledge, with Closed-Form Predictions and a Controlled Test

**Target venue:** ICLR 2027 (main track; theory + controlled empirical study)

---

## Abstract

A model "knows that it does not know" when some signal it can compute separates queries it would answer wrongly from queries it would answer correctly. Two such signals are in common use. The first is the model's own predictive distribution (maximum probability, entropy, verbalized confidence). The second is a probe trained on hidden states to predict correctness. We identify two ways in which these signals become unreliable, each for a structural reason, and make both precise. **(i) Ambiguity confound.** If a fraction α of the queries the model does know are genuinely ambiguous (aleatoric), and their predictive distributions are within total-variation distance δ of those produced by ignorance, then every detector that is a function of the predictive distribution alone has AUROC at most 1 − α(1−δ)²/2 (Proposition 1; the bound is tight). **(ii) Familiarity shortcut.** In a Gaussian model of hidden states that encode both familiarity F (how often the entity was seen) and knowledge K (whether the specific fact is stored), a correctness probe trained where F and K agree with probability π has a false-"knows" rate on familiar-but-unknown queries that we give in closed form. That rate increases monotonically in π and tends to a value above ½ exactly when familiarity is more salient than knowledge (Proposition 2). In-distribution accuracy stays high throughout, so ordinary validation cannot reveal the shortcut. Monte Carlo simulation matches both closed forms to within 0.004. We then test the predictions in a controlled synthetic knowledge world: Zipf-distributed entity popularity, partial fact coverage, latent entity types and aleatoric relations, with 3 worlds × 5-member ensembles of embedding MLPs trained on CPU. The results confirm the predicted pattern; Section 6 gives the measured values. Familiarity raises confidence without raising accuracy. Single-model confidence loses discrimination roughly linearly in α, while ensemble mutual information is unaffected by α. Correctness probes trained on "natural" data, where familiarity and correctness co-occur, fail on dissociated cells, and the size of the failure is predicted to first order by Proposition 2 applied to the estimated hidden-state geometry. A Good–Turing singleton estimate predicts the mass of never-seen facts, which bounds the abstention a well-calibrated system should exhibit. We close with a pre-registered protocol for testing the same predictions on LLMs. We ran no LLM experiments.

## 中文摘要

如果模型能计算出某种信号，把它会答错的问题和会答对的问题区分开，我们就说模型"知道自己不知道"。常用的信号有两类：一是模型自身的预测分布（最大概率、熵、口头置信度），二是在隐藏状态上训练、用来预测回答是否正确的探针。本文指出这两类信号会因结构性原因而失效，并分别给出形式化刻画。**（一）歧义混淆：** 设模型确实"知道"的问题中有比例 α 属于本身就有歧义的问题（偶然不确定性），且它们的预测分布与"无知"时的预测分布之间的全变差距离不超过 δ，那么任何只依赖预测分布的检测器，AUROC 都不超过 1 − α(1−δ)²/2（命题1，界是紧的）。**（二）熟悉度捷径：** 设隐藏状态同时编码"熟悉度" F（实体出现过多少次）和"知识" K（具体事实是否被记住），在高斯模型下，若训练数据中 F 与 K 一致的概率为 π，则正确性探针在"熟悉但不知道"的问题上的误判率（误判为"知道"）有闭式解。该误判率随 π 单调递增；当熟悉度比知识更显著时，其极限超过 1/2（命题2）。而探针在训练分布上的准确率始终很高，因此常规验证发现不了这一捷径。蒙特卡洛模拟与两个闭式解的误差在 0.004 以内。我们随后在一个可控的合成知识世界中检验这些预测：实体流行度服从 Zipf 分布，事实只被部分覆盖，实体带有潜在类型，并含有偶然性关系；在 CPU 上训练了 3 个世界 × 5 成员集成的嵌入 MLP。结果与预测一致（具体数值见第6节）：熟悉度提高了置信度，却没有提高准确率；单模型置信度的区分能力大致随 α 线性下降，而集成互信息不受 α 影响；在熟悉度与正确性同时出现的"自然"数据上训练的探针，在两者分离的单元上失效，失效幅度可以由命题2结合估计出的隐藏状态几何作一阶预测；Good–Turing 单例估计能预测从未见过的事实所占的比例，从而给出校准良好的系统应当拒答的比例。最后，我们给出在大语言模型上检验同一组预测的预注册方案。本文没有做任何大语言模型实验。

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
- **C3 (controlled test).** A synthetic knowledge world in which familiarity, fact exposure, type-based inferability and aleatoric ambiguity are set independently. We train ensembles of small MLPs and measure: (a) confidence and accuracy as functions of familiarity for seen and unseen facts; (b) AUROC of q-only vs ensemble signals as α varies; (c) correctness probes trained with varying shares of dissociated items, compared with Proposition 2 evaluated on the *estimated* hidden-state geometry; (d) a Good–Turing estimate of unknown-fact mass.
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

**E3 (probes).** Labels: K = correctness. Entities are split in half; probes are trained on one half and evaluated on the other. The *natural* pool is familiar-seen ∪ unseen-entity. The *dissociated* pool is familiar-unseen-fact ∪ rare-seen. A probe is trained on 1,500 items, a share π_d of them drawn from the dissociated pool, π_d ∈ {0, .02, .05, .1, .25, .5}, with 5 resamples each. Features: hidden layer (256-d); hidden plus output features (top-5 sorted log-probabilities and entropy); output features alone. The probe is L2-regularized logistic regression. We record the effective co-occurrence π_eff = P(F=K) of each training set. We also estimate the familiarity and knowledge vectors (a, b) from cell means of whitened hidden states (pooled within-cell covariance with 30% shrinkage), and evaluate the general Proposition 2 at each π_eff.

**E4 (Good–Turing).** We draw 200,000 fresh queries from the training-generating distribution and compare the singleton fraction N₁/N with the actual mass of never-seen facts, and break down error by fact exposure.

<!--RESULTS-->
