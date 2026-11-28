# Necessity Is Not Reliance: Disentangling Chain-of-Thought Faithfulness Tests with Ground-Truth Transformers

*Target venue: NeurIPS 2027 (Main Track; evaluation-methodology contribution). Revision 2.*

## Abstract

Whether a language model's chain of thought (CoT) reflects the computation behind its answer is usually assessed with behavioural interventions on the CoT text. One removes it (truncation, or "early answering"); the other edits it (corruption, or "adding mistakes"). Both are routinely read as measuring one quantity, "faithfulness".

We separate three properties:

- **Necessity (N):** does the answer degrade when the CoT is absent?
- **Reliance (S):** when the CoT is present, does the answer follow what it says?
- **Description (D):** does the internal computation pass through the intermediate the CoT states?

N and S are logically independent. D is not identifiable from CoT-text interventions. These two observations are elementary, and we state them as such. We also give a quantitative version of the fact that outcome-supervised training removes reliance when training CoTs are sometimes wrong: an ε-optimal predictor has training-support reliance at most pε/((p−1)η). Reliance is left unconstrained when CoTs are only ever absent.

We test these claims on CPU-scale 2-layer transformers that compute a two-step modular sum with a one-token CoT. Mechanism is either fixed by attention masks or measured with path cuts and interchange interventions. Three findings come from a pre-specified sweep of 10 new seeds per regime, with intervals computed over runs:

1. **Truncation and corruption disagree in a direction set by the training regime.** When training CoTs are sometimes absent but never wrong, 7 of 10 models are N−S+ at 8,000 steps (95% CI 0.35–0.93) and 4 of 10 at 4,000 steps. These models follow a corrupted CoT yet answer correctly without one. When training CoTs are sometimes wrong (η=0.2), 0 of 10 final models disagree in either direction.
2. **Truncation is not a measure of either route.** In 15 of 16 N−S+ models, blocking the answer's access to the CoT position drops accuracy to 0.04–0.23, yet replacing the CoT token with the absent marker leaves accuracy high. An embedding-interpolation test shows that the absent marker switches on a different computation.
3. **D dissociates from both behavioural tests.** Sixteen models are behaviourally indistinguishable: they ignore corrupted CoTs and are unaffected by truncation. Their interchange-intervention accuracy for the stated intermediate still ranges from 0.02 to 1.00 (median 0.77). In most of them, the answer reads the internal value of the intermediate at the very position where the CoT token is generated, but not the emitted token. Their CoT therefore describes the computation without being used by it.

We correct two first-round claims:

- "With noisy CoTs, all test disagreements are truncation false positives." The disagreeing models are mechanistically combined-route models, and the effect did not replicate.
- "Reliance is always acquired first." It holds in 45 of 46 runs.

Our cross-test identity barely improves on the trivial predictor "truncation accuracy ≈ ignore rate". We report it as a weak diagnostic. We ran no LLM experiments. The LLM protocol is a proposal with stated predictions, and nothing here establishes that the effects transfer.

## 中文摘要

判断语言模型的“思维链”（CoT）是否忠实反映其真实推理过程，目前主要依赖对 CoT 文本的行为干预：删除它（截断/“提前作答”）或篡改它（插入错误）。这两类测试常被当作同一个“忠实性”量来解读。

本文区分三个性质：

- **必要性 N**：去掉 CoT 后答案是否变差；
- **依赖性 S**：CoT 存在时答案是否服从其内容；
- **描述性 D**：内部计算是否经过 CoT 所陈述的中间量。

N 与 S 逻辑上相互独立；D 无法由只干预 CoT 文本的测试识别。这两点都是初等观察，我们按初等观察陈述。我们还给出一个定量结果：训练 CoT 有时出错（比例 η）时，ε-最优的结果监督预测器在训练支撑集上的依赖性不超过 pε/((p−1)η)；若 CoT 只会缺失而从不出错，依赖性不受约束。

我们在 CPU 规模的 2 层 Transformer 上检验上述论断。任务是两步模加法，CoT 只有一个 token；模型机制或由注意力掩码固定，或由路径切断与交换干预测量。以下三项发现来自预先设定的扫描（每个区间 10 个新种子，置信区间按运行计算）：

1. **截断测试与篡改测试的分歧方向由训练区间决定。** 当训练 CoT 有时缺失但从不出错时，8000 步时 10 个模型中有 7 个属于 N−S+（95% CI 0.35–0.93），4000 步时为 4/10。这类模型服从被篡改的 CoT，但没有 CoT 时仍能答对。当训练 CoT 有时出错时（η=0.2），10 个最终模型中没有一个在任何方向上出现分歧。
2. **截断测试不度量任何一条通路。** 在 16 个 N−S+ 模型中的 15 个里，阻断答案位置读取 CoT 位置会使准确率降到 0.04–0.23，而把 CoT 换成缺失标记后准确率依然很高。嵌入插值实验表明，缺失标记开启了另一套计算。
3. **D 与两种行为测试都不一致。** 16 个行为上无法区分的模型（都忽略被篡改的 CoT，也都不受截断影响）对所述中间量的交换干预准确率在 0.02 到 1.00 之间（中位数 0.77）。其中多数模型的答案读取的是生成 CoT token 的那个位置上的中间量内部表示，而不是被输出的 token 本身。也就是说，它们的 CoT 描述了计算，却没有被计算使用。

我们更正了第一轮的两项结论：

- “CoT 有噪声时，测试分歧全部是截断测试的假阳性”：出现分歧的模型在机制上是组合通路模型，而且该现象没有复现。
- “依赖总是先被习得”：在 46 次运行中 45 次成立。

跨测试恒等式相对平凡预测“截断准确率≈忽略率”只有很小改进，我们把它作为弱诊断报告。我们没有做任何大模型实验；面向大模型的协议只是附带可证伪预测的提案，本文结果不能说明这些效应会迁移到大模型。

## 1 Introduction

Chain-of-thought prompting and training (Wei et al., 2022; Nye et al., 2021) made intermediate reasoning text a standard part of model outputs. CoT monitoring is now proposed as a safety layer (Baker et al., 2025; Korbak et al., 2025). Monitoring is useful only if the visible reasoning is connected to the computation that produces the answer, and a substantial literature shows that often it is not:

- biasing features change answers without being mentioned (Turpin et al., 2023);
- reasoning models verbalise hints they used only a minority of the time (Chen et al., 2025);
- models produce post-hoc rationalisations on unbiased prompts (Arcuschin et al., 2025);
- internal computations can run in parallel with, or diverge from, the stated reasoning (Dutta et al., 2024; Lindsey et al., 2025).

The most widely used quantitative tools are behavioural interventions on the CoT text (Lanham et al., 2023): truncating the CoT, inserting mistakes, paraphrasing, or replacing it with filler.

The tests ask different counterfactual questions. Truncation asks what the model would answer *without* the CoT. Corruption asks what it answers when the CoT says something *different*. A model can compute the answer twice, once internally and once through the CoT, and let the CoT win when both are present. Such a model passes corruption and fails truncation.

The same split exists in rationale evaluation. There, comprehensiveness (prediction change when the rationale is removed) and sufficiency (prediction from the rationale alone) are distinguished (DeYoung et al., 2020). It also appears in Jia et al.'s (2026) information-flow diagnostics for CoT. What remains open is:

- how often the tests disagree in trained networks;
- in which direction they disagree, and under which training conditions;
- how each relates to the mechanism, including the "description" question that people usually mean by "the CoT reflects the real reasoning".

These questions require ground truth.

**Closest prior work.** Wang, Alazali & Zhong (2026) train small transformers from scratch on synthetic arithmetic-expression reasoning with noisy steps. They measure faithfulness by counterfactual intervention on reasoning steps. Under moderate noise they observe a training transition from faithful stepwise reasoning to unfaithful skip-step reasoning, with a mixed phase in between. Our Section 5.4 observation, that reliance is acquired first and lost later and more slowly at lower noise, is consistent with and largely anticipated by their result.

Our work differs in three respects:

1. **Two training manipulations.** We manipulate CoT *absence* (δ) separately from CoT *error* (η), which Prop. 3 predicts have different consequences.
2. **Removal vs editing.** We compare the removal test with the editing test and characterise their disagreement mechanistically.
3. **Description measured separately.** We measure D with interchange interventions and show that losing reliance (S) need not mean losing description (D). Many of our models that ignore the CoT token still route the answer through the internal intermediate that generated the token.

**Contributions.**

1. **Decomposition (Section 3).** We define N, S and D and record two elementary observations: N ⟂ S (Obs. 1) and D is not behaviourally identifiable (Obs. 2). We then give a quantitative proposition (Prop. 3). It bounds the training-support reliance of ε-optimal outcome-supervised predictors by pε/((p−1)η), and N by ε/δ. When η=0, S is unconstrained.
2. **Pre-specified run-level evidence (Section 5.2).** We use 10 new seeds per regime, treat training length as a factor, and report Clopper–Pearson intervals over runs. This replaces first-round checkpoint pooling.
3. **Mechanistic characterisation of the disagreement (Sections 5.3–5.5):**
   - presence gating, with an embedding-interpolation test;
   - a symmetric comparison of both tests against both path-cut references;
   - interchange-intervention measurement of D.
4. **Corrections and a weak diagnostic (Sections 5.6, 5.7).**
   - We correct the first-round "truncation false positive" and "reliance always first" claims.
   - We report that the cross-test identity barely improves on a trivial baseline.
   - We give a proposed LLM protocol with stated predictions (Section 6.2). We did not run it.

The experiments are small. We do not claim that specific numbers transfer to LLMs. We claim that the logical relations and the qualitative regime dependence are things any interpretation of LLM faithfulness scores must reckon with.

## 2 Related Work

**Behavioural CoT faithfulness tests.**

- Lanham et al. (2023) introduced early answering, adding mistakes, paraphrasing and filler substitution.
- Turpin et al. (2023) and Chen et al. (2025) test whether a CoT *mentions* a feature that demonstrably changed the answer.
- Arcuschin et al. (2025) detect unfaithfulness through inconsistent answers to paired questions.
- Tutek et al. (2025, EMNLP) unlearn the knowledge expressed by a reasoning step and measure the effect on the answer.
- Paul et al. (2024) measure and train for causal dependence of answers on reasoning steps.
- Jia et al. (2026) frame faithfulness as information flow along the prompt→CoT→answer path, with sufficiency, completeness and necessity diagnostics. Their necessity coincides with our N. Our contribution relative to theirs is narrower:
  - separating N from reliance on CoT *content* under conflict (S);
  - characterising when and why the two disagree against mechanistic references.

**Rationale evaluation.** DeYoung et al. (2020, ERASER) introduced comprehensiveness and sufficiency for extractive rationales. Comprehensiveness is the analogue of N. Sufficiency is the analogue of our strict internal-route cut (CoT route alone suffices), not of S. S asks about the direction of the answer when the rationale *conflicts* with the input, which neither ERASER metric asks.

**Meta-evaluation.**

- Bentham et al. (2024) show that the early-answering metric is strongly correlated with accuracy and sensitive to answer order.
- Parcalabescu & Frank (2024) argue that such tests measure self-consistency.
- Gur-Arieh et al. (2026) construct tasks whose intermediate computations are revealed and find most faithfulness metrics near chance.

We are complementary: we build models whose mechanism is known.

**Synthetic studies of (un)faithful reasoning.**

- Wang, Alazali & Zhong (2026) is the closest prior work (Section 1).
- Yee et al. (2024) dissociate faithful and unfaithful error recovery in LLMs.
- Dutta et al. (2024) find multiple parallel answer pathways in Llama-2 7B on multi-step reasoning, some drawing on the question context and some on the generated CoT. This is the LLM-scale counterpart of the two-route picture in our Section 3.3.

**Necessity and monitorability.**

- Emmons et al. (2025) argue that when CoT is computationally necessary, models struggle to evade monitors.
- Bürger et al. (2026) show monitoring can fail even when CoT is necessary.
- Korbak et al. (2025) and Baker et al. (2025) discuss how training pressure can erode monitorability.
- Barez et al. (2025; NeurIPS 2026 poster) argue that CoT is not by itself explainability.

**Expressivity and mechanism.**

- CoT increases what bounded-depth transformers can compute (Feng et al., 2023; Merrill & Sabharwal, 2024; Li et al., 2024).
- Filler tokens can carry hidden computation (Pfau et al., 2024).
- We use attention masking, path cuts, probes (Belinkov, 2022) and interchange interventions (Geiger et al., 2021).
- The training dynamics relate to modular-arithmetic circuit formation (Power et al., 2022; Nanda et al., 2023) and to simplicity bias and shortcut learning (Shah et al., 2020; Geirhos et al., 2020).

## 3 Theory

### 3.1 Setup and definitions

An input $x$ determines a correct intermediate $v^*(x)$ and answer $y^*(x)=g(v^*(x),x)$. A CoT $z$ states an intermediate value $v(z)$. A model $M$ defines an answer distribution $\pi_M(y\mid x,\tilde z)$ for $\tilde z\in\mathcal Z\cup\{\varnothing\}$, where $\varnothing$ is an absent CoT. Let $z^*(x)$ be the correct CoT and $z'\sim Q(\cdot\mid x)$ a corrupted CoT with $g(v(z'),x)\ne y^*(x)$.

**Definition 1 (Necessity).** $N(M)=\mathbb E_x[\pi_M(y^*\mid x,z^*)-\pi_M(y^*\mid x,\varnothing)]$, the accuracy drop under truncation.

**Definition 2 (Reliance).** $S(M)=\mathbb E_x\mathbb E_{z'\sim Q}[\pi_M(g(v(z'),x)\mid x,z')]$, the probability that the answer follows a corrupted CoT. We also define two companion quantities:

- $I(M)=\mathbb E[\pi_M(y^*\mid x,z')]$ ("ignore");
- $O=1-S-I$ ("other").

**Definition 3 (Description).** Let the model's computation be a causal graph over internal states.

- $M$ *describes via the CoT* if there is a set of internal sites $H$ such that two conditions hold:
  - $H$ carries information about the input only through the variables $v$ is a function of (here, $(a,b)$ and not $c$);
  - interchange interventions on $H$, which copy $H$ from a source input $x_s$ into a base input $x_b$, act on the answer as setting $v:=v^*(x_s)$ acts on $g$. That is, the output becomes $g(v^*(x_s),x_b)$.
- We measure D by the interchange-intervention accuracy (IIA) of this alignment (Geiger et al., 2021).
- The definition is stated for the answer computation under a given CoT condition (corrupted or absent). It therefore applies to presence-gated models, whose internal route is active only under the absent condition, by evaluating it under that condition.

N and S are what truncation and corruption estimate. D is what people often mean by "the CoT reflects the real reasoning process".

### 3.2 Two elementary observations

**Observation 1 (N and S are logically independent).** For every $(n,s)\in\{0,1\}^2$ there is a model with $N=n$, $S=s$.

*Construction.* Each case defines the answer as a function of whether a CoT is present.

| Case | CoT present ($\tilde z\neq\varnothing$) | CoT absent ($\tilde z=\varnothing$) |
|---|---|---|
| (1,1) | $g(v(\tilde z),x)$ | uniform |
| (0,0) | $y^*(x)$ | $y^*(x)$ |
| (0,1) | $g(v(\tilde z),x)$ | $y^*(x)$ |
| (1,0) | $y^*(x)$ | uniform |

The (1,0) case is presence gating of an internal route. Within Eq. (1) below, fix $\alpha-\beta=\operatorname{logit}(s)$ and let $\beta\to\infty$. Then $S\to s$ and $N\to0$. ∎

**Observation 2 (D is not behaviourally identifiable).** Any test that depends on $M$ only through $\{\pi_M(\cdot\mid x,\tilde z)\}$ gives the same value for some $M_1$ that describes via the CoT and some $M_2$ that does not.

*Construction.* Give $M_1$ an internal route that computes $h=v^*(x)$ and then $g(h,x)$. Give $M_2$ a lookup table indexed by the full $x$; it has no site that carries $(a,b)$-information without $c$ and is sufficient for $y$. Combine each with the same CoT-reading route and the same arbitration rule. ∎

Both observations are short, and we claim no technical depth for them. They fix the logical status of the debate:

- black-box CoT interventions can at most establish N and S;
- D needs white-box evidence.

Section 5.5 shows that the gap is large in practice, not only in principle.

### 3.3 Outcome-supervised training and reliance (quantitative)

**Training distribution.** Training data are $(x,\tilde z,y^*(x))$ with $x$ uniform on a finite training set $X_{tr}$. The CoT input is drawn as follows:

- $\tilde z=z^*(x)$ with probability $1-\eta-\delta$;
- $\tilde z$ uniform over all $p$ values (correct value included) with probability $\eta$;
- $\tilde z=\varnothing$ with probability $\delta$.

**Excess risk.** Let $\varepsilon(q)=\mathbb E[-\log q(y^*(x)\mid x,\tilde z)]$ be the answer cross-entropy of a predictor $q$. Since labels are deterministic, $\inf_q\varepsilon=0$ and $\varepsilon$ is the excess risk.

**Training-support quantities.** $S_{tr}$ and $N_{tr}$ denote $S$ and $N$ with $\mathbb E_x$ taken over $X_{tr}$. The corruption distribution $Q$ is uniform over the $p-1$ wrong values.

**Proposition 3.** For every predictor $q$:

- (a) If $\eta>0$, then $S_{tr}(q)\le \dfrac{p}{p-1}\cdot\dfrac{\varepsilon(q)}{\eta}$.
- (b) If $\delta>0$, then $N_{tr}(q)\le 1-\mathbb E_x\,q(y^*\mid x,\varnothing)\le \varepsilon(q)/\delta$.
- (c) If $\eta=0$, then for every $\varepsilon_0>0$ and every $s\in[0,1]$ there is a $q$ with $\varepsilon(q)\le\varepsilon_0$ and $S_{tr}(q)=s$. That is, the objective places no constraint on reliance. If also $\delta=0$, the same holds for $N_{tr}$.

*Proof.* (a) For any $(x,z)$, the follow probability is at most $1-q(y^*\mid x,z)\le-\log q(y^*\mid x,z)$. For a non-negative $f$, $\mathbb E_{z'\sim Q}f\le\frac{p}{p-1}\mathbb E_{z\sim U_p}f$. Hence

$$S_{tr}\le\tfrac{p}{p-1}\,\mathbb E_{x}\mathbb E_{z\sim U_p}[-\log q(y^*\mid x,z)]\le\tfrac{p}{p-1}\cdot\tfrac{\varepsilon}{\eta},$$

because the noisy component contributes $\eta\,\mathbb E_x\mathbb E_{z\sim U_p}[-\log q]\le\varepsilon$ to the risk.

(b) The same argument applied to the absent component, using $N\le1-\mathbb E\,q(y^*\mid x,\varnothing)$.

(c) Take a softmax predictor with logit $L$ on $y^*$ at all on-support inputs, so that $\varepsilon\to0$ as $L\to\infty$. At the off-support inputs $(x,z')$, set the answer to the mixture $s\,\delta_{g(v(z'),x)}+(1-s)\,\delta_{y^*}$, smoothed by the same softmax. These points carry zero training mass and do not affect $\varepsilon$. ∎

**What changed from Revision 1, and what the proposition does and does not say.**

- **Infimum, not minimiser.** The first version was stated for an exact cross-entropy minimiser, which softmax networks never attain. The bounds above apply to any finite-loss predictor, so they apply to the trained networks.
- **Training-support scope.** The statements are about training-support inputs. Our behavioural metrics are reported on held-out triples. We therefore also evaluated every model on its training triples (`results/r2_trainsupport.json`). Across all 26 round-1 final models and 40 round-2 checkpoints, follow and truncation accuracy on training and held-out triples differ by at most 0.034. The held-out numbers in Section 5 can thus be read as training-support numbers for the purpose of Prop. 3.
- **Empirical check of the bound.** We computed the exact $\varepsilon$ (expectation over all noise values) for all 38 η>0 models. The soft follow rate on training triples is below the bound in every case (`results/r2_bound.json`). The bound is loose, typically by an order of magnitude or more (between 7× and 70×), and it is vacuous when $\varepsilon/\eta$ is large.
- **The bound's role.** It scales as $1/\eta$, which is the reason to expect reliance to erode more slowly at small η (Section 5.4). It does not predict *when* erosion happens.
- **Regime (c).** Part (c) says that only inductive bias and implicit regularisation decide S when η=0. Section 5.2 shows that this decision is seed- and length-dependent, and that N−S+ models are not always stable.

### 3.4 A two-route random-utility model and a cross-test identity

Suppose the answer is drawn from a multinomial logit over $p$ answers with utilities

$$u(y)=\alpha\,\mathbb 1[y=g(v(\tilde z),x)]+\beta\,\mathbb 1[y=y^*(x)]+\varepsilon_y\qquad(1)$$

with i.i.d. Gumbel noise. Here $\alpha$ is the CoT-route strength (absent when $\tilde z=\varnothing$) and $\beta$ the internal-route strength. Then

$$(S,I,O)\propto(e^\alpha,e^\beta,p-2),\qquad P_{\text{trunc}}=\frac{e^\beta}{e^\beta+p-1}.$$

**Proposition 4 (cross-test identity).** Under Eq. (1),

$$\hat P_{\text{trunc}}=\frac{(p-2)I/O}{(p-2)I/O+p-1}.\qquad(2)$$

*Proof.* We have $e^\beta=(p-2)I/O$. Substitute. ∎

A departure of observed truncation accuracy from Eq. (2) indicates non-additive route combination, for example presence gating or an off-support absence token. Eq. (2) needs $O>0$. Section 5.7 shows that, in our data, it adds little over the trivial predictor $\hat P_{\text{trunc}}=I$.

## 4 Experimental Testbed

### 4.1 Task and model

**Task.** Each example is a triple $(a,b,c)\in\mathbb Z_{23}^3$, with intermediate $s=(a+b)\bmod 23$ and answer $y=(s+c)\bmod 23$. The sequence is `[a, b, c, SEP, s, ANS]`. The model predicts $s$ at SEP and $y$ at ANS. The answer is computable without the CoT, so CoT is not computationally necessary here; this is the regime in which faithfulness tests are most ambiguous. All $23^3$ triples are split 80/20 by seed, and reported metrics are on held-out triples unless stated otherwise.

**Model and training.**

- Architecture: pre-LN decoder-only transformer, 2 layers, width 64, 4 heads, MLP 256, learned positions.
- Optimiser: AdamW, lr $10^{-3}$, weight decay 0.5, 200-step warm-up.
- Batch size 256, on CPU.

### 4.2 Ground-truth controls (attention masks, all layers)

- **Reliant:** positions 4 and 5 attend only to {2, 4, 5}. This mask leaks (Section 5.1).
- **Reliant-strict:** as Reliant, and in addition position 2 attends only to itself. Ground truth S=1.
- **Post-hoc:** position 5 may not attend to position 4. Ground truth S=0.
- **Free:** ordinary causal mask.

### 4.3 Training-distribution knobs and runs

In each training sequence, the CoT *input* token is replaced with probability $\eta$ by a uniform random value, and with probability $\delta$ by a `MASK` token. Labels are always correct.

**Round 1 (exploratory; reported in Revision 1).**

- η ∈ {0, 0.05, 0.1, 0.2, 0.4} × δ ∈ {0, 0.3}, two seeds each, at 4,000 steps.
- Six adaptively added longer runs, decided after inspecting results: four at (0, 0.3) for 8,000 steps and two at (0.05, 0) for 12,000 steps.
- The six controls.
- Total: 26 free models and 32 models overall.

**Round 2 (pre-specified; new in this revision).**

- Before launching, we wrote an analysis plan with endpoints, seeds, lengths and the inference rule to `code/sweep_r2_spec.json` (SHA-256 and timestamp in `results/sweep_r2_spec.sha256`). It is not lodged with an external registry.
- Design: seeds 100–109 (disjoint from round 1), with training length treated as a factor (models evaluated at 4,000 and 8,000 steps).
- The original design had four regimes. After about 10 minutes, measured throughput showed it would take about 2 h. Before any checkpoint had been evaluated, we amended it (recorded as `amendment_1` in the spec):
  - regimes (0, 0) and (0.2, 0.3) were dropped;
  - regime (0, 0.3) keeps 10 seeds at 4k and 8k, and seeds 100–103 also continue to 16,000 steps;
  - regime (0.2, 0) keeps 10 seeds at 4k and 8k.
- Inference unit: the run. Checkpoints within a run are never pooled as independent observations.

### 4.4 Measurements

**Behavioural tests** (greedy; soft-probability versions also recorded):

- *Truncation:* the CoT token is replaced by `MASK`.
- *Corruption:* the CoT token is replaced by $s'\ne s$, and the answer is classified as follow / ignore / other.
- *Binary verdicts:*
  - N+ if acc_clean − acc_trunc > 0.5;
  - S+ if follow > 0.5.

**Alternative removal formats** (new). Truncation is also evaluated with the CoT position set to each of:

- the SEP token;
- the mean of the number-token embeddings;
- a zero token embedding (positional embedding only).

For δ=0 models none of these is on-support. No on-support "absent CoT" exists for them, and we do not claim otherwise.

**Mechanistic references** (true CoT present):

- *CoT-cut:* ANS cannot attend to position 4. Mechanistic N+ if acc_clean − acc_CoT-cut > 0.5.
- *Strict internal cut:* positions 4 and 5 cannot attend to {0, 1, 3}, and position 2 cannot attend to {0, 1}. Mechanistic S+ if accuracy > 0.5.

**Presence-gating tests** (new):

- *Embedding interpolation:* the CoT-position token embedding is set to $(1-\lambda)E[s']+\lambda E[\texttt{MASK}]$, and follow/ignore are tracked over λ ∈ [0, 1].
- *Activation patching:* the layer-1 residual at position 4 in a corrupted-CoT run is replaced by its value in the MASK run.

**Description (D) by interchange intervention** (new). Base and source triples have $c_s\ne c_b$ and $s_s\ne s_b$, and the CoT condition is corrupted in both. We copy the source's layer-1 residual at a site set $H$ into the base run. Outcomes are classified as follows:

| Outcome | Output | Meaning |
|---|---|---|
| *s-aligned* | $s_s+c_b$ | $H$ carries the intermediate, separately from $c$ |
| *y-src* | $s_s+c_s$ | $H$ carries the whole answer |
| unchanged | base output | $H$ is not used |

**Site sets.** Under the causal mask, positions {0, 1, 3} (a, b, SEP) carry no information about $c$ at layer 1. The primary D measure is the IIA at $H=\{0,1,3\}$. We also report single sites, and subspace patches at position 3 restricted to the top-22 directions of a linear probe for $s$ or to a random 22-dimensional subspace. We report D only for models whose answer under corruption is ≥90% correct, so that the internal route actually determines the answer.

**Linear probes** are as in Revision 1.

All numbers come from code in `code/` run in this environment. Each section gives its source file in `results/`.

## 5 Results

### 5.1 Controls, and a leak in one of them

**Table 1.** Controls (`results/final_table.md`, `results/r2_stats.json: mech_r1_controls`). IIA is reported only where the answer is computed by the internal route.

| Model | (η,δ) | Clean | Trunc | Follow | Ignore | CoT-cut | Int-cut (strict) | D: IIA at {0,1,3} |
|---|---|---|---|---|---|---|---|---|
| Post-hoc | (0,0) | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.041 | 0.141 |
| Post-hoc | (0.2,0.3) | 0.999 | 0.999 | 0.000 | 0.999 | 0.999 | 0.039 | 0.139 |
| Reliant | (0,0) | 1.000 | 0.039 | 1.000 | 0.000 | 0.043 | 1.000 | — |
| Reliant | (0.2,0.3) | 0.998 | 0.721 | 0.117 | 0.454 | 0.083 | 0.177 | — |
| Reliant-strict | (0,0) | 1.000 | 0.044 | 1.000 | 0.000 | 0.043 | 1.000 | — |
| Reliant-strict | (0.2,0.3) | 1.000 | 0.036 | 1.000 | 0.000 | 0.043 | 1.000 | — |

**Behaviour of the controls.**

- The post-hoc and reliant-strict controls behave as designed.
- The original reliant mask leaked under noisy/absent-CoT training: truncation accuracy 0.721, and 45% of corrupted CoTs ignored. The strict cut drops its accuracy from 0.998 (non-strict cut) to 0.177. This locates the path: $a,b$ reach the answer through the residual stream at $c$'s position. Training pressure against reliance found a path the designer did not intend.

**D in the post-hoc controls (new).** The post-hoc controls answer perfectly without reading the CoT, but their IIA for $s$ at the $c$-free sites is only 0.14. Most of their computation does not pass through a site that carries $s$ separately from $c$. In the terms of Wang et al. (2026), they mostly compute the answer in one "skip" step. Contrast this with the free models in Section 5.5.

### 5.2 Run-level rates from the pre-specified sweep

**Table 2.** Final-model behavioural quadrants: run-level fractions over 10 seeds (4 at 16k), with 95% Clopper–Pearson intervals (`results/r2_stats.json: round2.runlevel`; Fig. 5, `results/fig5_runlevel.png`).

| Regime | Steps | N+S+ | **N−S+** | **N+S−** | N−S− |
|---|---|---|---|---|---|
| η=0, δ=0.3 | 4,000 | 5/10 | **4/10** [0.12, 0.74] | 0/10 [0, 0.31] | 1/10 |
| η=0, δ=0.3 | 8,000 | 2/10 | **7/10** [0.35, 0.93] | 0/10 [0, 0.31] | 1/10 |
| η=0, δ=0.3 | 16,000 (seeds 100–103) | 0/4 | 2/4 [0.07, 0.93] | 0/4 | 2/4 |
| η=0.2, δ=0 | 4,000 | 2/10 | 0/10 [0, 0.31] | **0/10** [0, 0.31] | 8/10 |
| η=0.2, δ=0 | 8,000 | 0/10 | 0/10 | **0/10** [0, 0.31] | 10/10 |

**Regime (c), η=0 and δ=0.3.**

- **Replication.** The first-round claim that redundant-but-reliant models arise when training CoTs are absent but never wrong replicates on new seeds: 4/10 at the originally planned 4,000 steps, and 7/10 at 8,000.
- **Training length is a real factor.** Three of the four 4k→8k changes are N+S+→N−S+: seeds 102, 104 and 105 acquire the absent-CoT route late, as seed 4 did in round 1. The reviewer's point that round-1 seed 4 would have been classified N+S+ at 4,000 steps is correct (`round1.regc_long_quadrant_at_4000`). Our response is to treat length as a factor and report both.
- **Stability.** Of the three seeds tracked to 16,000 steps that were N−S+ at 8,000, two (100, 101) remain N−S+ with follow ≈ 1.0. Seed 102 lost reliance: follow fell from 0.626 to 0.202. Seed 103 was N−S− throughout. N−S+ is therefore **not** shown to be a stable endpoint under weight decay 0.5. Prop. 3(c) only says that the objective does not decide S; implicit regularisation evidently can, in some runs.
- **Within-run behaviour.** Over checkpoints with clean accuracy > 0.9 and step ≤ 8,000, 8 of 10 runs have at least one N−S+ checkpoint. The mean per-run fraction is 0.40 (bootstrap-over-runs 95% CI 0.22–0.60).

**η=0.2, δ=0.**

- No final model is N+S− at 4k or 8k (0/10 each). The first-round observation of two N+S− final models at this setting (round-1 seeds 0, 1; truncation 0.36/0.40) does not replicate as a final-model property.
- N+S− does occur transiently during training. All 10 runs have N+S− checkpoints, with a mean per-run fraction of 0.16 (95% CI 0.09–0.25). At 4,000 steps, four models have partial truncation accuracy (0.58–0.80) that falls short of the 0.5-drop threshold.

**Revision-1 checkpoint pooling, re-analysed.**

- Checkpoint-level statistics pooled autocorrelated checkpoints as independent units. In round 1, 22 of the 38 N−S+ checkpoints came from one run (`round1.regc_NminusSplus_ckpts_by_run`).
- We no longer report checkpoint-pooled p-values. The checkpoint Spearman correlation between truncation drop and follow (ρ=0.81) has a cluster-bootstrap (over runs) 95% interval of [0.62, 0.94] (`round1.ckpt_spearman_drop_follow`).
- The direction asymmetry of round-1 checkpoint disagreements holds at every verdict threshold from 0.3 to 0.7 (`round1.threshold_sweep`):
  - regime (c): N−S+ 22–27%, N+S− 0%;
  - η>0: N−S+ 0%, N+S− 9.1–12.8%.

### 5.3 Presence gating: why truncation is not a measure of either route

**Table 3.** Regime-(c) N−S+ models (round 1, 8,000 steps). CoT-cut accuracy is at chance although truncation accuracy is high.

| Seed | Clean | Trunc | Follow | CoT-cut | Int-cut (strict) | λ at which ignore ≥ 0.5 |
|---|---|---|---|---|---|---|
| 2 | 0.994 | 0.816 | 0.986 | 0.062 | 0.652 | 0.7 |
| 4 | 1.000 | 0.753 | 1.000 | 0.039 | 0.980 | 0.9 |
| 5 | 1.000 | 0.996 | 1.000 | 0.046 | 1.000 | 0.8 |

**What the cuts show.** Both rounds together give 16 N−S+ models or checkpoints (3 round-1 final models, plus 13 round-2 checkpoints at 4k, 8k or 16k). In 15 of them, blocking ANS from the CoT position while the true CoT is present drops accuracy to 0.04–0.23 (`round2.presence_gated` and `r2_models_*.json`). The exception is round-2 seed 102 at 8k, with CoT-cut 0.58. Yet the same models answer correctly when the CoT token is *replaced by* `MASK`. So the absent-CoT computation is not a parallel route that runs whenever the CoT is present.

**Interpolation test.** We moved the CoT-position embedding from a corrupted token toward `MASK` (Fig. 6, `results/fig6_gating.png`). In all 16 N−S+ models, the answer moves from following the corrupted CoT to the correct answer as λ increases. The ignore rate crosses 0.5 at λ = 0.6–0.9. At λ = 0.4 it is ≤ 0.11 in 15 models and 0.23 in seed 102.

**Patching test.** Patching the full layer-1 residual at position 4 from the MASK run into the corrupted run does *not* transfer the switch: ignore ≤ 0.09, except 0.33 for seed 102. So the gate is not stored at the CoT position after layer 1. It is read directly from the CoT-position embedding by layer-1 attention at later positions.

**Interpretation.** We call this mechanism **presence gating**. Truncation in these models measures a computation that is switched on by the absence marker and that does not run when any CoT token is present.

**Symmetric comparison of tests against references.** Table 4 compares each behavioural verdict with each mechanistic verdict, with the same 0.5 threshold everywhere (`mech_r1_final_free`, `round2.mech_step8000`).

**Table 4.** Agreement between behavioural and mechanistic verdicts.

| | Round 1 final (n=26) | Round 2, 8k (n=20) |
|---|---|---|
| Truncation (N) vs mechanistic N (CoT-cut) | 0.81 | 0.65 |
| Corruption (S) vs mechanistic S (strict cut) | 0.92 | 0.95 |
| Spearman: trunc acc vs CoT-cut acc | 0.83 | 0.58 |
| Spearman: follow vs strict-cut acc | 0.96 | 0.89 |
| Spearman: follow vs CoT-cut acc | −0.94 | −0.95 |

**Truncation mismatches.** All truncation mismatches in round 2 except one are N−S+ models: truncation says "not necessary", while cutting the CoT position destroys the answer. A mismatch also occurs in the opposite direction. At 4,000 steps in regime (0.2, 0), round-2 seeds 103 and 107 have truncation accuracy 0.80 and 1.00 but CoT-cut accuracy 0.09 and 0.42.

**Corruption agreement is partly definitional.** We accept the reviewer's point (W5). Follow under corruption and strict-cut accuracy both ask whether the answer is a function of the CoT token and $c$. The values are bimodal: 27% of round-1 final models have follow in (0.1, 0.9). The rank correlation therefore overstates agreement, and we no longer quote its p-value. What the comparison does show is narrower: the corruption test, run with both routes open, gives nearly the same verdict as the white-box cut that silences the internal route. In this testbed, the internal route does not contaminate the corruption verdict. Truncation has no comparably close white-box counterpart, because removing the token and blocking the position are different interventions in a presence-gated model.

### 5.4 Training dynamics, and positioning against Wang et al. (2026)

**Reliance first.** In 45 of 46 free runs (25/26 in round 1, 20/20 in round 2), P(follow) exceeds 0.5 at some logged checkpoint. Revision 1's statement that "training never produced a model that relied on the internal route from the start" was wrong. Round-1 run `add_free_e0.4_d0.3_s0` (η=0.4) never relied on the CoT: its maximum follow is 0.044, at step 250 (`round1.never_reliant_runs`). The corrected statement is: in all runs with η ≤ 0.2, reliance was acquired before any loss of it; at η=0.4 one of four runs never acquired it.

**Erosion.** Among round-1 free runs ending with follow < 0.5, 14 of 15 had first been reliant; the exception is the η=0.4 run above. In round 2, all 10 η=0.2 runs end with follow < 0.5, and all 10 had first been reliant. Erosion is slower at small η. With η=0.05, four of six round-1 models still follow at the end. This pattern is consistent with the 1/η scaling of Prop. 3(a).

**Relation to Wang et al.** The acquired-then-lost pattern is the counterpart, in our one-step setting, of the stepwise→skip-step transition under noise that Wang, Alazali & Zhong (2026) report. We regard that qualitative finding as theirs.

**What their framing does not cover.** Losing *reliance* on the CoT token is not the same as becoming *skip-step*. Section 5.5 shows that most of our models that ignore the CoT token still compute the answer through the internal intermediate.

### 5.5 Description (D) dissociates from both behavioural tests

D-IIA is measured on all models whose answer under a corrupted CoT is ≥ 90% correct: 8 from round 1 and 8 from round 2 at 8k (`round2.D_IIA_ignorers`). All 16 are behaviourally indistinguishable on both tests: follow ≤ 0.008 and truncation accuracy ≥ 0.936.

**Spread.** Their IIA for $s$ at the $c$-free site set {0, 1, 3} ranges from 0.016 to 1.000, with median 0.77. Ten exceed 0.5 and three are below 0.2.

**Controls.** A random 22-dimensional subspace patch at position 3 gives IIA 0.002–0.085. The post-hoc controls give 0.14.

**Where the intermediate lives.** In 15 of the 16 models, the IIA is carried by position 3 alone, the SEP position at which the model *generates* the CoT token. For example:

| Model | IIA at position 3 | IIA at {0,1,3} |
|---|---|---|
| Round-1 η=0.4, seed 1 | 0.998 | 0.998 |
| Round-1 η=0.1, δ=0.3, seed 1 | 0.59 | 0.59 |

A linear-probe subspace (top 22 probe directions) at position 3 recovers part of the effect, with IIA 0.0–0.95. The variable is therefore partly, but not wholly, linearly accessible.

**Interpretation.** These models compute $s$ at SEP to emit the CoT token, and the answer position then reads that internal $s$, not the emitted token. Their CoT is *descriptive* of the computation, because it states the value of the variable the answer is computed from. But it is *not relied on*: changing the emitted token changes nothing. A truncation test and a corruption test both call these models "post-hoc". Yet for 10 of the 16, the stated intermediate is the causal mediator of the answer.

Other models with the same behaviour do not route through $s$ (IIA < 0.2). This is Observation 2 realised in trained networks: identical N and S, very different D.

**Caveats.**

- IIA depends on the site set. A different alignment, such as a distributed or nonlinear one, could raise the low values.
- Our D measure is a lower bound on alignment with $s$.
- D-IIA correlates only weakly with behavioural measures within this group (Spearman with truncation accuracy 0.41, with ignore rate 0.63, n=16).

### 5.6 Correction: the N+S− models are combined-route models, not truncation false positives

Revision 1 called the η>0 disagreements "truncation false positives". We applied the symmetric mechanistic criterion. The two round-1 N+S− final models (η=0.2, δ=0) come out mechanistically N+S− as well:

| Model | CoT-cut | Strict int-cut | Mechanistic verdict |
|---|---|---|---|
| Seed 0 | 0.159 | 0.322 | N+ (CoT position needed), S− (CoT route alone insufficient) |
| Seed 1 | 0.314 | 0.250 | N+ (CoT position needed), S− (CoT route alone insufficient) |

These models combine the CoT position and the internal route. Truncation's verdict, that the CoT position is needed, is mechanistically correct, and we withdraw the "false positive" label.

The off-support explanation remains untested in the sense the reviewer asks for, because no on-support removal exists for δ=0 models. The alternative removal formats give truncation accuracies that vary with format: 0.21–0.57 across MASK, SEP, mean-embedding and zero-embedding formats for these two models (`round2.NplusSminus_models`). The truncation result for δ=0 models is therefore format-dependent. That is the most we can say. Combined with Section 5.2, where N+S− did not replicate as a final-model property, the first-round claim reduces to the following:

- N+S− is a transient training-phase state in all η=0.2 runs.
- Where it persists, it reflects combined-route models.

**Removal-format sensitivity in regime (c).** In 9 of the 10 regime-(c) models at 8k, truncation with SEP, mean-embedding or zero-embedding removal is at chance (0.04–0.15). Truncation with the trained `MASK` marker gives 0.28–1.00 in the same models. The remaining model, seed 103, gives 0.50–0.57 under all three alternative formats (`round2.removal_formats`). The N−S+ verdict therefore depends on the existence of a trained absence token.

### 5.7 The cross-test identity is a weak diagnostic

**Table 5.** Eq. (2) vs the trivial predictor $\hat P_{\text{trunc}}=I$ on graded checkpoints ($O\ge0.02$), evaluated against observed truncation accuracy (`round1.eq2_vs_trivial_*`, `round2.eq2_vs_trivial_*`). Each cell gives Pearson r / MAE.

| Checkpoints | n (runs) | Eq. (2) | Trivial $I$ |
|---|---|---|---|
| Round 1, excl. regime (c) | 189 (20) | 0.971 / 0.054 | 0.968 / 0.079 |
| Round 1, regime (c) | 31 (6) | 0.890 / 0.409 | 0.822 / 0.480 |
| Round 2, η=0.2, δ=0 | 169 (10) | 0.989 / 0.034 | 0.988 / 0.048 |
| Round 2, regime (c) | 152 (10) | 0.573 / 0.573 | 0.509 / 0.644 |

**Comparison with the baseline.** We confirm the reviewer's computation (W6). Almost all of Eq. (2)'s agreement comes from "truncation accuracy ≈ ignore rate". The logit model lowers MAE by about 30% and leaves correlation essentially unchanged.

**The soft-probability residual as a detector.** Used as a model-level detector of behavioural N/S disagreement (|residual| > 0.2), it gives:

| Set | Disagreeing models flagged | Precision | Recall |
|---|---|---|---|
| Round-2 8k models | 7 of 7 (2 false flags) | 0.78 | 1.0 |
| Round-1 final models | 3 of 5 (2 false flags) | 0.60 | 0.60 |

The false flags are regime-(c) N−S− models on which the tests agree. The residual therefore flags the δ>0, η=0 regime more than individual misleading models. We keep Eq. (2) only as a cheap consistency check, and no longer present it as a contribution that identifies the models for which a single score misleads.

## 6 Discussion

### 6.1 What this means for "is the CoT faithful?"

The question bundles three properties, and in this testbed they come apart in every pairing:

1. **N vs S.** Regime-(c) models are N−S+ in most seeds: they follow the CoT whenever it is present, yet answer without it. Section 5.3 shows that truncation probes a computation that only the absence marker switches on.
2. **S vs D.** Many models with S≈0 still compute the answer through the intermediate the CoT states, read internally at the position where the CoT token is generated (Section 5.5). These models are "unfaithful" by both behavioural tests. Yet their CoT is an accurate report of the causal intermediate, because it is produced from the same internal variable the answer uses.
3. **N vs mechanism.** Truncation agrees with a positional path cut for only 65–81% of models. It is sensitive to the format of the removal (Section 5.6).

**Operational reading.**

- Reliance (S), measured by on-support corruption with the three-way outcome recorded, is closest to what a CoT *monitor* needs: the visible text controls the answer.
- Necessity (N) answers a different question, and depends on how "absence" is represented.
- Description (D) is what "the CoT reflects the reasoning" most naturally means. It requires white-box measurement, and it can be high when both behavioural tests say "post-hoc".

**For LLMs these are hypotheses, not results.** Our models predict that outcome-trained models with sometimes-wrong sampled CoTs lose S on internally solvable items (Prop. 3(a); Section 5.4). This is consistent with Lanham et al.'s (2023) observation that larger models depend less on their CoT, and with Wang et al. (2026). The S≠D dissociation suggests that a loss of S need not imply that the CoT stops reflecting the computation. Measuring that requires interpretability tools at scale (Lindsey et al., 2025; Dutta et al., 2024).

### 6.2 A proposed LLM protocol with stated predictions (not run)

We did not run this protocol. The environment for this work had no GPU and no network access to model hubs, so not even a small open-weights model could be tested. P1–P4 are untested predictions.

1. **Items.** Tasks with an enumerable answer space and an editable decisive intermediate, split into two groups:
   - internally solvable items (no-CoT accuracy ≥ 80%);
   - CoT-necessary items.
2. **Tests.**
   - Truncation with at least two removal formats: an empty CoT, and filler of equal length.
   - Corruption of the decisive intermediate with the rest of the CoT kept consistent, recording follow/ignore/other.
   - Forced answering throughout.
3. **Reporting.** Report N, S, (I, O), the rates of N−S+ and N+S− items, and Eq. (2) next to the trivial $I$ baseline. Do not report a combined score. Where white-box access exists, add an interchange-based D measure at the position that generates the decisive intermediate.
4. **Predictions.**
   - **(P1)** On internally solvable items, the N−S+ rate depends strongly on the removal format. It is higher when the removal format resembles absent-reasoning contexts seen in training. This is our analogue of the trained `MASK` token.
   - **(P2)** Eq. (2) residuals are larger on internally solvable items than on CoT-necessary items.
   - **(P3)** Across checkpoints of an outcome-RL run, S on internally solvable items declines.
   - **(P4)** On internally solvable items with S≈0, attribution- or patching-based measures will often still find the stated intermediate mediating the answer (S≠D).
5. **Failure conditions.**
   - If P1 shows no format dependence, our presence-gating account does not transfer.
   - If P4 fails, the S≠D dissociation is specific to our architecture.

### 6.3 Relation to verbalisation-based tests

Hint-verbalisation tests (Turpin et al., 2023; Chen et al., 2025; Gur-Arieh et al., 2026) ask about *completeness*: does the CoT mention every factor that influenced the answer? Our framework does not cover completeness. A model can have S=1 and still hide an influence that shaped the CoT itself.

## 7 Limitations

- **External validity is minimal.** All of the following limit transfer to LLMs:
  - 2-layer, width-64 transformers;
  - one task (two-step modular addition);
  - a one-token CoT and a 23-way answer space;
  - in regime (c), a dedicated trained `MASK` token that marks an absent CoT.

  LLM early answering has no such trained token. Section 5.6 shows that our N−S+ verdicts depend on it. LLM CoTs are long, redundant and self-generated, and error correction in later steps (Yee et al., 2024) cannot appear here. We ran no LLM experiment, not even on a small open-weights model, so every statement about LLMs is a hypothesis.
- **Power.** The round-2 sweep has 10 seeds per regime, and the CIs remain wide; for example, the N−S+ rate at 8k has CI [0.35, 0.93]. Only two regimes were re-run because of compute. For η ∈ {0.05, 0.1, 0.4} and for (0, 0) and (0.2, 0.3), the evidence is round-1 only, with 2–4 seeds. Stability beyond 8k rests on 4 seeds, and one of the three N−S+ models among them lost reliance by 16k. We did not run 20k–50k-step training.
- **Adaptive analysis history.** Round 1 was adaptive. Round 2 followed a written plan, amended once for compute before any evaluation. The plan was not registered with a third party.
- **Thresholds.** Binary verdicts use 0.5. The round-1 direction asymmetry is robust from 0.3 to 0.7. We did not repeat the threshold sweep for round 2.
- **Mechanistic references are imperfect.**
  - Path cuts change activation distributions.
  - The corruption-vs-strict-cut agreement is partly definitional (Section 5.3).
  - D-IIA depends on the chosen site set and is a lower bound.
  - The presence-gating account rests on embedding interpolation and one patching test. We did not localise the gate to specific heads.
- **Theory is simple.** Observations 1–2 are elementary. Prop. 3 is a short bound that is loose by 7–70× empirically. It speaks to training-support inputs; we checked that held-out and training metrics agree to within 0.034.
- **The cross-test identity adds little** over a trivial baseline (Section 5.7).
- **No natural-language component.** We do not test paraphrase, filler or verbalisation.

## 8 Conclusion

Truncation and corruption tests of chain-of-thought faithfulness measure different properties, and neither measures whether the computation passes through the stated intermediate.

On small transformers whose mechanism we can inspect, three kinds of model appear:

- **N−S+ models.** When training CoTs are absent but never wrong, most seeds produce models that follow the CoT whenever it is present but answer correctly when a trained absence marker replaces it. The absence marker switches on a different computation. These models are not reliably stable under longer training.
- **Combined-route models.** When training CoTs are sometimes wrong, models lose reliance after first acquiring it. The first-round N+S− "false positive" models turn out to be combined-route models.
- **Descriptive but unrelied-on CoTs.** Many models that both tests call post-hoc nonetheless compute their answer through the internal value that generated the CoT token.

The narrow answer to "does the CoT faithfully reflect the model's reasoning?" therefore has three parts:

- behavioural tests can answer two of the three questions (N and S), and only separately;
- the third (D) needs white-box evidence;
- in our models, D can be high exactly where both behavioural tests say "unfaithful".

## References

Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., & Conmy, A. (2025). Chain-of-thought reasoning in the wild is not always faithful. arXiv:2503.08679.

Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., & Farhi, D. (2025). Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv:2503.11926.

Barez, F., et al. (2025). Chain-of-thought is not explainability. Preprint, Oxford Martin AI Governance Initiative (July 2025); NeurIPS 2026 poster.

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219.

Bentham, O., Stringham, N., & Marasović, A. (2024). Chain-of-thought unfaithfulness as disguised accuracy. *Transactions on Machine Learning Research*. arXiv:2402.14897.

Bürger, L., et al. (2026). Chain of thought monitoring can fail even when CoT is necessary. arXiv:2610.18105.

Chen, Y., Benton, J., Radhakrishnan, A., Uesato, J., Denison, C., Schulman, J., et al. (2025). Reasoning models don't always say what they think. arXiv:2505.05410.

DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). ERASER: A benchmark to evaluate rationalized NLP models. *ACL 2020*, 4443–4458.

Dutta, S., Singh, J., Chakrabarti, S., & Chakraborty, T. (2024). How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning. arXiv:2402.18312.

Emmons, S., Jenner, E., Elson, D. K., Saurous, R. A., Rajamanoharan, S., Chen, H., Shafkat, I., & Shah, R. (2025). When chain of thought is necessary, language models struggle to evade monitors. arXiv:2507.05246.

Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., & Wang, L. (2023). Towards revealing the mystery behind chain of thought: A theoretical perspective. *NeurIPS 2023*.

Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. *NeurIPS 2021*.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673.

Gur-Arieh, Y., Marasović, A., & Geva, M. (2026). Faithfulness metrics don't measure faithfulness: A meta-evaluation with ground truth. arXiv:2605.25052.

Jia, J., Benton, J., & Easley, E. (2026). Faithfulness as information flow: Evaluating and training faithful chain-of-thought reasoning. arXiv:2605.24286.

Korbak, T., Balesni, M., Barnes, E., Bengio, Y., et al. (2025). Chain of thought monitorability: A new and fragile opportunity for AI safety. arXiv:2507.11473.

Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., et al. (2023). Measuring faithfulness in chain-of-thought reasoning. arXiv:2307.13702.

Li, Z., Liu, H., Zhou, D., & Ma, T. (2024). Chain of thought empowers transformers to solve inherently serial problems. *ICLR 2024*.

Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., et al. (2025). On the biology of a large language model. *Transformer Circuits Thread*.

Merrill, W., & Sabharwal, A. (2024). The expressive power of transformers with chain of thought. *ICLR 2024*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.

Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., et al. (2021). Show your work: Scratchpads for intermediate computation with language models. arXiv:2112.00114.

Parcalabescu, L., & Frank, A. (2024). On measuring faithfulness or self-consistency of natural language explanations. *ACL 2024*.

Paul, D., West, R., Bosselut, A., & Faltings, B. (2024). Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. arXiv:2402.13950.

Pfau, J., Merrill, W., & Bowman, S. R. (2024). Let's think dot by dot: Hidden computation in transformer language models. *COLM 2024*. arXiv:2404.15758.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

Shah, H., Tamuly, K., Raghunathan, A., Jain, P., & Netrapalli, P. (2020). The pitfalls of simplicity bias in neural networks. *NeurIPS 2020*.

Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS 2023*.

Tutek, M., Hashemi Chaleshtori, F., Marasović, A., & Belinkov, Y. (2025). Measuring chain of thought faithfulness by unlearning reasoning steps. *EMNLP 2025*, 9935–9960. arXiv:2502.14829.

Wang, F., Alazali, A., & Zhong, Y. (2026). How does unfaithful reasoning emerge from autoregressive training? A study of synthetic experiments. arXiv:2602.01017.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

Yee, E., Li, A., Tang, C., Jung, Y. H., Paturi, R., & Bergen, L. (2024). Dissociation of faithful and unfaithful reasoning in LLMs. arXiv:2405.15092.
