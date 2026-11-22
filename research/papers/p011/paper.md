# Necessity Is Not Reliance: Disentangling Chain-of-Thought Faithfulness Tests with Ground-Truth Transformers

*Target venue: NeurIPS 2027 (Main Track; Datasets & Benchmarks–style evaluation-methodology contribution)*

## Abstract

Whether a language model's chain of thought (CoT) reflects the computation that actually produces its answer is usually assessed with behavioural interventions on the CoT text: removing it (truncation / "early answering") or editing it (corruption / "adding mistakes"). These tests are routinely read as measuring one quantity, "faithfulness". We argue, prove, and demonstrate on transformers whose mechanism is known that they measure two logically independent properties: **necessity** (does the answer degrade when the CoT is absent?) and **reliance** (when the CoT is present, does the answer follow what it says?). A third property, **description** (does the internal computation pass through the intermediate the CoT states?), is invisible to any test that only intervenes on CoT text and reads outputs. We (i) formalise the three properties and prove that necessity and reliance are independent, that description is not behaviourally identifiable, and that an outcome-supervised predictor trained to optimality on a task it can compute without the CoT becomes CoT-invariant whenever erroneous CoTs appear in training; (ii) derive a two-route random-utility model under which the three-way outcome of a corruption test (follow / ignore / other) predicts the truncation result, giving a falsifiable cross-test consistency check; and (iii) build a CPU-scale testbed of 2-layer transformers on a two-step modular task in which ground-truth reliance is fixed by attention masks or measured by mechanistic path cuts. RESULTS_PLACEHOLDER The practical consequence is that a single "faithfulness score" from truncation or corruption is not interpretable; we give a pre-registered protocol that reports necessity, reliance and a cross-test consistency statistic separately, and that we believe should replace single scores in LLM evaluations.

## 中文摘要

判断语言模型的“思维链”（CoT）是否忠实反映其真实推理过程，目前主要依赖对 CoT 文本的行为干预：删除它（截断/“提前作答”）或篡改它（插入错误）。这些测试通常被当作同一个量——“忠实性”——来解读。本文论证、证明并在机制已知的 Transformer 上验证：它们实际测量的是两个逻辑上相互独立的性质——**必要性**（去掉 CoT 后答案是否变差）与**依赖性**（CoT 存在时答案是否服从其内容）；而第三个性质**描述性**（内部计算是否经过 CoT 所陈述的中间量）对任何“只干预 CoT 文本、只观察输出”的测试都不可识别。我们 (i) 形式化这三种性质，证明必要性与依赖性相互独立、描述性在行为层面不可识别，并证明：若任务无需 CoT 即可计算，且训练数据中出现错误的 CoT，则以结果监督训练至最优的预测器必然对 CoT 不变；(ii) 提出双通路随机效用模型，使篡改测试的三类结果（服从/忽略/其他）可以预测截断测试的结果，从而给出可证伪的跨测试一致性检验；(iii) 构建 CPU 规模的实验平台：在两步模运算任务上训练 2 层 Transformer，其真实依赖关系由注意力掩码固定，或由机制性路径切断测量。RESULTS_ZH_PLACEHOLDER 结论是：单一的截断或篡改“忠实性分数”不可解释。我们给出一个预注册协议，分别报告必要性、依赖性及跨测试一致性统计量，建议在大模型评测中取代单一分数。

## 1 Introduction

Chain-of-thought prompting and training (Wei et al., 2022; Nye et al., 2021) made intermediate reasoning text a standard part of model outputs, and CoT monitoring is now proposed as a safety layer (Baker et al., 2025; Korbak et al., 2025). Monitoring is useful only if the visible reasoning is connected to the computation that produces the answer. A substantial literature shows that it often is not: biasing features change answers without being mentioned (Turpin et al., 2023), reasoning models verbalise hints they used only a minority of the time (Chen et al., 2025), models produce post-hoc rationalisations on unbiased prompts (Arcuschin et al., 2025), and interpretability work finds internal computations that diverge from the stated reasoning (Lindsey et al., 2025). The most widely used *quantitative* tools are behavioural interventions on the CoT text (Lanham et al., 2023): truncating the CoT ("early answering"), inserting mistakes, paraphrasing, or replacing it with filler. A model whose answer does not change under these interventions is described as "unfaithful", or as having "post-hoc" reasoning.

Two problems with this practice motivate this paper. First, the tests are validated almost exclusively against each other or against intuitions; when ground truth has been constructed, most metrics perform near chance (Gur-Arieh et al., 2026), and the early-answering metric is highly correlated with accuracy (Bentham et al., 2024). Parcalabescu & Frank (2024) argue that such tests measure output-level self-consistency rather than faithfulness to internal reasoning. Second, the tests ask different counterfactual questions. Truncation asks "what would the model answer *without* the CoT?"; corruption asks "what does the model answer when the CoT says something *different*?". A model that computes the answer twice—once internally, once through the CoT—and lets the CoT win when both are present will pass the corruption test and fail the truncation test. Whether such redundancy arises in trained networks, how often the two tests disagree, and what each one actually measures, are open empirical questions that cannot be settled on LLMs without access to ground truth.

We address them in a setting where ground truth is available by construction. Our sub-question is: **what do truncation and corruption tests measure, when do they disagree, and what training conditions determine the outcome?** We answer it with a formal analysis and a controlled experiment on small transformers.

**Contributions.**

1. **A three-property decomposition with proofs (Section 3).** We separate *necessity* (N), *reliance* (S) and *description* (D). We prove that N and S are logically independent (Prop. 1), that D is not identifiable by any test that intervenes on CoT text and observes outputs (Prop. 2), and that an outcome-supervised predictor trained to optimality on a task it can solve without the CoT is CoT-invariant on the whole input space whenever erroneous CoTs have positive probability in training, while with CoT-*absent* (but never erroneous) training data reliance is left undetermined by the objective (Prop. 3).
2. **A cross-test consistency check (Section 3.3).** Under a two-route random-utility (multinomial-logit) model, the three-way outcome distribution of a corruption test identifies the strengths of the CoT route and the internal route and therefore predicts the truncation result. Agreement or disagreement between predicted and observed truncation accuracy is a falsifiable diagnostic for whether a single "faithfulness" number is meaningful.
3. **A ground-truth testbed and its results (Sections 4–5).** We train 2-layer transformers on a two-step modular task, with CoT reliance fixed by attention masks (positive/negative controls) or left free and measured by mechanistic path cuts and linear probes. We vary the reliability (noise rate η) and availability (dropout rate δ) of the CoT in training and track both tests throughout training.
4. **A pre-registered protocol for LLMs (Section 6.2)** that reports N, S and the consistency statistic separately, with concrete predictions derived from Props. 1–3.

The paper is a measurement-methodology contribution. The experiments are deliberately small; we do not claim that the specific numbers transfer to LLMs, only that the logical relations and the qualitative training-condition dependence are properties that any interpretation of LLM faithfulness scores must reckon with.

## 2 Related Work

**Behavioural CoT faithfulness tests.** Lanham et al. (2023) introduced early answering, adding mistakes, paraphrasing and filler-token substitution, and found that dependence on the CoT varies widely across tasks and tends to decrease with model size. Turpin et al. (2023) and Chen et al. (2025) measure whether a CoT *mentions* a feature that demonstrably changed the answer (verbalisation tests). Arcuschin et al. (2025) detect unfaithfulness through logically inconsistent answers to paired questions. Tutek et al. (2025) remove the knowledge expressed by a reasoning step from the parameters by unlearning and measure the effect on the answer (parametric faithfulness). Paul et al. (2024) measure and train for causal dependence of answers on reasoning steps. Jia et al. (2026) frame faithfulness as information flow along the prompt→CoT→answer path and define sufficiency, completeness and necessity diagnostics; their "necessity" coincides with our N, and their emphasis that necessity is interventional matches ours, but they do not separate N from reliance on CoT *content* under conflict, nor validate against mechanistic ground truth.

**Meta-evaluation.** Bentham et al. (2024) show that the Lanham early-answering metric is strongly correlated with accuracy and sensitive to answer-option order. Parcalabescu & Frank (2024) argue that "faithfulness tests" for natural-language explanations measure self-consistency. Gur-Arieh et al. (2026) construct tasks whose outputs reveal the intermediate computations and find most faithfulness metrics near chance against those labels. Our work is complementary: rather than building labelled LLM data, we build models whose mechanism is known and ask what the tests measure *in principle* and under which training conditions they disagree.

**Necessity and monitorability.** Emmons et al. (2025) argue that when CoT is computationally necessary, models struggle to evade CoT monitors; Bürger et al. (2026) show monitoring failures persist even when CoT is necessary and propose a hierarchy of necessity levels. Korbak et al. (2025) and Baker et al. (2025) discuss how training pressures can erode monitorability. Barez et al. (2025) argue that CoT is not by itself explainability. Our Prop. 3 gives a simple formal reason why outcome-only training erodes reliance on tasks that are internally computable, and our experiment shows the erosion as a discrete switch during training.

**Expressivity of CoT.** Theory shows CoT strictly increases what bounded-depth transformers can compute (Feng et al., 2023; Merrill & Sabharwal, 2024; Li et al., 2024), and filler tokens can carry hidden computation (Pfau et al., 2024). These results concern *capability*; our setting is the complementary regime in which the answer is internally computable, which is where faithfulness tests are most needed and most ambiguous.

**Mechanistic ground truth and shortcuts.** We use attention masking to enforce mechanisms and path cuts and probes (Belinkov, 2022) to measure them, in the spirit of causal-abstraction analysis (Geiger et al., 2021). The training dynamics we observe—an easy route learned first, later replaced—relate to grokking and circuit formation in modular arithmetic (Power et al., 2022; Nanda et al., 2023) and to simplicity bias and shortcut learning (Shah et al., 2020; Geirhos et al., 2020).

## 3 Theory

### 3.1 Setup and definitions

An input $x$ determines a correct intermediate $v^*(x)$ and answer $y^*(x)=g(v^*(x),x)$. A CoT $z$ states an intermediate value $v(z)$. A model $M$ defines an answer distribution $\pi_M(y\mid x,\tilde z)$ for any CoT $\tilde z\in\mathcal Z\cup\{\varnothing\}$, where $\varnothing$ denotes an absent CoT. Behavioural tests are functionals of $\pi_M$ evaluated at intervened CoTs. Let $z^*(x)$ be the correct CoT ($v(z^*)=v^*$), and $z'$ a corrupted CoT with $v(z')\neq v^*(x)$ drawn from a corruption distribution $Q(\cdot\mid x)$ such that $g(v(z'),x)\neq y^*(x)$.

**Definition 1 (Necessity).** $N(M)=\mathbb E_x[\pi_M(y^*\mid x,z^*)-\pi_M(y^*\mid x,\varnothing)]$, the accuracy drop under truncation.

**Definition 2 (Reliance).** $S(M)=\mathbb E_x\mathbb E_{z'\sim Q}[\pi_M(g(v(z'),x)\mid x,z')]$, the probability that the answer follows a corrupted CoT. We also record $I(M)=\mathbb E[\pi_M(y^*\mid x,z')]$ ("ignore": the answer reflects the computation the CoT should have stated) and $O=1-S-I$ ("other").

**Definition 3 (Description).** Let the model's computation be a causal graph over internal states $h$. $M$ *describes* via the CoT if there exists an internal variable $h_v$ on every path from $x$ to $y$ that does not pass through the CoT token, such that $h_v$ encodes $v^*(x)$ and interchange interventions on $h_v$ act on $y$ as interventions on $v$ act on $g$ (a causal-abstraction alignment in the sense of Geiger et al., 2021). Informally: the answer is computed *through* the intermediate the CoT states, whether or not the CoT token itself is read.

N and S are what the truncation and corruption tests estimate. D is what people often mean by "the CoT reflects the real reasoning process".

### 3.2 Independence and non-identifiability

**Proposition 1 (N and S are logically independent).** For every $(n,s)\in\{0,1\}^2$ there is a model with $N=n$ and $S=s$; more generally, within the two-route model of Section 3.3, for any $s\in(0,1)$ and any $\epsilon>0$ there is a model with $S=s$ and $N<\epsilon$.

*Proof.* Constructively. (N=1,S=1): the answer is $g(v(\tilde z),x)$ if $\tilde z\neq\varnothing$ and uniform otherwise (pure CoT route). (N=0,S=0): the answer is $y^*(x)$ regardless of $\tilde z$ (pure internal route). (N=0,S=1): the answer is $g(v(\tilde z),x)$ if $\tilde z\ne\varnothing$, else $y^*(x)$ (redundant computation, CoT has precedence). (N=1,S=0): the answer is $y^*(x)$ if $\tilde z\ne\varnothing$, else uniform (the presence of a CoT gates an internal computation that ignores its content). For the continuous statement, use the model of Eq. (1) below with $\alpha-\beta=\log(s/(1-s))$ held fixed and $\beta\to\infty$: then $S\to s$ while $\pi(y^*\mid\varnothing)\to1$ and $\pi(y^*\mid z^*)\to1$, so $N\to0$. ∎

The (N=0, S=1) case is the one that matters in practice: a truncation-based score declares such a model "unfaithful"/"post-hoc" although, whenever the CoT is present, the answer is *entirely* determined by what the CoT says. The (N=1, S=0) case is a truncation false positive, and corresponds to the out-of-distribution concern raised about early answering (Bentham et al., 2024).

**Proposition 2 (Description is not behaviourally identifiable).** Let $\mathcal T$ be any test whose result depends on $M$ only through $\{\pi_M(\cdot\mid x,\tilde z)\}_{x,\tilde z}$. Then there exist models $M_1,M_2$ with $\mathcal T(M_1)=\mathcal T(M_2)$ for all such $\mathcal T$, where $M_1$ describes via the CoT (Def. 3) and $M_2$ does not.

*Proof.* Let $M_1$'s internal route compute $h_v=v^*(x)$ and then $g(h_v,x)$; let $M_2$'s internal route be a lookup table $x\mapsto y^*(x)$ with no intermediate variable that admits an interchange-intervention alignment with $v$ (e.g. a table indexed by the full input, whose internal states are one-hot codes of $x$; an interchange on any proper sub-part of the code produces an output inconsistent with $g$ for some input pair whenever $g$ is non-degenerate in $v$). Combine each internal route with the same CoT-reading route and the same arbitration rule. The input–output maps $(x,\tilde z)\mapsto\pi(\cdot\mid x,\tilde z)$ coincide, so every $\mathcal T$ agrees. ∎

Prop. 2 is elementary, but it fixes the logical status of the debate: claims that a CoT "reflects the real reasoning process" in the sense of D require white-box evidence (probes, interchange interventions, attribution graphs). Black-box CoT interventions can at most establish N and S.

**Proposition 3 (Outcome-optimal predictors are CoT-invariant when the task is internally computable).** Consider training data $(x,\tilde z,y^*(x))$ in which $\tilde z=z^*(x)$ w.p. $1-\eta-\delta$, $\tilde z$ is a uniformly random CoT w.p. $\eta$, and $\tilde z=\varnothing$ w.p. $\delta$, with $x$ having full support. Let $q^\star$ minimise the expected cross-entropy of $y$ over a hypothesis class that contains $(x,\tilde z)\mapsto\delta_{y^*(x)}$. Then:
(a) on the support of the training distribution $q^\star(\cdot\mid x,\tilde z)=\delta_{y^*(x)}$;
(b) if $\eta>0$, the support contains every $(x,z')$, hence $S(q^\star)=\Pr[g(v(z'),x)=y^*(x)]$, i.e. zero for corruptions defined as in Section 3.1, and $N(q^\star)=0$ if $\delta>0$;
(c) if $\eta=0,\delta>0$: $N(q^\star)=0$ but $S(q^\star)$ is unconstrained—every value in $[0,1]$ is attained by some minimiser;
(d) if $\eta=\delta=0$: both $N$ and $S$ are unconstrained.

*Proof.* The cross-entropy of a deterministic label is minimised pointwise, uniquely, by the point mass on the label at every point of positive probability, which gives (a). (b) follows because $\eta>0$ puts positive mass on every $(x,z')$. In (c) and (d), points with $\tilde z=z'\neq z^*(x)$ (and, in (d), $\tilde z=\varnothing$) have zero probability, so the loss does not depend on $q$ there; any completion is a minimiser. ∎

Prop. 3 predicts that reliance observed in trained models on internally computable tasks is either (i) a finite-training phenomenon (an inductive-bias preference for the CoT route that optimisation has not yet removed) or (ii) an off-support extrapolation in the regime where training CoTs are never wrong. It also predicts that truncation and corruption tests will diverge most in regime (c). Section 5 tests these predictions.

### 3.3 A two-route random-utility model and a cross-test identity

Phenomenologically, suppose the answer is drawn from a multinomial logit over the $p$ possible answers with utilities
$$u(y)=\alpha\,\mathbb 1[y=g(v(\tilde z),x)]+\beta\,\mathbb 1[y=y^*(x)]+\varepsilon_y,\qquad(1)$$
with i.i.d. Gumbel noise $\varepsilon_y$, a CoT-route strength $\alpha$ (absent when $\tilde z=\varnothing$) and an internal-route strength $\beta$. Then
$$P_{\text{clean}}=\frac{e^{\alpha+\beta}}{e^{\alpha+\beta}+p-1},\quad P_{\text{trunc}}=\frac{e^{\beta}}{e^{\beta}+p-1},\quad (S,I,O)=\frac{(e^{\alpha},e^{\beta},p-2)}{e^{\alpha}+e^{\beta}+p-2}.$$

**Proposition 4 (Cross-test identity).** Under Eq. (1), $\alpha=\log\frac{(p-2)S}{O}$ and $\beta=\log\frac{(p-2)I}{O}$ are identified from the corruption test alone, and the truncation accuracy is predicted as
$$\hat P_{\text{trunc}}=\frac{(p-2)I/O}{(p-2)I/O+p-1}.\qquad(2)$$
*Proof.* Divide the corruption probabilities by $O$ and substitute into $P_{\text{trunc}}$. ∎

Eq. (2) turns the two tests into a consistency check: if a model's observed truncation accuracy departs from $\hat P_{\text{trunc}}$, the routes do not combine additively—e.g., the model detects the absence of a CoT (presence gating, Prop. 1's N=1,S=0 case) or the truncation input is out of distribution. In either case a single "faithfulness" number is not meaningful for that model. In practice $O$ is often near 0 for well-trained models (Eq. (1) is then near-degenerate), so the check is informative mainly in graded regimes, e.g. during training or under sampling at temperature. We report where it is and is not informative.

## 4 Experimental Testbed

### 4.1 Task and model

Each example is a triple $(a,b,c)\in\mathbb Z_{23}^3$. The intermediate ("reasoning step") is $s=(a+b)\bmod 23$ and the answer is $y=(s+c)\bmod 23$. The sequence is `[a, b, c, SEP, s, ANS]`; the model predicts $s$ at the SEP position and $y$ at the ANS position, with cross-entropy on both. The answer is computable from the inputs alone (as $a+b+c$), so the task sits in the regime where CoT is *not* computationally necessary—the regime in which faithfulness tests are most needed and most ambiguous. All $23^3=12{,}167$ triples are split 80/20 into train and held-out sets by seed; every reported metric is on held-out triples.

The model is a pre-LayerNorm decoder-only transformer with 2 layers, width 64, 4 heads, MLP width 256, learned positional embeddings, trained with AdamW (lr $10^{-3}$, weight decay 0.5, $\beta=(0.9,0.98)$, 200-step warm-up), batch 256, for 4,000 steps (2,500 for controls), one CPU thread per run.

### 4.2 Ground-truth regimes (attention masks)

Masks are applied in every layer.
- **Reliant (positive control):** positions 4 (CoT token) and 5 (ANS) may attend only to positions {2, 4, 5}. Information about $a,b$ can reach the answer *only* through the CoT token. Ground truth: $S=1$ for any model that solves the task.
- **Post-hoc (negative control):** position 5 may not attend to position 4. The answer is causally independent of the CoT token. Ground truth: $S=0$, truncation has no effect.
- **Free:** ordinary causal mask. The model chooses its own mechanism, which we measure.

### 4.3 Training-distribution knobs

For each training sequence, the CoT *input token* is replaced with probability $\eta$ by a uniformly random value (unreliable CoT) and with probability $\delta$ by a `MASK` token (absent CoT). The answer label is always correct, and the CoT prediction target at SEP is always the true $s$. These knobs instantiate the three regimes of Prop. 3: $\eta>0$ (erroneous CoTs on-support), $\eta=0,\delta>0$ (CoT sometimes absent, never wrong), $\eta=\delta=0$. We use $\eta\in\{0,0.05,0.1,0.2,0.4\}$, $\delta\in\{0,0.3\}$, two seeds each (20 free models), plus reliant and post-hoc controls at $(\eta,\delta)=(0,0)$ and $(0.2,0.3)$.

### 4.4 Measurements

**Behavioural tests** (on held-out triples; greedy decoding of $y$):
- *Clean / generated:* accuracy with the true CoT token and with the model's own greedy CoT.
- *Truncation* (necessity): the CoT token is replaced by `MASK`. For $\delta=0$ models this token was never seen in training, which is exactly the out-of-distribution issue that affects early answering in LLMs.
- *Corruption* (reliance): the CoT token is replaced by $s'\sim\mathrm{Unif}(\mathbb Z_{23}\setminus\{s\})$; the output is classified as *follow* ($y=s'+c$), *ignore* ($y=s+c$) or *other*.

**Mechanistic ground truth** (white-box, with the true CoT token, so no out-of-distribution token is introduced):
- *CoT-route cut:* ANS cannot attend to the CoT position. Accuracy measures the sufficiency of the internal route.
- *Internal-route cut:* the CoT and ANS positions cannot attend to $a$, $b$ or SEP. Accuracy measures the sufficiency of the CoT route.
- *Linear probes* (ridge-regularised one-hot least squares, trained on 6,000 training triples, tested on held-out triples) on the residual stream at ANS after each layer, *under corrupted CoT*, decoding (i) the true intermediate $s$ (a white-box proxy for the description property D: the internal route represents the stated-when-correct intermediate) and (ii) the stated corrupted value $s'$. Chance is $1/23\approx0.043$.

**Checkpoint-level statistics.** All behavioural tests are logged every 250 steps. Over checkpoints with clean accuracy > 0.9 we label the truncation verdict "faithful" if the truncation drop exceeds 0.5, and the corruption verdict "faithful" if follow > 0.5, and report the disagreement rate. The cross-test identity (Eq. 2) is evaluated on checkpoints with $O\ge0.02$, where the multinomial-logit fit is non-degenerate.
