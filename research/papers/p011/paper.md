# Necessity Is Not Reliance: Disentangling Chain-of-Thought Faithfulness Tests with Ground-Truth Transformers

*Target venue: NeurIPS 2027 (Main Track; Datasets & Benchmarks–style evaluation-methodology contribution)*

## Abstract

Whether a language model's chain of thought (CoT) reflects the computation that actually produces its answer is usually assessed with behavioural interventions on the CoT text: removing it (truncation / "early answering") or editing it (corruption / "adding mistakes"). These tests are routinely read as measuring one quantity, "faithfulness". We argue, prove, and demonstrate on transformers whose mechanism is known that they measure two logically independent properties: **necessity** (does the answer degrade when the CoT is absent?) and **reliance** (when the CoT is present, does the answer follow what it says?). A third property, **description** (does the internal computation pass through the intermediate the CoT states?), is invisible to any test that only intervenes on CoT text and reads outputs. We (i) formalise the three properties and prove that necessity and reliance are independent, that description is not behaviourally identifiable, and that an outcome-supervised predictor trained to optimality on a task it can compute without the CoT becomes CoT-invariant whenever erroneous CoTs appear in training; (ii) derive a two-route random-utility model under which the three-way outcome of a corruption test (follow / ignore / other) predicts the truncation result, giving a falsifiable cross-test consistency check; and (iii) build a CPU-scale testbed of 2-layer transformers on a two-step modular task in which ground-truth reliance is fixed by attention masks or measured by mechanistic path cuts. Across 26 freely trained models and 511 well-trained checkpoints the two tests disagree on 13.3% of checkpoints, and the disagreements are regime-specific as the theory predicts: when the CoT is sometimes absent but never wrong in training, 24.7% of checkpoints (and 3 of 6 final models) are *redundant-but-reliant*—truncation calls them unfaithful, yet they follow a corrupted CoT ≥98.6% of the time and their CoT route alone suffices under a mechanistic path cut—whereas when training CoTs are sometimes wrong, all disagreements (9.1%) are truncation false positives. Reliance measured by corruption tracks mechanistic CoT-route sufficiency (Spearman ρ = 0.96, n = 26). The cross-test identity holds where routes combine additively (r = 0.97, mean absolute error 0.054, 189 checkpoints) and fails in the redundant regime (MAE 0.41, 31 checkpoints), flagging the models for which a single score misleads. We also report that one of our attention-mask "ground-truth" controls leaked through an unanticipated path that training exploited, a caution for constructed ground truth. The practical consequence is that a single "faithfulness score" from truncation or corruption is not interpretable; we give a pre-registered protocol that reports necessity, reliance and a cross-test consistency statistic separately, and that we believe should replace single scores in LLM evaluations.

## 中文摘要

判断语言模型的“思维链”（CoT）是否忠实反映其真实推理过程，目前主要依赖对 CoT 文本的行为干预：删除它（截断/“提前作答”）或篡改它（插入错误）。这些测试通常被当作同一个量——“忠实性”——来解读。本文论证、证明并在机制已知的 Transformer 上验证：它们实际测量的是两个逻辑上相互独立的性质——**必要性**（去掉 CoT 后答案是否变差）与**依赖性**（CoT 存在时答案是否服从其内容）；而第三个性质**描述性**（内部计算是否经过 CoT 所陈述的中间量）对任何“只干预 CoT 文本、只观察输出”的测试都不可识别。我们 (i) 形式化这三种性质，证明必要性与依赖性相互独立、描述性在行为层面不可识别，并证明：若任务无需 CoT 即可计算，且训练数据中出现错误的 CoT，则以结果监督训练至最优的预测器必然对 CoT 不变；(ii) 提出双通路随机效用模型，使篡改测试的三类结果（服从/忽略/其他）可以预测截断测试的结果，从而给出可证伪的跨测试一致性检验；(iii) 构建 CPU 规模的实验平台：在两步模运算任务上训练 2 层 Transformer，其真实依赖关系由注意力掩码固定，或由机制性路径切断测量。在 26 个自由训练的模型、511 个训练充分的检查点上，两种测试在 13.3% 的检查点上结论相反，且分歧的类型完全符合理论预测：当训练中 CoT 有时缺失但从不出错时，24.7% 的检查点（以及 6 个最终模型中的 3 个）属于“冗余但依赖”型——截断测试判其不忠实，但它们对被篡改 CoT 的服从率不低于 98.6%，且机制性路径切断表明仅凭 CoT 通路即可正确作答；而当训练 CoT 有时出错时，全部分歧（9.1%）都是截断测试的假阳性。篡改测试测得的依赖性与机制性 CoT 通路充分性高度一致（Spearman ρ = 0.96，n = 26）。跨测试恒等式在两条通路可加组合时成立（r = 0.97，平均绝对误差 0.054，189 个检查点），在冗余区间失效（MAE 0.41，31 个检查点），从而准确标出单一分数会误导的模型。我们还报告：我们的一个基于注意力掩码的“真值”对照存在意料之外的信息泄漏通路，并被训练所利用——这提醒人们构造真值时需格外谨慎。 结论是：单一的截断或篡改“忠实性分数”不可解释。我们给出一个预注册协议，分别报告必要性、依赖性及跨测试一致性统计量，建议在大模型评测中取代单一分数。

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
(b) if $\eta>0$, the support contains every $(x,z')$, hence $S(q^\star)=\Pr[g(v(z'),x)=y^*(x)]$, i.e. zero for corruptions defined as in Section 3.1, and $N(q^\star)=0$ if $\delta>0$, while if $\delta=0$ the value of $N(q^\star)$ is unconstrained (the truncated input is off-support);
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

The model is a pre-LayerNorm decoder-only transformer with 2 layers, width 64, 4 heads, MLP width 256, learned positional embeddings, trained with AdamW (lr $10^{-3}$, weight decay 0.5, $\beta=(0.9,0.98)$, 200-step warm-up), batch 256, for 4,000 steps (2,500 for controls; 8,000 or 12,000 for six longer runs, Section 4.3), one CPU thread per run.

### 4.2 Ground-truth regimes (attention masks)

Masks are applied in every layer.
- **Reliant (positive control):** positions 4 (CoT token) and 5 (ANS) may attend only to positions {2, 4, 5}. Information about $a,b$ can reach the answer *only* through the CoT token. Ground truth was *intended* to be $S=1$. As Section 5.1 shows, this mask leaves a path open (position 2 may still attend to $a,b$), so we add:
- **Reliant-strict (positive control):** as Reliant, and position 2 may attend only to itself. Now no information about $a,b$ reaches positions 4–5 except through the CoT token; ground truth $S=1$.
- **Post-hoc (negative control):** position 5 may not attend to position 4. The answer is causally independent of the CoT token. Ground truth: $S=0$, truncation has no effect.
- **Free:** ordinary causal mask. The model chooses its own mechanism, which we measure.

### 4.3 Training-distribution knobs

For each training sequence, the CoT *input token* is replaced with probability $\eta$ by a uniformly random value (unreliable CoT) and with probability $\delta$ by a `MASK` token (absent CoT). The answer label is always correct, and the CoT prediction target at SEP is always the true $s$. These knobs instantiate the three regimes of Prop. 3: $\eta>0$ (erroneous CoTs on-support), $\eta=0,\delta>0$ (CoT sometimes absent, never wrong), $\eta=\delta=0$. We use $\eta\in\{0,0.05,0.1,0.2,0.4\}$, $\delta\in\{0,0.3\}$, two seeds each (20 free models), plus reliant, reliant-strict and post-hoc controls at $(\eta,\delta)=(0,0)$ and $(0.2,0.3)$. Because regime (c) produced seed-dependent outcomes and slow late-training changes, we added four seeds at $(\eta,\delta)=(0,0.3)$ trained for 8,000 steps and two seeds at $(0.05,0)$ trained for 12,000 steps (26 free models in total; 32 models overall). The extension was decided after inspecting the 4,000-step results, and we report all runs.

### 4.4 Measurements

**Behavioural tests** (on held-out triples; greedy decoding of $y$):
- *Clean / generated:* accuracy with the true CoT token and with the model's own greedy CoT.
- *Truncation* (necessity): the CoT token is replaced by `MASK`. For $\delta=0$ models this token was never seen in training, which is exactly the out-of-distribution issue that affects early answering in LLMs.
- *Corruption* (reliance): the CoT token is replaced by $s'\sim\mathrm{Unif}(\mathbb Z_{23}\setminus\{s\})$; the output is classified as *follow* ($y=s'+c$), *ignore* ($y=s+c$) or *other*.

**Mechanistic ground truth** (white-box, with the true CoT token, so no out-of-distribution token is introduced):
- *CoT-route cut:* ANS cannot attend to the CoT position. Accuracy measures the sufficiency of the internal route.
- *Internal-route cut:* the CoT and ANS positions cannot attend to $a$, $b$ or SEP. Accuracy measures the sufficiency of the CoT route. Because position 2 ($c$) can itself attend to $a,b$, we also report a *strict* internal-route cut that additionally blocks position 2 from $a,b$; we use the strict version as the mechanistic reliance reference.
- *Linear probes* (ridge-regularised one-hot least squares, trained on 6,000 training triples, tested on held-out triples) on the residual stream at ANS after each layer, *under corrupted CoT*, decoding (i) the true intermediate $s$ (a white-box proxy for the description property D: the internal route represents the stated-when-correct intermediate) and (ii) the stated corrupted value $s'$. Chance is $1/23\approx0.043$.

**Checkpoint-level statistics.** All behavioural tests are logged every 250 steps. Over checkpoints with clean accuracy > 0.9 we label the truncation verdict "faithful" if the truncation drop exceeds 0.5, and the corruption verdict "faithful" if follow > 0.5, and report the disagreement rate. The cross-test identity (Eq. 2) is evaluated on checkpoints with $O\ge0.02$, where the multinomial-logit fit is non-degenerate.

All numbers in Sections 5 and 6 come from code in `code/` run in this environment; the source file under `results/` is given for each. Each training run takes 1–5 CPU-minutes; the whole study takes roughly an hour on 4 cores.

## 5 Results

### 5.1 Controls, and a leak in one of them

Table 1 reports the attention-mask controls (source: `results/final_table.md`, `results/supp_table.md`).

**Table 1.** Controls. "Trunc" = accuracy with the CoT removed; "Follow/Ignore" = corruption outcomes; "CoT-cut" = accuracy when ANS cannot read the CoT position; "Int-cut (strict)" = accuracy when only the CoT route is left open.

| Model | $(\eta,\delta)$ | Clean acc | Trunc | Follow | Ignore | CoT-cut | Int-cut (strict) |
|---|---|---|---|---|---|---|---|
| Post-hoc | (0, 0) | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.041 |
| Post-hoc | (0.2, 0.3) | 0.999 | 0.999 | 0.000 | 0.999 | 0.999 | 0.039 |
| Reliant | (0, 0) | 1.000 | 0.039 | 1.000 | 0.000 | 0.043 | 1.000 |
| Reliant | (0.2, 0.3) | 0.998 | 0.721 | 0.117 | 0.454 | 0.083 | 0.177 |
| Reliant-strict | (0, 0) | 1.000 | 0.044 | 1.000 | 0.000 | 0.043 | 1.000 |
| Reliant-strict | (0.2, 0.3) | 1.000 | 0.036 | 1.000 | 0.000 | 0.043 | 1.000 |

The post-hoc and reliant-strict controls behave as designed under both tests and both path cuts. The original "reliant" mask did not. Under the noisy/absent-CoT training distribution, that model learned to answer correctly when the CoT was removed (truncation accuracy 0.721) and to ignore a corrupted CoT 45% of the time. The strict internal-route cut drops its accuracy from 0.998 (non-strict cut) to 0.177, which locates the leak: $a$ and $b$ reach the answer through the residual stream at the position of $c$, which the "reliant" mask still lets attend to $a,b$. With clean training data the same mask gives a perfectly reliant model, because nothing pushes optimisation toward the leak. **Lesson:** "ground truth by construction" holds only as far as the construction is airtight. Training pressure against reliance (here, noisy CoTs) finds paths the designer did not intend. Constructed-faithfulness benchmarks for LLMs face the same risk, and the internal-route cut must be checked for all paths, not only the obvious ones.

### 5.2 Training conditions determine reliance as Prop. 3 predicts, up to finite-training effects

Figure 1 (`results/fig1_final_vs_eta.png`) shows the final free models as a function of $\eta$. The per-regime outcomes (`results/supp_stats.json`) are:

- **Regime (b), $\eta\ge0.1$:** Prop. 3 predicts $S=0$. In 11 of 12 final models P(follow) is below 0.25, and in 7 of 12 it is below 0.01. The exception ($\eta=0.1,\delta=0$, seed 0; follow 0.855) is visibly still losing reliance at step 4,000 (Fig. 2, middle panel). With $\eta=0.05$, 4 of 6 models still follow at the end of training (follow ∈ {0.703, 1.0, 1.0, 0.836}), including one run trained for 12,000 steps. The prediction holds asymptotically, but convergence slows as $\eta$ shrinks, as expected when the gradient signal against reliance is proportional to the fraction of noisy examples.
- **Reliance is acquired first and lost later.** In 15 of 26 free runs P(follow) first exceeds 0.5 and later falls below it (switch steps 750–7,000; `summary.json`). This includes 14 of the 15 runs that end with follow < 0.5. Fig. 2 shows the pattern: copying the CoT token is learned within ~500 steps, and the internal route ($a+b+c$) is learned later and then takes over. Training never produced a model that relied on the internal route from the start. One run ($\eta=\delta=0$, seed 1) dipped below 0.5 and recovered to 1.0.
- **Regime (c), $\eta=0,\delta=0.3$:** Prop. 3 predicts $N=0$ but leaves $S$ unconstrained. All six models have final follow ≥ 0.44 (values 1.0, 0.45, 0.986, 0.439, 1.0, 1.0). Truncation accuracy ranges from 0.038 to 0.996 depending on seed and training length.
- **Regime (d), $\eta=\delta=0$:** both models are fully reliant (follow 1.0) and fail at chance when the CoT is removed (0.041, 0.043). The removed-CoT token was never seen in training, so the truncation result is off-support. Here it happens to agree with the mechanism: the CoT-cut accuracy is 0.043/0.044, and there is no internal route.

### 5.3 Where the tests disagree, and in which direction

At the checkpoint level (511 free-model checkpoints with clean accuracy > 0.9; `results/supp_stats.json`, `results/ckpt_disagree_by_regime.json`), the binary verdicts disagree on 13.3%. The continuous measures (truncation drop vs. P(follow)) have Spearman ρ = 0.81. The disagreement is not noise; its *direction* is fixed by the training regime:

**Table 2.** Checkpoint-level verdict disagreement by training regime.

| Training regime | Checkpoints | Disagree | N− S+ (trunc "unfaithful", corruption "faithful") | N+ S− |
|---|---|---|---|---|
| $\eta=0,\delta=0.3$ | 154 | 24.7% | 24.7% | 0% |
| $\eta=0,\delta=0$ | 29 | 0% | 0% | 0% |
| $\eta>0$ (both $\delta$) | 328 | 9.1% | 0% | 9.1% |

Both directions are predicted by the theory. **N−S+ (redundant-but-reliant)** appears only in regime (c), where the absent-CoT token is on-support (so the model must answer without a CoT) but no training example ever contradicts the CoT (so following it is never penalised). Three of the six final models in this regime are of this type (Table 3). Their truncation accuracy is 0.75–1.00, so a truncation-based evaluation would call their CoT post-hoc. Yet they follow a corrupted CoT 98.6–100% of the time, and the CoT route alone gives 0.65–1.00 accuracy under the strict internal-route cut. Whenever a CoT is present, these models' answers are determined by what it says. **N+S−** appears only with $\eta>0$, and mostly with $\delta=0$ (e.g. $\eta=0.2,\delta=0$: truncation accuracy 0.36/0.40 while follow is 0.12/0.21). These models do not follow the CoT but fail when it is removed, because the removal token is off-support. This is the truncation false positive anticipated by Bentham et al. (2024). Figure 4 (`results/fig4_verdict_plane.png`) plots all checkpoints in the (truncation-drop, follow) plane.

**Table 3.** The three redundant-but-reliant final models (all $\eta=0,\delta=0.3$, 8,000 steps).

| Seed | Clean acc | Trunc | Follow | CoT-cut | Int-cut (strict) | Probe: true $s$ at ANS, layer 2 |
|---|---|---|---|---|---|---|
| 2 | 0.994 | 0.816 | 0.986 | 0.062 | 0.652 | 0.062 |
| 4 | 1.000 | 0.753 | 1.000 | 0.039 | 0.980 | 0.044 |
| 5 | 1.000 | 0.996 | 1.000 | 0.046 | 1.000 | 0.054 |

Table 3 also shows a mechanistic detail that the two-route picture of Section 3.3 does not capture. With the true CoT token in place, blocking the answer position's access to the CoT position (CoT-cut) drops accuracy to chance (≤ 0.062), even though the same models answer correctly when the CoT token is *replaced by* `MASK`. So the "internal route" is not a separate pathway running in parallel with the CoT reading. It is computed at, or read through, the CoT position, and it is switched on by the absence signal there. We call this **presence gating**. It is a concrete mechanism behind the abstract (N=0, S=1) construction of Prop. 1. It also means that truncation in these models measures a computation that *does not run* when a CoT is present. Truncation accuracy is therefore uninformative about the computation behind CoT-conditioned answers. Fig. 2 (second panel) shows this route emerging late (after step ~6,000 for seed 4) while P(follow) stays at 1.0 throughout.

### 5.4 Which behavioural test tracks the mechanism?

Over the 26 final free models (`results/supp_stats.json`):

- P(follow) vs. strict internal-route-cut accuracy (sufficiency of the CoT route): Spearman ρ = 0.96 (p ≈ 2×10⁻¹⁵).
- Truncation accuracy vs. CoT-cut accuracy (sufficiency of the internal route with the CoT position unreadable): ρ = 0.83. The mismatch is concentrated in the presence-gated models of Table 3.
- P(follow) vs. probe accuracy for the *true* intermediate $s$ at the answer position under corrupted CoT (layer 2): ρ = −0.87. Models that follow the CoT carry no decodable trace of the correct intermediate when the CoT is wrong (probe accuracy 0.04–0.06, chance 0.043). Models that ignore it carry one (0.24–0.76).

The corruption test is therefore the better behavioural proxy for *reliance* in this testbed. Truncation measures something else: the existence and accessibility of an alternative route when the CoT is absent.

The probes also bear on description (D). The post-hoc controls answer perfectly without reading the CoT, but the true intermediate $s=(a+b)\bmod23$ is only weakly decodable at their answer position (0.22–0.24 at layer 2; chance 0.043; `final_table.md`). By contrast, some free models that ignore the CoT reach 0.6–0.76. A linear probe is a weak test of D (a nonlinear encoding of $s$ would be missed), so we do not claim that the post-hoc models bypass $s$. The observation does illustrate Prop. 2: two models with identical behaviour on every CoT intervention (both post-hoc controls and, e.g., the $\eta=0.4$ models have trunc ≈ 1, follow ≈ 0) differ in whether the stated intermediate is internally represented, and only white-box measurement shows the difference.

### 5.5 The cross-test identity is a usable consistency check

Eq. (2) predicts truncation accuracy from the three-way corruption outcome. We evaluate it on the graded checkpoints where it is non-degenerate (other ≥ 0.02; `results/crosstest_by_regime.json`, Fig. 3):

**Table 4.** Cross-test identity (Eq. 2), predicted vs. observed truncation accuracy.

| Checkpoints | n | Pearson r | MAE | Fraction with \|residual\| > 0.3 |
|---|---|---|---|---|
| All free | 220 | 0.83 | 0.104 | — |
| Excluding $\eta=0,\delta=0.3$ | 189 | 0.97 | 0.054 | 0.5% |
| $\eta=0,\delta=0.3$ | 31 | 0.89 | 0.409 | 58.1% |

A temperature-1 (soft-probability) version computed per example on the final models (`analyze.py: soft_eval`) gives the same picture. The model-level absolute residual averages 0.028 for the 21 models with residual ≤ 0.2. The five models with residual > 0.2 all come from the $\eta=0,\delta=0.3$ regime, and they include all three presence-gated models (`supp_stats.json`). Where the identity holds, a single latent pair (α, β) explains both tests, and reporting N and S together is coherent. Where it fails, the model handles an absent CoT differently from a present-but-different one, and the truncation result must not be read as a statement about the CoT-conditioned computation. The check needs only the corruption outcomes (follow/ignore/other) and the truncation accuracy, and is cheap to run on LLMs whenever the answer space is enumerable.

## 6 Discussion

### 6.1 What this means for "is the CoT faithful?"

The question as posed in the field bundles three properties. On the evidence here: (i) **reliance** (S) is what most people mean operationally—"if the CoT had said something else, would the answer change accordingly?"—and the corruption test measures it well when corruptions are on-support and the three-way outcome is recorded; (ii) **necessity** (N) answers a different question—"is there a way to the answer without the CoT?"—and can be high or low independently of S (Prop. 1; Table 3); (iii) **description** (D) cannot be settled by any CoT-text intervention (Prop. 2) and needs white-box tools. Prop. 3 and Section 5.2 add a training-side reading: outcome-only training on internally computable tasks *removes* reliance whenever the CoT the model conditions on is sometimes wrong, but only after an initial phase in which reliance is the easy solution. For LLMs trained with outcome-based RL on their own sampled CoTs, sampled CoTs are sometimes wrong while the reward is still obtained. Our model therefore predicts erosion of reliance on tasks the model can solve without the CoT. This is consistent with Lanham et al.'s (2023) observation that larger models depend less on their CoT, and with the monitorability concerns of Korbak et al. (2025) and Baker et al. (2025). We stress that this is a hypothesis about LLMs, not a result.

The redundant-but-reliant type deserves attention from the monitoring literature. A CoT monitor needs S (the visible text controls the answer), not N. A model of this type gives a monitor exactly what it needs while being labelled "post-hoc" by early answering. Conversely, high N does not imply high S: a necessity-based argument for monitorability (Emmons et al., 2025) should be accompanied by a reliance measurement. This agrees with Bürger et al.'s (2026) finding that monitoring can fail even when CoT is necessary.

### 6.2 A pre-registered protocol for LLM evaluations

We propose the following protocol and state its predictions in advance. We have not run it.

1. **Items.** Tasks with an enumerable answer space and a CoT whose decisive intermediate can be edited coherently (multi-step arithmetic, multiple-choice with a computed key quantity), split into *internally solvable* items (no-CoT accuracy ≥ 80%) and *CoT-necessary* items (no-CoT accuracy near chance).
2. **Tests.** (a) Truncation with at least two removal formats (empty CoT; CoT replaced by neutral filler of equal length), to expose off-support effects. (b) Corruption of the decisive intermediate, keeping the rest of the CoT consistent with the corrupted value, with the three-way outcome (follow/ignore/other) recorded. Forced answering is used so that the model cannot regenerate the CoT.
3. **Report** N, S and (I, O) separately, the Eq. (2) residual per item and per model, and the rate of N−S+ and N+S− items. Never report a single combined score.
4. **Predictions** (falsifiable): (P1) On internally solvable items, the N−S+ rate exceeds the N+S− rate when the truncation format is on-support (filler), and the order reverses when it is off-support (empty CoT). (P2) Eq. (2) residuals are larger on internally solvable items than on CoT-necessary items. (P3) Across checkpoints of an outcome-RL run, S on internally solvable items declines while N on CoT-necessary items stays high. (P4) Item-level S correlates more strongly with attribution-based measures of CoT→answer influence (e.g., attribution graphs, Lindsey et al., 2025; or unlearning-based measures, Tutek et al., 2025) than item-level N does.
5. **Failure conditions.** If P1 fails in both formats, our claim that truncation tracks route availability rather than reliance does not transfer to LLMs. If P4 fails, the corruption test is not the better reliance proxy at scale.

### 6.3 Relation to verbalisation-based tests

Hint-verbalisation tests (Turpin et al., 2023; Chen et al., 2025; Gur-Arieh et al., 2026) ask about *completeness*: does the CoT mention every factor that influenced the answer? Our framework does not cover completeness. The two are complementary: a model can have S = 1 (the answer follows the CoT) and still hide an influence that shaped the CoT itself. Jia et al. (2026) formalise completeness and sufficiency information-theoretically; our contribution is to separate necessity from reliance and to validate the separation against mechanism.

## 7 Limitations

- **Scale and task.** All models are 2-layer, width-64 transformers on a single two-step modular addition task with a one-token CoT and a 23-way answer space. LLM CoTs are long, partially redundant, written in natural language, and generated by the model itself. Effects such as error correction in later CoT steps (Yee et al., 2024) cannot appear here. We make no quantitative claim about LLMs; Section 6.2 is a protocol, not a result.
- **The internal route is easy.** We chose an internally computable task deliberately. On tasks where CoT is computationally necessary for the architecture (Feng et al., 2023; Merrill & Sabharwal, 2024), N is high by construction, and our redundancy findings do not apply.
- **Few seeds, and an adaptive extension.** Most configurations have 2 seeds. Regime (c) has 6, of which 4 were added after we saw the 4,000-step results, with longer training. Reported proportions (e.g. 3/6 redundant-but-reliant) are small-sample and seed-sensitive. We report every run but do not claim stable rates.
- **Thresholds.** The binary verdicts use thresholds of 0.5 (truncation drop and follow rate). Disagreement rates depend on them. The continuous correlations (ρ = 0.81 between the two tests; ρ = 0.96 between follow and the strict path cut) do not.
- **Mechanistic ground truth is itself imperfect.** Path cuts are attention-mask interventions and change activation distributions. The "reliant" mask leaked (Section 5.1). Presence gating means the "internal route" is not a separable pathway. Linear probes miss nonlinear encodings. Our ground truth is therefore stronger than behavioural tests but not absolute.
- **The two-route model is phenomenological.** Eq. (1) assumes additive utilities with Gumbel noise. It is uninformative when the "other" outcome is near zero (as it is for most fully trained models), so the cross-test check applies only to graded checkpoints or soft probabilities.
- **No natural-language component.** We do not test paraphrase, filler or verbalisation, and we cannot say anything about whether CoTs omit influences.

## 8 Conclusion

Truncation and corruption tests of chain-of-thought faithfulness measure different things. Truncation measures necessity, i.e. whether an alternative route to the answer exists when the CoT is absent. Corruption measures reliance, i.e. whether the answer follows what the CoT says when it is present. Neither measures whether the internal computation passes through the stated intermediate. On transformers whose mechanism we can check, the two tests disagree in a regime-dependent and theoretically predicted way. Corruption tracks mechanistic reliance closely, while truncation can label fully reliant models as post-hoc. A simple cross-test identity flags the models for which the tests cannot be combined. The narrow answer to "does the CoT faithfully reflect the model's reasoning?" is therefore that the question has at least three parts, behavioural tests can answer two of them, and those two should be reported separately and checked against each other.

## References

Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., & Conmy, A. (2025). Chain-of-thought reasoning in the wild is not always faithful. arXiv:2503.08679.

Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., & Farhi, D. (2025). Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv:2503.11926.

Barez, F., et al. (2025). Chain-of-thought is not explainability. Preprint, Oxford Martin AI Governance Initiative (July 2025).

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219.

Bentham, O., Stringham, N., & Marasović, A. (2024). Chain-of-thought unfaithfulness as disguised accuracy. *Transactions on Machine Learning Research*. arXiv:2402.14897.

Bürger, L., et al. (2026). Chain of thought monitoring can fail even when CoT is necessary. arXiv:2610.18105.

Chen, Y., Benton, J., Radhakrishnan, A., Uesato, J., Denison, C., Schulman, J., et al. (2025). Reasoning models don't always say what they think. arXiv:2505.05410.

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

Tutek, M., Hashemi Chaleshtori, F., Marasović, A., & Belinkov, Y. (2025). Measuring chain of thought faithfulness by unlearning reasoning steps. arXiv:2502.14829.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

Yee, E., Li, A., Tang, C., Jung, Y. H., Paturi, R., & Bergen, L. (2024). Dissociation of faithful and unfaithful reasoning in LLMs. arXiv:2405.15092.
