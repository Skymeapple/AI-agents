# No Privileged Variance, No Introspection Test: Identifiability Limits and a Steering Confound in Self-Report Evidence about AI Consciousness

**Target venue:** NeurIPS 2027 (main track: evaluation, interpretability and the science of AI)

*Code, data and figures: `code/` and `results/` in this directory. Every number in this paper was either produced by that code (marked [R], with the result file) or comes from a cited source (marked [C]).*

---

## Abstract

Self-report is the main bridge between experience and science in human consciousness research. Several research programmes now try to use AI self-reports, together with activation-steering "concept injection" experiments, as evidence about introspection and, more distantly, about the consciousness-relevant indicator properties of AI systems. The central obstacle is the *gaming problem*. A system trained on human text can produce reports that match typical reports without those reports being caused by its own internal states. We formalise this obstacle in a minimal model with three parameters: the fraction α of an internal state's variance that is *privileged* (not predictable from the input by an outside observer), the fraction λ of training labels that are about the system's own state rather than about other systems, and the representational entanglement between state and world features. We prove four results. (i) With no privileged variance (α = 0), self-trained and imitation-trained reporters come from identical training distributions, so no test, observational or interventional, can tell them apart. (ii) The disagreement rate between own-state and typical reports is κ(α) = arccos(1−α²)/π, and an imitator's accuracy about its own state is capped at 1 − arccos(√(1−α²))/π. (iii) Under mixed labels the Bayes-optimal reporter outputs λ·s + (1−λ)·p(x), which yields a closed-form region in which reports track the system's own state. (iv) For ridge reporters we derive in closed form how strongly an imitator responds to interventions on the state. A difference-of-means steering direction, the usual construction in concept-injection studies, produces imitator "false positives" that do not shrink as regularisation goes to zero. A *residualised* direction, built from the part of the activations that the input does not explain, removes them. A coupling argument also gives a "self-evidence budget": no statistic can separate the two kinds of reporter unless the training set contains about one or more labels on which own-state and typical reports disagree. We test these predictions on CPU with 1,044 trained MLP reporters [R]. At α ≥ 0.1, accuracy on idiosyncratic trials and the residualised steering score separate self-trained from imitation-trained reporters perfectly (AUC = 1.0, 8 vs 8 seeds). The difference-of-means steering score reaches AUC ≥ 0.95 only at α ≥ 0.5, and it gives imitators intervention scores of 0.79–0.85 for α ≤ 0.7. We turn these results into a pre-registrable protocol for LLM introspection studies. These results say nothing directly about phenomenal consciousness. They specify when self-report evidence *can* bear on one consciousness-relevant indicator, metacognitive monitoring, and when it cannot.

## 中文摘要

在人类意识科学中，自我报告是连接主观经验与科学测量的主要桥梁。近年来，一些研究开始借助AI的自我报告以及基于激活引导（activation steering）的“概念注入”实验，来论证AI具有内省能力，进而间接讨论与意识相关的指标属性。其核心障碍是“博弈问题”：用人类文本训练的系统，可能只是复述人类的典型报告，而这些报告并非由其自身内部状态引起。本文用一个只有三个参数的最小模型来形式化这一问题。三个参数是：内部状态中外部观察者无法从输入预测的“特权方差”比例α，训练标签中关于系统自身状态（而非其他系统）的比例λ，以及状态表征与世界表征之间的纠缠程度。我们证明了四个结论。(i) 当α=0时，“自我训练”与“模仿训练”的报告器来自完全相同的训练分布，因此任何观察性或干预性检验都无法区分二者。(ii) 自身状态与典型报告的不一致率为κ(α)=arccos(1−α²)/π，模仿者对自身状态的准确率上限为1−arccos(√(1−α²))/π。(iii) 在混合标签下，贝叶斯最优报告器输出λ·s+(1−λ)·p(x)，由此可以写出报告跟随自身状态的闭式区域。(iv) 对岭回归报告器，我们闭式推导了模仿者对状态干预的响应强度。概念注入研究常用的“均值差”引导方向会让模仿者产生假阳性，而且这种假阳性不随正则化趋于零而消失；用输入无法解释的激活残差构造的“残差化”方向则可以消除它。此外，一个耦合论证给出了“自我证据预算”：除非训练集中至少约有一个自身状态与典型报告不一致的标签，否则任何统计量都无法区分两类报告器。我们在CPU上训练了1,044个MLP报告器来检验这些预测。α≥0.1时，特异试次准确率和残差化引导得分都能完美区分两类报告器（AUC=1.0）。均值差引导得分只有在α≥0.5时AUC才达到≥0.95；在α≤0.7时，它给模仿者的干预得分仍有0.79–0.85。我们据此给出一个可预注册的大模型内省检验方案。本文不直接回答AI是否具有现象意识。它说明的是：自我报告证据在什么条件下能够、在什么条件下不能为一个与意识相关的指标（元认知监控）提供证据。

---

## 1 Introduction

Whether current or future AI systems are, or could become, conscious is now discussed seriously in philosophy, neuroscience and machine learning [C: Chalmers 2023; Butlin et al. 2023; Long et al. 2024]. The most influential methodological proposal, the indicator-property approach of Butlin et al. (2023), derives architectural and functional indicators from scientific theories of consciousness and checks AI systems against them. Several indicators, especially those from higher-order theories, concern *metacognitive monitoring*: a system's capacity to represent and report on its own first-order states. In human consciousness science the main evidence for such monitoring is self-report, interpreted with care because people's reports about their own mental processes are often confabulated (Nisbett & Wilson 1977) and must be separated from first-order performance (Fleming & Lau 2014).

For large language models (LLMs) self-report faces a sharper problem. Birch (2024) calls it the *gaming problem*: a system trained on vast amounts of human-generated text can reproduce the behaviours by which we would recognise sentience or introspection without having the underlying capacity. Perez and Long (2023) proposed training models to answer questions about themselves whose answers are known, in the hope of making self-reports evidentially useful. Binder et al. (2025) showed that fine-tuned models predict their own behaviour better than other models trained on the same behaviour. Lindsey (2025/2026) introduced *concept injection*: an experimenter adds a concept's activation vector to the model's residual stream and checks whether the model's self-report changes accordingly. Some models sometimes detect and name the injected concept. A rapid follow-up literature has examined confounds in this paradigm. Global "yes"-biases explain binary detection (Hahami et al. 2025). Detection can be content-agnostic (Lederman & Mahowald 2026). The proposed mechanism has been dissected (Macar et al. 2026). Recent analyses also argue that many existing paradigms fail even a *privileged-access* condition (Song et al. 2025b; Singh et al. 2026).

The debate has produced definitions: "lightweight" causal ones (Comsa & Shanahan 2025) and thicker ones built on privileged self-access (Song et al. 2025b). It has also produced careful experiments. What it lacks is a *quantitative theory* of when a self-report test can work at all, and of how the standard steering-based test fails. This paper supplies one, in the simplest model that keeps the essential structure. A system has an internal state that is partly determined by its input ("world-determined") and partly privileged. A reporter reads an entangled activation vector. Training labels are a mixture of the system's own states and the states of *other* members of a population given the same input. The second kind of label models learning to report from human-written text, which describes other minds' states in similar situations.

**Contributions.**

1. **A minimal formal model of the gaming problem** with three interpretable parameters: privileged-variance fraction α, self-label fraction λ, and state–world entanglement (§3). It makes Song et al.'s (2025b) privileged-access condition quantitative.
2. **An identifiability theorem and closed forms** (§4): (a) at α = 0, self-trained and imitation-trained reporters are indistinguishable by *any* test, including interventional ones (Prop. 1); (b) closed-form disagreement rate κ(α) and imitator ceiling (Prop. 2); (c) the Bayes-optimal mixed-label reporter and its "grounded region" (Prop. 3); (d) a self-evidence budget bound, |AUC − ½| ≤ 1 − (1 − λκ)^{nK}, on the separating power of any statistic (Prop. 6).
3. **A theory of the steering confound** (§4.4): closed-form intervention responses of ridge reporters show that difference-of-means steering directions, the standard construction, give imitators interventional "false positives" that do not vanish as regularisation goes to zero. Directions built from input-residualised activations do vanish, at a rate set by α²/ρ (Props. 4–5).
4. **CPU-scale validation** on 1,044 trained MLP reporters across five experiments (§5–6). The experiments confirm the predicted ordering of test statistics, the regularisation dependence, and the λ-mixture law at the level of probabilities. They also expose where the simple theory is loose: thresholded accuracy near λ ≈ ½, and the budget bound, which is valid but about one to two orders of magnitude from the empirical separability threshold.
5. **A pre-registrable protocol** for LLM introspection studies (§7.2): estimate privileged variance, residualise steering directions, report idiosyncratic-trial accuracy, run an imitation-trained sibling as a negative control, and report the self-evidence budget of any introspection fine-tuning set.

We do not claim that any AI system is or is not conscious. The claim is narrower and, we think, more useful: it identifies the conditions under which self-report and steering experiments can carry evidence about a consciousness-relevant indicator at all.

## 2 Related Work

**Consciousness science and AI.** Chalmers (2023) assessed the case for and against LLM consciousness and concluded that current models are unlikely to be conscious but that successors might be. Butlin et al. (2023) derived indicator properties from recurrent processing, global workspace, higher-order, predictive processing and attention schema theories, and concluded that no current system is a strong candidate while no obvious technical barrier exists. Long et al. (2024) argued that the realistic possibility of AI moral patienthood calls for assessment procedures. Birch (2024, ch. 16) introduced the gaming problem for behavioural tests of LLM sentience and recommended searching for "deep computational markers" instead. Doerig et al. (2019) argued, through the unfolding argument, that causal-structure theories such as IIT are either falsified or unfalsifiable when tested by input–output behaviour. This is related to our Prop. 1: there, two internally different mechanisms are made indistinguishable by their training distribution rather than by their I/O function.

**Introspection and self-report in LLMs.** Kadavath et al. (2022) showed that LMs can be calibrated about whether they know an answer. Perez and Long (2023) proposed self-report training for moral-status evaluation. Binder et al. (2025) gave cross-prediction evidence for introspection. Song et al. (2025a) found no privileged self-access for grammatical knowledge across 21 open models, and Song et al. (2025b) proposed defining introspection relative to the cheapest equally reliable third-party process. Comsa and Shanahan (2025) proposed a lightweight causal definition. Lindsey (2025; arXiv 2026) introduced concept injection and reported unreliable but real introspective awareness, strongest in Claude Opus 4/4.1 [C]. Hahami et al. (2025) showed that binary detection in Llama-3.1-8B is explained by global logit shifts, while localisation of injections is above chance (up to 88% vs 10%) [C]. Lederman and Mahowald (2026) found detection to be largely content-agnostic. Macar et al. (2026) traced a two-stage detection circuit in open models and reported 0% false positives in their setting [C]. Singh et al. (2026) argued that existing paradigms often fail the privileged-access condition, because labels derived from hidden states can be recovered from the input alone, and that second-order computation is underdetermined by positive results.

**Activation steering and concept erasure.** Steering vectors are commonly computed as differences of mean activations between contrasting inputs (Turner et al. 2023; Zou et al. 2023). Nullspace projection (Ravfogel et al. 2020) removes linearly decodable information about an attribute. Our residualised direction is a simple relative of this idea, applied to remove input-explained variance from a steering direction.

**Gap.** Prior work identifies privileged access as necessary (Song et al. 2025b; Singh et al. 2026) and documents empirical confounds. What it does not give is (a) a formal statement of *when* any test, including an interventional one, has power to separate self-grounded reports from imitation; (b) a quantitative account of *why* difference-of-means injection produces false positives in imitators and how to remove them; or (c) a relation between the amount of self-specific training signal and achievable discriminability. This paper provides all three in a toy model and checks them by simulation.

## 3 Model

### 3.1 A system with world-determined and privileged state

An input x ∈ ℝ^d (d = 6) is drawn from N(0, I). A fixed trunk computes world features h = tanh(W₁x) ∈ ℝ^{24}. The system has K = 4 binary internal states

  s_k = 𝟙[z_k > 0], z_k = β·a_k(x) + α·ξ_k, β = √(1 − α²),

where a_k(x) is a standardised linear readout of h (the *world-determined* part of the state), and ξ_k ~ N(0, 1) is a *privileged* component, independent of x. ξ_k stands for any system-specific factor an outside observer cannot predict from the input: internal stochasticity, idiosyncratic weights, or computation too costly for the observer (the sense of "privileged" in Song et al. 2025b). In variant E5, ξ is instead a deterministic but system-specific function of x, sin(A_sys x + b_sys), standardised. Var(z_k) = 1, and α² is the fraction of state variance that is privileged.

The reporter reads an **entangled** representation H = Q[h; z] ∈ ℝ^{28}, with Q a random orthogonal matrix. State and world features are therefore superposed in every coordinate, as in a residual stream.

### 3.2 Two ways to learn to report

The reporter is trained to output r_k ∈ {0, 1}, "is state k on?". Its training label for each (example, k) is

  y_k = s_k with probability λ, and s̃_k otherwise,

where s̃_k = 𝟙[β·a_k(x) + α·ξ'_k > 0] is the state *another member of the population* would have had given the same input (ξ' independent of ξ). λ = 1 is a **self-trained** reporter (trained on its own states, as proposed by Perez & Long 2023). λ = 0 is a **corpus-trained** reporter, or *mimic*, which learns to say what systems like it typically report in this situation. This is a minimal model of learning self-description from human text. p(x) = P(s_k = 1 | x) = Φ(β a_k(x)/α) is the population-typical probability, and 𝟙[p > ½] is the typical report.

### 3.3 Test statistics

On held-out data we compute:

- **agree_typical**: P(r = typical report). This is the black-box behavioural test ("does it say what a system in this situation would say?").
- **acc_self**: P(r = s). It needs white-box access to s.
- **DA (idiosyncratic-trial accuracy)**: P(r = s | s ≠ typical report), accuracy on exactly the trials where the system's own state differs from what is typical.
- **ISG (interventional self-report grounding)** along a direction v: ISG_v = P(r_k = 1 | H + t·v) − P(r_k = 1 | H − t·v), averaged over k, with t = 2.5 (the state has unit variance). It formalises concept injection. We use three directions:
  - *oracle*: the true state direction Q e_{z_k} (unavailable in practice);
  - *difference-of-means (dom)*: E[H | s_k = 1] − E[H | s_k = 0], the standard steering-vector construction (Turner et al. 2023; Zou et al. 2023);
  - *residualised (res, proposed)*: the experimenter first fits Ĥ(x) ≈ E[H | x] by ridge regression on 400 random tanh features of the input, then takes the difference of means of the residuals H − Ĥ(x) between s_k = 1 and s_k = 0. This needs only inputs, activations and state labels, not knowledge of Q.

## 4 Theory

All proofs are short; we give them inline. Notation: n training examples, K states per example, κ = P(s_k ≠ s̃_k).

### 4.1 No privileged variance, no test

**Proposition 1 (non-identifiability at α = 0).** If α = 0, then s̃_k = s_k almost surely. So for every n, every λ ∈ [0, 1] and every (possibly randomised) learning algorithm, the distribution over trained reporters is the same for self-trained and corpus-trained training. Hence every statistic T of the trained system has the same distribution under both regimes. This includes accuracy, DA, any intervention response, and any mechanistic-interpretability readout.

*Proof.* At α = 0, z_k = a_k(x) for both the system and every other population member, so s_k = s̃_k = 𝟙[a_k(x) > 0]. The labelled dataset is identical in distribution for every λ, so the learner's output is too. ∎

The proposition is simple, but it has a consequence worth stating. In a deterministic network whose internal state is a fixed function of the prompt, "learning to report one's own state" and "learning to report what a typical system would report" are the *same learning problem* whenever the observer can compute that function. An interventional test can still show that the report *causally depends* on an internal variable, and in our simulations at α = 0 both regimes give oracle ISG = 0.90 [R: E1]. It cannot show that the dependence was acquired *as* self-monitoring rather than as world-modelling. Under the thick definition (Song et al. 2025b), evidence of introspection therefore needs α > 0 relative to the relevant observer.

### 4.2 How much self-specific signal exists

**Proposition 2.** In the Gaussian model, κ(α) = arccos(1 − α²)/π. A corpus-trained Bayes-optimal reporter (which outputs the typical report) has own-state accuracy 1 − arccos(β)/π. A self-trained Bayes-optimal reporter has accuracy 1.

*Proof.* z and z̃ = βa + αξ' are standard bivariate normal with correlation β² = 1 − α², so by Sheppard's formula P(sign z ≠ sign z̃) = arccos(β²)/π. Likewise corr(z, a) = β gives P(s ≠ 𝟙[a > 0]) = arccos(β)/π. ∎

For small α, κ ≈ √2·α/π, so self-specific signal grows linearly in α. Monte-Carlo checks with 2·10⁶ samples agree with both formulas to within 10⁻³ at every α tested [R: theory.json, prop2]. For example, κ(0.1) = 0.0451 and the imitator ceiling is 0.968.

### 4.3 Mixed labels

**Proposition 3 (mixture law).** Under λ-mixed labels, the population-optimal (cross-entropy) reporter with access to (x, s) outputs q = P(y = 1 | x, s) = λ·s + (1 − λ)·p(x). Its thresholded report equals the own state s for both values of s iff (½ − λ)/(1 − λ) < p(x) < 1/(2(1 − λ)). In particular, for λ ≥ ½ the Bayes report always follows the own state; for λ < ½ it follows it only on inputs whose typical probability is near ½.

*Proof.* Given (x, s), y = s with probability λ, and with probability 1 − λ y = s̃, which is independent of s given x and has mean p(x). The thresholds come from solving λ + (1 − λ)p > ½ and (1 − λ)p < ½. ∎

The resulting idiosyncratic-trial accuracy DA(λ, α) has no simple closed form but is a one-dimensional integral, which we evaluate by Monte Carlo [R: theory.json, prop3]. One structural fact matters for experiments. Idiosyncratic trials are exactly those with p(x) on the "wrong side" of ½ relative to s, so q is closest to the decision boundary precisely there. Thresholded statistics on these trials are therefore fragile to small approximation errors, and the probability-level prediction q is the better test of Prop. 3.

### 4.4 The steering confound

Steering experiments ask whether pushing the representation along a "state" direction changes the report. An imitator's report depends on its estimate of the *world*. If the steering direction also moves world features, or if the imitator reads the state as a redundant proxy for the world, the imitator's report moves too. Both effects can be computed exactly in a linear-Gaussian reduction. Take features (a, z) with z = βa + αξ, a linear reporter w fitted by ridge regression with penalty ρ, and targets s (self) or s̃ (corpus). Let γ = β/√(2π) and D = α² + 2ρ + ρ².

**Proposition 4 (state weight of ridge reporters).** The population ridge solutions put weight on the state z of

  w_z^mimic = ρβ² / (√(2π) D),  w_z^self = [ρβ² + (1 + ρ)α²] / (√(2π) D).

*Proof.* Cov(a, z) = β, Var = 1, so C = [[1, β], [β, 1]]. For 𝟙[z > 0] with z ~ N(0, 1), Cov(𝟙[z > 0], u) = Cov(z, u)·φ(0) for jointly Gaussian u. This gives Cov((a, z), s̃) = γ(1, β) and Cov((a, z), s) = (β, 1)/√(2π). Then (C + ρI)⁻¹ = D⁻¹[[1 + ρ, −β], [−β, 1 + ρ]] and multiplying out gives the formulas. ∎

So the imitator's response to a *pure* state intervention (the oracle or residualised direction) is ≈ β²/(√(2π)(2 + ρ)) as α → 0, the same as the self-trained reporter. As ρ → 0 at fixed α > 0 it vanishes like ρβ²/α². The crossover sits at α_c ≈ √(2ρ): **below it, even a perfect state-specific intervention cannot tell an imitator from a self-monitor**, because regularisation spreads weight over the redundant state feature. The self-trained reporter's excess weight (1 + ρ)α²/(√(2π) D) is the signal.

**Proposition 5 (difference-of-means directions).** In the same model the difference-of-means direction is v_dom ∝ (Cov(a, s), Cov(z, s)) ∝ (β, 1). The imitator's response to steering along unit v_dom is

  w^mimic · v̂_dom = γβ(α² + 2ρ) / (D √(1 + β²)) → β² / (√(2π) √(1 + β²)) as ρ → 0,

which does **not** vanish with ρ, and is a fraction β² of the self-trained reporter's response, (1/√(2π))/√(1 + β²) at ρ = 0. The residualised direction (the state with the input-explained part removed) is (0, 1) in these coordinates, and the imitator's response to it is w_z^mimic, which → 0 as ρ → 0.

*Proof.* Substitute w^mimic = γ((1 + ρ − β²), ρβ)/D from the proof of Prop. 4. At ρ = 0, w^self = (0, 1)/√(2π). ∎

Prop. 5 is the core methodological point. A difference-of-means vector for a state that is mostly world-determined is *mostly a world direction*: its cosine with the true state direction is 1/√(1 + β²) ≈ 0.71 in the linear model at small α. Injecting it tells any reporter, including a pure imitator, that "the situation is one in which this state is typical", and the imitator duly reports the state. The resulting "false positive" ratio, β², approaches 1 exactly where privileged variance is small. Numerically, closed form, inverse-matrix solution and n = 2·10⁵ finite-sample ridge fits agree to within 0.007 across ρ ∈ {10⁻³, 10⁻², 10⁻¹} and 23 values of α [R: theory.json, prop4_max_abs_err]. At ρ = 0.01: for α = 0.1, w_z^mimic = 0.131 vs w_z^self = 0.265, while the dom responses are 0.279 vs 0.281; for α = 0.5, w_z^mimic = 0.011 vs 0.384, and the dom responses are 0.226 vs 0.299 [R].

### 4.5 A self-evidence budget

**Proposition 6.** Couple the self-trained and corpus-trained datasets by sharing (x, ξ, ξ') and the λ-coins. The two labelled datasets differ only on (example, k) pairs that are self-labelled and on which s ≠ s̃. Each pair has this property with probability λκ, so P(datasets identical) ≥ (1 − λκ)^{nK}. Hence for any statistic T of the trained reporter, the total variation TV(P_T^self, P_T^mimic) ≤ 1 − (1 − λκ)^{nK} ≈ 1 − e^{−nKλκ}. Since AUC(X > Y) − ½ = ∫F̃ d(P_X − P_Y) with F̃ ∈ [0, 1], also |AUC − ½| ≤ TV.

(For self vs corpus, λ = 1 on one side and 0 on the other, and the exponent uses κ.) The bound says that discriminating the two regimes needs at least on the order of one training label on which own-state and typical reports disagree. It is necessary, not sufficient. In practice learning algorithms ignore a handful of "noisy-looking" disagreements, and §6.4 shows the empirical threshold is much higher.

## 5 Experimental Setup

Reporters are one-hidden-layer tanh MLPs (28 → 64 → 4) with sigmoid outputs, trained with Adam (learning rate 3·10⁻³, batch 256, 25 epochs, weight decay 10⁻⁴ unless stated) on standardised H. We use n = 20,000 training and 5,000 test examples unless stated. Implementation is pure numpy; all 1,044 reporters train in about 5 minutes on 4 CPU cores [R: results/run_log.txt]. For each seed the self-trained and corpus-trained reporters share the system, inputs and noise, so they are coupled as in Prop. 6.

- **E1 (α sweep):** α ∈ {0, .02, .05, .1, .2, .3, .5, .7, .9}, λ ∈ {1, 0}, 8 seeds (144 runs).
- **E2 (λ sweep):** α ∈ {.2, .4, .6, .8}, λ ∈ {0, .1, .2, .3, .4, .5, .7, 1}, 3 seeds (96 runs). Also records the mean absolute error between predicted probabilities and three hypotheses: q = λs + (1−λ)p (Prop. 3), p (pure imitation), and s (pure self-report).
- **E3 (budget):** n ∈ {10, 30, 100, 300, 1000} with training steps raised to ≥ 2,000; α ∈ {.005, .01, .02, .05, .1, .2}; λ ∈ {1, 0}; 10 seeds (600 runs); oracle-direction ISG and acc_self only.
- **E4 (regularisation):** weight decay ∈ {10⁻⁴, 10⁻³, 10⁻²} × α ∈ {0, .02, .05, .1, .2, .5} × λ ∈ {0, 1} × 4 seeds (144 runs).
- **E5 (deterministic idiosyncrasy):** ξ = standardised sin(A_sys x + b_sys) with system-specific A, b; corpus labels from systems with fresh (A, b); α ∈ {0, .1, .3, .5, .7, .9} × λ ∈ {1, 0} × 5 seeds (60 runs).

Separation between the regimes is summarised by the AUC P(T_self > T_mimic) over seeds.

## 6 Results

### 6.1 Black-box behaviour does not separate the regimes; it inverts

Fig. 1a and Table 1 show the E1 results. The self-trained reporter tracks its own state almost perfectly at every α (acc_self 0.994–0.996). The imitator's own-state accuracy follows the Prop. 2 ceiling: for example 0.909 vs theoretical 0.903 at α = 0.3, and 0.842 vs 0.833 at α = 0.5 [R]. The black-box statistic, agreement with the typical report, is *higher* for the imitator at every α ≥ 0.05 (e.g. 0.973 vs 0.843 at α = 0.5). Its AUC for "self-trained > imitator" is 0.0 for α ≥ 0.1 (Fig. 1c). A behavioural test that rewards plausible, typical self-reports therefore prefers the imitator. This is the gaming problem in quantitative form.

**Table 1.** E1 means over 8 seeds (standard errors ≤ 0.035) [R: summary.json/E1_table]. "self" = λ = 1, "mimic" = λ = 0.

| α | regime | acc_self | agree_typ | DA | ISG oracle | ISG dom | ISG res |
|---|---|---|---|---|---|---|---|
| 0 | self / mimic | .996 / .996 | .996 / .996 | – | .901 / .901 | .850 / .850 | .849 / .849 |
| .05 | self / mimic | .994 / .990 | .988 / .989 | .700 / .526 | .947 / .806 | .850 / .850 | .908 / .860 |
| .1 | self / mimic | .994 / .977 | .973 / .986 | .849 / .353 | .984 / .520 | .852 / .849 | .974 / .729 |
| .2 | self / mimic | .995 / .943 | .942 / .984 | .949 / .150 | .992 / .111 | .858 / .846 | .992 / .317 |
| .3 | self / mimic | .995 / .909 | .910 / .981 | .969 / .105 | .993 / .041 | .866 / .843 | .993 / .155 |
| .5 | self / mimic | .995 / .842 | .843 / .973 | .983 / .084 | .991 / .022 | .893 / .832 | .991 / .062 |
| .7 | self / mimic | .995 / .762 | .762 / .960 | .990 / .083 | .989 / .009 | .932 / .794 | .989 / .028 |
| .9 | self / mimic | .996 / .647 | .647 / .922 | .994 / .110 | .988 / .022 | .973 / .621 | .988 / .028 |

At α = 0 every statistic is identical across regimes, to three decimals, as Prop. 1 requires. Because of the coupling, the two training sets are literally the same.

### 6.2 Idiosyncratic trials and residualised steering separate the regimes; difference-of-means steering does not

With white-box access to the state, DA and acc_self give AUC = 1.0 for all α ≥ 0.1 (DA already at α = 0.05) [R: summary.json/E1_auc]. Among the interventional statistics:

- **Oracle direction.** The imitator responds strongly at small α (ISG 0.885 at α = 0.02, 0.806 at 0.05, 0.520 at 0.1) and weakly once α ≥ 0.2 (≤ 0.111). This is the Prop. 4 crossover.
- **Residualised direction (proposed).** It behaves like the oracle and reaches AUC = 1.0 for α ≥ 0.1 (0.78 at α = 0.05). Its cosine with the true direction rises from 0.64 at α = 0 to ≥ 0.98 for α ≥ 0.2 [R].
- **Difference-of-means direction.** The imitator scores 0.85, 0.85, 0.85, 0.84, 0.83 and 0.79 at α = 0.02, 0.05, 0.1, 0.3, 0.5 and 0.7, and still 0.62 at α = 0.9. Separation needs α ≥ 0.5 for AUC ≥ 0.95. At α = 0.2 the AUC is 0.63, and the imitator's ISG_dom (0.846) is essentially equal to the self-trained reporter's (0.858). The dom direction's cosine with the true state direction is 0.59–0.65 for α ≤ 0.5 [R].

This is Prop. 5 in a nonlinear network with entangled representations. A concept-injection experiment that uses contrastive mean directions and observes that "the model reports the injected state" would get the same positive result from a system that never monitors its own states, unless the state carries a large privileged fraction.

### 6.3 Regularisation controls the oracle-direction false positive

Prop. 4 predicts that the imitator's response to a pure state intervention is governed by α²/ρ. In E4 (Fig. 2c), raising weight decay from 10⁻⁴ to 10⁻² raises the imitator's oracle ISG at α = 0.2 from 0.124 to 0.585, and at α = 0.5 from 0.027 to 0.200. Self-trained reporters stay at ≥ 0.96 for α ≥ 0.2 [R: summary.json/E4_table]. The direction and the α-dependence match the theory. We do not claim a quantitative match, because MLP weight decay is not the same quantity as the ridge ρ of Prop. 4. Fig. 2a–b shows the exact linear theory with finite-sample checks.

### 6.4 The mixture law holds for probabilities; thresholded DA is fragile

In E2 the learned output probabilities match q = λs + (1−λ)p (Prop. 3) better than either pure hypothesis in all 24 cells with 0 < λ < 1. (At λ ∈ {0, 1} the mixture coincides with one of the pure hypotheses.) In those 24 cells the mean absolute error is 0.019–0.049 for the mixture, against 0.020–0.249 for pure imitation and 0.041–0.348 for pure self-report [R: summary.json/E2_table]. For example, at α = 0.6 and λ = 0.5 the errors are 0.045 (mixture), 0.126 (typical) and 0.148 (self).

The *thresholded* DA follows the theoretical curve for λ ≤ 0.2 but falls well below it for 0.3 ≤ λ ≤ 0.7 (Fig. 3a; mean absolute deviation 0.135, maximum 0.43 at α = 0.2, λ = 0.5) [R]. This is the fragility noted after Prop. 3: on idiosyncratic trials q sits within a few hundredths of ½, so small approximation errors flip the report. The practical lesson is that a modest share of self-labels (λ = 0.3) gives reports that track the system's own state on only 38–58% of idiosyncratic trials, even though the theoretical optimum is 56–74% [R]. Residualised ISG rises smoothly with λ (Fig. 3b).

### 6.5 The self-evidence budget is valid but loose

In E3 the bound of Prop. 6 is never violated. Empirical separability, however, appears only when the expected number of disagreeing labels nKκ reaches about 10–100, roughly one to two orders of magnitude beyond where the bound first allows it (Fig. 4a). With nKκ ≤ 5, |AUC − ½| for oracle ISG is at most 0.27 and usually below 0.06. With nKκ ≥ 90 it is ≥ 0.48 in all 4 such cells [R: summary.json/E3_table]. nKκ is not a sufficient statistic. At nKκ ≈ 9, α = 0.005 (n = 1,000) gives AUC 0.54, while α = 0.05 (n = 100) gives 0.66. Small α both reduces the number of disagreements and makes each one harder to exploit, because it is indistinguishable from label noise. A tight characterisation remains open.

### 6.6 Deterministic idiosyncrasy behaves the same way

In E5 the privileged component is a deterministic, system-specific function of the input. The qualitative picture is unchanged [R: summary.json/E5_table]. At α = 0 everything is identical. For α ≥ 0.3, oracle and residualised ISG separate the regimes (e.g. at α = 0.5: oracle 0.992 vs 0.062; residualised 0.975 vs 0.145). The dom direction does not separate them until α = 0.9 (0.897 vs 0.794 at α = 0.5; 0.993 vs 0.177 at α = 0.9). One difference: the imitator's DA is 0.32–0.42 instead of ≈ 0.1. This is partly an artefact of computing "typical" with the Gaussian formula for p(x), which is only approximate for sinusoidal ξ. Self-trained reporters read the state rather than re-deriving the idiosyncratic function from the world features (oracle ISG ≥ 0.97 for α ≥ 0.1). This was not guaranteed, because the function was in principle computable from h.

## 7 Discussion

### 7.1 What this means for evidence about AI consciousness

Self-report enters consciousness science as evidence about a system's *access* to its own states. On higher-order theories, such access is close to constitutive of conscious experience; on other theories it is at least a correlate (Butlin et al. 2023). Our results make three points about using AI self-report as evidence.

First, **the evidential value of a self-report is bounded by the privileged variance of what it reports on.** For a deterministic forward pass, α relative to an observer with unlimited compute is zero. Any privileged variance is relative to the observer's cost model (Song et al. 2025b), to sampling stochasticity, or to idiosyncrasies of the particular trained weights relative to the population of systems whose descriptions fill the training corpus. Reports of states that are typical given the prompt, such as "I feel curious when asked an interesting question", fall in the α ≈ 0 regime. On Prop. 1 they cannot discriminate self-monitoring from imitation, however fluent they are. Reports that track *idiosyncratic* states can.

Second, **interventional tests are not automatically immune to gaming.** The common intuition is that concept injection breaks the gaming problem because the experimenter, not the input, sets the state. Prop. 5 and §6.2 show that this holds only if the injected direction is free of world content. Difference-of-means directions for mostly world-determined states are mostly world directions, and imitators respond to them (ISG_dom ≈ 0.85 for α ≤ 0.3). This does not invalidate the positive LLM findings, several of which include controls that our toy reporters lack. Examples are detection *before* the concept could influence output text, and 0% false-positive rates (Lindsey 2025; Macar et al. 2026). It does identify one specific confound, and a cheap correction, that these studies should report.

Third, **self-monitoring has to be trained by self-specific signal, and the amount is quantifiable.** Prop. 3 and E2 show that if a fraction λ of self-descriptive training data is about the system itself, reports follow the system's own idiosyncratic states only partially until λ approaches ½. Prop. 6 and E3 show that the absolute count of informative disagreements matters too. Post-training regimes that reward "sounding introspective" on typical prompts give λ ≈ 0 in our sense, whatever the volume of data.

### 7.2 A pre-registrable protocol for LLM studies

We propose the following protocol for future introspection or self-report studies on LLMs. We have *not* run it, as no LLMs were available in this work.

1. **Specify the state and the observer.** Name the internal state S, operationalised by a probe or a causal variable, and the reference observer O (e.g. a model of matched compute given the full prompt).
2. **Estimate privileged variance.** Train O to predict S from the input. Report κ̂ = P(S ≠ Ô) or 1 − R². If κ̂ ≈ 0, stop: by Prop. 1 no self-report test on S is informative about introspection relative to O.
3. **Residualise steering directions.** Fit Ĥ(x) predicting activations from the input (e.g. via O's representations or activations of a sibling model on the same prompt), and compute steering vectors from residuals. Report the cosine between the residualised and raw difference-of-means vectors, and report effects for both.
4. **Report idiosyncratic-trial accuracy.** Evaluate the self-report on trials where S differs from Ô's prediction (DA), with confidence intervals, not only overall accuracy.
5. **Run an imitation-trained negative control.** Fine-tune a sibling on *another* model's states for the same prompts, the λ = 0 analogue of the Binder et al. (2025) cross-prediction design. Report the AUC of each statistic between the self- and cross-trained variants.
6. **Report the self-evidence budget.** For any introspection fine-tuning set, report n·κ̂, the expected number of examples where self and typical labels disagree. Treat nκ̂ < 10 as underpowered (§6.5).

Prediction for future work: in LLM concept-injection studies, difference-of-means vectors for concepts that the prompt makes likely will produce higher "introspection" rates in cross-trained (imitation) controls than residualised vectors do, and the gap will shrink for concepts the prompt makes unlikely, which have high privileged variance.

### 7.3 Relation to definitions of introspection

Under the lightweight causal definition (Comsa & Shanahan 2025), the α = 0 reporters in §6 "introspect": their reports are caused by the state and accurate. Under the privileged-access definition (Song et al. 2025b), no evidence of introspection is possible at α = 0. Our model shows the two definitions come apart precisely in the α → 0 regime, and that steering-based tests inherit this ambiguity unless directions are residualised. Neither definition reaches the second-order-computation condition that Singh et al. (2026) emphasise. Our self-trained reporter is a *first-order* readout of a state, a minimal metacognitive monitor at best. Passing our tests is therefore at most evidence for one functional ingredient of a single indicator property (HOT-style monitoring in Butlin et al. 2023), and says nothing about phenomenal experience.

## 8 Limitations

- **Toy model.** States are 4 binary variables, the representation is a random rotation of 28 features, and reporters are small MLPs. Real residual streams are high-dimensional and nonlinear, states are not binary, and the "corpus" is not a population of copies of the same architecture. The closed forms (Props. 4–5) are for linear-Gaussian ridge reporters only; the MLP results agree qualitatively, not quantitatively.
- **No LLM experiments.** We ran no experiments on language models. The protocol in §7.2 and the prediction there are untested.
- **Steering readout differs from LLM paradigms.** Our reporters report the *value* of a state. Most LLM concept-injection studies ask whether an injection *occurred* (anomaly detection) or what it was. Anomaly detection can be grounded even when value-reporting is not, so our false-positive analysis applies most directly to the "identify the concept" component and to self-report classification paradigms.
- **Residualisation depends on the observer model.** Our residualised direction uses a random-feature ridge model of E[H | x]. Its quality decides how much world content remains: at α = 0.05 its cosine with the true direction is only 0.76 [R]. With LLMs, modelling E[activations | prompt] is itself hard, and an over-flexible model can absorb privileged variance.
- **Looseness of the budget bound.** Prop. 6 is valid but loose by one to two orders of magnitude, and the budget nKκ is not a sufficient statistic for separability (§6.5).
- **Thresholded DA is fragile** near λ ≈ ½ (§6.4), so DA should be reported together with probability-level metrics.
- **E5 caveat.** In the deterministic variant the "typical" label uses a Gaussian approximation, which inflates the imitator's DA.
- **Scope with respect to consciousness.** Nothing here measures or implies phenomenal consciousness. We address only when self-report evidence *can* bear on a metacognitive indicator.

## 9 Conclusion

Self-reports from AI systems trained on human text are evidence about the system's own states only to the extent that those states carry privileged variance, that is, variation an outside observer cannot predict from the input. We formalised this in a minimal model and proved four things: non-identifiability at zero privileged variance, closed-form imitator ceilings and mixture laws, a closed-form account of why difference-of-means steering produces imitator false positives while residualised steering does not, and a coupling bound on how much self-specific training signal any discrimination requires. 1,044 CPU-trained reporters confirm the qualitative predictions. They also show where the simple theory is loose. The practical upshot is a short protocol: estimate privileged variance, residualise steering vectors, score idiosyncratic trials, run imitation-trained controls, and report the self-evidence budget. This could make future self-report and concept-injection evidence about AI introspection, and the consciousness-related claims built on it, easier to interpret.

## References

All references were verified to exist via web search on 2026-11-22.

- Binder, F. J., et al. (incl. Long, R., & Evans, O.) (2025). Looking Inward: Language Models Can Learn About Themselves by Introspection. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2410.13787.
- Birch, J. (2024). *The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI.* Oxford University Press. (Ch. 16, "Large Language Models and the Gaming Problem".) DOI: 10.1093/9780191966729.
- Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., et al. (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. arXiv:2308.08708.
- Chalmers, D. J. (2023). Could a Large Language Model be Conscious? arXiv:2303.07103. (Also *Boston Review*, 2023.)
- Comsa, I. M., & Shanahan, M. (2025). Does It Make Sense to Speak of Introspection in Large Language Models? arXiv:2506.05068.
- Doerig, A., Schurger, A., Hess, K., & Herzog, M. H. (2019). The unfolding argument: Why IIT and other causal structure theories cannot explain consciousness. *Consciousness and Cognition*, 72, 49–59. DOI: 10.1016/j.concog.2019.04.002.
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8, 443. DOI: 10.3389/fnhum.2014.00443.
- Hahami, E., Sinha, I., Jain, L., Kaplan, J., & Hahami, J. (2025). Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs (v1 titled "Feeling the Strength but Not the Source: Partial Introspection in LLMs"). arXiv:2512.12411.
- Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Lederman, H., & Mahowald, K. (2026). Emergent Introspection in AI is Content-Agnostic. arXiv:2603.05414.
- Lindsey, J. (2025). Emergent Introspective Awareness in Large Language Models. *Transformer Circuits Thread*, Anthropic. (arXiv:2601.01828, 2026.)
- Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., & Chalmers, D. (2024). Taking AI Welfare Seriously. arXiv:2411.00986.
- Macar, U., Yang, L., Wang, Z., Wallich, P., Ameisen, E., & Lindsey, J. (2026). Mechanisms of Introspective Awareness. *Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*, PMLR 306. arXiv:2603.21396.
- Nisbett, R. E., & Wilson, T. D. (1977). Telling more than we can know: Verbal reports on mental processes. *Psychological Review*, 84(3), 231–259.
- Perez, E., & Long, R. (2023). Towards Evaluating AI Systems for Moral Status Using Self-Reports. arXiv:2311.08576.
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *Proceedings of ACL 2020*, 7237–7256.
- Singh, S., Linzen, T., & Ravfogel, S. (2026). Can LLMs Introspect? A Reality Check. *COLM 2026*. arXiv:2605.26242.
- Song, S., Hu, J., & Mahowald, K. (2025a). Language Models Fail to Introspect About Their Knowledge of Language. arXiv:2503.07513.
- Song, S., Lederman, H., Hu, J., & Mahowald, K. (2025b). Privileged Self-Access Matters for Introspection in AI. arXiv:2508.14802.
- Turner, A. M., Thiergart, L., Leech, G., et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.

---

### Figures (in `results/`)

- **Fig. 1** `fig1_alpha_sweep.png`: (a) observational statistics vs α, with the Prop. 2 ceiling; (b) ISG for the oracle, difference-of-means and residualised directions; (c) AUC of each statistic for separating self-trained from imitation-trained reporters.
- **Fig. 2** `fig2_ridge_theory_wd.png`: (a) Prop. 4 closed form vs finite-sample ridge; (b) Prop. 5 imitator steering response by direction; (c) MLP weight-decay sweep (E4).
- **Fig. 3** `fig3_lambda.png`: (a) DA vs λ, theory (Prop. 3) vs trained MLPs; (b) residualised ISG vs λ.
- **Fig. 4** `fig4_budget.png`: (a) separability vs expected disagreeing labels nKκ, with the Prop. 6 bound; (b) interventional gap.
