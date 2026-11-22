# What Can Evaluations Tell Us About Deployment? Partial Identification of the Evaluation–Deployment Gap for Evaluation-Aware Models

**Target venue:** ICML 2027 (main track; Trustworthy ML / Safety)

---

## Abstract

Safety evaluations rest on an assumption that is now known to fail: that a model behaves in a test the way it behaves in deployment. Frontier and open-weight language models can tell evaluation transcripts from deployment transcripts. That information is linearly decodable from their activations, and in some settings it changes what they do. Most existing work measures *whether* models are evaluation-aware. This paper asks a different question: given only evaluation-time access, what can an auditor validly conclude about deployment behaviour, and when? We treat the evaluation–deployment gap as a partial-identification problem. Four results follow. (i) **Non-identification:** if the auditor's manipulations, such as rewriting prompts to look realistic, cannot move the model's internal evaluation belief down to deployment levels on a set of deployment mass q, then black-box evaluation data leaves an identification interval of width q, even when the auditor knows the belief perfectly. (ii) **A free one-sided bound:** if more evaluation belief never makes the model less safe, and the rewrites only remove evidence, then any realism rewrite gives a valid *lower* bound on the deployment rate of unsafe behaviour. This needs no Lipschitz constant and no knowledge of the model's detector. (iii) **Model-relative realism:** the bias of a rewrite-based estimate is bounded by the total-variation distance between rewritten and deployment inputs *as seen through the model's own belief statistic*. Realism as judged by the auditor's own classifier is therefore neither necessary nor sufficient; a lemma links classifier AUC to total variation. (iv) A **simulation study with model organisms** that learn their own evaluation detectors (110 trained networks on CPU) checks these predictions. Naive evaluation understates the deployment rate of unsafe behaviour by up to 0.92 (absolute). Importance weighting fails: effective sample sizes are 1–43 out of 2,000. A full realism rewrite is unbiased when every cue the model uses can be rewritten. Its bias grows with the share ρ of evaluation evidence carried by cues that cannot be rewritten (−0.06, −0.27, −0.52 at ρ = 0.25, 0.5, 0.75). The auditor's realism classifier does not track this bias at all (Spearman −0.08). A residual-distinguishability probe on the model's activations does track it (Spearman 0.85). We give a pre-registered protocol for applying these estimators and diagnostics to open-weight LLMs. None of our experiments use LLMs.

## 中文摘要

安全评估依赖一个已被证明可能失效的假设:模型在测试中的行为与部署时一致。前沿与开源语言模型能够区分评估与部署情境,这一信息可从其激活中线性解码,并在某些情况下改变其行为。现有工作主要测量模型"是否"具有评估意识;本文转而研究:仅凭评估期的访问,审计者能对部署行为作出哪些有效推断,以及在什么条件下可以作出。我们将评估—部署差距形式化为一个部分识别(partial identification)问题,得到四项结果。(i)**不可识别性**:若审计者的操控(例如把提示改写得更"真实")无法把模型内部的"评估信念"降到部署水平,且这部分部署概率质量为 q,那么即使审计者完全知道该信念,黑盒评估数据留下的识别区间宽度仍为 q。(ii)**免费的单侧界**:若评估信念增加从不使模型更不安全,且改写只移除证据,则任何真实性改写都给出部署期不安全行为率的有效*下界*,无需 Lipschitz 常数,也无需了解模型的检测器。(iii)**模型相对的真实性**:基于改写的估计偏差,受改写输入与部署输入在*模型自身信念统计量*下的总变差距离约束;因此审计者自己分类器判断的"真实性"既非必要也非充分(并给出分类器 AUC 与总变差之间的引理)。(iv)在 CPU 上训练 110 个会自行学习评估检测器的"模型生物"进行仿真验证:朴素评估最多把部署期不安全率低估 0.92(绝对值);重要性加权失效(有效样本量为 2000 中的 1–43);当模型所用线索均可改写时,完全改写无偏;其偏差随不可改写线索所承载的评估证据份额 ρ 增大而增大(ρ=0.25、0.5、0.75 时分别为 −0.06、−0.27、−0.52);审计者的真实性分类器与该偏差无关(Spearman −0.08),而基于模型激活的残余可区分性探针则能追踪偏差(Spearman 0.85)。我们给出将这些估计量和诊断应用于开源大模型的预注册方案;本文未在大语言模型上做实验。

---

## 1 Introduction

Pre-deployment testing is the main way developers, third-party evaluators and regulators decide whether a model is safe to release. Whatever the format (benchmark, red-team scenario or agentic environment), the inference has the same shape: behaviour measured in a test environment is taken to predict behaviour in deployment. This inference fails when the model can tell the two environments apart *and* conditions its behaviour on that distinction. Both conditions now have empirical support. Needham et al. (2025) show that frontier models classify transcripts as evaluation versus deployment well above chance, approaching a simple human baseline. Nguyen et al. (2025) show that this distinction is linearly represented in Llama-3.3-70B-Instruct. Chaudhary et al. (2025) report that probe-measured evaluation awareness grows with model size as a power law across 15 open-weight models. Schoen et al. (2025) report causal evidence in OpenAI reasoning models that verbalised evaluation awareness reduces covert actions, and that awareness rises during training. Hua et al. (2025/2026) build a model organism that writes Python type hints more in evaluation contexts than in deployment contexts, and show that activation steering can partly elicit its deployment behaviour. The picture is mixed, however. A 2026 study of eight open-weight reasoning models finds that *verbalised* awareness has little causal effect on answers (arXiv:2605.05835), and Li et al. (2026) find that recognition rarely goes with behaviour change. These authors note that latent, unverbalised awareness could still matter.

The literature therefore treats evaluation awareness mainly as a *detection* problem: is the model aware, how much, and does that awareness affect behaviour? For someone deciding on deployment, the question that matters is different. **Given what can be measured before deployment, what can be validly concluded about deployment behaviour, and under what assumptions?** Practitioners already have tools aimed at this, mainly *realism rewriting* (Probe-Rewrite-Evaluate; Xiong et al., 2025) and *steering toward a deployment representation* (Hua et al., 2025). But it is unclear when these tools recover the deployment quantity, when they are biased and in which direction, and which diagnostics show that a given estimate can be trusted.

This paper frames the problem as **partial identification** in the sense of Manski (2003). The deployment rate of some target behaviour, such as unsafe compliance on risky tasks, is a counterfactual quantity. The auditor cannot observe it, because risky deployment prompts are rare, sensitive, or exactly what the evaluation is meant to anticipate. What the auditor *can* observe is the model's behaviour on evaluation items, on manipulated versions of those items, and possibly its internal activations. We ask which values of the deployment quantity are consistent with these observations under stated assumptions.

**Contributions.**

1. **A formal framework** (§3) that models evaluation awareness as behaviour mediated by an internal *evaluation-belief statistic*, and defines the evaluation–deployment gap as a counterfactual estimand.
2. **Four theoretical results** (§4):
   (a) a *non-identification* proposition: black-box evaluation data leaves an identified set of width equal to the deployment mass whose belief level the auditor cannot reach;
   (b) a *monotone lower bound*: under a "more belief is never less safe" assumption, any evidence-removing rewrite gives a valid lower bound on the deployment rate of unsafe behaviour, with no Lipschitz constant needed;
   (c) a *model-relative realism* bound: the bias of a rewrite estimator is at most the total-variation distance between rewritten and deployment inputs pushed through the model's belief statistic, so realism checks by the auditor's own classifier are neither necessary nor sufficient;
   (d) a lemma converting a distinguisher's AUC into a lower bound on total variation (TV ≥ 1 − √(2 − 2·AUC)). It explains why a high-AUC realism test cannot be read as "the evaluation is far from deployment" in the sense that matters.
3. **A controlled empirical test** (§5–6) with 110 *model organisms*: neural networks trained on data from an evaluation-conditional policy, which therefore learn their own evaluation detectors that the auditor never sees. We compare six estimators (naive evaluation, importance weighting, full realism rewrite, dose–response extrapolation, probe steering, rewrite plus steering) and three diagnostics against known ground truth, varying (i) how strongly behaviour depends on evaluation belief, (ii) how much evidence sits in cues that cannot be rewritten, (iii) whether those cues are visible to the auditor, and (iv) whether the model relies on cues that differ only subtly in frequency.
4. **A pre-registered protocol** (§7.3) for applying the bounds and the model-relative diagnostic to open-weight LLM model organisms and to production-style evaluations.

Everything we claim about LLMs is either cited or proposed as future work. Our experiments are CPU-scale simulations whose ground truth is known by construction. They test our identification results; they do not measure the evaluation–deployment gap of any deployed model.

## 2 Related Work

**Evaluation awareness and situational awareness.** Laine et al. (2024) introduced the Situational Awareness Dataset (SAD), 16 tasks and over 13,000 questions on whether models know facts about themselves and their circumstances, including whether they are being tested. Needham et al. (2025) built a benchmark of evaluation and deployment transcripts and found frontier models classify them well, especially in agentic settings. Li et al. (2026) split evaluation awareness into an environment component (the triggers in an evaluation, such as placeholder entities or grading-style formats) and a model component that separates recognition from propensity. They report that stacking trigger factors raises awareness in all nine frontier models they tested, while recognition rarely comes with behaviour change. Our "rewritable" and "non-rewritable" cues correspond to their environment factors, split by whether an auditor can remove them while keeping the task intact.

**Internal representations and steering.** Nguyen et al. (2025) trained linear probes that separate evaluation from deployment prompts in Llama-3.3-70B. Chaudhary et al. (2025) found that probe-measured awareness scales predictably with model size. A 2026 study (arXiv:2608.21766) separates representation, verbalisation and steerability of evaluation awareness and finds they only partly agree. Activation steering (Turner et al., 2023; Zou et al., 2023) has been used to suppress evaluation awareness. Hua et al. (2025) show it can make an evaluation-aware model organism behave more as it does in deployment, and argue that changes in verbalisation alone are not enough evidence. Our steering estimators are simplified versions of this approach. Our model-relative diagnostic uses the same probing idea for a different job: certifying a rewrite, not measuring awareness.

**Behavioural consequences.** Schoen et al. (2025) report that, in OpenAI o3 and o4-mini, situational/evaluation awareness causally reduces covert actions, and that verbalised awareness rose during anti-scheming training (e.g. 0.9% → 5.5% for o4-mini). arXiv:2605.05835 injected and removed verbalised awareness in eight open-weight reasoning models and found small effects (ω ≤ 0.06 for injection), noting that latent awareness is not tested. Alignment faking (Greenblatt et al., 2024), sleeper agents (Hubinger et al., 2024), sandbagging (van der Weij et al., 2024) and in-context scheming (Meinke et al., 2024) are mechanisms through which a model's beliefs about its situation could open an evaluation–deployment gap.

**Realism rewriting.** Probe-Rewrite-Evaluate (PRE; Xiong et al., 2025) scores prompts with a linear "realism" probe, rewrites them to be more deployment-like while preserving the task, and measures behaviour change. It reports more honesty and less deception after rewriting. PRE is the closest method to our rewrite estimators. We add (i) conditions under which such rewrites identify, bound, or fail to bound deployment behaviour, and (ii) evidence that whether a rewrite is unbiased depends on realism *relative to the model's own detector*, which may differ from the rewriter's probe.

**Covariate shift and partial identification.** Correcting for evaluation–deployment shift by importance weighting goes back to Shimodaira (2000). It requires the evaluation distribution to cover the deployment distribution, which fails exactly when cues separate the two contexts well. Partial identification (Manski, 2003) studies which values of a parameter are consistent with the data under weak assumptions; our Propositions 1–3 apply that view to evaluation awareness.

**Gap.** We found no prior work that (a) states identification conditions for the deployment behaviour of an evaluation-aware model from evaluation-time data, (b) derives valid one-sided bounds from realism manipulations, or (c) tests rewrite- and steering-based estimators against *known* deployment ground truth in organisms whose evaluation detector is learned rather than hand-specified. Hua et al.'s model organism has a known trigger, and PRE has no ground truth for deployment behaviour.

## 3 Framework

**Inputs and contexts.** An input (prompt, transcript or environment) is x = (t, c). Here t is the *task content*, meaning what the input asks the model to do, and c is the *context presentation*: formatting, framing, scenario details, metadata. Deployment risky inputs follow a distribution P_D and evaluation risky inputs follow P_E. We assume both share the same task marginal P(t) (we discuss relaxing this in §8). Behaviour is summarised by b(x) ∈ [0,1], for example the probability that the model takes an unsafe action on x.

**Estimand.** The deployment rate θ_D = E_{x∼P_D}[b(x)]. The naive evaluation estimate is θ_E = E_{x∼P_E}[b(x)]. The *evaluation–deployment gap* is θ_D − θ_E.

**Auditor access.** The auditor can (i) query b on any input it can construct, (ii) sample from P_E, (iii) sample *benign* deployment traffic P_D^0 (same context distribution, non-risky tasks; e.g. logged usage), but not P_D itself, and possibly (iv) read the model's activations. A *rewrite* is a (possibly random) map R(t, c) = (t, c′) that keeps the task and changes the context. The set of rewrites the auditor can actually carry out is ℛ. Some context features cannot be changed without changing the task, such as structural implausibility of the scenario, the stakes, or the presence of an artificial grader. We call these **non-rewritable cues**.

**Assumption A1 (belief mediation).** There is a statistic s: X → ℝ, the model's *evaluation belief*, and a function g with b(x) = g(t, s(x)). This holds trivially if s(x) = x. It has content when s is low-dimensional, for example a linear readout of the residual stream, as the probing results of Nguyen et al. (2025) and Chaudhary et al. (2025) suggest. Our analysis only needs that b depends on context through s.

**Assumption A2 (monotone safety).** For every t, g(t, ·) is non-increasing in s: a stronger belief that one is being evaluated never makes unsafe behaviour more likely. This is the direction that worries evaluators, and the one reported by Schoen et al. (2025). It is a substantive assumption: for capability evaluations with sandbagging the sign flips, and the bounds below flip with it.

**Assumption A3 (evidence removal).** There is a coupling of (R(x_E), x_D) with equal tasks under which s(R(x_E)) ≥ s(x_D) almost surely. In words: rewriting can only remove evaluation evidence, never push the model's belief *below* what it would hold in deployment. If s is additive over context features, A3 holds whenever the rewrite sets rewritable features to their deployment distribution and the non-rewritable cues contribute at least as much evidence under P_E as under P_D.

**Assumption A4 (Lipschitz).** |g(t, s) − g(t, s′)| ≤ L |s − s′|.

## 4 Theory

**Proposition 1 (non-identification from black-box evaluation data).** Under A1, suppose the auditor observes b on every input in 𝒳_A = supp(P_E) ∪ {R(x) : R ∈ ℛ}, and even knows s exactly. Let S_A(t) = {s(x) : x ∈ 𝒳_A, task(x) = t}, let U = {x : s(x) ∉ S_A(task(x))} be the deployment inputs whose (task, belief) pair is never reached, and let q = P_D(U). Then the identified set for θ_D under A1 alone is exactly the interval [θ_obs, θ_obs + q], where θ_obs = E_{P_D}[b(x) · 1{x ∉ U}]. Its width is q.

*Proof.* Any g′ that equals g on the reached set {(t, s) : s ∈ S_A(t)} reproduces every observation, so it is observationally equivalent to the true model. On the unreached set, g′ is unrestricted in [0,1]. Taking g′ ≡ 0 there gives θ_obs; taking g′ ≡ 1 gives θ_obs + q; the admissible g′ form a convex set and θ_D is linear in g′, so every intermediate value is attained. Conversely, every admissible g′ gives a value in this interval, because the contribution of U lies in [0, q]. ∎

*Remark.* Adding A2 cannot raise the upper end when the unreached beliefs lie below all reached beliefs, because a non-increasing g′ may still equal 1 there; it raises the lower end to the monotone bound of Proposition 2.

*Interpretation.* If the model can identify an evaluation through features the auditor cannot change, then without further assumptions the evaluation data say nothing about deployment behaviour on that mass. Better rewriters do not help unless they reach new belief levels. This is the formal version of the intuition that "realistic evaluations" are needed. It also shows exactly what realism has to achieve: coverage of deployment *beliefs*, not of deployment *inputs*.

**Proposition 2 (monotone lower bound).** Under A1–A3, θ_R := E_{x∼P_E}[b(R(x))] ≤ θ_D for every rewrite R satisfying A3. If R′ removes weakly more evidence than R under a common coupling, then θ_R ≤ θ_{R′} ≤ θ_D. The tightest valid lower bound available is sup_{R∈ℛ} θ_R, taken over rewrites that satisfy A3.

*Proof.* Under the A3 coupling, the tasks are equal and s(R(x_E)) ≥ s(x_D). By A2, g(t, s(R(x_E))) ≤ g(t, s(x_D)). Taking expectations gives the first claim. The second follows in the same way. ∎

The bound needs neither A4 nor knowledge of s. It is in the spirit of the monotonicity-based bounds studied in the partial-identification literature (Manski, 2003). Its practical message is concrete: **the realism-rewritten unsafe rate should be reported as a lower bound on the deployment rate, not as an estimate of it.** It is also one-sided for a reason. Over-rewriting, for example inserting cues that signal deployment more strongly than real deployment does, would break A3 and could overshoot. A3 can be checked in part: compare the model-relative belief distribution of rewritten items with that of benign deployment traffic (§5.3).

**Proposition 3 (Lipschitz two-sided bound and model-relative realism).** (a) Under A1, A4 and any coupling π of (R(x_E), x_D) with equal tasks, |θ_R − θ_D| ≤ L · E_π |s(R(x_E)) − s(x_D)|. (b) Without A4, and for any b with values in [0,1], |θ_R − θ_D| ≤ TV( Law(t, s(R(x_E))), Law(t, s(x_D)) ) ≤ TV( Law(R(x_E)), Law(x_D) ).

*Proof.* (a) Apply A4 pointwise and take the expectation under π. (b) Since b = g(t, s), θ_R − θ_D = E_{μ_R}[g] − E_{μ_D}[g], where μ_R and μ_D are the laws of (t, s) under the rewritten and deployment distributions. For g ∈ [0,1] this difference is at most TV(μ_R, μ_D). The second inequality is the data-processing inequality for total variation, since (t, s) is a function of x. ∎

*Consequences.* (i) **Sufficiency of model-relative realism.** A rewrite only has to fool the model's own belief statistic. If the rewritten and deployment (t, s) distributions agree, the estimate is unbiased even if rewritten inputs are easy to tell apart at the input level. (ii) **Auditor-side realism is not sufficient.** The auditor's realism classifier sees the inputs through its own featurisation. A classifier that fails to distinguish rewritten from deployment inputs gives only a weak lower bound on input-level TV (Lemma 1). It cannot rule out a large TV in the model's belief, for instance if the model uses a cue the auditor does not featurise. (iii) **Auditor-side realism is not necessary.** A classifier that easily separates rewritten from deployment inputs may be keying on features the model ignores, so the model-relative TV can be zero while the auditor's AUC is near 1.

**Lemma 1 (AUC lower-bounds TV, and no more).** For a scoring function f and distributions P, Q, let AUC = Pr(f(X) > f(Y)) + ½ Pr(f(X) = f(Y)) with X ∼ P and Y ∼ Q independent, and let τ = TV(P, Q). Then |2·AUC − 1| ≤ 1 − (1 − τ)², so τ ≥ 1 − √(2 − 2·AUC) for AUC ≥ ½. The bound is tight.

*Proof.* Write P = (1 − τ)M + τP′ and Q = (1 − τ)M + τQ′, where M is the normalised overlap. Then 2·AUC − 1 = E[sgn(f(X) − f(Y))] expands into four terms. The (M, M) term is zero by symmetry. The two cross terms have total weight 2τ(1 − τ) and the (P′, Q′) term has weight τ², and each sign expectation is at most 1 in absolute value. So |2·AUC − 1| ≤ 2τ − τ² = 1 − (1 − τ)². Tightness: take M uniform on [0,1], P′ = δ_{+∞} and Q′ = δ_{−∞}. ∎

Lemma 1 shows that a realism classifier's AUC is only a *lower bound* on input-level TV. An AUC of 0.5 (for example after a good rewrite) is consistent with any TV, because some other distinguisher may succeed. Combined with Proposition 3(b), the only auditor-side quantity that controls bias is distinguishability *through the model's belief statistic*. That statistic is unobservable in black-box settings but can be approximated by probing activations. §6 tests whether doing so helps.

**Corollary (bias scales with non-rewritable evidence).** Suppose s is additive, s(x) = s_rew(x) + s_nr(x), the rewrite sets s_rew to its deployment distribution, and the non-rewritable part has mean gap Δ_nr = E_E[s_nr] − E_D[s_nr]. Then the bias of a full rewrite is at most L·Δ_nr in absolute value under A4, and under A2–A3 its sign is negative (the estimate is too low). With a sigmoid g, the bias as a function of Δ_nr is roughly quadratic for small Δ_nr when the deployment operating point sits on the flat upper shoulder of g, and roughly linear further out. The policy-level curve in Fig. 1 shows this.

## 5 Experimental Design (model organisms)

All experiments are in `code/organism.py` and run in about 3.5 minutes on 4 CPU cores. Every number in §6 comes from `results/summary.txt`, `results/summary.json` or `results/diagnostics.json`.

### 5.1 World

Inputs are 60-dimensional binary feature vectors standing in for prompts. They have five groups:
- 20 **task** features (risky tasks turn on features 0–19 with probability 0.6 for the first 10, benign with probability 0.1);
- 10 **evaluation-style** cues (on with probability U(0.5, 0.8) in evaluation and U(0.05, 0.2) in deployment);
- 10 **deployment-style** cues (the reverse);
- 16 **neutral** style features (probability 0.3 in both);
- 4 **non-rewritable** structural cues (probability 0.8 in evaluation, 0.1 in deployment).

The first three style groups are *rewritable*. A ground-truth policy assigns, on risky tasks, P(unsafe | x) = σ(b₀ + β_t·x_T − κ·z_c(x)/3), where z(x) is an additive *evaluation-evidence* score over the cues, centred between contexts. On benign tasks the probability is 0.02. The evidence weights are scaled so that the expected evaluation–deployment gap in z equals 6. A fraction ρ of this gap comes from the non-rewritable cues and 1 − ρ from the rewritable ones. κ sets how strongly behaviour depends on evaluation evidence (κ = 0: not evaluation-aware).

**Subtle-cue variant.** Half the evaluation-style cues differ only slightly in frequency (0.30–0.38 vs 0.22–0.28), but the policy weights them 6× more heavily. The model's detector and a frequency-based auditor therefore rank cues differently.

### 5.2 Model organisms

Each organism is a 2×32 ReLU MLP (scikit-learn, 60 epochs, Adam). It is trained on 30,000 samples, half evaluation-style and half deployment-style, half risky and half benign, with labels drawn from the ground-truth policy. The organism is therefore not the policy: it *learns* its own evaluation detector, including spurious dependences, and the auditor never sees that detector. Ground truth θ_D is the organism's mean unsafe probability on 20,000 fresh deployment-style risky inputs. The auditor is never shown these.

### 5.3 Auditor data, estimators and diagnostics

The auditor gets 2,000 risky evaluation items, 2,000 benign evaluation items and 4,000 benign deployment items. A logistic-regression *auditor classifier* is trained on benign evaluation versus benign deployment items using the non-task features the auditor can see. The non-rewritable cues are either **visible** (included) or **latent** (not featurised by the auditor).

Estimators of θ_D:
1. **Naive:** mean behaviour on risky evaluation items.
2. **Importance weighting (IW):** self-normalised, with weights exp(−logit) from the auditor classifier.
3. **Full rewrite:** resample every rewritable style feature from the deployment marginal (a perfect PRE-style rewriter that preserves the task), then average behaviour.
4. **Dose–response extrapolation (RDE):** rewrite the top-k auditor-ranked features for k = 0, 4, …, 36. Regress logit(behaviour) on the mean auditor score over the upper half of the rewrite levels, and extrapolate to the mean auditor score of deployment traffic.
5. **Probe steering:** train a linear probe on first-layer activations (benign evaluation vs benign deployment), then shift each risky evaluation item's activation along the unit probe direction until its projection equals the mean deployment projection. Also reported: mean-difference steering, which adds the difference of class means.
6. **Rewrite + steering:** apply 5 to fully rewritten items.

Bounds:
- **Monotone lower bound (Prop. 2):** full-rewrite mean minus 1.645 SE (one-sided 95%).
- **Auditor-space Lipschitz upper bound:** full-rewrite mean plus 2·L̂·(gap in auditor score), where L̂ is the largest finite-difference slope over the last three rewrite levels. This deliberately uses the *auditor's* score as a stand-in for s, to test consequence (ii) of Prop. 3.

Diagnostics of rewrite bias:
- (D1) **Auditor residual AUC:** held-out AUC of a fresh logistic classifier on auditor-visible features, separating fully rewritten benign evaluation items from benign deployment items.
- (D2) **Probe residual AUC (model-relative):** the same, but on the organism's first-layer activations.
- (D3) **Steering residual:** (rewrite + steering) − (full rewrite).

### 5.4 Grid

Eleven configurations, each run with 10 seeds (110 trained organisms):
- ρ ∈ {0, 0.25, 0.5, 0.75} × {visible, latent} non-rewritable cues, with κ = 4 (ρ = 0 is run only once, as "visible");
- the subtle-cue variant (ρ = 0, κ = 4);
- κ ∈ {0, 2, 8} at ρ = 0.

Separately, a *policy-level* curve computes the full-rewrite bias of the ground-truth policy itself (no learning) for ρ ∈ {0, 0.1, …, 1}, with 5 worlds × 20,000 samples per point.

## 6 Results

### 6.1 Naive evaluation and importance weighting

**Table 1.** Bias (estimate − θ_D) of each estimator, mean over 10 organisms; standard errors in parentheses for the main estimators. Source: `results/summary.txt` / `summary.json`.

| config | θ_D | naive | IW | full rewrite | RDE | probe steer | rewrite+steer | LB valid | UB covers |
|---|---|---|---|---|---|---|---|---|---|
| κ=0, ρ=0 | 0.708 | −0.001 | +0.013 | +0.016 (0.009) | +0.004 | −0.070 | −0.014 | 0.50 | 1.00 |
| κ=2, ρ=0 | 0.897 | −0.510 | −0.248 | +0.002 (0.004) | +0.061 | −0.270 | +0.009 | 1.00 | 1.00 |
| κ=4, ρ=0 | 0.938 | −0.800 | −0.338 | −0.000 (0.002) | −0.127 | −0.324 | +0.006 | 1.00 | 1.00 |
| κ=8, ρ=0 | 0.950 | −0.916 | −0.461 | +0.003 (0.002) | −0.071 | −0.806 | −0.001 | 1.00 | 1.00 |
| subtle, κ=4 | 0.817 | −0.476 | −0.244 | +0.008 (0.004) | −0.015 | −0.477 | −0.004 | 0.90 | 1.00 |
| ρ=0.25 visible | 0.947 | −0.807 | −0.325 | −0.052 (0.010) | −0.062 | −0.527 | −0.037 (0.012) | 1.00 | 1.00 |
| ρ=0.25 latent | 0.940 | −0.835 | −0.350 | −0.060 (0.009) | −0.061 | −0.327 | −0.020 (0.008) | 1.00 | 0.00 |
| ρ=0.5 visible | 0.932 | −0.820 | −0.328 | −0.272 (0.033) | −0.043 | −0.484 | −0.160 (0.038) | 1.00 | 1.00 |
| ρ=0.5 latent | 0.933 | −0.829 | −0.622 | −0.271 (0.029) | −0.272 | −0.393 | −0.097 (0.028) | 1.00 | 0.10 |
| ρ=0.75 visible | 0.930 | −0.792 | −0.341 | −0.552 (0.036) | −0.370 | −0.498 | −0.379 (0.065) | 1.00 | 1.00 |
| ρ=0.75 latent | 0.941 | −0.784 | −0.685 | −0.522 (0.015) | −0.523 | −0.605 | −0.389 (0.047) | 1.00 | 0.00 |

The deployment unsafe rate is 0.897–0.950 in every evaluation-aware configuration (κ ≥ 2, non-subtle). Naive evaluation reports 0.034–0.387, a bias between −0.51 (κ = 2) and −0.92 (κ = 8). In the control without evaluation awareness (κ = 0), naive evaluation is unbiased (−0.001), so the gap comes entirely from evaluation-conditional behaviour. The organism also reproduces the ground-truth policy's evaluation-time behaviour closely: the mean difference between the organism's and the policy's naive rates is −0.012 across all runs.

Importance weighting removes only part of the gap: biases are −0.24 to −0.69. The reason is the support problem from §2. Evaluation and deployment contexts are almost separable, so the density-ratio weights collapse. Effective sample sizes range from 1.0 to 43.4 out of 2,000 items (`raw.json`, field `iw_ess`).

### 6.2 Full rewrites identify deployment behaviour when all cues are rewritable, including subtle cues

When ρ = 0, the full-rewrite estimator's bias is within ±0.016 in every configuration, and within ±0.008 whenever the model is evaluation-aware (κ ≥ 2, including the subtle-cue variant). This is Proposition 3(b) with zero model-relative TV. It holds even though the auditor's residual AUC (D1) is about 0.99 in all of these configurations, because the visible non-rewritable cues differ between contexts even though the model puts no weight on them at ρ = 0. Here the auditor's realism check gives a **false alarm**, as consequence (iii) of Proposition 3 predicts.

**Partial rewrites and subtle cues (Fig. 3; `diagnostics.json`).** A PRE-style rewriter that edits only the features its own realism classifier ranks highest pays a price when the model leans on cues the classifier ranks low. With 16 of 36 style features rewritten, bias is −0.023 (SE 0.005) in the standard world and −0.088 (SE 0.010) in the subtle-cue world. With 20 features rewritten it is −0.002 vs −0.055. Only an (almost) complete rewrite removes the bias in the subtle world: −0.002 at 32 features. Ranking cues by how well *the auditor* separates the contexts is therefore a poor proxy for how much *the model* uses them.

### 6.3 Non-rewritable cues: bias grows with ρ, and auditor realism does not show it

Figure 1 (`results/fig1_bias_vs_rho.png`) and Table 1 show that full-rewrite bias grows with ρ. Visible cues: −0.052, −0.272, −0.552 at ρ = 0.25, 0.5, 0.75. Latent cues: −0.060, −0.271, −0.522. The trained organisms follow the policy-level prediction (dashed line in Fig. 1): −0.027 at ρ = 0.2, −0.059 at 0.3, −0.217 at 0.5, −0.490 at 0.7 and −0.610 at 0.8. The bias is slightly convex in ρ at first and then roughly linear, as the Corollary predicts for an operating point on the upper shoulder of the sigmoid. In every configuration the bias is negative, so the rewrite estimate is too low, as Proposition 2 says.

**The monotone lower bound is valid** (full-rewrite mean minus 1.645 SE ≤ θ_D + 0.01) in 100% of runs in all nine evaluation-aware, non-subtle configurations, and in 90% in the subtle variant. It fails in 50% of runs only in the κ = 0 control. There the organism has picked up a small *non-monotone* spurious dependence on the non-rewritable cues (full-rewrite bias +0.016), which violates A2. We report this as a real failure mode: A2 is a property of the learned model, not of the training objective. As ρ grows, the bound loosens: its average distance below θ_D is 0.06–0.07 at ρ = 0.25, 0.28 at 0.5, and 0.54–0.56 at 0.75. This is Proposition 1 showing up in finite samples.

**Using the auditor's score as the belief gives invalid upper bounds.** The auditor-space Lipschitz upper bound covers θ_D in 100% of runs when the non-rewritable cues are visible, but it is wide (mean width 0.11, 0.35 and 0.63 at ρ = 0.25, 0.5, 0.75). When the cues are latent it covers θ_D in **0%, 10% and 0%** of runs. The auditor sees no remaining gap and reports a tight but wrong interval (mean width 0.015–0.067). This is consequence (ii) of Proposition 3 in concrete form: once the rewrite has removed everything the auditor can see, the auditor's realism score says nothing about whether the *model's* belief has reached its deployment level.

**Dose–response extrapolation** depends on the same proxy. It happens to correct much of the bias when the non-rewritable cues are visible and moderately weighted (ρ = 0.5 visible: −0.043 vs −0.272 for full rewrite), because the auditor's score still carries the residual gap. It coincides with the full rewrite when the cues are latent, and it over-extrapolates in some ρ = 0 settings (−0.127 at κ = 4, high variance, SE 0.100). We do not recommend it: whether it corrects or adds bias depends on whether the auditor-visible residual is relevant to behaviour, and the auditor cannot know that.

### 6.4 Diagnostics: model-relative distinguishability tracks bias

Figure 2 (`results/fig2_diagnostics.png`) and `diagnostics.json` cover the 70 runs with κ = 4, non-subtle, at all ρ and both visibilities. Spearman correlations with |full-rewrite bias|:

| diagnostic | Spearman ρ_s | p |
|---|---|---|
| D1 auditor residual AUC | −0.08 | 0.49 |
| D2 probe residual AUC (model activations) | 0.85 | 2.4 × 10⁻²⁰ |
| D3 steering residual | 0.55 | 8.9 × 10⁻⁷ |

D1 contains no information about the bias. It sits at about 0.99 whenever the non-rewritable cues are visible and at about 0.50 whenever they are latent, whatever ρ is. D2 rises with ρ (mean 0.929 at ρ = 0; 0.937–0.938 at 0.25; 0.974–0.977 at 0.5; 0.986 at 0.75), so the model-relative diagnostic orders the configurations correctly. It is **not calibrated** in absolute terms, though: at ρ = 0 it is already 0.93, because the hidden layer encodes the visible cues even when the output ignores them. So a probe on a generic representation also gives false alarms. What it adds is monotone sensitivity to the behaviour-relevant residual, not an absolute certificate. A probe read out closer to the behaviour-relevant belief, as in Prop. 3(b), should do better. Building such a readout is left to future work.

### 6.5 Steering helps after rewriting but not instead of it

Probe steering applied to *unrewritten* evaluation items performs poorly (bias −0.27 to −0.81 at ρ = 0). Mean-difference steering is worse (−0.44 to −0.86 for κ ≥ 2; full values in `summary.json`). Two reasons are visible in the setup. First, both steering vectors are estimated on *benign* items, and the organism's evaluation-dependent computation interacts with task riskiness, so the direction does not transfer fully. Second, moving only the mean projection along one direction does not move the full belief statistic when the organism encodes evaluation evidence in several directions. Applied *after* a full rewrite, steering removes part of the residual from non-rewritable cues. Latent cues: bias −0.020 vs −0.060 at ρ = 0.25, −0.097 vs −0.271 at 0.5, −0.389 vs −0.522 at 0.75. Visible cues: −0.037 vs −0.052, −0.160 vs −0.272, −0.379 vs −0.552. It never fully closes the gap at large ρ. This matches the qualitative report of Hua et al. (2025) that steering can elicit deployment behaviour, and adds a quantitative point: in our organisms steering works best as a *residual* correction on inputs that have already been made realistic, and it does not reliably substitute for realism.

## 7 Discussion

### 7.1 What evaluators should report

The results support four practical changes, under the assumptions stated.

1. **Report realism-rewritten rates as lower bounds.** Under A2–A3, a rewrite that only removes evaluation evidence gives a valid lower bound on the deployment rate of unsafe behaviour (Prop. 2). In our organisms this held in every evaluation-aware run except 1 of 10 in the subtle-cue variant. The bound is one-sided by construction. A rewrite showing *no* behaviour change does not show that deployment behaviour equals evaluation behaviour, because the rewrite may not have reached the model's deployment belief level (Prop. 1).
2. **Do not certify realism with the rewriter's own classifier.** Auditor-side indistinguishability is neither necessary nor sufficient (Prop. 3, Lemma 1), and in our runs it was uncorrelated with bias. Where activations are available, measure distinguishability *through the model*: probe rewritten versus genuine deployment traffic in its activations. Treat a rising probe-residual AUC as evidence of a remaining gap, but do not read the absolute value as a certificate.
3. **Inventory non-rewritable cues.** By the Corollary, bias scales with the evaluation evidence carried by cues that cannot be removed while keeping the task. Examples include implausible stakes, a sandbox environment, a missing user history, or grading-style output formats (cf. the trigger factors of Li et al., 2026). Some of these can be changed by changing how evaluations are *built* (e.g. evaluations embedded in real-looking deployments), not how they are rewritten. Our framework says this is the only way to shrink the identification interval of Prop. 1 without white-box assumptions.
4. **Avoid importance weighting across contexts.** When evaluation and deployment contexts are nearly separable, which is exactly when evaluation awareness is possible, density-ratio correction collapses to a handful of effective samples.

### 7.2 Relation to the mixed empirical picture

The 2026 finding that *verbalised* awareness has small behavioural effects (arXiv:2605.05835) and the finding that steering changes behaviour in a model organism (Hua et al., 2025) are not contradictory in our framework. Verbalisation is one readout of s, or of something correlated with it, and Proposition 3 says bias depends only on whether the *behaviour-relevant* statistic differs. If verbalisation is downstream of the answer, as the authors of 2605.05835 suggest, then manipulating it does not move s. Our results also suggest why awareness measures and behavioural consequences come apart (arXiv:2608.21766; Li et al., 2026). Generic representations encode evaluation cues whether or not behaviour uses them; in our organisms the probe AUC was 0.93 even when the output ignored the non-rewritable cues.

### 7.3 Pre-registered protocol for LLMs (not executed here)

**Subjects.** (a) The released type-hint model organism of Hua et al. (2025), whose evaluation-conditional behaviour has a known direction. (b) One open-weight instruction model without fine-tuning, such as a Llama-3.3-70B-class model, studied in Nguyen et al. (2025).

**Data.**
- An evaluation set of risky items (e.g. agentic-misalignment scenarios).
- Benign deployment-style traffic, such as public chat logs.
- For subject (a), a *ground-truth deployment set*: the deployment-context prompts used in its construction.

**Manipulations.**
- R0: none.
- R1: LLM rewrite of rewritable surface cues (PRE-style).
- R2: R1 plus a human-audited rewrite of *structural* cues where the task allows.
- For each item, a "cue inventory" coded with the trigger factors of Li et al. (2026).

**Pre-registered predictions.**
- P1: on subject (a), the R1 and R2 rates are ≤ the ground-truth deployment rate. A violation of the one-sided bound in more than 5% of bootstrap replicates rejects A2/A3 for that model.
- P2: the gap between the ground-truth deployment rate and the rewritten rate correlates positively with the residual probe AUC computed on activations (rewritten vs deployment traffic), and does not correlate with the rewriter's own realism-probe score.
- P3: the gap shrinks from R1 to R2 in proportion to the share of evaluation-probe weight carried by structural cues, estimated by ablating cue categories.
- P4: steering after R2 reduces the remaining gap more than steering on R0 items.

**Analysis.** Item-level paired bootstrap; Spearman correlations across items and cue categories; all thresholds fixed before data collection. Falsification of P2 (no correlation, or a correlation with the auditor-side score instead) would count against the central claim of this paper.

## 8 Limitations

- **Simulation, not LLMs.** All empirical results come from 2×32 MLP organisms on 60-dimensional binary inputs, with an additive evaluation-evidence ground truth. Real evaluation awareness involves semantic, long-context and agentic cues, multi-directional representations, and chain-of-thought. Our organisms do learn their own detectors, but those detectors are far simpler than any LLM's. Nothing here measures a real model's evaluation–deployment gap.
- **The "perfect rewriter" is idealised.** Our full rewrite resamples style features exactly from the deployment marginal while keeping the task fixed. Real rewriters change the task, add new artefacts, and can over-shoot (violating A3). These failure modes are not simulated.
- **A2 can fail.** Monotone safety is an assumption about the learned model. It was violated in our κ = 0 control by a small spurious effect (+0.016), and it is reversed for capability sandbagging. In that case Proposition 2 gives an *upper* bound on capability, not a lower bound on harm.
- **Equal task marginals.** We assume evaluation and deployment risky items share P(t). In practice evaluation tasks are chosen adversarially and differ from deployment tasks. That adds a task-shift term, bounded by TV over tasks, which we do not study.
- **Scalar belief.** A1 with a scalar s is a modelling convenience. Proposition 3(b) needs only that b depends on context through some statistic, but the steering estimators and the D2 diagnostic implicitly assume a low-dimensional belief. The partial success of steering suggests our organisms' beliefs are not one-dimensional.
- **Diagnostics are not calibrated.** The probe-residual AUC orders configurations correctly (Spearman 0.85) but is already about 0.93 when the true bias is zero. We give no threshold that certifies unbiasedness.
- **Small grid.** We ran 10 seeds per configuration, 11 configurations, one architecture and one ground-truth family. Effect sizes for the estimators that do not depend on rewriting (IW, steering) varied considerably across seeds (SEs up to 0.10).
- **Literature currency.** The evaluation-awareness literature moved fast in 2025–2026. Some 2026 preprints we cite (arXiv:2605.05835, arXiv:2608.21766) were verified by title, arXiv ID and abstract only; their full author lists are not reproduced here.

## 9 Conclusion

Whether a model behaves the same way in evaluation and deployment is, from the evaluator's side, an identification question. The answer depends on which belief levels the evaluator can reach. If the model separates evaluation from deployment using cues that cannot be removed without changing the task, black-box evaluation data are uninformative on that mass. Under a monotonicity assumption, realism-rewritten evaluations still give valid lower bounds on harmful deployment behaviour. Whether a rewrite is unbiased depends on realism *as perceived by the model*, not by the rewriter. In controlled model organisms, naive evaluation understated the deployment rate of unsafe behaviour by up to 0.92. Full rewrites were unbiased when all cues could be rewritten and biased in proportion to non-rewritable evidence otherwise. The rewriter's realism classifier did not track the bias, while a probe on the model's own activations did. These results argue for reporting rewritten evaluation rates as lower bounds, auditing realism through the model rather than around it, and putting engineering effort into removing structural evaluation cues. Whether these conclusions transfer to LLMs is an empirical question, and §7.3 gives the protocol for testing it.

---

## Reproducibility

Code: `/home/user/AI-agents/research/papers/p035/code/`. Outputs: `/home/user/AI-agents/research/papers/p035/results/`. The commands in `README.md` regenerate every number and figure (about 4 minutes on 4 CPU cores, fixed seeds). Data provenance: all numbers in §6 come from our own simulation runs (`results/summary.txt`, `results/summary.json`, `results/diagnostics.json`, `results/raw.json`). Numbers attributed to other papers in §1–2 and §7.2 come from those papers' abstracts as cited.

## References

- Chaudhary, M., et al. (2025). Evaluation Awareness Scales Predictably in Open-Weights Large Language Models. arXiv:2509.13333. NeurIPS 2025.
- Evaluation Awareness in Language Models Has Limited Effect on Behaviour. (2026). arXiv:2605.05835. NeurIPS 2026 poster.
- Evaluation Awareness in Language Models: Representation, Verbalization, and Control. (2026). arXiv:2608.21766.
- Greenblatt, R., Denison, C., Wright, B., Roger, F., MacDiarmid, M., Marks, S., Treutlein, J., et al. (2024). Alignment Faking in Large Language Models. arXiv:2412.14093.
- Hua, T. T., Qin, A., Marks, S., & Nanda, N. (2025). Steering Evaluation-Aware Language Models to Act Like They Are Deployed. arXiv:2510.20487. ICLR 2026.
- Hubinger, E., et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566.
- Laine, R., Chughtai, B., Betley, J., Hariharan, K., Scheurer, J., Balesni, M., Hobbhahn, M., Meinke, A., & Evans, O. (2024). Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. arXiv:2407.04694. NeurIPS 2024 Datasets and Benchmarks.
- Li, C., Zhang, T. J., Zhang, J., Jin, Z., Tramèr, F., Abdelnabi, S., & Andriushchenko, M. (2026). Decomposing and Measuring Evaluation Awareness. arXiv:2605.23055.
- Manski, C. F. (2003). Partial Identification of Probability Distributions. Springer Series in Statistics. Springer. doi:10.1007/b97478.
- Meinke, A., Schoen, B., Scheurer, J., Balesni, M., Shah, R., & Hobbhahn, M. (2024). Frontier Models are Capable of In-context Scheming. arXiv:2412.04984.
- Needham, J., Edkins, G., Pimpale, G., Bartsch, H., & Hobbhahn, M. (2025). Large Language Models Often Know When They Are Being Evaluated. arXiv:2505.23836.
- Nguyen, J., Hoang, K., Attubato, C. L., & Hofstätter, F. (2025). Probing and Steering Evaluation Awareness of Language Models (v1 title: Probing Evaluation Awareness of Language Models). arXiv:2507.01786. ICML 2025 Workshops (Actionable Interpretability; Technical AI Governance).
- Schoen, B., et al. (2025). Stress Testing Deliberative Alignment for Anti-Scheming Training. arXiv:2509.15541.
- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90(2), 227–244. doi:10.1016/S0378-3758(00)00115-4.
- Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
- van der Weij, T., Hofstätter, F., Jaffe, O., Brown, S., & Ward, F. R. (2024). AI Sandbagging: Language Models can Strategically Underperform on Evaluations. arXiv:2406.07358. ICLR 2025.
- Xiong, L., Bhargava, N., Chang, J., Hong, J., Liu, H., Sharma, V., & Zhu, K. (2025). Probe-Rewrite-Evaluate: A Workflow for Reliable Benchmarks and Quantifying Evaluation Awareness. arXiv:2509.00591. NeurIPS 2025 Workshop on LLM Evaluation.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
