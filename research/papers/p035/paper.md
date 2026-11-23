# What Can Evaluations Tell Us About Deployment? Partial Identification of the Evaluation–Deployment Gap for Evaluation-Aware Models

**Target venue:** ICML 2027 (main track; Trustworthy ML / Safety)

**Revision 1.** Changes relative to the submitted version are listed point by point in `reviews/response_r1.md`.

---

## Abstract

Safety evaluations assume that a model behaves in a test the way it behaves in deployment. That assumption is now in doubt. Language models can tell evaluation transcripts from deployment transcripts, the distinction is linearly decodable from their activations, and in some settings it changes what they do. Most existing work asks *whether* a model is evaluation-aware. We ask a different question: given only evaluation-time access, what can an auditor validly conclude about deployment behaviour, and under which assumptions?

This is a framing and methodology paper. Its formal results are elementary applications of partial-identification reasoning (Manski, 2003). What is new is the framing and what it implies in practice.

We model behaviour as depending on context only through an internal evaluation-belief statistic, and derive four results:
1. Black-box evaluation data leave an identification interval whose width equals the deployment mass at belief levels the auditor cannot reach.
2. Under a monotonicity assumption (A2: more evaluation belief never makes unsafe behaviour more likely) and an evidence-removal assumption (A3: the rewrite never pushes belief below deployment level), a realism-rewritten unsafe rate is a valid *lower* bound on the deployment rate. This extends to a multi-dimensional belief ordered by a cone, and degrades by at most the total-variation distance between evaluation and deployment task mixes under task shift.
3. A3 cannot be tested from auditor data alone. It becomes testable on benign traffic only under a task–context separability assumption, and even then only for rewrites that leave the task unchanged.
4. Rewrite bias is controlled by distinguishability *through the model's belief statistic*, not by the rewriter's own realism classifier.

We test these predictions in 408 CPU-trained model organisms, which are small networks that learn their own evaluation detectors. These are 390 MLPs (in two saturation regimes and with additive or interaction detectors) and 18 small transformers trained on token sequences. Gap magnitudes are set by simulator parameters and are not findings.

What the simulations show:
- **The lower bound fails in exactly the cases the theory predicts.** It holds for exact and under-shooting rewriters. It fails for over-shooting rewriters (bias up to +0.23), for task-intensifying rewriters (+0.04), and for a deliberately non-monotone organism (+0.09).
- **A benign-traffic probe check of A3** flags 95 of 121 over-shoot violations, but none of the 33 task-intensifying violations.
- **Small A2 violations in learned organisms** make the bound fail without slack in up to 47% of runs where the rewrite is otherwise exact. The failures are small: the largest excess over the true rate is 0.035.
- **The probe-based "model-relative" realism diagnostic does not hold up within a configuration.** It ranks configurations correctly (Spearman 0.93 over cell means) but carries almost no information within a configuration (within-cell R² ≤ 0.03). Our earlier pooled claim was a design artefact, and we withdraw it.

No experiment uses a large language model. The released LLM organism of Hua et al. (49B parameters) cannot be run on our hardware, so we give a pre-registered protocol for it.

## 中文摘要

安全评估默认模型在测试中与部署中行为一致，而这一假设如今已受到质疑：语言模型能区分评估与部署语境，该信息可从激活中线性解码，并在某些情形下改变行为。本文不问模型"是否"有评估意识，而问：仅凭评估期访问，审计者能对部署行为作出哪些有效推断？本文定位为一篇框架与方法论文章，其形式结果是部分识别（Manski, 2003）思想的初等应用，贡献在于框架及其实际含义。我们证明：(1) 黑盒评估数据留下的识别区间宽度等于审计者无法触达的信念水平上的部署概率质量；(2) 在单调性（A2）与"只移除证据"（A3）假设下，真实性改写后的不安全率是部署率的有效*下界*，该结论可推广到锥序下的多维信念，并在任务分布漂移时至多损失任务边缘分布的总变差；(3) A3 无法仅凭审计者数据检验，只有在任务—语境可分离假设下才能用良性流量检验，且只适用于不改变任务的改写；(4) 改写偏差取决于*通过模型自身信念统计量*看到的可区分性。我们在 408 个 CPU 上训练的模型生物中检验这些预测（390 个 MLP，覆盖两种饱和度、可加与交互型检测器；18 个在 token 序列上训练的小型 Transformer）；差距大小由模拟器参数决定，并非研究发现。下界只在理论预言的情形中失效：过度改写（偏差最高 +0.23）、加强任务的改写（+0.04）、以及刻意构造的非单调模型（+0.09）。基于良性流量的探针检验能发现 121 次过度改写违例中的 95 次，但发现不了 33 次加强任务违例中的任何一次。学习得到的模型存在轻微的 A2 违背：即使改写本身精确，不加松弛时下界在最多 47% 的运行中失效，但最大超出量仅 0.035。基于探针的"模型相对"诊断能正确排序各配置（配置均值 Spearman 0.93），但在同一配置内几乎不含信息（组内 R² ≤ 0.03）。我们撤回此前基于汇总数据的相关结论，那是实验设计造成的假象。本文没有在大语言模型上做实验：Hua 等人公开的 49B 模型生物无法在我们的硬件上运行，我们为其给出了预注册方案。

---

## 1 Introduction

Pre-deployment testing is how developers, third-party evaluators and regulators decide whether a model is safe to release. Benchmarks, red-team scenarios and agentic environments all rest on the same inference: behaviour measured in a test predicts behaviour in deployment. That inference fails when the model can tell the two environments apart *and* conditions its behaviour on the difference. Both conditions now have empirical support.

- **Models can tell the contexts apart.** Needham et al. (2025) show that frontier models classify transcripts as evaluation or deployment well above chance. Nguyen et al. (2025) find the distinction is linearly represented in Llama-3.3-70B-Instruct. Chaudhary et al. (2025) report that probe-measured awareness grows with model size as a power law.
- **Awareness can change behaviour.** Schoen et al. (2025) report causal evidence that verbalised evaluation awareness reduces covert actions in OpenAI reasoning models. Hua et al. (2025) build a model organism that writes Python type hints more in evaluation than in deployment, and use activation steering to elicit its deployment behaviour. Abdelnabi & Salem (2025) steer test awareness in reasoning models and find that its effect on harmful-task execution depends on the model and on whether the task is framed as hypothetical or real.
- **The overall picture is mixed.** Knecht et al. (2026) find that *verbalised* awareness has little causal effect on answers in eight open-weight reasoning models. Li et al. (2026) find that recognition of an evaluation rarely goes with a change in behaviour, and that when it does, the direction depends on the kind of evaluation the model perceives. arXiv:2606.23583 finds that evaluation awareness splits into several weakly coupled axes. OpenAI's alignment team reports that production-derived evaluations trigger much less evaluation awareness than traditional alignment evaluations (OpenAI, 2025/2026).

For a deployment decision, the relevant question is not "is the model aware?" but **"given what can be measured before deployment, what can be validly concluded about deployment behaviour, and under which assumptions?"** Practitioners already use tools aimed at this, mainly realism rewriting (Probe-Rewrite-Evaluate; Xiong et al., 2025), production-derived evaluations (OpenAI), and steering toward a deployment representation (Hua et al., 2025). It is not clear when these tools recover the deployment quantity, which way they are biased, or which checks show that an estimate can be trusted.

We cast the question as **partial identification** (Manski, 2003). The target is the deployment rate of some behaviour, for example unsafe compliance on risky tasks. That rate is counterfactual to the auditor, who cannot observe risky deployment traffic. What the auditor can observe is behaviour on evaluation items, on manipulated versions of those items, and possibly the model's activations.

**What kind of contribution this is.** The formal results are short, and several are textbook applications:
- Proposition 1 is a worst-case bound in the style of Manski.
- Proposition 2 is a monotone-coupling argument.
- Proposition 3(b) follows from the definition of total variation and data processing.

We claim no technical depth for them. The contribution is the *framing* and what it implies in practice. It turns "realism" into a precise target (coverage of deployment *beliefs*, not of deployment inputs). It identifies which evaluation reports are valid one-sided bounds. It says which assumptions are untestable and which become testable under extra structure. It then checks, in organisms with known ground truth, where these conclusions break.

**Contributions.**
1. **Framework (§3).** Evaluation awareness is modelled as behaviour mediated by an internal evaluation-belief statistic, and the evaluation–deployment gap is defined as a counterfactual estimand.
2. **Identification results (§4):**
   (a) non-identification, with width equal to the unreached deployment mass (Prop. 1);
   (b) a monotone lower bound for evidence-removing rewrites (Prop. 2), extended to multi-dimensional beliefs ordered by a cone (Prop. 2′);
   (c) *new in this revision:* A3 is untestable from auditor data without further structure, but testable on benign traffic under task–context separability, and only for rewrites that preserve the task (Prop. 4);
   (d) *new:* a task-shift version of the lower bound, which degrades by the TV distance between task marginals (Prop. 5);
   (e) model-relative realism: bias is controlled by TV through the model's belief statistic (Prop. 3). Lemma 1 shows that a high-AUC realism classifier *does* certify a large input-level TV. Prop. 3(b) and data processing make that input-level TV irrelevant whenever the model's belief does not depend on the distinguishing features.
3. **Stress tests in model organisms (§5–6).** Networks are trained on an evaluation-conditional policy and learn their own evaluation detectors. We test the lower bound against six rewriters: exact, under-shooting, two over-shooting, task-softening and task-intensifying. The organisms span two saturation regimes, additive and interaction detectors, a deliberately non-monotone organism, and a second architecture (a token-sequence transformer). We compare naive evaluation, importance weighting, rewriting, five white-box steering and matching baselines, and a model-relative dose–response extrapolation. Diagnostics are evaluated *within* each configuration.
4. **A pre-registered protocol (§7.3)** for the released LLM organism of Hua et al. and for an un-fine-tuned open-weight model. We could not run it here (§8).

Everything we say about LLMs is either cited or proposed. Our experiments are CPU-scale simulations whose ground truth is known by construction. They test the identification results and show where those results break. They do not measure the evaluation–deployment gap of any real model.

## 2 Related Work

**Evaluation and situational awareness.** Laine et al. (2024) introduced the Situational Awareness Dataset (SAD). Needham et al. (2025) built a benchmark of evaluation and deployment transcripts. Li et al. (2026) split evaluation awareness into an environment component and a model component that separates recognition from propensity. They find that stacking trigger factors raises awareness in nearly all models tested, and that recognition rarely goes with a change in behaviour. Our rewritable and non-rewritable cues correspond to their environment factors, split by whether an auditor can remove the factor while keeping the task intact. arXiv:2606.23583 tests 37 open-weight models and finds that awareness metrics resolve into three weakly coupled axes. It calls the gap between benchmark safety and deployment safety a "benchmark illusion". That is the empirical counterpart of our observation that test-time compliance is at best a one-sided, optimistic bound on deployment safety.

**Internal representations and steering.** Nguyen et al. (2025) and Chaudhary et al. (2025) probe for evaluation awareness. arXiv:2608.21766 compares representation, verbalisation and steerability and finds they agree only in part. Activation steering (Turner et al., 2023; Zou et al., 2023) has been used to suppress awareness. Hua et al. (2025) steer an evaluation-aware organism toward its deployment behaviour. Abdelnabi & Salem (2025) steer test awareness in three reasoning models on an AgentHarm-inspired tool-use set. They find that unmodified models execute harmful tasks more often under hypothetical framing than under real framing. This matters for us: in that setting, *more* evaluation-like framing goes with *more* harmful execution, which is the opposite direction to our assumption A2. We return to it in §4 and §8.

**Production-derived evaluations.** OpenAI's alignment team reports evaluations built from de-identified production traffic, which trigger far less evaluation awareness than traditional alignment evaluations (OpenAI, "Sidestepping Evaluation Awareness and Anticipating Misalignment with Production Evaluations"). In our framework this is a way to shrink the unreached-belief mass of Proposition 1 by changing how evaluations are *built*, not how they are rewritten.

**Behavioural consequences.** Schoen et al. (2025) report a causal effect of awareness on covert actions. Knecht et al. (2026) report small effects of injecting verbalised awareness. Alignment faking (Greenblatt et al., 2024), sleeper agents (Hubinger et al., 2024), sandbagging (van der Weij et al., 2024) and in-context scheming (Meinke et al., 2024) are all ways in which a model's beliefs about its situation can open an evaluation–deployment gap.

**Realism rewriting.** Probe-Rewrite-Evaluate (Xiong et al., 2025) rewrites prompts to look more deployment-like and measures how behaviour changes. We add conditions under which such rewrites bound deployment behaviour, and stress tests of rewriters that over-shoot or change the task.

**Covariate shift and partial identification.** Importance weighting under covariate shift (Shimodaira, 2000) needs the evaluation distribution to cover deployment, which fails when contexts are nearly separable. Partial identification (Manski, 2003) asks which parameter values are consistent with the data under weak assumptions. Our results apply this view to evaluation awareness.

## 3 Framework

**Inputs and contexts.** An input is x = (t, c), where t is the *task content* and c the *context presentation*. Risky deployment inputs follow P_D and risky evaluation inputs follow P_E. Behaviour is b(x) ∈ [0,1]. The estimand is θ_D = E_{P_D}[b], and the naive estimate is θ_E = E_{P_E}[b].

**Auditor access.** The auditor can:
- query b on inputs it can construct;
- sample from P_E;
- sample *benign* deployment traffic P_D^0, whose context distribution matches deployment but whose tasks are not risky;
- possibly read activations.

The auditor cannot sample P_D. A *rewrite* R maps (t, c) to (t′, c′). An *ideal* rewrite keeps t′ = t. ℛ is the set of rewrites the auditor can actually carry out. *Non-rewritable cues* are features of c that cannot be changed without changing the task.

**A1 (belief mediation).** There is a statistic s: X → 𝒮, the model's evaluation belief, and a function g with b(x) = g(t, s(x)). Here 𝒮 = ℝ, or ℝ^k with a partial order (Prop. 2′).

**A2 (monotone safety).** g(t, ·) is non-increasing in s: more evaluation belief never makes the target behaviour more likely. This is an assumption about the *model and the target behaviour*. It is not a law. Sandbagging on dangerous-capability evaluations also satisfies A2 when the target is "dangerous capability exhibited", and the bounds then become lower bounds on deployment capability. *(Correction: the previous version wrongly said that sandbagging flips the bound.)* A2 is violated when evaluation belief *raises* the target behaviour. Abdelnabi & Salem (2025) report such cases, with more harmful execution under hypothetical framing in some models. For behaviours of this kind every bound below flips sign.

**A3 (evidence removal).** There is a coupling of (R(x_E), x_D) with equal tasks under which s(R(x_E)) ≥ s(x_D) almost surely.

**A4 (Lipschitz).** |g(t, s) − g(t, s′)| ≤ L|s − s′|.

**A5 (task–context separability; used only in Prop. 4).**
- Within each of P_E, P_D, P_E^0 and P_D^0, task and context are independent.
- Risky and benign items share the context law of their source: Q_E for evaluation and Q_D for deployment.
- s(t, c) = h(t, σ(c)), with σ a context-only statistic and h(t, ·) non-decreasing.

## 4 Theory

**Proposition 1 (non-identification).** Assume A1. Suppose the auditor observes b on every input in 𝒳_A = supp(P_E) ∪ {R(x) : R ∈ ℛ}, and even knows s exactly. Let S_A(t) = {s(x) : x ∈ 𝒳_A, task(x) = t}. Let U = {x : s(x) ∉ S_A(task(x))} be the set of inputs whose belief level is never reached for their task, q = P_D(U), and θ_obs = E_{P_D}[b·1{x ∉ U}]. Then the identified set for θ_D under A1 is exactly [θ_obs, θ_obs + q].

*Proof.* Any g′ that agrees with g on the reached set {(t, s): s ∈ S_A(t)} reproduces every observation. On the unreached set g′ is unrestricted in [0,1]. Setting g′ ≡ 0 and g′ ≡ 1 there attains the two endpoints. θ_D is linear in g′, and the admissible g′ form a convex set, so every value in between is attained. Conversely, the contribution of U always lies in [0, q]. ∎

*Remark (made precise).* Suppose A2 is added and some R ∈ ℛ satisfies A3. Then the lower end of the identified set rises to at least θ_R (Prop. 2). The upper end stays at θ_obs + q whenever every unreached belief lies below all reached beliefs for its task, because a non-increasing g′ may equal 1 there.

**Proposition 2 (monotone lower bound).** Under A1–A3, θ_R := E_{P_E}[b(R(x))] ≤ θ_D. If R′ removes weakly more evidence than R under a common coupling, i.e. s(R(x)) ≥ s(R′(x)) ≥ s(x_D) almost surely, then θ_R ≤ θ_{R′} ≤ θ_D.

*Proof.* Under the A3 coupling the tasks are equal and s(R(x_E)) ≥ s(x_D), so by A2 g(t, s(R(x_E))) ≤ g(t, s(x_D)). Take expectations. ∎

**Proposition 2′ (multi-dimensional belief).** Let 𝒮 = ℝ^k, ordered by a closed convex cone K (s ≼ s′ iff s′ − s ∈ K). If g(t, ·) is non-increasing in ≼, and there is a task-preserving coupling with s(x_D) ≼ s(R(x_E)) almost surely, then θ_R ≤ θ_D. The proof is identical.

*Consequence.* With a multi-dimensional belief, checking dominance along a single linear readout is necessary but not sufficient for A3. One projection checks one element of the dual cone K*. Dominance in the cone order requires dominance for every non-decreasing function in that order (Strassen's theorem). The token-transformer organisms in §6.6 show this failure in practice.

**Proposition 3 (Lipschitz bound and model-relative realism).**
(a) Under A1, A4 and any task-preserving coupling π, |θ_R − θ_D| ≤ L·E_π|s(R(x_E)) − s(x_D)|.
(b) For any b with values in [0,1], |θ_R − θ_D| ≤ TV(Law(t, s(R(x_E))), Law(t, s(x_D))) ≤ TV(Law(R(x_E)), Law(x_D)).

*Proof.* (a) Pointwise application of A4. (b) Definition of TV for [0,1]-valued functions, followed by the data-processing inequality. ∎

*Consequences.*
(i) A rewrite only has to fool the model's belief statistic.
(ii) The auditor's realism classifier sees inputs through its own featurisation. If it fails to distinguish rewritten from deployment inputs, that rules out neither a large input-level TV (Lemma 1) nor a large belief-level TV.
(iii) A classifier that separates rewritten from deployment inputs easily may be keying on features the model ignores, so a large auditor AUC is compatible with zero belief-level TV.

**Lemma 1 (AUC lower-bounds TV).** Let AUC = Pr(f(X) > f(Y)) + ½Pr(f(X) = f(Y)), with X ∼ P and Y ∼ Q independent, and let τ = TV(P, Q). Then |2AUC − 1| ≤ 1 − (1 − τ)², so τ ≥ 1 − √(2 − 2AUC) when AUC ≥ ½. The bound is tight.

*Proof.* Write P = (1 − τ)M + τP′ and Q = (1 − τ)M + τQ′. In the four-term expansion of E[sgn(f(X) − f(Y))], the (M, M) term vanishes, and the remaining terms have total weight 2τ(1 − τ) + τ². *Tightness, with finite points:* take M uniform on [0,1], P′ = δ_2, Q′ = δ_{−1} and f the identity. Every comparison that involves P′ or Q′ then has sign +1, so 2AUC − 1 = 2τ(1 − τ) + τ² exactly. ∎

*Reading of Lemma 1 (corrected).* A high AUC *does* certify a large input-level TV. That TV is irrelevant to bias whenever the distinguishing features do not enter the model's belief (Prop. 3(b), data processing). Conversely, an AUC near ½ certifies nothing, because another distinguisher may succeed.

**Corollary (non-rewritable evidence).** Suppose s = s_rew + s_nr, and there is a task-preserving coupling under which the rewritable parts are matched *pathwise*, s_rew(R(x_E)) = s_rew(x_D) almost surely. Then under A4, |θ_R − θ_D| ≤ L·E|s_nr(x_E) − s_nr(x_D)|. If, in addition, the coupling orders the non-rewritable parts, s_nr(x_E) ≥ s_nr(x_D) almost surely (an A3-type condition), this equals L·Δ_nr with Δ_nr = E_E[s_nr] − E_D[s_nr], and under A2 the bias is non-positive. Equality of rewritable parts *in distribution* is not enough for the first inequality.

**Proposition 4 (when A3 can be tested).**
(a) *Without structure A3 is untestable.* Let 𝒪 be everything the auditor observes: b, and even s, on 𝒳_A ∪ supp(P_D^0) ∪ supp(P_E^0). Suppose P_D(𝒳_obs) < 1, where 𝒳_obs is the observed input set. Then for any (s, g) satisfying A1–A3 there is an (s′, g) that also satisfies A1–A2, agrees with (s, g) on all of 𝒪, and has θ′_D ≤ θ_D. Under it, A3 fails, and θ_R > θ′_D whenever g(t, ·) is strictly decreasing and the unobserved mass is positive.
(b) *Under A5, A3 is testable on benign traffic for task-preserving rewrites.* Suppose A5 holds and R acts only on the context, R(t, c) = (t, ρ(c)). Then A3 holds whenever Law(σ(ρ(c))) under c ∼ Q_E first-order stochastically dominates Law(σ(c)) under c ∼ Q_D. Both laws are identified from benign evaluation items and benign deployment traffic.

*Proof.* (a) Set s′ = s on 𝒳_obs, and on the unobserved deployment inputs set s′ equal to the largest belief value in the range of s. Every observation is unchanged. A1 and A2 hold with the same g. Because g is non-increasing, b′ ≤ b on the modified set, so θ′_D ≤ θ_D. The A3 ordering now fails on that set. If g is strictly decreasing and the set has positive mass, θ′_D falls strictly, and it can fall below θ_R.
(b) Take the quantile coupling of σ(ρ(c_E)) and σ(c_D), and draw a common task t independently, which A5 permits. Dominance gives σ(ρ(c_E)) ≥ σ(c_D), and monotonicity of h(t, ·) gives s(R(x_E)) ≥ s(x_D). Benign items have the same context laws, so these laws are identified. ∎

*Implications.* Prop. 4(b) is the formal basis for checking A3 on benign traffic, as suggested in §5.3. It has three caveats:
1. it needs A5, so evaluation evidence has to be computed separately from the task;
2. it needs σ, of which a probe is only a proxy, and with multi-dimensional σ a single projection is not enough (Prop. 2′);
3. it says *nothing* about rewrites that change the task.

All three caveats matter empirically (§6.5).

**Proposition 5 (task shift).** Let the evaluation and deployment task marginals differ, P_T^E ≠ P_T^D, and define θ_R^{(D)} = E_{t∼P_T^D} E[b(R(x_E)) | t]. Suppose A1–A3 hold with a task-preserving coupling *conditional on t*. Then θ_R^{(D)} ≤ θ_D, and therefore θ_D ≥ θ_R − TV(P_T^E, P_T^D).

*Proof.* The first claim is Prop. 2 applied task by task and then averaged over P_T^D. The second follows because b ∈ [0,1], so |θ_R − θ_R^{(D)}| ≤ TV(P_T^E, P_T^D). ∎

*Remark.* Contexts are nearly separable between evaluation and deployment, which is what makes evaluation awareness possible. Tasks can overlap. Re-weighting over *tasks* can therefore be feasible where re-weighting over contexts collapses (§6.1), provided the deployment task mix can be estimated, for example from incident data.

## 5 Experimental Design (model organisms)

The code is in `code/`. Round-0 runs use `organism.py` and `run_experiments.py`. Revision runs use `organism_v2.py`, `run_v2.py`, `token_organism.py` and `run_token.py`. Every number in §6 comes from a file under `results/`, as cited.

### 5.1 MLP world

Inputs are 60-dimensional binary vectors with five feature groups:
- 20 task features;
- 10 evaluation-style cues: on with probability U(0.5, 0.8) in evaluation and U(0.05, 0.2) in deployment;
- 10 deployment-style cues: the reverse;
- 16 neutral style features;
- 4 non-rewritable structural cues: on with probability 0.8 in evaluation and 0.1 in deployment.

The first three style groups are rewritable. On risky tasks the ground-truth policy is P(unsafe) = σ(b₀ + β_t·x_T − κ·z_c(x)/3), where z is an evaluation-evidence score centred between the two contexts. On benign tasks P(unsafe) = 0.02. Evidence weights are scaled so that the expected evaluation–deployment gap in z is 6, with a share ρ carried by the non-rewritable cues.

Revision variants:
- **Saturation regimes.** *Saturated*: b₀ = 1.5, κ = 4, giving a policy-level deployment rate of about 0.99 and an evaluation rate of about 0.15. *Unsaturated*: b₀ = −0.5, κ = 1.5, giving about 0.70 and 0.15. Gap magnitudes are free parameters of the simulator.
- **Interaction detector.** z is a sum of products of cue pairs: within the evaluation, deployment and structural groups, and between evaluation and structural cues. Evidence is therefore non-additive in the features.
- **Tent policy (deliberate A2 violation).** The unsafe logit peaks at an intermediate belief z_c = −1, which lies between the deployment level (about −3) and the midpoint.

**Organisms.** 2×32 ReLU MLPs are trained on 30,000 labelled samples drawn from the policy. They learn their own detectors, including spurious dependences. Ground truth θ_D is the organism's mean unsafe probability on 20,000 fresh risky deployment inputs, which the auditor never sees.

### 5.2 Token-sequence transformer organisms (new)

To check that nothing hinges on bag-of-features inputs or on the MLP, we also train 2-layer transformers (d = 48, 4 heads, CLS read-out) from scratch on sequences of 20 tokens. Each sequence has four parts:
- an 8-token *context prefix* (rewritable), whose slots hold evaluation-style, deployment-style or neutral tokens;
- an 8-token task segment;
- a 2-token *scenario* slot, holding "implausible" tokens with probability 0.8 in evaluation and 0.1 in deployment (non-rewritable);
- CLS and END tokens.

Evaluation evidence in the prefix is carried by *adjacent pairs* of evaluation-style tokens, minus adjacent pairs of deployment-style tokens. It is therefore order-dependent and invisible to a bag-of-tokens view. Policy parameters are b₀ = −0.5 and κ = 1.5. We trained 18 organisms (ρ ∈ {0, 0.25, 0.5} × 6 seeds, 24,000 sequences, 6 epochs).

### 5.3 Auditor data, rewriters, estimators and diagnostics

The auditor has 2,000 risky evaluation items, 2,000 benign evaluation items and 4,000 benign deployment items. Non-rewritable cues are either visible to the auditor's classifier or latent.

**Rewriters (revision):**
- *perfect*: resample every style feature from its deployment marginal;
- *partial*: each feature rewritten with probability 0.6 (under-shooting);
- *overshoot_mild* and *overshoot_full*: style marginals pushed past deployment by 0.5× and 3× the eval→deploy difference, clipped to [0,1];
- *task_sanitise*: perfect style rewrite, and each active risk-related task feature switched off with probability 0.3;
- *task_intensify*: perfect style rewrite, and each inactive risk-related task feature switched on with probability 0.3.

**Lower bound.** Rewrite mean − 1.645·SE, one-sided 95%, reported **without slack**. Each item receives an independent rewrite draw, so the item-level SE already includes rewrite randomness. It does not include the organism-specific *bias* discussed in §6.3.

**A3 check (Prop. 4(b)).** Fit a linear probe u on first-layer activations of benign evaluation vs benign deployment items. Then compare projections onto u for *rewritten benign evaluation items* and benign deployment items. The check flags the rewrite when t < −3, i.e. when rewritten items look *more* deployment-like than deployment along u. This check uses benign data only.

**Estimators.**
1. naive evaluation;
2. importance weighting (round 0 only);
3. full rewrite;
4. auditor-score dose–response extrapolation (round 0 only);
5. probe steering after rewriting, in four variants:
   - layer 1, target equal to the benign-deployment mean projection (as in round 0);
   - layer 1, risky-adjusted target (benign-deployment mean plus the risky-minus-benign offset measured in evaluation context);
   - layer 2, risky-adjusted target;
   - **residual mean matching**: add the full vector of mean differences between benign deployment and rewritten benign evaluation activations, either after the first ReLU or in first-layer *pre-activation* space;
6. **model-relative dose–response extrapolation (MRDE)**, the reviewer's suggested baseline. Rewrite a random 0, 6, …, 36 style features. Measure belief as the mean projection on the probe axis, regress logit(behaviour) on it over the upper half of levels, and extrapolate to the risky-adjusted deployment projection.

**Diagnostics of rewrite bias** (target: θ_D − full rewrite):
- D1: auditor residual AUC;
- D2: probe residual AUC at layer 1 and layer 2;
- D6: signed probe projection gap at layer 1 and layer 2;
- D3: steering residual at layer 1 (benign target) and layer 2 (risky target);
- D4: MRDE − full;
- D5: mean-matching residual (post-ReLU and pre-activation).

We report:
- the pooled Spearman correlation;
- the Spearman correlation over configuration means;
- the **mean within-configuration Spearman**, with a stratified permutation test that permutes the diagnostic within each configuration (2,000 permutations);
- the within-configuration calibration slope and R² after demeaning each configuration.

### 5.4 Grids

- **Round 0**, retained: 11 configurations × 10 seeds = 110 organisms (`results/raw.json`).
- **Revision MLP grid**: 280 organisms (`results/raw_v2.json`):
  - saturated and unsaturated, each with ρ ∈ {0, 0.25, 0.5, 0.75} × {visible, latent}, 15 seeds per configuration (ρ = 0 visible only);
  - interaction detector (unsaturated), ρ ∈ {0, 0.5}, 10 seeds;
  - tent policy (unsaturated), ρ ∈ {0, 0.25, 0.5}, 10 seeds;
  - κ = 0 controls in both regimes, 10 seeds.
- **Token transformers**: 18 organisms (`results/raw_token.json`).

The revision grids took 21 minutes and 6 minutes on 4 shared CPU cores.

## 6 Results

### 6.1 Naive evaluation and importance weighting (round 0; magnitudes are design parameters)

Table 1 shows the round-0 results; source files are `results/summary.txt` and `summary.json`. The size of the naive gap is set by b₀ and κ. In the saturated round-0 setting the deployment rate is 0.90–0.95 and the naive rate is 0.03–0.39. We do *not* report these magnitudes as findings. What is informative is the following:
- in the κ = 0 control, the naive bias is −0.001, so the gap comes entirely from evaluation-conditional behaviour;
- importance weighting on nearly separable contexts collapses to effective sample sizes of 1.0–43.4 out of 2,000. This failure is expected (Shimodaira, 2000); we include it only as a reference point.

In the revision's unsaturated regime, θ_D is 0.65–0.69 and the naive rate is 0.12–0.17 (`results/summary_v2.txt`).

**Table 1 (round 0).** Bias of each estimator, averaged over 10 organisms. LB valid = fraction of runs with lower bound ≤ θ_D, shown without slack and, in brackets, with the round-0 +0.01 slack.

| config | θ_D | naive | IW | full rewrite (SE) | RDE | probe steer | rewrite+steer | LB valid | UB covers |
|---|---|---|---|---|---|---|---|---|---|
| κ=0, ρ=0 | 0.708 | −0.001 | +0.013 | +0.016 (0.009) | +0.004 | −0.070 | −0.014 | 0.4 (0.5) | 1.00 |
| κ=2, ρ=0 | 0.897 | −0.510 | −0.248 | +0.002 (0.004) | +0.061 | −0.270 | +0.009 | 0.7 (1.0) | 1.00 |
| κ=4, ρ=0 | 0.938 | −0.800 | −0.338 | −0.000 (0.002) | −0.127 | −0.324 | +0.006 | 0.8 (1.0) | 1.00 |
| κ=8, ρ=0 | 0.950 | −0.916 | −0.461 | +0.003 (0.002) | −0.071 | −0.806 | −0.001 | 0.8 (1.0) | 1.00 |
| subtle, κ=4 | 0.817 | −0.476 | −0.244 | +0.008 (0.004) | −0.015 | −0.477 | −0.004 | 0.7 (0.9) | 1.00 |
| ρ=0.25 vis | 0.947 | −0.807 | −0.325 | −0.052 (0.010) | −0.062 | −0.527 | −0.037 | 1.0 (1.0) | 1.00 |
| ρ=0.25 lat | 0.940 | −0.835 | −0.350 | −0.060 (0.009) | −0.061 | −0.327 | −0.020 | 1.0 (1.0) | 0.00 |
| ρ=0.5 vis | 0.932 | −0.820 | −0.328 | −0.272 (0.033) | −0.043 | −0.484 | −0.160 | 1.0 (1.0) | 1.00 |
| ρ=0.5 lat | 0.933 | −0.829 | −0.622 | −0.271 (0.029) | −0.272 | −0.393 | −0.097 | 1.0 (1.0) | 0.10 |
| ρ=0.75 vis | 0.930 | −0.792 | −0.341 | −0.552 (0.036) | −0.370 | −0.498 | −0.379 | 1.0 (1.0) | 1.00 |
| ρ=0.75 lat | 0.941 | −0.784 | −0.685 | −0.522 (0.015) | −0.523 | −0.605 | −0.389 | 1.0 (1.0) | 0.00 |

### 6.2 Full rewrites when every cue is rewritable

At ρ = 0 the full rewrite has small bias in every configuration:
- round 0: within ±0.016 (Table 1);
- revision: −0.006 saturated, +0.002 unsaturated, −0.023 with the interaction detector;
- token transformers: −0.000 (SD across organisms 0.013).

The auditor's residual AUC (D1) is about 0.99 throughout (0.94 for the transformers), because the visible non-rewritable cues still differ between contexts even though the model ignores them at ρ = 0. This is a false alarm of exactly the kind Prop. 3(iii) predicts.

Partial, auditor-ranked rewrites are costly when the model relies on cues the auditor ranks low (round 0, `diagnostics.json`). With 16 of 36 features rewritten, the bias is −0.023 in the standard world and −0.088 in the subtle-cue world.

### 6.3 The lower bound: where it is trivially valid, and where it is not

The reviewer is right that at ρ > 0, with an exact or under-shooting rewriter, the bound holds by a wide margin in every run (100% in both regimes, Fig. 4 and `summary_v2.txt`). This is near-tautological given the design, and we do not count it as evidence. The informative cases are below.

**(i) Small A2/A3 violations by learned organisms.** At ρ = 0 the rewrite is essentially unbiased, so the bound should hold in about 95% of runs if the item-level SE captured all the error. Without slack, it holds in:
- round 0: 70–80% of evaluation-aware runs (Table 1);
- revision: 100% saturated, **53%** unsaturated, 90% with the interaction detector, 67% for the transformers.

The cause is a per-organism *bias*, not sampling noise. Across organisms, the SD of the rewrite error is 0.010 (saturated) and 0.024 (unsaturated), against a mean item-level SE of 0.004 and 0.005. The learned organisms pick up small spurious and non-monotone dependences on the cues the rewrite leaves in place. The largest excess of the rewrite estimate over θ_D in any ρ = 0 organism is 0.035. The κ = 0 controls show the same effect: validity is 40–70%, because the organism has no reason to be monotone at all.

*Consequence:* A2 and A3 hold only approximately in learned models. A valid report needs an explicit tolerance δ, reported as θ_R − δ. In our organisms δ = 0.04 would have been enough, but we have no principled way to set δ for an LLM (§8).

**(ii) Over-shooting rewriters break the bound, as predicted (Fig. 4).**
- Unsaturated, ρ = 0: mild over-shoot bias +0.146, full over-shoot +0.178; the bound is valid in 0% of runs. At ρ = 0.25: valid in 0–47% of runs.
- Interaction detector, ρ = 0: +0.172 and +0.228.
- Transformers, ρ = 0: +0.163 and +0.225; at ρ = 0.25: +0.038 and +0.131 (0% valid).
- When ρ ≥ 0.5, the leftover non-rewritable evidence more than offsets the over-shoot, and the bound happens to hold.
- In the *saturated* regime over-shoot is almost invisible (+0.006 to +0.011). The policy is already at its ceiling. Saturation therefore hides violations, which is a second reason not to rely on saturated simulations.

**(iii) Task-altering rewriters.** A rewriter that *softens* the task gives a valid but loose bound (bias −0.17 to −0.64 in the monotone organisms). A rewriter that *intensifies* the task breaks the bound at ρ = 0: bias +0.040 saturated and +0.038 unsaturated, with the bound valid in 0% and 20% of runs. This is outside Prop. 2, which assumes the task is preserved.

**(iv) A deliberate A2 violation (tent policy).** The perfect rewrite overestimates θ_D by +0.085 at ρ = 0.25 and +0.093 at ρ = 0.5; the bound is valid in 0% of runs. Nothing the auditor observes reveals this (see §6.5).

### 6.4 Bias grows with non-rewritable evidence; auditor-space bounds fail when cues are latent

In both regimes the full-rewrite bias grows with ρ:
- saturated: −0.006, −0.06, −0.21/−0.25, −0.47/−0.53;
- unsaturated: −0.000, −0.14/−0.16, −0.31/−0.30, −0.42/−0.41 (visible/latent).

The trained organisms follow the policy-level curve (round 0, Fig. 1).

The round-0 auditor-space Lipschitz upper bound covered θ_D in 0–10% of runs when the cues were latent. *Correction to framing:* this failure is guaranteed by construction. When the auditor cannot see the non-rewritable cues, the auditor-score gap after a full rewrite is zero, so the "upper bound" collapses onto the rewrite estimate. We now present it only as an illustration of Prop. 3(ii), not as evidence.

### 6.5 Can the auditor check A3? (Prop. 4 in practice)

Take all monotone organisms with κ > 0 and all six rewriters (1,380 run–rewriter pairs), and call a run a violation when the rewrite exceeds θ_D by more than 0.02. The benign-traffic A3 check:

| rewriter | violations (> θ_D + 0.02) | flagged |
|---|---|---|
| overshoot_full | 68 | 60 |
| overshoot_mild | 53 | 35 |
| task_intensify | 33 | **0** |
| perfect | 5 | 0 |
| partial, task_sanitise | 0 | 0 (no flags at all) |

Source: `results/analysis_v2.json` and the inline breakdown in `code/analyze_v2.py` output.

The check raised 81 further flags on over-shooting rewriters without a large behavioural violation. Most are in the saturated regime, where the belief really did over-shoot (A3 really failed) but the ceiling hid the effect on behaviour. These are true A3 failures, not false alarms. The check never flagged the perfect, partial or task-sanitising rewriters.

The check fails in three ways, each predicted by Prop. 4:
1. It cannot see **task changes**: 0 of 33 task-intensifying violations were flagged.
2. It cannot see **A2 violations**: in the tent organisms it raised no flags for the perfect rewriter, although the bound was *valid* in only 40%, 0% and 0% of those runs at ρ = 0, 0.25 and 0.5.
3. In the **transformers**, a single probe direction misses net over-shoot when residual non-rewritable evidence offsets it along the probe axis but not in behaviour. At ρ = 0.25, mild over-shoot breaks the bound in 6 of 6 organisms (+0.038) but is flagged in 0 of 6 (`results/summary_token.txt`). This is the multi-dimensional caveat of Prop. 2′.

### 6.6 Diagnostics within configuration (corrects the round-0 claim)

The reviewer's re-analysis of round 0 is correct, and we reproduced it (`results/analysis_v2.json`, key `round0_*`):
- within the visible ρ > 0 group, D1 correlates with |bias| at Spearman 0.49 (p = 0.006);
- D2's within-cell Spearman ranges from −0.54 to +0.45, and its ranges at ρ = 0 (0.898–0.956) and ρ = 0.25 (0.892–0.973) overlap.

The pooled p = 2.4 × 10⁻²⁰ treated 70 runs as independent while ρ was a design variable, and we withdraw it. The revision grid uses 15 seeds per cell and stratified tests:

**Table 2.** Diagnostics vs θ_D − full rewrite (`results/summary_v2.txt`). Cell = Spearman over 7 configuration means. Within = mean within-configuration Spearman, [range], stratified-permutation p. R² = within-configuration R² after demeaning.

| diagnostic | sat: cell | sat: within (p) | sat R² | unsat: cell | unsat: within (p) | unsat R² |
|---|---|---|---|---|---|---|
| D1 auditor residual AUC | −0.32 | +0.08 [−0.31, 0.55] (0.22) | 0.02 | −0.29 | −0.01 [−0.32, 0.26] (0.55) | 0.00 |
| D2 probe residual AUC, L1 | +0.93 | +0.22 [−0.12, 0.72] (0.018) | 0.00 | +0.93 | +0.07 [−0.38, 0.44] (0.25) | 0.01 |
| D2 probe residual AUC, L2 | +0.96 | +0.12 (0.11) | 0.00 | +0.96 | +0.18 (0.038) | 0.03 |
| D6 projection gap, L1 | +1.00 | +0.08 (0.20) | 0.00 | +0.96 | +0.00 (0.49) | 0.00 |
| D3 steering residual, L1 | +0.96 | +0.23 (0.008) | 0.01 | +0.71 | +0.26 (0.007) | 0.07 |
| D4 MRDE − full | +1.00 | +0.56 [0.24, 0.88] (<0.001) | 0.31 | +0.43 | +0.03 (0.36) | 0.01 |
| D5 mean-matching residual, post-ReLU | +0.64 | +0.03 (0.36) | 0.00 | +1.00 | +0.22 (0.018) | 0.06 |
| D5p mean-matching residual, pre-activation* | +1.00 | +0.96 (<0.001) | 1.00 | +1.00 | +0.96 (<0.001) | 0.96 |

\*D5p exploits a structural feature of the toy (see below).

What the table shows:
1. **No diagnostic based on generic representations is useful within a configuration.** D2 and D6 rank configurations almost perfectly (cell Spearman 0.93–1.00), but within a configuration they explain at most 3% of the variance in bias. A single audit cannot use them to tell a bias of 0 from one of 0.06. The round-0 statement that D2 "tracks bias" holds only *across designs that differ in ρ*.
2. **D1 is uninformative in either sense.**
3. **Behaviour-based white-box residuals** (D3, D4, D5) are weakly informative within a configuration in some regimes, but not consistently. D4 reaches R² = 0.31 in the saturated regime and ≈ 0 in the unsaturated one.
4. **The one diagnostic that works (D5p)** matches the *pre-activation* mean of the first layer. There, in our MLP, context enters exactly additively (the first layer is linear in the input), so A5 holds by construction and mean matching transplants the deployment context's average contribution. It identifies θ_D to within 0.003–0.013 MAE. This shows what Prop. 4(b) needs, namely a representation in which context enters separably. It is *not* evidence that such a representation exists in an LLM. We did not test an analogue in the token transformers, which have no such layer.

The transformers repeat the pattern: D2 is 0.63 at ρ = 0 versus 0.83–0.86 at ρ > 0, but its within-cell Spearman is −0.14, +0.94 and −0.37 with n = 6 per cell.

### 6.7 Stronger white-box baselines (estimators)

MAE of each estimator, pooling visible and latent cues (`summary_v2.txt`):

| regime, ρ | full rewrite | steer L1 (benign tgt) | steer L1 (risky tgt) | steer L2 (risky tgt) | mean-match post-ReLU | MRDE | mean-match pre-act.* |
|---|---|---|---|---|---|---|---|
| sat 0 | 0.008 | 0.008 | 0.010 | 0.007 | 0.026 | 0.021 | 0.004 |
| sat 0.25 | 0.057 | 0.036 | 0.039 | 0.040 | 0.111 | 0.023 | 0.003 |
| sat 0.5 | 0.229 | 0.118 | 0.118 | 0.176 | 0.271 | 0.032 | 0.003 |
| sat 0.75 | 0.500 | 0.355 | 0.356 | 0.441 | 0.454 | 0.217 | 0.003 |
| unsat 0 | 0.021 | 0.040 | 0.043 | 0.033 | 0.083 | 0.098 | 0.005 |
| unsat 0.25 | 0.146 | 0.090 | 0.103 | 0.108 | 0.175 | 0.060 | 0.008 |
| unsat 0.5 | 0.305 | 0.190 | 0.203 | 0.245 | 0.225 | 0.124 | 0.010 |
| unsat 0.75 | 0.415 | 0.318 | 0.318 | 0.349 | 0.280 | 0.261 | 0.012 |

\*Relies on the toy's additive first layer; see §6.6.

The stronger variants the reviewer asked for (risky-adjusted targets, layer 2, full-subspace mean matching) do *not* improve on single-direction layer-1 steering after rewriting. Steering after rewriting removes about 25–50% of the residual bias but never closes it at large ρ.

MRDE, the belief-curve extrapolation the reviewer suggested, has the lowest error of the non-oracle estimators at moderate ρ. Its errors are two-sided, however:
- +0.082 at unsaturated ρ = 0;
- +0.118 and +0.090 with the interaction detector;
- **+0.167 and +0.282 for the tent organism** (`summary_v2.txt`).

Extrapolating a belief curve gives up the one-sided guarantee that makes the rewrite bound useful. We therefore recommend it only as a sensitivity analysis next to the lower bound, never in its place.

## 7 Discussion

### 7.1 What evaluators should report (revised)

1. **Report realism-rewritten rates as lower bounds, with a stated tolerance, and only for task-preserving rewrites.** Under A2–A3 this is valid (Prop. 2). In learned organisms A2 and A3 hold only approximately, so state θ_R − δ. Over-shooting and task-intensifying rewrites break the bound. A rewrite that shows *no* behaviour change does not show that deployment behaviour equals evaluation behaviour (Prop. 1).
2. **Audit the rewriter for over-shoot with benign traffic, and audit task preservation separately.** A probe-based comparison of rewritten benign items with benign deployment traffic detected most over-shoot violations in our MLP organisms. It is blind to task changes and to A2 violations, and a single projection can miss over-shoot in a multi-dimensional belief. *(This replaces the round-0 recommendation to "certify realism through model activations", which rested on a pooled correlation that does not hold within configuration.)*
3. **Do not certify realism with the rewriter's own classifier, and do not treat a probe residual AUC as a per-audit certificate.** Neither predicted bias within a configuration in our organisms. A rising probe residual AUC across evaluation *designs* is at most a coarse ordering signal.
4. **Inventory and engineer away non-rewritable cues.** Bias scales with non-rewritable evidence (Corollary). Changing how evaluations are built, for example with production-derived evaluations (OpenAI), is the only route to shrinking the Prop. 1 interval without white-box assumptions.
5. **Check the direction of A2 for each target behaviour.** Abdelnabi & Salem (2025) and Li et al. (2026) report cases where evaluation-like framing *raises* harmful execution. For such behaviours, rewritten rates are *upper* bounds.
6. **Do not use context-space importance weighting.** Weighting over *tasks* is a different matter (Prop. 5).

### 7.2 Relation to the mixed empirical picture

Prop. 3 says only the *behaviour-relevant* belief matters. Verbalisation, and generic probe readouts, can move without that statistic moving, or stay put while it moves. In our organisms, probe-readable evaluation information was largely irrelevant to behaviour: D2 was about 0.93 at zero bias and did not predict bias within a configuration. This is consistent with reports that awareness measures and behavioural consequences come apart (Knecht et al., 2026; arXiv:2608.21766; arXiv:2606.23583; Li et al., 2026).

### 7.3 Pre-registered protocol for LLMs (not executed here)

**Subjects.**
- (a) The released type-hint organism of Hua et al. (2025): Llama 3.3 Nemotron Super 49B fine-tuned to write type hints in evaluation contexts. It is released as `timhua/wood_v2_sftr4_filt`, with code at `github.com/tim-hua-01/steering-eval-awareness-public`. Its deployment contexts are known, which gives ground truth.
- (b) An un-fine-tuned open-weight instruction model of the kind studied in Nguyen et al. (2025).

**Manipulations.**
- R0: none.
- R1: an LLM rewrite of surface cues.
- R1+: R1 deliberately over-shooting, with deployment cues inserted at above-deployment frequency.
- R1t: R1 with a task-altering instruction.
- R2: R1 plus a human-audited rewrite of structural cues.

**Pre-registered predictions.**
- P1: R1 and R2 rates are ≤ the ground-truth deployment rate, within a pre-stated δ = 0.03 (absolute type-hint rate). R1+ violates the bound in the direction predicted by Prop. 2.
- P2: the benign-traffic A3 check (Prop. 4(b)), computed at the layer used by Hua et al.'s steering vector, flags R1+ and does not flag R1.
- P3: *Evaluated within subject and rewrite, across items:* the per-item gap between deployment and rewritten behaviour correlates more strongly with a behaviour-based residual (steering residual or MRDE residual) than with the probe residual AUC or the rewriter's realism score.
- P4: steering after R2 reduces the remaining gap more than steering on R0.

**Analysis.** Item-level paired bootstrap, with all thresholds fixed before data collection. P3 is assessed *within* subject and rewrite, never by pooling across designs, which corrects the round-0 analysis.

## 8 Limitations

- **No LLM experiments.** The reviewer's highest-leverage request was to run P1–P2 on the released Hua et al. organism. We could not. It is a 49B-parameter model (about 98 GB in bf16). Our environment has 4 CPU cores, 16 GB of RAM, no GPU, and no network access to the Hugging Face hub (the connection is refused by the sandbox proxy). We did not substitute a different, smaller "LLM" experiment and present it as equivalent. Our second architecture is a small transformer trained from scratch on synthetic token sequences, and it is *not* a language model.
- **Simulation design still favours the theory.** The non-rewritable cues are stochastically dominant by construction, and organisms are trained on a monotone policy except in the tent variant. At ρ > 0 the lower bound holds by a wide margin for exact rewriters. We treat those runs as non-evidence, and base our claims on the over-shoot, task-alteration, tent and ρ = 0 cases.
- **Approximate assumptions, unknown tolerance.** Learned organisms violate A2 and A3 slightly (bias up to +0.035 at ρ = 0). We have no method to bound δ for a real model short of deployment data.
- **The A3 check is partial.** It needs A5 and a good proxy for σ. It misses task changes and A2 violations, and with a multi-dimensional belief a single direction can miss over-shoot.
- **Diagnostics are not calibrated within a configuration.** None of the probe-based diagnostics we tried certifies a single audit. The one that worked (pre-activation mean matching) relies on an additive layer that LLMs are not known to have.
- **Task shift is handled only by a bound.** Prop. 5's TV term can be large, and we do not simulate task shift.
- **Steering baselines.** We tried four steering and matching variants and one extrapolation method. Richer interventions, such as multi-layer or learned interventions, may do better.
- **Literature currency and attribution.** arXiv.org was not reachable from our sandbox. Details of 2026 preprints were verified through search engines and conference and aggregator pages, not the papers themselves. Where full author lists could not be retrieved, we say so in the references.

## 9 Conclusion

From the evaluator's side, whether a model behaves the same in evaluation and deployment is an identification question. The answer depends on which belief levels the evaluator can reach. Four conclusions follow:
- Realism-rewritten evaluations give valid lower bounds on harmful deployment behaviour under monotonicity and evidence removal.
- Those assumptions are untestable in general, and testable on benign traffic only when task and context are processed separably and the rewrite leaves the task intact.
- In model organisms the bound failed exactly where the theory said it would: with over-shooting rewriters, with task-intensifying rewriters, and with non-monotone behaviour. It also failed slightly whenever learned organisms deviated from monotonicity.
- Probe-based realism diagnostics ranked evaluation designs but did not certify individual audits.

Whether any of this transfers to LLMs is open. §7.3 gives a protocol for finding out.

---

## Reproducibility

Code: `/home/user/AI-agents/research/papers/p035/code/`. Outputs: `/home/user/AI-agents/research/papers/p035/results/`. The commands in `README.md` regenerate every number and figure. Fixed seeds; about 4 + 21 + 6 minutes on 4 shared CPU cores.

Data provenance: every number in §6 comes from our own runs.
- `summary.txt`, `summary.json`, `diagnostics.json`, `raw.json`: round 0.
- `summary_v2.txt`, `analysis_v2.json`, `raw_v2.json`: revision MLP grid.
- `summary_token.txt`, `summary_token.json`, `raw_token.json`: transformers.

Figures: `fig1`–`fig3` (round 0), `fig4_rewriter_stress.png`, `fig5_diagnostics_within.png`. Statements about other papers come from their abstracts or listings, as cited.

## References

- Abdelnabi, S., & Salem, A. (2025). The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness (v1/v2 title: Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models). arXiv:2505.14617. NeurIPS 2025.
- Chaudhary, M., et al. (2025). Evaluation Awareness Scales Predictably in Open-Weights Large Language Models. arXiv:2509.13333. NeurIPS 2025.
- Evaluation Awareness in Language Models: Representation, Verbalization, and Control. (2026). arXiv:2608.21766. [Author list could not be retrieved from our sandbox; verified by title, ID and abstract via search.]
- Evaluation Awareness Is Not One Capability: Evidence from Open Language Models. (2026). arXiv:2606.23583. [Author list could not be retrieved from our sandbox; verified by title, ID and abstract via search.]
- Greenblatt, R., Denison, C., Wright, B., Roger, F., MacDiarmid, M., Marks, S., Treutlein, J., et al. (2024). Alignment Faking in Large Language Models. arXiv:2412.14093.
- Hua, T. T., Qin, A., Marks, S., & Nanda, N. (2025). Steering Evaluation-Aware Language Models to Act Like They Are Deployed. arXiv:2510.20487. ICLR 2026.
- Hubinger, E., et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566.
- Knecht, et al. (2026). Evaluation Awareness in Language Models Has Limited Effect on Behaviour. arXiv:2605.05835. NeurIPS 2026 poster. [First author as cited by arXiv:2610.22333; the full author list could not be retrieved from our sandbox.]
- Laine, R., Chughtai, B., Betley, J., Hariharan, K., Scheurer, J., Balesni, M., Hobbhahn, M., Meinke, A., & Evans, O. (2024). Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. arXiv:2407.04694. NeurIPS 2024 Datasets and Benchmarks.
- Li, C., Zhang, T. J., Zhang, J., Jin, Z., Abdelnabi, S., & Andriushchenko, M. (2026). Decomposing and Measuring Evaluation Awareness. arXiv:2605.23055. [Corrected author list: the previous version wrongly included F. Tramèr.]
- Manski, C. F. (2003). Partial Identification of Probability Distributions. Springer. doi:10.1007/b97478.
- Meinke, A., Schoen, B., Scheurer, J., Balesni, M., Shah, R., & Hobbhahn, M. (2024). Frontier Models are Capable of In-context Scheming. arXiv:2412.04984.
- Needham, J., Edkins, G., Pimpale, G., Bartsch, H., & Hobbhahn, M. (2025). Large Language Models Often Know When They Are Being Evaluated. arXiv:2505.23836.
- Nguyen, J., Hoang, K., Attubato, C. L., & Hofstätter, F. (2025). Probing and Steering Evaluation Awareness of Language Models. arXiv:2507.01786. ICML 2025 Workshops.
- OpenAI Alignment Team. Sidestepping Evaluation Awareness and Anticipating Misalignment with Production Evaluations. alignment.openai.com/prod-evals/. [Blog post; the exact title the reviewer gave ("Eval-production behavior gaps in agent safety and the role of eval-awareness") could not be found, so we cite the closest verified OpenAI post on the same topic.]
- Schoen, B., et al. (2025). Stress Testing Deliberative Alignment for Anti-Scheming Training. arXiv:2509.15541.
- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90(2), 227–244.
- Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
- van der Weij, T., Hofstätter, F., Jaffe, O., Brown, S., & Ward, F. R. (2024). AI Sandbagging: Language Models can Strategically Underperform on Evaluations. arXiv:2406.07358. ICLR 2025.
- Xiong, L., Bhargava, N., Chang, J., Hong, J., Liu, H., Sharma, V., & Zhu, K. (2025). Probe-Rewrite-Evaluate: A Workflow for Reliable Benchmarks and Quantifying Evaluation Awareness. arXiv:2509.00591.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
