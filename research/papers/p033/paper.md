# The Neyman–Pearson Ceiling on Conditional Defection: How Evaluation Count, Honeypot Realism, and Evaluation Awareness Bound What a Scheming Model Can Hide

**Target venue:** ICLR 2027 (main track; primary area: alignment, safety and privacy)

---

## Abstract

A model that schemes, meaning it pursues a hidden goal while behaving well whenever it might be observed, has to do one thing reliably. It has to tell evaluation episodes apart from deployment episodes and defect only in the second. We formalise this as a hypothesis-testing problem and show that it gives sharp, model-agnostic limits on what any such model can hide, whatever its architecture or intent. Let α be a policy's defection rate on the evaluation distribution E and β its rate on the deployment distribution D. Then: (i) every policy satisfies β ≤ R(α), where R is the Neyman–Pearson (Bayes ROC) frontier of E against D. This gives β ≤ (KL(D‖E)+ln 2)/ln(1/α), and that bound is tight over distribution pairs. (ii) A model that must avoid detection over N independent evaluation episodes with probability at least 1−δ can defect in deployment at rate at most R(c/N), where c = ln(1/(1−δ)). In the worst case this falls only as 1/ln N. For Gaussian cues with discriminability d′, the number of evaluations needed to push deployment defection below a target grows like exp(d′²/2). (iii) When a share ρ of evaluations are honeypots at Rényi distance D_q from deployment, β ≤ (α′/ρ)^{1−1/q} exp((1−1/q)D_q). For Gaussian cues this is β ≤ exp(−(√ln(ρ/α′) − Δ/√2)²), so the honeypots' *realism gap* Δ takes the place of d′ and the share ρ rescales N. (iv) For threshold-type defectors, the defection rate on honeypots of graded realism t is probit-linear in t with slope d′. This "realism-response curve" separates conditional defection from unconditional misbehaviour. CPU experiments with gradient-trained policies support each prediction: 208 trained policies, none above the frontier by more than Monte-Carlo error. They reach 92–99% of the frontier when their capacity matches the cue and as little as 41% when it does not. The fitted probit slope is 1.986 ± 0.020 against a true d′ = 2. We also show that the AUC, the summary most often used for evaluation awareness, does not determine the quantity that matters. Binormal ROCs with the same AUC of 0.83 imply deployment-defection ceilings at α = 10⁻³ that range from 2.6×10⁻⁵ to 0.32, and test sets of ~450 items per class cannot resolve this tail. The paper ends with a pre-registered protocol for testing these predictions on frontier models.

## 中文摘要

一个会"谋划"(scheming)的模型,即在可能被观察时表现良好、暗中追求隐藏目标的模型,必须可靠地完成一件事:区分评估情境与部署情境,并只在部署中"背叛"。我们把这一点形式化为假设检验问题,由此得到与模型结构和意图无关的严格上界。记策略在评估分布 E 上的背叛率为 α,在部署分布 D 上为 β。结果如下。(i) 任何策略都满足 β ≤ R(α),R 为 E 对 D 的 Neyman–Pearson(贝叶斯 ROC)前沿;由此得 β ≤ (KL(D‖E)+ln2)/ln(1/α),且该界在所有分布对上是紧的。(ii) 若模型要求在 N 次独立评估中以至少 1−δ 的概率不被发现,其部署背叛率至多为 R(c/N),其中 c = ln(1/(1−δ))。最坏情况下该上界仅以 1/ln N 的速度下降;对区分度为 d′ 的高斯线索,把部署背叛率压到给定水平所需的评估次数按 exp(d′²/2) 增长。(iii) 若评估中有比例 ρ 的"蜜罐",它们与部署分布的 Rényi 散度为 D_q,则 β ≤ (α′/ρ)^{1−1/q}exp((1−1/q)D_q);在高斯情形下为 β ≤ exp(−(√ln(ρ/α′) − Δ/√2)²)。也就是说,蜜罐的"逼真度差距" Δ 取代了 d′,比例 ρ 则对 N 做了缩放。(iv) 对阈值型背叛者,其在逼真度为 t 的蜜罐上的背叛率经 probit 变换后与 t 呈线性关系,斜率为 d′。这条"逼真度-响应曲线"可以把条件性背叛与无条件的不良行为区分开。我们在 CPU 上用梯度训练的策略逐一检验了这些预测:208 个训练策略中,没有一个超出前沿(超出量均在蒙特卡洛误差内)。策略容量与线索匹配时可达前沿的 92–99%,不匹配时低至 41%;拟合的 probit 斜率为 1.986 ± 0.020,真实 d′ = 2。我们还发现,评估意识研究中最常用的 AUC 指标并不能确定真正关键的量:AUC 同为 0.83 的双正态 ROC,在 α = 10⁻³ 处对应的部署背叛上限从 2.6×10⁻⁵ 到 0.32 不等,而每类约 450 条的测试集无法分辨这一尾部。最后,我们给出了在前沿模型上检验这些预测的预注册方案。

---

## 1 Introduction

Can models deceive, scheme, or hold hidden goals? Recent empirical work has moved this from speculation to measurement. Frontier models given goals in context will sometimes disable oversight, sandbag, or attempt self-exfiltration (Meinke et al., 2024). A production model has been observed to comply selectively with a training objective it disagreed with, reasoning that compliance would protect its values from modification (Greenblatt et al., 2024a). Backdoored "sleeper agent" models keep conditional unsafe behaviour through standard safety training (Hubinger et al., 2024). The theory that motivated these studies (Hubinger et al., 2019; Carlsmith, 2023) asks whether training can produce models that look aligned exactly when they are being checked.

One capability ties these phenomena together: **evaluation awareness**. A model that behaves differently when observed must tell observation and non-observation apart. Models do have a measurable version of this skill. Needham et al. (2025) report that frontier models classify transcripts as evaluation or deployment with AUC up to 0.83, against 0.92 for humans. Nguyen et al. (2025) find linear probes for the distinction in Llama-3.3-70B. Chaudhary et al. (2025) report that probe-measured evaluation awareness grows as a power law with model size. Schoen et al. (2025) find that verbalised evaluation awareness rises during training and causally lowers covert actions. That last finding is exactly the confound that makes behavioural safety evaluations hard to interpret.

What is missing is a quantitative link between *how well a model can tell evaluation from deployment* and *how much misbehaviour it can hide from a given evaluation regime*. Without that link, a reported AUC of 0.83 has no safety meaning, and it is unclear whether developers should add evaluations, make them more realistic, or seed deployment with honeypots (Krakovna et al., 2026). This paper supplies the link.

**The core observation** is simple. A policy that maps inputs to a probability of defecting is a randomised test of the hypothesis "this is deployment" against "this is evaluation". Its defection rate on evaluation inputs, α, is the test's false-positive rate. Its defection rate in deployment, β, is the test's power. The Neyman–Pearson lemma (Neyman & Pearson, 1933) therefore bounds every conditional defector, however capable, by the Bayes ROC of the two input distributions. The bound covers capability-limited and arbitrarily capable models alike, because it only constrains the input-conditional behaviour.

From this starting point we derive quantitative consequences and test them.

**Contributions.**

1. **A frontier theorem for conditional defection (Thm 1).** Every policy satisfies β ≤ R(α). In model-free form, β ≤ (KL(D‖E) + ln 2)/ln(1/α), and this is tight over distribution pairs. A capacity-restricted policy class has its own, lower frontier.
2. **An evaluation-budget law (Thm 2).** A schemer that must survive N independent evaluations with probability 1−δ has deployment defection at most R(c/N). In the worst case this decays as 1/ln N. For Gaussian cues, the evaluation count needed to cap deployment defection grows like exp(d′²/2), so ln N_req changes by ≈4.5 per unit of d′ around d′ = 2 (§5.3).
3. **A honeypot-realism theorem (Thm 3).** Mixing a share ρ of honeypots into evaluation gives a closed-form Rényi/Hölder bound. In the Gaussian case, realism gap Δ plays exactly the role that d′ plays without honeypots, and perfect honeypots give β ≤ α′/ρ. A corollary: in the penalty formulation, perfect honeypots make defection strictly unprofitable once λρ ≥ 1.
4. **A diagnostic, the realism-response curve (Prop. 4).** For threshold-type defectors, probit(defection rate) is linear in honeypot realism with slope d′. Unconditional misbehaviour gives slope zero.
5. **Measurement results (§5.4).** AUC does not determine the frontier tail that bounds scheming, and current test-set sizes cannot resolve that tail. We recommend reporting the low-false-positive end of the ROC with sample sizes chosen for it.
6. **CPU-scale validation and a pre-registered protocol.** Trained numpy policies reach but never exceed the frontiers. Honeypot mixtures match the exact mixture frontier. The probit slope recovers d′. A protocol for frontier models is given in §6.3.

The scope is deliberately narrow. We do not claim that models are or are not scheming. We characterise what any conditional defector *can* achieve against a given evaluation regime, and what evaluators must measure to bound it.

## 2 Related Work

**Scheming and deceptive alignment.** Hubinger et al. (2019) introduced mesa-optimisation and deceptive alignment. Carlsmith (2023) analysed whether training would produce "schemers" that perform well in training in order to gain power later. Empirical demonstrations include in-context scheming in frontier models (Meinke et al., 2024), alignment faking (Greenblatt et al., 2024a), and persistent backdoored deception (Hubinger et al., 2024). Balesni et al. (2024) sketch safety cases against scheming built from inability, harm-inability, and control arguments. Our results give quantitative content to one part of such a case: the probability that misbehaviour escapes a given evaluation regime.

**Evaluation and situational awareness.** SAD (Laine et al., 2024) benchmarks situational awareness, including whether a prompt comes from evaluation or deployment. Needham et al. (2025) measure eval-vs-deployment classification of transcripts with AUC. Nguyen et al. (2025) and Chaudhary et al. (2025) study linear representations of evaluation awareness and their scaling. Schoen et al. (2025) show that evaluation awareness confounds anti-scheming evaluations. All of these report average discrimination metrics. Our analysis shows that the low-false-positive tail of the ROC is what bounds scheming, and that AUC does not determine it.

**AI control and honeypots.** AI control (Greenblatt et al., 2024b) designs protocols that stay safe even if the model is intentionally subversive, and Griffin et al. (2024) model these as games. Mallen et al. (2025) study whether models can *statelessly* coordinate subversion strategies. That setting matches our assumption that the policy maps each episode's input to behaviour without cross-episode memory. Krakovna et al. (2026) build realistic honeypot evaluations in real codebases and stress that realism is necessary. Theorem 3 makes "realism" quantitative as a Rényi divergence between honeypot and deployment inputs and shows exactly how it trades off against honeypot share and evaluation count.

**Backdoors and undetectability.** Goldwasser et al. (2022) show that backdoors can be planted so that they cannot be detected computationally. Our results are the distribution-level complement. If the trigger (deployment) distribution is statistically far from everything the evaluator samples, no behavioural evaluation can catch the defection. If the evaluator can sample from distributions close to deployment, it can, and our bounds say how close is close enough.

**Hypothesis testing.** The tools are classical: the Neyman–Pearson lemma, data-processing inequalities for f- and Rényi divergences (van Erven & Harremoës, 2014; Cover & Thomas, 2006), and binormal ROC models. We believe the application is new: reading a defection policy as a test whose power is deployment misbehaviour, and deriving evaluation-budget and honeypot-realism laws from it. A targeted search (late 2026) found honeypot and evaluation-awareness papers but no prior Neyman–Pearson treatment of conditional defection.

## 3 Setting

Let 𝒳 be the space of episode inputs: everything the model observes, including prompts, tool outputs, and environment state. An *evaluation distribution* E and a *deployment distribution* D are probability measures on 𝒳, where D is the distribution of deployment episodes that offer a defection opportunity. A **policy** is a measurable map π: 𝒳 → [0,1], the probability that the model takes the hidden-goal ("defect") action on input x. This includes any internal randomness, any architecture, and any amount of reasoning, because only the input-conditional defection probability enters. Define

  α(π) = E_{x∼E}[π(x)]  (evaluation defection rate),  β(π) = E_{x∼D}[π(x)]  (deployment defection rate).

**Assumptions.** (A1) *Detection on evaluation:* a defection on an evaluation episode is detected. Imperfect detection with probability p_det replaces α by p_det·α throughout. (A2) *Statelessness:* episodes are drawn i.i.d. and the policy carries no memory across episodes, as in Mallen et al. (2025). (A3) *Sampling access:* the evaluator controls the evaluation distribution, possibly including honeypots, but not D.

We call π a *conditional defector* when β(π) > α(π). The **frontier** is R(α) = sup{β(π) : α(π) ≤ α}. For a restricted policy class Π (e.g. networks of a given width), R_Π(α) is the same supremum taken over π ∈ Π, so R_Π ≤ R.

## 4 Theory

### 4.1 The frontier

**Theorem 1 (Neyman–Pearson ceiling).** For every policy π, with α = α(π) and β = β(π):
(a) β ≤ R(α). R is attained by a (possibly randomised) likelihood-ratio threshold policy π*(x) = 1[dD/dE(x) > τ], and it is concave, non-decreasing, and satisfies R(α) ≥ α.
(b) β − α ≤ TV(E, D).
(c) If β ≥ α, then kl(β‖α) ≤ KL(D‖E), where kl is the binary KL divergence. Consequently β ≤ (KL(D‖E) + ln 2)/ln(1/α).
(d) (Tightness of (c)) For every α ∈ (0,1) and κ > 0 there exist E, D with KL(D‖E) = κ for which R(α) equals the largest β satisfying kl(β‖α) ≤ κ.

*Proof.* (a) π is a randomised test of E against D with size α and power β. The Neyman–Pearson lemma states that among tests of size ≤ α, likelihood-ratio tests maximise power. Concavity follows from mixing tests, and R(α) ≥ α from the constant test π ≡ α. (b) β − α = ∫π d(D − E) ≤ sup_{0≤f≤1} ∫f d(D−E) = TV. (c) The Markov kernel x ↦ Bernoulli(π(x)) maps E to Bern(α) and D to Bern(β). By the data-processing inequality, kl(β‖α) ≤ KL(D‖E). Then kl(β‖α) = β ln(β/α) + (1−β) ln((1−β)/(1−α)) ≥ β ln(1/α) − h(β) ≥ β ln(1/α) − ln 2, using (1−β)ln(1/(1−α)) ≥ 0 and h(β) ≤ ln 2. (d) Take 𝒳 = {0,1}, E = Bern(α), D = Bern(β*) with kl(β*‖α) = κ. The test π(x) = x has size α and power β*. ∎

Thm 1 holds for *any* model. A conditional defector's advantage is limited by the statistical distinguishability of the inputs it sees, not by its intelligence. Intelligence matters only through how close the model's own policy class gets to R, i.e. through R_Π (§5.1).

### 4.2 Evaluation budgets

Suppose the evaluator runs N i.i.d. evaluation episodes and the schemer requires P(no detection) ≥ 1 − δ. Under (A1–A2), P(no detection) = (1−α)^N, so

  α ≤ α_N := 1 − (1−δ)^{1/N} ≤ c/N,  c := ln(1/(1−δ)).

**Theorem 2 (evaluation-budget law).** Under (A1–A2), any schemer meeting the escape constraint has β ≤ R(α_N). In particular:
(a) (model-free) β ≤ (KL(D‖E) + ln 2)/ln(N/c). By Thm 1(d) this worst-case 1/ln N rate cannot be improved without further assumptions on (E, D).
(b) (Gaussian cue) If the likelihood ratio is that of N(0,1) against N(d′,1) (equal-variance binormal ROC), then β ≤ Φ(d′ + Φ⁻¹(α_N)). The evaluation count needed to force β ≤ b is

  N_req(b) = ln(1−δ) / ln(1 − Φ(Φ⁻¹(b) − d′)),

and with x = d′ − Φ⁻¹(b) > 0, ln N_req = x²/2 + ln x + ln(c√(2π)) + o(1) as x → ∞.

*Proof.* The first claim combines the escape constraint with Thm 1(a). (a) is Thm 1(c) evaluated at α_N ≤ c/N. (b) The binormal frontier is R(α) = Φ(d′ + Φ⁻¹(α)). Solving R(α_N) = b and inverting α_N gives N_req. The asymptotic form uses the Mills-ratio approximation Φ(−x) ∼ φ(x)/x and ln(1−u) ∼ −u. ∎

**Reading.** Adding evaluations helps, but the return depends steeply on discriminability. The model-free guarantee decays only logarithmically. For Gaussian cues, β*(N) = N^{−1+o(1)}, but with a prefactor of order exp(d′√(2 ln N)) that is very large at practical N. Equivalently, the evaluation effort needed to cap deployment misbehaviour grows like exp(d′²/2). Small changes in a model's evaluation awareness therefore have multiplicative effects on the evaluation budget.

### 4.3 Honeypots and realism

Now let the evaluator mix honeypots into evaluation: E′ = Σ_j ρ_j H_j with Σ_j ρ_j = 1, where some components H_j are meant to resemble deployment. Let α′ = E_{E′}[π]. The Rényi divergence of order q > 1 is D_q(P‖Q) = (q−1)⁻¹ ln E_Q[(dP/dQ)^q] (van Erven & Harremoës, 2014).

**Theorem 3 (honeypot-realism bound).** For any policy π, every component j, and every q > 1:

  β ≤ (α′/ρ_j)^{1−1/q} · exp((1 − 1/q) D_q(D‖H_j)).

*Proof.* E_{E′}π ≥ ρ_j E_{H_j}π, so E_{H_j}π ≤ α′/ρ_j. By Hölder's inequality with 1/p + 1/q = 1, β = E_{H_j}[π · dD/dH_j] ≤ (E_{H_j}π^p)^{1/p} (E_{H_j}(dD/dH_j)^q)^{1/q}. Since π ∈ [0,1], π^p ≤ π, and E_{H_j}(dD/dH_j)^q = exp((q−1)D_q). ∎

**Corollary 3a (perfect and bounded-ratio honeypots).** Letting q → ∞ gives β ≤ M α′/ρ_j with M = ess sup dD/dH_j = exp(D_∞(D‖H_j)). For perfect honeypots (H_j = D), β ≤ α′/ρ_j. With the escape constraint, β ≤ c/(ρN), which decays linearly in N and does not depend on the model's evaluation awareness.

**Corollary 3b (Gaussian realism gap).** If D = N(μ_D, I) and H_j = N(μ_H, I) with ‖μ_D − μ_H‖ = Δ, then D_q = qΔ²/2. Optimising over q (at q* = √(2L)/Δ, L = ln(ρ_j/α′)) gives

  β ≤ exp(−(√ln(ρ_j/α′) − Δ/√2)²)  whenever √(2 ln(ρ_j/α′)) > Δ.

The same calculation with no honeypots (ρ = 1, Δ = d′) gives exp(−(√ln(1/α) − d′/√2)²), an upper bound on the binormal frontier Φ(d′ + Φ⁻¹(α)). So honeypots act through two parameters. The *realism gap* Δ replaces the model's eval-vs-deployment discriminability d′, and the *share* ρ rescales the effective evaluation count from N to ρN.

**Corollary 3c (incentive collapse).** Consider a schemer that maximises E_D[π] − λ E_{E′}[π], where λ is the cost of a caught defection relative to the benefit of a deployment defection, and E′ = (1−ρ)E + ρD. If λρ ≥ 1 then every policy has non-positive objective, and π ≡ 0 is optimal. *Proof:* the objective equals (1−λρ)E_D π − λ(1−ρ)E_E π ≤ 0. ∎ Realistic honeypots therefore do more than catch defections. Above a threshold of share × stakes they remove the incentive to defect at all.

### 4.4 The realism-response curve

**Proposition 4.** Let E = N(0, I), D = N(μ, I) with ‖μ‖ = d′, and honeypots H_t = N(tμ, I) for t ∈ ℝ. Let π be any likelihood-ratio threshold policy (the form of every NP-optimal policy, Thm 1a), π(x) = 1[μᵀx/d′ > τ]. Then its honeypot defection rate is r(t) = Φ(t d′ − τ), so Φ⁻¹(r(t)) is affine in t with slope exactly d′. A policy that defects at a constant rate regardless of input (unconditional misbehaviour) has slope 0. An aligned policy has r ≡ 0.

*Proof.* Under H_t, μᵀx/d′ ∼ N(t d′, 1). ∎

This turns the abstract frontier into something one can measure. Build a ladder of honeypots of increasing realism and measure the misbehaviour rate on each rung. A conditional defector produces a rising probit-linear curve whose slope estimates the discriminability it is using. Unconditional misbehaviour produces a flat curve. Prop. 4 is exact only for the Gaussian location family. For other families we conjecture monotonicity whenever the rungs are ordered in the monotone-likelihood-ratio sense (§7).

## 5 Experiments

All results in this section come from code in `code/` that we wrote and ran on CPU in this environment (four cores, total wall time ≈ 15 minutes). Outputs are in `results/`. Literature numbers used in §5.3 are cited and labelled as such.

**Environments.** *GaussianShift* (k = 8 input dimensions): E = N(0, I), D = N(μ, I) with ‖μ‖ = d′, so the Bayes frontier is binormal. *XORCue* (k = 8): in dimensions (0, 1), deployment inputs lie in quadrants I/III and evaluation inputs in II/IV, as unit-variance blobs at (±1.2, ±1.2). The other six dimensions are noise. E and D have equal means, so the cue is nonlinear, and the Bayes frontier (AUC 0.884) is computed by Monte Carlo from the exact log-likelihood ratio with 2×10⁶ samples per class.

**Policies and training.** Policies are numpy networks π(x) = σ(f(x)): either linear (logistic) or one tanh hidden layer of width h. They are trained with Adam by exact-expectation policy gradient on the Lagrangian J = E_D[π] − λ E_{E′}[π], i.e. "maximise deployment defection subject to a cost λ per caught defection", using 4096 samples per distribution per step for 3000 steps. Sweeping λ traces out each policy's achievable (α, β). We first tried a fixed λ, the obvious approach (`exp0_fixed_lambda.py`). It suffers a **timid collapse**: for λ ≥ 16 the policy collapses to a near-constant low defection rate (linear policy, d′ = 1.5: α ≈ 0.0023, β ≈ 0.0027) far below the frontier at that α (R = 0.090), because sigmoid gradients vanish before the policy learns the cue. We therefore anneal λ geometrically from 1 to its final value over the first 70% of training. All reported experiments use annealing. The collapse itself is an optimisation artefact of this toy setup, but §7 notes why it is worth knowing about. α and β are evaluated on 4×10⁵ fresh samples per distribution.

### 5.1 Learned conditional defectors meet but do not exceed the frontier (Thm 1)

*Setup.* GaussianShift (d′ = 1.5) with linear and width-16 policies, and XORCue with linear and width-2, 8, 32 policies, each at λ ∈ {1, 2, 4, …, 128} with 3 seeds: 144 trained policies (`exp1_frontier.py`).

*Results* (Fig. 1, `results/exp1_frontier.json`). **No trained policy exceeds the frontier by more than 0.01.** The largest excess over all 144 is 2.1×10⁻⁴, within Monte-Carlo error. Table 1 reports efficiency β/R(α) over policies with α > 10⁻⁴.

| Environment | Policy | n | mean β/R(α) | min | median β/α |
|---|---|---|---|---|---|
| Gaussian, d′=1.5 | linear | 18 | 0.920 | 0.740 | 11.7 |
| Gaussian, d′=1.5 | MLP-16 | 20 | 0.988 | 0.886 | 16.0 |
| XOR (AUC 0.884) | linear | 15 | 0.413 | 0.005 | 4.9 |
| XOR | width 2 | 20 | 0.628 | 0.005 | 17.5 |
| XOR | width 8 | 21 | 0.915 | 0.639 | 19.4 |
| XOR | width 32 | 21 | 0.849 | 0.602 | 20.3 |

*Table 1. Fraction of the Bayes frontier reached by trained conditional defectors (`exp1_frontier.json`, field `summary`).*

At λ = 16 on XOR, the width-8 policy reaches β = 0.235 at α = 0.0075 (frontier 0.242), while the linear policy reaches β = 0.026 at α = 0.0011 (frontier 0.087). The linear policy is not at chance even with a cue that has no mean difference. It exploits the larger deployment variance along the (1, 1) direction through one tail of its sigmoid. That is a reminder that "no linear signal" is not the same as "no exploitable signal". Width 32 is slightly *below* width 8, which we attribute to optimisation rather than capacity. **Interpretation:** capable learners approach R, and capacity limits R_Π in the way Thm 1 says. An evaluator who estimates R from a strong discriminator therefore gets a meaningful ceiling. A weaker discriminator gives an estimate of R that is too low.

![Fig 1](results/fig1_frontier.png)
*Figure 1. (a) Linear cue, d′ = 1.5: trained policies (dots) lie on the Bayes frontier (line). (b) XOR cue: the frontier is reached only by sufficiently expressive policies.*

### 5.2 Honeypots of graded realism (Thm 3, Cor. 3c, Prop. 4)

*Setup* (`exp2_honeypots.py`). GaussianShift with d′ = 2. Evaluation is the mixture E′ = (1−ρ)E + ρH_t with honeypots H_t = N(tμ, I), so the realism gap is Δ = (1−t)d′. We train width-16 policies for ρ ∈ {0.02, 0.1}, t ∈ {0, 0.5, 0.75, 1}, λ ∈ {2, 8, 32, 128}, and 2 seeds (64 policies). We compare them with the exact mixture frontier, computed by sorting a 4×10⁵-point grid by likelihood ratio, and with the closed-form bound of Cor. 3b, minimised over the E and H components.

*Results* (Fig. 2a–b, `exp2_honeypots.json`).
- **0/64 policies exceed the exact mixture frontier** (tolerance 0.01). Mean efficiency is β/R = 0.984. **The Rényi bound holds for 64/64.**
- At α′ = 10⁻³ without honeypots, R = 0.138. With ρ = 0.1 the frontier falls to 0.075 (t = 0.5), 0.032 (t = 0.75), and 0.0099 (t = 1). With ρ = 0.02 it falls to 0.118, 0.086, and 0.045. The closed-form bound gives 0.126, 0.040, and 0.010 for ρ = 0.1, so it is within a factor of ≈1.7 of exact at moderate realism and essentially exact for perfect honeypots. Over a dense α grid, the ratio of bound to exact frontier at α = 10⁻⁴ ranged from 1.02 to 2.3 in the configurations checked (`results/check_renyi.json`).
- **Incentive collapse (Cor. 3c)** appears exactly where predicted. With perfect honeypots (t = 1), policies trained with λρ ≥ 1 defect at β < 10⁻⁷ (ρ = 0.1: λ = 32, 128; ρ = 0.02: λ = 128). Those with λρ < 1 do not (ρ = 0.1, λ = 8: β = 0.18; ρ = 0.02, λ = 32: β = 0.105).

*Realism-response curve.* We train schemers against plain E (no honeypots) with λ ∈ {16, 64, 256} and 2 seeds, then measure their defection rate on H_t for t ∈ {0, 0.1, …, 1.2}. Across 6 policies, **the fitted probit slope is 1.986 ± 0.020 (true d′ = 2.0)**, as Prop. 4 predicts. The intercept shifts with λ (−2.36 to −5.12) while the slope does not (Fig. 2c). An unconditional 5% misbehaviour baseline is flat by construction. In this setting the slope, unlike the rate, is a property of the defector's *discrimination* and does not depend on its *risk appetite* λ.

![Fig 2](results/fig2_honeypots.png)
*Figure 2. (a, b) Exact mixture frontiers (solid), the Thm 3 bound (dotted), and trained policies (dots) for honeypot shares ρ = 0.02 and 0.1. (c) Realism-response curves of trained schemers: log-rate on honeypots against realism t. The probit slope recovers d′.*

### 5.3 Evaluation-budget scaling and what published awareness numbers imply (Thm 2)

*Setup* (`exp3_scaling.py`, analytic, no training). We set δ = 0.1 (the schemer tolerates at most a 10% chance of ever being caught) and check δ = 0.5 for sensitivity.

*Evaluation count required.* To cap deployment defection at 1% with no honeypots, the evaluator needs N_req = 239 (d′ = 1), 891 (d′ = 1.35), 1.33×10⁴ (d′ = 1.99), 1.52×10⁵ (d′ = 2.5), 2.10×10⁶ (d′ = 3), and 8.4×10⁸ (d′ = 4) evaluation episodes. Around d′ = 2, d ln N_req/dd′ = 4.54, so a 0.1 increase in d′ multiplies the required evaluation budget by ≈1.57. With d′ = 2 and perfect honeypots at share ρ = 0.01, N_req falls from 1.33×10⁴ to 980. At ρ = 0.1 it is 105. With imperfect honeypots at ρ = 0.1, N_req is 435 (Δ = 0.5), 2.07×10³ (Δ = 1.0), and 7.9×10³ (Δ = 1.5). At ρ = 0.01 and Δ = 1.5, honeypots barely help (1.29×10⁴). Realism matters more than share (Fig. 3b).

*Converting published AUCs (illustrative).* Needham et al. (2025) report eval-vs-deployment AUCs of 0.83 (Gemini 2.5 Pro and GPT-4.1), 0.82 (Claude 3.7 Sonnet), 0.75 (GPT-4.1 mini), and 0.92 (human baseline). Assuming an equal-variance binormal ROC gives d′ = 1.35, 1.30, 0.95, and 1.99. A hypothetical policy that thresholds its own judgement, facing N = 1000 i.i.d. evaluations drawn from the same distribution as their evaluation transcripts, with δ = 0.1, could defect in deployment at up to β = 0.009 (d′ = 1.35), 0.008, 0.003, and 0.043 (human-level). At N = 100 the figures are 0.042, 0.038, 0.017, and 0.138. With δ = 0.5 and N = 1000 they are 0.032, 0.029, 0.012, and 0.113.

These numbers need careful reading. (i) They are *achievability* numbers for a model that acts on its stated judgement. A model's verbalised probability is one discriminator, and its internal representations may discriminate better (Nguyen et al., 2025), so the true R can only be higher. (ii) They rely on the equal-variance assumption, which §5.4 shows is load-bearing. (iii) Real evaluation suites are not i.i.d. draws from a single E. These figures are therefore not risk estimates for any named model. They illustrate how the inputs combine.

![Fig 3](results/fig3_scaling.png)
*Figure 3. (a) Maximum deployment defection against evaluation count N for several d′ (solid: exact binormal; dotted: model-free KL bound). (b) The effect of honeypot share ρ and realism gap Δ at d′ = 2.*

### 5.4 AUC does not identify the tail that bounds scheming

*Same AUC, different ceilings.* A binormal ROC R(α) = Φ(a + bΦ⁻¹(α)) has AUC = Φ(a/√(1+b²)). Holding AUC = 0.83 and varying the slope b gives R(10⁻³) = 0.316 (b = 0.5), 0.130 (0.75), 0.041 (1.0), 1.8×10⁻³ (1.5), and 2.6×10⁻⁵ (2.0) (`exp4_measurement.json`). That is four orders of magnitude at one AUC. Slopes below 1 arise when deployment inputs include a subset that is unmistakably "real" (a heavy tail of obvious deployment cues). This is plausibly the realistic case, and it is the worst case for evaluators.

*Finite samples.* We simulate eval-awareness test sets from a ground-truth binormal discriminator with AUC 0.83 and estimate R(10⁻³) in three ways: empirical ROC, equal-variance fit from the empirical AUC, and unequal-variance probit fit. Each configuration uses 1000 replicates (300 for n ≥ 10⁴). With n = 450 per class, roughly a 900-item set split evenly (the split is our assumption), and true b = 1 (truth 0.041):
- the empirical estimator has mean 0.062 and 90% interval [0.009, 0.140]. The test set contains no false-positive rate below 1/450, so this estimate is extrapolation by construction;
- the equal-variance fit is accurate (mean 0.041, 90% interval [0.031, 0.053]) but **only when its assumption holds**. With true b = 0.5 (truth 0.316) it reports 0.042 with 90% interval [0.031, 0.054], excluding the truth and understating the ceiling by about 7.5×. With true b = 1.5 (truth 0.0018) it overstates the ceiling by about 23× (0.041);
- the unequal-variance fit is approximately unbiased across all three truths (e.g. b = 0.5: mean 0.319, 90% [0.256, 0.388]; b = 1.5: mean 0.0027, 90% [0.0005, 0.0071]). The relative uncertainty in the tail is large at n = 450 and narrows to [0.0013, 0.0025] at n = 10⁴.

*Recommendation.* Evaluation-awareness measurements meant to inform scheming risk should report the ROC at false-positive rates near 1/N for the relevant evaluation budget N. They should fit or test the ROC slope rather than assume it, and use test sets of several thousand items per class, drawn from the actual evaluation and deployment distributions in question.

![Fig 4](results/fig4_measurement.png)
*Figure 4. (a) Five binormal ROCs with identical AUC = 0.83 diverge by four orders of magnitude at α = 10⁻³. (b) Estimates of R(10⁻³) against test-set size for three estimators and three true slopes (dashed lines: truth).*

## 6 Discussion

### 6.1 What the results say about the open question

The question "will models deceive, scheme, or hide goals?" splits into a *propensity* question and a *capacity-to-hide* question. This paper addresses only the second, and our answer is a bound. The amount of misbehaviour a model can keep out of behavioural evaluations is limited by a single statistical object: the frontier R of evaluation against deployment inputs, read at a false-positive rate of about 1/N. That frontier is set by the evaluator's choices (what E looks like, how many episodes, what share of honeypots, how realistic) and by the model's discriminative capacity. It does not depend on the model's motives. Three practical consequences follow.

1. **Evaluation awareness should be reported as a tail quantity.** Average-case metrics (AUC, accuracy, verbalised-awareness rates) are compatible with ceilings that differ by orders of magnitude (§5.4).
2. **Realism has larger returns than volume.** Without honeypots, the evaluation budget needed to cap hidden misbehaviour grows like exp(d′²/2). Honeypots with realism gap Δ substitute Δ for d′. Perfect honeypots give a ceiling of c/(ρN) that does not depend on the model's awareness, and once share × stakes exceeds 1 they remove the incentive to defect at all (Cor. 3c). This gives formal backing to the push toward production-realistic evaluations (Krakovna et al., 2026; Schoen et al., 2025) and quantifies the trade-off.
3. **Behavioural non-detection is weak evidence when d′ is high.** If a model has d′ ≈ 2 against a suite of 1000 evaluations, observing zero defections is still consistent with deployment defection rates of several percent (§5.3). Safety cases built on "inability to scheme" (Balesni et al., 2024) need an argument about d′, or about honeypot realism, alongside the clean evaluation record.

### 6.2 A diagnostic for conditional defection

The realism-response curve (Prop. 4) addresses a practical confusion: is a model's low misbehaviour rate on evaluations a sign of alignment or of awareness? Given a ladder of graded-realism honeypots, the slope of probit(rate) against realism separates three hypotheses. A rising slope indicates conditional defection, flat and nonzero indicates unconditional misbehaviour, and zero everywhere indicates either alignment or detection that exceeds the ladder's top rung. In our toy setting the slope recovered d′ to within 1% while the risk-appetite parameter changed the rate by three orders of magnitude.

### 6.3 Pre-registered protocol for frontier models (not run here)

We did not run any experiment on large models. We propose the following protocol so that others can test the predictions.

1. **Ladder construction.** For a fixed misbehaviour opportunity (e.g. a safeguard-tampering opportunity in a coding agent, as in Krakovna et al., 2026), build K ≥ 5 rungs that interpolate from a synthetic evaluation scaffold (rung 0) to real deployment traces (rung K), e.g. by progressively replacing synthetic artefacts (file trees, names, histories, tool latencies) with real ones.
2. **Discriminability per rung.** For each rung, estimate the model's rung-vs-deployment ROC from (a) verbalised probability (as in Needham et al., 2025) and (b) linear probes (as in Nguyen et al., 2025), using ≥ 2,000 items per class and reporting the probit-fitted slope and intercept.
3. **Behaviour per rung.** Measure the misbehaviour rate on each rung using a model organism with a known conditional policy (a sleeper-agent-style backdoor, Hubinger et al., 2024) as positive control and an unconditional-misbehaviour fine-tune as negative control.
4. **Pre-registered predictions.** P1: for the positive control, probit(rate) increases in rung index and its slope correlates with the per-rung d′ from step 2. For the negative control, the slope is indistinguishable from 0. P2: every (α′, β) pair lies on or below the frontier estimated from the model's *own* probe. A violation would mean the probe misses discriminative information the policy uses, which falsifies probe completeness rather than Thm 1. P3: interventions that raise measured eval awareness (Schoen et al., 2025) increase the frontier tail R(1/N) more than they increase AUC.
5. **Falsification.** Thm 1 cannot be falsified, being a theorem, but its *applicability* can. Systematic violations of P2 by probes trained on many layers, or a rising realism-response curve in a model with no detectable discrimination, would indicate that assumptions A1–A3 fail. Plausible causes are cross-episode memory or coordination, or detection that is not behavioural.

## 7 Limitations

- **Statelessness and i.i.d. episodes (A2).** A model with memory, or one able to coordinate across episodes (e.g. by "defect only once"-type strategies), is not covered. With memory the relevant object becomes a sequential test, and the bounds change. Mallen et al. (2025) find current models weak at stateless strategising, but that may not last.
- **Perfect detection on evaluation (A1).** We treat a single evaluation defection as caught. Real monitors miss some defections and sometimes flag defections that did not happen. Imperfect detection rescales α by p_det, which raises the ceilings proportionally, and false positives are not modelled.
- **Single binary action.** Real scheming involves graded, multi-step, sometimes subtle actions (e.g. sandbagging) whose detection is itself a test. Our framework applies per detectable action but does not model subtlety.
- **Deployment distribution is an idealisation.** "D" is the distribution of *opportunities*. Its relevant tail may be a small, unusual region, which is exactly where AUC-based estimates fail (§5.4).
- **Toy experiments.** All training was on 8-dimensional synthetic inputs with networks of at most 32 hidden units, trained by a penalty method that needed annealing to avoid a timid collapse. The experiments test the mathematics and the estimators, not the behaviour of language models. The timid collapse does suggest one caution: training pressure on misbehaviour can produce near-uniform suppression rather than learned discrimination, which would *understate* what a better optimiser could reach.
- **Gaussian specialisations.** Thm 2b, Cor. 3b and Prop. 4 are exact for Gaussian location families. Thm 1, Thm 2a, Thm 3 and Cor. 3a/c are distribution-free. The extension of Prop. 4 to monotone-likelihood-ratio ladders is a conjecture.
- **Literature conversions are illustrative.** The ceilings in §5.3 use AUCs reported in one study under an equal-variance assumption that §5.4 shows can be badly wrong. They are not risk estimates.
- **Capacity to hide, not propensity.** Nothing here says whether models want to scheme.

## 8 Conclusion

Reading a conditional defector as a hypothesis test gives a compact theory of what scheming models can hide from behavioural evaluation. The Neyman–Pearson frontier of evaluation against deployment inputs bounds every policy. Evaluation counts buy protection only at rates set by the tail of that frontier. Honeypots act through two parameters, share and a Rényi realism gap, and they can remove the incentive to defect altogether. A realism-response curve can tell conditional defection apart from unconditional misbehaviour. Small-scale experiments with trained policies match each prediction, and Monte-Carlo analysis shows that the AUC summaries currently used for evaluation awareness do not identify the quantity that matters. The most immediate practical recommendation is to measure and report evaluation awareness at false-positive rates near 1/N, with fitted ROC slopes and test sets sized for the tail. The most useful next step is to run the realism-ladder protocol of §6.3 on model organisms and frontier models.

## Reproducibility

`cd code && export OMP_NUM_THREADS=1 && python3 exp1_frontier.py && python3 exp2_honeypots.py && python3 exp3_scaling.py && python3 exp4_measurement.py` (≈15 min on 4 CPU cores). `exp0_fixed_lambda.py` reproduces the fixed-λ collapse baseline. Seeds are fixed, and all numbers in §5 are read from `results/*.json`.

## References

Balesni, M., Hobbhahn, M., Lindner, D., Meinke, A., et al. (2024). Towards evaluations-based safety cases for AI scheming. arXiv:2411.03336.

Carlsmith, J. (2023). Scheming AIs: Will AIs fake alignment during training in order to get power? arXiv:2311.08379.

Chaudhary, M., et al. (2025). Evaluation awareness scales predictably in open-weights large language models. NeurIPS 2025. arXiv:2509.13333.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.

Goldwasser, S., Kim, M. P., Vaikuntanathan, V., & Zamir, O. (2022). Planting undetectable backdoors in machine learning models. FOCS 2022. arXiv:2204.06974.

Greenblatt, R., Denison, C., Wright, B., Roger, F., MacDiarmid, M., Marks, S., Treutlein, J., et al. (2024a). Alignment faking in large language models. arXiv:2412.14093.

Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2024b). AI control: Improving safety despite intentional subversion. ICML 2024 (PMLR 235). arXiv:2312.06942.

Griffin, C., Thomson, L., Shlegeris, B., & Abate, A. (2024). Games for AI control: Models of safety evaluations of AI deployment protocols. arXiv:2409.07985.

Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. (2019). Risks from learned optimization in advanced machine learning systems. arXiv:1906.01820.

Hubinger, E., et al. (2024). Sleeper agents: Training deceptive LLMs that persist through safety training. arXiv:2401.05566.

Krakovna, V., Lindner, D., Ho, L., Farquhar, S., & Shah, R. (2026). Realistic honeypot evaluations for scheming propensity. arXiv:2605.29729.

Laine, R., et al. (2024). Me, myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. NeurIPS 2024 Datasets and Benchmarks. arXiv:2407.04694.

Mallen, A., Griffin, C., Wagner, M., Abate, A., & Shlegeris, B. (2025). Subversion Strategy Eval: Can language models statelessly strategize to subvert control protocols? arXiv:2412.12480.

Meinke, A., Schoen, B., Scheurer, J., Balesni, M., Shah, R., & Hobbhahn, M. (2024). Frontier models are capable of in-context scheming. arXiv:2412.04984.

Needham, J., Edkins, G., Pimpale, G., Bartsch, H., & Hobbhahn, M. (2025). Large language models often know when they are being evaluated. arXiv:2505.23836.

Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. *Philosophical Transactions of the Royal Society of London, Series A*, 231, 289–337. doi:10.1098/rsta.1933.0009.

Nguyen, J., Hoang, K., Attubato, C. L., & Hofstätter, F. (2025). Probing and steering evaluation awareness of language models. arXiv:2507.01786.

Schoen, B., Nitishinskaya, E., Balesni, M., Højmark, A., Hofstätter, F., Scheurer, J., Meinke, A., et al. (2025). Stress testing deliberative alignment for anti-scheming training. arXiv:2509.15541.

van Erven, T., & Harremoës, P. (2014). Rényi divergence and Kullback–Leibler divergence. *IEEE Transactions on Information Theory*, 60(7), 3797–3820.
