# Assessment Design Is the Binding Constraint: A Dynamic Model of Learning, Offloading, and Grading When Students Have Generative AI

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES). Alternative: *Economics of Education Review*.

---

## Abstract

How should education systems adapt to generative AI? Most proposals focus on what AI tools should do (tutor rather than answer) or on what students should learn (AI literacy). We argue that a third lever, the design of assessment, is the binding constraint, and we make the argument formal. We build a dynamic model in which a student with present-biased preferences chooses, period by period, what share of graded practice to offload to an AI of quality *q*. Offloading raises the practice score but transmits only a fraction κ ("retained engagement") of the learning that doing the work oneself would produce. The model gives five results. (P1) Total offloading is weakly increasing in AI quality under any course structure; the proof is a two-line revealed-preference argument. (P2) The correlation between an unproctored score and true skill has a closed form that collapses as *q* rises; under a uniform skill distribution it falls from 1 to 0.46 at *q* = 0.9. (P3) A closed-form last-period offloading rule, which we check against the full dynamic programme at 64,935 grid points with zero mismatches. (P4) When AI help has an expertise-reversal profile, κ(*s*) = *k*₀ − *k*₁*s*, the skill-maximising access schedule is a myopic *fading* rule: use AI only while κ(*s*) > 1. A present-biased student does not follow this rule. (P5) Partial guardrails can lower learning: raising κ below 1 can make a diligent student who would otherwise practise switch to offloading, so skill drops discontinuously. We calibrate κ by inverting the two moments reported in a field experiment with roughly a thousand high-school students (Bastani et al., PNAS 2025). The implied κ is 0–0.61 (median 0.305) for an unguarded GPT-4 interface and 1.0 in every feasible cell for a guardrailed tutor. In a simulated cohort of 6,000 heterogeneous students at *q* = 0.9, open AI access with 30% proctored weight lowers mean terminal skill by 27.6% relative to full engagement and by 43% in the bottom prior-skill tercile. The top–bottom skill gap rises from 0.199 to 0.352, and the correlation between course grade and skill falls from 0.93 to 0.68. Guardrails restore skill but not grade validity, and fading policies invert the take-home score (correlation −0.17 at *q* = 1). All numbers come from code we ran, except the two calibration moments, which come from the cited experiment. We state falsifiable predictions and a pre-registrable test protocol.

## 中文摘要

教育体系应如何适应AI时代？现有讨论多集中于"AI工具应如何设计"（辅导而非直接给答案）或"学生应学什么"（AI素养）。本文主张第三个杠杆——评估设计——才是约束性条件，并给出形式化论证。我们建立了一个动态模型：具有现时偏好（present bias）的学生在每一期选择将多少比例的计分练习外包给质量为 *q* 的AI。外包提高练习得分，但只保留自己完成任务所能带来学习量的一个比例 κ（"保留的生成性投入"）。主要结果如下。(P1) 在任何课程结构下，总外包量随AI质量弱单调递增（显示偏好论证，两行可证）。(P2) 无监考成绩与真实技能的相关系数有闭式解，并随 *q* 上升而崩塌：均匀技能分布下在 *q*=0.9 时由1降至0.46。(P3) 给出末期外包决策的闭式规则，并在64,935个网格点上与完整动态规划对照，零误差。(P4) 若AI帮助呈"专长逆转"特征 κ(*s*)=*k*₀−*k*₁*s*，则使技能最大化的AI使用方案是一个短视的"渐退（fading）"规则：仅当 κ(*s*)>1 时使用AI；现时偏好的学生不会自发遵循。(P5) 不完全的防护机制可能降低学习：在 κ<1 范围内提高 κ，会使原本会自己练习的勤奋学生转向外包，导致技能非连续下降。我们利用Bastani等（PNAS 2025）高中数学田野实验报告的两个矩对 κ 进行反向校准：无防护GPT-4界面的 κ 为0–0.61（中位数0.305），有防护辅导器在所有可行参数格中 κ=1.0。在6,000名异质学生的模拟队列中，当 *q*=0.9、监考权重30%时，开放AI使平均终期技能相对"完全投入"基准下降27.6%，最低先验技能三分位组下降43%；高低组技能差距由0.199扩大到0.352；课程成绩与技能的相关性由0.93降至0.68。防护型辅导恢复了技能但未恢复成绩效度；渐退政策甚至会使课后作业成绩与技能负相关（*q*=1时为−0.17）。除两个校准矩取自所引实验外，所有数字均来自本文运行的代码。文末给出可证伪预测与可预注册的检验方案。

---

## 1 Introduction

Generative AI now outperforms most students on most homework. That breaks an assumption built into how schools and universities organise learning: the graded artefact a student hands in (problem set, essay, program) is evidence that the student did the cognitive work that builds skill. Once a capable assistant can produce the artefact, the artefact stops being evidence of the work, and the incentive to do the work also changes.

Evidence on the learning cost is accumulating. In a randomized field experiment with high-school mathematics students in Turkey, Bastani et al. (2025) found that access to a standard GPT-4 chat interface raised scores on assisted practice by 48% but lowered scores on a later unassisted exam by 17% relative to a control group. A guardrailed "GPT Tutor" that gave teacher-designed hints instead of answers raised practice scores by 127% and left exam scores statistically indistinguishable from control. Lehmann, Cornelius and Sting (2024) report that students who use LLMs to *substitute* for learning activities cover more topics but understand each one less, whereas students who use them to *complement* their learning understand more. Policy guidance (UNESCO; Miao & Holmes, 2023) and practitioner frameworks such as the AI Assessment Scale (Perkins et al., 2024) recommend graduated permission levels for AI use. They do not say how much proctored assessment is needed, or when.

The question "how should education systems adapt to AI?" is too broad to answer directly. We narrow it to one question that can be studied formally:

> **Given an AI of quality *q* and a learning-retention profile κ, what combination of (i) proctored-assessment weight, (ii) tool guardrails, and (iii) access schedules keeps unassisted skill acquisition near its no-AI benchmark, and what happens to the informational value of grades?**

We answer it with a dynamic model of the student's offloading decision. The model has three ingredients drawn from established literatures. The first is a performance–learning distinction: assisted output is not learning (Bjork & Bjork, 2011; Risko & Gilbert, 2016). The second is present bias: students under-weight future skill (Laibson, 1997; O'Donoghue & Rabin, 1999). The third is multitask incentives: the grade rewards a mix of observable output and unobservable learning (Holmström & Milgrom, 1991). We calibrate the key unobserved parameter to published experimental moments and then run policy experiments on a simulated heterogeneous cohort.

**Contributions.**

1. **A tractable model of AI offloading in coursework** (Section 3), with a single primitive, κ, that summarises how much learning an AI-assisted episode retains. κ is identified from the two moments that AI-in-education experiments routinely report.
2. **Five formal results** (Section 4). P1 is a general monotone-comparative-statics result: better AI never reduces offloading. P2 is a closed form for the collapse of unproctored-score validity. P3 is an explicit last-period offloading rule. P4 is an optimal *fading* theorem under expertise reversal. P5 shows that partial guardrails can be worse than none for diligent students. P1, P2 (uniform case), P4 and P5 are proved; P3 is derived and checked numerically against the dynamic programme.
3. **An inverse calibration** (Section 5.1) of κ from Bastani et al. (2025). It suggests κ ≈ 0.3 for an unguarded assistant and κ ≈ 1 for a guardrailed tutor, across 36 configurations of the unobserved parameters.
4. **Cohort policy experiments and robustness checks** (Sections 5.2–5.5). These quantify skill loss, inequality and grade validity under seven regimes, and test how sensitive the headline effects are to horizon, learning rate, κ and the prior-skill distribution.
5. **Falsifiable predictions and a pre-registrable protocol** (Section 6) that separate our mechanism from alternatives.

The policy message is that tool design (guardrails) and assessment design (proctored weight, access schedules) are not substitutes. Guardrails protect *learning*; only assessment design protects the *information content of grades*. Partial measures on either side can backfire.

## 2 Related Work

**Empirical effects of generative AI on learning.** Bastani et al. (2025) is the cleanest causal evidence on the learning cost of unguarded access, and it supplies our calibration moments. Kestin et al. (2025) report that a research-designed AI physics tutor more than doubled median learning gains relative to in-class active learning in a randomized trial at Harvard. Read together, the two studies say that *design* determines whether AI helps or harms. Lehmann et al. (2024) document the substitute/complement distinction and report that LLM access widened the gap between students with low and high prior knowledge. Kosmyna et al. (2025, preprint) report lower recall of one's own essay and lower neural engagement after LLM-assisted writing; that study is small and its analyses have been publicly critiqued (see the comment arXiv:2601.00856), so we cite it only as suggestive. Correlational survey evidence links heavy AI use to lower critical-thinking scores (Gerlich, 2025). Outside education, Brynjolfsson, Li and Raymond (2025) and Dell'Acqua et al. (2023) find large productivity gains for less-experienced workers. Productivity gains of this kind are the *practice-score* side of our model and say nothing on their own about skill formation.

**Cognitive science of learning.** The distinction between performance during learning and durable learning is the core of the "desirable difficulties" literature (Bjork & Bjork, 2011). Cognitive offloading, the use of external tools to reduce internal processing demands, is reviewed by Risko and Gilbert (2016). The expertise reversal effect, in which guidance that helps novices can hurt more advanced learners, is reviewed by Kalyuga et al. (2003). It motivates our κ(*s*) specification. Scaffolding that is gradually withdrawn ("fading") goes back to Wood, Bruner and Ross (1976). Our P4 gives a decision-theoretic justification for fading AI access and shows that students will not choose it themselves.

**Economics of incentives and time preference.** Multitask agency (Holmström & Milgrom, 1991) explains why rewarding an easily measured output distorts effort away from an unmeasured one. In our setting the take-home score is the measured output and learning is the unmeasured one. Present-biased (β–δ) preferences (Laibson, 1997; O'Donoghue & Rabin, 1999) explain why students choose less practice than is in their own long-run interest. Our P1 uses the monotone comparative statics logic of Milgrom and Shannon (1994). P5 resembles risk compensation (Peltzman, 1975): a safety improvement changes behaviour in a way that offsets the improvement.

**Assessment policy.** Academic-integrity work (Cotton, Cotton & Shipway, 2024) and the AI Assessment Scale (Perkins et al., 2024) classify permissible AI use. UNESCO guidance (Miao & Holmes, 2023) calls for validating AI tools pedagogically. To our knowledge no prior work gives a quantitative model linking AI quality, tool design and proctored weight to skill formation and grade validity together. That is the gap this paper addresses.

## 3 Model

### 3.1 Primitives

A course has *T* graded practice periods followed by a proctored, AI-free exam. A student enters with unassisted skill *s*₀ ∈ [0,1]. In each period *t*, the student chooses an offload share *a*ₜ ∈ [0,1] of the practice task to hand to an AI of quality *q* ∈ [0,1], where *q* is the score the AI's work earns.

- **Practice score:** *P*ₜ = (1 − *a*ₜ) *s*ₜ + *a*ₜ *q*.
- **Learning:** *s*ₜ₊₁ = *s*ₜ + η *g*ₜ (1 − *s*ₜ), with *g*ₜ = (1 − *a*ₜ) + *a*ₜ κ(*s*ₜ), clipped at 0.

Doing the work oneself yields learning rate η on the remaining headroom (1 − *s*). An offloaded episode retains a fraction κ of that learning. κ < 1 describes a crutch (the AI produces the answer). κ = 1 describes a tool that preserves the generative work, for example a hint-giving tutor. κ > 1 describes a tool that teaches better than solitary practice, for example well-designed worked examples for a novice. We use either a constant κ or an expertise-reversal profile κ(*s*) = *k*₀ − *k*₁*s* with *k*₁ > 0.

### 3.2 Preferences and the grade

The course grade is *G* = (1 − *w*) · (1/*T*)Σₜ *P*ₜ + *w* · *s*_T, where *w* is the weight of the proctored exam. Without loss of generality the exam measures terminal skill directly; measurement noise is added in simulation. The student maximises

  *U* = Σₜ [ (1 − *w*)/*T* · *P*ₜ − *c* (1 − *a*ₜ) ] + β (*w* + *L*) *s*_T,

where *c* is the per-period effort cost of doing the work oneself, *L* is the long-run value of skill (later courses, labour market) in grade units, and β ∈ (0,1] discounts the future terminal payoff. This is the β–δ present-bias specification with δ = 1 inside a course. The planner (school) maximises *s*_T, or equivalently *L*·*s*_T. The full-engagement benchmark is *a*ₜ ≡ 0.

Because the payoff from skill is linear in *s*_T, the student's problem is a finite-horizon dynamic programme. We solve it by backward induction on a 401-point skill grid and a 21-point action grid (`code/model.py`).

### 3.3 What κ is, and how it is identified

κ captures, in one number, the interaction between the tool and how students use it. It is not a fixed property of the model's weights. The same GPT-4 has κ ≈ 0.3 behind an open chat box and κ ≈ 1 behind a hint-only prompt (Section 5.1). In an experiment with a no-AI control arm and constant behaviour within a session, the model maps (*a*, κ) to two observable ratios: the treatment/control practice-score ratio and the treatment/control unassisted-exam ratio. Given the unobserved (*s*₀, η*T*, *q*), these two moments pin down (*a*, κ). We treat (*s*₀, η*T*, *q*) as a sensitivity grid rather than estimating them.

## 4 Theoretical Results

**Proposition 1 (better AI never reduces offloading).** Fix any (*T*, η, κ(·), *w*, *L*, β, *c*, *s*₀) and any set of admissible action paths 𝒜 (which may encode access restrictions). Let **a**(*q*) be an optimal path at quality *q*. Then for *q*′ > *q*, Σₜ *a*ₜ(*q*′) ≥ Σₜ *a*ₜ(*q*).

*Proof.* Skill dynamics do not depend on *q*, so *U*(**a**; *q*) = *F*(**a**) + *q* · (1 − *w*)/*T* · Σₜ *a*ₜ for a function *F* independent of *q*. Optimality gives *U*(**a**′; *q*′) ≥ *U*(**a**; *q*′) and *U*(**a**; *q*) ≥ *U*(**a**′; *q*). Adding the two inequalities gives (*q*′ − *q*)(1 − *w*)/*T* · (Σ*a*′ₜ − Σ*a*ₜ) ≥ 0. ∎

P1 holds for every guardrail level, schedule and preference parameter, so improvements in AI capability always push in the direction of more offloading. Its counterpart, that terminal skill is weakly *decreasing* in *q* when κ ≤ 1, does not follow from the same argument, because *which* periods are offloaded matters. We checked it numerically (E1): over 432 configurations (β ∈ {0.1, 0.5, 0.9}, *c* ∈ {0.005, 0.02, 0.05}, *w* ∈ {0, 0.3, 0.6}, *s*₀ ∈ {0.1, 0.3, 0.5, 0.7}, κ ∈ {0, 0.3, 0.6, 0.9}) × 21 values of *q*, we found zero violations of either monotonicity (`results/summary.json`, E1).

**Proposition 2 (collapse of unproctored-score validity).** Suppose that on an unproctored task each student submits the better of their own work and the AI's, so the take-home score is *m* = max(*s*, *q*). If *s* ~ U[0,1], then

  corr(*m*, *s*) = [*q*³/2 + (1 − *q*³)/3 − (1 + *q*²)/4] / √{ [*q*³ + (1 − *q*³)/3 − ((1 + *q*²)/2)²] / 12 }.

*Proof.* Direct integration: E[*m*] = (1 + *q*²)/2, E[*sm*] = *q*³/2 + (1 − *q*³)/3, E[*m*²] = *q*³ + (1 − *q*³)/3, Var(*s*) = 1/12. ∎

The correlation is 1 at *q* = 0, 0.894 at *q* = 0.5, 0.631 at *q* = 0.8, 0.460 at *q* = 0.9 and 0.330 at *q* = 0.95. Monte Carlo with 200,000 draws matches the closed form to within 0.0017 (E2; Figure 2a). The collapse is much faster for a right-skewed skill distribution: under Beta(2,3) the correlation is 0.795, 0.318 and 0.126 at *q* = 0.5, 0.8 and 0.9, because most students sit below the AI. A composite grade with proctored weight *w* and grading noise of s.d. 0.1 keeps a correlation with skill of 0.517, 0.708 and 0.814 at *q* = 0.9 for *w* = 0.3, 0.5 and 0.7, against 0.895 at *q* = 0. With *w* = 0 the correlation is 0.003 (Figure 2b).

**Proposition 3 (last-period offloading rule).** In the final practice period, with a binary choice, the student offloads iff

  (1 − *w*)(*q* − *s*)/*T* + *c* > β (*w* + *L*) η (1 − κ)(1 − *s*).

Both sides are affine in *s*, so the offloading set is an interval. It widens with *q*, *c* and κ, and narrows with β, *L* and η.

*Derivation.* In the last period the continuation value is β(*w* + *L*)*s*_T. Offloading changes the flow payoff by (1 − *w*)(*q* − *s*)/*T* + *c* and changes *s*_T by −η(1 − κ)(1 − *s*). We verified the rule against the last-period policy of the full dynamic programme at 64,935 non-knife-edge grid points across 162 parameter combinations, with zero mismatches (S1, `results/sensitivity.json`).

Two features of P3 matter for policy. First, κ enters the *private* cost of offloading. A better tool makes offloading less costly to learning, so students offload more. P5 builds on this. Second, *w* enters twice: it lowers the grade reward for the take-home artefact and raises the stake on skill. When the AI is weak (*q* < *s*), raising *w* also lowers the grade cost of shirking. This produces the multitask effect in Section 5.2, where a higher proctored weight *lowers* skill at *q* = 0.

**Proposition 4 (optimal fading under expertise reversal).** Let κ(*s*) = *k*₀ − *k*₁*s* with *k*₁ > 0 and η(*k*₀ + *k*₁) < 1. Then the planner's skill-maximising access policy is the myopic rule *a*ₜ = 1 if κ(*s*ₜ) > 1 and *a*ₜ = 0 otherwise. Equivalently, use AI only while *s*ₜ < *s̄* = (*k*₀ − 1)/*k*₁.

*Proof.* Let *f*ₐ(*s*) = *s* + η[(1 − *a*) + *a*κ(*s*)](1 − *s*). For *a* = 0, *f*′ = 1 − η > 0. For *a* = 1, *f*′(*s*) = 1 − η(*k*₀ + *k*₁ − 2*k*₁*s*) ≥ 1 − η(*k*₀ + *k*₁) > 0. Since *f*ₐ is affine in *a*, *f*ₐ(*s*) is increasing in *s* for every *a* ∈ [0,1]. The terminal value *V*_T(*s*) = *s* is increasing. Suppose *V*ₜ₊₁ is increasing. Then *V*ₜ(*s*) = maxₐ *V*ₜ₊₁(*f*ₐ(*s*)) is attained by the *a* that maximises *f*ₐ(*s*), which maximises *g* = 1 + *a*(κ(*s*) − 1): *a* = 1 iff κ(*s*) > 1. *V*ₜ is then a maximum of increasing functions and is itself increasing. Induction completes the proof. ∎

Because κ(*s*) is decreasing in *s*, the policy is a *fading* schedule: full AI support for novices, withdrawn once the learner crosses *s̄*. In E4 (*k*₀ = 1.4, *k*₁ = 0.8, so *s̄* = 0.5), the planner's dynamic programme and the myopic rule agree to within 1.3×10⁻⁵ in value across 40 starting skills. Starting from *s*₀ = 0.15, fading reaches *s*_T = 0.648, against 0.624 for never using AI and 0.641 for always using it. A present-biased student (β = 0.3, *w* = 0.3, *q* = 0.9) offloads in all 20 periods, while the planner offloads in 12 (Figure 4). Even with a tool that is *better than solitary practice* for novices, students do not fade on their own, because the grade rewards the assisted artefact.

**Proposition 5 (partial guardrails can lower learning).** Consider a single binary period (or the last period) with state *s* < 1, and let κ_c be the value at which the P3 inequality holds with equality. If *q* ≥ *s* and the student would self-practise at κ = 0, that is β(*w* + *L*)η(1 − *s*) > (1 − *w*)(*q* − *s*)/*T* + *c*, then κ_c ∈ (0, 1). Terminal skill as a function of κ is constant at *s* + η(1 − *s*) on [0, κ_c), drops discontinuously to *s* + ηκ_c(1 − *s*) at κ_c, and is strictly below its κ = 0 level for every κ ∈ (κ_c, 1).

*Proof.* The student self-practises iff κ < κ_c, where κ_c = 1 − [(1 − *w*)(*q* − *s*)/*T* + *c*] / [β(*w* + *L*)η(1 − *s*)]. The bracketed gain is positive because *q* ≥ *s* and *c* > 0, so κ_c < 1; the self-practice hypothesis gives κ_c > 0. If the student offloads, skill is *s* + ηκ(1 − *s*) < *s* + η(1 − *s*) for κ < 1. ∎

P5 is a risk-compensation result. A guardrail that makes AI-assisted work *less damaging* also makes it *more attractive* to students who were previously deterred by the damage. Who is hurt is predictable: the diligent (high-β, low-*c*) students, who would otherwise have practised. Present-biased students offload at every κ, so for them guardrails only help. The multi-period simulations in Section 5.4 show the same effect in the full dynamic model.

## 5 Experiments

All experiments are CPU simulations of the model in Section 3, implemented in `code/` and run in this environment (total about 80 s of CPU time; command in `code/README.md`). Every number in this section is read from `results/summary.json`, `results/calibration_summary.json`, `results/calibration.csv`, `results/sensitivity.json` or `results/sensitivity_table.csv`, except the two calibration targets taken from Bastani et al. (2025).

### 5.1 Inverse calibration of κ (Figure 1)

**Targets (external, from Bastani et al., 2025).** GPT Base: practice ratio 1.48, exam ratio 0.83. GPT Tutor: practice ratio 2.27, exam ratio about 1.00 (the reported effect is not statistically distinguishable from zero).

**Procedure.** For each of 36 combinations of unobserved parameters (*s*₀ ∈ {0.2, 0.3, 0.4, 0.5}, η*T* ∈ {0.2, 0.4, 0.8}, *q* ∈ {0.7, 0.85, 1.0}; *T* = 10), we hold *a* and κ constant within the session, search a 401 × 401 grid over *a* ∈ [0,1] and κ ∈ [0,2], and call a cell *feasible* if the squared log error on both moments is below 10⁻⁴.

**Results.** GPT Base is feasible in 15 of 36 cells. The implied κ ranges from 0.00 to 0.61 with median 0.305, and the implied offload share *a* ranges from 0.355 to 1.0 with median 0.71. GPT Tutor is feasible in 9 of 36 cells, and the implied κ is 1.00 in every one, with *a* between 0.465 and 0.975 (median 0.79). Infeasible cells are mostly those with *q* too low to produce the reported practice gains. The ratio of exam effect to practice effect is −0.35 for GPT Base and 0 for GPT Tutor.

**Interpretation.** In this model, a zero exam effect under heavy use identifies κ = 1 almost by construction. The informative finding is therefore about the unguarded arm: across every feasible configuration, an open GPT-4 interface retained at most about 60% of the learning, and typically about 30%, of an equivalent episode of self-practice. We use κ = 0.305 ("open AI") and κ = 1 ("tutor") as baselines, and S2 varies κ_open over {0, 0.305, 0.61}.

### 5.2 Cohort policy experiment (Figure 5)

**Setup.** *N* = 6,000 students with *s*₀ ~ Beta(2,3). β is drawn uniformly from {0.1, 0.3, 0.5, 0.7, 0.9} and *c* from {0.005, 0.01, 0.02}. *T* = 20, η = 0.04, *L* = 1. Grade noise has s.d. 0.05. *q* ranges over 21 values in [0, 1]. We compare seven regimes:

1. *Full engagement* (benchmark; *a* ≡ 0 is enforced).
2. *No AI*: offloading is pure shirking (*q* = 0, κ = 0).
3. *Open AI*, *w* = 0.3, κ = 0.305.
4. *Open AI*, *w* = 0.7, κ = 0.305.
5. *Tutor*, κ = 1, *w* = 0.3.
6. *Tutor* with κ(*s*) = 1.4 − 0.8*s*, unrestricted.
7. *Tutor* κ(*s*) *with fading enforced*: AI is allowed only while κ(*s*) > 1.

Each student best-responds by solving the DP for their own (β, *c*).

**Table 1. Cohort outcomes at *q* = 0.9** (from `results/summary.json`, E5).

| Regime | mean *s*_T | bottom-tercile *s*_T | top–bottom gap | corr(grade, *s*_T) | corr(take-home, *s*_T) | mean grade | mean offload |
|---|---|---|---|---|---|---|---|
| Full engagement | 0.736 | 0.640 | 0.199 | 0.928 | 0.939 | 0.629 | 0.00 |
| No AI (shirking) | 0.727 | 0.613 | 0.226 | 0.931 | 0.934 | 0.620 | 0.02 |
| Open AI, *w* = 0.3 | 0.533 | 0.363 | 0.352 | 0.680 | 0.008 | 0.789 | 1.00 |
| Open AI, *w* = 0.7 | 0.576 | 0.435 | 0.293 | 0.837 | −0.195 | 0.653 | 0.82 |
| Tutor κ = 1, *w* = 0.3 | 0.736 | 0.640 | 0.199 | 0.481 | 0.003 | 0.851 | 1.00 |
| Tutor κ(*s*), unrestricted | 0.728 | 0.652 | 0.162 | 0.418 | −0.001 | 0.848 | 1.00 |
| Tutor κ(*s*) + fading | 0.744 | 0.660 | 0.179 | 0.437 | 0.126 | 0.737 | 0.27 |

**Findings.**

*(i) Open AI erodes skill, mostly at the bottom.* With *w* = 0.3, mean terminal skill falls monotonically in *q* from 0.727 (*q* = 0) to 0.533 (*q* ≥ 0.85), 27.6% below the benchmark. In the bottom prior-skill tercile the loss is 43% (0.363 vs 0.640), and the top–bottom gap widens by 77% (0.199 → 0.352). This matches the direction Lehmann et al. (2024) report: AI widened the gap between low- and high-prior-knowledge students.

*(ii) Grades inflate while their validity falls.* Under open AI the mean grade rises from 0.617 (*q* = 0) to 0.860 (*q* = 1), while corr(grade, skill) falls from 0.948 to 0.69 and the take-home component carries essentially no information (0.008 at *q* = 0.9). Raising *w* to 0.7 keeps corr(grade, skill) at 0.837, but the take-home score becomes *negatively* related to skill (−0.195), because low-skill students lean on AI most.

*(iii) Proctoring is a blunt instrument and has a multitask cost.* With *w* = 0.7, skill at *q* = 0.9 recovers only to 0.576. At *q* = 0 it is lower than with *w* = 0.3 (0.685 vs 0.727): a smaller grade weight on practice lowers the incentive to do the practice at all, in line with P3 and Holmström and Milgrom (1991).

*(iv) Guardrails restore learning but not grade validity.* A κ = 1 tutor fully preserves skill (0.736 at every *q*), yet corr(grade, skill) falls to 0.48 because practice scores are uniformly inflated. Tool design cannot replace assessment design if grades are to carry information.

*(v) Fading gives the best learning and the most inverted take-home scores.* Enforced fading gives the highest mean skill (0.744) and a narrower gap (0.179) than the benchmark. But because novices are permitted AI while stronger students are not, the take-home score is *inversely* related to skill at high *q* (−0.174 at *q* = 1), and corr(grade, skill) falls to 0.223 at *q* = 1. A fading regime has to be paired with grading that does not use assisted practice as evidence of skill.

### 5.3 How much proctoring is enough? (Figure 3)

We define *w*₉₅ as the smallest proctored weight (on a 0.02 grid) at which a student with *s*₀ = 0.3, β = 0.5 and *c* = 0.01 reaches at least 95% of benchmark terminal skill. With κ = 0, *w*₉₅ = 0 for *q* ≤ 0.5 and then rises steeply: 0.34, 0.58, 0.74, 0.80 and 0.84 at *q* = 0.6, 0.7, 0.8, 0.9 and 1.0. With κ = 0.3 the target is **infeasible for every *w* ≤ 1** once *q* ≥ 0.5, and with κ = 0.6 once *q* ≥ 0.4. With κ = 0.9, *w*₉₅ = 0 throughout, because even full offloading keeps skill above the 95% threshold. Holding *q* = 0.9 and varying κ, for β = 1 *w*₉₅ rises from 0.34 at κ = 0 to 1.0 at κ = 0.65, is infeasible for κ ∈ [0.7, 0.85], and drops to 0 for κ ≥ 0.9. For β = 0.5 it is 0.80–0.98 for κ ≤ 0.25 and infeasible up to κ = 0.85. For β = 0.25 it is infeasible below κ = 0.9.

So guardrails and proctoring are not substitutes over most of the range. Partial guardrails *raise* the proctored weight needed to deter offloading (P3), and only near-complete retention (κ ≳ 0.9 here) removes the need for proctoring. Both levers are needed, with a discontinuity at the high end.

### 5.4 Partial guardrails in the full dynamic model (Figure 6)

We scan κ ∈ [0, 0.95] at *q* = 0.9 and *w* = 0.3 over 27 configurations (β ∈ {0.25, 0.5, 1.0}, *c* ∈ {0.005, 0.01, 0.02}, *s*₀ ∈ {0.1, 0.3, 0.5}). Terminal skill is non-monotone in κ in 6 of 27 configurations: every configuration with β = 1 and *c* ≤ 0.01. For β ≤ 0.5, students offload fully at every κ, and skill rises smoothly with κ. For the diligent group the drop is large. With *s*₀ = 0.1 and *c* = 0.01, *s*_T falls from 0.602 at κ = 0 (no offloading) to 0.234 at κ = 0.2 (full offloading), and even at κ = 0.95 recovers only to 0.585, still below the κ = 0 level. This confirms P5 in the multi-period model. A guardrail can improve average outcomes while lowering outcomes for the students who were already doing the work.

### 5.5 Robustness (S2)

We re-ran the cohort comparison at *q* = 0.9 (*N* = 2,000; independent seed) while varying one factor at a time. To keep total learning potential fixed when varying *T*, we held η*T* = 0.8. Headline quantities (`results/sensitivity_table.csv`):

| Variant | skill loss, open *w* = 0.3 | skill loss, open *w* = 0.7 | fading gain | corr(grade, *s*_T), open | corr, benchmark |
|---|---|---|---|---|---|
| Baseline (*T* = 20, η = 0.04, κ_open = 0.305, Beta(2,3)) | 27.8% | 21.2% | +1.1% | 0.70 | 0.93 |
| *T* = 10 (η = 0.08) | 27.9% | 16.3% | +1.1% | 0.68 | 0.93 |
| *T* = 40 (η = 0.02) | 27.4% | 25.8% | +1.0% | 0.70 | 0.93 |
| η = 0.02 (*T* = 20) | 22.1% | 21.7% | +1.9% | 0.73 | 0.95 |
| η = 0.08 (*T* = 20) | 24.7% | 17.6% | +0.4% | 0.38 | 0.85 |
| κ_open = 0 | 42.0% | 25.9% | +1.1% | 0.46 | 0.93 |
| κ_open = 0.61 | 13.6% | 13.5% | +1.0% | 0.58 | 0.93 |
| *s*₀ ~ Beta(5,5) | 21.7% | 17.7% | +0.3% | 0.57 | 0.87 |
| *s*₀ ~ Beta(1.5,4) | 36.5% | 27.6% | +2.2% | 0.65 | 0.91 |

The qualitative pattern holds across all variants. Open access costs 13.6–42.0% of terminal skill. Raising *w* from 0.3 to 0.7 recovers only part of that loss and, at the upper end of κ, almost none. Fading gives small positive gains (0.3–2.2%). Grade validity under open access is 0.22–0.47 below the benchmark. The size of the skill loss depends most on κ_open and on the share of low-prior students, which are the two quantities a deployment should measure.

## 6 Discussion: Predictions and a Test Protocol

The model implies the following predictions. Each can be tested with the two-moment design of existing studies plus one added arm or measurement.

**Pred-1 (κ is identifiable and tool-specific).** For any AI tool, the ratio (exam effect)/(practice effect) estimated in a randomized trial with a no-AI control maps to κ through Section 5.1. A hint-only tutor and an answer-giving interface built on the *same* model should yield different κ estimates, which can be compared statistically.

**Pred-2 (offloading rises with capability even under fixed rules).** Holding course rules fixed, an upgrade in AI quality should weakly raise logged offloading (P1). A model-version change during a term is a natural experiment for this.

**Pred-3 (heterogeneous harm by prior skill).** Exam losses from open access should be concentrated in the lowest prior-achievement tercile (Table 1, findings (i); 43% vs 27.6% on average).

**Pred-4 (guardrail risk compensation).** Introducing a *partial* guardrail (one that lowers but does not eliminate answer-giving) should *reduce* exam performance for students with high measured conscientiousness or patience, while raising it for the least patient students (P5, Section 5.4). This interaction is the sharpest test separating our mechanism from a pure "tool quality" account, in which guardrails help everyone.

**Pred-5 (fading inverts take-home validity).** Under an access policy that permits AI for novices only, the correlation between take-home scores and proctored exam scores should fall and may become negative (Table 1, last row).

**Pre-registrable protocol.** The design is a 2 × 2 × 2 cluster-randomized trial at the course-section level:

- Factors: guardrail (open vs hint-only), proctored weight (0.3 vs 0.7), and access schedule (unrestricted vs fading with a skill-gated unlock).
- Measures: (a) logged offload share from interaction logs, classifying each turn as answer-request or hint-request; (b) practice scores; (c) a proctored, AI-free exam; (d) baseline prior achievement; (e) a validated present-bias or conscientiousness measure.
- Primary estimands: the exam treatment effect by arm, κ̂ per arm using Section 5.1, and the guardrail × patience interaction for Pred-4.
- Power: the guardrail × patience interaction is a within-arm moderation effect and will need many more sections than the main effects. We have not run a power analysis; a pre-registration should set the number of sections by simulating from the calibrated model with the expected intraclass correlation.

**What the model says about "how education systems should adapt."** Within the model's scope:

1. Treat assessment design as part of AI policy. Any regime that grades unproctored artefacts will see grade inflation and loss of validity that grows with *q* (P2). This happens even with perfect tutors (Table 1, row 5).
2. Procure tools on their measured κ, not on how good the AI's answers are. A high-*q*, low-κ tool is the worst combination for learning (P1, P3).
3. Do not rely on partial guardrails alone. They can hurt diligent students (P5) and raise the proctoring needed (Section 5.3).
4. Where tools are better than solitary practice for novices, schedule them, using skill-gated fading (P4), and stop treating assisted practice as evidence of skill.

## 7 Limitations

- **Stylised learning dynamics.** Skill is one-dimensional, learning is exponential toward a ceiling, and κ summarises the tool–student interaction in a single number. Real skills are multidimensional, and AI may help with some sub-skills (for example retrieval) while harming others (for example problem formulation). Forgetting, spacing and transfer are not modelled.
- **Calibration is thin.** κ is inferred from two published moments of one experiment (one country, one subject, one grade band, GPT-4-era tools). The unobserved parameters (*s*₀, η*T*, *q*) are swept rather than estimated, and only 15 of 36 (GPT Base) and 9 of 36 (GPT Tutor) cells are feasible. We do not propagate the experiment's sampling uncertainty. In particular, the tutor's null exam effect is consistent with κ somewhat below or above 1. κ = 1 for the tutor is largely identified by construction (Section 5.1).
- **Behavioural specification.** Students are sophisticated optimisers of a β-discounted objective with known parameters. Real students may be naive about their own present bias, may misjudge κ (a natural extension: perceived vs true κ), and may be motivated by things other than grades. Peer effects, teacher responses and AI-detection or sanction regimes are absent.
- **Assessment model.** The proctored exam measures skill without bias. In practice, proctored exams measure a narrow slice of skill, and oral or in-class formats have their own validity and equity problems. The take-home model in P2 assumes students submit max(*s*, *q*), which ignores partial blending.
- **Parameter choices in simulations are illustrative.** The expertise-reversal profile (*k*₀ = 1.4, *k*₁ = 0.8) is hypothetical and not calibrated. The fading gains (≈1%) depend on it and should not be read as an empirical forecast. The *w*₉₅ results are for one representative student.
- **No experiments on real learners or real models.** All results are simulations and proofs. The predictions in Section 6 are untested.
- **Proof scope.** P1, P2 (uniform case), P4 and P5 (single-period form) are proved. The monotonicity of terminal skill in *q* and the multi-period form of P5 are established numerically over finite grids only.

## 8 Conclusion

We formalised one sub-question of how education should adapt to AI: how assessment weight, tool guardrails and access schedules jointly determine skill formation and grade validity when a present-biased student can offload graded work to an AI. In a deliberately simple model, more capable AI always increases offloading. Unproctored scores lose their information content at a rate with a closed form. The best access policy under expertise reversal is a myopic fading rule that students will not choose for themselves. Partial guardrails can reduce learning for the most diligent students. Calibrated to the best available field experiment, the model suggests an unguarded assistant retains about 30% of the learning from self-practice. In simulated cohorts this translates into a loss of roughly a quarter of terminal skill, concentrated among weaker students, along with inflated and less informative grades. Guardrails and proctoring address different failures, learning and measurement respectively, and an education system adapting to AI needs both. The predictions in Section 6, especially the guardrail × patience interaction, are the most direct way to find out whether this account is right.

## Reproducibility

Code: `code/` (see `code/README.md`). Run `python3 calibrate.py && python3 experiments.py && python3 sensitivity.py && python3 fig6_guardrail.py` from `code/`; total CPU time is about 80 s. Outputs are in `results/`: `calibration.csv`, `calibration_summary.json`, `summary.json` (E1–E5), `sensitivity.json` and `sensitivity_table.csv` (S1–S3), and figures `fig1_calibration.png` through `fig6_guardrail.png`.

**Figures.** Fig. 1: inverse calibration of (*a*, κ). Fig. 2: unproctored-score informativeness (P2) and composite grades. Fig. 3: minimal proctored weight *w*₉₅. Fig. 4: fading theorem trajectories (P4). Fig. 5: cohort outcomes vs *q* across seven regimes. Fig. 6: terminal skill vs guardrail strength κ (P5).

## References

Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R. (2025). Generative AI without guardrails can harm learning: Evidence from high school mathematics. *Proceedings of the National Academy of Sciences*, 122(26), e2422633122. https://doi.org/10.1073/pnas.2422633122

Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. In M. A. Gernsbacher et al. (Eds.), *Psychology and the Real World: Essays Illustrating Fundamental Contributions to Society* (pp. 56–64). Worth Publishers.

Brynjolfsson, E., Li, D., & Raymond, L. (2025). Generative AI at work. *The Quarterly Journal of Economics*, 140(2), 889–942.

Cotton, D. R. E., Cotton, P. A., & Shipway, J. R. (2024). Chatting and cheating: Ensuring academic integrity in the era of ChatGPT. *Innovations in Education and Teaching International*, 61(2), 228–239.

Dell'Acqua, F., McFowland III, E., Mollick, E., Lifshitz-Assaf, H., Kellogg, K. C., Rajendran, S., Krayer, L., Candelon, F., & Lakhani, K. R. (2023). Navigating the jagged technological frontier: Field experimental evidence of the effects of AI on knowledge worker productivity and quality. Harvard Business School Working Paper 24-013.

Gerlich, M. (2025). AI tools in society: Impacts on cognitive offloading and the future of critical thinking. *Societies*, 15(1), 6.

Holmström, B., & Milgrom, P. (1991). Multitask principal–agent analyses: Incentive contracts, asset ownership, and job design. *Journal of Law, Economics, & Organization*, 7 (Special Issue), 24–52.

Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The expertise reversal effect. *Educational Psychologist*, 38(1), 23–31.

Kestin, G., Miller, K., Klales, A., Milbourne, T., & Ponti, G. (2025). AI tutoring outperforms in-class active learning: An RCT introducing a novel research-based design in an authentic educational setting. *Scientific Reports*, 15, 17458.

Kosmyna, N., Hauptmann, E., Yuan, Y. T., Situ, J., Liao, X.-H., Beresnitzky, A. V., Braunstein, I., & Maes, P. (2025). Your brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task. arXiv:2506.08872.

Laibson, D. (1997). Golden eggs and hyperbolic discounting. *The Quarterly Journal of Economics*, 112(2), 443–478.

Lehmann, M., Cornelius, P. B., & Sting, F. J. (2024). AI meets the classroom: When do large language models harm learning? SSRN Working Paper 4941259.

Miao, F., & Holmes, W. (2023). *Guidance for Generative AI in Education and Research*. Paris: UNESCO.

Milgrom, P., & Shannon, C. (1994). Monotone comparative statics. *Econometrica*, 62(1), 157–180.

O'Donoghue, T., & Rabin, M. (1999). Doing it now or later. *American Economic Review*, 89(1), 103–124.

Peltzman, S. (1975). The effects of automobile safety regulation. *Journal of Political Economy*, 83(4), 677–725.

Perkins, M., Furze, L., Roe, J., & MacVaugh, J. (2024). The Artificial Intelligence Assessment Scale (AIAS): A framework for ethical integration of generative AI in educational assessment. *Journal of University Teaching and Learning Practice*, 21(6).

Risko, E. F., & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences*, 20(9), 676–688.

Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry*, 17(2), 89–100.
