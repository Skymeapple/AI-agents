# The Deskilling Trap: Endogenous AI Reliance, Bistable Skill Dynamics, and When Better AI Makes Human–AI Teams Worse

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES) — formal-model and simulation track. (A shorter version would also fit ACM FAccT.)

**Contribution type:** theory (closed-form propositions) + simulation + consistency checks against published effect sizes. No human-subjects data were collected; no large models were run.

---

## Abstract

Concern that heavy use of AI assistants erodes human skill now rests on a growing but mixed body of evidence: a field experiment in which unrestricted GPT-4 access lowered later unaided exam performance, a randomized study in which AI help reduced conceptual mastery of a new programming library, and an observational study in which endoscopists' unassisted adenoma detection rate fell after AI was introduced. Most discussions treat deskilling as a smooth *dose–response*: more AI use, proportionally less skill. We argue that the defining feature of AI-induced deskilling is a *feedback loop*: people rely on AI more when their own skill is low, and relying more lowers skill further. We formalize this loop in a minimal one-dimensional dynamical model that couples (i) practice-driven skill acquisition and exponential forgetting (half-life calibrated to a 2025 meta-analysis of procedural skill decay) with (ii) a myopic reliance rule that compares unaided error plus effort against AI error after imperfect human verification. We prove that (1) for an interval of AI error rates the system is bistable, with a skilled and a deskilled equilibrium, and give the interval in closed form; (2) the system exhibits hysteresis, so that skills lost when AI improves are not regained when AI degrades by the same amount; (3) at the collapse point a marginal *improvement* in AI accuracy *raises* equilibrium task error discontinuously, by exactly the effort saving plus the lost verification value; and (4) a mandated share of unaided practice, or a design that raises learning during assisted work, removes the trap at a computable minimum dose, and the two are perfect substitutes. Numerical work shows that the bistability persists for smooth reliance rules only above a critical choice sharpness, that forward-looking agents avoid the trap that myopic agents fall into (myopic discounted cost up to 27% higher when starting skilled and up to 128% higher when starting deskilled), that hysteresis, not population bimodality, discriminates the feedback model from a no-feedback null in heterogeneous populations, and that generic early-warning signals are only weakly informative (AUC 0.56–0.64). A consistency check against the endoscopy study shows that practice-driven forgetting at meta-analytic rates cannot by itself produce the reported 21% relative drop within three months, which points to a fast, reversible, attentional component. We list seven falsifiable predictions and a pre-registrable test protocol.

## 中文摘要

关于"过度依赖AI是否会导致人类技能退化"的证据日益增多但结论不一：高中数学现场实验显示无限制使用GPT-4会降低之后独立考试成绩；随机实验显示AI辅助降低了对新编程库的掌握；观察性研究显示内镜医生在引入AI后、非AI辅助下的腺瘤检出率下降。现有讨论多把技能退化视为平滑的"剂量—反应"关系。本文认为AI致技能退化的本质是一个**反馈回路**：技能越低越依赖AI，越依赖AI技能越低。我们构建了一个最小的一维动力学模型，将(i)由练习驱动的技能习得与指数遗忘（遗忘半衰期依据2025年程序性技能衰退元分析校准）与(ii)近视型依赖决策（比较独立完成的错误率加努力成本与经人类不完全核查后的AI错误率）相耦合。我们证明：(1)在一段AI错误率区间内系统呈**双稳态**（熟练均衡与去技能均衡并存），并给出区间的闭式解；(2)系统存在**滞后效应**：AI变好时失去的技能，在AI以同等幅度变差时并不会恢复；(3)在崩溃点，AI准确率的边际**提升**反而使均衡任务错误率**跳升**，跳幅恰为努力节省与核查价值损失之和；(4)强制一定比例的独立练习，或提高辅助工作中的学习保留率，可在可计算的最低剂量下消除陷阱，二者完全可替代。数值实验表明：平滑依赖规则下的双稳态仅在决策陡峭度超过临界值时存在；具前瞻性的个体能避开近视个体陷入的陷阱（近视个体的贴现成本最高高出27%（从熟练状态出发）或128%（从去技能状态出发））；在异质人群中，区分反馈模型与无反馈零模型的是滞后效应而非人群双峰性；通用的临界预警信号仅弱有效（AUC 0.56–0.64）。与内镜研究的一致性检验表明，按元分析遗忘速率的练习驱动遗忘无法单独在三个月内产生所报告的21%相对下降，提示存在快速、可逆的注意力成分。文末给出七条可证伪预测与可预注册的检验方案。

---

## 1 Introduction

The worry that automation erodes the skills of the people it assists is old. Bainbridge's "ironies of automation" observed that automating routine work deprives operators of the practice they need for the rare cases where they must take over (Bainbridge, 1983), and a 2018 retrospective concluded that these ironies remain unresolved (Strauch, 2018). Generative AI has made the question both broader and more urgent, because the tasks it absorbs (writing, coding, diagnosis, tutoring) are cognitive tasks whose skills are known to decay without use. In a 2025 meta-analysis of procedural skill retention, half of the initial gains in accuracy-based performance were lost after roughly 6.5 months of non-use (Psychological Bulletin, 2025; Arthur et al., 1998 for the earlier review).

Direct evidence on AI is accumulating. In a field experiment with nearly a thousand high-school students, an unrestricted GPT-4 tutor improved practice performance but reduced later unaided exam grades by 17%, while a tutor with learning-oriented guardrails largely avoided the harm (Bastani et al., 2025). In a randomized study of developers learning a new asynchronous Python library, those with AI assistance scored 50% on a follow-up quiz against 67% for those coding by hand, with interaction patterns that kept users cognitively engaged preserving learning (Shen & Tamkin, 2026). In a retrospective multicentre study, endoscopists' adenoma detection rate (ADR) in unassisted colonoscopies fell from 28.4% to 22.4% in the three months after AI-assisted colonoscopy was introduced (Budzyń et al., 2025). Surveys and lab studies report reduced self-reported critical-thinking effort (Lee et al., 2025), correlations between AI use and lower critical-thinking scores mediated by cognitive offloading (Gerlich, 2025), and lower neural engagement during AI-assisted essay writing (Kosmyna et al., 2025). Macnamara et al. (2024) argued that AI assistance may accelerate skill decay and hinder skill development *without the performer being aware of it*.

The evidence is also mixed. The endoscopy result is observational, and critics noted differences in patient mix and colonoscopy volume between periods. Correlational studies cannot separate selection from causation. And the existing literature lacks a *mechanistic* account that says when deskilling should occur, how large and how reversible it should be, and what minimum intervention prevents it. Without such an account, the field's empirical results cannot be reconciled: one study's large effect and another's null are both "consistent" with the vague claim that over-reliance may harm skill.

**The gap we address.** We isolate one mechanism that, we argue, distinguishes AI-induced deskilling from ordinary disuse: *reliance is endogenous to skill*. People rely on an AI more when they find the task harder for themselves (Risko & Gilbert, 2016, review evidence that offloading decisions track perceived internal ability), and relying more removes the practice that maintains ability. This is a positive feedback loop. Positive feedback loops generically produce tipping points, multiple equilibria and hysteresis (Strogatz, 2015; Scheffer et al., 2009). We ask:

> *Under what conditions does endogenous reliance on an AI assistant produce a self-reinforcing "deskilling trap"; what are the consequences for human–AI team performance; and what is the minimal intervention that removes the trap?*

**Contributions.**

1. **A minimal, calibrated model** of the reliance–skill loop (Section 3) with five interpretable ingredients: learning from practice, exponential forgetting, partial learning during assisted work (κ), imperfect verification of AI output (ρ), and an effort saving from reliance (Δc). The forgetting rate is taken from a published meta-analysis.
2. **Closed-form results for the step-reliance limit** (Section 4): an exact bistability interval of AI error rates [ε_c, ε_r) (Proposition 1), strictly positive hysteresis whenever assisted work teaches less than unassisted work (Proposition 2), a discontinuous *increase* in equilibrium task error caused by a marginal AI *improvement* at the collapse point, with an exact jump size (Proposition 3, the "improvement paradox"), and the minimal practice floor that removes the trap, with a perfect-substitution relation between practice floors and engagement-preserving design (Proposition 4). All four are checked numerically on 395 random parameter draws.
3. **Numerical results for smooth reliance, forward-looking agents and heterogeneous populations** (Section 5): a critical reliance sharpness β_c below which no bistability occurs; the cost of myopia relative to a dynamic-programming benchmark; the finding that hysteresis, not population bimodality, separates the feedback model from a no-feedback null; recovery of practice-driven deficits within months once AI is withdrawn; and only weak early-warning signals.
4. **A quantitative consistency check against published effect sizes** (Section 6): practice-driven forgetting at meta-analytic rates cannot by itself produce the 21% relative ADR drop reported within three months for endoscopy, while implied learning-retention parameters from the education and coding studies suggest κ ≈ 0.5–0.8.
5. **Seven falsifiable predictions and a pre-registrable protocol** (Section 7) that discriminate the feedback mechanism from dose–response accounts.

We state the scope plainly: this is a stylized model. Its value lies in identifying which quantities need to be measured (κ, ρ, the reliance sharpness β, and the forgetting rate) and what qualitative signatures to look for. It does not show that any particular profession is in a deskilling trap.

## 2 Related Work

**Automation, complacency and skill retention.** Human-factors research established that automation changes operator skill and attention. Parasuraman and Manzey (2010) integrated automation complacency and automation bias, showing that imperfect decision aids produce both omission and commission errors and that the effects are not removed by simple practice. Casner et al. (2014) found in a Boeing 747 simulator study that instrument scanning and manual control skills were reasonably well retained, while the cognitive skills of manual flying, such as tracking position without a map display, degraded with automation use. Bainbridge (1983) and Strauch (2018) frame the central irony that the more reliable the automation, the less practised the human for the cases where it fails. Our model formalizes this irony as a bifurcation and adds what is new about generative AI: reliance is chosen task by task, and the choice depends on the user's current skill.

**Skill decay.** Arthur et al. (1998) reviewed factors affecting skill loss over periods of non-use and found larger losses for cognitive than physical tasks. The 2025 Psychological Bulletin meta-analysis of procedural skill (1,344 effect sizes from 457 reports) estimated half-lives of gains of about 6.5 months for accuracy, 11 months for mixed and 13 months for speed measures. We use these as our forgetting-rate calibration.

**Cognitive offloading and digital tools.** Risko and Gilbert (2016) define cognitive offloading and note that people offload when they believe it more efficient than internal processing, beliefs that do not always match reality. Sparrow et al. (2011) reported that people remember less of information they expect to be able to find later. Dahmani and Bohbot (2020) found that greater lifetime GPS use was associated with worse spatial memory during self-guided navigation, and in a small three-year follow-up with steeper decline. These studies document offloading and its correlates but do not model the feedback from skill to offloading.

**Generative AI and learning or skill.** Bastani et al. (2025), Shen and Tamkin (2026), Budzyń et al. (2025), Lee et al. (2025), Gerlich (2025) and Kosmyna et al. (2025) are described in Section 1. Dell'Acqua et al. (2023) showed in a pre-registered experiment with 758 consultants that GPT-4 improved performance on tasks inside the AI's capability "frontier" but reduced correct solutions by 19 percentage points on a task outside it, which is the regime in which human verification skill matters. Macnamara et al. (2024) set out the theoretical case for AI-accelerated decay. Our model supplies the dynamics missing from these accounts and turns their qualitative concerns into conditions that can be tested.

**Critical transitions.** Scheffer et al. (2009) reviewed generic early-warning signals (rising variance and autocorrelation from critical slowing down) before fold bifurcations in ecological, climate and financial systems. We test whether these signals would be usable for individual deskilling, and find them weak at realistic noise and observation lengths.

**What is new here.** To our knowledge, no prior work (i) derives the bistability region, hysteresis width and minimal-intervention dose in closed form for endogenous AI reliance, (ii) shows the discontinuous team-error increase at the collapse point, or (iii) checks a deskilling mechanism quantitatively against published effect sizes. We did not find an earlier formal treatment of this feedback loop in the sources we surveyed. Our literature search was web-based and not exhaustive, so we make this claim with that caveat.

## 3 Model

Time *t* is measured in months. A worker performs a stream of tasks of a given type. Their **unaided skill** is s ∈ [0, 1].

**Errors.** The unaided error rate is linear in skill,

  h(s) = h₀ + h₁(1 − s),

with an irreducible floor h₀. The AI makes errors at rate ε. When the worker relies on the AI, they verify its output and catch an error with probability ρs, where ρ ∈ (0, 1] is verification efficacy. Spotting someone else's error is harder than avoiding one's own, a point consistent with the automation-bias literature (Parasuraman & Manzey, 2010). The AI-assisted error rate is

  a(s; ε) = ε(1 − ρs).

**Effort.** Unaided work costs an extra effort Δc > 0 relative to assisted work (in error-rate-equivalent loss units). This captures the reduced cognitive effort reported by knowledge workers (Lee et al., 2025).

**Myopic reliance.** The advantage of relying on the AI for a task is

  D(s; ε) = h(s) + Δc − a(s; ε).

The worker's reliance share is r(s) = 1[D > 0] in the **step model**, or r(s) = 1/(1 + e^(−βD)) in the **smooth model**, where β is the sharpness of the reliance response. A practice floor u_min caps reliance at r ≤ 1 − u_min.

**Skill dynamics.** Unassisted tasks provide full practice. Assisted tasks provide a fraction κ ∈ [0, 1) of it, which is how much learning survives when the AI does most of the work (Shen & Tamkin, 2026, report that engaged interaction patterns preserve learning). Effective practice is p(r) = (1 − r) + κr, and

  ds/dt = α p(r(s)) (1 − s) − δ s,   (1)

with learning rate α and forgetting rate δ. With fixed practice p, skill relaxes exponentially to s(p) = αp/(αp + δ). We write s_H = s(1) = α/(α + δ) for the skilled level and s_L = s(κ) for the fully reliant level.

**Reliance threshold.** Solving D(θ) = 0 gives the threshold below which the worker relies:

  θ(ε) = (h₀ + h₁ + Δc − ε) / (h₁ − ερ).   (2)

*Condition A*, ρ(h₀ + h₁ + Δc) < h₁, makes θ strictly decreasing in ε, so a better AI raises the skill below which people hand over the task. It holds for all baseline and sensitivity settings used here.

**Baseline parameters.** δ = ln2/6.5 per month, the accuracy half-life from the 2025 meta-analysis [sourced]. α = 9δ, giving s_H = 0.9 [assumed]. κ = 0.15, ρ = 0.5, h₀ = 0.05, h₁ = 0.6, Δc = 0.03 [assumed; varied in Section 5.6]. All results below are produced by the code in `code/` and saved under `results/` (file names in brackets).

## 4 Theory: the step-reliance limit

Throughout this section r = 1[D > 0] and Condition A holds.

**Proposition 1 (bistability interval).** Let ε_c = (h₀ + h₁ + Δc − s_H h₁)/(1 − s_H ρ) and ε_r = (h₀ + h₁ + Δc − s_L h₁)/(1 − s_L ρ). Then:
(a) s_H is a stable equilibrium if and only if ε ≥ ε_c;
(b) s_L is a stable equilibrium if and only if ε < ε_r;
(c) the system is bistable exactly for ε ∈ [ε_c, ε_r), and has no other stable equilibria.

*Proof.* On {s ≥ θ} the worker never relies (p = 1), so the drift α(1 − s) − δs has the unique zero s_H. On {s < θ} the worker always relies (p = κ), and the unique zero is s_L. A zero of a region's drift is an equilibrium of (1) exactly when it lies inside that region: s_H ≥ θ or s_L < θ respectively. Each is then stable, because the drift is strictly decreasing in s within each region. If s_H < θ, the drift is negative on [θ, 1], so every trajectory starting there enters {s < θ}. If s_L ≥ θ, the drift is positive on [0, θ), so every trajectory enters {s ≥ θ}. At s = θ the drift is discontinuous with the same sign on both sides unless one of the above holds, so θ itself is never an equilibrium. By Condition A, θ is strictly decreasing in ε, so θ(ε) ≤ s_H ⇔ ε ≥ ε_c and θ(ε) > s_L ⇔ ε < ε_r, where ε_c and ε_r solve θ = s_H and θ = s_L from (2). ∎

**Proposition 2 (hysteresis).** If κ < 1 then s_L < s_H and ε_c < ε_r. A worker at s_H whose AI improves past ε_c collapses to s_L, and returns only after the AI degrades past ε_r > ε_c. The hysteresis width is ε_r − ε_c > 0.

*Proof.* s(p) is strictly increasing in p, so κ < 1 implies s_L < s_H. θ is strictly decreasing, so the preimages satisfy ε_c < ε_r. ∎

**Proposition 3 (improvement paradox).** At the collapse point, a marginal improvement of the AI (ε falling from just above to just below ε_c) raises the equilibrium task error of an initially skilled worker discontinuously, from h(s_H) to a(s_L; ε_c). The jump is exactly

  Δerror = Δc + ε_c ρ (s_H − s_L) > 0.

Moreover, there is a band ε ∈ (ε_b, ε_c), with ε_b = h(s_H)/(1 − ρ s_L) < ε_c, in which the deskilled, AI-reliant equilibrium has higher task error than an unaided skilled worker. If the AI becomes unavailable, the deskilled worker's error exceeds the skilled worker's by h₁(s_H − s_L).

*Proof.* At ε_c, D(s_H; ε_c) = 0, i.e. a(s_H; ε_c) = h(s_H) + Δc. Hence a(s_L; ε_c) − h(s_H) = a(s_L; ε_c) − a(s_H; ε_c) + Δc = ε_c ρ (s_H − s_L) + Δc. Since a(s_L; ε) is increasing in ε and exceeds h(s_H) at ε_c, the unique ε_b solving a(s_L; ε_b) = h(s_H) lies below ε_c. The outage statement follows from h. ∎

The intuition is that the marginal worker accepts a *higher* error rate in exchange for the effort saving Δc, which is individually rational under myopia. Once they rely, they lose the skill that made their verification effective, so the AI-assisted error itself rises. Better AI does lower error eventually: below ε_b the deskilled equilibrium beats unaided skilled work. The harm is confined to a band of AI quality, and that band is where current tools plausibly sit for frontier-adjacent tasks (Dell'Acqua et al., 2023).

**Proposition 4 (minimal practice floor; substitution).** With a floor u_min, the reliant equilibrium is s_L(u_min) = s(p_min) with p_min = u_min + κ(1 − u_min). For ε ∈ [ε_c, ε_r), the deskilled equilibrium is eliminated if and only if

  u_min ≥ u*(ε) = [δθ(ε)/(α(1 − θ(ε))) − κ] / (1 − κ),

which is feasible (u* < 1) iff θ(ε) < s_H. More generally, guaranteeing every stable equilibrium a skill of at least s̄ < s_H requires u_min ≥ [δs̄/(α(1 − s̄)) − κ]/(1 − κ). Floors and engagement design enter only through p_min = u_min + κ(1 − u_min), so they are perfect substitutes along iso-p_min curves.

*Proof.* Under the floor, reliance on {s < θ} is 1 − u_min, so the drift there has the zero s(p_min). By the argument of Proposition 1, this is an equilibrium iff s(p_min) < θ. Solving s(p) ≥ θ for p gives p ≥ δθ/(α(1 − θ)), and solving for u_min gives the expression. Replacing θ by s̄ gives the target version. ∎

**Numerical verification** [exp2_verify_props.json]. On 395 random parameter draws satisfying Condition A (forgetting half-life uniform on 3–13 months; α, κ, h₀, h₁, ρ, Δc drawn from wide ranges), the number of stable equilibria found by root-finding on (1) matched Proposition 1 in 1,830 of 1,830 checks. The jump formula of Proposition 3 matched the numeric difference to 1.7 × 10⁻¹⁶. The floor u* of Proposition 4 was sufficient in 370/370 cases, and a floor 0.02 lower left the trap in place in 370/370 cases.

**Baseline values** [exp1_bifurcation.json]. s_H = 0.90, s_L = 0.574 (64% of s_H). ε_c = 0.255, ε_r = 0.470, so the hysteresis width is 0.216. At collapse, equilibrium error jumps from 0.110 to 0.181 (Δerror = 0.071). The worse-than-unaided band is ε ∈ (0.154, 0.255). The outage error gap is 0.195. Minimal practice floors are u* = 0.92, 0.53, 0.28, 0.13 and 0.03 at ε = 0.26, 0.30, 0.35, 0.40 and 0.45. Maintaining s ≥ 0.8 regardless of AI quality requires u_min = 0.35 at κ = 0.15 [exp2]. Figure 3 [fig3_practice_floor.png] shows how rapidly the required floor falls as κ rises.

## 5 Numerical Results

### 5.1 Smooth reliance: bistability needs a sharp response

With logistic reliance, the bistable window shrinks as β falls and converges to the closed-form interval as β → ∞ [exp2]. The windows are [0.305, 0.332] at β = 40; [0.294, 0.401] at β = 100; [0.275, 0.442] at β = 300; [0.263, 0.460] at β = 1000; [0.258, 0.467] at β = 3000; and [0.255, 0.470] in the step limit, against the closed form [0.2545, 0.4704]. There is a critical sharpness β_c below which no AI quality yields bistability [exp7_beta_critical.json]. At baseline β_c = 28.7. It rises steeply with κ: 9.2 (κ = 0), 28.7 (0.15), 54.9 (0.3), 116 (0.5), 273 (0.7). In units, β is the inverse of the reliance advantage (in error-rate points) that multiplies the odds of relying by e. β = 29 means the odds change by e per 3.5 points of error-rate advantage. This is a *measurable* quantity: it is the slope of a logistic regression of trial-level reliance on the difference between one's own and the AI's expected error. Whether real users are above β_c is therefore an empirical question that the protocol in Section 7 addresses.

Figure 1(a) [fig1_bifurcation.png] shows the S-shaped equilibrium curves. In a slow quasi-static sweep at β = 100, skill collapses when ε falls to 0.293 and recovers only when ε rises back to 0.404 (Fig. 1b). Figure 1(c) shows the improvement paradox in the step model.

### 5.2 Critical slowing down near the fold

On the skilled branch, the local recovery rate −f′(s*) is 1.07 per month far from the fold (ε = 0.6). It falls to 0.015 per month at ε = 0.294 for β = 100 and to 0.072 at ε = 0.305 for β = 40 [exp1; fig2_critical_slowing.png]. Close to collapse, a skilled worker therefore takes far longer to recover from a perturbation such as a period of leave. This is a within-person prediction; Section 5.5 tests whether it is usable.

### 5.3 Myopic versus forward-looking reliance

We solved the forward-looking worker's problem, minimizing ∫e^(−r_d t)[r a(s) + (1 − r)(h(s) + Δc)]dt subject to (1), by value iteration on a 401-point skill grid with 21 reliance levels and dt = 0.1 month [exp3_planner.json; fig4_planner.png]. We used three discount rates (5%, 30% and 200% per year), the last a strongly present-biased agent.

* *Starting skilled (s₀ = 0.9).* For ε ∈ {0.22, 0.24, 0.25}, below the myopic collapse point, the myopic worker collapses to s_L while forward-looking workers at all three discount rates stay at s_H. The myopic discounted cost is higher by 11.6/10.0/5.1% at ε = 0.22 and by 26.8/24.9/19.5% at ε = 0.25 (5%/30%/200% per year). At ε = 0.20 only the most patient planner stays skilled (myopia excess 1.4%). For ε ≤ 0.18 all agents rely fully, because the AI is good enough that keeping one's skill is not worth the effort.
* *Starting deskilled (s₀ = s_L).* The myopic worker stays trapped for all ε up to 0.45 (ε_r = 0.47), while forward-looking workers invest in practice and climb to s_H for every ε ≥ 0.20 (5%/yr) or ε ≥ 0.22 (other rates). The myopic excess cost grows with ε, reaching 128/123/106% at ε = 0.45.

The trap is therefore a *myopia* phenomenon. A worker who values their own future skill escapes it. This places the intervention target on decision environments, such as defaults, practice scheduling and feedback on one's own skill, rather than on AI accuracy.

### 5.4 Heterogeneous stochastic populations

We simulated N = 5,000 workers with heterogeneous forgetting half-lives (log-normal, median 6.5 months), skilled levels, κ ∈ U[0.05, 0.30] and effort savings, with β = 100 and skill noise σ = 0.02 per √month. AI error falls from 0.55 to 0.10 over months 60–300, holds, and returns to 0.55 over months 360–600 [exp4_abm.json; fig5_population.png]. A **null model** is identical except that reliance responds to the AI's quality and the worker's *own pre-AI* skill, not to current skill. This removes the feedback while keeping heterogeneity and the same reliance function.

* **Hysteresis discriminates.** At the same AI error ε = 0.30, mean skill relative to own pre-AI level is 0.853 on the way down and 0.661 on the way back up in the feedback model (share of workers below 80% of their pre-AI skill: 34% vs 85%). In the null model the figures are 0.961 vs 0.950 (5.2% vs 8.0%), a small lag only (Fig. 5c).
* **Bimodality does not discriminate.** The bimodality coefficient exceeds the 5/9 benchmark in both models (maximum 0.96 feedback, 0.75 null). In heterogeneous populations, different individual thresholds produce bimodal cross-sections even without feedback. This is a negative result that we did not anticipate. It means cross-sectional skill distributions alone cannot establish a deskilling trap.
* **Policies.** A 30% unaided-practice floor raises the minimum mean relative skill from 0.648 to 0.874. Raising κ by 0.3 through engagement-preserving design raises it to 0.894. Both nearly eliminate the hysteresis loop (Fig. 5c).
* **Withdrawal.** If the AI is withdrawn at month 300 (mean relative skill 0.649), the population regains 90% of the deficit within 3 months. A practice-driven deficit is quickly reversible once practice is forced. In this model the "trap" lies in the *choice* to practise, not in a loss of capacity.

### 5.5 Early-warning signals are weak

For 200 stochastic single-worker runs (σ = 0.01, β = 100) with ε drifting from 0.45 to 0.22 over 50 years, we computed rolling-window variance and lag-1 autocorrelation of detrended monthly skill. Each series was cut 12 months before collapse (median collapse month 413) and compared with matched runs at fixed ε = 0.45 [exp4; fig6_ews.png]. Kendall-τ trends were positive before collapse (median τ_var = 0.16, τ_AC1 = 0.16) and near zero in controls (−0.05, 0.00). Discrimination was weak, however: AUC 0.64 for variance and 0.56 for autocorrelation. Even with a perfect monthly measure of unaided skill, which is rarely available, generic early warnings would be a poor individual screening tool. Direct measurement of κ, β and ρ is more promising.

### 5.6 Sensitivity

One-at-a-time ±50% changes [exp6_sensitivity.json; fig8_sensitivity_kappa.png]:

* s_L/s_H ranges from 0.45 (κ × 0.5) to 0.77 (δ × 0.5).
* The hysteresis width ranges from 0.09 (h₁ × 0.5) to 0.34 (h₁ × 1.5), and stays positive in every case.
* ε_c moves most with verification efficacy: 0.18 at ρ × 0.5, 0.43 at ρ × 1.5. Better verification makes collapse happen at a *worse* AI, because good verifiers find reliance attractive earlier.
* The improvement-paradox jump ranges from 0.045 to 0.135.

In the step model the window never closes for κ < 1 (Proposition 2). In the smooth model, larger κ closes it unless β is large (Section 5.1).

## 6 Consistency Checks Against Published Effect Sizes

All numbers in this section are computed by `exp5_calibration.py` from published summary statistics. We assume (A1) that the outcome measure is proportional to skill s. These checks test whether the mechanism is consistent with the data. They do not estimate parameters.

**Endoscopy (Budzyń et al., 2025).** The published counts give an unassisted-ADR ratio of (145/648)/(226/795) = 0.787 after about three months of AI exposure. Under (1), the largest possible decline is pure forgetting (u = κ = 0).

* If the ADR reflects the *mean* skill over the three-month window, a ratio of 0.787 requires a forgetting half-life of at most **4.1 months**. If it reflects end-of-window skill, the half-life must be at most **8.6 months** [exp5_calibration.json]. The meta-analytic half-lives are 6.5 months for accuracy and 11–13 months for speed and mixed measures (Psychological Bulletin, 2025).
* Endoscopists also kept performing non-AI colonoscopies during the exposure period. We assume that the 648 non-AI and 734 AI-assisted post-exposure procedures reported for the study fall in the same window, which gives an unassisted share u ≈ 0.47. With this share and κ = 0, the model predicts window-mean ratios of 0.926, 0.948 and 0.967 for half-lives of 3, 6.5 and 13 months, and an *asymptotic* ratio of 0.898 (at s_H = 0.9). **No forgetting rate reproduces 0.787** (Fig. 7, fig7_calibration.png).

The reported drop is therefore too large and too fast to be explained by loss of practice alone at meta-analytic forgetting rates. Three explanations remain. (i) A fast attentional or motivational component, in which habituated reliance on AI prompts lowers vigilance during unassisted procedures. (ii) Confounding, such as the changes in case mix and volume raised by critics. (iii) Much faster decay of perceptual-vigilance skills than in the meta-analysis. Explanation (i) predicts rapid reversibility. Our model predicts rapid reversal for practice-driven deficits too (Section 5.4), so a recovery that is *slow* after withdrawal would point to capacity loss. Either way, the check suggests that "deskilling" in this case is not mainly slow skill erosion. A crossover trial with repeated measurements after withdrawal, as Budzyń et al. themselves call for, would separate these accounts.

**Learning studies: implied κ.** For a short acquisition episode starting at baseline skill b (as a fraction of the control group's final skill), the linearized model gives κ = (R − b)/(1 − b), where R is the ratio of AI-group to control-group outcomes. For Shen and Tamkin (2026), R = 50/67 = 0.746, giving κ = 0.75, 0.64 and 0.49 for b = 0, 0.3 and 0.5. For Bastani et al. (2025), R ≈ 0.83 (a 17% reduction), giving κ = 0.83, 0.76 and 0.66. These implied values, crude as they are, are well above our pessimistic baseline κ = 0.15. In the smooth model they raise the critical sharpness to β_c ≈ 116–273 (Section 5.1). If such κ values are typical, a deskilling *trap* requires very sharp reliance responses, and the more common outcome is a moderate, smooth, reversible skill reduction. We consider this an important correction to alarmist readings of the literature. It also identifies κ as the quantity to measure.

## 7 Falsifiable Predictions and a Test Protocol

The model makes predictions that distinguish feedback-driven deskilling from a dose–response account (DR), in which skill loss is proportional to exposure and independent of the user's skill trajectory.

* **P1 (path dependence).** At equal current AI accuracy, users who experienced improving accuracy have lower unaided skill and higher reliance than users who experienced degrading accuracy. DR predicts no difference beyond exposure-time lags.
* **P2 (nonlinearity).** Unaided skill as a function of AI accuracy has a steep segment, a jump in the limit, near a threshold that moves with ρ, Δc and h. DR predicts a smooth, roughly linear relation.
* **P3 (who falls first).** Users with lower baseline skill, or higher effort costs, cross the reliance threshold first and lose a larger share of skill. This matches the junior-developer sample of Shen and Tamkin (2026) and the younger, more AI-dependent users in Gerlich (2025), and should hold within-sample after controlling for exposure.
* **P4 (improvement paradox).** In a population near the threshold, an AI upgrade raises team error on tasks where verification matters (outside the capability frontier) before it lowers it.
* **P5 (substitution).** The practice floor u and the engagement parameter κ affect long-run skill only through u + κ(1 − u). Interventions with equal predicted p_min yield equal long-run unaided skill.
* **P6 (reversibility).** Practice-driven deficits recover within a few forgetting time constants once unaided practice is forced. Slow recovery falsifies the practice mechanism as the main driver.
* **P7 (critical sharpness).** Bistability and hysteresis appear only in groups whose estimated reliance sharpness exceeds β_c(κ).

**Protocol (pre-registrable).** A multi-session online experiment with a perceptual or diagnostic classification task that has a learnable skill component, for example synthetic lesion detection with feedback.

* *Design.* Participants are randomized to (a) ascending versus descending AI-accuracy schedules over 8–10 sessions spread across several weeks, which tests P1 and P2; (b) practice floors u ∈ {0, 0.2, 0.4} crossed with an engagement manipulation that requires a prediction before seeing the AI's answer, which raises κ and tests P5; and (c) a final AI-withdrawal block with repeated probes, which tests P6.
* *Measures.* Unaided probe trials interleaved in every session give s(t). Trial-level reliance choices give β through a logistic regression on the expected-error difference. AI-error catch rates give ρ. Learning on assisted versus unassisted trials gives κ.
* *Analysis.* The pre-registered primary test is the descending-minus-ascending difference in unaided skill at matched accuracy (P1). Secondary tests are the u–κ iso-p_min equivalence (P5, as an equivalence test) and recovery half-life after withdrawal (P6).
* *Power.* Sample size should be set from pilot estimates of between-person variance in s. We do not report a power calculation because we have no pilot data.

## 8 Discussion

**Answering the motivating question.** Does over-reliance on AI cause skill decline? The model gives a conditional answer. Wherever assisted work teaches less than unassisted work (κ < 1), reliance lowers equilibrium skill, from s_H to s(p). That part is close to tautological and is what most of the existing evidence shows. The non-obvious claims concern *when the decline becomes self-reinforcing*: this requires myopic, skill-sensitive reliance that is sharp enough (β > β_c), AI quality in a specific band, and low learning during assisted work. Inside that regime, small changes in AI quality or in effort costs can tip whole groups into a deskilled state that does not reverse when conditions return. Team performance can also get *worse* when the AI gets better. Outside that regime, deskilling is graded and reversible.

**Implications for design and policy.** (i) Interventions should target the decision environment and the learning content of assisted work, κ, rather than AI accuracy. Guardrailed tutoring (Bastani et al., 2025) and engaged interaction patterns (Shen & Tamkin, 2026) are, in our terms, increases in κ. (ii) Proposition 4 gives a principled way to size mandated manual practice, as aviation does for manual flying, from three measurable quantities. (iii) Because the improvement paradox lives in the band of AI quality just better than the collapse point, AI deployments in verification-heavy settings should monitor unaided performance and team error *after* upgrades rather than assume monotone gains. (iv) Generic early-warning indicators will not substitute for periodic unaided assessment.

**Reconciling mixed evidence.** The model predicts heterogeneity across settings. Whether deskilling appears depends on κ, ρ, Δc, the fraction of practice retained, and the AI's position relative to ε_c. Settings in which clinicians keep performing many unassisted procedures (high u) should show small practice-driven effects. Our endoscopy consistency check suggests that the large reported drop, if it is real, involves more than practice loss.

## 9 Limitations

* **Stylization.** Skill is one-dimensional and linear in error. Real expertise has components (perceptual, procedural, metacognitive) that decay at different rates (Casner et al., 2014; Arthur et al., 1998). Verification skill might decay differently from production skill. We model a single s for both.
* **Behavioural assumptions.** The myopic reliance rule ignores social norms, organizational mandates, metacognitive miscalibration (users may misjudge their own s, which Macnamara et al. (2024) argue is likely) and trust dynamics. Miscalibration could either strengthen or weaken the trap, and we did not model it.
* **Parameter values.** Only the forgetting rate is taken from data. α, κ, ρ, h₀, h₁ and Δc are assumed. The implied κ values in Section 6 rely on assumption A1 and on unknown baselines. The bistability results hold across our sensitivity ranges in the step model but need a sharp reliance response in the smooth model, and we have no empirical estimate of β.
* **The consistency checks are coarse.** They use summary statistics, assume proportionality between outcomes and skill, and in the endoscopy case assume that the AI-assisted and non-AI post-exposure procedures fall in the same window. We could not access the full text of every study, and the endoscopy study is observational and contested.
* **No new empirical data.** All results are analytical or simulated. The predictions are untested, and the protocol in Section 7 has not been piloted.
* **Population model.** Workers do not interact. Skill transmission through mentoring, which may matter for future generations of experts, is absent. This is plausibly the most important omitted channel for long-run societal deskilling.

## 10 Conclusion

We formalized AI-induced deskilling as a feedback loop between skill and endogenous reliance. In its sharp-response limit the loop yields closed-form bistability, hysteresis, a discontinuous increase in team error when the AI improves, and an exact minimum practice dose that removes the trap. Numerically, the trap requires myopia and a sharp reliance response. Hysteresis, not bimodality, is its population signature. It is reversible once practice is restored, and generic early-warning signals detect it only weakly. Published effect sizes suggest that learning during assisted work may be higher than pessimistic accounts assume. They also suggest that at least one prominent deskilling result is too large and too fast to be practice loss alone. The most useful next step is empirical: measure κ, ρ and the reliance sharpness β in controlled multi-session experiments, as specified in Section 7.

## Reproducibility

All code is in `code/`. Run `bash code/run_all.sh`. Seeds are fixed, the run takes a few minutes on one CPU core, and it needs Python 3 with numpy, scipy and matplotlib. Outputs are written to `results/` (JSON summaries, logs and fig1–fig8). Every number in Sections 4–6 is taken from these files, except the values labelled as sourced from the cited studies.

## References

All references below were checked to exist via web search in November 2026.

- Arthur, W., Jr., Bennett, W., Jr., Stanush, P. L., & McNelly, T. L. (1998). Factors that influence skill decay and retention: A quantitative review and analysis. *Human Performance*, 11(1), 57–101. doi:10.1207/s15327043hup1101_3
- Bainbridge, L. (1983). Ironies of automation. *Automatica*, 19(6), 775–779. doi:10.1016/0005-1098(83)90046-8
- Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, Ö., & Mariman, R. (2025). Generative AI without guardrails can harm learning: Evidence from high school mathematics. *Proceedings of the National Academy of Sciences*, 122(26), e2422633122.
- Budzyń, K., et al. (2025). Endoscopist deskilling risk after exposure to artificial intelligence in colonoscopy (multicentre, observational study). *The Lancet Gastroenterology & Hepatology*, 10(10), 896–903. doi:10.1016/S2468-1253(25)00133-5
- Casner, S. M., Geven, R. W., Recker, M. P., & Schooler, J. W. (2014). The retention of manual flying skills in the automated cockpit. *Human Factors*, 56(8), 1506–1516. doi:10.1177/0018720814535628
- Dahmani, L., & Bohbot, V. D. (2020). Habitual use of GPS negatively impacts spatial memory during self-guided navigation. *Scientific Reports*, 10. doi:10.1038/s41598-020-62877-0
- Dell'Acqua, F., et al. (2023). Navigating the jagged technological frontier: Field experimental evidence of the effects of AI on knowledge worker productivity and quality. Harvard Business School Technology & Operations Mgt. Unit Working Paper No. 24-013 (SSRN 4573321); later published in *Organization Science*, doi:10.1287/orsc.2025.21838.
- Gerlich, M. (2025). AI tools in society: Impacts on cognitive offloading and the future of critical thinking. *Societies*, 15(1), 6. doi:10.3390/soc15010006
- Kosmyna, N., et al. (2025). Your brain on ChatGPT: Accumulation of cognitive debt when using an AI assistant for essay writing task. arXiv:2506.08872.
- Lee, H.-P., Sarkar, A., Tankelevitch, L., Drosos, I., Rintel, S., Banks, R., & Wilson, N. (2025). The impact of generative AI on critical thinking: Self-reported reductions in cognitive effort and confidence effects from a survey of knowledge workers. In *Proceedings of CHI '25*, Article 1121. doi:10.1145/3706598.3713778
- Macnamara, B. N., et al. (2024). Does using artificial intelligence assistance accelerate skill decay and hinder skill development without performers' awareness? *Cognitive Research: Principles and Implications*, 9, 46. doi:10.1186/s41235-024-00572-8
- Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional integration. *Human Factors*, 52(3), 381–410. doi:10.1177/0018720810376055
- *Procedural skill retention and decay: A meta-analytic review* (2025). *Psychological Bulletin*, 151(6), 696–736. doi:10.1037/bul0000481 (PubMed 40455501). [Author list not confirmed in our search; cited by title and DOI. Cited in the text as "Psychological Bulletin, 2025".]
- Risko, E. F., & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences*, 20(9), 676–688. doi:10.1016/j.tics.2016.07.002
- Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical transitions. *Nature*, 461, 53–59. doi:10.1038/nature08227
- Shen, J. H., & Tamkin, A. (2026). How AI impacts skill formation. arXiv:2601.20245.
- Sparrow, B., Liu, J., & Wegner, D. M. (2011). Google effects on memory: Cognitive consequences of having information at our fingertips. *Science*, 333(6043), 776–778. doi:10.1126/science.1207745
- Strauch, B. (2018). Ironies of automation: Still unresolved after all these years. *IEEE Transactions on Human-Machine Systems*, 48(5), 419–433. doi:10.1109/THMS.2017.2732506
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.
