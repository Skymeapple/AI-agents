# How Many Targets Can a Human Meaningfully Approve? Throughput Bounds on Meaningful Human Control in AI-Assisted Targeting

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES), full paper track

---

## Abstract

Most states, the ICRC and the main policy instruments now agree that the boundary of acceptable military AI runs through human control. The US standard is "appropriate levels of human judgment over the use of force." Elsewhere the standard is "meaningful human control." Neither phrase says how much review a human must give before an approval counts. We address one tractable piece of that gap: AI decision-support systems that nominate targets for human approval. We build a signal-detection model of the approval step. The reviewer's incremental discriminability grows with review time along a standard speed-accuracy curve. The reviewer sets an approval criterion either to meet an operational hit quota or to minimise expected cost, and automation bias enters as a shift in prior log-odds. The model gives closed-form results. (i) Catching a fraction κ of machine errors while approving a fraction h of valid nominations requires incremental discriminability d′ ≥ Φ⁻¹(h) + Φ⁻¹(κ), whatever the machine's precision. (ii) This implies a maximum review rate per operator, λ_max = 3600η / t*. If the reviewer's asymptotic discriminability D does not exceed that threshold, no review rate satisfies the standard. (iii) A cost-minimising reviewer facing a more precise machine needs more discriminability to catch half of its errors. In a large region of parameter space, a more precise machine even increases unlawful approvals per nomination. (iv) Under fixed staffing, unlawful approvals grow superlinearly in nomination volume: the volume elasticity exceeds 1. In our simulations, burstier demand lowers the catch rate at the same mean load. All results were checked by Monte Carlo; the maximum absolute error over 40 configurations × 10⁶ draws was 0.0013. We applied the model to publicly reported figures: about 20 seconds of review per AI-nominated target and about 90% sampled precision. Across our sensitivity grid, the model then puts the share of machine errors caught at between 6% and 88%, and the result turns on two unmeasured human-performance parameters. We therefore propose a concrete audit criterion, an estimator with tested coverage, and a pre-registered protocol for measuring those parameters. The contribution is a way to turn "meaningful human control" into numbers that can be audited. It is not a measurement of any actual system.

## 中文摘要

各国、红十字国际委员会（ICRC）以及主要政策文件普遍认为，军事人工智能的可接受边界取决于"人类控制"。美国的标准是"对使用武力保持适当水平的人类判断"，其他场合的标准是"有意义的人类控制"。但这两种表述都没有说明，人类需要审查到什么程度，批准才算数。本文研究这一缺口中可以处理的一部分：由人工智能决策支持系统提名目标、再由人类审批的流程。我们为审批环节建立了信号检测模型。审查者的增量辨别力 d′ 随审查时间按经典的速度-准确率曲线增长。审查者设定批准标准的方式有两种：满足作战命中配额，或使期望成本最小。自动化偏差表现为先验对数几率的偏移。模型给出以下闭式结果。（1）要在批准比例 h 的有效提名的同时拦截比例 κ 的机器错误，需要 d′ ≥ Φ⁻¹(h)+Φ⁻¹(κ)，且与机器精度无关。（2）由此得到每名操作员审查速率的上限 λ_max = 3600η/t*。若审查者的渐近辨别力 D 不超过该阈值，任何审查速率都无法满足标准。（3）"精度悖论"：对成本最小化的审查者来说，机器越精确，拦截其一半错误所需的辨别力越高。在相当大的参数区域内，更精确的机器甚至会增加每次提名的非法批准数。（4）人员编制固定时，非法批准数随提名量超线性增长（弹性大于 1）。在我们的模拟中，平均负荷相同时，需求越呈突发性，拦截率越低。所有结果都经过蒙特卡洛验证：40 种配置、每种 10⁶ 次抽样，最大绝对误差为 0.0013。我们把模型用于公开报道的数据：每个人工智能提名目标约 20 秒审查时间、抽样精度约 90%。在我们的敏感性网格中，模型得出的机器错误拦截比例在 6% 到 88% 之间，结果取决于两个尚未测量的人类绩效参数。因此，我们提出一个具体的审计标准、一个经过覆盖率检验的估计量，以及一个用于测量这两个参数的预注册实验方案。本文的贡献是把"有意义的人类控制"转化为可审计的数量，而不是对任何实际系统的测量。

---

## 1 Introduction

The question "where is the boundary of AI use in military and autonomous weapons?" has drawn an unusually convergent answer in diplomacy, even though deep disagreements remain about prohibition. UN General Assembly resolution 79/62 (December 2024) and resolution 80/57 (December 2025) on lethal autonomous weapons systems both passed by large majorities [UNGA 2024; UNGA 2025]. The ICRC recommends prohibiting unpredictable and anti-personnel autonomous weapon systems and regulating all others, including through limits that keep a human able to intervene [ICRC 2021; ICRC 2025]. US policy, DoD Directive 3000.09 (2023), requires systems to be designed so that commanders and operators can exercise "appropriate levels of human judgment over the use of force" [DoD 2023]. The 2023 Political Declaration on Responsible Military Use of AI and Autonomy, which dozens of states have endorsed, frames its commitments around human responsibility and accountability [US State Department 2023]. Across these instruments, the boundary is drawn at *human control*.

Human control is easy to state and hard to verify. Critics have noted that 3000.09 never defines "appropriate" [HRW 2023]. Scholars have offered philosophical criteria such as the "tracking" and "tracing" conditions of Santoni de Sio and van den Hoven [2018] and operational accounts of targeting practice [Ekelhof 2019]. These accounts say what human control is *for*. They do not tell an auditor whether a given approval workflow delivers it.

The gap became concrete with reported AI decision-support systems that *nominate* targets for human approval. According to +972 Magazine and Local Call, which relied on six intelligence officers, the Israeli military's "Lavender" system marked up to about 37,000 people as suspected militants. Officers said they spent roughly 20 seconds per target, often only checking that the target was male. A sampled check reportedly found about 90% accuracy [Abraham 2024]. The Israel Defense Forces dispute the account of human involvement. In another case, a former chief of staff described an earlier system that generated about 100 targets a day, against about 50 per year before it [as reported by NPR, Brumfiel 2023]. We do not adjudicate these reports. We use them because they make a precise question unavoidable: *how much review per nomination, by whom, under what incentives, is needed before a human approval makes a measurable difference to whether machine errors become strikes?*

This paper answers that question inside a stated model. We treat "meaningful" in the narrow, testable sense of *error interception*. A human approval stage is meaningful at level κ if, in expectation, it vetoes at least a fraction κ of the machine's erroneous nominations. This is a necessary condition, not a sufficient one. Interception is far from all that meaningful human control involves. A human who intercepts no machine errors is not exercising control over the error-generating process, however the workflow is labelled. A necessary condition is enough to draw a boundary. It does so from the outside: it marks workflows that *fail* any reasonable standard.

**Contributions.**

1. **A model of the approval step (Section 3).** The reviewer's incremental discriminability d′(t) follows the shifted-exponential speed-accuracy function [Wickelgren 1977]. The criterion is set in one of two regimes. In the *quota* regime it meets an operational hit rate. In the *Bayes* regime it minimises expected cost, with automation bias as a prior shift.
2. **Closed-form thresholds and a throughput bound (Section 4, Propositions 1–3).** Under a hit quota h, intercepting κ of errors requires d′ ≥ Φ⁻¹(h)+Φ⁻¹(κ), independent of machine precision. This yields the maximum review rate λ_max per operator. When the reviewer's asymptotic discriminability D falls below that threshold, *no* review rate suffices (an "irreducible" regime).
3. **Two counter-intuitive structural results (Propositions 4–6).** First, a precision paradox: with a cost-minimising reviewer, better machines need more discerning humans to catch the same share of errors. Second, a sign result: when the reviewer leans toward veto, a more precise machine *increases* unlawful approvals per nomination. Third, under fixed staffing, unlawful approvals have volume elasticity above one. Simulation adds a fourth point: bursty demand lowers interception at constant mean load.
4. **Validation and sensitivity analysis (Section 5).** Monte Carlo verification of every closed form. A 216-configuration sensitivity grid. A robustness check with a non-saturating, diffusion-type speed-accuracy curve. An illustrative calculation using the reported 20-second figure, with the explicit result that the answer depends on two human-performance parameters nobody has measured for this task.
5. **An audit criterion and a pre-registered protocol (Section 6).** A veto-rate inequality that any auditor can check from logs. A stratified-audit estimator of the catch rate with tested interval coverage. An experimental protocol, with falsifiable predictions, for measuring the missing parameters.

We do not claim that the model describes any real system. We claim that it identifies *which quantities* an approval-based boundary depends on, gives the functional form of that dependence, and shows that current policy language never mentions those quantities.

## 2 Related Work

**Normative accounts of meaningful human control.** Santoni de Sio and van den Hoven [2018] ground meaningful human control (MHC) in guidance control and propose two conditions. *Tracking* requires the system to respond to the relevant moral reasons of relevant humans. *Tracing* requires outcomes to be traceable to a human who understands the system. Ekelhof [2019] argues that MHC has to be assessed across the military targeting cycle, not at a single moment of trigger-pulling. Scharre [2018] surveys the spectrum from human-in-the-loop to human-out-of-the-loop systems and the risks of speed and scale. Verhagen et al. [2024] note that designers of variable-autonomy systems lack *metrics* for MHC. Our work supplies one metric for one stage, the approval of machine nominations, and connects it to Ekelhof's point: the throughput upstream of approval determines whether approval can be meaningful.

**Policy instruments.** DoD Directive 3000.09 [DoD 2023], the Political Declaration [US State Department 2023], the ICRC's position [ICRC 2021; 2025] and the UNGA resolutions [UNGA 2024; 2025] all invoke human judgment or control without operational thresholds. Our propositions express such a threshold in terms of review time, staffing and quota. A regulator could write those quantities into a rule.

**Automation bias and complacency.** Skitka, Mosier and Burdick [1999] documented omission and commission errors with automated aids. Parasuraman and Manzey [2010] integrate complacency and bias in an attentional framework and report that neither training nor instructions eliminate bias. Goddard, Roudsari and Wyatt [2012] list workload and time pressure among the mediators. Cummings [2004] analyses automation bias in time-critical military decision support. Hawley [2017] describes the Patriot fratricides and the "dangerous middle ground" in which an undertrained crew turns an automated system into a de facto fully automated one. In a nine-country experiment with 9,000 adults, Horowitz and Kahn [2024] find a non-linear, Dunning-Kruger-like relationship between AI knowledge and automation bias in national-security tasks. These studies establish the phenomenon. Our model converts it into a bias parameter α and shows how α raises the required discriminability.

**Signal detection and responsibility quantification.** Signal detection theory (SDT) [Green & Swets 1966] separates sensitivity from response bias. Langer, Baum and Schlicker [2025] propose SDT as the framework for analysing human oversight of AI outputs. Douer and Meyer's ResQu model [Douer & Meyer 2018] uses information theory to quantify the human's causal responsibility in automated systems. It shows that this responsibility is small when the automation's capabilities greatly exceed the human's. Our work differs from ResQu in three ways. (a) We make human sensitivity a *function of review time* and derive the throughput bound. (b) We model the institutional criterion-setting regime (quota versus Bayes) explicitly and show that results differ qualitatively between them. (c) We target an auditable rate cap rather than a responsibility share. The precision paradox (Proposition 4) is consistent in spirit with ResQu's finding that better automation shrinks the human's contribution. Here it appears as a concrete threshold and as a sign reversal in harm.

**Speed-accuracy tradeoffs.** Wickelgren [1977] established the speed-accuracy tradeoff (SAT) function, the dependence of accuracy on processing time, as the main measurement object. The shifted-exponential form we use is the standard way to summarise SAT curves measured with experimenter-imposed deadlines. Because the form is an assumption, we also test a √t growth law of the kind implied by diffusion models.

**Gap.** We found no prior work that derives, from an explicit human-performance model, a *maximum number of nominations per operator-hour* consistent with a stated level of human error interception, or that analyses how that bound interacts with machine precision, operational quotas and demand burstiness.

## 3 Model

### 3.1 Setting

A machine produces a stream of *nominations*: proposed targets, or proposed engagements for a defensive system. Each nomination is *valid* (lawful and correctly identified) with probability q, the machine's precision; otherwise it is a *machine error*. A human reviewer sees each nomination for t seconds and either approves it, so that it proceeds to engagement, or vetoes it. We consider only this approval stage. Errors elsewhere, such as in weaponeering, collateral-damage estimation or execution, are outside the model.

### 3.2 Reviewer evidence

Following SDT, the reviewer's evidence about a nomination is a scalar

x ~ N(+d′(t)/2, 1) if the nomination is valid, x ~ N(−d′(t)/2, 1) if it is an error,

and the reviewer approves iff x > c. The quantity d′(t) is the reviewer's **incremental discriminability**: their ability to separate valid nominations from machine errors *given that the machine has already nominated them*. This conditioning matters. If the reviewer sees only what the machine saw, or only the machine's output, incremental discriminability can be near zero even for a highly skilled analyst. Opacity, the absence of independent evidence and the lack of a ground-truth channel all lower D, the asymptote below.

**Speed-accuracy function.** We model

d′(t) = D·(1 − exp(−(t−δ)/τ)) for t > δ, and 0 otherwise, (1)

where D is the asymptotic incremental discriminability with unlimited time, τ is the time constant of evidence accumulation, and δ is the non-decision time (reading, orienting). Form (1) is the standard SAT summary [Wickelgren 1977]. As a robustness check we also use d′(t) = k·√(t−δ), which never saturates.

### 3.3 Throughput

An operator who devotes a fraction η of each hour to review and faces λ nominations per hour has

t = 3600·η / λ seconds per nomination. (2)

### 3.4 Criterion regimes

**Quota regime.** The organisation, or the operator under pressure, sets c so that a fixed fraction h of valid nominations is approved (the hit rate), for example because rejecting valid targets is visible and costly to operational tempo. Then c = d′/2 − Φ⁻¹(h), and the false-approval rate is

FA = Φ(Φ⁻¹(h) − d′). (3)

**Bayes regime.** The reviewer minimises expected cost, with C_F the cost of approving an error (an unlawful or mistaken strike), C_M the cost of vetoing a valid nomination, and r = C_F/C_M. Using subjective prior log-odds that may be inflated by automation bias α ≥ 0, the reviewer approves iff the log-likelihood ratio d′·x exceeds

ln β = ln((1−q)/q) + ln r − α, (4)

that is, c = ln β / d′. Then FA = Φ(−d′/2 − ln β/d′) and the hit rate is Φ(d′/2 − ln β/d′). If ln β < 0 the reviewer leans toward approval (*approve-default*); if ln β > 0 the reviewer leans toward veto (*veto-default*).

**Outcome measures.** The *catch rate* κ = 1 − FA is the share of machine errors the human vetoes. *Unlawful approvals per nomination* are U = (1−q)·FA, and per operator-hour they are λ·U. The *veto rate* is V = q(1−H) + (1−q)(1−FA), where H is the hit rate.

**Definition (κ-meaningful approval).** An approval stage is κ-meaningful if its expected catch rate is at least κ. The definition is deliberately minimal. Section 7 discusses what it leaves out.

## 4 Theoretical Results

**Proposition 1 (quota threshold).** In the quota regime with hit rate h, the catch rate is κ(d′) = Φ(d′ − Φ⁻¹(h)). The approval stage is κ-meaningful iff

d′ ≥ d*_Q(h, κ) := Φ⁻¹(h) + Φ⁻¹(κ). (5)

The threshold does not depend on machine precision q. As d′ → 0 the catch rate tends to 1 − h: the vetoes become random with respect to validity.

*Proof.* From (3), 1 − FA = 1 − Φ(Φ⁻¹(h) − d′) = Φ(d′ − Φ⁻¹(h)), which is increasing in d′. Setting it ≥ κ gives (5). The limit follows by setting d′ = 0. ∎

Representative values are d*_Q(0.95, 0.5) = 1.645, d*_Q(0.95, 0.9) = 2.926 and d*_Q(0.99, 0.9) = 3.608 (computed; `results/summary.json`, E3). A d′ of 1.645 means the reviewer's evidence separates a valid nomination from a machine error with an area under the ROC curve of Φ(1.645/√2) ≈ 0.88. This is substantial independent discrimination on items that have *already passed* the machine's filter.

**Proposition 2 (throughput bound and irreducibility).** Under (1) and (2), the approval stage is κ-meaningful iff each operator handles at most

λ_max = 3600·η / t*, with t* = δ + τ·ln(D / (D − d*)), (6)

nominations per hour, where d* is the relevant threshold (d*_Q in the quota regime). If D ≤ d*, then λ_max = 0: no review time, however long, makes the stage κ-meaningful.

*Proof.* d′(t) in (1) is continuous and strictly increasing for t > δ, with supremum D. If D > d*, the unique solution of d′(t) = d* is t*, and d′(t) ≥ d* iff t ≥ t*. Substituting (2) gives λ ≤ 3600η/t*. If D ≤ d*, then d′(t) < d* for all t. ∎

The irreducible case is the formal counterpart of a familiar qualitative worry: when the human has no independent evidence (D small), keeping a human "in the loop" does not intercept errors at any tempo. Note the policy consequence: no rate limit or minimum review time can satisfy the definition in that case. The only remedies change D, for example by giving the reviewer independent intelligence sources, explanations they can act on, or feedback on ground truth.

**Proposition 3 (Bayes regime: precision paradox and automation bias).** In the approve-default Bayes regime (ln β < 0), the stage is κ-meaningful iff

d′ ≥ d*_B := z + √(z² + 2|ln β|), z = Φ⁻¹(κ). (7)

d*_B is increasing in q and in α, and decreasing in r. As d′ → 0 the catch rate tends to 0: the reviewer approves everything. In the veto-default regime the catch rate tends to 1 as d′ → 0, and the system becomes unusable because valid nominations are also vetoed.

*Proof.* With c = ln β/d′, 1 − FA = Φ(d′/2 + ln β/d′). The condition Φ(·) ≥ κ is d′/2 − |ln β|/d′ ≥ z. Multiplying by d′ > 0 gives the quadratic d′²/2 − z·d′ − |ln β| ≥ 0, whose positive root is (7). |ln β| = ln(q/(1−q)) − ln r + α is increasing in q and α and decreasing in r, and (7) is increasing in |ln β|. For the limits, c → −∞ (respectively +∞) as d′ → 0 when ln β < 0 (respectively > 0). ∎

The *precision paradox* is the monotonicity in q. A cost-minimising reviewer who correctly believes the machine is more accurate rationally needs more contrary evidence before vetoing. To catch half of a better machine's errors therefore takes a more discerning human. For instance, with r = 1, d*_B(κ = 0.5) is 2.10 at q = 0.9 and 3.03 at q = 0.99. Automation bias acts as an extra increment α to |ln β|. With q = 0.9 and r = 5, raising α from 0 to 1 nat raises d*_B from 1.08 to 1.78 (E4).

**Proposition 4 (sign reversal: better machines can increase unlawful approvals).** In the Bayes regime with fixed d′ > 0, unlawful approvals per nomination U(q) = (1−q)·Φ(a), with a = −d′/2 − ln β(q)/d′, satisfy

dU/dq > 0 ⟺ φ(a)/Φ(a) > q·d′. (8)

A sufficient condition is ln β ≥ d′²(q − ½). Since q − ½ < ½, every veto-default configuration with ln β ≥ ½d′² has dU/dq > 0.

*Proof.* dlnβ/dq = −1/(q(1−q)), so da/dq = 1/(d′q(1−q)), and dU/dq = −Φ(a) + (1−q)φ(a)·da/dq = −Φ(a) + φ(a)/(d′q). This gives (8). For the sufficient condition, if ln β ≥ 0 then a < 0, and Gordon's inequality [Gordon 1941] gives φ(a)/Φ(a) > −a = d′/2 + ln β/d′. This exceeds q·d′ whenever ln β ≥ d′²(q − ½). ∎

The intuition: a reviewer who leans toward veto, because they rightly weigh wrongful strikes heavily, relaxes the criterion as the machine's precision rises. Approvals of the remaining errors can rise faster than the number of errors falls. In absolute terms these per-nomination rates are small in the veto-default region (Section 5.3). The practical point is that "a more accurate AI" does not monotonically reduce harm once the human stage adapts to it. Claims of accuracy improvement therefore do not, on their own, show that a system has become safer.

**Proposition 5 (superlinear harm in volume).** Under fixed staffing and either regime, unlawful approvals per operator-hour, 𝒰(λ) = λ(1−q)FA(t(λ)), have elasticity

ε(λ) = d ln 𝒰 / d ln λ = 1 + d ln FA / d ln λ ≥ 1,

with strict inequality wherever t(λ) > δ and d′(t(λ)) < D, in the quota regime and in the approve-default Bayes regime.

*Proof.* FA is decreasing in d′ in both cases: by (3) in the quota regime, and in the approve-default Bayes regime because ∂FA/∂d′ = −φ(a)(1/2 − ln β/d′²) < 0 when ln β < 0. d′ is increasing in t on (δ, ∞), and t is decreasing in λ by (2). So FA is non-decreasing in λ, strictly so where d′ is strictly increasing. ∎

Thus doubling the nomination volume without doubling the review staff *more than doubles* unlawful approvals, because each review becomes shallower. As λ → ∞, ε → 1 and the human stage becomes transparent: 𝒰 → λ(1−q)h in the quota regime, and λ(1−q) in the approve-default Bayes regime.

**Proposition 6 (audit inequality).** For any reviewer, V ≥ (1−q)κ. Hence an observed veto rate V_obs and an audited precision q̂ give the bound κ ≤ V_obs / (1−q̂).

*Proof.* V = q(1−H) + (1−q)κ ≥ (1−q)κ. ∎

This is elementary, but it needs only logs and a precision audit, which some operators report already. An approval stage that vetoes 1% of nominations of a machine with 10% errors catches at most 10% of those errors, whatever its reviewers believe.

## 5 Computational Results

All numbers in this section come from `code/experiments.py`, run in this environment (≈10 s CPU; seed 20261122). Outputs are in `results/summary.json` and the figures referenced below. Every parameter value for D, τ, δ, h, r and α is an *assumption swept over a grid*, not an estimate. No study we know of measures them for target vetting.

### 5.1 Validation (E1)

We compared the closed-form hit and false-approval rates against Monte Carlo simulation with 10⁶ draws per class across 40 configurations: both regimes, d′ ∈ {0.25, …, 3}, several (q, r, α), and h ∈ {0.9, 0.95, 0.99}. The maximum absolute discrepancy was 0.0013, which is consistent with sampling error at n = 10⁶. Substituting the thresholds d*_Q and d*_B back in reproduced the target κ to within 1.1 × 10⁻¹⁶. For Proposition 4, the analytic condition (8) matched the sign of a numerical derivative at all 3 × 90,000 grid points (agreement 1.000; E4b).

### 5.2 Catch rate versus review time, and the throughput bound (E2, E3)

Figure 1 (`results/fig1_catch_vs_time.png`) plots the catch rate against review time for D ∈ {1, 2, 3, 4} and τ ∈ {15, 60, 240} s, with δ = 2 s and quota h = 0.95. At t = 20 s the catch rate varies from 0.058 (D = 1, τ = 240 s) to 0.875 (D = 4, τ = 15 s). At τ = 60 s it lies between 0.083 and 0.272 for all D. In other words, a 20-second review intercepts most machine errors only if the reviewer's evidence both accumulates quickly (τ ≈ 15 s) and is ultimately strong (D ≥ 3). If either condition fails, the reviewer approves most errors.

Table 1 reports t* and λ_max (η = 1) for κ = 0.5 and h = 0.95 (d*_Q = 1.645).

| D | τ (s) | t* (s) | λ_max (nominations / operator-hour) |
|---|---|---|---|
| 1 | any | ∞ (irreducible) | 0 |
| 2 | 15 | 27.9 | 128.9 |
| 2 | 60 | 105.7 | 34.1 |
| 2 | 240 | 416.8 | 8.6 |
| 3 | 15 | 13.9 | 258.6 |
| 3 | 60 | 49.7 | 72.5 |
| 3 | 240 | 192.7 | 18.7 |
| 4 | 15 | 9.9 | 362.0 |
| 4 | 60 | 33.8 | 106.6 |
| 4 | 240 | 129.1 | 27.9 |

*Table 1. Computed from Eq. (6); δ = 2 s. Source: `results/e3_lambda_max_grid.csv`.*

Over the full grid of 216 configurations (D × τ × δ × h ∈ {0.9, 0.95, 0.99} × κ ∈ {0.5, 0.8, 0.9}), 50% were irreducible (D ≤ d*). Among the feasible ones, the median t* was 71 s, and only 14.8% had t* ≤ 20 s. Figure 2 (`results/fig2_lambda_max.png`) maps λ_max over a continuum of (D, τ). The irreducibility boundary D = d* is a vertical wall, and contours of 10, 30, 60 and 180 nominations per hour run across the feasible region.

**Robustness to the SAT form.** With the non-saturating law d′ = k√(t−δ), and k calibrated so that d′(60 s) ∈ {1, 2, 3}, we get t* = 164.3, 42.6 and 20.0 s respectively (h = 0.95, κ = 0.5). No configuration is irreducible under this law, by construction. Otherwise the conclusions are unchanged in kind: reaching κ = 0.5 in 20 s requires a reviewer who would reach d′ = 3 within a minute.

### 5.3 Precision paradox and sign reversal (E4, E4b)

Figure 3 (`results/fig3_precision_paradox.png`) shows the Bayes regime at d′ = 1. With r = 5, the catch rate falls from 0.765 at q = 0.8 to 0.465 at q = 0.9 and 0.006 at q = 0.99. Unlawful approvals per nomination are *higher* at q = 0.9 (0.0535) than at q = 0.8 (0.0470), and lower again at q = 0.99 (0.0099). With r = 20, unlawful approvals per nomination are 0.0035, 0.0097 and 0.0086 at q = 0.8, 0.9 and 0.99. With r = 1, the catch rate at q = 0.9 is 0.045, barely above zero.

Figure 5 (`results/fig5_nonmonotone_region.png`) maps the region where dU/dq > 0 over q ∈ [0.5, 0.995] and r ∈ [1, 1000] (log-uniform grid). It covers 86.7%, 86.5% and 78.9% of the grid at d′ = 0.5, 1 and 2. It contains the whole veto-default region at d′ = 0.5 and d′ = 1 and extends slightly into the approve-default region. These shares depend on the grid chosen and are not population frequencies. The robust statement is the location of the region: everywhere the reviewer is veto-leaning and evidence is weak.

Automation bias at q = 0.9 and r = 5 lowers the catch rate at d′ = 1 from 0.465 (α = 0) to 0.278 (α = 0.5), 0.138 (α = 1) and 0.018 (α = 2).

### 5.4 Volume elasticity (E5)

Figure 4 (`results/fig4_volume_elasticity.png`) shows unlawful approvals per operator-hour against nomination rate in the quota regime (q = 0.9, h = 0.95). The peak elasticity was 1.54 (D = 2, τ = 15 s; at λ ≈ 150 per hour), 2.31 (D = 3, τ = 60 s; λ ≈ 38 per hour) and 3.57 (D = 4, τ = 240 s; λ ≈ 9.5 per hour). In the steep region, a 10% increase in nominations per reviewer raises unlawful approvals by roughly 15–36%. For better reviewers (larger D) the peak is higher and occurs at lower volume: the more a reviewer's quality depends on time, the more is lost when time is taken away.

### 5.5 Demand burstiness (E6)

We simulated 2,000 days of hourly nominations to a cell of N = 10 operators (η = 0.8, preferred review time 120 s, D = 3, τ = 60 s, δ = 2 s, q = 0.9, h = 0.95), with mean demand 300 per hour. Nominations are time-sensitive and must be cleared within the hour, so review time is t_h = min(120, 3600ηN / A_h). Under constant load (t = 96 s), the catch rate was 0.767. With Poisson arrivals (coefficient of variation 0.06) it was 0.766. With negative-binomial surges at the same mean, the error-weighted catch rate fell to 0.697 (CV 0.45), 0.552 (CV 1.00) and 0.379 (CV 1.82). Unlawful approvals per day rose from 168 to 218, 321 and 443. Most errors arrive during surges, which are exactly when reviews are shortest. Staffing for mean load therefore overstates interception. A rate cap of the form (6) has to bind on peak rates, not averages.

### 5.6 Illustrative calculation with reported figures (E7)

We take the reported figures at face value, as journalistic allegations disputed by the IDF: about 37,000 nominations, about 20 s of review each, and about 90% sampled precision [Abraham 2024]. That implies about 205.6 reviewer-hours in total and about 3,700 erroneous nominations. Table 2 gives the reviewer-hours that the model says would be required.

| D | τ (s) | hours for κ = 0.5 | hours for κ = 0.9 |
|---|---|---|---|
| 1 | any | irreducible | irreducible |
| 2 | 15 | 287 | irreducible |
| 2 | 60 | 1,086 | irreducible |
| 3 | 15 | 143 | 592 |
| 3 | 60 | 511 | 2,307 |
| 3 | 240 | 1,981 | 9,166 |
| 4 | 60 | 347 | 832 |

*Table 2. Reviewer-hours for 37,000 nominations at quota h = 0.95, δ = 2 s (E7).*

At 20 s per nomination, the model's catch rate ranges from 0.06 to 0.88 across our grid (Section 5.2). The reported effort is *consistent with* meaningful interception only in the corner of fast, strong independent evidence (D ≥ 3, τ ≈ 15 s). It is inconsistent with interception at the κ = 0.9 level under every configuration in the grid. Even the best case, D = 4 and τ = 15 s, needs 21.7 s. For comparison, the reported earlier rate of about 100 nominations per day would require 0.3–24.8 reviewer-hours per day for κ ∈ {0.5, 0.9} at D ∈ {3, 4} across all τ in the grid. So volume as such is not the problem. The product of volume and review depth, given D, is. We stress that we do not know D or τ for this or any targeting task. That is exactly the measurement gap Section 6 addresses.

### 5.7 Auditing the catch rate (E8)

Proposition 6 gives an upper bound from logs alone. For a point estimate, we simulated a stratified ground-truth audit: n approved and n vetoed nominations are sampled and checked, and the results are combined with the logged veto rate V. The setting was q = 0.9, d′ = 1 and h = 0.95, giving a true κ of 0.260 and V = 0.071. The estimator κ̂ = V·p̂_v / (V·p̂_v + (1−V)·p̂_a), where p̂_a and p̂_v are the audited error shares among approved and vetoed nominations, had a mean (SD) of 0.295 (0.136) at n = 50 per stratum, 0.268 (0.052) at n = 200 and 0.262 (0.024) at n = 800, over 1,000 replications. Nominal 95% bootstrap intervals covered the true value in 92.7%, 93.8% and 94.3% of replications respectively. Auditing a few hundred cases per stratum therefore pins down the catch rate to about ±0.1. This is achievable for after-action review, and the ICRC's and other proposals for operational limits could require it.

## 6 From Model to Boundary: An Operational Proposal and a Test Protocol

**A rule that can be written down.** The model suggests stating the human-control requirement for nomination-and-approval systems as three auditable commitments:

1. *Declared interception level.* The operator declares a minimum catch rate κ (for example 0.5 or 0.9) and a hit quota h.
2. *Measured reviewer performance.* Before deployment, the operator measures (D, τ, δ) for the actual reviewer population, task and interface, using the protocol below.
3. *Derived rate cap, binding at peak.* The operator enforces per-reviewer nomination rates ≤ λ_max from Eq. (6), computed at peak demand, together with logging that supports the audit in Section 5.7 and the inequality of Proposition 6.

If the measured D ≤ d*, the system fails the human-control requirement *as an approval system*, whatever the staffing. It must then either be redesigned to give reviewers independent evidence or be treated, for legal and policy purposes, as autonomous at the selection stage. This turns the model's irreducibility result into a boundary rule. It also explains, in quantitative terms, why the ICRC's emphasis on predictability and on limits to scale and duration [ICRC 2021] belongs in the same framework: scale enters through λ, duration and surge through the burstiness effect.

**Pre-registered protocol for measuring (D, τ, δ) and testing the propositions.** *Task:* participants vet machine-nominated items in a synthetic target-identification task (for example, distinguishing simulated military vehicles from civilian vehicles in degraded imagery, with structured "intelligence" cues), with machine nominations at controlled precision q ∈ {0.8, 0.9, 0.97}. *Manipulation:* response deadlines t ∈ {3, 6, 10, 20, 40, 80, 160} s (response-signal method), instructions inducing either a hit quota or a cost framing (r ∈ {1, 5, 20}), and an automation-bias manipulation (described machine reliability). *Population:* lay participants, plus a sample with relevant professional training, given the knowledge-dependence that Horowitz and Kahn [2024] report. *Pre-registered predictions:*
- P1 (form): d′(t) estimated per deadline is fit better by Eq. (1) than by a constant, and estimated D differs across interface conditions (independent evidence versus machine output only), with D near zero in the machine-output-only condition.
- P2 (quota invariance): under quota instructions, the catch rate at fixed t does not depend on q (Proposition 1).
- P3 (precision paradox): under cost framing, the catch rate at fixed t decreases with q (Proposition 3).
- P4 (sign reversal): under high r and short t, unlawful approvals per nomination are higher at q = 0.9 than at q = 0.8 (Proposition 4).
- P5 (volume): with a fixed session time and the number of items varied, the elasticity of false approvals with respect to item count exceeds 1 (Proposition 5).

Each prediction can fail. Failure of P2 or P3 would show that real reviewers do not set criteria in either of our regimes, which would itself be policy-relevant.

## 7 Discussion

**What the model says about "where the boundary is".** The boundary of acceptable AI use in targeting is often discussed in terms of the *machine*: its autonomy level, accuracy and predictability. Our results show that for nomination-and-approval systems, a necessary condition for human control is a property of the *socio-technical pipeline*: the product of nomination volume per reviewer and the reviewer's time-dependent incremental discriminability, filtered through the institution's criterion-setting regime. Three implications follow.

First, *accuracy claims do not substitute for review capacity*. Under a quota, the required discriminability is independent of machine precision (Proposition 1). Under a cost framing, a more precise machine *raises* the discriminability needed to catch its errors (Proposition 3) and can raise per-nomination harm (Proposition 4). Asserting that a system is "90% accurate" therefore says nothing about whether the humans approving its outputs exercise control.

Second, *speed and scale are not neutral efficiency gains*. Proposition 5 and the burstiness simulation show that, at fixed staffing, scaling nomination output degrades the human stage faster than it adds nominations. "Machine-speed targeting" is therefore in tension with human-control commitments as a matter of arithmetic, not only of sentiment. A per-reviewer rate cap binding at peak is the simplest instrument consistent with the model.

Third, *some workflows cannot be fixed by slowing down*. When reviewers lack independent evidence, D is small, and Proposition 2 says that no review time suffices. Such systems are effectively autonomous at the selection stage. The legal and policy analysis that applies to autonomous selection should apply to them, whatever the formal presence of a human approver. This sharpens the concern behind the "dangerous middle ground" [Hawley 2017] and the ResQu observation that the human's share of causal responsibility shrinks as automation outperforms them [Douer & Meyer 2018].

**Relationship to normative accounts.** Error interception is closest to the *tracking* condition of Santoni de Sio and van den Hoven [2018]. A human stage that cannot discriminate machine errors cannot make the system track the reasons that make a target unlawful. Our framework does not address tracing (who understood and is accountable), and it does not settle what κ ought to be. That is a normative choice which the model only makes explicit and costable.

## 8 Limitations

- **Parameters unmeasured.** D, τ and δ are unknown for any real targeting task, so all quantitative conclusions are conditional on the swept grids. The 50% irreducibility share and the 14.8% "≤ 20 s" share depend on our grid. Different grids would give different shares.
- **Equal-variance Gaussian SDT.** Real evidence may be multi-dimensional, unequal-variance or heavy-tailed. Item difficulty may correlate with machine confidence, which would probably make the machine's errors harder for humans to detect than its average nomination and lower the effective D.
- **Independence across items.** We treat nominations as independent. In reality, errors can be correlated (a systematic misclassification of a population, as in the training-data gap reported for an earlier system [Brumfiel 2023]), and reviewer fatigue and learning vary over a shift. Correlated errors make κ a poor summary of harm concentration.
- **Two stylised criterion regimes.** Organisations may use mixtures, quotas that drift over time, or rubber-stamp norms that neither regime captures. Automation bias is modelled only as a prior shift. It may also reduce attention, which would lower d′ and act through D and τ.
- **Approval stage only.** Proportionality assessment, collateral-damage estimation, timing decisions (for example, when to strike a located target) and weapon choice are outside the model. Harm per unlawful approval is not modelled.
- **Necessary, not sufficient.** κ-meaningfulness is a floor. A high catch rate does not establish lawfulness, legitimacy or accountability.
- **Reported figures.** The Lavender and Gospel figures are journalistic reports based on anonymous sources and are disputed by the IDF. We use them only to illustrate the calculation, not to estimate any system's actual catch rate.
- **No human-subject data.** We have not run the protocol in Section 6. All numerical results are from the model and simulation.

## 9 Conclusion

"Meaningful human control" and "appropriate human judgment" are where most of the international community has drawn the boundary of military AI. For AI systems that nominate targets for human approval, we have shown that this boundary has measurable content. A human stage intercepts a fraction κ of machine errors only if the reviewer's incremental discriminability exceeds Φ⁻¹(h)+Φ⁻¹(κ) (quota regime) or z+√(z²+2|ln β|) (cost regime). That translates into a maximum number of nominations per reviewer-hour. Better machines, higher volume and burstier demand each make the requirement harder to meet, and when reviewers lack independent evidence it cannot be met at all. The quantities that decide whether control is meaningful are review time, reviewer discriminability, quotas and peak load. Current policy language does not mention them, yet each can be measured and audited. We propose measuring them.

## References

- Abraham, Y. (2024, April 3). "'Lavender': The AI machine directing Israel's bombing spree in Gaza." *+972 Magazine* and *Local Call*.
- Brumfiel, G. (2023, December 14). "Israel is using an AI system to find targets in Gaza. Experts say it's just the start." *NPR*. https://www.npr.org/2023/12/14/1218643254/israel-is-using-an-ai-system-to-find-targets-in-gaza-experts-say-its-just-the-st
- Cummings, M. L. (2004). "Automation bias in intelligent time critical decision support systems." Presented at the AIAA 1st Intelligent Systems Technical Conference, September 2004. Reston, VA: American Institute of Aeronautics and Astronautics.
- Department of Defense (2023). *DoD Directive 3000.09: Autonomy in Weapon Systems* (effective January 25, 2023). Office of the Under Secretary of Defense for Policy. https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodd/300009p.PDF
- Douer, N., & Meyer, J. (2018). "The Responsibility Quantification (ResQu) Model of Human Interaction with Automation." arXiv:1810.12644.
- Ekelhof, M. (2019). "Moving beyond semantics on autonomous weapons: Meaningful human control in operation." *Global Policy*, 10(3), 343–348. https://doi.org/10.1111/1758-5899.12665
- Goddard, K., Roudsari, A., & Wyatt, J. C. (2012). "Automation bias: a systematic review of frequency, effect mediators, and mitigators." *Journal of the American Medical Informatics Association*, 19(1), 121–127. https://doi.org/10.1136/amiajnl-2011-000089
- Gordon, R. D. (1941). "Values of Mills' ratio of area to bounding ordinate and of the normal probability integral for large values of the argument." *Annals of Mathematical Statistics*, 12, 364–366.
- Green, D. M., & Swets, J. A. (1966). *Signal Detection Theory and Psychophysics*. New York: Wiley.
- Hawley, J. K. (2017). *Patriot Wars: Automation and the Patriot Air and Missile Defense System*. Center for a New American Security (Ethical Autonomy Series), January 2017. https://www.cnas.org/publications/reports/patriot-wars
- Horowitz, M. C., & Kahn, L. (2024). "Bending the automation bias curve: A study of human and AI-based decision making in national security contexts." *International Studies Quarterly*, 68(2), sqae020. https://doi.org/10.1093/isq/sqae020
- Human Rights Watch (2023, February 14). "Review of the 2023 US Policy on Autonomy in Weapons Systems." https://www.hrw.org/news/2023/02/14/review-2023-us-policy-autonomy-weapons-systems
- International Committee of the Red Cross (2021, May 12). *ICRC Position on Autonomous Weapon Systems*. Geneva. https://www.icrc.org/en/document/icrc-position-autonomous-weapon-systems
- International Committee of the Red Cross (2025, October). *Autonomous Weapon Systems and International Humanitarian Law: Selected Issues* (Position Paper). https://www.icrc.org/sites/default/files/media_file/2025-10/ICRC-Position_Paper-Autonomous_Weapon_Systems_and_IHL-Selected_issues_Oct2025.pdf
- Langer, M., Baum, K., & Schlicker, N. (2025). "Effective human oversight of AI-based systems: A signal detection perspective on the detection of inaccurate and unfair outputs." *Minds and Machines*, 35(1), 1. https://doi.org/10.1007/s11023-024-09701-0
- Parasuraman, R., & Manzey, D. H. (2010). "Complacency and bias in human use of automation: An attentional integration." *Human Factors*, 52(3), 381–410. https://doi.org/10.1177/0018720810376055
- Santoni de Sio, F., & van den Hoven, J. (2018). "Meaningful human control over autonomous systems: A philosophical account." *Frontiers in Robotics and AI*, 5, 15. https://doi.org/10.3389/frobt.2018.00015
- Scharre, P. (2018). *Army of None: Autonomous Weapons and the Future of War*. New York: W. W. Norton.
- Skitka, L. J., Mosier, K. L., & Burdick, M. (1999). "Does automation bias decision-making?" *International Journal of Human-Computer Studies*, 51(5), 991–1006. https://doi.org/10.1006/ijhc.1999.0252
- United Nations General Assembly (2024). Resolution 79/62, *Lethal autonomous weapons systems*, adopted 2 December 2024 (166–3–15). https://digitallibrary.un.org/record/4071100
- United Nations General Assembly (2025). Resolution 80/57, *Lethal autonomous weapons systems*, adopted 1 December 2025 (164–6–7). https://digitallibrary.un.org/record/4095989
- US Department of State (2023). *Political Declaration on Responsible Military Use of Artificial Intelligence and Autonomy* (launched February 2023).
- Verhagen, R. S., Neerincx, M. A., & Tielman, M. L. (2024). "Meaningful human control and variable autonomy in human-robot teams for firefighting." *Frontiers in Robotics and AI*, 11, 1323980. https://doi.org/10.3389/frobt.2024.1323980
- Wickelgren, W. A. (1977). "Speed-accuracy tradeoff and information processing dynamics." *Acta Psychologica*, 41(1), 67–85.

---

*Reproducibility.* `cd code && python3 experiments.py` regenerates every number and figure (see `code/README.md`). Numbers labelled as reported (37,000; 20 s; 90%; 100/day; vote tallies) come from the cited sources. All other numbers are model outputs saved in `results/summary.json`.
