# The Incentive Budget of Blame: Noisy Failure Attribution, Limited Liability, and the Apportionment of Liability for AI Agent Errors

*Target venue: ACM Conference on Fairness, Accountability, and Transparency (FAccT) — law-and-economics / socio-technical track.*

## Abstract

When an AI agent's mistake causes a loss, at least three parties stand behind it: the model developer, the deployer who built the scaffold and granted permissions, and the user who delegated the task. Proposals for dividing liability among them usually assume that a court, an insurer or a contract can tell whose layer failed. Published benchmarks say otherwise: the best automated method for attributing a multi-agent failure to the responsible agent is right about 53% of the time (Zhang et al., 2025). We ask what such a noisy forensic signal can and cannot do for incentives. We model liability as a budget-balanced rule that maps an attribution signal, described by a confusion matrix C, into shares of the loss. Four results follow. (1) Without rebates, the set of incentive profiles a rule can induce is a polytope fixed by C alone; the sum of the parties' internalised shares is at most B(C) = Σ_a max_j C_aj, which interpolates between Holmström's team impossibility (B = 1) and full internalisation (B = n) and is monotone in Blackwell's order. (2) A "debiased" rule S = C⁻¹ restores first-best incentives and balances ex post, but it requires the blamed party to pay a multiple of the loss and the others to receive rebates; at 53% accuracy with three parties the multiple is 2.54. (3) With exponential care technology the deadweight loss of dilution is τ(1/θ − 1 + ln θ) per party, second-order in attribution error. In our simulations noisy attributed-party-pays liability beats channelling all liability to one party in every one of 300 random parameter draws, and channelling wins only when care has low leverage. (4) When parties can also catch each other's errors, attributing the whole loss to the originator removes every incentive to oversee. In a closed-form two-party model the efficient originator share is one half when the two technologies are equally elastic, attribution accuracy above that share has no value, and under the attributed-party-pays rule welfare can fall as attribution improves. On the 58 hand-crafted Who&When logs, log-only heuristics reach 57–64% accuracy while leaving the orchestrator internalising 0–11% of the failures it causes, so accuracy is the wrong metric for liability purposes; per-origin recall is the right one. All technology parameters are assumed; the results are comparative statics, not calibrated estimates.

## 中文摘要

智能体出错造成损失时，背后至少站着三方：模型开发者、搭建脚手架并授予权限的部署者，以及委托任务的用户。现有的责任分摊方案大多默认法院、保险人或合同能够判断是哪一层出了问题。但公开基准显示，目前最好的自动归因方法识别多智能体失败的责任方，准确率只有约 53%（Zhang 等，2025）。本文研究这样一个带噪声的取证信号对激励能起多大作用。我们把责任规则建模为预算平衡的映射：输入是归因信号（由混淆矩阵 C 描述），输出是各方承担损失的份额。主要结论有四点。（1）不允许返还时，规则能够实现的激励组合构成一个只由 C 决定的多面体，各方内部化份额之和不超过 B(C) = Σ_a max_j C_aj；该上界在 Holmström 团队不可能定理（B = 1）与完全内部化（B = n）之间插值，并随 Blackwell 信息序单调。（2）“去偏”规则 S = C⁻¹ 可以恢复最优激励并在事后保持预算平衡，但要求被归责方支付损失的若干倍、其他方获得返还；在三方、准确率 53% 时，这个倍数是 2.54。（3）在指数型注意技术下，激励稀释造成的无谓损失为每方 τ(1/θ − 1 + ln θ)，是归因误差的二阶小量。在 300 组随机参数的模拟中，带噪声的“被归责方全赔”规则全部优于把责任集中到单一方；只有当注意的杠杆很低时，集中责任才占优。（4）当各方还能拦截彼此的错误时，把全部损失归给错误源头会使监督激励降为零。在一个有闭式解的两方模型中，两种技术弹性相同时源头方的最优份额是一半，高于该份额的归因准确率没有价值，而且在“被归责方全赔”规则下，归因越准福利反而可能越低。在 Who&When 的 58 条手工系统日志上，仅依赖日志的启发式归因准确率为 57–64%，但编排者只内部化了自身所致失败的 0–11%。因此就责任认定而言，应报告的指标是按来源计算的召回率，而不是总体准确率。文中技术参数均为假设值，结论属于比较静态分析，不是校准后的估计。

## 1 Introduction

An AI agent books the wrong flights, deletes a production database, or sends a payment to the wrong counterparty. Someone has lost money. The agent itself cannot pay, and the humans and firms behind it each have a story about why the failure belongs to someone else: the developer points to a misconfigured scaffold, the deployer to a model that ignored its instructions, both to a user who wrote an ambiguous request and did not check the result. Courts have so far handled the simple case, where one firm stands behind the system. In *Moffatt v. Air Canada* (2024 BCCRT 149) the tribunal held the airline liable for its chatbot's misstatement and rejected the argument that the chatbot was a separate entity. The harder case is the one agentic systems make routine, in which several independent parties each supply a layer of the system that acted.

The legal literature on this question is now substantial (Kolt, 2026; Ayres and Balkin, 2024; Herbosch, 2025; Yao, 2026; IMDA, 2026), and the regulatory picture is in motion: the revised EU Product Liability Directive (Directive (EU) 2024/2853) brings software and AI systems within strict product liability, while the proposed AI Liability Directive, which would have eased proof of fault and causation, was dropped from the Commission's 2025 work programme. Most proposals share an assumption that this paper examines: that after the fact one can determine *whose* failure it was, and apportion accordingly. Yao (2026), for example, treats the interaction log as the primary evidence for locating where the trajectory left the authorised task.

How good is that evidence? The one benchmark built for the purpose, Who&When (Zhang et al., 2025), reports that the best automated method names the failure-responsible agent correctly in 53.5% of failure logs and the decisive step in 14.2%. Human forensic work will do better at higher cost, but the direction is clear: attribution among the layers of an agentic system is a noisy signal, and its noise level can be measured.

This paper asks a narrow question that the broad debate leaves open. **Given a forensic attribution signal of known quality, what incentive profiles can any liability rule that conditions on it induce, how much welfare is lost to the noise, and which rule should be used?** We answer with a formal model, numerical checks, and a small analysis of real failure logs. The question is framed in terms of deterrence: liability rules are judged by the care they induce in the parties who can prevent the loss, with the victim fully compensated in every rule we consider.

**Contributions.**

1. *An incentive budget (Theorem 1).* For budget-balanced rules without rebates, the set of achievable incentive profiles depends on the forensic channel only. Its size in the direction that matters, the sum of the shares of own-caused losses that parties internalise, is at most B(C) = Σ_a max_j C_aj, which equals n times the best achievable attribution accuracy under a uniform prior. The bound is tight, monotone under garbling, and equals 1 for an uninformative signal and n for a perfect one.
2. *Debiased apportionment and its price (Theorem 2).* The rule S = C⁻¹ gives every party exactly its own expected harm and balances ex post. It is the liability analogue of backward loss correction for noisy labels. It needs rebates and super-compensatory payments whose size we compute, which yields the attribution accuracy below which the rule exceeds a party's assets.
3. *The cost of dilution versus the cost of channelling (Proposition 3).* A closed-form deadweight loss shows dilution is a second-order cost, while removing a party's incentive altogether is a first-order one. We give the accuracy threshold at which attribution-based liability overtakes single-party channelling and the forensic cost per incident that attribution is worth.
4. *Oversight breaks origin-based liability (Proposition 4).* When parties can catch each other's errors, no budget-balanced rule reaches the first best, with or without rebates and even with perfect attribution. Assigning the whole loss to the originator induces zero monitoring. In a two-party closed form the efficient split and the value of attribution accuracy are explicit.
5. *A measurement implication, checked on real logs (Section 6).* Attribution benchmarks report accuracy. Incentives depend on per-origin recall. On Who&When the two diverge sharply.

Contribution type: a formal model with proved results, validated numerically, with an empirical illustration on public data. The functional forms are stylised and the technology parameters are assumed, so the quantitative results describe the model and are not forecasts.

## 2 Related Work

**Economics of tort law.** Shavell (1980) compares strict liability and negligence for a single injurer. Landes and Posner (1980) extend the analysis to joint tortfeasors and show that negligence-type rules can induce efficient care by all when courts observe care. Shavell (1986) shows that injurers with assets below the harm take too little care. Polinsky and Shavell (1998) derive the principle that damages should be scaled by the inverse of the probability of being held liable. Our Theorem 2 is the multi-party, misattribution version of that principle: scaling by an inverse probability becomes multiplication by an inverse confusion matrix, and the correction can be made budget-balanced.

**Moral hazard in teams.** Holmström (1982) shows that no budget-balanced rule for sharing a joint outcome induces efficient effort from every team member when only the outcome is observed. Legros and Matthews (1993) characterise when partnerships can nonetheless reach or approach efficiency. A liability rule among the parties behind an agent is a sharing rule for a negative joint output, and the forensic signal is a monitoring technology. Theorem 1 quantifies how far a given signal relaxes Holmström's constraint under limited liability, and Proposition 4 recovers his impossibility inside each incident when the loss has several but-for contributors.

**Liability for AI and AI agents.** Buiten, de Streel and Peitz (2023) apply the standard framework to AI and evaluate the EU proposals. Weil (2024) argues for strict liability and punitive damages for advanced AI developers. Ayres and Balkin (2024) argue that AI should be treated as a risky agent without intentions, with objective standards applied to the humans behind it. Kolt (2026) analyses AI agents through agency law and identifies liability, visibility and inclusivity as governing principles. Herbosch (2025) gives a law-and-economics account specific to agents and concludes that targeted adjustments to tort law suffice. The IMDA (2026) discussion paper sets out the allocation problem along the value chain of developers, deployers and end-users. De Chiara, Manna and Singh (2026) is the closest formal work: a two-party model of a developer and a human expert who can supervise a generative model, in which operator liability gives first-best supervision incentives and developer liability distorts them. We differ in taking the *identity of the failing layer* as uncertain and asking what a noisy signal about it is worth, with any number of parties.

**Responsibility and failure attribution in computer science.** Chockler and Halpern (2004) define degrees of responsibility and blame in structural causal models. Triantafyllou, Singla and Radanovic (2021) study blame attribution in multi-agent sequential decision making and show that standard methods such as the Shapley value fail incentive properties. Chan et al. (2024) propose agent identifiers, real-time monitoring and activity logs as infrastructure for accountability. Zhang et al. (2025) introduce the Who&When benchmark for automated failure attribution, and Cemri et al. (2025) give a taxonomy of fourteen multi-agent failure modes grouped into system design, inter-agent misalignment and task verification. These works measure or define attribution. We ask what a given quality of attribution buys in incentives.

**Learning with noisy labels.** Natarajan et al. (2013) construct unbiased loss estimators under class-conditional label noise, and Patrini et al. (2017) express the correction as multiplication by the inverse noise transition matrix. Theorem 2 is the same algebra applied to money: the "label" is the blamed party and the "loss" is the damages award.

**Gap.** We found no prior work that characterises the set of incentive profiles implementable by liability rules conditioned on a misclassifying attribution signal, or that connects the reported accuracy of agent failure attribution to welfare. That is the gap addressed here.

## 3 Model

### 3.1 Parties, care and harm

There are n risk-neutral parties, indexed i = 1..n; the running example is n = 3 (developer, deployer, user). A task is executed once. Party i chooses care x_i ≥ 0 at unit cost: evaluation and training effort for a developer, guardrails and permission scoping for a deployer, careful specification for a user. An incident that *originates* in layer i occurs with probability h_i(x_i), decreasing and convex. Incidents are rare, so the probability of a loss is P = Σ_j h_j(x_j) and each loss has one origin. An incident causes loss L to a victim, who may be the user or a third party. Section 5 adds the possibility that other parties catch an error before it causes loss.

Care is not observed by courts. This rules out negligence rules based on a standard of care, which is the informational premise of the paper: the adjudicator sees the loss and whatever forensic analysis of the logs reveals about its origin, and nothing else. Where care is cheaply verifiable, the classical results of Landes and Posner (1980) apply and attribution matters less.

The first best minimises Σ_i x_i + L Σ_j h_j(x_j), so each party should equate −h_i′(x_i) L to 1.

### 3.2 The forensic channel

After a loss, a forensic process emits a signal a ∈ {1..n} naming a layer. Its quality is the confusion matrix

C[a, j] = Pr(signal = a | origin = j),

whose columns sum to one. The symmetric channel with accuracy q has C[j, j] = q and C[a, j] = r = (1 − q)/(n − 1) otherwise. We treat C as known and not manipulable; Section 8 returns to this.

### 3.3 Liability rules

A liability rule is a matrix S, where S[i, a] is the share of L paid by party i when the signal is a. *Budget balance* (the victim receives exactly L and nothing flows to or from outsiders) means each column of S sums to one. *No rebates* means S ≥ 0, so no party profits from an incident. Tort awards satisfy both. Contracts among the parties (indemnities, warranties) can relax the second but not the first, and they can condition only on the same verifiable signal. The analysis therefore covers any arrangement among the parties, legal or contractual, and the Coasean objection that parties will contract around the default rule changes which S is chosen but not the set of S available.

Familiar regimes are points in this space. *Channelling* to party i sets S[i, a] = 1 for all a: this is respondeat-superior-style deployer liability, developer-only product liability, or terms of service that put all responsibility on the user. *Attributed-party-pays* sets S = I. *Equal shares* sets S[i, a] = 1/n.

Define M = S C. Then M[i, j] is the expected share that party i bears of a loss originating at j, and party i's expected cost is

x_i + L Σ_j M[i, j] h_j(x_j).

Only the diagonal term depends on x_i, so the privately optimal care solves −h_i′(x_i) L θ_i = 1, where

θ_i = M[i, i] = Σ_a S[i, a] C[a, i]

is party i's *internalisation rate*. The first best needs θ_i = 1 for every party.

## 4 Results for the Origin Model

### 4.1 The incentive budget

**Theorem 1.** Let Θ(C) be the set of internalisation profiles θ achievable by budget-balanced rules without rebates.

(a) Θ(C) is a convex polytope, and for any weights w ≥ 0, max over Θ(C) of Σ_i w_i θ_i equals Σ_a max_i w_i C[a, i].

(b) In particular Σ_i θ_i ≤ B(C) := Σ_a max_i C[a, i], with equality attained. 1 ≤ B(C) ≤ n; B(C) = 1 exactly when the signal is uninformative (all columns equal), and B(C) = n exactly when it identifies the origin with certainty.

(c) If C′ = G C for a column-stochastic G (C′ is a garbling of C), then Θ(C′) ⊆ Θ(C).

(d) B(C)/n is the highest accuracy with which any decision rule can identify the origin from the signal when origins are equally likely a priori.

*Proof.* (a) θ is linear in S, and the feasible S form a product of n simplices, one per signal, so Θ(C) is a convex polytope. Σ_i w_i θ_i = Σ_a Σ_i S[i, a] w_i C[a, i], and for each a the inner sum is a convex combination of the numbers w_i C[a, i], maximised by putting the whole share on the largest. (b) Set w = 1. Since max_i C[a, i] ≥ C[a, 1] and Σ_a C[a, 1] = 1, B ≥ 1, with equality only if C[a, i] ≤ C[a, 1] for all a and i, which together with equal column sums forces equal columns. B ≤ Σ_a Σ_i C[a, i] = n, with equality only if each row has a single non-zero entry. (c) Given S′ for C′, the rule S = S′ G is non-negative, has unit column sums, and S C = S′ G C = S′ C′. (d) Under a uniform prior the Bayes rule picks argmax_j C[a, j] and is correct with probability (1/n) Σ_a max_j C[a, j]. ∎

The theorem says that a forensic channel comes with a fixed budget of incentive. An uninformative signal gives a total of one unit of internalisation to be divided among n parties, which is Holmström's (1982) result. A perfect signal gives n units, enough for everyone. In between, each point of lost attribution accuracy removes a point of incentive that no reallocation of shares can recover. Part (c) says that degrading the evidence, for instance by logging less, can only shrink what is implementable.

**Corollary 1.** If the signal is imperfect, no budget-balanced rule without rebates induces first-best care by all parties.

For the symmetric channel, B = nq when q ≥ 1/n, and the attributed-party-pays rule gives θ_i = q for all i.

*Numerical check (code run here; `results/exp1_theory_checks.json`).* On 2,000 random channels with n between 2 and 6, a linear program maximising Σθ_i returned B(C) to within 6.7 × 10⁻¹⁶, and no random garbling increased any weighted LP value (0 violations in 2,000).

### 4.2 Debiased apportionment

**Theorem 2.** Suppose C is invertible and rebates are allowed. The rule S = C⁻¹ (i) satisfies ex-post budget balance, (ii) gives M = I, so that each party's expected liability equals exactly L h_i(x_i) and care is first-best, and (iii) is the only rule with M = I. (iv) It has a negative entry unless C is a permutation matrix. (v) For the symmetric channel the blamed party pays L(1 − r)/(q − r) and every other party receives L r/(q − r). A party with assets A can meet the payment only if

q ≥ (n − 2 + κ)/(κ n − 1), where κ = A/L.

*Proof.* (i) 1ᵀC = 1ᵀ implies 1ᵀ = 1ᵀC⁻¹. (ii) and (iii) are immediate from M = S C. (iv) A non-negative matrix with a non-negative inverse is a monomial matrix, and unit column sums make it a permutation. (v) C = (q − r) I + r 11ᵀ, and since (q − r) + n r = 1 the Sherman–Morrison formula gives C⁻¹ = (I − r 11ᵀ)/(q − r). The asset condition rearranges (n − 2 + q)/(n q − 1) ≤ κ. ∎

The rule charges the blamed party more than the loss and distributes the excess to the others, so that being wrongly blamed some of the time and wrongly rewarded the rest of the time cancel in expectation. It is the Polinsky–Shavell multiplier with the surplus routed to co-defendants so that the books balance, and it is algebraically the backward loss correction of Patrini et al. (2017).

*Numerical check (`exp1_theory_checks.json`).* We simulated 4,000,000 tasks with assumed unequal hazards (0.02, 0.01, 0.005), n = 3, L = 100 and a symmetric channel at q = 0.5344, producing 139,381 incidents. Under attributed-party-pays, mean payments per task were (1.42, 1.10, 0.96) against own-origin losses of (2.02, 0.98, 0.49): the rule pulls every party toward the average. Under the debiased rule they were (2.01, 0.97, 0.50). Payments summed to L in every incident under both rules. The cost is exposure: the blamed party pays 2.54 L and the others receive 0.77 L each, and the per-incident standard deviation of a party's payment rises from 49.1 to 162.9.

The multiples grow quickly as accuracy falls (`exp4_whowhen.json`, computed from accuracies published by Zhang et al., 2025, under an assumed symmetric channel). At the 53.44% of their best method on hand-crafted systems, the multiple is 2.54 with three parties and 7.77 with two. At the 36.21% of their binary-search method it is 15.8 with three parties. Debiasing is therefore a tool for well-capitalised parties in a contract (an indemnity schedule with a multiplier and a rebate), and Shavell's (1986) judgment-proof problem bounds its use against individuals.

### 4.3 How much does dilution cost?

To price the shortfall we need a care technology. Let h_i(x) = b_i e^(−x/τ_i). Here b_i is the incident probability with no care and τ_i measures how much care it takes to cut the hazard by a factor e; ρ_i = L b_i/τ_i is the *leverage* of care, the expected loss with no care relative to the cost scale of care.

**Proposition 3.** (a) A party with internalisation rate θ chooses x = τ ln(θρ) if θρ > 1 and x = 0 otherwise. Its contribution to social cost exceeds the first best by

D(θ) = τ (1/θ − 1 + ln θ) if θρ > 1, and D(θ) = τ (ρ − 1 − ln ρ) if θρ ≤ 1.

(b) D(θ) = τ (1 − θ)²/2 + O((1 − θ)³) near θ = 1. For any smooth strictly convex technology the loss is likewise second-order in 1 − θ.

(c) With n identical parties and a symmetric channel, attributed-party-pays has lower deadweight loss than channelling to one party if and only if n (1/q − 1 + ln q) < (n − 1)(ρ − 1 − ln ρ), for qρ > 1.

*Proof.* (a) The party minimises x + θ L b e^(−x/τ), giving e^(−x/τ) = 1/(θρ) and h = τ/(θL). Social cost is τ ln(θρ) + τ/θ, against τ ln ρ + τ at θ = 1. At a corner the cost is L b = τρ. (b) Differentiate twice at θ = 1; the general statement is the envelope theorem, since θ = 1 minimises social cost in x. (c) Under attributed-party-pays θ_i = q for all i. Under channelling one party has θ = 1 and n − 1 have θ = 0. ∎

Dilution is cheap and exclusion is expensive. A party who internalises only half of the harm it causes still takes most of the efficient care, because the last units of care are the least valuable. A party who internalises none takes none.

*Numbers (`results/exp2_regimes.json`; all parameters assumed: n = 3, L = 100, b_i = 0.1).* With τ_i = 1 (ρ = 10) the first-best social cost is 9.91. Figure 1 (`results/fig1_regimes.png`) and Table 1 give deadweight loss as a percentage of it.

**Table 1. Deadweight loss as % of first-best cost, three symmetric parties, ρ = 10 (computed).**

| Attribution accuracy q | Channel to one party | Equal shares | Attributed party pays | Optimal no-rebate rule | Debiased rule: payment / L |
|---|---|---|---|---|---|
| 0.34 | 135.2 | 27.3 | 26.1 | 26.1 | 67.0 |
| 0.40 | 135.2 | 27.3 | 17.7 | 17.7 | 7.0 |
| 0.5344 | 135.2 | 27.3 | 7.4 | 7.4 | 2.54 |
| 0.70 | 135.2 | 27.3 | 2.2 | 2.2 | 1.55 |
| 0.80 | 135.2 | 27.3 | 0.8 | 0.8 | 1.29 |
| 0.90 | 135.2 | 27.3 | 0.2 | 0.2 | 1.12 |

At the accuracy of today's best published attribution method, attributed-party-pays loses 7.4% of first-best cost. Raising accuracy to 0.80 removes about nine-tenths of that loss. Channelling loses 135% whatever the accuracy, because two of three parties take no care. Equal shares, which ignores the signal, loses 27%.

The threshold in Proposition 3(c) lies below chance accuracy unless leverage is low. For n = 3 it is q* = 0.561 at ρ = 2 and 0.252 at ρ = 5, the latter below 1/3. For n = 2 and n = 5 at ρ = 2 it is 0.602 and 0.534. A grid search agrees to three decimals wherever q* exceeds 1/n. So channelling is the better simple rule only when the care available to the excluded parties has little leverage, which is the cheapest-cost-avoider logic that Herbosch (2025) invokes, with a number attached: the other parties' care must be worth less than about twice its cost scale.

*Heterogeneous parties.* With τ = (3, 1, 0.3), a developer whose care is expensive to scale and a user whose care is cheap and quickly exhausted, the optimal rule departs from attributed-party-pays when accuracy is low. At q = 0.5344 it charges the developer 31% of the loss when the signal blames the user, giving θ = (0.61, 0.53, 0.37) and reducing deadweight loss from 9.3% to 8.0% of first-best cost. At q ≥ 0.85 the optimal rule is attributed-party-pays again. The incentive budget is tilted toward the party with the largest τ, for whom a given dilution is most costly.

*Sensitivity.* Across 300 random draws (ρ_i log-uniform on [2, 100], τ_i log-uniform on [0.3, 3], random diagonally dominant channels with mean diagonal between 0.40 and 0.95), attributed-party-pays beat the best channelling rule in every draw. The optimal no-rebate rule was attributed-party-pays itself in 61.7% of draws and never a pure channelling rule. Median deadweight loss was 0.94% of first-best cost for the optimal rule, 0.96% for attributed-party-pays and 104.6% for the best channelling rule. The correlation between B(C) and the optimal rule's deadweight loss was −0.90.

*What is attribution worth?* Channelling needs no forensic work. Attribution-based liability breaks even with channelling when the forensic cost per incident equals the difference in deadweight loss divided by the incident rate. With three symmetric parties at q = 0.5344 this is 2.26 L at ρ = 10, 0.72 L at ρ = 5 and 33.5 L at ρ = 100. At ρ = 2 it is negative (−0.02 L): channelling is better even if attribution is free, and attribution is worth 0.14 L per incident only once q reaches 0.80. Outside the low-leverage case, the model says it is worth spending more than the loss itself on working out who caused it. The reason is deterrence, and has nothing to do with fairness to the parties.

*A judgment-proof user.* If the user can pay at most a fraction of L, the user's internalisation rate cannot exceed that fraction. With the symmetric baseline and q = 0.5344, the optimal rule's deadweight loss rises from 0.73 (in cost units; 7.4%) with no cap to 1.43 at a cap of 0.5 L, 2.93 at 0.2 L and 6.91 at 0.05 L. At the smallest cap the user falls below the corner θρ = 1 and takes no care, and the optimal rule gives up on the user and uses the freed budget on the other two parties (θ = 0.65 each, above q). Even with perfect attribution the loss is 6.70 at that cap. For consumer users facing losses far above their means, the model supports placing liability on the developer and deployer and accepting the lost user care as a cost.

*Validity of the rare-event approximation.* In an exact version of the model, in which layers fail independently and the origin is the first failing layer, Nash care levels differ from the closed form by at most 3.8% and social cost by at most 0.39% across q ∈ {0.4, 0.5344, 0.7, 0.9, 1.0} (`exp1_theory_checks.json`).

## 5 Layered Oversight: When the Originator Should Not Pay Everything

Agentic systems are built in layers so that one layer catches another's mistakes: a deployer's guardrail blocks a model's bad tool call, and a user reviews a plan before approving it. Suppose each party i also chooses monitoring effort y_i ≥ 0, and party k's monitoring fails to stop an error that originated elsewhere with probability E_k(y_k), decreasing and convex with E_k(0) = 1. The probability of a loss originating at j is h_j(x_j) Π_{k≠j} E_k(y_k). The signal still reports on the origin.

**Proposition 4.** (a) Party i's incentive to monitor is proportional to its off-diagonal shares M[i, j], j ≠ i. If these are zero, as under attributed-party-pays with perfect attribution, then y_i = 0 in every equilibrium.

(b) The first best requires M[i, j] = 1 for every pair (i, j) such that i has a productive action against losses originating at j. Every budget-balanced rule, with or without rebates and for any channel, has Σ_i M[i, j] = 1 for each j. So if any two parties can each reduce the same kind of loss, no budget-balanced rule based on the origin signal reaches the first best, even when attribution is perfect.

(c) Let n = 2 with identical parties, h(x) = b(1 + x/τ)^(−β) and E(y) = (1 + y/σ)^(−γ), and let each party bear share m of losses it originates and 1 − m of the other's. In an interior symmetric equilibrium

x = mβΛ − τ, y = (1 − m)γΛ − σ, Λ^(1+β+γ) = K m^(−β) (1 − m)^(−γ),

with K = L b τ^β σ^γ β^(−β) γ^(−γ), and social cost per party is (1 + mβ + (1 − m)γ)Λ − τ − σ. If β = γ this is minimised at m = 1/2 for all τ and σ.

(d) Under a symmetric two-party channel of accuracy q the achievable m lie in [1 − q, q]. If m* is the efficient share, attribution accuracy above max(m*, 1 − m*) has no value under the optimal rule.

*Proof.* (a) Party i's cost is x_i + y_i + L Σ_j M[i, j] h_j Π_{k≠j} E_k, and y_i enters only the terms with j ≠ i. (b) The social first-order conditions are the private ones with every relevant M[i, j] replaced by 1. Columns of M = S C sum to 1ᵀS C = 1ᵀC = 1ᵀ. (c) The first-order conditions are mβΛ = τ + x and (1 − m)γΛ = σ + y, where Λ = L h E is the expected loss per origin. Substituting into Λ gives the expression. With β = γ the factor 1 + mβ + (1 − m)γ is constant and Λ is minimised where m(1 − m) is largest. (d) m = αq + (1 − α)(1 − q) for a rule that puts share α on the blamed party, and α ranges over [0, 1]. ∎

Part (b) is Holmström's theorem inside a single incident. The forensic signal answers "where did the error start?", but with a layered defence a loss has several but-for contributors, the originator and each layer that failed to stop it, and one unit of loss cannot be charged in full to each. This connects to the structural account of Chockler and Halpern (2004), in which each of k jointly necessary contributors bears a fractional degree of responsibility. The economic counterpart is that the fractions must sum to one and each party's incentive is scaled down accordingly.

*Numbers (`results/exp3_oversight.json`, `results/exp3b_optimal_share.json`; parameters assumed: b = 0.1, τ = σ = 1; equilibria by damped best-response iteration, which does not use the closed form).* Table 2 reports deadweight loss as a function of the originator's effective share m. Where the equilibrium is interior, the closed form matches the numerical equilibrium cost to within 3 × 10⁻⁶ in relative terms. The efficient share m* is located on a grid of step 0.025 (plus 0.5344), so it is accurate to that resolution.

**Table 2. Layered oversight, two symmetric parties: deadweight loss as % of first-best cost (computed).**

| L·b | β, γ | efficient share m* | at m* | m = 0.5344 | m = 0.80 | m = 1 (originator pays all) |
|---|---|---|---|---|---|---|
| 100 | 1, 1 | 0.50 | 6.8 | 7.0 | 26.6 | 59.3 |
| 100 | 1, 0.5 | 0.55 | 6.5 | 6.6 | 16.5 | 17.9 |
| 100 | 1, 0.25 | 1.00 | 3.7 | 6.1 | 4.4 | 3.7 |
| 100 | 1, 0.1 | 1.00 | 0.0 | 5.2 | 0.7 | 0.0 |
| 100 | 0.5, 1 | 0.45 | 6.5 | 7.5 | 30.8 | 146.4 |
| 100 | 1, 2 | 0.475 | 6.4 | 7.3 | 38.7 | 173.6 |
| 100 | 2, 1 | 0.53 | 6.3 | 6.3 | 12.6 | 11.9 |
| 10 | 1, 1 | 0.50 | 8.4 | 8.7 | 20.2 | 19.3 |
| 10 | 1, 0.5 | 1.00 | 2.1 | 8.1 | 2.9 | 2.1 |

Three things stand out. First, when monitoring is about as productive as origination care (the five rows with γ ≥ β/2 at the higher stake), the efficient originator share is close to one half, and charging the originator everything costs between 12% and 174% of first-best cost against 6–7% at the efficient share. Second, under attributed-party-pays the effective share m equals the attribution accuracy q, so the columns of Table 2 can be read as accuracies: with β = γ = 1 and L b = 100, deadweight loss is 7.0% at q = 0.5344, 26.6% at q = 0.80 and 59.3% at q = 1. Under that rule, better attribution makes outcomes worse, because noise was acting as an accidental sharing rule that kept overseers on the hook. Third, the result reverses when monitoring is unproductive (γ ≤ 0.25 here, or γ = 0.5 at the lower stake): there the originator should pay everything, the first best has little or no monitoring, and the origin model of Section 4 is the right guide. Which regime an application is in is an empirical question about how much errors are caught downstream.

The same patterns appear with asymmetric parties. In a configuration where one party originates most errors and the other is the natural overseer (b = (0.15, 0.02), σ = (3, 0.5), L = 100), the best originator share under perfect attribution is 0.55 and gives a deadweight loss of 7.9%, against 38.3% when the originator pays all. With three symmetric parties, the originator-pays-all rule loses 75.7% under perfect attribution and 35.2% at q = 0.5344, and the best rule in the family is close to equal shares (22.7%) (`exp3_oversight.json`; Figure 2, `results/fig2_oversight.png`; Figure 3, `results/fig3_share.png`).

The practical reading is that a liability rule for layered systems needs two components: a share that follows the origin and a share borne by parties who had a reasonable opportunity to catch the error. This is close in structure to comparative fault and to contribution among joint tortfeasors. De Chiara, Manna and Singh (2026) reach a related conclusion from a different model, that developer-only liability undermines supervision by the operator.

## 6 What Real Attribution Channels Deliver

Theorem 1 makes the confusion matrix, not the accuracy, the object of interest. To see whether the difference matters in practice we computed confusion matrices on the Who&When data (Zhang et al., 2025), which contain 184 annotated failure logs: 58 from a hand-crafted system with fixed roles (Orchestrator, WebSurfer, FileSurfer, Assistant) and 126 from algorithm-generated systems whose agents differ from task to task. We ran no language model. We applied attribution heuristics that read only who spoke when, and compared them with the published accuracies of the LLM-based methods. In the hand-crafted logs the annotated responsible agent is the WebSurfer in 33 cases, the Orchestrator in 18, the Assistant in 4 and the FileSurfer in 3 (`results/exp4_whowhen.json`).

**Table 3. Log-only heuristics on the 58 hand-crafted Who&When failure logs (computed; 95% bootstrap intervals from 2,000 resamples).**

| Heuristic | Accuracy | B(C)/n | Internalisation under attributed-party-pays: Orchestrator, WebSurfer, FileSurfer, Assistant |
|---|---|---|---|
| Always blame the Orchestrator | 0.31 [0.21, 0.43] | 0.25 | 1.00, 0.00, 0.00, 0.00 |
| Always blame the WebSurfer | 0.57 [0.45, 0.71] | 0.25 | 0.00, 1.00, 0.00, 0.00 |
| Blame the first worker to act | 0.60 [0.47, 0.72] | 0.35 [0.25, 0.52] | 0.06, 1.00, 0.33, 0.00 |
| Blame the most active worker | 0.60 [0.47, 0.72] | 0.35 [0.25, 0.52] | 0.06, 1.00, 0.33, 0.00 |
| Blame the majority worker in the second half | 0.64 [0.52, 0.76] | 0.44 [0.28, 0.57] | 0.11, 1.00, 0.67, 0.00 |
| Blame the last worker to act | 0.64 [0.52, 0.76] | 0.60 [0.48, 0.76] | 0.06, 0.94, 0.67, 0.75 |

Two observations follow. First, trivial heuristics reach 57–64% accuracy on these logs, the same range as or above the 53.44–55.17% that Zhang et al. report for their best LLM-based method on the same subset. The comparison is loose, since the samples are small and the intervals wide, but it shows that accuracy on this benchmark is dominated by the base rate of the most frequently responsible agent. Second, accuracy and incentive budget come apart. "Always blame the WebSurfer" is 57% accurate and has the budget of an uninformative signal (B/n = 0.25, the minimum for four parties). The two heuristics with 64% accuracy have budgets of 0.44 and 0.60. Under every heuristic except the constant one that always blames it, the Orchestrator internalises 0–11% of the failures it causes, although it is responsible for 31% of them. If the roles were supplied by different firms, the firm supplying the orchestrator would face almost no liability-driven incentive under any of these attribution methods, and Proposition 3 says that is where the large welfare losses are.

Under attributed-party-pays the sum of internalisation rates is the sum of per-origin recalls, n times the balanced accuracy, and B(C) is its upper envelope over rules. **We therefore recommend that failure-attribution benchmarks report the per-origin confusion matrix and B(C)/n alongside accuracy.** A method tuned for accuracy on an imbalanced benchmark can be useless, or worse, as the basis for apportioning liability.

On the algorithm-generated logs, where roles vary and only accuracy can be computed, blaming the most active agent is correct in 54.0% of 126 cases, the first speaker in 49.2% and the last speaker in 44.4%, against 29.1% for uniform guessing (matching the random baseline reported by Zhang et al.) and 51.12–54.33% for their best method.

Translating the published accuracies into the model under an assumed symmetric channel: at q = 0.5344 each party internalises 53% of the harm it causes under attributed-party-pays, the dilution loss factor 1/q − 1 + ln q is 0.245 in units of τ per party, and debiasing requires payments of 2.54 L (three parties) or 2.23 L (four). At the 32.75% of the step-by-step method the loss factor is 0.94 and debiasing with four parties requires 7.5 L.

## 7 Discussion

**An answer to the question.** In this model, "how should liability be determined when an agent's mistake causes loss?" has a conditional answer.

1. *Decide first whether the loss was of a kind that other layers could have caught.* If not (the origin model), liability should follow the best available attribution of origin, even when that attribution is poor. Attribution accurate little more than half the time already recovers most of the efficient care, and single-party channelling is justified only where the other parties' care has little leverage or where they cannot pay.
2. *If other layers could have caught it,* part of the loss should be borne by those layers regardless of where the error began, and the origin share should be well below one. In the symmetric benchmark it is one half.
3. *Among well-capitalised firms, contracts can do better than the tort default* by using debiased indemnities: multipliers above one for the party blamed and rebates to the others, calibrated to the measured confusion matrix of the agreed forensic procedure.
4. *Where the user is an individual and the loss is large relative to the user's means,* the user's share must be small for compensation to be real, and the lost user care is a cost to be recovered through design (confirmation prompts, spending limits) and not through liability.

This is compatible with the direction of Directive (EU) 2024/2853, which channels defect liability to producers, for losses to consumers. It is in tension with pure channelling in business-to-business deployments, where all parties can pay and all can take care.

**Falsifiable predictions and a test protocol.** The model yields predictions that can be tested without waiting for case law.

- P1. Holding the rule at attributed-party-pays, care by each party rises with attribution accuracy when errors cannot be caught downstream, and total welfare loss falls roughly with the square of the error rate.
- P2. When downstream catching is possible and productive, monitoring effort falls as attribution accuracy rises under attributed-party-pays, and total losses can rise.
- P3. A debiased payment schedule equalises expected payments with own-caused harm and raises care by the least-blamed party relative to attributed-party-pays.
- P4. Attribution methods with equal accuracy but different per-origin recall induce different care, with the largest shortfall at the party with the lowest recall.

A laboratory protocol: three-person groups (or three LLM-based agents, each with its own budget) repeatedly choose costly care and monitoring levels from a menu with the hazard functions of Sections 3 and 5 implemented by the experimenter. Losses are assigned by a rule S applied to a signal drawn from a known C. The design crosses accuracy q ∈ {0.4, 0.55, 0.8, 1.0} with rule ∈ {channelling, attributed-party-pays, debiased, half-origin and half-shared} and with catching ∈ {off, on}, and the pre-registered outcomes are mean care, mean monitoring and realised total cost against the equilibrium values computed by the code released with this paper. We have not run this experiment.

**Logging as a public good.** Theorem 1(c) implies that every party weakly prefers, behind a veil of ignorance about who will be blamed, a more informative forensic channel. After the fact, a party that expects to be the origin prefers a noisier one. This supports mandatory activity logging and agent identifiers of the kind proposed by Chan et al. (2024) on incentive grounds, independent of their use for monitoring.

## 8 Limitations

- **Stylised technology, assumed parameters.** The exponential and power-law forms were chosen for closed forms. L, b, τ, σ, β and γ are illustrative, and no number in Sections 4 and 5 is an estimate of a real deployment. The second-order property of dilution losses is general. The magnitudes, the thresholds and the value m* = 1/2 are specific to the functional forms. Table 2 shows that conclusions about oversight flip with the relative productivity of monitoring, which we do not measure.
- **Care is unobservable by assumption.** Where a court can verify a standard of care (documented evaluations, required guardrails), negligence rules can do better than anything studied here. Our results apply to the residual in which only outcomes and logs are available.
- **Risk neutrality and no insurance.** The debiased rule raises payment variance threefold in our example. With risk-averse parties or costly capital this is a real cost that we do not price. Liability insurance would change both the asset constraints and the incentives.
- **No activity level, entry or innovation effects.** Liability also affects whether and how much agents are deployed and who enters the market. We hold activity fixed. We also do not model litigation costs beyond the break-even calculation, or the costs of delay.
- **The forensic channel is exogenous and honest.** In practice the deployer controls the logs and parties can shape the evidence. Strategic manipulation of C is outside the model, and it is likely to matter.
- **One origin per incident, known n.** Real failures can have several originating errors, and the boundaries between "model", "scaffold" and "instruction" errors are contested. Step-level attribution, where published accuracy is far lower (14.2%), is not modelled.
- **The empirical illustration is small and indirect.** Who&When agents are components of one system, not separately owned legal parties. The hand-crafted subset has 58 logs, with only 4 and 3 failures for two of the roles, so the per-role recalls for those roles are very imprecise. Its labels are human annotations of a judgement that is itself uncertain. We ran no LLM-based attribution method. For those we have only published accuracies, and translating them through a symmetric channel is an assumption that Table 3 suggests is optimistic about balance.
- **Equilibrium computation.** Equilibria in Section 5 were found by damped best-response iteration from fixed starting points with small residuals (below 10⁻⁸). We did not prove uniqueness, and other equilibria may exist. The optimal no-rebate rule is found by multi-start local search on a non-convex objective and is not certified globally optimal.
- **Compensation and fairness are held fixed.** Every rule we compare compensates the victim in full, and we rank rules by deterrence only. Corrective-justice views on which a party should pay only for what it actually caused would reject the debiased rule's rebates and its over-payment by the blamed party, whatever its incentive properties.

## 9 Conclusion

Liability for agent errors depends on an evidentiary question that can be measured: how well can the failing layer be identified after the fact? We showed that the quality of that identification sets a hard budget on the incentives any budget-balanced liability rule can provide, that the budget can be bought back with multipliers and rebates only by parties able to bear the exposure, and that in the simplest setting even poor attribution is far better for deterrence than assigning liability to a single party. We also showed where that conclusion stops. Once layers are expected to catch each other's errors, tracing the loss to its origin and charging the originator in full is the wrong rule, and more accurate tracing makes it worse. The contribution is a framework and a set of comparative statics. Turning it into quantitative guidance needs two measurements that do not yet exist: per-origin confusion matrices for forensic attribution across separately owned layers, and estimates of how much downstream oversight actually catches.

## References

Ayres, I., and Balkin, J. M. (2024). The Law of AI is the Law of Risky Agents without Intentions. *University of Chicago Law Review Online*, November 2024.

Blackwell, D. (1953). Equivalent Comparisons of Experiments. *Annals of Mathematical Statistics*, 24(2), 265–272.

Buiten, M., de Streel, A., and Peitz, M. (2023). The Law and Economics of AI Liability. *Computer Law & Security Review* (April 2023 issue).

Cemri, M., et al. (2025). Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657; presented at NeurIPS 2025.

Chan, A., Ezell, C., Kaufmann, M., Wei, K., Hammond, L., Bradley, H., Bluemke, E., Rajkumar, N., Krueger, D., Kolt, N., Heim, L., and Anderljung, M. (2024). Visibility into AI Agents. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)*. doi:10.1145/3630106.3658948.

Chockler, H., and Halpern, J. Y. (2004). Responsibility and Blame: A Structural-Model Approach. *Journal of Artificial Intelligence Research*, 22.

De Chiara, A., Manna, E., and Singh, S. (2026). Mitigating Generative AI Hallucinations. Working paper (earlier version: UB Economics Working Paper E25/492, University of Barcelona, 2025).

Directive (EU) 2024/2853 of the European Parliament and of the Council of 23 October 2024 on liability for defective products and repealing Council Directive 85/374/EEC. *Official Journal of the European Union*.

European Commission (2025). Commission Work Programme 2025, adopted 11 February 2025, listing the proposal for an AI Liability Directive for withdrawal (as reported by IAPP and Euronews, 2025).

Herbosch, M. (2025). Liability for AI Agents. *North Carolina Journal of Law & Technology*, 26(3), 391.

Holmström, B. (1982). Moral Hazard in Teams. *Bell Journal of Economics*, 13(2), 324–340.

Infocomm Media Development Authority of Singapore (IMDA) (2026). *Legal Responsibility for AI Agents*. Discussion paper, May 2026.

Kolt, N. (2026). Governing AI Agents. *Notre Dame Law Review*, 101, 335. arXiv:2501.07913.

Landes, W. M., and Posner, R. A. (1980). Joint and Multiple Tortfeasors: An Economic Analysis. *Journal of Legal Studies*, 9(3), 517.

Legros, P., and Matthews, S. A. (1993). Efficient and Nearly-Efficient Partnerships. *Review of Economic Studies*, 60(3), 599–611.

*Moffatt v. Air Canada*, 2024 BCCRT 149 (British Columbia Civil Resolution Tribunal, 14 February 2024).

Natarajan, N., Dhillon, I. S., Ravikumar, P., and Tewari, A. (2013). Learning with Noisy Labels. *Advances in Neural Information Processing Systems 26*, 1196–1204.

Patrini, G., Rozza, A., Menon, A. K., Nock, R., and Qu, L. (2017). Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1944–1952.

Polinsky, A. M., and Shavell, S. (1998). Punitive Damages: An Economic Analysis. *Harvard Law Review*, 111(4), 869–962.

Shavell, S. (1980). Strict Liability versus Negligence. *Journal of Legal Studies*, 9(1), 1–25.

Shavell, S. (1986). The Judgment Proof Problem. *International Review of Law and Economics*, 6(1), 45–58.

Triantafyllou, S., Singla, A., and Radanovic, G. (2021). On Blame Attribution for Accountable Multi-Agent Sequential Decision Making. *Advances in Neural Information Processing Systems 34*. arXiv:2107.11927.

Weil, G. (2024). Tort Law as a Tool for Mitigating Catastrophic Risk from Artificial Intelligence. SSRN working paper, January 2024.

Yao, Y. (2026). Acting with AI: An Interaction-Based Framework for Agentic Tort Liability. arXiv:2606.00518.

Zhang, S., et al. (2025). Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267. arXiv:2505.00212.
