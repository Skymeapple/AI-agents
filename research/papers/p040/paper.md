# The Public Writes a Shortlist: Certification Limits and Authorship in Public-Input AI Constitutions

**Target venue:** ACM Conference on Fairness, Accountability, and Transparency (FAccT 2027), full paper track.

---

## Abstract

Public-input processes such as Collective Constitutional AI (CCAI) are presented as letting "the public," rather than developers, write the principles that steer a language model. A constitution, however, can only be attributed to the public to the extent that the public's votes actually *determine* it. We make this testable. We treat each seat of a k-statement constitution as a ranking claim, "statement s is among the top k," and use bootstrap confidence sets for ranks (Mogstad et al., 2024) to ask which seats the votes certify, which they rule out, and which they leave open.

On the public CCAI data (1,002 participants, 275 statements, 34,563 votes), the votes certify **none** of the 58 seats at the 95% level, whether by simultaneous or per-statement rank intervals, and under every scoring rule we tried (overall agreement, agreement among decided votes, minimum agreement across opinion groups). Of the 58 seats:
- only 15 statements are selected in at least 95% of participant resamples;
- 149 statements are reliably excluded;
- the remaining 58 − 15 = 43 seats are filled from a *contested band* of 111 statements.

A procedure-free lower bound (Theorem 1) shows this is not an artefact of a conservative method. With about 80 votes per statement, about 34 of the 58 seats cannot be certified with 80% power by *any* valid procedure.

Theorem 1 also shows that the width of this band depends on n, the number of participants, only through **m, the number of votes each statement receives**. It is Θ(m^(−1/2)) up to a √(log M) factor, where M is the number of statements. Across CCAI and 17 public Polis conversations (n = 100–5,398), contested statements per seat fall with m (log-log slope −0.41, t = −4.3) but not with n (t = 1.1). Polis agendas grow with participation (M ∝ n^0.68), so per-statement exposure grows much more slowly than headcount.

The contested band is also where the designer acts. Across 72 aggregation pipelines that all fit CCAI's published description, 89% of designer disagreements fall on contested statements. Meanwhile 10 of the 15 statements the votes reliably select are dropped by at least one of 79 defensible designs.

A CCAI-calibrated simulation suggests what it would take to fix this:
- certifying about half of the 58 seats needs on the order of 3,200 votes per statement, about 25,000 participants at CCAI's per-participant voting rate;
- routing votes to rank-uncertified statements roughly doubles the number of certified seats at large budgets, with no false certifications in our runs. Routing votes to statements near the cutoff, by contrast, does produce false certifications.

The audit also finds two further problems:
- the opinion-group labels released with CCAI are statistically indistinguishable from random, consistent with a row-misalignment in the released notebook;
- count-based proportional rules on sparse Polis data turn organizer-controlled exposure into organizer authorship.

Our answer to "who writes the constitution?" is therefore quantitative. At current exposure levels, the public writes a shortlist, and sampling noise and the designer choose from it. Public-input constitutions should report which seats the votes certify.

## 中文摘要

"集体宪法 AI"（CCAI）等公众参与机制常被描述为由"公众"而非开发者来书写约束语言模型的原则。然而，只有当公众的投票在统计意义上**确定**了宪法的内容时，宪法才能归属于公众。本文把这一点变为可检验的命题：把 k 条宪法中的每个"席位"视为一个排序断言（"语句 s 位列前 k"），利用参与者自助法下的排序置信集（Mogstad 等，2024）判断哪些席位被投票"认证"、哪些被排除、哪些悬而未决。

在公开的 CCAI 数据（1002 名参与者、275 条语句、34563 张票）上，无论采用同时还是逐条的排序区间，也无论采用哪种计分规则（总体同意率、剔除"跳过"后的同意率、各意见群体中的最低同意率），投票在 95% 水平上**一个席位都无法认证**。在 58 个席位中：
- 仅 15 条语句在至少 95% 的重抽样中入选；
- 149 条被稳定排除；
- 其余 43 个席位从 111 条"争议带"语句中产生。

一个与具体方法无关的下界（定理 1）表明，这并非保守方法造成的假象：在每条语句约 80 票的曝光下，58 个席位中约 34 个无论用何种有效程序都无法以 80% 的检验力认证。

定理 1 还表明，这条争议带的宽度只通过**每条语句获得的票数 m**依赖于参与人数 n，量级为 Θ(m^(−1/2))，至多相差一个 √(log M) 因子（M 为语句数）。在 CCAI 与 17 个公开 Polis 对话（n = 100–5398）中，每席位的争议语句数随 m 下降（双对数斜率 −0.41，t = −4.3），而与 n 无关（t = 1.1）。Polis 的议程随参与人数扩张（M ∝ n^0.68），因此单条语句的曝光增长远慢于人数增长。

设计者恰恰在争议带中发挥作用。在 72 条均符合 CCAI 公开描述的聚合管线之间，89% 的分歧落在争议语句上；而投票稳定选中的 15 条语句中，有 10 条至少被 79 种可辩护设计中的一种剔除。

基于 CCAI 校准的模拟给出了改进所需的量级：
- 要认证 58 个席位中的约一半，每条语句约需 3200 票，按 CCAI 的人均投票量折合约 2.5 万名参与者；
- 把投票定向分配给排序尚未认证的语句，可在较大预算下使已认证席位约增加一倍，且在我们的运行中未出现错误认证；而只向截断线附近的语句分配投票则会产生错误认证。

审计还发现两个问题：
- CCAI 公开数据中的意见群体标签与随机标签无法区分，与公开笔记本中的行顺序错位相符；
- 在稀疏的 Polis 数据上，按原始计数运行的比例代表规则会把组织者掌控的曝光转化为组织者的作者身份。

因此，"谁在书写宪法"有一个定量的回答：在当前曝光水平下，公众写下的是一份候选名单，由抽样噪声和设计者从中挑选。公众参与式宪法应当报告哪些席位得到了投票的认证。

---

## 1 Introduction

Constitutional AI trains a model against a written list of principles instead of per-example human labels (Bai et al., 2022). This moves much of a model's normative content into a short natural-language document, which raises the question in our title: who should write it?

Gabriel (2020) argued that the central normative problem of alignment is not finding the true moral theory but finding a *fair process* for choosing principles. Conitzer et al. (2024) make the same case for constitutional AI specifically. Developers now publish their constitutions; Anthropic's January 2026 revision, for example, is credited to named internal authors (Anthropic, 2026). Developers and civil-society groups have also run public-input experiments:
- CCAI, with about 1,000 U.S. adults on Polis (Huang et al., 2024);
- OpenAI's Democratic Inputs grants (OpenAI, 2024);
- collective-dialogue policy drafting (Konya et al., 2023).

These processes are usually described by *who participated*. Their legitimacy claim, "the public chose these principles," also has a statistical precondition: the public's votes must actually *determine* which principles were chosen. Suppose a different, equally valid draw of participants would have produced a substantially different list. Then the list is not "what the public chose." It is one draw from what the public might have chosen, and the remaining choices were made by noise and by whoever designed the aggregation pipeline. The same reasoning lies behind reporting confidence intervals for league tables of schools, hospitals or neighbourhoods. Mogstad et al. (2024) showed that many published rankings do not survive it.

We therefore ask: **in a deployed public-input process, which seats of the resulting constitution do the public's votes statistically determine, what governs how many, and who fills the rest?**

**Contributions.**

1. **A certification audit for public-input constitutions** (Section 3). Each seat is a claim that a statement is in the top k. We use participant-level bootstrap rank-confidence sets (Mogstad et al., 2024) to classify every statement as certified-in, certified-out or uncertified, together with a simpler bootstrap stability measure. A design multiverse and a pipeline-integrity check then show who fills the uncertified seats.

2. **Theorem 1: an exposure law for certification** (Section 4). In a simple model, any valid certification procedure leaves uncertified a band of width at least (1 − α − β)·√(2q(1−q)/m) around the selection cutoff q, where m is a statement's number of votes and α, β are the error rates. A Hoeffding procedure certifies everything outside a band of width O(√(log M / m)). So the number of seats the public can certify depends on n only through m. Under a Polis-style voting budget with growing agendas, it improves only slowly with headcount.

3. **Measurements on CCAI** (Section 6).
   - (a) No seat of a 58-statement constitution is certified. 15 statements are stable, 149 are reliably excluded, and 43 seats come from a 111-statement contested band. Theorem 1 puts about 34 seats beyond any valid procedure at CCAI's exposure.
   - (b) 89% of the disagreements among 72 description-consistent designs fall on contested statements.
   - (c) The released opinion groups are statistically random, which is consistent with a row-misalignment in the released notebook. All label-dependent results are now deterministic and reported across 50 clustering seeds.
   - (d) Count-based proportional rules amplify organizer-controlled exposure into organizer authorship.

4. **A cross-process test** (Section 6.4). On CCAI plus 17 public Polis conversations, contested statements per seat track per-statement exposure (t = −4.3) and not headcount (t = 1.1), as Theorem 1 predicts. It also partly explains a round-1 observation, that designer influence relative to sampling influence rises with process size: the sampling component of that ratio tracks exposure, not headcount.

5. **Design consequences** (Sections 6.5 and 7). A CCAI-calibrated simulation gives the votes per statement needed to certify a given share of seats. It also shows that routing votes to uncertified statements, in the spirit of adaptive top-k identification (Kalyanakrishnan et al., 2012) and adaptive elicitation (Halpern et al., 2023), roughly doubles certified seats at large budgets.

We do not claim that public input is illegitimate or that CCAI's final constitution was wrong. That constitution also went through manual merging steps that are not in the released data. Our claim is narrower: whether the public's votes determine a seat is measurable, and at the exposure levels of current processes most seats are not determined.

## 2 Related Work

**Constitutions and public input for AI.**
- *Constitutional AI and CCAI.* Constitutional AI (Bai et al., 2022) put a written principle list at the centre of alignment training. CCAI (Huang et al., 2024) sourced a "public" constitution through Polis from about 1,000 U.S. adults. It retained statements that passed a consensus threshold within both Polis opinion groups, then de-duplicated and merged them by hand (Anthropic, 2023).
- *Other public-input processes.* OpenAI's Democratic Inputs program funded ten such prototypes (OpenAI, 2024). Konya et al. (2023) combined collective dialogues with LLM-drafted policy and reported 75–81% support for the resulting guidelines.
- *Extracting principles from preferences.* Inverse Constitutional AI (Findeis et al., 2025) compresses preference data into principles.

**Who shapes Polis outputs.**
- *Gelauff (2026).* An analysis of 119 Polis conversations argues that outcomes are co-shaped by the developer, the convener and the participant, and notes that exported clusterings cannot be reproduced from the votes alone. Our design and agenda channels are close in spirit. What we add is a statistical criterion (certification), a theorem on what governs it, and a decomposition of *where* designer discretion acts (the contested band).
- *Blair et al. (2026).* They show that Polis-style bridging scores depend on the chosen partition. They propose partition-robust bridging metrics that hold up under sparse votes. Our Proposition 2 (partition leverage) is a simpler, statement-level version of that concern.

**Representation in committee selection.**
- *ABC voting.* Choosing k statements from approval votes is approval-based committee (ABC) voting (Lackner & Skowron, 2023). Justified representation (JR) formalizes when a cohesive group is owed a seat (Aziz et al., 2017).
- *Equal Shares.* The Method of Equal Shares satisfies strong proportionality axioms (Peters & Skowron, 2020).
- *JR in ranking.* Revel et al. (2025) impose a JR constraint on comment ranking and test it in deployment. This is close to the "consensus core plus coverage" hybrid that we examined in round 1 and now report only as one design in the multiverse.
- *Incomplete votes.* Halpern et al. (2023) show that with incomplete votes, non-adaptive elicitation needs impractically many votes for representation guarantees, and they propose adaptive routing. Our Theorem 1 is the analogous statement for *top-k certification* rather than proportional guarantees, and our routing simulation is a concrete instance of their recommendation.
- *Generative social choice and pluralism.* Fish et al. (2024), Sorensen et al. (2024) and Kirk et al. (2024) treat related questions.

**Inference for ranks and top-k identification.**
- *Rank confidence sets.* Mogstad et al. (2024) construct marginal and simultaneous confidence sets for ranks from estimated means. They show that several widely reported rankings of US neighbourhoods are not statistically supported. We apply their construction to statements in a public-input process, with participants as the resampling unit.
- *Top-k identification.* PAC top-k identification in bandits, for example LUCB (Kalyanakrishnan et al., 2012), gives sample-complexity results for identifying the best arms adaptively. Our Theorem 1 is an elementary, non-adaptive, finite-sample version tailored to Bernoulli votes.

As far as we know, these tools have not been applied to AI-constitution processes. Nor has anyone shown that certification there is governed by exposure rather than headcount.

**Researcher degrees of freedom.** Multiverse analysis (Steegen et al., 2016), specification curves (Simonsohn et al., 2020) and the garden of forking paths (Gelman & Loken, 2013) motivate our designer channel.

## 3 The Certification Audit

**Data model.**
- There are N participants and M statements.
- V ∈ {+1, −1, 0, ∅}^{N×M} records agree, disagree, pass and *not seen*.
- m_s is the number of voters who saw statement s.
- A design d maps V to a set W of k statements: a score (agreement estimator, optional opinion-group partition and consensus rule) followed by top-k, or a committee rule.

**Certification.**
- *Score and bootstrap.* For a design with a per-statement score θ_s (for example, agreement rate, or minimum agreement across groups for a fixed partition), we compute the estimate θ̂ and B = 1,000 bootstrap replicates, resampling *participants* with replacement. Participants are the sampling unit, and their votes on different statements are dependent.
- *Rank bounds.* Following Mogstad et al. (2024), we form studentized bootstrap confidence intervals for all pairwise differences θ_s − θ_j, using the bootstrap covariance. Let c be the critical value. A statement's rank interval is [1 + #{j: θ_j − θ_s significantly > 0}, M − #{j: θ_s − θ_j significantly > 0}].
- *Two versions.* In the *marginal* version, c is the 95% quantile of the bootstrap maximum of |t| over the pairs that involve s, so each statement's interval is valid on its own. In the *simultaneous* version, the maximum runs over all pairs, so every interval is valid jointly.
- *Classification.* s is **certified-in** if its upper rank bound is at most k, **certified-out** if its lower rank bound exceeds k, and **uncertified** otherwise.
- *Stability.* As a lenient, non-inferential companion, we report bootstrap stability π_s, the share of bootstrap draws in which s is in the top k. **Stable-in** means π_s ≥ 0.95, **stable-out** means π_s ≤ 0.05, and the rest form the **contested band**.
- *Threshold mode.* For constitutions defined by a threshold ("agreement above τ in every group"), s is certified above τ if its simultaneous lower confidence bound exceeds τ.

**Designer channel.** The multiverse is unchanged from round 1, so it is not tuned to the new analysis.
- *Opinion groups:* 12 Polis-style groupings, from 2 PCA conventions × 2 missing-vote imputations × K ∈ {2, 3, 4}.
- *Agreement estimators:* 3 (agree/seen, agree/decided, Laplace).
- *Consensus rules:* 2 (maximin, product).

That gives **72 consensus designs**, all consistent with "statements with high agreement in every opinion group." We add 7 others:
- approval-rate ranking with 3 estimators;
- sequential PAV, the Method of Equal Shares (MES) and greedy Chamberlin–Courant coverage (CC), on raw approvals;
- a consensus-core plus coverage hybrid.

The total is 79 designs. We report:
- designer-swap distance: the mean Jaccard distance between two designs on the same participant sample;
- sample-swap distance: the mean Jaccard distance between two bootstrap samples under the same design;
- where designer disagreements fall relative to the certification classes.

The k-means seed is now held fixed across bootstrap draws, so that sample-swap does not absorb clustering-initialization noise (reviewer M2). We measure that noise separately.

**Agenda and integrity channels.** We report:
- organizer and seed shares;
- exposure, measured as votes per statement and its dependence on submission order;
- seed shares inside each design's constitution;
- an integrity check of released intermediate artefacts (opinion-group labels) against a permutation null.

## 4 Theory

### 4.1 An exposure law for certification

*Model.* Statement s receives m_s votes. Each vote is an independent Bernoulli(p_s) "agree" indicator, and votes on different statements are independent. Let T_k(p) be the set of the k statements with the largest p (ties broken arbitrarily). For each s, let q_s be the k-th largest value of {p_j : j ≠ s}. Statement s is in the top k if p_s > q_s and out if p_s < q_s.

A *certification procedure* maps the votes to a set C. It has *per-statement level α* if P_p(s ∈ C) ≤ α whenever s ∉ T_k(p).

**Theorem 1 (certification band).**

(i) *Impossibility.* For every procedure of per-statement level α, every p and every s ∈ T_k(p):
P_p(s ∈ C) ≤ α + (p_s − q_s)·√(m_s / (2 q_s(1 − q_s))).
Hence a seat can be certified with power at least 1 − β only if p_s − q_s ≥ (1 − α − β)·√(2 q_s(1 − q_s)/m_s).

(ii) *Achievability.* Let ε_j = √(log(2M/α)/(2m_j)). Certify s if #{j ≠ s : p̂_s − ε_s > p̂_j + ε_j} ≥ M − k. This procedure has familywise error at most α. If all m_j = m, it certifies, with probability at least 1 − α, every s with p_s − p_(k+1) > 4ε, where p_(k+1) is the (k+1)-th largest rate.

*Proof.*

(i) Fix s ∈ T_k(p) and ε > 0, and let p′ equal p except that p′_s = q_s − ε. Under p′, s has at least k statements above it, so s ∉ T_k(p′) and P_{p′}(s ∈ C) ≤ α. The two data distributions differ only in the m_s votes on s. So |P_p(s ∈ C) − P_{p′}(s ∈ C)| ≤ TV(Bin-sequence(m_s, p_s), Bin-sequence(m_s, p′_s)). By Pinsker's inequality, this is at most √(m_s·KL(p_s ‖ p′_s)/2). The χ² bound gives KL(p ‖ p′) ≤ (p − p′)²/(p′(1 − p′)). Let ε → 0.

(ii) By Hoeffding's inequality and a union bound, the event E = {|p̂_j − p_j| ≤ ε_j for all j} has probability at least 1 − α. On E, p̂_s − ε_s > p̂_j + ε_j implies p_s > p_j. So a certified s exceeds at least M − k others and lies in T_k(p): the familywise error is at most α. Conversely, on E, if p_s − p_(k+1) > 4ε, then for each of the M − k statements j with p_j ≤ p_(k+1) we have p̂_s − p̂_j ≥ p_s − p_j − 2ε > 2ε, so s is certified. ∎

**Corollary 1 (exposure, not headcount).** Suppose the true rates are i.i.d. draws from a density f that is continuous and positive at the cutoff quantile q, all m_j = m, and k = ρM. Then the expected number of top-k seats that no level-α procedure can certify with power 1 − β is M·f(q)·(1 − α − β)·√(2q(1 − q)/m)·(1 + o(1)). By (ii), the expected number left uncertified by the Hoeffding procedure is at most M·f(q)·O(√(log M/m)).

Both bounds depend on the participant count n only through m. Now take a Polis-style budget: each participant casts v votes, and the agenda grows as M ∝ n^γ. Then m = vn/M ∝ n^(1−γ), and the uncertifiable *fraction* of seats scales as n^(−(1−γ)/2). With γ̂ = 0.68 estimated across the 18 processes (Section 6.4), a tenfold increase in participants shrinks it by only about 31%.

*Remarks.*
- (1) Theorem 1 is elementary: a two-point Le Cam argument and a union bound. Its value lies in which quantity it identifies (m, not n) and in the fact that its prediction can be tested on data (Section 6.4).
- (2) Votes by the same participant on different statements are not independent in practice. Our empirical procedure resamples participants and does not rely on this assumption. Only the theorem's constants do.
- (3) The factor f(q) explains why high-agreement pools such as CCAI's (78.9% of votes are "agree") are hard to certify: many statements pile up near the cutoff.
- (4) Consensus scores (minimum agreement across groups) are estimated from the smaller group's ≈ α_min·m votes, so by (i) their band is wider by a factor of about α_min^(−1/2).

### 4.2 Partition leverage

**Proposition 2 (partition leverage).** Fix statement s with overall agreement p among its viewers, and α ∈ (0, 1].
- (a) Every group G that contains at least a share α of the viewers has agreement a_G(s) ≥ max{0, 1 − (1 − p)/α}.
- (b) Every partition has min_j a_j(s) ≤ p.
- (c) If (1 − p) ≤ α, the bound in (a) is attained up to rounding by a group of size ⌈αN⌉ that contains all non-agreers.

*Proof.* (a) G contains at most (1 − p)N non-agreers. (b) p is a convex combination of the group rates. (c) Construct G directly. ∎

This bounds exposure to the choice of partition one statement at a time. It does not bound the effect of a realized partition, because one partition cannot push every statement to its floor at once and Polis partitions are endogenous k-means outputs. With CCAI's minority share α = 0.382 and τ = 0.72, a statement is partition-robust only if p > 0.893. 65 of the 210 statements with p > 0.72 meet this. We report this as exposure, not as an effect. Blair et al. (2026) develop partition-robust bridging metrics that address this concern directly.

*Consensus filters and JR (remark).* A τ-consensus filter gives each group, and any faction spread thinly across groups, a veto. For a fixed partition with 1/2 < τ < 1 − 1/k, one can construct approval profiles in which every admissible committee violates JR. The construction is given in Appendix A. It assumes an exogenous partition, and the violations we observe on real data (Section 6.6) occur equally under plain approval voting. So we treat this as a known tension (Ovadya & Thorburn, 2023) rather than a contribution.

## 5 Data and Setup

**CCAI.** We use the anonymized export released with Huang et al. (2024), repository `saffronh/ccai`:
- `clean_votes.csv`: 1,002 participants, after the authors removed low-quality participants and those with 25 or fewer votes;
- `clean_comments.csv`: 275 moderated-in statements.

**Polis openData.** We use every conversation in the Computational Democracy Project's openData repository that meets four conditions:
- it has a participants-votes export;
- it has at least 100 participants with at least 7 votes each;
- it has at least 40 moderated-in statements seen by at least 10 voters each.

That gives **17 conversations, with n = 100–5,398**. This includes the 5,398-participant `march-on` conversation, which round 1 excluded for runtime. For each, k = clip(round(0.2M), 10, 58).

**Procedure.**
- *Certification:* B = 1,000 participant bootstraps per score.
- *Designer × sample analysis:* B = 40 for CCAI and B = 20 per openData conversation.
- *Integrity:* 50 k-means seeds. PCA uses a deterministic full SVD, which fixes the round-1 non-determinism (reviewer F1).
- *Simulations:* planning and routing simulations (Section 6.5) use empirical-Bayes-shrunk CCAI rates as ground truth, from a beta-binomial fit with α̂ = 8.31 and β̂ = 2.19.

All numbers come from the scripts in `code/` (run order in `code/README.md`), and all outputs are under `results/`. The total CPU time is about 1.5 hours on 4 cores.

## 6 Results

### 6.1 Agenda and exposure (CCAI)

**Votes.** The release has 34,563 observed votes, so the matrix is 12.5% observed. Of these votes, 78.9% are agree, 13.2% disagree and 7.9% pass.

**Who wrote the pool.**
- 61 of the 275 statements (22.2%) come from the organizer account.
- 21 of those were posted before any participant statement. This matches the "21 seed statements" in Anthropic (2023).
- The remaining 214 statements were written by 156 participants (15.6% of the 1,002).

**Exposure.** Exposure is very unequal and follows submission order.
- Seeds were seen by 386 voters on average, against 104 for the other statements.
- Spearman ρ(submission order, number of viewers) = −0.83.
- The median statement has m = 81 viewers.

This exposure profile governs everything below.

### 6.2 The votes certify no seat of a 58-statement CCAI constitution

Table 1 and Figure 1 summarize the certification audit for k = 58, the length of Anthropic's 2023 constitution that CCAI targeted.

*Table 1. Certification of a 58-seat CCAI constitution, B = 1,000 participant bootstraps, 95% level. "cert-in/out" use rank-confidence sets; "stable/contested" use bootstrap inclusion frequency. Source: `results/certify_ccai.json`.*

| Score | cert-in (marginal) | cert-in (simult.) | cert-out (marginal) | stable-in | contested | stable-out |
|---|---|---|---|---|---|---|
| agree/seen (CCAI estimator) | **0** | 0 | 59 | 15 | 111 | 149 |
| agree/decided | 0 | 0 | 42 | 5 | 137 | 133 |
| min-group agree/seen, re-aligned groups | 0 | 0 | 37 | 14 | 117 | 144 |
| min-group agree/seen, released groups | 0 | 0 | 28 | 12 | 130 | 133 |

**No seat is certified, under any score.** Even the statement with the highest estimated agreement (0.976 from 125 viewers) has a marginal 95% rank interval of [1, 158].

**The votes do much better at exclusion than at selection.** 59 statements are certified *out* by marginal rank intervals, and 149 are stable-out.

**Exposure decides which statements get certified.** The 15 stable-in statements had 174 viewers on average, against 110 for contested statements.

**The finding holds across constitution sizes.**
- At k = 10, 20 and 30, no statement is certified-in, and 0, 1 and 3 statements are stable-in.
- At k = 100 and 150, there are still no certified-in statements, with 38 and 85 stable-in.

**Threshold mode.** CCAI's constitution is defined by a threshold ("consensus above τ in both groups"). With re-aligned groups, 181 statements have minimum-group agreement above 0.72, but only 39 are certified above 0.72 with simultaneous 95% bounds (43 of 188 with the released groups). For overall agreement, 110 of 210 are certified above 0.72.

**This is not an artefact of a conservative procedure.** Theorem 1(i), applied with the empirical-Bayes rates as truth, CCAI's actual per-statement exposure, α = 0.05 and β = 0.2, puts **34 of the 58 seats** inside the band where *no* level-0.05 procedure has 80% power. With raw rates as truth the figure is 30. The rank-interval method certifies 0, the bound allows at most about 24, and bootstrap stability, which makes no validity claim, finds 15.

### 6.3 The designer acts in the contested band

**Designer disagreements fall on contested statements.** On the full data, the 72 description-consistent consensus designs give 58-statement constitutions with mean pairwise Jaccard similarity 0.569 (minimum 0.275). Only 10 statements are in all 72, and 136 are in at least one. We classify each statement by its av|seen stability:
- **89.4%** of the statements on which two consensus designs disagree are contested (84.2% over all 79 designs);
- 4.0% are stable-in;
- 6.6% are stable-out.

The contested band holds 40% of the statements (111 of 275) but almost nine-tenths of the designer disagreement. Of the 136 statements that any consensus design selects, 103 are contested.

**Some designs override the votes.** 10 of the 15 stable-in statements are excluded by at least one of the 79 designs, and only 7 are selected by all 72 consensus designs. Measured with each design's own bootstrap (B = 40, π ≥ 0.95), consensus designs fill on average 11.0 of their 58 seats with vote-stable statements (range 3–23). Approval-rate designs fill 16.7 (8–21), and count-based proportional rules fill 36.3 (28–41). The proportional rules buy that stability by favouring high-exposure statements (Section 6.6).

**Which knobs matter.** Table 2 shows that single technical knobs which a reader cannot infer from the CCAI paper's prose, such as PCA centring and imputation, each move about 30% of the constitution. Redrawing the sample moves more.

*Table 2. Mean Jaccard distance between 58-statement constitutions: designer-swap (same sample, one knob changed) and sample-swap (same design, sample redrawn). CCAI, B = 40, n = 1,002, k-means seed fixed across draws. Source: `results/ccai_results.json`.*

| Knob varied | designer-swap | sample-swap |
|---|---|---|
| min vs product consensus | 0.246 | 0.508 |
| PCA centring convention | 0.300 | 0.528 |
| missing-vote imputation | 0.308 | 0.531 |
| agreement estimator (3) | 0.353 | 0.541 |
| number of groups K (2/3/4) | 0.379 | 0.564 |
| rule family (consmin, AV, PAV, MES) | 0.560 | 0.390 |
| all 72 consensus designs | 0.440 | 0.565 |
| all 79 designs | 0.465 | 0.551 |

**k-means seed noise is small.** Changing only the k-means seed on the full data moves a consensus constitution by a mean Jaccard distance of 0.023 (maximum 0.21). Decoupling the seed from the bootstrap index left the CCAI swap distances unchanged to three decimals, so reviewer M2's concern does not affect the CCAI numbers.

**The three channels largely interact.** In a two-way variance decomposition of inclusion over designs × samples (consensus family), the designer main effect is 12.9%, the sample main effect 35.5% and the interaction 51.6%. The large interaction is the contested band seen from another angle: which contested statements a design picks depends on the sample.

### 6.4 Across 18 processes, certification tracks exposure, not headcount

For CCAI and the 17 openData conversations we compute the av|seen certification statistics at k = clip(0.2M, 10, 58) (`results/certify_opendata.csv`, Figure 2).

**Exposure predicts the contested band.**
- In an OLS regression of log(contested statements per seat) on log n and log m (median votes per statement), the coefficient on **log m is −0.41 (t = −4.31)**, close to Theorem 1's −0.5. The coefficient on **log n is 0.08 (t = 1.13)**. R² = 0.58.
- Rank correlations: ρ(contested/seat, m) = −0.57 (p = 0.013) and ρ(contested/seat, n) = −0.07 (p = 0.78). Partial Spearman: −0.60 for m given n, and 0.23 for n given m.
- The share of seats certified-in by marginal rank intervals has ρ = 0.79 with m (p = 10⁻⁴) and ρ = 0.25 with n (p = 0.32).

**Only high-exposure conversations certify many seats.** Only two conversations have m above 500:
- `march-on` (m = 2,336) certifies 15 of 44 seats;
- `vtaiwan.uberx` (m = 568) certifies 5 of 20.

Most conversations certify 0–2 seats.

**Why exposure lags headcount.** Across the 18 processes, the agenda grows as M ∝ n^0.68, and median exposure grows only as m ∝ n^0.40 (ρ(n, m) = 0.42, p = 0.08). This is the budget mechanism of Corollary 1.

**Robustness.**

| Variant | Coefficient on log m | Spearman ρ(contested/seat, m) |
|---|---|---|
| Keep one Austrian conversation (n = 14) | t = −3.32 | |
| Drop CCAI | t = −3.62 | |
| Drop `march-on`, the single most influential point | t = −2.20 | −0.49 (p = 0.044) |
| Drop `march-on` and control for the disagreement share | t = −1.41, n.s. | |

The last row shows that without `march-on`, exposure and agreement level cannot be fully separated in this small corpus. The coefficient on log n is never significant (|t| ≤ 1.2).

**The round-1 scale regularity is partly an exposure effect.** Round 1 reported that the ratio of designer-swap to sample-swap distance rises with n. We re-ran it with the k-means seed decoupled from the bootstrap index and with `march-on` added (`results/opendata_stats.json`, `results/opendata_ratio_exposure.json`):
- The consensus-family ratio still rises with n (ρ = 0.89, p = 1.5×10⁻⁶) and also with m (ρ = 0.59, p = 0.013). In OLS of log ratio on log n and log m, both contribute (t = 5.4 and 4.9).
- Its *sample* component tracks exposure: sample-swap falls with m (ρ = −0.66, p = 0.004) but not with n (ρ = −0.23, p = 0.37). The designer component tracks neither (ρ = 0.20 with n, −0.35 with m, both n.s.).
- The largest ratio is `march-on`'s, at 1.54 (95% Monte-Carlo interval 1.50–1.59). There exposure is highest, sampling noise is small and designer discretion remains.

We no longer claim that "designer choices dominate at scale" as a general law:
- Over the consensus family, designer-swap is *not* significantly larger than sample-swap (Wilcoxon p = 0.49).
- Over the full 79-design multiverse it is (p = 0.004).
- Monte-Carlo 95% intervals for the consensus ratio lie above 1 in 8 of 17 conversations and below 1 in 3.
- Ratios range from 0.80 to 1.54.
- Across 100 random sub-multiverses per conversation (half of the consensus designs plus a random subset of the other rules), the 5–95% range runs from 0.79 to 1.87. The multiverse's composition therefore moves the ratio by as much as the conversation does (`results/opendata_ratio_ci.csv`).

The defensible statement is narrower: whatever the multiverse, sampling noise falls with exposure and designer discretion does not.

### 6.5 What it would take: exposure planning and routing

**Uniform exposure.** We simulate CCAI-like data with M = 275 and empirical-Bayes rates as truth, giving every statement m votes. We average 3 replicates with a parametric bootstrap of B = 400 (`results/certify_planning.csv`; Figure 3, left).

| votes per statement m | equivalent n at CCAI's 34.5 votes/participant | certified-in (marginal) | stable-in | Thm 1: seats beyond any procedure |
|---|---|---|---|---|
| 100 | 800 | 0 | 13.3 | 36 |
| 400 | 3,200 | 1.7 | 29.0 | 25 |
| 800 | 6,400 | 9.0 | 36.3 | 15 |
| 1,600 | 12,800 | 18.3 | 41.7 | 11 |
| 3,200 | 25,500 | 27.3 | 44.0 | 8 |
| 6,400 | 51,000 | 33.0 | 48.3 | 4 |

Certifying about half the seats (27 of 58) with valid rank intervals takes about 3,200 votes per statement. At CCAI's per-participant rate, that is about 25,000 participants, or, more cheaply, many more votes per participant on fewer statements.

**Routing.** We compare uniform exposure with two adaptive rules. The first is *rank-aware routing* (`results/routing_sim.csv`; Figure 3, right):
- give every statement 30 votes;
- then send batches of 2,750 votes to statements whose approximate rank interval still straddles k, weighted by their standard error.

The second, *cutoff-proximity routing*, sends each batch to statements whose 2-standard-error interval contains the estimated cutoff.

All arms are evaluated with the same marginal rank-certification procedure, against the known truth. Means over 4 replicates (false certifications in parentheses where non-zero):

| total votes | arm | certified-in | certified-out | stable-in | contested |
|---|---|---|---|---|---|
| 34,563 (CCAI) | uniform | 0.0 | 67.3 | 17.3 | 98.0 |
| | rank-aware | 0.0 | 75.5 | 16.8 | 95.8 |
| | cutoff-proximity | 4.3 (1.5 false) | 34.0 | 29.8 | 66.3 |
| 138,252 (4×) | uniform | 3.3 | 115.8 | 29.8 | 62.3 |
| | rank-aware | 7.5 | 132.3 | 34.3 | 50.5 |
| | cutoff-proximity | 4.3 (1.5 false) | 37.5 | 49.3 | 20.3 |
| 276,504 (8×) | uniform | 10.5 | 141.5 | 36.8 | 42.8 |
| | rank-aware | 19.5 | 168.0 | 40.5 | 34.0 |
| | cutoff-proximity | 5.3 (0.8 false) | 36.8 | 53.5 | 9.0 |

Rank-aware routing roughly doubles certified seats at 4–8× CCAI's budget, and none of its runs produced a false certification, either in or out. At CCAI's own budget it does not help, because the binding constraint is the total number of votes.

The naive alternative, routing to statements near the estimated cutoff, looks better on bootstrap stability: at 8× it leaves 53.5 seats stable and a contested band of only 9. But it certifies few seats and produces false certifications in every budget setting (0.8–1.5 per run on average). Statements far from the cutoff keep wide intervals, and adaptive concentration makes the naive bootstrap overconfident. Routing should therefore target rank uncertainty, not closeness to the cutoff, and stability alone is not a safe success metric for adaptive designs.

### 6.6 Representation and exposure capture

**JR.** No design violates JR on CCAI's observed approvals. This is because the approvals are sparse: few groups of N/k or more voters are *observed* to share an approval. Across the 17 conversations, consensus designs and approval-rate designs violate JR in the same 6 conversations (the five Austrian climate conversations and `march-on`). PAV, MES, CC and the hybrid never do. Since plain approval voting fails exactly where consensus filtering does, we attribute these violations to non-proportional rules generally, not to consensus filtering specifically (reviewer M4).

**Exposure capture.** Count-based proportional rules treat "unseen" as "disapproved," so statements shown to more voters collect more approvals. On CCAI, seeds are 7.6% of the pool but 31.0–34.5% of the PAV, MES and CC constitutions. Across the 15 processes that have seed statements, the median ratio of a rule's seed share to the pool seed share is:
- **3.09** for count-based PAV and **3.19** for count-based CC;
- 1.09 for approval-rate ranking.

We tested two exposure-normalized variants (reviewer M5; `results/exposure_rules.json`):
- **Imputed utilities** (unseen replaced by the statement's agree rate) remove the over-representation (0.99×; Wilcoxon p = 6×10⁻⁵ against count-based PAV). However, PAV then collapses onto approval-rate ranking (median Jaccard 0.95) and violates JR in 3 of 18 processes.
- **Inverse-propensity weighting** keeps JR (0 violations) but only halves the over-representation (2.01×).

On data where exposure is neither random nor complete, no post-hoc normalization we tried delivers exposure-neutrality and proportionality together. This is a concrete instance of Halpern et al.'s (2023) argument for adaptive elicitation.

**Group parity.** Measured with the re-aligned groups, consensus designs keep the two groups' endorsement rates within 0.0002–0.025 of each other. Proportional designs and the hybrid produce gaps of 0.16–0.18.

### 6.7 Pipeline integrity: the released opinion groups look random

**Which labels the filter used.** The CCAI notebook stores opinion-group labels in `clean_votes.csv` (619/383 participants). The group-consensus columns used for constitution building are computed from these labels.

**The suspected bug.** The notebook sorts the vote matrix by participant before PCA and k-means, then assigns the labels *positionally* back to the unsorted data frame.

**Replication.** We replicated this with a deterministic PCA and 50 k-means seeds:
- assigning the labels positionally reproduces the released labels with **ARI 0.83–0.96** (median 0.92);
- assigning them to the correct participants gives **ARI between −0.005 and −0.003**.

**The released groups carry no opinion structure.**
- Under the released labels, the two groups agree at almost identical rates (0.788 and 0.790).
- Their mean per-statement agreement difference is 0.0704. This lies inside a 200-permutation null (mean 0.0678, sd 0.0035, max 0.0792).
- Re-aligned labels give rates of 0.828 and 0.730 and a separation of 0.103–0.109 across seeds, about 10 null standard deviations above the null mean.

**Consequences.** At τ = 0.72, the released-label set has 188 statements. The re-aligned set has 175–182 across seeds (181 for seed 0). The two sets differ by 16–28 statements (median 23) out of a union of 192–197. Their Jaccard similarity is 0.86–0.92, and the two top-58 lists have Jaccard similarity 0.71–0.76 (`results/ccai_realignment_by_seed.csv`).

The released-label filter is closer to a plain majoritarian threshold: Jaccard 0.895 with the overall-agreement set, against 0.83–0.87 for re-aligned labels.

**Further notebook details (reviewer m3, m5).**
- The released notebook's cluster-assignment step is complete. The round-1 statement that it was truncated was wrong.
- The notebook selects K by silhouette score, whereas our replica fixes K = 2.
- Its group-consensus helper would raise an error, rather than return 0, if no group member agreed with a statement.

**Limits.** Our replica uses a recent scikit-learn, and the final CCAI constitution went through manual merging that is not in the release. The finding should be confirmed by the original authors, and it may not affect the published constitution. In light of Section 6.2, its practical importance is limited in any case: with either labelling, most seats are uncertified.

## 7 Discussion

### 7.1 What this says about "who should write the constitution"

**Authorship requires determination.** A constituency writes a seat of the constitution only if its votes determine that seat. In CCAI the public's votes determine which statements are *out* far better than which are *in*. They reliably exclude about 150 of 275 statements and reliably select about 15. The remaining seats are filled from a contested band by sampling noise and by designer choices that the published description does not pin down.

**More participants is not the fix.** By Theorem 1 and the cross-process evidence, certification depends on exposure per statement. Polis-style processes dilute exposure as agendas grow. Three levers help:
- capping or curating the agenda;
- asking each participant for more votes;
- routing votes to uncertified statements.

**The designer's role is to resolve ties the public left open, and that role should be explicit.** Two honest options exist:
- (a) declare the contested seats as designer or curator choices, for example by deliberation or a transparent lottery among statistically tied statements;
- (b) collect more votes on exactly those statements.

Presenting the contested band's resolution as "the public decided" overstates the public's role.

### 7.2 Recommendations: a certification statement

We suggest that public-input constitution efforts publish five items:
1. Exposure statistics: votes per statement, and the seed-to-non-seed exposure ratio.
2. The pre-registered pipeline and its score.
3. For each seat, its certification class (certified, stable, contested) under participant resampling, and the number of uncertified seats.
4. The designer-swap distance over a declared multiverse, restricted to the contested band.
5. An integrity check of intermediate artefacts against a null, such as the group-separation test of Section 6.7, which takes a few lines of code.

Before data collection, the planning table in Section 6.5 (or the released simulation) gives the votes per statement needed for a target number of certified seats.

### 7.3 Falsifiable predictions

**P1 (exposure law).** In new Polis-style processes with pre-registered k and scoring, the number of contested statements per seat falls with median votes per statement, with a log-log slope between −0.3 and −0.6, and does not fall with n once m is controlled. *Falsified* if, across at least 20 new processes, the partial rank correlation with m given n is not negative.

**P2 (routing).** In a randomized deployment, routing a fixed share of votes to rank-uncertified statements yields more certified seats than uniform routing at equal total votes, for budgets of at least about 4× M × 100.

**P3 (exposure capture).** With seeds shown to a random 50% of participants, the seed over-representation under count-based proportional rules falls toward 1.

## 8 Limitations

- **One AI-constitution process.** CCAI is, to our knowledge, the only AI-constitution public-input process with released vote-level data. The 17 openData conversations are civic deliberations. We use them only to test the exposure law, which concerns sampling rather than topic. The law rests on 18 processes with correlated n, M and agreement levels. Without the single highest-exposure conversation, the exposure effect is still significant (t = −2.2), but it becomes non-significant once the disagreement share is also controlled.
- **Statements, not principles.** CCAI's published constitution came from manual de-duplication and merging. Certification of statements does not transfer one-to-one to merged principles. Merging near-duplicates could make some contested seats moot, because two tied statements may express the same principle. This is the main reason the 30–56% designer-swap figures and the "0 certified seats" result should not be read as statements about the published constitution's wording.
- **What certification means.** Rank intervals answer "would a new sample from the same population, with the same exposure process, rank this statement in the top k?" They do not capture what different participants would have *written*, recruitment differences, or framing. The bootstrap also treats the realized exposure pattern as fixed, although in Polis it is partly endogenous. Mogstad et al.'s (2024) intervals are conservative, so we report bootstrap stability and the procedure-free Theorem 1 bound alongside them.
- **Theorem 1's model.** Theorem 1 assumes independent Bernoulli votes across statements. Our empirical procedures do not, but the planning simulation does (parametric, independent statements). Its "votes needed" figures depend on the empirical-Bayes truth: the beta prior smooths the rate distribution, and raw rates give similar numbers (Section 6.5, `results/certify_planning.csv`).
- **The routing simulation.** It is a simulation, not a deployment, with 4 replicates per cell. Adaptive sampling can invalidate naive bootstrap intervals, as the cutoff-proximity arm shows. Rank-aware routing produced no false certifications in our runs, but we have not proved validity for any adaptive design.
- **The multiverse is ours.** Designer-swap distances depend on which designs are admitted. The certification results do not depend on the multiverse, and the "89% of disagreements are contested" figure is a statement about where disagreement falls, not about its size.
- **The integrity finding.** It rests on a replica built with a recent library version and has not been confirmed by the original authors.
- **No model training.** We measure the constitution text, not the behaviour of models trained on it.

## 9 Conclusion

"Who should write the constitution?" has an empirical precondition: knowing whether the people named as authors actually determined it. We showed how to test this seat by seat, proved that the answer is governed by per-statement exposure rather than headcount, and confirmed that prediction across 18 public deliberations.

In the one AI-constitution process with public votes, the votes determine a shortlist, not a constitution. No seat of a 58-statement constitution is certified, about 34 are beyond any valid procedure at the realized exposure, and the designer's discretion operates almost entirely within the contested band. Public input can carry the legitimacy claimed for it only if processes are designed for enough exposure, or routing, to certify their seats, and if they report which seats were certified.

## Appendix A: Consensus filters versus JR, and the consensus-core hybrid

**Proposition A1.** Let k ≥ 3 and 1/2 < τ < 1 − 1/k. Fix any partition 𝒫 of N voters into g groups, with N ≥ g/(1 − τ − 1/k). Then there is an approval profile under which every committee of at most k τ-admissible statements violates JR.

*Proof.*
1. From each group G_j, put ⌊(1 − τ)|G_j|⌋ members into a faction F. Then |F| ≥ (1 − τ)N − g ≥ N/k.
2. Members of F approve only a statement c_F. All other voters approve every statement in a set C with |C| ≥ k.
3. Each c ∈ C has agreement at least τ in every group, so it is admissible.
4. c_F has agreement at most 1 − τ < τ in every group, so it is not admissible.
5. Any committee W ⊆ C therefore leaves F, a cohesive group of size at least N/k, unrepresented. ∎

The partition here is fixed before the profile is constructed. On this profile, an endogenous k-means partition would tend to isolate F as its own group. The violation then still occurs whenever |F| ≥ N/k, because a statement approved by only one group is inadmissible for every τ > 0.

**Proposition A2.** Let W₀ be any k₀ < k statements, and let U be the set of voters who approve nothing in W₀. Complete W₀ to size k by greedy coverage. If |U| < (k − k₀ + 1)N/k, the result satisfies JR.

*Proof.* Suppose a JR violation is witnessed by a set X with |X| ≥ N/k, all of whose members approve some c ∉ W. Then X ⊆ U. Each of the k − k₀ greedy steps covers at least N/k new voters from U \ X. Hence |U| ≥ (k − k₀ + 1)N/k, a contradiction. ∎

On CCAI the condition of Proposition A2 is vacuous (|U| = 15). The hybrid also loses the group parity of consensus rules (Section 6.6). We therefore report it only as one design in the multiverse. Revel et al. (2025) study JR-constrained ranking in depth.

---

## Figures

* **Figure 1** (`results/fig_cert1_ccai_rank_ci.png`): Marginal 95% rank-confidence intervals for all 275 CCAI statements. Colour shows bootstrap stability. The dashed line is the 58-seat cut.
* **Figure 2** (`results/fig_cert2_exposure_law.png`): Contested statements per seat against participants n (left) and median votes per statement m (right), for CCAI (orange), five Austrian conversations (purple) and 12 other Polis conversations.
* **Figure 3** (`results/fig_cert3_planning_routing.png`): Left, certified and stable seats against uniform votes per statement in a CCAI-calibrated simulation, with the Theorem 1 ceiling. Right, rank-aware routing against uniform exposure.
* **Figure 4** (`results/fig1_multiverse_jaccard.png`): Pairwise Jaccard similarity of 58-statement CCAI constitutions under 79 designs.
* **Figure 5** (`results/fig5_group_separation.png`): Group-separation statistic for released labels, re-aligned labels and 200 random relabelings.

## Data and Code Availability

All code is in `code/` (run order in `code/README.md`), and all outputs are in `results/`. Every number in Sections 5–6 is computed by this code from the public CCAI release and the Polis openData repository (CC BY 4.0, The Computational Democracy Project). The exceptions are facts attributed to cited sources, such as the 21 seed statements, the 58-principle target and Konya et al.'s support figures. Set `OMP_NUM_THREADS=1`.

## References

Anthropic (2023). *Collective Constitutional AI: Aligning a Language Model with Public Input* (research blog post). https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input

Anthropic (2026). *Claude's new constitution.* January 21, 2026. https://www.anthropic.com/news/claude-new-constitution

Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017). Justified representation in approval-based committee voting. *Social Choice and Welfare*, 48(2), 461–485. https://doi.org/10.1007/s00355-016-1019-3

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073.

Blair, C., de Raaij, J., Procaccia, A. D., Rubin-Toles, M., Shah, N., Si, M., & Wang, S. (2026). The structure of bridging. *Proceedings of the 27th ACM Conference on Economics and Computation (EC 2026)*. Preprint: https://www.cs.toronto.edu/~nisarg/papers/bridging.pdf

Computational Democracy Project. *openData: Data exports from "open data" Polis conversations.* GitHub repository, https://github.com/compdemocracy/openData (accessed November 2026; CC BY 4.0).

Conitzer, V., Freedman, R., Heitzig, J., Holliday, W. H., Jacobs, B. M., Lambert, N., Mossé, M., Pacuit, E., Russell, S., Schoelkopf, H., Tewolde, E., & Zwicker, W. S. (2024). Position: Social choice should guide AI alignment in dealing with diverse human feedback. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 9346–9360.

Findeis, A., Kaufmann, T., Hüllermeier, E., Albanie, S., & Mullins, R. (2025). Inverse Constitutional AI: Compressing preferences into principles. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2406.06560.

Fish, S., Gölz, P., Parkes, D. C., Procaccia, A. D., Rusak, G., Shapira, I., & Wüthrich, M. (2024). Generative social choice. *Proceedings of the 25th ACM Conference on Economics and Computation (EC 2024)*. arXiv:2309.01291.

Gabriel, I. (2020). Artificial intelligence, values, and alignment. *Minds and Machines*. https://doi.org/10.1007/s11023-020-09539-2

Gelauff, L. (2026). It takes three to converse: Empirical observations on how the developer, the convener and the participant shaped 119 Polis conversations. arXiv:2608.28368.

Gelman, A., & Loken, E. (2013). The garden of forking paths: Why multiple comparisons can be a problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis was posited ahead of time. Unpublished manuscript, Department of Statistics, Columbia University.

Halpern, D., Kehne, G., Procaccia, A. D., Tucker-Foltz, J., & Wüthrich, M. (2023). Representation with incomplete votes. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(5), 5657–5664. arXiv:2211.15608.

Huang, S., Siddarth, D., Lovitt, L., Liao, T. I., Durmus, E., Tamkin, A., & Ganguli, D. (2024). Collective Constitutional AI: Aligning a language model with public input. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)*. https://doi.org/10.1145/3630106.3658979. Data and notebook: https://github.com/saffronh/ccai

Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012). PAC subset selection in stochastic multi-armed bandits. *Proceedings of the 29th International Conference on Machine Learning (ICML 2012)*, 655–662.

Kirk, H. R., et al. (2024). The PRISM alignment dataset: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track*. arXiv:2404.16019.

Konya, A., Schirch, L., Irwin, C., & Ovadya, A. (2023). Democratic policy development using collective dialogues and AI. arXiv:2311.02242.

Lackner, M., & Skowron, P. (2023). *Multi-Winner Voting with Approval Preferences.* SpringerBriefs in Intelligent Systems. Springer. https://doi.org/10.1007/978-3-031-09016-5

Mogstad, M., Romano, J. P., Shaikh, A. M., & Wilhelm, D. (2024). Inference for ranks with applications to mobility across neighbourhoods and academic achievement across countries. *The Review of Economic Studies*, 91(1), 476–518. https://doi.org/10.1093/restud/rdad006

OpenAI (2024). *Democratic inputs to AI grant program: lessons learned and implementation plans.* January 16, 2024. https://openai.com/index/democratic-inputs-to-ai-grant-program-update

Ovadya, A., & Thorburn, L. (2023). Bridging systems: Open problems for countering destructive divisiveness across ranking, recommenders, and governance. Knight First Amendment Institute; arXiv:2301.09976.

Peters, D., & Skowron, P. (2020). Proportionality and the limits of welfarism. *Proceedings of the 21st ACM Conference on Economics and Computation (EC '20)*, 793–794. Full version arXiv:1911.11747.

Revel, M., Milli, S., Lu, T., Watson-Daniels, J., & Nickel, M. (2025). Representative ranking for deliberation in the public sphere. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267, 51583–51613. arXiv:2503.18962.

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4, 1208–1214. https://doi.org/10.1038/s41562-020-0912-z

Sorensen, T., et al. (2024). Position: A roadmap to pluralistic alignment. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 46280–46302.

Steegen, S., Tuerlinckx, F., Gelman, A., & Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*, 11(5), 702–712. https://doi.org/10.1177/1745691616658637
