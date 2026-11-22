# Who Writes the Constitution? An Authorship Audit of Public-Input AI Constitutions, with Evidence from Collective Constitutional AI

**Target venue:** ACM Conference on Fairness, Accountability, and Transparency (FAccT 2027), full paper track.

---

## Abstract

Public-input processes such as Collective Constitutional AI (CCAI) are presented as a way for "the public," not developers, to write the principles that steer a language model. But the public does not write a constitution by voting alone. Organizers write seed statements, a few participants write most of the rest, a random sample decides which statements get voted on and by whom, and a designer picks the clustering, the agreement estimator, the threshold and the aggregation rule that turn votes into text. We propose an **authorship audit** that splits these channels apart and measures each one. The audit has four parts: (i) **agenda authorship**, meaning who wrote the candidate pool and how exposure was allocated; (ii) **sample authorship**, meaning how much the constitution changes when the same process draws a different set of participants (estimated by bootstrap); (iii) **designer authorship**, meaning how much it changes across a multiverse of defensible aggregation pipelines run on the same votes; and (iv) **pipeline integrity**, meaning whether the released code does what it says. We prove three short results. The first bounds how far a designer's choice of opinion groups can move a statement's group-consensus score. The second shows that consensus filtering, which gives every group a veto, cannot guarantee justified representation (JR) whenever the threshold τ satisfies 1/2 < τ < 1 − 1/k. The third gives a condition under which a "consensus core plus coverage completion" rule restores JR. We apply the audit to the public CCAI data (1,002 participants, 275 statements, 34,563 votes) and replicate it on 16 public Polis conversations. On CCAI, 72 consensus pipelines that all fit the published description ("statements passing a consensus threshold in both groups") yield 58-statement constitutions with mean pairwise Jaccard similarity 0.57 (minimum 0.27). Only 10 statements appear in all 72. Redrawing the participants changes the constitution even more: the mean Jaccard distance is 0.57 between bootstrap samples, against 0.44 between designers. Across the 16 Polis conversations, the ratio of designer to sampling influence rises with the number of participants (Spearman ρ = 0.95). Designer choices dominate in every conversation with at least about 900 participants. Proportional rules applied to raw approval counts on sparse Polis data amplify exposure: organizers' 21 seed statements make up 7.6% of the CCAI pool but 31–35% of the constitutions these rules select. Finally, the group labels in the released CCAI data are statistically indistinguishable from random labels. They match a row-ordering misalignment in the released notebook (adjusted Rand index 0.83–0.94 with a replica of the misaligned pipeline, versus −0.004 with the correctly aligned one). Re-aligning the labels changes 26 of the 195 statements that pass the 0.72 threshold under either version. These are measurements of one process and 16 others, not a verdict on public input. They show that "who writes the constitution" has a quantitative answer, and that the designer's share of authorship should be reported alongside the public's.

## 中文摘要

"集体宪法 AI"（CCAI）等公众参与机制常被描述为让"公众"而非开发者来书写约束语言模型的原则。然而宪法并非仅由投票写成。组织者撰写种子语句，少数参与者贡献了大部分其余语句，随机抽样决定了哪些语句被谁投票，设计者还要选择聚类方法、同意率估计量、阈值与聚合规则，把投票变成文本。本文提出一种**"作者身份审计"**方法，把上述渠道拆开并分别度量。审计包括四个部分：（1）**议程作者**：候选语句由谁撰写，曝光如何分配；（2）**样本作者**：同一流程换一批参与者后宪法变化多少（自助法估计）；（3）**设计者作者**：对同一批投票运行一组均可辩护的聚合管线，宪法变化多少（多元宇宙分析）；（4）**管线完整性**：公开代码是否真正实现了其声称的功能。我们证明三个简短命题。其一，界定设计者通过选择"意见群体"划分能把一条语句的群体共识得分移动多远。其二，"每组一票否决"式的共识过滤在 1/2 < τ < 1 − 1/k 时无法保证正当代表性（JR）。其三，给出"共识核心加覆盖补全"规则恢复 JR 的条件。我们把审计应用于公开的 CCAI 数据（1002 名参与者、275 条语句、34563 张票），并在 16 个公开 Polis 对话上复现。在 CCAI 上，72 条均符合论文描述（"两组内均超过共识阈值"）的管线所生成的 58 条宪法，平均两两 Jaccard 相似度仅 0.57（最低 0.27），只有 10 条语句出现在全部 72 部宪法中。重新抽取参与者带来的变化更大：自助样本之间的平均 Jaccard 距离为 0.57，设计者之间为 0.44。在 16 个 Polis 对话中，设计者影响与抽样影响之比随参与人数上升（Spearman ρ = 0.95），在参与者约 900 人以上的对话中设计者影响均占主导。在稀疏 Polis 数据上直接按原始赞成票计数运行的比例代表规则会放大曝光偏差：组织者的 21 条种子语句占 CCAI 候选池的 7.6%，却占这些规则所选宪法的 31–35%。最后，CCAI 公开数据中的群体标签在统计上与随机标签无法区分，与公开笔记本中的行顺序错位相吻合（与错位管线复现结果的调整兰德指数为 0.83–0.94，与正确对齐结果为 −0.004）。纠正对齐后，在任一版本下通过 0.72 阈值的 195 条语句中有 26 条发生变化。这些结果是对一个流程及另外 16 个流程的测量，并非对公众参与本身的否定。它们说明"谁在书写宪法"有定量答案，设计者所占的作者份额应与公众份额一并报告。

---

## 1 Introduction

Constitutional AI trains a model against a written list of principles instead of per-example human labels (Bai et al., 2022). That moves a large part of a model's normative content into a short natural-language document. The question in our title follows directly: who should write it? Gabriel (2020) argued that the central normative problem of alignment is not finding the true moral theory but finding a *fair process* for choosing principles, and he named social choice as one candidate. Conitzer et al. (2024) make the same case for constitutional AI specifically, asking which humans should give input, in what form, and how it should be aggregated. Developers now publish their constitutions (e.g., Anthropic's January 2026 revision, credited to named internal authors; Anthropic, 2026). Developers and civil-society groups have also run public-input experiments: CCAI with about 1,000 U.S. adults on Polis (Huang et al., 2024), OpenAI's Democratic Inputs grants (OpenAI, 2024), and collective-dialogue policy drafting (Konya et al., 2023).

These experiments are usually described by *who participated*, for example "a representative sample of U.S. adults." Participation is not authorship, though. In a Polis-style process the text that ends up in a constitution passes through several hands. Organizers write seed statements. A small fraction of participants write most new statements. A sampling and routing process decides which participants see which statements. A designer then chooses how to cluster participants into "opinion groups," how to estimate agreement from sparse votes, what threshold or size to use, and which aggregation rule to apply. Each of these is an authorship channel. If the designer's degrees of freedom move the constitution as much as the public's votes do, then calling the result "the public's constitution" overstates the public's role, much as an unreported garden of forking paths overstates a statistical finding (Gelman & Loken, 2013; Steegen et al., 2016).

We narrow the open question "who should write the constitution?" to a measurable one: **in a deployed public-input process, how much of the resulting constitution is determined by the participants and how much by the designer, and how do the two compare as the process scales?** We answer it for CCAI, whose anonymized data and processing notebook are public, and replicate the main comparison on 16 public Polis conversations.

**Contributions.**

1. **An authorship-audit methodology** (Section 3) that splits a public-input constitution into four channels: agenda, sample, designer and pipeline integrity. Each channel comes with a concrete statistic: author concentration and exposure share; bootstrap "sample-swap" distance; multiverse "designer-swap" distance; and a two-way variance decomposition of statement inclusion over designs × samples. The code runs on any Polis export.
2. **Three formal results** (Section 4). (P1) A tight bound on how far a designer's choice of opinion groups can move a statement's minimum-group agreement. Applied to CCAI, it shows 145 of the 210 statements above the 0.72 agreement level are *partition-vulnerable*. (P2) An impossibility result: consensus filtering cannot guarantee JR when 1/2 < τ < 1 − 1/k, for *any* partition, and for any τ > 0 if the dissenting faction is itself an opinion group of size at least N/k. (P3) A sufficient condition under which a consensus core completed by greedy coverage satisfies JR. This is a constructive design that keeps bridging and guarantees JR.
3. **Measurements on CCAI** (Section 6). (a) The designer multiverse produces substantially different constitutions from the same votes. (b) Sampling noise at n ≈ 1,000 is even larger, mainly because each statement is seen by only about 100 voters. (c) Proportional rules on raw approval counts hand roughly a third of the constitution to organizer seeds through exposure. (d) The released group labels are consistent with a row-misalignment bug, which made the "consensus in both groups" filter run on effectively random groups.
4. **A cross-process regularity with a falsifiable prediction** (Section 6.6). Across 16 Polis conversations, designer influence relative to sampling influence rises with participant count (ρ = 0.95, p = 2×10⁻⁸; the effect survives controlling for the number of statements). We predict that in larger public-input processes the designer, not the sample, becomes the main determinant of the constitution. Section 7.3 gives a pre-registrable test protocol.

We do not claim that public input is illegitimate or that CCAI's final constitution was wrong. That constitution also passed through manual merging steps that are not in the released data. Our claim is narrower: authorship can be measured, and current reporting leaves out the designer's share.

## 2 Related Work

**Constitutions and public input for AI.** Constitutional AI (Bai et al., 2022) put a written principle list at the centre of alignment training. CCAI (Huang et al., 2024) sourced a "public" constitution through Polis from a sample of about 1,000 U.S. adults recruited to be representative by age, gender, income and geography. It retained statements that passed a consensus threshold within both Polis opinion groups, then removed duplicates and merged statements by hand toward the 58-principle length of Anthropic's 2023 constitution (Anthropic research post on CCAI; Huang et al., 2024). OpenAI's Democratic Inputs program funded ten teams to prototype such processes and reported, among other lessons, that public views shifted often (OpenAI, 2024). Konya et al. (2023) combined collective dialogues with LLM-drafted policy clauses and reported 75–81% support for the resulting guidelines. Inverse Constitutional AI (Findeis et al., 2025) goes the other way, compressing preference data into principles. That makes the extraction procedure, rather than any participant, the proximate author. Our audit is complementary to all of these: it does not propose a new elicitation process but measures how authorship is distributed within one.

**Social choice for alignment and pluralism.** Conitzer et al. (2024) argue that social choice should guide how diverse feedback, including constitutional principles, is aggregated. Sorensen et al. (2024) distinguish Overton, steerable and distributional pluralism. PRISM (Kirk et al., 2024) collects individualized, demographically annotated feedback and shows that "whose feedback" changes conclusions. Generative social choice (Fish et al., 2024) proves representation guarantees for statement slates when an LLM can answer preference queries. Bakker et al. (2022) and Tessler et al. (2024) train models to write group-consensus statements. None of these quantify the designer's discretion over the aggregation pipeline on real public-input data, which is our focus.

**Proportional committee selection.** Choosing k statements from approval votes is approval-based committee (ABC) voting (Lackner & Skowron, 2023). Justified representation (JR) and extended JR (Aziz et al., 2017) formalize the idea that any sufficiently large, cohesive group deserves representation. Proportional Approval Voting and the Method of Equal Shares (Peters & Skowron, 2020) satisfy strong proportionality axioms. Halpern et al. (2023) show that with incomplete votes, as on Polis, non-adaptive elicitation needs impractically many votes for representation guarantees, and they propose adaptive routing. Our Proposition 2 is in the spirit of the classical result that many popular ABC rules fail JR. It is specific to the *consensus filters* (bridging rules; Ovadya & Thorburn, 2023) used in Polis-style constitution writing, and Proposition 3 gives a hybrid repair.

**Who is in the room.** Flanigan et al. (2021) design sortition algorithms that balance representativeness with fair selection probabilities. They address *sample* authorship at the recruitment stage. Our bootstrap channel measures the downstream consequence: how much the output changes when the sample changes.

**Researcher degrees of freedom.** Multiverse analysis (Steegen et al., 2016), specification-curve analysis (Simonsohn et al., 2020) and the garden of forking paths (Gelman & Loken, 2013) show that defensible analytic choices can drive conclusions. We carry this idea into AI governance, treating the designer of a public-input pipeline as a researcher whose forking paths are part of the authorship of the outcome.

## 3 The Authorship Audit

**Data model.** There are N participants and M statements. V ∈ {+1, −1, 0, ∅}^{N×M} records agree, disagree, pass and *not seen*. A constitution is a set W of k statements, or the set passing a threshold τ. A *design* d is a full specification of the pipeline from V to W. A *sample* b is a multiset of participants drawn from the population the process claims to represent.

**Channel A: agenda.** We report (i) the share of candidate statements written by the organizer account, separating *seed* statements posted before any participant statement from later ones; (ii) the concentration of participant authorship (share of participants who wrote any retained statement; share written by the top 10% of authors); (iii) exposure, meaning the mean number of voters who saw a statement, and its correlation with submission order; and (iv) for each design, the share of the constitution drawn from seeds and from the organizer account, compared with their share of the pool.

**Channel B: sample.** We resample participants with replacement (B bootstrap draws of size n) and rerun each design. The *sample-swap distance* of design d is the mean Jaccard distance 1 − |W∩W′|/|W∪W′| between constitutions from two independent samples under d. Varying n traces how sample authorship shrinks with scale.

**Channel C: designer.** We fix a sample and vary the design over a declared multiverse 𝒟. The *designer-swap distance* is the mean Jaccard distance between constitutions from two designs on the same sample. For a design-free summary, we arrange inclusion indicators I[d, b, s] ∈ {0, 1} in a D × B × M array and decompose their total variance (summed over statements) into a designer main effect, a sample main effect and an interaction, as in two-way ANOVA. Swap distances are easier to interpret. The decomposition shows how much of each kind of variation is *systematic*.

**Channel D: pipeline integrity.** We re-execute the released processing code (or a faithful replica), compare its intermediate outputs (here, opinion-group labels) with the released ones, and check those outputs against a null distribution (random relabelings).

**The multiverse used here.** Every design picks k = 58 statements, the size of Anthropic's 2023 constitution, which CCAI targeted. It is built from the knobs a careful designer reading Huang et al. (2024) could defensibly choose:

* *Opinion groups*: two PCA conventions found in Polis practice. Participants are embedded either in statement space with columns centred ("stmt"), or with the PCA fitted on the transposed matrix as in the CCAI notebook ("part"). Missing votes are imputed as 0 or as the column mean, and k-means uses K ∈ {2, 3, 4}. That gives 12 groupings.
* *Agreement estimator*: agree/seen (CCAI notebook), agree/(agree+disagree), or Laplace-smoothed (agree+1)/(seen+2) (the Polis group-aware-consensus form).
* *Consensus rule*: maximize the minimum group agreement ("consmin"), or the product of group agreements ("gac").

That yields 12 × 3 × 2 = **72 consensus designs**, all of which implement "statements with high agreement in every opinion group." We add 7 non-consensus designs for comparison: approval voting by rate (AV, 3 estimators), sequential PAV, the Method of Equal Shares (MES, completed by approval count), greedy Chamberlin–Courant coverage (CC), and the hybrid of Proposition 3 (41-statement consensus core plus CC completion). That makes **79 designs** in total. The proportional rules run on raw approvals with unseen treated as non-approval, the usual default. Section 6.4 shows why this matters.

**Representation metrics.** On observed approvals (agree votes) we compute a JR ratio: the size of the largest group of voters who approve nothing in W but share an approved statement outside W, divided by N/k. JR is violated when the ratio is at least 1. We also compute each participant's endorsement rate of the selected statements they saw, and the gap in mean endorsement between the two opinion groups.

## 4 Theory

Throughout, we assume full information for each statement s within the voters considered. Let p(s) be the fraction of those voters who agree with s. A partition 𝒫 = {G₁, …, G_g} of voters is chosen by the designer. Let a_j(s) be the agreement rate within G_j. A τ-consensus filter admits s iff min_j a_j(s) ≥ τ.

**Proposition 1 (partition leverage).** Fix s with overall agreement p, and let α ∈ (0, 1]. (i) Every group G with |G| ≥ αN has a_G(s) ≥ max{0, 1 − (1 − p)/α}. (ii) Every partition has min_j a_j(s) ≤ p. (iii) If (1 − p)N ≤ αN, there is a group of size ⌈αN⌉ whose agreement equals 1 − (1 − p)N/⌈αN⌉, which approaches the bound in (i) up to rounding.

*Proof.* (i) G contains at most (1 − p)N non-agreers, so a_G(s) ≥ (|G| − (1 − p)N)/|G| ≥ 1 − (1 − p)/α. (ii) p = Σ_j (|G_j|/N) a_j(s) is a convex combination, so its minimum term is at most p. (iii) Put all (1 − p)N non-agreers in G and fill it with agreers up to ⌈αN⌉. ∎

*Corollary.* For one statement considered alone, a designer who may choose any partition whose smallest group has share at least α can place s's minimum-group score anywhere in roughly [1 − (1 − p)/α, p]. With threshold τ, s is **partition-robust** only if p > 1 − α(1 − τ). With CCAI's minority-group share α = 0.382 and τ = 0.72, robustness requires p > 0.893. One partition cannot push every statement to its floor at once, so the bound measures exposure to the choice of partition, not the effect of a realized choice. Section 6.3 measures the realized effect.

**Proposition 2 (consensus filters versus JR).** Let k ≥ 3 and 1/2 < τ < 1 − 1/k. For every partition 𝒫 of N voters into g groups with N ≥ g/(1 − τ − 1/k), there is an approval profile under which *every* committee of size at most k made only of τ-admissible statements violates JR. In addition, for every τ > 0: if a group of 𝒫 has at least N/k members who jointly approve a statement that no other group approves, and none of them approves an admissible statement, then every admissible committee violates JR.

*Proof.* For each group G_j, put ⌊(1 − τ)|G_j|⌋ of its members in a faction F. Then |F| ≥ Σ_j ((1 − τ)|G_j| − 1) = (1 − τ)N − g ≥ N/k by the assumption on N. Members of F approve only a statement c_F. All other voters approve every statement in a set C with |C| ≥ k and do not approve c_F. Each c ∈ C has a_j(c) ≥ 1 − ⌊(1 − τ)|G_j|⌋/|G_j| ≥ τ in every group, so it is admissible. c_F has a_j(c_F) ≤ 1 − τ < τ (because τ > 1/2), so it is not admissible. Any committee W ⊆ C leaves F, a cohesive group of size at least N/k that jointly approves c_F ∉ W, without any approved member: a JR violation. For the second claim, note that a statement approved only by members of G_j has agreement 0 in every other group and is inadmissible for every τ > 0. So if G_j has at least N/k members, they jointly approve such a statement, and none of them approves an admissible statement, then every admissible committee violates JR. ∎

Proposition 2 makes precise a tension that bridging-based processes acknowledge informally (Ovadya & Thorburn, 2023). A τ-consensus filter gives every group, and any large enough faction spread thinly across groups, a veto over statements that only it endorses. Proportional representation requires the opposite: a cohesive group of size N/k is owed a seat. With k = 58 the condition τ < 1 − 1/k ≈ 0.983 covers every threshold used in practice.

**Proposition 3 (consensus core plus coverage completion).** Let W₀ be any k₀ < k statements, and let U be the set of voters who approve nothing in W₀. Complete W₀ to size k by greedy coverage, each step adding the statement approved by the most voters not yet covered. If |U| < (k − k₀ + 1)N/k, the final committee W satisfies JR.

*Proof.* Suppose JR fails. Then there is a set X of voters with |X| ≥ N/k, all approving some c ∉ W and none approving anything in W. So X ⊆ U and X stays uncovered throughout. At each of the k − k₀ greedy steps, c was available and would have covered at least |X| ≥ N/k new voters, so the chosen statement covered at least N/k new voters. These newly covered sets are disjoint subsets of U \ X. Hence |U| ≥ |X| + (k − k₀)N/k ≥ (k − k₀ + 1)N/k, a contradiction. ∎

Proposition 3 gives a concrete recipe: choose the consensus core size k₀ so that its uncovered set satisfies the inequality, then fill the remaining seats by coverage. This keeps bridging content and guarantees JR. On sparse data, "approve" should be read as "observed to approve." Halpern et al. (2023) explain why guarantees on incomplete votes require adaptive elicitation.

## 5 Data and Experimental Setup

**CCAI.** We use the anonymized export released by the CCAI authors (repository `saffronh/ccai`): `clean_votes.csv` (1,002 participants after the authors' removal of low-quality participants and those with 25 or fewer votes) and `clean_comments.csv` (275 moderated-in statements). All numbers below are computed by our code from these files unless marked otherwise.

**Polis openData.** We use every conversation in the Computational Democracy Project's openData repository with a participants-votes export, at least 100 participants with at least 7 votes each (the Polis convention), and at least 40 moderated-in statements seen by at least 10 voters. That gives 16 conversations (n = 100–2,957). The 6,289-participant `march-on` conversation is excluded for runtime (n > 4,000 after filtering), two conversations fail the size filters, and one has no participants-votes export. For each conversation the constitution size is k = clip(round(0.2M), 10, 58).

**Procedure.** For CCAI we run all 79 designs on the full data; B = 40 bootstrap samples of size n = 1,002 × 79 designs; a sample-size curve with B = 25 bootstrap samples at each n ∈ {100, 200, 400, 700}; threshold-mode analyses at τ ∈ {0.60, …, 0.80}; a pipeline-integrity audit; and the representation audit. For openData we use B = 20 per conversation. Total CPU time is about 15 minutes on 4 cores (single-threaded BLAS). Code, seeds and the run order are in `code/README.md`, and every output is under `results/`.

## 6 Results

### 6.1 Agenda authorship (CCAI)

The released data contain 34,563 observed votes, so the participant × statement matrix is 12.5% observed. Of these votes, 78.9% are agree, 13.2% disagree and 7.9% pass. Of the 275 retained statements, **61 (22.2%) come from the organizer account** (author-id 0). **21 of these were posted before any participant statement.** That matches the "21 seed statements" in the CCAI blog post. The other 40 were posted from the same account during the conversation, and the data do not tell us whether they were organizer-written or re-entered participant text. The remaining 214 were written by **156 participants (15.6% of the 1,002)**. The top 10% of these authors wrote 28.5% of participant-authored statements, and one participant wrote 16.

Exposure is very unequal and follows submission order. **Seed statements were seen by 386 voters on average, against 104 for all other statements.** The Spearman correlation between submission order and number of voters is −0.83 (p ≈ 10⁻⁷¹). Every rate-based rule therefore estimates most statements' support from about 100 votes, while seeds are estimated from almost 400. Sections 6.4 and 6.5 show that this asymmetry drives both sampling noise and some rules' preference for organizer-written text.

### 6.2 Pipeline integrity: the released opinion groups look random

The CCAI notebook re-clusters participants after cleaning and stores the labels in `clean_votes.csv` (group 0 = 619 participants, group 1 = 383). The group-consensus columns of `clean_comments.csv` are computed from these labels, and these columns are the input for building the constitution. We found the following.

* The rows of `clean_votes.csv` are not sorted by participant id. The notebook sorts the vote matrix by participant before PCA and k-means, then assigns the resulting labels *positionally* back to the unsorted data frame.
* Replicating the notebook's clustering on the sorted matrix and assigning positionally reproduces the released labels with **adjusted Rand index (ARI) 0.83–0.94** across five k-means seeds. Assigning them to the correct participants gives **ARI −0.004** with the released labels.
* Under the released labels the two groups agree at almost identical rates (0.788 and 0.790). Their mean per-statement absolute agreement difference is **0.0704**, inside the null distribution from 200 random relabelings (mean 0.0678, sd 0.0035, max 0.0792; Figure 5). Re-aligned labels give rates of 0.826 and 0.728 and a separation of **0.1065**, more than ten null standard deviations above the null mean.

The released "consensus in both groups" scores were therefore computed on groups that carry no opinion structure. The filter reduces almost to a majoritarian threshold: at τ = 0.72, the released-label set has Jaccard 0.895 with the plain overall-agreement set, against 0.838 for the re-aligned set. Re-aligning changes the τ = 0.72 set (agree/seen estimator) from 188 to 176 statements, with Jaccard 0.867: 19 statements drop out and 7 enter. These 26 changes are out of 195 statements that pass under either version. Under a fixed size of 58, the two top-58 lists have Jaccard 0.73. Examples of statements that pass only under the released labels are in `results/ccai_misalignment_examples.csv`. One is "The AI should protect humans," with group agreement 0.88/0.70 under re-aligned labels.

We stress two limits. First, the final CCAI constitution was produced from these scores by further manual de-duplication and merging, which are not in the release. We therefore cannot say whether, or how, the published constitution would change. Second, our replica uses scikit-learn k-means. Imperfect ARI (below 1) is expected from k-means initialization and library-version differences. Still, the near-zero ARI under correct alignment and the null-like separation under the released labels are hard to explain otherwise. We report this so that it can be checked. Channel D exists to catch exactly this kind of error, which affects authorship but is invisible in the output.

### 6.3 Designer authorship: same votes, different constitutions

**Threshold mode.** At τ = 0.72, the 36 technical variants of the CCAI recipe (12 groupings × 3 estimators) admit **between 96 and 220 statements**. Across those 36 variants plus the released and re-aligned label sets, **233 statements pass in at least one variant and only 56 pass in all.** The estimator alone matters a great deal. Holding the grouping fixed, switching among the three estimators (for example, counting passes against agreement as the CCAI notebook does, or excluding them) changes the size of the admitted set by 42–77 statements.

**Fixed size (k = 58).** Figure 1 shows pairwise Jaccard similarity among the 79 designs on the full data. Within the 72 consensus designs the **mean pairwise Jaccard is 0.569 (minimum 0.275)**. **Only 10 statements are in all 72 constitutions, while 136 are in at least one.** Over all 79 designs the mean is 0.545 (minimum 0.105), 7 statements are always included, and 162 are ever included. Changing k does not remove the effect: the mean consensus-family Jaccard is 0.50 at k = 30, 0.57 at k = 58 and 0.68 at k = 100.

**Partition leverage (Proposition 1).** Of the 210 statements whose overall agreement exceeds 0.72, only **65 are partition-robust** at α = 0.382 (overall agreement above 0.893). The other **145 could fall below 0.72** under some admissible two-group split.

**Which knobs matter.** Table 1 changes one knob at a time around a CCAI-like baseline (participant-centred PCA, zero imputation, K = 2, agree/seen, consmin) on the same bootstrap samples.

*Table 1. Mean Jaccard distance between 58-statement constitutions when one knob is changed (same sample), and when the sample is redrawn (same design). CCAI, B = 40 bootstraps, n = 1,002. Source: `results/ccai_results.json` (`decomp_by_knob`).*

| Knob varied | designer-swap | sample-swap |
|---|---|---|
| min vs product consensus | 0.246 | 0.508 |
| PCA centring convention | 0.301 | 0.528 |
| missing-vote imputation | 0.308 | 0.531 |
| agreement estimator (3) | 0.353 | 0.541 |
| number of groups K (2/3/4) | 0.381 | 0.564 |
| rule family (consmin, AV, PAV, MES) | 0.560 | 0.390 |
| all 72 consensus designs | 0.441 | 0.565 |
| all 79 designs | 0.466 | 0.551 |

A single technical knob that a reader of the paper could not infer from its text, such as the PCA centring convention or the imputation rule, changes about 30% of the constitution. The choice of rule family changes 56%.

### 6.4 Sample authorship: at n ≈ 1,000, the sample still writes much of the text

Redrawing participants moves the constitution *more* than switching between consensus designs (Table 1, Figure 2). The mean sample-swap distance is 0.565 against a designer-swap distance of 0.441. The mean Jaccard similarity between a bootstrap constitution and the full-data constitution under the same consensus design is 0.54. Both distances fall as n grows. From n = 100 to n = 1,002, consensus-family sample-swap falls from 0.729 to 0.565 and designer-swap from 0.556 to 0.441. The ratio stays between 0.71 and 0.78 throughout. In the two-way variance decomposition for the consensus multiverse at n = 1,002, the designer main effect is 12.9% of inclusion variance, the sample main effect 35.5% and the interaction 51.6%. The large interaction means that *which* statements a given design picks depends on which sample it sees.

The reason is exposure, not headcount. CCAI's statements are highly agreeable (78.9% of votes are agree), and most are seen by only about 100 voters. The top 58 is therefore chosen from a dense band of near-tied estimates, and the noise in each estimate (standard error about 0.04 at p = 0.8 and 100 votes) is comparable to the gaps between them. Exposure also decides which *rule families* are stable. Rules that score raw approval counts (PAV, MES, CC) are much less sensitive to resampling: bootstrap-to-full Jaccard is 0.85, 0.73 and 0.82, against about 0.59 for rate-based rules. They buy that stability by favouring high-exposure statements. The Spearman correlation between a statement's number of voters and its inclusion probability is 0.80 for proportional rules, against 0.30 for consensus rules.

Curve fits of the form a + b·n^(−c) to the five-point sample-size curve suggest that designer-swap levels off at about 0.23–0.30 while sample-swap keeps falling slowly (fitted exponent about 0.1). Under these fits, designer influence would overtake sampling influence on CCAI-like data only at tens of thousands of participants. With five points and three parameters, this extrapolation is weakly identified, and we report it only as a direction.

### 6.5 Representation and agenda effects by rule family

Figure 3 summarizes the audit.

* **JR on observed approvals.** No design violates JR on CCAI. The largest JR ratio is 0.35, from a consensus design. Sparsity is the reason: few groups of 17 or more voters (N/k) are *observed* to share an approval while approving nothing in W, and only 0.3–1.2% of participants have no endorsed statement selected under any design. Across the 16 Polis conversations, though, **consensus designs violate JR in 5 conversations** (all five Austrian climate conversations, in 65–97% of consensus designs), **and AV designs violate it in the same 5.** **PAV, MES, CC and the hybrid never do** (`results/opendata_decomposition.csv`). This matches Proposition 2's prediction that consensus filtering offers no JR guarantee, and Proposition 3's guarantee for the hybrid.
* **Agenda capture through exposure.** Seeds are 7.6% of the CCAI pool. They make up **31.0–34.5% of the constitutions chosen by PAV, MES and CC and 27.6% of the hybrid's**, against 3.4–12.1% (mean 8.4%) for consensus designs. Organizer-account statements make up 45–47% under proportional rules. On sparse, order-dependent Polis data, "proportional" rules on raw counts therefore shift authorship toward the organizers. The cause is that seeds were shown to everyone, not that participants preferred them.
* **Group parity.** Measured with the re-aligned opinion groups, consensus designs keep the two groups' endorsement rates within 0.0003–0.022 of each other, and AV within 0.015–0.025. Proportional designs and the hybrid produce gaps of 0.16–0.17, mostly because they include high-exposure, more divisive seed statements. With the PCA groups from the statement-centred convention, the gaps are 0.03–0.06 for consensus and 0.21 for proportional designs.

No rule wins on every axis. Consensus filters give the best group parity and the least agenda capture but have no JR guarantee and do fail JR on real data. Count-based proportional rules satisfy JR but hand the agenda to whoever controlled exposure. A defensible pipeline needs exposure-normalized approvals, adaptive routing (Halpern et al., 2023), or both, before proportional guarantees mean what they claim.

### 6.6 Replication across 16 Polis conversations: designer share rises with scale

Figure 4 and `results/opendata_summary.txt` report the designer-swap and sample-swap distances per conversation.

* Median consensus-family designer-swap distance is 0.409 (range 0.275–0.539). Median sample-swap distance is 0.414 (range 0.264–0.522). Over the full 79-design multiverse the medians are 0.441 and 0.408, and designer-swap is significantly larger (Wilcoxon p = 0.008; larger in 12 of 16 conversations).
* **The designer/sample ratio rises steeply with the number of participants**: Spearman ρ = 0.95 (p = 1.8 × 10⁻⁸) for the consensus family and 0.93 for the full multiverse. In every conversation with at least 900 participants, designer influence exceeds sampling influence (ratio 1.03–1.14). The ratio is below 1 in the smallest conversations (0.85 at n = 100).
* Conversation size is correlated with the number of statements (ρ = 0.81) and inversely with vote density. In an OLS regression of log ratio on log n and log M, log n stays significant (coefficient 0.059, t = 3.3) and log M does not (t = 1.25). With 16 conversations we cannot fully separate n from density (ρ(ratio, density) = −0.86).
* In the variance decomposition, the median designer main effect is 25% of inclusion variance and the median sample main effect is 20% (consensus family). For the full multiverse they are 31% and 17%.

CCAI is an outlier, with ratio 0.78 at n = 1,002. This fits Section 6.4: its unusually high agreement and low per-statement exposure inflate sampling noise. CCAI averages 126 voters per statement (104 for non-seed statements), against a median of 192 across the 16 conversations.

## 7 Discussion

### 7.1 What this says about "who should write the constitution"

The normative question cannot be answered by naming a constituency, because a constituency does not write. Authorship passes through channels, and they can be measured. Three implications follow from the measurements.

*First, the designer is a co-author whose share is currently unreported, and it grows with scale.* In processes with roughly 1,000 or more participants, a different but equally defensible pipeline moves the constitution at least as much as a different sample of the public does. Several of those choices, such as PCA centring, imputation and whether passes count against agreement, cannot be recovered from a paper's prose. A public-input constitution that reports only "N representative participants" omits the channel that, at scale, dominates.

*Second, small processes are sample-authored.* At the sizes typical of deliberative experiments (100–500 participants), and in high-agreement processes such as CCAI even at 1,000, which statements make the cut depends heavily on who happened to vote on what. Sortition improves who is in the room (Flanigan et al., 2021). Exposure routing determines how many of them speak to each statement, and that matters at least as much.

*Third, the designer's choice of rule is a choice of whose agenda wins.* Consensus filters protect between-group parity at the cost of minority representation (Proposition 2; JR violations in 5 of 16 conversations). Count-based proportional rules guarantee JR but convert organizer control of exposure into organizer authorship. Neither is neutral, and the choice should be justified and disclosed, not presented as "the public decided."

### 7.2 Recommendations (an authorship statement)

We suggest that public-input constitution efforts publish an **authorship statement** with five items. (1) Seed share and exposure distribution: the share of the pool written by organizers and the ratio of seed to non-seed exposure. (2) The pre-registered pipeline, together with the designer-swap distance over a declared multiverse of reasonable alternatives. (3) The bootstrap sample-swap distance at the realized n. (4) A representation audit (JR and EJR on observed approvals, and group parity). (5) An integrity check of intermediate artifacts against a null, such as the group-separation test of Section 6.2, which takes a few lines of code. Proposition 3 gives one pipeline that combines bridging with a JR guarantee, provided approvals are exposure-normalized or elicited adaptively.

### 7.3 Falsifiable predictions and a test protocol

**P1 (scale).** In public-input processes on Polis-like platforms with fixed k, the designer-swap/sample-swap ratio over a pre-declared multiverse increases with n holding exposure fixed. It exceeds 1 for n ≳ 1,000 when the median statement receives at least about 150 votes. *Falsified* if, across at least 20 new conversations with pre-registered multiverses, the rank correlation between n and the ratio, partialling out per-statement exposure, is not positive.

**P2 (exposure capture).** Under count-based proportional rules, the seed share of the constitution exceeds the seed share of the pool by a factor that grows with the seed-to-non-seed exposure ratio. With exposure-normalized approvals (for example, approval rate among viewers, or adaptive routing), the factor falls to about 1. *Test:* run a Polis conversation with randomized seed exposure (seeds shown to a random 50% of participants) and compare.

**P3 (veto).** In conversations with at least three opinion groups, τ-consensus filters violate JR on observed approvals more often than in two-group conversations, at matched k and density. *Test:* the openData corpus plus new conversations, with pre-registered K selection.

**Protocol.** For a new process: freeze the multiverse 𝒟 before data collection (the 79-design space here is a template); collect data with exposure logging; compute the four channels with the released code; report swap distances with bootstrap CIs; publish the authorship statement with the constitution.

## 8 Limitations

* **One deployed AI-constitution process.** CCAI is, to our knowledge, the only AI-constitution public-input process with released vote-level data. The 16 openData conversations are civic deliberations, not AI constitutions. We use them only to test the scale regularity, which does not depend on the topic.
* **The multiverse is ours.** Designer-swap distance depends on which designs count as "defensible." We restricted the main multiverse to designs that fit CCAI's published description, and we report single-knob effects so readers can apply their own weights. A narrower multiverse would give smaller designer shares. Equal weighting of designs is also a choice.
* **Statements, not principles.** CCAI's final constitution came from manual de-duplication, merging and rewording of the selected statements into a smaller set of principles. That curation is a further, unmeasured designer channel. Our Jaccard distances on statements may overstate (if merging collapses near-duplicates) or understate (if rewording adds content) differences between final principle lists.
* **Bootstrap as a model of resampling the public.** Bootstrapping the realized sample approximates drawing a new sample from the same recruitment process. It cannot capture differences in recruitment, platform, framing, or which statements would have been *written* by a different sample. The agenda would differ too, so our sample-swap distances probably *understate* true sample authorship.
* **Sparse votes.** JR and endorsement metrics treat "not seen" as "not approved." This understates cohesive groups and is why no design violates JR on CCAI. The exposure process is non-random (seeds first, later statements to fewer voters), so every rate estimate inherits it.
* **The integrity finding.** Our replica uses a recent scikit-learn rather than the authors' environment, and the released notebook truncates part of the cluster-assignment step, so our reconstruction of it is an inference. The evidence (ARI 0.83–0.94 with the positional replica versus −0.004 aligned, and null-like separation) is strong but not proof. The authors are best placed to confirm or refute it, and it may not affect the final constitution.
* **Small cross-conversation sample.** Sixteen conversations, with n, M and density correlated. The scale regularity should be treated as a hypothesis with strong preliminary support, not an established law. One large conversation was excluded for runtime.
* **No model training.** We measure the constitution text, not the behaviour of models trained on it. How much statement-level differences change trained-model behaviour is open.

## 9 Conclusion

"Who should write the constitution?" has an empirical precondition: knowing who *does* write it. We introduced an authorship audit that separates agenda, sample, designer and pipeline-integrity channels. We proved that consensus filtering and proportional representation conflict, and gave a repair that keeps bridging while guaranteeing JR. Applied to the only public AI-constitution vote data and to 16 civic Polis processes, the audit gives four results. Defensible designer choices change about 30–56% of a 58-statement constitution. At CCAI's scale the sample changes even more. Designer influence overtakes sampling influence as processes grow. And the choice of rule decides whether organizers or minorities gain. The audit also surfaced a likely data-alignment error in the released CCAI pipeline. Public input can confer legitimacy only if the designer's authorship is measured and disclosed along with the public's.

---

## Figures

* **Figure 1** (`results/fig1_multiverse_jaccard.png`): Pairwise Jaccard similarity of 58-statement CCAI constitutions under 79 pipelines on identical votes.
* **Figure 2** (`results/fig2_swap_vs_n.png`): Designer-swap versus sample-swap distance as a function of participant count (CCAI bootstrap).
* **Figure 3** (`results/fig3_representation_audit.png`): JR ratio, seed share and between-group endorsement gap by rule family (CCAI).
* **Figure 4** (`results/fig4_opendata.png`): Designer versus sample influence across 16 Polis conversations, and the ratio against n.
* **Figure 5** (`results/fig5_group_separation.png`): Group-separation statistic for released CCAI labels, re-aligned labels and 200 random relabelings.

## Data and Code Availability

All code is in `code/` (run order in `code/README.md`), and all outputs are in `results/`. Every number in Sections 5–6 is computed by this code from the public CCAI release and the Polis openData repository (CC BY 4.0, The Computational Democracy Project), except figures attributed to cited sources (e.g., 21 seed statements, recruitment details, the 58-principle target, 75–81% support in Konya et al.). Run with `OMP_NUM_THREADS=1`.

## References

Anthropic (2023). *Collective Constitutional AI: Aligning a Language Model with Public Input* (research blog post). https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input

Anthropic (2026). *Claude's new constitution.* January 21, 2026. https://www.anthropic.com/news/claude-new-constitution

Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017). Justified representation in approval-based committee voting. *Social Choice and Welfare*, 48(2), 461–485. https://doi.org/10.1007/s00355-016-1019-3

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073.

Bakker, M. A., et al. (2022). Fine-tuning language models to find agreement among humans with diverse preferences. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2211.15006.

Computational Democracy Project. *openData: Data exports from "open data" Polis conversations.* GitHub repository, https://github.com/compdemocracy/openData (accessed November 2026; CC BY 4.0).

Conitzer, V., Freedman, R., Heitzig, J., Holliday, W. H., Jacobs, B. M., Lambert, N., Mossé, M., Pacuit, E., Russell, S., Schoelkopf, H., Tewolde, E., & Zwicker, W. S. (2024). Position: Social choice should guide AI alignment in dealing with diverse human feedback. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 9346–9360.

Findeis, A., Kaufmann, T., Hüllermeier, E., Albanie, S., & Mullins, R. (2025). Inverse Constitutional AI: Compressing preferences into principles. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2406.06560.

Fish, S., Gölz, P., Parkes, D. C., Procaccia, A. D., Rusak, G., Shapira, I., & Wüthrich, M. (2024). Generative social choice. *Proceedings of the 25th ACM Conference on Economics and Computation (EC 2024)*. arXiv:2309.01291.

Flanigan, B., Gölz, P., Gupta, A., Hennig, B., & Procaccia, A. D. (2021). Fair algorithms for selecting citizens' assemblies. *Nature*, 596(7873), 548–552. https://doi.org/10.1038/s41586-021-03788-6

Gabriel, I. (2020). Artificial intelligence, values, and alignment. *Minds and Machines*. https://doi.org/10.1007/s11023-020-09539-2

Gelman, A., & Loken, E. (2013). The garden of forking paths: Why multiple comparisons can be a problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis was posited ahead of time. Unpublished manuscript, Department of Statistics, Columbia University.

Halpern, D., Kehne, G., Procaccia, A. D., Tucker-Foltz, J., & Wüthrich, M. (2023). Representation with incomplete votes. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(5), 5657–5664. arXiv:2211.15608.

Huang, S., Siddarth, D., Lovitt, L., Liao, T. I., Durmus, E., Tamkin, A., & Ganguli, D. (2024). Collective Constitutional AI: Aligning a language model with public input. *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)*. https://doi.org/10.1145/3630106.3658979. Data and notebook: https://github.com/saffronh/ccai

Kirk, H. R., et al. (2024). The PRISM alignment dataset: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track*. arXiv:2404.16019.

Konya, A., Schirch, L., Irwin, C., & Ovadya, A. (2023). Democratic policy development using collective dialogues and AI. arXiv:2311.02242.

Lackner, M., & Skowron, P. (2023). *Multi-Winner Voting with Approval Preferences.* SpringerBriefs in Intelligent Systems. Springer. https://doi.org/10.1007/978-3-031-09016-5

OpenAI (2024). *Democratic inputs to AI grant program: lessons learned and implementation plans.* January 16, 2024. https://openai.com/index/democratic-inputs-to-ai-grant-program-update

Ovadya, A., & Thorburn, L. (2023). Bridging systems: Open problems for countering destructive divisiveness across ranking, recommenders, and governance. Knight First Amendment Institute; arXiv:2301.09976.

Peters, D., & Skowron, P. (2020). Proportionality and the limits of welfarism. *Proceedings of the 21st ACM Conference on Economics and Computation (EC '20)*, 793–794. Full version arXiv:1911.11747.

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4, 1208–1214. https://doi.org/10.1038/s41562-020-0912-z

Small, C., Bjorkegren, M., Erkkilä, T., Shaw, L., & Megill, C. (2021). Polis: Scaling deliberation by mapping high dimensional opinion spaces. *RECERCA. Revista de Pensament i Anàlisi*, 26(2), 1–26. https://doi.org/10.6035/recerca.5516

Sorensen, T., et al. (2024). Position: A roadmap to pluralistic alignment. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 46280–46302.

Steegen, S., Tuerlinckx, F., Gelman, A., & Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science*, 11(5), 702–712. https://doi.org/10.1177/1745691616658637

Tessler, M. H., Bakker, M. A., et al. (2024). AI can help humans find common ground in democratic deliberation. *Science*, 386(6719). https://doi.org/10.1126/science.adq2852
