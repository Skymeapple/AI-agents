# How Small Can a Distilled Model Be? Metric Dependence, a Loose Capacity Bound, and What Soft Labels Buy in a Zipfian Skill Model

**Target venue:** NeurIPS 2027 (Main Track)

**Revision note (round 2).** This version replaces the round-1 headline "distillation saves data and compute, not capacity". New long-training experiments with 5 seeds (E6) contradict that headline at non-floor widths. Changes are listed point by point in `reviews/response_r1.md`.

---

## Abstract

How small can a distilled model be and still keep frontier capabilities? We study this in a skill-frequency model. A capability is a set of M discrete skills used with Zipfian frequency p_i ∝ i^-(1+α). We find that the answer depends on three things: how the evaluation weights skills, how much of the domain the teacher covers, and which resource limits the student. Our results, stated with their limits:

**(P1) Capacity bound.** A standard counting / rate–distortion bound holds for every training procedure: bN ≥ Σ_{i∈S} R_i(ε). We calibrate it in our testbed. Reaching the "learned" threshold needs about 17 bits per skill under a Gaussian test channel (a heuristic estimate, not a proven lower bound). Trained students need 6–8 bits per parameter under per-channel post-training quantisation, which is 231–344 bits per learned skill. The bound therefore holds with up to about 16× slack. It is valid but loose, so it does not by itself fix the achievable student size.

**(P2) Aggregate vs. tail metrics, finite domain.** We derive the error ratio for finite M: E_S/E_T = (r^α − κ^α)/(1 − κ^α), where r is the skill-compression factor and κ = K_T/M is the fraction of the domain the teacher covers. The round-1 asymptotic r^α is the κ → 0 limit, and it underpredicted our testbed by 1.7–3×. Across 80 student pairs the finite-M formula cuts the median absolute log-error from 0.235 to 0.088. Teacher coverage changes the answer a lot. At α = 0.5, doubling the aggregate error permits a 4× cut in skills when κ → 0, but only 1.5× when κ = 0.6. We apply the general (non-power-law) form to the public PopQA popularity data. If usage follows Wikipedia page views, knowing only the top 10% of entities gives 84% view-weighted accuracy but only 10% entity-balanced accuracy.

**(P3) Where distillation helps.** Under an assumed threshold model, distillation multiplies the per-skill sample efficiency. Empirically, the data needed to reach the same skill count is about 8× smaller with oracle soft labels (7.7–8.9× for K = 10–70; seed-resampled fit 6.5–9.5). The threshold-model exponent is misspecified, however: the point-wise γ falls from 10.5 to 6.5 as D grows. The gain grows with label entropy under both a KL-based and an accuracy-based skill criterion.

**Capacity regime, the main empirical correction.** At unlimited data, 100k steps and 5 seeds, soft labels let students retain more skills than hard labels at non-floor widths. With oracle soft labels, retention is 1.17× (h = 32) and 1.41× (h = 48). With a real trained teacher it is 1.19× and 1.27×. All 95% bootstrap CIs exclude zero. At the smallest converged widths the advantage is 1.0× (h = 8) and 1.13–1.17× (h = 16, CI includes zero). With 2.5× more training (250k steps, 2 seeds per arm), the h = 48 advantage of teacher-distilled over hard-label students narrows from 1.27× to 1.10× (125.5 vs 114.5 skills). At h = 128, where capacity is not binding, hard-label students rise from 49% of the oracle-soft skill count (40k steps) to 84% (200k steps). The data are consistent with a slowly closing, largely optimisation-limited gap, but our runs ended before it closed. Distilled students never learned a skill their teacher lacked: none of the 1,249 learned student–skill cases across 20 teacher-distilled students fell outside the teacher's set. Within the slack that P1 leaves, the supervision signal therefore changes how many skills a fixed architecture reaches. We could not show that this advantage disappears with training, so we do not claim that distillation leaves achievable capacity unchanged.

Everything is measured in a CPU testbed (400 skills, MLPs with 404–76k parameters, 222 training runs). We give LLM predictions and a protocol that needs only inference on public checkpoints. We did not run them.

## 中文摘要

蒸馏后的小模型能小到什么程度、同时仍保有前沿能力？我们在一个"技能—频率"模型中研究这个问题：能力是 M 个离散技能，使用频率服从 Zipf 分布 p_i ∝ i^-(1+α)。答案取决于三件事：评测如何给技能加权、教师覆盖了领域的多大比例、学生受哪种资源限制。以下是我们的结论及其适用范围。

**(P1) 容量界。** 一个标准的计数/率失真界对任何训练方式都成立：bN ≥ Σ_{i∈S} R_i(ε)。我们在实验环境中对其做了标定。按高斯测试信道估计，达到"学会"阈值约需每技能 17 比特（这是启发式估计，不是已证明的下界）。训练好的学生在逐通道训练后量化下需要每参数 6–8 比特，即每个已学会技能 231–344 比特。可见该界成立，但最多松约 16 倍：它有效但不紧，单靠它定不出可达的学生规模。

**(P2) 总体与长尾指标（有限领域）。** 我们推导了有限 M 下的误差比：E_S/E_T = (r^α − κ^α)/(1 − κ^α)，其中 r 是技能压缩倍数，κ = K_T/M 是教师覆盖的领域比例。第一轮使用的渐近式 r^α 是 κ → 0 的极限，在本实验环境中低估了 1.7–3 倍。在 80 对学生上，有限 M 公式把对数误差的中位绝对值从 0.235 降到 0.088。教师覆盖率对结论影响很大：α = 0.5 时，总体误差翻倍在 κ → 0 时允许砍掉 4 倍技能，在 κ = 0.6 时只允许 1.5 倍。我们把一般（非幂律）形式用于公开的 PopQA 流行度数据：若使用频率与维基百科浏览量成正比，只掌握前 10% 的实体即可得到 84% 的浏览量加权准确率，但实体均衡准确率只有 10%。

**(P3) 蒸馏在哪里有用。** 在假定的阈值模型下，蒸馏提高每个技能的样本效率。实测中，用 oracle 软标签达到相同技能数所需的数据约少 8 倍（K = 10–70 时为 7.7–8.9 倍；按种子重采样的拟合区间为 6.5–9.5）。但阈值模型的指数设定有误：逐点 γ 随 D 增大从 10.5 降到 6.5。无论用基于 KL 还是基于准确率的技能判据，该增益都随标签熵增大。

**容量区间：主要的实证修正。** 在数据无限、训练 10 万步、5 个种子的条件下，在非地板宽度上，软标签使学生比硬标签保留更多技能。用 oracle 软标签时为 1.17 倍（h = 32）和 1.41 倍（h = 48）；用真实训练的教师时为 1.19 倍和 1.27 倍；所有 95% bootstrap 置信区间都不含零。在已收敛的最小宽度上，优势为 1.0 倍（h = 8）和 1.13–1.17 倍（h = 16，置信区间含零）。训练量增加到 2.5 倍（25 万步，每组 2 个种子）后，h = 48 时教师蒸馏学生相对硬标签学生的优势从 1.27 倍缩小到 1.10 倍（125.5 对 114.5 个技能）。在容量并非瓶颈的 h = 128 上，硬标签学生学到的技能数从 oracle 软标签学生的 49%（4 万步）升至 84%（20 万步）。这些结果与一个缓慢收窄、主要受优化限制的差距相符，但在我们的运行结束时，差距尚未消失。 被蒸馏的学生从未学会教师不会的技能：在 20 个教师蒸馏学生的 1,249 个已学会技能中，没有一个落在教师的技能集合之外。也就是说，在 P1 留下的余量之内，监督信号会改变固定架构能学到的技能数。我们未能证明这一优势会随训练消失，因此不主张"蒸馏不改变可达容量"。

所有结果都来自 CPU 实验环境（400 个技能，404–7.6 万参数的 MLP，共 222 次训练）。我们给出了针对大语言模型的预测，以及只需在公开模型上做推理的检验方案，但没有实际运行。

---

## 1 Introduction

Small distilled models now post benchmark numbers that recently required much larger systems.
- Gemma 2 trained its 2B and 9B models with knowledge distillation and reports them as competitive with models 2–3× larger [Gemma Team 2024].
- DeepSeek-R1's 1.5B distilled model reports 28.9% on AIME 2024 and 83.9% on MATH-500 [DeepSeek-AI 2025].
- Xiao et al. [2025] fit a "densing law": the parameter count needed for a fixed benchmark score halves roughly every 3.5 months.

Read naively, these results suggest that frontier capability compresses with no visible floor.

Other work points the other way:
- QA accuracy degrades for rare entities [Kandpal et al. 2023; Mallen et al. 2023].
- Language models store about 2 bits of knowledge per parameter [Allen-Zhu & Li 2025].
- Compressed networks keep top-line accuracy while losing the long tail [Hooker et al. 2019]. This loss is concentrated on memorised, atypical examples [Dam et al. 2023], as Feldman's [2019] long-tail theory of memorisation predicts.
- Distillation can hurt rare subgroups [Lukasik et al. 2021] and sparsely represented domains [Zhou et al. 2024].

We make the relation between these strands precise in a minimal model and test it in a controlled setting. We do not measure frontier LLMs; we have no GPU and no model access in this environment.

**Sub-question.** Suppose capabilities are discrete skills with power-law usage. What limits the number of skills a student of size N retains when a teacher trains it? Which of these limits does distillation lift? How does the answer depend on the evaluation metric and on the teacher's coverage?

**Contributions** (stated at their actual strength):

1. **A finite-domain aggregate–tail law (Prop. 2) and its validation.** The formula E_S/E_T = (r^α − κ^α)/(1 − κ^α) shows that the admissible compression depends on the teacher's coverage κ as well as on α. In 80 student pairs it fits much better than the asymptotic r^α. A general-p version applied to PopQA popularity data shows that the gap is large for a real knowledge distribution. The algebra is elementary. The contribution is the correction and its test, not the mathematics.
2. **A calibrated, procedure-independent capacity bound (Prop. 1).** The bound is textbook. We supply what round 1 lacked: an estimate of the per-skill rate in our testbed and a measurement of the students' effective bits per parameter. Together they show the bound holds with about 16× slack.
3. **Controlled measurements of what soft labels buy.** We measure:
   - an exponent-free data multiplier of about 8 at matched skill count;
   - its growth with label entropy under two skill criteria;
   - a capacity-regime advantage of 1.17–1.41× more skills at 100k steps (5 seeds, oracle and real teacher), narrowing to 1.10× at 250k steps but not closing within our budget.

   This last result contradicts the round-1 slogan, and we report it as such.
4. **Testable LLM predictions and a protocol** (Section 7) that need only inference on public checkpoints. These are untested.

We do not claim a mechanistic derivation of the distillation scaling law of Busbridge et al. [2025]. Section 4.3 explains why our threshold model does not determine its exponent.

## 2 Related Work

**Compression and the long tail.** Hooker et al. [2019] show that pruned and quantised image classifiers keep top-line accuracy while disproportionately losing performance on a small set of "pruning identified exemplars" from the under-represented long tail. Dam et al. [2023] connect the mismatches between full and compressed models to memorised training examples. Feldman [2019] proves that memorisation is needed for near-optimal generalisation when subpopulation frequencies are long-tailed, and Feldman & Zhang [2020] give empirical support. Our Proposition 2 is a frequency-weighted, closed-form version of the head–tail effect these papers show empirically. It adds the dependence on teacher coverage κ, which they do not address. Our capacity bound is the information-theoretic counterpart of Feldman's memorisation argument: tail skills must be stored, and storage costs bits. We do not claim the head–tail divergence itself as new.

**Knowledge distillation.** Hinton et al. [2015] introduced training on softened teacher outputs. Menon et al. [2021] explain the benefit as variance reduction under Bayes class-probabilities. Phuong & Lampert [2019] analyse linear students. Distillation often fails to make students match teachers [Stanton et al. 2021]. Large teacher–student gaps can hurt [Cho & Hariharan 2019; Mirzadeh et al. 2020]. Distillation can harm low-sample subgroups [Lukasik et al. 2021]. BalDistill [Zhou et al. 2024] rebalances head and tail domains in sequence-level LLM distillation, which is direct evidence that tail domains are where distilled LLMs lose. Our oracle soft labels are exactly the Bayes-probability setting of Menon et al. We additionally use a real trained teacher.

**Distillation and scaling laws.** Busbridge et al. [2025] find that distillation beats supervised learning only up to a compute level that scales with student size. Our threshold model gives a qualitatively similar crossover, but we cannot match their exponent (Section 4.3). Our capacity-regime results (Section 6.5) also show a soft-label advantage that does not disappear within our budget. Sardana et al. [2024] motivate training small models long. Xiao et al. [2025] measure capability density on aggregate benchmarks. Proposition 2 implies that density measured this way depends on the benchmark's frequency weighting and on the reference model's coverage.

**Quantised skills.** The quantization model [Michaud et al. 2023] and Hutter [2021] derive power-law learning from Zipf-distributed discrete features. Our Proposition 3 is that model with a supervision-dependent threshold. Allen-Zhu & Li [2025] measure about 2 bits/parameter for factual knowledge. Our students need 6–8 bits/parameter under naive post-training quantisation. That figure is an upper bound on the information stored, not a capacity measurement.

**Long-tail knowledge in LLMs.** Kandpal et al. [2023] and Mallen et al. [2023] link QA accuracy to entity popularity or document counts. We use the PopQA popularity data [Mallen et al. 2023] in Section 6.6.

**What remains open.** Prior work shows the head–tail effect empirically. As far as we found, it does not give (i) a closed-form error-ratio law that includes teacher coverage and is validated against controlled runs, or (ii) a controlled separation of data-, optimisation- and capacity-regime effects of soft labels with a real teacher. That is the scope of this paper.

## 3 Model

**Skills.** Skills i = 1…M are invoked with probability p_i = i^-(1+α)/Z_M. Skill i has parameter θ_i, and a model's competence on it is s_i ∈ [0,1].

**Evaluation functionals.**
- The aggregate score is A = Σ_i p_i s_i, with aggregate error E = 1 − A.
- The tail (uniform) score on a set of size M' is U_{M'} = M'^-1 Σ_{i≤M'} s_i.

**Students and teachers.**
- A student has N parameters representable to b bits.
- Supervision consists of hard labels sampled from the true distribution, or soft labels. Soft labels are either the true conditional distribution ("oracle") or a trained teacher's predictive distribution.
- D is the number of distinct training inputs.
- Skill i is *learned* if s_i ≥ τ = 0.75.
- K_T is the teacher's learned count, and κ = K_T/M is its coverage.

## 4 Theory

### 4.1 Proposition 1: a procedure-independent capacity bound, and how loose it is

**Proposition 1.** Let θ_1…θ_M be independent. Let ω be the student's parameters after any (possibly randomised) training procedure using any teacher, data or queries, with ω taking at most 2^{bN} values. Let θ̂_i = g_i(ω), and let d_i have rate–distortion function R_i. If E d_i(θ_i, θ̂_i) ≤ ε for all i ∈ S, then bN ≥ Σ_{i∈S} R_i(ε).

*Proof.*
1. I(θ_S; ω) ≤ H(ω) ≤ bN.
2. By the chain rule and independence, I(θ_S; ω) = Σ_k I(θ_{i_k}; ω, θ_{<k}) ≥ Σ_k I(θ_{i_k}; ω).
3. By data processing, I(θ_i; ω) ≥ I(θ_i; θ̂_i).
4. By the definition of R_i, I(θ_i; θ̂_i) ≥ R_i(ε). ∎

This is a standard counting argument; we do not claim it as new. It is useful here because it holds for every teaching signal (soft labels, features, synthetic data, reasoning traces). It is only informative, however, once b and R_i are calibrated. Round 1 did not calibrate either. We now do (Section 6.1, `analyze_r2.py`, `P1_calibration` and `quantisation`).

- *Rate per skill.* A skill W_t ∈ R^{4×4} has 12 identifiable degrees of freedom: softmax is invariant to a shared shift across classes, and each degree has variance 0.75 after centring. The Gaussian rate–distortion function at relative MSE ε is 6 log2(1/ε) bits. We pass Gaussian test-channel reconstructions through the testbed's own KL-based score. Half of the skills cross the learned threshold at ε* = 0.136, which corresponds to **R ≈ 17 bits per skill**. This is a calibration, not a proof. The bound needs R under the KL-threshold distortion, and the Gaussian channel is one scheme that reaches it, so 17 bits estimates rather than lower-bounds that rate.
- *Effective bits per parameter.* We quantise trained students uniformly with one scale per output channel. We take the smallest bit width that keeps ≥ 90% of learned skills. This gives b = 6–8 for all 30 E6 models (h = 32 and 48, all three supervision types), or **231–344 bits per learned skill** (median 273). Post-training quantisation gives only an upper bound on the bits a network needs.
- *Consequence.* The students store skills at most about 16× less efficiently than the rate estimate (median 273 / 17). P1 is therefore a valid floor, but in this testbed it sits far below what gradient-trained MLPs achieve. Within this slack, architecture, optimiser and supervision signal can all change the achievable skill count. Section 6.5 shows that the supervision signal does.

### 4.2 Proposition 2: aggregate vs. tail retention in a finite domain

Assume top-K retention: a model that has learned K skills has learned exactly skills 1…K, all-or-none. Then E(K) = Σ_{K<i≤M} p_i.

**Proposition 2 (finite M).** Let the teacher learn the top K_T skills, with κ = K_T/M, and let the student learn the top K_S = K_T/r. Using the integral approximation Σ_{K<i≤M} i^-(1+α) ≈ (K^-α − M^-α)/α, which has relative error O(1/K):

  E_S/E_T = (r^α − κ^α)/(1 − κ^α) · (1 + O(1/K_S)),  U_{K_T}(student) = 1/r.

At error ratio 1+δ, the admissible skill compression is r* = [(1+δ)(1 − κ^α) + κ^α]^{1/α}.

*Proof.* E(K) ∝ K^-α − M^-α. Divide, and substitute K_S = K_T/r and M = K_T/κ. ∎

*Remarks.*
1. As κ → 0, r* → (1+δ)^{1/α}. This is the round-1 statement, now explicitly scoped to a teacher that covers a vanishing fraction of the domain. For κ > 0, r* is always smaller: truncation removes tail mass, so each dropped skill is a larger share of the remaining error.
2. For a general frequency vector p (not a power law), E_S/E_T = Σ_{i>K_S} p_i / Σ_{i>K_T} p_i. Section 6.6 applies this to PopQA.
3. Top-K retention is an assumption. In our testbed the teacher has learned 89% of its top-241 skills, its highest learned rank is 386, and its error is 0.0122 against 0.0114 for an exact prefix. Students are 75–91% prefix-consistent. Section 6.2 tests the proposition with these imperfections present.

**Table 1.** Admissible skill compression r* at aggregate-error ratio 1+δ, by teacher coverage κ (`summary_r2.json`, `prop2_table_finite`). The κ = 0 rows are the round-1 asymptotic values.

| α | κ | δ = 0.25 | δ = 0.5 | δ = 1 | δ = 2 |
|---|---|---|---|---|---|
| 0.25 | 0 | 2.44 | 5.06 | 16.0 | 81.0 |
| 0.25 | 0.1 | 1.51 | 2.21 | 4.27 | 12.4 |
| 0.25 | 0.6 | 1.13 | 1.26 | 1.57 | 2.36 |
| 0.5 | 0 | 1.56 | 2.25 | 4.00 | 9.00 |
| 0.5 | 0.1 | 1.37 | 1.80 | 2.84 | 5.61 |
| 0.5 | 0.6 | 1.12 | 1.24 | 1.50 | 2.10 |
| 1.0 | 0 | 1.25 | 1.50 | 2.00 | 3.00 |
| 1.0 | 0.1 | 1.23 | 1.45 | 1.90 | 2.80 |
| 1.0 | 0.6 | 1.10 | 1.20 | 1.40 | 1.80 |

Coverage interacts strongly with α. A teacher that already covers most of a narrow domain leaves little room for "free" compression under an aggregate metric. A teacher that covers a small fraction of a large, flat domain leaves a lot. Both cases are plausible for LLMs, depending on the domain.

### 4.3 Proposition 3: a threshold model of when soft labels help

Following Michaud et al. [2023], assume the following. Skills are learned in frequency order. Skill i becomes learnable once D·p_i ≥ n_m, with n_soft = n_hard/γ. Then K_D(D; m) = (D/(n_m Z))^{1/(1+α)}, and K* = min(K_cap, K_D, K_opt). (a) In the data-limited regime, soft labels multiply K by γ^{1/(1+α)}. (b) The hard-label student reaches K_cap at D* = n_hard Z K_cap^{1+α}. The proof is substitution, and the content lies entirely in the assumptions. We test those assumptions rather than the algebra.

What the model does **not** give:
- It does not give a relation D* ∝ N^{1+α} unless K_cap ∝ N. Our capacity estimates do not support proportionality: parameters per learned skill range from 81 (h = 8) to 35 (h = 48), and about 55 at h = 128.
- It does not give an exponent comparable with Busbridge et al. [2025], whose law is in compute and teacher cross-entropy, not in skill counts.
- It does not predict whether hard and soft supervision reach the same K at the capacity limit. P1 bounds both from above but is about 16× loose, so equal achievable capacity is an *additional* assumption. Round 1 made that assumption implicitly; Section 6.5 tests it.

## 5 Experimental Testbed

All numbers in Sections 5–6 come from code we wrote and ran on a shared 4-core CPU.
- Raw outputs: `results/all_runs.json`, `results/e5_runs.json` and `results/e6_runs_{A,B,C}.json`.
- Summaries: `results/summary.json` (round 1) and `results/summary_r2.json` (round 2).
- Total: 222 training runs.

**World.** M = 400 skills, α = 0.5 unless stated. Input z ~ N(0, I_4), and p(y|t,z) = softmax(β W_t z) over 4 classes, with β = 2 by default. There are two skill families:
- *random*: W_t i.i.d. N(0,1), which is incompressible;
- *lowrank*: W_t = Σ_{k≤2} c_{tk} B_k with shared bases.

**Students.** A 2-hidden-layer ReLU MLP of width h. Its input is z and a fixed, non-learned random 32-dimensional code for t. All parameters are shared across skills. Students range from 404 parameters (h = 8) to 21,764 (h = 128); the teacher has 76,292 (h = 256).

**Supervision.**
- *hard*: sampled labels.
- *soft (oracle)*: the true p(·|t,z).
- *soft (teacher)*: the predictive distribution of an h = 256 network trained for 30k steps on fresh hard labels (K_T = 241, A = 0.988, κ = 0.60).

Training uses Adam (lr 10^-2, cosine schedule) with batch 512. With finite D, we early-stop on a same-label-type validation set.

**Metrics.**
- s_t = clip(1 − KL_t/KL0_t, 0, 1) over 200 fixed test inputs per skill, where KL0_t is the KL of the uniform predictor. A skill is learned if s_t > 0.75.
- Because KL0 depends on β, we also report an accuracy-based criterion in E3: argmax agreement with the Bayes classifier ≥ 0.9.

**Experiments.**
- E1–E5 are the round-1 experiments:
  - E1: capacity regime, oracle soft labels, 45k steps.
  - E2: data regime, 10k steps.
  - E3: β sweep.
  - E4: α sweep.
  - E5: 45k steps at h ≤ 16, 2 seeds.
- **E6 is new.** It uses unlimited data (fresh samples every step) and world seed 0, the teacher's world. Seeds vary initialisation and the data stream.
  - E6-A: h ∈ {32, 48}, 100k steps, hard / oracle-soft / teacher-soft, 5 seeds each.
  - E6-B: h = 16 at 45k steps (hard and oracle-soft with 5 seeds, teacher-soft with 2); teacher-distilled students at h = 8 and 24 (2 seeds); h = 128 at 40k steps (hard and oracle-soft, 2 seeds).
  - E6-C: h = 48 at 250k steps (hard and teacher-soft, 2 seeds); h = 128 hard at 200k steps (2 seeds).
- We log K every 10k steps. These checkpoints fall inside a cosine schedule, so they are not equivalent to shorter runs.
- Statistics: 95% bootstrap CIs (10,000 resamples) on differences of means, Welch t-tests and Mann–Whitney tests. With 5 seeds per arm, the smallest attainable two-sided Mann–Whitney p is 0.008.

## 6 Results

### 6.1 Capacity: frequency-ordered learning, and P1's slack

In E1 (random family, oracle soft labels, 45k steps), K grows from 5.0 (404 parameters) to 110.0 (4,324 parameters). Learning is frequency-ordered: the Spearman correlation between rank and score is −0.46, −0.65, −0.82 and −0.93 for h = 8–48.

Parameters per learned skill are not constant:
- 81 at h = 8;
- 39 at h = 48 after 45k steps, and 35 after 100k steps with oracle soft labels (E6);
- about 55 at h = 128, where oracle-soft students learn 391–395 of the 400 skills in 40k steps.

Round 1 quoted "about 40 parameters per skill". That figure was a single unconverged point, and we no longer use it as a constant. At equal parameter count, the low-rank family fits 2.8–3.8× more skills: 61.5 vs 20.0 at 932 parameters, and 216.5 vs 56.5 at 2,372.

*P1 calibration (Section 4.1).*
- Rate estimate: R ≈ 17 bits per skill.
- Effective precision: all 30 E6-A models need 6–8 bits per parameter (per-channel quantisation, ≥ 90% of skills kept), which is 231–344 bits per learned skill.
- By supervision type: hard-label models mostly need 6 bits (h = 48) or 6–7 bits (h = 32); soft-label models mostly need 7 bits.
- Bits per learned skill at h = 48 are about 295 for hard and about 244 for oracle soft, because soft-label students store more skills in the same parameters.

The bound holds with at most about 16× slack. P1 cannot be violated, so this is a measurement of looseness, not a test that could fail.

### 6.2 Aggregate metrics hide tail loss, and the finite-M law predicts by how much

The round-1 evidence, that A ≈ Σ_{i≤K} p_i (MAE 0.016), mostly confirms frequency ordering, as the reviewer noted. The operative claim is the error ratio, which we now test directly. Figure 6 (`results/fig6_prop2_finiteM.png`) compares predicted with observed E_S/E_T over 80 ordered pairs. The pairs are capacity-regime students at the same α with K ≥ 3, plus every student against the trained teacher.

| predictor | median abs log-error | median signed log-error |
|---|---|---|
| asymptotic r^α (round 1) | 0.235 | −0.235 (underpredicts by about 21%) |
| finite-M (Prop. 2) | 0.088 | +0.058 (overpredicts by about 6%) |

The finite-M prediction is closer in 78% of pairs. Its small overprediction has two sources. Students earn partial credit on skills just below threshold, and the teacher's learned set is not an exact prefix, which inflates E_T relative to top-K.

**Table 2.** Students *distilled from the trained teacher* (E6, unlimited data), compared with that teacher (`E6_teacher_table`). Round 1 compared oracle-trained students with an independently trained teacher; this table replaces it. Ranges are over seeds.

| width | params | compression | steps | seeds | K | aggregate A | observed E_S/E_T | r^α | finite-M | teacher skills retained | skills outside teacher's set |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 4,324 | 17.6× | 100k | 5 | 107–116 | 0.965–0.970 | 2.45–2.85 | 1.44–1.50 | 2.97–3.24 | 44–48% | 0 |
| 32 | 2,372 | 32.2× | 100k | 5 | 57–67 | 0.941–0.950 | 4.11–4.83 | 1.90–2.06 | 5.01–5.72 | 24–28% | 0 |
| 24 | 1,588 | 48.0× | 45k | 2 | 38–39 | 0.919–0.920 | 6.54–6.69 | 2.49–2.52 | 7.64–7.79 | 16% | 0 |
| 16 | 932 | 81.9× | 45k | 2 | 20–21 | 0.865–0.879 | 9.96–11.06 | 3.39–3.47 | 11.7–12.0 | 8–9% | 0 |
| 8 | 404 | 188.8× | 45k | 2 | 4–5 | 0.743–0.759 | 19.8–21.1 | 6.9–7.8 | 27.6–31.2 | 2% | 0 |

Two observations follow.
- A 32× smaller distilled student is within 3.8–4.7 aggregate points of the teacher, yet retains only a quarter of the teacher's skills. The finite-M law reproduces this error ratio to within about 20%. The round-1 asymptotic formula misses it by more than 2×.
- None of the 998 learned student–skill cases in Table 2 falls outside the teacher's learned set. The count is 1,249 when the two 250k-step teacher-distilled students of E6-C are included. Distillation did not transfer any skill the teacher lacked.

### 6.3 Data regime: soft labels as a data multiplier, and γ is not a constant

E2 (h = 128, 10k steps, 2 seeds) gives the following skill counts, soft vs hard: 33.5 vs 7.0 at D = 10^4, 68.0 vs 16.5 at 3·10^4, and 181.0 vs 77.0 at 3·10^5.

**γ is not constant in the threshold form.** The point-wise γ = (K_soft/K_hard)^{1.5} falls with D: 10.5, 8.4, 6.5 and 3.6 at D = 10^4, 3·10^4, 10^5 and 3·10^5. The last value is depressed because soft students approach their 10k-step ceiling. With the exponent left free, the slopes differ: 0.64–0.82 for hard and 0.57–0.66 for soft (ranges over all 4,096 seed resamplings). The exponent of the threshold model is therefore misspecified for at least one method. The fixed-exponent fit gives a median γ of 7.8, with a 2.5–97.5% range of 6.5–9.5 over seed resamplings. This is a seed-resampling range, not a full confidence interval, because only 2 seeds are available per cell.

**An exponent-free multiplier is stable.** We interpolate each method's K(D) curve in log–log space and compute D_hard(K)/D_soft(K) at matched K. The ratio is 8.9, 7.7, 8.4 and 8.2 at K = 10, 20, 40 and 70. We therefore report the result as "soft labels reduce the data needed for a given skill count by about 8× at β = 2", not as a constant threshold multiplier.

*Real teacher.* At h = 128, students distilled from the trained teacher track the oracle-soft students up to D = 3·10^4 (60 vs 68 skills), then fall behind (138 vs 181 at 3·10^5). The teacher's coverage (241 skills) and its imperfect probabilities on rarer skills limit what it can transfer.

*Crossover.* Round 1 predicted D* from 10k-step ceilings, which are optimisation-limited, and never observed a crossover. We recompute D* = n_hard Z K_cap^{1.5} from the 100k-step E6 ceilings (`E6_capacity_Dstar`), giving D* ≈ 2.3·10^5 (h = 32) and 6.5·10^5 (h = 48). These are beyond the E2 grid for those widths. We no longer claim that the observed relative gaps confirm the prediction. At h = 16 the gap plateaued at 0.24 for D ≥ 3·10^4, and at h = 128 it was 1.0 at D = 10^3 and 3·10^3 (where K_hard = 0) and 0.57–0.79 for D ≥ 10^4.

### 6.4 The soft-label advantage grows with label entropy, under two skill criteria

E3 (h = 128, D = 10^4, 2 seeds) varies β. KL0 changes with β, so s_t is not comparable across β. We therefore also count skills by accuracy: argmax agreement with Bayes ≥ 0.9 (`E3_acc`).

| β | Bayes entropy (nats) | K_hard / K_soft (KL criterion) | implied γ, KL | K_hard / K_soft (accuracy criterion) | implied γ, accuracy |
|---|---|---|---|---|---|
| 8 | 0.14 | 9.0 / 12.5 | 1.6 | 4.5 / 7.5 | 2.2 |
| 4 | 0.27 | 9.5 / 19.0 | 2.8 | 2.0 / 12.5 | 15.6 |
| 2 | 0.51 | 7.0 / 33.5 | 10.5 | 1.5 / 16.0 | 34.8 |
| 1 | 0.84 | 2.0 / 41.5 | ≥ 95 | 0 / 20.5 | ≥ 93 (floor) |
| 0.5 | 1.14 | 0.5 / 43.0 | ≥ 280 (floor) | 0 / 18.0 | ≥ 76 (floor) |

The *direction* holds under both criteria: the advantage of soft labels grows as a sampled label discards more of the distribution. This agrees with the variance-reduction account [Menon et al. 2021]. The *magnitude* depends on the criterion: at β = 2 the implied γ is 10.5 under KL and 34.8 under accuracy. For β ≤ 1 the hard-label counts are at or near zero, so the implied γ values are floors. We do not report a single γ(entropy) curve.

### 6.5 Capacity regime: soft labels raise the achievable skill count within our budget

Round 1 concluded from E5 (h ≤ 16, 2 seeds) that distillation "does not raise the capacity ceiling". E6 tests this at non-floor widths with 5 seeds and a real teacher (Figure 7, `results/fig7_e6_ceiling.png`, where filled points are E6 means with seed ranges and open squares are the 10k-step E2 values for reference; data in `E6` and `E6_compare`).

| width (params) | steps | hard K | oracle-soft K | teacher-soft K | soft − hard [95% CI] | teacher − hard [95% CI] |
|---|---|---|---|---|---|---|
| 8 (404) | 45k | 5.0 (2 seeds, E5) | 5.0 (E5) | 4.5 | 0 | — |
| 16 (932) | 45k | 18.2 (14–23) | 21.2 (20–24) | 20.5 (2 seeds) | +3.0 [−0.2, 6.4], 1.16× | +2.3 [−0.8, 5.3], 1.13× |
| 32 (2,372) | 100k | 52.8 (50–55) | 61.8 (59–65) | 62.6 (57–67) | +9.0 [6.4, 11.6], 1.17×, MW p = 0.012 | +9.8 [6.0, 13.4], 1.19×, MW p = 0.012 |
| 48 (4,324) | 100k | 87.8 (77–97) | 123.8 (120–131) | 111.6 (107–116) | +36.0 [29.2, 43.0], 1.41×, MW p = 0.008 | +23.8 [17.4, 30.4], 1.27×, MW p = 0.008 |
| 128 (21,764) | 40k | 193.5 (183–204) | 393.0 (391–395) | — | +199.5, 2.03× (2 seeds) | — |
| 48 (4,324) | 250k | 114.5 (113–116) | — | 125.5 (121–130) | — | +11.0, 1.10× (2 seeds; MW p = 0.33) |
| 128 (21,764) | 200k | 331.5 (322–341) | 393.0 at 40k | — | hard reaches 84% of the 40k oracle-soft count | — |

What the data show:

1. **The advantage does not vanish at non-floor widths within 100k steps.** It is statistically clear at h = 32 and h = 48, both for oracle labels and for a real teacher. At h = 16 (5 seeds) the round-1 1.14× is reproduced as 1.16×, and the CI now includes zero because the hard-label runs vary widely (14–23).
2. **The gap narrows slowly with training.** Mean K at the 10k-step checkpoint vs the final value (hard / teacher-soft):
   - h = 32: 29.2 / 36.8 (ratio 0.79) → 52.8 / 62.6 (0.84);
   - h = 48: 39.0 / 56.0 (0.70) → 87.8 / 111.6 (0.79).

   At h = 48, hard students at 100k steps reach about what oracle-soft students had at 40k steps. Hard labels thus cost roughly a 2–2.5× step multiplier here.
3. **With more training the gap keeps narrowing.**
   - At h = 48 and 250k steps (2 seeds per arm), hard-label students reach 114.5 skills against 125.5 for teacher-distilled students. That is 1.10×, down from 1.27× at 100k steps, and not significant with 2 seeds.
   - Across budgets, the hard/teacher ratio at h = 48 rises monotonically: 0.70 (10k checkpoint), 0.79 (100k), 0.91 (250k).
   - At h = 128, oracle-soft students learn 98% of the world in 40k steps, so capacity is not binding. Hard-label students there rise from 193.5 skills at 40k steps to 331.5 at 200k steps, from 49% to 84% of the oracle-soft count.
   - In every case the hard-label curves are still rising when training ends (Figure 7).
4. **Teacher coverage caps distillation.** Teacher-distilled students never learn a skill outside the teacher's set. At h = 48 they fall below oracle-soft students (111.6 vs 123.8), because the teacher's distribution is imperfect on its rarer skills.

**Interpretation.** P1 bounds all methods by the same bit budget, but the bound is about 16× loose (Section 6.1). Nothing forces hard and soft supervision to reach the same achievable skill count for a fixed architecture, optimiser and step budget, and in our runs they do not. We cannot distinguish two explanations within our compute: (i) hard-label training would catch up given still more steps (a slow optimisation-limited frontier), or (ii) soft targets let gradient descent find more storage-efficient solutions at the same size, raising the achievable capacity. The extra-long runs lean towards (i): the hard/soft ratio rises monotonically with budget, and at h = 128 hard-label students recover most of the gap with 5× more steps. The quantisation data lean weakly towards (ii): at h = 48, soft-label students store more skills per bit. Neither is decisive. **We therefore withdraw the round-1 claim that distillation "saves data and compute, not capacity".** What survives:
- no teaching signal can exceed the P1 bit budget;
- at the smallest, converged widths the advantage is small (≤ 1.17×, not significant at h = 16 with 5 seeds);
- at larger widths, soft labels raise the skill count reached by a given budget, by 1.2–1.4× (oracle) or 1.2–1.3× (teacher) at 100k steps;
- this advantage shrinks with longer training (1.10× at 250k steps for h = 48), which is consistent with soft labels acting mainly as a step multiplier. We could not show that the gap reaches zero.

### 6.6 How large is the aggregate–tail gap for a real knowledge distribution? (PopQA, public data)

We cannot evaluate LLMs here. To check whether Proposition 2's effect is large in a real domain, we apply its general-p form to the PopQA popularity data [Mallen et al. 2023]. The data are 14,267 questions; s_pop is the monthly Wikipedia page views of the subject entity (file `data_ext/popqa/popQA.tsv`, analysis in `analyze_r2.py`, `popqa`).

- **Not a single power law.** Ranked by popularity, the local log–log slope is −0.74 over ranks 10–1,000, −1.96 over 1,000–5,000 and −3.31 over 5,000–12,000. The power-law form of Proposition 2 is therefore at best a local approximation. The general form needs no such assumption.
- **The head–tail gap is large.** Suppose usage is proportional to page views and a model knows exactly the top fraction f of entities. Then view-weighted accuracy vs entity-balanced accuracy is:
  - 38% vs 1% at f = 0.01;
  - 71% vs 5% at f = 0.05;
  - 84% vs 10% at f = 0.1;
  - 94% vs 20% at f = 0.2;
  - 99.2% vs 50% at f = 0.5.
- **Compression ratios diverge.** If a teacher knows the top 50% and a student the top 10% (a 5× skill compression), the view-weighted error ratio is 18.5, while entity-balanced accuracy falls 5×.

This is arithmetic on a public popularity vector under the top-K assumption, not a measurement of any model. It shows that the effect is not an artefact of our chosen α: in a real long-tail knowledge domain, usage-weighted and entity-balanced scores of the same model can differ by an order of magnitude.

## 7 Discussion

### 7.1 So how small?

In this model the answer has three parts.

1. **For the skills you must keep, size has a floor set by their information content (P1).** This floor is procedure-independent, but in our testbed it is about 16× below what trained networks achieve. In practice the binding constraint is how efficiently a given architecture and training signal store skills, not the information-theoretic floor itself.
2. **How many skills you may drop depends on the metric and on the teacher's coverage (P2).** With α = 0.5, doubling the aggregate error permits a 4× cut in skills when the teacher covers a negligible fraction of the domain, but only 1.5× at 60% coverage. Usage-weighted benchmarks report much larger "compression" than entity-balanced ones, by an order of magnitude for PopQA-like popularity. Compression and capability-density claims [Xiao et al. 2025] should therefore state the benchmark's weighting.
3. **What soft labels buy depends on the binding resource.**
   - When data is limited, about an 8× data saving at moderate entropy, growing with entropy.
   - When the step budget is limited, roughly a 2–2.5× step saving at h = 48.
   - In the capacity regime, a 1.2–1.4× higher skill count at 100k steps. This narrows to 1.10× at 250k steps (h = 48), and we could not show that it vanishes.
   - In all regimes, never a skill the teacher lacks.

### 7.2 Falsifiable predictions for LLMs and a protocol (untested)

- **L1 (frequency-stratified deficit).** The distilled-student vs teacher accuracy gap grows with entity rarity (PopQA popularity bins; Kandpal et al. document-count bins). *Falsified if* the gap is flat across bins.
- **L2 (coverage-dependent mismatch).** Compression ratios computed from usage-weighted benchmarks exceed those from entity-balanced benchmarks. Per Prop. 2 (general-p form with the domain's popularity vector), the excess shrinks as the teacher's coverage of the domain grows. *Falsified if* the two ratios agree.
- **L3 (bounded but non-zero capacity-regime gain).** At matched student size and long training, distilled students lead non-distilled ones on tail knowledge. The lead narrows with training budget but does not reverse, and it never extends to entities the teacher fails on. *Falsified if* distilled students answer tail items their teacher gets wrong at above-chance rates, or if the lead grows with budget.
- **L4 (entropy dependence).** The fixed-budget distillation advantage is larger in domains with higher teacher next-token entropy.

*Protocol (inference only).*
1. Evaluate teacher, distilled student and a same-size non-distilled model on all PopQA items, for example Gemma 2 27B/9B/2B [Gemma Team 2024].
2. Bin items by log-popularity. Fit a monotone trend test for L1.
3. Compute view-weighted and uniform accuracy, and compare compression ratios with the Prop. 2 general-p prediction from the observed per-item correctness of the teacher (L2).
4. For L3, cross-tabulate the student's correct answers against the teacher's.

We could not run this protocol: there is no model access and no GPU in this environment.

## 8 Limitations

- **Toy setting.** The skills are random bilinear-softmax maps routed by fixed codes. The students are 2-layer MLPs with at most 21,764 parameters, and the teacher has 76,292. Real capabilities are not cleanly discrete, are not independent, and compose.
- **The capacity question is unresolved.** E6 shows a soft-label advantage at 100k steps (h = 32, 48), which narrows to 1.10× at 250k steps (h = 48, 2 seeds) but had not closed when our runs ended. We cannot tell a slow optimisation frontier from a genuine difference in achievable capacity. Runs long enough to settle this at h ≥ 48 exceeded our compute budget.
- **P1 is loose and its calibration is heuristic.** The 17-bits-per-skill rate is computed for a Gaussian test channel mapped through our KL threshold; it is not a proven lower bound for that distortion. The 6–8 bits per parameter comes from naive post-training quantisation and is only an upper bound on the information a network needs. Quantisation-aware training could lower it.
- **P2 assumes top-K, all-or-none retention.** Real learned sets are 75–91% prefix-consistent. The finite-M law overpredicts the error ratio by a median of about 6%.
- **P3's exponent is misspecified.** We report an exponent-free data multiplier instead. The E2/E3 cells have 2 seeds, so their γ ranges are seed resamplings, not confidence intervals. E3 magnitudes depend on the skill criterion, and for β ≤ 1 they are floor-limited.
- **One teacher.** We study a single trained teacher (h = 256). Teacher-size effects [Cho & Hariharan 2019; Mirzadeh et al. 2020] are not studied.
- **Seeds.** The E6 capacity-regime comparisons at h = 16, 32 and 48 have 5 seeds per arm. The h = 8, 24 and 128 rows and the E1–E5 experiments have 2.
- **No LLM evidence.** The PopQA analysis is arithmetic on public popularity data, not a model evaluation. Every LLM statement is a prediction.

## 9 Conclusion

"How small can distillation compress frontier capability?" has a definite answer only once three things are fixed: the retained skill set, the metric's weighting, and the teacher's coverage.
- A procedure-independent bit bound sets a floor. In our testbed it is valid but about 16× loose.
- A finite-domain aggregate–tail law predicts how much compression an aggregate metric hides. It corrects a 2–3× error in our round-1 asymptotic formula, and applied to PopQA popularity it implies order-of-magnitude head–tail gaps.
- Soft labels act as a roughly 8× data multiplier and a roughly 2× step multiplier. At non-floor widths they also raise the skill count a given architecture reaches within a fixed budget: by 1.2–1.4× at 100k steps, narrowing to 1.10× at 250k steps. They never take a student beyond the teacher's own skill set.

The practical implication is that compression claims should be reported with frequency-stratified and coverage-aware evaluation. The open scientific question is whether the capacity-regime advantage of soft labels persists at convergence.

## Reproducibility

```
python3 code/run_all.py; python3 code/run_e5.py; python3 code/analyze.py      # round 1 (E1-E5)
python3 code/run_e6.py A; python3 code/run_e6.py B; python3 code/run_e6.py C   # round 2 (E6)
python3 code/analyze_r2.py                                                     # round-2 analyses -> results/summary_r2.json, fig6, fig7
```

- Every testbed number in Sections 5–6 is read from `results/summary.json`, `results/summary_r2.json` or the raw run files.
- The PopQA numbers use the public PopQA file [Mallen et al. 2023].
- Literature numbers in Section 1 come from the cited sources.

## References

- Allen-Zhu, Z., & Li, Y. (2025). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. *ICLR 2025*. arXiv:2404.05405.
- Busbridge, D., Shidani, A., Weers, F., Ramapuram, J., Littwin, E., & Webb, R. (2025). Distillation Scaling Laws. *ICML 2025*, PMLR 267:5977–6045. arXiv:2502.08606.
- Cho, J. H., & Hariharan, B. (2019). On the Efficacy of Knowledge Distillation. *ICCV 2019*, 4793–4801.
- Dam, H., Joseph, V., Bhaskara, A., Gopalakrishnan, G., Muralidharan, S., & Garland, M. (2023). Understanding the Effect of the Long Tail on Neural Network Compression. arXiv:2306.06238.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
- Elhage, N., et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
- Feldman, V. (2019). Does Learning Require Memorization? A Short Tale about a Long Tail. arXiv:1906.05271.
- Feldman, V., & Zhang, C. (2020). What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. *NeurIPS 2020*. arXiv:2008.03703.
- Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Hooker, S., Courville, A., Clark, G., Dauphin, Y., & Frome, A. (2019). What Do Compressed Deep Neural Networks Forget? arXiv:1911.05248.
- Hutter, M. (2021). Learning Curve Theory. arXiv:2102.04074.
- Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *ICML 2023*. arXiv:2211.08411.
- Lukasik, M., Bhojanapalli, S., Menon, A. K., & Kumar, S. (2021). Teacher's pet: understanding and mitigating biases in distillation. arXiv:2106.10494.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. *ACL 2023*. arXiv:2212.10511.
- Menon, A. K., Rawat, A. S., Reddi, S., Kim, S., & Kumar, S. (2021). A statistical perspective on distillation. *ICML 2021*, PMLR 139:7632–7642.
- Michaud, E. J., Liu, Z., Girit, U., & Tegmark, M. (2023). The Quantization Model of Neural Scaling. *NeurIPS 2023*. arXiv:2303.13506.
- Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020). Improved Knowledge Distillation via Teacher Assistant. *AAAI 2020*, 34(04), 5191–5198.
- Phuong, M., & Lampert, C. H. (2019). Towards Understanding Knowledge Distillation. *ICML 2019*, PMLR 97:5142–5151.
- Sardana, N., Portes, J., Doubov, S., & Frankle, J. (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. *ICML 2024*, PMLR 235:43445–43460.
- Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). Does Knowledge Distillation Really Work? *NeurIPS 2021*.
- Xiao, C., et al. (2025). Densing law of LLMs. *Nature Machine Intelligence*, 7, 1823–1833. arXiv:2412.04315.
- Zhou, Y., Zhu, J., Xu, P., Liu, X., Wang, X., Koutra, D., Ai, W., & Huang, F. (2024). Multi-Stage Balanced Distillation: Addressing Long-Tail Challenges in Sequence-Level Knowledge Distillation. *Findings of EMNLP 2024*, 3315–3333. arXiv:2406.13114.
