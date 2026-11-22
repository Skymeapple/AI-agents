# Capacity, Not Teaching, Bounds Compression: A Skill-Frequency Theory of How Far Distillation Can Shrink Models

**Target venue:** NeurIPS 2027 (Main Track)

---

## Abstract

How small can a distilled model be and still have frontier capabilities? We argue that the question has no single answer. The answer depends on (i) which capabilities are measured and how they are weighted, and (ii) which resource limits the student: its parameters, its training data, or its optimisation budget. We formalise this in a skill-frequency model in which a capability is a set of discrete skills used with Zipfian frequency p_i ∝ i^-(1+α). We prove three results. **(P1)** Any training procedure, including distillation from a perfect teacher, obeys an information-theoretic capacity bound: a student with N parameters storing b bits per parameter can retain a set S of independent skills to distortion ε only if bN ≥ Σ_{i∈S} R_i(ε), where R_i is the skill's rate–distortion function. **(P2)** Frequency-weighted ("aggregate") and uniform ("tail") benchmarks diverge under compression. If the student is allowed an aggregate error 1+δ times the teacher's, it can drop a factor r = (1+δ)^{1/α} of the teacher's skills, and the tail benchmark falls to 1/r. **(P3)** Distillation multiplies the per-skill sample efficiency by a factor γ. This extends the data-limited skill frontier by γ^{1/(1+α)} but leaves the capacity frontier unchanged, so the advantage of distillation disappears beyond a data budget D* ∝ K_cap^{1+α}. We test these predictions in a controlled CPU testbed: 400 stochastic skills, MLP students from 404 to 76k parameters, and 166 training runs. In the testbed, (a) skills are acquired in frequency order, and the number retained grows with parameters at about 40 parameters per incompressible skill. Compressible (low-rank) skills fit about 3× more per parameter. (b) Knowing only how many skills a student learned predicts its aggregate score to within 0.016 (mean absolute error). A student with 32× fewer parameters than the teacher comes within 5 points of the teacher's aggregate score but keeps only 22–25% of the teacher's skills. (c) Soft labels gave γ ≈ 8.4 at moderate label entropy, and the implied γ rose from 1.6 to over 90 as the Bayes entropy increased. (d) With long training, the soft-label advantage over hard labels shrank from 2.1× more skills learned (at D = 10^4) to 1.14× (h = 16) and 1.0× (h = 8) at unlimited data. So the question "how small?" comes down to how incompressible the retained skill set is. Distillation saves data and compute, not capacity. We give four falsifiable predictions for LLMs and a test protocol.

## 中文摘要

蒸馏后的小模型能小到什么程度，同时仍保有前沿能力？本文认为这个问题没有单一答案。答案取决于两点：(i) 衡量哪些能力、以何种权重衡量；(ii) 学生模型受哪种资源限制：参数量、训练数据，还是优化预算。我们用一个"技能—频率"模型来形式化这一点：能力被视为一组离散技能，各技能的使用频率服从 Zipf 分布 p_i ∝ i^-(1+α)。我们证明了三个结论。**(P1)** 任何训练方式（包括向完美教师蒸馏）都受信息论容量界约束：若参数量为 N、每参数 b 比特的学生要以失真 ε 保留一组独立技能 S，则必须满足 bN ≥ Σ_{i∈S} R_i(ε)，其中 R_i 为该技能的率失真函数。**(P2)** 压缩时，按频率加权的"总体"基准与均匀加权的"长尾"基准会分离：若允许学生的总体误差为教师的 1+δ 倍，它可以丢掉教师技能的 r = (1+δ)^{1/α} 倍，而长尾基准降至 1/r。**(P3)** 蒸馏使单个技能的样本效率提高 γ 倍。这使数据受限时的技能前沿扩大 γ^{1/(1+α)} 倍，但不改变容量前沿；因此当数据量超过 D* ∝ K_cap^{1+α} 后，蒸馏的优势消失。我们在可控的 CPU 实验环境中检验这些预测：400 个随机技能，参数量 404 到 7.6 万的 MLP 学生，共 166 次训练。在该环境中：(a) 技能按频率顺序习得；可保留的技能数随参数增长，约每 40 个参数对应一个不可压缩技能，而可压缩（低秩）技能每参数可多容纳约 3 倍。(b) 仅凭学生学会的技能数即可预测其总体得分，平均绝对误差 0.016。参数量比教师少 32 倍的学生，总体得分与教师相差不到 5 个百分点，却只保留了教师 22–25% 的技能。(c) 在中等标签熵下软标签的 γ ≈ 8.4；随贝叶斯熵增大，隐含的 γ 从 1.6 升至 90 以上。(d) 长时间训练下，软标签相对硬标签的优势从数据量 10^4 时多学 2.1 倍技能，缩小到数据无限时的 1.14 倍（h=16）和 1.0 倍（h=8）。因此，"能压缩到多小"取决于所需保留技能集合的不可压缩程度：蒸馏节省的是数据和算力，而不是容量。文末给出了四个针对大语言模型的可证伪预测及检验方案。

---

## 1 Introduction

Small distilled models now post benchmark numbers that recently required much larger systems. Gemma 2 trained its 2B and 9B models with knowledge distillation instead of next-token prediction and reports them as competitive with models 2–3× larger [Gemma Team 2024]. DeepSeek-R1 fine-tuned dense models from 1.5B to 70B parameters on about 800K teacher-generated samples. The 1.5B distilled model reports 28.9% on AIME 2024 and 83.9% on MATH-500 [DeepSeek-AI 2025]. Xiao et al. [2025] fit an empirical "densing law": the parameter count needed to reach a fixed benchmark score has halved roughly every three to four months (3.3 months in the preprint, 3.5 months in the journal version). Read naively, these results suggest that frontier capability compresses with no visible floor.

Two other strands of work point the other way. Factual question answering degrades sharply for rare entities [Kandpal et al. 2023; Mallen et al. 2023], and scaling mostly helps popular knowledge [Mallen et al. 2023]. Controlled studies find that language models store about 2 bits of factual knowledge per parameter, with capacity growing linearly in model size [Allen-Zhu & Li 2025]. In classical distillation, students often fail to match teachers even when they have the capacity [Stanton et al. 2021]. Very large teachers can also be poor teachers for very small students [Cho & Hariharan 2019; Mirzadeh et al. 2020]. Distillation can hurt rare subgroups [Lukasik et al. 2021]. Distillation scaling laws [Busbridge et al. 2025] find that distillation beats supervised training only up to a compute level that grows with student size.

These two strands are not in contradiction. Our claim is that they measure different things. We make this precise with a minimal theory and test it in a controlled experiment. We do not attempt to measure frontier LLMs: we have no GPU and run no large models. We isolate the mechanism in a small setting, derive quantitative predictions, and give a protocol for testing them at scale.

**Sub-question.** In a world where capabilities are discrete skills with power-law usage frequencies, what limits how many skills a student of size N retains when trained by a teacher? Which of these limits can distillation lift? How does the answer depend on the evaluation metric?

**Contributions.**

1. **A procedure-independent capacity bound (Proposition 1).** A rate–distortion argument shows that no teaching signal lets a student retain more independent skills than its parameter bits allow. This includes soft labels, unlimited teacher queries, and chain-of-thought traces. The lower bound on student size is the sum of the retained skills' rates, so it depends on how *compressible* the skills are, not on how they are taught.
2. **Aggregate–tail divergence (Proposition 2).** Under Zipfian skill usage with exponent 1+α, a student allowed aggregate error 1+δ times the teacher's can drop a factor (1+δ)^{1/α} of the teacher's skills, and a uniform tail benchmark falls by the same factor. For α = 0.5, doubling aggregate error permits a 4× cut in skills. For α = 0.25, it permits 16×. The answer to "how small?" is therefore metric-dependent in a predictable way.
3. **A three-limit account of when distillation helps (Proposition 3).** The skills learned are K* = min(K_cap(N), K_D(D, γ), K_opt), where distillation raises only the data-limited frontier, by γ^{1/(1+α)}, and in practice also the optimisation-limited frontier. This yields a crossover budget D* ∝ K_cap^{1+α} beyond which distillation stops helping. It gives a mechanistic account of the crossover reported empirically by Busbridge et al. [2025].
4. **A controlled testbed and 166 CPU training runs** that test each proposition (Sections 5–6). The testbed has 400 Zipf-weighted stochastic skills, controllable skill compressibility and label entropy, oracle and trained teachers, and students from 404 to 76,292 parameters. Code and raw per-skill outputs are released.
5. **Four falsifiable predictions for LLM distillation and a concrete evaluation protocol** (Section 7.2), based on frequency-stratified evaluation.

## 2 Related Work

**Knowledge distillation.** Hinton et al. [2015] introduced training a student on a teacher's softened output distribution and argued that "dark knowledge" in non-argmax probabilities carries information. Statistical analyses explain the benefit as variance reduction: the student's risk under Bayes class-probabilities has lower variance than under one-hot labels [Menon et al. 2021]. For linear students, Phuong & Lampert [2019] prove fast convergence. Our parameter γ, the per-skill sample-efficiency gain, is the skill-level counterpart of this variance reduction. We measure how it depends on label entropy (Section 6.4). Empirically, distillation often fails to make students match teachers [Stanton et al. 2021]. Large teacher–student gaps can hurt [Cho & Hariharan 2019], which motivated teacher assistants [Mirzadeh et al. 2020]. Distillation can harm low-sample subgroups [Lukasik et al. 2021]. Our work does not propose a new distillation method. It asks what *no* method can do, and where the gains of existing methods come from.

**Distillation and scaling laws.** Busbridge et al. [2025] fit a distillation scaling law and find that "distillation outperforms supervised learning up to a compute level that scales predictably with student size." Proposition 3 derives such a crossover from a skill-frequency mechanism and predicts its exponent. Gemma 2 [Gemma Team 2024] uses distillation to train small models beyond compute-optimal token counts. Inference-aware scaling [Sardana et al. 2024] motivates training small models long. Xiao et al. [2025] measure "capability density" against reference-model scaling curves on aggregate benchmarks. Proposition 2 implies that density measured this way depends on the frequency weighting of the benchmark.

**Quantised skills and power laws.** The quantization model [Michaud et al. 2023] explains power-law scaling by discrete "quanta" learned in order of decreasing use frequency, with Zipfian frequencies. Hutter [2021] derives data learning curves from Zipf-distributed features. We build directly on this view and add three things it lacks: a teacher, a procedure-independent capacity bound, and the distinction between aggregate and tail evaluation. Allen-Zhu & Li [2025] measure about 2 bits/parameter knowledge capacity in language models. This empirical constant enters Proposition 1 as b. Superposition [Elhage et al. 2022] is one way networks store more features than neurons. It does not evade the bit-counting bound.

**Long-tail knowledge.** Kandpal et al. [2023] show strong correlational and causal dependence of QA accuracy on the number of relevant pretraining documents. Mallen et al. [2023] show that scaling mainly improves popular knowledge in PopQA. These results motivate our tail benchmark, and our predictions P1–P2 extend them to distilled students.

**Gap.** We could not find prior work that combines (i) a bound on what distillation *cannot* transfer, valid for every teaching signal, with (ii) an explicit account of how the answer to "how small?" depends on the evaluation weighting and the binding resource, and that tests both in a controlled setting.

## 3 Model

**Skills.** A capability domain consists of skills i = 1, 2, …, M. When the domain is used, skill i is invoked with probability p_i = i^-(1+α) / Z_M, with α > 0 and Z_M = Σ_{j≤M} j^-(1+α). Z_∞ = ζ(1+α). Skill i is specified by a parameter θ_i, for example a mapping from inputs to output distributions. A model's competence on skill i is a score s_i ∈ [0,1].

**Two evaluation functionals.** The *aggregate* score A = Σ_i p_i s_i weights skills by use frequency. It is the analogue of loss on held-out natural data, or of a benchmark sampled from the usage distribution. The *tail* score over the first M skills, U_M = M^-1 Σ_{i≤M} s_i, weights skills uniformly. It is the analogue of an entity-balanced benchmark such as PopQA, or of a curated benchmark of rare skills. Write E = 1 − A for aggregate error.

**Students and teachers.** A student has N real parameters representable to b bits each. A teacher provides training signal on inputs: hard labels sampled from the true output distribution, or its own predictive distribution (soft labels). A data budget D is the number of distinct training inputs. We say skill i is *learned* if s_i ≥ τ; we use τ = 0.75.

## 4 Theory

### 4.1 A capacity bound that no teacher can lift

**Proposition 1 (procedure-independent capacity bound).** Let θ_1, …, θ_M be independent random skill parameters. Let ω be the student's parameters after any training procedure, possibly randomised and using any teacher, any data, and any number of queries. Suppose ω takes at most 2^{bN} values. For each skill i, let θ̂_i = g_i(ω) be the student's implied estimate of θ_i, and let d_i be a distortion measure with rate–distortion function R_i(ε). If E d_i(θ_i, θ̂_i) ≤ ε for all i ∈ S, then

  bN ≥ Σ_{i∈S} R_i(ε).

*Proof.* H(ω) ≤ bN, so I(θ_S; ω) ≤ bN. For independent θ_i, the chain rule gives I(θ_S; ω) = Σ_k I(θ_{i_k}; ω | θ_{i_1..i_{k-1}}). Each term equals I(θ_{i_k}; ω, θ_{<k}) − I(θ_{i_k}; θ_{<k}) = I(θ_{i_k}; ω, θ_{<k}) ≥ I(θ_{i_k}; ω), using independence. By data processing, I(θ_i; ω) ≥ I(θ_i; θ̂_i). By the definition of the rate–distortion function, achieving expected distortion ≤ ε requires I(θ_i; θ̂_i) ≥ R_i(ε). ∎

*Remarks.* (i) The bound treats training as an arbitrary channel from (θ, teacher, data) to ω, so it holds for soft labels, logits, intermediate-feature matching, synthetic data, and reasoning traces alike. (ii) It concerns *independent* skill content. If skills share structure, as in the low-rank family below, R_i should be read as the conditional rate given the shared part. The bound then becomes Σ_i R_i(ε | shared) + R(shared), which can be much smaller. Compressibility, not teaching, sets the limit. (iii) With R_i = R for all i, at most K_cap ≤ bN/R skills are retained. Using the measured b ≈ 2 bits/parameter for factual knowledge [Allen-Zhu & Li 2025], a student that must retain M independent facts of R bits each needs at least MR/2 parameters, however it is trained.

### 4.2 Aggregate versus tail retention

**Proposition 2 (aggregate–tail divergence).** Let M = ∞. Suppose a teacher has learned exactly the top K_T skills and a student exactly the top K_S = K_T / r skills, with r ≥ 1. Then as K_S → ∞,

  E_S / E_T → r^α  and  U_{K_T}(student) = 1/r.

Hence, if the student may have aggregate error at most (1+δ)·E_T, the largest admissible skill-compression factor is r* = (1+δ)^{1/α}, and the student's tail score on the teacher's skill set is (1+δ)^{-1/α}.

*Proof.* E(K) = Σ_{i>K} p_i. Integral comparison gives K^-α/α ≥ Σ_{i>K} i^-(1+α) ≥ (K+1)^-α/α, so E(K) = K^-α/(α ζ(1+α)) · (1 + O(1/K)). The ratio of the errors is therefore (K_T/K_S)^α (1+o(1)) = r^α. The tail statement is immediate because the student scores 1 on K_S of the K_T skills and 0 on the rest. ∎

Table 1 evaluates r* (computed in `analyze.py`, `prop2_table`). The flatter the usage distribution (smaller α), the more skills a student can drop before an aggregate metric notices.

**Table 1.** Admissible skill-compression r* = (1+δ)^{1/α} (equal to 1/tail retention) at aggregate-error ratio 1+δ.

| α | δ = 0.25 | δ = 0.5 | δ = 1 | δ = 2 |
|---|---|---|---|---|
| 0.25 | 2.44 | 5.06 | 16.0 | 81.0 |
| 0.5 | 1.56 | 2.25 | 4.0 | 9.0 |
| 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |

### 4.3 When distillation helps: three frontiers

Following the quantization model [Michaud et al. 2023], assume skills are acquired in order of decreasing frequency. Assume also that skill i becomes learnable once its expected number of training examples D·p_i exceeds a threshold n_m that depends on the supervision method m. Write n_hard = n_h for hard labels and n_soft = n_h/γ for soft labels, so γ ≥ 1 is the per-skill sample-efficiency gain of distillation.

**Proposition 3 (frontiers).** Under these assumptions, the number of learned skills is K* = min(K_cap(N), K_D(D; m)), where

  K_D(D; m) = (D / (n_m Z))^{1/(1+α)}.

Therefore (a) in the data-limited regime, distillation multiplies the skill frontier by γ^{1/(1+α)}. (b) In the capacity-limited regime, distillation has no effect on K*, by Proposition 1. (c) The hard-label student reaches the capacity frontier at D* = n_h Z K_cap^{1+α}, and the distillation advantage vanishes for D ≥ D*. Since K_cap ∝ N for incompressible skills (Prop. 1), D* ∝ N^{1+α}. In compute terms (C ≈ 6ND), C* ∝ N^{2+α}.

*Proof.* Skill i is learnable from data iff D p_i ≥ n_m, that is, i ≤ (D/(n_m Z))^{1/(1+α)}. Frequency ordering and the capacity bound give the minimum. Parts (a)–(c) follow by substitution. ∎

In practice there is a third, *optimisation-limited* frontier K_opt(steps; m): for a finite number of gradient steps, lower-variance soft targets can also reach more skills [Phuong & Lampert 2019]. We observe this regime in Section 6.3 and separate it from the capacity frontier with long training in Section 6.5.

## 5 Experimental Testbed

All numbers in Sections 5–6 come from code we wrote and ran on a 4-core CPU (`code/`). Raw outputs are in `results/all_runs.json` and `results/e5_runs.json`, and the summary is in `results/summary.json`. Total compute is about 44 minutes of wall-clock time on 4 cores.

**World.** M = 400 skills with Zipf frequencies; α = 0.5 unless stated. A query is (t, z) with skill index t ~ p and input z ~ N(0, I_4). The true output distribution is p(y | t, z) = softmax(β W_t z) over C = 4 classes, with β = 2 by default. Two skill families:

- *random*: W_t has i.i.d. N(0,1) entries, so each skill carries 16 independent Gaussian numbers and the family is incompressible.
- *lowrank*: W_t = Σ_{k=1}^{2} c_{tk} B_k with shared bases B_k and per-skill coefficients c_t ∈ R^2. Each skill carries 2 numbers given the shared structure.

**Students.** A two-hidden-layer ReLU MLP of width h. Its input is z and a fixed, non-learned random 32-dimensional code for skill t. All student parameters are shared across skills; there is no per-skill embedding table, so parameter counts measure shared capacity. Parameter counts range from 404 (h = 8) to 76,292 (h = 256).

**Supervision.**
- *Hard*: labels sampled from p(y | t, z).
- *Soft (oracle)*: the true distribution p(· | t, z), an idealised perfect teacher.
- *Soft (trained teacher)*: the predictive distribution of an h = 256 network trained for 30,000 steps on fresh hard-labelled samples. This teacher learns 241 of 400 skills, with aggregate score A = 0.988.

Training uses Adam (lr 10^-2, cosine schedule) with batch 512. With a finite budget D, early stopping uses a validation set of size max(D/4, 2000) drawn with the *same* label type. D = "unlimited" means fresh samples at every step.

**Metrics.** For skill t, s_t = clip(1 − KL_t / KL0_t, 0, 1). KL_t is the student's mean KL divergence from the true distribution over 200 fixed test inputs, and KL0_t is the KL of the uniform predictor. A skill is learned if s_t > 0.75. We report K (number of skills learned), the aggregate A = Σ p_t s_t, and the tail K/M or U = mean_t s_t.

**Experiments.**
- E1: capacity regime; unlimited oracle soft labels, 45k steps, widths 8–48, both families, 2 seeds; plus 90k-step convergence checks.
- E2: data regime; D ∈ {10^3, …, 3·10^5, unlimited}, hard vs oracle soft, widths 16/32/128, 2 seeds, 10k steps; plus the trained teacher at h = 128.
- E3: label-entropy sweep, β ∈ {0.5, …, 8}, h = 128, D = 10^4.
- E4: α ∈ {0.25, 1.0}, capacity regime.
- E5: long-training (45k steps) hard-vs-soft comparison at h ∈ {8, 16}, D ∈ {10^4, 10^5, unlimited}, 2 seeds.

## 6 Results

### 6.1 Skills are acquired in frequency order and capacity scales with parameters

Figure 1a (`results/fig1_capacity.png`) plots skills learned against student parameters in the capacity regime (E1). For the random family, K grows from 5.0 (404 parameters) to 110.0 (4,324 parameters). The log–log slope over this range is 1.28. Parameters per learned skill fall from 81 at h = 8 to 39 at h = 48, which is about 2.5 parameters per independent Gaussian number of skill content. Figure 1b shows that per-skill scores decline with frequency rank. The Spearman correlation between rank and score is −0.46, −0.65, −0.82 and −0.93 for h = 8, 16, 32, 48. The correlation is weaker for small h because most skills tie at score 0. Learning is frequency-ordered, as the quantization model assumes.

*Convergence.* Doubling training to 90k steps changes K from 21 to 22 at h = 16 but from 53 to 63 at h = 32 (seed 0). The E1 values at h ≥ 32 are therefore lower bounds on capacity, and the slope of 1.28 partly reflects incomplete convergence of the larger students. The h ≤ 16 points are essentially converged.

*Compressibility.* At equal parameter count, the low-rank family fits 2.8–3.8× more skills: 61.5 vs 20.0 at 932 parameters, and 216.5 vs 56.5 at 2,372 parameters. At 4,324 parameters it nearly saturates the world, at 375.5 of 400 skills. The per-skill information ratio is 8× (16 vs 2 numbers). The observed gain is smaller, which we attribute to routing and decoding overhead that the student pays per skill regardless of content. As Proposition 1 predicts, the attainable compression is governed by how compressible the skills are.

### 6.2 Aggregate metrics hide tail loss, as Proposition 2 predicts

Figure 2a (`results/fig2_head_tail.png`) plots the aggregate score A against the fraction of skills learned K/M for 16 capacity-regime runs across α ∈ {0.25, 0.5, 1.0}. The solid curves are the zero-parameter top-K prediction A_pred = Σ_{i≤K} p_i, computed from each run's own K. It matches the observed A with a mean absolute error of 0.016 (`head_tail_A_abs_err`). Most of the residual comes from partial credit on skills just below threshold.

Compared against the trained teacher (76,292 parameters, 241 skills, A = 0.988), the divergence is large (Table 2, `compression_vs_teacher`):

**Table 2.** Students trained with unlimited oracle soft labels (E1, random family, 2 seeds) compared with the trained teacher.

| width | params | compression (teacher/student params) | aggregate A | aggregate-error ratio | teacher skills retained |
|---|---|---|---|---|---|
| 48 | 4,324 | 17.6× | 0.969–0.970 | 2.4–2.5 | 44–47% |
| 32 | 2,372 | 32.2× | 0.938–0.946 | 4.4–5.1 | 22–25% |
| 24 | 1,588 | 48.0× | 0.920–0.921 | 6.5–6.6 | 14–17% |
| 16 | 932 | 81.9× | 0.871–0.884 | 9.5–10.6 | 8–9% |
| 8 | 404 | 188.8× | 0.726–0.767 | 19–22 | 2–3% |

A 32× smaller student sits within 5 aggregate points of the teacher and has lost three-quarters of the teacher's skills. A compression claim based on the aggregate column would be correct for that metric and would still overstate how much capability was preserved.

### 6.3 In the data-limited regime, distillation acts as a sample-efficiency multiplier

Figure 3 (`results/fig3_data_regime.png`) shows E2. At h = 128, which is not capacity-bound in this range, soft-label students learn more skills at every D: 33.5 vs 7.0 at D = 10^4, 68.0 vs 16.5 at 3·10^4, and 181.0 vs 77.0 at 3·10^5. We fit the threshold rule K = (D/(n_m Z))^{1/1.5} to points with 3 < K < 0.8·K_unlimited. The fit gives n_hard ≈ 188 and n_soft ≈ 22.5 samples per skill, so **γ ≈ 8.4**. The predicted exponent is 1/(1+α) = 0.667. Over 10^3–10^5, the soft-label curve has empirical slope ≈ 0.62 (7.5 → 131 skills); over 10^4–10^5, the hard-label curve has slope ≈ 0.73 (7.0 → 37.5).

With the *trained* teacher, students track the oracle-soft students up to D = 3·10^4 (60 vs 68 skills). They then fall behind, reaching 138 vs 181 at D = 3·10^5. This is consistent with the teacher's own frontier of 241 skills and its imperfect predictive distribution on rarer skills. Distillation cannot transfer skills the teacher lacks, and it transfers the teacher's weak skills less well.

*Crossover.* Proposition 3 predicts that the soft-label advantage disappears for D ≥ D* = n_h Z K_cap^{1.5}. Taking K_cap as each width's 10k-step unlimited-data soft-label ceiling, this gives D* ≈ 2.7·10^4 (h = 16), 1.1·10^5 (h = 32) and 1.6·10^6 (h = 128). The observed relative gap (K_soft − K_hard)/K_soft shrinks as predicted:
- h = 16: from 0.45 at D = 10^4 to 0.24–0.25 for D ≥ 3·10^4.
- h = 32: from 0.54 at 10^4 to 0.38, 0.28 and 0.18 at 3·10^4, 10^5 and 3·10^5.
- h = 128 (D* far beyond the grid): 0.57–0.79 throughout.

The gap does not reach zero at 10k steps, because the hard-label ceiling under unlimited data is also lower: 13 vs 15 skills (h = 16) and 27 vs 38 (h = 32). This is the optimisation-limited frontier: with a fixed step budget, noisy hard-label gradients reach fewer skills. Section 6.5 removes this confound.

### 6.4 The distillation multiplier grows with label entropy

E3 (Figure 4, `results/fig4_noise.png`) varies the inverse temperature β of the true skills at D = 10^4 and h = 128. As β decreases, the Bayes conditional entropy H(y | t, z) rises from 0.14 to 1.14 nats, and the advantage of soft labels grows sharply.

| β | Bayes entropy (nats) | K_hard | K_soft | implied γ = (K_soft/K_hard)^{1.5} |
|---|---|---|---|---|
| 8 | 0.14 | 9.0 | 12.5 | 1.6 |
| 4 | 0.27 | 9.5 | 19.0 | 2.8 |
| 2 | 0.51 | 7.0 | 33.5 | 10.5 |
| 1 | 0.84 | 2.0 | 41.5 | ≥ 95 |
| 0.5 | 1.14 | 0.5 | 43.0 | ≥ 280 (floor effect) |

The β = 2 value (10.5) agrees with the independent fit in Section 6.3 (8.4). For β ≤ 1, hard-label students learn almost nothing at this D, so the implied γ is a floor-limited lower bound rather than a precise estimate. There is also a confound: lower β makes the target logits smaller in norm, which may make skills easier to fit under soft supervision, and the rise of K_soft as β falls partly reflects this. The qualitative conclusion is robust: distillation saves data in proportion to how much of the teacher's distribution a sampled label throws away. This is consistent with the variance-reduction view of Menon et al. [2021].

### 6.5 With enough data and training, distillation does not raise the capacity ceiling

E5 (Figure 5, `results/fig5_ceiling.png`) repeats the hard-vs-soft comparison with 45k steps at small widths (mean over 2 seeds):

| width | D = 10^4 hard / soft | D = 10^5 hard / soft | unlimited hard / soft |
|---|---|---|---|
| h = 8 (404 params) | 4.0 / 3.5 | 4.0 / 5.0 | 5.0 / 5.0 (A: 0.743 / 0.747) |
| h = 16 (932 params) | 5.0 / 10.5 | 16.0 / 18.0 | 17.5 / 20.0 (A: 0.864 / 0.878) |

At h = 8 the student is capacity-bound at every D, and soft labels make no difference. At h = 16, soft labels double the skills learned at D = 10^4 (10.5 vs 5.0). At unlimited data the difference shrinks to 2.5 skills, a factor of 1.14, or 1.4 aggregate points. This residual may reflect incomplete convergence of the noisier hard-label run, but we cannot rule out a small capacity-regime benefit of soft targets. Within seed noise, the result supports Proposition 3(b): distillation helps students that are short of data or optimisation, and helps little or not at all once they are short of parameters.

## 7 Discussion

### 7.1 So how small?

In this model, the honest answer to "how small can distillation make a frontier-capable model?" has three parts.

1. **For the skills you must keep, size is bounded below by their incompressible content** (Prop. 1). Distillation, synthetic data and longer training do not move this bound. They change only how close a student of given size gets to it. Our testbed shows a factor of about 3 between incompressible and low-rank skills at equal parameter count. Real capabilities plausibly span a much wider range: algorithmic procedures with short descriptions at one end, long-tail facts at the other.
2. **How many skills you may drop depends on how you score** (Prop. 2). Aggregate metrics weighted by natural usage permit large compression; with α = 0.5, a 4× cut in skills costs only a 2× rise in aggregate error. Uniform tail metrics expose the loss one-for-one. Reported compression results, including "capability density" trends [Xiao et al. 2025], should therefore name the frequency weighting of their benchmarks. Our model predicts that density gains measured on head-heavy benchmarks will be larger than those measured on tail-balanced ones.
3. **Distillation is a data and compute multiplier, not a capacity multiplier** (Prop. 3). Its value is largest when the student is far below its capacity frontier and the teacher's outputs are high-entropy. It shrinks once the student is trained to its frontier, at a budget that grows as K_cap^{1+α}. The impressive small distilled models in the literature are consistent with this picture. Under our account, they run near their capacity frontier on head skills, such as common reasoning patterns in mathematics benchmarks, while dropping tail skills that aggregate benchmarks under-weight.

### 7.2 Falsifiable predictions for LLMs and a test protocol

We have not tested these on LLMs. They are stated so that others can test them.

- **P1 (frequency-stratified deficit).** For a distilled student and its teacher, the per-item accuracy gap grows monotonically with item rarity, for example with log-popularity bins of PopQA [Mallen et al. 2023] or the document-count bins of Kandpal et al. [2023]. Within each bin, the gap is approximately all-or-none across items. *Falsified if* the gap is flat across popularity bins.
- **P2 (head–tail compression mismatch).** Take a model family with distilled students (for example Gemma 2 2B/9B vs 27B [Gemma Team 2024], or DeepSeek-R1-Distill-Qwen-1.5B/7B/14B/32B [DeepSeek-AI 2025]). The effective compression ratio computed from a frequency-weighted benchmark exceeds that computed from a popularity-balanced benchmark of the same domain by roughly the factor of Prop. 2 for the fitted α. *Falsified if* the two ratios agree.
- **P3 (no capacity gain from distillation).** At matched student size and very long training, distilled and non-distilled students converge on tail-knowledge benchmarks, while distilled students lead at short budgets. The crossover budget grows superlinearly in student size, D* ∝ N^{1+α}. *Falsified if* distilled students keep a tail-knowledge advantage that does not shrink with training budget.
- **P4 (entropy dependence).** The distillation advantage at fixed budget is larger for domains where the teacher's next-token entropy is higher. *Falsified if* advantage and teacher entropy are uncorrelated across domains.

*Protocol.*
1. Choose a domain with an external popularity signal (PopQA entities, or n-gram or document counts in an open pretraining corpus).
2. Evaluate teacher, distilled student and same-size non-distilled student per item.
3. Fit α from the popularity–usage relation and compute A (usage-weighted) and U (uniform).
4. Test P1 with a monotone-trend test across bins and P2 by comparing effective compression ratios.
5. For P3, use open intermediate checkpoints, where available, to trace the gap against training tokens.

All of these steps require only inference on public checkpoints.

## 8 Limitations

- **Toy setting.** Our skills are random bilinear-softmax maps routed by a fixed code, and our students are 2-layer MLPs with at most 76k parameters. Real capabilities are not cleanly discrete, are not used independently, and can be composed. Composition can make a small model's effective skill count exceed what an independent-skill count suggests, and Proposition 1 deliberately ignores this.
- **Frequency ordering and thresholds are assumptions.** Proposition 3 assumes skills are learned in frequency order with a sharp per-skill sample threshold. Our data support this approximately (Spearman −0.65 to −0.93; fitted exponents 0.62–0.73 vs the predicted 0.667) but not exactly.
- **Convergence.** E1 points at h ≥ 32 were not converged at 45k steps (53 → 63 skills at 90k steps for h = 32), so the log–log slope of 1.28 should not be read as a scaling exponent. E5 used only h ≤ 16 because longer runs for wider students exceeded our compute budget. The 1.14× residual soft-label advantage at h = 16 is not resolved.
- **The threshold metric is a choice.** The learned/not-learned threshold τ = 0.75 affects K; A and U are threshold-free and are reported alongside it.
- **γ estimates.** The β-sweep values of γ rest on a single D and 2 seeds, have floor effects for β ≤ 1, and are confounded by logit-norm changes.
- **Few seeds.** Most configurations have 2 seeds. Error bars are min/max over seeds and are not confidence intervals.
- **Teacher realism.** The "oracle" teacher is an idealisation. Our single trained teacher shows that teacher imperfection matters (138 vs 181 skills at D = 3·10^5), but we do not study teacher-size effects [Cho & Hariharan 2019; Mirzadeh et al. 2020].
- **No LLM evidence.** Every LLM-level statement in Section 7 is a prediction, not a finding.

## 9 Conclusion

The question "how small can distillation compress frontier capability?" has a precise answer only once the retained skill set and its weighting are fixed. In a Zipfian skill model, three results hold. Capacity is bounded by the incompressible content of the retained skills, whatever the teacher. Aggregate metrics allow skill compression of (1+δ)^{1/α} at error ratio 1+δ, and tail metrics fall by the same factor. Distillation raises the data- and compute-limited frontiers by a factor that grows with label entropy, but not the capacity frontier. A controlled CPU testbed supports each claim quantitatively. Its weakest points are the convergence of the larger students and the single-width test of the capacity ceiling. The practical implication is that compression claims should be reported with frequency-stratified evaluations. The scientific implication is that the floor on model size is set by what must be remembered, not by how it is taught.

## Reproducibility

`README.md` lists the commands: `python3 code/run_all.py`, `python3 code/run_e5.py`, `python3 code/analyze.py`. Every number in Sections 5–6 is read from `results/summary.json` or the raw run files. Literature numbers in Section 1 (Gemma 2, DeepSeek-R1, densing law, 2 bits/parameter) come from the cited sources and are labelled as such.

## References

- Allen-Zhu, Z., & Li, Y. (2025). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2404.05405.
- Busbridge, D., Shidani, A., Weers, F., Ramapuram, J., Littwin, E., & Webb, R. (2025). Distillation Scaling Laws. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:5977–6045. arXiv:2502.08606.
- Cho, J. H., & Hariharan, B. (2019). On the Efficacy of Knowledge Distillation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4793–4801. doi:10.1109/ICCV.2019.00489.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
- Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Hutter, M. (2021). Learning Curve Theory. arXiv:2102.04074.
- Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *Proceedings of the 40th International Conference on Machine Learning (ICML)*. arXiv:2211.08411.
- Lukasik, M., Bhojanapalli, S., Menon, A. K., & Kumar, S. (2021). Teacher's pet: understanding and mitigating biases in distillation. arXiv:2106.10494.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. *Proceedings of the 61st Annual Meeting of the ACL (Long Papers)*. arXiv:2212.10511.
- Menon, A. K., Rawat, A. S., Reddi, S., Kim, S., & Kumar, S. (2021). A statistical perspective on distillation. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:7632–7642.
- Michaud, E. J., Liu, Z., Girit, U., & Tegmark, M. (2023). The Quantization Model of Neural Scaling. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2303.13506.
- Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020). Improved Knowledge Distillation via Teacher Assistant. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), 5191–5198. arXiv:1902.03393.
- Phuong, M., & Lampert, C. H. (2019). Towards Understanding Knowledge Distillation. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:5142–5151.
- Sardana, N., Portes, J., Doubov, S., & Frankle, J. (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:43445–43460. arXiv:2401.00448.
- Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). Does Knowledge Distillation Really Work? *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*.
- Xiao, C., Cai, J., Zhao, W., Zeng, G., Lin, B., Zhou, J., Zheng, Z., Han, X., Liu, Z., & Sun, M. (2025). Densing law of LLMs. *Nature Machine Intelligence*, 7, 1823–1833. arXiv:2412.04315.
