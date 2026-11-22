# Silent Latents: Feature Absorption Is What Sparse Coding Optimises For, and Activations Alone Cannot Certify Its Absence

**Target venue:** NeurIPS (main track; Theory / Interpretability). Secondary fit: ICLR.

---

## Abstract

Sparse autoencoders (SAEs) are the main tool for reading the "thoughts" of language models. A latent that fires on a concept is taken as evidence that the model represents the concept, and its silence is often read as evidence that it does not. *Feature absorption* (Chanin et al., 2024) breaks the second inference: a latent that looks monosemantic fails to fire on part of its concept's support, because a more specific latent has taken over its direction. We study when absorption arises and whether it can be detected. We use a minimal model: a parent concept P and a child concept C with orthogonal directions, where C usually implies P, coded by the non-negative lasso objective that ReLU SAEs approximate. We prove four results and check each numerically.

1. **Closed form at fixed magnitudes.** For fixed feature magnitudes, the fully absorbed dictionary has strictly lower loss than the faithful one on every co-occurrence input when a + c − √(a² + c²) > λ/4. Here a and c are the magnitudes and λ is the sparsity weight.
2. **The faithful dictionary is never stationary.** Its directional derivative toward absorption equals −E[1_{P∧C}·w·(c − λ/2)₊], which is non-zero whenever parent and child co-occur. This holds however often the child also appears alone and however variable the magnitudes are.
3. **The optimal child atom does not depend on q.** With the parent atom fixed, the optimal child atom is independent of q = P(child | parent).
4. **A hierarchy-violation threshold.** We give a closed-form rate ρ† of child-without-parent occurrences at which the faithful dictionary overtakes the fully absorbed one. The observed transition is first-order and occurs at about 0.55–0.6 ρ†.

Between the extremes, magnitude variability produces partial absorption. The parent latent fires on only 0–92% of co-occurrence inputs, and its precision stays at 1.0 in every configuration we ran (35 settings of magnitude variation and λ). Trained numpy ReLU SAEs match the exact optimum to within 1.5° in decoder angle.

We then prove a non-identifiability result. A fully absorbed hierarchy produces exactly the same activation distribution as a pair of mutually exclusive, non-orthogonal features, which have no false negatives. So no test that sees only activations or SAE weights can certify that absorption is absent. In a 48-feature toy model (5 seeds, 240 latents), all 18 latents with recall below 0.9 are hierarchy parents with precision 1.0. A weight-only absorption score separates them from healthy latents with AUROC 0.97, but drops to 0.86 once exclusive non-orthogonal pairs are included, because it flags those pairs exactly as the theorem predicts. A linear probe on the raw activations recovers the parent with recall ≈ 1.0 in the same settings where the SAE latent reaches only 0.40. The information is present in the model. The dictionary does not read it out. We conclude with falsifiable predictions and a test protocol for LLM SAEs.

## 中文摘要

稀疏自编码器（SAE）是"读懂"语言模型内部表征的主要工具。研究者通常把某个潜变量的激活视为模型表征了某概念的证据，而它的沉默也常被解读为模型未表征该概念。"特征吸收"（feature absorption）现象动摇了后一种推断：看似单义的潜变量在其概念的一部分支撑集上不激活，因为更具体的潜变量吸收了它的方向。本文在一个最小模型中研究吸收何时出现、能否被检测。模型包含父概念 P 与子概念 C，二者方向正交，且子概念通常蕴含父概念；编码目标是 ReLU SAE 所近似的非负 lasso。我们证明四个结论并逐一做数值验证。（1）幅值固定时，只要 a+c−√(a²+c²) > λ/4，完全吸收的字典在每个共现输入上的损失都严格低于忠实字典。（2）忠实字典不是驻点：它朝吸收方向的方向导数为 −E[1_{P∧C}·w·(c−λ/2)₊]，只要父子概念共现就不为零，与子概念单独出现的频率和幅值波动无关。（3）固定父原子时，最优子原子与共现率 q = P(子|父) 无关。（4）我们给出"违反层级"比率的闭式阈值 ρ†，忠实字典在该点超过完全吸收的字典；实际观察到的是一阶相变，位于约 0.55–0.6 ρ† 处。幅值波动在两个极端之间产生部分吸收：父潜变量只在 0–92% 的共现输入上激活，而我们运行的全部 35 组配置中其精确率都保持 1.0。训练得到的 ReLU SAE 与精确最优解的解码器角度相差不超过 1.5°。随后我们证明一个不可辨识性定理：完全吸收的层级结构与一对互斥、非正交的特征产生完全相同的激活分布，而后者不存在漏检。因此，任何只看激活或 SAE 权重的检验都无法证明吸收不存在。在 48 特征的玩具模型中（5 个随机种子，共 240 个潜变量），召回率低于 0.9 的 18 个潜变量全部是层级中的父特征，精确率均为 1.0。一个仅基于权重的吸收分数能以 AUROC 0.97 把它们与正常潜变量区分开；但加入互斥非正交特征对后降至 0.86，因为它如定理所预言地把这些特征对也标记为可疑。在同样的设置下，直接作用于原始激活的线性探针对父概念的召回率约为 1.0，而 SAE 潜变量只有 0.40：信息存在于模型中，只是字典没有把它读出来。最后，我们给出可证伪的预测和面向大模型 SAE 的检验方案。

---

## 1 Introduction

Can we actually read what a model is "thinking"? The current answer of mechanistic interpretability is a qualified yes. Sparse dictionary learning decomposes activations into many latents, and each latent fires on a narrow, human-nameable concept (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024; Gao et al., 2024). A reading built on such a dictionary makes two different kinds of claim:

- **Positive claims.** "Latent *k* fired, so the model represented concept *k* here." These rest on the latent's *precision*.
- **Negative claims.** "Latent *k* did not fire, so the model did not represent *k* here." These rest on its *recall*. Negative claims carry much of the safety weight. A deception or "dangerous-capability" latent that stays silent is reassuring only if silence means absence.

Chanin et al. (2024) documented a failure mode that attacks negative claims directly. In *feature absorption*, a latent that tracks a general concept (e.g. "starts with S") does not fire on some tokens that clearly have the property (e.g. "short"). A token-specific latent has absorbed the general direction. Absorption has since motivated new architectures, such as Matryoshka SAEs (Bussmann et al., 2025), and benchmark metrics such as SAEBench (Karvonen et al., 2025). Related distortions have been described for correlated features (*feature hedging*, Chanin et al., 2025a) and for a mis-set L0 (Chanin & Garriga-Alonso, 2025). Two questions remain open:

1. **When exactly is absorption optimal, and how much of it should we expect?** Existing theory shows that absorption lowers the sparsity penalty for hierarchical features (Chanin et al., 2024). It does not say when it is the global optimum, why it is often *partial* in practice, or which statistics of the data control it.
2. **Can absorption be detected without ground-truth labels?** If not, every unlabeled SAE reading carries an uncertified recall gap.

We answer both questions in the smallest model that contains the phenomenon. The model has two concepts with orthogonal directions in a hierarchy, and a two-atom dictionary trained on the exact non-negative-lasso objective that a ReLU SAE amortises. The setting is deliberately minimal so that every claim can be proved or computed to high accuracy on a CPU. We then check that trained SAEs reach the predicted optima, and that the detection result carries over to a 48-feature model.

**Contributions.**

1. **Exact characterisation of absorption in the minimal model (§4.1–4.4).**
   - At fixed magnitudes, full absorption beats the faithful dictionary in closed form (Prop. 1), and numerically it is the global optimum for every λ we tested.
   - The faithful dictionary is *never* a stationary point when parent and child co-occur (Prop. 2). We give the gradient in closed form and match it to finite differences to within 0.5%.
   - With the parent atom fixed, the optimal child atom does not depend on q = P(child | parent) (Prop. 3).
   - The rate ρ at which the child occurs without its parent controls a first-order transition. We give a closed-form reference threshold ρ† (Prop. 4) and locate the actual jump numerically.
2. **Partial absorption from magnitude variability (§5.1).** Varying the coefficient of variation of feature magnitudes moves the optimum continuously from full absorption to near-faithful. The parent latent then fires on a contingent subset of co-occurrence inputs, decided by the magnitude ratio a/c. Its precision stays at 1.0 throughout, so standard precision-based interpretability checks cannot see the gap. Trained ReLU SAEs reproduce the exact optimum.
3. **Non-identifiability theorem (§4.5).** Every fully absorbed hierarchy has an activation-equivalent model of mutually exclusive non-orthogonal features in which the same latent has perfect recall. No procedure that uses only activations, or SAE weights fit to them, can certify that absorption is absent. Certification needs an extra inductive bias, such as orthogonality of hierarchical concepts (Park et al., 2025), or extra information, such as concept labels or interventions.
4. **Empirical test of detection in a 48-feature model (§5.3).** A weight-only score detects absorbed parents well (AUROC 0.97). It also raises the false alarms the theorem predicts on exclusive non-orthogonal pairs (AUROC 0.86 overall). A supervised linear probe shows the "missing" information is linearly present in the activations.
5. **Falsifiable predictions and a protocol for LLM SAEs (§6).**

All numbers come from code in `code/` that we ran on a CPU. The outputs are in `results/`, and each number cites its source file.

## 2 Related Work

**Dictionary learning for interpretability.** Sparse coding goes back to Olshausen & Field (1996). Its use to decompose neural-network activations in superposition (Elhage et al., 2022) was established by Cunningham et al. (2023) and Bricken et al. (2023), and scaled by Templeton et al. (2024) and Gao et al. (2024). SAE features have been used to build causal circuits (Marks et al., 2025). Sharkey et al. (2025) list the faithfulness and completeness of SAE decompositions as central open problems.

**Absorption, hedging and related pathologies.** Chanin et al. (2024) introduced feature absorption. They proved that absorption reduces the L1 penalty for a two-feature hierarchy, and measured it on hundreds of LLM SAEs using a first-letter task with known labels. They also found that varying SAE width or sparsity does not remove it. Our Prop. 1 is a sharpened, closed-form version of their observation that includes shrinkage. Props. 2–4, the magnitude-variability mechanism for partial absorption, and the non-identifiability theorem are, to our knowledge, new.

Other related work:
- *Feature hedging* (Chanin et al., 2025a) is a reconstruction-driven distortion for correlated features in narrow SAEs.
- Chanin & Garriga-Alonso (2025) show that a wrong L0 mixes correlated features.
- Matryoshka SAEs (Bussmann et al., 2025) reduce absorption by nesting dictionaries; Chanin et al. (2025a) argue that they trade absorption for hedging.
- SAEBench (Karvonen et al., 2025) includes an absorption metric based on first-letter probes. It needs labels, which is consistent with our impossibility result.
- Shrinkage, the systematic underestimation of activations by L1, was studied by Wright & Sharkey (2024) and addressed by Gated SAEs (Rajamanoharan et al., 2024). Our "magnitude under-reading" (§5.2) is a different effect: the parent latent under-reads the parent by up to 46% at full recall, which is much more than the shrinkage term λ/2.

While surveying we also found a 2026 arXiv listing (arXiv:2609.29551, "HiPACE") that reportedly derives an L0-penalty phase boundary for absorption. We could not retrieve its text to check its claims, so we do not rely on it or compare against it.

**Identifiability.** Locatello et al. (2019) proved that unsupervised disentanglement is impossible without inductive biases. Our Prop. 5 is a concrete, SAE-specific instance of that principle. It is specific to *recall certification* and comes with an explicit equivalent model. Cui et al. (2025) give conditions (extreme sparsity, enough width) under which the SAE optimum recovers ground-truth features. Tang et al. (2025) argue that sparse dictionary learning is non-identifiable in general and propose anchoring. Hierarchies violate the independence that recovery results typically assume, and our results show what goes wrong in that case. Hindupur et al. (2025) argue that each SAE architecture builds in assumptions about concept geometry. Leask et al. (2025) show that SAEs do not find canonical units.

**Concept geometry.** Park et al. (2025) report that hierarchically related concepts in LLMs are represented orthogonally: the child's difference vector is orthogonal to the parent vector. Our model takes this geometry as its ground truth. Our identifiability result says that this orthogonality is exactly the kind of prior that would make absorption detectable from weights.

**Probing versus SAEs.** Kantamneni et al. (2025) found that SAE-based probes do not consistently beat linear-probe baselines. Our §5.3 gives one mechanism for that: a linear probe reads the parent concept that the SAE latent misses.

## 3 Model

**Data.** Let e_P and e_C be orthonormal (in E2/E3 they are embedded in ℝ^8 and ℝ^64). Each input falls into one of four classes:

| class | input | probability |
|---|---|---|
| P-only | x = a·e_P | π(1−q) |
| P∧C | x = a·e_P + c·e_C | πq |
| C-only (hierarchy violation) | x = c·e_C | πqρ |
| none | x = 0 | remainder |

Magnitudes a, c ~ max(0.05, 1 + σ·N(0,1)) are independent. σ is the coefficient of variation (CV). We use π = 0.5 throughout. With ρ = 0 the hierarchy is strict, C ⇒ P, as in "short" ⇒ "starts with S".

**Objective.** A dictionary is two unit-norm atoms d₁ = (cos θ₁, sin θ₁) and d₂ = (cos θ₂, sin θ₂) in the (e_P, e_C) plane. The *exact sparse-coding loss* is

  L(D) = E_x [ min_{z ≥ 0} ‖x − D z‖² + λ·1ᵀz ].   (1)

A ReLU SAE with unit-norm decoder rows and no decoder bias minimises the same expression with an amortised encoder z = ReLU(Wₑx + b) in place of the inner minimum. So (1) lower-bounds the SAE loss for any given dictionary, and E2 measures the gap. Atom d₁ is the "parent latent" (θ₁ ≈ 0) and d₂ the "child latent".
- θ₂ = 90° is the **faithful** dictionary.
- θ₂ = 45° (for a = c) is the **fully absorbed** dictionary: d₂ points at the P∧C input itself.

**Exact codes.** With two atoms, the inner problem in (1) is solved exactly by enumerating the supports ∅, {1}, {2}, {1,2}. For a single active unit atom with correlation b = dᵀx > λ/2, the optimal code is z = b − λ/2 and the loss is ‖x‖² − (b − λ/2)². With both atoms active and Gram matrix G, the loss is ‖x‖² − rᵀG⁻¹r with r = Dx − λ/2 (`code/theory.py`). We find the global optimum over (θ₁, θ₂) on a 29 × 91 grid followed by a 21 × 21 local refinement, with θ₁ ∈ [−20°, 20°] and θ₂ ∈ [0°, 90°].

**Readout metrics.** The parent latent "fires" when z₁ > τ = 0.1. We report:
- **recall_PC**: the firing rate on P∧C inputs, i.e. recall on the part of P's support that absorption threatens;
- **precision**: P(parent present | fires);
- **rel_act_PC**: mean z₁/a on P∧C inputs, i.e. how much of the parent's magnitude the latent reports.

## 4 Theory

### 4.1 Full absorption beats faithfulness at fixed magnitudes

**Proposition 1.** Take a P∧C input x = a e_P + c e_C with a, c > λ/2.
- Under the faithful dictionary (e_P, e_C), the per-input loss is ℓ_F = λ(a + c) − λ²/2.
- Under the absorbed dictionary (e_P, x/‖x‖), it is ℓ_A = λ√(a² + c²) − λ²/4.
- On P-only inputs the two dictionaries have the same loss. On C-only inputs the faithful dictionary is no worse.

Hence ℓ_A < ℓ_F if and only if

  a + c − √(a² + c²) > λ/4.   (2)

*Proof.* Faithful dictionary: G = I, so both codes are b_i − λ/2 and the loss is a² + c² − (a − λ/2)² − (c − λ/2)² = λ(a + c) − λ²/2. Absorbed dictionary: the single atom x/‖x‖ has b = ‖x‖, so the loss is ‖x‖² − (‖x‖ − λ/2)² = λ‖x‖ − λ²/4. The two-atom support cannot do better, because the atom already spans x and adding e_P only adds penalty. On a P-only input, both dictionaries code with e_P alone; Lemma A shows that the child atom is never used there. ∎

For a = c = 1, condition (2) reads 0.586 > λ/4, so it holds for any λ < 2.34. That covers every practical regime. The closed forms match the exact numerical lasso to 1e−15 for six (a, c, λ) combinations [E1: `results/e1_theory.json`, `fixed_closed_form`].

**Lemma A (P-only inputs ignore the child atom).** Fix d₁ = e_P and d₂ = (cos θ, sin θ) with θ ∈ (0, π/2]. For x = a e_P, the optimal code has z₂ = 0.
*Proof.* Take any code with z₂ > 0. Replace z₂d₂ by (z₂ cos θ) e_P, i.e. move z₂ cos θ into z₁. The penalty falls by λz₂(1 − cos θ) ≥ 0. The reconstruction error falls by (z₂ sin θ)², because the off-axis component is removed and the on-axis component is unchanged. Both changes are strictly positive for θ > 0. ∎

**Numerical global optimum.** At σ = 0 and ρ = 0, the exact global optimum is θ₂* = 45.0° (θ₁* = 0°) for every λ ∈ {0.02, 0.05, 0.1, 0.2, 0.4}. The parent latent's recall on P∧C is exactly 0, its precision is 1.0, and the optimal loss is 18–20% below the faithful dictionary's (e.g. 0.1179 vs 0.1456 at λ = 0.1) [E1 grid]. Full absorption is therefore not a training artefact: it is the global minimiser of the objective. This is consistent with Chanin et al.'s (2024) finding that changing width or sparsity does not remove it.

### 4.2 The faithful dictionary is never stationary

**Proposition 2.** Fix d₁ = e_P and parameterise d₂ = (sin ε, cos ε), so ε = 0 is faithful and ε > 0 tilts the child atom toward the parent. Then

  ∂L/∂ε |_{ε=0} = − πq · E[ w(a) · (c − λ/2)₊ | P∧C ],  with w(a) = λ if a > λ/2, and w(a) = 2a otherwise.   (3)

This is strictly negative whenever P(P∧C, c > λ/2) > 0, for every ρ ≥ 0 and every magnitude distribution.

*Proof.* By the envelope theorem, the derivative of the per-input optimal value with respect to d₂ is −2 z₂ (x − Dz) at the optimal code. The derivative of d₂ with respect to ε at 0 is e_P. Take each class in turn:
- **P∧C input with both codes active:** the residual is (λ/2, λ/2), so the contribution is −2 z₂ · λ/2 = −λ z₂ with z₂ = c − λ/2.
- **P∧C input with a ≤ λ/2:** only d₂ is active, the residual's e_P component is a, and the contribution is −2a z₂.
- **C-only input:** the residual's e_P component is 0.
- **P-only input:** z₂ = 0 for small ε (Lemma A).

Taking expectations gives (3). ∎

Formula (3) matches central finite differences of the exact loss (2 × 10⁵ samples) to within 0.4% in all four tested settings. For example, at σ = 0, ρ = 0, λ = 0.1: 0.023846 (finite differences) vs 0.023847 (formula). At σ = 0.4, ρ = 0.5, λ = 0.2: 0.04447 vs 0.04449 [E1 `derivative_check`]. (The finite difference is taken backward in θ₂, hence the sign.)

**Consequence.** Hierarchy violations (ρ > 0) and magnitude noise (σ > 0) can move the optimum toward the faithful dictionary, but never all the way to it. The optimum always keeps some tilt. Our numerical optima confirm this: even at ρ = 2 (the child appears alone twice as often as with the parent), θ₂* = 88.7° and the parent latent under-reads the parent by 7% on P∧C inputs [E1 `rho_sweep`].

### 4.3 Co-occurrence rate does not matter

**Proposition 3.** With d₁ = e_P fixed, the minimiser of L over d₂ does not depend on q.
*Proof.* By Lemma A, the P-only term does not depend on d₂. The P∧C and C-only terms enter L with weights πq and πqρ, so after dividing by πq the objective over d₂ is free of q. ∎

When d₁ is also free, the parent atom tilts slightly (|θ₁*| ≤ 3°), so the invariance is only approximate. Numerically, θ₂* stays within 52.7°–53.3° and recall_PC within 0.66–0.69 as q goes from 0.1 to 0.9 (σ = 0.2, λ = 0.1) [E1 `q_sweep`]. Trained SAEs give θ₂ = 54.2°–54.4° [E2 `q_sweep`].

**Interpretation.** Absorption is driven by whether the hierarchy is ever violated (ρ) and by how variable the magnitudes are (σ, §5.1). It is *not* driven by how strongly the concepts co-occur. A child that accompanies its parent on 10% of occurrences is absorbed as much as one that accompanies it on 90%.

### 4.4 A first-order transition in the violation rate ρ

**Proposition 4.** At fixed magnitudes (a, c), let θ = atan(c/a). The faithful dictionary has lower expected loss than the fully absorbed dictionary (e_P, x/‖x‖) if and only if ρ > ρ†, where

  ρ† = [λ(a + c − √(a² + c²)) − λ²/4] / [(c − λ/2)² − (c sin θ − λ/2)₊²].   (4)

*Proof.* The P-only term is equal for the two dictionaries (Lemma A). Per P∧C input, the faithful dictionary loses Δ = ℓ_F − ℓ_A, the numerator (Prop. 1). Per C-only input x = c e_C:
- the faithful dictionary loses c² − (c − λ/2)²;
- the absorbed dictionary can only use d₂, whose correlation with x is c sin θ, so it loses c² − (c sin θ − λ/2)₊².

The difference is the denominator. The C-only rate is ρ times the P∧C rate. ∎

For a = c = 1, (4) gives ρ† ≈ 1.17λ for small λ. Specifically, ρ† = 0.059, 0.119 and 0.243 for λ = 0.05, 0.1 and 0.2.

Prop. 4 compares two specific dictionaries. The true optimum also has *tilted* near-faithful dictionaries available, which do better than the exactly faithful one (Prop. 2), so the true transition comes earlier. A fine sweep at σ = 0 (ρ in steps of 0.01, common random numbers) gives a **discontinuous** jump [E1b: `results/e1b_rho_fine.json`, Fig. 3]:

| λ | ρ† | jump between ρ = | θ₂* just before → just after | recall_PC before → after |
|---|---|---|---|---|
| 0.05 | 0.059 | 0.03 and 0.04 | 45.4° → 60.4° | 0 → 1 |
| 0.1 | 0.119 | 0.06 and 0.07 | 45.8° → 54.8° | 0 → 1 |
| 0.2 | 0.243 | 0.13 and 0.14 | 48.9° → 54.0° | 0 → 1 |

So the observed transition is at ρ_obs ≈ 0.55–0.6 ρ†. It scales linearly with λ, as (4) predicts. Before the jump, the optimum stays nearly fully absorbed (θ₂* drifts by ≤ 4°) and the parent latent never fires on P∧C. The model is either silent on P∧C or fires on all of it. We do not have a closed form for ρ_obs. Deriving it would require solving for the optimal tilted competitor.

### 4.5 Absorption cannot be certified from activations

**Proposition 5 (non-identifiability).** Consider two generative models on ℝ^d:

- **M_H (hierarchy):** orthonormal features e_P, e_C; σ = 0, ρ = 0, a = c = 1. Inputs are e_P with probability π(1−q), e_P + e_C with probability πq, and 0 otherwise.
- **M_X (exclusive):** two mutually exclusive features f_A = e_P and f_B = (e_P + e_C)/√2. f_A is active alone with magnitude 1 and probability π(1−q). f_B is active alone with magnitude √2 and probability πq.

The two models induce *identical* distributions over x. The optimal two-atom dictionary is therefore the same under both: (e_P, (e_P + e_C)/√2): Prop. 1 shows it beats the faithful dictionary, and §4.1 finds numerically that it is the global optimum. Under M_H, latent 1 is a parent latent with **recall 1 − q** on its concept, and every miss is silent: it falls on a P∧C input where no other "P" latent fires. Under M_X, the same latent has **recall 1** on its concept f_A. Consequently, no test T whose input is the activation distribution, samples from it, or any function of these (including trained SAE weights and codes) can both:
- (i) flag every dictionary with a recall gap above some δ < q, and
- (ii) never flag a dictionary whose latents all have perfect recall.

*Proof.* The input distributions are equal by construction: e_P + e_C = √2 · f_B. T therefore returns the same output under both models, but (i) and (ii) require different outputs. ∎

The construction extends beyond two features. Given any data whose optimal dictionary absorbs a hierarchy, re-describe each absorbed P∧C configuration as its own "feature" pointing along the absorbed atom, and make the features mutually exclusive. This gives an activation-equivalent model whose features are non-orthogonal. What separates M_H from M_X is the **geometry prior**, namely whether concepts are orthogonal, or **information outside the activations**: concept labels, or interventions that show the model's downstream computation uses a P component on P∧C inputs. The orthogonality of hierarchical concepts reported by Park et al. (2025) is such a prior. However, superposition makes features non-orthogonal in general (Elhage et al., 2022), so the prior cannot be assumed blindly. In short, an SAE reading can certify positive claims, but it cannot certify negative claims without labels, causal evidence or a geometric assumption.

## 5 Experiments

All code is numpy/scipy on a 4-core CPU, and the full run takes about 20 minutes. Commands are in `README.md`.

- **E1** computes exact optima of (1) on N = 12,000 samples per configuration, with all-zero inputs removed because they have zero loss under every dictionary.
- **E1b** is the fine ρ sweep (N = 20,000).
- **E2** trains ReLU SAEs with 2 latents on the model embedded by a random rotation in ℝ^8. Training uses unit-norm decoder rows, no decoder bias, Adam (lr 3e−3, linear decay over the last 20%), batch 512 and 5,000 steps. We keep the best of 3 seeds by training loss and evaluate on 40,000 fresh samples.
- **E3** is the many-feature model in §5.3.

### 5.1 Magnitude variability produces partial absorption (E1, E2; Fig. 1)

At σ = 0 the optimum is fully absorbed (§4.1). As σ grows, θ₂* rises continuously toward 90°, and the parent latent begins to fire on some P∧C inputs. The table below gives exact optima [E1 grid; `results/e1_theory.json`]. Precision is 1.0 in all 35 grid cells.

| CV σ | λ = 0.02: θ₂* / recall_PC | λ = 0.1: θ₂* / recall_PC | λ = 0.4: θ₂* / recall_PC |
|---|---|---|---|
| 0.0 | 45.0° / 0.00 | 45.0° / 0.00 | 45.0° / 0.00 |
| 0.05 | 47.1° / 0.30 | 45.7° / 0.07 | 45.0° / 0.00 |
| 0.1 | 50.6° / 0.71 | 47.7° / 0.37 | 45.6° / 0.08 |
| 0.2 | 58.0° / 0.88 | 53.3° / 0.69 | 48.1° / 0.34 |
| 0.4 | 73.4° / 0.92 | 65.2° / 0.81 | 57.0° / 0.61 |
| 0.6 | 82.9° / 0.90 | 75.3° / 0.84 | 64.8° / 0.67 |

**Mechanism.** The absorbed child atom points along the *typical* P∧C direction. When a P∧C input has a relatively large parent magnitude (a/c above the atom's slope), the residual has a positive e_P component, and the parent latent fires to supply it. When a/c is below the slope, the parent latent stays silent. So **whether the parent latent fires on P∧C depends on the incidental ratio a/c**, not on whether the parent is present. A larger λ makes absorption stronger at every σ, because sparsity pressure is what drives it. Recall is not perfectly monotone in σ at small λ (0.92 at σ = 0.4 vs 0.90 at σ = 0.6, λ = 0.02). We have not analysed this.

**Trained SAEs reach the exact optimum.** Across the 12 cells of the E2 grid (σ ∈ {0, 0.1, 0.2, 0.4}, λ ∈ {0.05, 0.1, 0.2}):
- the trained SAE's child-atom angle is within 1.5° of the exact optimum (exactly equal at σ = 0);
- its recall_PC is within 0.12 of the exact value;
- the decoder stays in the (e_P, e_C) plane (off-plane mass < 4 × 10⁻⁸);
- the amortisation gap, i.e. SAE loss versus exact codes on the SAE's own dictionary, is 0.25–0.44% of the loss [E2: `results/e2_sae.json`].

The SAE's angle is consistently slightly *higher* (0.75°–1.47°), so the amortised encoder absorbs slightly less than the exact optimum. Precision is 1.0 in all E2 cells.

### 5.2 Hierarchy violations and magnitude under-reading (E1, E2; Fig. 2)

Fig. 2b–c and the E1/E2 ρ sweeps (λ = 0.1) show the following:
- At σ = 0, recall_PC is 0 for ρ ≤ 0.05 and 1 for ρ ≥ 0.1 (the first-order jump of §4.4).
- At σ = 0.2, recall_PC rises smoothly: 0.68, 0.74, 0.83 and 0.98 at ρ = 0, 0.02, 0.05 and 0.1 [E1 `rho_sweep`].

**Full recall does not imply faithful readout.** The parent latent's activation on P∧C inputs, relative to the true parent magnitude (rel_act_PC), is 0.54 at ρ = 0.1, 0.75 at ρ = 0.2, 0.87 at ρ = 0.5 and 0.93 at ρ = 2 (σ = 0) [E1]. Trained SAEs give 0.544, 0.757, 0.872 and 0.933 [E2]. The pure-shrinkage reference is 1 − (λ/2)/a = 0.95. So at ρ = 0.1, a latent with perfect recall reports only about half of the parent's magnitude on co-occurrence inputs. The rest is carried by the child latent's tilted decoder.

Fig. 2a shows the q-invariance (Prop. 3). The exact optimum and the trained SAEs are flat in q.

### 5.3 Many features: silent false negatives and the limits of weight-only detection (E3; Fig. 4)

**Setup.** We use 48 ground-truth features in ℝ^64:

- **12 hierarchical pairs** (P(parent) = 0.06), covering a full factorial of q ∈ {0.3, 0.6}, σ ∈ {0, 0.2, 0.4} and ρ ∈ {0, 0.5}.
- **4 correlated, non-hierarchical pairs**: each feature alone with probability 0.04, jointly with probability 0.03.
- **4 mutually exclusive pairs** with non-orthogonal directions (cos = 1/√2) and magnitude √2 on the second feature. These are the M_X construction of Prop. 5; ground truth says they have *no* false negatives.
- **8 independent features.**

Directions are orthonormal except for the exclusive pairs. We train matched-width (48-latent) ReLU SAEs with λ = 0.1 for 5,000 steps over 5 seeds, which gives 240 latents. Latents are matched to features by Hungarian assignment on decoder cosine.

**Weight-only score.** We use the score

  A_i = max_{j≠i} [ cos(d_i, d_j) − (w_iᵀd_j)/(w_iᵀd_i) ],

where d are decoder rows and w are encoder columns. It is large when another latent's decoder overlaps latent i's direction but latent i's encoder does not respond to that decoder. This is the weight signature of absorption.

**Results** [E3: `results/e3_many.json`]:

- **Silent false negatives.** 18 of 240 latents have recall < 0.9, and *all 18* are hierarchy parents. All 240 latents have precision 1.0 (minimum 1.000). By configuration, parent-latent recall restricted to co-occurrence inputs is:
  - 0.00 for σ = 0, ρ = 0 (full absorption; overall recall 0.71 at q = 0.3 and 0.40 at q = 0.6, i.e. ≈ 1 − q, as Prop. 5 states);
  - 0.73–0.74 for σ = 0.2, ρ = 0;
  - 0.83–0.84 for σ = 0.4, ρ = 0;
  - 0.97–1.00 for every ρ = 0.5 configuration.

  This is the ordering predicted by §4–5.2. The absorbing child latents point at 45° to the child feature (minimum cosine 0.707), but they still have recall ≥ 0.976 for the child, so naive cosine-matching would label them as good child latents.
- **Detection versus false alarms.** Excluding exclusive pairs, A separates low-recall latents from the rest with AUROC 0.969. With exclusive pairs included, AUROC falls to 0.865. The exclusive-pair latents score A ≈ 0.52–0.69 (mean 0.68 for the first feature, 0.63 for the second), which is in the same range as fully absorbed parents (≈ 0.65–0.70), yet they have recall 1.000 (Fig. 4b). This is Prop. 5 made concrete: the weights of an absorbed hierarchy and of an exclusive non-orthogonal pair look the same. A simpler score, the maximum decoder cosine, behaves similarly (AUROC 0.939 without exclusive pairs, 0.814 with them).
- **The information is in the activations.** A least-squares linear probe for each parent concept, trained on 20,000 labelled raw activations, has precision 1.0 and recall ≥ 0.99 for σ ≤ 0.2. That includes the σ = 0, ρ = 0 parents, where the SAE latent's recall is 0.40–0.71. At σ = 0.4, the probe's recall falls to 0.86–0.88. We attribute this to our crude thresholded least-squares probe (fixed threshold 0.5 under magnitude noise), not to missing information. A logistic probe would be the fair baseline, and we did not run one.

## 6 Discussion

**What this says about "reading the model's thoughts".** On the question *can we really read what the model is thinking?*, the results give a split answer. Sparse dictionaries are a trustworthy *positive* reader: in every configuration we ran, a firing parent latent meant the parent was present (precision 1.0). They are not a trustworthy *negative* reader. A silent latent is consistent with the concept being present and carried by a more specific latent's direction. Whether that happens is decided by data statistics the analyst usually cannot see: whether the hierarchy is ever violated (ρ), and how variable feature magnitudes are (σ). It is not decided by how semantically "strong" the relation is (q). By Prop. 5, no amount of staring at activations or SAE weights can settle the matter without an extra assumption or extra information. Interpretability-based safety arguments of the form "the X latent never fired" therefore need either:
- a labelled recall audit for X on held-out data where X is known to be present, or
- interventional evidence, or
- an explicit and defended geometric prior.

**Falsifiable predictions for LLM SAEs.** Our model predicts the following for real SAEs, for example on the first-letter task of Chanin et al. (2024), where the parent is "starts with letter L" and the children are token identities.

- **P1 (q-invariance).** Across child tokens, absorption rates do *not* correlate with P(child | parent), after controlling for ρ and magnitude CV.
- **P2 (ρ = 0 ⇒ absorption).** Token-identity children never occur without their first letter (ρ = 0), so absorption should be common and should persist across widths and L0 values. Chanin et al. report the latter. Children with ρ > 0, such as semantic features that can appear without the lexical parent, should be absorbed much less, with a sharp rather than gradual transition in ρ.
- **P3 (magnitude CV ⇒ partial absorption).** Among absorbing pairs, the fraction of co-occurrence tokens on which the parent latent still fires should increase with the across-context CV of the child latent's activation. On each token, the parent should fire when the parent-to-child magnitude ratio is high.
- **P4 (under-reading).** Where the parent latent does fire on co-occurrence tokens, its activation should be systematically lower than on parent-only tokens by more than the shrinkage scale. The decoder of the child latent should have a positive component along the parent latent's decoder.
- **P5 (false alarms).** Weight-only absorption scores should flag some pairs whose labelled audit shows no recall gap. These are pairs of mutually exclusive concepts with correlated directions.

**Test protocol (pre-registered; not run here).** We did not run this protocol; it requires LLM activations and GPU compute that this environment lacks.

1. Use public SAEs (e.g. from SAEBench) on Gemma-2-2B for the first-letter task.
2. For each letter-parent latent and each child token, estimate:
   - ρ from a lexicon, where the "child without parent" events come from a semantic child concept;
   - q from corpus frequencies;
   - σ from the CV of the child latent's activation across 1,000 contexts.
3. Measure recall_PC with SAEBench-style labelled probes.
4. Regress recall_PC on (log q, ρ, σ).
   - P1 predicts a q-coefficient that is not significantly different from 0.
   - P3 predicts a positive σ-coefficient.
5. Test P4 with paired comparisons of parent activation magnitude on co-occurrence versus parent-only tokens.
6. Test P5 by computing the score A for all latent pairs and auditing the top 100 flagged pairs with labels.

Rejecting P1 or P3 would show that the minimal model misses a dominant mechanism in real SAEs.

**Implications for SAE design.** Prop. 2 shows that faithfulness is never stationary under an L1 sparsity objective whenever parents and children co-occur. Architectural fixes therefore have to change the objective, not the optimiser. Examples are nesting, as in Matryoshka SAEs (Bussmann et al., 2025), hierarchy-aware penalties, or L0 penalties without shrinkage. Prop. 3 suggests that reweighting by co-occurrence frequency would not help. Prop. 4 suggests that training data containing even a few percent of hierarchy violations could remove absorption altogether, at the cost of some magnitude under-reading. That last suggestion is a hypothesis, and we did not test it.

## 7 Limitations

- **Minimal model.** Our theory covers one parent and one child with orthogonal directions, a two-atom dictionary and an L1 penalty. We did not analyse multiple children per parent, deeper hierarchies, non-orthogonal hierarchical directions, TopK/JumpReLU/Gated SAEs (Gao et al., 2024; Rajamanoharan et al., 2024), or dictionaries wider than the number of features. L0-style penalties remove shrinkage but keep the sparsity incentive behind Prop. 1. We expect, but have not shown, that full absorption still dominates under them.
- **Numerical optima.** The global-optimum claims in §4.1 and §5.1 come from grid search plus local refinement (resolution about 0.1° in θ₂ and 0.14° in θ₁ after refinement) on finite samples (N = 12,000–20,000). They are not proofs. Sampling noise causes small non-monotonicities, such as the 0.90 vs 0.92 recall noted in §5.1.
- **ρ† is not the transition.** Prop. 4 compares two specific dictionaries. The actual first-order transition happens at about 0.55–0.6 ρ†, and we have no closed form for it.
- **Magnitude model.** Magnitudes are clipped Gaussians with mean 1 and are independent between parent and child. Correlated magnitudes, which may be common in LLMs, change the a/c ratio distribution and therefore the degree of partial absorption.
- **E3 is still a toy.** It has 48 features, one λ, one training recipe and 5 seeds. Results are pooled over seeds without confidence intervals. The weight-only score A is one heuristic among many. The linear-probe baseline is a crude thresholded least-squares probe, and a logistic probe would be fairer.
- **No LLM experiments.** We have no GPU and no LLM access here. Predictions P1–P5 are untested, and the extent to which real-SAE absorption follows our mechanisms is unknown.
- **Scope of Prop. 5.** Prop. 5 excludes tests that use only the activation distribution. It does not exclude tests that use the model's downstream computation, for example checking whether ablating the parent direction on P∧C inputs changes outputs the way it does on P-only inputs. Such tests use information beyond the activation distribution and are a promising route.

## 8 Conclusion

In a model small enough to solve exactly, feature absorption is the optimum that sparse coding is designed to find, not a training bug. Full absorption is optimal when feature magnitudes are fixed and the hierarchy is never violated. Partial absorption follows from magnitude variability. Faithfulness is never a stationary point while parent and child co-occur. The co-occurrence rate itself does not matter, and a small rate of hierarchy violations triggers a discontinuous switch. Because an absorbed hierarchy is activation-equivalent to a pair of exclusive non-orthogonal features, the resulting silent false negatives cannot be certified away from activations or weights alone. Reading a model's "thoughts" with sparse dictionaries is reliable for what latents say. What they leave unsaid needs labels, interventions or a defended geometric prior before it can be trusted.

## Reproducibility and Data Provenance

Every numerical claim comes from scripts in `code/` that we ran in this environment:

| results file | script | used in |
|---|---|---|
| `results/e1_theory.json` | `exp1_theory.py` | §4.1–4.3, §5.1, §5.2 |
| `results/e1b_rho_fine.json` | `exp1b_rho_fine.py` | §4.4 |
| `results/e2_sae.json` | `exp2_sae_two_feature.py` | §5.1, §5.2 |
| `results/e3_many.json` | `exp3_many_features.py` | §5.3 |

Figures are produced by `make_figures.py`: `results/fig1_phase.png`, `fig2_q_rho.png`, `fig3_rho_transition.png` and `fig4_many.png`. Seeds are fixed in each script. No external data is used.

## References

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Tamkin, A., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features/
- Bussmann, B., Nabeshima, N., Karvonen, A., & Nanda, N. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *Proceedings of ICML 2025*, PMLR 267. arXiv:2503.17547.
- Chanin, D., Wilken-Smith, J., et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507. NeurIPS 2025.
- Chanin, D., Dulka, T., & Garriga-Alonso, A. (2025a). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
- Chanin, D., & Garriga-Alonso, A. (2025). Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders. arXiv:2508.16560. ICML 2026 (PMLR 306).
- Cui, J., Zhang, Q., Wang, Y., & Wang, Y. (2025). On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond. arXiv:2506.15963.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600. ICLR 2024.
- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*; arXiv:2209.10652.
- Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093. ICLR 2025.
- Hindupur, S. S. R., Lubana, E. S., Fel, T., & Ba, D. (2025). Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry. arXiv:2503.01822. NeurIPS 2025.
- Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N. (2025). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. *Proceedings of ICML 2025*, PMLR 267. arXiv:2502.16681.
- Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, Y.-T., Farrell, E., et al., & Nanda, N. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *Proceedings of ICML 2025*, PMLR 267. arXiv:2503.09532.
- Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. ICLR 2025. arXiv:2502.04878.
- Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B., & Bachem, O. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. *Proceedings of ICML 2019*, PMLR 97, 4114–4124. arXiv:1811.12359.
- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. ICLR 2025. arXiv:2403.19647.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609. doi:10.1038/381607a0.
- Park, K., Choe, Y. J., Jiang, Y., & Veitch, V. (2025). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. ICLR 2025. arXiv:2406.01506.
- Rajamanoharan, S., Conmy, A., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014. Published at NeurIPS 2024 as "Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders".
- Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., et al. (2025). Open Problems in Mechanistic Interpretability. arXiv:2501.16496.
- Tang, Y., Saini, H., Liao, Y., Liu, D., et al. (2025). On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability. arXiv:2512.05534.
- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Wright, B., & Sharkey, L. (2024). Addressing Feature Suppression in SAEs. AI Alignment Forum (MATS Winter 2023–24). https://www.alignmentforum.org/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes
