# Interpolation Is Relative to a Representation: Vacuous Input Hulls, Exact Anti-Interpolation in Modular Addition, and What a Readout-Hull Certificate Does and Does Not Add to the Logit Margin

**Target venue:** International Conference on Learning Representations (ICLR 2027), main track

**Revision note (round 2).** This version replaces the round-1 claim that grokking networks "generalize first and interpolate later". New controls (Section 6.2) show that this certification lag is reproduced, to within one 500-step checkpoint in all 14 grokked MLP runs, by a single fixed logit-margin threshold. The paper now reports the lag as a large-margin phenomenon. The response to reviewers (`reviews/response_r1.md`) lists every change.

---

## Abstract

People keep asking whether neural networks "really generalize" or "only interpolate". The question has no answer until one says *in which representation* interpolation is measured, and *which notion* of interpolation is meant. We give one negative result, one measurement and one set of controls.

**(i) Theory (negative).** Proposition 1 is elementary. If the input set is in convex position (one-hot or multi-hot codes, points on a sphere), no unseen input ever lies in the convex hull of a training set, so hull-based "interpolation" at the input is decided before any model is trained. Proposition 2 is the main formal result. On modular addition with one-hot inputs, the input-space 1-nearest-neighbour rule and the nearest-class-hull rule are *exactly 0%* accurate on held-out pairs. The exception has probability 6.8·10⁻³³ per pair at p = 53 and a 50% split. Networks nonetheless reach 100%.

**(ii) Measurement.** We measure layer-wise nearest-class-hull accuracy I_ℓ, together with a readout-hull certificate. The certificate is a standard Cauchy–Schwarz margin bound. It checks whether a test representation lies close enough to its class's training hull that no perturbation of that size could flip the decision. We show that the full-space certificate C is *not invariant* to readout-invisible directions: appending invisible units drives C on digits from 0.876 to 0.001 without changing a single logit. We therefore introduce a readout-subspace version C_vis that is invariant to such units. Exact membership in C_vis means that the test point's logit-difference vector is a convex mixture of the logit-difference vectors of same-label training points.

**(iii) Findings** (CPU-scale numpy/PyTorch models; all code and outputs released).

- *Where interpolation appears.* On modular addition, I_ℓ is exactly 0 at the input and ≤ 0.07 at layer 1. At the last hidden layer it tracks test accuracy.
- *Controls.* We ran 17 MLP runs (5 seeds at fraction 0.5; fractions 0.6 and 0.7; weight decay 0.3, 1 and 3) and three grokked 1-layer transformers. C crosses 50% 3,450 steps after test accuracy at fraction 0.5 (95% CI [2,940, 3,960], 5 seeds), 2,250 at fraction 0.6 and 1,583 at fraction 0.7, and the lag tends to be shorter with stronger weight decay (2,833 at weight decay 3; wide CI). However, the fraction of test points whose logit margin exceeds a fixed τ = 8 reproduces C's 50% crossing within ≤ 500 steps in all 14 grokked MLP runs (correlation of lags r = 0.97). A single τ = 6.25 does the same in all 3 grokked transformers. The "certification lag" is therefore a growth-of-margin phenomenon, not evidence for a distinct uncertified mode of generalization. With weight decay 0.3, networks reach 80–85% test accuracy with C ≈ 0 and small test margins.
- *Information beyond the margin.* At the checkpoint where C ≈ 0.5, the margin predicts certification per point with AUROC 0.86–0.91, not 1. On digits (5 seeds; an MLP and a small CNN; six perturbations), a hull-contrast score adds significantly to the logit margin for error detection in all 12 model×perturbation conditions (leave-one-seed-out gain 0.02–0.21 AUROC; p < 0.05 in 12/12). It beats max-softmax, energy, Mahalanobis and deep-kNN baselines in 11/12 conditions.

Our answer to the title question is deflationary. Input-space "interpolation" is vacuous or exactly misleading for discrete tasks. In the last hidden layer, test points of well-generalizing networks become relative interpolations. Whether they become *certified* ones is, to first order, a matter of logit-margin growth. Hull geometry nonetheless carries modest per-point information about errors that the margin does not.

## 中文摘要

“神经网络是真正泛化，还是本质上是高级插值？”在说明*在哪个表示中*、用*哪种*插值概念之前，这个问题没有答案。本文给出一个负面理论结果、一种度量，以及一组对照实验。

**（i）理论（负面）。** 命题1较为初等：若输入集合处于凸位置（如 one-hot/multi-hot 编码、球面上的点），任何未见输入都不在训练集凸包内，因此输入层的凸包“插值”在训练之前就已确定。命题2是本文主要的形式化结果：在 one-hot 编码的模加法上，输入空间的 1-最近邻和最近类凸包分类器在留出样本上的准确率*恰为 0%*。例外情形的概率在 p = 53、50% 划分下为每对 6.8·10⁻³³。而网络可达 100%。

**（ii）度量。** 我们逐层度量最近类凸包准确率 I_ℓ，并使用一个读出层凸包证书。它本质上是标准的 Cauchy–Schwarz 间隔界：检验测试表示是否离同类训练凸包足够近，使得任何同等大小的扰动都无法改变决策。我们证明全空间证书 C 对读出层不可见的方向*不具不变性*：在 digits 上添加不可见单元可使 C 从 0.876 降到 0.001，而任何 logit 都不变。因此我们引入读出子空间版本 C_vis，它对这类单元不变。C_vis 意义下的精确成员关系意味着：测试点的 logit 差向量是同类训练点 logit 差向量的凸组合。

**（iii）结果**（CPU 规模的 numpy/PyTorch 模型，代码与输出全部公开）。

- *插值出现在哪里。* 在模加法上，I_ℓ 在输入层恰为 0，第1层不超过 0.07，最后隐藏层则与测试准确率同步。
- *对照实验。* 共 17 个 MLP 运行（训练比例 0.5 下 5 个种子；比例 0.6、0.7；权重衰减 0.3、1、3）和 3 个单层 Transformer。C 越过 50% 的时间比测试准确率晚：比例 0.5 时 3,450 步（95% CI [2,940, 3,960]，5 个种子），0.6 时 2,250 步，0.7 时 1,583 步；权重衰减越强，滞后趋于越短（权重衰减 3 时为 2,833 步，置信区间较宽）。然而，“logit 间隔超过固定阈值 τ = 8 的测试点比例”在全部 14 个 grokking 的 MLP 运行中，都在 ≤ 500 步内复现了 C 的 50% 越过时间（滞后相关系数 r = 0.97）；单一阈值 τ = 6.25 在全部 3 个 grokking 的 Transformer 中同样如此。因此“认证滞后”是间隔增长现象，而不是存在一种独立的、不可认证的泛化模式的证据。权重衰减为 0.3 时，网络测试准确率达到 80–85%，但 C ≈ 0，测试间隔也很小。
- *间隔之外的信息。* 在 C ≈ 0.5 的检查点，间隔对逐点认证的预测 AUROC 为 0.86–0.91，而非 1。在 digits 上（5 个种子；MLP 与小型 CNN；6 种扰动），凸包对比分数在全部 12 个“模型×扰动”条件下都显著补充了 logit 间隔对错误检测的信息（留一种子 AUROC 增益 0.02–0.21；12/12 条件 p < 0.05）。它在 11/12 条件下优于最大 softmax、能量、Mahalanobis 和深度 kNN 基线。

我们对题目问题的回答是收缩性的：对离散任务，输入空间的“插值”是空洞的，甚至恰好误导。在最后隐藏层，泛化良好网络的测试点成为相对意义上的插值；它们能否成为*被认证的*插值，在一阶上取决于 logit 间隔的增长。尽管如此，凸包几何仍携带少量间隔所不具备的逐点错误信息。

---

## 1 Introduction

Whether neural networks "genuinely generalize" or "merely interpolate" is debated constantly. One camp argues that deep networks fit their training data densely and generalize by local interpolation over a task manifold ("direct fit"; Hasson et al., 2020). Others read failures on counterfactual or compositional variants of familiar tasks as evidence of recitation rather than reasoning (Wu et al., 2024; Dziri et al., 2023; Lake & Baroni, 2018). Throughout, "interpolation" is used as if it were a well-defined property of a model's behaviour on a test point.

It is not. The standard formal definition calls a test point an interpolation if it lies in the convex hull of the training inputs. Balestriero, Pesenti and LeCun (2021) showed that under this definition test points in high dimension almost surely fall outside the hull. Yousefzadeh (2021; 2022a; 2022b) argued that interpolation in this sense is inadequate for understanding deep-learning generalization, and found that test images lie outside but close to the training hull in feature space. Bonnasse-Gahot (2022) found most test points inside the hull of a low-dimensional latent code of the last hidden layer, but argued that membership was not the relevant criterion. Hwang and Shin (2024) used hull distance as an uncertainty score.

This paper asks three narrower questions that these works leave open:

1. Can input-space interpolation be not merely rare but *provably vacuous*, or even *exactly wrong*, on a natural task?
2. In which layer of a trained network do test points become interpolations, and does a decision-linked notion of interpolation carry information beyond the network's own logit margin?
3. During grokking, when does decision-linked interpolation appear relative to generalization, and is that timing explained by simpler quantities?

Our contributions, stated with their limits:

- **C1 (Vacuity; Proposition 1, elementary).** For inputs in convex position, hull membership of unseen inputs is impossible. This follows directly from extremality and is implicit in Balestriero et al. (2021). We state it because it settles the discrete-input case completely. It concerns the *one-hot input*, not learned embeddings.
- **C2 (Exact anti-informativeness; Proposition 2).** On one-hot modular addition, input-space 1-NN and nearest-class-hull rules are *exactly* wrong on every held-out pair, except with probability 6.8·10⁻³³ per pair at p = 53 and a 50% split. Networks reach 100%. We are not aware of a prior exact statement of this kind. It is the paper's main formal result.
- **C3 (Readout-hull certificates and their invariance).** Proposition 3 is a standard affine-margin/Cauchy–Schwarz bound applied to a hull point; we claim no novelty for the inequality itself. We show that the natural full-space statistic C depends on readout-invisible directions (Proposition 4, with an empirical construction). We give a readout-subspace version C_vis (Proposition 3′) that is invariant to them. Exact C_vis membership has a literal reading: the test logit-difference vector is a convex mixture of same-label training logit-difference vectors.
- **C4 (Controls for the certification lag).** In 17 MLP runs and 3 transformers, we compare the certificate's timing with margin thresholds, test loss and NC1. **The lag is reproduced by a fixed logit-margin threshold.** We report it as a margin phenomenon and withdraw the round-1 interpretation ("generalization that is not interpolation, then consolidation into interpolation").
- **C5 (Per-point information beyond the margin).** On digits, with an MLP and a CNN, six perturbations, five seeds, per-seed confidence intervals and standard baselines, hull-relative scores add significantly to the logit margin for error detection.

Our position on the title question is deflationary. The question is ill-posed until a representation and a notion of interpolation are named. Once they are named, input-space interpolation is uninformative for discrete algorithmic tasks, and last-layer interpolation tracks accuracy. Certified interpolation largely tracks the logit margin. We make no claims about large models; Section 7.3 states predictions only.

## 2 Related Work

**Interpolation and the convex hull.** Balestriero et al. (2021) formalized interpolation as convex-hull membership. They showed it almost surely fails in high dimension, with the governing quantity being the dimension of the smallest affine subspace containing the data, and concluded that the definition is a poor indicator of generalization. Yousefzadeh (2021) reported that most test images lie outside the training hull in pixel, wavelet and feature spaces. Yousefzadeh (2022b), a short abstract, argues directly that interpolation is not adequate to understand deep-network generalization. Yousefzadeh (2022a) found that last-hidden-layer test features are close to the training hull. Bonnasse-Gahot (2022) found most test points inside the hull in an autoencoder latent space of the last layer. Hwang and Shin (2024) proposed hull-based uncertainty for test selection.

What we add relative to these works:

- Proposition 2 is an exact, below-chance failure of relative input-space interpolation on a natural task.
- We show that a decision-linked hull certificate must be restricted to the readout-visible subspace to be well defined.
- We give controls showing that its training-time behaviour is largely explained by the logit margin.
- We measure per-point information beyond the margin against standard baselines.

Representation-space hull proximity tracking accuracy, hull distance as an uncertainty score, and late-phase feature compression are all known. We do not claim them.

**Margins and late-phase training.** Gradient descent on cross-entropy drives homogeneous networks toward maximum-margin directions (Lyu & Li, 2020). Lyu et al. (2024) prove that grokking can arise from a transition between an early kernel-like regime and a late max-margin regime, with weight decay controlling the timing. Nanda et al. (2023) describe a "cleanup" phase after circuit formation. Our controls (Section 6.2) put the certification lag inside this known picture: it is reproduced by a margin threshold.

**Neural collapse.** Papyan, Han and Donoho (2020) documented the collapse of last-layer features to class means in the terminal phase of training. In our modular-addition runs the NC1 statistic does *not* decrease after grokking (Section 6.2), so collapse in the NC1 sense does not explain the certificate's timing here. Margin growth does.

**Error and OOD detection.** We compare with maximum softmax probability (Hendrycks & Gimpel, 2017), the energy score (Liu et al., 2020), class-conditional Mahalanobis distance (Lee et al., 2018) and deep k-nearest-neighbour distance (Sun et al., 2022). We use them as baselines for a *property of a measurement*, not to claim a state-of-the-art detector.

**Extrapolation by architecture.** Xu et al. (2021) proved that ReLU MLPs become linear far from the training support. That is a statement about the function outside the training support in input space. Ours concerns where test points sit relative to the training set in representation space.

**Grokking.** Power et al. (2022) discovered delayed generalization. Nanda et al. (2023) reverse-engineered the Fourier algorithm of transformers on modular addition. Liu et al. (2022) related generalization to structured representations, Liu, Michaud and Tegmark (2023) to weight norm, and Gromov (2023) gave analytic solutions for MLPs.

**Interpolation regime and memorization.** "Interpolation" also names fitting the training data exactly (Belkin, 2021). Zhang et al. (2017) showed that networks can fit random labels. We use the geometric sense of the word. Our random-label control is the analogue of Zhang et al.'s experiment within our metrics.

## 3 Setting and Definitions

Let f(x) = Wᵀφ(x) + b be a classifier with K classes, penultimate representation φ: 𝒳 → ℝ^d, readout W = [w₁,…,w_K] ∈ ℝ^{d×K} and bias b. φ_ℓ denotes the representation after layer ℓ, with φ₀ the input. 𝒯 = {(x_i, y_i)} is the training set and 𝒯_y = {x_i : y_i = y}.

**Definition 1 (hull interpolation; Balestriero et al., 2021).** x is an *interpolation in layer ℓ* if φ_ℓ(x) ∈ conv{φ_ℓ(x_i) : x_i ∈ 𝒯}. It is a *class-conditional interpolation* if φ_ℓ(x) ∈ conv φ_ℓ(𝒯_y).

**Definition 2 (relative hull distance).** d_ℓ(x, c) = dist(φ_ℓ(x), conv φ_ℓ(𝒯_c)) / r_ℓ, where r_ℓ is the RMS distance of training representations to their mean. The normalization is a convention. The classifier below is invariant to it, so our layer comparisons rest on I_ℓ and not on raw d values.

**Definition 3 (nearest-class-hull classifier, profile, depth).** ĉ_ℓ(x) = argmin_c d_ℓ(x, c).

- The *interpolation profile* is the vector of accuracies I_ℓ of ĉ_ℓ.
- The *interpolation depth* is the smallest ℓ with I_ℓ within ε = 0.02 of the network's accuracy.

I_ℓ is scale-free in each layer, because argmin is unaffected by r_ℓ.

**Definition 4 (certified fractions).** Using Proposition 3 with h the nearest point of conv φ(𝒯_y), a test point is *certified* if (2) holds. C is the certified fraction in the full representation space. C_vis is the same quantity after projecting all representations onto the readout-visible subspace (Proposition 3′). With the predicted label ŷ in place of y, both are label-free.

## 4 Theory

### 4.1 The hull definition is vacuous for discrete inputs (elementary)

A set S ⊂ ℝ^n is in *convex position* if every s ∈ S is an extreme point of conv S.

**Proposition 1.** Let S be in convex position, 𝒯 ⊂ S finite and x ∈ S \ 𝒯. Then x ∉ conv 𝒯.

*Proof.* conv 𝒯 ⊆ conv(S \ {x}), and x ∉ conv(S \ {x}) by extremality. ∎

**Examples.**

- (a) The vertices of {0,1}^n are in convex position. This covers one-hot codes and concatenations of one-hot blocks, so every fixed-length token sequence *under one-hot encoding* is covered.
- (b) Any subset of a sphere is in convex position.

The proposition is about the one-hot input itself. Learned embedding layers are generally not in convex position, so it says nothing about them. We checked it numerically with an exact LP: 0 of 40 held-out modular-addition inputs lay in the training hull, in all six Experiment-1 runs (`results/summary.json`, `input_lp_in_hull_frac`). Continuous data show the same in practice: on digits, 0 of 300 test images per seed lay in the hull of the 600 training images or of their own class (5 seeds).

### 4.2 Relative interpolation can be exactly anti-informative

Let p be prime and let e(a, b) ∈ {0,1}^{2p} concatenate the one-hot codes of a and b. The label is c = (a + b) mod p.

**Proposition 2.** Let (a, b) be a held-out pair with label c. Suppose the training set contains some pair (a, b′) with b′ ≠ b or some (a′, b) with a′ ≠ a. Then the input-space 1-NN classifier and the nearest-class-hull classifier ĉ₀ both assign (a, b) a label ≠ c.

*Proof.*

1. *Same-label training points share no coordinate with the test point.* If (a′, b′) ≠ (a, b) has label c and a′ = a, then b′ ≡ c − a ≡ b, a contradiction. Symmetrically b′ ≠ b. So every same-label training point is at squared distance 4.
2. *The same-label hull is at squared distance more than 2.* Any point h of the same-label hull has h_a = h_{p+b} = 0. Each block of h is a probability vector supported on at most n_c coordinates, where n_c is the number of same-label training points. So each block has squared norm at least 1/n_c, and ‖e(a, b) − h‖² = 2 + ‖h‖² ≥ 2 + 2/n_c > 2.
3. *Some other-label point is at squared distance 2.* The assumed training pair (a, b′) has label a + b′ ≢ c and lies at squared distance 2, and likewise for (a′, b).

So both the nearest point and the nearest class hull belong to a wrong class. ∎

**Exact failure probability.** The condition fails only if none of the 2(p − 1) pairs sharing a coordinate with (a, b) is in the training set. With n_tr training pairs drawn uniformly without replacement from the p² − 1 pairs other than (a, b), the probability is

  C(p² − 1 − 2(p − 1), n_tr) / C(p² − 1, n_tr).

For p = 53 this is 6.8·10⁻³³ at a 50% split (n_tr = 1,404), 2.3·10⁻²⁴ at 40% and 2.0·10⁻⁴³ at 60%, computed exactly with integer binomials. The expected number of held-out pairs violating the condition at 50% is 9.6·10⁻³⁰.

**Corollary.** On one-hot modular addition, relative input-space interpolation has accuracy exactly 0 on held-out pairs (with the probability above), which is below chance (1/p). Any network that generalizes here does so *against* input-space proximity. I₀ and input 1-NN accuracy do not depend on the network, so they are constant over training by construction. We measured I₀ = 1-NN accuracy = 0.000 in every real-label run (`max_L0_hull_nearest_acc_over_training` = 0).

### 4.3 Readout-hull certificates

**Proposition 3 (hull-margin bound; standard).** Let x have label y and let h = Σ_i λ_i φ(x_i) with x_i ∈ 𝒯_y, λ ≥ 0, Σλ_i = 1, and δ = ‖φ(x) − h‖. Write m_{yc}(v) = (w_y − w_c)ᵀv + b_y − b_c. Then for every c ≠ y:

  m_{yc}(φ(x)) ≥ Σ_i λ_i m_{yc}(φ(x_i)) − ‖w_y − w_c‖ δ,   (1)

and if min_{c≠y}[m_{yc}(h) − ‖w_y − w_c‖δ] > 0 (2), the network predicts y on x.

*Proof.* Margins are affine and Σλ_i = 1. Apply Cauchy–Schwarz to (w_y − w_c)ᵀ(φ(x) − h). ∎

**What (2) does and does not say (revised in response to review).** The conclusion "the network predicts y" can be read off the logits. With ŷ in place of y it is tautological. The content of (2) is geometric. φ(x) lies within δ of a convex mixture of same-label training representations, and the whole ball of radius δ around that mixture lies on the y side of every pairwise decision boundary. In other words, the ratio of distance-to-hull to the hull point's normalized margin is below 1. This is a definition of "decision-robust hull proximity", not a theorem about mechanism. An uncertified correct prediction is not shown to be non-interpolative. We no longer describe uncertified correct decisions as "generalization that is not interpolation".

**Proposition 3′ (readout-subspace certificate).** Let U ∈ ℝ^{d×k} have orthonormal columns spanning span{w_c − w_1 : c}. Then m_{yc}(φ) = m̃_{yc}(Uᵀφ), where m̃ uses the readout UᵀW and the same bias. Hence Proposition 3 holds with every representation replaced by Uᵀφ, and with δ_vis = dist(Uᵀφ(x), conv Uᵀφ(𝒯_y)) ≤ δ. Moreover, Uᵀφ(x) ∈ conv Uᵀφ(𝒯_y) if and only if the logit-difference vector (f_c(x) − f_1(x))_c lies in the convex hull of the logit-difference vectors of the same-label training points.

*Proof.* Each w_y − w_c lies in span U, so (w_y − w_c)ᵀφ = (w_y − w_c)ᵀUUᵀφ. Projection is a contraction that maps the hull onto the projected hull, so δ_vis ≤ δ. Write D = [w_c − w_1]_c = U S Vᵀ (thin SVD restricted to non-zero singular values). The logit differences are Dᵀφ + const = V S Uᵀφ + const, an injective affine image of Uᵀφ. Injective affine maps preserve and reflect convex-hull membership. ∎

C_vis is the more literal "interpolation of training decisions". With δ_vis = 0, the network's output on x, modulo a shift common to all classes, *is* a convex combination of its outputs on same-label training points. C_vis ≥ C pointwise for the existential version of the certificate.

**Proposition 4 (C is not invariant to readout-invisible directions).** Call the full-space certificate *existentially satisfied* if some hull point satisfies (2). Appending to φ any features with zero readout weight leaves every logit unchanged, can only decrease the existential full-space certificate, and leaves the readout-subspace certificate unchanged.

*Proof.* Take a padded hull point h′ with residual δ′. Its first-block projection is a hull point of the original representation with residual ≤ δ′ and identical margins. So padded certification implies original certification. U is unchanged by padding with zero readout rows. ∎

Section 6.4 shows the effect is large. Padding raises the relative residual, and C falls from 0.876 to 0.001 on digits while not one logit changes. C is therefore a property of the representation *and* the readout, including directions the decision ignores. We report both C (for continuity with round 1) and C_vis.

**Remark (why only the readout).** For a nonlinear head after layer ℓ < L − 1, no analogue of (1) holds without a Lipschitz bound, and the convex-combination reading is lost. For earlier layers we report only I_ℓ.

## 5 Experimental Setup

**Common settings.**

- *Models and optimiser.* Fully connected ReLU networks in numpy with softmax cross-entropy, full-batch AdamW (β = (0.9, 0.98), decoupled weight decay *applied to all parameters including biases*) and Gaussian initialisation with variance 1/fan-in. The transformer and CNN use PyTorch (CPU).
- *Hull computations.* Projections use `scipy.optimize.nnls` with a weighted sum-to-one row, followed by renormalisation, so h is always a valid hull point and imperfect solves only make certificates conservative. Membership uses `scipy.optimize.linprog` (HiGHS).
- *Provenance.* All numbers come from scripts in `code/`, run in this environment. The machine has 4 CPU cores, shared with other jobs. Raw outputs are in `results/`.

**Experiment 1 (modular addition, round 1; unchanged).**

- *Task and model:* p = 53, architecture 106–128–128–53, lr 10⁻³, weight decay 1.0, 12,000 steps.
- *Runs:* fraction 0.5 with seeds 0–2; fractions 0.4 and 0.6 with seed 0; a random-label control.
- *Measurements:* every 500 steps, on 250 held-out pairs, I_ℓ, 1-NN accuracy and C. Test accuracy uses the full held-out set.

**Experiment 4 (controls; new).**

- *Model:* same architecture and optimiser as Experiment 1.
- *Runs (17):*
  - fraction 0.5 with seeds 0–4 at weight decay 1;
  - fractions 0.6 and 0.7 with seeds 0–2 at weight decay 1;
  - weight decay 3 (seeds 0–2, 12,000 steps) and 0.3 (seeds 0–2, 24,000 steps) at fraction 0.5.
- *Measurements:* every 250 steps, on a fixed random subset of 700 held-out pairs:
  - test accuracy (full held-out set) and test and train cross-entropy;
  - C and C_vis;
  - margin-threshold fractions M_abs(τ) = Pr[m(φ(x)) > τ] for τ ∈ {2, 5, 10};
  - M_q, the fraction of test margins above the q-quantile of the *training* margins at the same checkpoint (q = 0.1, 0.5);
  - NC1 = tr(Σ_W Σ_B⁺)/K on training features and on test features (relative to training class means);
  - weight norm.
- *Per-point data:* every 500 steps we store per-point margins and certification for a post-hoc threshold fit over τ ∈ [0, 40].
- *Crossing steps:* we use *sustained* crossings, i.e. the first checkpoint after which the statistic stays ≥ the threshold until the end of training.

**Experiment 5 (transformer; new).**

- *Model:* a one-layer transformer following the setup of Nanda et al. (2023), scaled down for CPU. Tokens are [a, b, =]; d_model = 128, 4 heads, d_mlp = 512; no LayerNorm; full-batch AdamW with lr 10⁻³, weight decay 1, β = (0.9, 0.98) and 10-step warm-up.
- *Representation:* φ is the final residual stream at "=", and the readout is the unembedding (no bias).
- *Runs:* p = 53, fraction 0.5, seeds 0–2, 6,000 steps. Fraction 0.3 (seeds 0–2) was also run; see Section 6.3.
- *Measurements:* the same as Experiment 4.

**Experiments 2 and 6 (digits).**

- *Data:* UCI 8×8 digits (Alpaydin & Kaynak, 1998); 600 training and 1,197 test images per seed; inputs x/16 − 0.5; seeds 0–4.
- *Models:*
  - the round-1 MLP (64–128–64–10, 1,500 full-batch steps, weight decay 10⁻²);
  - new: a CNN (conv3×3-32, conv3×3-64, max-pool, fc-128, fc-10; AdamW lr 10⁻³, weight decay 10⁻⁴, 60 epochs, batch 32, no augmentation).
- *Perturbations of the test set:* none; right shift by 1 and by 2 px; down shift by 1 px; additive Gaussian noise with σ = 0.25 of the pixel range, clipped; rotation by 15°.
- *Error-detection scores:* −max-softmax; −logit margin (top-1 minus top-2); energy (−logsumexp); class-conditional Mahalanobis (shared covariance, minimum over classes); deep-kNN distance (L2-normalised features, k = 1 and 10); hull distance d(x, ŷ); hull contrast (d(x, ŷ) minus the smallest d over the other classes); and the certificate slacks for C and C_vis.
- *Statistics:* AUROC per seed, reported as mean ± s.d. with 95% t-intervals over seeds. To test whether a score adds to the margin, we fit a leave-one-seed-out logistic regression on the margin with and without the score. Features are z-scored with the same seed's unlabelled, unperturbed test statistics. We report the paired AUROC gain across seeds.
- *Padding construction:* 256 readout-invisible units ReLU(αGφ₁) are appended to the MLP's penultimate layer, with α ∈ {1, 4, 16}.

**Experiment 3 (diagnostic, round 1; unchanged).** Residuals, the readout-visible fraction of ‖r‖², and margins at φ(x) and h, against training-set size.

## 6 Results

### 6.1 Where interpolation appears (Experiment 1)

**Table 1. Final-checkpoint profile on modular addition** (250 held-out pairs; step 12,000; Experiment 1).

| run | test acc | I₀ (input) | I₁ | I₂ | 1-NN acc (L2) | median d₀ / d₁ / d₂ | C |
|---|---|---|---|---|---|---|---|
| frac 0.5, seed 0 | 0.998 | 0.000 | 0.048 | 1.000 | 1.000 | 1.03 / 0.87 / 0.40 | 0.892 |
| frac 0.5, seed 1 | 0.999 | 0.000 | 0.044 | 0.996 | 0.996 | 1.03 / 0.88 / 0.33 | 0.964 |
| frac 0.5, seed 2 | 0.999 | 0.000 | 0.012 | 1.000 | 1.000 | 1.03 / 0.88 / 0.41 | 0.808 |
| frac 0.6, seed 0 | 1.000 | 0.000 | 0.068 | 1.000 | 1.000 | 1.03 / 0.84 / 0.24 | 1.000 |
| frac 0.4, seed 0 (not grokked) | 0.138 | 0.000 | 0.000 | 0.096 | 0.060 | 1.03 / 0.96 / 0.89 | 0.000 |
| frac 0.5, random labels | 0.019 | 0.024 | 0.012 | 0.008 | 0.012 | 0.88 / 0.92 / 0.96 | 0.000 |

Three points follow.

1. **The input is exactly anti-interpolative** (Proposition 2). I₀ is constant by construction.
2. **Layer 1 is not interpolative even after grokking** (I₁ ≤ 0.07).
3. **I₂ and layer-2 1-NN accuracy track test accuracy.** I₂ crosses 0.5 at the same 500-step checkpoint as test accuracy in the three fraction-0.5 runs (6,500/6,500, 6,500/6,500, 7,500/7,500). For fraction 0.6 it crosses one checkpoint later (3,500 vs 3,000). The interpolation depth is L − 1 = 2.

This is correlational. Accurate networks with roughly clustered, linearly separable last-layer features will have high I₂ almost automatically. We read it as locating *where* test points become relative interpolations, not as a mechanism. The random-label control stays at chance throughout, so the metrics do not rise merely because training data are fitted.

### 6.2 The certification lag is a margin phenomenon (Experiment 4)

**Table 2. Lag of the 50% crossing after test accuracy crosses 50%** (steps; mean [95% t-CI] over seeds; 250-step checkpoints; sustained crossings).

| condition (seeds, grokked) | C | C_vis | M_abs(2) | M_abs(5) | M_abs(10) | test loss ≤ 0.1 | final acc | final C | final C_vis |
|---|---|---|---|---|---|---|---|---|---|
| frac 0.5, wd 1 (5, 5) | 3,450 [2,940, 3,960] | 3,250 [2,759, 3,741] | 650 [480, 820] | 1,800 [1,540, 2,060] | 4,625 [3,797, 5,453] (4/5) | 2,600 [2,129, 3,071] | 0.999 | 0.916 ± 0.064 | 0.936 ± 0.048 |
| frac 0.6, wd 1 (3, 3) | 2,250 [1,629, 2,871] | 2,250 [1,629, 2,871] | 417 [58, 775] | 917 [558, 1,275] | 2,750 [2,129, 3,371] | 1,500 [879, 2,121] | 1.000 | 1.000 | 1.000 |
| frac 0.7, wd 1 (3, 3) | 1,583 [1,225, 1,942] | 1,417 [1,058, 1,775] | 250 | 500 | 1,583 [1,225, 1,942] | 750 | 1.000 | 1.000 | 1.000 |
| frac 0.5, wd 3 (3, 3) | 2,833 [837, 4,830] | 2,250 [1,008, 3,492] | 417 [58, 775] | 1,167 [808, 1,525] | never | 1,417 [468, 2,365] | 0.995 | 0.956 ± 0.022 | 0.977 ± 0.010 |
| frac 0.5, wd 0.3 (3, 2 by step 24k) | never | never | 1,625 (2 runs) | never | never | never | 0.80, 0.85, 0.48 | ≤ 0.004 | ≤ 0.011 |
| 1-layer transformer, frac 0.5 (3, 3) | 917 [−32, 1,865] | 917 [−32, 1,865]V | 83 | 417 | 4,250 (2/3) | 1,000 | 0.999 | 0.803 ± 0.156 | 0.803 ± 0.156V |

Per-seed lags for C at fraction 0.5, wd 1 are 4,000, 3,250, 3,750, 3,000 and 3,250. Sources: `results/controls_summary.json` (`groups`) and the per-run `results/controls_*.json`.

**Finding 1: the lag is robust, and it shrinks with data and with weight decay.** With 3–5 seeds per condition, C crosses 50% after test accuracy in every grokked run. The mean lag falls from 3,450 to 2,250 to 1,583 steps as the training fraction goes from 0.5 to 0.6 to 0.7, with non-overlapping CIs between 0.5 and 0.7. This replaces round 1's single-seed trend. With weight decay 3, grokking is earlier and the lag tends to be shorter. With weight decay 0.3, two of three runs grok only after about 20,000 steps and end at 80–85% test accuracy with C ≤ 0.004.

**Finding 2: a fixed margin threshold reproduces the lag.** Every fixed-threshold statistic M_abs(τ) also lags test accuracy, and the lag grows with τ: 650, 1,800 and 4,625 steps for τ = 2, 5 and 10 at fraction 0.5. We fitted one threshold to all MLP runs, using the per-point margins stored every 500 steps. The pooled best fit is τ = 8.0 logits (RMSE 0.091 between the curves of M_τ and C). With this single τ:

- the 50% crossing of M_τ coincides with that of C, or precedes it by exactly one 500-step checkpoint, in all 14 grokked MLP runs;
- the mean absolute difference is 286 steps;
- the Pearson correlation of the two lags across runs is r = 0.97.

For C_vis the best fit is τ = 7.75 (r = 0.90; mean absolute difference 321 steps). Source: `controls_summary.json`, `global_tau`.

**Finding 3: training-relative margins and NC1 do not explain it, but absolute margins do.** M_q10 and M_q50 (test margin above the 10th or 50th percentile of *training* margins) never reach 0.5 at the end of training: training margins stay larger (median 15.4 against a test median of 10.5 at fraction 0.5). NC1 on test features does not fall after grokking. Its half-decline happens during memorization, 5,500–7,000 steps *before* test accuracy crosses 50% at fraction 0.5. Its final mean value, 15.7 at fraction 0.5 and 20.7 at fraction 0.6, is far from collapse and slightly *above* its post-memorization minimum (Fig. 4). Test loss falls below 0.1 between test-accuracy crossing and certification (lag 2,600 at fraction 0.5). Weight norm peaks during memorization (65–76 at steps 1,250–1,750 for weight decay ≤ 1) and then declines smoothly, with no feature at the certificate crossing (`results/controls_*.json`).

**Finding 4: per point, C is close to, but not the same as, a margin threshold.** At the checkpoint where C is closest to 0.5, we compute the AUROC with which the test margin predicts certification among correctly classified points. It is 0.90 ± 0.01 at fraction 0.5 (wd 1), 0.87–0.88 at wd 3 and fraction 0.6, and 0.86 at fraction 0.7. The best single-threshold rule agrees with certification on 80–82% of points. The certificate also depends on the hull residual, as constructed, but its *timing over training* is captured by the margin.

**Interpretation (revised).** Round 1 read the gap between test accuracy and C as a window of "generalization that is not interpolation", followed by "consolidation into interpolation". The controls do not support that reading over a simpler one. After grokking, logit margins on held-out points keep growing under weight decay, consistent with late-phase margin maximisation (Lyu & Li, 2020; Lyu et al., 2024) and with Nanda et al.'s (2023) cleanup. Any conservative margin-based statistic, C included, crosses a fixed level later than accuracy. We therefore withdraw the mechanistic interpretation. What remains is a robust, quantified ordering: accuracy first, decision-robust hull proximity later. Its timing is predicted by a logit-margin threshold of about 8.

### 6.3 One-layer transformer (Experiment 5)

Three one-layer transformers at fraction 0.5 grok quickly: test accuracy crosses 50% at steps 1,250, 500 and 1,250, and ends at 0.996–1.000 by step 6,000. C crosses 50% after test accuracy in all three runs, which is what round-1 prediction P1 expected. The lags are 1,250, 500 and 1,000 steps (C_vis: 1,000, 500 and 1,000).

The margin account again fits. A single threshold τ = 6.25, fitted on the transformer runs, gives the same 500-step crossing as C in all three runs. At the checkpoint where C ≈ 0.5, the margin predicts per-point certification with AUROC 0.91 ± 0.03.

Three differences from the MLP stand out:

- C is non-monotone. It oscillates by up to 0.2 between checkpoints after grokking (Fig. 4), which is typical of full-batch Adam instabilities. Final C is 0.80 ± 0.16.
- Test NC1 *rises* after grokking (final mean 52), so the residual-stream features at "=" are not collapsing.
- C_vis exceeds C (0.85 ± 0.13). The residual stream contains readout-invisible content, as Proposition 4 would lead one to expect.

At fraction 0.3, none of the three runs had grokked by step 6,000 (test accuracy 0.065), and C = C_vis = 0 throughout. That window is too short for p = 53 at this fraction, so we draw no conclusion from those runs.

### 6.4 Is C informative beyond accuracy and margin? (Experiments 4 and 6)

**A well-generalizing network with C ≈ 0: yes, but only with small margins.** The weight-decay-0.3 runs reach 80% and 85% test accuracy (seeds 0 and 1) with C ≤ 0.004 and C_vis ≤ 0.011. Their median test margins are 3.7 and 4.5. These networks generalize without certification, but the margin explains why. We found no natural network with both high accuracy and large test margins but low C_vis.

**A function-identical network with C ≈ 0: yes, by padding (Proposition 4).** We append 256 readout-invisible ReLU units to the digits MLP. The maximum logit change is below 10⁻¹³, so the network computes the same function.

**Table 3. Padding construction** (digits MLP; mean ± s.d. over 5 seeds).

| α | padding norm / φ norm | C | C_vis |
|---|---|---|---|
| no padding | 0 | 0.876 ± 0.007 | 0.915 ± 0.006 |
| 1 | 0.32 | 0.855 ± 0.010 | 0.915 ± 0.006 |
| 4 | 1.27 | 0.559 ± 0.020 | 0.915 ± 0.006 |
| 16 | 5.08 | 0.001 ± 0.001 | 0.915 ± 0.006 |

C can therefore be driven to zero without changing accuracy or margins. This is a defect of C as a functional quantity, and the reason we now recommend C_vis. C_vis is exactly invariant. Even so, its timing in grokking is likewise reproduced by a margin threshold (Section 6.2).

**Per-point information beyond the margin: modest but significant.** The more useful question is whether hull geometry tells us anything about *individual predictions* that the logit margin does not. Table 4 answers it for error detection.

**Table 4. Digits error detection: AUROC, mean over 5 seeds (± s.d.).** The final column is the leave-one-seed-out gain of margin + hull contrast over margin alone (mean ± s.d. over seeds; paired t-test p).

| model / perturbation (acc) | −MSP | −logit margin | energy | Mahalanobis | kNN (k = 10) | hull contrast | C_vis slack | LOSO gain of hull contrast |
|---|---|---|---|---|---|---|---|---|
| MLP / none (0.971) | 0.956 ± .008 | 0.956 ± .008 | 0.940 | 0.767 | 0.908 | **0.975 ± .008** | 0.961 | +0.019 ± .006 (p = 0.001) |
| MLP / right 1 (0.445) | 0.687 ± .045 | 0.686 | 0.721 | 0.629 | 0.759 | **0.810 ± .019** | 0.783 | +0.130 ± .056 (p = 0.007) |
| MLP / right 2 (0.094) | 0.482 ± .095 | 0.482 | 0.495 | 0.470 | 0.504 | **0.672 ± .083** | 0.597 | +0.205 ± .126 (p = 0.02) |
| MLP / down 1 (0.444) | 0.693 | 0.691 | 0.717 | 0.556 | 0.737 | 0.756 ± .030 | 0.729 | +0.068 ± .021 (p = 0.002) |
| MLP / noise 0.25 (0.886) | 0.892 | 0.891 | 0.855 | 0.586 | 0.820 | **0.922 ± .006** | 0.897 | +0.031 ± .007 (p < 0.001) |
| MLP / rot 15° (0.633) | 0.774 | 0.772 | 0.659 | 0.672 | 0.654 | **0.799 ± .022** | 0.786 | +0.029 ± .021 (p = 0.03) |
| CNN / none (0.977) | 0.958 ± .011 | 0.956 | 0.936 | 0.727 | 0.913 | **0.978 ± .008** | 0.954 | +0.022 ± .005 (p < 0.001) |
| CNN / right 1 (0.597) | 0.760 | 0.753 | 0.783 | 0.568 | 0.788 | **0.866 ± .037** | 0.829 | +0.115 ± .030 (p = 0.001) |
| CNN / right 2 (0.159) | 0.399 | 0.401 | 0.385 | 0.461 | 0.575 | **0.610 ± .024** | 0.612 | +0.167 ± .025 (p < 0.001) |
| CNN / down 1 (0.502) | 0.707 | 0.703 | 0.735 | 0.492 | 0.688 | **0.759 ± .031** | 0.687 | +0.057 ± .029 (p = 0.01) |
| CNN / noise 0.25 (0.910) | 0.898 | 0.897 | 0.840 | 0.630 | 0.816 | **0.923 ± .015** | 0.889 | +0.026 ± .010 (p = 0.004) |
| CNN / rot 15° (0.753) | 0.809 | 0.803 | 0.762 | 0.616 | 0.780 | **0.836 ± .043** | 0.806 | +0.034 ± .014 (p = 0.006) |

Source: `results/digits2_summary.json`. MLP / down 1 is the one condition where kNN (k = 1) is higher (0.759 vs 0.756); kNN (k = 1) is not shown in the table. Errors per condition, pooled over 5 seeds: 171 (MLP / none) and 137 (CNN / none), up to 5,420 (MLP / right 2).

Three observations:

- **Hull contrast adds to the margin everywhere.** Its leave-one-seed-out gain is positive in all 12 conditions and significant at p < 0.05 in all of them. In distribution the gain is small (+0.02 AUROC). Under shift it is large (+0.06 to +0.21).
- **It beats the standard baselines in 11/12 conditions.** In distribution, the round-1 differences (0.956 vs 0.975) are now supported per seed: the paired difference is +0.020 ± 0.006 (p = 0.0015) for the MLP and +0.021 ± 0.005 (p < 0.001) for the CNN.
- **The certificate's own slack adds less.** C_vis adds less than hull contrast: significant in 10/12 conditions, with smaller gains. The *contrast* between hulls, not the certificate threshold, carries most of the extra information.

**Table 5. Digits profile and certification** (mean ± s.d. over 5 seeds).

| model / condition | acc | I₀ (pixels) | I_pen | C | C_vis | acc on C-certified / uncertified | acc on C_vis-certified / uncertified |
|---|---|---|---|---|---|---|---|
| MLP / none | 0.971 ± .005 | 0.984 ± .004 | 0.979 | 0.876 ± .007 | 0.915 ± .006 | 0.998 / 0.785 ± .052 | 0.994 / 0.726 |
| CNN / none | 0.977 ± .005 | 0.984 ± .004 | 0.982 | 0.855 ± .008 | 0.918 ± .008 | 0.997 / 0.856 ± .031 | 0.996 / 0.768 |
| MLP / right 1 | 0.445 ± .034 | 0.559 ± .023 | 0.501 | 0.147 ± .021 | 0.330 | 0.909 / 0.364 | 0.738 / 0.299 |
| CNN / right 1 | 0.597 ± .027 | 0.559 ± .023 | 0.665 | 0.195 ± .034 | 0.340 | 0.958 / 0.510 | 0.904 / 0.439 |
| MLP / rot 15° | 0.633 ± .072 | 0.786 ± .039 | 0.647 | 0.285 | 0.454 | 0.929 / 0.517 | 0.843 / 0.463 |
| CNN / rot 15° | 0.753 ± .079 | 0.786 ± .039 | 0.756 | 0.262 | 0.521 | 0.979 / 0.673 | 0.920 / 0.573 |

The round-1 numbers are reproduced exactly with the same seeds: 5,245 of 5,985 predictions certified, with 171 errors. Uncertified accuracy is 0.785 as the mean over seeds and 0.786 when pooled; round 1 used both numbers without saying which was which.

**Digits interpolation depth.** The reviewer asked whether "depth 0" was an artefact of a weak MLP. The stronger CNN (0.977) still does not beat the pixel-space nearest-class-hull classifier (0.984). Because I₀ = 0.984 and no classifier exceeds 1.0, depth 0 under ε = 0.02 holds for *any* network on this split. The substantive statement is only that pixel-space relative interpolation already nearly solves this easy task. Under 1-px shifts the CNN extrapolates better than pixel-space interpolation (0.597 vs 0.559), while the MLP does worse (0.445). So the round-1 remark that "the learned representation extrapolates worse than input interpolation" holds for the MLP, not in general.

### 6.5 Certificate diagnostics (Experiment 3), with null baselines

**Table 6. Certificate diagnostics** (seed 0; up to 400 test points; penultimate layer; *medians* over test points).

| task | train per class | test acc | C | hull point correct | median δ/r | median visible fraction of ‖r‖² (null: rank/d) | median margin at φ(x) / at h |
|---|---|---|---|---|---|---|---|
| modadd, frac 0.5 | 26.5 | 0.998 | 0.908 | 1.000 | 0.40 | 0.94 (0.41) | 10.4 / 16.3 |
| modadd, frac 0.7 | 37.1 | 1.000 | 1.000 | 1.000 | 0.18 | 0.93 (0.41) | 14.2 / 16.3 |
| modadd, frac 0.9 | 47.7 | 1.000 | 1.000 | 1.000 | 0.14 | 0.82 (0.41) | 14.4 / 16.0 |
| digits, n = 100 | 10 | 0.918 | 0.680 | 1.000 | 0.30 | 0.76 (0.14) | 15.0 / 21.3 |
| digits, n = 300 | 30 | 0.955 | 0.818 | 1.000 | 0.19 | 0.69 (0.14) | 21.0 / 24.2 |
| digits, n = 600 | 60 | 0.974 | 0.868 | 1.000 | 0.17 | 0.63 (0.14) | 25.0 / 26.9 |
| digits, n = 1,200 | 120 | 0.983 | 0.915 | 1.000 | 0.12 | 0.53 (0.14) | 29.8 / 30.6 |

The null baseline is the expected visible fraction for an isotropic random residual. It equals rank/d: 52/128 = 0.41 for modular addition and 9/64 = 0.14 for digits (an analytic value, not measured).

- **Residuals are concentrated in readout-visible directions.** Observed visible fractions are 2–2.3× the null for modular addition and 3.8–5.4× for digits. The hull residual is not isotropic noise.
- **More data shrink the residual.** δ/r falls and C rises with training points per class.
- **The out-of-hull residual typically reduces the margin.** The median margin at φ(x) is below the median at h in every row. This is a statement about medians.
- **Hull-point correctness is a sanity check.** The hull point is always correct: margins are affine, so a convex mixture of correctly classified training points is correctly classified, given a perfectly fitted training set.

## 7 Discussion

### 7.1 What this says about "generalization vs. interpolation"

**(a) Input-space interpolation is the wrong yardstick for discrete tasks.** It is vacuous (Proposition 1) and can be exactly anti-correlated with the right answer (Proposition 2, Table 1). Arguments of the form "the model succeeded on inputs unlike its training data, so it does not merely interpolate" are uninformative when "unlike" is measured by input geometry. For continuous data, strict membership fails as well, while relative interpolation can be excellent (Table 5).

**(b) In the last layer of networks that generalize, test points become relative interpolations.** I₂ tracks test accuracy in grokking. This is what one expects of any accurate network with clustered, linearly separable last-layer features. It locates *where* the representation makes test points look like training points of their class (only the last layer, in modular addition), but it is not by itself a mechanism.

**(c) Certified interpolation arrives later, and the reason is the margin.** Round 1 claimed that "generalization comes first in an uncertifiable form and consolidates into interpolation". The controls show that this ordering is what one gets from any conservative large-margin statistic while margins keep growing. We now state only the following. In every grokked run, decision-robust hull proximity is reached after accuracy, and a single logit-margin threshold predicts when.

**(d) Hull geometry adds a little per-point information.** As a per-prediction diagnostic, between-hull contrast in the readout's input space adds significantly to the logit margin for error detection under all six perturbations, for both an MLP and a CNN. This is the strongest *positive* evidence that "is this test point an interpolation of same-class training points in the learned space?" is not merely a restatement of "is the margin large?". The effect is modest in distribution (+0.02 AUROC) and larger under shift.

### 7.2 Practical use

- *Auditing.* Report C_vis, not C, as the fraction of decisions whose logit-difference vector lies close enough to the hull of same-class training logit-difference vectors to be decision-robust. Report it alongside the margin distribution, since the two are strongly related.
- *Error detection.* Hull contrast at the penultimate layer is a cheap complement to softmax confidence: one small NNLS problem per class.
- *Not recommended:* using the certification lag as a "progress measure" distinct from margin growth. Our controls show it is not distinct.

### 7.3 Falsifiable predictions for larger models (not tested)

- **P1′ (margin reduction).** In larger transformers on algorithmic tasks, the step at which C_vis crosses 50% is predicted, within one checkpoint, by a fixed logit-margin threshold fitted on other seeds.
  *Falsified if* C_vis's timing deviates systematically from every fixed-threshold M_τ.
- **P2 (depth profile).** For tasks that require a learned algorithm, interpolation depth equals the last layer. For perceptual tasks with input-level similarity structure, it is shallow.
- **P3 (counterfactual tasks).** For an LLM on default versus counterfactual task variants (Wu et al., 2024), with answer-class hulls built from final-layer representations of default items:
  - C_vis is lower on counterfactual items;
  - hull contrast adds to the answer-token logit margin for error prediction, with leave-one-split-out evaluation and bootstrap CIs.

  *Falsified if* the gain over the margin is not significant.

## 8 Limitations

- **Scale and architecture.** We used small MLPs, one small CNN and a one-layer transformer at p = 53, all on CPU. We have not tested deeper transformers or language models.
- **Checkpoint resolution and samples.**
  - Lags are quantized to 250 steps (Experiment 4) or 500 steps (the threshold fit).
  - C is computed on 700 of about 1,400 held-out pairs. Test accuracy uses the full set.
  - Seeds per condition are 3–5. CIs with 3 seeds are wide (e.g. weight decay 3).
- **Weight decay 0.3 runs** were stopped at 24,000 steps before convergence. One of the three runs had not grokked (test accuracy 0.48). We do not know their final C.
- **Margin thresholds were fitted post hoc.** τ = 8 was fitted on the same runs it is evaluated on. It is a single parameter shared by all 14 runs, and the per-condition best fits range from 7.5 to 10.
- **NC1 as computed** (pseudo-inverse of the between-class covariance in 128 dimensions with 53 classes) is one of several collapse statistics. Others might behave differently.
- **The certificate is sufficient, not necessary, and conservative.** C depends on readout-invisible directions (Proposition 4). C_vis fixes that but remains a Cauchy–Schwarz bound.
- **The relative-distance normalisation r_ℓ is a convention.** Layer comparisons rest on the scale-free I_ℓ. Raw d values across layers of different widths should not be compared quantitatively.
- **Digits** is a small, easy dataset, and our perturbations are synthetic. The error-detection baselines are standard but untuned beyond defaults (k = 10, shared covariance). The leave-one-seed-out regression uses only two features.
- **No large-model evidence.** Nothing here establishes how much of a large language model's behaviour is certified interpolation.

## 9 Conclusion

"Does the model generalize or merely interpolate?" becomes answerable only once one names the representation and the notion of interpolation.

- In input space, for discrete tasks, the answer is fixed in advance (Proposition 1) or exactly wrong (Proposition 2).
- In the last hidden layer, test points of generalizing networks become relative interpolations.
- *Certified* interpolation arrives later in grokking, but a fixed logit-margin threshold predicts when. We therefore no longer claim a distinct regime of uncertified generalization.
- The full-space certificate is not invariant to readout-invisible directions. The readout-subspace version is, and it has a literal reading as convex mixing of training outputs.
- Hull-relative scores add modest but significant per-point information beyond the margin.

We think the useful replacement for the interpolation/generalization dichotomy is the measurable profile reported here: the layer at which test points become relative interpolations, together with what hull geometry adds to the margin.

## References

Alpaydin, E., & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49

Balestriero, R., Pesenti, J., & LeCun, Y. (2021). Learning in High Dimension Always Amounts to Extrapolation. arXiv:2110.09485.

Belkin, M. (2021). Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. *Acta Numerica*, 30, 203–248. arXiv:2105.14368.

Bonnasse-Gahot, L. (2022). Interpolation, extrapolation, and local generalization in common neural networks. arXiv:2207.08648.

Dziri, N., et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. *NeurIPS 36*. arXiv:2305.18654.

Gromov, A. (2023). Grokking modular arithmetic. arXiv:2301.02679.

Hasson, U., Nastase, S. A., & Goldstein, A. (2020). Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks. *Neuron*, 105(3), 416–434. https://doi.org/10.1016/j.neuron.2019.12.002

Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*. arXiv:1610.02136.

Hwang, and Shin (2024). Uncertainty Measurement of Deep Learning System based on the Convex Hull of Training Sets. arXiv:2405.16082.

Lake, B., & Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. *ICML 2018*, PMLR 80, 2873–2882. arXiv:1711.00350.

Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS 2018*. arXiv:1807.03888.

Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020). Energy-based Out-of-distribution Detection. *NeurIPS 2020*. arXiv:2010.03759.

Liu, Z., Kitouni, O., Nolte, N., Michaud, E., Tegmark, M., & Williams, M. (2022). Towards Understanding Grokking: An Effective Theory of Representation Learning. *NeurIPS 35*. arXiv:2205.10343.

Liu, Z., Michaud, E. J., & Tegmark, M. (2023). Omnigrok: Grokking Beyond Algorithmic Data. *ICLR 2023*. arXiv:2210.01117.

Lyu, K., & Li, J. (2020). Gradient Descent Maximizes the Margin of Homogeneous Neural Networks. *ICLR 2020*. arXiv:1906.05890.

Lyu, K., Jin, J., Li, Z., Du, S. S., Lee, J. D., & Hu, W. (2024). Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking. *ICLR 2024*. arXiv:2311.18817.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*. arXiv:2301.05217.

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS*, 117(40), 24652–24663. https://doi.org/10.1073/pnas.2015509117

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. *ICML 2022*, PMLR 162, 20827–20840. arXiv:2204.06507.

Wu, Z., Qiu, L., Ross, A., Akyürek, E., Chen, B., Wang, B., Kim, N., Andreas, J., & Kim, Y. (2024). Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks. *NAACL 2024*, 1819–1862. https://doi.org/10.18653/v1/2024.naacl-long.102

Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K., & Jegelka, S. (2021). How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks. *ICLR 2021*. arXiv:2009.11848.

Yousefzadeh, R. (2021). Deep Learning Generalization and the Convex Hull of Training Sets. arXiv:2101.09849.

Yousefzadeh, R. (2022a). Decision boundaries and convex hulls in the feature space that deep learning functions learn from images. arXiv:2202.04052.

Yousefzadeh, R. (2022b). Deep Learning Generalization, Extrapolation, and Over-parameterization. arXiv:2203.10366.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*. arXiv:1611.03530.

(Chollet, 2019, cited in round 1, was dropped together with the sentence that cited it.)

---

### Figures (in `results/`)

- **Figure 1** (`fig1_modadd_dynamics.png`, round 1): test accuracy, layer-wise nearest-class-hull accuracy, layer-2 1-NN accuracy, input-space accuracy (identically 0) and C over training. Panels: fraction 0.5 (seed 0), fraction 0.4 (not grokked) and the random-label control.
- **Figure 2** (`fig2_cert_vs_acc.png`, round 1): C against test accuracy for all real-label Experiment-1 runs, *including the non-grokked fraction-0.4 run*, which stays near the origin.
- **Figure 3** (`fig3_digits.png`, round 1): digits MLP under shift.
- **Figure 4** (`fig4_controls.png`, new): test accuracy, C, C_vis, M_abs(5), M_q10, normalised NC1 (test) and normalised test loss over training. Panels: MLP at weight decay 1, 3 and 0.3, and the one-layer transformer.
- **Figure 5** (`fig5_digits_auroc.png`, new): digits error-detection AUROC by score, perturbation and model, with 95% CIs over seeds.

### Data and code provenance

Every number in Sections 4–6 was produced by the scripts in `code/`, run in this environment. Sources by experiment:

| experiment | script | output |
|---|---|---|
| 1 | `exp_modadd.py` | `results/modadd_*.json`, `summary.json` |
| 2 | `exp_digits.py` | `digits.json` |
| 3 | `exp_diag.py` | `diag.json` |
| 4 | `exp_controls.py`, `analyze_controls.py` | `controls_mlp_*.json`, `controls_summary.json` |
| 5 | `exp_transformer.py` | `controls_tf_*.json` |
| 6 | `exp_digits2.py`, `analyze_digits2.py` | `digits2.json`, `digits2_summary.json` |

The exact failure probability in Section 4.2 was computed with Python integer binomials (one line; reproduced in the response to reviewers). The null visible fractions in Table 6 are analytic. No number is taken from external sources except the bibliographic facts in Sections 1–2.
