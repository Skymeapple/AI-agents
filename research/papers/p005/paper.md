# Interpolation Is Relative to a Representation: Vacuous Input Hulls, a Readout Interpolation Certificate, and the Order in Which Grokking Networks Generalize and Then Interpolate

**Target venue:** International Conference on Learning Representations (ICLR 2027), main track

---

## Abstract

People keep asking whether neural networks "really generalize" or "only interpolate", but the question has no answer until one says *in which space* interpolation is measured. The geometric definition (a test point interpolates if it lies in the convex hull of the training set) is known to put almost every high-dimensional test point outside the hull (Balestriero et al., 2021). We show that for discrete inputs the definition is not just rarely satisfied but *vacuous*, and that natural relative versions of it can be *anti-informative*. We then propose a measurement that makes the question decidable for a trained network. **(i) Theory.** Proposition 1: whenever the input space is in convex position (one-hot or multi-hot token encodings, any points on a sphere), no unseen input lies in the hull of any training set, whatever the model or task. Proposition 2: on modular addition with one-hot inputs, both the input-space 1-nearest-neighbour classifier and the nearest-class-hull classifier score *exactly 0%* on held-out pairs, while networks reach 100%. Proposition 3 gives a *readout interpolation certificate*. If a test point's penultimate representation lies within distance δ of the convex hull of the same-label training representations, and the hull point's logit margin is larger than ‖w_y − w_c‖·δ, then the network's decision is provably a convex interpolation of its training decisions. **(ii) Experiments** (numpy MLPs on CPU, all code released). On modular addition (p = 53), interpolation appears only at the last hidden layer. There, nearest-hull accuracy tracks test accuracy throughout grokking. Layer 1 stays at 1–7%, and the input stays at exactly 0%. The certificate *lags* generalization: the certified fraction crosses 50% 2,500–4,000 optimisation steps after test accuracy does, and reaches 81–100% at the end of training. So rule-like, uncertifiable generalization comes first, and certified interpolation follows during later consolidation. On the UCI 8×8 digits, 0 of 300 test images per seed lie in the pixel-space hull. Yet the pixel-space nearest-class-hull classifier (98.4%) beats the network (97.1%). 87.6% of test predictions are certified, with accuracy 99.75% on certified versus 78.6% on uncertified points. Under a 1–2 pixel shift, the certified fraction falls from 87.6% to 14.7% and then 0.3%. In error detection, a hull-contrast score beats max-softmax (AUROC 0.81 vs 0.69 at a 1-pixel shift; 0.67 vs 0.48 at 2 pixels). Our answer to the title question: "interpolation vs. generalization" is a property of a representation, not of a model. It can be measured layer by layer. On the tasks we studied, generalization consists in building a representation in which test points become certifiable interpolations.

## 中文摘要

“神经网络是真正泛化，还是本质上只是高级插值？”这个问题在说明“插值是在哪个空间中度量的”之前没有确定答案。基于凸包的几何定义（测试点落入训练集凸包即为插值）已知在高维中几乎从不成立（Balestriero 等，2021）。我们证明：对离散输入，该定义不仅很少成立，而且是*空洞的*，并且它的自然相对化版本甚至可以是*反向信息的*。在此基础上，我们提出一种使该问题对给定网络可判定的度量方法。**理论**：命题1：只要输入空间处于凸位置（如 one-hot/multi-hot 词元编码、球面上的点），任何未见输入都不在任何训练集的凸包内，与模型和任务无关。命题2：在 one-hot 编码的模加法任务上，输入空间的最近邻分类器和最近类凸包分类器在留出样本上的准确率*恰好为 0%*，而网络可达 100%。命题3给出一个*读出层插值证书*：若测试点的倒数第二层表示到同类训练表示凸包的距离为 δ，且凸包投影点的 logit 间隔大于 ‖w_y − w_c‖·δ，则网络的决策可被证明是其训练决策的凸插值。**实验**（CPU 上的 numpy MLP，代码全部公开）：在模加法（p = 53）上，插值只出现在最后一个隐藏层。在该层，最近凸包准确率在整个 grokking 过程中与测试准确率同步；第1层始终只有 1–7%，输入层恒为 0%。证书*滞后*于泛化：被认证比例越过 50% 的时间比测试准确率越过 50% 晚 2,500–4,000 步，训练结束时达到 81–100%。也就是说，先出现类似规则、无法认证的泛化，随后在后期的巩固阶段才变成可认证的插值。在 UCI 8×8 手写数字上，每个种子的 300 张测试图像中没有一张落在像素空间凸包内；但像素空间的最近类凸包分类器（98.4%）反而优于网络（97.1%）。87.6% 的测试预测可被认证，其准确率为 99.75%，未认证样本仅为 78.6%。在 1–2 像素平移下，认证比例由 87.6% 降至 14.7%，再降至 0.3%。在错误检测上，凸包对比分数优于最大 softmax 概率（1 像素平移时 AUROC 0.81 对 0.69；2 像素时 0.67 对 0.48）。我们的回答是：“插值还是泛化”是某个表示的属性，而不是模型的属性，可以逐层度量。在我们研究的任务上，泛化表现为学到一个能让测试点成为可认证插值的表示。

---

## 1 Introduction

Whether neural networks "genuinely generalize" or "merely interpolate" is debated constantly. One camp argues that deep networks fit their training data densely and generalize by local interpolation over a task manifold ("direct fit"; Hasson et al., 2020). Others read failures on counterfactual or compositional variants of familiar tasks as evidence of recitation rather than reasoning (Wu et al., 2024; Dziri et al., 2023; Lake & Baroni, 2018). Still others argue that a genuinely intelligent system should be measured by skill-acquisition efficiency beyond the training distribution (Chollet, 2019). Throughout, "interpolation" is used as if it were a well-defined property of a model's behaviour on a test point.

It is not. The standard formal definition calls a test point an interpolation if it lies in the convex hull of the training inputs. Balestriero, Pesenti and LeCun (2021) showed that under this definition, test points in high dimension almost surely fall outside the hull. "Interpolation" then never happens, so it cannot explain anything. Later work moved the hull into a learned feature space. Yousefzadeh (2021; 2022) found that test images lie outside but *close to* the training hull in feature space. Bonnasse-Gahot (2022) found that in a low-dimensional latent code of the last hidden layer most test points do fall inside the hull, but that membership itself did not track accuracy as well as other proximity measures. Two things are still missing:

1. **A principled reason to prefer one space over another.** Hull membership in some feature space is a descriptive statistic. Nothing links it to the network's *decision*.
2. **Dynamics.** We do not know whether a network generalizes *because* test points become interpolations, or whether test points become interpolations *after* the network has generalized.

We make this concrete in two settings where the answer can be computed. One is algorithmic (modular addition, where grokking separates memorization from generalization in time; Power et al., 2022). The other is natural (handwritten digits). Our contributions are:

- **C1 (Vacuity, Proposition 1).** If the set of possible inputs is in convex position, which holds for one-hot and multi-hot encodings of discrete tokens and for any points on a sphere, then no unseen input lies in the hull of any training set. Under the hull definition, *every* generalization by a model with discrete inputs (including every language model at its input layer) is "extrapolation", whatever the model does. The definition carries no information in this regime.
- **C2 (Anti-informativeness, Proposition 2).** For modular addition with one-hot inputs, the natural relative versions of interpolation, nearest neighbour and nearest-class-hull, are not merely uninformative. They are *exactly* wrong on every held-out pair, for any training split satisfying a mild condition. Networks nonetheless reach 100%. The experiments confirm 0.000 at every checkpoint of every run.
- **C3 (A decision-linked definition, Proposition 3).** For any network with a linear readout, we derive a *readout interpolation certificate*. It is a checkable sufficient condition, computable by one nonnegative least-squares problem per test point, under which the network's decision on a test point is provably a convex interpolation of its decisions on same-label training points. This ties "interpolation" to the network's actual output.
- **C4 (Measurement: interpolation profile and lag).** We measure, layer by layer, how far test representations are from same-label training hulls and how often they are certified. In grokking MLPs, interpolation appears only at the last hidden layer and in lock-step with test accuracy. The *certificate* lags test accuracy by 2,500–4,000 steps across four runs. Generalization therefore comes first in an uncertifiable form, and certified interpolation emerges during consolidation. Random-label controls stay at chance on every metric.
- **C5 (Real data and shift).** On 8×8 digits (5 seeds), 0/300 test images per seed lie in the pixel-space hull. Even so, interpolation in a relative sense explains the task already at the input layer. The certificate separates reliable from unreliable predictions (99.75% vs 78.6% accuracy). It collapses under small pixel shifts, and a hull-contrast score detects errors under shift better than softmax confidence.

Our position on the title question is deflationary but constructive. "Does the model generalize or interpolate?" is a category error unless a representation is named. Once one is named, the question has a measurable answer, and the layer at which test points become certified interpolations (the *interpolation depth*) is a meaningful property of a task-model pair. On the tasks we studied, learning to generalize amounted to learning a representation in which test points become interpolations. That representation is reached only after a phase of generalization that is not certifiably interpolative. We make no claims here about large models; Section 7.3 gives a protocol and falsifiable predictions for them.

## 2 Related Work

**Interpolation and the convex hull.** Balestriero et al. (2021) formalized interpolation as convex-hull membership. They showed it almost surely fails in high dimension, with the governing quantity being the dimension of the smallest affine subspace containing the data rather than the manifold's intrinsic dimension, and concluded that the definition is a poor indicator of generalization. Yousefzadeh (2021) reported that most test images of standard datasets lie outside the training hull in pixel, wavelet and network-feature spaces. Yousefzadeh (2022) found that in the last-hidden-layer feature space test images are close to the training hull and far from decision boundaries. Bonnasse-Gahot (2022) projected last-layer activations to a low-dimensional autoencoder latent space and found most test points inside the hull there, but argued that membership was not the relevant criterion. Hwang and Shin (2024) proposed hull-based uncertainty scores for test selection. Our work differs in four ways. Proposition 1 proves vacuity (not just rarity) for discrete inputs. Proposition 2 proves that relative versions can be exactly anti-informative. Proposition 3 connects hull proximity to the *decision* through a sound certificate rather than using it as a correlate. And we study how all of this evolves during training.

**Interpolation regime and memorization.** "Interpolation" also names fitting the training data exactly (Belkin, 2021). Zhang et al. (2017) showed that networks can fit random labels, so the capacity to memorize does not by itself explain generalization. We use the geometric sense of the word. Our random-label control is the direct analogue of Zhang et al.'s experiment within our metrics.

**Extrapolation by architecture.** Xu et al. (2021) proved that ReLU MLPs become linear far from the training support. They therefore extrapolate only functions that are linear along those directions, and success on algorithmic extrapolation requires suitable nonlinearities in the architecture or the input representation. That is a statement about the function *outside* the training support in input space. Ours concerns where test points sit relative to the training set in *representation* space.

**Grokking and its mechanism.** Power et al. (2022) discovered delayed generalization on small algorithmic datasets. Nanda et al. (2023) reverse-engineered a Fourier-feature algorithm for modular addition in transformers and split training into memorization, circuit formation and cleanup. Liu et al. (2022) related generalization to structured representations, and Liu, Michaud and Tegmark (2023) to weight norm. Gromov (2023) gave analytic solutions for two-layer MLPs that grok modular arithmetic. Our certificate lag (C4) is a new, model-agnostic progress measure. It needs no knowledge of the Fourier algorithm, and its timing matches the qualitative picture of a later cleanup phase.

**Neural collapse.** Papyan, Han and Donoho (2020) documented that last-layer features collapse to class means during the terminal phase of training. Collapse would trivially place test features near same-class training hulls. Our measurement quantifies how far from collapse the test representations are (median relative hull distance 0.24–0.41 in modular addition) and whether the residual is small enough, relative to the readout geometry, to certify the decision.

**"Reasoning vs. reciting" in language models.** Counterfactual-task studies (Wu et al., 2024) and compositional studies (Dziri et al., 2023; Lake & Baroni, 2018) diagnose behaviour. Our method is complementary: it diagnoses *representations*, and Section 7.3 proposes how to apply it to them.

## 3 Setting and Definitions

Let f(x) = Wᵀφ(x) + b be a classifier with K classes, penultimate representation φ: 𝒳 → ℝ^d, readout W = [w₁,…,w_K] ∈ ℝ^{d×K}, and bias b. More generally, φ_ℓ denotes the representation after layer ℓ, with φ₀ the input. Let 𝒯 = {(x_i, y_i)} be the training set and 𝒯_y = {x_i : y_i = y}.

**Definition 1 (hull interpolation; Balestriero et al., 2021).** x is an *interpolation in layer ℓ* if φ_ℓ(x) ∈ conv{φ_ℓ(x_i) : x_i ∈ 𝒯}. It is a *class-conditional interpolation* if φ_ℓ(x) ∈ conv φ_ℓ(𝒯_y) for its label y.

**Definition 2 (relative hull distance).** d_ℓ(x, c) = dist(φ_ℓ(x), conv φ_ℓ(𝒯_c)) / r_ℓ. Here r_ℓ is the root-mean-square distance of training representations to their mean, so d is comparable across layers.

**Definition 3 (nearest-class-hull classifier and interpolation profile).** ĉ_ℓ(x) = argmin_c d_ℓ(x, c). The *interpolation profile* of a trained network on a test set is the vector of accuracies I_ℓ of ĉ_ℓ over layers ℓ = 0,…,L−1. The *interpolation depth* is the smallest ℓ at which I_ℓ is within ε of the network's own accuracy (we use ε = 0.02). In words, I_ℓ answers: if the task were solved by assigning each test point to the class whose training examples it lies closest to as a convex mixture, in layer ℓ, how well would that do?

Definitions 2–3 are *relative*. They replace membership, which Proposition 1 shows can be vacuous, with comparisons between hulls. But I_ℓ is still only a correlate of the network's behaviour. Definition 4 ties it to the decision.

**Definition 4 (certified interpolation).** A test point x with label y is *certified* if the condition of Proposition 3 holds for φ(x) and 𝒯_y. The *certified fraction* C is the fraction of test points that are certified. With the predicted label ŷ in place of y, the certificate is label-free, and it can hold for at most one class (Remark 2).

## 4 Theory

### 4.1 The hull definition is vacuous for discrete inputs

A set S ⊂ ℝ^n is in *convex position* if every s ∈ S is an extreme point of conv S, i.e. s ∉ conv(S \ {s}).

**Proposition 1 (vacuity).** Let S be in convex position, 𝒯 ⊂ S finite, and x ∈ S \ 𝒯. Then x ∉ conv 𝒯.

*Proof.* conv 𝒯 ⊆ conv(S \ {x}), and x ∉ conv(S \ {x}) by extremality. ∎

**Examples.**

- (a) Vertices of the hypercube {0,1}^n are in convex position. This covers one-hot encodings and concatenations of one-hot blocks, so every fixed-length token sequence under one-hot encoding is covered, as is the (a, b) encoding of Section 5.
- (b) Any subset of a sphere is in convex position, because the sphere is strictly convex. This covers L2-normalized embeddings, and RMS-normalized vectors with the gain removed.

**Consequence.** For a model whose inputs are discrete tokens, *every* correct answer on an unseen input is "extrapolation" under Definition 1, at the input layer, for every model and every task. The verdict is fixed before the model is trained, so it can say nothing about the model. We checked this numerically with an exact linear-programming feasibility test: 0 of 40 held-out modular-addition inputs per run lay in the training hull, in all six runs (`results/summary.json`, field `input_lp_in_hull_frac`).

The digits experiment shows that the same thing happens in practice, though not provably, for continuous 64-dimensional data. For each of 5 seeds, 0 of 300 test images lay in the hull of the 600 training images, and 0 of 300 lay in the hull of their own class (`input_in_hull_uncond`, `input_in_hull_same_class`). This is consistent with Balestriero et al. (2021).

### 4.2 Relative interpolation can be exactly anti-informative

Let p be prime and let e(a, b) ∈ {0,1}^{2p} be the concatenation of the one-hot vectors of a and b. The label is c = (a + b) mod p.

**Proposition 2 (anti-informativeness).** Let (a, b) be a held-out pair with label c. Suppose the training set contains some pair (a, b′) with b′ ≠ b or some pair (a′, b) with a′ ≠ a. Then the input-space 1-NN classifier and the input-space nearest-class-hull classifier ĉ₀ both assign (a, b) a label ≠ c.

*Proof.*

1. *Same-label training points share no coordinate with the test point.* If (a′, b′) ≠ (a, b) has label c and a′ = a, then b′ ≡ c − a ≡ b, which is a contradiction. Symmetrically b′ ≠ b. So for every same-label training point, ‖e(a, b) − e(a′, b′)‖² = 4.
2. *The same-label hull is farther than √2.* Let h be any point of the hull of the same-label training points. Each such point has zeros at coordinates a and p + b. So h_a = h_{p+b} = 0, and the first block of h is a probability vector supported on at most n_c coordinates, where n_c ≥ 1 is the number of same-label training points. Its squared norm is therefore at least 1/n_c, and the same holds for the second block. Hence ‖e(a, b) − h‖² = 1 + 1 + ‖h‖² ≥ 2 + 2/n_c > 2.
3. *Some other-label point is at distance exactly √2.* The assumed training pair (a, b′) has label a + b′ ≢ c and lies at squared distance 2 from e(a, b), and likewise for (a′, b).

So both the nearest training point and the nearest class hull belong to a wrong class. ∎

The condition fails only if none of the 2(p − 1) pairs sharing a coordinate with (a, b) is in the training set. For p = 53 and a 50% split, the probability of that is about 2^{−104}. Hence:

**Corollary.** On modular addition with one-hot inputs, relative input-space interpolation has accuracy exactly 0 on held-out pairs, which is below chance (1/p). Any network that generalizes here does so *against* input-space proximity. Experiment 1 observes I₀ = 0.000 and 1-NN accuracy 0.000 at every one of 25 checkpoints in every real-label run (`max_L0_hull_nearest_acc_over_training` = 0).

### 4.3 A readout interpolation certificate

**Proposition 3 (certificate).** Let x have label y and let h = Σ_i λ_i φ(x_i) with x_i ∈ 𝒯_y, λ_i ≥ 0, Σ_i λ_i = 1, and δ = ‖φ(x) − h‖. Define the logit margins m_{yc}(v) = (w_y − w_c)ᵀv + b_y − b_c. Then for every c ≠ y:

  m_{yc}(φ(x)) ≥ Σ_i λ_i m_{yc}(φ(x_i)) − ‖w_y − w_c‖ δ.        (1)

Consequently, if

  min_{c≠y} [ m_{yc}(h) − ‖w_y − w_c‖ δ ] > 0,        (2)

the network predicts y on x.

*Proof.* Margins are affine in the representation, and Σλ_i = 1, so m_{yc}(h) = Σ_i λ_i m_{yc}(φ(x_i)). Then m_{yc}(φ(x)) = m_{yc}(h) + (w_y − w_c)ᵀ(φ(x) − h) ≥ m_{yc}(h) − ‖w_y − w_c‖δ by Cauchy–Schwarz. ∎

**Remark 1 (what the certificate says).** When (2) holds, the network's decision on x equals the decision it makes at a convex mixture of *same-label training representations*, and the perturbation δ that separates x from that mixture is provably too small to change the decision. In this precise, decision-relevant sense the prediction is an interpolation of training decisions.

If every training point of class y is classified correctly, then m_{yc}(h) > 0 for every h in the same-label hull. So the hull point itself is always classified correctly. Our diagnostic in Section 6.3 measures this as 100% in every setting. That is a sanity check on the implementation, not a finding. Everything therefore rests on δ.

**Remark 2 (label-free use).** Use ŷ = argmax f(x) in place of y. If (2) holds for a class c, the network predicts c, so at most one class can be certified. The certificate is *sound* for any h in the hull. In our implementation h comes from a nonnegative least-squares solve with a heavily weighted sum-to-one row, followed by renormalisation of λ. The resulting h is always a valid hull point, so an imperfect solve can only make the certificate more conservative, never unsound.

**Remark 3 (why only the readout).** For a nonlinear head g after layer ℓ < L − 1, g(h) is not a convex combination of training outputs. An analogue of (1) would need a Lipschitz bound on g and would lose the "interpolation of training decisions" reading. For earlier layers we therefore report only the descriptive profile I_ℓ.

**Remark 4 (conservativeness).** Inequality (1) uses Cauchy–Schwarz over the whole residual r = φ(x) − h. Components of r orthogonal to span{w_y − w_c} do not affect the decision. In Section 6.3 we measure what fraction of ‖r‖² the readout actually sees.

## 5 Experimental Setup

All experiments use fully connected ReLU networks implemented in numpy, with softmax cross-entropy, full-batch AdamW (β = (0.9, 0.98), decoupled weight decay), and Gaussian initialisation with variance 1/fan-in. Everything runs on 4 CPU cores; the full pipeline takes roughly 10 minutes. Hull projections use `scipy.optimize.nnls`, and exact membership uses `scipy.optimize.linprog` (HiGHS). Code: `code/`; raw outputs: `results/`.

**Experiment 1 (modular addition).**

- *Task:* p = 53, input e(a, b) ∈ {0,1}^{106}, architecture 106–128–128–53.
- *Optimisation:* lr 10⁻³, weight decay 1.0, 12,000 steps.
- *Runs:*
  - training fraction 0.5 with seeds 0, 1, 2;
  - fractions 0.4 and 0.6 with seed 0;
  - a random-label control: fraction 0.5, each of the p² pairs given an independent uniformly random label.
- *Measurements:* every 500 steps we compute, on a fixed subset of 250 held-out pairs, I_ℓ and the 1-NN accuracy for ℓ = 0 (input), 1 and 2 (penultimate), the median relative distance d_ℓ(x, y), and the certified fraction C at layer 2. Test accuracy is reported on the full held-out set.

**Experiment 2 (digits).**

- *Data:* UCI optical handwritten digits, 8×8, 1,797 images as packaged in scikit-learn (Alpaydin & Kaynak, 1998). Each seed uses a random split of 600 training and 1,197 test images. Inputs are scaled as x/16 − 0.5.
- *Model and training:* architecture 64–128–64–10, lr 10⁻³, weight decay 10⁻², 1,500 steps, 5 seeds.
- *Shift probe:* we also evaluate every test image shifted right by 1 and by 2 pixels (zero fill).
- *Measurements:* I_ℓ for ℓ = 0, 1, 2; exact LP membership for 300 test images in raw pixel space; the certified fraction under the true and under the predicted label; and accuracy on certified versus uncertified points.
- *Error detection:* we compare four scores for detecting misclassifications, by pooled AUROC: negative max-softmax; 1-NN distance in layer 2; hull distance d₂(x, ŷ); and *hull contrast*, defined as d₂(x, ŷ) minus the smallest d₂ over the other classes.

An earlier run used per-pixel standardisation. Under the shift probe this blew up near-constant border pixels, so it was superseded. Its outputs are kept in `results/digits_perpixelstd_prelim*` and are qualitatively similar.

**Experiment 3 (diagnostic).** The purpose is to explain the certificate's failures. We retrain modular addition (fractions 0.5, 0.7, 0.9; 12,000 steps) and digits (600 of 1,797 images; training sizes 100, 300, 600, 1,200) with seed 0. For up to 400 test points we record:

- the relative residual δ / r;
- the fraction of ‖r‖² lying in span{w_c − w_y};
- the margins at φ(x) and at h.

## 6 Results

All numbers below are produced by the released code and stored in `results/summary.json`, `results/diag.json` and the per-run JSON files.

### 6.1 Experiment 1: in modular addition, interpolation appears only at the last layer, and certification lags generalization

**Grokking reproduced.** With training fraction 0.5, all three seeds fit the training set by step 500 and grok later. Test accuracy crosses 50% at steps 6,500, 6,500 and 7,500, and ends at 0.998, 0.999 and 0.999. With fraction 0.6, test accuracy crosses 50% at step 3,000 and reaches 1.000. With fraction 0.4 the network had not grokked by step 12,000 (test accuracy 0.138). The random-label control stays at chance (0.019; chance = 1/53 ≈ 0.019).

**Table 1. Final-checkpoint interpolation profile on modular addition** (250 held-out pairs; step 12,000).

| run | test acc | I₀ (input) | I₁ | I₂ | 1-NN acc (L2) | median d₀ / d₁ / d₂ | certified C |
|---|---|---|---|---|---|---|---|
| frac 0.5, seed 0 | 0.998 | 0.000 | 0.048 | 1.000 | 1.000 | 1.03 / 0.87 / 0.40 | 0.892 |
| frac 0.5, seed 1 | 0.999 | 0.000 | 0.044 | 0.996 | 0.996 | 1.03 / 0.88 / 0.33 | 0.964 |
| frac 0.5, seed 2 | 0.999 | 0.000 | 0.012 | 1.000 | 1.000 | 1.03 / 0.88 / 0.41 | 0.808 |
| frac 0.6, seed 0 | 1.000 | 0.000 | 0.068 | 1.000 | 1.000 | 1.03 / 0.84 / 0.24 | 1.000 |
| frac 0.4, seed 0 (not grokked) | 0.138 | 0.000 | 0.000 | 0.096 | 0.060 | 1.03 / 0.96 / 0.89 | 0.000 |
| frac 0.5, random labels | 0.019 | 0.024 | 0.012 | 0.008 | 0.012 | 0.88 / 0.92 / 0.96 | 0.000 |

Four observations follow.

1. **The input layer is exactly anti-interpolative.** In every real-label run, I₀ = 0 and 1-NN accuracy = 0 at all 25 checkpoints, as Proposition 2 predicts.
2. **The first hidden layer is still not interpolative after grokking.** I₁ ≤ 0.07 even when the network is at 100%, and median relative hull distances in layer 1 (0.84–0.88) are barely below those at the input (1.03). Layer 1 of the grokked network computes features of a and b that the readout combines in a rule-like way. Test pairs are not mixtures of same-label training pairs there.
3. **The penultimate layer becomes interpolative in lock-step with generalization.** I₂ and the layer-2 1-NN accuracy track test accuracy over the whole trajectory (Fig. 1, left). Both cross 0.5 at the same checkpoint as test accuracy in all four grokking runs (`step_L2hull_ge_0.5` vs `step_test_acc_ge_0.5`: 6,500/6,500, 6,500/6,500, 7,500/7,500, and 3,500 vs 3,000 for frac 0.6). The interpolation depth is therefore L − 1 = 2: the network places test pairs among same-sum training pairs only in its last representation.
4. **Certification lags.** The certified fraction C stays near 0 while test accuracy rises, and climbs only later (Fig. 1, red). Lags at the 50% level:

   | run | test acc ≥ 0.5 | C ≥ 0.5 | lag |
   |---|---|---|---|
   | frac 0.5, seed 0 | step 6,500 | step 10,500 | 4,000 |
   | frac 0.5, seed 1 | step 6,500 | step 9,500 | 3,000 |
   | frac 0.5, seed 2 | step 7,500 | step 11,000 | 3,500 |
   | frac 0.6, seed 0 | step 3,000 | step 5,500 | 2,500 |

   At the 95% level, frac 0.6 goes from step 4,000 (test accuracy) to 7,500 (C), and seed 1 from 8,500 to 11,500. Seeds 0 and 2 do not reach C ≥ 0.95 within 12,000 steps (final C 0.892 and 0.808). Fig. 2 plots C against test accuracy: every trajectory runs below the diagonal.

**Interpretation.** In the window between the two crossings, the network classifies most held-out pairs correctly, and its penultimate representation already places them nearer the correct class hull than any other. Yet the decisions are *not* certifiable as interpolations: the residual from the same-label hull is too large, relative to the margin, for (2) to hold. The network is right, but not *because* of convex interpolation. It relies on how the readout weighs directions in which test points lie outside the hull. Continued training under weight decay shrinks the residual (median d₂ ends at 0.24–0.41) until the same decisions become certified. This ordering, uncertified generalization first and certified interpolation later, is what one would expect if structure-building (circuit formation) precedes consolidation (cleanup) as described by Nanda et al. (2023). Our measure detects it without any knowledge of the Fourier algorithm. The random-label control shows that the metrics do not rise simply because the training data are fitted.

### 6.2 Experiment 2: on digits, interpolation explains the task at the input already, and the certificate separates reliable predictions

**Table 2. Digits, 5 seeds** (mean ± s.d. across seeds; 1,197 test images per seed).

| condition | network acc | I₀ (pixels) | I₁ | I₂ | certified C (pred. label) | acc on certified | acc on uncertified |
|---|---|---|---|---|---|---|---|
| no shift | 0.971 ± 0.005 | 0.984 ± 0.003 | 0.983 ± 0.003 | 0.979 ± 0.004 | 0.876 ± 0.006 | 0.9975 ± 0.001 | 0.785 ± 0.047 |
| shift 1 px | 0.445 ± 0.030 | 0.559 ± 0.020 | 0.537 ± 0.027 | 0.501 ± 0.025 | 0.147 ± 0.019 | 0.909 ± 0.052 | 0.364 ± 0.033 |
| shift 2 px | 0.094 ± 0.010 | 0.151 ± 0.022 | 0.151 ± 0.029 | 0.147 ± 0.010 | 0.003 ± 0.003 | (20 points pooled; 0.55) | 0.093 ± 0.011 |

Pooled over seeds without shift, 5,245 of 5,985 test predictions are certified. Their accuracy is 0.9975, against 0.786 for the 740 uncertified ones.

**Hull membership is never satisfied, yet relative interpolation solves the task.** Not one of 1,500 test images (300 per seed) lies in the pixel-space training hull, or in its own class's hull. Yet the pixel-space nearest-class-hull classifier reaches 98.4%, *higher* than the trained MLP (97.1%). The interpolation depth of this task is therefore 0. Digits and modular addition sit at opposite ends: in one, test points are already relative interpolations in the input; in the other, they become so only at the last layer, and input proximity is exactly misleading. Asking "does the model generalize or interpolate?" without naming the task and the layer conflates these two very different situations.

**Certification identifies reliable predictions.** On unshifted data, 87.6% of predictions are certified, with accuracy 99.75% against 78.5% for the uncertified ones. Without shift the network makes 171 errors in total across seeds. Most of them fall in the uncertified 12%.

**Certification collapses under shift faster than accuracy does.** A 1-pixel shift reduces accuracy from 0.971 to 0.445 and the certified fraction from 0.876 to 0.147. The certified predictions remain 91% accurate, against 36% for uncertified ones. At 2 pixels, accuracy is 0.094, below the chance level of 0.1, and essentially nothing is certified (0.3%). Even under shift, the pixel-space nearest-class-hull classifier beats the network (0.559 vs 0.445 at 1 px; 0.151 vs 0.094 at 2 px). The MLP's learned representation thus extrapolates *worse* than naive input-space interpolation on this shift. This is consistent with Xu et al.'s (2021) observation that MLPs do not extrapolate outside the training support.

**Table 3. Error detection, pooled AUROC over 5 seeds** (n = 5,985 per condition).

| condition (errors) | −max softmax | 1-NN distance (L2) | hull distance d₂(x, ŷ) | hull contrast |
|---|---|---|---|---|
| no shift (171) | 0.956 | 0.932 | 0.960 | **0.975** |
| shift 1 px (3,324) | 0.686 | 0.782 | 0.806 | **0.808** |
| shift 2 px (5,420) | 0.482 | 0.501 | 0.511 | **0.669** |

Hull-based scores detect errors at least as well as softmax confidence in distribution and clearly better under shift. At 2 px, softmax confidence is *anti*-informative (AUROC < 0.5): the network is more confident on its errors. Hull contrast still separates errors from correct predictions (0.669). We report this as a property of the measurement, not as a state-of-the-art OOD detector; we did not compare with dedicated detectors.

### 6.3 Experiment 3: why certification fails, and how it scales with data

**Table 4. Certificate diagnostics** (seed 0; up to 400 test points; penultimate layer).

| task | train per class (mean) | test acc | certified C | hull point correct | median δ/r | median visible fraction of ‖r‖² | median margin at φ(x) / at h |
|---|---|---|---|---|---|---|---|
| modadd, frac 0.5 | 26.5 | 0.998 | 0.908 | 1.000 | 0.40 | 0.94 | 10.4 / 16.3 |
| modadd, frac 0.7 | 37.1 | 1.000 | 1.000 | 1.000 | 0.18 | 0.93 | 14.2 / 16.3 |
| modadd, frac 0.9 | 47.7 | 1.000 | 1.000 | 1.000 | 0.14 | 0.82 | 14.4 / 16.0 |
| digits, n = 100 | 10 | 0.918 | 0.680 | 1.000 | 0.30 | 0.76 | 15.0 / 21.3 |
| digits, n = 300 | 30 | 0.955 | 0.818 | 1.000 | 0.19 | 0.69 | 21.0 / 24.2 |
| digits, n = 600 | 60 | 0.974 | 0.868 | 1.000 | 0.17 | 0.63 | 25.0 / 26.9 |
| digits, n = 1,200 | 120 | 0.983 | 0.915 | 1.000 | 0.12 | 0.53 | 29.8 / 30.6 |

Five patterns stand out.

1. As guaranteed by Remark 1, the hull point h is always classified correctly (100% everywhere). Every certification failure is due to the residual δ.
2. The residual shrinks as training data per class grows, and the certified fraction rises with it: from 0.68 to 0.92 on digits, and from 0.91 to 1.00 on modular addition. More data fill the hull, as an interpolation account predicts.
3. On modular addition, the readout "sees" most of the residual (median 82–94% of ‖r‖²). On digits it sees a decreasing share (76% down to 53%). In other words, the uncertified part of the modular-addition representation is aligned with the decision directions: test points differ from interpolations in exactly the directions that matter.
4. The margin at the actual test representation is smaller than at its hull projection in every setting. The out-of-hull residual reduces the margin rather than increasing it, so the test point is classified "despite" its residual, not "because" of it.
5. The modular-addition 0.5 result here (C = 0.908 on 400 points) agrees with the Experiment 1 value for the same configuration (0.892 on 250 points).

## 7 Discussion

### 7.1 What this says about "generalization vs. interpolation"

Our results support three claims, each tied to the evidence above.

**(a) Input-space interpolation is the wrong yardstick.** For discrete inputs it is vacuous (Proposition 1). In algorithmic tasks it can be exactly anti-correlated with the right answer (Proposition 2, Table 1). For continuous data, strict membership fails too, while relative interpolation can be excellent (Table 2). Arguments of the form "the model succeeded on inputs unlike its training data, so it does not merely interpolate" are therefore uninformative when the measure of "unlike" is input geometry.

**(b) In the representations of networks that generalize, test points become interpolations, and with enough data or training the decisions become certifiably so.** Fully grokked modular-addition networks reach 81–100% certified at the end of training, and 100% at training fractions ≥ 0.6–0.7. On digits, 87.6% of decisions are certified. Read this way, generalization *is* interpolation, but in a learned space. The network's achievement is to construct that space. This reconciles the "direct fit" view (Hasson et al., 2020) with the observation that grokking networks implement algorithms (Nanda et al., 2023): the algorithm *is* the map into a space where interpolation works.

**(c) The order matters, and it is not "interpolation first".** In grokking, correct generalization appears *before* it can be certified as interpolation, by 2,500–4,000 steps. During that window, a rule-like computation (the readout exploiting the structure of a representation in which test points still lie well outside same-label hulls) produces correct answers that convex mixing of training decisions cannot account for. The simplest description of this window is "generalization that is not interpolation". It then consolidates into interpolation. On this evidence, neither the pure-interpolation nor the pure-generalization reading of the title question is right for these models. Both regimes occur, in a consistent order, and our certificate tells them apart per test point and per checkpoint.

### 7.2 Practical use

The certificate costs one small NNLS problem per test point and class, and gives a per-prediction diagnostic. Certified predictions were 99.75% accurate in distribution and 91% accurate after a 1-pixel shift, against 79% and 36% for uncertified ones. Hull contrast stayed informative where softmax confidence inverted. We see two uses:

- an *auditing* statistic, reporting what fraction of a model's decisions on an evaluation set are certifiable interpolations of its training decisions;
- a *training progress measure*, since certification lag flags a model that is right for reasons not yet consolidated.

### 7.3 Falsifiable predictions and a protocol for larger models

We did not run large models. The following predictions can falsify our account.

- **P1 (lag in transformers).** In one-layer and two-layer transformers trained on modular addition, as in Nanda et al. (2023), with the final residual stream as φ, the certified fraction crosses 50% strictly after test accuracy does. The lag should shrink as the training fraction grows, as here (4,000 → 2,500 steps from fraction 0.5 to 0.6).
  *Falsified if* certification precedes or coincides with generalization across seeds.
- **P2 (depth profile).** For tasks requiring a learned algorithm, interpolation depth equals the last layer. For perceptual tasks with input-level similarity structure, it is shallow.
  *Falsified if* algorithmic tasks show I_ℓ close to accuracy at early layers.
- **P3 (counterfactual tasks).** Take an LLM evaluated on default versus counterfactual task variants (Wu et al., 2024). Build per-answer-class hulls from final-layer representations of a reference set of default-variant items, which act as the "training" set. Then (i) the certified fraction is lower on counterfactual than on default items, and (ii) within each variant, accuracy is higher on certified than on uncertified items.
  *Falsified if* certified and uncertified accuracy do not differ, which would mean the certificate does not track the mechanism.

*Protocol for P3.* The readout is the unembedding restricted to the answer tokens, so Proposition 3 applies exactly. The hull uses representations of items with the same correct answer. Report C, accuracy split by certification status, and the hull-contrast AUROC, with bootstrap confidence intervals over items. Pre-register the reference set and the answer-token set.

## 8 Limitations

- **Scale and architecture.** All experiments use small MLPs (≤ 2 hidden layers, width ≤ 128) trained in numpy on CPU. We have not shown that the lag or the depth profile hold for transformers, convolutional networks, or language models. Section 7.3 states those as predictions, not results.
- **Few seeds and subsampling.** Experiment 1 has 3 seeds at fraction 0.5 and one seed each for the other conditions. Metrics are computed on a fixed 250-pair subsample, so the step resolution of the lag is 500 steps. Lag values are reported per run without confidence intervals. We did not sweep learning rate, weight decay or width. With fraction 0.4, grokking did not occur within the step budget, so we cannot say how the lag behaves near the grokking threshold.
- **The certificate is sufficient, not necessary, and conservative.** It uses Cauchy–Schwarz on the whole residual. An uncertified correct decision is not proven to be non-interpolative; it is only not certified. Table 4 shows that the readout sees 53–94% of the residual, so a sharper bound might certify more points, especially on digits. Statement (c) of Section 7.1 should be read with this in mind: what lags is *certifiable* interpolation under our bound. The relative measure I₂, which is not a certificate, does *not* lag.
- **Hull geometry in high dimension.** With 10–120 training points per class in 64–128 dimensions, same-label hulls are low-dimensional simplices, and both membership and distance depend on the number of points per class (Table 4). The relative distance normalisation r_ℓ is a convention. Comparisons across layers of different widths should be made with care.
- **Only the readout layer carries a decision-level guarantee.** For earlier layers (Remark 3), I_ℓ is descriptive.
- **Small real dataset and synthetic shift.** The digits data are small and easy, and the pixel-shift probe is one specific, synthetic distribution shift. The error-detection comparison omits dedicated OOD and failure-prediction baselines.
- **No large-model evidence.** Nothing here establishes how much of a large language model's behaviour is certified interpolation.

## 9 Conclusion

"Does the model generalize or merely interpolate?" becomes answerable once one names the space in which interpolation is measured and requires the measure to be linked to the model's decision. We showed that the input-space convex-hull definition is vacuous for discrete inputs and can be exactly anti-informative. We gave a readout interpolation certificate that ties hull proximity to decisions, and we measured it through training and under shift. In grokking networks, test points become interpolations only in the last representation. That happens in step with generalization, while *certified* interpolation arrives thousands of steps later. On digits, certified decisions are nearly always right, and certification collapses under shift. The interpolation/generalization dichotomy is therefore better replaced by a measurable profile: at which layer, and from which point in training, a model's correct decisions become certifiable interpolations of its training decisions.

## References

Alpaydin, E., & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49

Balestriero, R., Pesenti, J., & LeCun, Y. (2021). Learning in High Dimension Always Amounts to Extrapolation. arXiv:2110.09485.

Belkin, M. (2021). Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. *Acta Numerica*, 30, 203–248. arXiv:2105.14368.

Bonnasse-Gahot, L. (2022). Interpolation, extrapolation, and local generalization in common neural networks. arXiv:2207.08648.

Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547.

Dziri, N., et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2305.18654.

Gromov, A. (2023). Grokking modular arithmetic. arXiv:2301.02679.

Hasson, U., Nastase, S. A., & Goldstein, A. (2020). Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks. *Neuron*, 105(3), 416–434. https://doi.org/10.1016/j.neuron.2019.12.002

Hwang and Shin (2024). Uncertainty Measurement of Deep Learning System based on the Convex Hull of Training Sets. arXiv:2405.16082.

Lake, B., & Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, 2873–2882. arXiv:1711.00350.

Liu, Z., Kitouni, O., Nolte, N., Michaud, E., Tegmark, M., & Williams, M. (2022). Towards Understanding Grokking: An Effective Theory of Representation Learning. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2205.10343.

Liu, Z., Michaud, E. J., & Tegmark, M. (2023). Omnigrok: Grokking Beyond Algorithmic Data. *International Conference on Learning Representations (ICLR 2023)*. arXiv:2210.01117.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations (ICLR 2023)*. arXiv:2301.05217.

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *Proceedings of the National Academy of Sciences*, 117(40), 24652–24663. https://doi.org/10.1073/pnas.2015509117

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

Wu, Z., Qiu, L., Ross, A., Akyürek, E., Chen, B., Wang, B., Kim, N., Andreas, J., & Kim, Y. (2024). Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks. *Proceedings of NAACL 2024 (Volume 1: Long Papers)*, 1819–1862. https://doi.org/10.18653/v1/2024.naacl-long.102

Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K., & Jegelka, S. (2021). How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks. *International Conference on Learning Representations (ICLR 2021)*. arXiv:2009.11848.

Yousefzadeh, R. (2021). Deep Learning Generalization and the Convex Hull of Training Sets. arXiv:2101.09849 (NeurIPS 2020 Workshop on Deep Learning through Information Geometry).

Yousefzadeh, R. (2022). Decision boundaries and convex hulls in the feature space that deep learning functions learn from images. arXiv:2202.04052.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *International Conference on Learning Representations (ICLR 2017)*. arXiv:1611.03530.

---

### Figures (in `results/`)

- **Figure 1** (`fig1_modadd_dynamics.png`): test accuracy, layer-wise nearest-class-hull accuracy, layer-2 1-NN accuracy, input-space accuracy (identically 0), and certified fraction over training, shown for fraction 0.5 (seed 0), fraction 0.4, and the random-label control.
- **Figure 2** (`fig2_cert_vs_acc.png`): certified fraction plotted against test accuracy along training for all real-label runs. All trajectories lie below the diagonal, which is the certification lag.
- **Figure 3** (`fig3_digits.png`): digits results. Left: accuracy, I₀/I₁/I₂ and certified fraction under shift. Right: error-detection AUROC by score.

### Data and code provenance

Every number in Sections 4–6 was produced by the scripts in `code/` in this environment. The values come from `results/summary.json` (Experiments 1–2), the per-run `results/modadd_*.json` files, `results/digits.json`, and `results/diag.json` (Experiment 3). No number is taken from external sources except the bibliographic facts in Sections 1–2.
