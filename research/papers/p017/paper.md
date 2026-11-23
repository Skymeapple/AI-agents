# What a Probe Gets for Free: Suffix Ceilings, Placement Floors, and When Decodable World States Are Evidence of a World Model

**Target venue:** NeurIPS 2027 (Main Track; Evaluations & Interpretability). Code and all raw outputs: `code/`, `results/` (round-2 tables in `results/summary_r2.md`).

## Abstract

A common argument that a sequence model "has a world model" is that a probe decodes the latent world state from its activations, and that the decoding beats an untrained-network baseline. We argue that the right comparison is with the *information an encoder gets for free from its architecture*, and we make this precise.

**Theory.** For any recurrent encoder with fading memory, and for any finite-receptive-field model, probe accuracy is bounded by the *k-suffix ceiling* Acc\*\_k: the Bayes accuracy attainable from the last k tokens, plus an explicit margin term (Theorem 1, Corollary 2). The ceiling is tight. An *untrained* contracting encoder with random tanh features attains Acc\*\_k exactly with a linear probe once its width reaches the number of reachable k-suffixes (Proposition 4). What training buys is therefore memory at sub-exponential width, not decodability as such.

**Gridworld with exact ceilings.** We test the theory in a torus gridworld where Acc\*\_k is computed exactly from the true finite-time prior.
- *Truncation test.* Theorem 1's prediction holds in all 324 cells, with a maximum excess of +0.002.
- *Untrained encoders.* Across 9 configurations of untrained d = 512 reservoirs, including orthogonal near-isometries that keep long memory, linear probes never exceed the 4-step ceiling. MLP probes reach at most 7 steps.
- *Trained GRUs.* Across 40 independently trained GRUs, the trained models reach effective suffix lengths of 15–24 steps after 30 epochs.
- *Behaviour precedes state.* After 10 epochs the landmark-world GRUs predict the next observation with accuracy 0.959 ± 0.018, but their linear state probe reaches only 0.537 ± 0.196.
- *Interventions.* Across independent runs, the gain of probe-subspace interventions over random-subspace interventions tracks probe accuracy (r = 0.88, 95% CI [0.81, 0.93]).

**Othello.** For transformers the free information is different. Attention can read off an order-free, parity-split bag of the moves, that is, *where each player has placed discs*, without simulating a single flip.
- We trained a small Othello GPT on CPU. Its top-1 move is legal 92.4% of the time.
- Its linear board probe reaches 0.812 per square, against 0.698 for its own untrained initialisation, which is the conventional baseline.
- A decoder that sees only disc placement and the move index, and simulates no flips, already reaches 0.879. A nonlinear probe on the trained model reaches 0.835, so it stays below this floor as well.
- Stratifying by how long ago each square last changed shows the probe's error is spread evenly over square ages. In contrast, decoders limited to the last k moves fail precisely on old squares.
- ⟦LONG⟧

The untrained-twin baseline thus understated the free information by 18 points. A world-model claim needs to clear the floor appropriate to the architecture, stratified by how much history the answer requires.

## 中文摘要

判断序列模型"拥有世界模型"的常见论据是：探针能从激活中解码出潜在世界状态，而且解码准确率高于未训练网络这一基线。本文认为，正确的比较对象是*编码器凭其结构即可免费获得的信息*，并对此给出了精确刻画。

**理论。** 对任何具有衰减记忆的循环编码器，以及任何感受野有限的模型，探针准确率都不超过 *k-后缀上限* Acc\*\_k（仅凭最近 k 个词元可达到的贝叶斯准确率）加上一个显式的间隔项（定理 1、推论 2）。该上限是紧的：当宽度达到可达 k-后缀的个数时，带随机 tanh 特征的*未训练*压缩编码器配合线性探针即可恰好达到 Acc\*\_k（命题 4）。因此，训练带来的是以次指数宽度实现的记忆，而不是可解码性本身。

**可精确计算上限的网格世界。** 我们在环面网格世界中检验该理论；在这里，Acc\*\_k 可依据真实的有限时间先验精确计算。
- *截断检验*：定理 1 的预测在全部 324 个单元中均成立，最大超出量为 +0.002。
- *未训练编码器*：在 9 种 d = 512 的未训练储备池配置中（包括保持长记忆的正交近等距映射），线性探针从未超过 4 步窗口的上限，MLP 探针最多达到 7 步。
- *训练过的 GRU*：在 40 个独立训练的 GRU 中，训练 30 轮后有效后缀长度达到 15–24 步。
- *行为先于状态*：训练 10 轮后，地标世界中的 GRU 预测下一观测的准确率为 0.959 ± 0.018，但线性状态探针仅为 0.537 ± 0.196。
- *干预*：在各独立运行之间，探针子空间干预相对随机子空间干预的增益与探针准确率高度相关（r = 0.88，95% 置信区间 [0.81, 0.93]）。

**黑白棋（Othello）。** 对 Transformer 而言，免费信息有所不同：注意力可以读出一个与顺序无关、按奇偶拆分的走子集合，即*双方各在哪些格子落过子*，而无需模拟任何一次翻转。
- 我们在 CPU 上训练了一个小型 Othello GPT，其首选走法合法的比例为 92.4%。
- 其线性棋盘探针的逐格准确率为 0.812，而作为常规基线的自身未训练初始化仅为 0.698。
- 一个只看落子位置与步数、完全不模拟翻转的解码器即可达到 0.879。训练模型上的非线性探针为 0.835，同样低于这一下限。
- 按格子距上次变化的时间分层可见：探针的误差在各个"年龄"上分布均匀；而只看最近 k 步的解码器恰恰在"年老"的格子上失败。
- ⟦LONG-ZH⟧

由此可见，未训练孪生模型这一基线把免费信息低估了 18 个百分点。世界模型的主张需要越过与模型结构相匹配的下限，并按答案所需历史的长短分层检验。

⟦BODY⟧
