# Where Unification Stops Paying: A Capacity–Data Model of Competition and Synergy in Unified Multimodal Models, with Small-Scale Tests

**Target venue:** NeurIPS 2027 (main track; theory + controlled experiments)

**Code and data:** `code/` (run instructions in `code/README.md`). Every number below is either read from a JSON file in `results/` produced by those scripts in this environment, or attributed to a cited paper. Revision 2 added random designs, a fine-grid threshold study, a re-run ReLU study, and small transformers trained on real image/audio/text token streams (`reviews/response_r1.md`). Revision 3 adds a validation-selected-rank experiment, a direct test of Cor. 3, three more transformer seeds (5 in total) with test-set bootstrap, and an optimisation control; it also narrows several claims (`reviews/response_r2.md`).

---

## Abstract

Unified "any-to-any" models put vision, audio, video, action and text tokens through one shared network. Mixed-modal scaling studies report that two modalities trained together can do worse than separately trained models of the same size ("competition") or better ("synergy"), and that the sign changes with model size. We study a deliberately minimal, analytically solvable model of this. Each modality is a linear regression task that reads a shared rank-$r$ core through its own head; its target depends on $k_s$ features shared across modalities and $k_p$ private ones. The derivations are elementary (Eckart–Young, the reduced-rank-regression degrees-of-freedom count, and the BBP spiked-matrix transition); what the paper contributes is what they imply when combined, plus tests of those implications. (i) At the population level the interaction is zero for $r\le k_s$, positive on the window $k_s<r<k_s+Mk_p$, and largest at $r=k_s+k_p$, which gives a modality budget $M^*=\lfloor (r-k_s)/k_p\rfloor$. (ii) A degrees-of-freedom (df) heuristic predicts that, at oracle width, finite-data synergy comes only from pooled estimation of shared directions and is bounded by an explicit ratio $\Gamma$. With no shared structure, a fully provisioned dense model pays a head-interference penalty. (iii) Under misalignment $\theta$ of the shared directions, synergy needs $\sin^2\theta\lesssim C/n$. (iv) Under power-law spectra, the capacity-bound loss ratio is the scale-free constant $R=(\phi M^{1/\alpha}+(1-\phi)M)^{\alpha}/M$, and data-bound synergy starts at a capacity that grows like $n^{1/\alpha}$. (v) Pooling lowers the spectral recovery threshold for shared features and raises it for private ones.

**Evidence.** In the linear model itself, Monte Carlo agrees with the df heuristic to within 2.3% (Exp. A), 3.9% (B) and 7.0% (C) maximum relative error on the loss ratio at $n=1000$. Under power-law spectra the error is 5–11% for intermediate $n$ ($2\cdot10^3$–$1.3\cdot10^5$), because a power-law spectrum always has directions near the noise edge. With random Gaussian designs the predictions hold for $n\ge 500$ after replacing $n$ by $n-d-1$. A fine-grid threshold study matches the asymptotic spiked-matrix prediction of the pooling shift (2.23–2.27 against 2.27–2.32 for shared features, 1.64–1.68 against 1.72–1.76 for private ones, at $d=256$). Selecting each model's rank on a validation set reproduces the oracle-width ratios to within 1%, whereas fixing both models at the same rank gives much lower ratios (0.53–0.68 against 0.94–0.96), so the bounded synergy applies to size-tuned comparisons and not to the fixed-size comparisons that are usually published. One-hidden-layer ReLU networks reproduce the competition window and its peak at width $k_s+k_p$, but not the model's constants. At matched (oracle) width their ratio is 3–82% above the predicted $\Gamma$, with competition where synergy is predicted for $M=8$; only the weak lower bound holds. Contrary to Prop. 1, they compete at and below width $k_s$. A control with no private features shows the same competition below $k_s$, and a linear model with the teacher's unequal feature strengths predicts its sign (not its size), so this is a failure of the equal-strength idealisation. Small decoder-only transformers (12k–442k parameters, 5 seeds) trained on real image (MNIST), audio (spoken digits) and text token streams show a small image-side competition (positive in 36 of 40 seed-level comparisons, at most 5% of the loss) that declines with model size and vanishes at the largest size with more data. In the one cell trained to convergence with a tuned learning rate it shrinks by about 40%. This observation does not discriminate our model from other capacity-sharing accounts. Audio shows no consistent interaction. The predicted rise-then-fall with model size (P1) and the modality-budget effect (P4) are *not* observed, and the P4 test is underpowered. At the single checkpoint a deployed model would use, the dominant interaction is overfitting-rate mismatch between modalities, a mechanism outside the model. Video and action are not studied. We give a pre-registered protocol for testing the remaining predictions at scale. None of this establishes that real unified transformers obey the model, and the question of where they stop paying remains open; the paper offers a solvable instance of capacity competition with explicit constants, and a record of where those constants stop transferring.

## 中文摘要

统一的"任意模态到任意模态"模型把视觉、音频、视频、动作和文本token送入同一个共享网络。混合模态缩放律研究发现：两种模态联合训练时，可能比同等规模的单独训练模型更差（"竞争"），也可能更好（"协同"），且符号随模型规模变化。我们研究一个刻意简化、可解析求解的模型：每个模态是一个线性回归任务，通过各自的输出头读取秩为 $r$ 的共享核心；目标依赖 $k_s$ 个跨模态共享特征和 $k_p$ 个私有特征。推导本身是初等的（Eckart–Young定理、降秩回归的自由度计数、BBP尖峰矩阵相变）；本文的贡献在于把它们组合后得到的推论，以及对这些推论的检验。(i) 总体层面上，交互项在 $r\le k_s$ 时为零，在窗口 $k_s<r<k_s+Mk_p$ 内为正，在 $r=k_s+k_p$ 处最大，由此给出模态预算 $M^*=\lfloor (r-k_s)/k_p\rfloor$。(ii) 自由度启发式预测：在最优宽度下，有限数据的协同只来自共享方向的合并估计，且上界为显式比值 $\Gamma$；若无共享结构，即使容量充足，稠密模型也要付出"输出头干扰"代价。(iii) 共享方向错位角为 $\theta$ 时，协同要求 $\sin^2\theta\lesssim C/n$。(iv) 在幂律谱下，容量受限时的损失比为与规模无关的常数 $R$，数据受限时的协同出现在按 $n^{1/\alpha}$ 增长的容量处。(v) 合并训练降低共享特征的谱恢复阈值，提高私有特征的阈值。证据：在线性模型内部，$n=1000$ 时模拟与自由度启发式的损失比最大相对误差为2.3%（实验A）、3.9%（B）、7.0%（C）；幂律谱下中等 $n$ 的误差为5–11%。随机高斯设计下，把 $n$ 换成 $n-d-1$ 后，预测在 $n\ge500$ 时成立。细网格阈值实验与尖峰矩阵渐近预测一致（$d=256$ 时共享特征移位2.23–2.27，预测2.27–2.32）。在验证集上为每个模型选择秩，所得比值与最优宽度口径相差不超过1%；而把两个模型固定在同一秩时比值低得多（0.53–0.68，对比0.94–0.96），因此"协同有界"的结论适用于各自调好规模的比较，而不适用于文献中常见的固定规模比较。单隐层ReLU网络重现了竞争窗口及其在宽度 $k_s+k_p$ 处的峰值，但没有重现模型的常数：在最优宽度口径下，其比值比预测的 $\Gamma$ 高3–82%，$M=8$ 时在预测为协同之处仍是竞争，只有较弱的下界成立。与命题1不符，在宽度不超过 $k_s$ 时它们仍表现出竞争；去掉私有特征的对照实验中，宽度小于 $k_s$ 时竞争依旧，而采用教师网络不等特征强度的线性模型能预测其符号（但不能预测其大小），因此这是"特征强度相等"这一理想化假设的失败。在真实图像（MNIST）、音频（语音数字）和文本token流上训练的小型解码器Transformer（1.2万–44万参数，5个随机种子）中，图像模态表现出小幅竞争（40次种子级配对比较中36次为正，不超过损失的5%），它随模型增大而减弱，在数据较多的最大模型上消失；在唯一一个训练至收敛并调过学习率的单元中，该效应缩小约40%。这一观察不能把本文模型与其他"共享容量"解释区分开。音频没有一致的交互。随模型规模先升后降的预测（P1）和模态预算效应（P4）均未观察到，且P4的检验功效不足。在实际部署所用的单一检查点上，主导交互来自各模态过拟合速度的不匹配，这一机制不在模型之内。本文未研究视频和动作模态。这些结果并不证明真实的统一Transformer服从该模型，统一模型的极限在何处仍是未解问题；本文给出的是容量竞争的一个带显式常数的可解实例，以及这些常数在何处失效的记录。

---

## 1 Introduction

One line of work in multimodal AI aims at a single network that reads and writes every modality. Gato serialised 604 tasks across modalities and embodiments into one token stream [Reed et al., 2022]. Chameleon trained an early-fusion token model on interleaved image–text data [Chameleon Team, 2024]. Emu3 argued that next-token prediction alone suffices for multimodal generation and perception [Emu3 Team, 2024]. BAGEL reported capabilities that emerge with scale in a unified pre-trained model [Deng et al., 2025]. The question is where this approach runs out. Is one model with $N$ parameters as good as $M$ specialists of size $N$, as good as one specialist of size $N$, or worse than both, and what decides which?

The best empirical evidence comes from mixed-modal scaling laws. Aghajanyan et al. [2023] trained over 250 models on seven modalities and added an interaction term to unimodal scaling laws. They call two modalities *competing* when joint training does worse than separate models trained on the same per-modality tokens, *synergistic* in the opposite case, and the boundary the *competition barrier*. Shukor et al. [2025] trained 457 native multimodal models. They found that early and late fusion scale similarly, that mixture-of-experts (MoE) models learn modality-specific weights and perform better, and that MoEs with learned (modality-agnostic) routing beat MoEs routed by modality. MoMa [Lin et al., 2024] and Mixture-of-Transformers [Liang et al., 2025] separate parameters by modality and report large compute savings. Multilingual modelling has a close analogue in the *curse of multilinguality* [Conneau et al., 2020; Chang et al., 2024], now modelled by capacity-aware scaling laws [Fernandes et al., 2023; Longpre et al., 2025].

**Scope and claim.** We do not claim that real unified transformers are linear shared-core models, and the "limits" derived below are limits *of that model*. The paper offers three things. (1) A minimal model in which competition, synergy, the barrier, a modality budget and a scale-invariant capacity tax can all be computed. (2) A list of quantitative predictions from it (P1–P6). (3) Tests of those predictions at three levels of realism: Monte Carlo inside the model (including random designs), one-hidden-layer ReLU networks, and small decoder-only transformers trained on real image, audio and text token streams. We report where the predictions fail. The transformer level is far too small (at most 442k parameters) to say anything about production-scale unified models, and video and action tokens are not studied at all. For evidence at scale we rely on others' work, in particular the controlled study of Han et al. [2026], which trains unified models up to 13.5B parameters and reports that whether modalities help or hurt each other depends on data complexity, with capacity competition for complex data. Our paper should be read as a solvable instance of that capacity-competition picture with explicit constants, not as an answer to where real unified models stop paying.

**Novelty, stated plainly.** Proposition 1 follows from Ky Fan / Eckart–Young once the model is written down. The finite-data analysis is the standard reduced-rank-regression df count [Izenman, 1975] applied to two matrix shapes. Remark 4 applies the rectangular spiked-matrix results of Benaych-Georges and Nadakuditi [2012] and Gavish and Donoho [2014]. The scale invariance of Proposition 3 comes from the self-similarity of a power-law spectrum, as in solvable scaling-law models [Bahri et al., 2024; Maloney et al., 2022; Paquette et al., 2024]. The qualitative picture, that tasks sharing a bottleneck compete unless they are redundant, is established: empirically at scale [Han et al., 2026; Hao et al., 2026] and information-theoretically [Khan, 2026]. What we believe is new is modest: closed-form constants inside one linear instance. These are the capacity tax $R$, $\kappa$; the alignment tolerance $\sin^2\theta_c\propto 1/n$; the separation of head interference from capacity as a cost of dense unification; and tests that show where these constants stop transferring.

**Contributions.**
1. A solvable shared-core model with exact population interaction (Prop. 1), modality budget (Cor. 1) and misalignment spectrum (Lemma 2).
2. A finite-data df heuristic (Heuristic 1), with bounded oracle-width synergy (Cor. 2), head interference (Cor. 3), alignment tolerance (Cor. 4), and a fixed-width variant (Remark 2). A simulation with validation-selected rank shows which of the two conventions a practitioner's comparison corresponds to (Sec. 5.4).
3. A scale-invariant capacity tax under power-law spectra (Prop. 3) and a data-bound crossover $r_\times\propto n^{1/\alpha}$.
4. Pooling shifts of the spectral recovery threshold (Remark 4), tested on a fine grid against asymptotic predictions.
5. Tests at three levels of realism, with errors reported across all experiments. The quantitative constants hold only inside the linear model; in ReLU networks and small transformers only qualitative features survive, and some predictions fail.
6. A pre-registered protocol (Section 7) for the predictions this paper cannot test.

## 2 Related Work

**Unified multimodal models.** Gato, Chameleon, Emu3 and BAGEL share most parameters across modalities; BAGEL already uses separate understanding and generation experts. MoMa and Mixture-of-Transformers separate non-embedding parameters by modality while keeping global attention; Liang et al. [2025] report matching a dense Chameleon-7B baseline with 55.8% of the FLOPs. Uni-X [Hao et al., 2026] measures gradient conflict between vision and text in a shared autoregressive transformer, finds it concentrated in the shallow and deep layers, and makes those layers modality-specific around a shared middle. That layout is the deep-network counterpart of our "shared core with modality-specific heads", and its conflict at the output end is related to our head-interference term, although gradient conflict is an optimisation quantity that our static model does not contain.

**Controlled studies of synergy and competition at scale.** Han et al. [2026] is the closest empirical work. In controlled unified-pretraining experiments, validated with 13.5B-parameter MoE models trained on 2T tokens, they report (per their abstract) that data complexity largely determines whether modalities help or hinder each other, and that shared attention and normalisation with modality-specific feed-forward layers promote synergy. This addresses our P1- and P6-type questions at a scale we cannot reach, and it does so without our model. We could read only the abstract and summaries of that paper, so we do not claim that its results confirm or refute any specific formula here. Our architectural discussion (Sec. 6) adds nothing empirical to theirs.

**Scaling laws and solvable models of them.** Kaplan et al. [2020] and Hoffmann et al. [2022] established power-law loss curves. Bahri et al. [2024] distinguish variance-limited and resolution-limited regimes. Maloney et al. [2022] derive power-law loss from a power-law data spectrum in a random-feature model. Paquette et al. [2024] give a solvable phase diagram of compute-optimal scaling. Our Proposition 3 uses the same spectral idealisation but asks a different question: the ratio between a pooled and a per-task spectrum truncated at the same rank. Aghajanyan et al. [2023] model mixed-modal interaction with an additive term. In multilingual NMT, Fernandes et al. [2023] find that changing the language mixture changes only the multiplicative constant of the scaling law, not its exponent. That is an empirical analogue of the scale-free ratio in Proposition 3, although in a different domain. ATLAS [Longpre et al., 2025] fits capacity-aware multilingual scaling laws and a cross-lingual transfer matrix.

**Theory of multi-task and multimodal learning.** Huang et al. [2022] prove modality competition in the optimisation dynamics of late-fusion networks. Wang et al. [2020] show that modalities overfit at different rates. Wu et al. [2020] analyse shared-module models. They show that misaligned task covariances cause negative transfer, and they *do* analyse how the capacity of the shared module affects transfer. (Our first version said otherwise; that was wrong.) Our addition to their analysis is the exact dependence on $M$, on the shared/private split, on data, and on power-law spectra. Ildiz et al. [2024] study multitask representation learning with heterogeneous task covariances in high dimensions and find multiple-descent risk curves and inverse scaling for individual tasks, a task-competition phenomenon that is close to ours in spirit. Ilbert et al. [2024] use random-matrix theory to obtain exact risks for multi-task regression with task weights $W_t=W_0+V_t$ (a shared part plus a task-specific deviation) in the proportional limit, and split the test risk into a signal term and a noise term responsible for negative transfer. Their decomposition is additive and has no rank constraint; ours is a shared/private split of directions under a rank constraint, treated only by a df heuristic, so their finite-data results are rigorous where ours are not. Khan [2026] gives an information-theoretic capacity–redundancy identity for multi-task learning: with limited shared capacity and weakly redundant tasks, negative transfer follows, which motivates private capacity for unrelated tasks. This is a distribution-free statement of the idea behind our Cor. 1; our budget $M^*$ is its explicit form in one linear model. Tripuraneni et al. [2021] and Du et al. [2021] bound the benefit of shared linear representations at the true representation dimension, so capacity competition does not arise there.

**Shared structure across modalities.** The Platonic Representation Hypothesis [Huh et al., 2024] would make the shared fraction $\phi$ large. The modality gap [Liang et al., 2022] corresponds to $\theta>0$.

**Curse of multilinguality.** Conneau et al. [2020] and Chang et al. [2024] document that adding languages hurts once capacity is limited; this is the language-only analogue of our window and budget.

## 3 Model

**Data.** There are $M$ modalities. Each tokenizer maps inputs into a common $d$-dimensional space, $x\sim\mathcal N(0,I_d)$. Modality $m$ has targets $y=W_mx+\varepsilon$, with $W_m\in\mathbb R^{q\times d}$, $\varepsilon\sim\mathcal N(0,\sigma^2 I_q)$, and $n$ samples per modality.

**Shared and private structure.** Let $S\in\mathbb R^{d\times k_s}$ and $P_1,\dots,P_M\in\mathbb R^{d\times k_p}$ have orthonormal, mutually orthogonal columns, and
$$W_m = s\,[A_m\;B_m]\,[S_m\;P_m]^\top,\qquad S_m=\cos\theta\,S+\sin\theta\,E_m ,$$
where $[A_m\,B_m]$ has orthonormal columns and $E_m$ (orthonormal, orthogonal to everything else) is a modality-specific "gap" direction. Write $k=k_s+k_p$ and $\phi=k_s/k$.

**Models.** *Unified:* $\hat y = V_m U^\top x$ with one $U\in\mathbb R^{d\times\rho}$, $\rho\le r$, and heads $V_m$. *Specialist:* each modality has its own $U_m$ of width $\le r$. This is the same-size, same-data comparison of Aghajanyan et al. [2023]. The per-modality excess risk is $L=\frac1M\sum_m\|\hat W_m-W_m\|_F^2$, the interaction is $\Delta=L_{\rm u}-L_{\rm s}$, and the ratio $L_{\rm u}/L_{\rm s}$ is $>1$ for competition. **Width convention.** Unless stated otherwise, both models use the *oracle* width $\rho\le r$ that minimises their risk ("oracle width"). Remark 2 and Section 5.7 also use the *fixed-width* convention $\rho=r$, which is how neural networks are usually compared. Oracle width is not available in practice; Sec. 5.4 shows that choosing the rank on a validation set gives nearly the same ratios, so "oracle width" stands for "each model's size tuned up to the cap $r$", not for "both models at size $r$".

## 4 Theory

### 4.1 Population (infinite data)

Write $G_m=W_m^\top W_m$, $G=\sum_m G_m$.

**Lemma 1 (variational form).** For orthonormal $U$, $\min_{V_m}\|W_m-V_mU^\top\|_F^2=\|W_m\|_F^2-\mathrm{tr}(U^\top G_mU)$, so the unified population risk is $\frac1M\sum_{i>\rho}\lambda_i(G)$ and the specialist risk is $\frac1M\sum_m\sum_{i>\rho}\lambda_i(G_m)$.

*Proof.* The optimal head is $V_m=W_mU$; sum over $m$ and apply Ky Fan's maximum principle [Eckart and Young, 1936]. $\square$

**Proposition 1 (exact population interaction, $\theta=0$).**
$$\Delta_{\rm pop}(r)=\begin{cases}0,& r\le k_s\\ s^2 (r-k_s)(1-\tfrac1M),& k_s\le r\le k\\ s^2\,(k_s+Mk_p-r)/M,& k\le r\le k_s+Mk_p\\ 0,& r\ge k_s+Mk_p,\end{cases}$$
with maximum $s^2(M-1)k_p/M$ at $r=k$.

*Proof.* $G=s^2(M\,SS^\top+\sum_mP_mP_m^\top)$ has eigenvalues $Ms^2$ ($\times k_s$) and $s^2$ ($\times Mk_p$); each $G_m$ has $k$ eigenvalues $s^2$. Apply Lemma 1 case by case. $\square$

**Corollary 1 (modality budget).** For $r\ge k$, a capacity-$r$ core matches specialists at the population level iff $M\le M^*=\lfloor (r-k_s)/k_p\rfloor$.

**Lemma 2 (misalignment spectrum).** With $c=\cos^2\theta$, each shared feature contributes to $G$ one eigenvalue $s^2(1+(M-1)c)$ and $M-1$ eigenvalues $s^2(1-c)$ (Gram matrix $(1-c)I+c\mathbf 1\mathbf 1^\top$). $\square$

**Remark 3 (unequal feature strengths: competition below $k_s$).** The tie for $r\le k_s$ in Prop. 1 depends on the equal-strength, orthonormal-head idealisation. Lemma 1 holds for arbitrary $W_m$, and Ky Fan's inequality $\sum_{i\le\rho}\lambda_i(\sum_mG_m)\le\sum_m\sum_{i\le\rho}\lambda_i(G_m)$ gives $\Delta_{\rm pop}(r)\ge0$ at every $r$, with equality only if one $r$-dimensional subspace is a top-$r$ eigenspace of every $G_m$. If modalities weight the shared features differently, each specialist keeps its own strongest $r$ directions, the unified core must compromise, and there is competition even when all features are shared and $r<k_s$. Section 5.7 shows that this, and not the private features, accounts for the sign of the interaction that narrow ReLU networks show.

### 4.2 Finite data

**Exact reduction (orthogonal design).** If $X_m^\top X_m=nI$, which requires $n\ge d$, then $\hat W_m^{\rm OLS}=W_m+(\sigma/\sqrt n)Z_m$. The unified least-squares estimator is the rank-$\rho$ truncated SVD of the stacked $[\hat W_1;\dots;\hat W_M]$, and each specialist is the truncated SVD of its own $\hat W_m$. **When $n<d$ no such design exists.** In that range we read the same Gaussian-perturbation model as an **SNR model**, in which $n_{\rm eff}=\sigma^2/\mathrm{Var}$ is only an inverse noise variance. Section 5.4 checks the conclusions with genuine random Gaussian designs ($n>d+1$). There the OLS noise has mean covariance $\sigma^2 I/(n-d-1)$, which suggests replacing $n$ by $n-d-1$.

**Heuristic 1 (df approximation; not a theorem).** At high SNR, first-order perturbation of the truncated SVD gives excess risk = discarded signal $+\frac{\sigma^2}{n}\times$ the dimension of the rank-$\rho$ manifold:
$$L_{\rm u}(r)=\min_{\rho\le r}\tfrac1M\Big[\textstyle\sum_{i>\rho}\lambda_i(G)+\tfrac{\sigma^2}{n}\rho\,(d+Mq-\rho)\Big],\quad L_{\rm s}(r)=\tfrac1M\sum_m\min_{\rho\le r}\Big[\textstyle\sum_{i>\rho}\lambda_i(G_m)+\tfrac{\sigma^2}{n}\rho\,(d+q-\rho)\Big].$$
It ignores directions near the noise edge, so it should fail at low SNR and, under power-law spectra, for the near-threshold tail. Section 5 shows both failures.

**Remark 1 (finite-sample synergy below $k_s$).** Proposition 1's "zero below $k_s$" is a population statement. With finite $n$, Heuristic 1 predicts synergy for $r\le k_s$, because both models then estimate the same shared directions and the unified model pools $M$ modalities. In Exp. A ($n=200$, $r=2,3,4$) the simulated ratios are 0.92, 0.88, 0.83 against predicted 0.93, 0.89, 0.84 (`analysis_r2.json`).

**Corollary 2 (oracle-width synergy is bounded).** If $r\ge k_s+Mk_p$ and $\theta=0$, the oracle-width ratio is
$$\Gamma=\frac{\rho_u(d+Mq-\rho_u)}{M\,k(d+q-k)},\qquad \rho_u=k_s+Mk_p ,$$
independent of $n$. Since $\Gamma=\frac{\rho_u}{Mk}\cdot\frac{d+Mq-\rho_u}{d+q-k}$, where the first factor is $1-\phi(1-1/M)$ and the second is $\ge1$ exactly when $q\ge k_p$, we have $\Gamma\ge 1-\phi(1-1/M)$ **provided $q\ge k_p$**, with the bound approached as $d/(Mq)\to\infty$. The condition holds automatically in the model ($W_m$ has rank $k\le q$). Without it the bound is false ($q\to0$ would push $\Gamma$ below it).

**Remark 2 (fixed width is a different comparison).** If both models are forced to use $\rho=r$ with $r>k$ (a specialist's rank cannot exceed its output dimension, so it is capped at $\min(r,q)$; in Exp. A that cap, $q=16$, binds for $r=20,24$), the specialist also pays $\frac{\sigma^2}{n}(r-k)(\cdots)$ for $r-k$ noise directions, and the unified model pays less per modality because those directions are shared. The fixed-width ratio can then fall far below $\Gamma$. In the Exp. A configuration at $n=1000$, Heuristic 1 gives fixed-width ratios of 0.53 ($r=20$) and 0.61 ($r=24$), against an oracle-width $\Gamma=0.94$ (`analysis_r2.json`). Cor. 2 bounds only the oracle-width comparison. Published fixed-size comparisons such as Aghajanyan et al. [2023] are of the fixed-width kind, so for them the relevant statement is this remark, not the "bounded synergy" of Cor. 2 (see Sec. 5.4 for a simulation of all three conventions).

**Corollary 3 (head interference).** If $\phi=0$ and capacity is ample, $\Gamma=(d+Mq-Mk_p)/(d+q-k_p)>1$ whenever $(M-1)(q-k_p)>0$.

**Corollary 4 (alignment tolerance; first order).** Assume $k_s+Mk_p\le r$ and that, at small $\theta$, the oracle unified model does not spend rank on the $(M-1)k_s$ gap directions. The core may have room for some of them (at $r=30$ in Exp. C, for 10 of the 12), but each one costs $\frac{\sigma^2}{n}(d+Mq-\rho)$ in variance and carries only $s^2\sin^2\theta$ of signal, so including it does not pay at small $\theta$. To first order in $\sin^2\theta$, the unified model loses $\frac{(M-1)k_s}{M}s^2\sin^2\theta$ and gains $G_n=\frac{\sigma^2}{n}\big[k(d+q-k)-\rho_u(d+Mq-\rho_u)/M\big]$, so synergy requires $\sin^2\theta<\sin^2\theta_c=\frac{M\,G_n}{(M-1)k_s s^2}\propto\frac1n$.

### 4.3 Power-law spectra: a scale-invariant capacity tax

Suppose feature energies $\lambda_j\propto j^{-\alpha}$ ($\alpha>1$), a fraction $\phi$ of them shared by all $M$ modalities, so one modality's counting function is $N_{\rm s}(\lambda)=\lambda^{-1/\alpha}$.

**Proposition 3.** In the continuum limit, at the population level, for every $r$ below the total feature count,
$$\frac{L_{\rm u}(r)}{L_{\rm s}(r)}=R(\phi,M,\alpha)=\frac{\big(\phi M^{1/\alpha}+(1-\phi)M\big)^{\alpha}}{M},\qquad \kappa=R^{1/(\alpha-1)},\qquad \frac{M}{\kappa}=\Big(\frac{M}{\phi M^{1/\alpha}+(1-\phi)M}\Big)^{\alpha/(\alpha-1)}\ge 1 .$$
*Proof.* The pooled counting function is $c_u\lambda^{-1/\alpha}$ with $c_u=\phi M^{1/\alpha}+(1-\phi)M$; truncating a spectrum with counting function $c\lambda^{-1/\alpha}$ at rank $r$ leaves $\frac{c^\alpha r^{1-\alpha}}{\alpha-1}$. Divide and solve for $\kappa$. $\square$

In this model, while data are abundant, competition does not fade with scale. A unified model needs $\kappa$ times one specialist's capacity, which is still $M/\kappa$ times less than $M$ specialists.

**Prediction (data-bound crossover).** Under Heuristic 1, the unified oracle width $\rho_u^*(n)$ is where the pooled spectrum crosses $\sigma^2(d+Mq)/n$, so $\rho_u^*\propto n^{1/\alpha}$ and $r_\times(n)\propto n^{1/\alpha}$.

### 4.4 Remark 4: pooling shifts spectral thresholds

For an $a\times b$ noise matrix with entry variance $v$ plus a rank-one spike of singular value $\vartheta$, the spike is detectable when $\vartheta^2>v\sqrt{ab}$ [Benaych-Georges and Nadakuditi, 2012]. With $v=\sigma^2/n$: a specialist ($q\times d$) detects a feature of energy $s^2$ when $n>\sigma^2\sqrt{qd}/s^2$. In the stacked unified matrix ($Mq\times d$), a shared feature has $\vartheta^2=Ms^2$, so its detection threshold is $\sqrt M$ times lower; a private feature's is $\sqrt M$ times higher. *Detection* is not *useful recovery*. With oracle hard truncation, a component lowers the error only if $y^2<2xy\,c_uc_v$ in the normalised notation of Gavish and Donoho [2014]: $x$ is the normalised spike, $y$ the limiting empirical singular value, and $c_u,c_v$ the limiting singular-vector overlaps. That needs a larger SNR. The asymptotic shift in the SNR at which a given fraction of the signal has been recovered therefore depends on the aspect ratios. For $d=4q$, $M=4$ it is 2.27–2.32 for shared features and 1.72–1.76 for private ones, not exactly $\sqrt M=2$. We test these values in Section 5.6.

## 5 Experiments

All experiments run on CPU. Unless stated, $s=\sigma=1$ and both models use the oracle width. Theory values come from `common.theory_losses` (Heuristic 1 on the exact spectrum of $G$). **Error metric** (used throughout, unless a sentence says otherwise): $|{\rm sim}-{\rm th}|/{\rm th}$ of the loss ratio. **Ratio definition:** `exp_linear.py` averages per-run ratios and `exp_gap.py` divides mean losses. In Exp. A the two differ by at most 0.09% (`analysis_r2.json`). **Window convention:** a window "$a$–$b$" lists the capacities with ratio $>1$; the code stores it as (first $r$ with ratio $>1$, first later $r$ with ratio $<1$) $=(a, b+1)$.

**Summary of theory–simulation agreement (all linear experiments; `analysis_r2.json`).**

| Experiment | $n$ | cells | max error | median error |
|---|---|---|---|---|
| A (capacity sweep) | 200 / 1000 / 5000 / $10^5$ | 23 each | 11.7% / 2.3% / 1.4% / 0.9% | 2.4% / 2.2% / 1.2% / 0.9% |
| B (modality budget) | 1000 / 5000 / $10^5$ | 14 each | 3.9% / 1.2% / 0.5% | 0.5% / 0.4% / 0.2% |
| C (misalignment grid) | 1000 | 140 | 7.0% | 1.7% |
| D (full capacity, SNR model) | 50 / 100 / 200 / 500 / 1000 / 3000 / $10^4$ / $10^5$ | 1 each | 21% / 8.9% / 11% / 4.3% / 1.8% / 0.26% / 0.22% / 0.34% | – |
| E2 (power law) | 500 / 2000 / 8000 / 32000 / 128000 / $10^7$ | 12 each | 5.5 / 5.5 / 5.9 / 6.6 / 11.3 / 0.65% | 0.9 / 4.1 / 4.6 / 5.3 / 3.7 / 0.1% |
| Random design (Sec. 5.4) | 500 / 1000 / 3000 | 8 each | see text (≤ 4.6%) | – |

### 5.1 Exp. A: the competition window

Setup: $d=64$, $q=16$, $k_s=k_p=4$, $M=4$; $r=2..24$; 30 repetitions (`linear_results.json`, `figA_capacity_sweep.png`).

| $n$ | window (sim) | window (df) | population window | ratio at $r=8$ (sim / df) | ratio at $r=24$ (sim / df) |
|---|---|---|---|---|---|
| 200 | $r\ge 6$, does not close by 24 | 6–18 | 5–19 | 1.40 / 1.46 | 1.05 / 0.94 |
| 1 000 | 5–19 | 5–19 | 5–19 | 5.50 / 5.62 | 0.96 / 0.94 |
| 5 000 | 5–19 | 5–19 | 5–19 | 26.1 / 26.5 | 0.94 / 0.94 |
| 100 000 | 5–19 | 5–19 | 5–19 | 516.5 / 521.3 | 0.94 / 0.94 |

The peak sits at $r=k=8$, and its height grows roughly linearly in $n$ because the specialist loss there is pure variance. For $n\ge1000$ the full-capacity ratio approaches $\Gamma=0.9375$; the gain is only about 6% because $Mq=64$ equals $d$. At $n=200$ the df heuristic fails (Sec. 5.4).

### 5.2 Exp. B: the modality budget

Setup: $d=128$, $q=16$, $k_s=k_p=4$, $r=16$; $M=1..14$; 20 repetitions. Corollary 1 predicts $M^*=3$. Simulated and df $M^*$ are both 3 for every $n\in\{10^3,5\cdot10^3,10^5\}$. The ratio at $M=4$ jumps to 1.58, 5.22, 92.5 (sim; df 1.57, 5.24, 92.6).

### 5.3 Exp. C: misalignment and alignment tolerance

Setup: $d=64$, $q=16$, $k_s=k_p=4$, $M=4$, $n=1000$, $\theta\in\{0^\circ,10^\circ,\dots,90^\circ\}$, $r\in\{4,6,\dots,30\}$, 12 repetitions (`figC_misalignment.png`). The df heuristic gets the sign right in 140/140 cells. **Most cells are far from the boundary, so that count is not informative on its own.** Only 15 cells have $|\log{\rm ratio}|<0.1$ under the df prediction; the sign agrees in all 15. For each of the 14 capacities, the smallest grid angle at which unification loses is the same in simulation and theory (0° for $6\le r\le18$, 10° for $r\ge20$, 20° for $r=4$), but the angle grid is only 10° wide. The boundary is located more precisely by the fine sweep (`exp_gap.py`, same $d$, $q$, $k_s$, $k_p$, $M$; $r=30$, 41 angles in $[0^\circ,20^\circ]$, 16 repetitions):

| $n$ | $\theta_c$ closed form | $\theta_c$ df (numerical) | $\theta_c$ simulation |
|---|---|---|---|
| 300 | 11.5° | 11.5° | none (no synergy at any $\theta$) |
| 1 000 | 6.3° | 6.3° | 4.8° |
| 3 000 | 3.6° | 3.6° | 3.5° |
| 10 000 | 2.0° | 2.0° | 2.1° |
| 30 000 | 1.1° | 1.1° | 1.1° |

For $n\ge3000$ the simulated $\theta_c$ is within 0.1° of the prediction, consistent with $\theta_c\propto n^{-1/2}$. At $n\le1000$ the low-SNR failure appears again.

### 5.4 Exp. D, random designs and width conventions: where the df heuristic holds

*SNR model (orthogonal design, Exp. D).* At full capacity ($r=24$, Exp. A setup) the error of the predicted loss ratio is 21%, 8.9%, 11%, 4.3%, 1.8%, 0.26%, 0.22% and 0.34% for $n_{\rm eff}=50, 100, 200, 500, 10^3, 3\cdot10^3, 10^4, 10^5$ (the metric of the summary table). An earlier version quoted a different quantity here, the error of the unified *loss* relative to its simulated value (17%, 8%, 19%, 9%, 4.6%, 1.8%, 0.8%, 0.4%); the ratio errors are smaller at large $n$ because errors in the two losses partly cancel. At $n_{\rm eff}<d=64$ this is the SNR model of Sec. 4.2, not a sample size.

*Random Gaussian designs* (`exp_random_design.py`, `random_design_results.json`; 6 repetitions; specialists by exact reduced-rank regression, unified by alternating least squares). Here $n$ is a real sample size, $n\in\{100,200,500,1000,3000\}$, and theory uses $n_{\rm eff}=n-d-1$. The theory value is computed from the last repetition's $W_m$; this is harmless because the spectra of $G$ and $G_m$ are identical across repetitions.

| $n$ | ratio $r=4$ (sim / df) | ratio $r=8$ | ratio $r=16$ | ratio $r=24$ (sim [95% CI] / df) | df with $n$ instead of $n_{\rm eff}$, $r=8$ |
|---|---|---|---|---|---|
| 100 | 0.86 / 0.94 | 0.86 / 0.94 | 0.86 / 0.94 | 0.86 [0.83, 0.87] / 0.94 | 0.91 |
| 200 | 0.80 / 0.79 | 1.12 / 1.12 | 1.07 / 1.01 | 1.03 [1.00, 1.06] / 0.94 | 1.46 |
| 500 | 0.92 / 0.91 | 2.74 / 2.68 | 1.59 / 1.53 | 0.98 [0.96, 1.00] / 0.94 | 3.02 |
| 1000 | 0.96 / 0.96 | 5.28 / 5.29 | 2.43 / 2.40 | 0.96 [0.93, 0.98] / 0.94 | 5.63 |
| 3000 | 0.99 / 0.99 | 15.9 / 15.7 | 5.95 / 5.87 | 0.93 [0.91, 0.95] / 0.94 | 16.0 |

With random designs, the window (ratio $>1$ for $6\le r\le16$, $<1$ at $r=20$), the peak at $r=8$ and $\Gamma$ all survive for $n\ge500$, with errors up to 4.6% (largest at $r=20$–24, $n=500$). The $n-d-1$ correction matters (at $n=500$, $r=8$, using $n$ predicts 3.02 against a simulated 2.74). At $n=200$, random designs reproduce the low-SNR failure of the SNR model: at full capacity, unification still loses slightly (1.03) where the df heuristic predicts 0.94. At $n=100$, specialists recover no signal at all; the unified model recovers some and wins (0.86), which is Remark 4's regime. ALS is not guaranteed to reach the global optimum, so the unified losses are, if anything, overestimates.

*Width conventions: oracle, validation-selected and fixed rank* (`exp_valrank.py`, `valrank_results.json`; Exp. A setup, SNR model, 40 repetitions, ratio of mean losses, bootstrap 95% CI over repetitions). The oracle rank uses the true risk, which no practitioner has. We therefore also select the rank on an independent validation estimate with $n_{\rm val}=n/4$ (one rank for the unified model, one per specialist), and compare with no selection at all ($\rho=r$, specialist capped at $q$).

| $n$ | convention | competition window | ratio $r=16$ | ratio $r=20$ | ratio $r=24$ [95% CI] |
|---|---|---|---|---|---|
| 1000 | oracle | 5–19 | 2.49 | 0.964 | 0.964 [0.958, 0.971] |
| 1000 | validation-selected | 5–19 | 2.48 | 0.963 | 0.963 [0.956, 0.969] |
| 1000 | fixed | 5–17 | 1.43 | 0.553 | 0.684 [0.681, 0.687] |
| 1000 | df heuristic, oracle / fixed | 5–19 / 5–17 | 2.51 / 1.41 | 0.937 / 0.527 | 0.937 / 0.609 |
| 5000 | oracle | 5–19 | 9.39 | 0.941 | 0.941 [0.935, 0.946] |
| 5000 | validation-selected | 5–19 | 9.38 | 0.940 | 0.940 [0.934, 0.946] |
| 5000 | fixed | 5–19 | 5.31 | 0.532 | 0.670 [0.668, 0.673] |

Validation-selected rank reproduces the oracle ratio to within 0.2% at every capacity for $n\ge1000$, and within 1.0% at $n=200$. So the oracle convention is not the source of the optimism in Cor. 2: a practitioner who tunes each model's rank gets the same bounded synergy (about 4–6% here). The fixed convention gives a different answer, a ratio of 0.53–0.68 beyond the window, as Remark 2 says. The df heuristic for fixed rank is accurate at $r=20$ (0.527 against 0.553 and 0.532) but too low at $r=24$ (0.609 against 0.684 and 0.670, a 10–12% error). At $n=200$ the fixed-rank ratios are 0.65 ($r=20$) and 0.74 ($r=24$), 22–24% above the heuristic, while oracle and validation-selected ranks both give 1.05 (no synergy; the low-SNR failure of Exp. D). We did not investigate the $r=24$ discrepancy further. The practical reading is that the model's "bounded synergy" describes size-tuned comparisons, and published fixed-size comparisons can show much larger synergy for a reason that has nothing to do with shared structure: the specialist is over-sized for its task.

### 5.5 Exp. E: power-law spectra

**E1 (numerical check of the continuum step only).** E1 compares the closed form of Prop. 3 with discrete sums over the *same* spectral model ($2\times10^5$ features). It verifies the integral approximation, not the model. At $r=1000$ the discrete ratio matches $R$ within 0.19% ($\alpha=2$), 0.70% ($\alpha=3$) and 4.1% ($\alpha=1.5$, which has a heavier truncated tail). Representative values:

| $\alpha$ | $\phi$ | $M$ | $R$ | $\kappa$ | $M/\kappa$ |
|---|---|---|---|---|---|
| 2 | 0.25 | 4 | 3.06 | 3.06 | 1.31 |
| 2 | 0.50 | 4 | 2.25 | 2.25 | 1.78 |
| 2 | 0.75 | 4 | 1.56 | 1.56 | 2.56 |
| 2 | 0.50 | 8 | 3.66 | 3.66 | 2.18 |
| 3 | 0.50 | 4 | 5.45 | 2.33 | 1.71 |
| 1.5 | 0.50 | 4 | 1.47 | 2.17 | 1.85 |

**E2 (finite data, finite feature count).** Setup: $d=512$, $q=96$, $M=4$, $\phi=0.5$, $\alpha=2$, 48 shared and $4\times48$ private features, 4 repetitions. With only 48 features per class, the *population* ratio is not flat. Over $r=2$–64 it rises monotonically from 1.6 to 4.64, so it is **not** "close to $R=2.25$". Flatness needs many more features than this setup has (E1). Simulation against the df heuristic, at all six $n$: the full-capacity ($r=240$) plateau is 0.90/0.91 ($n=500$), 0.94/0.90 ($2\cdot10^3$), 0.93/0.89 ($8\cdot10^3$), 0.91/0.86 ($3.2\cdot10^4$), 0.89/0.80 ($1.28\cdot10^5$) and 0.80/0.80 ($10^7$). Maximum errors are 5.5–11.3% for $2\cdot10^3\le n\le1.28\cdot10^5$, against 0.65% at $10^7$. The df heuristic thus *overstates* synergy by up to 0.09 in ratio at intermediate $n$. The same pattern appears with the full rank grid of `exp_crossover.py` (plateau 0.891 at $n=1.28\cdot10^5$), so it is not a grid artefact. Our explanation: a power-law spectrum always has retained directions just above the noise edge, where first-order perturbation overestimates how well they are recovered, and that tail vanishes only once every feature is far above the edge ($n=10^7$).

**E3 (crossover scaling).** For 9 values of $n\in[10^3,2.56\cdot10^5]$, the smallest $r$ beyond the peak with simulated ratio $<1$ (10 repetitions, $r$-grid step 2; `crossover_results.json`). Rows whose predicted $\rho_u^*$ hits the rank cap of 240 are excluded in the script, which leaves 8 points for $\alpha=2$ and 4 for $\alpha=1.5$.

| $\alpha$ | points | slope (sim) | 95% CI, bootstrap over reps | grid-resolution band | slope (df $\rho_u^*$) | asymptotic $1/\alpha$ |
|---|---|---|---|---|---|---|
| 2 | 8 | 0.567 | [0.567, 0.571] | 0.567–0.618 | 0.577 | 0.5 |
| 1.5 | 4 | 0.750 | [0.737, 0.750] | 0.750–0.809 | 0.782 | 0.667 |

The bootstrap CI is narrow because, with 10 repetitions, the crossover rarely moves between grid points. The dominant uncertainty is the step-2 grid (the true crossover lies in $(r_\times-2, r_\times]$), shown as a band. Both slopes exceed $1/\alpha$ by 13–20%, which we read as pre-asymptotic behaviour (the df prediction is above $1/\alpha$ by a similar amount). The $\alpha=1.5$ slope is fragile: it rests on 4 points spanning less than one decade of $n$ ($10^3$–$8\cdot10^3$), and dropping any one of them moves it between 0.72 and 0.79 (`analysis_r3.json`). Together with the grid band, all we can say for $\alpha=1.5$ is that the slope lies somewhere in 0.72–0.81, above $1/\alpha=0.667$; it does not test the exponent. For $\alpha=2$ the leave-one-out range is 0.549–0.573. $r_\times/\rho_u^*$ lies in 0.54–0.65 ($\alpha=2$).

### 5.6 Exp. G: pooling and spectral thresholds (Remark 4), fine grid

The round-1 version of this experiment used a coarse grid and read "onsets" by eye; we replace it. Setup (`exp_threshold_fine.py`): full capacity, $M=4$, total signal energy 8 per modality, purely shared ($k_s=8$) or purely private ($k_p=8$) structure. We use 40 log-spaced SNR values and 40 repetitions at $d=64,q=16$, plus a finite-size check at $d=256,q=64$ (12 repetitions). This is the SNR model: $n_{\rm eff}$ is an inverse noise variance. The statistic is the $n_{\rm eff}$ at which an estimator's mean loss first falls below a fraction $f$ of the signal (log-interpolated), and the shift is the ratio of these values. CIs come from bootstrap over repetitions. Asymptotic predictions use the formulas of Sec. 4.4 (`figG_threshold_fine.png`).

| case | $f$ | shift, $d=64$ (95% CI) | shift, $d=256$ (95% CI) | asymptotic |
|---|---|---|---|---|
| shared (unified earlier) | 0.9 | 2.06 [2.03, 2.09] | 2.23 [2.20, 2.25] | 2.27 |
| shared | 0.5 | 2.16 [2.13, 2.18] | 2.27 [2.25, 2.29] | 2.32 |
| private (unified later) | 0.9 | 1.37 [1.36, 1.38] | 1.68 [1.67, 1.69] | 1.76 |
| private | 0.5 | 1.37 [1.35, 1.38] | 1.64 [1.63, 1.65] | 1.72 |

Both shifts go in the predicted direction, and at $d=256$ they lie within 2–5% of the asymptotic values. At $d=64$ the private shift falls short (1.37 against 1.76). There the 32 private spikes fill half of the $64\times64$ stacked matrix, outside the fixed-rank regime of the asymptotics; quadrupling $d$ brings it to 1.64–1.68. Round 1 observed onsets "about 3× above BBP". That is expected: useful recovery under hard truncation needs more SNR than detection. Asymptotically the specialist reaches 90% residual loss at $n_{\rm eff}=118$, against the detection threshold $\sqrt{qd}=32$; the simulated value is 96.

**Direct test of Cor. 3 (head interference).** Earlier versions supported Cor. 3 with a number from the superseded round-1 threshold experiment. We now test it directly (`exp_headint.py`, `headint_results.json`): $\phi=0$, ample capacity, oracle width, 20 repetitions, five configurations $(d,q,k_p,M)$. Simulated ratio at $n=10^3$ / $10^4$ / $10^5$ against the Cor. 3 value: $(64,16,8,4)$ 1.344 / 1.330 / 1.328 against 1.333; $(64,16,8,2)$ 1.126 / 1.115 / 1.113 against 1.111; $(64,16,4,4)$ 1.530 / 1.490 / 1.487 against 1.474; $(128,16,4,8)$ 1.730 / 1.605 / 1.593 against 1.600; and $(64,8,8,4)$, where $q=k_p$ and Cor. 3 predicts no interference, 1.000 at every $n$. The error is at most 1.2% for $n\ge10^4$ and up to 8.1% at $n=10^3$ (the $M=8$ configuration).

### 5.7 Exp. F: one-hidden-layer ReLU networks

Teacher: $y_m=A_m\,\mathrm{relu}(S^\top x)+B_m\,\mathrm{relu}(P_m^\top x)+\varepsilon$, $q=8$, $k_s=k_p=6$, $\sigma=0.5$. Students: width-$h$ one-hidden-layer ReLU networks; the unified network shares the hidden layer and has per-modality heads. Adam, minibatch 256, 2000 steps, early stopping on validation, noiseless test MSE.

**Round 1** ($d=32$, random unit teacher directions, 3 seeds, one learning rate; `mlp_results.json`) had two problems. (a) For $M=8$ the teacher has 54 directions in $d=32$, so the orthogonality assumption fails. (b) We compared *fixed-width* ratios, as low as 0.49, with the oracle-width bound of Cor. 2. Under the oracle-width convention that Cor. 2 actually addresses, the round-1 ratios at the largest width are 0.84 ($M=2$), 1.13 ($M=4$), 1.68 ($M=8$) at $n=1024$, and 1.03, 1.03, 1.49 at $n=4096$ (`analysis_r2.json`). None is below the bound $1-\phi(1-1/M)$ (0.75, 0.625, 0.56), and most show *less* synergy than the df value. The round-1 claim that the nonlinear synergy was "stronger than the linear $\Gamma$" compared the wrong conventions and the wrong configuration (Exp. A's $\Gamma=0.94$), and we withdraw it.

**Round 2** (`exp_mlp2.py`, `mlp2_summary.json`): $d=64$ for every $M$, so $k_s+Mk_p\le54\le d$ holds; exactly orthonormal teacher directions; 4 seeds; two learning rates ($10^{-3}$, $3\cdot10^{-3}$) selected per model, width and modality on validation loss; $n\in\{1024,4096\}$. Corollary 2's formula for this configuration gives 0.78, 0.69, 0.69 for $M=2,4,8$. This is only a formal analogue, since $q<k$ and a linear map cannot represent the teacher ($q=8\ge k_p=6$, so the bound of Cor. 2 still applies to the formula). The $d\to\infty$ bound is 0.75, 0.625, 0.56.

| $n$ | $M$ | peak width (ratio) | ratio at $h=k_s=6$ | crossover, fixed width* ($k_s+Mk_p$) | min fixed-width ratio [95% CI] (at $h$) | oracle-width ratio at $h=64$ [95% CI] | Cor. 2 formula / $d\to\infty$ bound |
|---|---|---|---|---|---|---|---|
| 1024 | 2 | 8 (1.45) | 1.38 | 16 (18) | 0.57 [0.49, 0.59] (24) | 0.80 [0.69, 0.92] | 0.78 / 0.75 |
| 1024 | 4 | 8 (1.75) | 1.58 | 16 (30) | 0.53 [0.49, 0.54] (32) | 0.89 [0.81, 0.97] | 0.69 / 0.625 |
| 1024 | 8 | 8 (2.05) | 1.74 | 24 (54) | 0.68 [0.65, 0.70] (48) | 1.27 [1.18, 1.39] | 0.69 / 0.56 |
| 4096 | 2 | 12 (3.29) | 1.65 | 24 (18) | 0.59 [0.56, 0.61] (48) | 0.89 [0.84, 0.94] | 0.78 / 0.75 |
| 4096 | 4 | 12 (6.52) | 2.02 | 32 (30) | 0.42 [0.39, 0.45] (32) | 0.80 [0.76, 0.84] | 0.69 / 0.625 |
| 4096 | 8 | 12 (9.52) | 2.05 | 48 (54) | 0.34 [0.32, 0.36] (64) | 1.17 [1.07, 1.27] | 0.69 / 0.56 |

*Smallest width after the peak from which the seed-mean fixed-width ratio stays below 1 at every larger width. Widths tested: 2, 4, 6, 8, 12, 16, 24, 32, 48, 64. The bootstrap CI of this crossover collapses onto the reported grid point in 5 of 6 cells; for $n=1024$, $M=2$ it is [12, 64].

What carries over from the linear model. (a) With enough data ($n=4096$), the competition peak sits at $h=12=k_s+k_p$, and its height grows with $M$ (3.3, 6.5, 9.5). At $n=1024$ it moves to $h=8$, consistent with a data-bound regime. (b) At $n=4096$ the fixed-width crossover grows with $M$ (24, 32, 48), roughly tracking $k_s+Mk_p$ (18, 30, 54) on a coarse grid. (c) Under the oracle-width convention, no cell goes below the $d\to\infty$ synergy bound of Cor. 2. This is a weak test: the bound is a floor, and any model with less synergy than the linear one passes it. We do not count it as support for the quantitative theory (see (f)). (d) The fixed-width ratios fall far lower (to 0.34). This is the Remark 2 mechanism: wide specialists degrade more than the wide unified network. It is real synergy in a fixed-size comparison, but Cor. 2 does not bound it.

What fails. (e) **The tie at and below $h=k_s$ (Prop. 1) fails**: the networks show clear competition, 1.38–2.05 at $h=6$, growing with $n$ and $M$. Round 1 reported only "1.2–1.4 at $h\le4$" and understated this. (f) **The quantitative synergy prediction $\Gamma$ of Cor. 2 fails.** At $h=64$ the oracle-width ratio exceeds the formula value by 3%, 29% and 82% ($n=1024$; $M=2,4,8$) and by 15%, 17% and 68% ($n=4096$); the bootstrap CI contains the formula value in only one of the six cells ($n=1024$, $M=2$; `analysis_r3.json`). For $M=8$ the unified network is still in competition at $h=64>k_s+Mk_p=54$ (1.27 and 1.17, both CIs above 1), where the linear model predicts synergy. ReLU networks apparently need more width than the feature count to represent 54 ReLU features; the theory has no account of this. Learning-rate selection chose $3\cdot10^{-3}$ for 73–93% of unified-network widths, so a wider learning-rate grid might change the details.

**Why narrow networks compete: a control and a partial explanation.** In an earlier version we guessed that the competition in (e) arises because a narrow shared layer must also serve the private features. A control experiment refutes that guess (`exp_mlp_narrow.py`, `mlp_narrow_results.json`; round-2 protocol, $n=4096$, widths 2, 4, 6, 4 seeds). With a teacher that has **no private features** ($k_p=0$), the competition below $k_s$ is still there:

| teacher | $M$ | ReLU ratio at $h=2$ / $4$ / $6$ [95% CI at $h=6$] | linear analogue with the same head distribution, $r=2$ / $4$ / $6$ |
|---|---|---|---|
| $k_p=6$ | 2 | 1.20 / 1.43 / 1.63 [1.53, 1.74] | 1.42 / 2.35 / 5.68 |
| $k_p=6$ | 4 | 1.33 / 1.69 / 2.03 [1.94, 2.15] | 1.59 / 3.07 / 9.01 |
| $k_p=6$ | 8 | 1.35 / 1.71 / 2.05 [1.90, 2.17] | 1.72 / 3.51 / 11.3 |
| $k_p=0$ | 2 | 1.28 / 1.40 / 0.90 [0.55, 1.55] | 1.32 / 2.32 / 0.56 |
| $k_p=0$ | 4 | 1.29 / 1.78 / 0.26 [0.19, 0.35] | 1.57 / 3.15 / 0.34 |
| $k_p=0$ | 8 | 1.31 / 1.72 / 0.18 [0.16, 0.20] | 1.77 / 3.99 / 0.23 |

The cause is the one in Remark 3. The ReLU teacher has Gaussian heads, so each modality weights the shared features differently, whereas Prop. 1 assumes equal strengths. The right-hand column is the linear model with heads drawn from the same distribution as the ReLU teacher's (`exp_hetero.py`, `hetero_results.json`; Monte Carlo at $\sigma=0.5$, $n=4096$, oracle rank, 20 teacher draws; the df heuristic agrees with it to within 0.7%, and with orthonormal heads the population ratio is exactly 1 for $r<k_s$). It has the same sign as the ReLU point estimate in all 18 cells: competition below $k_s$ for both teachers, competition at $h=k_s$ when there are private features, and a switch to pooling synergy at $h=k_s$ when there are none. In one cell ($k_p=0$, $M=2$, $h=6$) the ReLU ratio, 0.90, is not distinguishable from 1, so 17 cells are clear. The magnitudes do not transfer. The linear analogue overstates the competition by a factor of 1.03–1.35 at $h=2$, 1.65–2.3 at $h=4$ and 3.5–5.5 at $h=6$ with private features, and we have no quantitative account of the ReLU values. So finding (e) is a failure of the equal-strength idealisation of Prop. 1 and not of the shared-core picture; but it shows that Prop. 1's window edge at $k_s$ should not be expected in any network whose modalities weight shared features unequally.

### 5.8 Exp. H: small transformers on real image, audio and text tokens

*Data* (`prep_tokens.py`; downloaded public datasets, not redistributed). **Image:** MNIST digits, 2×2-pooled to 14×14, 4 grey levels, raster order. **Audio:** Free Spoken Digit Dataset recordings, 25 ms / 10 ms frames of 24 log band energies, vector-quantised with a 64-code k-means codebook. Held-out audio comes from recordings 45–49 of every speaker and digit. **Text:** Tiny Shakespeare characters. **Fashion:** Fashion-MNIST with the image tokenizer. The four sources occupy disjoint ranges of one 144-token vocabulary, and each window starts with a modality tag. These are real image, audio and text data with simple tokenizers. They are not video or action, and the models are tiny.

*Models and protocol* (`exp_transformer.py`). Pre-LN decoder-only transformers with 2 layers, context 64 and $d_{\rm model}\in\{16,32,64,128\}$ (12k–442k parameters). Each step uses 16 windows per modality, so a unified model and a specialist of the same size see the same per-modality tokens for the same number of steps (the Aghajanyan et al. convention). Training data are the first $D\in\{25{,}000,\,100{,}000\}$ tokens of each modality's stream. Training uses AdamW at lr $2\cdot10^{-3}$, at most 3000 steps, and stops after 600 steps without improvement in any modality. We report test cross-entropy (nats/token) on a held-out half, at each modality's own best-validation checkpoint. Reporting at a modality's own checkpoint removes the confound that modalities overfit at different rates [Wang et al., 2020]; the single joint checkpoint is also reported. Each cell has 5 seeds: seeds 0 and 1 from revision 2 (`transformer_runs.jsonl`) and seeds 2–4 added in this revision with the same code path (`exp_transformer_r3.py`, `transformer_runs_r3.jsonl`; its `check` mode reproduces a revision-2 run bit for bit). That is 140 trained models. For seeds 2–4 we also stored per-window test losses. The held-out test half has 160 windows for image and 98 for audio (about 6k tokens). The interaction is $\Delta_m=L_{{\rm u},m}-L_{{\rm s},m}$ in nats/token. We use differences, not ratios of excess loss, because the irreducible entropy is unknown.

**Results** (`transformer_summary_r3.json`, `figH_transformer_r3.png`; $\Delta$ in nats/token at each modality's own checkpoint; mean ± seed SE over 5 seeds, with the $t$-based 95% CI on 4 degrees of freedom in brackets and the number of seeds with $\Delta>0$. Specialist losses for reference: image 0.451–0.486, audio 1.79–1.96).

| $D$ | $d_{\rm model}$ (params) | $\Delta_{\rm image}$ [95% CI], seeds $>0$ | $\Delta_{\rm audio}$ [95% CI], seeds $>0$ | $\Delta_{\rm image}$, joint ckpt | $\Delta_{\rm audio}$, joint ckpt |
|---|---|---|---|---|---|
| 25k | 16 (12k) | +0.024 ± 0.005 [+0.010, +0.037], 5/5 | −0.019 ± 0.008 [−0.042, +0.004], 0/5 | +0.024 ± 0.005 | −0.016 ± 0.009 |
| 25k | 32 (37k) | +0.018 ± 0.002 [+0.013, +0.023], 5/5 | −0.008 ± 0.006 [−0.024, +0.009], 1/5 | +0.040 ± 0.003 | −0.000 ± 0.008 |
| 25k | 64 (123k) | +0.018 ± 0.003 [+0.010, +0.026], 5/5 | −0.007 ± 0.007 [−0.027, +0.012], 3/5 | +0.072 ± 0.005 | −0.005 ± 0.006 |
| 25k | 128 (442k) | +0.008 ± 0.002 [+0.002, +0.015], 5/5 | +0.001 ± 0.011 [−0.028, +0.030], 2/5 | +0.082 ± 0.004 | +0.001 ± 0.011 |
| 100k | 16 (12k) | +0.018 ± 0.006 [+0.001, +0.035], 5/5 | +0.011 ± 0.004 [+0.000, +0.021], 5/5 | +0.019 ± 0.006 | +0.008 ± 0.005 |
| 100k | 32 (37k) | +0.014 ± 0.003 [+0.006, +0.023], 5/5 | +0.002 ± 0.006 [−0.015, +0.019], 2/5 | +0.016 ± 0.004 | −0.001 ± 0.007 |
| 100k | 64 (123k) | +0.015 ± 0.002 [+0.008, +0.021], 5/5 | −0.004 ± 0.005 [−0.018, +0.010], 2/5 | +0.022 ± 0.004 | −0.000 ± 0.005 |
| 100k | 128 (442k) | −0.001 ± 0.002 [−0.007, +0.005], 1/5 | −0.004 ± 0.006 [−0.019, +0.012], 2/5 | +0.023 ± 0.003 | +0.005 ± 0.003 |

*Statistics.* Revision 2 had 2 seeds per cell and claimed "more than 2 SE in 7 of 8 cells". With 2 seeds that statistic has one degree of freedom and carries no inferential weight, and we withdraw it. We now report three things. (i) *Seed-level sign counts*, the statistic the reviewer recommended: $\Delta_{\rm image}>0$ in 36 of 40 paired unified–specialist comparisons (exact two-sided binomial $p=2\cdot10^{-7}$; in the 2-seed data it was 15 of 16). All four negative comparisons are in one cell, the largest model at $D$=100k. For audio the count is 17 of 40 ($p=0.43$). (ii) *Seed-level CIs*, with the seed as the unit: the mean over the 8 cells is +0.014 [+0.010, +0.019] nats/token for image and −0.003 [−0.008, +0.001] for audio. (iii) *Test-set sampling uncertainty*: a paired bootstrap over test windows with the seeds held fixed (seeds 2–4) gives 95% half-widths of about 0.004 for image and 0.010 for audio. The image interval excludes zero in 7 of 8 cells (the exception is again 100k/128, [−0.006, +0.001]). The audio interval excludes zero in 3 of 8 cells, with both signs (25k/16 and 25k/32 negative, 100k/16 positive). The two sources of uncertainty are of similar size and we have not combined them, and the 16 cell-level tests are not corrected for multiplicity.

*Modality addition (P4), $d_{\rm model}=32$, $D$=25k, 5 seeds.* Per-modality $\Delta$ (mean ± SE):

| $M$ | image | audio | text | fashion | mean over modalities present |
|---|---|---|---|---|---|
| 2 | +0.018 ± 0.002 | −0.008 ± 0.006 | – | – | +0.005 ± 0.003 |
| 3 | +0.014 ± 0.003 | −0.014 ± 0.004 | +0.029 ± 0.014 | – | +0.010 ± 0.006 |
| 4 | +0.018 ± 0.005 | −0.015 ± 0.007 | +0.018 ± 0.012 | +0.035 ± 0.004 | +0.014 ± 0.004 |

*Optimisation control* (`transformer_converge.jsonl`; $d_{\rm model}=32$, $D$=25k, seeds 0–2). The main grid uses one learning rate and a 3000-step cap, and 83 of the 90 models that include the image modality reach their best image checkpoint at step ≥2400, so the grid mixes capacity with optimisation. In one cell we therefore raised the cap to 9000 steps and selected the learning rate from $\{10^{-3},2\cdot10^{-3},4\cdot10^{-3}\}$ per model and modality on validation loss. No run reached the new cap (best image checkpoints at steps 3200–5200). $\Delta_{\rm image}$ is +0.010 ± 0.003 (per seed +0.010, +0.014, +0.005; $t$-CI [−0.002, +0.022]), against +0.017 ± 0.003 for the same three seeds under the main protocol. $\Delta_{\rm audio}$ is −0.007 ± 0.006.

What we observe, and how it relates to the predictions:
1. **Image is in small, consistent competition, and the effect is partly an optimisation effect.** The sign is positive in 36 of 40 comparisons and the size is 0–5% of the loss. In the one cell where we trained to convergence with a tuned learning rate it stays positive in all 3 seeds but shrinks by about 40% (+0.017 to +0.010), and with 3 seeds its CI includes zero. We have this control for one cell only, so we cannot say how much of the competition elsewhere in the grid would survive convergence. Revision 2 called this effect "what Cor. 3 predicts" and wrote that the two modalities have $\phi\approx0$. We withdraw both statements. We did not measure $\phi$. Cor. 3 is about oracle width, ample capacity and per-modality heads, whereas these models have fixed width and one shared output layer. A small loss for one modality when a second is added is what almost any account of shared capacity or shared optimisation predicts, so the observation does not discriminate our model from the alternatives. **Audio shows no consistent interaction** (17 of 40 positive, mean −0.003 [−0.008, +0.001]); if anything it is slightly helped at the smaller budget (6 of 20 positive at $D$=25k). The model treats modalities symmetrically and does not explain this asymmetry.
2. **P1 (rise then fall with size) is not observed.** Image competition only declines with size: the per-seed slope of $\Delta_{\rm image}$ against $\log_2$(parameters) is negative in 5 of 5 seeds at both budgets (−0.0026 [−0.0050, −0.0003] and −0.0033 [−0.0058, −0.0009] nats per doubling), and at $D$=100k the largest model shows none (−0.001 ± 0.002). So the competition is not "persistent at every size", as revision 2 put it. A rising branch would have to lie below our smallest model. Because the size axis is confounded with optimisation (previous paragraph), this is not a clean test of P1 either.
3. **The data trend points the wrong way for P3, but the comparison is confounded.** Under the model, more data per modality makes a fixed-size model less data-bound and should *increase* competition at fixed size. Observed image competition is equal or smaller at $D$=100k than at 25k. Since most image runs are compute-limited, we do not count this as a test of P3.
4. **P4 is not supported, and the test is weak.** P4 says that adding modalities to a fixed-size model eventually raises the loss of *every* modality. From $M=2$ to $M=4$, the seed-paired change in $\Delta$ is +0.000 [−0.015, +0.015] for image and −0.007 [−0.018, +0.004] for audio; from $M=3$ to $M=4$ it is −0.010 [−0.042, +0.022] for text (`budget_changes`). The mean over the modalities present does rise (+0.005 to +0.014; paired change +0.009 [+0.000, +0.018]), but only because the added modalities, text and fashion, are themselves in competition; no existing modality gets worse. The reviewer noted two oddities in the 2-seed data: text competition falling from +0.038 to +0.016 when fashion was added, and image competition falling when Fashion-MNIST was added. With 5 seeds the first is +0.029 to +0.018, well inside its CI, and the second has disappeared (+0.018 at both $M=2$ and $M=4$). These intervals could hide effects of ±0.015 nats, as large as the image effect itself. The sweep has one size and one budget, and with $D$=25k these models overfit audio and text within 400–1900 steps, which suggests a data-bound regime where the model does not predict capacity competition. So this is an underpowered null, not evidence against a modality budget.
5. **The largest effect is one the model does not contain.** At the single joint checkpoint that a deployed unified model would use, image competition grows with size at $D$=25k (+0.024 → +0.082, positive in 40 of 40 comparisons over both budgets). The cause is that for $d_{\rm model}\ge32$ audio overfits after 400–1200 steps while image keeps improving almost to the step limit, so the jointly selected checkpoint is early for image. This is the overfitting-rate mismatch of Wang et al. [2020]. At small data it outweighs the capacity effects that our model describes.

## 6 Discussion: what the model says about the limits of unified models

All statements in this section are statements about the model, supported to the degree shown in Section 5.

1. **Capacity limit.** In the capacity-bound regime a unified core hosts about $(r-k_s)/k_p$ modalities (Cor. 1; exact in Exp. B). Under power-law spectra this becomes a constant capacity tax $\kappa$ that does not shrink with scale (Prop. 3). If real models behaved like this, scaling alone would not remove competition; it would set its price.
2. **Statistical limit.** With each model's size tuned (oracle or validation-selected width, which agree; Sec. 5.4), the model's only source of synergy is pooled estimation of shared directions, bounded by Cor. 2 and reduced by head interference (Cor. 3). The ReLU networks show less synergy than the model's $\Gamma$, and for $M=8$ none at all, so $\Gamma$ is not a usable quantitative prediction outside the linear model. Fixed-size comparisons are different: there, a unified model can also win because it wastes less of its width on noise directions (Remark 2), and the ReLU networks, the fixed-rank simulation and the fixed-rank df model all show large fixed-width synergy. Published comparisons, including Aghajanyan et al., are fixed-size. For them the headline "bounded synergy" does not apply, and the fixed-width mechanism can contribute to reported synergy. We have not shown that it does.
3. **Alignment limit.** The tolerated misalignment shrinks as $n^{-1/2}$ (Cor. 4, Exp. C).

**Architectural implications (model-based, untested at scale).** Modality-specific parameters remove the private-capacity competition and the head-interference term. The model therefore predicts that such capacity helps most when $\phi$ is small and $M$ large. Shukor et al. [2025] find that MoEs with *learned* routing outperform MoEs with *hand-assigned modality* routing. The model is silent on that comparison, because it has no within-modality heterogeneity, and a learned router can in principle recover the shared/private partition and more. We therefore restate P6 in terms of modality-specific *capacity*, however it is allocated, and do not claim that hand-assigned routing is preferable. Two recent empirical papers bear on this more directly than our model does. Han et al. [2026] report that shared attention and normalisation with modality-specific feed-forward layers promote synergy, and Uni-X [Hao et al., 2026] reports that separating the shallow and deep layers by modality reduces gradient conflict. Both agree in direction with P6, but they were obtained without our model, and we have not checked whether their gains vary with $M$ or with shared structure as P6 requires.

**Falsifiable predictions.**
- **P1:** At fixed per-modality data, the unified–specialist interaction against model size is non-monotone: it rises to a peak near the specialist's saturation size and then falls.
- **P2:** In the data-abundant regime, the excess-loss ratio is approximately constant across model sizes (Prop. 3).
- **P3:** The size at which a unified model crosses the barrier scales as $D^{1/\alpha}$.
- **P4:** Adding modalities to a fixed-size dense model eventually increases per-modality loss for every modality, and the onset moves out linearly with capacity.
- **P5:** Rotating part of one modality's embedding space eliminates synergy at an angle that shrinks with more data.
- **P6:** Modality-specific capacity (hand-assigned or learned) helps more as $M$ grows and as $\phi$ falls.

**Status of the predictions after this paper.**

| prediction | linear Monte Carlo | ReLU networks | small transformers (real data) |
|---|---|---|---|
| Window and peak at $k_s+k_p$ (P1, capacity-bound) | yes (Exp. A; random designs) | yes at $n=4096$ | **not observed**: monotone decline over 12k–442k params (5 seeds); size confounded with optimisation |
| Scale-free ratio $R$ (P2) | yes, given many features (E1); not with 48 (E2) | not tested | not tested (no irreducible-loss estimate) |
| $r_\times\propto n^{1/\alpha}$ (P3) | slope 0.57 vs 0.5 (pre-asymptotic) | crossover grows with $n$ (coarse grid) | not testable (compute-limited runs) |
| Modality budget (P4) | exact (Exp. B) | crossover grows with $M$ | **not observed** at $d_{\rm model}=32$, $D$=25k (5 seeds; underpowered) |
| Alignment tolerance $\propto n^{-1/2}$ (P5) | yes for $n\ge3000$ | not tested | not tested |
| Oracle-width synergy level $\Gamma$ (Cor. 2) | yes ($n\ge1000$); same with validation-selected rank | **quantitative prediction fails** (ratio 3–82% above $\Gamma$; competition at $M=8$); only the weak lower bound holds | – |
| Fixed-width synergy (Remark 2) | yes at $r=20$; df 10–12% too low at $r=24$ | qualitatively (ratios down to 0.34) | not tested |
| Head interference with $\phi=0$ (Cor. 3) | yes (direct test, Sec. 5.6: within 1.2% for $n\ge10^4$) | – | not tested: $\phi$ was not measured, and the observed image-side competition does not discriminate Cor. 3 from other capacity-sharing accounts |
| Zero interaction below $k_s$ (Prop. 1) | population only, and only for equal feature strengths (Remark 3); df predicts finite-$n$ synergy (confirmed) | **fails** (competition 1.4–2.0 at $h=k_s$); sign explained by unequal feature strengths (Remark 3; 17 of 18 cells clear), magnitude not | – |

## 7 Pre-registered test protocol for real unified models

*Not executed here (no GPU).* (1) For $M\in\{2,3,4\}$ modalities (text, image tokens, audio tokens, action tokens), train dense decoder-only models at 5 sizes (20M–1B) on a fixed per-modality token budget $D$, at 3 values of $D$, plus unimodal models on the same grid. (2) Estimate $\alpha$ from the eigenspectrum of unimodal hidden representations, and $\phi$ as the fraction of spectral energy captured by CCA between paired modalities (image–caption, audio–transcript) above a fixed threshold. Estimate the irreducible loss per modality from the unimodal size sweep, so that excess-loss ratios can be formed. (3) Compute $R$, $\kappa$ and the crossover exponent $1/\alpha$. (4) **Pass criteria:** P2, the capacity-bound excess-loss ratio is within a factor 1.5 of $R$ and the 95% CI of its log–log slope against size contains 0. P3, the 95% CI of the fitted crossover exponent overlaps $[0.5/\alpha,1.5/\alpha]$. P4, at the smallest size, per-modality loss increases monotonically once $M$ exceeds a fitted budget. P5, the critical rotation angle decreases with $D$. (5) Report both fixed-size and best-size-up-to-$N$ comparisons (Remark 2). (6) Repeat with a modality-expert variant and a learned-routing MoE to test P6.

## 8 Limitations

- **Linear, single-bottleneck model.** Real unified models are deep and nonlinear and can use superposition [Elhage et al., 2022], which could soften the window edges of Prop. 1. The quantitative constants already fail in one-hidden-layer ReLU networks: the synergy level $\Gamma$ is missed by up to 82%, and the lower window edge at $k_s$ does not exist there.
- **Idealised structure.** Equal-strength orthogonal features, isotropic inputs and a single $\phi$. Proposition 3's flatness needs many features; with 48 per class (E2) the population ratio is far from flat.
- **The finite-data theory is a heuristic.** It fails at low SNR (Exp. A at $n=200$, Exp. C at $n\le1000$, random designs at $n=200$). Under power-law spectra it overstates synergy by 5–11% at intermediate $n$. Oracle width selection is optimistic for both models. Below $n=d$ the orthogonal-design model is only an SNR model; random designs were tested only for $n>d+1$, and with ALS for the unified model.
- **Threshold shifts.** Asymptotic predictions are reached only when the number of spikes is small relative to the matrix dimensions ($d=256$, not $d=64$).
- **Nonlinear evidence is small.** The ReLU study uses one teacher family and one optimiser, with 4 seeds. The transformer study has 2-layer models up to 442k parameters, two data budgets, 5 seeds, simple tokenizers and one learning rate with at most 3000 steps, so that 83 of 90 image-containing models were still improving near the step limit. A converged, learning-rate-tuned control exists for one cell only, and there the image effect shrank by about 40%. The P4 sweep has one size and one budget. There is no measurement of $\alpha$, $\phi$ or irreducible loss, so P2, P3 and P5 are untested on real data, $R$ and $\kappa$ were not compared with any measured quantity, and the link between the image-side competition and Cor. 3 is not established. At this scale P1 and P4 were not observed.
- **The research question is not answered for real systems.** Video and action, which motivate the question, are absent. Nothing here tests a model above 442k parameters. What the paper establishes is confined to the linear model; for evidence at scale see Han et al. [2026].
- **Cell-level tests are uncorrected.** The transformer table has 16 cell-level intervals without multiplicity correction; seed and test-window uncertainty are reported separately, not combined.
- **No optimisation dynamics.** The mechanisms of Huang et al. [2022] and Wang et al. [2020] are absent from the theory. Our per-modality early stopping removes only the second.
- **Comparison convention.** Same-size specialists match neither total parameters nor compute. Under power-law spectra, $\kappa$ and $M/\kappa$ (Sec. 5.5) convert the same-size comparison into a same-total-capacity one, inside the model only; we give no conversion to equal compute. Most statements are for size-tuned (oracle or validation-selected) width, whereas published comparisons are fixed-size; Remark 2 and Sec. 5.4 show that the two conventions give different answers (0.94–0.96 against 0.53–0.68 in one configuration).

## 9 Conclusion

A linear shared-core model with shared and private features produces competition, synergy, a competition barrier, a modality budget, and a scale-free capacity tax under power-law spectra. Its finite-data predictions hold inside the model where SNR is high, with random designs after an $n-d-1$ correction, and, for the pooling threshold shifts, up to finite-size effects that shrink with dimension. One-hidden-layer ReLU networks reproduce the window, its peak and the growth of the crossover with $M$, but not the model's constants: the synergy level $\Gamma$ is missed by up to 82%, and the tie below $k_s$ does not occur. A control experiment and Remark 3 trace the latter to unequal feature strengths across modalities, which predicts its sign but not its size. Their large fixed-width synergy comes from a different mechanism (over-sized specialists), which the fixed-rank simulation reproduces; validation-selected rank behaves like oracle rank. Small transformers trained on real image, audio and text tokens show a small image-side competition (positive in 36 of 40 seed-level comparisons) that declines with model size, shrinks when training is run to convergence in the one cell we checked, and does not discriminate our model from other capacity-sharing accounts. They show no consistent audio-side effect, no competition window, and no modality-budget effect at the scales we could afford. Overfitting-rate mismatch, which the model omits, dominates single-checkpoint comparisons. The paper therefore contributes a solvable instance with explicit constants and a record of where those constants stop transferring. It does not settle where real unified models stop paying; Section 7 says how to test the remaining predictions, and Han et al. [2026] is the place to look for evidence at scale.

## References

- Aghajanyan, A., Yu, L., Conneau, A., Hsu, W.-N., Hambardzumyan, K., Zhang, S., Roller, S., Goyal, N., Levy, O., and Zettlemoyer, L. (2023). Scaling Laws for Generative Mixed-Modal Language Models. *ICML 2023*, PMLR 202:265–279. arXiv:2301.03728.
- Bahri, Y., Dyer, E., Kaplan, J., Lee, J., and Sharma, U. (2024). Explaining neural scaling laws. *Proceedings of the National Academy of Sciences*, 121(27):e2311878121. arXiv:2102.06701.
- Benaych-Georges, F. and Nadakuditi, R. R. (2012). The singular values and vectors of low rank perturbations of large rectangular random matrices. *Journal of Multivariate Analysis*, 111:120–135. arXiv:1103.2221.
- Chameleon Team (2024). Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv:2405.09818.
- Chang, T. A., Arnett, C., Tu, Z., and Bergen, B. K. (2024). When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages. *EMNLP 2024*. arXiv:2311.09205.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., and Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*, pp. 8440–8451. arXiv:1911.02116.
- Deng, C., Zhu, D., Li, K., et al. (2025). Emerging Properties in Unified Multimodal Pretraining (BAGEL). arXiv:2505.14683.
- Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., and Lei, Q. (2021). Few-Shot Learning via Learning the Representation, Provably. *ICLR 2021*. arXiv:2002.09434.
- Eckart, C. and Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3):211–218.
- Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. arXiv:2209.10652.
- Emu3 Team (BAAI) (2024). Emu3: Next-Token Prediction is All You Need. arXiv:2409.18869.
- Fernandes, P., Ghorbani, B., Garcia, X., Freitag, M., and Firat, O. (2023). Scaling Laws for Multilingual Neural Machine Translation. *ICML 2023*, PMLR 202:10053–10071. arXiv:2302.09650.
- Gavish, M. and Donoho, D. L. (2014). The Optimal Hard Threshold for Singular Values is $4/\sqrt3$. *IEEE Transactions on Information Theory*, 60(8):5040–5053. arXiv:1305.5870.
- Han, J., Tong, S., Fan, D., Chen, M., Torr, P., Kokkinos, F., and Lewis, M. (2026). Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes. arXiv:2608.05000.
- Hao, J., Liu, H., Xiao, X., Huang, Q., and Yu, J. (2026). Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models. *ICLR 2026*. arXiv:2509.24365.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.
- Huang, Y., Lin, J., Zhou, C., Yang, H., and Huang, L. (2022). Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably). *ICML 2022*, PMLR 162:9226–9259. arXiv:2203.12221.
- Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). Position: The Platonic Representation Hypothesis. *ICML 2024*, PMLR 235:20617–20642. arXiv:2405.07987.
- Ilbert, R., Tiomoko, M., Louart, C., Odonnat, A., Feofanov, V., Palpanas, T., and Redko, I. (2024). Analysing Multi-Task Regression via Random Matrix Theory with Application to Time Series Forecasting. *NeurIPS 2024*. arXiv:2406.10327.
- Ildiz, M. E., Zhao, Z., and Oymak, S. (2024). Understanding Inverse Scaling and Emergence in Multitask Representation Learning. *AISTATS 2024*, PMLR 238:4726–4734.
- Izenman, A. J. (1975). Reduced-rank regression for the multivariate linear model. *Journal of Multivariate Analysis*, 5(2):248–264.
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
- Khan, A. (2026). Capacity and Redundancy Trade-offs in Multi-Task Learning. *UAI 2026*. arXiv:2607.16554.
- Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., and Zou, J. (2022). Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. *NeurIPS 2022*. arXiv:2203.02053.
- Liang, W., Yu, L., Luo, L., et al. (2025). Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models. *Transactions on Machine Learning Research*. arXiv:2411.04996.
- Lin, X. V., Shrivastava, A., Luo, L., et al. (2024). MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts. arXiv:2407.21770.
- Longpre, S., Kudugunta, S., Muennighoff, N., Ebrahimi, S., et al. (2025). ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality. arXiv:2510.22037 (ICLR 2026).
- Maloney, A., Roberts, D. A., and Sully, J. (2022). A Solvable Model of Neural Scaling Laws. arXiv:2210.16859.
- Paquette, E., Paquette, C., Xiao, L., and Pennington, J. (2024). 4+3 Phases of Compute-Optimal Neural Scaling Laws. *NeurIPS 2024*. arXiv:2405.15074.
- Reed, S., Zolna, K., Parisotto, E., et al. (2022). A Generalist Agent. *Transactions on Machine Learning Research*. arXiv:2205.06175.
- Shukor, M., Fini, E., Turrisi da Costa, V. G., Cord, M., Susskind, J., and El-Nouby, A. (2025). Scaling Laws for Native Multimodal Models. *ICCV 2025*. arXiv:2504.07951.
- Tripuraneni, N., Jin, C., and Jordan, M. I. (2021). Provable Meta-Learning of Linear Representations. *ICML 2021*, PMLR 139. arXiv:2002.11684.
- Wang, W., Tran, D., and Feiszli, M. (2020). What Makes Training Multi-Modal Classification Networks Hard? *CVPR 2020*, pp. 12695–12705. arXiv:1905.12681.
- Wu, S., Zhang, H. R., and Ré, C. (2020). Understanding and Improving Information Transfer in Multi-Task Learning. *ICLR 2020*. arXiv:2005.00944.

**Datasets used (downloaded at run time, not redistributed):** MNIST (LeCun et al.), Fashion-MNIST (Xiao, Rasul and Vollgraf, 2017), Free Spoken Digit Dataset (Jackson et al., GitHub `Jakobovski/free-spoken-digit-dataset`), Tiny Shakespeare (Karpathy, `char-rnn`).

---

### Appendix: reproducibility and provenance of numbers

| Paper item | Script | Output file |
|---|---|---|
| Exp. A–D, error summary | `code/exp_linear.py`, `code/analyze_r2.py` | `results/linear_results.json`, `results/analysis_r2.json` |
| Alignment tolerance | `code/exp_gap.py` | `results/gap_results.json` |
| Random designs (5.4) | `code/exp_random_design.py` | `results/random_design_results.json` |
| Exp. E1–E2 | `code/exp_powerlaw.py` | `results/powerlaw_results.json` |
| Exp. E3 | `code/exp_crossover.py` | `results/crossover_results.json` |
| Exp. G (fine grid) | `code/exp_threshold_fine.py` (`TF_SCALE=4` for $d=256$) | `results/threshold_fine_results.json`, `results/threshold_fine_results_d256.json` |
| Exp. G round 1 (superseded) | `code/exp_threshold.py` | `results/threshold_results.json` |
| Exp. F round 1, convention re-analysis | `code/exp_mlp.py`, `code/analyze_mlp.py`, `code/analyze_r2.py` | `results/mlp_results.json`, `results/mlp_summary.json`, `results/analysis_r2.json` |
| Exp. F round 2 | `code/exp_mlp2.py`, `code/analyze_mlp2.py` | `results/mlp2_results_n*.json`, `results/mlp2_summary.json` |
| Exp. H (transformers), seeds 0–1 | `code/prep_tokens.py`, `code/exp_transformer.py`, `code/analyze_transformer.py` | `results/transformer_runs.jsonl`, `results/transformer_summary.json` (2-seed summary, superseded), `results/figH_transformer.png` |
| Exp. H, seeds 2–4, optimisation control, 5-seed analysis | `code/exp_transformer_r3.py`, `code/analyze_transformer_r3.py` | `results/transformer_runs_r3.jsonl`, `results/transformer_converge.jsonl`, `results/transformer_summary_r3.json`, `results/figH_transformer_r3.png` |
| Width conventions (5.4) | `code/exp_valrank.py` | `results/valrank_results.json` |
| Direct test of Cor. 3 (5.6) | `code/exp_headint.py` | `results/headint_results.json` |
| Narrow ReLU control and linear analogue (5.7) | `code/exp_mlp_narrow.py`, `code/exp_hetero.py` | `results/mlp_narrow_results.json`, `results/hetero_results.json` |
| E3 slope fragility, ReLU vs $\Gamma$ | `code/analyze_r3.py` | `results/analysis_r3.json` |

Statements about other papers come from the cited sources and were not reproduced here.
