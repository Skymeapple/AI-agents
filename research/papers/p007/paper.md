# Where Unification Stops Paying: A Capacity–Data Theory of Competition and Synergy in Unified Multimodal Models

**Target venue:** NeurIPS 2027 (main track; theory + controlled experiments)

**Code and data:** `code/` (run instructions in `code/README.md`); every number below is read from a JSON file in `results/` produced by those scripts, unless it is attributed to a cited paper.

---

## Abstract

Unified "any-to-any" models, which put vision, audio, video, action and text tokens through one shared network, are often described as limited only by scale. Empirical mixed-modal scaling studies report something more specific. Two modalities trained together can do worse than two separately trained models of the same size ("competition") or better ("synergy"), and the sign changes with model size. We give a minimal analytically solvable account of this. Each modality is a regression task that reads a shared core of bounded capacity $r$ through its own head, and its target depends on $k_s$ features shared across modalities and $k_p$ private features. We prove five results. (i) The population interaction is zero below $k_s$, positive on a *competition window* $k_s<r<k_s+Mk_p$, and peaks at $r=k_s+k_p$ with per-modality excess loss $(M-1)k_p s^2/M$. A capacity-$r$ core can therefore absorb at most $M^*=\lfloor (r-k_s)/k_p\rfloor$ modalities before it loses to specialists. (ii) With finite data, synergy is a purely statistical effect: pooled estimation of shared directions. It is bounded above by an explicit degrees-of-freedom ratio, and with no shared structure a fully provisioned unified model pays a "head-interference" penalty. (iii) When the shared directions of different modalities are misaligned by an angle $\theta$, synergy survives only for $\sin^2\theta \lesssim C/n$. The alignment needed for synergy therefore becomes *stricter* as data grows. (iv) Under power-law feature spectra, the population loss ratio between a unified model and specialists does not depend on scale: $R=(\phi M^{1/\alpha}+(1-\phi)M)^{\alpha}/M$. This is a constant "capacity tax" $\kappa=R^{1/(\alpha-1)}$, so in this model competition does not shrink with scale while data are abundant, and synergy appears only once the model is data-bound, at a crossover capacity that scales as $n^{1/\alpha}$. (v) Pooling lowers the spectral detection threshold for shared features and raises it for private ones. Monte Carlo simulations match the theory's loss ratios within 2.3% (median 1–2%) for $n\ge 1000$; its sign predictions match on all 140 cells of a misalignment–capacity grid; the closed-form tax matches within 0.7% for $\alpha\in\{2,3\}$. One-hidden-layer ReLU networks show the predicted competition window, a peak at width $k_s+k_p$, and a crossover width that grows with $M$ and $n$. We end with a pre-registered protocol for testing these predictions on real unified models.

## 中文摘要

统一的"任意模态到任意模态"模型把视觉、音频、视频、动作和文本token送入同一个共享网络，常被认为其上限只取决于规模。然而，混合模态缩放律研究报告了更具体的现象：两种模态联合训练时，可能比同等规模的两个单独训练的模型更差（"竞争"），也可能更好（"协同"），且符号随模型规模而变化。我们为此给出一个可解析求解的最简模型：每个模态是一个回归任务，通过各自的输出头读取一个容量为 $r$ 的共享核心；其目标依赖 $k_s$ 个跨模态共享特征和 $k_p$ 个模态私有特征。我们证明了五个结果。(i) 总体交互项在 $r\le k_s$ 时为零，在"竞争窗口" $k_s<r<k_s+Mk_p$ 内为正，并在 $r=k_s+k_p$ 处达到峰值 $(M-1)k_p s^2/M$；因此，容量为 $r$ 的核心至多能容纳 $M^*=\lfloor (r-k_s)/k_p\rfloor$ 个模态，超过后便不如专用模型。(ii) 在有限数据下，协同纯粹是统计效应（共享方向的合并估计），其上界由一个显式的自由度比值给出；若不存在共享结构，即使容量充足，统一模型也要付出"输出头干扰"代价。(iii) 若各模态的共享方向相互偏离角度 $\theta$，协同只在 $\sin^2\theta\lesssim C/n$ 时成立，即数据越多，协同所需的对齐越严格。(iv) 在幂律特征谱下，统一模型与专用模型的总体损失比与规模无关：$R=(\phi M^{1/\alpha}+(1-\phi)M)^{\alpha}/M$，相当于恒定的"容量税" $\kappa=R^{1/(\alpha-1)}$。因此在该模型中，数据充足时竞争不会随规模消失；只有当模型受数据限制时才会出现协同，且转折容量按 $n^{1/\alpha}$ 增长。(v) 合并训练会降低共享特征的谱检测阈值，并提高私有特征的阈值。蒙特卡洛模拟在 $n\ge1000$ 时与理论损失比的误差不超过2.3%（中位数1–2%）；在140格的错位–容量相图上符号预测全部一致；容量税闭式解在 $\alpha\in\{2,3\}$ 时误差不超过0.7%。单隐层ReLU网络同样呈现预测的竞争窗口、在宽度 $k_s+k_p$ 处的峰值，以及随 $M$ 和 $n$ 增大的转折宽度。最后，我们给出在真实统一模型上检验这些预测的预注册方案。

---

## 1 Introduction

One line of work in multimodal AI aims at a single network that reads and writes every modality: images, audio, video, actions and text. Gato serialised 604 tasks across modalities and embodiments into one token stream for a single transformer [Reed et al., 2022]. Chameleon trained an early-fusion token model on interleaved image–text data [Chameleon Team, 2024]. Emu3 argued that next-token prediction alone suffices for multimodal generation and perception [Emu3 Team, 2024]. BAGEL reported capabilities that emerge with scale in a unified pre-trained model [Deng et al., 2025]. The obvious question is where this approach runs out. Is one model with $N$ parameters as good as $M$ specialists of size $N$, as good as one specialist of size $N$, or worse than both, and what decides which?

The best empirical evidence comes from mixed-modal scaling laws. Aghajanyan et al. [2023] trained over 250 models on seven modalities and added an interaction term to unimodal scaling laws. They call two modalities *competing* when joint training does worse than training separate models on the same per-modality tokens, *synergistic* in the opposite case, and they call the boundary the *competition barrier*. Their 30B speech–text model crossed it. Shukor et al. [2025] trained 457 native multimodal models and found that early and late fusion scale similarly, and that mixture-of-experts models, which learn modality-specific weights, perform significantly better. Architectures that separate parameters by modality, such as MoMa [Lin et al., 2024] and Mixture-of-Transformers [Liang et al., 2025], report large compute savings over dense baselines at matched quality. Multilingual language modelling shows a close analogue, the *curse of multilinguality*: adding languages helps up to a point and then hurts, and more capacity relieves it [Conneau et al., 2020; Chang et al., 2024].

These observations lack a mechanistic theory that says *when* competition turns into synergy, *how many* modalities a given capacity can hold, and *what* happens under scaling. Existing theory covers neighbouring questions. Huang et al. [2022] prove that gradient-descent dynamics in late-fusion networks can learn only a subset of modalities, an optimisation phenomenon. Multi-task representation-learning theory [Tripuraneni et al., 2021; Du et al., 2021] proves that a shared representation helps, but it fixes the representation dimension at the true one, so capacity competition never arises. Wu et al. [2020] show that task misalignment causes negative transfer in shared-module models, but they do not study how the effect scales with capacity, data or the number of tasks.

**Sub-question.** We study one question that a model can settle. *In a network whose modalities share a core of bounded capacity and keep their own heads, what is the sign and size of the competition/synergy interaction as a function of capacity $r$, data per modality $n$, number of modalities $M$, the fraction of structure modalities share, and their misalignment? How does that interaction behave under power-law scaling?*

**Contributions.**

1. **An exactly solvable shared-core model** (Section 3). It makes "shared" and "private" structure precise, and we compute the population interaction exactly (Proposition 1). The interaction vanishes below $k_s$, is positive on a competition window $(k_s, k_s+Mk_p)$, peaks at $r=k_s+k_p$, and gives a *modality budget* $M^*=\lfloor (r-k_s)/k_p\rfloor$ (Corollary 1). Misalignment turns shared structure into a "gap" spectrum that we derive in closed form (Lemma 2).
2. **Finite-data theory** (Proposition 2). Under orthogonal design, the unified least-squares estimator is *exactly* a truncated SVD of the stacked per-modality estimates. A degrees-of-freedom (df) model then predicts that synergy is a bounded statistical effect (Corollary 2). With no shared structure, a unified model at full capacity still pays a head-interference penalty (Corollary 3). The alignment tolerance obeys $\sin^2\theta_c\propto 1/n$ (Corollary 4).
3. **A scale-invariant capacity tax** under power-law spectra (Proposition 3): $R=(\phi M^{1/\alpha}+(1-\phi)M)^\alpha/M$. Data-bound synergy sets in at a crossover capacity $r_\times\propto n^{1/\alpha}$.
4. **Threshold shifts from pooling** (Remark 4). Pooling shifts the spectral (BBP-type) detection thresholds: down by $\sqrt M$ for shared features and up for private ones.
5. **Validation.** We check the theory against Monte Carlo simulation (7 experiment families) and against one-hidden-layer ReLU networks, and we report where the theory fails (low SNR; widths below $k_s$ in the nonlinear case).
6. **A pre-registered protocol** (Section 7) for testing predictions P1–P6 on real unified models.

We do not claim that real unified transformers are linear shared-core models. We claim that the model isolates a mechanism, capacity competition offset by pooled estimation, that is sufficient to produce the qualitative phenomena reported empirically, and that it makes quantitative, falsifiable predictions beyond them.

## 2 Related Work

**Unified multimodal models.** Gato [Reed et al., 2022], Chameleon [Chameleon Team, 2024], Emu3 [Emu3 Team, 2024] and BAGEL [Deng et al., 2025] are early-fusion or unified-decoder models that share most parameters across modalities. BAGEL already uses two transformer experts, one for understanding and one for generation. MoMa [Lin et al., 2024] and Mixture-of-Transformers [Liang et al., 2025] separate non-embedding parameters by modality while keeping global attention. Liang et al. report matching a dense Chameleon-7B baseline with 55.8% of the FLOPs. In our model, such designs remove the head-interference and private-capacity terms (Section 6).

**Scaling laws.** Kaplan et al. [2020] and Hoffmann et al. [2022] established power-law dependence of loss on parameters and data. Aghajanyan et al. [2023] extended this to mixed-modal training with an additive interaction term and the competition barrier. Shukor et al. [2025] studied native multimodal scaling across fusion strategies. Our Proposition 3 gives a generative account of a scale-invariant interaction ratio when data are abundant, and of a data-driven crossover.

**Theory of multimodal and multi-task learning.** Huang et al. [2022] prove modality competition in the optimisation dynamics of late-fusion networks. Wang et al. [2020] show empirically that multimodal networks can underperform the best unimodal network because modalities overfit at different rates. Wu et al. [2020] analyse shared-module multi-task models and identify misalignment as a cause of negative transfer. Tripuraneni et al. [2021] and Du et al. [2021] bound the benefit of shared low-dimensional linear representations. We add *bounded capacity* and *growing $M$* to this linear setting, and we work out exactly what follows from them.

**Shared structure across modalities.** The Platonic Representation Hypothesis [Huh et al., 2024] argues that representations from different modalities converge, which would make $\phi$ (the shared fraction) large. The modality gap [Liang et al., 2022] shows that embeddings of different modalities can stay systematically apart, which corresponds to $\theta>0$ in our model. Corollary 4 shows that synergy is very sensitive to this angle.

**Curse of multilinguality.** Conneau et al. [2020] and Chang et al. [2024] document that adding languages hurts once capacity is limited, and that the effect depends on similarity between languages. This is the language-only analogue of our competition window and modality budget.

**Tools.** Population results use the Eckart–Young / Ky Fan variational characterisation [Eckart and Young, 1936]. The finite-sample df model is the standard tangent-space parameter count of reduced-rank regression [Izenman, 1975]. Remark 4 uses the rectangular spiked-matrix phase transition of Benaych-Georges and Nadakuditi [2012]. Superposition [Elhage et al., 2022] is a mechanism that the linear model leaves out by construction (Section 8).

## 3 Model

**Data.** There are $M$ modalities. Each modality's tokenizer maps its inputs into a common $d$-dimensional space, so $x\sim\mathcal N(0,I_d)$. Modality $m$ has targets $y=W_mx+\varepsilon$, with $W_m\in\mathbb R^{q\times d}$ and $\varepsilon\sim\mathcal N(0,\sigma^2 I_q)$, and $n$ i.i.d. samples per modality. The targets stand for whatever the model must predict for that modality: next tokens, actions, reconstructions.

**Shared and private structure.** Let $S\in\mathbb R^{d\times k_s}$ and $P_1,\dots,P_M\in\mathbb R^{d\times k_p}$ have orthonormal columns that are mutually orthogonal. Write
$$W_m = s\,[A_m\;B_m]\,[S_m\;P_m]^\top,\qquad S_m=\cos\theta\,S+\sin\theta\,E_m ,$$
where $[A_m\,B_m]\in\mathbb R^{q\times(k_s+k_p)}$ has orthonormal columns and $E_m$ (orthonormal, orthogonal to everything else) is the modality-specific "gap" direction. When $\theta=0$, the $k_s$ shared features are literally the same input directions in every modality. Let $k=k_s+k_p$ be the number of features one modality needs, and $\phi=k_s/k$ the shared fraction.

**Models.** Both models are linear networks with a bottleneck of width at most $r$ (the "core capacity").

- *Unified:* $\hat y = V_m U^\top x$ with **one** $U\in\mathbb R^{d\times\rho}$, $\rho\le r$, shared by all modalities, and modality heads $V_m\in\mathbb R^{q\times\rho}$.
- *Specialist:* each modality has its own $U_m$ of width $\le r$.

This is the comparison of Aghajanyan et al. [2023]: one joint model of size $r$ against separate models of the same size trained on the same per-modality data. The per-modality excess risk under $x\sim\mathcal N(0,I)$ is $L=\frac1M\sum_m\|\hat W_m-W_m\|_F^2$. The **interaction** is $\Delta=L_{\rm u}-L_{\rm s}$, and the **ratio** $L_{\rm u}/L_{\rm s}$ is $>1$ for competition and $<1$ for synergy (the barrier is ratio 1). For finite $n$, both models choose the oracle width $\rho\le r$ that minimises their risk, so capacity is an upper bound that is never wasted on fitting noise. Section 8 discusses this choice.

## 4 Theory

### 4.1 Population (infinite data)

Write $G_m=W_m^\top W_m$ and $G=\sum_m G_m$.

**Lemma 1 (variational form).** For orthonormal $U\in\mathbb R^{d\times\rho}$, $\min_{V_m}\|W_m-V_mU^\top\|_F^2=\|W_m\|_F^2-\mathrm{tr}(U^\top G_mU)$. Hence the unified population risk is $\frac1M\big(\mathrm{tr}\,G-\max_U \mathrm{tr}(U^\top GU)\big)=\frac1M\sum_{i>\rho}\lambda_i(G)$, and the specialist risk is $\frac1M\sum_m\sum_{i>\rho}\lambda_i(G_m)$.

*Proof.* Given $U$, the optimal head is $V_m=W_mU$, and $\|W_m-W_mUU^\top\|^2=\|W_m\|^2-\mathrm{tr}(U^\top G_mU)$ because $UU^\top$ is an orthogonal projector. Summing over $m$ and applying Ky Fan's maximum principle (equivalently Eckart–Young [1936]) gives the result. $\square$

**Proposition 1 (exact population interaction, $\theta=0$).** With $s^2$ the per-feature signal energy,
$$\Delta_{\rm pop}(r)=\begin{cases}0,& r\le k_s\\ s^2 (r-k_s)(1-\tfrac1M),& k_s\le r\le k\\ s^2\,(k_s+Mk_p-r)/M,& k\le r\le k_s+Mk_p\\ 0,& r\ge k_s+Mk_p.\end{cases}$$
$\Delta_{\rm pop}\ge 0$ everywhere, with maximum $s^2(M-1)k_p/M$ at $r=k$. Since $L_{\rm s}=0$ there, the population ratio is infinite at $r=k$.

*Proof.* Because $[A_m\,B_m]$ has orthonormal columns, $G_m=s^2(SS^\top+P_mP_m^\top)$, so $G=s^2(M\,SS^\top+\sum_mP_mP_m^\top)$. Its eigenvalues are $Ms^2$ with multiplicity $k_s$, then $s^2$ with multiplicity $Mk_p$. Each $G_m$ has $k$ eigenvalues equal to $s^2$. Lemma 1 gives $L_{\rm u}=\frac1M[Ms^2(k_s-r)_++s^2\min(Mk_p,\,Mk_p+k_s-r)]$ for $r\le k_s+Mk_p$ and $L_{\rm s}=s^2(k-r)_+$. Subtracting case by case gives the result. $\square$

The **competition window** is $(k_s,\,k_s+Mk_p)$. Below $k_s$, both models spend their capacity on the shared features (for the unified model these are the top eigen-directions, carrying $M$ times the energy), so they tie. Above $k_s+Mk_p$, the unified core holds everything. In between, the unified model must ration $r-k_s$ slots among $Mk_p$ private features, while each specialist needs only $k_p$.

**Corollary 1 (modality budget).** For $r\ge k$, a unified core of capacity $r$ matches specialists at the population level if and only if $M\le M^*=\lfloor (r-k_s)/k_p\rfloor$. For larger $M$, the per-modality loss $s^2(k_s+Mk_p-r)/M$ increases with $M$ towards $s^2k_p$.

**Lemma 2 (misalignment spectrum).** For $\theta>0$, write $c=\cos^2\theta$. Each shared feature $j$ contributes to $G$ one eigenvalue $s^2(1+(M-1)c)$ and $M-1$ eigenvalues $s^2(1-c)$.

*Proof.* The $M$ vectors $v_m=\cos\theta\,s_j+\sin\theta\,e^{(m)}_j$ are unit vectors with pairwise inner products $c$. Their Gram matrix $(1-c)I+c\mathbf 1\mathbf 1^\top$ has the stated eigenvalues, and $\sum_m v_mv_m^\top$ has the same non-zero spectrum. $\square$

So misalignment moves energy from the pooled shared direction into $(M-1)k_s$ "gap" directions of strength $s^2\sin^2\theta$. A unified model must spend capacity on these or pay for them as approximation error. A specialist sees no gap.

### 4.2 Finite data

**Exact reduction.** Use an orthogonal design, $X_m^\top X_m=nI$ (a standard idealisation; Section 8). The OLS estimate is then $\hat W_m=W_m+(\sigma/\sqrt n)Z_m$ with i.i.d. Gaussian $Z_m$, and the unified least-squares problem $\min_{U,V}\sum_m\|Y_m-X_mUV_m^\top\|^2$ has the same minimiser as $\min\sum_m\|\hat W_m-V_mU^\top\|^2$. The unified estimator is therefore the rank-$\rho$ truncated SVD of the stacked matrix $\hat W_{\rm st}=[\hat W_1;\dots;\hat W_M]\in\mathbb R^{Mq\times d}$, and each specialist is the truncated SVD of its own $\hat W_m$. Both are reduced-rank regressions [Izenman, 1975] on matrices with different shapes.

**Proposition 2 (df model).** At high SNR (all retained signal singular values well above the noise edge), first-order perturbation of the truncated SVD gives an excess risk equal to the discarded signal plus $\frac{\sigma^2}{n}\times$ the dimension of the manifold of rank-$\rho$ matrices of that shape. The per-modality risks are
$$L_{\rm u}(r)=\min_{\rho\le r}\frac1M\Big[\sum_{i>\rho}\lambda_i(G)+\frac{\sigma^2}{n}\rho\,(d+Mq-\rho)\Big],\qquad L_{\rm s}(r)=\frac1M\sum_m\min_{\rho\le r}\Big[\sum_{i>\rho}\lambda_i(G_m)+\frac{\sigma^2}{n}\rho\,(d+q-\rho)\Big].$$
The unified model estimates one $d$-dimensional subspace from $M$ modalities, which is the source of synergy. It also estimates $Mq\rho$ head parameters, including weights that must learn to ignore other modalities' features. We use Proposition 2 as a heuristic approximation and test it numerically (Section 5).

**Corollary 2 (synergy is bounded).** If $r\ge k_s+Mk_p$ and $\theta=0$, the ratio at full capacity is
$$\Gamma=\frac{\rho_u(d+Mq-\rho_u)}{M\,k(d+q-k)},\qquad \rho_u=k_s+Mk_p ,$$
independent of $n$. When $d\gg Mq$, $\Gamma\to\phi/M+(1-\phi)=1-\phi(1-1/M)$. In the df model, then, the largest relative gain from unification is the shared fraction, discounted by $1/M$, and the head term $Mq$ shrinks it further.

**Corollary 3 (head interference).** If $\phi=0$ (no shared structure) and capacity is ample, $\Gamma=(d+Mq-Mk_p)/(d+q-k_p)>1$ whenever $(M-1)(q-k_p)>0$. A dense unified model with no shared structure is worse than specialists even with unlimited capacity. Routing that restricts each head to its own modality's features (modality experts) removes this term.

**Corollary 4 (alignment tolerance).** Let $k_s+Mk_p\le r<k_s+Mk_p+(M-1)k_s$, so the core cannot also represent the gap directions. To first order in $\sin^2\theta$, the unified model loses $\frac{(M-1)k_s}{M}s^2\sin^2\theta$ in approximation and gains $G_n=\frac{\sigma^2}{n}\big[k(d+q-k)-\rho_u(d+Mq-\rho_u)/M\big]$ in variance. Synergy therefore requires
$$\sin^2\theta<\sin^2\theta_c=\frac{M\,G_n}{(M-1)k_s s^2}\;\propto\;\frac1n .$$
**The more data there is, the more precisely modalities must be aligned for unification to pay.**

### 4.3 Power-law spectra: a scale-invariant capacity tax

Real data have no clean $k_s,k_p$. A more realistic picture is a power-law spectrum of feature energies [Kaplan et al., 2020]. Suppose features have energies $\lambda_j\propto j^{-\alpha}$ ($\alpha>1$). A fraction $\phi$ of them (by density) are shared across all $M$ modalities and the rest are private, so the counting function of one modality is $N_{\rm s}(\lambda)=\lambda^{-1/\alpha}$.

**Proposition 3.** In the continuum limit and at the population level, for every capacity $r$ smaller than the total feature count,
$$\frac{L_{\rm u}(r)}{L_{\rm s}(r)}=R(\phi,M,\alpha)=\frac{\big(\phi M^{1/\alpha}+(1-\phi)M\big)^{\alpha}}{M},$$
which does not depend on $r$. Equivalently, a unified model needs $\kappa=R^{1/(\alpha-1)}$ times the capacity of a single specialist to match it. Relative to $M$ specialists of total capacity $Mr$, it saves a factor $M/\kappa=\big(\tfrac{M}{\phi M^{1/\alpha}+(1-\phi)M}\big)^{\alpha/(\alpha-1)}\ge 1$.

*Proof.* In the pooled Gram matrix $G$, shared features have energy $M\lambda$, with counting function $\phi(\lambda/M)^{-1/\alpha}$, and private features contribute $M(1-\phi)\lambda^{-1/\alpha}$. Hence $N_{\rm u}(\lambda)=c_u\lambda^{-1/\alpha}$ with $c_u=\phi M^{1/\alpha}+(1-\phi)M$. For a counting function $c\lambda^{-1/\alpha}$, keeping the top $r$ eigenvalues sets the cutoff $\lambda^*=(c/r)^\alpha$, and the residual is $\int_0^{\lambda^*}\lambda\,|dN|=\frac{c}{\alpha-1}(\lambda^*)^{1-1/\alpha}=\frac{c^\alpha r^{1-\alpha}}{\alpha-1}$. Dividing the unified residual by $M$ and taking the ratio gives $R=c_u^\alpha/M$. Setting $c_u^\alpha r_u^{1-\alpha}/M=r_s^{1-\alpha}$ gives $\kappa$. Finally $M/\kappa=(M/c_u)^{\alpha/(\alpha-1)}$, and $c_u\le M$ because $M^{1/\alpha}\le M$. $\square$

$R=1$ when $\phi=1$, and $R=M^{\alpha-1}$ when $\phi=0$. In this model, **when data are abundant (capacity-bound), competition does not fade with scale**: the loss ratio is a constant set by $\phi$, $M$ and $\alpha$. Synergy needs the data-bound regime, where the oracle width is set by $n$ and not by $r$.

**Prediction (data-bound crossover).** Under Proposition 2, the unified model's useful width $\rho_u^*(n)$ is where the pooled spectrum crosses $\sigma^2(d+Mq)/n$, so $\rho_u^*\propto n^{1/\alpha}$. Once $r$ exceeds a constant multiple of $\rho_u^*(n)$, capacity stops binding and the ratio drops below 1. Hence $r_\times(n)\propto n^{1/\alpha}$.

### 4.4 Remark 4: pooling shifts detection thresholds

For an $a\times b$ matrix with i.i.d. noise of variance $v$ plus a rank-one spike of singular value $\vartheta$, the spike is asymptotically detectable when $\vartheta^2>v\sqrt{ab}$ [Benaych-Georges and Nadakuditi, 2012]. Here $v=\sigma^2/n$. A specialist ($a=q$, $b=d$) detects a feature of energy $s^2$ when $n>\sigma^2\sqrt{qd}/s^2$. In the stacked unified matrix ($a=Mq$), a shared feature has $\vartheta^2=Ms^2$, giving the threshold $\sigma^2\sqrt{qd}/(s^2\sqrt M)$, which is $\sqrt M$ lower. A private feature keeps $\vartheta^2=s^2$, giving $\sigma^2\sqrt{Mqd}/s^2$, which is $\sqrt M$ higher. Unification therefore makes shared features visible earlier and private features later. This is a low-data effect that the df model leaves out.

## 5 Experiments

All experiments run on CPU. Linear experiments use the exact reduction of Section 4.2: we sample $\hat W_m=W_m+\sigma Z_m/\sqrt n$, compute both estimators, and pick the oracle width $\rho\le r$ by true risk. Unless stated, $s=\sigma=1$. Theory values come from `common.theory_losses` (Proposition 2 evaluated on the exact spectrum of $G$).

### 5.1 Exp. A: the competition window (linear)

Setup: $d=64$, $q=16$, $k_s=k_p=4$, $M=4$; $r=2..24$; 30 repetitions per cell (`results/linear_results.json`, Fig. `figA_capacity_sweep.png`).

| $n$ | competition window (sim) | window (df theory) | population window | ratio at $r=k=8$ (sim / theory) | ratio at $r=24$ (sim / theory) | max rel. error of ratio |
|---|---|---|---|---|---|---|
| 200 | $r\ge 6$, does not close by 24 | 6–18 | 5–19 | 1.40 / 1.46 | 1.05 / 0.94 | 10.5% |
| 1 000 | 5–19 | 5–19 | 5–19 | 5.50 / 5.62 | 0.96 / 0.94 | 2.3% |
| 5 000 | 5–19 | 5–19 | 5–19 | 26.1 / 26.5 | 0.94 / 0.94 | 1.5% |
| 100 000 | 5–19 | 5–19 | 5–19 | 516.5 / 521.3 | 0.94 / 0.94 | 0.9% |

(Window = range of $r$ with ratio $>1$; the population window is the open interval $(k_s, k_s+Mk_p)=(4,20)$.) As Proposition 1 predicts, the peak sits exactly at $r=k=8$, and its height grows roughly linearly in $n$ because the specialist loss at $r=k$ is pure variance. For $n\ge1000$, the full-capacity synergy ratio approaches $\Gamma=20\cdot108/(4\cdot8\cdot72)=0.9375$ (Corollary 2): the gain from unification is only about 6%, because the head term $Mq=64$ is as large as $d$. At $n=200$, the df model breaks down and the simulated ratio stays above 1 even at full capacity (Section 5.6).

### 5.2 Exp. B: the modality budget

Setup: $d=128$, $q=16$, $k_s=k_p=4$, $r=16$; $M=1..14$; 20 repetitions (Fig. `figB_modality_budget.png`). Corollary 1 predicts $M^*=(16-4)/4=3$. The simulated and theoretical $M^*$ both equal **3** for every $n\in\{10^3,5\cdot10^3,10^5\}$. The ratio at $M=3$ is 0.81, 0.79 and 0.79 (sim). At $M=4$ it jumps to 1.58, 5.22 and 92.5 (sim; theory 1.57, 5.24, 92.6). Data scarcity does not move the budget in this regime: the variance gain is too small to offset even one missing private feature.

### 5.3 Exp. C: misalignment phase diagram and alignment tolerance

Setup: $M=4$, $n=1000$, $\theta\in[0^\circ,90^\circ]$ (10 values), $r\in\{4,\dots,30\}$ (14 values), 12 repetitions (Fig. `figC_misalignment.png`). The df theory predicts the **sign** of $\log(L_{\rm u}/L_{\rm s})$ correctly in **140/140** cells. At $\theta=0$ there is synergy for $r\ge20$ (log-ratio $-0.05$). At $\theta=10^\circ$ the log-ratio is already $+0.11$ at every $r\ge 20$, and at $\theta\ge20^\circ$ unification loses at every capacity tested, including $r=4$ (Lemma 2: the gap moves energy out of the pooled direction).

A finer sweep (`exp_gap.py`, $r=30$, 41 angles in $[0^\circ,20^\circ]$, 16 repetitions) tests Corollary 4:

| $n$ | $\theta_c$ closed form | $\theta_c$ df theory (numerical) | $\theta_c$ simulation |
|---|---|---|---|
| 300 | 11.5° | 11.5° | none (no synergy at any $\theta$) |
| 1 000 | 6.3° | 6.3° | 4.8° |
| 3 000 | 3.6° | 3.6° | 3.5° |
| 10 000 | 2.0° | 2.0° | 2.1° |
| 30 000 | 1.1° | 1.1° | 1.1° |

The simulation confirms $\theta_c\propto n^{-1/2}$ for $n\ge 3000$. At $n=300$ the df model's predicted synergy does not occur at all, which is the same low-SNR failure as in Exp. A.

### 5.4 Exp. D: where the df model is accurate

At full capacity ($r=24$, setup of Exp. A), the relative error of the predicted unified loss is 17%, 8%, 19%, 9%, 4.6%, 1.8%, 0.8% and 0.4% for $n=50,100,200,500,10^3,3\cdot10^3,10^4,10^5$. The predicted *ratio* is off in sign at $n=200$ (sim 1.04, theory 0.94). The df model is reliable for $n\gtrsim 1000$, i.e. per-feature SNR $s^2n/(\sigma^2\sqrt{dq})\gtrsim 30$.

### 5.5 Exp. E: power-law spectra

**E1 (closed form).** We use discrete spectra with $2\times10^5$ features, $\alpha\in\{1.5,2,3\}$, $\phi\in\{.25,.5,.75\}$, $M\in\{2,4,8\}$, and $r$ from 10 to $10^4$ (Fig. `figE1_powerlaw_tax.png`). At $r=1000$ the population ratio matches $R$ within 0.19% ($\alpha=2$), 0.70% ($\alpha=3$) and 4.1% ($\alpha=1.5$; the heavier tail is more affected by truncation at $2\times 10^5$ features). Across $r=100\ldots10^4$ the ratio is nearly flat, as predicted. Representative values:

| $\alpha$ | $\phi$ | $M$ | $R$ | capacity tax $\kappa$ | saving vs $M$ specialists $M/\kappa$ |
|---|---|---|---|---|---|
| 2 | 0.25 | 4 | 3.06 | 3.06 | 1.31 |
| 2 | 0.50 | 4 | 2.25 | 2.25 | 1.78 |
| 2 | 0.75 | 4 | 1.56 | 1.56 | 2.56 |
| 2 | 0.50 | 8 | 3.66 | 3.66 | 2.18 |
| 3 | 0.50 | 4 | 5.45 | 2.33 | 1.71 |
| 1.5 | 0.50 | 4 | 1.47 | 2.17 | 1.85 |

So under these assumptions, with $\phi=0.5$ and four modalities, a dense unified model needs about 2.2–2.3 times the capacity of one specialist (about 56–58% of the total capacity of four specialists, for $\alpha=2$ and $3$) to match per-modality loss in the capacity-bound regime.

**E2 (finite data).** Setup: $d=512$, $q=96$, $M=4$, $\phi=0.5$, $\alpha=2$, 48 shared and $4\times48$ private features (Fig. `figE2_powerlaw_finite_data.png`). With $n=500$ the ratio falls below 1 at $r=8$ and plateaus at 0.90 (sim). With $n=10^7$ it stays above 1 up to $r=200$ and reaches 64 at $r=96$, where specialists have exhausted their 96 features. That spike is a finite-feature-count effect, the discrete analogue of Proposition 1's peak. Between $r=2$ and $64$ the population ratio stays between 1.6 and 4.64, close to $R=2.25$, as Proposition 3 predicts.

**E3 (crossover scaling).** For 9 values of $n\in[10^3,2.56\times10^5]$, we find the smallest $r$ beyond the competition peak with simulated ratio $<1$ (`crossover_results.json`). For $\alpha=2$, over the 8 points not limited by the finite feature count, $\log r_\times$ against $\log n$ has slope **0.567**. The df-predicted $\rho_u^*(n)$ has slope 0.577, and the asymptotic value is $1/\alpha=0.5$. The ratio $r_\times/\rho_u^*$ is stable at 0.54–0.65. For $\alpha=1.5$ only 4 points are unsaturated; their slopes are 0.74 (sim) and 0.78 (df), against an asymptotic 0.67.

### 5.6 Exp. G: pooling and detection thresholds (Remark 4)

Setup: full capacity ($r=40$), $d=64$, $q=16$, $M=4$, total signal energy 8 per modality, 40 repetitions (`threshold_results.json`).

- *Purely private* ($k_s=0,k_p=8$). Specialists start recovering signal at $n=96$ (loss 7.17 of 8). The unified model recovers nothing until $n=128$ (7.30 of 8). For $n\ge 256$ the ratio settles at 1.41, 1.37 and 1.34 (theory: $4/3$, Corollary 3). This is a **head-interference penalty at full capacity with no capacity competition at all**.
- *Purely shared* ($k_s=8,k_p=0$). The unified model starts recovering at $n=48$ (loss 7.04), while specialists recover nothing until $n=96$. The ratio then drops to 0.42 (theory 0.4167).

The direction of both threshold shifts matches Remark 4. The shared-feature shift is about $2\times=\sqrt M$, as predicted. The private-feature shift is smaller (about $1.3\times$) than the asymptotic $\sqrt M$, a finite-size effect we do not model.

### 5.7 Exp. F: nonlinear networks

Teacher: $y_m=A_m\,\mathrm{relu}(S^\top x)+B_m\,\mathrm{relu}(P_m^\top x)+\varepsilon$ with $d=32$, $q=8$, $k_s=k_p=6$, $\sigma=0.5$. Students: one-hidden-layer ReLU networks of width $h\in\{2,\dots,64\}$. The unified network shares the hidden layer and has per-modality heads; specialists have their own. We use Adam with minibatches of 256, 2000 steps, early stopping on a validation set, and report noiseless test MSE over 3 seeds (`mlp_summary.json`, Fig. `figF_mlp.png`). Mean ratios $L_{\rm u}/L_{\rm s}$:

| $n$ | $M$ | peak width | peak ratio | first width with ratio <1 after peak | $k_s+Mk_p$ | min ratio |
|---|---|---|---|---|---|---|
| 256 | 2 | 4 | 1.20 | 8 | 18 | 0.89 |
| 1024 | 2 | 8 | 1.56 | 16 | 18 | 0.61 |
| 4096 | 2 | 12 | 4.92 | 24 | 18 | 0.67 |
| 256 | 4 | 4 | 1.29 | 12 | 30 | 0.84 |
| 1024 | 4 | 12 | 2.57 | 24 | 30 | 0.55 |
| 4096 | 4 | 12 | 8.41 | 32 | 30 | 0.51 |
| 256 | 8 | 4 | 1.30 | 8 | 54 | 0.79 |
| 1024 | 8 | 12 | 3.20 | 32 | 54 | 0.69 |
| 4096 | 8 | 12 | 10.41 | 48 | 54 | 0.49 |

Four predictions carry over. (a) When data are sufficient ($n\ge1024$, $M\ge4$), the competition peak sits at $h=12=k_s+k_p$, and the peak height grows with $n$ and $M$. (b) At $n=4096$, the crossover width grows with $M$ (24, 32, 48), close to $k_s+Mk_p$ (18, 30, 54); the width grid is coarse. (c) At small $n$ the model is data-bound, and the crossover moves to small widths (8–12), as the data-bound mechanism of Section 4.3 predicts. (d) Above the crossover, unification gives real synergy, with ratios down to 0.49.

One prediction fails: below $k_s$ the theory predicts a tie, but the networks show mild competition (ratio 1.2–1.4 at $h\le4$). We suspect this is optimisation and feature-selection behaviour that the linear model does not have (Section 8). The nonlinear synergy is also stronger than the linear $\Gamma$ (0.49 against about 0.94). This is consistent with Corollary 2's dependence on $q/d$: here $q/d=0.25$ and shared ReLU features are costlier to estimate than heads. We have not tested that explanation.

## 6 Discussion: what this says about the limits of unified models

The results separate three distinct limits.

1. **Capacity limit (modality budget).** In the capacity-bound regime, a unified core can host about $(r-k_s)/k_p$ modalities before it loses to same-size specialists (Corollary 1, confirmed exactly in Exp. B). Under power-law spectra this becomes a constant capacity tax $\kappa$ that does not shrink with scale (Proposition 3). If this holds for real models, "just scale" does not remove competition. It only sets the parameter price, $\kappa$ times one specialist, which is still cheaper than $M$ specialists by $M/\kappa$.
2. **Statistical limit (bounded synergy).** In the model, the only source of synergy is better estimation of shared directions. It is bounded by Corollary 2 and reduced by head interference (Corollary 3). Synergy therefore appears when models are **data-bound**, and the crossover capacity grows as $n^{1/\alpha}$ (Exp. E3). This offers one consistent reading of the empirical observation that competition diminishes and synergy appears at larger model sizes [Aghajanyan et al., 2023]: at a fixed per-modality token budget, larger models are more data-bound. We have not checked their training budgets against this account. It is a hypothesis, not a reanalysis.
3. **Alignment limit.** Synergy needs the shared structure to be shared in the *same coordinates* after tokenisation. The tolerated misalignment shrinks as $n^{-1/2}$ (Corollary 4, Exp. C). In this model, the observed modality gap [Liang et al., 2022] is not a cosmetic property of embeddings: it directly sets whether unification pays.

**Architectural implications (model-based, not tested at scale).** Modality-specific parameters (MoE, MoT, MoMa) act on exactly the two penalty terms. Private capacity stops competing for the shared core, and heads stop paying the $Mq\rho$ interference cost. This is consistent with the reported advantages of modality-aware sparsity [Shukor et al., 2025; Lin et al., 2024; Liang et al., 2025]. The theory also predicts that such designs help most when $\phi$ is small and $M$ is large, and least when $\phi\to1$.

**Falsifiable predictions.**

- **P1:** At fixed per-modality data, the unified/specialist ratio against model size is non-monotone: it rises to a peak near the specialist's "saturation size" and then falls.
- **P2:** In the data-abundant regime, the ratio is approximately constant across model sizes (Proposition 3) and does not decay to 1.
- **P3:** The size at which a unified model crosses the competition barrier scales as $D^{1/\alpha}$, where $D$ is the per-modality data and $\alpha$ is the spectral decay of the representation.
- **P4:** Adding modalities to a fixed-size dense model eventually *increases* per-modality loss for every modality (Corollary 1), and the onset moves out linearly with model capacity.
- **P5:** Artificially misaligning one modality's input embeddings (applying a random rotation to part of the embedding space) eliminates synergy at an angle that shrinks with more data.
- **P6:** Modality-specific routing helps more as $M$ grows and as $\phi$ falls.

## 7 Pre-registered test protocol for real unified models

*Not executed here (no GPU).* (1) For $M\in\{2,3,4\}$ modalities (text, image tokens, audio tokens, action tokens), train dense decoder-only models at 5 sizes (e.g. 20M–1B) on a fixed per-modality token budget $D$, at 3 values of $D$. Also train unimodal models with the same grid. (2) Estimate $\alpha$ from the eigenspectrum of unimodal hidden representations, and estimate $\phi$ as the fraction of spectral energy captured by CCA between paired modalities (image–caption, audio–transcript) above a fixed threshold. (3) Compute the predicted $R$ and $\kappa$ (Proposition 3) and the crossover exponent $1/\alpha$. (4) **Pass criteria:** for P2, the measured capacity-bound ratio lies within a factor 1.5 of $R$ and has no significant trend with size (slope of log-ratio against log-size has a 95% CI containing 0). For P3, the fitted crossover exponent has a 95% CI that overlaps $[0.5/\alpha,\,1.5/\alpha]$. For P4, at the smallest size, per-modality loss increases monotonically once $M$ exceeds a fitted budget. For P5, the critical rotation angle decreases with $D$. Any of these failing falsifies the corresponding part of the theory as a description of transformers. (5) Repeat with an MoT variant to test P6.

## 8 Limitations

- **Linear, single-bottleneck model.** Real unified models are deep and nonlinear, and they can use **superposition** [Elhage et al., 2022] to pack more features than dimensions. Superposition could soften the hard capacity cliffs of Proposition 1, so the window edges should be read as idealised. The ReLU experiment agrees with the window and peak but shows competition below $k_s$, which the linear theory does not predict.
- **Equal-strength, orthogonal features and isotropic inputs.** Proposition 1 assumes equal feature energies and orthonormal heads. Proposition 3 relaxes the energies but keeps the shared/private dichotomy with a single $\phi$. Real modalities share structure in graded ways that $\theta$ only partly captures.
- **Orthogonal design and oracle width.** The exact reduction uses $X_m^\top X_m=nI$. Random Gaussian designs add fluctuations we did not simulate. Both models choose their width by oracle, which favours neither model but is more optimistic than practical model selection.
- **The df model is a heuristic.** It is accurate for $n\gtrsim1000$ in our settings, but at low SNR it mispredicts the sign of the interaction (Exp. A at $n=200$; Exp. C at $n=300$). Remark 4 identifies one cause but does not give a quantitative low-SNR theory. The private-feature threshold shift we observed is smaller than its asymptotic prediction.
- **No optimisation dynamics.** We study minimisers, not gradient-descent dynamics. The modality competition of Huang et al. [2022] and the overfitting-rate mismatch of Wang et al. [2020] are separate mechanisms, absent here, that could add to the capacity competition we describe.
- **Comparison convention.** We follow Aghajanyan et al. [2023] in comparing against specialists *of the same size*, which matches neither total parameters nor compute. Section 5.5 gives the conversion to $M$ specialists.
- **No real-model evidence.** All results are from simulations we ran. The connections to published large-model findings are qualitative, and the test protocol in Section 7 has not been run.
- **Small nonlinear study.** It uses 3 seeds, one teacher family and a coarse width grid, so the crossover widths are resolved only to the grid.

## 9 Conclusion

A shared-core model with shared and private features is enough to reproduce competition, synergy and the competition barrier, and it makes the limits of unification precise. In this model, a capacity-$r$ core holds about $(r-k_s)/k_p$ modalities. Under power-law spectra it charges a constant capacity tax that scale alone does not remove. Its synergy comes from data-bound pooled estimation and is bounded. That synergy also needs alignment between modalities that tightens as $1/\sqrt n$ in angle. The theory matches Monte Carlo simulation closely where its high-SNR assumption holds, and qualitatively matches nonlinear networks. It yields six falsifiable predictions and a protocol for testing them at scale.

## References

- Aghajanyan, A., Yu, L., Conneau, A., Hsu, W.-N., Hambardzumyan, K., Zhang, S., Roller, S., Goyal, N., Levy, O., and Zettlemoyer, L. (2023). Scaling Laws for Generative Mixed-Modal Language Models. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:265–279. arXiv:2301.03728.
- Benaych-Georges, F. and Nadakuditi, R. R. (2012). The singular values and vectors of low rank perturbations of large rectangular random matrices. *Journal of Multivariate Analysis*, 111:120–135. arXiv:1103.2221.
- Chameleon Team (2024). Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv:2405.09818.
- Chang, T. A., Arnett, C., Tu, Z., and Bergen, B. K. (2024). When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages. *Proceedings of EMNLP 2024*. arXiv:2311.09205.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., and Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of ACL 2020*, pp. 8440–8451. arXiv:1911.02116.
- Deng, C., Zhu, D., Li, K., et al. (2025). Emerging Properties in Unified Multimodal Pretraining (BAGEL). arXiv:2505.14683.
- Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., and Lei, Q. (2021). Few-Shot Learning via Learning the Representation, Provably. *International Conference on Learning Representations (ICLR)*. arXiv:2002.09434.
- Eckart, C. and Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3):211–218.
- Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. arXiv:2209.10652.
- Emu3 Team (BAAI) (2024). Emu3: Next-Token Prediction is All You Need. arXiv:2409.18869.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.
- Huang, Y., Lin, J., Zhou, C., Yang, H., and Huang, L. (2022). Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably). *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9226–9259. arXiv:2203.12221.
- Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). Position: The Platonic Representation Hypothesis. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:20617–20642. arXiv:2405.07987.
- Izenman, A. J. (1975). Reduced-rank regression for the multivariate linear model. *Journal of Multivariate Analysis*, 5(2):248–264.
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
- Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., and Zou, J. (2022). Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. *Advances in Neural Information Processing Systems 35 (NeurIPS)*. arXiv:2203.02053.
- Liang, W., Yu, L., Luo, L., et al. (2025). Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models. *Transactions on Machine Learning Research*. arXiv:2411.04996.
- Lin, X. V., Shrivastava, A., Luo, L., et al. (2024). MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts. arXiv:2407.21770.
- Reed, S., Zolna, K., Parisotto, E., et al. (2022). A Generalist Agent. *Transactions on Machine Learning Research*. arXiv:2205.06175.
- Shukor, M., Fini, E., Turrisi da Costa, V. G., Cord, M., Susskind, J., and El-Nouby, A. (2025). Scaling Laws for Native Multimodal Models. *Proceedings of ICCV 2025*. arXiv:2504.07951.
- Tripuraneni, N., Jin, C., and Jordan, M. I. (2021). Provable Meta-Learning of Linear Representations. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139. arXiv:2002.11684.
- Wang, W., Tran, D., and Feiszli, M. (2020). What Makes Training Multi-Modal Classification Networks Hard? *Proceedings of CVPR 2020*, pp. 12695–12705. arXiv:1905.12681.
- Wu, S., Zhang, H. R., and Ré, C. (2020). Understanding and Improving Information Transfer in Multi-Task Learning. *International Conference on Learning Representations (ICLR)*. arXiv:2005.00944.

---

### Appendix: reproducibility and provenance of numbers

| Paper item | Script | Output file |
|---|---|---|
| Exp. A–D (Tables in 5.1–5.4, Figs A–C) | `code/exp_linear.py` | `results/linear_results.json`, `results/linear_log.txt` |
| Alignment tolerance table | `code/exp_gap.py` | `results/gap_results.json` |
| Exp. E1–E2 | `code/exp_powerlaw.py` | `results/powerlaw_results.json`, `figE1_*.png`, `figE2_*.png` |
| Exp. E3 | `code/exp_crossover.py` | `results/crossover_results.json` |
| Exp. G | `code/exp_threshold.py` | `results/threshold_results.json` |
| Exp. F | `code/exp_mlp.py`, `code/analyze_mlp.py` | `results/mlp_results.json`, `results/mlp_summary.json`, `results/figF_mlp.png` |

Statements about other papers (sizes, FLOP savings, qualitative findings) come from the cited sources and were not reproduced here.
