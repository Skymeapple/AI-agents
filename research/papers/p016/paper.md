# The Stability Gap Is a Property of the Path, Not the Objective: An Exact Theory of Transient Forgetting Under Replay in Linear Models

**Target venue:** ICML (main track, theory of deep learning / continual learning)

**Code and data:** `code/` (run `bash code/run_all.sh`); all numbers in this paper are produced by those scripts and saved under `results/`.

---

## Abstract

Continual learners that rehearse old data, and even learners trained on the full union of old and new data, show a *stability gap*: as soon as training on a new task starts, accuracy on old tasks drops sharply and then partly or fully recovers (De Lange et al., 2023). This transient matters for any deployed learner that is evaluated while it trains. It is currently explained only empirically. We give an exact account of the stability gap in the simplest setting where it can occur, overparameterized linear regression with replay. The joint objective there is convex and has a solution with zero loss on both tasks, so terminal forgetting is zero by construction. We prove five results. (i) The old-task loss along the gradient-flow path has a closed form. For rank-one tasks it reduces to $G=\tfrac12 a c^2 s^2 (1-r)^2 r^{2r/(1-r)}$, where $r$ is the ratio of the two Hessian eigenvalues. (ii) Short-time law: the initial forgetting rate is reduced by replay only by the factor $(1-\rho)^2$, because the replay gradient vanishes at the old optimum. (iii) Existence criterion: a gap occurs for some initial error if and only if the new-task Hessian fails to leave the old task's null space invariant. Commuting task Hessians, for example, give no gap at all. (iv) Small replay does not make the gap shallower: as the replay weight $\rho\to0$, the gap depth is bounded below by a quantity that tends to the forgetting incurred *without* replay, while the slowest recovery mode has time constant at least $1/(\rho\|\Sigma_1\|)$. (v) Exact Gauss–Newton (natural-gradient) preconditioning of the replay objective removes the gap completely for every $\rho>0$. With damping, a residual gap remains that is controlled by the ratio of damping to the replay-scaled curvature. Diagonal (Adam-like) preconditioning carries no such guarantee. Simulations confirm the closed forms to numerical precision ($<10^{-6}$ relative error), and the bounds hold on every random instance tested (they are valid but loose). In two-layer networks trained on real data (sklearn digits; domain- and class-incremental; 4 seeds), the one-step law predicts the measured first-step forgetting to within 2% across 56 runs. Gap depth is as insensitive to small replay weights as the theory predicts: at $\rho=0.01$, 55–74% of the no-replay damage remains. Flow-like damped Gauss–Newton reduces the gap by 36–60% relative to GD at $\rho\in\{0.1,0.3\}$, against 15–22% for Adam. The exact-elimination results, however, do not transfer to small replay weights or to full Gauss–Newton steps. The theory reframes the stability gap as an artefact of Euclidean gradient geometry. It also yields falsifiable predictions for large models, which we state as a pre-registered protocol.

## 中文摘要

持续学习中的"稳定性缺口"(stability gap)指：模型一开始学习新任务，旧任务性能就骤降，随后部分或完全恢复。这一现象即使在经验回放乃至对新旧数据做联合训练时也会出现(De Lange 等，2023)，但至今只有经验性解释。本文在最简单的可出现该现象的设定(带回放的过参数化线性回归)中给出精确理论。该设定的联合目标是凸的，并且存在同时完美拟合两个任务的解，因此终态遗忘按构造为零。我们证明了五个结果。(i) 梯度流路径上的旧任务损失有闭式解；秩一任务的缺口深度为 $G=\tfrac12 a c^2 s^2(1-r)^2r^{2r/(1-r)}$。(ii) 短时律：由于旧任务最优点处回放梯度为零，回放只把初始遗忘速率降低 $(1-\rho)^2$ 倍。(iii) 存在性判据：当且仅当新任务 Hessian 不保持旧任务零空间不变时，存在会产生缺口的初始误差；例如两个任务的 Hessian 可交换时缺口恒为零。(iv) 小回放不会让缺口变浅：回放权重 $\rho\to0$ 时，缺口深度收敛到无回放时的遗忘量，而最慢恢复模态的时间常数至少为 $1/(\rho\|\Sigma_1\|)$。(v) 对回放目标做精确 Gauss–Newton(自然梯度)预条件，对任意 $\rho>0$ 都能完全消除缺口；带阻尼的预条件按阻尼大小成比例地消除缺口；对角(类 Adam)预条件则不能。数值实验以 $<10^{-6}$ 的相对误差验证了闭式解；渐近界在全部 50 个随机实例上成立(界成立但偏松)。我们还在两层神经网络(sklearn digits，域增量与类增量设定，4 个随机种子)上做了 CPU 规模的检验。在 56 次运行中，一步律对首步遗忘的预测误差在 2% 以内；缺口深度对小回放权重不敏感，与理论一致：$\rho=0.01$ 时仍保留无回放损伤的 55–74%。在 $\rho\in\{0.1,0.3\}$ 时，类流(小步长)阻尼 Gauss–Newton 相对 GD 把缺口减小 36–60%，Adam 只减小 15–22%。但精确消除缺口的结论不能推广到小回放权重或完整 Gauss–Newton 步长。本文把稳定性缺口重新解释为欧氏梯度几何的产物，并以预注册方案的形式给出可在大模型上证伪的预测。

---

## 1 Introduction

Whether a model can keep learning without catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999) is usually judged by *end-of-task* accuracy. The model is trained on task 2 and then evaluated on task 1. By that measure, rehearsal (replay) of old data is one of the most reliable remedies known (Rolnick et al., 2019; Wang et al., 2024). Continual *evaluation*, which measures old-task performance at every iteration, gives a different picture. De Lange, van de Ven and Tuytelaars (2023) found that replay, regularization and other methods all show a large but transient loss of old-task performance immediately after a task switch, followed by recovery. They called this the **stability gap**. Later work found it in class-incremental learning with pre-trained models (Harun & Kanan, 2024), in joint incremental training on *homogeneous* tasks drawn from a single distribution (Kamath et al., 2024), in continual pre-training of language models (Gupta et al., 2023), and concentrated largely in the linear classification head (Łapacz et al., 2025). A system that serves predictions while it learns is exposed during the trough, and the trough is invisible to end-of-task metrics.

Existing explanations are qualitative. De Lange et al. (2023) note that the gradient of a stability term is zero at the old optimum, so at a task switch only the plasticity gradient acts. Hess et al. (2023) argue that the continual-learning community should ask not only *what* to optimize but *how*. Remedies have been engineered empirically: layerwise proximal updates (Yoo et al., 2024), changes to the replay schedule and head (Harun & Kanan, 2024), and gain modulation (Rodriguez-Garcia et al., 2025). To our knowledge there is no model in which the stability gap can be computed exactly, its existence characterized, and its dependence on the replay ratio and the optimizer derived. The linear-regression theory of forgetting (Evron et al., 2022; Lin et al., 2023; Goldfarb & Hand, 2023; Mahdaviyeh et al., 2025) analyses the *endpoints* of sequential training, not the path between them.

**This paper.** We study the *trajectory* of gradient-based training on a replay objective $(1-\rho)L_2+\rho L_1$, starting from a minimizer of $L_1$, in overparameterized linear regression where both tasks can be fit exactly. This is the most favourable possible setting for replay. The objective is convex, the terminal old-task loss is zero for every $\rho>0$, and any rise in $L_1$ is therefore purely transient. If a stability gap appears here, it cannot be blamed on non-convexity, insufficient buffer size, representation drift, or an unreachable joint optimum.

**Contributions.**

1. **Exact path formula and closed form (§3, Prop. 1, Thm. 1).** The old-task loss is $L_1(t)=\tfrac12\|\Sigma_1^{1/2}e^{-H_\rho t}e_0\|^2$, with $H_\rho=(1-\rho)\Sigma_2+\rho\Sigma_1$. For rank-one tasks we obtain the gap depth and peak time in closed form, and they match numerical integration to $<10^{-6}$ relative error (E1).
2. **A short-time law (Thm. 2).** $L_1(t)=\tfrac12(1-\rho)^2t^2\|\Sigma_1^{1/2}\Sigma_2e_0\|^2+O(t^3)$, and the analogous statement holds exactly for one discrete step. Replay reduces initial forgetting only by $(1-\rho)^2$, which is 19% at $\rho=0.1$. This makes the qualitative argument of De Lange et al. quantitative.
3. **An existence criterion (Thm. 3).** A gap occurs for some admissible initial error if and only if $\Sigma_2$ does not leave $\ker\Sigma_1$ invariant (equivalently, $[\Sigma_2,P_{\mathrm{range}\,\Sigma_1}]\neq0$). Commuting task Hessians produce no gap, which we verify numerically.
4. **Small replay does not shrink the gap; it only shortens it (Thm. 4, Prop. 5).** $G(\rho)\ge F_\infty-O(\rho\log(1/\rho))$, where $F_\infty$ is the terminal forgetting *without* replay. The slowest recovering mode has time constant $\ge1/(\rho\|\Sigma_1\|)$.
5. **Preconditioning removes the gap (Thm. 6).** Exact Gauss–Newton flow on the replay objective has $L_1\equiv0$ for every $\rho>0$. Damped Gauss–Newton has a gap bounded by $O((\delta/\lambda^+_{\min})^2)$. Diagonal preconditioning gives no such guarantee and, in the linear model, does no better than GD.
6. **A nonlinear check and a pre-registered protocol (§6–7).** We test the predictions on two-layer networks trained on real data (CPU scale). The one-step law and the small-$\rho$ insensitivity transfer, curvature-aware flow-like updates roughly halve the gap at moderate $\rho$, and exact elimination does not transfer. We also list falsifiable predictions for large-scale continual training.

The contribution is a mechanistic theory of one phenomenon in a deliberately simple model, together with predictions and evidence about which parts survive nonlinearity. It is not a new continual-learning method.

## 2 Related Work

**Catastrophic forgetting and replay.** Forgetting under sequential training was identified by McCloskey & Cohen (1989) and reviewed by French (1999). Mitigation families include regularization such as EWC (Kirkpatrick et al., 2017), gradient projection such as GEM and OGD (Lopez-Paz & Ranzato, 2017; Farajtabar et al., 2020), and replay (Rolnick et al., 2019). Wang et al. (2024) give a recent survey, and van de Ven et al. (2022) define the task-, domain- and class-incremental scenarios used in our experiments.

**The stability gap.** De Lange et al. (2023) introduced continual evaluation and identified the gap. Kamath et al. (2024) showed that it persists in joint incremental training of homogeneous tasks. Harun & Kanan (2024) studied it with pre-trained networks and proposed mitigations. Łapacz et al. (2025) located most of it in the linear head. Earlier, Caccia et al. (2022) traced abrupt representation change at task boundaries under experience replay to new-class gradients disrupting old representations. Gupta et al. (2023) observed an analogous loss spike when continuing to pre-train language models. Rodriguez-Garcia et al. (2025) attenuate it with a two-timescale gain mechanism, and Yoo et al. (2024) with a proximal-point replay optimizer. Hess et al. (2023) argued for studying *how* continual objectives are optimized. All of these are empirical or conceptual. Our results give these observations a solvable model: why the gap survives full replay, why it is insensitive to small replay weights, and why curvature-aware optimizers help.

**Theory of forgetting in linear and kernel models.** Evron et al. (2022) computed exact forgetting for sequences of overparameterized linear regression tasks trained to convergence, and related it to principal angles between task subspaces. Lin et al. (2023) derived expected forgetting and generalization error under Gaussian features. Goldfarb & Hand (2023) analysed tasks related by random orthogonal transformations. Mahdaviyeh et al. (2025) showed that replay can *increase* terminal forgetting in overparameterized linear regression. Doan et al. (2021) analysed forgetting through the NTK overlap matrix, and Lee et al. (2021) used a teacher–student model to show that intermediate task similarity is worst. All of these study the converged solution of each phase. We study the *transient within a phase* under a joint (replay) objective. In that regime the endpoint analyses predict zero forgetting, yet the path forgets. Our no-replay limit ($\rho=0$) recovers the endpoint forgetting of Evron et al. as a special case (Prop. 1).

**Curvature-aware optimization in continual learning.** Natural-gradient and Kronecker-factored curvature (Martens & Grosse, 2015) underlie Fisher-based continual methods such as NCL (Kao et al., 2021), which casts gradient projection as a Fisher trust region. Mirzadeh et al. (2020) relate forgetting to the old task's Hessian along the displacement. Theorem 6 gives a sharp reason why curvature-aware updates should specifically suppress the *stability gap*: under the joint Gauss–Newton metric, an arbitrarily small replay weight makes the update invisible to the old task.

## 3 Setting

There are two regression tasks with design matrices $X_k\in\mathbb R^{n_k\times d}$ and second-moment matrices $\Sigma_k=X_k^\top X_k/n_k\succeq0$, for $k=1,2$. We assume **joint realizability**: there is $w^\star$ with $X_kw^\star=y_k$ for both $k$. This holds generically when $n_1+n_2\le d$. The losses are $L_k(w)=\tfrac1{2n_k}\|X_kw-y_k\|^2=\tfrac12(w-w^\star)^\top\Sigma_k(w-w^\star)$. Phase 1 ends at any interpolator $\bar w_1$ of task 1, so $L_1(\bar w_1)=0$. Phase 2 minimizes the **replay objective**

$$L_\rho(w)=(1-\rho)L_2(w)+\rho L_1(w),\qquad \rho\in[0,1),$$

where $\rho$ is the effective replay weight (the fraction of each batch's loss carried by old data). It is optimized by gradient flow $\dot w=-\nabla L_\rho(w)$, or by gradient descent with step $\eta$. Define the error $e=w-w^\star$, its initial value $e_0=\bar w_1-w^\star\in\ker\Sigma_1$, and the joint Hessian $H_\rho=(1-\rho)\Sigma_2+\rho\Sigma_1$.

**Definition (stability gap).** The *gap depth* is $G(\rho)=\sup_{t\ge0}L_1(w(t))-L_1(w(\infty))$. For realizable replay ($\rho>0$) we show $L_1(w(\infty))=0$, so $G=\sup_tL_1$. We also use the *no-replay forgetting curve* $F(\tau)=\tfrac12\|\Sigma_1^{1/2}e^{-\Sigma_2\tau}e_0\|^2$ and its limit $F_\infty=\tfrac12\|\Sigma_1^{1/2}P_{\ker\Sigma_2}e_0\|^2$, which is the endpoint forgetting studied by Evron et al. (2022).

## 4 Theory

All proofs are elementary. We give them in full because the results are only useful if they are checkable.

**Proposition 1 (path and endpoints).** Under gradient flow, $e(t)=e^{-H_\rho t}e_0$ and $L_1(t)=\tfrac12\|\Sigma_1^{1/2}e^{-H_\rho t}e_0\|^2$. Moreover $L_1(0)=0$. For $\rho\in(0,1)$, $L_1(\infty)=0$. For $\rho=0$, $L_1(\infty)=F_\infty$.

*Proof.* $\nabla L_\rho(w)=H_\rho e$, so $\dot e=-H_\rho e$. As $t\to\infty$, $e^{-H_\rho t}\to P_{\ker H_\rho}$. For $\rho\in(0,1)$, $H_\rho$ is a positive combination of PSD matrices, so $\ker H_\rho=\ker\Sigma_1\cap\ker\Sigma_2$ and $\Sigma_1P_{\ker H_\rho}=0$. For $\rho=0$, $\ker H_0=\ker\Sigma_2$. $\square$

Replay with any positive weight therefore eliminates *terminal* forgetting in this model. Everything that follows concerns the transient.

**Theorem 1 (closed form for rank-one tasks).** Let $\Sigma_1=a\,uu^\top$ and $\Sigma_2=b\,vv^\top$ with unit vectors at angle $\theta\in(0,\pi/2)$, working in $\mathrm{span}\{u,v\}$, and let $e_0=c\,u_\perp$. Let $0<\lambda_1<\lambda_2$ be the eigenvalues of $H_\rho$ with orthonormal eigenvectors $q_1,q_2$, and write $s=(u^\top q_1)(q_1^\top u_\perp)$ and $r=\lambda_1/\lambda_2$. Then
$$L_1(t)=\tfrac12ac^2s^2\left(e^{-\lambda_1t}-e^{-\lambda_2t}\right)^2,\qquad t^\star=\frac{\ln(\lambda_2/\lambda_1)}{\lambda_2-\lambda_1},\qquad G=\tfrac12ac^2s^2(1-r)^2r^{2r/(1-r)}.$$
As $\rho\to0$, $G\to\tfrac12ac^2\sin^2\theta\cos^2\theta=F_\infty$.

*Proof.* $u^\top e^{-Ht}u_\perp=\sum_ie^{-\lambda_it}(u^\top q_i)(q_i^\top u_\perp)$. Since $u\perp u_\perp$, the two coefficients sum to $u^\top u_\perp=0$, so they are $s$ and $-s$. Maximizing $(e^{-\lambda_1t}-e^{-\lambda_2t})^2$ gives $t^\star$ and the value $r^{r/(1-r)}(1-r)$. As $\rho\to0$, $q_2\to v$, $q_1\to v_\perp$ and $r\to0$, so $r^{2r/(1-r)}\to1$ and $s^2\to\sin^2\theta\cos^2\theta$. $\square$

The gap is the difference of a fast exponential (the new-task mode, rate $\lambda_2\approx b$) and a slow one (the replay-restored mode, rate $\lambda_1=O(\rho)$). Its depth depends only on the eigenvalue ratio $r$ and the eigenvector geometry $s$. As in the endpoint theory (Evron et al., 2022; Lee et al., 2021), the dependence on task angle is non-monotone and largest at intermediate similarity.

**Theorem 2 (short-time law).** For any $\rho\in[0,1)$,
$$L_1(t)=\tfrac12(1-\rho)^2t^2\,\|\Sigma_1^{1/2}\Sigma_2e_0\|^2+O(t^3).$$
For one GD step of size $\eta$, $L_1(w_1)=\tfrac12\eta^2(1-\rho)^2\|\Sigma_1^{1/2}\Sigma_2e_0\|^2$ exactly.

*Proof.* $\Sigma_1^{1/2}e^{-Ht}e_0=\Sigma_1^{1/2}(e_0-tHe_0)+O(t^2)$. Since $\Sigma_1e_0=0$, $He_0=(1-\rho)\Sigma_2e_0$ and $\Sigma_1^{1/2}e_0=0$. For GD, $e_1=(I-\eta H)e_0$ and the same cancellation is exact. $\square$

The replay gradient $\rho\Sigma_1e$ vanishes at the old optimum, so replay cannot oppose the first steps; it only removes the factor $(1-\rho)$ from the new-task push. This is a quantitative form of the zero-stability-gradient argument of De Lange et al. (2023). It predicts that the initial slope of old-task degradation, normalized by $(1-\rho)^2$, is independent of $\rho$.

**Theorem 3 (existence criterion).** The following are equivalent: (a) for some (equivalently, every) $\rho\in[0,1)$, $L_1(t)=0$ for all $t\ge0$ and all $e_0\in\ker\Sigma_1$; (b) $\Sigma_2\ker\Sigma_1\subseteq\ker\Sigma_1$; (c) $\Sigma_2P_1=P_1\Sigma_2$, where $P_1$ is the orthogonal projector onto $\mathrm{range}\,\Sigma_1$. In particular, $[\Sigma_1,\Sigma_2]=0$ implies there is never a gap. Conversely, for a specific $e_0$, $\Sigma_1\Sigma_2e_0\ne0$ implies $G(\rho)>0$ for every $\rho\in[0,1)$.

*Proof.* (b)⇒(a): $\Sigma_1$ annihilates $\ker\Sigma_1$, so $H_\rho$ maps $\ker\Sigma_1$ into itself, and so does $e^{-H_\rho t}$. Hence $\Sigma_1e(t)=0$. (a)⇒(b): by Theorem 2, $L_1\equiv0$ forces $\Sigma_1^{1/2}\Sigma_2e_0=0$, so $\Sigma_1\Sigma_2e_0=0$ for all $e_0\in\ker\Sigma_1$. (b)⇔(c): a symmetric matrix leaves a subspace invariant if and only if it leaves the orthogonal complement invariant, which holds if and only if it commutes with the projector. The last claim follows from Theorem 2 because the $t^2$ coefficient is positive. $\square$

So with full (realizable) joint training, a gap is generic. It is absent only under an exact alignment between the new task's curvature and the old task's null space. The gap belongs to the path: both endpoints have $L_1=0$.

**Theorem 4 (small replay does not shrink the gap).** For $\rho\in(0,1)$,
$$\sqrt{2G(\rho)}\;\ge\;\sup_{\tau\ge0}\Big[\sqrt{2F(\tau)}-\tfrac{\rho\tau}{1-\rho}\|\Sigma_1\|^{3/2}\|e_0\|\Big].$$
Let $\mu$ be the smallest nonzero eigenvalue of $\Sigma_2$. Taking $\tau=\log(1/\rho)/\mu$ gives
$$\sqrt{2G(\rho)}\ge\sqrt{2F_\infty}-\|\Sigma_1\|^{1/2}\|e_0\|\Big(\rho+\tfrac{\|\Sigma_1\|}{\mu(1-\rho)}\rho\log\tfrac1\rho\Big),$$
so $\liminf_{\rho\to0}G(\rho)\ge\sup_\tau F(\tau)\ge F_\infty$.

*Proof.* Let $A=(1-\rho)\Sigma_2$ and $B=\rho\Sigma_1$. Duhamel's formula $e^{-(A+B)t}-e^{-At}=-\int_0^te^{-(A+B)(t-s)}Be^{-As}\,ds$, with $\|e^{-(A+B)s}\|,\|e^{-As}\|\le1$ for PSD exponents, gives $\|e^{-H_\rho t}e_0-e^{-At}e_0\|\le\rho t\|\Sigma_1\|\|e_0\|$. Since $e^{-At}e_0=e^{-\Sigma_2\tau}e_0$ with $\tau=(1-\rho)t$, the reverse triangle inequality applied to $\|\Sigma_1^{1/2}\cdot\|$ gives the first bound. For the second, $\|\Sigma_1^{1/2}(e^{-\Sigma_2\tau}-P_{\ker\Sigma_2})e_0\|\le\|\Sigma_1\|^{1/2}e^{-\mu\tau}\|e_0\|$. $\square$

**Proposition 5 (recovery is slow).** If $F_\infty>0$, then $\lambda^+_{\min}(H_\rho)\le\rho\|\Sigma_1\|$, and for all $t$, $L_1(t)\le\tfrac12\|\Sigma_1\|\|e_0\|^2e^{-2\lambda^+_{\min}t}$. The slowest mode of recovery therefore has time constant at least $1/(\rho\|\Sigma_1\|)$.

*Proof.* Let $z=P_{\ker\Sigma_2}e_0$. Then $\Sigma_1z\neq0$ because $F_\infty>0$. Split $z=z_K+z_R$ with $z_K\in\ker H_\rho\subseteq\ker\Sigma_2$. Then $z_R\in\ker\Sigma_2\cap\mathrm{range}\,H_\rho$ and $\Sigma_1z_R=\Sigma_1z\neq0$. The Rayleigh quotient on $\mathrm{range}\,H_\rho$ is $z_R^\top H_\rho z_R/\|z_R\|^2=\rho\,z_R^\top\Sigma_1z_R/\|z_R\|^2\le\rho\|\Sigma_1\|$. The decay bound follows because $\Sigma_1^{1/2}P_{\ker H_\rho}=0$. $\square$

Together, Theorems 2 and 4 and Proposition 5 describe replay at small $\rho$. The fall is almost as deep as without replay, it begins at nearly the same rate, and the recovery takes time of order $1/\rho$. Small replay buffers buy eventual recovery, not protection during the transition.

**Theorem 6 (preconditioning).** Consider preconditioned flow $\dot w=-P\nabla L_\rho(w)$.
(a) *Exact Gauss–Newton/Newton*, $P=H_\rho^+$: for every $\rho\in(0,1)$, $L_1(t)=0$ for all $t$. The same holds for discrete steps $w\leftarrow w-\eta H_\rho^+\nabla L_\rho$ with any $\eta\in(0,2)$.
(b) *Damped*, $P=(H_\rho+\delta I)^{-1}$: $G\le\tfrac12\|\Sigma_1\|\|e_0\|^2\big(\delta/(e\,\lambda^+_{\min}(H_\rho))\big)^2$.
(c) For a general $P\succ0$, $L_1(t)=\tfrac12(1-\rho)^2t^2\|\Sigma_1^{1/2}P\Sigma_2e_0\|^2+O(t^3)$. A diagonal $P$ does not in general zero this coefficient.

*Proof.* (a) $H^+H=P_R$, the projector onto $\mathrm{range}\,H_\rho$, so $e(t)=P_Ke_0+e^{-t}P_Re_0$ (for discrete steps, $(1-\eta)^k$ in place of $e^{-t}$). Then $\Sigma_1e(t)=(1-e^{-t})\Sigma_1P_Ke_0+e^{-t}\Sigma_1e_0=0$, since $P_Ke_0\in\ker H_\rho\subseteq\ker\Sigma_1$ and $\Sigma_1e_0=0$. (b) $PH$ has the eigenvectors of $H$ with rates $\alpha_i=\lambda_i/(\lambda_i+\delta)$. Comparing with (a) mode by mode, $|e^{-\alpha t}-e^{-t}|\le(1-\alpha)t\,e^{-\alpha t}\le(1-\alpha)/(\alpha e)=\delta/(\lambda e)$. (c) This is Theorem 2 with $H$ replaced by $PH$. $\square$

Part (a) says that, in the metric defined by the *joint* curvature, the old task's curvature $\rho\Sigma_1$ (however small, provided it is nonzero) reshapes the update so that it moves only in directions the old task cannot see. In this model the stability gap is caused by the Euclidean geometry of gradient descent, not by the replay objective. Part (b) adds a practical constraint: because $\lambda^+_{\min}(H_\rho)\le\rho\|\Sigma_1\|$ (Prop. 5), the damping must be small *relative to the replay weight*. This leads to a falsifiable scaling prediction: the damping needed to suppress the gap scales like $\delta\propto\rho$.

## 5 Linear Experiments

All numbers in this section come from `code/linear_theory.py` and `code/linear_extras.py`. The raw outputs are in `results/linear_results.json` and `results/linear_extras.json`.

**E1: closed form (Theorem 1).** We take $a=b=c=1$ and $\theta=45^\circ$, and integrate gradient flow on a grid of step $0.01$. Table 1 compares the closed form with numerical integration.

*Table 1. Rank-one tasks, $\theta=45^\circ$. Closed form vs. numerical gradient flow.*

| $\rho$ | $G$ closed form | $G$ numeric | $t^\star$ closed | $t^\star$ numeric | $G/F_\infty$ |
|---|---|---|---|---|---|
| 0 | — | 0.12500 | — | (monotone) | 1.000 |
| 0.01 | 0.117324 | 0.117324 | 5.352 | 5.35 | 0.939 |
| 0.03 | 0.106624 | 0.106624 | 4.328 | 4.33 | 0.853 |
| 0.1 | 0.081529 | 0.081529 | 3.318 | 3.32 | 0.652 |
| 0.3 | 0.042213 | 0.042213 | 2.626 | 2.63 | 0.338 |
| 0.5 | 0.020667 | 0.020667 | 2.493 | 2.49 | 0.165 |

Agreement is within $10^{-6}$ relative error. The peak time is resolved to the grid spacing. At $\rho=0.01$ the trough reaches 94% of the forgetting without replay, as Theorem 4 predicts. The recovery time constant $1/\lambda_{\min}(H_\rho)$ has log–log slope $-0.9998$ against $\rho$ for small $\rho$ (Prop. 5). Figure 1 (`results/fig1_linear2d.png`) shows the trajectories and the closed-form $G(\rho,\theta)$ for $\theta\in\{15^\circ,\dots,75^\circ\}$. Depth is largest at intermediate angles for every $\rho$.

**E2: high-dimensional random tasks.** We use $d=100$ and 10 seeds. Task inputs lie in 60-dimensional subspaces that share 20 directions, with anisotropic scales in $[0.3,2]$ and $n_1=n_2=40$ samples. The initial error $e_0\in\ker\Sigma_1$ is random with unit norm. Results:

- *Gap depth vs. replay weight* (Figure 2b). Mean ± sd over seeds of $G(\rho)/F_\infty$: 0.884±0.086 ($\rho=0.003$), 0.768±0.105 (0.01), 0.604±0.113 (0.03), 0.383±0.095 (0.1), 0.257±0.070 (0.2), 0.187±0.052 (0.3), 0.103±0.030 (0.5), 0.050±0.015 (0.7). Without replay the trajectory overshoots its own endpoint slightly ($\sup_tF/F_\infty=1.019\pm0.036$). Even 3% replay leaves 60% of the no-replay damage in the transient.
- *Terminal loss* is $<2\times10^{-7}$ for all $\rho\ge0.003$ (Prop. 1). The damage is therefore entirely transient.
- *Commuting control* (Theorem 3). Both task Hessians are built in a shared eigenbasis with overlapping supports. Then $|L_1(t)|<10^{-17}$ for every $\rho$ and seed: no gap.
- *Short-time law* (Theorem 2). The measured $L_1(10^{-3})/10^{-6}$ matches $\tfrac12(1-\rho)^2\|\Sigma_1^{1/2}\Sigma_2e_0\|^2$ to within 0.76% for every $\rho$ and seed. The residual is consistent with the $O(t^3)$ term.
- *Recovery time* (Figure 2c). The time for $L_1$ to fall to 10% of its peak is 587, 200, 81, 34, 22, 18 and 16 for $\rho=0.003,\dots,0.5$. The log–log slope is $-0.79$ over $\rho\le0.1$ and $-0.89$ between the two smallest $\rho$. This approaches the $-1$ predicted by Prop. 5 as $\rho\to0$. At larger $\rho$ the recovery time saturates at the new-task time scale.
- *Bounds.* The lower bound of Theorem 4 holds in all 50 (seed, $\rho$) pairs with $\rho\in[10^{-4},10^{-2}]$. It is valid but loose: at $\rho=10^{-4}$ the bound is 0.0754 while the exact $G$ is 0.0861 and $F_\infty=0.0863$; at $\rho=10^{-2}$ the bound is 0.0083 while $G=0.0657$. The damped-preconditioning upper bound (Thm 6b) also holds everywhere but is loose by orders of magnitude.

**E3: preconditioning (Theorem 6).** We run discrete updates with a fixed step: GD with $\eta=1/\lambda_{\max}(H_\rho)$, and preconditioned variants with $\eta=0.5$. Table 2 reports the mean $G$ over 10 seeds. Every method drives $L_2$ below $10^{-4}$ of its initial value.

*Table 2. Gap depth $G$ by optimizer in the linear model (mean of 10 seeds). $\delta$ is relative to $\lambda_{\max}(H_\rho)$.*

| method | $\rho=0.01$ | $\rho=0.1$ | $\rho=0.3$ |
|---|---|---|---|
| GD | 0.0664 | 0.0341 | 0.0182 |
| diagonal preconditioner (Adam-like) | 0.0761 | 0.0379 | 0.0195 |
| damped GN, $\delta=10^{-1}$ | 0.0576 | 0.0195 | 0.0066 |
| damped GN, $\delta=10^{-2}$ | 0.0309 | 0.0049 | 0.0011 |
| damped GN, $\delta=10^{-3}$ | 0.0069 | 0.00053 | 0.00008 |
| damped GN, $\delta=10^{-4}$ | 0.00072 | 0.00002 | 0.000002 |
| exact GN ($H_\rho^+$) | 0 (to $10^{-15}$) | 0 | 0 |

Exact Gauss–Newton removes the gap. For fixed $\delta$ the residual gap grows sharply as $\rho$ falls, because $\lambda^+_{\min}(H_\rho)$ shrinks with $\rho$. At $\rho=0.01$ it falls roughly linearly in $\delta$ over this range. Diagonal preconditioning is slightly *worse* than GD. GN variants reach $L_2<1\%$ of its initial value in 4–7 steps, against 15–22 for GD. The reduction in the gap is not bought with slower learning.

**E4: beyond realizability.** We also test an under-parameterized noisy case: $d=30$, $n_1=n_2=40$, label noise $\sigma=0.3$, and partially shared teachers. Here $L_1(\infty)>0$, and we measure excess loss over task 1's least-squares optimum. The trajectory still overshoots its own endpoint, by 0.17, 0.53, 0.29 and 0.09 (mean of 10 seeds) for $\rho=0.01, 0.1, 0.3, 0.5$. These are small relative to the endpoint (10.2, 5.8, 2.8, 1.5) because the two tasks conflict strongly. A transient on top of terminal forgetting therefore persists without realizability, but the clean "zero at both ends" picture is specific to the realizable case.


## 6 Nonlinear Test: Two-Layer Networks on Real Data

**Design.** The linear theory makes five claims, and we ask which of them survive in a nonlinear network trained on real data. The model is a 64–16–10 tanh MLP (1,210 parameters) trained on the sklearn `digits` dataset (1,797 8×8 images; 1,200 for training, 597 for testing). We use squared loss on one-hot targets, which keeps the Gauss–Newton matrix $J^\top J$ the exact analogue of $\Sigma_k$ (Hui & Belkin, 2021, show that square loss is competitive for classification). Figure 3 (`results/fig3_mlp_curves.png`) shows the old-task loss and accuracy trajectories, and Figure 4 (`results/fig4_mlp_scaling.png`) the scaling and optimizer comparisons. We use two scenarios (van de Ven et al., 2022):

- **perm** (domain-incremental): task 2 applies a fixed random pixel permutation to a disjoint half of the training images. The 10-way head is shared.
- **split** (class-incremental): task 1 is classes 0–4 and task 2 is classes 5–9, with a shared 10-way head.

Phase 1 runs 4,000 full-batch GD steps on task 1. Phase 2 optimizes $(1-\rho)L_2+\rho L_1$ with the *entire* task-1 training set replayed. This is incremental joint training, the most favourable replay regime and the one in which Kamath et al. (2024) observed the gap. We compare three optimizers: GD with $\eta=1/\lambda_{\max}(G_\rho(\theta_0))$ for 4,000 steps; Adam (lr $10^{-3}$) for 4,000 steps; and damped Gauss–Newton (GGN) for 60–150 steps. We run 4 seeds × 7 values of $\rho$. The gap depth $G=\max_tL_1(t)-L_1(0)$ is measured on the old task's training loss, and accuracy is measured on the old task's test set. All numbers come from `results/mlp_summary.json` (mean ± sd over 4 seeds).

**A stability gap appears under full joint replay.** In split with GD at $\rho=0.1$, old-task test accuracy falls from 99.3% to a minimum of 46.3±6.0% and recovers to 87.5±2.2% by step 4,000. At $\rho=0.3$ the corresponding values are 99.3% → 93.6% → 95.5%. In perm with GD at $\rho=0.1$, accuracy goes 94.5% → 78.1% → 84.8%. Unlike the realizable linear model, recovery here is incomplete within the budget, so $G$ includes a terminal part. For split at $\rho=0.1$, the transient part is $G-(L_1(T)-L_1(0))=0.377-0.141=0.236$.

**Prediction 1 (one-step law, Thm 2): confirmed quantitatively.** For a network at a task-1 minimum, Theorem 2 predicts that the first GD step raises $L_1$ by $\tfrac12\eta^2(1-\rho)^2g_2^\top G_1g_2$, where $g_2$ is the new-task gradient and $G_1$ the old-task Gauss–Newton matrix at $\theta_0$. Across all 56 GD runs (2 scenarios × 4 seeds × 7 values of $\rho$), the ratio of measured to predicted increase is 0.994 on average (range 0.979–1.000) (Figure 4b). Replay acts on the first step only through the factor $(1-\rho)^2$. The small shortfall below 1 comes from the neglected term $\rho\,g_1$ and from curvature beyond second order.

**Prediction 2 (small replay leaves most of the gap, Thm 4): confirmed qualitatively.** $G(\rho)/G(0)$ for GD is 0.76 / 0.55 / 0.37 / 0.24 / 0.11 / 0.06 in perm and 0.89 / 0.74 / 0.58 / 0.39 / 0.18 / 0.09 in split, for $\rho=0.003, 0.01, 0.03, 0.1, 0.3, 0.5$ (Figure 4a). This is close to the linear model's 0.88 / 0.77 / 0.60 / 0.38 / 0.19 / 0.10. In split with $\rho\le0.01$, old-task accuracy reaches 0% at its minimum, exactly as without replay. In Figure 3, the curves for $\rho\in\{0,0.003,0.01\}$ are indistinguishable for the first ~30 steps, which is the time-scale separation of Theorem 4 seen directly.

**Prediction 3 (slow recovery, Prop. 5): consistent, but not measured precisely.** With $\rho\le0.03$, half-recovery from the peak was not reached within 4,000 steps in any run (all 4 seeds, both scenarios, GD and Adam). It was reached by step 600–1,600 at $\rho\ge0.1$ in split. We do not claim to have verified a $1/\rho$ law in the network: the runs are censored, and terminal forgetting is nonzero, which confounds the recovery time.

**Prediction 4 (curvature-aware updates shrink the gap, Thm 6): confirmed for flow-like steps, refuted for large steps.** Table 3 gives the gap depth by optimizer.

*Table 3. Gap depth $G$ in the MLP (mean ± sd over 4 seeds). Final $L_2$ is in parentheses.*

| scenario, $\rho$ | GD (4000 steps) | Adam (4000) | GGN $\delta{=}10^{-3}$, lr 0.2 (150) | GGN $\delta{=}10^{-3}$, lr 1 (60) | GGN $\delta{=}10^{-2}$, lr 1 (60) |
|---|---|---|---|---|---|
| perm, 0.01 | 0.429±0.026 (0.040) | 0.317±0.016 (0.022) | 0.385±0.012 (0.022) | 0.441±0.025 (0.015) | 0.420±0.015 (0.035) |
| perm, 0.1 | 0.183±0.032 (0.047) | 0.142±0.020 (0.029) | **0.118±0.007** (0.032) | 0.205±0.044 (0.023) | 0.174±0.027 (0.047) |
| perm, 0.3 | 0.087±0.020 (0.063) | 0.074±0.014 (0.039) | **0.050±0.004** (0.048) | 0.109±0.027 (0.039) | 0.081±0.019 (0.060) |
| split, 0.01 | 0.706±0.027 (0.011) | 0.633±0.018 (0.004) | 0.672±0.013 (0.004) | 0.715±0.016 (0.003) | 0.708±0.017 (0.009) |
| split, 0.1 | 0.377±0.018 (0.027) | 0.308±0.015 (0.013) | **0.183±0.008** (0.013) | 0.384±0.041 (0.010) | 0.341±0.018 (0.024) |
| split, 0.3 | 0.173±0.012 (0.042) | 0.146±0.011 (0.024) | **0.069±0.005** (0.029) | 0.246±0.042 (0.022) | 0.152±0.012 (0.040) |

GGN with a small step (lr 0.2, which approximates the preconditioned *flow*) reduces the gap relative to GD by 36–42% in perm and by 51–60% in split at $\rho\in\{0.1,0.3\}$, while also reaching a lower or comparable new-task loss. Adam, a diagonal preconditioner, reduces it by 15–22%. With a full step (lr 1), GGN does *not* help, and is sometimes worse than GD: the peak occurs at step 1. The discrete-step guarantee of Theorem 6(a) depends on the loss being exactly quadratic, and it fails once one Gauss–Newton step leaves the region where the linearization holds. At $\rho=0.01$, GGN-flow with $\delta=10^{-3}$ helps little (−10% in perm, −5% in split), and Adam does better there (−26% and −10%). Part of Adam's advantage may come from its different effective step size. The weak GGN effect at small $\rho$ matches Theorem 6(b) and Prop. 5, which say the damping must be small relative to $\rho$.

**Prediction 5 (damping must scale with $\rho$).** We test it directly with GGN-flow (lr 0.2, 150 steps) at $\rho\in\{0.01,0.1\}$ and $\delta\in\{10^{-3},10^{-4},10^{-5}\}$, using 2 seeds per scenario (`code/mlp_damping.py`).

*Table 4. Damping sweep for GGN-flow (lr 0.2, 150 steps; mean of seeds 0–1). Gap depth $G$, with final $L_2$ in parentheses.*

| scenario, $\rho$ | $\delta=10^{-3}$ | $\delta=10^{-4}$ | $\delta=10^{-5}$ |
|---|---|---|---|
| perm, 0.01 | 0.382 (0.018) | 0.369 (0.012) | 0.391 (0.010) |
| perm, 0.1 | 0.120 (0.030) | 0.119 (0.020) | 0.119 (0.016) |
| split, 0.01 | 0.684 (0.004) | 0.667 (0.002) | 0.626 (0.002) |
| split, 0.1 | 0.187 (0.015) | 0.149 (0.007) | 0.138 (0.005) |

The result is mixed and mostly negative for the prediction. In split, smaller damping reduces the gap monotonically: −26% at $\rho=0.1$ and −9% at $\rho=0.01$ going from $10^{-3}$ to $10^{-5}$. In perm, the gap is flat in $\delta$. At $\rho=0.01$, even $\delta=10^{-5}$ does not reproduce the near-elimination seen in the linear model (Table 2): the old-task accuracy minimum is still 43% in perm and 1% in split. The linear guarantee therefore does **not** transfer to small replay weights in this network. A plausible explanation, which we have not tested, is that once the trajectory moves far from $\theta_0$, the Gauss–Newton matrix no longer describes how the old task's outputs change, so the invariance argument of Theorem 6(a) holds only locally.

**Summary of transfer.** In the network, the one-step law (Thm 2) holds to within 2%. The insensitivity of gap depth to small $\rho$ (Thm 4) holds qualitatively, with numbers close to the linear model's. Curvature-aware flow-like updates roughly halve the gap at $\rho\in\{0.1,0.3\}$. The exact-elimination results (Thm 6a at small $\rho$, and the discrete full-step guarantee) do not transfer.



## 7 Discussion

**What the theory explains.** Several features of the stability gap have been reported empirically without a common explanation: it occurs under full joint training (Kamath et al., 2024), modest replay does not remove it (De Lange et al., 2023; Harun & Kanan, 2024), it concentrates in the linear head (Łapacz et al., 2025), and optimizer-level changes reduce it (Yoo et al., 2024; Rodriguez-Garcia et al., 2025). In our model they follow from a single structural fact. When the curvatures of the two tasks are not aligned (Theorem 3), the replay objective has a fast new-task mode and a slow old-task mode with very different time scales. Euclidean gradient flow follows the fast mode first and the slow, replay-driven correction later. The depth of the trough is set by the ratio of the time scales (Theorem 1 and Theorem 4), not by whether the optimum is shared. Łapacz et al.'s head localization fits this picture: a shared linear head trained on new classes is exactly a linear least-squares problem of the kind analysed here.

**Why "more replay" is a weak lever.** Theorem 4 and Prop. 5 show that a small replay weight purchases eventual recovery, at a time cost of order $1/\rho$, and buys almost no protection during the transition. In both the linear model and the MLP, $\rho=0.01$ leaves 55–94% of the no-replay damage at its peak (94% in the 2-D model, 77% in the $d=100$ model, 55–74% in the MLP). A system served while it learns, such as a continually updated recommender or language model, is evaluated during exactly that window.

**Why geometry is a strong lever.** Theorem 6(a) is the sharpest statement in the paper. In the realizable linear model, any positive replay weight combined with the *joint* Gauss–Newton metric removes the gap completely. This gives a principled reason why Fisher- and curvature-based continual learners (Kirkpatrick et al., 2017; Kao et al., 2021) and proximal replay methods (Yoo et al., 2024) should be especially effective against the *transient*, separately from their effect on terminal forgetting. Two qualifications came from the experiments rather than from the theory. In a nonlinear network the guarantee holds only for flow-like steps, since a full GN step overshoots the region where the linearization holds. And the required damping shrinks with $\rho$. In the linear model the corollary is that curvature estimates used to precondition replay should include the old-task curvature at weight $\rho$, with damping below $\rho\cdot\lambda_{\max}(\Sigma_1)$. In our network, reducing the damping helped in only one of the two scenarios (Table 4), so this corollary is unconfirmed outside the linear model.

**Pre-registered predictions for large models (not tested here).** We state the following as a protocol for groups with GPU resources. Setting: continual pre-training or fine-tuning with a replay mixture weight $\rho$ (e.g. Gupta et al., 2023), with the old-domain loss logged every step. **P1 (one-step law):** immediately after a domain switch, the old-domain loss increase over the first $k$ steps, divided by $(1-\rho)^2$, is independent of $\rho$ (for $k$ small enough that the curvature is approximately constant). **P2 (depth insensitivity):** for $\rho\le0.05$ the peak old-loss increase is at least 50% of the no-replay increase at the same step. **P3 (recovery scaling):** the time to recover half the peak scales as $\rho^{-\alpha}$ with $\alpha$ close to 1 for small $\rho$. **P4 (geometry):** at matched new-domain loss, a K-FAC/Shampoo-type preconditioner whose curvature statistics include the replay data shows a smaller gap than AdamW, and the advantage shrinks when the damping exceeds about $\rho$ times the top old-task curvature. **P5 (existence):** tasks whose gradient covariances are nearly co-diagonalizable (measurable through the commutator norm $\|[\hat\Sigma_2,\hat P_1]\|$ of the empirical NTK/GGN restricted to a sketch) show smaller gaps. Each of these would be falsified by a measured exponent, ratio or ordering outside the stated range.


## 8 Limitations

- **The model is deliberately minimal.** The theorems are for linear regression with exact realizability, deterministic gradient flow or full-batch GD, and two tasks. We do not analyse SGD noise, momentum, learning-rate schedules, finite replay buffers (we replay the entire old set with weight $\rho$), or more than two tasks.
- **The nonlinear test is small.** It uses one architecture with 1,210 parameters, one dataset (sklearn digits), squared loss, 4 seeds (2 for the damping test) and full-batch training. We have not shown that the quantitative results (e.g. a 51–60% gap reduction for GGN) transfer to convolutional networks, cross-entropy, or stochastic training. The cross-entropy GGN attempt we tried first diverged under saturated softmax outputs, and we switched to squared loss for that reason.
- **The damping-scaling prediction failed in the network.** At $\rho=0.01$, even $\delta=10^{-5}$ did not suppress the gap in the MLP (Table 4). We offer a hypothesis (nonlocal curvature change) but have not tested it.
- **Recovery-time scaling is not established in the network.** The runs were censored at 4,000 steps, and terminal forgetting is nonzero.
- **The bounds are loose.** The Theorem 4 lower bound and the Theorem 6(b) upper bound hold on every instance tested, but they are not tight. The closed form (Theorem 1) is exact only for rank-one tasks.
- **The GGN comparison is not compute-matched.** GGN steps cost far more than GD steps ($O(P^3)$ per step here). Our claim concerns the *trajectory geometry*, not wall-clock efficiency. Practical use would require K-FAC-style approximations, whose effect on the gap we did not measure.
- **Novelty.** To our knowledge, no prior work gives a closed-form or provable characterization of the stability gap in linear models. Our search (late 2026) found only empirical and conceptual treatments, but we cannot exclude concurrent work.


## 9 Conclusion

Asking whether a model can learn continually without catastrophic forgetting means asking about the whole learning trajectory, not only its endpoints. In the simplest setting where this can be posed exactly, overparameterized linear regression with replay, replay removes terminal forgetting completely, yet the path forgets almost as much as training without replay. Small replay weights mostly lengthen recovery rather than reduce the trough. The gap exists exactly when the new task's curvature does not preserve the old task's null space. In this model it is an artefact of Euclidean gradient geometry that exact Gauss–Newton preconditioning removes. In two-layer networks on real data, the one-step law holds to within 2%, and the insensitivity of the gap to small replay weights reappears. Flow-like curvature-aware updates halve the gap at moderate replay weights. Exact elimination, however, does not survive at small replay weights, even with very small damping. These results turn the stability gap from an empirical curiosity into a quantity that can be predicted, together with concrete tests for large-scale continual training.


## References

All references were checked for existence (authors, title, venue) by web search in November 2026.

1. Caccia, L., Aljundi, R., Asadi, N., Tuytelaars, T., Pineau, J., & Belilovsky, E. (2022). New insights on reducing abrupt representation change in online continual learning. *ICLR 2022*. arXiv:2104.05025.
2. De Lange, M., van de Ven, G. M., & Tuytelaars, T. (2023). Continual evaluation for lifelong learning: Identifying the stability gap. *ICLR 2023*. arXiv:2205.13452.
3. Doan, T., Abbana Bennani, M., Mazoure, B., Rabusseau, G., & Alquier, P. (2021). A theoretical analysis of catastrophic forgetting through the NTK overlap matrix. *AISTATS 2021*, PMLR 130:1072–1080.
4. Evron, I., Moroshko, E., Ward, R., Srebro, N., & Soudry, D. (2022). How catastrophic can catastrophic forgetting be in linear regression? *COLT 2022*, PMLR 178:4028–4079.
5. Farajtabar, M., Azizan, N., Mott, A., & Li, A. (2020). Orthogonal gradient descent for continual learning. *AISTATS 2020*, PMLR 108:3762–3773.
6. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128–135.
7. Goldfarb, D., & Hand, P. (2023). Analysis of catastrophic forgetting for random orthogonal transformation tasks in the overparameterized regime. *AISTATS 2023*, PMLR 206:2975–2993.
8. Gupta, K., Thérien, B., Ibrahim, A., et al. (2023). Continual pre-training of large language models: How to (re)warm your model? arXiv:2308.04014.
9. Harun, M. Y., & Kanan, C. (2024). Overcoming the stability gap in continual learning. *Transactions on Machine Learning Research*. arXiv:2306.01904.
10. Hess, T., Tuytelaars, T., & van de Ven, G. M. (2023). Two complementary perspectives to continual learning: Ask not only what to optimize, but also how. *Proceedings of the 1st ContinualAI Unconference*, PMLR 249:37–61.
11. Hui, L., & Belkin, M. (2021). Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks. *ICLR 2021*. arXiv:2006.07322.
12. Kamath, S., Soutif-Cormerais, A., van de Weijer, J., & Raducanu, B. (2024). The expanding scope of the stability gap: Unveiling its presence in joint incremental learning of homogeneous tasks. *CVPR Workshops 2024*, 4182–4186.
13. Kao, T.-C., Jensen, K. T., van de Ven, G. M., Bernacchia, A., & Hennequin, G. (2021). Natural continual learning: Success is a journey, not (just) a destination. *NeurIPS 2021*, 28067–28079.
14. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980.
15. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
16. Łapacz, W., Marczak, D., Szatkowski, F., & Trzciński, T. (2025). Exploring the stability gap in continual learning: The role of the classification head. *WACV 2025*, 7551–7560. arXiv:2411.04723.
17. Lee, S., Goldt, S., & Saxe, A. (2021). Continual learning in the teacher-student setup: Impact of task similarity. *ICML 2021*, PMLR 139:6109–6119.
18. Lin, S., Ju, P., Liang, Y., & Shroff, N. (2023). Theory on forgetting and generalization of continual learning. *ICML 2023*, PMLR 202.
19. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS 2017*, 6470–6479.
20. Mahdaviyeh, Y., Lucas, J., Ren, M., Tolias, A. S., Zemel, R., & Pitassi, T. (2025). Replay can provably increase forgetting. arXiv:2506.04377.
21. Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*, PMLR 37:2408–2417.
22. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.
23. Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. *NeurIPS 2020*.
24. Rodriguez-Garcia, A., Ghosh, A., & Ramaswamy, S. (2025). Dynamic gain neuromodulation attenuates stability gaps during continual joint training (earlier title: Noradrenergic-inspired gain modulation attenuates the stability gap in joint training). arXiv:2507.14056.
25. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *NeurIPS 2019*. arXiv:1811.11682.
26. van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). Three types of incremental learning. *Nature Machine Intelligence*, 4, 1185–1197.
27. Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE TPAMI*. doi:10.1109/TPAMI.2024.3367329.
28. Yoo, J., Liu, Y., Wood, F., & Pleiss, G. (2024). Layerwise proximal replay: A proximal point method for online continual learning. *ICML 2024*, PMLR 235:57199–57216.

