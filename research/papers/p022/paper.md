# Verification Blind Spots Bound Error Recovery in Long-Running Agents: Interval Invariance, a Correctness Ceiling, and Power-Law Returns to Re-Verification

**Target venue:** NeurIPS (main track; theory + controlled simulation)

---

## Abstract

Long-running autonomous agents recover from mistakes by verifying their work, checkpointing what passes, and rolling back what fails. The classical theory of checkpointing (Young, Daly, and the silent-error work of Benoit, Robert and colleagues) optimizes *time*. It assumes either that errors are detected at once or that a guaranteed verifier exists. Agents have neither. Their errors are silent, and their verifiers (tests, LLM judges, self-review) have blind spots. We study the *verify–checkpoint–rollback* (VCR) model with silent errors, per-error heterogeneous detectability, false alarms, and a step hazard that may grow with context age. We prove four results. (1) **Interval invariance:** with constant hazard, the probability that the final state is clean is exactly $\theta_m^N$, where $\theta_m=(1-p)/(1-p+p\mu_m)$ and $\mu_m=\mathbb E[(1-d)^m]$. It does not depend on the checkpoint interval, false-alarm rate, or any cost. Checkpoint frequency buys time, not correctness. (2) **Blind-spot ceiling:** no policy driven by verifier outputs (adaptive intervals, deep rollback, any number of re-checks) can exceed $((1-p)/(1-p+p\rho))^N\approx e^{-Np\rho}$, where $\rho$ is the mass of errors the verifier ensemble can never flag. (3) **Power-law returns:** when detectability has density $\propto d^{a-1}$ near zero, repeated verification lowers the miss rate only as $m^{-a}$. False alarms make the time cost grow geometrically in $m$. (4) **Context-age hazard:** when the hazard grows linearly with context age, the time-optimal interval moves from the square-root law $T^*\propto c^{1/2}$ toward $T^*\propto c^{1/3}$. Resuming from a checkpoint with a fresh context beats restoring a context snapshot beyond a crossover horizon. A step-level simulator agrees with the closed forms in 96 configurations; the largest relative makespan error is 0.32%. We also measure detectability on 102 AST mutants of 8 programs, using test-sampling verifiers. One test pool leaves a blind spot of 11.5% of non-equivalent mutants. A second, edge-case pool removes the joint blind spot entirely. With the combined verifier and 8 calls, the simulated 1000-step correctness rises from 0.49 to 0.99. In this toy setting, verifier *diversity* matters more than verification *frequency*. We end with a falsifiable test protocol for LLM agents.

## 中文摘要

长时间自主运行的智能体通常通过“验证—检查点—回滚”来做错误恢复：先验证工作，通过的部分保存为检查点，失败则回滚。经典检查点理论（Young、Daly，以及 Benoit、Robert 等人关于静默错误的工作）以运行时间为优化目标，并假设错误会被立即发现，或存在一个保证正确的验证器。智能体两者都没有：它的错误是静默的，验证器（单元测试、LLM 评审、自我检查）存在盲区。本文建立了一个“验证—检查点—回滚”（VCR）模型，包含静默错误、逐错误异质的可检测性、误报，以及随上下文长度增长的出错率，并证明四个结果。（1）**间隔不变性**：在恒定出错率下，最终状态正确的概率恰为 $\theta_m^N$，与检查点间隔、误报率和各项开销都无关。也就是说，检查点频率只影响时间，不影响正确性。（2）**盲区上限**：任何只依据验证器输出做决策的策略（自适应间隔、深度回滚、任意多次复验）的正确率都不可能超过 $((1-p)/(1-p+p\rho))^N\approx e^{-Np\rho}$，其中 $\rho$ 是验证器组合永远无法发现的错误所占比例。（3）**复验的幂律回报**：若可检测性在零附近的密度为 $d^{a-1}$，重复验证只能使漏检率以 $m^{-a}$ 的速度下降，而误报会使时间成本随 $m$ 几何增长。（4）**上下文老化**：当出错率随上下文长度线性增长时，最优间隔从平方根律变为趋近立方根律；超过某个任务长度后，“以全新上下文从检查点恢复”优于“恢复上下文快照”。逐步模拟器在 96 组配置下与闭式解一致（运行时间的最大相对误差为 0.32%）。我们还在 8 个程序的 102 个语法树变异体上测量了可检测性。单一测试池对 11.5% 的非等价变异体完全不可见；加入一个边界用例测试池后，联合盲区降为零，1000 步任务的正确率从 0.49 提升到 0.99（8 次验证）。在这一玩具设定中，验证器的多样性比验证频率更重要。文末给出了一个可在 LLM 智能体上检验这些预测的实验协议。

---

## 1 Introduction

Agents now run for hours and across many context windows. Kwa et al. [5] report that the length of tasks frontier agents complete at 50% reliability has doubled roughly every seven months. Ord [6] shows that a constant per-minute failure hazard explains much of that data. Sinha et al. [7] find that per-step accuracy itself degrades over long executions, partly because models condition on their own earlier mistakes. Engineering practice has converged on a recovery loop. The agent does a chunk of work, checks it (tests, a judge model, self-review), externalizes progress (a git commit, a progress file [15]), and restarts from the last good point if something breaks. Recent systems make rollback a first-class runtime primitive (AgentRewind [18]; rollback-with-reflection [19]). Empirical studies trace where errors enter long trajectories and which controls reach them [20].

This loop is the agent version of checkpoint/restart, a problem studied in high-performance computing for fifty years [1,2,3,4]. Agents break two assumptions of that theory.

- **No guaranteed verifier.** HPC silent-error patterns end every period with a *guaranteed* verification that detects all silent errors [3]. For agents no such oracle exists. Unit tests miss behaviors, LLM judges share blind spots with the generator [10,11], and intrinsic self-correction is unreliable [9]. Stroebl et al. [8] show that resampling against an imperfect verifier has an accuracy ceiling in single-shot generation. We ask what the analogous ceiling is for *long-horizon* recovery, where errors accumulate and recovery decisions are sequential.
- **Hazard depends on the agent's own state.** Machine failures arrive as a process the scheduler does not influence. An agent's error rate depends on how long its current context is [7,14]. Whether a rollback restores the old context or starts afresh therefore changes the hazard itself.

**Question.** For a long-running agent whose verifiers are imperfect in a structured way, which recovery design choices affect *end-to-end correctness*, which affect only *time*, and what bounds correctness no matter how recovery is scheduled?

**Contributions.**

1. **A model** (§3). The verify–checkpoint–rollback (VCR) model has silent persistent errors, a per-error *detectability* distribution $\mathcal D$ (including a blind-spot atom $\rho$ at $d=0$), false alarms, verification/checkpoint/rollback/re-orientation costs, and a context-age-dependent hazard with two resumption semantics ("fresh" and "snapshot").
2. **Interval invariance** (Theorem 1). Under constant hazard, end-to-end correctness is exactly $\theta_m^N$. It is independent of the checkpoint interval $T$, the false-alarm rate $f$, and all costs. We also give an exact makespan formula (Proposition 1).
3. **A blind-spot ceiling** (Theorem 2). No verifier-driven policy, however adaptive, beats $((1-p)/(1-p+p\rho))^N$. The bound is tight. The only levers are a lower step hazard $p$ and a smaller *joint* blind spot $\rho$, which requires verifier diversity.
4. **Power-law returns to re-verification** (Proposition 3). Heterogeneous detectability makes miss rates fall as $m^{-a}$ rather than geometrically. With false alarms, the time cost grows as $(1-f)^{-m}$.
5. **Context-age effects** (Propositions 4–5). The optimal interval moves from a $c^{1/2}$ toward a $c^{1/3}$ law. Under snapshot restoration, correctness decays super-exponentially in task length, and fresh-context resumption becomes time-optimal beyond a crossover horizon.
6. **Validation and a measurement** (§6). An independent step-level simulator checks every closed form (96 configurations × 3000 runs). We measure a non-synthetic detectability distribution on 102 program mutants and use it end to end. We finish with a pre-registered protocol for testing the predictions on LLM agents (§7.3).

All numbers below come from code in `code/` with outputs in `results/`. Each is labeled with its source file.

## 2 Related Work

**Checkpointing theory.** Young [1] derived the first-order optimal checkpoint interval $\sqrt{2\delta M}$ for fail-stop failures. Daly [2] gave higher-order corrections. For *silent* errors, Benoit, Robert and colleagues combine checkpoints with verifications, including partial verifications with recall below one. They characterize optimal patterns under the assumption that each pattern ends with a guaranteed verification [3,4]. Our time results (Proposition 1, the $\beta=0$ case of Proposition 4) are agent-flavored instances of that theory. What is new is (i) dropping the guaranteed verifier, which makes *correctness* rather than time the binding objective, (ii) per-error heterogeneous detectability rather than a single recall, and (iii) a hazard that depends on context age, together with two resumption semantics.

**Long-horizon agent reliability.** METR's time-horizon measurements [5], Ord's constant-hazard reading of them [6], and Sinha et al.'s long-execution study [7] motivate the per-step hazard model. Vending-Bench [13] documents performance breakdown over long horizons even when each sub-task is simple. The Chroma "context rot" report [14] documents performance degradation as input length grows. MAST [12] classifies multi-agent failures and includes a "task verification" category. Our model gives a quantitative account of why weak verification caps long-run reliability.

**Recovery mechanisms for agents.** Reflexion [16] uses verbal feedback across trials. LATS [17] searches over action trees with environment feedback, which is a form of backtracking. AgentRewind [18] restores agent context and environment snapshots together. Rollback-with-reflection [19] keeps reflection memory across rollbacks. These are mechanisms. Our results are design constraints that apply to any of them. In particular, the "fresh vs snapshot" distinction (§5.3) maps directly onto whether a system restores the old context or resumes from externalized state [15,18].

**Imperfect verifiers.** Stroebl et al. [8] show that resampling against unit tests with false positives has an accuracy ceiling. Kim et al. [11] find that different LLMs' errors are strongly correlated: when two models both err, they agree 60% of the time on one leaderboard dataset. Self-Correction Bench [10] reports that models fail to correct their own errors far more often than identical external ones. These findings motivate a blind-spot atom and correlated blind spots across verifiers. Theorem 2 extends the single-shot ceiling of [8] to sequential, adaptive recovery over long horizons. It also identifies the joint blind spot as the binding quantity. Mutation analysis [21] supplies our non-synthetic error population.

## 3 The Verify–Checkpoint–Rollback Model

**Progress and errors.** The task requires $N$ committed steps. Each executed step takes one time unit. A step executed at context age $a$ (steps since the current context began) introduces an error with probability $p(a)$. We use $p(a)=p_0(1+\beta a)$, capped at 1; $\beta=0$ is the constant-hazard case. Errors are *silent* (nothing signals them) and *persistent* (a corrupted committed state remains corrupted). Each error $e$ draws a detectability $d_e\sim\mathcal D$ independently. We write $\rho=\Pr_{\mathcal D}(d=0)$ for the blind-spot mass and $\mu_m=\mathbb E_{\mathcal D}[(1-d)^m]$ for the probability that an error survives $m$ verifier calls.

**Verifier.** One verifier call on a segment flags it if a false alarm fires (probability $f$, independent) or if any error $e$ in the segment is detected (independently with probability $d_e$). A call costs $V$.

**Policy (baseline schedule).** Execute a segment of $T$ steps, then make $m$ verifier calls. If none flags, write a checkpoint (cost $C$) and continue. Otherwise roll back to the last checkpoint (cost $R$) and re-execute the segment with fresh randomness. Two resumption semantics:
- *fresh*: every (re)start from a checkpoint opens a new context at age 0 and pays a re-orientation cost $K$. This corresponds to resuming from externalized state such as a progress file [15].
- *snapshot*: the context is restored with the checkpoint, so age continues from the committed progress. This corresponds to restoring context snapshots [18].

**Verifier-driven policies (general class).** In Theorem 2 we allow any policy whose decisions are measurable with respect to its own randomness and the history of verifier outputs. Decisions may include segment lengths, the number of calls, which earlier checkpoint to roll back to, re-verification of old segments, and stopping.

**Objectives.** The final state is *clean* if none of the $N$ committed steps contains an error. We study $P_{\rm clean}$ and the expected makespan $\mathbb E[\mathcal T]$.

**Interpretation.** $\mathcal D$ summarizes verifier quality *per error*. A homogeneous recall $r$ corresponds to $\mathcal D=\delta_r$. A unit-test suite that never exercises some behavior puts mass at 0. An LLM judge that catches some error types sometimes and others almost never gives a spread-out $\mathcal D$ with mass near 0.

## 4 Correctness Results

### 4.1 Interval invariance

**Theorem 1 (interval invariance).** Under the baseline schedule with constant hazard $p$ and any $T\ge1$, $m\ge1$, $f<1$, and costs $V,C,R,K\ge0$, under either resumption semantics,
$$P_{\rm clean}=\theta_m^{\,N},\qquad \theta_m=\frac{1-p}{1-p+p\,\mu_m}.$$
With age-dependent hazard and fresh resumption, $P_{\rm clean}=\prod_{\text{segments}}\prod_{a=0}^{L-1}\theta_m(p(a))$, where $\theta_m(q)=(1-q)/(1-q+q\mu_m)$. With snapshot resumption, $P_{\rm clean}=\prod_{x=0}^{N-1}\theta_m(p(x))$, where $x$ is committed progress.

*Proof.* Consider one segment attempt of length $L$ with step hazards $q_1,\dots,q_L$. Steps err independently, and each error survives all $m$ calls with probability $\mathbb E[(1-d)^m]=\mu_m$, independently across errors. A call raises no false alarm with probability $1-f$. So
$$\Pr(\text{pass})=(1-f)^m\prod_{i=1}^L\big(1-q_i+q_i\mu_m\big),\qquad \Pr(\text{pass}\wedge\text{clean})=(1-f)^m\prod_{i=1}^L(1-q_i).$$
Attempts at the same segment are i.i.d. because re-execution uses fresh randomness and the same hazards (for fresh semantics, ages restart at 0; for snapshot semantics, ages restart at the checkpoint's age). The committed content of a segment is therefore distributed as one attempt conditioned on passing. It is clean with probability $\prod_i (1-q_i)/(1-q_i+q_i\mu_m)$. The factor $(1-f)^m$ cancels. Segments are independent, so the product over segments gives the claim. With constant hazard every factor is $\theta_m$, and there are $N$ of them. $\square$

**Reading.** Checking more often, checkpointing more often, false alarms, and cheaper rollbacks all change *how long* the run takes. Under constant hazard, none of them changes *how likely the result is to be right*. Correctness is fixed by the per-step hazard and by how well a passing segment's errors are screened ($\mu_m$). Shorter segments mean more checks per step but fewer steps per check, and the two effects cancel exactly. To first order, $P_{\rm clean}\approx\exp(-Np\mu_m)$. Verification multiplies the effective hazard by $\mu_m$. The 50%-correctness horizon therefore grows by a factor of $1/\mu_m$, up to $O(p)$ corrections (§6.6).

**Proposition 1 (exact makespan).** Let $s_j$ be the pass probability of segment $j$ (length $L_j$). Then
$$\mathbb E[\mathcal T]=\sum_j\Big[\frac{L_j+mV+K}{s_j}+R\Big(\frac1{s_j}-1\Big)+C\Big],$$
with $K=0$ under snapshot semantics. *Proof:* the number of attempts is geometric with mean $1/s_j$; each attempt costs $L_j+mV+K$, each failure adds $R$, and the single success adds $C$. $\square$

### 4.2 The blind-spot ceiling

**Theorem 2 (blind-spot ceiling).** Let the hazard be constant at $p$, and let $\rho=\Pr(d=0)$. For every verifier-driven policy that terminates with $N$ committed steps,
$$P_{\rm clean}\;\le\;\Big(\frac{1-p}{1-p+p\rho}\Big)^{N}\;\approx\;e^{-Np\rho}.$$
The bound is attained, for every $T$, $m\ge1$ and $f<1$, by the baseline schedule whenever $\mathcal D$ is supported on $\{0,1\}$.

*Proof.* Classify each executed step as clean (probability $1-p$), *visible* error ($d>0$; probability $p(1-\rho)$), or *blind* error ($d=0$; probability $p\rho$). These categories are i.i.d. across executed steps because the hazard is constant, and they are independent of the policy's randomness. Let $\mathcal G$ be the $\sigma$-algebra generated by the policy's randomness, the visible-error indicators and their detectabilities, the false-alarm coins, and the detection coins. Blind errors are never detected, so every verifier output is $\mathcal G$-measurable. Hence every decision is $\mathcal G$-measurable, and so is the identity of the $N$ committed steps. Given $\mathcal G$, each executed step that is not a visible error is blind with probability $p\rho/(1-p+p\rho)$, independently across steps. Therefore
$$\Pr(\text{all committed steps clean}\mid\mathcal G)=\mathbf 1\{\text{no committed visible error}\}\cdot\Big(\tfrac{1-p}{1-p+p\rho}\Big)^N.$$
Taking expectations gives the bound. If $\mathcal D$ is supported on $\{0,1\}$, any $m\ge1$ calls flag every visible error, so no visible error is ever committed and the indicator is 1. $\square$

**Corollary (diversity is the lever).** Suppose the policy draws on a set of verifiers $\mathcal V$, and $\rho$ is the mass of errors invisible to every verifier in $\mathcal V$. Then the theorem applies with that joint $\rho$. Repeating the same verifier, re-checking old checkpoints with it, adapting intervals, or rolling back deeper cannot push correctness above $e^{-Np\rho}$. Adding a verifier whose blind spot does not overlap the existing ones can. If blind spots were independent the joint mass would be the product of the individual masses. Correlated errors across models [11] suggest it will often be larger, and our mutation data (§6.5) shows the overlap directly.

**Remark (age hazard).** With an age-dependent hazard the same argument goes through step by step, with ceiling $\prod_{\text{committed steps}}(1-p_t)/(1-p_t+p_t\rho)$. That product now depends on the policy through the ages at which committed steps were executed. This is the only channel through which scheduling can affect the ceiling (§5).

### 4.3 Returns to re-verification

If $\rho=0$, Theorem 2 imposes no ceiling, and $P_{\rm clean}\to1$ as $m\to\infty$ by Theorem 1. The question is how fast.

**Proposition 3 (power-law returns).** If $\mathcal D$ has no atom at 0 and a density $g$ with $g(d)\sim\kappa\, d^{a-1}$ as $d\downarrow0$, then $\mu_m\sim\kappa\,\Gamma(a)\,m^{-a}$. For $\mathcal D=\mathrm{Beta}(a,b)$, $\mu_m=B(a,b+m)/B(a,b)\sim\frac{\Gamma(a+b)}{\Gamma(b)}m^{-a}$. A homogeneous verifier ($\mathcal D=\delta_r$) gives $\mu_m=(1-r)^m$ instead. If $d\ge d_{\min}>0$ almost surely, then $\mu_m\le(1-d_{\min})^m$.

*Proof.* $\mu_m=\int_0^1(1-d)^m g(d)\,dd$. Substitute $d=u/m$ and apply dominated convergence (a standard Laplace-type argument): $\mu_m\sim\kappa m^{-a}\int_0^\infty u^{a-1}e^{-u}du$. The Beta identity is the Beta integral, and the asymptotic follows from $\Gamma(b+m)/\Gamma(a+b+m)\sim m^{-a}$. $\square$

**Consequence.** Halving the residual hazard costs a factor $2^{1/a}$ more verifier calls: 4× for $a=1/2$, 10× for $a=0.3$. Meanwhile each call that a clean segment must survive costs a factor $(1-f)^{-1}$ in expected attempts (Proposition 1, $s\le(1-f)^m$). Time cost is therefore geometric in $m$ while correctness gains are polynomial. The time-correctness frontier bends sharply (Figure 1c), and the useful number of re-checks is small. This mirrors the "fewer than 10 samples" result of [8] in a sequential setting.

## 5 Time Results Under Context-Age Hazard

### 5.1 Optimal interval

For small segment failure probability, Proposition 1 gives the per-step overhead
$$\frac{\mathbb E[\mathcal T]}{N}-1\approx\frac{c}{T}+(1-\mu_m)\,p_0\Big(T+\frac{\beta T^2}{2}\Big)\Big(1+\frac{mV+R}{T}\Big)+fm(\cdot),\qquad c=mV+C+K.$$
Only *detected* errors cost time, which is why the factor is $(1-\mu_m)$. Undetected errors cost correctness instead: a weaker verifier makes the run *faster* and *less correct*.

**Proposition 4 (interval scaling).** Keep the leading terms. The first-order optimal interval solves $c=(1-\mu_m)p_0T^2(1+\beta T)$. Hence $T^*\approx\sqrt{c/((1-\mu_m)p_0)}$ when $\beta T^*\ll1$ (the silent-error analogue of Young's law, with no factor 2 because a detected silent error wastes the whole segment), and $T^*\approx\big(c/((1-\mu_m)p_0\beta)\big)^{1/3}$ when $\beta T^*\gg1$.

So when context aging dominates, a 10× higher per-segment overhead justifies only about 2.2× longer segments ($10^{1/3}$), not 3.2× ($10^{1/2}$).

### 5.2 Age hazard couples the interval to correctness

Under fresh resumption, Theorem 1 gives $P_{\rm clean}=\big[\prod_{a<T}\theta_m(p(a))\big]^{N/T}\approx\exp\{-Np_0\mu_m(1+\beta(T-1)/2)\}$. Correctness now *decreases* with $T$. Short segments keep each context young. The time-optimal $T^*$ is no longer correctness-neutral, and there is a genuine Pareto frontier over $T$ (Figure 2c).

### 5.3 Fresh vs snapshot resumption

**Proposition 5.** Under snapshot resumption the committed step at progress $x$ is executed at age $x$, so $P_{\rm clean}=\prod_{x<N}\theta_m(p_0(1+\beta x))\approx\exp\{-p_0\mu_m(N+\beta N^2/2)\}$. This decays super-exponentially in $N$, whatever $T$ is. The segment pass probability $s_j$ decays with $j$, so the per-step makespan grows without bound in $N$. Under fresh resumption the per-step makespan is independent of $N$ (up to the last segment), at the price of re-orientation $K$ per attempt. There is therefore a crossover horizon $N^\times$ beyond which fresh resumption is faster. Beyond a (generally different) horizon it is also more correct.

This formalizes the intuition behind harnesses that restart agents from a progress file rather than carrying a long context [15]. It also identifies a failure mode of high-fidelity context-snapshot restoration: restoring a long context restores its hazard. The model does not capture what fresh resumption loses (information not externalized); we fold that cost into $K$ and into a possible increase of $p_0$, a limitation discussed in §8.

## 6 Experiments

All experiments run on one CPU core; the full pipeline (`code/run_all.sh`) takes about 12 minutes. The simulator (`code/simulator.py`) is written independently of the closed forms (`code/theory.py`). It samples individual steps, errors, detectabilities, false alarms and verifier coins, and executes rollbacks and checkpoints explicitly.

### 6.1 E1: closed forms vs simulation

*Source: `results/e1_validation.json`.* We compare Theorem 1 and Proposition 1 against the simulator on a full factorial of 96 configurations: {fresh, snapshot} × $\beta\in\{0,\ \beta>0\}$ ($\beta=0.05$ fresh, $0.002$ snapshot) × $T\in\{5,20\}$ × $m\in\{1,3\}$ × three detectability laws (point $r{=}0.8$; point $r{=}0.9$ with $\rho{=}0.1$; Beta(0.5,1) with $\rho{=}0.05$) × $f\in\{0,0.05\}$. Fixed values: $N=400$, $p_0=0.004$, $V=1$, $C=0.5$, $R=1$, $K=2$ (fresh). Each configuration uses 3000 runs. The largest relative makespan error is **0.32%** and the largest absolute error in $P_{\rm clean}$ is **0.021**. Of the configurations, 93.8% (makespan) and 95.8% (correctness) fall within 2 standard errors, close to the 95% expected for an exact formula. Mean z-scores are −0.23 and −0.10.

### 6.2 E2: interval invariance and the ceiling

*Source: `results/e2_e3_floor_powerlaw.json`, Figure 1a.* Setting: $N=1000$, $p=0.002$, point detectability $r=0.7$ with blind spot $\rho=0.1$, $f=0.02$. Theorem 1 predicts $P_{\rm clean}=0.4765$ for $m=1$ and $0.7795$ for $m=3$ at every $T$. Simulated values (2000 runs each) for $T\in\{1,2,5,10,25,50,100\}$ are 0.4735, 0.4655, 0.4715, 0.4695, 0.4775, 0.466, 0.4655 ($m=1$) and 0.784, 0.783, 0.761, 0.779, 0.786, 0.7895, 0.7985 ($m=3$). Every value is within about 2 SE ($\approx$0.011 and 0.009) of the prediction, and none shows a trend in $T$. Without verification the probability is 0.1351.

We also ran an *adaptive* policy. After a flag it halves $T$ and triples $m$; after two consecutive passes it doubles $T$. It reached 0.500, 0.508 and 0.534 for starting $T\in\{5,20,50\}$. All lie below the Theorem 2 ceiling of **0.8184**. A verifier that detects every non-blind error ($r=1$, same $\rho$) reached 0.8205 and 0.819 at $T=5$ and $T=50$, matching the ceiling as the theorem's tightness clause predicts.

### 6.3 E3: power-law returns and the cost frontier

*Source: `results/e2_e3_floor_powerlaw.json`, `results/e2b_replication.json`, Figure 1b–c.* For Beta(0.3,1), Beta(0.5,1) and Beta(1,1), the fitted log-log slopes of $\mu_m$ over $m\in[30,1000]$ are −0.299, −0.497 and −0.993. These match Proposition 3's $-a$. Adding a 2% blind-spot atom to Beta(0.5,1) flattens the slope to −0.375 over the same range as $\mu_m$ approaches 0.02. A homogeneous verifier with the same single-call recall drops below $10^{-6}$ by $m\approx20$.

Monte Carlo checks at $m\in\{1,4,16\}$ ($N=1000$, $p=0.002$, Beta(0.5,1), 3000 runs) gave 0.262 vs 0.263, 0.415 vs 0.443, and 0.644 vs 0.648 (simulated vs predicted). The $m=4$ value deviated by about 3 SE. Two replications with 4000 runs each gave 0.432 and 0.4365 (SE 0.008), within 1.4 SE of the prediction, so we treat the first deviation as sampling noise.

Cost frontier ($T=10$, Beta(0.5,1)): raising $m$ from 1 to 100 raises $P_{\rm clean}$ from 0.263 to 0.838. Expected makespan grows from 1158 to 11,250 time units when $f=0$, to 85,190 when $f=0.02$, and to $4.3\times10^8$ when $f=0.1$. Correctness at each $m$ is identical across $f$, as Theorem 1 says.

### 6.4 E4: context-age hazard

*Source: `results/e4_age_hazard.json`, Figure 2.*

**Interval scaling (Fig. 2a).** We used $p_0=10^{-3}$, $\mu=0.24$ ($r=0.8$, $\rho=0.05$, $m=1$), $N=2\times10^5$, and minimized the exact makespan over $T$ for overheads $c\in[0.1,100]$. Fitted slopes of $\log T^*$ against $\log c$ over the upper half of the range are 0.47 ($\beta=0$), 0.36 ($\beta=0.01$) and 0.28 ($\beta=0.1$). The first-order formula of Proposition 4 tracks the exact optimum closely at small $c$. At large $c$ it overestimates $T^*$; for $\beta=0.1$ and $c=100$ it gives 106 against an exact 81. In that regime the segment failure probability is no longer small, and the exact optimum is shorter than the first-order formula predicts. The qualitative claim (the exponent falls from about 1/2 toward 1/3 and below as aging grows) holds. The exact exponent 1/3 is a first-order asymptotic, not a measured constant.

**Fresh vs snapshot (Fig. 2b).** Parameters: $p_0=10^{-3}$, $\beta=0.01$, $V=2$, $C=1$, $R=1$, $f=0.01$, $K=5$ for fresh, each policy at its own optimal $T$. Snapshot is faster for short tasks (1.133 vs 1.222 time units per step at $N=100$) and is roughly tied at $N=500$ (1.204 vs 1.209). Fresh is faster from $N=1000$ on (1.209 vs 1.270). By $N=20{,}000$ snapshot costs 2.49 per step while fresh holds at 1.209. Correctness diverges even more: at $N=2000$, 0.515 (fresh) vs 0.0048 (snapshot); with no verification and no checkpointing it is $2.4\times10^{-10}$. A simulator spot check at $N=2000$ (1000 runs) gave 1.210 vs 1.209 time per step and 0.523 vs 0.515 correctness for fresh, and 1.372 vs 1.375 and 0.004 vs 0.0048 for snapshot.

**Coupling (Fig. 2c).** With $\beta=0.05$ and $N=5000$ under fresh resumption, $P_{\rm clean}$ falls monotonically with $T$, from 0.30 at $T=1$, while time per step is minimized at an intermediate $T=52$ (1.276 per step), where $P_{\rm clean}$ has fallen to 0.065. This is the Pareto structure of §5.2.

### 6.5 E5: detectability measured on program mutants

*Source: `results/e5_mutation.json`, `results/e5_detectabilities.npz`, Figure 3.* We want a $\mathcal D$ that was not chosen by us, so we treat errors as first-order AST mutants (operator swaps, comparison swaps, ±1 constants, boolean-return flips) of 8 small reference functions: gcd, binary search, run-length encoding, primality, merge, maximum subarray, clamped sum, leap year. This produced 102 mutants. A verifier call runs $k=2$ tests drawn without replacement from a pool of $n=200$, so a mutant that fails $\kappa$ pool tests has $d=1-\binom{n-\kappa}{2}/\binom{n}{2}$. Pool A uses typical random inputs; pool B uses edge-case-oriented inputs. Six mutants were killed by no test in either pool. On manual inspection all six are semantically equivalent on integer inputs, so they are not errors:
- `//2`→`//1` in binary search, which degrades it to a correct linear scan;
- an unreachable `<`→`<=`;
- `<=`→`<` in merge, indistinguishable for equal integers;
- `>`→`>=` in max-subarray;
- two clamp boundary swaps.

We exclude them, leaving 96 real errors.

| verifier (per call) | blind spot $\rho$ | mean $d$ | $\mu_1$ | $\mu_4$ | $\mu_{36}$ | $\mu_{621}$ |
|---|---|---|---|---|---|---|
| A: 2 tests from typical pool | **0.115** | 0.629 | 0.371 | 0.248 | 0.163 | 0.115 |
| B: 2 tests from edge-case pool | 0.021 | 0.603 | 0.397 | 0.110 | 0.022 | 0.021 |
| A+B: 2 + 2 tests | **0.000** | 0.817 | 0.183 | 0.025 | 5.7e-6 | 1.1e-62 |

Three findings bear on the theory.
1. **Blind spots are real and large for a natural verifier.** For pool A, 11.5% of real errors have $d=0$. Adding more tests from the same pool cannot change this, since $\rho$ does not depend on $k$. Re-verification therefore plateaus at 0.115, as Theorem 2 requires.
2. **Heterogeneity near zero.** In pool A, 9.4% of errors have $0<d<0.05$. Over the non-blind errors, $\mu_m$ has log-log slope −0.55 on $m\in[3,100]$, a power-law-like decay. An atom+Beta fit to pool A gives $a=0.37$, $b=0.17$, $\rho=0.088$, which underestimates the measured atom (0.115). The parametric family is only approximate. Because the pools are finite, $d\ge d_{\min}\approx0.01$ for every non-blind mutant, so all curves eventually decay geometrically (Proposition 3, last clause), as A+B shows.
3. **Diversity removes the joint blind spot.** The A+B verifier has $\rho=0$. (On the 102 mutants *before* removing equivalent ones, the joint blind mass was 0.059, against 0.013 predicted by independence, a 4.5× overlap. That overlap consisted entirely of equivalent mutants. A naive count would therefore over-state verifier correlation, and real-world blind-spot estimates need an equivalence check.)

**End to end.** We ran the simulator with errors drawn from this empirical $\mathcal D$ ($N=1000$, $p=0.002$, $T=20$, 2000 runs). Results:

| verifier | $m$ | simulated $P_{\rm clean}$ | Theorem 1 | Theorem 2 ceiling |
|---|---|---|---|---|
| A | 1 | 0.494 | 0.475 | 0.795 |
| A | 8 | 0.649 | 0.650 | 0.795 |
| A+B | 1 | 0.703 | 0.693 | 1.0 |
| A+B | 8 | 0.989 | 0.989 | 1.0 |

The A+B verifier costs twice as many tests per call as A. Even so, A+B with $m=1$ (4 tests per segment) beats A with $m=8$ (16 tests per segment). In this setting, diversity buys more than repetition at a quarter of the test budget.

### 6.6 E6: implied horizon multipliers

*Source: `results/e6_horizon_multiplier.json`, analytic.* Theorem 1 implies that the 50%-correctness horizon of a verified agent is $\ln 2/(-\ln\theta_m)$, against $\ln2/(-\ln(1-p))$ without verification. The ratio is $\approx1/\mu_m$, and the ceiling is $1/\rho$. At $p=10^{-3}$:
- $r=0.8$, $\rho=0.05$: multipliers 4.17, 17.4 and 19.99 for $m=1,3,10$ (ceiling 20).
- $r=0.95$, $\rho=0.2$: multipliers 4.17, 5.00 and 5.00.

A near-perfect but blind-spotted verifier is worth less than a mediocre verifier with a small blind spot once re-checks are allowed. Combined with constant-hazard readings of measured time horizons [5,6], this suggests that some of the horizon gains attributed to "ability to adapt to mistakes" [5] could be decomposed into changes in $p$ and changes in $\mu$. We have not tested this; §7.3 describes how one could.

## 7 Discussion

### 7.1 Design implications (conditional on the model)

1. **Measure correctness separately from throughput.** Checkpoint interval and rollback machinery are time levers (Theorem 1). Reporting only task completion time, or tuning recovery against it, can push toward *weaker* verification, because undetected errors cost no time (§5.1).
2. **Estimate the joint blind spot $\rho$, not just recall.** A single recall number hides the quantity that bounds long-run reliability (Theorem 2). Spend verification budget on *orthogonal* verifiers (different test generators, different judge families, execution-based checks next to model-based ones) before spending it on repetition. Correlated LLM errors [11] and self-correction blind spots [10] make same-model self-review a particularly poor diversifier.
3. **Expect polynomial returns from re-asking a judge.** If an LLM judge's per-error detection probabilities are spread toward 0, $k$ re-samples reduce missed errors only like $k^{-a}$ (Proposition 3). False alarms make the cost geometric, so a small $m$ is optimal.
4. **Prefer fresh-context resumption for long tasks when the hazard grows with age.** Restoring a long context restores its hazard (Proposition 5). Externalized state plus a fresh context bounds per-step cost and slows the correctness decay.

### 7.2 Relation to the HPC literature

When $\rho=0$, $\mathcal D=\delta_r$ and $\beta=0$, our time analysis reduces to known silent-error patterns with partial verification [3,4]. What does not carry over is the role of the guaranteed verification. In HPC it makes the correctness objective trivial. For agents it does not exist, and correctness becomes governed by the invariants of §4. The age-dependent hazard and the fresh/snapshot distinction have no direct HPC counterpart. In HPC, restarting a process does not change the failure rate of the hardware.

### 7.3 A pre-registered protocol for LLM agents

We did not run LLM agents (no API or GPU was available). The predictions can be tested as follows.

- **P1 (interval invariance).** On a long, decomposable task with a programmatic ground truth (for example, a multi-file refactor with a hidden full test suite, or a long arithmetic or state-tracking execution of the kind in [7]), run the agent with a fixed verifier (visible tests or an LLM judge) at checkpoint intervals $T\in\{1,4,16,64\}$ steps, with fresh resumption and short segments so aging is negligible. Prediction: hidden-suite correctness does not vary with $T$ beyond sampling error; wall-clock time does. Falsified if correctness varies monotonically with $T$ (by more than 2 SE at $n\ge200$ episodes per arm) when $\beta\approx0$.
- **P2 (ceiling).** Estimate per-error detectability by injecting known errors (as in [10]) and querying the verifier $m$ times. Estimate $\hat\rho$ as the fraction of injected errors never flagged in $m_{\max}$ calls, after an equivalence check. Prediction: no recovery policy that uses only that verifier exceeds $e^{-N\hat p\hat\rho}$.
- **P3 (power law).** The empirical $\hat\mu_m$ for an LLM judge decays polynomially with an exponent $a<1$, rather than as $(1-\hat r)^m$.
- **P4 (diversity).** Adding a judge from a different model family lowers $\hat\rho_{\rm joint}$ by less than independence predicts [11], but more than doubling calls to the first judge does.
- **P5 (resumption).** On tasks spanning several context windows, fresh-context resumption from a progress file gives higher correctness than context-snapshot restoration once $N$ exceeds a crossover, and this crossover moves earlier for models with stronger measured self-conditioning [7].

## 8 Limitations

- **Independence assumptions.** Errors occur independently across steps, and detection events are independent across calls given $d$. Real agents show self-conditioning [7], meaning errors make further errors likelier, and judges may be deterministic or correlated across calls. Theorem 2's proof needs only that blind errors do not influence verifier outputs and that the hazard is not driven by blind errors. Theorem 1's exact product form needs per-step independence. Error cascades would make long segments worse than our model predicts.
- **Persistent, binary errors.** We treat corruption as binary and permanent. Real agents sometimes repair errors incidentally, and some errors are harmless. Equivalent mutants (§6.5) are an instance of the latter.
- **Rollback is assumed perfect and side-effect free.** We ignore irreversible external actions. Studies of real rollback report re-issued committed actions and broken security dependencies [20,22]. In such settings rollback has its own failure modes outside our model.
- **Fresh resumption is idealized.** The cost of losing un-externalized information is folded into a constant $K$. In practice it may raise $p_0$ after resumption, which would move the crossover of §5.3.
- **The detectability measurement is a toy.** 96 non-equivalent mutants of 8 small functions, with test-sampling verifiers, is far from LLM agents and LLM judges. It shows that blind spots and near-zero detectability arise naturally, and that the theory's predictions hold with a $\mathcal D$ we did not design. It does not estimate $\rho$ for any real agent. The two test pools were written by us. Equivalence of the 6 excluded mutants was judged by manual inspection, not proof.
- **Asymptotics.** The $c^{1/3}$ law is first order. The exact optima in E4 deviate when segment failure probability is not small.
- **No LLM experiments.** All agent-level claims are model predictions. The protocol in §7.3 is how they should be tested.

## 9 Conclusion

For long-running agents, error recovery is limited less by *how* work is checkpointed and rolled back than by *what the verifiers can see*. Under constant hazard, checkpoint scheduling cannot change end-to-end correctness (Theorem 1). No verifier-driven recovery policy can beat the ceiling set by the verifiers' joint blind spot (Theorem 2). Re-asking the same verifier gives power-law returns at geometric cost (Proposition 3). When error rates grow with context age, the time-optimal interval shortens toward a cube-root law, and resuming from externalized state with a fresh context wins on long tasks (Propositions 4–5). The simulations and a mutation-based measurement agree with these results. In the toy setting, a diverse pair of cheap verifiers lifted 1000-step correctness from 0.49 to 0.99, while repeating one verifier eight times reached only 0.65. Whether the same is true of LLM agents and LLM judges is an empirical question, and §7.3 gives the protocol to answer it.

## References

[1] J. W. Young. A first order approximation to the optimum checkpoint interval. *Communications of the ACM*, 17(9):530–531, 1974.

[2] J. T. Daly. A higher order estimate of the optimum checkpoint interval for restart dumps. *Future Generation Computer Systems*, 22(3):303–312, 2006.

[3] A. Benoit, A. Cavelan, Y. Robert, H. Sun. Optimal resilience patterns to cope with fail-stop and silent errors. *IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2016.

[4] A. Benoit, S. K. Raina, Y. Robert. Efficient checkpoint/verification patterns. *International Journal of High Performance Computing Applications*, 2015.

[5] T. Kwa, B. West, et al. Measuring AI ability to complete long tasks. arXiv:2503.14499, 2025 (v3, 2026, retitled "Measuring AI Ability to Complete Long Software Tasks").

[6] T. Ord. Is there a half-life for the success rates of AI agents? arXiv:2505.05115, 2025.

[7] A. Sinha, A. Arun, S. Goel, S. Staab, J. Geiping. The illusion of diminishing returns: Measuring long horizon execution in LLMs. arXiv:2509.09677, 2025.

[8] B. Stroebl, S. Kapoor, A. Narayanan. Inference scaling fLaws: The limits of LLM resampling with imperfect verifiers. arXiv:2411.17501, 2024 (published at ICLR 2026 as "The Limits of Inference Scaling Through Resampling").

[9] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, D. Zhou. Large language models cannot self-correct reasoning yet. *ICLR*, 2024.

[10] K. Tsui. Self-Correction Bench: Revealing and addressing the self-correction blind spot in LLMs. arXiv:2507.02778, 2025.

[11] E. M. Kim, A. Garg, K. Peng, N. Garg. Correlated errors in large language models. *ICML*, PMLR 267:30038–30066, 2025.

[12] M. Cemri, M. Z. Pan, S. Yang, et al. Why do multi-agent LLM systems fail? arXiv:2503.13657, 2025 (NeurIPS 2025 Datasets and Benchmarks).

[13] A. Backlund, L. Petersson. Vending-Bench: A benchmark for long-term coherence of autonomous agents. arXiv:2502.15840, 2025.

[14] K. Hong, A. Troynikov, J. Huber. Context rot: How increasing input tokens impacts LLM performance. Chroma technical report, July 2025. https://research.trychroma.com/context-rot

[15] Anthropic. Effective harnesses for long-running agents. Anthropic Engineering blog, November 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

[16] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, S. Yao. Reflexion: Language agents with verbal reinforcement learning. *NeurIPS*, 2023.

[17] A. Zhou, K. Yan, M. Shlapentokh-Rothman, H. Wang, Y.-X. Wang. Language agent tree search unifies reasoning, acting, and planning in language models. *ICML*, PMLR 235:62138–62160, 2024.

[18] Y. Zhuang, K. Chen, Y. Duan, S. Zheng, J. Li, X.-Y. Zhang. AgentRewind: Recoverable execution for long-horizon LLM agents. arXiv:2608.14380, 2026.

[19] Rollback the world, keep the reflection: Rollback-induced reflection for long-horizon LLM agents. arXiv:2609.18304, 2026. (Author list not retrieved; cited by title and identifier.)

[20] Dependable long-horizon LLM agents: A trajectory-centered empirical study of where errors enter and which controls reach them. arXiv:2611.13254, 2026. (Author list not retrieved; cited by title and identifier.)

[21] Y. Jia, M. Harman. An analysis and survey of the development of mutation testing. *IEEE Transactions on Software Engineering*, 37(5):649–678, 2011.

[22] Safe to resume? Breaking execution continuity of agent execution via rollback. arXiv:2608.29381, 2026. (Author list not retrieved; cited by title and identifier.)

---

### Figures

- **Figure 1** (`results/fig_floor_powerlaw.png`): (a) simulated $P_{\rm clean}$ vs $T$, with the Theorem 1 predictions, adaptive policies, the $r=1$ verifier and the Theorem 2 ceiling; (b) $\mu_m$ for Beta and homogeneous verifiers; (c) time–correctness frontier over $m$ for three false-alarm rates.
- **Figure 2** (`results/fig_age_hazard.png`): (a) exact optimal $T^*$ vs overhead $c$ (points) and the first-order formula (lines); (b) per-step time under fresh vs snapshot resumption vs $N$; (c) Pareto set over $T$ under age hazard.
- **Figure 3** (`results/fig_mutation.png`): (a) histogram of per-call detectability for 96 non-equivalent mutants under pools A and B; (b) re-verification curves for A, B, A+B, a homogeneous comparator and the atom+Beta fit.

### Reproducibility

`cd code && ./run_all.sh` regenerates every number and figure (about 12 CPU-minutes; fixed seeds). See `code/README.md`.
