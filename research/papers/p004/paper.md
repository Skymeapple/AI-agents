# Information, Learnability, Verification: Three Ceilings on How Far Synthetic Data Can Stretch the Human Data Stock

*Target venue: International Conference on Learning Representations (ICLR 2027), main track (theory plus controlled experiments)*

---

## Abstract

Forecasts put full use of the stock of high-quality public human text at somewhere between 2026 and 2032 (Villalobos et al., 2024). Synthetic data is the most common answer to what happens next. The model-collapse literature, though, shows that naive recursive training degrades models. The verification literature shows that filtering can prevent this. Neither says how much human data a given synthetic pipeline is worth. We give an accounting framework that answers this through a **data multiplier** $M$: the factor by which a pipeline increases the human-equivalent sample size. We separate three ceilings on $M$. (i) **Information ceiling.** If the synthetic data depends on the target only through the human corpus, it cannot reduce Bayes risk (Prop. 1), and any gain is a gain in learnability. (ii) **Learnability ceiling.** In a Zipfian exposure model of factual knowledge, faithful source-grounded paraphrasing $r$ times is *exactly* equivalent to a human corpus $M(r)=(1-(1-s)^{1+r})/s$ times larger, for every frequency distribution and corpus size. Here $s$ is the learner's per-exposure retention, so $M\le 1/s$ (Prop. 2). A generator with retention $s_g$ caps $M$ at $1+s_g(1-s)/s$, and a learner re-reading its own memories gets $M=1$. (iii) **Verification ceiling.** Verifier-filtered self-generation adds at most $\sum_j I(C_j;V_j)$ bits, where $I(C;V)$ is the mutual information between a sample's correctness and the verifier's accept bit (Thm. 2). Under a plurality readout, a fixed generator's accepted stream is correct only if $aq>b\,w_{\max}$ (Prop. 3). Self-training on an accumulating pool is a generalized Pólya urn whose error decays as $t^{-(1-b/a)}$ in the number of proposals $t$ (Prop. 4, mean-field). We check each result numerically. The closed forms match Monte Carlo to within 0.002 accuracy, and the fitted urn exponents are 1.00, 0.75, 0.50 and 0.50 against predictions of 1, 0.75, 0.5 and 0.5. In 144 self-training runs of a logistic-regression learner on a structured "skill" task and an unstructured "knowledge" task, the multiplier goes from 19.7× with a perfect verifier to about 1× with an uninformative one. Accuracy gain tracks verifier information (Spearman $\rho=0.89$ and $0.83$). Unverified self-training *loses* accuracy on the knowledge task (0.892→0.750). Accumulating data is not uniformly safer than replacing it. The implication is that synthetic data stretches the human stock by a bounded factor set by learner inefficiency for knowledge, and by verifier information for skills. A multiplier of 10 delays the data horizon by only about 3 years at current scaling rates.

## 中文摘要

据预测，高质量公开人类文本存量将在2026至2032年间被用尽（Villalobos 等，2024）。合成数据被普遍视为接续方案，但"模型崩溃"研究表明朴素的递归训练会退化，"验证"研究表明过滤可以阻止退化。两类工作都没有回答一个核心问题：一条合成数据流水线究竟相当于多少人类数据？本文提出一个信息核算框架，以**数据乘数** $M$（合成流水线使"人类等效样本量"扩大的倍数）为核心量，并区分三类上限。（一）**信息上限**：若合成数据只经由人类语料依赖于目标，则它无法降低贝叶斯风险（命题1），任何收益都只是"可学性"收益。（二）**可学性上限**：在描述事实知识的齐普夫曝光模型中，忠实的、以源文档为依据的 $r$ 倍改写，**严格等价于**把人类语料扩大 $M(r)=(1-(1-s)^{1+r})/s$ 倍，且对任意频率分布与语料规模都成立；其中 $s$ 为学习者对每次曝光的保持率，因此 $M\le 1/s$（命题2）。保持率为 $s_g$ 的生成器把 $M$ 限制在 $1+s_g(1-s)/s$；学习者复述自身记忆则 $M=1$。（三）**验证上限**：经验证器过滤的自生成数据至多提供 $\sum_j I(C_j;V_j)$ 比特的新信息，其中 $I(C;V)$ 为样本正确性与验证器接受信号之间的互信息（定理2）。在多数（众数）读出下，固定生成器被接受的数据流正确的条件是 $aq>b\,w_{\max}$（命题3）。在累积数据池上的自训练是一个广义 Pólya 罐子模型，其错误率随提议数 $t$ 按 $t^{-(1-b/a)}$ 衰减（命题4，平均场结果）。数值验证表明：闭式解与蒙特卡洛的准确率差异在0.002以内；罐子模型拟合指数为1.00、0.75、0.50、0.50，预测值为1、0.75、0.5、0.5。在结构化"技能"任务与无结构"知识"任务上共144次逻辑回归自训练实验中，数据乘数从完美验证器下的19.7倍降至无信息验证器下的约1倍；准确率增益与验证器信息量高度相关（Spearman $\rho=0.89$ 与 $0.83$）。未经验证的自训练使知识任务准确率**下降**（0.892→0.750），而"累积"策略也并非总比"替换"更安全。结论是：合成数据只能把人类数据存量放大有限倍数，对知识类能力由学习者效率决定，对技能类能力由验证器信息决定；按当前扩展速度，10倍乘数仅能把数据枯竭时间推迟约3年。

---

## 1 Introduction

Frontier language models have been trained on ever larger human corpora. Villalobos et al. (2024) estimate the effective stock of quality-adjusted public human text at about 300T tokens (90% CI 100T–1000T). They project that compute-optimal training runs will use datasets of roughly that size between 2026 and 2032, and earlier under heavy over-training. Training compute for frontier models has grown 4–5× per year (Sevilla & Roldán, 2024), and compute-optimal dataset size grows roughly as the square root of compute (Hoffmann et al., 2022). The question is therefore near-term: **can synthetic data take over once human data runs out?**

The literature points in two directions. The *collapse* line shows that models trained recursively on their own outputs lose the tails of the distribution and degrade (Shumailov et al., 2024; Alemohammad et al., 2024). It also shows that synthetic data changes the scaling laws themselves (Dohmatob et al., 2024). The *rescue* line shows that keeping the real data and accumulating synthetic data bounds the damage (Gerstgrasser et al., 2024; Kazdan et al., 2025; Bertrand et al., 2024). It also shows that verification or selection lets synthetic data improve models (Feng et al., 2025; Yi et al., 2025), and that self-improvement is governed by a generation–verification gap (Song et al., 2025) or a "sharpening" mechanism (Huang et al., 2025). In practice, rephrased and synthetic corpora have improved data efficiency (Maini et al., 2024; Yang et al., 2025; Gunasekar et al., 2023). Bootstrapped reasoning with answer checking works in practice too (Zelikman et al., 2022; Gulcehre et al., 2023).

None of these results gives a planner what the "running out of data" question actually requires: **an exchange rate**. How many human tokens is a synthetic pipeline worth, what limits that number, and does the limit depend on the kind of capability? Without such an accounting, "synthetic data will fill the gap" cannot be checked, and "synthetic data causes collapse" cannot be acted on.

**This paper.** We define the *data multiplier* $M$ of a synthetic pipeline as the ratio of the human-equivalent sample size it reaches to the human sample size it started with. We then bound and compute $M$ in models simple enough to solve exactly and rich enough to separate the mechanisms. The central conceptual point is that a synthetic pipeline can deliver value through three channels with different ceilings:

1. **Information**: new facts about the target. This is impossible without an external signal.
2. **Learnability**: making information already in the corpus easier for an imperfect learner to absorb. This is bounded by how inefficient the learner is.
3. **Verification**: injecting new information through a check that is cheaper than producing the answer. This is bounded by the verifier's mutual information with correctness.

**Contributions.**

- **C1 (framework and information ceiling).** We formalize the data multiplier. We state the garbling bound (Prop. 1): synthetic data that depends on the target only through the human corpus cannot lower Bayes risk, so $M_{\text{Bayes}}=1$. Every reported gain from such data is therefore a learnability gain.
- **C2 (an exact amplification identity).** In a Zipfian exposure model of factual knowledge, $r$-fold faithful source-grounded paraphrasing is *exactly* equivalent to multiplying the human corpus by $M(r)=(1-(1-s)^{1+r})/s$. This holds for any frequency distribution and any corpus size (Prop. 2). It yields the learnability ceiling $1/s$, a generator-retention ceiling $1+s_g(1-s)/s$, and $M=1$ for self-recall. We check it against Monte Carlo and characterize the effect of paraphrase fidelity, including a sharp failure at fidelity $1/2$ when errors are systematic.
- **C3 (verification ceiling and dynamics).** We bound the new information from verified self-generation by $\sum_j I(C_j;V_j)$ (Thm. 2). We give a plurality condition $aq>b\,w_{\max}$ for a fixed generator (Prop. 3). We derive a power-law error decay $t^{-(1-b/a)}$ for accumulate-mode self-training as a generalized Pólya urn (Prop. 4), and confirm the exponent by simulation.
- **C4 (controlled experiments).** We run 144 self-training runs with a real learner on structured (skill) and unstructured (knowledge) tasks. We measure $M$ directly through learning-curve inversion, show that gain tracks verifier information, and document two departures from common heuristics. Unverified self-training is harmful rather than neutral for knowledge, and *replace* can beat *accumulate* when the verifier is imperfect.
- **C5 (horizon accounting and falsifiable predictions).** We convert multipliers into delays of the human-data horizon, $\Delta t=\ln M/\ln g_D$. We state predictions and a test protocol for large models (§6.1) that could falsify the framework.

We do not claim that synthetic data cannot sustain progress. We claim that its contribution can be measured in human-data units, and that the measurement is bounded by quantities we name: learner retention, generator retention, paraphrase fidelity and verifier information. Where those quantities are large the bounds are loose; where they are small the bounds bind.

## 2 Related Work

**Data stock and scaling.** Villalobos et al. (2024) estimate the stock of human public text and the timing of its exhaustion. They note synthetic data, multimodal transfer and data efficiency as possible ways out. Hoffmann et al. (2022) establish compute-optimal scaling with tokens and parameters scaled together. Muennighoff et al. (2023) show that up to about four epochs of repetition cost little loss, and they model the decaying value of repeated tokens. Our multiplier $M$ is the synthetic-data analogue of their "effective data" for repetition. Our learnability ceiling also gives a mechanism for why repetition and paraphrase eventually saturate.

**Model collapse.** Shumailov et al. (2024) show that indiscriminate recursive training causes irreversible loss of distributional tails. Alemohammad et al. (2024) show self-consuming loops degrade without enough fresh real data. Bertrand et al. (2024) prove stability of iterative retraining when enough real data is kept. Dohmatob et al. (2024) show how synthetic data truncates the tails and changes scaling laws. Gerstgrasser et al. (2024) show that accumulating rather than replacing data bounds test error in linear models and avoids collapse empirically. Kazdan et al. (2025) compare replace and accumulate regimes more broadly and find that small amounts of synthetic data can help when real data is scarce. Our E2 results add a qualification: with an *imperfect verifier*, accumulating keeps early low-precision synthetic data, and replacing can be better (§5.2).

**Verification and self-improvement.** Feng et al. (2025) argue that scaling with synthesized data requires verification, and show that imperfect verifiers can still prevent collapse. Yi et al. (2025) analyze verifier-filtered retraining in linear regression and VAEs and find near-term gains that eventually plateau. Song et al. (2025) formalize self-improvement through a generation–verification gap and measure it across models. Huang et al. (2025) model self-improvement as *sharpening*: the model acts as its own verifier and amortizes inference-time selection. They note explicitly that this cannot create new information. Zelikman et al. (2022, STaR) and Gulcehre et al. (2023, ReST) are practical bootstrapping loops built on answer correctness or reward. We add an *information budget* (Thm. 2) that turns verifier error rates $(a,b)$ into an upper bound on new information. We also add an urn-dynamics law whose exponent depends only on $b/a$.

**Synthetic data for efficiency and knowledge.** Phi-1 trained a code model largely on filtered web data plus synthetic textbooks (Gunasekar et al., 2023). WRAP rephrases web documents and reports about 3× faster pre-training (Maini et al., 2024). EntiGraph's synthetic continued pre-training turns a small corpus into a much larger synthetic one and improves closed-book QA (Yang et al., 2025). Kandpal et al. (2023) show that LLM factual accuracy depends strongly on how many relevant pre-training documents a fact has. Allen-Zhu & Li (2025) report that reaching their measured knowledge capacity requires many exposures per fact. Our Prop. 2 formalizes why rephrasing helps knowledge: it closes the gap between *seen* and *stored*. It also shows the limit: it cannot reach facts the source never contained.

## 3 Framework: Data Multipliers and the Information Ceiling

**Setup.** A target $\theta$ (the world, or the true function) generates a human corpus $D_h\sim P_\theta^{\,n}$. A synthetic pipeline produces $D_s\sim Q(\cdot\mid D_h, U, V)$. Here $U$ is independent randomness (prompts, sampling, the generator's own training noise) and $V$ is an optional *external signal*: verifier outputs, environment feedback, retrieval. A learner $\mathcal A$ maps data to a predictor with risk $R_{\mathcal A}(\cdot)$.

**Definition (data multiplier).** Let $\rho_{\mathcal A}(n)$ be the expected risk of $\mathcal A$ trained on $n$ human samples. Assume it is decreasing. The human-equivalent sample size of a pipeline is $N_{\text{eff}}=\rho_{\mathcal A}^{-1}\big(\mathbb E\,R_{\mathcal A}(D_h\cup D_s)\big)$, and the **data multiplier** is $M=N_{\text{eff}}/n$. $M$ depends on the learner. That dependence is the point, as Prop. 2 shows.

**Proposition 1 (information ceiling: garbling).** If $V$ is absent, so that $\theta\to D_h\to D_s$ is a Markov chain, then $I(\theta;D_h,D_s)=I(\theta;D_h)$. For any loss, the Bayes risk achievable from $(D_h,D_s)$ equals that achievable from $D_h$. Hence the multiplier of the Bayes-optimal learner is $M_{\text{Bayes}}=1$.

*Proof.* The mutual-information statement is the data processing inequality (Cover & Thomas, 2006, §2.8) together with the chain rule: $I(\theta;D_h,D_s)=I(\theta;D_h)+I(\theta;D_s\mid D_h)$, and the second term is zero by the Markov property. For risk, any decision rule $\delta(D_h,D_s)$ is matched by the randomized rule $\delta'(D_h)=\delta(D_h,\tilde D_s)$ with $\tilde D_s\sim Q(\cdot\mid D_h,U)$ drawn internally. That rule has the same joint law with $\theta$ and hence the same risk. Conversely $D_h$-measurable rules are a subset. $\square$

Prop. 1 is elementary, but it organizes everything that follows. Without an external signal, **every** gain from synthetic data is a gain in *learnability*. It moves a non-Bayes learner toward what the human corpus already supported, so it is bounded by the gap $\rho_{\mathcal A}(n)-\rho_{\text{Bayes}}(n)$. This is consistent with Huang et al.'s (2025) observation that sharpening "cannot create new information." Prop. 1 states it as a ceiling on $M$ for any learner that is already efficient.

## 4 Theory: Learnability and Verification Ceilings

### 4.1 A Zipfian exposure model of knowledge

There are $K$ facts. Fact $i$ is queried with frequency $p_i$ (any distribution; we use Zipf with exponent $\alpha$ in experiments) and has an answer among $k$ options. A human corpus of $N$ samples gives fact $i$ a count $c_i\sim\text{Poisson}(Np_i)$, independently across facts. The learner **stores** each exposure independently with probability $s\in(0,1]$, its *per-exposure retention*. A stored fact is answered correctly, and an unstored fact is guessed (accuracy $1/k$). This abstracts the empirical regularity that factual accuracy rises with the number of supporting documents (Kandpal et al., 2023) and that storage takes many exposures (Allen-Zhu & Li, 2025). Small $s$ means a sample-inefficient learner. Accuracy is

$$A(N;s)=1-\Big(1-\tfrac1k\Big)\sum_i p_i\,e^{-sNp_i},$$

since the number of storing exposures is a thinned Poisson with mean $sNp_i$. The **coverage ceiling** $A(N;1)$ is the accuracy of a learner that stores everything it sees once. By Prop. 1, no source-only synthetic pipeline can exceed it for facts, because unseen facts carry no information in $D_h$.

**Synthetic amplification.** A generator re-expresses every human exposure $r$ times, for example by paraphrasing or by placing the fact in new contexts as in WRAP or EntiGraph. Each paraphrase is an independent exposure for the learner. The generator itself retains a given human exposure with probability $s_g$. Source-grounded generation, with the document in context, has $s_g=1$.

**Proposition 2 (amplification identity and learnability ceiling).** In this model, for every $(p_i)$, $N$, $K$ and $k$, the amplified corpus gives exactly the accuracy of a plain human corpus of size $M\cdot N$ with

$$M(r;s,s_g)=\frac{1-(1-s)\big[(1-s_g)+s_g(1-s)^{r}\big]}{s}\;\xrightarrow[r\to\infty]{}\;1+\frac{s_g(1-s)}{s}.$$

For source-grounded generation ($s_g=1$) this is $M(r)=(1-(1-s)^{1+r})/s\le 1/s$. If the generator is the learner re-expressing its *own* stored facts, then $M=1$.

*Proof.* A given human exposure fails to produce a stored copy if the learner misses the original (probability $1-s$) and either the generator dropped it ($1-s_g$) or the learner missed all $r$ paraphrases ($s_g(1-s)^r$). Call this failure probability $f$. By Poisson thinning, the number of exposures of fact $i$ that lead to storage is $\text{Poisson}(Np_i(1-f))$. So $\Pr[\text{fact }i\text{ unstored}]=e^{-Np_i(1-f)}$, and this equals the plain-corpus value $e^{-sN'p_i}$ exactly when $N'=N(1-f)/s$, for every $i$ simultaneously. The limit follows from $(1-s)^r\to0$. For self-recall, the facts the learner can re-express are exactly the facts already stored, which are already answered correctly, so accuracy is unchanged. $\square$

Prop. 2 has three consequences. (i) **The multiplier is a property of the learner, not of the data distribution.** It does not depend on $\alpha$ or $N$. Rephrasing should help sample-inefficient learners (small $s$) a lot and efficient learners little. (ii) **A teacher caps the student.** With $s_g<1$ the ceiling is $1+s_g(1-s)/s$. An independent model of the same quality ($s_g=s$) still gives $M\to2-s$ through pooling of independent retention, whereas a learner re-reading its own memories gives exactly $M=1$. (iii) **No pipeline in this class reaches unseen facts.** The ceiling is coverage, not truth.

**Fidelity.** Suppose a fraction $1-\phi$ of paraphrases carries a single systematic wrong answer, and the learner answers with the label it has stored most often. The correct label then wins asymptotically if and only if $\phi>1/2$. This is a special case of Prop. 3 below with $a=b=1$ and $w_{\max}=1-\phi$. We quantify the finite-$r$ behaviour numerically in §5.1.

### 4.2 Verification: an information budget

Next, let the pipeline query an external verifier. For proposal $j$ (a prompt $x_j$ and a candidate answer $\hat y_j$ drawn from the current model), let $C_j=\mathbb 1[\hat y_j\text{ correct under }\theta]$. The verifier returns $V_j\in\{0,1\}$ with $\Pr[V_j=1\mid C_j=1]=a$ (true-positive rate) and $\Pr[V_j=1\mid C_j=0]=b$ (false-positive rate). Its randomness is independent of everything else given $C_j$.

**Theorem 2 (verification ceiling).** Let $\mathcal F_{j}$ denote everything observed before proposal $j$. Then

$$I(\theta;D_h,D_s,V_{1:m})\;\le\;I(\theta;D_h)+\sum_{j=1}^{m}I(C_j;V_j\mid\mathcal F_j,x_j,\hat y_j)\;\le\;I(\theta;D_h)+\sum_{j=1}^m \big[h(aq_j+b(1-q_j))-q_jh(a)-(1-q_j)h(b)\big],$$

where $q_j=\Pr[C_j=1\mid\mathcal F_j,x_j,\hat y_j]$ and $h$ is the binary entropy. In particular, when $a=b$ the verifier adds no information and Prop. 1 applies. Each proposal adds at most one bit.

*Proof.* By the chain rule, $I(\theta;D_h,D_s,V_{1:m})=I(\theta;D_h)+\sum_j I(\theta;x_j,\hat y_j,V_j\mid\mathcal F_j)$. Given $\mathcal F_j$, the proposal $(x_j,\hat y_j)$ is generated without reference to $\theta$, so $I(\theta;x_j,\hat y_j\mid\mathcal F_j)=0$. What remains is $I(\theta;V_j\mid\mathcal F_j,x_j,\hat y_j)$. Conditioned on the proposal, $V_j$ depends on $\theta$ only through $C_j$, so the Markov chain $\theta\to C_j\to V_j$ and the data processing inequality bound it by $I(C_j;V_j\mid\cdot)$. That quantity is the mutual information of a binary asymmetric channel with input prior $q_j$, which gives the explicit expression. $\square$

Thm. 2 turns the qualitative claim "verification is needed" (Feng et al., 2025) into a budget. Two corollaries matter for the data-exhaustion question. **(a) Knowledge is expensive to verify.** For a fact absent from the corpus, the model's guess is correct with $q\approx1/k$. Even a perfect verifier then supplies only $h(1/k)$ bits per proposal, while the fact itself is worth $\log_2 k$ bits. Worse, a verifier with low false-positive rate on facts must itself contain the facts, so it is a repackaged source of human (or retrieved) data. **(b) Skills can be cheap to verify.** When checking is easier than producing (unit tests, proof checkers, arithmetic), $q$ is moderate, $b$ can be near 0, and each proposal can carry close to $h(q)$ bits without needing human answers. The asymmetry between knowledge and skills therefore comes from *the cost of a low-$b$ verifier*, not from the learner.

**Proposition 3 (plurality condition for a fixed generator).** Fix a prompt. A generator outputs the correct label with probability $q$, and wrong label $y$ with probability $w_y$, where $\sum_y w_y=1-q$ and $w_{\max}=\max_y w_y$. Filter its outputs through an $(a,b)$ verifier. As the number of proposals grows, the correct label is almost surely the plurality label of the accepted stream if $aq>b\,w_{\max}$, and almost surely not if $aq<b\,w_{\max}$.

*Proof.* Accepted counts per label are sums of i.i.d. Bernoulli variables with means $aq$ (correct) and $b w_y$ (wrong $y$). Apply the strong law of large numbers to each count. $\square$

Prop. 3 contains the fidelity threshold of §4.1 ($a=b=1$, $q=\phi$, $w_{\max}=1-\phi$). It also shows that systematic errors (large $w_{\max}$) are much more dangerous than diffuse ones. For a uniformly guessing generator on $k$ classes, $w_{\max}=(1-q)/(k-1)$, so even a weak verifier satisfies the condition.

**Proposition 4 (verified self-training as a generalized Pólya urn; mean-field).** For one prompt, let the pool hold $C$ correct and $W$ wrong labels. Each step the model samples a label in proportion to the pool (a tabular maximum-likelihood learner sampling at temperature 1), the verifier filters it, and accepted labels are added to the pool (accumulate). In the mean-field limit, $\dot C=aC/T$ and $\dot W=bW/T$ with $T=C+W$, so $\tfrac{d}{dt}\ln(C/W)=(a-b)/T$. If $a>b$, then $C/T\to1$, $T\sim at$, and

$$\frac{W}{T}\;\asymp\;t^{-(1-b/a)}.$$

If $a=b$, the expected correct share is a martingale and does not improve, which is the classical Pólya urn.

*Derivation.* Integrate $\tfrac{d}{dt}\ln(C/W)=(a-b)/T$ with $T\approx at$ to obtain $\ln(C/W)\approx\tfrac{a-b}{a}\ln t+\text{const}$. Then $W/T\approx W/C$. $\square$ This is a mean-field argument, not a rigorous almost-sure statement. We check it by simulation in §5.3.

Prop. 4 gives a **scaling law for verified self-training**: the error falls as a power of the synthetic volume, with an exponent set entirely by the verifier ratio $b/a$. With $b=0$ the exponent is 1, the same rate as fresh labelled data on a single item. With $b/a=1/2$ the exponent is 1/2, so reaching the same error takes quadratically more proposals.

## 5 Experiments

All code is in `code/` and all outputs in `results/`. Every number in this section was produced by those scripts in this environment, on CPU. Seeds are fixed.

### 5.1 E1: Amplification identity and fidelity (knowledge model)

**Setup.** $K=10^5$ facts, $k=10$, Zipf $\alpha=1.1$, $N=2\times10^5$ human samples. We compare the closed form of Prop. 2 with a Monte Carlo simulation that samples every exposure and storage event (3 replicates). We recover $M$ independently by numerically inverting $A(\cdot;s)$ ("N-equivalent").

**Results** (`results/e1_knowledge.json`, Fig. 1).

- *Identity.* For $s\in\{0.05,0.2,0.5\}$ and $r\in\{0,1,4,16,64\}$ (15 cells), the closed-form accuracy matches Monte Carlo to within 0.0019. The largest discrepancy, at $s=0.05,r=0$ (0.7818 vs 0.7836), is within 1.4 MC standard deviations. The N-equivalent multiplier equals $M(r)$ to about $10^{-9}$ relative error. For example, $s=0.05$, $r=64$ gives $M=19.29$, approaching the ceiling $1/s=20$, and $s=0.2$ saturates at $M=5.00$.
- *Invariance.* At $s=0.1$, $r=16$, $M=8.332$ for every $\alpha\in\{0.8,1.1,1.5\}$ and $N\in\{10^4,10^6\}$, as Prop. 2 predicts.
- *Ceilings.* At $s=0.1$, plain accuracy is 0.818 and the coverage ceiling is 0.923. Amplification closes that gap and stops there. The ceiling is set by unseen facts.
- *Generator retention* (learner $s=0.1$). The ceilings $1+s_g(1-s)/s$ are reached at $r=256$: $M=1.90,\,3.70,\,6.40,\,10.0$ for $s_g=0.1,0.3,0.6,1.0$.
- *Fidelity* (systematic hallucination, majority readout, 4 MC replicates). With $\phi=0.9$, $M$ rises from 1.76 ($r=1$) to 10.05 ($r=256$). With $\phi=0.7$, it rises from 1.39 to 9.74. Below-ceiling fidelity *slows* amplification without lowering the asymptote, as long as $\phi>1/2$. At $\phi=0.5$ accuracy *falls* from 0.819 to 0.688 as $r$ grows, giving $M=0.10$. At $\phi=0.4$ it collapses to 0.019 at $r=256$, below the 0.1 guessing rate, because stored systematic errors outvote the truth. This is the threshold predicted by Prop. 3.

### 5.2 E2: Verified self-training with a real learner (skill vs knowledge)

**Tasks.** *Skill*: $x\in\{\pm1\}^{20}$, $y=\arg\max(W^\star x+b^\star)$ over 8 classes, so the answer has learnable structure. *Knowledge*: $x$ is a one-hot item among 400, with labels drawn uniformly at random from 8 classes. There is no structure, and human prompts follow Zipf(1.1) over items. **Learner**: L2-regularized multinomial logistic regression trained to convergence (L-BFGS). **Loop**: 8 rounds. Each round draws 2,000 fresh prompts (prompts are cheap, answers are scarce), samples labels from the model's own softmax, filters with an $(a,b)$ verifier, and adds the accepted pairs to the pool (accumulate) or swaps them for the previous round's synthetic data (replace). The model is then retrained from scratch on human plus synthetic data. **Metrics**: test accuracy (Zipf-weighted for knowledge); $N_{\text{eff}}$ from the plain learning curve averaged over 3 seeds; and the verifier information $\sum_j I(C_j;V_j)$ evaluated at each round's empirical $q$. There are 3 seeds per cell and 144 runs in total (`results/e2_table.txt`).

**Table 1.** Selected E2 results (mean over 3 seeds; s.d. in parentheses). $N_{\text{eff}}$ is the human-equivalent sample size from learning-curve inversion. "inf" means above the largest human budget on the curve (51,200).

| task | mode | $n_0$ | $(a,b)$ | acc$_0$ | acc$_8$ | precision | verifier kbits | $N_{\text{eff}}$ | $M$ |
|---|---|---|---|---|---|---|---|---|---|
| skill | acc. | 64 | (1, 0) | 0.544 | 0.932 (0.012) | 1.000 | 12.4 | 1263 | 19.7 |
| skill | acc. | 64 | (1, 0.05) | 0.544 | 0.890 (0.015) | 0.980 | 11.9 | 617 | 9.6 |
| skill | acc. | 64 | (1, 0.15) | 0.544 | 0.816 (0.003) | 0.928 | 10.4 | 297 | 4.6 |
| skill | acc. | 64 | (1, 0.3) | 0.544 | 0.719 (0.013) | 0.833 | 8.0 | 162 | 2.5 |
| skill | acc. | 64 | (1, 0.6) | 0.544 | 0.608 (0.030) | 0.660 | 3.9 | 87 | 1.4 |
| skill | acc. | 64 | (1, 1) | 0.544 | 0.538 (0.039) | 0.496 | 0 | 68 | 1.07 |
| skill | acc. | 64 | (0.5, 0.5) | 0.544 | 0.520 (0.032) | 0.487 | 0 | 64 | 1.00 |
| skill | acc. | 24 | (1, 0) | 0.365 | 0.898 (0.050) | 1.000 | 13.0 | 699 | 29.1 |
| skill | acc. | 256 | (1, 0) | 0.800 | 0.954 (0.005) | 1.000 | 11.4 | 2438 | 9.5 |
| skill | replace | 64 | (1, 0.15) | 0.544 | 0.898 (0.008) | 0.942 | 9.6 | 697 | 10.9 |
| skill | replace | 64 | (1, 0.6) | 0.544 | 0.702 (0.021) | 0.681 | 3.9 | 148 | 2.3 |
| knowledge | acc. | 200 | (1, 0) | 0.768 | 0.957 (0.009) | 1.000 | 14.8 | 7183 | 35.9 |
| knowledge | acc. | 200 | (1, 0.3) | 0.768 | 0.862 (0.024) | 0.517 | 5.3 | 578 | 2.9 |
| knowledge | acc. | 200 | (1, 1) | 0.768 | 0.668 (0.012) | 0.162 | 0 | 76 | 0.38 |
| knowledge | acc. | 800 | (1, 0) | 0.892 | 0.981 (0.005) | 1.000 | 14.8 | inf | – |
| knowledge | acc. | 800 | (1, 0.15) | 0.892 | 0.952 (0.005) | 0.733 | 8.6 | 4951 | 6.2 |
| knowledge | acc. | 800 | (1, 1) | 0.892 | 0.750 (0.009) | 0.181 | 0 | 178 | 0.22 |
| knowledge | replace | 800 | (1, 1) | 0.892 | 0.804 (0.006) | 0.194 | 0 | 316 | 0.39 |
| knowledge | replace | 800 | (1, 0) | 0.892 | 0.949 (0.006) | 1.000 | 14.4 | 3915 | 4.9 |

**Findings.**

1. *The multiplier is set by verifier quality.* On the skill task with $n_0=64$, $M$ falls monotonically from 19.7 (perfect verifier) to 9.6, 4.6, 2.5 and 1.4 as $b$ rises to 0.05, 0.15, 0.3 and 0.6. It is about 1 (1.07 and 1.00) for the two uninformative verifiers $a=b$. Prop. 1 and Thm. 2 predict no gain at $a=b$, and none appears.
2. *Gain tracks verifier information.* Across the accumulate cells, the Spearman correlation between $\sum I(C;V)$ and accuracy gain is 0.89 for skill (24 cells, $p=3.9\times10^{-9}$) and 0.83 for knowledge (16 cells, $p=7\times10^{-5}$). Within each $n_0$ it ranges from 0.92 to 0.97 (`results/e2_spearman.txt`; Fig. 2c). Bits are not a sufficient statistic: the conversion rate from bits to accuracy differs between tasks and budgets.
3. *Smaller human budgets get larger multipliers but lower ceilings.* With a perfect verifier, $M=29.1$, 19.7 and 9.5 for $n_0=24$, 64 and 256, while final accuracy rises (0.898, 0.932, 0.954). This is the learnability pattern of Prop. 2: synthetic data helps most where the learner is furthest from its data-supported optimum.
4. *Knowledge can be learned by trial and error, but only through the verifier.* With $b=0$, the knowledge task rises from 0.892 to 0.981 ($n_0=800$). No generalization is possible here, so every learned fact came from the verifier, which in this setting *is* a perfect oracle on facts (Thm. 2, corollary (a)). The sustained gains at $b=0.3$ for $n_0=200$ (precision 0.52) agree with Prop. 3: the model's wrong guesses on unseen items are spread over 7 labels, so $w_{\max}$ is small.
5. *Unverified self-training hurts, and accumulation does not prevent this.* With $a=b=1$, knowledge accuracy falls from 0.892 to 0.750 under accumulate ($M=0.22$) and to 0.804 under replace. Uniform synthetic prompts reach rare items whose sampled labels are noise. Accumulating keeps every round's noise, while replacing discards it. On the skill task, *replace* beats *accumulate* at $b=0.15$ (0.898 vs 0.816) and $b=0.6$ (0.702 vs 0.608), because early rounds have low precision. With a perfect verifier, accumulate wins (0.932 vs 0.925 for skill, 0.981 vs 0.949 for knowledge). Whether accumulate is safer than replace (Gerstgrasser et al., 2024) therefore depends on whether the synthetic stream's precision rises over time.

### 5.3 E3: The urn law (script `e4_urn_scaling.py`)

We simulate Prop. 4 with 200 replicate urns of $2\times10^5$ proposals each, starting from 3 correct and 3 wrong labels, and fit the decay of the mean wrong share for $t\ge2000$ (`results/e4_urn.json`, Fig. 4). Fitted exponents against the prediction $1-b/a$: $(a,b)=(1,0)$: 1.001 vs 1; $(1,0.25)$: 0.751 vs 0.75; $(1,0.5)$: 0.498 vs 0.5; $(0.5,0.25)$: 0.496 vs 0.5; $(1,1)$: 0.0003 vs 0, with the wrong share staying at 0.499. At $(1,0.75)$ the fit is 0.216 against 0.25. Convergence to the asymptote is slowest when $a-b$ is small, so this is the expected pre-asymptotic bias. The case $(0.5,0.25)$ confirms that only the *ratio* $b/a$ matters for the exponent. The absolute rate $a$ only rescales time.

### 5.4 E4: What multipliers buy at the horizon (script `e3_horizon.py`)

If a pipeline multiplies effective data by $M$ and compute-optimal dataset size grows by $g_D$ per year, the date at which human data binds moves by $\Delta t=\ln M/\ln g_D$. We take $g_D=\sqrt{4}$ to $\sqrt5$, i.e. 2.0–2.24× per year. This follows from 4–5× per year compute growth (Sevilla & Roldán, 2024) and $D\propto C^{1/2}$ (Hoffmann et al., 2022). It is a derived quantity that assumes compute-optimal scaling, not a measured dataset growth rate. Results (`results/e3_horizon.json`, Fig. 3):

| source of multiplier | $M$ | delay at 2.0×/yr | delay at 2.24×/yr |
|---|---|---|---|
| 4-epoch repetition (Muennighoff et al., 2023) | 4 | 2.0 yr | 1.7 yr |
| stock at 95th pct vs central (1000T/300T) | 3.3 | 1.7 yr | 1.5 yr |
| source-grounded paraphrase, $s=0.5$ | ≤2 | ≤1.0 yr | ≤0.9 yr |
| source-grounded paraphrase, $s=0.1$ | ≤10 | ≤3.3 yr | ≤2.9 yr |
| teacher $s_g=0.5$, learner $s=0.1$ | ≤5.5 | ≤2.5 yr | ≤2.1 yr |
| source-grounded paraphrase, $s=0.01$ | ≤100 | ≤6.6 yr | ≤5.7 yr |

Under exponential demand growth, multipliers buy time only logarithmically. Even an unrealistically efficient factor of 100 buys under 7 years. Mapping the knowledge-model multiplier onto *token counts* in a pre-training scaling law is itself an assumption (§7). The table should be read as the arithmetic of the horizon, not as a forecast.

## 6 Discussion

**Answer to the motivating question.** Our framework says synthetic data *can* stand in for human data, but only through channels whose capacities can be named and in principle measured:

- For **knowledge** (facts whose truth is not derivable), synthetic data without an external source cannot add information (Prop. 1). Its value is bounded by learner inefficiency: at most $1/s$ for source-grounded rephrasing and at most $1+s_g(1-s)/s$ for teacher distillation (Prop. 2). It also cannot reach facts the corpus lacks. Verifiers for knowledge must already contain the knowledge, which makes them retrieval of human data under another name.
- For **verifiable skills** (mathematics, code, formal reasoning, games), the external signal can come from a verifier that is cheaper than the answer. The gain is then bounded by verifier information rather than by the human stock (Thm. 2), and it accrues at a rate set by $b/a$ (Prop. 4). Here synthetic data really does extend the frontier. The scarce resource becomes verifier coverage and compute for proposals.

This explains, within one account, why reasoning-focused synthetic pipelines with checkers (Zelikman et al., 2022; Gulcehre et al., 2023) and rephrasing pipelines (Maini et al., 2024; Yang et al., 2025) have succeeded. It also explains why recursive, unfiltered self-training collapses (Shumailov et al., 2024): the latter has $a=b$, hence zero information, and with systematic errors it falls below the threshold of Prop. 3.

**Learner dependence is a prediction, not a nuisance.** Prop. 2 implies that the same rephrased corpus yields a *smaller* multiplier for a more sample-efficient learner. If larger models retain more per exposure, as the knowledge-capacity literature suggests in some regimes (Allen-Zhu & Li, 2025), the knowledge benefit of rephrasing should *shrink with scale*. Skill benefits from verification should not shrink this way, because their ceiling is the verifier's, not the learner's.

**Accumulate vs replace.** Prior work found accumulate safer than replace in settings where synthetic and real data are both kept (Gerstgrasser et al., 2024; Kazdan et al., 2025). Our E2 adds the case of an imperfect verifier whose precision rises over rounds. Here, keeping early low-precision batches hurts and replacing them helps. A practical corollary is to weight or expire synthetic batches by their estimated precision.

### 6.1 Falsifiable predictions and a test protocol

**P1 (learner-dependence of rephrasing).** For a fixed source corpus and a fixed rephrasing pipeline, the closed-book QA gain from synthetic continued pre-training saturates at a multiplier that *decreases* with model scale. **Protocol**: (1) estimate $s$ for models of several sizes by measuring recall of planted facts as a function of exposure count, fitting $1-(1-s)^c$; (2) train with $r\in\{0,1,4,16,64\}$ paraphrases per document; (3) invert each model's own plain learning curve to obtain $M(r)$; (4) test against $M(r)=(1-(1-s)^{1+r})/s$. *Falsified if* measured $M$ exceeds $1/\hat s$ by more than the estimation error, or rises with scale.

**P2 (coverage ceiling).** Facts absent from the source corpus show no gain from any source-only pipeline, at any $r$, beyond the effect of answer priors. *Falsified if* planted held-out facts become answerable without retrieval.

**P3 (urn exponent).** In verified self-training on a domain with a tunable verifier, for example code with test suites of controlled false-positive rate, the residual error on a fixed problem set decays as (proposals)$^{-(1-b/a)}$ in the accumulate regime. *Falsified if* the fitted exponent depends on $a$ at fixed $b/a$, or does not fall toward 0 as $b\to a$.

**P4 (information sufficiency, weak form).** Across verifier settings, accuracy gains are rank-ordered by $\sum I(C;V)$. *Falsified if* a verifier with lower measured information consistently produces larger gains on the same task.

## 7 Limitations

- **Toy models.** The exposure model treats facts as independent and storage as independent per exposure. Real LLMs share representations across facts, can infer some facts from others (which blurs our knowledge/skill split), and likely retain repeated or near-duplicate exposures with diminishing probability. The "independent paraphrase" assumption is optimistic for $r\gg1$. A paraphrase-diversity discount would lower $M$ further, so our ceilings are, if anything, generous.
- **Small learners.** E2 uses multinomial logistic regression with 20 input dimensions (skill) or 400 one-hot items (knowledge), 8 rounds and 3 seeds. We ran no neural-network or LLM experiments, and nothing here shows that the measured multipliers transfer to them. The predictions in §6.1 are the route to testing that.
- **$N_{\text{eff}}$ estimation.** Multipliers come from interpolating a 3-seed mean learning curve on a log grid. Values near the curve's ends (e.g. $M\approx1$ cells, "inf" cells) are imprecise. Small departures from 1 at $a=b$ (1.07, 0.96) are within that imprecision.
- **Prop. 4 is mean-field.** We did not prove almost-sure rates. The simulation agrees closely except at small $a-b$, where convergence is slow. The urn also models a tabular learner. Learners that generalize couple items, which can speed things up (as in E2 skill) or entrench systematic errors.
- **Thm. 2 is an upper bound on information, not on accuracy.** E2 shows that bits are rank-predictive but not sufficient: the conversion rate differs by task and budget. We do not provide a tight bits-to-risk conversion.
- **Horizon arithmetic.** The delay table assumes compute-optimal scaling ($D\propto C^{1/2}$), constant compute growth, and that a knowledge-model multiplier applies to pre-training tokens. Over-training, multimodal data, private data and changes in compute growth would all move the numbers. The sourced inputs (stock, 2026–2032 window, 4–5× compute growth) carry their own uncertainty, which we inherit.
- **No estimate of real $s$, $s_g$, $a$, $b$.** We do not estimate these for any deployed model, so we make no quantitative claim about how much any existing pipeline extends any real horizon.

## 8 Conclusion

The question "can synthetic data take over once human data runs out?" becomes answerable once it is phrased as an exchange rate. A synthetic pipeline's data multiplier is limited by three ceilings. The information ceiling says that no external signal means no new information. The learnability ceiling says that rephrasing is worth at most $1/s$ human corpora for a learner with per-exposure retention $s$, exactly so in a Zipfian exposure model, and less for a weaker teacher. The verification ceiling says that a verifier contributes at most $I(C;V)$ bits per proposal, and that verified self-training converges as $t^{-(1-b/a)}$. Controlled experiments confirm each piece, and they show that unverified self-training can actively destroy knowledge even when data is accumulated. For the data-exhaustion debate, our results suggest that synthetic data will extend progress mainly in domains where checking is cheaper than knowing. For factual breadth, synthetic data improves how well existing human knowledge is absorbed but does not replace new human (or retrieved) knowledge. Multipliers of any plausible size move the horizon by only a few years. Measuring $s$, $s_g$, $a$ and $b$ at scale is the natural next step, and §6.1 gives a protocol.

## Reproducibility

`code/README.md` lists the commands. Total CPU time is about 20 minutes on 4 cores, almost all of it E2. All figures (`results/fig1_knowledge.png`, `fig2_selftrain.png`, `fig3_horizon.png`, `fig4_urn.png`) and tables (`results/e2_table.txt`, `e2_spearman.txt`, `e1_knowledge.json`, `e3_horizon.json`, `e4_urn.json`) are produced by the scripts.

## References

Alemohammad, S., et al. (incl. Baraniuk, R. G.) (2024). Self-Consuming Generative Models Go MAD. *International Conference on Learning Representations (ICLR 2024)*. arXiv:2307.01850.

Allen-Zhu, Z., & Li, Y. (2025). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2404.05405.

Bertrand, Q., Bose, A. J., Duplessis, A., Jiralerspong, M., & Gidel, G. (2024). On the Stability of Iterative Retraining of Generative Models on their own Data. *International Conference on Learning Representations (ICLR 2024)*. arXiv:2310.00429.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Dohmatob, E., Feng, Y., Yang, P., Charton, F., & Kempe, J. (2024). A Tale of Tails: Model Collapse as a Change of Scaling Laws. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235, 11165–11197. arXiv:2402.07043.

Feng, Y., Dohmatob, E., Yang, P., Charton, F., & Kempe, J. (2025). Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2406.07515.

Gerstgrasser, M., et al. (2024). Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. arXiv:2404.01413.

Gulcehre, C., et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998.

Gunasekar, S., et al. (2023). Textbooks Are All You Need. arXiv:2306.11644.

Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.

Huang, A., Block, A., Foster, D. J., Rohatgi, D., Zhang, C., Simchowitz, M., Ash, J. T., & Krishnamurthy, A. (2025). Self-Improvement in Language Models: The Sharpening Mechanism. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2412.01951.

Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, 15696–15707. arXiv:2211.08411.

Kazdan, J., Schaeffer, R., Dey, A., Gerstgrasser, M., Rafailov, R., Donoho, D. L., & Koyejo, S. (2025). Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, PMLR 267. arXiv:2410.16713.

Maini, P., Seto, S., Bai, H., Grangier, D., Zhang, Y., & Jaitly, N. (2024). Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), Long Papers*, 14044–14072. arXiv:2401.16380.

Muennighoff, N., Rush, A. M., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., & Raffel, C. (2023). Scaling Data-Constrained Language Models. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2305.16264.

Sevilla, J., & Roldán, E. (2024). Training Compute of Frontier AI Models Grows by 4-5x per Year. Epoch AI. https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755–759. doi:10.1038/s41586-024-07566-y.

Song, Y., Zhang, H., Eisenach, C., Kakade, S., Foster, D., & Ghai, U. (2025). Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2412.02674.

Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., & Hobbhahn, M. (2024). Will we run out of data? Limits of LLM scaling based on human-generated data. arXiv:2211.04325 (v2, June 2024).

Yang, Z., Band, N., Li, S., Candès, E., & Hashimoto, T. (2025). Synthetic continued pretraining. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2409.07431.

Yi, B., Liu, Q., Cheng, Y., & Xu, H. (2025). Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence. arXiv:2510.16657 (presented at ICLR 2026).

Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2203.14465.
