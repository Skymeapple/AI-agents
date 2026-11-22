# Exchange Rates for Synthetic Data: Coverage, Learnability and Verification Limits on Stretching the Human Data Stock

*Target venue: International Conference on Learning Representations (ICLR 2027), main track (theory plus controlled experiments)*

*Revision 1. A point-by-point response to the round-1 review is in `reviews/response_r1.md`.*

---

## Abstract

Forecasts put full use of the stock of high-quality public human text somewhere between 2026 and 2032, an 80% interval (Villalobos et al., 2024). Synthetic data is the most common answer to what happens next. The question this raises is an exchange rate: how many human tokens is a synthetic pipeline worth? We call this the **data multiplier** $M$. We give simple accounting tools for $M$ and test them, including on a small transformer. **Theory.** Most of our formal statements are standard results applied to this question, and we label them that way: data processing (no external signal, no new information), an exact information accounting identity for verifier-filtered generation, and the known asymptotics of the randomly reinforced urn (Muliere et al., 2006; May & Flournoy, 2009), whose rate gives a residual-error exponent of $1-b/a$ for verified self-training. Two statements are new as far as we know. (i) The 1/s ceiling on paraphrase multipliers, and its independence from the frequency distribution, hold only under memoryless (geometric) storage. Under threshold storage the multiplier depends on the frequency distribution and on corpus size, and it diverges for rare facts. Only the *coverage ceiling* (facts absent from the corpus stay unlearned) holds for every storage law. (ii) A verifier with a description length of $L$ bits can add at most $L$ bits about the target, however many proposals it checks. Knowledge verifiers must therefore grow with the number of facts, while a program verifier for a skill need not. **Experiments.** (a) In 78 runs of a 0.67M-parameter transformer trained on planted facts, a source-grounded paraphrase is worth about one fresh occurrence at $r=1$ ($M=2.09$, 90% CI 1.98–2.19). Paraphrases beat verbatim repetition (2.97 vs 2.66 at $r=3$). Recall saturates *exactly at the coverage ceiling* (0.813 vs 0.812; 0.934 vs 0.934). The multiplier depends on corpus size and frequency distribution (1.80 to 3.75 at $r=3$), which falsifies Prop. 2's invariance for this learner. Fitted per-exposure retention is not a learner constant: it rises from 0.03 to 0.21 as the corpus grows eightfold. (b) In a verifier-design experiment that decouples information from $b/a$, gain is ranked by $b/a$ and precision (Spearman 0.96–0.99), not by verifier information (0.78–0.85). Our earlier claim that "gain tracks information" does not survive. Information is a ceiling, not a predictor, and the urn law's $b/a$ is the better predictor. (c) At equal verifier description length (~500–670 bits), a quantized program verifier raises skill accuracy by 0.29–0.35. A lookup verifier for facts raises knowledge accuracy by 0.05 at 600 bits and needs 1,200 bits (all 400 facts) for 0.18. **Implication.** Published rephrasing gains (about 3× for WRAP; 5–10× speed-ups to a fixed loss for a one-third rephrased mix) and our tiny-LM multipliers (2–3.8×) correspond to roughly 1.3–3.3 years of horizon delay if compute-optimal dataset size keeps growing 2.0–2.24× per year. That figure is arithmetic on externally measured inputs, not a forecast. Beyond coverage, knowledge gains need new information, which comes from human data, retrieval or a verifier that stores the facts. Skills can get it from compact verifiers.

## 中文摘要

据预测，高质量公开人类文本存量将在2026至2032年间（80%区间）被用尽（Villalobos 等，2024）。合成数据被普遍视为接续方案，核心问题是"汇率"：一条合成数据流水线相当于多少人类数据？我们称之为**数据乘数** $M$，并给出核算工具和检验，包括一个小型 Transformer 实验。**理论**：多数形式结论是标准结果在此问题上的应用，文中均如实标注：数据处理不等式（无外部信号则无新信息）、验证器过滤生成的精确信息核算恒等式，以及随机强化罐子模型的已知渐近结果（Muliere 等，2006；May & Flournoy，2009），由此得到验证自训练的残差指数 $1-b/a$。据我们所知，有两点是新的。（一）改写乘数的 $1/s$ 上限及其与频率分布无关的性质，只在无记忆（几何）存储下成立；在阈值存储下，乘数依赖频率分布和语料规模，且对稀有事实发散。对任何存储规律都成立的只有**覆盖上限**：语料中没有的事实学不到。（二）描述长度为 $L$ 比特的验证器，无论检查多少提议，至多提供 $L$ 比特关于目标的信息；因此知识验证器的规模必须随事实数增长，而技能的程序验证器不必。**实验**：（a）在78次训练（0.67M参数 Transformer，植入事实）中，$r=1$ 时一次源文改写约等于一次新出现（$M=2.09$，90%置信区间1.98–2.19）；改写优于逐字重复（$r=3$ 时2.97对2.66）；召回率**恰好饱和于覆盖上限**（0.813对0.812；0.934对0.934）；乘数随语料规模和频率分布变化（$r=3$ 时从1.80到3.75），对该学习器证伪了命题2的不变性；拟合的单次曝光保持率不是学习器常数，语料扩大8倍时从0.03升到0.21。（b）在信息量与 $b/a$ 解耦的验证器设计实验中，增益的排序由 $b/a$ 和精度决定（Spearman 0.96–0.99），而不是验证器信息量（0.78–0.85）。我们此前"增益随信息量变化"的结论不成立：信息量是上限，不是预测量，罐子模型的 $b/a$ 是更好的预测量。（c）在相同验证器描述长度（约500–670比特）下，量化程序验证器把技能准确率提高0.29–0.35；事实查表验证器在600比特时只把知识准确率提高0.05，提高0.18需要1200比特（全部400个事实）。**含义**：已发表的改写收益（WRAP约3倍；三分之一改写数据混合在达到同一损失时提速5–10倍）和本文小模型乘数（2–3.8倍），若计算最优数据集规模继续每年增长2.0–2.24倍，相当于把数据枯竭期推迟约1.3–3.3年。这是基于外部测量输入的算术，不是预测。在覆盖范围之外，知识的增益需要新信息，来源只能是人类数据、检索或存储事实本身的验证器；技能则可以从紧凑的验证器获得新信息。

---

## 1 Introduction

Villalobos et al. (2024) estimate the stock of quality-adjusted public human text at about 300T tokens (90% CI 100T–1000T). They project that compute-optimal training runs will reach datasets of that size between 2026 and 2032, an 80% interval, and earlier under heavy over-training. Frontier training compute has grown 4–5× per year (Sevilla & Roldán, 2024), and compute-optimal dataset size grows roughly as the square root of compute (Hoffmann et al., 2022). So the question is near-term: **can synthetic data take over once human data runs out?**

The literature points two ways. The *collapse* line shows that recursive training on a model's own outputs loses the tails of the distribution (Shumailov et al., 2024; Alemohammad et al., 2024) and changes the scaling laws (Dohmatob et al., 2024). The *rescue* line shows several things. Accumulating real and synthetic data bounds the damage (Gerstgrasser et al., 2024; Kazdan et al., 2025; Bertrand et al., 2024). Verification lets synthetic data help (Feng et al., 2025; Yi et al., 2025). Self-improvement is governed by a generation–verification gap (Song et al., 2025) or by sharpening (Huang et al., 2025). Rephrased corpora improve data efficiency (Maini et al., 2024; Yang et al., 2025; Kang et al., 2025).

A planner needs an **exchange rate**: how many human tokens a synthetic pipeline is worth, and what limits that number. This paper gives a vocabulary for that exchange rate, a few accounting results, and tests of those results. The tests include one that falsifies part of our own earlier model.

**What is standard and what is new.** Several statements below are textbook facts applied to this question. Fact 1 is data processing. Lemma 3 is the chain rule. Remark 1 is the strong law. Prop. 5 is an application of randomly-reinforced-urn asymptotics. We label them so and do not count them as contributions. Our contributions are:

- **C1 (when the paraphrase ceiling holds, and when it does not).** In an exposure model of factual knowledge, the identity $M(r)=(1-(1-s)^{1+r})/s$ and its independence from the frequency distribution follow directly from assuming memoryless per-exposure storage (Prop. 2). We show it is *only* that assumption. Under threshold storage the multiplier depends on the frequency distribution and on $N$, approaches the memoryless value for frequent facts and diverges as $\lambda^{-(1-1/m)}$ for rare ones (Prop. 2b). The statement that survives every storage law is the coverage ceiling (Prop. 2c).
- **C2 (verifier description length).** A verifier of description length $L$ bits adds at most $L$ bits about the target, for any number of proposals (Prop. 4). This makes precise the claim that a knowledge verifier "must contain the facts": its size has to grow with the number of facts learned beyond the corpus. A program verifier for a structured skill does not.
- **C3 (a neural test of the knowledge model).** A 0.67M-parameter transformer trained on planted facts (78 runs) gives controlled measurements of per-exposure retention $s$, paraphrase multipliers and the coverage ceiling in one controlled setting. The coverage ceiling holds exactly. Prop. 2's invariance fails. $s$ is not a learner constant.
- **C4 (what predicts verified self-training gains).** A verifier grid that decouples information from $b/a$ shows that gain is ranked by $b/a$ and precision, as the urn rate predicts, and not by information. This overturns our own earlier claim. A verifier-cost experiment shows the knowledge/skill asymmetry at equal verifier description length.
- **C5 (anchored horizon arithmetic).** We convert multipliers into horizon delays using externally measured rephrasing gains and our own measurements, each labelled by provenance.

## 2 Related Work

**Data stock and scaling.** Villalobos et al. (2024) estimate the human text stock and when it will be used up. Hoffmann et al. (2022) give compute-optimal scaling. Muennighoff et al. (2023) find that up to about four epochs of repetition are nearly as good as fresh data, and they model the decaying value of repeated tokens. Our $M$ is the synthetic-data analogue of their effective data.

**Synthetic data for pre-training and knowledge.** WRAP rephrases web documents and reports about 3× faster pre-training (Maini et al., 2024). Kang et al. (2025) run a large study of synthetic pre-training data across more than 1,000 models. They find rephrased data alone is not faster than natural web text, while a mix of about one-third rephrased data reaches the same validation loss 5–10× faster at larger budgets. EntiGraph turns a small corpus into a large synthetic one for continued pre-training (Yang et al., 2025). Kandpal et al. (2023) relate factual accuracy to the number of supporting documents. Allen-Zhu & Li (2024, Part 3.1) show on synthetic biographies that knowledge is reliably *extractable* only when facts appear in diverse forms (paraphrasing, shuffling). Allen-Zhu & Li (2025, Part 3.3) measure knowledge capacity, which requires many exposures per fact. Our E5 uses the same planted-fact methodology at a much smaller scale. Its paraphrase-beats-repetition result agrees with their diversity finding.

**Model collapse and accumulation.** Shumailov et al. (2024), Alemohammad et al. (2024), Bertrand et al. (2024), Dohmatob et al. (2024), Gerstgrasser et al. (2024) and Kazdan et al. (2025) characterise when recursive training degrades, and show that accumulating real data bounds the damage.

**Verification, self-improvement and information.** Feng et al. (2025) show that scaling with synthesized data needs verification and that imperfect verifiers can still prevent collapse. Yi et al. (2025) analyze verifier-filtered retraining and find near-term gains that converge to the verifier's "knowledge center". Our E6 skill result (the learner converges to the quantized verifier's own accuracy) is an instance of this. Song et al. (2025) formalize the generation–verification gap. Huang et al. (2025) model self-improvement as sharpening: extracting knowledge that is latent in the model rather than acquiring new information. STaR (Zelikman et al., 2022) and ReST (Gulcehre et al., 2023) are practical verified-bootstrapping loops. Gan & Liu (2025) give an information-theoretic account of synthetic data in post-training, the "reverse bottleneck", in which generalization gain is tied to the information gain contributed by the generative model. Our Fact 1 and Lemma 3 share that information view. We use it for a different question: an exchange rate against human data, and the cost of the verifier.

**Randomly reinforced urns.** In a two-colour urn where a drawn colour is reinforced by a random amount with colour-dependent mean, Muliere, Paganoni & Secchi (2006) derive asymptotics for the composition, and May & Flournoy (2009) give asymptotics, including convergence rates of the number of draws of each colour, for the response-adaptive design that this urn generates. Verified self-training on one prompt with a tabular learner is such an urn, with reinforcement means $a$ (correct) and $b$ (wrong). Prop. 5 is an application of these results, not a new theorem.

## 3 Framework and the Information Ceiling

**Setup.** A target $\theta$ generates a human corpus $D_h\sim P_\theta^{\,n}$. A pipeline produces $D_s\sim Q(\cdot\mid D_h,U,V)$, where $U$ is independent randomness and $V$ is an optional external signal (a verifier, an environment, retrieval).

**Definition (data multiplier).** Let $\rho_{\mathcal A}(n)$ be the expected risk of learner $\mathcal A$ on $n$ human samples, assumed decreasing. Define $N_{\text{eff}}=\rho_{\mathcal A}^{-1}(\mathbb E\,R_{\mathcal A}(D_h\cup D_s))$ and $M=N_{\text{eff}}/n$. $M$ depends on the learner.

**Fact 1 (garbling; standard).** If $\theta\to D_h\to D_s$ is a Markov chain, then $I(\theta;D_h,D_s)=I(\theta;D_h)$, and the Bayes risk from $(D_h,D_s)$ equals that from $D_h$. *Proof:* chain rule plus data processing (Cover & Thomas, 2006, §2.8). For risk, simulate $D_s$ internally from $D_h$; this is Blackwell garbling. $\square$

Fact 1 is used only to sort gains into types. Without an external signal, any gain is a *learnability* gain, bounded by $\rho_{\mathcal A}(n)-\rho_{\text{Bayes}}(n)$. This matches Huang et al.'s (2025) view of self-improvement as extracting knowledge that is already latent.

## 4 Theory

### 4.1 Knowledge: exposure model, its limits, and the coverage ceiling

There are $K$ facts with query frequencies $p_i$ and $k$ answer options. A corpus of $N$ samples gives fact $i$ a count $c_i\sim\text{Poisson}(Np_i)$. A storage law $g(c)=\Pr[\text{stored}\mid c\text{ exposures}]$ describes the learner. An unstored fact is guessed with accuracy $1/k$. Accuracy is $A(N)=1-(1-\tfrac1k)\sum_i p_i\,\mathbb E[1-g(c_i)]$. A source-grounded generator ($s_g=1$) re-expresses each human exposure $r$ times. A generator of retention $s_g$ keeps a given exposure with probability $s_g$.

**Prop. 2 (memoryless storage; follows directly from the assumption).** If each exposure is stored independently with probability $s$, i.e. $g(c)=1-(1-s)^c$, then for every $(p_i)$, $N$, $K$ and $k$ the amplified corpus has exactly the accuracy of a plain corpus of size $MN$ with $M=\big(1-(1-s)[(1-s_g)+s_g(1-s)^r]\big)/s$. This tends to $1+s_g(1-s)/s$, and for $s_g=1$ to $1/s$. *Proof:* Poisson thinning. Each human exposure independently leads to storage with probability $1-f$, so $\Pr[\text{unstored}]=e^{-Np_i(1-f)}=e^{-sN'p_i}$ with $N'=N(1-f)/s$. $\square$

The invariance in $(p_i)$ and $N$ is a restatement of the memoryless assumption, as the reviewer noted. What matters is whether real learners satisfy it. The next statement shows the invariance is fragile, and E5 tests it.

**Prop. 2b (threshold storage breaks invariance).** Suppose a fact is stored iff at least $m\ge2$ of its exposures are "storage events", each an independent Bernoulli($s$). Let $M(\lambda)$ be the per-fact multiplier, i.e. the plain exposure rate that matches the amplified unstored probability of a fact with rate $\lambda=Np_i$, divided by $\lambda$. Then for $1+r\ge m$:
(a) as $\lambda\to0$, $M(\lambda)\sim (m!\,P_r)^{1/m}\,s^{-1}\,\lambda^{-(1-1/m)}\to\infty$, where $P_r=\Pr[\text{Bin}(1+r,s)\ge m]$;
(b) as $\lambda\to\infty$, $M(\lambda)\to(1-(1-s)^{1+r})/s$, the memoryless value.
Hence the corpus-level $M$ depends on $(p_i)$ and $N$, and it can exceed $1/s$.

*Proof.* Plain: storage events are Poisson($s\lambda'$), so $\Pr[\text{stored}]=\Pr[\text{Pois}(s\lambda')\ge m]\sim(s\lambda')^m/m!$ as $\lambda'\to0$. Amplified: storage events are compound Poisson, with $\text{Pois}(\lambda)$ human exposures each contributing $\text{Bin}(1+r,s)$ events. As $\lambda\to0$ the event "at least two human exposures" has probability $O(\lambda^2)$. So $\Pr[\text{stored}]=\lambda P_r+O(\lambda^2)$. Equating gives (a). For (b), $\Pr[\text{unstored}]$ is dominated by the zero-event term up to polynomial factors. For the plain corpus this is $e^{-s\lambda'}\,\text{poly}(\lambda')$. For the amplified corpus it is $e^{-\lambda(1-(1-s)^{1+r})}\,\text{poly}(\lambda)$, because a human exposure yields zero events with probability $(1-s)^{1+r}$. Matching exponents gives $\lambda'/\lambda\to(1-(1-s)^{1+r})/s$. $\square$

E7 computes this exactly by Panjer's recursion (`results/e7_threshold.json`). At $s=0.1$ and $r=7$, the memoryless model gives $M=5.70$ for every $(\alpha,N)$. Under $m=2$, $M$ ranges from 5.87 to 8.71 over $\alpha\in\{0.6,1,1.4\}$ and $N\in\{10^4,10^5,10^6\}$. The per-fact $M$ is 21.0 at $\lambda=0.1$ and 5.73 at $\lambda=100$. Under $m=3$ the per-fact $M$ at $\lambda=0.1$ is 32.4, which exceeds $1/s=10$.

**Prop. 2c (coverage ceiling; any storage law).** If the generator only re-expresses facts present in $D_h$, then for every storage law the amplified accuracy is at most $A_{\text{cov}}(D_h)=\sum_i p_i\big[\mathbb 1(c_i>0)+\tfrac1k\mathbb 1(c_i=0)\big]$. *Proof:* a fact with $c_i=0$ produces no exposures of any kind, so it is guessed. $\square$

This is elementary too, but it is the one knowledge-side statement that does not depend on the storage model, and E5 finds it binding.

**Fidelity.** If a fraction $1-\phi$ of paraphrases carries one systematic wrong answer and the learner outputs its most-often-stored label, then as $r\to\infty$ the correct label wins iff $\phi>1/2$. (The single original exposure is always correct, so this holds only in the limit $r\to\infty$. At finite $r$ and $\phi=1/2$ the outcome depends on tie-breaking and on the original exposure.)

### 4.2 Verification: accounting identity, description length, urn dynamics

For proposal $j$ (prompt $x_j$, candidate $\hat y_j$ drawn without reference to $\theta$ given the past $\mathcal F_j$), let $C_j=\mathbb 1[\hat y_j\text{ correct}]$, a function of $(\theta,x_j,\hat y_j)$. The verifier returns $V_j$ with $\Pr[V_j=1\mid C_j=1]=a$ and $\Pr[V_j=1\mid C_j=0]=b$, with independent noise.

**Lemma 3 (accounting identity; elementary).**
$$I(\theta;D_h,D_s,V_{1:m})=I(\theta;D_h)+\sum_{j=1}^m I(C_j;V_j\mid\mathcal F_j,x_j,\hat y_j)\;\le\;I(\theta;D_h)+\sum_{j=1}^m \mathcal I_{a,b}\big(\mathbb E\,q_j\big),$$
where $q_j=\Pr[C_j=1\mid\mathcal F_j,x_j,\hat y_j]$ and $\mathcal I_{a,b}(q)=h(aq+b(1-q))-qh(a)-(1-q)h(b)$. The conditional mutual information in the middle term is itself an expectation over $(\mathcal F_j,x_j,\hat y_j)$: it equals $\mathbb E[\mathcal I_{a,b}(q_j)]$.

*Proof.* Chain rule. Given $\mathcal F_j$, the proposal carries no information about $\theta$. Given the proposal, $I(\theta;V_j\mid\cdot)=I(\theta,C_j;V_j\mid\cdot)=I(C_j;V_j\mid\cdot)+I(\theta;V_j\mid C_j,\cdot)=I(C_j;V_j\mid\cdot)$. The second equality holds because $C_j$ is a function of $\theta$ given the proposal. The last holds because $V_j$ depends on $\theta$ only through $C_j$. So this step is an equality, which corrects the inequality in our earlier version. The final inequality is concavity of $\mathcal I_{a,b}$ in $q$ (Jensen). $\square$

**Prop. 4 (verifier description length bounds total new information).** Suppose the verifier is a deterministic function $V_j=\nu_d(x_j,\hat y_j)$ of a description $d$ of length $L$ bits, plus noise independent of $\theta$. Suppose proposals depend on $\theta$ only through $(D_h,V_{1:j-1})$. Then for any number of proposals $m$,
$$I(\theta;D_h,D_s,V_{1:m})\;\le\;I(\theta;D_h)+L.$$
*Proof.* $(D_s,V_{1:m})$ is a function of $(D_h,d,U,\text{noise})$, with $U$ and the noise independent of $(\theta,d,D_h)$. By data processing, $I(\theta;D_h,D_s,V_{1:m})\le I(\theta;D_h,d)=I(\theta;D_h)+I(\theta;d\mid D_h)\le I(\theta;D_h)+H(d)\le I(\theta;D_h)+L$. $\square$

*Consequences.* For $K'$ independent uniform facts on $k$ options that are absent from the corpus, learning a fraction $\eta$ of them requires $I\ge\eta K'\log_2k-O(1)$ bits (Fano), so $L\ge\eta K'\log_2k-O(1)$. A knowledge verifier is a store of the facts, and its size grows with the number of facts. For a structured skill whose target has a short description (here, a linear teacher with 168 parameters), a verifier of fixed $L$ can label an unbounded input space. Lemma 3 bounds new information per proposal; Prop. 4 bounds total new information by the verifier's size. New information is at most $\min\{\sum_j\mathbb E\,\mathcal I_{a,b}(q_j),\,L\}$. Both are upper bounds on *information*. As E6 shows, neither predicts *accuracy gain* well.

**Remark 1 (plurality; strong law).** If a fixed generator outputs the correct label with probability $q$ and wrong label $y$ with probability $w_y$, the accepted stream's plurality is a.s. correct asymptotically iff $aq>b\,w_{\max}$ (with equality left undetermined). Systematic errors (large $w_{\max}$) are the dangerous case.

**Prop. 5 (verified self-training is a randomly reinforced urn; application of known results).** For one prompt, let the pool hold $C$ correct and $W$ wrong labels. Each step the learner samples a label in proportion to the pool. The verifier accepts with probability $a$ (correct) or $b$ (wrong), and accepted labels are added. This is a two-colour randomly reinforced urn with Bernoulli reinforcements of means $a>b$ (Muliere et al., 2006; May & Flournoy, 2009). In the mean-field limit $\tfrac{d}{dt}\ln(C/W)=(a-b)/T$ with $T\sim at$, giving
$$W/T\asymp t^{-(1-b/a)}.$$
May & Flournoy (2009) prove rigorous convergence-rate results for the number of draws of each colour in this urn; these are the rigorous counterpart of the mean-field rate (the inferior colour's count growing like a power of $t$ set by the ratio of mean reinforcements, here $t^{b/a}$). We did not re-derive their theorem and rely on it only for this correspondence. We state Prop. 5 only as the mapping "verifier $\leftrightarrow$ reinforcement means", which yields the exponent $1-b/a$, and we check it by simulation (E3). The prediction we test in the learner experiments is that $b/a$, not $a$ or information, governs the rate, while $a$ only rescales time.

## 5 Experiments

All code is in `code/` and all outputs are in `results/`. Every number in this section comes from those scripts, run in this environment on CPU, unless a citation is given. Experiment numbers match script names. We present them in the order of the argument.

### 5.1 E1: Numerical check of Prop. 2 inside its own model

E1 is a code check, not evidence about real learners. A Monte Carlo of the same stochastic model tests the closed form for errors, and the "N-equivalent" check inverts the closed form, so agreement there is guaranteed algebraically. Setup: $K=10^5$, $k=10$, Zipf 1.1, $N=2\times10^5$, 3 replicates. Across 15 cells, closed form and MC agree to within 0.0019. The largest gap (0.7818 vs 0.7836, $s=0.05,r=0$) is 1.34 population s.d. of the 3 replicates, or about 1.9 standard errors using the $n-1$ s.d. The N-equivalent multiplier equals $M(r)$ to a relative error of $2\times10^{-11}$. Fidelity under a majority readout (4 replicates): with $\phi=0.9$ or $0.7$, $M$ still rises (to 10.05 and 9.74 at $r=256$). At $\phi=0.5$ accuracy falls from 0.819 to 0.688. At $\phi=0.4$ it collapses to 0.019. The $\phi\le1/2$ failures agree with the large-$r$ limit in §4.1. At $\phi=0.5$ the limit is undetermined, and the observed decline comes from tie-breaking and the finite-$r$ dilution of the original exposure, not from Remark 1.

### 5.2 E5: Does a small transformer obey the exposure model? (main test of C1/C3)

**Setup** (`e5_tinylm_retention.py`). There are 1,000 persons × 2 attributes = 2,000 facts, and each value is uniform over 16 options. Each sentence states one fact in one of 8 surface templates (different filler tokens; two word orders), with the value as the last token. Fact frequencies are Zipf($\alpha$) over facts, randomly permuted, and counts are Poisson($Np_i$). The model is a 2-layer causal transformer (width 128, 4 heads, 0.67M parameters), trained with full next-token loss, AdamW (lr $2\times10^{-3}$), batch 64, for a fixed 4 epochs over whatever corpus it is given. Recall is the argmax over the 16 values of the attribute, averaged over the 8 templates. Accuracy is weighted by query frequency, as in §4.1. Plain corpora have $N\in\{1,2,4,8,16,32,64\}\times10^3$. Amplified corpora add $r$ copies per human sentence, either in *other* templates ("paraphrase", source-grounded) or in the *same* template ("repetition"). $M$ comes from inverting the 3-seed mean plain curve, with 90% CIs from a 2,000-sample bootstrap over seeds of both curves. There are 78 runs, taking 17 min on 4 shared cores (`results/e5_tinylm.json`, `e5_analysis.json`, Fig. 5).

**Table 2.** Tiny transformer, paraphrase vs repetition (3 seeds). "Coverage" is $A_{\text{cov}}$ of Prop. 2c for the human corpus.

| $\alpha$ | $N_0$ | $r$ | paraphrase acc | $M$ (90% CI) | repetition acc | $M$ (90% CI) | coverage | memoryless pred. $M$ |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 2000 | 1 | 0.761 | 2.09 (1.98–2.19) | 0.755 | 2.01 (1.94–2.12) | 0.812 | 1.89 |
| 1.0 | 2000 | 3 | 0.813 | 2.97 (2.82–3.13) | 0.797 | 2.66 (2.55–2.80) | 0.812 | 3.38 |
| 1.0 | 2000 | 7 | 0.811 | 2.92 (2.78–3.08) | 0.793 | 2.59 (2.47–2.72) | 0.811 | 5.50 |
| 1.0 | 8000 | 1 | 0.920 | 1.61 (1.51–1.70) | 0.916 | 1.56 (1.48–1.64) | 0.932 | 1.59 |
| 1.0 | 8000 | 3 | 0.934 | 1.80 (1.72–1.88) | 0.908 | 1.47 (1.22–1.70) | 0.934 | 2.14 |
| 0.6 | 2000 | 1 | 0.322 | 2.15 (1.94–2.33) | – | – | 0.707 | 1.99 |
| 0.6 | 2000 | 3 | 0.692 | 3.75 (3.66–3.85) | – | – | 0.708 | 3.94 |

"Memoryless pred." uses the geometric $s$ fitted on the plain corpus of size $N_0$. Each corpus occurrence is seen 4 times (epochs).

**Findings.**

1. *The coverage ceiling binds exactly.* Paraphrase recall saturates at the coverage ceiling: 0.813 and 0.811 against 0.812 at $N_0=2000$, and 0.934 against 0.934 at $N_0=8000$. Going from $r=3$ to $r=7$ adds nothing. Facts never seen in the human corpus stay at chance, as Prop. 2c requires.
2. *One paraphrase is worth about one fresh occurrence, and paraphrase beats repetition.* At $r=1$, $M\approx2$ (2.09, 2.15; 1.61 when $N_0=8000$ is already near coverage). Surface diversity helps: $M=2.97$ vs 2.66 at $r=3$, 2.92 vs 2.59 at $r=7$ and 1.80 vs 1.47 at $N_0=8000$. Prop. 2 treats the two identically, so it misses this effect. The effect agrees qualitatively with Allen-Zhu & Li (2024).
3. *Prop. 2's invariance is falsified for this learner.* At $r=3$, $M$ is 2.97, 1.80 and 3.75 across $(\alpha,N_0)$, where Prop. 2 predicts one value. Most of the variation follows the headroom below the coverage ceiling: the gap between plain accuracy at $N_0$ and $A_{\text{cov}}$.
4. *Per-exposure retention is not a learner constant.* Fitted geometric $s$ rises with corpus size: 0.029, 0.075, 0.124 and 0.205 for $N=2,4,8,16\times10^3$ at $\alpha=1$, and 0.0025 to 0.28 at $\alpha=0.6$. Under fixed-epoch training a larger corpus also means more optimizer steps, so "retention" reflects the training regime as well as the architecture. On per-fact likelihood, the geometric law and the $m=2$ threshold law trade places from one $N$ to the next, and $m\ge3$ is clearly worse. So storage here is close to memoryless with at most a mild threshold, not strongly thresholded. Recall at 1, 2 and 3 occurrences is 0.25, 0.49 and 0.74 ($\alpha=1$, $N=4000$).
5. *The memoryless model predicts $M$ well at $r=1$ and overpredicts it beyond.* At $r=1$ the predictions are 1.89, 1.59 and 1.99 against measured 2.09, 1.61 and 2.15. At $r\ge3$ the predictions are too high (3.38 vs 2.97, 5.50 vs 2.92), because the network reaches coverage faster than a constant-$s$ model allows. In short, the memoryless model gets the order of magnitude, and coverage sets the ceiling, not $1/s$.

E5 is a 0.67M-parameter model on synthetic facts. It is an existence test of the model's mechanisms in a neural learner, not an estimate of $s$ for frontier LLMs.

### 5.3 E2: Verified self-training with a convex learner (original experiment, re-analysed)

**Setup** (unchanged). *Skill*: $x\in\{\pm1\}^{20}$ and $y=\arg\max(W^\star x+b^\star)$ over 8 classes. *Knowledge*: 400 one-hot items with uniform random labels, and Zipf(1.1) human prompts. The learner is L2-regularised multinomial logistic regression. The loop runs 8 rounds of 2,000 fresh prompts with labels sampled from the model, an $(a,b)$ verifier, accumulate or replace, and retraining. $N_{\text{eff}}$ comes from inverting the plain learning curve. There are 3 seeds and 144 runs. A re-run by the reviewer reproduced every number bit-for-bit.

**Table 1.** Selected E2 results (mean over 3 seeds; s.d. in parentheses). "inf" means above the largest budget on the human curve (51,200). Those cells are excluded from $M$ summaries, which biases aggregates downward.

| task | mode | $n_0$ | $(a,b)$ | acc$_0$ | acc$_8$ | precision | verifier kbits (marginal $q$) | $N_{\text{eff}}$ | $M$ |
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

**Findings that stand.**
(i) Uninformative verifiers ($a=b$) give $M\approx1$ or less, as Fact 1 and Lemma 3 require.
(ii) Smaller human budgets get larger multipliers but lower final accuracy (29.1, 19.7 and 9.5 for $n_0=24$, 64 and 256).
(iii) Unverified self-training *hurts* on the knowledge task even when data is accumulated (0.892 to 0.750). Replace beats accumulate when early-round precision is low (skill, $b=0.15$: 0.898 vs 0.816; $b=0.6$: 0.702 vs 0.608), and accumulate wins with a perfect verifier.

**Finding withdrawn.** We previously wrote that "gain tracks verifier information" and supported it with Spearman $\rho=0.89/0.83$ and $p$-values. The reviewer is right that 6 of the 8 verifiers per budget are $(1,b)$ with increasing $b$, so information, $-b/a$, $a-b$ and precision are nearly collinear. Our re-analysis (`results/e2_predictors.txt`) confirms this. Pooled, $-b/a$ gives 0.868 (skill) and 0.857 (knowledge) against 0.894 and 0.830 for information. Within budgets, $-b/a$ averages 0.978 and 0.970 against 0.938 and 0.970. The $p$-values treated non-independent cells as independent and should not be used. The E2 bit counts use each round's *marginal* $q$, which by Lemma 3 is an upper bound on the conditional quantity. E6 replaces this analysis.

**What E2 does not show.** Its verifier is a free exact oracle on both tasks. The knowledge task's large multiplier ($M=35.9$) shows only that an oracle *storing all 400 facts* transmits them. This is consistent with Prop. 4 ($L=400\times3=1200$ bits) but is not evidence of a knowledge/skill asymmetry. E6 tests the asymmetry directly.

### 5.4 E6: Verifier design. What predicts gain, and what does a verifier cost?

**Part A: decorrelated grid** (`e6_verifier_design.py`). Same learner and loop as E2, at skill $n_0=64$ and knowledge $n_0=200$, with 5 seeds. The grid is $a\in\{0.25,0.5,1\}\times b/a\in\{0,0.2,0.5,0.8\}$ (12 settings), which separates information (which grows with $a$ at fixed $b/a$) from $b/a$. Information is computed both from marginal $q$ and from per-proposal $q_j=$ the learner's own probability of its sampled label, a proxy for Lemma 3's conditional term. There are 120 runs.

| predictor (Spearman over 12 settings, seed-averaged) | skill | knowledge |
|---|---|---|
| $\sum_j\mathcal I_{a,b}(q_j)$, conditional | 0.783 | 0.811 |
| $\sum$ marginal-$q$ information | 0.783 | 0.853 |
| $a-b$ | 0.863 | 0.821 |
| $-b/a$ (ties: 4 groups of 3) | 0.972 | 0.972 |
| precision of accepted stream | 0.993 | 0.958 |

The partial rank correlation of gain with $b/a$ given conditional information is 0.993 (skill) and 0.936 (knowledge). The partial correlation of gain with information given $b/a$ is 0.948 and 0.497. In 2,000 seed-bootstrap resamples, information never ranked gain better than $b/a$ ($P=0.000$ on both tasks; `results/e6_summary.txt`). At fixed $b/a=0$, multiplying $a$ by four multiplies information by about 8 (skill: 0.94 to 7.62 kbits) but raises the gain only from 0.359 to 0.389. At fixed $a=1$, raising $b/a$ from 0 to 0.8 cuts the gain from 0.389 to 0.027.

*Interpretation.* With proposals plentiful (16,000 over 8 rounds), the binding constraint is the *purity* of the accepted stream, not the number of bits. This is what the urn rate predicts: $a$ rescales time, and $b/a$ sets the asymptotic error. Lemma 3 and Prop. 4 remain valid *ceilings*, but they do not predict gains in this regime. We withdraw the earlier prediction P4 ("gains are rank-ordered by information") and replace it with P4′ below. On the knowledge task, $b/a=0.8$ verifiers *lower* accuracy (gains −0.041 to −0.058) even though they carry positive information. The accepted wrong labels are concentrated on items the learner has never seen.

**Part B: verifier cost.** Each verifier is given an explicit description length.
- *Knowledge, lookup verifier:* stores the labels of a random fraction $\kappa$ of the 400 items ($3\kappa\cdot400$ bits). It is exact on covered items and rejects everything else.
- *Skill, program verifier:* the teacher $(W^\star,b^\star)$ quantized to $\beta$ bits per parameter ($168\beta$ bits). It accepts iff the quantized teacher agrees with the proposed label. Its errors are systematic, not independent noise.

There are 5 seeds per level and 50 runs.

| verifier | bits | acc$_0$ → acc$_8$ (s.d.) | gain | empirical $(a,b)$ | ceiling |
|---|---|---|---|---|---|
| knowledge lookup, $\kappa=0.125$ | 150 | 0.769 → 0.785 (0.025) | +0.015 | (0.30, 0) | coverage bound 0.803 |
| knowledge lookup, $\kappa=0.25$ | 300 | 0.769 → 0.753 (0.019) | −0.016 | (0.49, 0) | 0.830 |
| knowledge lookup, $\kappa=0.5$ | 600 | 0.769 → 0.819 (0.025) | +0.050 | (0.73, 0) | 0.890 |
| knowledge lookup, $\kappa=1$ | 1200 | 0.769 → 0.951 (0.014) | +0.182 | (1, 0) | 1.000 |
| skill program, $\beta=1$ | 168 | 0.547 → 0.560 (0.019) | +0.012 | (0.82, 0.60) | verifier agreement 0.552 |
| skill program, $\beta=2$ | 336 | 0.547 → 0.659 (0.039) | +0.111 | (0.87, 0.52) | 0.652 |
| skill program, $\beta=3$ | 504 | 0.547 → 0.834 (0.009) | +0.286 | (0.93, 0.29) | 0.836 |
| skill program, $\beta=4$ | 672 | 0.547 → 0.901 (0.011) | +0.353 | (0.96, 0.14) | 0.921 |
| skill program, float32 | 5376 | 0.547 → 0.935 (0.014) | +0.388 | (1, 0) | 1.000 |

*Findings.*
(1) At comparable description length (504–672 bits), the program verifier gives the skill learner +0.29 to +0.35. The lookup verifier gives the knowledge learner +0.05 at 600 bits. The lookup verifier has to store all 400 facts (1,200 bits) to reach +0.18, and its required size grows linearly with the number of facts, while the program verifier's size is independent of the size of the input space. This is the asymmetry of Prop. 4, now measured rather than argued.
(2) The skill learner converges to the verifier's own agreement with the truth (0.834 vs 0.836; 0.901 vs 0.921; 0.659 vs 0.652), an instance of Yi et al.'s (2025) convergence to the verifier's knowledge center.
(3) Knowledge accuracy stays well below the union-coverage bound at partial coverage (0.819 vs 0.890 at $\kappa=0.5$), and the $\kappa=0.25$ result is within seed noise of zero gain. In this learner the lookup verifier is a poor channel even for facts it holds, because uniform proposals for covered items are only rarely correct at first ($q\approx1/8$ on unseen items).

The comparison holds the learner and loop fixed but not the human budgets ($n_0$ differs by task), so gains are compared within, not across, absolute scales.

### 5.5 E3: Urn exponent (simulation of Prop. 5)

We ran 200 replicate urns of $2\times10^5$ proposals each, starting from 3 correct and 3 wrong labels (`e3_urn_scaling.py`, `results/e3_urn.json`, Fig. 3). For speed the pool is frozen within blocks of $n/200$ proposals between checkpoints, so this is an approximate simulation. Fitted exponents for $t\ge2000$, against $1-b/a$:
- $(1,0)$: 1.001 vs 1
- $(1,0.25)$: 0.751 vs 0.75
- $(1,0.5)$: 0.498 vs 0.5
- $(0.5,0.25)$: 0.496 vs 0.5
- $(1,1)$: 0.0003 vs 0
- $(1,0.75)$: 0.216 vs 0.25 (slow pre-asymptotic convergence)

The $(0.5,0.25)$ case shows that only $b/a$ sets the exponent. This is the same structure that E6 Part A finds in a learner that generalizes.

### 5.6 E4: Horizon arithmetic with labelled anchors

With dataset growth $g_D$ per year, a multiplier $M$ moves the date at which human data binds by $\Delta t=\ln M/\ln g_D$. We take $g_D=2.0$–$2.24$, derived from 4–5×/yr compute growth (Sevilla & Roldán, 2024) and $D\propto C^{1/2}$ (Hoffmann et al., 2022). This is an assumption, not a measured dataset growth rate. Results are in `results/e4_horizon.json` and Fig. 4.

| multiplier source (provenance) | $M$ | delay at 2.0×/yr | at 2.24×/yr |
|---|---|---|---|
| WRAP rephrasing, ~3× pre-training speed-up (Maini et al., 2024) [external, speed-up not strict $M$] | 3 | 1.6 yr | 1.4 yr |
| 4 epochs of repetition ≈ fresh data (Muennighoff et al., 2023) [external] | 4 | 2.0 yr | 1.7 yr |
| one-third rephrased mix, 5–10× speed-up to equal val. loss (Kang et al., 2025) [external, speed-up not strict $M$] | 5–10 | 2.3–3.3 yr | 2.0–2.9 yr |
| E5 tiny LM: saturated paraphrase multiplier ($\alpha=1$, $N_0=2000$) [this paper] | 2.92 | 1.5 yr | 1.3 yr |
| E5 tiny LM: largest paraphrase multiplier observed [this paper] | 3.75 | 1.9 yr | 1.6 yr |
| stock at upper end of 90% CI (1000T vs 300T) (Villalobos et al., 2024) [external] | 3.3 | 1.7 yr | 1.5 yr |
| hypothetical memoryless ceiling $1/s$, $s=0.01$ [model only] | 100 | 6.6 yr | 5.7 yr |

The external numbers are training speed-ups to a target loss in specific setups. They are not human-data multipliers in our strict sense (same learner, same loss, matched human budget), and a speed-up can over- or understate $M$. The range they span (about 3–10×) maps to about 1.4–3.3 years. Because $\Delta t$ is logarithmic in $M$, even a tenfold error in $M$ shifts the delay by only about 3 years.

## 6 Discussion

**Answer to the motivating question, at the strength the evidence supports.**

- *Knowledge.* Without an external signal, synthetic data adds no information (Fact 1). It can still help a learner absorb what the corpus contains. In our tiny transformer one paraphrase was worth about one fresh occurrence, and diverse paraphrases beat repetition. That help stops at the coverage ceiling, which bound exactly in E5. Beyond coverage, new facts must come from somewhere. A verifier can supply them only by storing them (Prop. 4, E6 Part B), so it is retrieval of human data under another name. The $1/s$ ceiling and the multiplier's independence from the data distribution are *not* robust: they fail under threshold storage (Prop. 2b) and in E5.
- *Skills with compact verifiers.* A verifier whose description is short relative to the inputs it can label can inject information without human answers (E6: 504–672 bits gave gains of 0.29–0.35). In our experiments the rate at which that information turns into accuracy is governed by the purity $b/a$ of the accepted stream, not by the bit count. The learner converges to the verifier's own accuracy. The scarce resources become verifier accuracy and coverage, not human tokens.
- *Horizon.* Measured rephrasing gains of about 3–10× correspond to about 1.4–3.3 years of delay under compute-optimal growth assumptions. Verified skills are not limited by the human stock in this accounting, but by verifiers.

**Learner dependence.** Prop. 2 predicted that more sample-efficient learners gain less from rephrasing. E5 refines this: fitted retention depends on the training regime (it rises with corpus size under fixed epochs), and the binding limit was coverage. So the prediction is better stated as "rephrasing gains shrink as the plain learner approaches its coverage ceiling" (finding 3 of E5).

### 6.1 Predictions and protocols (updated)

- **P1 (coverage ceiling).** For any source-only pipeline, closed-book accuracy on facts absent from the source stays at the answer prior. *Status:* confirmed in E5 (tiny transformer). *Protocol for LLMs:* plant held-out facts in a continued-pre-training corpus and test recall without retrieval.
- **P2 (headroom, not 1/s).** The paraphrase multiplier at a given $r$ scales with the headroom $A_{\text{cov}}-A(N_0)$ and is not invariant in $(\alpha,N)$. *Status:* consistent with E5. *Falsified if* $M(r)$ is constant across corpus sizes for a learner with a measurable threshold-like storage law.
- **P3 (urn exponent).** In verified self-training with a tunable verifier (for example code with test suites of controlled false-positive rate), the residual error decays as (proposals)$^{-(1-b/a)}$ in accumulate mode. *Falsified if* the exponent depends on $a$ at fixed $b/a$.
- **P4′ (purity predicts gain; replaces P4).** At a fixed proposal budget that is large relative to the human budget, gains are rank-ordered by $b/a$ (or precision) rather than by verifier information. *Status:* supported in E6 for a convex learner.
- **P5 (verifier size).** The number of facts a verified pipeline can add beyond its corpus is at most $L/\log_2k$ for a verifier of description length $L$. *Status:* implied by Prop. 4 and consistent with E6 Part B.

**Pre-registered LLM protocol (not run here; no GPU).** (1) Estimate $s$ for 2–3 model sizes from planted-fact recall versus exposure count. Fit both geometric and threshold laws, and report how $s$ depends on training length. (2) Paraphrase with $r\in\{0,1,3,7\}$ and compare with repetition. (3) Invert each model's plain curve to get $M$, and check saturation at $A_{\text{cov}}$. (4) For code, estimate $(a,b)$ of unit-test verifiers by mutating test suites, and test P3/P4′.

## 7 Limitations

- **Tiny neural evidence only.** E5 uses a 0.67M-parameter transformer, synthetic facts, 8 templates and 4 epochs. Its multipliers (1.5–3.8) are not estimates for LLMs. Fixed-epoch training ties corpus size to optimizer steps, which partly explains why fitted $s$ grows with $N$. Paraphrases differ only in fillers and word order, which is far less diverse than real rephrasing.
- **Convex learner for verification.** E2 and E6 use logistic regression, 8 rounds, and 3 or 5 seeds. We did not run STaR/ReST-style loops on neural models, and we have no external baselines for verified selection (e.g. Feng et al.'s setting). E6's bootstrap CIs are narrow because the 12 settings are widely spaced. They do not cover other learners.
- **The verifier-cost comparison is stylised.** Lookup and quantized-program verifiers are two points in a large design space, and the human budgets differ between tasks. Prop. 4 bounds information, not accuracy.
- **Theory depth.** Fact 1, Lemma 3, Remark 1 and Prop. 5 are standard or direct applications (data processing, chain rule, strong law, randomly reinforced urns). Prop. 2 restates its own assumption. Prop. 2b and Prop. 4 are short arguments. Prop. 2b's asymptotics are for a stylised threshold law that E5 does not strongly support (E5 storage was close to memoryless with at most $m=2$).
- **Horizon arithmetic.** It rests on compute-optimal scaling, constant compute growth, and external speed-ups that are not strictly human-data multipliers. Over-training, multimodal and private data would change the numbers.
- **No measurement on deployed systems.** We estimate no $s$, $a$ or $b$ for any real LLM pipeline.

## 8 Conclusion

Phrased as an exchange rate, the question "can synthetic data take over?" separates into parts. Without external signals, synthetic data only improves how well the human corpus is absorbed. In a small transformer, that improvement was about one fresh occurrence per paraphrase, larger for diverse paraphrases than for repetition, and capped exactly by the corpus's coverage. The memoryless model's $1/s$ ceiling and its distribution-invariance did not hold. New facts need new information, and a verifier provides it only by storing it. Verified skills escape the human-data limit when a compact verifier exists, but gains then follow the purity $b/a$ of the accepted stream (the urn rate), not the verifier's bit count, and saturate at the verifier's own accuracy. Measured rephrasing gains of about 3–10× buy roughly 1.4–3.3 years of horizon under current scaling assumptions. The open measurement problem is now specific: coverage, retention as a function of training regime, and verifier purity $b/a$, measured on real LLM pipelines.

## Reproducibility

`code/README.md` lists every command. Total CPU time is about 1.5–2 hours on 4 shared cores: E2 18–45 min, E5 ~17 min, E6 ~28 min, the rest under a minute. Seeds are fixed. Figures: `results/fig1_knowledge.png`, `fig2_selftrain.png`, `fig3_urn.png`, `fig4_horizon.png`, `fig5_tinylm.png`, `fig6_verifier_design.png`.

## References

Alemohammad, S., et al. (2024). Self-Consuming Generative Models Go MAD. *ICLR 2024*. arXiv:2307.01850.

Allen-Zhu, Z., & Li, Y. (2024). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235, 1067–1077. arXiv:2309.14316.

Allen-Zhu, Z., & Li, Y. (2025). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. *ICLR 2025*. arXiv:2404.05405.

Bertrand, Q., Bose, A. J., Duplessis, A., Jiralerspong, M., & Gidel, G. (2024). On the Stability of Iterative Retraining of Generative Models on their own Data. *ICLR 2024*. arXiv:2310.00429.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Dohmatob, E., Feng, Y., Yang, P., Charton, F., & Kempe, J. (2024). A Tale of Tails: Model Collapse as a Change of Scaling Laws. *ICML 2024*, PMLR 235, 11165–11197. arXiv:2402.07043.

Feng, Y., Dohmatob, E., Yang, P., Charton, F., & Kempe, J. (2025). Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification. *ICLR 2025*. arXiv:2406.07515.

Gan, Z., & Liu, Y. (2025). Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective. *ICLR 2025*. arXiv:2410.01720.

Gerstgrasser, M., et al. (2024). Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. arXiv:2404.01413.

Gulcehre, C., et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998.

Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.

Huang, A., Block, A., Foster, D. J., Rohatgi, D., Zhang, C., Simchowitz, M., Ash, J. T., & Krishnamurthy, A. (2025). Self-Improvement in Language Models: The Sharpening Mechanism. *ICLR 2025*. arXiv:2412.01951.

Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. *ICML 2023*, PMLR 202, 15696–15707. arXiv:2211.08411.

Kang, F., Ardalani, N., Kuchnik, M., Emad, Y., Elhoushi, M., Sengupta, S., Li, S.-W., Raghavendra, R., Jia, R., & Wu, C.-J. (2025). Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls. *Proceedings of EMNLP 2025 (Main)*, 10739–10758. arXiv:2510.01631.

Kazdan, J., Schaeffer, R., Dey, A., Gerstgrasser, M., Rafailov, R., Donoho, D. L., & Koyejo, S. (2025). Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World. *ICML 2025*, PMLR 267. arXiv:2410.16713.

Maini, P., Seto, S., Bai, H., Grangier, D., Zhang, Y., & Jaitly, N. (2024). Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling. *ACL 2024*, 14044–14072. arXiv:2401.16380.

May, C., & Flournoy, N. (2009). Asymptotics in response-adaptive designs generated by a two-color, randomly reinforced urn. *The Annals of Statistics*, 37(2), 1058–1078. doi:10.1214/08-AOS596.

Muennighoff, N., Rush, A. M., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., & Raffel, C. (2023). Scaling Data-Constrained Language Models. *NeurIPS 2023*. arXiv:2305.16264.

Muliere, P., Paganoni, A. M., & Secchi, P. (2006). A randomly reinforced urn. *Journal of Statistical Planning and Inference*, 136, 1853–1874.

Sevilla, J., & Roldán, E. (2024). Training Compute of Frontier AI Models Grows by 4-5x per Year. Epoch AI. https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755–759.

Song, Y., Zhang, H., Eisenach, C., Kakade, S., Foster, D., & Ghai, U. (2025). Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models. *ICLR 2025*. arXiv:2412.02674.

Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., & Hobbhahn, M. (2024). Will we run out of data? Limits of LLM scaling based on human-generated data. arXiv:2211.04325 (v2).

Yang, Z., Band, N., Li, S., Candès, E., & Hashimoto, T. (2025). Synthetic continued pretraining. *ICLR 2025*. arXiv:2409.07431.

Yi, B., Liu, Q., Cheng, Y., & Xu, H. (2025). Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence. arXiv:2510.16657.

Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning. *NeurIPS 2022*. arXiv:2203.14465.
