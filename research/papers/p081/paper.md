# Headroom Budgets: Most of the 4–5×/Year Growth in Frontier Training Compute Comes From Bounded Factors, and This Fixes When It Ends

**Target venue:** *Nature Machine Intelligence* (Analysis article). A shorter version would also fit the AAAI/ACM Conference on AI, Ethics, and Society (AIES).

**Code and data:** `code/` (run `bash code/run_all.sh`, about 30 s on 4 CPU cores); all outputs are in `results/`.

---

## Abstract

From 2020 to 2025, the compute used to train the largest AI models grew about 4–5× per year. Whether this can go on is usually argued one bottleneck at a time: power, chips, money, data or latency. We argue that this framing hides a simpler structure. We write frontier training compute as two accounting identities, one financial and one physical. Each identity is a product of *secular* factors, which can keep growing for a long time (world GDP, hardware price-performance, energy efficiency, deliverable power), and *bounded share* factors, which have ceilings (the share of GDP spent on AI compute, the share of that spending used for the largest run, precision and utilisation gains, and run duration). We prove three results. (i) The long-run growth rate is the smallest of the channels' sums of secular growth rates; bounded factors do not affect it. (ii) The total above-trend gain a channel can deliver, its *headroom budget*, depends only on the ratios between the caps and today's values, not on how fast those ratios are closed. Faster growth now therefore moves the slowdown earlier and does not raise the eventual level. (iii) Lengthening runs saves power but has no effect on cost, so the power required per run speeds up once run durations stop growing. We calibrate the model with verified published estimates and add explicit assumptions for the caps. About 78% (90% interval 75–82%) of the recent log-growth comes from bounded factors. The secular rate is about 1.38× per year. The financial headroom budget is 2.6 (2.0–3.2) orders of magnitude, which lasts about 5 (3.8–6.5) years at today's above-trend pace. In a Monte Carlo over all parameters, the median year in which annual growth first falls below 3× is 2028.8, and below 2× it is 2030.2 (90% interval 2028.7–2033.0). The largest run reaches a median of 10^29.1 FLOP in mid-2030. As a check, the model's implied growth in power per run (2.3×/yr) matches Epoch AI's independent estimate (2.2×/yr). A variance-based sensitivity analysis shows that the dominant uncertainty depends on the horizon. Power deliverable in 2030 dominates the 2030 level. A poorly characterised data-movement ceiling dominates the 2035 level (total-order Sobol index 0.78). Surprisingly, weaker AI revenue growth lowers the 2030 level by up to 0.27 OOM but slightly *delays* the deceleration. We end with dated, falsifiable predictions and a test protocol.

## 中文摘要

2020–2025年间，最大规模AI模型的训练算力以每年约4–5倍的速度增长。这一增长能否持续，通常按单一瓶颈分别讨论：电力、芯片、资金、数据或延迟。本文认为，这种讨论方式掩盖了一个更简单的结构。我们把前沿训练算力写成两个核算恒等式，一个是金融恒等式，一个是物理恒等式。每个恒等式都是两类因子的乘积。第一类是可以长期增长的“长期因子”，包括世界GDP、硬件性价比、能效和可交付电力。第二类是有上限的“份额因子”，包括AI算力投资占GDP的比例、最大单次训练占投资的比例、精度与利用率收益，以及训练时长。我们证明三个结论。(i) 长期增长率等于各通道长期因子增长率之和中的最小值，与份额因子无关。(ii) 每个通道能提供的超趋势总增量，即“余量预算”，只取决于上限与当前值之比，而与逼近上限的速度无关。因此，当前增长越快，减速来得越早，但最终水平不会更高。(iii) 延长训练时长可以节省电力，但不影响成本；一旦时长停止增长，单次训练所需电力的增速会加快。我们用经过核实的公开估计校准模型，并对各上限给出明确假设。结果显示，近期对数增长中约78%（90%区间75–82%）来自份额因子。长期增长率约为每年1.38倍。金融通道的余量预算为2.6（2.0–3.2）个数量级，按当前超趋势速度约可维持5（3.8–6.5）年。在对全部参数的蒙特卡洛模拟中，年增速首次跌破3倍的中位年份为2028.8年，跌破2倍的中位年份为2030.2年（90%区间2028.7–2033.0）。2030年中，最大训练规模的中位数为10^29.1 FLOP。作为检验，模型推出的单次训练功率增速（每年2.3倍）与Epoch AI的独立估计（每年2.2倍）一致。基于方差的全局敏感性分析表明，主导不确定性的因素随时间跨度变化：2030年的算力水平主要取决于2030年可交付的电力；2035年的水平主要取决于一个尚缺乏刻画的数据传输上限（总效应Sobol指数0.78）。一个出乎意料的结果是：AI收入增长较弱时，2030年水平最多降低0.27个数量级，但减速反而略为推迟。最后，我们给出带日期的可证伪预测和检验方案。

---

## 1 Introduction

From 2010 to 2024, the compute used to train notable machine learning models grew about 4–5× per year. For frontier language models since 2020 the figure is about 5× per year (Sevilla & Roldán, 2024; Epoch AI, 2026a). This is much faster than hardware price-performance, which doubles about every 2–3 years (Hobbhahn & Besiroglu, 2022), and much faster than the economy. The question "can compute demand keep growing exponentially, and where is the bottleneck?" has so far been answered mostly one constraint at a time. The literature covers chip and power feasibility to 2030 (Sevilla et al., 2024), electricity (EPRI & Epoch AI, 2025; IEA, 2025), training data (Villalobos et al., 2024), data movement and latency (Erdil & Schneider-Joseph, 2024), and cost (Cottier et al., 2024; Thompson et al., 2020). These studies establish that each constraint could plausibly be met up to some scale. They are less suited to two questions that matter for planning and policy:

1. **How much of the current growth rate is structurally transient?** A growth rate that combines a rising spending share with improving hardware will slow when the share stops rising, even if no physical wall is ever hit.
2. **Is "which bottleneck binds?" well posed, and what determines when growth slows rather than whether it slows?**

We give a compact, falsifiable answer. The key observation is that frontier training compute satisfies two accounting identities. One is financial: compute equals the amortised cost of the run times FLOP per dollar. The other is physical: compute equals power times duration times FLOP per joule. In each identity, some factors are *shares*, which are bounded above, and some are *secular*, meaning they can grow indefinitely at modest rates. This "growth accounting" view, standard in macroeconomics, has not to our knowledge been applied to frontier compute. It turns the open-ended bottleneck question into three tractable sub-questions: how big the remaining headroom is, how fast it is being used up, and what the secular floor is.

**Contributions.**

1. **A growth-accounting decomposition of the 4–5×/yr trend** (Section 5.1). We use only verified published rates. Secular factors (GDP and hardware price-performance) explain about 22% of current log-growth. Bounded factors explain about 78% (90% interval 75–82%). The implied long-run financial growth rate is 1.38×/yr (1.32–1.45).
2. **A formal model with three propositions** (Section 4). These cover the asymptotic growth rate (Proposition 1), *headroom invariance*: the eventual level does not depend on how fast the bounded factors grow, so faster growth now means an earlier slowdown (Proposition 2), and the role of run duration as a power-saving but cost-neutral factor (Proposition 3). Proposition 3 yields a consistency check against independent data. Our implied growth in power per run is 2.33×/yr (2.03–2.67), and Epoch AI's independent historical estimate is 2.2×/yr (EPRI & Epoch AI, 2025).
3. **A calibrated three-channel Monte Carlo with global sensitivity analysis** (Sections 5.2–5.6). We give distributions for the year growth falls below 3× and 2×, for the level reached in 2028–2040, and for which constraint binds over time. A Sobol analysis shows that the dominant uncertainty shifts with the horizon: from deliverable power (2030) to the data-movement ceiling (2035). It also shows that the slowdown date is mostly an *interaction* effect: the first-order indices sum to 0.33.
4. **A counter-intuitive conditional result on revenue** (Section 5.6). Over a wide grid of AI revenue growth paths, weaker revenue lowers the 2030 compute level by up to 0.27 OOM but *delays* the date at which growth drops below 2×/yr. This is a direct consequence of Proposition 2, and it means that "the growth rate slowed" and "the level fell short" are different empirical signals.
5. **Pre-registered, dated predictions and a test protocol** (Section 6). They can be checked against public trackers between 2027 and 2031.

We do not claim that frontier AI progress will stop. Algorithmic efficiency (Ho et al., 2024), inference-time compute and post-training are outside our quantity of interest, which is *physical training FLOP of the largest run*. We also do not claim our cap assumptions are right. Our claim is narrower: given the structure, the timing and level of the slowdown are set by a small number of quantities, and we identify which ones and how much each matters.

## 2 Related Work

**Compute trends.** Sevilla et al. (2022) documented a shift from Moore's-law growth to about 6-month doubling in the deep-learning era. Epoch AI's later estimate puts notable-model growth at 4.1×/yr (90% CI 3.7–4.6) from 2010 to 2024, and frontier models at 4–5×/yr (Sevilla & Roldán, 2024). Epoch's current trends page reports about 5×/yr for frontier language models since 2020 and chip price-performance doubling about every 2.2 years (Epoch AI, 2026a). The largest known run in mid-2025 was Grok 4, at about 5×10^26 FLOP, with estimated amortised cost of about $490M and energy of about 310 GWh (Epoch AI, 2025a). Training-run length for frontier LLMs has grown about 1.4×/yr (90% CI 1.3–1.5) (Epoch AI, 2025b). We use these as anchors and growth rates.

**Costs.** Cottier et al. (2024) estimate that the amortised hardware-plus-energy cost of frontier training runs has grown 2.4×/yr since 2016 (90% CI 2.0–2.9). The difference between 4–5×/yr compute growth and 2.4×/yr cost growth is what our accounting splits into hardware and residual terms. Thompson et al. (2020) argued earlier that deep learning's compute dependence would become economically and environmentally unsustainable. Our model makes the economic part of that argument quantitative and time-resolved.

**Feasibility studies of specific constraints.** Sevilla et al. (2024) conclude that 2×10^29 FLOP runs are likely feasible by 2030, that power is likely to bind first and chip manufacturing second, and that a latency wall lies at 3×10^30–10^32 FLOP. Erdil & Schneider-Joseph (2024) model data-movement limits. Utilisation drops noticeably above about 10^28 FLOP for 3-month runs, with a practical ceiling of 10^28–10^31 that more aggressive batch scaling could raise. Villalobos et al. (2024) estimate the effective stock of public human text at about 300T tokens (90% CI 100T–1000T), fully used between 2026 and 2032. Muennighoff et al. (2023) show that up to about 4 epochs of repetition cost little. For power, the EPRI–Epoch report (EPRI & Epoch AI, 2025) finds that frontier-run power has grown more than 2× per year, with 1–2 GW by 2028 and 4–16 GW by 2030 if trends continue. Pilz et al. (2025) find leading AI supercomputers' performance doubling every ~9 months, cost and power each doubling about yearly, and performance per watt improving 1.34×/yr. The IEA (2025) projects global data-centre electricity rising from about 415 TWh (2024) to about 945 TWh (2030), roughly 15%/yr. Our contribution is not a new estimate of any single constraint. It is a framework that puts them on a common scale (log-headroom and log-rate), proves how they interact, and shows which uncertainties matter at which horizon.

**Economics of AI investment and growth.** The GATE integrated assessment model (Erdil et al., 2025) has endogenous compute investment and finds that optimal investment could exceed 10% of world GDP. Nordhaus (2021) tests whether IT-driven growth is approaching a "singularity" and finds the key parameter to be the substitutability between information and conventional inputs. Our model is far simpler than GATE and is not a growth model. It treats GDP as exogenous, so it addresses the "pre-transformative" regime in which AI compute spending is still a small share of output. The cap on the investment share is our stand-in for the question GATE models endogenously, and we vary it over an order of magnitude (1–10% of GDP).

**Revenue evidence.** Epoch AI reports that combined OpenAI and Anthropic revenue tripled in 2024 and grew more than fourfold in 2025, followed by an acceleration in 2026 (Epoch AI, 2026b). It also reports that Anthropic's compute spending has grown about 4× per year (Epoch AI, 2026c). Hyperscaler capital expenditure was reported at about $410B in 2025, with about $720–740B guided for 2026 (press compilations; see Section 3.3). We use these only to calibrate the initial growth of AI compute investment and in a revenue-constrained variant.

**Sensitivity analysis.** We use variance-based global sensitivity analysis with the Saltelli et al. (2010) first-order estimator and the Jansen (1999) total-effect estimator.

**Gap.** We are not aware of prior work that (a) decomposes the frontier compute trend into bounded and secular components with uncertainty, (b) proves that the eventual level does not depend on the speed of share growth, or (c) attributes the uncertainty in *when* growth slows to specific parameters with a global sensitivity analysis.

## 3 Model

### 3.1 Two accounting identities

Let $C(t)$ be the training compute (FLOP) of the largest training run that starts at time $t$.

**Financial identity.** Let $K$ be the run's amortised hardware-plus-energy cost (Cottier et al.'s definition). Then
$$C = K \cdot q, \qquad q = \text{FLOP per amortised dollar}.$$
With $N$ chips of peak throughput $r$ and price $p$, utilisation $u$, duration $T$ and amortisation life $L$, we have $C = N r u T$ and $K \approx N p\,T/L$. So $q \approx (r/p)\,u\,L$. **Run duration cancels**: under amortised accounting a longer run costs proportionally more. We write $r/p = h$ (price-performance, secular). The remaining factors, including utilisation, lower numerical precision, and the gap between list price and the effective price paid, go into a residual $x$. We assume $x$ is bounded. We decompose the cost further as
$$K = \varphi \cdot \sigma \cdot Y,$$
where $Y$ is world GDP, $\sigma = I/Y$ is AI compute investment $I$ as a share of GDP, and $\varphi = K/I$ is the largest run's share of that investment. Thus
$$C_{\text{fin}}(t) = \varphi(t)\,\sigma(t)\,Y(t)\,q_0\,e^{g_h t}\,x(t). \tag{1}$$

**Physical identity.** With average facility power $P$ (W), duration $T$ (s) and delivered FLOP per joule $e$,
$$C_{\text{pow}}(t) = P_{\max}(t)\, T(t)\, e(t). \tag{2}$$
Here $P_{\max}(t)$ is the most power that can be delivered to a single (possibly geographically distributed) run. The power channel is what *can* be run given the power available. The financial channel is what *can be paid for*.

**Engineering ceiling.** $C_{\text{lat}}$ is a ceiling from data movement and latency (Erdil & Schneider-Joseph, 2024; Sevilla et al., 2024).

The realised compute is
$$C(t) = \min\{C_{\text{fin}}(t),\; C_{\text{pow}}(t),\; C_{\text{lat}}\}. \tag{3}$$
We do not model chip supply as a separate channel. A chip-supply shortage shows up as a higher effective price (lower $h$) or as a cap on $\varphi$, which is a simplification we discuss in Section 8.

### 3.2 Bounded and secular factors

Each factor $z$ is one of two kinds:

- **Secular:** $z(t)=z_0 e^{b t}$, which grows indefinitely. Here this means $Y$ ($g_Y$), $h$ ($g_h$), $e$ ($g_e$) and $P_{\max}$ ($g_{P}$).
- **Bounded:** $s(t)=\min(s_0 e^{a t},\bar s)$, which grows at its current rate until it reaches a cap. Here this means $\sigma$, $\varphi$, $x$ and $T$.

The hard-cap form is deliberately simple. Smooth logistic saturation changes when growth slows by a fraction of the saturation width, but it leaves Propositions 1–2 unchanged (they depend only on limits). The two investment shares are calibrated jointly. Their combined growth must reproduce observed cost growth $g_K$: $(g_\sigma + g_Y) + g_\varphi = g_K$, where $g_\sigma + g_Y = g_I$ is the growth of investment. The residual's growth is identified from $g_x = g_C - g_K - g_h$.

### 3.3 Calibration

Table 1 lists every parameter. Each is marked **[S]** if its range comes from a verified source or **[A]** if it is an assumption. All ranges are sampled independently (uniform or log-uniform).

**Table 1. Parameters** (mid-2025 = $t_0$).

| Symbol | Range | Tag | Basis |
|---|---|---|---|
| $C_0$ | 3–8 ×10^26 FLOP | S | Grok 4 ≈ 5×10^26 (Epoch AI, 2025a) |
| $g_C$ (current) | ln 4 – ln 5 | S | 4–5×/yr (Sevilla & Roldán, 2024; Epoch AI, 2026a) |
| $g_K$ | ln 2.0 – ln 2.9 | S | 2.4×/yr, 90% CI 2.0–2.9 (Cottier et al., 2024) |
| $g_h$ | ln2/2.9 – ln2/2.0 | S | price-performance doubling 2.0–2.9 yr (Hobbhahn & Besiroglu, 2022; Epoch AI, 2026a: 2.2 yr) |
| $g_e$ | ln 1.30 – ln 1.45 | S | 1.34×/yr performance per watt for AI supercomputers (Pilz et al., 2025) |
| $g_Y$ | 2.5–3.5%/yr | S | world growth 3.2% (2025), 3.1% (2026) (IMF, 2025) |
| $g_T$ | ln 1.3 – ln 1.5 | S | 1.4×/yr, 90% CI 1.3–1.5 (Epoch AI, 2025b) |
| $g_I$ (initial) | 0.35–0.70 | A | 2025→2026 big-tech capex ≈ $410B → ≈$725B implies ≈0.57 (press compilations) |
| $\bar\sigma$ | 1–10% of GDP | A | GATE explores >10% (Erdil et al., 2025) |
| $\bar\varphi$ | 1–5% | A | Today about 0.12% ($490M / $410B) |
| $\bar x$ | 1.5–4× | A | e.g. FP8→FP4 = 2× |
| $\bar T$ | 180–365 days | A | Epoch argues runs above ~9 months are suboptimal (Epoch AI, 2025b) |
| $P_{\max}(2030)$ | 1–16 GW | S | 1–5 GW campuses (Sevilla et al., 2024); 4–16 GW trend demand (EPRI & Epoch AI, 2025) |
| $g_P$ | 0.10–0.40 | A | data-centre electricity +15%/yr (IEA, 2025); US AI power 5→50 GW by 2030 (EPRI & Epoch AI, 2025) |
| $C_{\text{lat}}$ | 10^29–10^32 | S | 10^28–10^31 for 3-month runs (Erdil & Schneider-Joseph, 2024); 3×10^30–10^32 (Sevilla et al., 2024). Lower range tested in Section 5.5 |

Fixed anchors are Grok 4 cost $K_0 = \$4.9\times10^8$ and energy $E_0 = 310$ GWh (Epoch AI, 2025a), a run length of $T_0 = 90$ days, $I_0 = \$4.1\times10^{11}$ and $Y_0 = \$1.15\times10^{14}$. $T_0$ is an assumption chosen to be consistent with 310 GWh at the 100–150 MW per frontier run reported by EPRI & Epoch AI (2025). $I_0$ is 2025 capex of Microsoft, Alphabet, Amazon and Meta from press compilations. It is used as a proxy for AI compute investment even though it includes non-AI capex. $Y_0$ is an order-of-magnitude assumption for world nominal GDP. These anchors set $\varphi_0 = 1.2\times10^{-3}$ and $\sigma_0 = 3.6\times10^{-3}$, and $e_0 = C_0/E_0$. Errors in $I_0$ or $Y_0$ move $\varphi_0$ and $\sigma_0$ in opposite directions and largely cancel in the product $\varphi\sigma$. They matter only through which cap binds first.

**Revenue-limited variant.** AI revenue grows at $g_R(t)=g_{R\infty}+(g_{R0}-g_{R\infty})e^{-t/\tau}$, where $g_{R0}\in[\ln 2.5,\ln 4]$ [S: 3× in 2024, >4× in 2025; Epoch AI, 2026b], $g_{R\infty}\in[0.05,0.20]$ [A] and $\tau\in[1.5,6]$ yr [A]. Investment grows at $g_I$ unless the ratio of investment to revenue (normalised to 2025) would exceed $\lambda_{\max}\in[1,4]$ [A]. If it would, investment grows with revenue.

## 4 Theory

Group the factors by channel $k\in\{\text{fin},\text{pow},\text{lat}\}$. Let $\mathcal{S}_k$ be the bounded factors and $\mathcal{G}_k$ the secular factors. Define the channel's secular rate $B_k=\sum_{j\in\mathcal G_k} b_j$, its active bounded rate $A_k(t)=\sum_{i\in\mathcal S_k} a_i\,\mathbb 1[t<\tau_i]$ with exhaustion times $\tau_i = \ln(\bar s_i/s_{i0})/a_i$, and its **headroom budget** $H_k = \sum_{i\in\mathcal S_k}\ln(\bar s_i/s_{i0})$. Then $\frac{d}{dt}\ln C_k = B_k + A_k(t)$ almost everywhere.

**Proposition 1 (asymptotic rate).** If all caps are finite and $t^\star=\max_i\tau_i$, then for $t>t^\star$ every channel is affine in $\ln C_k$ with slope $B_k$. Hence $\lim_{t\to\infty}\frac1t\ln C(t)=\min_k B_k$. This is the minimum over channels of the sum of secular rates and does not depend on any bounded factor.

*Proof.* For $t>t^\star$, $\ln C_k(t) = \ln C_k(0) + H_k + B_k t$. The minimum of finitely many affine functions is eventually the one with the smallest slope (ties do not affect the slope). ∎

With our calibration, $B_{\text{fin}} = g_Y + g_h \approx \ln 1.38$ and $B_{\text{pow}} = g_P + g_e \approx \ln(1.1\text{–}1.5)+\ln(1.3\text{–}1.45)$. A hard engineering ceiling has $B_{\text{lat}}=0$. So if the data-movement ceiling is truly fixed, single-run compute eventually stops growing. If engineering keeps raising it, the long-run rate is about 1.4×/yr, a Moore's-law-like rate rather than 4–5×.

**Proposition 2 (headroom invariance and front-loading).** Hold the caps and initial values fixed. For any $t>t^\star$, $C_k(t)$ does not depend on the bounded growth rates $\{a_i\}$. Increasing any $a_i$ weakly raises $C_k$ for $t<t^\star$, leaves it unchanged for $t>t^\star$, and lowers $\tau_i$ by $\ln(\bar s_i/s_{i0})\,\Delta(1/a_i)$. The total above-trend gain is fixed at $H_k$. Faster growth today buys the same orders of magnitude sooner and then decelerates sooner.

*Proof.* Each $\ln s_i(t) = \min(\ln s_{i0}+a_i t, \ln \bar s_i)$ is non-decreasing in $a_i$ and equals $\ln\bar s_i$ for $t\geq\tau_i$. Summing over factors and taking the minimum over channels (a monotone operation) gives the claims. ∎

This is elementary, but it has a consequence that is often missed in public discussion: *acceleration of investment is evidence of an earlier deceleration, not of a higher plateau*, unless the caps themselves move. It also means that the **speed** of the slowdown and the **level** reached are separate observables (Section 5.6). The numerical check in `results/theory_check.json` reproduces $\log_{10}C(2040)=31.4035$ for investment growth of 1.42×, 1.65× and 2.01×/yr, identical to the analytic value. Their 2028 values are 28.37, 28.57 and 28.64 (Fig. 7).

**Proposition 3 (duration saves power, not money).** Under amortised costing $q$ does not depend on $T$ (Section 3.1). The power the run requires is $P_{\text{req}} = C/(eT)$, so
$$g_{P,\text{req}} = g_C - g_e - g_T \quad\text{while } T<\bar T,\qquad g_{P,\text{req}} = g_C - g_e \quad\text{after}.$$
When duration stops growing, either the growth of power per run speeds up by a factor of $e^{g_T}\approx 1.4$, or compute growth slows by that factor. ∎

**Empirical check of Proposition 3.** Using only the source ranges in Table 1, the implied historical growth of power per run is $\exp(g_C-g_e-g_T)$ = **2.33×/yr (90% interval 2.03–2.67)**. EPRI & Epoch AI (2025) independently estimate 2.2×/yr. The agreement supports the identity and our choice that the residual $x$ (precision, utilisation, pricing) enters the financial channel but not the energy channel. When we let $x$ also raise FLOP per joule, the implied power growth falls to about 1.7×/yr, which is inconsistent with the observed 2.2×. We therefore use the former specification throughout. Once durations reach the cap, sustaining 4–5× compute growth would require 3.26× (2.90–3.66) per year growth in power per run.

**Corollary (naive runway).** If channel $k$ binds and its bounded factors all grow until exhausted, the years of above-trend growth left are at least $H_k / \max_t A_k(t)$. With all bounded rates active today, this is $H_k/A_k(0)$.

## 5 Results

All numbers below come from code in `code/` and were produced in this environment. JSON outputs are in `results/`. Unless stated, intervals are 5th–95th percentiles over parameter uncertainty (Table 1).

### 5.1 Growth accounting (Fig. 1; `results/accounting.json`)

Table 2 shows the decomposition of the 2020–2025 trend in log-growth per year ($2\times10^5$ draws).

| Component | Log-growth/yr, median (90%) | Share of total |
|---|---|---|
| World GDP (secular) | 0.030 (0.025–0.034) | 2.0% |
| Hardware FLOP/$ (secular) | 0.29 (0.24–0.34) | 19.5% |
| Spending share of GDP, $\varphi\sigma$ (bounded) | 0.85 (0.68–1.02) | 56.7% |
| Precision/utilisation/pricing residual $x$ (bounded) | 0.33 (0.11–0.54) | 21.7% |
| **Total** | 1.50 (1.40–1.60) ≈ 4.5× | 100% |

Secular factors account for **21.5% (18.0–25.4%)** of current growth, and bounded factors for **78.5% (74.6–82.0%)**. The financial channel's secular rate is **1.38× per year (1.32–1.45)**. Today's above-trend excess is 0.51 OOM/yr (0.46–0.56). The financial headroom budget is $H_{\text{fin}}/\ln 10$ = **2.6 OOM (2.0–3.2)**. Dividing the two gives a naive runway of **5.1 years (3.8–6.5)**, i.e. to about 2029–2032.

The individual exhaustion times (`results/theory_check.json`) are as follows. The residual $x$ runs out first: median 2.8 yr (1.2–8.3), and it is the first financial cap to bind in 71% of draws. The investment share $\sigma$ follows at 4.4 yr (2.2–7.9), first in 24% of draws. The largest-run share $\varphi$ comes last at 8.3 yr (4.5–27.8). Run duration reaches its cap after 3.3 yr (2.1–4.6), which by Proposition 3 tightens the power channel from about 2028–2029.

### 5.2 Trajectories and deceleration (Figs. 2, 4; `results/mc_summary.json`)

In the main (exogenous-investment) specification ($N=20{,}000$):

| Quantity | Median | 90% interval |
|---|---|---|
| $\log_{10}C$, mid-2028 | 28.46 | 28.11–28.77 |
| $\log_{10}C$, mid-2030 | 29.14 | 28.63–29.66 |
| $\log_{10}C$, mid-2035 | 30.02 | 29.14–30.78 |
| $\log_{10}C$, mid-2040 | 30.43 | 29.14–31.48 |
| First year trailing growth < 3×/yr | 2028.8 | 2027.4–2030.6 |
| First year trailing growth < 2×/yr | 2030.2 | 2028.7–2033.0 |

The median trailing 12-month growth factor is 4.46× (mid-2026), 4.22× (2027), 3.19× (2028), 2.47× (2029), 1.88× (2030), 1.73× (2031), 1.58× (2033) and 1.39× (2035). After about 2037, single-run compute stops growing in the median draw because the fixed data-movement ceiling binds (see Section 5.5 for what happens without it). Every draw falls below 2×/yr by 2045.

**External consistency.** Two outputs can be compared with independent forecasts that were not used in calibration. The 2030 median of 10^29.14 ≈ 1.4×10^29 FLOP is close to Sevilla et al.'s (2024) "2×10^29 likely feasible by 2030". The implied median power of the largest run in 2028 is 1.29 GW (0.67–2.31) (`results/predictions.json`), within EPRI & Epoch AI's (2025) "1–2 GW by 2028". These are consistency checks, not validation, because the sources share some inputs with ours.

### 5.3 Which constraint binds? (Fig. 3)

The share of draws in which each channel binds changes over time:

| Year | Financial | Power | Latency/data movement |
|---|---|---|---|
| 2028 | 73% | 27% | 0% |
| 2030 | 44% | 48% | 8% |
| 2035 | 29% | 27% | 44% |
| 2040 | 20% | 6% | 74% |

At the moment growth first drops below 2×/yr, the binding channel is power in 50% of draws, financial in 37% and latency in 13%. So **the question "which bottleneck?" has no single answer**. The answer shifts with the horizon, and around 2030 the financial and power channels are about equally likely to bind. This fits a market in which spending and power build-out are planned together. In our model the two channels are separate, but their caps are similar in size, so small changes in assumptions switch which one binds.

### 5.4 Global sensitivity (Fig. 5; `results/sobol.json`)

We ran Saltelli sampling with $N=8192$ base samples, 15 parameters and 139k model runs, with 200 bootstrap resamples for confidence intervals. The total-order indices $S_T$ are:

- **$\log_{10}C(2030)$:** $P_{\max}(2030)$ 0.45 [0.44, 0.47]; $\bar\sigma$ 0.21; $g_K$ 0.17; $C_0$ 0.14; $g_I$ 0.13. The sum of first-order indices is 0.76.
- **$\log_{10}C(2035)$:** $C_{\text{lat}}$ **0.78** [0.75, 0.81]. Everything else is ≤0.10.
- **Year growth < 2×:** $g_P$ 0.37; $P_{\max}(2030)$ 0.35; $g_K$ 0.28; $\bar\sigma$ 0.27; $g_I$ 0.26; $g_e$ 0.23. The sum of first-order indices is only **0.33**.

Three implications follow. (a) The near-term level depends mostly on power, plus the observable current growth rates. (b) Beyond about 2033 the level depends mostly on the parameter with the widest and least-studied range, the data-movement ceiling. That makes it the highest-value target for measurement. (c) The *timing* of the slowdown is mostly an interaction effect: two-thirds of its variance cannot be attributed to any single parameter. This is a formal version of "it depends on which bottleneck binds", and it predicts that single-constraint analyses will disagree about timing even when they agree on every individual input.

### 5.5 Robustness to the engineering ceiling (`results/mc_summary_clatlow.json`, `results/mc_summary_nolat.json`)

- **Lower ceiling** ($C_{\text{lat}}\in[10^{28},10^{31}]$, Erdil & Schneider-Joseph's 3-month-run range): the 2030 median is 10^28.94, the 2035 median 10^29.48, and growth falls below 2× in 2029.7 (2028.3–2032.2). Latency is binding in 38% of draws by 2030.
- **No ceiling:** the 2035 median is 10^30.31 and the 2040 median 10^31.23 (30.50–31.92). Growth falls below 2× in 2030.3 (2028.7–2033.8), and long-run growth settles at **1.40× per year** by 2045. This matches Proposition 1's $\min_k B_k$.

So the slowdown date is robust to the ceiling (±0.5 yr), but the 2035–2040 level depends heavily on it (about 1–1.7 OOM).

### 5.6 Revenue constraints: lower level, later slowdown (Fig. 6; `results/revenue_stress.json`)

With the priors above, the revenue-limited variant is almost identical to the main run (2030 median 10^29.13, slowdown year 2030.2). The reason is that recent AI revenue growth (3–4× per year) is faster than investment growth (1.4–2× per year), so the ratio of investment to revenue *falls* in the near term and the revenue constraint does not bind before the share caps do. To find out when revenue *would* matter, we stress-tested a grid of initial revenue growth from 1.25× to 4× per year and decay times $\tau$ from 1 to 8 years, with $\lambda_{\max}=1.5$ and long-run revenue growth of 10%/yr. In the most pessimistic cell (1.25× per year, $\tau=1$), the 2030 median falls from 10^29.13 to **10^28.86** (−0.27 OOM). The median slowdown date, however, moves *later*, from 2030.2 to 2030.4 (up to 2030.75 at $\tau=8$). This follows from Proposition 2: if investment grows more slowly, the financial headroom is used up more slowly, so the above-trend phase is lower but lasts longer.

**Implication for measurement.** Analysts who watch for "a slowdown in compute growth" as a sign of weak AI economics are watching the wrong signal. Weak economics show up first as a *lower level* compared with the counterfactual path, while the growth rate stays above 2× for longer.

## 6 Pre-registered Predictions and Test Protocol

All probabilities come from the main specification (`results/predictions.json`). "Largest run" means the largest training compute estimate for a publicly known model in Epoch AI's database at the stated date.

| # | Prediction | Model probability / interval |
|---|---|---|
| P1 | Largest run exceeds 10^28 FLOP by mid-2028 | 0.99 |
| P2 | Growth in the year to mid-2028 is ≥3× | 0.58 (median 3.2×; 90%: 2.1–4.6×) |
| P3 | Largest run exceeds 10^29 FLOP by mid-2030 | 0.67 |
| P4 | Largest run does **not** exceed 10^30 FLOP by mid-2030 | 0.999 |
| P5 | Trailing annual growth falls below 2× before mid-2030 / before mid-2032 | 0.60 / 0.93 |
| P6 | Largest run does **not** exceed 10^31 FLOP by mid-2035 | 0.99 |
| P7 | Power drawn by the largest run in 2030 is 1.1–8.8 GW (90%), median 2.7 GW | interval |
| P8 (structural) | Within about 1 year of frontier run durations levelling off, either power per run grows ≥2.9×/yr or compute growth falls by ≥1.3× | Proposition 3 |
| P9 (structural) | Faster capex growth (a higher $g_I$) is followed by an *earlier*, not later, drop in compute growth, unless an announced cap (e.g. a multi-GW site or a capex/GDP level) is raised at the same time | Proposition 2 |

**Test protocol.**

1. **Data.** Use (i) training compute, cost and duration for the top-10 runs from Epoch AI's model database; (ii) hyperscaler and AI-lab capex from company filings; (iii) world GDP from the IMF WEO; and (iv) per-site power from EPRI/IEA or company disclosures. Update twice a year.
2. **Identity check.** Each half-year, compute the residual $\Delta\ln C-\Delta\ln K-\Delta\ln h$. The model predicts that this residual (the $x$ term) declines toward 0 within about 3 years. If it stays above 0.2/yr through 2029, the boundedness assumption for $x$ is rejected.
3. **Level check.** If the realised $\log_{10}C$ at mid-2028 lies outside [28.11, 28.77], the joint calibration is rejected. The cause is then attributed by re-estimating $g_K$ and $g_I$ from the new data and re-running `code/run_all.sh`.
4. **Cap tracking.** Track $\sigma$ (AI compute investment/GDP) and $\varphi$ (largest-run cost/investment). Under Proposition 2, an announced *increase in caps* (for example a sovereign commitment of several percent of GDP) should move the predicted plateau. An increase in *rates* should not.
5. **Decision rule for P5.** Use the trailing 12-month ratio of the largest-run compute between consecutive frontier releases, smoothed with a 6-month window. A sustained value ≥3× through mid-2031 would contradict the model at roughly the 95% level and would mean that at least one cap is larger than our prior allows.

## 7 Discussion

**Answer to the motivating question.** Can compute demand keep growing exponentially? In our framework, *physical training compute of the largest run can keep growing exponentially only at roughly the secular rate*, about 1.4× per year. That is a Moore's-law-like rate, not the current 4–5× per year. The current rate is mostly (about 78%) driven by shares that are rising toward ceilings. The question "where is the bottleneck?" has a horizon-dependent answer. In the late 2020s the financial and power channels are about equally likely to bind. In the 2030s the data-movement and latency ceiling dominates *if* it is as low as current estimates suggest. The quantity that matters most for the 2035 level, $C_{\text{lat}}$, is also the one with the widest range: three orders of magnitude.

**Why the timing is robust but the level is not.** The slowdown date is mainly set by quantities we can observe today: the current growth rates $g_K$ and $g_I$, plus near-term power. It is anchored by the size of the headroom budget, which spans only about 1.2 OOM across the prior (2.0–3.2). The level after 2033 depends on caps that are poorly understood. So the forecasting community can be fairly confident that growth will slow by the early 2030s and should be much less confident about where single-run compute ends up.

**What this does *not* say about AI progress.** Effective compute also includes algorithmic progress: Ho et al. (2024) estimate a halving of required compute every 1–2 years for pretraining. It also includes the growing compute used for experiments, inference and reinforcement learning. Epoch AI reports that final training runs were about 10% of OpenAI's 2024 R&D compute (Epoch AI, 2025c). If the largest-run share $\varphi$ is close to its cap, labs can still increase total compute (σ) and use it for things other than one big run. Our result is about one widely tracked scalar, not about capabilities.

**Policy relevance.** Compute-threshold regulation (for example thresholds at 10^25–10^26 FLOP) implicitly assumes continued 4–5× growth. Under our model, a threshold set today at 10^29 would be crossed with probability 0.67 by mid-2030. A threshold at 10^30 would almost certainly not be crossed by then, and would be crossed with probability about 0.51 by mid-2035. Rules indexed to the frontier should therefore expect the rate at which thresholds are crossed to fall by a factor of about 2–3 in log-growth around 2030.

## 8 Limitations

1. **Caps are assumptions.** $\bar\sigma$, $\bar\varphi$, $\bar x$, $\bar T$, $g_P$ and the initial investment growth are marked [A]. We vary them over wide ranges and report Sobol indices, but the 2035–2040 outputs are conditional on these priors. In particular, if AI drives a GATE-style transition in which GDP growth itself becomes endogenous, $g_Y$ and $\bar\sigma$ are no longer exogenous and Proposition 1's secular rate could rise substantially.
2. **Hard caps and constant rates.** Real shares saturate smoothly, and secular rates drift: hardware price-performance may slow, and grid build-out may speed up. Propositions 1–2 hold for any monotone saturation, but the numerical timing would shift by a fraction of the saturation width.
3. **The latency ceiling is modelled as fixed and hard.** In reality it degrades utilisation gradually and can be raised by new architectures, interconnects or training algorithms (Erdil & Schneider-Joseph, 2024). Our "no ceiling" run brackets this, but the main-case result that single-run compute stops growing after about 2037 should not be read as a forecast.
4. **Chips and data are not separate channels.** Chip supply (advanced packaging and memory) enters only through prices and the $\varphi$ cap. Pretraining data limits (Villalobos et al., 2024) are not modelled because synthetic data, multimodal data and RL compute can avoid them. A Chinchilla-style data ceiling would add a fourth channel similar to $C_{\text{lat}}$.
5. **Anchors are uncertain.** Grok 4's compute and cost are estimates based on vague public statements (Epoch AI, 2025a). Capex includes non-AI spending. World GDP is a round number. Recent revenue and capex figures come from company announcements and press reports rather than audited statements. Some 2026 sources we cite (Epoch AI, 2026b,c) are blog-style data insights, not peer-reviewed.
6. **Independence of priors.** We sample parameters independently. In reality, a high $g_I$ probably comes with a high $\bar\sigma$ and a high $g_P$, because investors who expect large returns build power too. Positive correlations would widen the outcome distributions.
7. **No strategic or geopolitical dynamics.** Export controls, racing between labs or states, and financial shocks (for example a correction in AI valuations) are outside the model, except insofar as they change caps or rates.
8. **We ran no experiments on AI systems.** This is a quantitative model calibrated to published data. All model outputs are computed in this repository. All empirical inputs are cited.

## 9 Conclusion

Frontier training compute has grown 4–5× per year because two bounded quantities have been rising: the share of economic output spent on AI compute (and on the single largest run), and a precision/utilisation/pricing residual. Ordinary hardware and economic growth account for only about a fifth of the trend. Treating the problem as growth accounting with bounded and secular factors gives three simple results. The long-run rate is the slowest channel's secular rate, about 1.4× per year. The above-trend gain is a fixed headroom budget of 2–3 OOM, so faster growth now ends sooner. And run duration saves power rather than money, so power tightens as durations level off. Calibrated with published estimates, the model puts the drop below 2× per year around 2030 (2028.7–2033.0). Power and spending are about equally likely to bind at that point, and the poorly characterised data-movement ceiling dominates the level thereafter. These are falsifiable predictions, and the most useful single measurement for long-horizon forecasts is a better estimate of that ceiling.

---

## References

All references were checked to exist via web search on 2026-11-22.

- Cottier, B., Rahman, R., Fattorini, L., Maslej, N., & Owen, D. (2024). The rising costs of training frontier AI models. arXiv:2405.21015.
- EPRI & Epoch AI (2025). *Scaling Intelligence: The Exponential Growth of AI's Power Needs*. Electric Power Research Institute white paper, August 2025. https://restservice.epri.com/publicattachment/94532 (see also Epoch AI, "Power demands of frontier AI training", https://epoch.ai/blog/power-demands-of-frontier-ai-training).
- Epoch AI (2025a). Grok 4 training resources (data insight and model entry). https://epoch.ai/data-insights/grok-4-training-resources ; https://epoch.ai/models/grok-4
- Epoch AI (2025b). The longest training run / training-run length trends. https://epoch.ai/data-insights/longest-training-run
- Epoch AI (2025c). Most of OpenAI's 2024 compute went to experiments (data insight, Oct. 10, 2025); see Epoch AI "AI companies" data hub, https://epoch.ai/data/ai-companies
- Epoch AI (2026a). Trends in Artificial Intelligence (updated Feb. 5, 2026). https://epoch.ai/data/trends
- Epoch AI (2026b). An update on AI's most important number. *Gradient Updates*. https://epoch.ai/gradient-updates/an-update-on-ais-most-important-number
- Epoch AI (2026c). Anthropic's compute spending is increasing by 4x per year (data insight, updated Nov. 18, 2026). https://epoch.ai/data-insights/anthropic-openai-compute-spend-growth
- Erdil, E., & Schneider-Joseph, D. (2024). Data movement limits to frontier model training. arXiv:2411.01137.
- Erdil, E., Potlogea, A., Besiroglu, T., et al. (2025). GATE: An integrated assessment model for AI automation. arXiv:2503.04941.
- Hobbhahn, M., & Besiroglu, T. (2022). Trends in GPU price-performance. Epoch AI. https://epoch.ai/blog/trends-in-gpu-price-performance
- Ho, A., Besiroglu, T., Erdil, E., Owen, D., Rahman, R., Guo, Z. C., Atkinson, D., Thompson, N., & Sevilla, J. (2024). Algorithmic progress in language models. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). arXiv:2403.05812.
- IEA (2025). *Energy and AI*. International Energy Agency, Paris. https://www.iea.org/reports/energy-and-ai
- IMF (2025). *World Economic Outlook, October 2025: Global Economy in Flux, Prospects Remain Dim*. International Monetary Fund.
- Jansen, M. J. W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1–2), 35–43.
- Muennighoff, N., Rush, A. M., Barak, B., Le Scao, T., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T., & Raffel, C. (2023). Scaling data-constrained language models. *NeurIPS 2023*. arXiv:2305.16264.
- Nordhaus, W. D. (2021). Are we approaching an economic singularity? Information technology and the future of economic growth. *American Economic Journal: Macroeconomics*, 13(1), 299–332.
- Pilz, K. F., Sanders, J., Rahman, R., & Heim, L. (2025). Trends in AI supercomputers. arXiv:2504.16026.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.
- Sevilla, J., Heim, L., Ho, A., Besiroglu, T., Hobbhahn, M., & Villalobos, P. (2022). Compute trends across three eras of machine learning. *2022 International Joint Conference on Neural Networks (IJCNN)*. arXiv:2202.05924.
- Sevilla, J., & Roldán, E. (2024). Training compute of frontier AI models grows by 4–5x per year. Epoch AI. https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year
- Sevilla, J., Besiroglu, T., Cottier, B., You, J., Roldán, E., Villalobos, P., & Erdil, E. (2024). Can AI scaling continue through 2030? Epoch AI. https://epoch.ai/blog/can-ai-scaling-continue-through-2030
- Thompson, N. C., Greenewald, K., Lee, K., & Manso, G. F. (2020). The computational limits of deep learning. arXiv:2007.05558.
- Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., & Hobbhahn, M. (2024). Position: Will we run out of data? Limits of LLM scaling based on human-generated data. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235. arXiv:2211.04325.

Press compilations used only for the 2025–2026 capex anchor (not peer-reviewed): "Four US tech giants spent roughly $410 billion on capital expenditure last year … steering toward roughly $725 billion in 2026", Silicon Canals (2026), https://siliconcanals.com/m-four-us-tech-giants-spent-roughly-410-billion-on-capital-expenditure-last-year-much-of-it-tied-to-ai-and-are-steering-toward-roughly-725-billion-in-2026/

---

### Figures (in `results/`)

- Fig. 1 `fig1_accounting.png`: decomposition of current growth into secular and bounded components, and the naive runway distribution.
- Fig. 2 `fig2_fan.png`: fan chart of largest-run compute, 2025–2045, for both variants, against the 4.5× extrapolation and the secular trend.
- Fig. 3 `fig3_binding.png`: share of draws in which each channel binds, over time.
- Fig. 4 `fig4_slowdown.png`: median trailing annual growth factor and distribution of the year growth first drops below 2×.
- Fig. 5 `fig5_sobol.png`: total and first-order Sobol indices for the 2030 level, the 2035 level and the slowdown year.
- Fig. 6 `fig6_revenue_stress.png`: median slowdown year over a grid of AI revenue growth paths.
- Fig. 7 `fig7_frontloading.png`: numerical illustration of Proposition 2.
