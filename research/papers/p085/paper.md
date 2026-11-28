# Batching Cannot Amortize Context: A Capacity-Limited Floor on LLM Decode Cost and What It Implies for How Far Inference Prices Can Fall

**Target venue:** MLSys 2027 (Conference on Machine Learning and Systems), research track

---

## Abstract

At a fixed capability level, the price of LLM inference has fallen by roughly an order of magnitude per year (a16z, 2024; Epoch AI, 2025; Epoch AI, 2026). Forecasting how much further it can fall requires a model of what bounds the cost of producing one token. We derive a closed-form, capacity-limited floor on the cost of autoregressive decoding. In the model, an accelerator group with price $P$, effective FLOP rate $F'$, effective memory bandwidth $B'$ and memory capacity $M$ serves a model with weight bytes $W$, active parameters $N_a$, attention FLOPs $a$ per context token and KV-cache bytes $\kappa$ per context token. The floor is $c^*(L)=P\max\{(2N_a+aL)/F',\; L\kappa M/((M-W)B')\}$. Its structural message is that batching amortizes weight reads but not context reads. Above a crossover context length $L^*$, the cost per token is linear in $L\kappa$ and in dollars per unit of bandwidth. It does not depend on parameter count or FLOP throughput, except through the weight-occupancy factor $M/(M-W)$. For Llama-3-70B on 8×H100, $L^*\approx1.6$k tokens, so typical reasoning and agentic contexts sit in this KV-bound regime.

We derive elasticities for each cost lever: weight quantization, KV quantization, KV-architecture compression, sparse attention, multi-token verification and hardware. We check four predictions on a CPU toy decoder (3 replicates, 43 configurations). (i) Per-token time saturates with batch size at a context-proportional floor. (ii) The slope of that floor tracks KV bytes: the measured slope ratio is 3.92 for a 4× change in KV heads. (iii) The sensitivity of per-token time to weight size falls from 0.82 at L=256 to 0.30 at L=4096. (iv) Verifying several positions per pass divides the KV term, giving a measured 3.4× speedup at τ=4. A two-term traffic model predicts the 43 measured step times with a mean absolute log error of 0.098.

Applied to published model and hardware parameters, the model gives the following. On fixed hardware, the algorithmic headroom on the floor of a GQA-70B-class model is about 5.6× at 4k context and about 24× at 32k (Monte Carlo medians; 90% intervals [3.0, 10.6] and [9.2, 57]). For an MLA model (DeepSeek-V3) on B200, further KV compression yields at most about 3×, because the attention compute term binds. Hardware improves the KV-bound floor by only about 1.26× per year under historical bandwidth trends, against about 1.73× per year in the compute-bound regime. At late-2026 rental prices, B200 and H100 offer essentially the same bandwidth per dollar. We state falsifiable predictions and a test protocol.

## 中文摘要

在固定能力水平下，大语言模型推理价格大约每年下降一个数量级。要判断它还能下降多少，需要一个刻画"生成一个 token 的成本下界"的模型。本文推导了自回归解码在显存容量约束下的闭式成本下界：$c^*(L)=P\max\{(2N_a+aL)/F',\;L\kappa M/((M-W)B')\}$。其中 $P$ 为硬件价格，$F'$ 和 $B'$ 为有效算力和有效带宽，$M$ 为显存容量，$W$ 为权重字节数，$N_a$ 为激活参数量，$a$ 为每个上下文 token 的注意力 FLOPs，$\kappa$ 为每个上下文 token 的 KV 缓存字节数。

核心结论是：批处理可以摊薄权重读取，但不能摊薄上下文（KV）读取。当上下文长度超过临界值 $L^*$ 时，每 token 成本与 $L\kappa$ 及"每单位带宽的价格"成正比。除权重占用因子 $M/(M-W)$ 外，它与参数量和算力无关。以 Llama-3-70B 在 8×H100 上为例，$L^*\approx1.6$k，因此推理型和智能体任务的长上下文处于 KV 受限区间。

我们给出了各类降本手段的弹性：权重量化、KV 量化、KV 结构压缩、稀疏注意力、多 token 验证和硬件。我们在 CPU 上用玩具解码器（3 次重复，43 种配置）检验了四个预测。(i) 每 token 时间随批大小增加而饱和于与上下文成正比的下界。(ii) 下界斜率随 KV 字节数变化：KV 头数改变 4 倍时，斜率比为 3.92。(iii) 权重大小的弹性从 L=256 时的 0.82 降至 L=4096 时的 0.30。(iv) 每次前向验证多个位置可以分摊 KV 读取，τ=4 时实测加速 3.4 倍。一个双项访存模型预测 43 个实测步时，平均绝对对数误差为 0.098。

将模型应用于公开的模型和硬件参数，得到以下结果。在固定硬件上，GQA-70B 级模型的算法降本空间约为：4k 上下文 5.6 倍，32k 上下文 24 倍（蒙特卡洛中位数）。对已采用 MLA 的 DeepSeek-V3（B200），进一步压缩 KV 最多带来约 3 倍收益，因为注意力计算项会成为瓶颈。按历史带宽趋势，硬件每年只能使 KV 受限区间的下界下降约 1.26 倍，而计算受限区间约为 1.73 倍。按 2026 年末的租赁价格，B200 与 H100 每美元带宽几乎相同。我们给出了可证伪的预测和检验方案。

---

## 1 Introduction

How much cheaper can LLM inference get? The empirical record shows rapid declines at fixed capability. Appenzeller (a16z, 2024) estimated roughly 10× per year. Epoch AI (2025) found 9–900× per year depending on benchmark and threshold. Epoch AI's September 2026 report estimates about 47% per quarter, or about 13× per year (Epoch AI, 2026). Gundlach et al. (2025) separate algorithmic efficiency from hardware and competition effects. Trend extrapolation, however, says nothing about *where the floor is*. It cannot tell us which workloads will keep getting cheaper and which will stall, or which engineering levers still have headroom.

This paper takes a bottom-up approach. We ask: *for a given model and a given accelerator, what is the lowest achievable cost per generated token, and how does that floor respond to each lever the field is pulling?* The answer turns on one fact about decoding that the systems literature knows well but that has not been carried through to the economics of price forecasting. Batching amortizes the reading of model weights across the sequences in a batch. It does not amortize the reading of each sequence's KV cache. Each sequence's context must be streamed from memory once per generated token, whatever the batch size. Memory capacity bounds how many sequences fit, so the best achievable cost per token has a floor proportional to context length times KV bytes per token. In this regime, the floor does not depend on how many FLOPs the hardware has or how many parameters the model has.

This matters for the forecasting question because workloads are shifting toward long contexts: reasoning traces, agents, retrieval and code. If decode costs in that regime are governed by bandwidth per dollar and bytes per context token, rather than FLOPs per dollar and parameters, then many of the cost reductions the field has relied on will have little effect there. Those include weight quantization, faster tensor cores and, in part, smaller models.

**Contributions.**

1. **A closed-form capacity-limited decode-cost floor (Section 3).** We prove that, under a standard roofline step-time model with memory-capacity and latency constraints, cost per token is non-increasing in batch size. Its minimum is $c^*(L)=P\max\{(2N_a+aL)/F',\;\Omega\, L\kappa/B'\}$, where $\Omega=\max\{M/(M-W),\tau_{\max}B'/(\tau_{\max}B'-W)\}$ is a weight-occupancy factor. We derive the crossover context $L^*$ and the full set of elasticities of the floor with respect to model and hardware quantities. We also extend the floor to multi-token verification (speculative decoding / multi-token prediction) and sparse attention.
2. **A CPU mechanism experiment testing four structural predictions (Section 5).** On a numpy/OpenBLAS toy decoder, we test: (T1) saturation at a context-proportional floor; (T2) floor slope proportional to KV bytes; (T3) declining sensitivity to weight size as context grows; (T4) division of the KV term by verified positions. We report where an independently calibrated roofline fails, and why: it misses non-fused attention-score traffic.
3. **A sourced quantitative headroom analysis (Section 6).** We apply the model to Llama-3-70B (GQA) and DeepSeek-V3 (MLA) on H100 and B200, and report the following. We give a regime map. We decompose a lever "ladder" step by step. A Monte Carlo over lever ranges gives the distribution of remaining algorithmic headroom on fixed hardware. We compare bandwidth and FLOPs per rental dollar across hardware generations. We project the per-year hardware contribution in each regime.
4. **Falsifiable predictions and a test protocol (Section 7).** These cover context-dependent serving cost, the limited value of weight-only quantization at long context, slower price declines for long-output workloads, the ceiling on KV compression once MLA-class designs are adopted, and quadratic per-task cost growth with reasoning length.

We state plainly what this is not. It is not a physical (thermodynamic) bound: "floor" means the minimum cost on a *given* hardware/model pair under the roofline abstraction. It is also not an estimate of how much smaller models can get at fixed capability. Our model treats the model as given, and capability-preserving shrinkage is in practice a large part of the observed decline (Section 8).

## 2 Related Work

**Empirical price trends.** Appenzeller (2024) documented roughly 10× per year declines in price at constant MMLU performance ("LLMflation"). Epoch AI's March 2025 analysis of six benchmarks found declines of 9–900× per year depending on the threshold. Epoch AI (2026), "The plunging price of thought," reports about 47% per quarter over three years. Gundlach, Lynch, Mertens and Thompson (2025) estimate price-performance trends from April 2024–November 2025 token prices. Their revised version emphasizes that the price of running *frontier* models can rise as models grow and reasoning demands increase. A March 2026 economic analysis of token pricing (arXiv:2603.28576) attributes most cost decline to software and architecture rather than GPU hardware. We have not been able to read that paper's full text and cite it only as context. All of this work is top-down. Ours is a bottom-up floor model meant to explain *which* parts of these trends can continue.

**Analytical models of inference.** Pope et al. (2023) gave an analytical model of partitioning and memory-time trade-offs for large-transformer inference on TPU v4. They noted that multi-query attention's smaller KV cache enables much longer contexts. Shazeer (2019) motivated multi-query attention by the memory-bandwidth cost of repeatedly loading keys and values during incremental decoding. Yuan et al. (2024) survey efficient inference through the roofline lens of Williams, Waterman and Patterson (2009) and release LLM-Viewer. Erdil (2025) builds a detailed model of inference economics, including network latency and parallelism, and computes Pareto frontiers of serial speed versus cost per token. Our contribution is narrower and complementary. We isolate the capacity-limited throughput floor in closed form, show that its long-context limit is independent of parameters and FLOPs, derive lever elasticities, and turn these into price-trend predictions. We claim novelty for that framing and those consequences, not for observing that decoding is memory-bound.

**Levers.** KV-cache size has been attacked by MQA (Shazeer, 2019), GQA (Ainslie et al., 2023), multi-head latent attention (MLA; DeepSeek-AI, 2024, which reports a 93.3% KV reduction relative to DeepSeek 67B), and KV quantization (KIVI, 2-bit; Liu et al., 2024). Effective context has been attacked by attention-sink streaming (Xiao et al., 2024) and by learned sparse attention: DeepSeek Sparse Attention (DSA) selects top-k entries, with k=2048 in the released system (DeepSeek-AI, 2025). Speculative decoding verifies several drafted tokens per target pass (Leviathan et al., 2023). Memory management (PagedAttention/vLLM; Kwon et al., 2023) and prefill/decode disaggregation (DistServe; Zhong et al., 2024) raise achievable batch and utilization. IO-aware attention kernels (FlashAttention; Dao et al., 2022) avoid materializing attention scores in HBM. Each lever enters our model as a specific parameter.

**Hardware trends.** Gholami et al. (2024) report that peak server FLOPS grew about 3.0× per 2 years while DRAM bandwidth grew about 1.6× per 2 years: the "memory wall." Sardana et al. (2024) show that inference demand shifts compute-optimal training toward smaller, longer-trained models. That is the main route by which capability-preserving model shrinkage reduces inference cost, and our model treats it as exogenous.

## 3 Model

### 3.1 Setting and step-time model

Consider one serving replica: a group of $G$ accelerators running one model with ideal tensor parallelism. Per device, the hardware has peak dense FLOP rate $F$ at the compute precision used, HBM bandwidth $B$, HBM capacity $M_{\rm dev}$ and rental price $p$ per hour. Aggregate effective quantities are $F'=\eta_F G F$, $B'=\eta_B G B$, $M=GM_{\rm dev}$ and $P=Gp/3600$ dollars per second. The efficiencies $\eta_F,\eta_B\in(0,1]$ capture achievable fractions of peak.

The model has $N_{\rm tot}$ parameters stored at $q_w$ bytes each, so weight bytes are $W=N_{\rm tot}q_w$, and $N_a$ parameters are active per token. Each context token contributes $\kappa$ bytes of KV cache, summed over layers, and costs $a$ attention FLOPs per generated token. For grouped-query attention with $n_\ell$ layers, $n_h$ query heads, $n_{kv}$ KV heads and head dimension $d_h$:
$$\kappa_{\rm GQA}=2n_\ell n_{kv} d_h q_{kv},\qquad a_{\rm GQA}=4n_\ell n_h d_h,$$
so the **attention arithmetic intensity** is $I=a/\kappa=2g/q_{kv}$, where $g=n_h/n_{kv}$. For MLA with latent dimension $d_c$ and decoupled RoPE dimension $d_r$, using weight absorption,
$$\kappa_{\rm MLA}=n_\ell(d_c+d_r)q_{kv},\qquad a_{\rm MLA}=2n_\ell n_h(2d_c+d_r).$$

A decode step for a batch of $b$ sequences, each with context $L$, takes
$$T(b)=\max\Big\{\frac{b(2N_a+aL)}{F'},\;\frac{W+bL\kappa}{B'}\Big\}. \tag{1}$$
We make three simplifications. All weights are read each step; for MoE at large batch, almost all experts are touched. Activation memory is ignored. The cost of writing the new KV entry, which is $O(b\kappa)$ against the $O(bL\kappa)$ read, is also ignored. Feasibility requires capacity, $W+bL\kappa\le M$, and optionally a time-per-output-token target, $T(b)\le\tau_{\max}$. The cost per generated token is $c(b)=P\,T(b)/b$.

### 3.2 The floor

**Proposition 1 (capacity-limited floor).** $c(b)$ is non-increasing in $b$. Let $b^\dagger=\min\{(M-W)/(L\kappa),\;(\tau_{\max}B'-W)/(L\kappa),\;\tau_{\max}F'/(2N_a+aL)\}$ be the largest feasible batch, treated as continuous. Then
$$c^*(L)=c(b^\dagger)=P\max\Big\{\frac{2N_a+aL}{F'},\;\frac{W/b^\dagger+L\kappa}{B'}\Big\}.$$
If the capacity constraint binds, the memory term equals $\Omega L\kappa/B'$ with $\Omega=M/(M-W)$. If the latency-bandwidth constraint binds, $\Omega=\tau_{\max}B'/(\tau_{\max}B'-W)$.

*Proof.* $c(b)=P\max\{(2N_a+aL)/F',\,(W/b+L\kappa)/B'\}$. The first argument is constant in $b$ and the second is decreasing, so the maximum is non-increasing and the infimum over the feasible set is reached at its largest element $b^\dagger$. Substituting $b^\dagger=(M-W)/(L\kappa)$ gives $W/b^\dagger+L\kappa=L\kappa\,W/(M-W)+L\kappa=L\kappa M/(M-W)$. The latency case is identical with $M$ replaced by $\tau_{\max}B'$. ∎

**Corollary 1 (no amortization of context).** For any batch size and any capacity, $c(b)\ge P\,L\kappa/B'$. Batching can at best remove the weight-read term; the context term is per sequence. The one exception is a *shared* prefix read once per batch (prefix caching). If a fraction $\phi$ of context is shared, $\kappa$ is replaced by $(1-\phi)\kappa+\phi\kappa/b$.

**Corollary 2 (crossover context).** Let $s_m=\Omega\kappa/B'$ and $s_c=a/F'$ be the per-context-token slopes of the two terms. If $s_m>s_c$, the floor is memory (KV) bound for
$$L>L^*=\frac{2N_a/F'}{s_m-s_c}=\frac{2N_a}{\kappa\,\Omega\,\beta-a},\qquad \beta=F'/B'.$$
If $s_m\le s_c$, it is compute-bound at all $L$. Equivalently, the long-context floor is KV-bound iff $I<\Omega\beta$: attention intensity is below the machine balance inflated by weight occupancy. For GQA in BF16, $I=g$ (8 for Llama-3-70B), far below the H100 BF16 balance of roughly 300. MLA with weight absorption raises $I$ to about 242 FLOP/byte for DeepSeek-V3 at BF16 KV, which is close to machine balance. That is why MLA's remaining KV headroom is small (Section 6.4).

**Proposition 2 (elasticities of the floor).** Write $\varepsilon_x=\partial\ln c^*/\partial\ln x$. In the KV-bound regime under capacity binding ($L>L^*$):
$$\varepsilon_\kappa=\varepsilon_L=1,\quad \varepsilon_{B}=-1,\quad \varepsilon_{F}=\varepsilon_{N_a}=0,\quad \varepsilon_{W}=\frac{W}{M-W},\quad \varepsilon_{M}=-\frac{W}{M-W}.$$
In the compute-bound regime: $\varepsilon_F=-1$, $\varepsilon_{N_a}=2N_a/(2N_a+aL)$, $\varepsilon_L=aL/(2N_a+aL)$, and $\varepsilon_\kappa=\varepsilon_B=\varepsilon_W=0$.
*Proof.* Direct differentiation of the two branches of Proposition 1. ∎

Weight-only quantization therefore acts at long context only through $\Omega$, by freeing capacity for more sequences. Its benefit is bounded by $\Omega$ itself: about 1.28 for Llama-3-70B BF16 on 8×H100, so even infinitely compressed weights gain at most 1.28×. By contrast, cutting $\kappa$ has unit elasticity until $L^*$ moves past $L$.

**Proposition 3 (multi-token verification).** Suppose each pass verifies $\tau$ positions per sequence, emits $A\le\tau$ tokens in expectation, and incurs drafting overhead $\delta$. The KV cache is read once per pass. Then
$$c^*_{\rm spec}=\frac{P(1+\delta)}{A}\max\Big\{\frac{\tau(2N_a+aL)}{F'},\;\Omega\frac{L\kappa}{B'}\Big\}.$$
In the KV-bound regime, the gain is $A/(1+\delta)$ as long as $\tau(2N_a+aL)/F'\le\Omega L\kappa/B'$. In the compute-bound regime, verification multiplies cost by $\tau(1+\delta)/A\ge1$, so a rational operator does not use it. Thus speculative decoding is a KV-regime lever. Its value at large batch depends on context length, not only on latency.

**Sparse attention.** Reading only $L_{\rm eff}<L$ context tokens per step (for example, top-k selection) replaces $L$ by $L_{\rm eff}$ in *both* the $a$ and $\kappa$ terms, plus the cost of an indexer. Among the levers considered here, it is the only one that lowers both branches of the long-context floor.

**Per-task cost.** A task that generates $T$ tokens after a prompt of $L_0$ tokens costs $\sum_{t}c^*(L_0+t)$. In the KV regime this is $\approx P\Omega\kappa(L_0T+T^2/2)/B'$, which is *quadratic* in reasoning length.

## 4 Experimental and Analytical Setup

### 4.1 CPU mechanism experiment

**Purpose.** The analytic model's GPU numbers depend on published specifications, not measurements. We therefore test its *structural* predictions on hardware we can measure: a 4-core Intel Xeon (2.1 GHz, 260 MiB L3 as reported by `lscpu`, 15 GB RAM), using numpy with OpenBLAS 0.3.34, in FP32.

**Toy decoder.** 8 layers, $d=1024$, 16 query heads of dimension 64, SwiGLU FFN of width 2816 (varied to 1408 and 5632 in T3), and $n_{kv}\in\{16,4,1\}$. Weight bytes are 273–688 MB, above the L3 size, so weights stream from DRAM. KV bytes per context token are $\kappa=2\cdot8\cdot n_{kv}\cdot64\cdot4$ = 64, 16 or 4 KiB. One decode step computes QKV projections, grouped attention against a pre-filled cache (score matmul, softmax, value matmul), the output projection and the FFN for $b\cdot\tau$ query rows.

**Calibration, done independently of the toy model.** Three quantities are calibrated: GEMM throughput ($2048^3$ matmul), GEMV bandwidth (a 512 MB matrix–vector product), and attention-kernel bandwidth (score, softmax and value on a synthetic 16×16-head, 2048-token cache).

**Grid and replication.** Batch $b\in\{1,\dots,128\}$; context $L\in\{256,\dots,4096\}$, capped at $bL\le$ 32k–128k tokens depending on $n_{kv}$; verified positions $\tau\in\{1,2,4,8,16\}$ at $b=8$. A separate run with a 1-token cache measures the weight-stage time $t_w(b)$. Each point is the median of 7 timed steps after one warm-up. The whole experiment was run 3 times (seeds 0–2), and we report the median across replicates. Total runtime is about 8 minutes.

### 4.2 Analytical application

**Configurations.** All are sourced. Llama-3-70B: 80 layers, 64 query heads, 8 KV heads, head dimension 128 (model `config.json`); 70.6B parameters at BF16. DeepSeek-V3: 671B total and 37B active parameters (DeepSeek-AI, 2024b); 61 layers, 128 heads, `kv_lora_rank` 512, RoPE head dimension 64 (model `config.json`); FP8 weights and BF16 KV. H100 SXM: 989 TFLOP/s dense BF16, 1979 dense FP8, 3.35 TB/s, 80 GB. B200: 4.5 PFLOP/s dense FP8, 8 TB/s, 180 GB usable of 192 nominal. Rental prices are $2.5 per hour for H100 (sensitivity range $2–4) and $6 per hour for B200 ($4–10). These are mid-points of late-2026 tracker and aggregator ranges (Section 9 discusses their reliability). Baseline efficiencies are $\eta_F=0.6$ and $\eta_B=0.8$ (varied to 0.4–0.8 and 0.6–0.9). The latency target is $\tau_{\max}=50$ ms per decode step (varied 20–100 ms and none).

**Lever ranges for the Monte Carlo.** Each lever is sampled log-uniform, 20,000 draws:
- KV quantization divisor of $\kappa$: 1.5–8. Endpoints correspond to roughly FP8 and to 2-bit (KIVI).
- KV-architecture divisor: 1–4. GQA-8→MLA is 3.56× in KV elements for this geometry. In the Monte Carlo, this lever does *not* add attention FLOPs; that is optimistic for MLA, whose compute effect we model separately with the real DeepSeek-V3 configuration.
- Sparse-attention divisor of $L$: 1–8.
- Tokens accepted per verification pass: 1.2–3.5, with $\tau=\lceil1.6A\rceil$ positions verified and $\delta=0.1$; used only when it lowers cost.
- Weight-bytes divisor: 1–4.
- Compute-precision multiplier on $F$: 1–4.
- Utilization multiplier on $\eta$: 1–1.2.

Code for everything is in `code/`; every number below is produced by `analysis.py` or `analyze_cpu.py` and stored in `results/`.

## 5 Results I: CPU Mechanism Tests

All numbers in this section are measured in this environment (`results/cpu_summary.json`; Figure 1 is `results/fig_cpu_tests.png`). The calibration medians across replicates (min–max) are: GEMM 390 GFLOP/s (334–432), GEMV bandwidth 43.1 GB/s (30.4–47.7), and attention-kernel bandwidth 12.7 GB/s (10.6–12.8).

**T1: batching saturates at a context-proportional floor.** At $n_{kv}=16$ and $L=4096$, per-token time falls from 38.6 ms at $b=1$ to 28.1 ms at $b=8$. The KV-only floor $L\kappa/B_{\rm attn}$ is 21.1 ms. At $L=1024$ it falls from 19.7 ms to 8.4 ms at $b=32$, against a KV-only floor of 5.3 ms. At every point, per-token time stays above the KV-only floor, and the gap shrinks with $b$, as Corollary 1 predicts. One non-monotonicity is a CPU artifact: $b=2$ is slower than $b=1$ (weight-stage step 37.9 ms vs 11.3 ms). This is because OpenBLAS uses an efficient multithreaded GEMV for one row but a poorly parallelized small-M GEMM for two.

**T2: the floor slope tracks KV bytes.** We take per-token time at the largest feasible batch as a function of $L$. It is linear in $L$ with $R^2=0.999$, $0.90$ and $0.99998$ for $n_{kv}=16,4,1$. The slopes are 6.61, 1.69 and 1.07 µs per context token. The 16:4 slope ratio is **3.92**, against a predicted 4. The 4:1 ratio is only 1.58 (predicted 4), and the 16:1 ratio is 6.2 (predicted 16).

The discrepancy at small $n_{kv}$ has an identifiable cause. Our numpy attention materializes the score tensor, which has $n_h$ (not $n_{kv}$) entries per context token per layer, and passes over it several times in softmax. That adds a per-context-token traffic term $\sigma$ independent of $n_{kv}$. We fit the two-term model slope $=(\kappa+\sigma)/B_{\rm eff}$ to the $n_{kv}=16$ and $n_{kv}=1$ slopes. This is a post-hoc fit of 2 parameters, and it gives $B_{\rm eff}=11.1$ GB/s, close to the independently calibrated 12.7 GB/s, and $\sigma\approx7.8$ KB per context token. Out of sample, it predicts an $n_{kv}=4$ slope of 2.18 µs against 1.69 measured. Combined with the *measured* weight-stage time $t_w(b)$, the two-term model predicts all 43 measured step times with median measured/predicted 0.96, a 10–90% range of [0.88, 1.21], and mean absolute log error 0.098. The fully independent roofline, which uses $\kappa$ only and GEMV bandwidth for weights, underpredicts: median ratio 1.41 at $n_{kv}=16$ and 3.05 at $n_{kv}=1$.

The lesson for the GPU model is this. The quantity that matters is *all* per-context-token bytes moved per generated token. IO-aware fused kernels (Dao et al., 2022) remove $\sigma$, so on GPUs with such kernels $\kappa$ is the right quantity. Any implementation that materializes per-head state raises the effective $\kappa$.

**T3: weight size stops mattering as context grows.** We vary FFN width so that weight bytes span 273–688 MB, with $\kappa$ fixed. At $L=256$, the elasticity $\partial\ln t/\partial\ln W$ is 0.81 ($b=1$) and 0.82 ($b=8$). At $L=4096$ it falls to 0.33 ($b=1$) and **0.30** ($b=8$). This is the qualitative content of Proposition 2: as the context term grows, weight reduction loses its leverage. The CPU elasticities are larger than the GPU model would give, because the CPU weight stage is dominated by inefficient small-M GEMMs rather than pure streaming.

**T4: verifying several positions per pass divides the KV term.** At $b=8$ and $L=4096$, verifying $\tau$ positions per pass (counting every position as accepted, so this is the upper bound $A=\tau$) gives per-token speedups of 1.65, **3.41**, 5.01 and 5.67 for $\tau=2,4,8,16$. The independently calibrated roofline predicts 2.0, 3.84, 6.96 and 11.7. At $L=512$, the measured speedups are 1.68, 2.70, 3.78 and 5.51 against predicted 2.0, 3.35, 4.87 and 6.15. Speedups are larger at long context for moderate $\tau$, and saturate once the weight and compute stage binds, as Proposition 3 predicts. The roofline over-predicts at large $\tau$ because the CPU's attention kernel does not become cheaper per row as rows are added.

## 6 Results II: Sourced Quantitative Analysis

All numbers are model outputs from `analysis.py` (`results/analysis_results.json`) given the sourced parameters of Section 4.2. They are not measurements of production systems.

### 6.1 Regime map and floors (Table 1, Figure 2)

**Table 1.** Floor cost $c^*$ ($ per 1M output tokens) at a 50 ms per-step latency target.

| Model / hardware | κ (bytes/ctx tok) | I = a/κ | β = F'/B' | W/M | L* | L=1k | L=4k | L=16k | L=64k | L=128k |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3-70B, 8×H100 BF16 | 327,680 | 8 | 221 | 0.22 | 1,561 | 0.168 (C) | 0.446 | 1.78 | 7.14 | 14.3 |
| DeepSeek-V3, 16×H100 FP8 | 70,272 | 242 | 443 | 0.52 | 1,527 | 0.053 (C) | 0.157 | 0.627 | 2.51 | 5.02 |
| DeepSeek-V3, 8×B200 FP8 | 70,272 | 242 | 422 | 0.47 | 1,921 | 0.056 (C) | 0.140 | 0.561 | 2.25 | 4.49 |

(C) marks the compute-bound regime; all other entries are KV/memory-bound.

All three configurations cross into the KV-bound regime at 1.5–1.9k tokens. Beyond that, the floor is linear in $L$: a 128k-context token costs about 85× (Llama) or 80–95× (DeepSeek-V3) as much as a 1k-context token on the same hardware. At capacity, the unamortizable term $L\kappa/B'$ makes up 78% of the Llama floor, and 48–53% of the DeepSeek-V3 floor. The remainder is the weight term amortized over $b^\dagger$, which is large for DeepSeek-V3 because its weights fill about half of memory ($\Omega\approx1.9$–2.1).

As a sanity anchor, which is not a validation: third-party aggregators list Llama-3.3-70B output at about $0.30–0.40 per 1M tokens (DeepInfra listings, late 2026). The model floor at 1–2k context is $0.17–0.22. Prices at short context are therefore within roughly 2× of this hardware floor. The gap is consistent with margins, utilization below peak, and prefill costs that the decode floor ignores.

**Sensitivity.** Across $\eta_F\in[0.4,0.8]$ and $\eta_B\in[0.6,0.9]$, the crossover $L^*$ for Llama on H100 ranges from 867 to 2,687 tokens, and the 32k floor ranges from $3.17 to $4.76. Tightening the latency target to 20 ms raises the 32k floor from $3.57 to $4.15, because the latency-bandwidth constraint replaces capacity in $\Omega$. Targets of 30 ms or looser do not change the floor. All floors scale linearly with $/device-hour.

### 6.2 Elasticities (Table 2)

**Table 2.** Numerical elasticities of $c^*$ (from `elasticities()`).

| Config | L | κ | W | N_a | F | B | M | L |
|---|---|---|---|---|---|---|---|---|
| Llama-70B / 8×H100 | 1k | 0 | 0 | 0.98 | −1 | 0 | 0 | 0.02 |
| Llama-70B / 8×H100 | 32k | 1 | 0.28 | 0 | 0 | −1 | −0.28 | 1 |
| DeepSeek-V3 / 16×H100 | 32k | 1 | 1.10 | 0 | 0 | −1 | −1.10 | 1 |
| DeepSeek-V3 / 8×B200 | 32k | 1 | 0.87 | 0 | 0 | −1 | −0.87 | 1 |

At 1k context, cost responds to parameters and FLOPs. At 32k it responds to KV bytes, bandwidth and memory occupancy, and *not* to FLOPs or active parameters. For DeepSeek-V3, the weight elasticity at long context is close to 1, because its FP8 weights fill half of memory. Weight compression (for example, to FP4) is therefore valuable for large MoE models, but only through the capacity it frees for KV.

### 6.3 Lever ladder and single-lever effects (Llama-3-70B, 8×H100)

**Single levers (factor reduction of the floor).**

| Lever | L=1k | L=4k | L=32k | L=128k |
|---|---|---|---|---|
| weights 4-bit (W/4) | 1.00 | 1.21 | 1.21 | 1.21 |
| KV 4-bit (κ/4) | 1.00 | 2.51 | 4.00 | 4.00 |
| KV-architecture compression (κ/3.56) | 1.00 | 2.51 | 3.56 | 3.56 |
| sparse attention (L/4) | 1.01 | 2.65 | 4.00 | 4.00 |
| multi-token verification (2.5 tok/pass) | 1.00 | 1.43 | 2.27 | 2.27 |
| FP8 compute (F×2) | 1.51 | 1.00 | 1.00 | 1.00 |
| next-gen hardware at equal price (F, B, M ×2) | 2.00 | 2.28 | 2.28 | 2.28 |

**Cumulative ladder at L=32k:**

| Step | Cumulative factor | $/1M tokens |
|---|---|---|
| Baseline | 1.0× | $3.57 |
| + FP8 weights | 1.14× | |
| + KV 4-bit | 4.57× | |
| + KV-architecture compression | 13.4×, now compute-bound | |
| + sparse attention L/4 | 18.8× | |
| + multi-token verification | 18.8×, unused (compute-bound) | |
| + FP8 compute | 37.5× | |
| + utilization ×1.15 | 43.1× | $0.083 |

**At L=4k** the same ladder saturates much earlier: 1.14× → 2.51× (compute-bound after KV-4-bit) → 2.65× → 5.3× (FP8 compute) → 6.1×.

The ladder shows the central dynamic. KV levers are worth a great deal at long context, but they are **exhaustible**. Once $L^*$ passes the operating context, the floor returns to $2N_a/F'$ per token: the short-context floor. From there, only FLOPs per dollar, fewer active parameters, or reading fewer tokens help.

### 6.4 An MLA model has much less KV headroom left

For DeepSeek-V3 on 8×B200, the attention intensity $I\approx242$ is already close to the machine balance $\beta\approx422$. Halving $\kappa$ (for example, FP8 KV) gives 1.58× at 4k and 2.0× at 32k and 128k. Quartering or cutting it 8× gives at most 2.88× (32k) and 3.16× (128k), after which the floor is compute-bound. FP4 weights give 1.44× at all long contexts, through capacity. Beyond about 3×, the remaining long-context lever for an MLA-class model is to *read fewer tokens*: sparse attention reduces both branches. This is consistent with the subsequent DeepSeek-V3.2 change, which added top-k sparse attention (DeepSeek-AI, 2025). That consistency is not evidence for our model, since the design decision had many inputs.

### 6.5 Monte Carlo: algorithmic headroom on fixed hardware

These figures are for the Llama-3-70B-class baseline on 8×H100. "Still KV-bound" is the fraction of draws in which the KV term still binds after the levers are applied.

| Context | Median factor | 90% interval | Still KV-bound | Most influential levers (Spearman ρ with log factor) |
|---|---|---|---|---|
| 4k | 5.6× | [3.0, 10.6] | 6% | compute precision 0.95 |
| 32k | 24× | [9.2, 57] | 47% | sparsity 0.64, KV quantization 0.41, compute precision 0.41, KV architecture 0.33 |
| 128k | 37× | [9.9, 105] | 69% | sparsity 0.71, KV quantization 0.44, KV architecture 0.36 |

Algorithmic headroom on a fixed model and fixed hardware is therefore context-dependent. It is single-digit at short-to-moderate context, where the remaining lever is arithmetic precision. It is one to two orders of magnitude at long context, where it comes mostly from reading fewer and smaller KV entries. These ranges inherit the assumed lever ranges, which encode judgement about what is achievable without quality loss. Figure 3 (`results/fig_mc_headroom.png`) shows the 32k distribution.

### 6.6 Hardware: the KV-bound floor improves slowly

At the mid-point rental prices, H100 delivers 1.34 TB/s per $/hr and B200 1.33 TB/s per $/hr. FP8 FLOP/s per $/hr are 0.79 and 0.75 PFLOP/s respectively. Across the price ranges, bandwidth per $/hr spans 0.84–1.68 for H100 and 0.80–2.0 for B200. At late-2026 market prices, the newer generation therefore offers no clear per-dollar improvement in either resource. Its gains appear as larger capacity, which lowers $\Omega$, and as per-replica speed.

Over longer horizons, we apply Gholami et al.'s historical per-device trends (FLOPS 3.0× per 2 yr; DRAM bandwidth 1.6× per 2 yr) and *assume* constant real price per device. Under those assumptions, the compute-bound floor halves every 1.26 years (1.73× per year), and the KV-bound floor halves every 2.95 years (1.26× per year). With ±10% per year device-price drift, the KV-bound rate ranges from 1.15× to 1.41× per year.

Compare these with the observed ~10–13× per year price decline at fixed capability (Appenzeller, 2024; Epoch AI, 2026). In the KV-bound regime, hardware could account for only about a tenth of that rate in log terms (ln 1.26 / ln 13 ≈ 0.09). The rest must come from algorithms, smaller models at equal capability, utilization and margins.

### 6.7 Per-task cost grows quadratically in reasoning length

For Llama-70B on H100 at the floor, with a 1k-token prompt, generating T=1k, 4k, 16k and 64k tokens costs $0.00018, $0.0013, $0.016 and $0.23 per task. That is a 64× increase in tokens for a 1,275× increase in cost, and an average per-token price rising from $0.18 to $3.59 per 1M tokens. If reasoning lengths grow, per-*task* cost can rise even while the per-token floor at fixed context falls. This is a mechanism behind the observation that frontier per-task prices can rise even as fixed-capability prices fall (Gundlach et al., 2025, v2).

## 7 Falsifiable Predictions and Test Protocol

**P1 (context-linear serving cost).** For a fixed open-weight model on fixed hardware at saturating load, maximum decode throughput per GPU falls as $1/L$ beyond $L^*$. The slope matches $\Omega\kappa/B'$ to within the achievable-bandwidth factor $\eta_B$. *Test:* run vLLM or SGLang at maximum feasible batch for Llama-3-70B and an MLA model, over $L\in\{1,2,4,8,16,32,64\}$k. Fit throughput$^{-1}$ against $L$. The prediction fails if the slope ratio between models differs from the $\kappa\Omega$ ratio by more than 30%. Corollary: flat per-token list pricing across context implies cross-subsidy from short to long contexts, unless sparse attention or prefix sharing is deployed.

**P2 (weight-only quantization has bounded long-context value).** Under the same protocol, moving from BF16 to 4-bit weights raises throughput at 32k context by at most $\Omega_{\rm BF16}/\Omega_{\rm 4bit}$: about 1.2× for 70B on 8×H100. Moving KV to 4 bits raises it about 4× until compute binds. *Falsified if* weight-only 4-bit yields more than 1.5× at 32k.

**P3 (slower price declines for long-output workloads).** In public price-per-task series (for example, Epoch-style fixed-capability frontiers), the decline rate for workloads requiring long outputs or contexts will be lower than for short-context tasks. Once KV levers are exhausted, the gap should widen. *Test:* split fixed-capability price series by mean tokens per task (above or below 8k). Compare log-linear decline rates over ≥ 2 years.

**P4 (ceiling on KV compression for MLA-class models).** For models with attention intensity $I$ within 2× of machine balance, further KV quantization yields at most about 3× serving-cost reduction at ≤128k context. Larger gains will come only from sparse attention, which reads fewer tokens. *Falsified if* FP4 or 2-bit KV on an MLA model yields more than 4× throughput at 32k without sparsity.

**P5 (quadratic per-task cost).** For agentic tasks with total generated length $T$, measured serving cost per task scales as $T^{\alpha}$ with $\alpha\in[1.6,2]$ for $T$ well above $L^*$. *Test:* instrument a serving stack and regress log GPU-seconds per task on log $T$.

## 8 Discussion: so how much further can inference cost fall?

The model decomposes the question into four factors.

1. **Price over floor (margin and utilization).** At short context, prices for a commodity open model sit within roughly 2× of our hardware floor (Section 6.1). For the frontier, we have no comparable data.
2. **Algorithmic headroom on the floor, at fixed model and hardware.** About 3–10× at ≤4k context. About 10–60× at 32k for a GQA model. Under 3× from further KV compression alone for an MLA-class model, with sparse attention as the remaining lever.
3. **Hardware per dollar.** About 1.26× per year in the KV-bound regime and about 1.73× per year in the compute-bound regime under historical trends with constant device prices. Late-2026 rental prices show no per-dollar gain from H100 to B200.
4. **Capability-preserving model shrinkage.** Our floor model does not bound this factor. In the compute-bound regime it acts through $N_a$. In the KV-bound regime it acts *only* through $\kappa$ and $\Omega$: a smaller model with the same KV geometry is barely cheaper at long context.

Hence: for short-context workloads, the field still has many years of decline available from hardware, precision and smaller models. For long-context workloads, the decline will increasingly depend on attention design, meaning bytes per context token and tokens read per step, rather than on compute. The ~10× per year fixed-capability decline is unlikely to persist uniformly across workloads unless sparse attention or a comparable change to what is read per step keeps delivering. Historical hardware trends contribute about 1.26× per year in that regime.

## 9 Limitations

- **The floor is model-relative, not physical.** It assumes a fixed model and accelerator, the roofline abstraction, and ideal tensor parallelism. It ignores communication (all-reduce costs, which Erdil (2025) and Pope et al. (2023) model), prefill cost, MoE load imbalance, activation memory, KV-cache fragmentation (cf. PagedAttention), and scheduling under variable demand. Including these would raise the floor and shift $L^*$.
- **Efficiency parameters are assumed.** $\eta_F$ and $\eta_B$ are not measured on GPUs. We report sensitivity (Section 6.1), but measured kernel efficiencies in the long-context decode regime could fall outside our ranges.
- **Rental prices are noisy and partly unverified.** H100 and B200 prices come from trackers and aggregators whose dates and figures conflict (on-demand B200 listings range from about $2 to $16 per hour). All dollar floors scale linearly with price. The per-dollar hardware comparison in Section 6.6 is especially sensitive to them.
- **Lever ranges encode judgement.** The Monte Carlo ranges are assumptions about what is achievable *without quality loss*. We do not model accuracy degradation from KV quantization, sparsity or architecture change. The "KV-architecture" lever ignores MLA's extra attention FLOPs (we model those only for DeepSeek-V3 directly).
- **The CPU experiment validates structure, not GPU magnitudes.** It uses FP32 numpy on 4 cores. Attention is not fused, which adds score traffic σ (Section 5, T2). The weight stage suffers small-M GEMM inefficiency, and timings are noisy in a shared virtual machine (we observed occasional transient slowdowns, hence the 3-replicate medians). The two-term model has 2 post-hoc fitted parameters. T4 assumes all verified positions are accepted, so it is an upper bound on speculative gain.
- **The trend projection is a scenario.** The 1.26× and 1.73× per year figures extrapolate Gholami et al.'s per-device historical rates under constant real device prices. Packaging changes (HBM stacking, SRAM-heavy or processing-near-memory designs) could break the bandwidth trend.
- **The price anchor is weak.** The comparison to listed Llama-3.3-70B prices uses third-party aggregator figures. It is an anchor, not a validation.
- **Verification of sources.** Several sources (Epoch AI pages, arXiv:2603.28576) were verified through search listings and secondary coverage rather than full-text reading, because those hosts were not reachable from this environment.

## 10 Conclusion

The cost of decoding a token has two floors. The compute floor is set by active parameters per FLOP-dollar. The context floor is set by KV bytes per context token per bandwidth-dollar, and batching cannot amortize it. Above about 1.5–2k tokens of context, the second floor governs for today's open models on today's GPUs. Its long-context cost is linear in context, and it is unaffected by more FLOPs, more parameters pruned, or weight quantization beyond the capacity they free. The levers that work there, KV quantization, KV-architecture compression, sparse attention and multi-token verification, still offer one to two orders of magnitude for GQA-class models. They are exhaustible, and MLA-class models have already used much of them. Under historical memory-bandwidth trends, hardware alone improves this floor by only about a quarter per year. The question "how much further can inference cost fall?" therefore has a context-dependent answer, and we have given predictions under which the answer can be checked.

## Reproducibility

`code/README.md` contains the full commands. CPU experiment: `python3 cpu_experiment.py {0,1,2}` (~8 min total); `python3 analyze_cpu.py`; `python3 analysis.py`. Outputs are in `results/`: `cpu_runs_rep*.json`, `cpu_calibration_rep*.json`, `cpu_summary.json`, `analysis_results.json`, `fig_cpu_tests.png`, `fig_floor_vs_context.png` and `fig_mc_headroom.png`.

**Data provenance.**
- Sections 5–6 and Tables 1–2: computed by code in this repository.
- Hardware specifications, model configurations, prices and trend rates: from the sources below.
- Lever ranges: assumptions, with ranges anchored to cited methods.

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *Proceedings of EMNLP 2023*, 4895–4901. arXiv:2305.13245.

Appenzeller, G. (2024, November 12). Welcome to LLMflation – LLM inference cost is going down fast. Andreessen Horowitz. https://a16z.com/llmflation-llm-inference-cost/

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2205.14135.

DeepSeek-AI (2024a). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.

DeepSeek-AI (2024b). DeepSeek-V3 Technical Report. arXiv:2412.19437. Model configuration: https://huggingface.co/deepseek-ai/DeepSeek-V3 (config.json).

DeepSeek-AI (2025). DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. arXiv:2512.02556.

Epoch AI (2025, March). LLM inference prices have fallen rapidly but unequally across tasks. https://epoch.ai/data-insights/llm-inference-price-trends

Epoch AI (Emberson, L., & Roodman, D.) (2026, September 22). The plunging price of thought. https://epoch.ai/publications/the-plunging-price-of-thought

Erdil, E. (2025). Inference economics of language models. arXiv:2506.04645.

Gholami, A., Yao, Z., Kim, S., Hooper, C., Mahoney, M. W., & Keutzer, K. (2024). AI and Memory Wall. *IEEE Micro*, 44(3), 33–39. arXiv:2403.14123.

Gundlach, H., Lynch, J., Mertens, M., & Thompson, N. (2025). The Price of Progress: Algorithmic Efficiency and the Falling Cost of AI Inference (v2 retitled "The Price of Progress: Price Performance and the Future of AI", 2026). arXiv:2511.23455.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *Proceedings of SOSP 2023*. arXiv:2309.06180.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *Proceedings of ICML 2023*, PMLR 202:19274–19286. arXiv:2211.17192.

Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., & Hu, X. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. *Proceedings of ICML 2024*, PMLR 235:32332–32344. arXiv:2402.02750.

Meta (2024). Meta-Llama-3-70B model configuration (config.json: 80 layers, hidden size 8192, 64 attention heads, 8 key-value heads). Hugging Face model hub.

NVIDIA (2022–2024). H100 Tensor Core GPU datasheet; Blackwell B200 specifications (as summarized at https://inferencex.semianalysis.com/chips/h100 and https://inferencex.semianalysis.com/chips/b200).

Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2023). Efficiently Scaling Transformer Inference. *Proceedings of Machine Learning and Systems 5 (MLSys 2023)*. arXiv:2211.05102.

Sardana, N., Portes, J., Doubov, S., & Frankle, J. (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. *Proceedings of ICML 2024*, PMLR 235:43445–43460. arXiv:2401.00448.

Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150.

Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful Visual Performance Model for Multicore Architectures. *Communications of the ACM*, 52(4), 65–76. doi:10.1145/1498765.1498785.

Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. *ICLR 2024*. arXiv:2309.17453.

Yuan, Z., Shang, Y., Zhou, Y., Dong, Z., Zhou, Z., Xue, C., Wu, B., Li, Z., Gu, Q., Lee, Y. J., Yan, Y., Chen, B., Sun, G., & Keutzer, K. (2024). LLM Inference Unveiled: Survey and Roofline Model Insights. arXiv:2402.16363.

Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024). DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. *Proceedings of OSDI 2024*. arXiv:2401.09670.

**Pricing sources (data, not literature).**
- H100 rental ranges: SemiAnalysis GPU index (https://gpu-index.semianalysis.com/); IntuitionLabs H100 rental comparison (https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison).
- B200 rental ranges: Together AI HGX B200 page (https://www.together.ai/gpu/nvidia-hgx-b200); amcompute B200 price index (https://www.amcompute.com/gpu-cloud-prices/b200).
- Llama-3.3-70B API listings: Requesty DeepInfra listing (https://www.requesty.ai/models-new/deepinfra/meta-llama-llama-3.3-70b-instruct).
