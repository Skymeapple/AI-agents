# The Significant Quantity of Compute: Material-Accountancy Limits on an "IAEA for AI"

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES 2027), full paper track

**Code and data:** `code/` (run `python3 run_all.py`), outputs in `results/`. All model numbers below are tagged **[C]** (computed by our code and saved in `results/results.json`) or **[S]** (taken from a cited source).

---

## Abstract

Proposals for an international AI agency "like the IAEA" usually argue by analogy, without asking whether the IAEA's verification method still works when the controlled material is computing hardware. We take the IAEA's own method, nuclear material accountancy, and apply it to AI chips. We define a *Significant Quantity of Compute* (SQC): the number of H100-equivalent chips needed to finish a threshold-crossing training run within a fixed time window. We then ask when diverting one SQC from a declared global chip stock can be detected at the IAEA-style goal of 90% detection probability with a 5% false-alarm rate. Under Gaussian inventory error, detection needs the *accountancy ratio* ρ = SQC/(σS) to be at least ρ* ≈ 2.93, where S is the stock and σ the relative inventory uncertainty. Put simply, a threshold-crossing run must need a share of world compute about three times larger than the accounting error. We prove that the threshold rule alone decides the sign of ρ's long-run trend:

- ρ falls by a factor of about 6.5 per year (90% Monte Carlo range 4.5–15.5) when the agreement protects a fixed capability level.
- ρ falls by about 2.3 per year under a fixed physical-FLOP cap.
- ρ rises when the cap tracks the frontier trend (in 98.5% of Monte Carlo draws).

Institutional improvements such as a larger budget, pooled national intelligence or better measurement multiply ρ by some factor M. That only moves the verification horizon by ln M/λ, so the agency buys "log-time". We anchor σ empirically. Epoch AI's 90% interval on chips smuggled to China through 2025 implies an inventory uncertainty of about 2% of the global stock. At that level, a 2026-frontier-scale run uses about 2.2% of world compute and gives ρ ≈ 1.1, which means about a 29% detection probability. GPT-4-level nonproliferation would need σ ≈ 5×10⁻⁶. We also translate the problem into IAEA inspection capacity and analyse hardware-enabled mechanisms (HEMs), which remove the stock-growth term but not the algorithmic-progress term. Our conclusion is narrow. An IAEA-like agency can plausibly verify a frontier-relative ceiling, provided accounting improves several-fold. It cannot verify nonproliferation of a fixed capability level through compute accounting beyond a few years, whatever its budget. We give four falsifiable predictions and a test protocol.

## 中文摘要

关于建立"类似国际原子能机构（IAEA）的国际AI安全机构"的提议，多停留在类比层面，很少追问：当受控对象从核材料变为算力芯片时，IAEA的核心核查方法是否依然有效。本文把IAEA的核心方法"核材料衡算"移植到AI芯片上，提出"算力重要量"（Significant Quantity of Compute, SQC）：在给定时间窗口内完成一次越过阈值的训练所需的H100等效芯片数。在高斯盘存误差下，要以IAEA式目标（检出概率90%、误报率5%）发现从全球申报库存中转移一个SQC，必须满足衡算比 ρ = SQC/(σS) ≥ ρ* ≈ 2.93，其中S为库存，σ为相对盘存不确定度。直观地说，越阈训练所需占全球算力的份额，必须约为衡算误差的三倍。我们证明，ρ的长期走势方向完全由阈值规则决定：若保护对象是固定能力水平，ρ每年约下降6.5倍（蒙特卡洛90%区间4.5–15.5）；若采用固定物理FLOP上限，每年约下降2.3倍；若上限随前沿趋势移动，ρ上升（98.5%的抽样）。预算、情报汇集、测量精度等机构改进只能把ρ乘以某个倍数M，核查期限因此只延长 ln M/λ，即机构只能"买到对数时间"。以Epoch AI对截至2025年流入中国的走私芯片的90%区间为经验锚，盘存不确定度约为全球库存的2%。在此水平下，2026年前沿规模的训练约占全球算力的2.2%，ρ≈1.1，检出概率约29%；而要核查"GPT-4级能力不扩散"，则需σ≈5×10⁻⁶。本文还把问题换算为IAEA视察能力，并分析硬件内置治理机制（HEM）：它能消除库存增长项，却不能消除算法进步项。结论是：若衡算精度提高数倍，IAEA式机构有望核查"相对前沿的上限"；但对固定能力水平的不扩散，无论预算多大，算力衡算在数年之后都无法核查。文末给出四条可证伪预测及检验方案。

---

## 1 Introduction

People keep calling for "an IAEA for AI". Heads of AI companies, the UN Secretary-General's High-level Advisory Body [UN HLAB 2024] and many academic proposals [Ho et al. 2023; Trager et al. 2023; Maas & Villalobos 2023] have all done so. The IAEA's distinctive function is not standard-setting or scientific consensus, which other bodies also provide. It is *verification*: an independent judgement that declared nuclear material has not been diverted. The safeguards system rests on a quantitative idea. A *significant quantity* (SQ) is the amount of material from which a weapon cannot be excluded: 8 kg of plutonium or 25 kg of U-235 in highly enriched uranium [S: IAEA Safeguards Glossary]. Material balance accounting tries to detect the loss of one SQ within a *timeliness goal*, which is one month for unirradiated direct-use material [S: IAEA Safeguards Glossary], at a specified detection probability and false-alarm rate [Avenhaus, von Stengel & Zamir 2002].

This paper asks one sub-question of the broader question "Do we need an international AI safety agency like the IAEA?":

> *Can an IAEA-style accountancy-and-inspection regime verify compute-based AI agreements, for which kinds of agreement, and for how long?*

We answer by building the AI analogue of the SQ and putting it into the IAEA's own detection logic. The answer is sharp. Which rule the agreement protects matters far more than how strong the agency is.

**Contributions.**

1. **A formal bridge from nuclear safeguards to compute governance.** We define the Significant Quantity of Compute (SQC) and the *accountancy ratio* ρ = SQC/(σS). We show that the IAEA-style detection goal (P_D = 0.9, α = 0.05) is equivalent to ρ ≥ 2.93 (Proposition 1). In words, verifiability is the share of world compute that a violating run needs, divided by the accounting error.
2. **A sign theorem for threshold rules** (Proposition 2). ρ decays exponentially at rate λ = ln2/h + ln g_S for fixed-capability (nonproliferation) agreements, including a pause once it begins. It decays at λ = ln g_S for fixed physical-FLOP caps, and at λ = ln g_S − ln g_F for frontier-trend caps, which is negative on current trends. Here h is the algorithmic-efficiency halving time, g_S the growth of the chip stock and g_F the growth of frontier training compute.
3. **A log-time result for institutions** (Proposition 3). Budget, intelligence pooling and measurement precision act multiplicatively on ρ. They therefore extend the verification horizon only by ln M/λ. Higher stakes raise the required ρ* only like √(2 ln(G/C)) (Proposition 4).
4. **Empirical calibration.** We use sourced trend estimates (algorithmic progress, frontier compute growth, chip stock, smuggling) and IAEA operating statistics. A Monte Carlo over 200,000 parameter draws plus a PRCC sensitivity analysis shows which inputs matter.
5. **Extensions that map to institutional design choices.** These cover protracted diversion (the CUMUF problem), jurisdictional decomposition (the largest compute holders are the hardest to verify), facility-inspection capacity in IAEA units, and hardware-enabled mechanisms.
6. **Four falsifiable predictions** with a concrete test protocol (Section 7).

We do not claim to settle whether an international agency is *desirable*. Its other functions, such as scientific assessment, standard-setting, crisis communication and access, are outside our model. We isolate the verification function, because that function is what makes the IAEA analogy distinctive and what most needs quantifying.

## 2 Related Work

**Institutional design.** Ho et al. [2023] set out four models of international AI institutions, including an Advanced AI Governance Organization patterned partly on the IAEA. Maas and Villalobos [2023] survey seven institutional functions and their precedents. Trager et al. [2023] propose an International AI Organization that certifies *jurisdictions* rather than firms, enforced through export controls. The UN HLAB final report [2024] recommends a globally inclusive, distributed governance architecture supported by a small AI office in the UN Secretariat. These works discuss verification qualitatively.

**Verification mechanisms.** Shavit [2023] proposes on-chip weight snapshots, training-transcript verification and supply-chain monitoring to detect large training runs. Baker [2023] draws lessons from nuclear arms-control verification and concludes that the challenges could be manageable with preparation. Wasil et al. [2024] catalogue ten verification methods, grouped as national technical means, access-dependent methods and hardware-dependent methods. Baker et al. [2025] describe six largely independent verification layers for rules on large-scale AI development. Sastry et al. [2024] argue that compute is detectable, excludable, quantifiable and concentrated, which makes it a natural governance lever. Kulp et al. [2024] analyse hardware-enabled governance mechanisms (HEMs). Heim and Koessler [2024] analyse training-compute thresholds as regulatory triggers.

**Erosion of thresholds.** Pilz, Heim and Brown [2023/2025] show that falling compute requirements diffuse capabilities to more actors ("access effect"). Koopmanschap and Barten [2026] define an "enforcement breaking point", the FLOP threshold below which a policy stops working. They argue qualitatively that detecting hidden capacity is the first sub-problem to become intractable. Gan [2026] models a bilateral pause as a game with costly monitoring. Gan shows that compute growth during a pause shortens the window in which cooperation can be sustained, and calls this the "agreement clock".

**Game theory.** Inspection games [Avenhaus et al. 2002] formalise verification as hypothesis testing against a strategic inspectee and underpin IAEA material balance evaluation. Armstrong, Bostrom and Shulman [2016] model AI races in which teams trade safety for speed.

**Gap.** Several works note that falling thresholds make verification harder [Pilz et al.; Koopmanschap & Barten; Gan]. None ports the IAEA's *quantitative* accountancy criterion (SQ, detection probability, false alarm, timeliness) to compute, calibrates it against an empirical inventory-uncertainty anchor, or separates the effect of the *threshold rule* (the sign of λ) from the effect of *institutional strength* (the intercept). Gan's "clock" is closest to our work. It concerns a bilateral pause and deterrence through response time. Our object is an agency's accountancy test, and we compare four threshold rules, jurisdictions, inspection capacity and HEMs.

## 3 Model

### 3.1 Primitives

Time t is measured in years. Compute is counted in H100-equivalents (H100e). One H100e delivers r = 989 TFLOP/s in dense BF16 [S: NVIDIA H100 datasheet; 1,979 TFLOP/s with sparsity, half without]. A training run achieves model-FLOPs utilisation u. Our baseline is u = 0.4, close to the 46.2% reported for PaLM [S: Chowdhery et al. 2022], and we vary it over [0.3, 0.5] (assumption).

*Threshold rules.* Let F_R(t) be the physical FLOP needed at time t to cross the threshold under rule R:

- **Fixed capability** (nonproliferation of a capability level): F_cap(t) = F₀·2^{−(t−t₀)/h}. The reference is F₀ = 2.1×10²⁵ FLOP, Epoch AI's estimate for GPT-4 [S], with t₀ = 2023.0 (assumption). The halving time h has a median of 8 months [S: Ho et al. 2024]. We model h as lognormal, with a 90% interval spanning the 5–14 month range reported in the arXiv version of that work.
- **Fixed physical-FLOP cap:** F_flop = 10²⁵ FLOP, the systemic-risk presumption threshold of EU AI Act Art. 51 [S].
- **Pause** at the 2026 frontier: the frontier FLOP of 2026 is frozen in capability terms, so after 2026 it erodes exactly like a fixed capability.
- **Frontier-trend cap:** F_front(t) = F₀·g_F^{t−t₀}, with g_F ≈ 4×/yr since 2018 [S: Sevilla & Roldán 2024]. We sample g_F from [3.0, 5.3]. The upper bound of 5.3×/yr is their 2010–2024 frontier estimate.

*Significant Quantity of Compute.* For a window τ within which a violating run must finish:

  **SQC_R(t) = F_R(t) / (r·u·τ).**

The baseline is τ = 90 days. Monte Carlo draws use τ ∈ {30, 90, 180, 365} days. The window plays the role of the IAEA timeliness goal. It is the time by which the agency must detect the run to keep a response possible. With baseline values, one H100e delivers 3.08×10²¹ FLOP per 90 days [C].

*Declared stock.* S(t) = S₀·g_S^{t−2026}. We take S₀ = 20.3×10⁶ H100e, the compute delivered by the end of 2025 [S: Epoch AI]. The growth rate g_S has a baseline of 2.3×/yr, the NVIDIA computing-power stock since 2019 [S: Epoch AI], and is sampled from [2.0, 3.3]. The upper bound of 3.3×/yr is all-designer capacity growth since 2022 [S: Epoch AI].

### 3.2 The accountancy test

The agency keeps a book inventory of declared chips. The *material unaccounted for*, MUF = book − physical, is approximately N(D, (σS)²), where D is the diverted quantity and σ the relative 1-sigma inventory uncertainty. This uncertainty bundles measurement, reporting, loss and unexplained flows, as in nuclear material balance evaluation [Avenhaus et al. 2002]. The agency runs a one-sided Neyman–Pearson test at false-alarm probability α.

**Definition (accountancy ratio).** ρ_R(t) = SQC_R(t) / (σ·S(t)).

**Proposition 1 (detection criterion).** For a diversion of one SQC, P_D = Φ(ρ − z_{1−α}). Hence P_D ≥ p* if and only if ρ ≥ ρ* = z_{1−α} + z_{p*}. For α = 0.05 and p* = 0.9, ρ* = 1.645 + 1.282 = **2.926** [C]. Equivalently, the run's share of world compute, SQC/S, must be at least ρ*·σ.

*Proof.* Under H₀, MUF/(σS) ~ N(0,1), and the test rejects when MUF/(σS) > z_{1−α}. Under H₁ with D = SQC, the statistic is N(ρ, 1), so P_D = 1 − Φ(z_{1−α} − ρ) = Φ(ρ − z_{1−α}). ∎

We checked Proposition 1 by simulation (200,000 draws per point). At ρ = 1, 2, 2.926 and 4, the analytic P_D values are 0.260, 0.639, 0.900 and 0.991, and the simulated values are 0.259, 0.638, 0.899 and 0.991 [C].

### 3.3 Dynamics: the sign theorem

**Proposition 2 (sign of erosion).** For every rule, ln ρ_R(t) is linear in t (for the pause, from the pause date on) with slope −λ_R, where

- λ_cap = λ_pause = ln2/h + ln g_S,
- λ_flop = ln g_S,
- λ_front = ln g_S − ln g_F.

*Proof.* Substitute the definitions: ln ρ = ln F_R − ln(r u τ σ S₀) − (t−2026) ln g_S. The term ln F_R has slope −ln2/h, 0 and ln g_F respectively. ∎

At baseline, λ_cap = 1.87/yr, meaning ρ shrinks 6.5× per year. λ_flop = 0.83/yr and λ_front = −0.55/yr [C]. In the Monte Carlo (Section 4.2), λ_front < 0 in 98.5% of draws.

**Corollary (horizon).** When λ > 0, the last date at which the detection goal holds is t* = t₁ + ln(ρ(t₁)/ρ*)/λ. When λ < 0 and ρ(t₁) < ρ*, the goal first becomes achievable at the same formula, and the date lies in the future.

A frontier-trend cap cannot have λ_front < 0 forever. ρ = SQC/(σS) ≤ 1/σ, and the frontier run's share of world compute cannot exceed 1. At baseline, a 90-day frontier run needs 2.2% of the stock in 2026, 6.5% in 2028 and 19.7% in 2030 [C]. Before 2030, then, g_F must fall to g_S, which sends λ_front to 0, and ρ_front levels off at (frontier share)/σ.

### 3.4 Institutions buy log-time

**Proposition 3 (log-time).** Any change that multiplies ρ by a constant M > 1 moves the horizon by Δt* = ln M/λ. Examples are lowering σ by a factor M, a budget that buys M-times-better measurement, or pooling m independent, equally precise inventory sources, which gives M = √m because the pooled variance is σ²/m.

*Proof.* This follows directly from the corollary. ∎

The policy reading is that institutional strength changes the intercept of ln ρ(t), while the threshold rule sets its slope.

### 3.5 Stakes and deterrence

In the canonical inspection game [Avenhaus et al. 2002], a state gains G from an undetected violation and loses C if caught. It is deterred exactly when P_D ≥ G/(G+C).

**Proposition 4.** The required ratio is ρ*(k) = z_{1−α} + Φ⁻¹(k/(1+k)), with k = G/C. As k → ∞, Φ⁻¹(1 − 1/(1+k)) ~ √(2 ln k). Higher stakes therefore raise ρ* only sub-logarithmically, and by Proposition 3 they cost horizon only as √(2 ln k)/λ.

### 3.6 Extensions

**(a) Protracted diversion.** A diverter can spread D over k balance periods. Assume independent per-period errors. The cumulative-MUF (CUMUF) test then has ratio ρ/√k. A per-period test with Šidák-corrected false alarms does worse.

**(b) Jurisdictions.** A state that diverts from its own declared stock S_j faces noise σ_j S_j, not σS. Its ratio is therefore ρ_j = SQC/(σ_j S_j). Verification is hardest for the largest holders.

**(c) Facility inspection.** Suppose an agency makes I inspections per year at random among N candidate sites, and each visit detects an ongoing run with probability q. Detection within window τ has probability 1 − exp(−qIτ/N). The goal p* is therefore met only if N ≤ N_max = qIτ/(−ln(1−p*)). We bound the number of candidate sites above by S/SQC, the number of simultaneous threshold-crossing runs the stock could host.

**(d) Hardware-enabled mechanisms.** Suppose every chip shipped after t_H carries a working HEM. A fraction ε of post-t_H chips is effectively uncontrolled through tampering or defeat. Legacy chips retire exponentially with mean life ℓ = 5 yr (assumption). The noise term then becomes √((σ_L L(t))² + (ε(S(t) − S(t_H)))²), where L(t) = S(t_H)e^{−(t−t_H)/ℓ}. If ε = 0, the decay rate under fixed-capability or pause rules becomes λ = ln2/h − 1/ℓ. HEMs remove the stock-growth term but leave the algorithmic term in place.

### 3.7 An empirical anchor for σ

Nobody publishes an inventory uncertainty for AI chips. We use the closest observable proxy. Epoch AI estimates that 660,000 H100e were smuggled to China through the end of 2025, with a 90% CI of 290,000–1.6 million [S]. The width of that interval is a direct statement of how uncertain informed analysts are about where a large block of chips sits. Treating it as approximately normal gives a 1-sigma of (1.6M − 0.29M)/(2·1.645) ≈ 398,000 H100e. That is **σ_emp ≈ 1.96%** of the end-2025 stock [C]. The median smuggled quantity alone is 3.25% of the stock [C].

For comparison, nuclear material balance studies of bulk-handling plants commonly assume about 1% measurement uncertainty at input [S: KIT near-real-time accountancy study]. The AI-chip number is not a measured σ; it is a stand-in for the current state of knowledge. We therefore report results for σ from 10⁻⁶ to 5%.

## 4 Results

### 4.1 Baseline values in 2026

**Table 1.** SQC, required inventory precision and detection at the empirical σ, at t = 2026.0, baseline parameters [C].

| Rule | SQC (H100e, 90 d) | Share of world stock | σ required for ρ ≥ ρ* | ρ at σ_emp ≈ 2% | P_D at σ_emp |
|---|---|---|---|---|---|
| Fixed capability (GPT-4 level) | 302 | 1.5×10⁻⁵ | 5.1×10⁻⁶ | 7.6×10⁻⁴ | ≈0.05 (= α) |
| Physical cap 10²⁵ FLOP | 3,250 | 1.6×10⁻⁴ | 5.5×10⁻⁵ | 8.2×10⁻³ | ≈0.05 |
| Pause / frontier (2026) | 437,000 | 2.15×10⁻² | 7.35×10⁻³ | 1.08 | 0.29 |

Three facts stand out.

1. A GPT-4-level run in a 90-day window now needs about 302 H100e. That is roughly 38 eight-GPU servers, against a stock of 20 million.
2. Even the 2026 frontier scale fails the detection goal at today's empirical inventory uncertainty, with ρ ≈ 1.08 and P_D ≈ 0.29. It would pass if σ fell to about 0.73%, a 2.7-fold improvement.
3. GPT-4-level nonproliferation would need inventory accounting accurate to five parts per million, far below anything nuclear safeguards achieve for bulk material.

Figure 1 (`results/fig1_sqc_vs_floor.png`) plots SQC_R(t) against the detectable-diversion floor ρ*σS(t) for σ ∈ {2%, 0.1%, 10⁻⁵}. Under the fixed-capability rule, the SQC curve falls below one 8-GPU server in 2029.5 at baseline [C], with a Monte Carlo 90% interval of 2026.5–2033.9.

### 4.2 Monte Carlo and sensitivity

We drew 200,000 parameter vectors: h (lognormal), g_S ~ U[2.0, 3.3], g_F ~ U[3.0, 5.3], u ~ U[0.3, 0.5], τ from {30, 90, 180, 365} days, and σ log-uniform. Results [C]:

- **λ_cap:** median 2.01/yr, 90% interval [1.51, 2.74]. This is a 7.5× annual fall in ρ (4.5×–15.5×).
- **λ_front:** median −0.45/yr, 90% interval [−0.81, −0.08]. P(λ_front < 0) = 0.985.
- **Fixed-capability horizon** (back-cast with the stock growth law; Section 6): at σ = 10⁻⁴ the median is 2024.3 [2023.6, 2025.4]; at σ = 10⁻³ it is 2023.2 [2022.4, 2024.1]. In at least 95% of draws, compute accounting with σ ≥ 10⁻⁴ could no longer verify GPT-4-level nonproliferation by 2026.
- **Pause horizon** (a pause at the 2026 frontier): at σ = 10⁻³ the median is 2026.9 [2026.1, 2027.9]; at σ = 10⁻⁴ it is 2028.0 [2027.1, 2029.3]. A pause whose verification rests on chip accounting stays verifiable for only about one to two years after it starts, even with accounting 20–200 times better than today's proxy.
- **Frontier cap ρ in 2026 at σ = 2%:** median 0.84 [0.17, 5.8].

*PRCC of the pause horizon* [C]:

| Input | PRCC |
|---|---|
| log σ | −0.98 |
| τ | −0.86 |
| h | +0.78 |
| g_F | +0.67 |
| g_S | −0.48 |
| u | −0.27 |

Measurement precision and the choice of window dominate, followed by the speed of algorithmic progress.

Figure 2 (`results/fig2_horizon_vs_sigma.png`) shows t* as a function of σ for each rule, with the Monte Carlo band for the fixed-capability rule.

### 4.3 Log-time

By Proposition 3, a tenfold improvement in accountancy SNR (M = 10) buys **14.8 months** under the fixed-capability rule and 33 months under the physical cap [C]. M = 1000 buys 44 months and 100 months respectively. Pooling ten independent national inventories in one agency (M = √10) buys 7.4 months under the fixed-capability rule [C]. See Figure 3 (`results/fig3_logtime.png`).

The result cuts both ways for the "do we need an agency" debate. Pooling is a real economy of scope that a single national verifier cannot match. Against an eroding threshold, though, its value is measured in months.

### 4.4 Stakes

ρ* rises slowly with the gain-to-penalty ratio k = G/C [C]:

| k | 1 | 9 | 99 | 999 | 10⁴ |
|---|---|---|---|---|---|
| ρ* | 1.64 | 2.93 | 3.97 | 4.74 | 5.36 |

Note that k = 9 reproduces the conventional 90% goal. Moving from k = 9 to k = 10⁴ (a "winner-takes-all" race) costs ln(5.36/2.93)/λ, which is 3.9 months under the fixed-capability rule. Stakes matter much less than the threshold rule.

### 4.5 Protracted diversion

With ρ = ρ* for the full diversion, spreading it over k balance periods gives the following detection probabilities [C, 200,000 simulations]:

| k | CUMUF P_D (simulated / analytic) | Per-period test P_D |
|---|---|---|
| 1 | 0.900 / 0.900 | 0.899 |
| 3 | 0.519 / 0.518 | 0.334 |
| 12 | 0.213 / 0.212 | 0.098 |

A τ-window run assembled from chips borrowed across several reporting periods therefore needs either shorter balance periods or continuous (near-real-time) accounting. The IAEA developed near-real-time accountancy for bulk-handling plants for exactly this reason. In AI the analogue is continuous telemetry, which brings us back to HEMs.

### 4.6 Jurisdictions

At σ_emp and the 2026 frontier scale, the ratio for diversion inside a jurisdiction holding share s of world compute is [C]:

| s | 5% | 25% | 50% | 75% |
|---|---|---|---|---|
| ρ_j | 21.9 | 4.39 | 2.19 | 1.46 |

Small holders can be verified even with today's poor accounting. Holders of half or more of world compute cannot. The fixed-capability rule fails everywhere (ρ_j ≤ 0.015). This structural fact matters for institutional design. An IAEA-style regime is easiest to run against the states whose compliance matters least for frontier risk, which mirrors the NPT's experience that verification is easiest where weapons ambitions are weakest.

### 4.7 Facility inspection in IAEA units

In 2023 the IAEA carried out 3,136 in-field verifications and spent 14,302 days in the field at more than 1,300 facilities and locations outside facilities [S: IAEA SIR 2023]. Its regular budget request is about €428M per year, of which safeguards is about 39% [S: VCDNP], roughly €167M [C].

Take I = 3,136 inspections per year and per-visit detection q = 0.9. The facility regime then meets P_D = 0.9 over at most N_max = 302 candidate sites for a 90-day window, or 102 for a one-month window [C]. The upper bound S/SQC on candidate sites (Figure 4, `results/fig4_inspection.png`) gives the budget, in multiples of IAEA-scale inspection capacity, needed to keep up [C]:

| Rule | 2026 | 2028 | 2030 |
|---|---|---|---|
| Fixed capability | 223× | 9,400× | 400,000× |
| Pause (2026 frontier) | 0.15× | 6.5× | 275× |

S/SQC is an upper bound; the true number of sites able to host one SQC depends on how facility sizes are distributed, which we could not measure (Section 6). For a frontier cap, the bound stays below 50 sites through 2030.

### 4.8 Hardware-enabled mechanisms

Figure 5 (`results/fig5_hem.png`) assumes a best case: every chip shipped from 2026 carries a HEM, and legacy chips retire with a mean life of 5 years [C].

- **Pause, legacy σ_L = 2%:** the goal is never met (ρ = 1.08 in 2026, falling).
- **Pause, legacy σ_L = 0.2%:** the horizon is 2027.55 with HEMs and ε = 0, against 2026.69 without HEMs at the same σ. HEMs add about 10 months.
- **Pause, σ_L = 0.2%, ε = 10⁻³:** the horizon is 2027.14.
- **Rate under HEMs:** the remaining decay rate is ln2/h − 1/ℓ = 0.84/yr. The algorithmic term persists.
- **Frontier cap:** the goal is met from about 2026.6–2026.7 even with σ_L = 2% and ε = 1%. HEMs plus a frontier-relative rule are mutually reinforcing.

## 5 Discussion

**What the analogy gets right.** The IAEA works because three conditions hold together. Its SQ is a *physical constant*: 8 kg of plutonium was a weapon's worth in 1970 and still is. Its material is concentrated in a modest number of declared facilities (about 1,300 [S]). And its accountancy error, around 1% of throughput at bulk facilities, is small relative to one SQ at most facilities.

Compute meets the second and third conditions only for frontier-scale quantities, and for those only with better accounting than exists today. It fails the first condition entirely. The SQC of a *fixed capability* is not constant: it falls by a factor of 2.8 per year from algorithmic progress alone, while the haystack grows 2.3–3.3× per year. The resulting λ of about 1.9/yr compounds to a factor of about 6.5 per year. No institutional parameter in our model offsets that except by buying ln M/λ of time.

**What this implies for "is an IAEA for AI needed?"** The verification function of an IAEA-like body is quantitatively viable for one class of agreement: *ceilings relative to the frontier*, such as "no single run above X% of world compute" or "no run above the agreed frontier trend". For these, ρ is flat or rising, and the binding constraint is getting σ below about a third of the frontier share. Today that means about 0.7%, roughly a 3× improvement over our empirical proxy. Pooling national intelligence (M = √m), chip registries and HEMs are plausible routes to that level, and this is where a multilateral agency adds value that bilateral verification cannot.

The same function is not viable, beyond a few years, for agreements that *freeze a capability level*. That includes nonproliferation of a fixed dangerous capability and a pause that is not continually renegotiated. This holds at any plausible budget. Such agreements need a different verification object: models and deployments (evaluations, licensing, know-your-customer rules) rather than chips. Alternatively, they must accept that compute accounting only covers a decreasing tail of actors. This refines Koopmanschap and Barten's qualitative "enforcement breaking point" into a dated, parameterised criterion. It also complements Gan's [2026] "agreement clock": our clock is the agency's accountancy horizon, and its speed is set by the threshold rule.

**Design recommendations that follow from the model.** These are conditional on the model's assumptions.

- (i) Write treaty thresholds relative to aggregate compute or the frontier, not as fixed FLOP numbers. Proposition 2 shows that a fixed FLOP cap erodes at λ = ln g_S even without algorithmic progress.
- (ii) Put a measurable inventory-uncertainty target, a "σ budget", into the agency's mandate, as IAEA International Target Values do for measurement systems.
- (iii) Prioritise continuous accounting over periodic balances (Section 4.5).
- (iv) Expect verification to be hardest for the largest holders. Design reciprocal measures, such as mutual HEMs and co-located inspectors, for them specifically, rather than imposing uniform obligations.
- (v) Treat any compute-based nonproliferation commitment as having a sunset date, set by t*.

## 6 Limitations

- **σ is a proxy.** Our empirical anchor is the width of a modelled smuggling estimate. It is not a measured inventory error of any real accounting system. A purpose-built registry might do much better, which is why we report results across five orders of magnitude.
- **Gaussian, single-test model.** Real MUF errors are correlated over time, partly systematic, and the IAEA combines accountancy with containment and surveillance. Non-Gaussian tails or systematic biases would change ρ*, though not the sign result in Proposition 2.
- **Algorithmic progress is measured on perplexity.** Ho et al. [2024] measure compute savings for reaching a fixed language-modelling loss. Dangerous capabilities may erode faster or slower, for example through distillation or post-training gains. Our fixed-capability results inherit this uncertainty. h is the second-most important parameter after σ and τ in the PRCC.
- **Back-casting.** Fixed-capability horizons earlier than 2026 come from extrapolating the 2026-anchored stock law backwards. They should be read as "already past", not as precise historical dates.
- **Physical consistency of the frontier rule.** At g_F > g_S, frontier runs would consume about 20% of world compute by 2030. The trend must therefore slow, and λ_front → 0. Our frontier-rule results beyond about 2030 are illustrative only.
- **Site counts.** We bound candidate sites by S/SQC because we could not obtain data on facility size distributions. The true count is lower, and the budget multiples in Section 4.7 are upper bounds.
- **Window τ.** Distributed or decentralised training, rented cloud compute and longer runs all change the effective τ. τ is the second most influential parameter, so this is a real sensitivity rather than a detail.
- **Scope.** We model only the accountancy function. We say nothing about an agency's legitimacy, scientific assessment, standard-setting, benefit-sharing, or the politics of creating it. Those may justify an agency even where verification is weak.
- **No large-model experiments.** All results are analytical, Monte Carlo or calibration results computed on CPU.

## 7 Falsifiable predictions and test protocol

Each prediction states a measurable quantity, a data source and a refutation condition.

- **P1 (fixed-capability infeasibility).** The first open-weights model to match GPT-4 on a pre-registered benchmark suite, using at most 2.46×10²² training FLOP (8 H100e × 90 days at u = 0.4), will appear by 2034. The model's 90% interval is 2026.5–2033.9 [C].
  - *Test:* Use Epoch AI's model database. Fix the suite, for example the benchmarks used in the original GPT-4 technical report, and evaluate held-out versions to limit contamination.
  - *Refuted if:* no such model exists by 2034.0.
- **P2 (accounting quality).** Absent mandatory HEMs, published 90% intervals for unaccounted or diverted AI chips will imply σ ≥ 0.5% of global stock through 2028.
  - *Test:* Apply the Section 3.7 transformation to each annual smuggling or diversion estimate from Epoch AI or similar sources.
  - *Refuted if:* any year's implied σ < 0.5%.
- **P3 (frontier share).** The share of world H100e stock needed for the largest training run in a 90-day window will rise from 2026 to 2028. Equivalently, λ_front < 0 (P = 0.985 [C]).
  - *Test:* Each year, divide the largest disclosed or estimated run (Epoch AI) by stock (Epoch AI chip sales).
  - *Refuted if:* the 2028 share is below the 2026 share. In that case frontier-relative verification gets *harder*, and our main positive recommendation weakens.
- **P4 (institutional behaviour).** Any intergovernmental AI body created with a compute-based verification mandate will, within three years, either redefine its thresholds relative to aggregate or frontier compute, or shift verification to model or deployment evaluation.
  - *Test:* Code mandate documents and their amendments.
  - *Refuted if:* the body keeps a fixed absolute FLOP threshold verified by compute accounting for more than three years while claiming effective verification.

## 8 Conclusion

Should there be an IAEA for AI? Our answer concerns the IAEA's core function, verification by accountancy. That function transfers to AI chips only for agreements whose threshold scales with the frontier or with world compute. For those agreements, the work is to cut inventory uncertainty from about 2% to below 1%, and a multilateral agency that pools national data and runs continuous accounting is well placed to do it. For agreements that freeze a capability level, compute accounting fails within a few years at any plausible budget, because the significant quantity of compute shrinks while the haystack grows. Institutions buy only log-time against that combined rate. These agreements need a different verification target. The broader question of whether an IAEA for AI is needed can therefore be narrowed to a measurable one: what is σ, and is the threshold rule's λ negative? We provide the tools to answer it.

---

## References

All references were checked to exist via web search on 2026-11-22. Data sources are listed with their publishers.

1. Armstrong, S., Bostrom, N., & Shulman, C. (2016). Racing to the precipice: a model of artificial intelligence development. *AI & Society*, 31(2), 201–206. doi:10.1007/s00146-015-0590-y
2. Avenhaus, R., von Stengel, B., & Zamir, S. (2002). Inspection games. In R. Aumann & S. Hart (Eds.), *Handbook of Game Theory with Economic Applications*, Vol. 3, Ch. 51, pp. 1947–1987. Elsevier. doi:10.1016/S1574-0005(02)03014-X
3. Baker, M. (2023). Nuclear Arms Control Verification and Lessons for AI Treaties. arXiv:2304.04123.
4. Baker, M., Kulp, G., Marks, O., Brundage, M., & Heim, L. (2025). Verifying International Agreements on AI: Six Layers of Verification for Rules on Large-Scale AI Development and Deployment. arXiv:2507.15916 (also RAND WR-A4077-1).
5. Bengio, Y., et al. (2025). International AI Safety Report. arXiv:2501.17805.
6. Chowdhery, A., et al. (2022). PaLM: Scaling Language Modeling with Pathways. arXiv:2204.02311.
7. Epoch AI (2025). The stock of computing power from NVIDIA chips is doubling every 10 months. Data insight, 13 Feb 2025. https://epoch.ai/data-insights/nvidia-chip-production
8. Epoch AI (2026). Global AI computing capacity is doubling every 7 months. Data insight, 9 Jan 2026. https://epoch.ai/data-insights/ai-chip-production
9. Epoch AI. AI Chip Sales data explorer and data-centre research (end-2025 delivered compute ≈ 20.3M H100e). https://epoch.ai/data/ai-chip-sales
10. Epoch AI. Chip smuggling (estimate of compute smuggled to China through 2025). https://epoch.ai/publications/chip-smuggling
11. Epoch AI. Models over 1e25 FLOP (GPT-4 training compute estimate 2.1×10²⁵). https://epoch.ai/data-insights/models-over-1e25-flop
12. European Union (2024). Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 51. https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-51
13. Gan, L. (2026). Agreements Have a Clock: Compute Growth and the Verification of AI Restraint. arXiv:2611.04272.
14. Heim, L., & Koessler, L. (2024). Training Compute Thresholds: Features and Functions in AI Regulation. arXiv:2405.10799.
15. Ho, A., Besiroglu, T., Erdil, E., et al. (2024). Algorithmic progress in language models. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). arXiv:2403.05812.
16. Ho, L., Barnhart, J., Trager, R., Bengio, Y., Brundage, M., Carnegie, A., Chowdhury, R., Dafoe, A., Hadfield, G., Levi, M., & Snidal, D. (2023). International Institutions for Advanced AI. arXiv:2307.04699.
17. International Atomic Energy Agency (2022). *IAEA Safeguards Glossary*, 2022 edition. Vienna: IAEA.
18. International Atomic Energy Agency (2024). *Safeguards Implementation Report for 2023* and *Safeguards Statement for 2023*. Vienna: IAEA.
19. Koopmanschap, R., & Barten, O. (2026). How to Catch a GPU: A Taxonomy of Verification and Enforcement Mechanisms for International AI Agreements. arXiv:2607.22619.
20. Kulp, G., et al. (2024). Hardware-Enabled Governance Mechanisms: Developing Technical Solutions to Exempt Items Otherwise Classified Under Export Control Classification Numbers 3A090 and 4A090. RAND Working Paper WR-A3056-1.
21. Maas, M. M., & Villalobos, J. J. (2023). International AI Institutions: A Literature Review of Models, Examples, and Proposals. Legal Priorities Project / Institute for Law & AI, AI Foundations Report.
22. NVIDIA (2023). NVIDIA H100 Tensor Core GPU datasheet.
23. Pilz, K., Heim, L., & Brown, N. (2023). Increased Compute Efficiency and the Diffusion of AI Capabilities. arXiv:2311.15377; published in *Proceedings of the AAAI Conference on Artificial Intelligence* 39 (2025). doi:10.1609/aaai.v39i26.34971
24. Sastry, G., Heim, L., Belfield, H., et al. (2024). Computing Power and the Governance of Artificial Intelligence. arXiv:2402.08797.
25. Sevilla, J., & Roldán, E. (2024). Training Compute of Frontier AI Models Grows by 4–5x per Year. Epoch AI. https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year
26. Shavit, Y. (2023). What does it take to catch a Chinchilla? Verifying Rules on Large-Scale Neural Network Training via Compute Monitoring. arXiv:2303.11341.
27. Trager, R., Harack, B., Reuel, A., Carnegie, A., Heim, L., Ho, L., Kreps, S., Lall, R., Larter, O., Ó hÉigeartaigh, S., Staffell, S., & Villalobos, J. J. (2023). International Governance of Civilian AI: A Jurisdictional Certification Approach. arXiv:2308.15514.
28. United Nations High-level Advisory Body on Artificial Intelligence (2024). *Governing AI for Humanity: Final Report*. New York: United Nations.
29. Vienna Center for Disarmament and Non-Proliferation (2023). Navigating the IAEA Budget Process (briefing). https://vcdnp.org/wp-content/uploads/2023/08/print_navigating_the_iaea_budget_process_rev.1.pdf
30. Wasil, A. R., Reed, T., Miller, J. W., & Barnett, P. (2024). Verification methods for international AI agreements. arXiv:2408.16074.
31. Karlsruhe Institute of Technology. Study of near-real-time accountancy (NRTA) for reprocessing plants of different sizes (assumes 1% input measurement uncertainty and a 5% false-alarm rate). KIT repository record: https://publikationen.bibliothek.kit.edu/270022357
