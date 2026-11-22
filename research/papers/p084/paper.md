# Weakest-Link Geography: A Leontief Disruption Model of AI-Accelerator Supply Risk from Logic, CoWoS Packaging and HBM

**Target venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES 2027), quantitative socio-technical track. (Alternative: *Risk Analysis*.)

---

## Abstract

How risky is the supply chain for frontier AI accelerators? Public discussion mostly asks "how concentrated is TSMC?" That question misses how the supply chain is built. An accelerator needs three inputs in fixed proportions: advanced logic dies, CoWoS-class 2.5D packaging, and high-bandwidth memory (HBM). AI demand also takes very different shares of each input. Public estimates say the four largest AI chip designers used about 12% of advanced logic output in 2025 but about 90% of CoWoS capacity and HBM supply. We build a weekly Monte Carlo model of accelerator output. Output is Leontief in the three inputs. Each input is made at sites in Taiwan, Korea and elsewhere, with capacity shares taken from public industry estimates. Five hazard classes are simulated: site incidents, Taiwan earthquakes, Korean regional shocks, HBM yield or qualification shocks, and a Taiwan cut-off (blockade or quarantine). The cut-off probability is a swept scenario parameter, not a forecast. We compare two allocation regimes, priority (AI outbids other users) and proportional rationing. We prove three simple propositions. (1) *Slack asymmetry*: under priority allocation, AI output survives the loss of up to 1−s of an input's capacity, where s is AI's share of that input. That is 88% for logic but only 10% for CoWoS and HBM. (2) *Weakest-link geography*: during a regional cut-off, output equals the smallest outside-region coverage ratio across inputs. The value of hardening inputs is therefore supermodular, and hardening one input alone can be worth zero. (3) *Buffer bound*: an inventory of b weeks removes at most b/E[D] of the expected loss from a full outage of mean length E[D]. In simulation (100,000 years per configuration), every non-cut-off hazard combined causes an expected annual loss of 0.07% (priority) to 0.19% (proportional) of accelerator output. A 1%/yr Taiwan cut-off raises this to 0.33–0.45% and accounts for 57–76% of the expected loss and almost all of the 99% CVaR (27–29%). Under priority allocation, hardening logic alone reduces expected loss by about 0%. Hardening logic and packaging together reduces it by 75%; the interaction term is 52 percentage points. Analytically, keeping 50% of AI output through a Taiwan cut-off needs about 6% of advanced logic capacity (on TSMC-compatible processes) outside Taiwan, but about 45% of CoWoS-class capacity outside Taiwan. At the industry capacity TrendForce reports for 2026, that is 77–90 thousand wafers per month. The policy implication is that the binding geographic risk sits in packaging and memory, not in front-end logic fabs. We state falsifiable predictions and an event-study protocol to test them.

## 中文摘要

先进AI加速器的供应链风险有多大？公共讨论通常只问"台积电有多集中"，这忽略了供应链的结构。一颗加速器需要按固定比例组合三种投入：先进逻辑芯片、CoWoS类2.5D封装和高带宽内存（HBM），而AI需求在三者中所占份额差别很大。公开估计显示，2025年四大AI芯片设计商约占先进逻辑产量的12%，却约占CoWoS产能和HBM供应的90%。我们构建了一个按周计算的蒙特卡洛模型：加速器产出对三种投入取Leontief最小值；各投入在台湾、韩国及其他地区的产能份额来自公开行业估计；模拟五类冲击，即工厂事故、台湾地震、韩国区域冲击、HBM良率/认证冲击和台湾断供（封锁/隔离）。台湾断供概率作为情景参数扫描，不作预测。我们比较两种分配机制：优先分配（AI出价压过其他用户）和按比例配给。我们证明三个简单命题。（1）"冗余不对称"：在优先分配下，某投入损失不超过1−s的产能时AI产出不受影响（s为AI占该投入的份额），即逻辑可承受88%，CoWoS与HBM仅10%。（2）"最弱环节地理"：区域断供期间，产出等于各投入区域外覆盖率的最小值，因此加固各投入的价值是超模的，单独加固某一投入的价值可能为零。（3）"库存上界"：b周库存最多消除平均长度为E[D]的完全停产所致期望损失的b/E[D]。在模拟中（每种配置10万年），所有非断供冲击合计造成的年期望产出损失为0.07%（优先分配）至0.19%（按比例配给）。加入每年1%的台湾断供后，期望损失升至0.33–0.45%，其中57–76%来自断供，99% CVaR（27–29%）几乎全部来自断供。在优先分配下，只加固逻辑使期望损失下降约0%，同时加固逻辑与封装下降75%，交互项为52个百分点。解析结果表明：要在台湾断供期间保住50%的AI产出，台湾以外只需约6%的先进逻辑产能（须为台积电兼容工艺），却需要约45%的CoWoS类产能；按TrendForce报告的2026年行业产能计算，约为每月7.7–9万片晶圆。政策含义是，具有约束力的地理风险在封装和内存，而不在前道逻辑晶圆厂。文中给出可证伪的预测和事件研究检验方案。

---

## 1 Introduction

Frontier AI depends on a small number of physical goods. Accelerators such as NVIDIA's H100/B200 or Google's TPUs need advanced logic dies made at TSMC on 3–5 nm-class processes. They also need HBM stacks, made by SK hynix, Samsung and Micron, and CoWoS-class 2.5D packaging, which joins logic and memory on one interposer and is overwhelmingly done in Taiwan [1, 2, 13]. Several sources document the concentration. All sub-10 nm capacity was in Taiwan (92%) and South Korea (8%) in the SIA/BCG baseline [4]. Compute-governance work treats this concentration as a lever for policy [6]. Macroeconomic scenario work puts the global cost of a Taiwan conflict at about 10% of world GDP, with the semiconductor shock as a central channel [18]. What is missing is a *quantitative, structure-aware* answer to the question this paper addresses: **how large is the supply risk to AI-accelerator output specifically, and which part of the supply chain carries it?**

Most risk discussion counts concentration input by input ("TSMC makes most advanced logic", "three firms make all HBM"). We argue that input-by-input counts mislead for AI accelerators, for two structural reasons that published data now let us quantify.

1. **Complementarity.** Inputs combine in fixed proportions. A logic die without a CoWoS slot and HBM stacks is not an accelerator. Firm-level evidence from the 2011 Tōhoku earthquake shows that imported-input supply chains behave close to Leontief in the short run [10], and that disruptions spread along supply chains [9]. When inputs cannot substitute for each other, the network structure determines how micro shocks become aggregate losses [8, 11].
2. **Asymmetric slack.** AI takes very different shares of each input. Epoch AI estimates that in 2025 the four largest AI chip designers took about 90% of global CoWoS capacity and HBM supply, but only about 12% of advanced logic die output [1]. If AI demand can outbid smartphones and PCs for logic, the logic stage has a large hidden buffer for AI. CoWoS and HBM have almost none.

Together these imply that the *geography of the weakest link* matters more than the concentration of any one input. Our sub-question is narrow and tractable: **given public capacity-share data, how does the joint structure of complementarity, asymmetric slack, regional co-location and inventory determine the distribution of AI-accelerator output losses, and which mitigations actually move that distribution?**

**Contributions.**

- **C1 (model).** A transparent, reproducible disruption model of AI-accelerator output. It has three Leontief inputs, 21 sites in three regions, five hazard classes, two allocation regimes (priority vs proportional), an inventory buffer, and a foundry-porting lag. Every parameter is tagged as sourced, assumed, or a scenario sweep (Section 3; code in `code/`).
- **C2 (theory).** Three propositions with short proofs (Section 4): (P1) slack asymmetry, giving a tolerance of 1−s_i per input; (P2) weakest-link regional resilience, with a closed-form capacity requirement o_i ≥ y·s_i and *supermodularity* of the value of hardening inputs; (P3) a b/E[D] upper bound on what inventory can do against long outages.
- **C3 (quantification).** Monte Carlo estimates of expected loss, VaR and CVaR across Taiwan cut-off probabilities from 0% to 5%/yr, a decomposition by hazard class, Shapley-value attribution of hardening across inputs, and one-at-a-time sensitivity analysis (Section 5). The main empirical-model finding: routine hazards are largely absorbed, while tail risk is almost entirely a Taiwan cut-off risk that runs through *packaging*, not logic.
- **C4 (policy translation and falsifiable predictions).** In physical units, keeping 50% of AI output through a Taiwan cut-off requires about 76.5–90k wafers/month of CoWoS-class capacity outside Taiwan at 2026 industry scale, against roughly 6% of advanced logic. We also state predictions that an event study can falsify (Section 6.3).

We do not claim to forecast conflict. The Taiwan cut-off probability is a swept parameter. Our contribution is the conditional structure: how much risk each structural feature creates *given* a hazard environment, and which interventions are complements.

## 2 Related Work

**AI compute supply chains and governance.** Khan and Mann [5] set out why leading-edge, specialised AI chips matter and why their supply chain is concentrated in a few allied economies. Sastry et al. [6] argue that compute is a governance lever partly because it is "produced via an extremely concentrated supply chain". Miller [7] gives the historical and geopolitical background, with TSMC at the centre. Epoch AI's scaling analysis [3] treats chip manufacturing as one of four constraints on scaling to 2030. Its 2026 data insight [1] and component dataset [2] identify logic dies, HBM and CoWoS as the three binding inputs and estimate AI's share of each. These works describe concentration and bottlenecks in steady state. None of them models *disruption*, meaning stochastic capacity loss, recovery, inventory and allocation, or shows how complementarity among the three inputs shapes the loss distribution.

**Industry structure data.** The SIA/BCG study [4] is the standard source for regional capacity shares (92% of sub-10 nm capacity in Taiwan). It also counts more than 50 points in the semiconductor chain where one region holds over 65% share. TrendForce reports give HBM vendor shares (SK hynix 52.3%, Samsung 28.7%, Micron 19% projected for 2025) [14] and CoWoS capacity (about 70–75k wafers/month in 2025; 120–140k at TSMC plus 50–60k at OSAT partners in 2026) [15]. Micron's largest HBM site, which does DRAM, test and HBM3E stacking, is in Taichung, Taiwan [16]. Arizona-made TSMC dies are still sent to Taiwan for advanced packaging, and US CoWoS capacity is not expected before roughly 2027–2029 [17].

**Production networks and shock propagation.** Acemoglu et al. [8] show that input-output linkages can turn idiosyncratic shocks into aggregate fluctuations. Baqaee and Farhi [11] show that with low elasticities of substitution, shocks to critical sectors have strongly non-linear, fat-tailed aggregate effects. Carvalho et al. [9] and Boehm, Flaaen and Pandalai-Nayar [10] use the 2011 Japan earthquake to show propagation along supply chains and near-Leontief input complementarity. Our model applies the Leontief limit to a three-input product, which matches the physical bill of materials of an accelerator.

**Operations research on disruption risk.** Snyder et al. [13] review 180 works on OR/MS disruption models. Tomlin [12] shows that the best mitigation depends on whether disruptions are frequent and short or rare and long. Inventory suits the first case; sourcing diversification suits the second. Simchi-Levi, Schmidt and Wei [19] introduced time-to-recover (TTR) based risk exposure, applied at Ford. We adapt TTR-style site modelling and test Tomlin's frequency/duration dichotomy against AI-specific data. Proposition 3 states its inventory half in closed form for lognormal durations.

**Gap.** We found no prior work that (i) combines the three-input Leontief structure of AI accelerators with (ii) AI's asymmetric demand shares and (iii) regional co-location of inputs in a stochastic disruption model. Prior work also does not draw out the implication that hardening one input, such as building front-end fabs abroad, can have near-zero marginal resilience value without matching packaging capacity.

## 3 Model

### 3.1 Production and allocation

Time is discrete in weeks, t = 1..52, and we simulate independent years. There are three inputs, i ∈ {logic, cowos, hbm}. Input i is produced at sites j with baseline capacity shares κ_ij (Σ_j κ_ij = 1) and availability a_j(t) ∈ [0,1]. Usable capacity is K_i(t) = Σ_j κ_ij a_j(t) π_j(t). The term π_j(t) is 1 for sites that run TSMC-compatible processes. For non-TSMC logic sites (Samsung, Intel), π_j(t) is 0 except after a porting lag during a Taiwan cut-off (Section 3.3).

AI demand is normalised to 1 accelerator-unit per week. It requires a fraction s_i of input i's total baseline capacity: s_logic = 0.12, s_cowos = s_hbm = 0.90 [1]. We study two allocation regimes for the capacity that remains:

- **Priority.** AI outbids all other users, so its input coverage is r_i(t) = min(K_i(t), 1)/s_i. This is the upper bound on AI's ability to take slack from smartphones, PCs and other users.
- **Proportional.** Every user is rationed pro rata, so r_i(t) = (K_i(t)/K_i^0)(1+h), with a small surge headroom h = 0.05.

Potential output is Leontief: y(t) = min_i r_i(t). An inventory of finished or in-transit accelerators, up to B weeks of demand (default B = 4), is drawn down when y < 1 and refilled from any output above 1. Deliveries are d(t) = min(1, y(t) + inv(t)). The annual loss is ℓ = 1 − (1/52) Σ_t d(t).

### 3.2 Sites and capacity shares

| Input | Site group | Region | Capacity share | Basis |
|---|---|---|---|---|
| logic | TSMC Taiwan (6 clusters) | TW | 0.90 | [S] 92% of sub-10 nm in Taiwan [4]; TSMC "vast majority" of 3/5 nm [2]; split [A] |
| logic | TSMC Arizona | US | 0.03 | [A] |
| logic | non-TSMC advanced (2) | KR/other | 0.07 | [A]; usable only after porting lag |
| cowos | TSMC + OSAT Taiwan (6) | TW | 0.95 | [S] TSMC and partner capacity in Taiwan [15, 17]; 5% outside is [A] |
| cowos | outside Taiwan | other | 0.05 | [A] |
| hbm | SK hynix (2) | KR | 0.523 | [S] [14] |
| hbm | Samsung | KR | 0.287 | [S] [14] |
| hbm | Micron | TW | 0.19 | [S] share [14]; largest HBM site in Taichung [16]; all-TW gating is [A] |

### 3.3 Hazards

Each hazard removes a fraction of a site's capacity for a lognormal duration (median m, log-sd σ), with a rectangular profile.

| Hazard | Scope | Rate | Loss fraction | Duration median / σ (weeks) | Anchors |
|---|---|---|---|---|---|
| site incident | one site | 0.05 /site/yr [A] | U(0.5, 1) | 10 / 0.7 [A] | Renesas Naka fire 2021: ~97 days to 100% capacity [20]; Thai floods 2011: Toyota 42 d, Honda 174 d [19]; HDD output ~80% of pre-flood by Feb 2012 [21] |
| Taiwan earthquake | all TW sites | 0.075/yr [A from S] | U(0.3, 0.9) | 1 / 0.6 | 1999 Chi-Chi: 6–11 day outage, mainly power [22, 23]; 2024 Hualien: >70% tool recovery in 10 h, >80% next day [24]. Two events in ~27 yr |
| Korea regional | all KR sites | 0.01/yr [A] | U(0.3, 1) | 4 / 1.0 [A] | none; swept |
| HBM yield/qualification | one HBM supplier | 0.15 /supplier/yr [A] | U(0.2, 0.6) | 13 / 0.5 [A] | Samsung 12-Hi HBM3E qualification difficulty; share 41% → 28.7% [14] |
| Taiwan cut-off | all TW sites | p_TW ∈ {0, 0.5, 1, 2, 5}%/yr **[P]** | 1 | 26 / 0.8 [A] | scenario only; see [18] for blockade-scale macro estimates |

During a Taiwan cut-off, non-TSMC advanced logic becomes usable for AI designs only after a porting (redesign and requalification) lag of 52 weeks [A]. Within the annual horizon this means it is effectively unavailable. Section 4 analyses the post-porting case analytically.

### 3.4 Experiments

E1: risk metrics versus p_TW under both regimes. E2: each hazard class alone and leave-one-out. E3: counterfactual "hardening", in which one or more inputs never lose capacity, plus Shapley attribution. E4: analytic regional cut-off output. E5: inventory sweep with the closed form. E6: one-at-a-time sensitivity over ±ranges of 12 parameters. E7: slack tolerance curves. Each configuration uses 100,000 simulated years, a fixed seed, and common random numbers across configurations. Monte Carlo standard errors of the mean loss are ≤ 0.024 percentage points (`results/E1.csv`). The total runtime is about 3 minutes on 4 CPUs.

## 4 Theory

All three propositions are elementary. Their value is in making the structural drivers explicit and checkable against data.

**Proposition 1 (Slack asymmetry).** Under priority allocation, suppose input i alone loses a fraction λ of capacity and the other inputs are intact. Then AI output is y = min{1, (1−λ)/s_i}. Output is unaffected iff λ ≤ 1 − s_i, and beyond that threshold output falls with slope 1/s_i. Under proportional allocation, y = 1 − λ for every input, before headroom.

*Proof.* Under priority, r_i = (1−λ)/s_i, and the other inputs have r_k = 1/s_k ≥ 1. So y = min(1, r_i). The threshold and slope follow directly. Under proportional allocation, r_i = (1−λ)(1+h), which gives 1−λ with h = 0. ∎

*Numbers.* With s from [1], the tolerances are 88% for logic and 10% for CoWoS and HBM (`results/results.json`, E7). A 20% loss of CoWoS or HBM capacity cuts AI output to 88.9% under priority. A 20% logic loss has no effect (Fig. 6). The proportional regime has no such asymmetry. So **which inputs matter for AI depends on how scarce capacity is allocated**. This testable implication is developed in Section 6.3.

**Proposition 2 (Weakest-link geography).** Let region R be fully cut off. Let o_i be the share of input i's capacity outside R that AI designs can use immediately. Under priority allocation:

(a) y_R = min_i min{1, o_i/s_i};
(b) keeping output fraction y requires o_i ≥ y·s_i for **every** i;
(c) ∂y_R/∂o_i = 0 for every i not in argmin_i o_i/s_i;
(d) more generally, consider any distribution of capacity shocks and no inventory. Let V(S) be the expected reduction in output loss from making the inputs in S immune to all hazards. Then V is supermodular: V(S∪{i}) − V(S) ≤ V(T∪{i}) − V(T) for S ⊆ T, i ∉ T.

*Proof.* (a)–(c) follow from y = min_i r_i with r_i = o_i/s_i, capped at 1. For (d), fix a realisation and a week, and write c_i = min(1, r_i) ≤ 1. Hardening S sets c_i = 1 for i ∈ S, so output is f(S) = min_{i∉S} c_i, with the minimum over the empty set equal to 1. Write m(X) = min_{i∈X} c_i. Then f(S∪{i}) − f(S) = m(S^c∖{i}) − min(c_i, m(S^c∖{i})) = (m(S^c∖{i}) − c_i)^+. For S ⊆ T, T^c∖{i} ⊆ S^c∖{i}, so m(T^c∖{i}) ≥ m(S^c∖{i}), and the increment is weakly larger at T. Weekly loss is 1 − f(S), so the weekly loss reduction is f(S) − f(∅), which inherits the supermodularity of f. Summing over weeks and taking expectations preserves it. ∎

*Numbers (analytic, E4).* In the 2025–26 baseline (o_logic = 0.03 TSMC-Arizona, o_cowos = 0.05, o_hbm = 0.81), a Taiwan cut-off leaves y = 5.6% under priority, bound by CoWoS (0.05/0.90). Tripling non-Taiwan TSMC logic to 0.09 leaves y unchanged at 5.6%. Moving 30% of CoWoS outside Taiwan *without* moving logic gives y = 25%, now bound by logic (0.03/0.12). Doing both, logic 0.09 and CoWoS 0.30, gives 33%. For y = 50% the requirement (b) is o_logic ≥ 0.06, o_cowos ≥ 0.45 and o_hbm ≥ 0.45. HBM already meets it (0.81 outside Taiwan). At the 2026 industry CoWoS-class capacity of 170–200k wafers/month reported by TrendForce [15], o_cowos ≥ 0.45 means about 76.5–90k wafers/month outside Taiwan; y = 80% means 122–144k. The symmetric case is a full Korea cut-off: logic and CoWoS are unaffected, HBM outside Korea is 0.19, and y = 0.19/0.90 = 21%. **Korea is a second, HBM-mediated near-single point of failure**, almost as severe as Taiwan for AI output, but through a different input. These are mechanical consequences of the stated shares, not forecasts. In particular, they assume that demand from other users can be fully displaced (priority) and that the outside capacity is qualified for current designs.

**Proposition 3 (Inventory cannot cover long outages).** Consider a full outage of random duration D with E[D] < ∞, an initial buffer of b weeks, and no refill during the outage. The lost output is (D−b)^+, and the fraction of expected loss the buffer removes is 1 − E[(D−b)^+]/E[D] ≤ b/E[D]. For D ~ LogNormal(μ, σ²), E[(D−b)^+] = E[D]Φ(d₁) − bΦ(d₂), with d₁ = (μ+σ²−ln b)/σ and d₂ = d₁ − σ.

*Proof.* (D−b)^+ ≥ D − b pointwise, so E[(D−b)^+] ≥ E[D] − b, which gives the bound. The lognormal expression is the standard partial-expectation (Black–Scholes call) formula. ∎

*Numbers (E5, closed form).* A 4-week buffer removes 99.1% of the expected loss from a quake-type outage (median 1 wk), 30.6% from a site incident (median 10 wk), and 11.1% from a cut-off (median 26 wk, E[D] = 35.8 wk; bound 4/35.8 = 11.2%). This is Tomlin's dichotomy [12] made quantitative for AI chips. Stockpiling accelerators is cheap insurance against earthquakes and site incidents and close to useless against a sustained cut-off.

## 5 Results

All numbers in this section are produced by `code/run_all.py` and `code/shapley.py`. Raw outputs are in `results/E1.csv`–`E6.csv`, `results/results.json`, `results/shapley.json` and the logs.

### 5.1 Headline risk (E1, Fig. 1)

| Regime | p_TW (/yr) | E[annual loss] | VaR₉₉ | CVaR₉₉ | CVaR₉₉.₉ | P(loss > 10%) | share of E[loss] from cut-off years |
|---|---|---|---|---|---|---|---|
| priority | 0 | 0.074% | 2.6% | 5.9% | 15.2% | 0.10% | – |
| priority | 0.5% | 0.208% | 4.4% | 17.7% | 60.3% | 0.49% | 61% |
| priority | 1% | 0.325% | 7.2% | 27.2% | 70.1% | 0.79% | 76% |
| priority | 2% | 0.573% | 20.6% | 42.3% | 76.7% | 1.49% | 87% |
| priority | 5% | 1.352% | 43.4% | 59.9% | 82.5% | 3.69% | 95% |
| proportional | 0 | 0.188% | 5.6% | 9.0% | 17.7% | 0.26% | – |
| proportional | 1% | 0.454% | 10.0% | 29.2% | 72.0% | 1.00% | 57% |
| proportional | 5% | 1.532% | 45.1% | 61.8% | 84.6% | 3.97% | 88% |

Three observations. First, **routine risk is small**. Without a Taiwan cut-off, expected annual loss is 0.07–0.19% of accelerator output, and a loss above 10% happens about once per 400–1,000 years. Second, the loss distribution is extremely skewed. Even at p_TW = 0.5%/yr, the cut-off scenario supplies most of the expected loss and nearly all of the 99.9% CVaR. Third, years with a cut-off lose an average of 26–28% of annual output. This figure is truncated at the calendar year (Section 7). During the cut-off itself, output falls to the 5.6% floor of Proposition 2, or 3% under proportional allocation, less whatever inventory covers.

### 5.2 Which hazards matter (E2, Fig. 2)

With p_TW = 1%/yr, each hazard simulated alone gives the following expected annual loss and CVaR₉₉:

| Hazard alone | priority E[loss] | priority CVaR₉₉ | proportional E[loss] | proportional CVaR₉₉ |
|---|---|---|---|---|
| site incidents | 0.012% | 1.2% | 0.046% | 4.0% |
| Taiwan earthquake | 0.000% | 0.0% | 0.000% | 0.01% |
| Korea regional | 0.015% | 1.5% | 0.016% | 1.6% |
| HBM yield/qualification | 0.029% | 2.9% | 0.065% | 5.4% |
| Taiwan cut-off (1%/yr) | 0.246% | 24.6% | 0.254% | 25.4% |

Earthquakes on the calibrated scale (outages of days) are fully absorbed by a 4-week buffer. This agrees with the observed absence of visible AI-supply consequences after the 2024 Hualien earthquake [24]. That agreement is a weak consistency check, not a validation. Among routine hazards, **HBM yield/qualification shocks are the largest contributor** under both regimes. Their effect is larger under proportional allocation, which cannot draw on CoWoS/HBM headroom. Hazards interact super-additively. Removing site incidents from the full model lowers expected loss by 0.030 pp under priority, more than their 0.012 pp standalone effect. The reason is that concurrent shocks deplete the shared buffer and compound through the min operator.

### 5.3 Where hardening pays: complementarity (E3)

We make sets of inputs immune to all hazards, including the cut-off. Reduction relative to the full model with p_TW = 1%:

| Hardened | priority ΔE[loss] | priority ΔCVaR₉₉ | proportional ΔE[loss] | proportional ΔCVaR₉₉ |
|---|---|---|---|---|
| logic | −0.0% | −0.0% | −5.9% | −2.5% |
| cowos | −22.6% | −24.1% | −4.3% | −0.6% |
| hbm | −25.3% | −10.6% | −42.2% | −11.4% |
| logic + cowos | −74.9% | −76.2% | −62.5% | −68.3% |
| cowos + hbm | −47.2% | −36.7% | −43.6% | −12.5% |
| all three | −100% | −100% | −100% | −100% |

Hardening logic alone is worth essentially nothing under priority allocation, because of logic's 88% slack and because CoWoS in the same region binds during the cut-off. Under proportional allocation it is worth very little. Hardening logic and CoWoS together removes three quarters of expected loss and of CVaR₉₉. Shapley attribution (`results/shapley.json`) gives, for expected loss under priority, logic 26%, CoWoS 49% and HBM 25%. The logic–CoWoS interaction term, V({L,P}) − V({L}) − V({P}), is +52 pp of baseline expected loss in both regimes, and +52 pp (priority) or +65 pp (proportional) for CVaR₉₉. This is the empirical counterpart of Proposition 2(d). **Standalone ("one input at a time") resilience assessments systematically undervalue joint investments.** The Shapley value assigns logic a quarter of the credit even though its standalone value is zero.

### 5.4 Weakest-link map (E4, Fig. 3)

Fig. 3 plots y_R (Prop. 2a) over the share of TSMC-process logic outside Taiwan and the share of CoWoS-class capacity outside Taiwan. The iso-resilience contours are L-shaped, as Leontief structure implies. Movement along either axis alone soon hits a wall. The 2025–26 point (3%, 5%) sits in the corner, at 5.6% output. The logic axis saturates quickly: 6% of logic outside Taiwan already supports 50% output, because s_logic is small. The packaging axis has to travel much further, to 45%.

### 5.5 Inventory (E5, Fig. 4)

Under proportional allocation, expected loss from routine hazards falls steeply with buffer size: 2.45% with no buffer, 0.19% at 4 weeks, 0.02% at 8 weeks. Expected loss from the cut-off barely moves: 0.32% with no buffer, 0.25% at 4 weeks, 0.19% at 8 weeks. This matches Proposition 3. The sensitivity results agree. Buffer size is the most influential parameter for *expected* loss (priority 0.82% at 1 week versus 0.20% at 8 weeks), but its effect on CVaR₉₉ is bounded by b/52 per year.

### 5.6 Sensitivity (E6, Fig. 5)

One-at-a-time ranges at p_TW = 1% (priority), with expected loss and then CVaR₉₉:
- buffer 1→8 wk: 0.82%→0.20%; 32.9%→19.5%.
- cut-off probability 0.5→2%: 0.21%→0.57%; 17.7%→42.3%.
- cut-off median 13→52 wk: 0.23%→0.40%; 18.1%→34.9%.
- HBM yield-shock rate 0.075→0.30: 0.30%→0.43%; 26.7%→28.9%.
- site-incident rate and duration: 0.30–0.41%; CVaR₉₉ 26.7–28.9%.
- AI shares of CoWoS/HBM over 0.80–0.95: 0.27–0.41%. These have no effect under proportional allocation, as Prop. 1 predicts.
- non-Taiwan CoWoS 0→15%: 0.34%→0.29%; CVaR₉₉ 28.9%→23.7%.

The ranking is stable. Tail risk is governed almost entirely by the Taiwan cut-off parameters and the buffer. Mean risk is governed by the buffer, the cut-off parameters and the HBM and site-incident parameters. No routine-hazard assumption within these ranges moves CVaR₉₉ by more than about 2.6 pp.

### Figures

- Fig. 1 `results/fig1_exceedance.png`: exceedance curves of annual loss for p_TW ∈ {0, 1, 5}%/yr, under both regimes.
- Fig. 2 `results/fig2_hazard_decomposition.png`: expected loss and CVaR₉₉ for each hazard simulated alone.
- Fig. 3 `results/fig3_weakest_link.png`: AI output during a Taiwan cut-off versus outside-Taiwan logic and CoWoS shares (Prop. 2).
- Fig. 4 `results/fig4_buffers.png`: inventory buffer effectiveness, simulated and closed form (Prop. 3).
- Fig. 5 `results/fig5_tornado.png`: one-at-a-time sensitivity of expected loss.
- Fig. 6 `results/fig6_slack.png`: slack asymmetry (Prop. 1).

## 6 Discussion

### 6.1 So how large is the risk?

Within this model, the answer has two parts. **Routine disruption risk to AI-accelerator output is small and largely self-insured.** The asymmetric slack in logic, the CoWoS/HBM headroom available under priority allocation, and a few weeks of inventory absorb fires, earthquakes of the 1999/2024 scale and most yield shocks. Expected losses are on the order of 0.1–0.2% of annual output. **The risk is dominated by a single correlated geographic scenario.** At any cut-off probability of 0.5%/yr or more it supplies most of the expected loss and nearly all of the tail. In that scenario AI output falls to roughly 3–6% of demand for its duration, and today's front-end diversification does essentially nothing to prevent that. A second, less discussed concentration exists in Korea, through HBM. A Korean cut-off would leave about 21% of output.

### 6.2 Policy implications

1. **Count resilience in completed accelerators, not in wafers.** Programmes that add advanced logic capacity outside Taiwan (such as TSMC Arizona) buy close to zero cut-off resilience for AI until CoWoS-class packaging, and HBM stacking for Micron's share, are co-located. Reports that Arizona dies are shipped back to Taiwan for packaging [17] show this concretely. The Shapley interaction of +52 pp quantifies the cost of sequencing these investments separately.
2. **The cheap part of AI resilience is logic; the expensive part is packaging.** Because s_logic ≈ 0.12, a modest offshore logic footprint, about 6% of advanced logic on TSMC-compatible processes, covers half of AI demand under priority. Packaging needs about 45% of capacity, or 76–90k wafers/month at 2026 scale.
3. **Stockpiles are for earthquakes, not blockades.** A buffer of 4–8 weeks removes most routine loss (Prop. 3, Fig. 4) but at most b/E[D] of a sustained cut-off.
4. **Allocation rules are a policy variable.** Priority allocation is what makes AI robust to logic shocks. Proportional rationing, for example under export or "fair share" regimes, removes that robustness (Table E3: proportional regime). Governments that would ration chips in a crisis should note that the rationing rule changes which inputs are critical.
5. **HBM is the routine-risk hotspot and the Korea-risk channel.** Among non-cut-off hazards, HBM yield and qualification shocks contribute the most. Three vendors, concentrated in one country, is a structurally fragile arrangement.

### 6.3 Falsifiable predictions and test protocol

The model makes predictions that public data can falsify.

- **H1 (slack asymmetry).** In a supply event that removes 10–50% of *advanced logic* capacity for weeks, AI-accelerator shipments and lead times will not fall measurably. Prices of consumer devices on the same nodes will rise or their volumes will fall. A comparable percentage loss of CoWoS or HBM capacity will reduce accelerator shipments roughly one-for-one beyond a 10% threshold. *If logic shocks visibly cut AI accelerator output, priority allocation is rejected.*
- **H2 (weakest link).** Announcements that expand non-Taiwan logic capacity, with no packaging counterpart, will not change market-implied measures of AI-supply tail risk, such as option-implied volatility of AI-accelerator vendors around Taiwan-Strait tension events. Announcements of non-Taiwan CoWoS-class capacity will.
- **H3 (buffer).** Short outages (under 2 weeks, as in 2024) will produce no quarterly shipment shortfall. Outages longer than about 3 months will produce shortfalls roughly proportional to (D − b).

*Protocol.* (i) Build an event list from company disclosures (fab incidents, earthquakes, HBM qualification news) and TrendForce capacity notes. (ii) Measure AI-accelerator output using vendors' quarterly data-centre revenue divided by ASP estimates, plus Epoch's chip-sales data [2]. Measure consumer-device output on the same nodes using smartphone/PC shipment trackers. (iii) Estimate local projections of each outcome on event-implied capacity loss, separately by input, and test for the threshold at λ ≈ 1 − s_i (H1) and the slope 1/s_i beyond it. (iv) Pre-register the threshold values from Proposition 1 before looking at the data.

### 6.4 Relation to macro estimates

Bloomberg Economics' scenarios put the global cost of a Taiwan blockade at about 5% of world GDP, and a war at about 10% [18]. Our model is complementary and much narrower. It says nothing about GDP. It shows that, for AI compute specifically, the blockade scenario is close to a total stop, with output above the floor only to the extent of pre-positioned inventory and qualified outside capacity.

## 7 Limitations

1. **Most hazard parameters are assumptions.** Rates and durations of site incidents, Korean regional shocks and HBM yield shocks are calibrated loosely to a handful of anchor events (Section 3.3), not estimated statistically. The Taiwan cut-off probability and duration are scenario parameters. We report sweeps, but readers should treat absolute numbers as conditional on the stated assumptions. The structural conclusions (Props. 1–3, the complementarity ranking) hold for any parameter values.
2. **Capacity shares are approximate and changing fast.** HBM shares are 2025 *projections* [14]. CoWoS capacity estimates come from press reports and differ by about 20% across sources [15]. The 3%/5% outside-Taiwan shares for TSMC-process logic and CoWoS are our assumptions. AI's input shares come from a single source [1] and will change as AI moves to more advanced nodes.
3. **Annual truncation.** Years are simulated independently, and a cut-off is truncated at year end. This understates the loss from multi-year cut-offs (median 52 weeks gives only a 0.40% expected loss in E6). A multi-year horizon would raise tail metrics.
4. **Missing inputs.** We omit ABF substrates, EUV tools and spares (ASML), specialty chemicals and gases, interposers, power and water. We also omit tool damage that lengthens recovery. Each could add an independent weak link and would only *increase* risk under the Leontief structure.
5. **Idealised allocation and porting.** The two regimes bound reality. Long-term contracts, export controls and politics lie between them. The 52-week porting lag is a single assumed number. We also ignore the possibility that the outside capacity itself is not qualified for current designs.
6. **No demand side, prices or substitution.** We measure physical output shortfalls only. We model no price responses, no substitution toward older accelerators or non-CoWoS designs, and no endogenous capacity investment after a shock. Baqaee–Farhi-style non-linearities [11] suggest the welfare cost could be much larger than the output shortfall, but we do not compute it.
7. **No empirical validation.** The model agrees with the 2024 earthquake non-event, but that is a single weak check. Section 6.3 is a proposal, not a result.

## 8 Conclusion

The question "how risky is the advanced-chip supply chain for AI?" cannot be answered by listing input concentrations. An AI accelerator is a Leontief bundle of logic, packaging and HBM. AI takes a small share of logic but nearly all of packaging and HBM. Inputs are co-located by region. Under these features, routine disruption risk is small and largely absorbed. Tail risk is concentrated in one correlated regional scenario, and it runs through packaging (and, for Korea, HBM) rather than front-end logic. The value of hardening inputs is supermodular: diversifying one input in isolation can be worth zero, while co-locating logic and packaging outside Taiwan removes about three quarters of modelled risk. We provide the model, proofs, code and a falsification protocol so that these claims can be revised as better capacity and hazard data appear.

## Data and code availability

All code is in `code/` (`python3 run_all.py; python3 shapley.py`, about 3 minutes on 4 CPUs). All outputs are in `results/`. Source-derived parameters are tagged in `code/params.py`.

## References

[1] Somala, V. (2026, March 12). *Advanced packaging and HBM, not logic dies, were the bottlenecks on AI chip production in 2025.* Epoch AI Data Insight. https://epoch.ai/data-insights/ai-chip-supply-chain-constraints

[2] Epoch AI. (2026). *Data on AI chip components.* https://epoch.ai/data/ai-chip-components

[3] Sevilla, J., et al. (2024). *Can AI scaling continue through 2030?* Epoch AI. https://epoch.ai/blog/can-ai-scaling-continue-through-2030

[4] Varas, A., Varadarajan, R., et al. (2021). *Strengthening the Global Semiconductor Supply Chain in an Uncertain Era.* Boston Consulting Group and Semiconductor Industry Association. https://www.semiconductors.org/strengthening-the-global-semiconductor-supply-chain-in-an-uncertain-era/

[5] Khan, S. M., & Mann, A. (2020). *AI Chips: What They Are and Why They Matter.* Center for Security and Emerging Technology, Georgetown University. https://cset.georgetown.edu/publication/ai-chips-what-they-are-and-why-they-matter/

[6] Sastry, G., Heim, L., Belfield, H., Anderljung, M., Brundage, M., Hazell, J., O'Keefe, C., et al. (2024). Computing Power and the Governance of Artificial Intelligence. arXiv:2402.08797.

[7] Miller, C. (2022). *Chip War: The Fight for the World's Most Critical Technology.* Scribner.

[8] Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The network origins of aggregate fluctuations. *Econometrica*, 80(5), 1977–2016. doi:10.3982/ECTA9623

[9] Carvalho, V. M., Nirei, M., Saito, Y. U., & Tahbaz-Salehi, A. (2021). Supply chain disruptions: Evidence from the Great East Japan Earthquake. *Quarterly Journal of Economics*, 136(2), 1255–1321. doi:10.1093/qje/qjaa044

[10] Boehm, C. E., Flaaen, A., & Pandalai-Nayar, N. (2019). Input linkages and the transmission of shocks: Firm-level evidence from the 2011 Tōhoku earthquake. *Review of Economics and Statistics*, 101(1), 60–75.

[11] Baqaee, D. R., & Farhi, E. (2019). The macroeconomic impact of microeconomic shocks: Beyond Hulten's theorem. *Econometrica*, 87(4), 1155–1203. doi:10.3982/ECTA15202

[12] Tomlin, B. (2006). On the value of mitigation and contingency strategies for managing supply chain disruption risks. *Management Science*, 52(5), 639–657. doi:10.1287/mnsc.1060.0515

[13] Snyder, L. V., Atan, Z., Peng, P., Rong, Y., Schmitt, A. J., & Sinsoysal, B. (2016). OR/MS models for supply chain disruptions: A review. *IIE Transactions*, 48(2), 89–109. doi:10.1080/0740817X.2015.1067735

[14] TrendForce. (2025, August 18). NVIDIA reportedly drives 27% of SK hynix revenue in 1H25 (HBM share projections: SK hynix 52.3%, Samsung 28.7%, Micron 19% for 2025). TrendForce News. https://www.trendforce.com/news/2025/08/18/news-nvidia-reportedly-drives-27-of-sk-hynix-revenue-in-1h25-cementing-ai-chip-partnership

[15] TrendForce. (2026, June 15). TSMC CoWoS supply-demand gap reportedly seen narrowing from 20% to 10% by end-2026 as capacity expands. TrendForce News. https://www.trendforce.com/news/2026/06/15/news-tsmc-cowos-supply-demand-gap-reportedly-seen-narrowing-from-20-to-10-by-end-2026-as-capacity-expands

[16] Tom's Hardware. (2024). Micron to expand production of HBM3E memory across the world to increase HBM market share: Report. https://www.tomshardware.com/pc-components/gpus/micron-to-expand-production-of-hbm3e-memory-across-the-world-to-increase-hbm-market-share-report

[17] Tom's Hardware. (2024, October). Amkor and TSMC team up for advanced packaging in the U.S. — CoWoS and InFO to make AI and HPC CPUs. https://www.tomshardware.com/tech-industry/amkor-and-tsmc-team-up-for-advanced-packaging-in-the-u-s-cowos-and-info-to-make-ai-and-hpc-cpus ; see also TrendForce (2025, June 10), TSMC speeds up Arizona expansion, yet U.S. packaging plant sites reportedly remain up in the air. https://www.trendforce.com/news/2025/06/10/news-tsmc-speeds-up-arizona-expansion-yet-u-s-packaging-plant-sites-reportedly-remain-up-in-the-air/

[18] Bloomberg Economics. (2024, January). Xi, Biden and the $10 trillion cost of war over Taiwan. Bloomberg (reprinted, e.g., Insurance Journal, 11 Jan 2024: https://www.insurancejournal.com/news/international/2024/01/11/755295.htm); updated in Bloomberg (2026, February 10), The $10 trillion fight: Modeling a US-China war over Taiwan. https://www.bloomberg.com/news/articles/2026-02-10/the-10-trillion-fight-modeling-a-us-china-war-over-taiwan

[19] Simchi-Levi, D., Schmidt, W., & Wei, Y. (2014). From superstorms to factory fires: Managing unpredictable supply-chain disruptions. *Harvard Business Review*, 92(1–2), 96–101.

[20] Renesas Electronics. (2021). Update 10: Notice regarding the semiconductor manufacturing factory (Naka Factory) fire — production capacity. https://www.renesas.com/en/about/press-room/update-10-notice-regarding-semiconductor-manufacturing-factory-naka-factory-fire-production-capacity

[21] The Register. (2012, February 24). [Report on hard-disk-drive output recovery after the 2011 Thai floods] (descriptive title). https://www.theregister.co.uk/2012/02/24/hdd_thailand_q1/

[22] United Microelectronics Corp. (1999, September 22). [Press release on the impact of the 21 September 1999 earthquake] (descriptive title). https://www.umc.com/en/News/press_release/Content/corporate/19990922

[23] TSMC. (1999, September). TSMC reports power still out at Hsinchu Science-Based Park. https://pr.tsmc.com/english/news/2166

[24] Fortune. (2024, April 3). [Report on TSMC operations after the 3 April 2024 Hualien earthquake] (descriptive title; reports TSMC's statement that tool recovery exceeded 70% within ten hours). https://fortune.com/2024/04/03/tsmc-taiwan-earthquake-nvidia-apple-chip-semiconductor-manufacturing/
