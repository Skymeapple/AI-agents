# Idle Accelerators Are Expensive: Pricing Compute Flexibility in Clean-Power Supply for Islanded AI Data Centers

**Target venue:** *Joule* (Analysis article)

**Code and data:** `code/` and `results/` in this repository (all numbers in this paper are produced by those scripts or taken from the cited sources; each number is labelled [model] or given a citation).

---

## Abstract

Electricity demand from data centers is growing faster than grids can connect new supply, and AI developers are turning to nuclear contracts, renewables-plus-storage, and on-site ("islanded") power. A common proposal for closing the gap is load flexibility: if AI campuses curtail a small fraction of their consumption in scarce hours, the system needs less firm capacity. Existing studies treat the flexible share or the shifting window as an exogenous parameter. We argue that for AI campuses the right primitive is instead a price, the *value of lost compute* (VoLC): the capital cost of accelerators that sit idle while power is curtailed. Public hardware price and power estimates imply a VoLC of roughly \$400–970 per MWh of facility energy (central \$600/MWh) [model], counting accelerator capital only. That is an order of magnitude above typical energy costs. We build an hourly linear capacity-expansion model of an islanded AI campus in which curtailment is chosen endogenously and priced at VoLC. The supply options are solar, wind, Li-ion storage, gas turbines and a firm clean resource standing in for small modular reactors (SMRs). We prove three properties of the model: a scarcity-price ceiling, the concavity identity that the flexibility dividend equals the integral of the optimal curtailment curve, and a resource-cost bound on how many hours per year curtailment can occur. We then solve the model with NREL resource data (NSRDB and WIND Toolkit, weather years 2010–2013) and NREL ATB 2024 costs for three sites: Abilene TX, Loudoun County VA and Phoenix AZ. At VoLC in the empirical range, optimal curtailment is a small fraction of a percent of annual energy, and it lowers total supply cost by about 1% or less. Flexibility therefore does not substitute for firm clean power or for long-duration storage at high carbon-free shares. Only at VoLC of about \$250/MWh, which corresponds to heavily depreciated hardware, does curtailment become structural (10–18% of energy), and then it lowers costs substantially. Because the target is a share of *served* energy, curtailment also tightens the carbon-free constraint, which raises the effective cost of flexibility above VoLC. We list falsifiable predictions and a protocol for testing them against interconnection-agreement and fleet-telemetry data.

## 中文摘要

数据中心电力需求的增长速度超过了电网接入新电源的速度，AI 企业正转向核电合约、"可再生能源+储能"以及园区自建（离网）电源。常见的建议是利用负荷灵活性：若 AI 园区在电力紧张时段削减一小部分用电，系统所需的稳定（firm）电源容量即可减少。现有研究通常把可灵活调节的负荷比例或可平移时间窗作为外生参数。本文认为，对 AI 园区而言，更合适的基本量是一个价格，即"算力损失价值"（VoLC）：削减用电期间闲置加速器的资本成本。根据公开的硬件价格与功率估计，仅计入加速器资本时，VoLC 约为每 MWh 设施用电 400–970 美元（中心值约 600 美元），比典型电能成本高一个数量级。我们构建了离网 AI 园区的逐小时线性容量扩展模型，削减量由模型内生决定并按 VoLC 计价，可选电源包括光伏、风电、锂电池储能、燃气轮机，以及代表小型模块化反应堆（SMR）的稳定清洁电源。我们证明了模型的三个性质：稀缺电价上限；"灵活性红利等于最优削减曲线的积分"这一凹性恒等式；以及由资源成本给出的年削减小时数上界。随后，我们使用 NREL 资源数据（NSRDB 与 WIND Toolkit，2010–2013 气象年）和 NREL ATB 2024 成本数据，对德州阿比林、弗吉尼亚州劳登县和亚利桑那州凤凰城三个站点求解模型。在经验 VoLC 范围内，最优削减量不到年用电量的百分之一，总供电成本仅降低约 1% 或更少；因此在高无碳比例下，灵活性不能替代稳定清洁电源或长时储能。只有当 VoLC 约为 250 美元/MWh（对应大幅折旧的硬件）时，削减才会成为结构性手段（占用电量 10–18%）并显著降低成本。由于目标是"已服务电量"中的无碳比例，削减还会收紧无碳约束，使灵活性的有效成本高于 VoLC。文末给出可证伪的预测，以及基于并网协议和机群遥测数据的检验方案。

---

## 1 Introduction

US data centers used about 176 TWh in 2023, roughly 4.4% of national electricity consumption. Lawrence Berkeley National Laboratory projects 325–580 TWh by 2028, or 6.7–12% of the national total [Shehabi et al. 2024]. Globally, the IEA estimates data-center consumption at about 415 TWh in 2024, rising to about 945 TWh by 2030 in its Base Case [IEA 2025]. EPRI's 2026 update projects data centers at 9–17% of US electricity by 2030, about 60% above its 2024 estimates [EPRI 2024; EPRI 2026]. The binding constraint is often speed rather than annual energy. At the end of 2023, about 2.6 TW of generation and storage was waiting in US interconnection queues [Rand et al. 2024]. Firm clean supply is arriving on long lead times. Microsoft's 20-year agreement to restart the 835 MW Three Mile Island Unit 1 (now the Crane Clean Energy Center) targets around 2028 [Constellation 2024]. Google's agreement with Kairos Power for up to 500 MW of SMRs targets a first unit by 2030 [Google 2024]. This mismatch has motivated two responses that are now central to the debate.

The first response is *on-site or islanded supply*. An off-grid campus combines solar, storage and gas backup and avoids the interconnection queue entirely. One industry white paper argues that such microgrids could plausibly supply a large share of AI-training demand [Baranko et al. 2024]. The second response is *load flexibility*. Norris et al. [2025] estimate that the 22 largest US balancing areas could absorb 76 GW of new load if that load accepts curtailment averaging 0.25% of its maximum annual energy, rising to 98 GW at 0.5% and 126 GW at 1%. Field trials report that software can cut the power of an AI cluster by 25% for three hours [Colangelo et al. 2025]. Energy-system models show that spatial and temporal load shifting lowers the cost of 24/7 carbon-free procurement [Riepin, Brown & Zavala 2025] and of system operation [Knittel, Senga & Wang 2025].

**The gap.** These studies treat flexibility as a *quantity*, such as a share of load that can be shifted, a shifting window, or an allowed curtailment rate, and they impose it exogenously. For AI campuses, though, flexibility has a clear and very large *price*. When a training or inference cluster is curtailed, the accelerators keep depreciating while producing nothing. Hardware amortization is the main cost of frontier training runs; energy is a small share [Cottier et al. 2024]. If the opportunity cost of an idle accelerator-hour is several times the cost of the electricity it would have used, the economically chosen amount of flexibility may be much smaller than the amounts assumed in the literature. Whether flexibility can stand in for firm clean power is then an empirical question about this price, and it has not been posed in a form that a model can answer.

**This paper.** We ask a narrow question: *for an islanded AI campus that must reach a given carbon-free share, how much curtailment is cost-optimal when curtailment is priced at the capital cost of idle compute, and how does that change the roles of renewables, storage, gas and firm clean (nuclear) supply?* We study islanded campuses for three reasons. They are the deployment mode most directly motivated by interconnection delays. They are a self-contained system, so the trade-offs are not hidden inside the rest of a grid. And they bound what a single developer can achieve without the grid.

**Contributions.**

1. **Formulation.** We define an hourly capacity-expansion linear program (LP) for an islanded campus in which compute curtailment is a decision variable priced at the value of lost compute *v*, and in which the carbon-free target applies to *served* energy (§3).
2. **Theory.** We prove three properties of this model (§4). (P1) Hourly scarcity prices are capped at an *effective* VoLC, v + (1−φ)μ, which exceeds v whenever the carbon-free constraint binds; under a share-based target, curtailment is therefore more expensive than its face value. (P2) Optimal total cost is concave in *v*, and the "flexibility dividend" equals the integral of the optimal curtailment curve; this gives two-sided bounds that we verify on every solved instance. (P3) For any available resource with annualized fixed cost F and availability a, the number of hours per year with curtailment of any depth is at most F/(a(v − c)).
3. **A sourced estimate of VoLC.** From public price and power estimates for a GB200 NVL72 rack, we estimate \$403–968 per MWh of facility energy (central \$597/MWh) [model]. This counts accelerator capital only and is therefore a lower bound (§3.4).
4. **Empirics on real weather data.** We solve 210 main-sweep instances plus 100 robustness instances. They cover three US data-center regions, NREL NSRDB irradiance and WIND Toolkit wind for 2010–2013, and NREL ATB 2024 cost projections for 2030. We find that at empirical VoLC, curtailment is a small fraction of a percent of energy and lowers total supply cost by about 1% or less. It does not replace firm clean power or storage. Flexibility becomes structural only at VoLC around \$250/MWh (§5).
5. **Robustness and falsifiable predictions.** We validate against full hourly resolution, test designs on out-of-sample weather years, and vary the main cost inputs. We then state predictions, with a test protocol, about where flexibility commitments by AI campuses should and should not appear (§6).

We do not claim that grid-connected flexibility is unimportant. A 0.25% curtailment commitment that unlocks a grid connection years early can be worth far more than its energy cost. Our result concerns flexibility as a substitute for *supply capacity* inside a clean-energy portfolio, and we return to this distinction in §6.

## 2 Related Work

**Data-center demand.** Bottom-up US estimates [Shehabi et al. 2024], global estimates [IEA 2025] and utility-sector scenarios [EPRI 2024; EPRI 2026] agree on rapid growth with wide uncertainty bands. Our model is per MW of load and does not depend on which projection is correct.

**Firm low-carbon resources and storage.** Capacity-expansion studies find that firm low-carbon resources (nuclear, geothermal, gas with CCS) lower the cost of deep decarbonization by 10–62%, and that batteries and demand flexibility did not substitute for them in the scenarios studied [Sepulveda et al. 2018]. Ziegler et al. [2019] show that a wind–solar mix can supply cost-competitive baseload in Texas and Arizona only if storage energy capacity costs fall to about \$20/kWh. Our islanded-campus results are consistent with both findings. The new element is that we let demand flexibility compete on price.

**24/7 carbon-free procurement.** Xu et al. [2024] show that hourly matching drives much larger system-level emission reductions than annual matching, and that it raises participants' cost premiums. Riepin, Brown & Zavala [2025] show that spatio-temporal shifting across a network of data centers lowers the cost of 24/7 matching, by about €1.29/MWh per additional percentage point of flexible load. In both lines of work flexibility is exogenous, a fixed share or window that carries no opportunity cost.

**Data-center flexibility.** Google's carbon-intelligent computing system shifts temporally flexible workloads using day-ahead "virtual capacity curves" [Radovanović et al. 2021]. Norris et al. [2025] quantify grid headroom unlocked by curtailable load. Colangelo et al. [2025] demonstrate a 25% three-hour power reduction on a 256-GPU A100 cluster. Knittel, Senga & Wang [2025] find in GenX that flexible data centers lower system costs, but they can raise emissions when the grid has a low renewable share. None of these studies prices curtailment at the capital cost of idle accelerators, and none derives the optimal amount of curtailment from that price.

**Training cost structure.** Cottier et al. [2024] estimate that amortized hardware dominates the cost of frontier training runs (with R&D staff at 29–49% of total amortized cost for the four models studied in depth), and that energy is a small component. This cost structure is the economic basis for our VoLC.

**Off-grid AI campuses.** Baranko et al. [2024] (an industry white paper, not peer-reviewed) argue that off-grid solar microgrids with gas backup could serve AI training. Their analysis, as reported, uses fixed solar shares. We optimize the full portfolio and treat flexibility endogenously.

## 3 Model

### 3.1 Setting

A campus has flat potential electricity demand L (we normalize L = 1 MW; all results scale linearly). It is islanded: all supply is on site. The planner chooses capacities K = (S, W, P_b, E_b, G_ct, G_cc, N). These are solar PV (MW_ac, single-axis tracking), wind (MW), Li-ion power (MW) and energy (MWh), simple-cycle gas (CT), combined-cycle gas (CC), and a firm clean resource N with hourly availability a = 0.93. N stands in for SMRs, though it could equally be enhanced geothermal. For each hour t = 1..T the planner chooses gas dispatch g_t^ct and g_t^cc, storage charge c_t, discharge d_t and state of charge e_t, firm output n_t, and curtailed compute load u_t ∈ [0, L].

### 3.2 Linear program

$$
\min\; \sum_k F_k K_k \;+\; \sum_t \Big[ c^{ct} g_t^{ct} + c^{cc} g_t^{cc} + c^{N} n_t + v\,u_t \Big]
$$

subject to, for all t:

- balance: S·σ_t + W·ω_t + n_t + g_t^{ct} + g_t^{cc} + d_t − c_t ≥ L − u_t  (dual λ_t ≥ 0)
- storage: e_t = e_{t−1} + η_c c_t − d_t/η_d (cyclic), 0 ≤ e_t ≤ E_b, 0 ≤ c_t, d_t ≤ P_b
- limits: g_t^{ct} ≤ G_ct, g_t^{cc} ≤ G_cc, n_t ≤ a N, 0 ≤ u_t ≤ L

and the carbon-free-energy (CFE) constraint on **served** energy

$$
\sum_t (g_t^{ct} + g_t^{cc}) \;\le\; (1-\varphi) \sum_t (L - u_t) \qquad (\text{dual } \mu \ge 0).
$$

Here σ_t and ω_t are hourly solar and wind capacity factors, F_k are annualized fixed costs, c^· are variable costs, φ is the CFE target, and v is the VoLC. Surplus generation is spilled (the balance is an inequality). Storage charging from gas is allowed, and the CFE accounting charges the gas at generation. Round-trip efficiency is 85%.

### 3.3 Data and cost inputs

*Resource data.* Hourly global, direct and diffuse irradiance, air temperature and wind speed come from the NSRDB v3.0.6 (4 km, 30 min, aggregated to hourly) [Sengupta et al. 2018]. 100 m wind speed, temperature and pressure come from the WIND Toolkit CONUS v1.0.0 (2 km, hourly) [Draxl et al. 2015]. Both were read directly from NREL's public AWS buckets at the grid cells nearest Abilene, TX (32.45°N, 99.73°W), Loudoun County, VA (39.04°N, 77.49°W) and Phoenix, AZ (33.45°N, 112.07°W), for 2010–2013 (the overlapping years). Abilene is the site of a large AI campus under construction, and Loudoun is the largest US data-center market. PV output is modelled with pvlib [Anderson et al. 2023]: single-axis tracking with backtracking (GCR 0.35), isotropic transposition, SAPM cell temperature, PVWatts DC model with γ = −0.35%/K, 14% DC losses, DC/AC ratio 1.3 and 96% inverter efficiency. Wind output uses a generic normalized power curve (cut-in 3 m/s, rated 11 m/s, cut-out 25 m/s), a density-corrected hub-height speed and 15% wake, availability and electrical losses. Resulting mean capacity factors for 2012 are 28.1% / 33.1% (Abilene PV / wind), 23.1% / 20.6% (Loudoun) and 30.3% / 10.0% (Phoenix) [model; `results/capacity_factor_summary.json`]. The generic power curve is conservative compared with modern low-specific-power turbines (§7).

*Costs.* All capital and fixed O&M costs are NREL ATB 2024 (v3) Moderate-scenario, Market-case values for cost year 2030, read from the official parquet release [NREL 2024]. They are: tracking PV \$1,193/kW_ac and \$18.0/kW-yr; land-based wind (techno-resource class 4) \$1,408/kW and \$29.3/kW-yr; Li-ion storage \$433/kW plus \$255/kWh (split from the ATB 2-h and 4-h system costs of \$942/kW and \$1,451/kW); gas CT \$1,270/kW with heat rate 9.72 MMBtu/MWh; gas CC (1-on-1 H-frame) \$1,669/kW with heat rate 6.04 MMBtu/MWh; and "Nuclear – Small" \$9,650/kW and \$136/kW-yr. We annualize at a uniform 7% real WACC, with lifetimes of 30 years (PV, wind, gas), 15 years (storage) and 40 years (firm). We assume a gas price of \$3.5/MMBtu and a firm-resource variable cost of \$13.6/MWh, and we exclude tax credits. These choices are assumptions, and we vary them in §5.5. The CO₂ factor for gas is 53.06 kg/MMBtu [EPA 2024]. At 93% availability, the annualized firm cost corresponds to about \$119/MWh [model].

*Solver and resolution.* LPs are solved with HiGHS [Huangfu & Hall 2018] through SciPy. The main sweeps use 2-hour time steps (T = 4,380), with capacity factors averaged within each step. We report the error against full hourly resolution in §5.6.

### 3.4 The value of lost compute

We define VoLC as the capital cost per MWh of facility energy of accelerators idled by curtailment:

$$
v_{\text{cap}} = \frac{P_{\text{rack}}\cdot \mathrm{CRF}(r, \ell)}{p_{\text{rack}} \cdot \mathrm{PUE}\cdot 8760}.
$$

Analyst and press estimates put a GB200 NVL72 rack at about \$3.0M (HSBC, May 2024, as reported by Tom's Hardware); later press reports give \$2.8–3.4M [Tom's Hardware 2024]. Rack power is about 120 kW per NVIDIA's DGX GB200 documentation and up to 132 kW in vendor specifications [NVIDIA 2025; HPE 2025]. We sweep price {2.8, 3.0, 3.4} \$M, power {120, 132} kW, amortization life {4, 5, 6} years, cost of capital {8, 10, 12}% and PUE {1.1, 1.2, 1.3}, which gives 162 combinations. VoLC ranges from \$403 to \$968/MWh with a median of \$615/MWh. The central case (\$3.0M, 126 kW, 5 years, 10%, PUE 1.2) gives \$597/MWh [model; `results/volc_summary.json`]. This is a *lower bound* on the economic cost of curtailment. It omits building and cooling capital, networking, staff, and the margin forgone on inference revenue, and it ignores schedule costs of delaying a training run. Our sweep over v ∈ {250, 500, 1000, 2000, 10000} \$/MWh covers depreciated hardware at the low end (\$250) and an effectively inflexible campus at the high end (\$10,000).

## 4 Theory

Let (P_v) denote the LP of §3.2 with VoLC v, let Z(v) be its optimal value (including curtailment cost), and let U*(v) = Σ_t u_t* be the optimal annual curtailment for any optimal solution. Write Z(∞) for the optimum of the inflexible problem (u ≡ 0), which is feasible whenever any dispatchable capacity is allowed.

**Proposition 1 (effective scarcity-price ceiling).** At any optimal primal–dual pair, every hour with u_t* < L satisfies

$$
\lambda_t \le v_{\text{eff}} \equiv v + (1-\varphi)\,\mu ,
$$

with equality whenever u_t* > 0. Hence curtailment occurs only in hours whose shadow price reaches v_eff. When the CFE constraint binds (μ > 0), v_eff > v.

*Proof.* The variable u_t appears in the objective with coefficient v, in the balance row with coefficient +1 (it relaxes the requirement), and in the CFE row with coefficient (1−φ) (curtailment reduces served energy and therefore the allowed gas). Its bounds are 0 ≤ u_t ≤ L, with duals ν_t^lo, ν_t^up ≥ 0. Stationarity of the Lagrangian gives v + (1−φ)μ − λ_t − ν_t^lo + ν_t^up = 0. If u_t < L, complementary slackness gives ν_t^up = 0, so λ_t = v_eff − ν_t^lo ≤ v_eff. If, in addition, u_t > 0, then ν_t^lo = 0 and λ_t = v_eff. ∎

*Interpretation.* With a share-based clean target, each curtailed MWh also removes (1−φ) MWh of gas allowance. The planner therefore values curtailment at v plus the marginal cost of that lost allowance. At φ = 0.9 and μ ≈ \$200/MWh (a typical value in our Abilene runs), v_eff exceeds v by about \$20/MWh. At φ = 1 the target is an absolute ban on gas, and the correction vanishes. A target expressed in absolute emissions rather than as a share would remove this penalty on flexibility.

**Proposition 2 (concavity and the flexibility dividend).** Z(v) is non-decreasing, concave and piecewise linear in v. U*(v) is non-increasing. For v₁ < v₂,

$$
U^*(v_2)\,(v_2 - v_1) \;\le\; Z(v_2) - Z(v_1) \;\le\; U^*(v_1)\,(v_2 - v_1).
$$

There is a finite v̄ with U*(v) = 0 for all v ≥ v̄, and the flexibility dividend satisfies Z(∞) − Z(v) = ∫_v^{v̄} U*(s) ds ≤ (v̄ − v) U*(v).

*Proof.* For a fixed feasible x, the objective C(x) + v U(x) is affine in v. Z is the pointwise minimum of such functions and is therefore concave. It is non-decreasing because U(x) ≥ 0. Let x₁ and x₂ be optimal at v₁ and v₂. Optimality of x₂ at v₂ gives Z(v₂) ≤ C(x₁) + v₂U(x₁) = Z(v₁) + (v₂ − v₁)U(x₁), which is the upper bound. Optimality of x₁ at v₁ gives Z(v₁) ≤ C(x₂) + v₁U(x₂) = Z(v₂) − (v₂ − v₁)U(x₂), which is the lower bound. Combining the two yields (v₂ − v₁)(U(x₁) − U(x₂)) ≥ 0, so U* is non-increasing. Because the LP has finitely many bases, Z is piecewise linear with slope U*(v) almost everywhere. Since U* ≥ 0 is non-increasing and Z is bounded above by Z(∞), U* must equal 0 beyond some v̄. Integrating the slope gives the identity, and monotonicity gives the final bound. ∎

*Use.* Proposition 2 turns a sweep over v into an exact accounting of what flexibility is worth. If the optimal curtailment at the empirical VoLC is small, the remaining dividend is at most (v̄ − v)·U*(v). In particular, flexibility cannot be worth much in a system where the planner chooses to use little of it.

**Proposition 3 (duration bound).** Suppose resource k has annualized fixed cost F_k per MW, constant hourly availability a_k, variable cost c_k < v and unbounded capacity. Suppose also that either (i) k is carbon-free, or (ii) k is gas and φ = 0. Then for every Δ > 0, the number of hours with u_t* ≥ Δ satisfies

$$
D(\Delta) \;\le\; \frac{F_k}{a_k\,(v - c_k)} .
$$

*Proof.* Take ε ∈ (0, Δ] and perturb an optimal solution. Add ε/a_k MW of resource k, dispatch it at ε in each of the D(Δ) hours with u_t ≥ Δ, and lower u_t by ε in those hours. The hourly balance is unchanged and u_t stays non-negative. In case (i), the CFE row's left side changes by −(1−φ)εD ≤ 0. In case (ii), with φ = 0, it changes by εD − εD = 0. The perturbation is therefore feasible, and its cost change is ε[F_k/a_k + D(c_k − v)]. This must be ≥ 0 by optimality, which gives the bound. ∎

*Numbers.* With the ATB-derived firm cost (F = \$860k/MW-yr, a = 0.93, c = \$13.6/MWh), the bound is 937 h/yr at v = \$1,000 and 469 h/yr at v = \$2,000. With a gas CT (F = \$127k/MW-yr, c = \$40.9/MWh) and φ = 0, it is 133 h/yr at v = \$1,000. The bounds hold for any weather year and any site, which makes them useful screening rules. A gas-backed campus with VoLC near \$1,000/MWh should never plan to curtail in more than about 130 hours per year.
