# Research portfolio review

_Generated 2026-09-18 by sciscout._

## Ranking

| # | Direction | Priority | Importance | Urgency | Confidence | TRL | P(market) | E[MOIC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Quantum error correction | **0.69** | 0.58 | 0.91 | 88% | 3 | 27% | 1.54x |
| 2 | Protein structure and design models | **0.69** | 0.59 | 0.86 | 88% | 5 | 26% | 5.59x |
| 3 | Machine learning climate emulators | **0.66** | 0.56 | 0.86 | 88% | 5 | 25% | 5.40x |
| 4 | Solid-state sodium batteries | **0.66** | 0.52 | 0.95 | 88% | 6 | 31% | 1.46x |
| 5 | CRISPR base and prime editing | **0.65** | 0.55 | 0.84 | 88% | 6 | 10% | 1.15x |
| 6 | Direct air carbon capture | **0.63** | 0.48 | 0.94 | 88% | 5 | 25% | 1.46x |
| 7 | Perovskite tandem photovoltaics | **0.63** | 0.49 | 0.91 | 88% | 5 | 27% | 1.56x |
| 8 | CAR-T for solid tumours | **0.61** | 0.48 | 0.87 | 88% | 7 | 12% | 1.25x |
| 9 | Neuromorphic photonic computing | **0.61** | 0.52 | 0.79 | 88% | 3 | 24% | 1.48x |
| 10 | Targeted protein degradation | **0.61** | 0.48 | 0.87 | 88% | 5 | 10% | 1.11x |
| 11 | Green hydrogen electrolysis catalysts | **0.61** | 0.47 | 0.88 | 88% | 3 | 24% | 1.53x |
| 12 | Topological qubits | **0.55** | 0.42 | 0.80 | 88% | 2 | 24% | 1.56x |
| 13 | Precision fermentation proteins | **0.54** | 0.42 | 0.78 | 88% | 5 | 30% | 4.99x |
| 14 | Gene-edited climate-resilient crops | **0.52** | 0.39 | 0.80 | 88% | 5 | 28% | 4.66x |
| 15 | Gut microbiome therapeutics | **0.50** | 0.39 | 0.75 | 88% | 5 | 11% | 1.25x |
| 16 | Ambient-pressure high-Tc superconductivity | **0.50** | 0.44 | 0.60 | 88% | 2 | 21% | 1.46x |

Priority combines importance and urgency geometrically, so a direction cannot rank highly on one axis alone. Confidence is the share of scoring inputs that were measured rather than assumed — read it alongside priority, because a confident 0.5 is a better basis for a decision than an unfounded 0.8.

## Quantum error correction

**Priority 0.69** (importance 0.58, urgency 0.91) | evidence confidence 88% | 84 works | 20 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 2-7, mean 4.4

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.48 | _measured_ — 75th-percentile normalised citation impact 2.14x expected across 84 works with citation data, against other tracks' works only; 84 compared within their own discipline and 0 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.48 | _measured_ — publication volume CAGR +22% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: computer science, physics, quantum physics |
| importance | evidence | 0.66 | _measured_ — mean publication-status weight 0.66 across 84 works, from 20 independent groups |
| importance | foundationality | 0.88 | _ASSUMED_ — 49 of 84 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 20 independent groups published in the last 12 months |
| urgency | acceleration | 0.76 | _measured_ — 25 works in the last 12 months vs historical mean 14.0/year (ratio 1.79) |
| urgency | industry_entry | 1.00 | _measured_ — 85% of works have at least one industry-affiliated author |

### Path to market

Sector: **Semiconductor devices and computing hardware**. Nominally **11.8 years** and **$752M** to market if every stage is survived, with cumulative technical success of **9.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Device demonstration | 1-4 | PERT(2, 4, 7) | PERT(2e+06, 1e+07, 4e+07) | Beta~(0.25, 0.40, 0.60) | _ASSUMED_ |
| Process integration and yield | 4-7 | PERT(2, 4, 8) | PERT(3e+07, 1.5e+08, 6e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| Qualification and ramp | 7-9 | PERT(1.5, 3, 6) | PERT(1e+08, 4e+08, 1.5e+09) | Beta~(0.45, 0.65, 0.85) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — Fab access or construction dominates cost and is largely fixed.
- _high_ **ecosystem** — A new device needs design tools, models and a supply chain before anyone can buy it.
- _medium_ **export_control** — Advanced-node and equipment export controls can foreclose entire markets.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$403M | -$114M | -$4M |
| Investor MOIC | 0.00x | 0.00x | 4.05x |
| Years to market _(successes)_ | 2.2 | 5.1 | 10.9 |
| Capital consumed _(successes)_ | $228M | $595M | $1.02B |
| Investor stake at exit _(successes)_ | 1.06% | 1.93% | 4.31% |

- Probability of reaching market: **26.7%**
- Expected investor multiple across all outcomes: **1.54x** (**5.77x** conditional on reaching market). Sampling error at this draw count: 1.47x to 1.61x (90% interval); the top 1% of draws contribute 28% of it
- Probability the project destroys value: **91.4%**
- Starting readiness simulated per draw: TRL 1 11%, TRL 2 13%, TRL 3 15%, TRL 4 15%, TRL 5 12%, TRL 6 15%, TRL 7 11%, TRL 8 7%, TRL 9 2%
- 1.6% of draws start with no development stage left and carry 30% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Device demonstration: 22.4%
- Process integration and yield: 37.0%
- Qualification and ramp: 13.7%
- reached market: 26.8%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.32 | +0.25 | _measured_ |
| Process integration and yield: cost (USD) | -0.23 | -0.10 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.19 | -0.48 | _ASSUMED_ |
| TAM (USD) | +0.15 | +0.46 | _ASSUMED_ |
| peak market share | +0.12 | +0.39 | _ASSUMED_ |
| exit revenue multiple | +0.07 | +0.17 | _ASSUMED_ |
| Device demonstration: cost (USD) | -0.05 | +0.02 | _ASSUMED_ |
| Device demonstration: P(success) | -0.04 | -0.02 | _ASSUMED_ |

**Where the spread comes from:** 38% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 62% is uncertainty about the inputs: 15.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **15.3% from assumptions**, and 31.1% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Protein structure and design models

**Priority 0.69** (importance 0.59, urgency 0.86) | evidence confidence 88% | 109 works | 24 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.46 | _measured_ — 75th-percentile normalised citation impact 2.04x expected across 109 works with citation data, against other tracks' works only; 93 compared within their own discipline and 16 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.49 | _measured_ — publication volume CAGR +23% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: biochemistry, computer science, structural biology |
| importance | evidence | 0.66 | _measured_ — mean publication-status weight 0.66 across 109 works, from 24 independent groups |
| importance | foundationality | 1.00 | _ASSUMED_ — 88 of 109 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 24 independent groups published in the last 12 months |
| urgency | acceleration | 0.61 | _measured_ — 26 works in the last 12 months vs historical mean 18.2/year (ratio 1.43) |
| urgency | industry_entry | 1.00 | _measured_ — 90% of works have at least one industry-affiliated author |

### Path to market

Sector: **Software, algorithms and applied machine learning**. Nominally **2.8 years** and **$51M** to market if every stage is survived, with cumulative technical success of **27.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Productisation | 5-8 | PERT(0.5, 1.5, 3) | PERT(2e+06, 1e+07, 4e+07) | Beta~(0.35, 0.55, 0.75) | _ASSUMED_ |
| Scale and distribution | 8-9 | PERT(0.5, 1, 2.5) | PERT(5e+06, 2.5e+07, 1.2e+08) | Beta~(0.30, 0.50, 0.70) | _ASSUMED_ |

**Constraints**

- _high_ **competition** — Low technical barriers mean fast replication; advantage comes from data, distribution or switching costs rather than the result itself.
- _medium_ **talent** — Small teams with scarce skills; key-person risk is concentrated.
- _medium_ **compute_cost** — Training and serving costs can dominate unit economics and move with hardware markets.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$40M | -$8M | $345M |
| Investor MOIC | 0.00x | 0.00x | 18.32x |
| Years to market _(successes)_ | 0.7 | 2.8 | 4.7 |
| Capital consumed _(successes)_ | $12M | $45M | $79M |
| Investor stake at exit _(successes)_ | 8.06% | 10.55% | 13.85% |

- Probability of reaching market: **25.5%**
- Expected investor multiple across all outcomes: **5.59x** (**21.92x** conditional on reaching market). Sampling error at this draw count: 5.49x to 5.69x (90% interval); the top 1% of draws contribute 11% of it
- Probability the project destroys value: **75.4%**
- Starting readiness simulated per draw: TRL 1 10%, TRL 2 12%, TRL 3 13%, TRL 4 13%, TRL 5 18%, TRL 6 14%, TRL 7 10%, TRL 8 7%, TRL 9 2%
- 2.3% of draws start with no development stage left and carry 12% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Research to working system: 19.5%
- Productisation: 32.3%
- Scale and distribution: 23.0%
- reached market: 25.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Productisation: cost (USD) | -0.15 | -0.00 | _ASSUMED_ |
| Scale and distribution: P(success) | +0.08 | +0.02 | _ASSUMED_ |
| Scale and distribution: cost (USD) | -0.08 | -0.05 | _ASSUMED_ |
| Research to working system: cost (USD) | -0.07 | -0.01 | _ASSUMED_ |
| TAM (USD) | +0.05 | +0.69 | _ASSUMED_ |
| peak market share | +0.04 | +0.57 | _ASSUMED_ |
| Research to working system: P(success) | -0.02 | -0.02 | _ASSUMED_ |
| Productisation: P(success) | +0.02 | +0.01 | _ASSUMED_ |

**Where the spread comes from:** 65% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 35% is uncertainty about the inputs: 8.6% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **12.1% from assumptions**, and 14.2% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Machine learning climate emulators

**Priority 0.66** (importance 0.56, urgency 0.86) | evidence confidence 88% | 85 works | 22 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.3

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.38 | _measured_ — 75th-percentile normalised citation impact 1.47x expected across 85 works with citation data, against other tracks' works only; 69 compared within their own discipline and 16 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.56 | _measured_ — publication volume CAGR +32% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: atmospheric science, computer science, environmental science |
| importance | evidence | 0.69 | _measured_ — mean publication-status weight 0.69 across 85 works, from 22 independent groups |
| importance | foundationality | 0.74 | _ASSUMED_ — 42 of 85 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 22 independent groups published in the last 12 months |
| urgency | acceleration | 0.60 | _measured_ — 20 works in the last 12 months vs historical mean 14.2/year (ratio 1.41) |
| urgency | industry_entry | 1.00 | _measured_ — 75% of works have at least one industry-affiliated author |

### Path to market

Sector: **Software, algorithms and applied machine learning**. Nominally **2.8 years** and **$51M** to market if every stage is survived, with cumulative technical success of **27.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Productisation | 5-8 | PERT(0.5, 1.5, 3) | PERT(2e+06, 1e+07, 4e+07) | Beta~(0.35, 0.55, 0.75) | _ASSUMED_ |
| Scale and distribution | 8-9 | PERT(0.5, 1, 2.5) | PERT(5e+06, 2.5e+07, 1.2e+08) | Beta~(0.30, 0.50, 0.70) | _ASSUMED_ |

**Constraints**

- _high_ **competition** — Low technical barriers mean fast replication; advantage comes from data, distribution or switching costs rather than the result itself.
- _medium_ **talent** — Small teams with scarce skills; key-person risk is concentrated.
- _medium_ **compute_cost** — Training and serving costs can dominate unit economics and move with hardware markets.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$40M | -$7M | $328M |
| Investor MOIC | 0.00x | 0.00x | 17.79x |
| Years to market _(successes)_ | 0.7 | 2.9 | 4.7 |
| Capital consumed _(successes)_ | $14M | $46M | $79M |
| Investor stake at exit _(successes)_ | 8.04% | 10.53% | 13.59% |

- Probability of reaching market: **24.7%**
- Expected investor multiple across all outcomes: **5.40x** (**21.88x** conditional on reaching market). Sampling error at this draw count: 5.30x to 5.49x (90% interval); the top 1% of draws contribute 11% of it
- Probability the project destroys value: **76.2%**
- Starting readiness simulated per draw: TRL 1 9%, TRL 2 14%, TRL 3 16%, TRL 4 13%, TRL 5 17%, TRL 6 13%, TRL 7 9%, TRL 8 6%, TRL 9 2%
- 2.1% of draws start with no development stage left and carry 11% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Research to working system: 21.2%
- Productisation: 32.2%
- Scale and distribution: 22.3%
- reached market: 24.3%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Productisation: cost (USD) | -0.15 | +0.00 | _ASSUMED_ |
| Scale and distribution: P(success) | +0.08 | +0.02 | _ASSUMED_ |
| Research to working system: cost (USD) | -0.08 | -0.02 | _ASSUMED_ |
| Scale and distribution: cost (USD) | -0.07 | -0.05 | _ASSUMED_ |
| TAM (USD) | +0.05 | +0.69 | _ASSUMED_ |
| peak market share | +0.04 | +0.57 | _ASSUMED_ |
| Research to working system: P(success) | -0.02 | -0.02 | _ASSUMED_ |
| Productisation: P(success) | +0.02 | +0.00 | _ASSUMED_ |

**Where the spread comes from:** 66% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 34% is uncertainty about the inputs: 8.3% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **11.6% from assumptions**, and 14.2% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Solid-state sodium batteries

**Priority 0.66** (importance 0.52, urgency 0.95) | evidence confidence 88% | 89 works | 22 groups

Readiness: TRL 6 (Demonstrated in relevant environment); 80% credible range 3-7, mean 5.3

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.40 | _measured_ — 75th-percentile normalised citation impact 1.62x expected across 89 works with citation data, against other tracks' works only; 88 compared within their own discipline and 1 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.66 | _measured_ — publication volume CAGR +47% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: chemistry, electrical engineering, materials science |
| importance | evidence | 0.68 | _measured_ — mean publication-status weight 0.68 across 89 works, from 22 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 89 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 22 independent groups published in the last 12 months |
| urgency | acceleration | 0.86 | _measured_ — 31 works in the last 12 months vs historical mean 14.8/year (ratio 2.09) |
| urgency | industry_entry | 1.00 | _measured_ — 85% of works have at least one industry-affiliated author |

### Path to market

Sector: **Energy generation, storage and industrial hardware**. Nominally **9.8 years** and **$790M** to market if every stage is survived, with cumulative technical success of **21.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Pilot and scale-up | 4-7 | PERT(3, 5, 10) | PERT(2e+07, 8e+07, 3e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| First commercial plant | 7-9 | PERT(2, 4, 8) | PERT(1e+08, 5e+08, 2e+09) | Beta~(0.40, 0.60, 0.80) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — First-of-a-kind plants are capital-intensive and rarely financeable on venture terms alone.
- _high_ **supply_chain** — Critical-mineral and specialised-equipment lead times can set the schedule regardless of technical readiness.
- _high_ **permitting** — Siting, grid interconnection and environmental permitting routinely add multi-year delays.
- _high_ **incumbent_cost** — Must beat an incumbent whose costs keep falling; the target moves during development.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$520M | -$90M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 3.98x |
| Years to market _(successes)_ | 3.0 | 5.5 | 11.7 |
| Capital consumed _(successes)_ | $270M | $670M | $1.20B |
| Investor stake at exit _(successes)_ | 0.83% | 1.65% | 3.67% |

- Probability of reaching market: **30.7%**
- Expected investor multiple across all outcomes: **1.46x** (**4.76x** conditional on reaching market). Sampling error at this draw count: 1.41x to 1.52x (90% interval); the top 1% of draws contribute 24% of it
- Probability the project destroys value: **91.8%**
- Starting readiness simulated per draw: TRL 1 4%, TRL 2 5%, TRL 3 10%, TRL 4 9%, TRL 5 15%, TRL 6 27%, TRL 7 21%, TRL 8 7%, TRL 9 1%
- 1.1% of draws start with no development stage left and carry 22% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Lab validation: 10.8%
- Pilot and scale-up: 39.0%
- First commercial plant: 19.7%
- reached market: 30.5%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.40 | +0.04 | _measured_ |
| First commercial plant: cost (USD) | -0.26 | -0.61 | _ASSUMED_ |
| Pilot and scale-up: cost (USD) | -0.19 | -0.06 | _ASSUMED_ |
| TAM (USD) | +0.17 | +0.46 | _ASSUMED_ |
| peak market share | +0.13 | +0.36 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.16 | _ASSUMED_ |
| First commercial plant: P(success) | +0.04 | +0.01 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.04 | -0.02 | _ASSUMED_ |

**Where the spread comes from:** 35% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 65% is uncertainty about the inputs: 16.0% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **20.0% from assumptions**, and 28.6% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## CRISPR base and prime editing

**Priority 0.65** (importance 0.55, urgency 0.84) | evidence confidence 88% | 102 works | 26 groups

Readiness: TRL 6 (Demonstrated in relevant environment); 80% credible range 1-6, mean 4.9

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.71 | _measured_ — 75th-percentile normalised citation impact 4.42x expected across 102 works with citation data, against other tracks' works only; 87 compared within their own discipline and 15 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.42 | _measured_ — publication volume CAGR +15% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: genetics, medicine, molecular biology |
| importance | evidence | 0.71 | _measured_ — mean publication-status weight 0.71 across 102 works, from 26 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 102 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 26 independent groups published in the last 12 months |
| urgency | acceleration | 0.55 | _measured_ — 22 works in the last 12 months vs historical mean 17.0/year (ratio 1.29) |
| urgency | industry_entry | 1.00 | _measured_ — 91% of works have at least one industry-affiliated author |

### Path to market

Sector: **Therapeutics and biologics**. Nominally **8.1 years** and **$446M** to market if every stage is survived, with cumulative technical success of **8.4%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Phase I | 6-7 | PERT(1, 1.7, 3) | PERT(8e+06, 2.5e+07, 6e+07) | Beta~(0.45, 0.52, 0.60) | _sourced_ |
| Phase II | 7-8 | PERT(1.5, 2.5, 4.5) | PERT(2e+07, 6e+07, 1.5e+08) | Beta~(0.24, 0.29, 0.35) | _sourced_ |
| Phase III and filing | 8-9 | PERT(2, 3.5, 6) | PERT(1e+08, 3e+08, 8e+08) | Beta~(0.45, 0.55, 0.65) | _sourced_ |

**Constraints**

- _high_ **regulatory** — Approval is a hard gate with no partial credit; timelines are set by the agency, not by the programme.
- _high_ **capital** — Phase III spend is concentrated and non-recoverable if the readout misses.
- _high_ **reimbursement** — Approval does not imply payment; payer coverage decisions can gate revenue for years.
- _medium_ **ip** — Effective exclusivity is the patent term minus development time, so slow programmes earn less even when they succeed.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$114M | -$25M | -$4M |
| Investor MOIC | 0.00x | 0.00x | 0.50x |
| Years to market _(successes)_ | 0.0 | 6.0 | 10.6 |
| Capital consumed _(successes)_ | $0 | $368M | $589M |
| Investor stake at exit _(successes)_ | 1.93% | 3.81% | 15.00% |

- Probability of reaching market: **10.3%**
- Expected investor multiple across all outcomes: **1.15x** (**11.24x** conditional on reaching market). Sampling error at this draw count: 1.09x to 1.22x (90% interval); the top 1% of draws contribute 37% of it
- Probability the project destroys value: **94.9%**
- Starting readiness simulated per draw: TRL 1 7%, TRL 2 8%, TRL 3 9%, TRL 4 13%, TRL 5 20%, TRL 6 26%, TRL 7 11%, TRL 8 5%, TRL 9 2%
- 1.6% of draws start with no development stage left and carry 40% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Discovery and target validation: 9.1%
- Preclinical: 25.3%
- Phase I: 22.8%
- Phase II: 25.4%
- Phase III and filing: 7.2%
- reached market: 10.1%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.44 | +0.40 | _measured_ |
| Preclinical: cost (USD) | -0.10 | -0.02 | _ASSUMED_ |
| Phase I: cost (USD) | -0.08 | -0.04 | _sourced_ |
| Phase II: cost (USD) | -0.06 | -0.04 | _sourced_ |
| peak market share | +0.06 | +0.42 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.55 | _ASSUMED_ |
| Preclinical: P(success) | -0.06 | -0.04 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.05 | -0.28 | _sourced_ |

**Where the spread comes from:** 37% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 63% is uncertainty about the inputs: 23.5% from not knowing the starting readiness level, 0.0% from measured inputs, 0.6% from sourced figures, **3.5% from assumptions**, and 34.9% from combinations of inputs that no single one accounts for.

23 of 24 simulation inputs are flagged as placeholders awaiting a real number.

## Direct air carbon capture

**Priority 0.63** (importance 0.48, urgency 0.94) | evidence confidence 88% | 68 works | 18 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.42 | _measured_ — 75th-percentile normalised citation impact 1.71x expected across 68 works with citation data, against other tracks' works only; 68 compared within their own discipline and 0 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.51 | _measured_ — publication volume CAGR +27% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: chemical engineering, chemistry, environmental science |
| importance | evidence | 0.67 | _measured_ — mean publication-status weight 0.67 across 68 works, from 18 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 68 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 18 independent groups published in the last 12 months |
| urgency | acceleration | 0.84 | _measured_ — 23 works in the last 12 months vs historical mean 11.3/year (ratio 2.03) |
| urgency | industry_entry | 1.00 | _measured_ — 84% of works have at least one industry-affiliated author |

### Path to market

Sector: **Energy generation, storage and industrial hardware**. Nominally **9.8 years** and **$790M** to market if every stage is survived, with cumulative technical success of **21.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Pilot and scale-up | 4-7 | PERT(3, 5, 10) | PERT(2e+07, 8e+07, 3e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| First commercial plant | 7-9 | PERT(2, 4, 8) | PERT(1e+08, 5e+08, 2e+09) | Beta~(0.40, 0.60, 0.80) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — First-of-a-kind plants are capital-intensive and rarely financeable on venture terms alone.
- _high_ **supply_chain** — Critical-mineral and specialised-equipment lead times can set the schedule regardless of technical readiness.
- _high_ **permitting** — Siting, grid interconnection and environmental permitting routinely add multi-year delays.
- _high_ **incumbent_cost** — Must beat an incumbent whose costs keep falling; the target moves during development.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$427M | -$71M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 3.62x |
| Years to market _(successes)_ | 2.9 | 7.6 | 13.0 |
| Capital consumed _(successes)_ | $235M | $671M | $1.21B |
| Investor stake at exit _(successes)_ | 0.88% | 1.79% | 4.57% |

- Probability of reaching market: **25.2%**
- Expected investor multiple across all outcomes: **1.46x** (**5.80x** conditional on reaching market). Sampling error at this draw count: 1.39x to 1.53x (90% interval); the top 1% of draws contribute 31% of it
- Probability the project destroys value: **93.0%**
- Starting readiness simulated per draw: TRL 1 7%, TRL 2 9%, TRL 3 16%, TRL 4 15%, TRL 5 20%, TRL 6 15%, TRL 7 11%, TRL 8 5%, TRL 9 2%
- 1.7% of draws start with no development stage left and carry 34% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Lab validation: 17.8%
- Pilot and scale-up: 41.8%
- First commercial plant: 15.5%
- reached market: 24.9%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.41 | +0.10 | _measured_ |
| Pilot and scale-up: cost (USD) | -0.21 | -0.07 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.19 | -0.57 | _ASSUMED_ |
| TAM (USD) | +0.13 | +0.43 | _ASSUMED_ |
| peak market share | +0.11 | +0.34 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.18 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.03 | -0.02 | _ASSUMED_ |
| Lab validation: cost (USD) | -0.03 | +0.00 | _ASSUMED_ |

**Where the spread comes from:** 34% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 66% is uncertainty about the inputs: 22.3% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **13.7% from assumptions**, and 29.6% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Perovskite tandem photovoltaics

**Priority 0.63** (importance 0.49, urgency 0.91) | evidence confidence 88% | 88 works | 24 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 4.7

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.48 | _measured_ — 75th-percentile normalised citation impact 2.15x expected across 88 works with citation data, against other tracks' works only; 88 compared within their own discipline and 0 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.51 | _measured_ — publication volume CAGR +27% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: chemistry, materials science, physics |
| importance | evidence | 0.64 | _measured_ — mean publication-status weight 0.64 across 88 works, from 24 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 88 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 24 independent groups published in the last 12 months |
| urgency | acceleration | 0.73 | _measured_ — 25 works in the last 12 months vs historical mean 14.7/year (ratio 1.70) |
| urgency | industry_entry | 1.00 | _measured_ — 80% of works have at least one industry-affiliated author |

### Path to market

Sector: **Energy generation, storage and industrial hardware**. Nominally **9.8 years** and **$790M** to market if every stage is survived, with cumulative technical success of **21.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Pilot and scale-up | 4-7 | PERT(3, 5, 10) | PERT(2e+07, 8e+07, 3e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| First commercial plant | 7-9 | PERT(2, 4, 8) | PERT(1e+08, 5e+08, 2e+09) | Beta~(0.40, 0.60, 0.80) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — First-of-a-kind plants are capital-intensive and rarely financeable on venture terms alone.
- _high_ **supply_chain** — Critical-mineral and specialised-equipment lead times can set the schedule regardless of technical readiness.
- _high_ **permitting** — Siting, grid interconnection and environmental permitting routinely add multi-year delays.
- _high_ **incumbent_cost** — Must beat an incumbent whose costs keep falling; the target moves during development.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$449M | -$75M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 3.78x |
| Years to market _(successes)_ | 2.8 | 6.9 | 12.6 |
| Capital consumed _(successes)_ | $226M | $667M | $1.21B |
| Investor stake at exit _(successes)_ | 0.87% | 1.77% | 4.56% |

- Probability of reaching market: **26.7%**
- Expected investor multiple across all outcomes: **1.56x** (**5.83x** conditional on reaching market). Sampling error at this draw count: 1.48x to 1.63x (90% interval); the top 1% of draws contribute 31% of it
- Probability the project destroys value: **92.6%**
- Starting readiness simulated per draw: TRL 1 8%, TRL 2 10%, TRL 3 11%, TRL 4 11%, TRL 5 23%, TRL 6 17%, TRL 7 13%, TRL 8 5%, TRL 9 2%
- 1.9% of draws start with no development stage left and carry 36% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Lab validation: 15.6%
- Pilot and scale-up: 41.5%
- First commercial plant: 16.6%
- reached market: 26.3%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.39 | +0.11 | _measured_ |
| Pilot and scale-up: cost (USD) | -0.21 | -0.06 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.20 | -0.57 | _ASSUMED_ |
| TAM (USD) | +0.14 | +0.43 | _ASSUMED_ |
| peak market share | +0.11 | +0.34 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.17 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.03 | -0.02 | _ASSUMED_ |
| Pilot and scale-up: duration (y) | +0.03 | +0.04 | _ASSUMED_ |

**Where the spread comes from:** 33% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 67% is uncertainty about the inputs: 22.2% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **14.7% from assumptions**, and 30.2% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## CAR-T for solid tumours

**Priority 0.61** (importance 0.48, urgency 0.87) | evidence confidence 88% | 81 works | 24 groups

Readiness: TRL 7 (Prototype demonstrated in operational environment); 80% credible range 2-7, mean 4.9

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.53 | _measured_ — 75th-percentile normalised citation impact 2.60x expected across 81 works with citation data, against other tracks' works only; 72 compared within their own discipline and 9 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.35 | _measured_ — publication volume CAGR +5% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: immunology, medicine, oncology |
| importance | evidence | 0.70 | _measured_ — mean publication-status weight 0.70 across 81 works, from 24 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 81 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 24 independent groups published in the last 12 months |
| urgency | acceleration | 0.64 | _measured_ — 20 works in the last 12 months vs historical mean 13.5/year (ratio 1.48) |
| urgency | industry_entry | 1.00 | _measured_ — 89% of works have at least one industry-affiliated author |

### Path to market

Sector: **Therapeutics and biologics**. Nominally **6.3 years** and **$418M** to market if every stage is survived, with cumulative technical success of **16.0%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Phase II | 7-8 | PERT(1.5, 2.5, 4.5) | PERT(2e+07, 6e+07, 1.5e+08) | Beta~(0.24, 0.29, 0.35) | _sourced_ |
| Phase III and filing | 8-9 | PERT(2, 3.5, 6) | PERT(1e+08, 3e+08, 8e+08) | Beta~(0.45, 0.55, 0.65) | _sourced_ |

**Constraints**

- _high_ **regulatory** — Approval is a hard gate with no partial credit; timelines are set by the agency, not by the programme.
- _high_ **capital** — Phase III spend is concentrated and non-recoverable if the readout misses.
- _high_ **reimbursement** — Approval does not imply payment; payer coverage decisions can gate revenue for years.
- _medium_ **ip** — Effective exclusivity is the patent term minus development time, so slow programmes earn less even when they succeed.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$141M | -$26M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 1.71x |
| Years to market _(successes)_ | 0.0 | 5.3 | 10.2 |
| Capital consumed _(successes)_ | $0 | $362M | $586M |
| Investor stake at exit _(successes)_ | 1.85% | 3.48% | 15.00% |

- Probability of reaching market: **11.8%**
- Expected investor multiple across all outcomes: **1.25x** (**10.62x** conditional on reaching market). Sampling error at this draw count: 1.18x to 1.32x (90% interval); the top 1% of draws contribute 36% of it
- Probability the project destroys value: **94.0%**
- Starting readiness simulated per draw: TRL 1 7%, TRL 2 9%, TRL 3 10%, TRL 4 12%, TRL 5 17%, TRL 6 18%, TRL 7 18%, TRL 8 7%, TRL 9 2%
- 1.7% of draws start with no development stage left and carry 40% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Discovery and target validation: 10.3%
- Preclinical: 24.3%
- Phase I: 18.5%
- Phase II: 26.8%
- Phase III and filing: 8.4%
- reached market: 11.7%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.45 | +0.38 | _measured_ |
| Preclinical: cost (USD) | -0.09 | -0.02 | _ASSUMED_ |
| Phase II: cost (USD) | -0.08 | -0.04 | _sourced_ |
| TAM (USD) | +0.07 | +0.56 | _ASSUMED_ |
| peak market share | +0.07 | +0.43 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.06 | -0.27 | _sourced_ |
| Phase I: cost (USD) | -0.06 | -0.03 | _sourced_ |
| Preclinical: P(success) | -0.05 | -0.04 | _ASSUMED_ |

**Where the spread comes from:** 40% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 60% is uncertainty about the inputs: 21.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.7% from sourced figures, **4.3% from assumptions**, and 33.7% from combinations of inputs that no single one accounts for.

23 of 24 simulation inputs are flagged as placeholders awaiting a real number.

## Neuromorphic photonic computing

**Priority 0.61** (importance 0.52, urgency 0.79) | evidence confidence 88% | 39 works | 16 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 1-6, mean 4.0

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.28 | _measured_ — 75th-percentile normalised citation impact 0.94x expected across 39 works with citation data, against other tracks' works only; 39 compared within their own discipline and 0 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.39 | _measured_ — publication volume CAGR +10% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: computer science, electrical engineering, physics |
| importance | evidence | 0.71 | _measured_ — mean publication-status weight 0.71 across 39 works, from 16 independent groups |
| importance | foundationality | 1.00 | _ASSUMED_ — 39 of 39 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.96 | _measured_ — 13 independent groups published in the last 12 months |
| urgency | acceleration | 0.44 | _measured_ — 7 works in the last 12 months vs historical mean 6.5/year (ratio 1.08) |
| urgency | industry_entry | 1.00 | _measured_ — 59% of works have at least one industry-affiliated author |

### Path to market

Sector: **Semiconductor devices and computing hardware**. Nominally **11.8 years** and **$752M** to market if every stage is survived, with cumulative technical success of **9.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Device demonstration | 1-4 | PERT(2, 4, 7) | PERT(2e+06, 1e+07, 4e+07) | Beta~(0.25, 0.40, 0.60) | _ASSUMED_ |
| Process integration and yield | 4-7 | PERT(2, 4, 8) | PERT(3e+07, 1.5e+08, 6e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| Qualification and ramp | 7-9 | PERT(1.5, 3, 6) | PERT(1e+08, 4e+08, 1.5e+09) | Beta~(0.45, 0.65, 0.85) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — Fab access or construction dominates cost and is largely fixed.
- _high_ **ecosystem** — A new device needs design tools, models and a supply chain before anyone can buy it.
- _medium_ **export_control** — Advanced-node and equipment export controls can foreclose entire markets.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$368M | -$96M | -$4M |
| Investor MOIC | 0.00x | 0.00x | 3.75x |
| Years to market _(successes)_ | 2.1 | 6.2 | 11.8 |
| Capital consumed _(successes)_ | $214M | $606M | $1.03B |
| Investor stake at exit _(successes)_ | 1.09% | 2.00% | 4.89% |

- Probability of reaching market: **23.6%**
- Expected investor multiple across all outcomes: **1.48x** (**6.29x** conditional on reaching market). Sampling error at this draw count: 1.42x to 1.55x (90% interval); the top 1% of draws contribute 31% of it
- Probability the project destroys value: **92.3%**
- Starting readiness simulated per draw: TRL 1 11%, TRL 2 16%, TRL 3 21%, TRL 4 16%, TRL 5 11%, TRL 6 10%, TRL 7 8%, TRL 8 6%, TRL 9 2%
- 1.8% of draws start with no development stage left and carry 35% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Device demonstration: 28.0%
- Process integration and yield: 36.3%
- Qualification and ramp: 11.9%
- reached market: 23.7%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.34 | +0.26 | _measured_ |
| Process integration and yield: cost (USD) | -0.21 | -0.12 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.15 | -0.45 | _ASSUMED_ |
| TAM (USD) | +0.13 | +0.46 | _ASSUMED_ |
| peak market share | +0.10 | +0.38 | _ASSUMED_ |
| Device demonstration: cost (USD) | -0.08 | +0.01 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.17 | _ASSUMED_ |
| Device demonstration: P(success) | -0.06 | -0.02 | _ASSUMED_ |

**Where the spread comes from:** 37% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 63% is uncertainty about the inputs: 19.1% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **12.8% from assumptions**, and 30.8% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Targeted protein degradation

**Priority 0.61** (importance 0.48, urgency 0.87) | evidence confidence 88% | 99 works | 28 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 3-7, mean 5.0

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.45 | _measured_ — 75th-percentile normalised citation impact 1.94x expected across 99 works with citation data, against other tracks' works only; 83 compared within their own discipline and 16 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.45 | _measured_ — publication volume CAGR +19% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: biochemistry, medicine, pharmacology |
| importance | evidence | 0.69 | _measured_ — mean publication-status weight 0.69 across 99 works, from 28 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 99 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 27 independent groups published in the last 12 months |
| urgency | acceleration | 0.62 | _measured_ — 24 works in the last 12 months vs historical mean 16.5/year (ratio 1.45) |
| urgency | industry_entry | 1.00 | _measured_ — 95% of works have at least one industry-affiliated author |

### Path to market

Sector: **Therapeutics and biologics**. Nominally **10.8 years** and **$458M** to market if every stage is survived, with cumulative technical success of **3.8%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Preclinical | 3-6 | PERT(1, 2.5, 5) | PERT(3e+06, 1e+07, 3e+07) | Beta~(0.30, 0.45, 0.65) | _ASSUMED_ |
| Phase I | 6-7 | PERT(1, 1.7, 3) | PERT(8e+06, 2.5e+07, 6e+07) | Beta~(0.45, 0.52, 0.60) | _sourced_ |
| Phase II | 7-8 | PERT(1.5, 2.5, 4.5) | PERT(2e+07, 6e+07, 1.5e+08) | Beta~(0.24, 0.29, 0.35) | _sourced_ |
| Phase III and filing | 8-9 | PERT(2, 3.5, 6) | PERT(1e+08, 3e+08, 8e+08) | Beta~(0.45, 0.55, 0.65) | _sourced_ |

**Constraints**

- _high_ **regulatory** — Approval is a hard gate with no partial credit; timelines are set by the agency, not by the programme.
- _high_ **capital** — Phase III spend is concentrated and non-recoverable if the readout misses.
- _high_ **reimbursement** — Approval does not imply payment; payer coverage decisions can gate revenue for years.
- _medium_ **ip** — Effective exclusivity is the patent term minus development time, so slow programmes earn less even when they succeed.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$122M | -$27M | -$4M |
| Investor MOIC | 0.00x | 0.00x | 0.61x |
| Years to market _(successes)_ | 0.0 | 6.2 | 10.6 |
| Capital consumed _(successes)_ | $0 | $376M | $598M |
| Investor stake at exit _(successes)_ | 1.94% | 3.65% | 15.00% |

- Probability of reaching market: **10.3%**
- Expected investor multiple across all outcomes: **1.11x** (**10.69x** conditional on reaching market). Sampling error at this draw count: 1.04x to 1.17x (90% interval); the top 1% of draws contribute 35% of it
- Probability the project destroys value: **95.0%**
- Starting readiness simulated per draw: TRL 1 5%, TRL 2 6%, TRL 3 8%, TRL 4 14%, TRL 5 21%, TRL 6 21%, TRL 7 19%, TRL 8 5%, TRL 9 1%
- 1.3% of draws start with no development stage left and carry 36% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Discovery and target validation: 7.5%
- Preclinical: 25.7%
- Phase I: 20.2%
- Phase II: 28.8%
- Phase III and filing: 7.6%
- reached market: 10.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.46 | +0.35 | _measured_ |
| Preclinical: cost (USD) | -0.10 | -0.03 | _ASSUMED_ |
| Phase II: cost (USD) | -0.08 | -0.04 | _sourced_ |
| Phase I: cost (USD) | -0.07 | -0.03 | _sourced_ |
| TAM (USD) | +0.07 | +0.57 | _ASSUMED_ |
| peak market share | +0.07 | +0.42 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.06 | -0.30 | _sourced_ |
| Preclinical: P(success) | -0.05 | -0.04 | _ASSUMED_ |

**Where the spread comes from:** 41% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 59% is uncertainty about the inputs: 22.0% from not knowing the starting readiness level, 0.0% from measured inputs, 0.7% from sourced figures, **3.6% from assumptions**, and 32.7% from combinations of inputs that no single one accounts for.

23 of 24 simulation inputs are flagged as placeholders awaiting a real number.

## Green hydrogen electrolysis catalysts

**Priority 0.61** (importance 0.47, urgency 0.88) | evidence confidence 88% | 80 works | 20 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 1-6, mean 4.2

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.43 | _measured_ — 75th-percentile normalised citation impact 1.83x expected across 80 works with citation data, against other tracks' works only; 80 compared within their own discipline and 0 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.46 | _measured_ — publication volume CAGR +20% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: chemical engineering, chemistry, materials science |
| importance | evidence | 0.66 | _measured_ — mean publication-status weight 0.66 across 80 works, from 20 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 80 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 19 independent groups published in the last 12 months |
| urgency | acceleration | 0.68 | _measured_ — 21 works in the last 12 months vs historical mean 13.3/year (ratio 1.57) |
| urgency | industry_entry | 1.00 | _measured_ — 68% of works have at least one industry-affiliated author |

### Path to market

Sector: **Energy generation, storage and industrial hardware**. Nominally **14.2 years** and **$797M** to market if every stage is survived, with cumulative technical success of **9.7%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Lab validation | 1-4 | PERT(2, 4, 8) | PERT(1e+06, 5e+06, 2e+07) | Beta~(0.25, 0.45, 0.65) | _ASSUMED_ |
| Pilot and scale-up | 4-7 | PERT(3, 5, 10) | PERT(2e+07, 8e+07, 3e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| First commercial plant | 7-9 | PERT(2, 4, 8) | PERT(1e+08, 5e+08, 2e+09) | Beta~(0.40, 0.60, 0.80) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — First-of-a-kind plants are capital-intensive and rarely financeable on venture terms alone.
- _high_ **supply_chain** — Critical-mineral and specialised-equipment lead times can set the schedule regardless of technical readiness.
- _high_ **permitting** — Siting, grid interconnection and environmental permitting routinely add multi-year delays.
- _high_ **incumbent_cost** — Must beat an incumbent whose costs keep falling; the target moves during development.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$398M | -$62M | -$2M |
| Investor MOIC | 0.00x | 0.00x | 3.60x |
| Years to market _(successes)_ | 2.6 | 7.8 | 13.4 |
| Capital consumed _(successes)_ | $188M | $662M | $1.20B |
| Investor stake at exit _(successes)_ | 0.89% | 1.87% | 5.50% |

- Probability of reaching market: **23.9%**
- Expected investor multiple across all outcomes: **1.53x** (**6.40x** conditional on reaching market). Sampling error at this draw count: 1.46x to 1.60x (90% interval); the top 1% of draws contribute 33% of it
- Probability the project destroys value: **93.0%**
- Starting readiness simulated per draw: TRL 1 9%, TRL 2 11%, TRL 3 19%, TRL 4 18%, TRL 5 15%, TRL 6 11%, TRL 7 8%, TRL 8 6%, TRL 9 2%
- 2.1% of draws start with no development stage left and carry 40% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Lab validation: 21.8%
- Pilot and scale-up: 40.0%
- First commercial plant: 14.5%
- reached market: 23.6%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.40 | +0.13 | _measured_ |
| Pilot and scale-up: cost (USD) | -0.19 | -0.06 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.17 | -0.55 | _ASSUMED_ |
| TAM (USD) | +0.12 | +0.41 | _ASSUMED_ |
| peak market share | +0.10 | +0.33 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.17 | _ASSUMED_ |
| Lab validation: cost (USD) | -0.05 | +0.01 | _ASSUMED_ |
| Pilot and scale-up: duration (y) | +0.03 | +0.04 | _ASSUMED_ |

**Where the spread comes from:** 32% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 68% is uncertainty about the inputs: 25.3% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **12.1% from assumptions**, and 30.9% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Topological qubits

**Priority 0.55** (importance 0.42, urgency 0.80) | evidence confidence 88% | 36 works | 14 groups

Readiness: TRL 2 (Technology concept formulated); 80% credible range 1-6, mean 3.9

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.39 | _measured_ — 75th-percentile normalised citation impact 1.58x expected across 36 works with citation data, against other tracks' works only; 34 compared within their own discipline and 2 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.30 | _measured_ — publication volume CAGR -3% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: condensed matter physics, physics, quantum physics |
| importance | evidence | 0.68 | _measured_ — mean publication-status weight 0.68 across 36 works, from 14 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 36 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.96 | _measured_ — 13 independent groups published in the last 12 months |
| urgency | acceleration | 0.48 | _measured_ — 7 works in the last 12 months vs historical mean 6.0/year (ratio 1.17) |
| urgency | industry_entry | 1.00 | _measured_ — 83% of works have at least one industry-affiliated author |

### Path to market

Sector: **Semiconductor devices and computing hardware**. Nominally **11.8 years** and **$752M** to market if every stage is survived, with cumulative technical success of **9.5%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Device demonstration | 1-4 | PERT(2, 4, 7) | PERT(2e+06, 1e+07, 4e+07) | Beta~(0.25, 0.40, 0.60) | _ASSUMED_ |
| Process integration and yield | 4-7 | PERT(2, 4, 8) | PERT(3e+07, 1.5e+08, 6e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| Qualification and ramp | 7-9 | PERT(1.5, 3, 6) | PERT(1e+08, 4e+08, 1.5e+09) | Beta~(0.45, 0.65, 0.85) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — Fab access or construction dominates cost and is largely fixed.
- _high_ **ecosystem** — A new device needs design tools, models and a supply chain before anyone can buy it.
- _medium_ **export_control** — Advanced-node and equipment export controls can foreclose entire markets.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$372M | -$97M | -$4M |
| Investor MOIC | 0.00x | 0.00x | 3.87x |
| Years to market _(successes)_ | 2.0 | 5.9 | 11.7 |
| Capital consumed _(successes)_ | $197M | $598M | $1.02B |
| Investor stake at exit _(successes)_ | 1.08% | 2.00% | 5.07% |

- Probability of reaching market: **24.2%**
- Expected investor multiple across all outcomes: **1.56x** (**6.45x** conditional on reaching market). Sampling error at this draw count: 1.49x to 1.64x (90% interval); the top 1% of draws contribute 32% of it
- Probability the project destroys value: **92.0%**
- Starting readiness simulated per draw: TRL 1 16%, TRL 2 19%, TRL 3 12%, TRL 4 11%, TRL 5 15%, TRL 6 11%, TRL 7 8%, TRL 8 6%, TRL 9 2%
- 1.9% of draws start with no development stage left and carry 37% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Device demonstration: 27.6%
- Process integration and yield: 35.9%
- Qualification and ramp: 12.1%
- reached market: 24.3%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.33 | +0.27 | _measured_ |
| Process integration and yield: cost (USD) | -0.21 | -0.11 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.15 | -0.45 | _ASSUMED_ |
| TAM (USD) | +0.14 | +0.45 | _ASSUMED_ |
| peak market share | +0.11 | +0.38 | _ASSUMED_ |
| Device demonstration: cost (USD) | -0.08 | +0.01 | _ASSUMED_ |
| exit revenue multiple | +0.06 | +0.16 | _ASSUMED_ |
| Device demonstration: P(success) | -0.06 | -0.02 | _ASSUMED_ |

**Where the spread comes from:** 36% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 64% is uncertainty about the inputs: 19.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **12.9% from assumptions**, and 31.7% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Precision fermentation proteins

**Priority 0.54** (importance 0.42, urgency 0.78) | evidence confidence 88% | 61 works | 18 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 5.0

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.28 | _measured_ — 75th-percentile normalised citation impact 0.94x expected across 61 works with citation data, against other tracks' works only; 43 compared within their own discipline and 18 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.41 | _measured_ — publication volume CAGR +14% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: biotechnology, chemical engineering, food science |
| importance | evidence | 0.68 | _measured_ — mean publication-status weight 0.68 across 61 works, from 18 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 61 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 17 independent groups published in the last 12 months |
| urgency | acceleration | 0.39 | _measured_ — 10 works in the last 12 months vs historical mean 10.2/year (ratio 0.98) |
| urgency | industry_entry | 1.00 | _measured_ — 90% of works have at least one industry-affiliated author |

### Path to market

Sector: **Agriculture, food and bio-production**. Nominally **8.1 years** and **$102M** to market if every stage is survived, with cumulative technical success of **24.8%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Field trials or pilot production | 4-7 | PERT(2, 4, 7) | PERT(5e+06, 2e+07, 7e+07) | Beta~(0.25, 0.45, 0.65) | _ASSUMED_ |
| Regulatory approval and market entry | 7-9 | PERT(1.5, 3.5, 8) | PERT(1e+07, 5e+07, 2.5e+08) | Beta~(0.35, 0.55, 0.75) | _ASSUMED_ |

**Constraints**

- _high_ **regulatory** — Novel food and gene-edited crop rules differ sharply by jurisdiction; approval in one market implies little about another.
- _high_ **seasonality** — Field validation is paced by growing seasons and cannot be accelerated with capital.
- _high_ **margin** — Competing against commodity agricultural inputs leaves little pricing headroom.
- _medium_ **consumer_acceptance** — Public acceptance of edited or fermented products can gate adoption independently of approval.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$58M | -$14M | $250M |
| Investor MOIC | 0.00x | 0.00x | 16.45x |
| Years to market _(successes)_ | 2.5 | 6.4 | 10.4 |
| Capital consumed _(successes)_ | $30M | $83M | $152M |
| Investor stake at exit _(successes)_ | 5.20% | 7.82% | 11.42% |

- Probability of reaching market: **29.8%**
- Expected investor multiple across all outcomes: **4.99x** (**16.77x** conditional on reaching market). Sampling error at this draw count: 4.90x to 5.09x (90% interval); the top 1% of draws contribute 12% of it
- Probability the project destroys value: **73.5%**
- Starting readiness simulated per draw: TRL 1 6%, TRL 2 7%, TRL 3 9%, TRL 4 8%, TRL 5 24%, TRL 6 22%, TRL 7 16%, TRL 8 6%, TRL 9 2%
- 1.5% of draws start with no development stage left and carry 9% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Strain or trait development: 11.3%
- Field trials or pilot production: 35.7%
- Regulatory approval and market entry: 23.1%
- reached market: 29.8%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Field trials or pilot production: cost (USD) | -0.18 | -0.04 | _ASSUMED_ |
| Regulatory approval and market entry: cost (USD) | -0.12 | -0.14 | _ASSUMED_ |
| TAM (USD) | +0.10 | +0.65 | _ASSUMED_ |
| Regulatory approval and market entry: P(success) | +0.09 | -0.01 | _ASSUMED_ |
| peak market share | +0.08 | +0.53 | _ASSUMED_ |
| exit revenue multiple | +0.04 | +0.21 | _ASSUMED_ |
| Strain or trait development: cost (USD) | -0.02 | +0.01 | _ASSUMED_ |
| Field trials or pilot production: duration (y) | +0.02 | -0.04 | _ASSUMED_ |

**Where the spread comes from:** 51% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 49% is uncertainty about the inputs: 12.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **13.1% from assumptions**, and 23.1% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Gene-edited climate-resilient crops

**Priority 0.52** (importance 0.39, urgency 0.80) | evidence confidence 88% | 59 works | 20 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.6

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.21 | _measured_ — 75th-percentile normalised citation impact 0.64x expected across 59 works with citation data, against other tracks' works only; 40 compared within their own discipline and 19 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.39 | _measured_ — publication volume CAGR +10% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: agronomy, genetics, plant science |
| importance | evidence | 0.70 | _measured_ — mean publication-status weight 0.70 across 59 works, from 20 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 59 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 17 independent groups published in the last 12 months |
| urgency | acceleration | 0.46 | _measured_ — 11 works in the last 12 months vs historical mean 9.8/year (ratio 1.12) |
| urgency | industry_entry | 1.00 | _measured_ — 83% of works have at least one industry-affiliated author |

### Path to market

Sector: **Agriculture, food and bio-production**. Nominally **8.1 years** and **$102M** to market if every stage is survived, with cumulative technical success of **24.8%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Field trials or pilot production | 4-7 | PERT(2, 4, 7) | PERT(5e+06, 2e+07, 7e+07) | Beta~(0.25, 0.45, 0.65) | _ASSUMED_ |
| Regulatory approval and market entry | 7-9 | PERT(1.5, 3.5, 8) | PERT(1e+07, 5e+07, 2.5e+08) | Beta~(0.35, 0.55, 0.75) | _ASSUMED_ |

**Constraints**

- _high_ **regulatory** — Novel food and gene-edited crop rules differ sharply by jurisdiction; approval in one market implies little about another.
- _high_ **seasonality** — Field validation is paced by growing seasons and cannot be accelerated with capital.
- _high_ **margin** — Competing against commodity agricultural inputs leaves little pricing headroom.
- _medium_ **consumer_acceptance** — Public acceptance of edited or fermented products can gate adoption independently of approval.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$55M | -$13M | $222M |
| Investor MOIC | 0.00x | 0.00x | 15.54x |
| Years to market _(successes)_ | 2.5 | 6.8 | 11.0 |
| Capital consumed _(successes)_ | $29M | $84M | $153M |
| Investor stake at exit _(successes)_ | 5.26% | 7.89% | 11.62% |

- Probability of reaching market: **27.6%**
- Expected investor multiple across all outcomes: **4.66x** (**16.90x** conditional on reaching market). Sampling error at this draw count: 4.57x to 4.75x (90% interval); the top 1% of draws contribute 12% of it
- Probability the project destroys value: **75.6%**
- Starting readiness simulated per draw: TRL 1 7%, TRL 2 8%, TRL 3 16%, TRL 4 15%, TRL 5 18%, TRL 6 16%, TRL 7 12%, TRL 8 6%, TRL 9 2%
- 1.6% of draws start with no development stage left and carry 10% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Strain or trait development: 15.7%
- Field trials or pilot production: 35.5%
- Regulatory approval and market entry: 21.2%
- reached market: 27.6%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Field trials or pilot production: cost (USD) | -0.18 | -0.05 | _ASSUMED_ |
| Regulatory approval and market entry: cost (USD) | -0.11 | -0.15 | _ASSUMED_ |
| TAM (USD) | +0.09 | +0.64 | _ASSUMED_ |
| Regulatory approval and market entry: P(success) | +0.08 | -0.01 | _ASSUMED_ |
| peak market share | +0.07 | +0.53 | _ASSUMED_ |
| starting TRL | -0.04 | +0.32 | _measured_ |
| exit revenue multiple | +0.04 | +0.21 | _ASSUMED_ |
| Strain or trait development: cost (USD) | -0.04 | +0.01 | _ASSUMED_ |

**Where the spread comes from:** 50% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 50% is uncertainty about the inputs: 13.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **11.7% from assumptions**, and 24.6% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.

## Gut microbiome therapeutics

**Priority 0.50** (importance 0.39, urgency 0.75) | evidence confidence 88% | 62 works | 22 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 4.7

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.28 | _measured_ — 75th-percentile normalised citation impact 0.96x expected across 62 works with citation data, against other tracks' works only; 57 compared within their own discipline and 5 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.27 | _measured_ — publication volume CAGR -9% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: immunology, medicine, microbiology |
| importance | evidence | 0.70 | _measured_ — mean publication-status weight 0.70 across 62 works, from 22 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 62 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.99 | _measured_ — 17 independent groups published in the last 12 months |
| urgency | acceleration | 0.30 | _measured_ — 8 works in the last 12 months vs historical mean 10.3/year (ratio 0.77) |
| urgency | industry_entry | 1.00 | _measured_ — 84% of works have at least one industry-affiliated author |

### Path to market

Sector: **Therapeutics and biologics**. Nominally **10.8 years** and **$458M** to market if every stage is survived, with cumulative technical success of **3.8%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Preclinical | 3-6 | PERT(1, 2.5, 5) | PERT(3e+06, 1e+07, 3e+07) | Beta~(0.30, 0.45, 0.65) | _ASSUMED_ |
| Phase I | 6-7 | PERT(1, 1.7, 3) | PERT(8e+06, 2.5e+07, 6e+07) | Beta~(0.45, 0.52, 0.60) | _sourced_ |
| Phase II | 7-8 | PERT(1.5, 2.5, 4.5) | PERT(2e+07, 6e+07, 1.5e+08) | Beta~(0.24, 0.29, 0.35) | _sourced_ |
| Phase III and filing | 8-9 | PERT(2, 3.5, 6) | PERT(1e+08, 3e+08, 8e+08) | Beta~(0.45, 0.55, 0.65) | _sourced_ |

**Constraints**

- _high_ **regulatory** — Approval is a hard gate with no partial credit; timelines are set by the agency, not by the programme.
- _high_ **capital** — Phase III spend is concentrated and non-recoverable if the readout misses.
- _high_ **reimbursement** — Approval does not imply payment; payer coverage decisions can gate revenue for years.
- _medium_ **ip** — Effective exclusivity is the patent term minus development time, so slow programmes earn less even when they succeed.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$117M | -$22M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 1.08x |
| Years to market _(successes)_ | 0.0 | 5.3 | 10.7 |
| Capital consumed _(successes)_ | $0 | $358M | $585M |
| Investor stake at exit _(successes)_ | 1.89% | 3.69% | 15.00% |

- Probability of reaching market: **10.8%**
- Expected investor multiple across all outcomes: **1.25x** (**11.57x** conditional on reaching market). Sampling error at this draw count: 1.18x to 1.32x (90% interval); the top 1% of draws contribute 38% of it
- Probability the project destroys value: **94.4%**
- Starting readiness simulated per draw: TRL 1 8%, TRL 2 10%, TRL 3 11%, TRL 4 14%, TRL 5 19%, TRL 6 15%, TRL 7 15%, TRL 8 6%, TRL 9 2%
- 1.9% of draws start with no development stage left and carry 45% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Discovery and target validation: 11.5%
- Preclinical: 27.3%
- Phase I: 18.3%
- Phase II: 24.6%
- Phase III and filing: 7.4%
- reached market: 10.8%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.44 | +0.43 | _measured_ |
| Preclinical: cost (USD) | -0.11 | -0.01 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.55 | _ASSUMED_ |
| peak market share | +0.06 | +0.42 | _ASSUMED_ |
| Phase II: cost (USD) | -0.06 | -0.03 | _sourced_ |
| Preclinical: P(success) | -0.06 | -0.04 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.05 | -0.27 | _sourced_ |
| Phase I: cost (USD) | -0.05 | -0.04 | _sourced_ |

**Where the spread comes from:** 35% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 65% is uncertainty about the inputs: 24.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.6% from sourced figures, **3.8% from assumptions**, and 36.3% from combinations of inputs that no single one accounts for.

23 of 24 simulation inputs are flagged as placeholders awaiting a real number.

## Ambient-pressure high-Tc superconductivity

**Priority 0.50** (importance 0.44, urgency 0.60) | evidence confidence 88% | 32 works | 12 groups

Readiness: TRL 2 (Technology concept formulated); 80% credible range 1-6, mean 3.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.52 | _measured_ — 75th-percentile normalised citation impact 2.47x expected across 32 works with citation data, against other tracks' works only; 31 compared within their own discipline and 1 against the whole corpus at that age, which does not correct for field |
| importance | momentum | 0.35 | _measured_ — publication volume CAGR +4% fitted over 6 years |
| importance | breadth | 0.49 | _measured_ — 2 distinct disciplines represented: condensed matter physics, materials science |
| importance | evidence | 0.63 | _measured_ — mean publication-status weight 0.63 across 32 works, from 12 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 32 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 0.92 | _measured_ — 10 independent groups published in the last 12 months |
| urgency | acceleration | 0.29 | _measured_ — 4 works in the last 12 months vs historical mean 5.3/year (ratio 0.75) |
| urgency | industry_entry | 0.44 | _measured_ — 22% of works have at least one industry-affiliated author |

### Path to market

Sector: **Energy generation, storage and industrial hardware**. Nominally **14.2 years** and **$797M** to market if every stage is survived, with cumulative technical success of **9.7%**.

| Stage | TRL | Duration (y) | Cost | P(success) | Basis |
| --- | --- | --- | --- | --- | --- |
| Lab validation | 1-4 | PERT(2, 4, 8) | PERT(1e+06, 5e+06, 2e+07) | Beta~(0.25, 0.45, 0.65) | _ASSUMED_ |
| Pilot and scale-up | 4-7 | PERT(3, 5, 10) | PERT(2e+07, 8e+07, 3e+08) | Beta~(0.20, 0.35, 0.55) | _ASSUMED_ |
| First commercial plant | 7-9 | PERT(2, 4, 8) | PERT(1e+08, 5e+08, 2e+09) | Beta~(0.40, 0.60, 0.80) | _ASSUMED_ |

**Constraints**

- _high_ **capital** — First-of-a-kind plants are capital-intensive and rarely financeable on venture terms alone.
- _high_ **supply_chain** — Critical-mineral and specialised-equipment lead times can set the schedule regardless of technical readiness.
- _high_ **permitting** — Siting, grid interconnection and environmental permitting routinely add multi-year delays.
- _high_ **incumbent_cost** — Must beat an incumbent whose costs keep falling; the target moves during development.

### Investment case

_20,000 Monte Carlo draws, seed 20260918._

| Measure | P10 | P50 | P90 |
| --- | --- | --- | --- |
| Project NPV | -$327M | -$46M | -$2M |
| Investor MOIC | 0.00x | 0.00x | 3.15x |
| Years to market _(successes)_ | 0.0 | 8.4 | 14.7 |
| Capital consumed _(successes)_ | $0 | $660M | $1.20B |
| Investor stake at exit _(successes)_ | 0.93% | 2.01% | 15.00% |

- Probability of reaching market: **20.7%**
- Expected investor multiple across all outcomes: **1.46x** (**7.06x** conditional on reaching market). Sampling error at this draw count: 1.38x to 1.54x (90% interval); the top 1% of draws contribute 35% of it
- Probability the project destroys value: **93.8%**
- Starting readiness simulated per draw: TRL 1 20%, TRL 2 23%, TRL 3 13%, TRL 4 12%, TRL 5 11%, TRL 6 8%, TRL 7 6%, TRL 8 4%, TRL 9 2%
- 2.2% of draws start with no development stage left and carry 44% of the expected multiple. Entry terms are the same in every draw, so those are worlds in which an early-stage price buys a finished technology: if that share is large, narrow the TRL with an analyst override before reading the multiple

**Where the programme dies**

- Lab validation: 30.7%
- Pilot and scale-up: 36.6%
- First commercial plant: 12.4%
- reached market: 20.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| starting TRL | -0.38 | +0.16 | _measured_ |
| Pilot and scale-up: cost (USD) | -0.15 | -0.07 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.14 | -0.51 | _ASSUMED_ |
| TAM (USD) | +0.11 | +0.40 | _ASSUMED_ |
| Lab validation: cost (USD) | -0.10 | +0.01 | _ASSUMED_ |
| peak market share | +0.08 | +0.31 | _ASSUMED_ |
| exit revenue multiple | +0.05 | +0.17 | _ASSUMED_ |
| Lab validation: P(success) | -0.05 | -0.00 | _ASSUMED_ |

**Where the spread comes from:** 30% of the variance in project NPV is gate luck — which side of each gate the programme lands on — and would remain if every input were known exactly. The other 70% is uncertainty about the inputs: 27.7% from not knowing the starting readiness level, 0.0% from measured inputs, 0.0% from sourced figures, **9.5% from assumptions**, and 32.3% from combinations of inputs that no single one accounts for.

17 of 18 simulation inputs are flagged as placeholders awaiting a real number.
