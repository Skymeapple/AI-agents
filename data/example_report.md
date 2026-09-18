# Research portfolio review

_Generated 2026-09-18 by sciscout._

## Ranking

| # | Direction | Priority | Importance | Urgency | Confidence | TRL | P(market) | E[MOIC] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Quantum error correction | **0.68** | 0.56 | 0.91 | 88% | 3 | 10% | 0.37x |
| 2 | Protein structure and design models | **0.67** | 0.57 | 0.86 | 88% | 5 | 28% | 5.39x |
| 3 | Machine learning climate emulators | **0.66** | 0.56 | 0.86 | 88% | 5 | 28% | 5.39x |
| 4 | Solid-state sodium batteries | **0.65** | 0.50 | 0.95 | 88% | 6 | 21% | 0.70x |
| 5 | Direct air carbon capture | **0.63** | 0.48 | 0.94 | 88% | 5 | 21% | 0.70x |
| 6 | Perovskite tandem photovoltaics | **0.62** | 0.48 | 0.91 | 88% | 5 | 21% | 0.70x |
| 7 | Neuromorphic photonic computing | **0.61** | 0.52 | 0.79 | 88% | 3 | 10% | 0.37x |
| 8 | Green hydrogen electrolysis catalysts | **0.61** | 0.47 | 0.88 | 88% | 3 | 9% | 0.40x |
| 9 | Targeted protein degradation | **0.60** | 0.47 | 0.87 | 88% | 6 | 8% | 0.52x |
| 10 | CRISPR base and prime editing | **0.60** | 0.47 | 0.84 | 88% | 6 | 8% | 0.52x |
| 11 | CAR-T for solid tumours | **0.58** | 0.45 | 0.87 | 88% | 7 | 16% | 0.84x |
| 12 | Precision fermentation proteins | **0.56** | 0.45 | 0.78 | 88% | 5 | 25% | 3.74x |
| 13 | Gene-edited climate-resilient crops | **0.56** | 0.44 | 0.80 | 88% | 5 | 25% | 3.74x |
| 14 | Topological qubits | **0.55** | 0.43 | 0.80 | 88% | 2 | 10% | 0.37x |
| 15 | Gut microbiome therapeutics | **0.52** | 0.41 | 0.75 | 88% | 5 | 4% | 0.33x |
| 16 | Ambient-pressure high-Tc superconductivity | **0.50** | 0.44 | 0.60 | 88% | 2 | 9% | 0.40x |

Priority combines importance and urgency geometrically, so a direction cannot rank highly on one axis alone. Confidence is the share of scoring inputs that were measured rather than assumed — read it alongside priority, because a confident 0.5 is a better basis for a decision than an unfounded 0.8.

## Quantum error correction

**Priority 0.68** (importance 0.56, urgency 0.91) | evidence confidence 88% | 84 works | 20 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 2-7, mean 4.4

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.41 | _measured_ — 75th-percentile normalised citation impact 1.67x expected across 84 works with citation data |
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
| Project NPV | -$186M | -$16M | -$5M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 9.7 | 11.7 | 13.9 |
| Capital consumed _(successes)_ | $433M | $733M | $1.12B |
| Investor stake at exit _(successes)_ | 1.05% | 1.75% | 2.87% |

- Probability of reaching market: **9.6%**
- Expected investor multiple across all outcomes: **0.37x** (**3.85x** conditional on reaching market)
- Probability the project destroys value: **98.4%**

**Where the programme dies**

- Device demonstration: 58.9%
- Process integration and yield: 26.1%
- Qualification and ramp: 5.3%
- reached market: 9.6%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Device demonstration: cost (USD) | -0.35 | -0.02 | _ASSUMED_ |
| Device demonstration: P(success) | -0.12 | -0.03 | _ASSUMED_ |
| Process integration and yield: cost (USD) | -0.11 | -0.27 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.49 | _ASSUMED_ |
| Device demonstration: duration (y) | +0.05 | +0.13 | _ASSUMED_ |
| peak market share | +0.05 | +0.43 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.04 | -0.52 | _ASSUMED_ |
| discount rate | +0.03 | +0.06 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## Protein structure and design models

**Priority 0.67** (importance 0.57, urgency 0.86) | evidence confidence 88% | 109 works | 24 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.40 | _measured_ — 75th-percentile normalised citation impact 1.60x expected across 109 works with citation data |
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
| Project NPV | -$46M | -$12M | $368M |
| Investor MOIC | 0.00x | 0.00x | 17.97x |
| Years to market _(successes)_ | 2.0 | 2.7 | 3.6 |
| Capital consumed _(successes)_ | $26M | $49M | $80M |
| Investor stake at exit _(successes)_ | 7.20% | 9.65% | 12.32% |

- Probability of reaching market: **27.5%**
- Expected investor multiple across all outcomes: **5.39x** (**19.58x** conditional on reaching market)
- Probability the project destroys value: **73.2%**

**Where the programme dies**

- Productisation: 45.3%
- Scale and distribution: 27.2%
- reached market: 27.5%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Productisation: cost (USD) | -0.27 | -0.01 | _ASSUMED_ |
| Scale and distribution: P(success) | +0.10 | +0.02 | _ASSUMED_ |
| Scale and distribution: cost (USD) | -0.09 | -0.06 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.71 | _ASSUMED_ |
| peak market share | +0.04 | +0.58 | _ASSUMED_ |
| Productisation: duration (y) | +0.02 | -0.04 | _ASSUMED_ |
| gross margin | +0.02 | +0.07 | _ASSUMED_ |
| exit revenue multiple | +0.01 | +0.24 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Machine learning climate emulators

**Priority 0.66** (importance 0.56, urgency 0.86) | evidence confidence 88% | 85 works | 22 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.3

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.37 | _measured_ — 75th-percentile normalised citation impact 1.44x expected across 85 works with citation data |
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
| Project NPV | -$46M | -$12M | $368M |
| Investor MOIC | 0.00x | 0.00x | 17.97x |
| Years to market _(successes)_ | 2.0 | 2.7 | 3.6 |
| Capital consumed _(successes)_ | $26M | $49M | $80M |
| Investor stake at exit _(successes)_ | 7.20% | 9.65% | 12.32% |

- Probability of reaching market: **27.5%**
- Expected investor multiple across all outcomes: **5.39x** (**19.58x** conditional on reaching market)
- Probability the project destroys value: **73.2%**

**Where the programme dies**

- Productisation: 45.3%
- Scale and distribution: 27.2%
- reached market: 27.5%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Productisation: cost (USD) | -0.27 | -0.01 | _ASSUMED_ |
| Scale and distribution: P(success) | +0.10 | +0.02 | _ASSUMED_ |
| Scale and distribution: cost (USD) | -0.09 | -0.06 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.71 | _ASSUMED_ |
| peak market share | +0.04 | +0.58 | _ASSUMED_ |
| Productisation: duration (y) | +0.02 | -0.04 | _ASSUMED_ |
| gross margin | +0.02 | +0.07 | _ASSUMED_ |
| exit revenue multiple | +0.01 | +0.24 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Solid-state sodium batteries

**Priority 0.65** (importance 0.50, urgency 0.95) | evidence confidence 88% | 89 works | 22 groups

Readiness: TRL 6 (Demonstrated in relevant environment); 80% credible range 3-7, mean 5.4

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.36 | _measured_ — 75th-percentile normalised citation impact 1.38x expected across 89 works with citation data |
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
| Project NPV | -$362M | -$91M | -$32M |
| Investor MOIC | 0.00x | 0.00x | 2.09x |
| Years to market _(successes)_ | 7.7 | 9.8 | 12.1 |
| Capital consumed _(successes)_ | $376M | $742M | $1.26B |
| Investor stake at exit _(successes)_ | 0.75% | 1.35% | 2.83% |

- Probability of reaching market: **20.9%**
- Expected investor multiple across all outcomes: **0.70x** (**3.36x** conditional on reaching market)
- Probability the project destroys value: **96.2%**

**Where the programme dies**

- Pilot and scale-up: 64.7%
- First commercial plant: 14.4%
- reached market: 20.9%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Pilot and scale-up: cost (USD) | -0.49 | -0.15 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.17 | -0.65 | _ASSUMED_ |
| TAM (USD) | +0.13 | +0.48 | _ASSUMED_ |
| peak market share | +0.10 | +0.38 | _ASSUMED_ |
| Pilot and scale-up: duration (y) | +0.09 | +0.10 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.08 | -0.00 | _ASSUMED_ |
| discount rate | +0.05 | +0.04 | _ASSUMED_ |
| exit revenue multiple | +0.05 | +0.19 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Direct air carbon capture

**Priority 0.63** (importance 0.48, urgency 0.94) | evidence confidence 88% | 68 works | 18 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.40 | _measured_ — 75th-percentile normalised citation impact 1.60x expected across 68 works with citation data |
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
| Project NPV | -$362M | -$91M | -$32M |
| Investor MOIC | 0.00x | 0.00x | 2.09x |
| Years to market _(successes)_ | 7.7 | 9.8 | 12.1 |
| Capital consumed _(successes)_ | $376M | $742M | $1.26B |
| Investor stake at exit _(successes)_ | 0.75% | 1.35% | 2.83% |

- Probability of reaching market: **20.9%**
- Expected investor multiple across all outcomes: **0.70x** (**3.36x** conditional on reaching market)
- Probability the project destroys value: **96.2%**

**Where the programme dies**

- Pilot and scale-up: 64.7%
- First commercial plant: 14.4%
- reached market: 20.9%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Pilot and scale-up: cost (USD) | -0.49 | -0.15 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.17 | -0.65 | _ASSUMED_ |
| TAM (USD) | +0.13 | +0.48 | _ASSUMED_ |
| peak market share | +0.10 | +0.38 | _ASSUMED_ |
| Pilot and scale-up: duration (y) | +0.09 | +0.10 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.08 | -0.00 | _ASSUMED_ |
| discount rate | +0.05 | +0.04 | _ASSUMED_ |
| exit revenue multiple | +0.05 | +0.19 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Perovskite tandem photovoltaics

**Priority 0.62** (importance 0.48, urgency 0.91) | evidence confidence 88% | 88 works | 24 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 4.7

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.44 | _measured_ — 75th-percentile normalised citation impact 1.90x expected across 88 works with citation data |
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
| Project NPV | -$362M | -$91M | -$32M |
| Investor MOIC | 0.00x | 0.00x | 2.09x |
| Years to market _(successes)_ | 7.7 | 9.8 | 12.1 |
| Capital consumed _(successes)_ | $376M | $742M | $1.26B |
| Investor stake at exit _(successes)_ | 0.75% | 1.35% | 2.83% |

- Probability of reaching market: **20.9%**
- Expected investor multiple across all outcomes: **0.70x** (**3.36x** conditional on reaching market)
- Probability the project destroys value: **96.2%**

**Where the programme dies**

- Pilot and scale-up: 64.7%
- First commercial plant: 14.4%
- reached market: 20.9%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Pilot and scale-up: cost (USD) | -0.49 | -0.15 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.17 | -0.65 | _ASSUMED_ |
| TAM (USD) | +0.13 | +0.48 | _ASSUMED_ |
| peak market share | +0.10 | +0.38 | _ASSUMED_ |
| Pilot and scale-up: duration (y) | +0.09 | +0.10 | _ASSUMED_ |
| Pilot and scale-up: P(success) | -0.08 | -0.00 | _ASSUMED_ |
| discount rate | +0.05 | +0.04 | _ASSUMED_ |
| exit revenue multiple | +0.05 | +0.19 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Neuromorphic photonic computing

**Priority 0.61** (importance 0.52, urgency 0.79) | evidence confidence 88% | 39 works | 16 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 1-6, mean 4.0

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.29 | _measured_ — 75th-percentile normalised citation impact 1.00x expected across 39 works with citation data |
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
| Project NPV | -$186M | -$16M | -$5M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 9.7 | 11.7 | 13.9 |
| Capital consumed _(successes)_ | $433M | $733M | $1.12B |
| Investor stake at exit _(successes)_ | 1.05% | 1.75% | 2.87% |

- Probability of reaching market: **9.6%**
- Expected investor multiple across all outcomes: **0.37x** (**3.85x** conditional on reaching market)
- Probability the project destroys value: **98.4%**

**Where the programme dies**

- Device demonstration: 58.9%
- Process integration and yield: 26.1%
- Qualification and ramp: 5.3%
- reached market: 9.6%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Device demonstration: cost (USD) | -0.35 | -0.02 | _ASSUMED_ |
| Device demonstration: P(success) | -0.12 | -0.03 | _ASSUMED_ |
| Process integration and yield: cost (USD) | -0.11 | -0.27 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.49 | _ASSUMED_ |
| Device demonstration: duration (y) | +0.05 | +0.13 | _ASSUMED_ |
| peak market share | +0.05 | +0.43 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.04 | -0.52 | _ASSUMED_ |
| discount rate | +0.03 | +0.06 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## Green hydrogen electrolysis catalysts

**Priority 0.61** (importance 0.47, urgency 0.88) | evidence confidence 88% | 80 works | 20 groups

Readiness: TRL 3 (Experimental proof of concept); 80% credible range 1-6, mean 4.2

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.43 | _measured_ — 75th-percentile normalised citation impact 1.79x expected across 80 works with citation data |
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
| Project NPV | -$125M | -$9M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 11.7 | 14.1 | 16.8 |
| Capital consumed _(successes)_ | $392M | $749M | $1.27B |
| Investor stake at exit _(successes)_ | 1.02% | 1.82% | 3.24% |

- Probability of reaching market: **9.2%**
- Expected investor multiple across all outcomes: **0.40x** (**4.33x** conditional on reaching market)
- Probability the project destroys value: **98.3%**

**Where the programme dies**

- Lab validation: 54.9%
- Pilot and scale-up: 29.2%
- First commercial plant: 6.7%
- reached market: 9.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Lab validation: cost (USD) | -0.30 | +0.02 | _ASSUMED_ |
| Lab validation: P(success) | -0.12 | +0.01 | _ASSUMED_ |
| Pilot and scale-up: cost (USD) | -0.09 | -0.14 | _ASSUMED_ |
| Lab validation: duration (y) | +0.07 | +0.10 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.48 | _ASSUMED_ |
| discount rate | +0.06 | +0.12 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.05 | -0.63 | _ASSUMED_ |
| peak market share | +0.04 | +0.37 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## Targeted protein degradation

**Priority 0.60** (importance 0.47, urgency 0.87) | evidence confidence 88% | 99 works | 28 groups

Readiness: TRL 6 (Demonstrated in relevant environment); 80% credible range 4-7, mean 5.2

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.41 | _measured_ — 75th-percentile normalised citation impact 1.69x expected across 99 works with citation data |
| importance | momentum | 0.45 | _measured_ — publication volume CAGR +19% fitted over 6 years |
| importance | breadth | 0.63 | _measured_ — 3 distinct disciplines represented: biochemistry, medicine, pharmacology |
| importance | evidence | 0.69 | _measured_ — mean publication-status weight 0.69 across 99 works, from 28 independent groups |
| importance | foundationality | 0.00 | _ASSUMED_ — 0 of 99 works use platform/method/enabling language. This is a lexical proxy: properly, foundationality is how much downstream work builds on a result, which needs citation-graph data this corpus does not carry |
| urgency | competition | 1.00 | _measured_ — 27 independent groups published in the last 12 months |
| urgency | acceleration | 0.62 | _measured_ — 24 works in the last 12 months vs historical mean 16.5/year (ratio 1.45) |
| urgency | industry_entry | 1.00 | _measured_ — 95% of works have at least one industry-affiliated author |

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
| Project NPV | -$124M | -$40M | -$16M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 6.9 | 8.1 | 9.5 |
| Capital consumed _(successes)_ | $291M | $433M | $636M |
| Investor stake at exit _(successes)_ | 1.92% | 3.00% | 4.47% |

- Probability of reaching market: **8.2%**
- Expected investor multiple across all outcomes: **0.52x** (**6.36x** conditional on reaching market)
- Probability the project destroys value: **97.3%**

**Where the programme dies**

- Phase I: 48.3%
- Phase II: 36.4%
- Phase III and filing: 7.1%
- reached market: 8.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Phase I: cost (USD) | -0.30 | -0.07 | _sourced_ |
| Phase II: cost (USD) | -0.13 | -0.08 | _sourced_ |
| TAM (USD) | +0.07 | +0.64 | _ASSUMED_ |
| peak market share | +0.06 | +0.52 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.04 | -0.42 | _sourced_ |
| Phase I: P(success) | -0.04 | -0.03 | _sourced_ |
| discount rate | +0.03 | -0.03 | _ASSUMED_ |
| Phase I: duration (y) | +0.03 | +0.04 | _sourced_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 91% to sourced figures, and **9% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## CRISPR base and prime editing

**Priority 0.60** (importance 0.47, urgency 0.84) | evidence confidence 88% | 102 works | 26 groups

Readiness: TRL 6 (Demonstrated in relevant environment); 80% credible range 3-7, mean 5.1

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.44 | _measured_ — 75th-percentile normalised citation impact 1.89x expected across 102 works with citation data |
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
| Project NPV | -$124M | -$40M | -$16M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 6.9 | 8.1 | 9.5 |
| Capital consumed _(successes)_ | $291M | $433M | $636M |
| Investor stake at exit _(successes)_ | 1.92% | 3.00% | 4.47% |

- Probability of reaching market: **8.2%**
- Expected investor multiple across all outcomes: **0.52x** (**6.36x** conditional on reaching market)
- Probability the project destroys value: **97.3%**

**Where the programme dies**

- Phase I: 48.3%
- Phase II: 36.4%
- Phase III and filing: 7.1%
- reached market: 8.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Phase I: cost (USD) | -0.30 | -0.07 | _sourced_ |
| Phase II: cost (USD) | -0.13 | -0.08 | _sourced_ |
| TAM (USD) | +0.07 | +0.64 | _ASSUMED_ |
| peak market share | +0.06 | +0.52 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.04 | -0.42 | _sourced_ |
| Phase I: P(success) | -0.04 | -0.03 | _sourced_ |
| discount rate | +0.03 | -0.03 | _ASSUMED_ |
| Phase I: duration (y) | +0.03 | +0.04 | _sourced_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 91% to sourced figures, and **9% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## CAR-T for solid tumours

**Priority 0.58** (importance 0.45, urgency 0.87) | evidence confidence 88% | 81 works | 24 groups

Readiness: TRL 7 (Prototype demonstrated in operational environment); 80% credible range 2-7, mean 4.9

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.41 | _measured_ — 75th-percentile normalised citation impact 1.70x expected across 81 works with citation data |
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
| Project NPV | -$235M | -$63M | -$27M |
| Investor MOIC | 0.00x | 0.00x | 2.41x |
| Years to market _(successes)_ | 5.2 | 6.3 | 7.6 |
| Capital consumed _(successes)_ | $257M | $403M | $606M |
| Investor stake at exit _(successes)_ | 1.51% | 2.36% | 3.76% |

- Probability of reaching market: **16.2%**
- Expected investor multiple across all outcomes: **0.84x** (**5.21x** conditional on reaching market)
- Probability the project destroys value: **93.7%**

**Where the programme dies**

- Phase II: 70.1%
- Phase III and filing: 13.7%
- reached market: 16.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Phase II: cost (USD) | -0.53 | -0.09 | _sourced_ |
| TAM (USD) | +0.13 | +0.64 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.11 | -0.39 | _sourced_ |
| peak market share | +0.10 | +0.50 | _ASSUMED_ |
| Phase II: duration (y) | +0.05 | +0.01 | _sourced_ |
| exit revenue multiple | +0.05 | +0.21 | _ASSUMED_ |
| gross margin | +0.03 | +0.06 | _ASSUMED_ |
| ramp to peak (y) | -0.03 | -0.11 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 90% to sourced figures, and **10% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Precision fermentation proteins

**Priority 0.56** (importance 0.45, urgency 0.78) | evidence confidence 88% | 61 works | 18 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 5.1

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.38 | _measured_ — 75th-percentile normalised citation impact 1.50x expected across 61 works with citation data |
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
| Project NPV | -$56M | -$19M | $148M |
| Investor MOIC | 0.00x | 0.00x | 12.68x |
| Years to market _(successes)_ | 6.1 | 8.0 | 10.0 |
| Capital consumed _(successes)_ | $52M | $96M | $162M |
| Investor stake at exit _(successes)_ | 4.65% | 7.12% | 10.49% |

- Probability of reaching market: **24.8%**
- Expected investor multiple across all outcomes: **3.74x** (**15.06x** conditional on reaching market)
- Probability the project destroys value: **78.8%**

**Where the programme dies**

- Field trials or pilot production: 55.0%
- Regulatory approval and market entry: 20.2%
- reached market: 24.8%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Field trials or pilot production: cost (USD) | -0.38 | -0.08 | _ASSUMED_ |
| Regulatory approval and market entry: cost (USD) | -0.09 | -0.16 | _ASSUMED_ |
| TAM (USD) | +0.09 | +0.69 | _ASSUMED_ |
| Regulatory approval and market entry: P(success) | +0.08 | -0.01 | _ASSUMED_ |
| peak market share | +0.08 | +0.57 | _ASSUMED_ |
| Field trials or pilot production: duration (y) | +0.05 | -0.08 | _ASSUMED_ |
| exit revenue multiple | +0.04 | +0.24 | _ASSUMED_ |
| discount rate | +0.03 | -0.19 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Gene-edited climate-resilient crops

**Priority 0.56** (importance 0.44, urgency 0.80) | evidence confidence 88% | 59 works | 20 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 1-6, mean 4.6

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.36 | _measured_ — 75th-percentile normalised citation impact 1.35x expected across 59 works with citation data |
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
| Project NPV | -$56M | -$19M | $148M |
| Investor MOIC | 0.00x | 0.00x | 12.68x |
| Years to market _(successes)_ | 6.1 | 8.0 | 10.0 |
| Capital consumed _(successes)_ | $52M | $96M | $162M |
| Investor stake at exit _(successes)_ | 4.65% | 7.12% | 10.49% |

- Probability of reaching market: **24.8%**
- Expected investor multiple across all outcomes: **3.74x** (**15.06x** conditional on reaching market)
- Probability the project destroys value: **78.8%**

**Where the programme dies**

- Field trials or pilot production: 55.0%
- Regulatory approval and market entry: 20.2%
- reached market: 24.8%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Field trials or pilot production: cost (USD) | -0.38 | -0.08 | _ASSUMED_ |
| Regulatory approval and market entry: cost (USD) | -0.09 | -0.16 | _ASSUMED_ |
| TAM (USD) | +0.09 | +0.69 | _ASSUMED_ |
| Regulatory approval and market entry: P(success) | +0.08 | -0.01 | _ASSUMED_ |
| peak market share | +0.08 | +0.57 | _ASSUMED_ |
| Field trials or pilot production: duration (y) | +0.05 | -0.08 | _ASSUMED_ |
| exit revenue multiple | +0.04 | +0.24 | _ASSUMED_ |
| discount rate | +0.03 | -0.19 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

14 of 14 simulation inputs are flagged as placeholders awaiting a real number.

## Topological qubits

**Priority 0.55** (importance 0.43, urgency 0.80) | evidence confidence 88% | 36 works | 14 groups

Readiness: TRL 2 (Technology concept formulated); 80% credible range 1-6, mean 3.9

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.40 | _measured_ — 75th-percentile normalised citation impact 1.62x expected across 36 works with citation data |
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
| Project NPV | -$186M | -$16M | -$5M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 9.7 | 11.7 | 13.9 |
| Capital consumed _(successes)_ | $433M | $733M | $1.12B |
| Investor stake at exit _(successes)_ | 1.05% | 1.75% | 2.87% |

- Probability of reaching market: **9.6%**
- Expected investor multiple across all outcomes: **0.37x** (**3.85x** conditional on reaching market)
- Probability the project destroys value: **98.4%**

**Where the programme dies**

- Device demonstration: 58.9%
- Process integration and yield: 26.1%
- Qualification and ramp: 5.3%
- reached market: 9.6%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Device demonstration: cost (USD) | -0.35 | -0.02 | _ASSUMED_ |
| Device demonstration: P(success) | -0.12 | -0.03 | _ASSUMED_ |
| Process integration and yield: cost (USD) | -0.11 | -0.27 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.49 | _ASSUMED_ |
| Device demonstration: duration (y) | +0.05 | +0.13 | _ASSUMED_ |
| peak market share | +0.05 | +0.43 | _ASSUMED_ |
| Qualification and ramp: cost (USD) | -0.04 | -0.52 | _ASSUMED_ |
| discount rate | +0.03 | +0.06 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.

## Gut microbiome therapeutics

**Priority 0.52** (importance 0.41, urgency 0.75) | evidence confidence 88% | 62 works | 22 groups

Readiness: TRL 5 (Validated in relevant environment); 80% credible range 2-7, mean 4.7

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.36 | _measured_ — 75th-percentile normalised citation impact 1.38x expected across 62 works with citation data |
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
| Project NPV | -$72M | -$16M | -$6M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 9.3 | 10.7 | 12.2 |
| Capital consumed _(successes)_ | $299M | $441M | $652M |
| Investor stake at exit _(successes)_ | 2.51% | 4.23% | 6.24% |

- Probability of reaching market: **3.7%**
- Expected investor multiple across all outcomes: **0.33x** (**8.97x** conditional on reaching market)
- Probability the project destroys value: **98.8%**

**Where the programme dies**

- Preclinical: 54.5%
- Phase I: 21.9%
- Phase II: 16.7%
- Phase III and filing: 3.2%
- reached market: 3.7%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Preclinical: cost (USD) | -0.34 | -0.05 | _ASSUMED_ |
| Preclinical: P(success) | -0.11 | -0.04 | _ASSUMED_ |
| Phase I: cost (USD) | -0.06 | -0.07 | _sourced_ |
| Preclinical: duration (y) | +0.06 | +0.07 | _ASSUMED_ |
| peak market share | +0.03 | +0.52 | _ASSUMED_ |
| TAM (USD) | +0.03 | +0.66 | _ASSUMED_ |
| Phase III and filing: cost (USD) | -0.02 | -0.43 | _sourced_ |
| exit revenue multiple | +0.02 | +0.17 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 4% to sourced figures, and **96% to assumptions**.

20 of 20 simulation inputs are flagged as placeholders awaiting a real number.

## Ambient-pressure high-Tc superconductivity

**Priority 0.50** (importance 0.44, urgency 0.60) | evidence confidence 88% | 32 works | 12 groups

Readiness: TRL 2 (Technology concept formulated); 80% credible range 1-6, mean 3.5

### Why it scores this way

| Axis | Sub-score | Value | Basis |
| --- | --- | --- | --- |
| importance | impact | 0.52 | _measured_ — 75th-percentile normalised citation impact 2.45x expected across 32 works with citation data |
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
| Project NPV | -$125M | -$9M | -$3M |
| Investor MOIC | 0.00x | 0.00x | 0.00x |
| Years to market _(successes)_ | 11.7 | 14.1 | 16.8 |
| Capital consumed _(successes)_ | $392M | $749M | $1.27B |
| Investor stake at exit _(successes)_ | 1.02% | 1.82% | 3.24% |

- Probability of reaching market: **9.2%**
- Expected investor multiple across all outcomes: **0.40x** (**4.33x** conditional on reaching market)
- Probability the project destroys value: **98.3%**

**Where the programme dies**

- Lab validation: 54.9%
- Pilot and scale-up: 29.2%
- First commercial plant: 6.7%
- reached market: 9.2%

**What drives the outcome** (rank correlation with project NPV)

| Input | All draws | Conditional on success | Provenance |
| --- | --- | --- | --- |
| Lab validation: cost (USD) | -0.30 | +0.02 | _ASSUMED_ |
| Lab validation: P(success) | -0.12 | +0.01 | _ASSUMED_ |
| Pilot and scale-up: cost (USD) | -0.09 | -0.14 | _ASSUMED_ |
| Lab validation: duration (y) | +0.07 | +0.10 | _ASSUMED_ |
| TAM (USD) | +0.06 | +0.48 | _ASSUMED_ |
| discount rate | +0.06 | +0.12 | _ASSUMED_ |
| First commercial plant: cost (USD) | -0.05 | -0.63 | _ASSUMED_ |
| peak market share | +0.04 | +0.37 | _ASSUMED_ |

**Provenance of this forecast:** 0% of the spread traces to measured inputs, 0% to sourced figures, and **100% to assumptions**.

17 of 17 simulation inputs are flagged as placeholders awaiting a real number.
