# Ten world problems, simulated to 2050

A coupled model of ten global problems, stepped a year at a time from 2025 to 2050 across 4,000 random draws. Baseline rates of change are fitted from global data (World Bank, Our World in Data, NASA); levels with no open series are anchored on cited agency reports; the links between problems and the effect of each policy lever are assumptions.

**This is not a forecast.** Of the 43 uncertain inputs, 5 are fitted from data, 4 come from a cited source and 34 are judgement calls. The current-trajectory numbers say where measured trends lead if they persist. The lever and scenario numbers say what the assumptions imply, and are only as good as those assumptions.

## Where the world stands

| Problem | Indicator | Latest value | Grade | Basis |
| --- | --- | --- | --- | --- |
| Climate change | Global warming | 1.38 degC above pre-industrial | sourced | warming_2024: Indicators of Global Climate Change, human-induced warming in 2024 |
| Climate change | CO2 emissions | 43.3 GtCO2 per year | observed | co2_emissions in 2025 |
| War | Deaths from organised violence | 245 thousand per year | sourced | conflict_deaths: UCDP annual totals for state-based, non-state and one-sided violence |
| Extreme poverty | People under $3 a day | 830 million | observed | poverty_rate in 2024, carried forward to 2025 |
| Hunger | Undernourished people | 641 million | sourced | undernourishment_2025: State of Food Security and Nutrition in the World 2026, 645 million people |
| Pandemics and resistance | Deaths from drug-resistant infections | 1.22 million per year | sourced | amr_deaths_2021: GRAM study, Lancet 2024 |
| Biodiversity loss | Forest cover (habitat proxy) | 31.10 % of land | observed | forest_share in 2023, carried forward to 2025 |
| Water and sanitation | People without safe drinking water | 2,163 million | observed | safe_water_rate in 2024, carried forward to 2025 |
| Water and sanitation | People without safe sanitation | 3,410 million | observed | safe_sanitation_rate in 2024, carried forward to 2025 |
| Forced displacement | Forcibly displaced people | 117.8 million | sourced | displaced_total: UNHCR Global Trends headline totals |
| Inequality | Wealth share of the richest 10% | 75.0 % | sourced | top10_wealth_share_2025: World Inequality Report 2026 |
| Context | nuclear warheads 2026 | 12,187 warheads, world inventory in January 2026 | sourced | SIPRI Yearbook 2026 |
| Context | military spending 2025 | 2,887 billion US$ | sourced | SIPRI military expenditure database, April 2026 release |
| Context | state conflicts 2025 | 65 active conflicts involving a state | sourced | UCDP, highest count since 1946 |
| Context | warming 2100 current policies | 2.8 degC by 2100 | sourced | UNEP Emissions Gap Report 2026, current-policies projection |
| Context | lacking safe sanitation 2024 | 3.4 billion people | sourced | WHO/UNICEF Joint Monitoring Programme 2025 |
| Context | species at risk | 1,000,000 species threatened with extinction | sourced | IPBES Global Assessment 2019 |

People counts for the base year are the latest observed rate applied to the 2025 population, so they differ slightly from agency headline counts.

## Three scenarios in 2050

- **Current trajectory**: Recent trends continue and no lever moves.
- **Cooperation**: Every lever at 0.7 - a sustained, coordinated push on all fronts.
- **Fragmentation**: Every lever at -0.3 - effort is withdrawn as cooperation breaks down.

Median, with the 10th to 90th percentile range in brackets.

| Indicator | 2025 | Current trajectory | Cooperation | Fragmentation |
| --- | --- | --- | --- | --- |
| Global warming (degC above pre-industrial) | 1.38 | **2.04** (1.89 to 2.19) | **1.80** (1.70 to 1.91) | **2.19** (2.01 to 2.39) |
| CO2 emissions (GtCO2 per year) | 43.3 | **53.6** (48.7 to 59.1) | **22.1** (17.5 to 27.3) | **78.7** (69.2 to 89.2) |
| Deaths from organised violence (thousand per year) | 245 | **143** (77 to 262) | **58** (30 to 111) | **211** (111 to 393) |
| People under $3 a day (million) | 830 | **448** (321 to 613) | **165** (113 to 236) | **691** (486 to 980) |
| Undernourished people (million) | 641 | **925** (565 to 1,444) | **318** (183 to 504) | **1,482** (884 to 2,442) |
| Deaths from drug-resistant infections (million per year) | 1.22 | **1.94** (1.74 to 2.17) | **1.05** (0.85 to 1.30) | **2.53** (2.21 to 2.88) |
| Forest cover (habitat proxy) (% of land) | 31.10 | **30.02** (29.89 to 30.12) | **30.70** (30.49 to 30.92) | **29.73** (29.56 to 29.86) |
| People without safe drinking water (million) | 2,163 | **1,795** (1,678 to 1,944) | **1,047** (897 to 1,222) | **2,273** (2,065 to 2,528) |
| People without safe sanitation (million) | 3,410 | **2,421** (2,237 to 2,647) | **1,411** (1,205 to 1,655) | **3,065** (2,767 to 3,435) |
| Forcibly displaced people (million) | 117.8 | **194.6** (159.9 to 238.7) | **100.4** (76.3 to 127.1) | **259.8** (210.5 to 328.5) |
| AI oversight gap index (2025 = 1) (index) | 1.00 | **4.09** (1.78 to 11.09) | **0.98** (0.37 to 2.82) | **7.71** (3.23 to 20.93) |
| Wealth share of the richest 10% (%) | 75.0 | **75.0** (73.7 to 76.3) | **70.6** (68.8 to 72.4) | **76.9** (75.5 to 78.2) |

**Pandemics** are drawn as chance events. A pandemic on the scale of COVID-19 strikes at least once by 2050 in 41% of draws. Where one does, median deaths are 10 million on the current trajectory and 6 million under Cooperation.

## Where the model disagrees with published projections

- **Hunger in 2030.** The model has 669 million undernourished (611 to 727); FAO projects 510 to 520 million. The model is more pessimistic because the typical annual change over the last ten years is close to flat, and it then adds assumed climate and conflict pressure. Hunger is the result most sensitive to which years the trend is fitted on.
- **Warming, checked against the past.** Fed observed emissions for 1995-2025 (1,143 GtCO2), the warming equation gives 0.62 degC. The fitted NASA temperature trend over those years is 0.73 degC. The central setting therefore under-predicts by 16%, inside the sampled range but worth knowing.
- **Emissions.** Extending the recent trend has emissions still rising in 2050. Published current-policy pathways have them flat or falling, because they count policies not yet visible in the data.

## What each lever moves

Each lever is pushed to full extra effort on its own and compared draw by draw with the current trajectory. Cells are the median change in 2050; a dot is under half a percent.

| Lever | Global warming | Deaths from organised violence | People under $3 a day | Undernourished people | Deaths from drug-resistant infections | Cumulative pandemic deaths since 2025 | Forest cover (habitat proxy) | People without safe drinking water | People without safe sanitation | Forcibly displaced people | AI oversight gap index (2025 = 1) | Wealth share of the richest 10% | Indicators moved 5%+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reduce conflict | · | -55% | -16% | -29% | -1% | · | · | -8% | -8% | -54% | · | · | 6 |
| Tax and redistribute | · | -16% | -32% | -19% | · | · | · | -1% | -1% | -5% | · | -8% | 5 |
| Finance development | · | -20% | -54% | -55% | · | · | · | -1% | -1% | -6% | · | · | 4 |
| Cut emissions | -15% | -5% | -6% | -11% | · | · | · | -3% | -3% | -3% | · | · | 3 |
| Build water and sanitation | · | · | · | · | -9% | · | · | -47% | -47% | · | · | · | 3 |
| Invest in health security | · | · | · | · | -52% | -49% | · | · | · | · | · | · | 2 |
| Govern advanced AI | · | · | · | · | · | · | · | · | · | · | -87% | · | 1 |
| Protect ecosystems | · | · | · | · | · | · | +3% | · | · | · | · | · | 0 |

The breadth ranking is a property of the assumed links, not a finding about the world: a lever reaches other problems only through couplings whose sizes are judgement calls. What it does show is which claims about cross-cutting leverage the assumptions support and which they do not. Forest cover moves slowly, so the ecosystem lever falls under the 5% bar even though it reverses twenty-five years of loss.

## What each result rests on

Approximate share of the spread in each current-trajectory result traced to inputs of each grade, and the inputs that matter most. Chance covers conflict shocks and whether a pandemic strikes.

| Indicator | Observed | Sourced | Assumed | Chance and interaction | Main drivers |
| --- | --- | --- | --- | --- | --- |
| Global warming | 4% | 81% | 14% | 2% | tcre (sourced), nonco2_multiplier (assumed), emissions_trend (observed) |
| CO2 emissions | 99% | 0% | 0% | 0% | emissions_trend (observed) |
| Deaths from organised violence | 0% | 3% | 2% | 95% | hunger_trend (sourced) |
| People under $3 a day | 66% | 2% | 15% | 18% | poverty_trend (observed), climate_to_poverty (assumed), tcre (sourced) |
| Undernourished people | 0% | 70% | 15% | 15% | hunger_trend (sourced), climate_to_hunger (assumed), tcre (sourced) |
| Deaths from drug-resistant infections | 0% | 97% | 2% | 0% | amr_growth (sourced) |
| Cumulative pandemic deaths since 2025 | 0% | 2% | 1% | 97% | pandemic_annual_probability (sourced) |
| Forest cover (habitat proxy) | 2% | 6% | 90% | 1% | climate_to_forest (assumed), tcre (sourced), forest_trend (observed) |
| People without safe drinking water | 1% | 7% | 58% | 34% | climate_to_water (assumed), tcre (sourced), conflict_to_water (assumed) |
| People without safe sanitation | 22% | 5% | 42% | 30% | climate_to_water (assumed), sanitation_trend (observed), tcre (sourced) |
| Forcibly displaced people | 0% | 4% | 24% | 72% | displacement_return_rate (assumed), hunger_trend (sourced), conflict_to_displacement (assumed) |
| AI oversight gap index (2025 = 1) | 0% | 0% | 100% | 0% | ai_gap_growth (assumed) |
| Wealth share of the richest 10% | 0% | 0% | 100% | 0% | inequality_drift (assumed) |

## Limitations

- **The couplings are assumed.** 34 of the 43 inputs are judgement calls, including every link between problems and every lever effect. Change them in `worldsim/params.yaml` and rerun.
- **Trends are extrapolated.** A rate measured over ten or twenty years is carried for twenty-five. Nothing in the model knows about policies adopted but not yet visible in the data.
- **Two problems have no usable series.** The AI oversight gap and the wealth-share drift are placeholders that show a direction, and their results are 100% assumption.
- **Forest cover stands in for biodiversity.** It is the only global habitat series available here. It says nothing about species loss inside standing forest, or about oceans.
- **Global averages hide the problem.** Poverty, hunger and unsafe water are concentrated in fragile states; a world model cannot say where.
- **Levers have no cost.** The model says what an effort would move, not what it would take to mount.
- **Some history is recalled, not ingested.** Conflict and displacement totals before 2024 are approximate report figures flagged for review in `data/world/sourced.yaml`. They set the spread of the conflict process.
- **Nuclear risk is not simulated.** It is a low-probability catastrophe with no base rate to fit.

## Every uncertain input

| Input | Value | Grade | Basis |
| --- | --- | --- | --- |
| emissions_trend | 0.008489 | observed | emissions_trend: median annual log change; 10-year +0.0085, 20-year +0.0092, to 2025 |
| poverty_trend | -0.04151 | observed | poverty_trend: median annual log change; 10-year -0.0388, 20-year -0.0616, to 2024 |
| hunger_trend | -0.009832 | sourced | hunger_trend: median annual log change; 10-year -0.0058, 20-year -0.0358, to 2025; last point from undernourishment_2025: State of Food Security and Nutrition in the World 2026, 645 million people (https://www.fao.org/newsroom/detail/un-report--global-hunger-levels-ease-for-third-consecutive-year-as-regional-disparities-persist/en) |
| water_trend | -0.01803 | observed | water_trend: median annual log change; 10-year -0.0182, 20-year -0.0175, to 2024 |
| sanitation_trend | -0.02427 | observed | sanitation_trend: median annual log change; 10-year -0.0245, 20-year -0.0219, to 2024 |
| forest_trend | -0.001212 | observed | forest_trend: median annual log change; 10-year -0.0012, 20-year -0.0012, to 2023 |
| tcre | 0.45 | sourced | tcre: transient climate response to cumulative emissions, likely range and best estimate (IPCC AR6 WG1 Summary for Policymakers, D.1.1) |
| nonco2_multiplier | 1.2 | assumed | nonco2_multiplier: methane and other gases net of aerosol cooling add roughly a fifth on top of CO2 warming **needs review** |
| conflict_to_poverty | 0.01167 | assumed | conflict_to_poverty: war destroys livelihoods; a doubling of deaths slows poverty reduction by under one point a year **needs review** |
| climate_to_poverty | 0.02333 | assumed | climate_to_poverty: crop losses and disasters fall hardest on the poor **needs review** |
| inequality_to_poverty | 0.001333 | assumed | inequality_to_poverty: growth reaches the poor more slowly when gains concentrate **needs review** |
| poverty_to_hunger | 0.5 | assumed | poverty_to_hunger: hunger is largely a purchasing-power problem **needs review** |
| conflict_to_hunger | 0.01667 | assumed | conflict_to_hunger: most acute food crises are in conflict zones **needs review** |
| climate_to_hunger | 0.03333 | assumed | climate_to_hunger: heat and drought cut yields in the tropics **needs review** |
| hunger_to_conflict | 0.3333 | assumed | hunger_to_conflict: food price spikes are associated with unrest **needs review** |
| climate_to_conflict | 0.1167 | assumed | climate_to_conflict: literature finds a positive but contested temperature-conflict link **needs review** |
| inequality_to_conflict | 0.02333 | assumed | inequality_to_conflict: grievance channel; weakly evidenced at the global level **needs review** |
| conflict_to_water | 0.005833 | assumed | conflict_to_water: coverage in fragile states lags far behind **needs review** |
| climate_to_water | 0.01167 | assumed | climate_to_water: drought and salinisation stress supplies **needs review** |
| sanitation_to_amr | 0.1667 | assumed | sanitation_to_amr: fewer infections means less antibiotic use and less spread **needs review** |
| climate_to_forest | 0.0006667 | assumed | climate_to_forest: fire and dieback **needs review** |
| conflict_to_displacement | 0.6833 | assumed | conflict_to_displacement: displacement rises with violence but less than proportionally **needs review** |
| climate_to_displacement | 0.1833 | assumed | climate_to_displacement: climate stress compounds conflict displacement **needs review** |
| conflict_reversion | 0.3167 | assumed | conflict_reversion: war intensity spikes and subsides over a few years **needs review** |
| displacement_return_rate | 0.05167 | assumed | displacement_return_rate: displacement is protracted; most situations last over a decade **needs review** |
| amr_growth | 0.0177 | sourced | amr_growth: mode is the rate implied by the GRAM forecast of 1.14M (2021) to 1.91M (2050); the range around it is our own (https://healthdata.org/node/12105) |
| pandemic_annual_probability | 0.02083 | sourced | pandemic_annual_probability: historical-record estimate of about 2% a year (Marani et al., PNAS 2021, Intensity and frequency of extreme novel epidemics) **needs review** |
| pandemic_deaths | 12.67 | assumed | pandemic_deaths: centred below COVID-19's roughly 15 million excess deaths in 2020-21 with a tail several times larger **needs review** |
| pandemic_to_poverty | 0.06667 | assumed | pandemic_to_poverty: anchored on COVID-19 adding roughly a tenth to extreme poverty **needs review** |
| ai_gap_growth | 0.05833 | assumed | ai_gap_growth: no measured series exists; this index is a placeholder for the direction of travel **needs review** |
| inequality_drift | 0 | assumed | inequality_drift: the share has moved little for decades **needs review** |
| mitigation_effect | 0.051 | assumed | mitigation_effect: UNEP's 45% cut by 2035 implies about 6% a year; treated as near the ceiling of what is achievable **needs review** |
| peacebuilding_effect | 0.7 | assumed | peacebuilding_effect: sustained mediation and peacekeeping roughly halving long-run deaths **needs review** |
| peacebuilding_return_effect | 0.03167 | assumed | peacebuilding_return_effect: settlements allow returns **needs review** |
| development_poverty_effect | 0.03 | assumed | development_poverty_effect: transfers and debt relief roughly doubling the recent pace **needs review** |
| development_hunger_effect | 0.01583 | assumed | development_hunger_effect: nutrition programmes and smallholder support **needs review** |
| health_amr_effect | 0.03 | assumed | health_amr_effect: the GRAM study's better-care scenario averts a large share of deaths **needs review** |
| health_pandemic_effect | 0.4833 | assumed | health_pandemic_effect: surveillance and fast vaccine manufacture **needs review** |
| conservation_effect | 0.001217 | assumed | conservation_effect: enough to halt and slightly reverse the recent loss rate **needs review** |
| wash_effect | 0.025 | assumed | wash_effect: roughly doubling the recent pace of coverage gains **needs review** |
| ai_governance_effect | 0.08333 | assumed | ai_governance_effect: sized to close the gap at the modal growth rate **needs review** |
| redistribution_inequality_effect | 0.25 | assumed | redistribution_inequality_effect: coordinated wealth and corporate minimum taxes **needs review** |
| redistribution_poverty_effect | 0.0105 | assumed | redistribution_poverty_effect: proceeds spent on transfers **needs review** |
