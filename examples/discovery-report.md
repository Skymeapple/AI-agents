# Discovered directions

_16 directions covering 1174 works (100%); 0 works too isolated to place; vocabulary 487 terms_

| Direction | Works | Cohesion | Sector | Routing confidence |
| --- | --- | --- | --- | --- |
| design / structure / models | 109 | 0.58 | software_ai | 76% |
| editing / base / prime | 102 | 0.64 | biopharma | 100% |
| targeted / degradation / vitro | 99 | 0.55 | biopharma | 100% |
| sodium / solid-state / batteries | 89 | 0.61 | energy_hardware | 77% |
| perovskite / tandem / photovoltaics | 88 | 0.57 | energy_hardware | 87% |
| emulators / learning / machine | 85 | 0.59 | software_ai | 79% |
| error / correction / quantum | 84 | 0.60 | semiconductors | 76% |
| solid / tumour / car-t | 81 | 0.59 | biopharma | 100% |
| hydrogen / electrolysis / catalysts | 80 | 0.61 | energy_hardware | 100% |
| carbon / capture / air | 68 | 0.63 | energy_hardware | 100% |
| therapeutics / gut / microbiome | 62 | 0.57 | biopharma | 100% |
| fermentation / proteins / precision | 61 | 0.57 | agriculture_food | 88% |
| crops / field / gene-edited | 59 | 0.59 | agriculture_food | 73% |
| photonic / computing / neuromorphic | 39 | 0.60 | semiconductors | 86% |
| topological / qubits / materials | 36 | 0.59 | semiconductors | 88% |
| superconductivity / transition / ambient-pressure | 32 | 0.70 | energy_hardware | 100% |

| Sector | Directions |
| --- | --- |
| energy_hardware | 5 |
| biopharma | 4 |
| semiconductors | 3 |
| software_ai | 2 |
| agriculture_food | 2 |

Every direction routed to a modelled sector. Note that this says the vocabulary matched, not that the routing is right -- lexical classification is a weak method and the sector drives every cost and timeline downstream, so spot-check the assignments.

## Discovery quality against known labels

- Directions recovered: **16 of 16** (100%)
- Mean cluster purity: **1.00**
- Works placed in a direction: **100%**
- Completeness (each true direction kept in one cluster): **1.00**; V-measure **1.00**
- Adjusted Rand index: **1.00** (1 is a perfect match, 0 is chance; unlike purity it cannot be raised by splitting directions into fragments)

## Sensitivity to the similarity threshold

| Threshold | Directions | Works placed | Agreement with the run above |
| --- | --- | --- | --- |
| 0.096 | 15 | 100% | 0.99 |
| 0.128 | 16 | 100% | 1.00 |
| 0.160 | 16 | 100% | 1.00 |
| 0.192 | 16 | 100% | 1.00 |
| 0.224 | 16 | 100% | 1.00 |

Agreement is the adjusted Rand index against the clustering at the threshold in use. It needs no labels: if it stays near 1 the directions are a property of the corpus, and if it falls away they are a property of the threshold.

## Ranked by priority

| # | Direction | Priority | Importance | Urgency | Confidence | TRL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | error / correction / quantum | 0.69 | 0.58 | 0.91 | 88% | 3 |
| 2 | design / structure / models | 0.69 | 0.59 | 0.86 | 88% | 5 |
| 3 | emulators / learning / machine | 0.66 | 0.56 | 0.86 | 88% | 5 |
| 4 | sodium / solid-state / batteries | 0.66 | 0.52 | 0.95 | 88% | 6 |
| 5 | editing / base / prime | 0.65 | 0.55 | 0.84 | 88% | 6 |
| 6 | carbon / capture / air | 0.63 | 0.48 | 0.94 | 88% | 5 |
| 7 | perovskite / tandem / photovoltaics | 0.63 | 0.49 | 0.91 | 88% | 5 |
| 8 | solid / tumour / car-t | 0.61 | 0.48 | 0.87 | 88% | 7 |
| 9 | photonic / computing / neuromorphic | 0.61 | 0.52 | 0.79 | 88% | 3 |
| 10 | targeted / degradation / vitro | 0.61 | 0.48 | 0.87 | 88% | 5 |
| 11 | hydrogen / electrolysis / catalysts | 0.61 | 0.47 | 0.88 | 88% | 3 |
| 12 | topological / qubits / materials | 0.55 | 0.42 | 0.80 | 88% | 2 |
| 13 | fermentation / proteins / precision | 0.54 | 0.42 | 0.78 | 88% | 5 |
| 14 | crops / field / gene-edited | 0.52 | 0.39 | 0.80 | 88% | 5 |
| 15 | therapeutics / gut / microbiome | 0.50 | 0.39 | 0.75 | 88% | 5 |
| 16 | superconductivity / transition / ambient-pressure | 0.50 | 0.44 | 0.60 | 88% | 2 |

