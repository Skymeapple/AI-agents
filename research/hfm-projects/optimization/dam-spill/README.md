# Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_cascade_spill_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU, as a reduced model.
- **Data access:** Columbia Basin Research DART query service: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** A real fish-passage model (COMPASS relationships) and hourly data.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work optimizes spill operations at eight federal dams on the Snake and Columbia rivers to safely pass juvenile salmon and steelhead during spring migration. Spillway passage is safer than turbine passage.

- **Metric:** `mean_instance_score` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.7199
- **HFM effort:** 66 proposals, 60 scored, 20 accepted improvements, about $2,979 of compute
- **HFM's current best idea:** 在p2172的LWG、LGS、LMN和BON内层五动作选择中，按发布时预测上限在可达区间中的连续位置软加权成熟损失，相比原等权内层选择提高完整练习集mean_instance_score。

## Existing theory and methods

COMPASS passage model, model-predictive control of reservoir cascades, gas-cap constraints.

## Public data

Columbia Basin Research DART daily river conditions for the eight forebay and tailwater gauges (public; data courtesy of the US Army Corps of Engineers).

## What limits progress

Gas forecast error is as large as the margin joint planning would use; fish survival relationships are not public in a ready form.

## Route to a step change

Joint optimisation of all eight dams over the fish travel horizon with gas limits as hard constraints.

## Experiment and results

Code: [`dam_spill.py`](dam_spill.py) in this folder. It ran on free GitHub Actions compute; the log is in [`research/compute/results/dam-spill/`](../../../compute/results/dam-spill/run.log).

- **Data:** daily flow, spill and dissolved gas at the forebay and tailwater gauges of all eight dams, 1 April to 20 June, 2012–2024 (Columbia Basin Research DART; data courtesy of the US Army Corps of Engineers). Models were fitted on 2012–2021 and tested on 2022–2024.
- **Part 1, measured: predicting dissolved gas.** Error in percentage points of saturation on the held-out years, mean over the eight dams:

  | Prediction of tailwater gas | Error |
  | --- | --- |
  | Linear in spill and forebay gas | **1.80** |
  | Saturating form (gas from spill levels off, as in the Corps' model) | 2.02 (better at 4 of 8 dams) |
  | Yesterday's measured value (not usable to plan a different spill) | 1.40 |

  - The next dam's forebay gas is predicted from the upstream release with an error of 2.2 to 4.5 points, against 4.1 to 6.6 for the seasonal mean. Between 79% and 114% of the excess gas released upstream arrives at the next dam, so gas does carry down the river.
  - **The physically motivated saturating form did not beat a straight line** on unseen years.
- **Part 2, model study: dam-by-dam against joint spill decisions.** The fitted gas model is used to choose each day's spill share at all eight dams under a tailwater gas cap. Survival values are assumed round numbers (spillway 0.98, other routes 0.94), not fitted.

  | Gas cap | Dam by dam | Joint (exact) | Change in passage loss | Days joint is better |
  | --- | --- | --- | --- | --- |
  | 120% | 0.7944 | 0.7973 | −1.4% | 79/243 |
  | 125% | 0.8147 | 0.8147 | 0.0% | 4/243 |

  - **Coordination across dams is worth little under this model:** 1.4% at the tighter cap and nothing at 125%, where the cap rarely binds and every dam spills near its maximum.
  - Raising the cap from 120% to 125% is worth about seven times more than coordinating (0.7944 → 0.8147).
- **Reading:** the route proposed above, joint optimisation of all eight dams, did not pay off in this reduced model. The gas prediction error (about 2 points) is as large as the margin that coordination would exploit.
- **Limits of this test:** daily averages, no travel-time dynamics beyond a fitted lag, no power or flow constraints, and no real fish-passage model. HFM's instances reward things this model does not contain.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **In this reduced model:** joint planning removes at most 1.4% of passage loss.
- **Where the room is:** the cap level and the accuracy of the gas forecast, not the optimiser.

## Next steps

- Hourly data and spill patterns by gate, which is how the Corps' gas model is used.
- Replace the assumed survival values with COMPASS relationships.
- Test whether weather (wind, temperature) explains the gas prediction error.
