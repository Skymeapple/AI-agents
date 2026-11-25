# Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration

- **Field:** optimization
- **HFM track:** `discovery_live_live_optimization_cascade_spill_auto` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## Readiness

- **State:** Data verified; pipeline or simulator still to build.
- **Data access:** Columbia Basin Research DART query service: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU.
- **Still missing:** A reduced fish-passage and dissolved-gas model built from DART data.
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

Columbia Basin Research DART (dam flows, spill, dissolved gas, fish counts) and US Army Corps data are public.

## What limits progress

Dam-by-dam decisions.

## Route to a step change

Joint optimisation of all eight dams over the fish travel horizon with gas limits as hard constraints.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs the fish-passage and dissolved-gas models; public data exist but the model is not yet assembled.

## How far this can go

Bounded by a perfect-forecast joint optimisation.

## Next steps

Assemble a reduced passage model from DART data.
