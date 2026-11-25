# PM2.5 Sensor Calibration Transfer

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_pm_calibration_transfer_auto` (paused on HFM)
- **Project status:** Open: public data exist, experiment not yet built

## Readiness

- **State:** Data verified; pipeline or simulator still to build.
- **Data access:** SensEURCity on Zenodo (5.7 GB): Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU on free compute.
- **Still missing:** The multi-site calibration pipeline.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

An air agency calibrates low-cost particle sensors to report hourly PM2.5 mass (µg/m³), monthly mean for standard comparison, and concentration tallies for advisories and exposure studies.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.6945
- **HFM effort:** 31 proposals, 29 scored, 10 accepted improvements, about $871 of compute
- **HFM's current best idea:** 保持 p1531 的湿度公式、健康选择和训练目标不变，仅以当前实例的环境温度中位数调节有界增益，可提高原始全部57个月的留一站平均官方 skill。

## Existing theory and methods

Barkjohn (US EPA) correction, kappa-Koehler hygroscopic growth, hierarchical calibration across sites.

## Public data

SensEURCity (Zenodo 17858205), PurpleAir co-location sets on Zenodo, US EPA AirSensor data.

## What limits progress

Aerosol type changes the optical response by site and season.

## Route to a step change

Infer hygroscopicity per site from particle-size bins; share calibration across sites hierarchically.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a multi-site co-location dataset; candidates are large (SensEURCity is 5.7 GB) and were not processed.

## How far this can go

Reference monitors themselves agree to about 10%, which bounds hourly skill.

## Next steps

Run leave-one-site-out calibration on SensEURCity.
