# Battery Voltage Model for Cold Operating Conditions

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_cold_cell_voltage` (active on HFM)
- **Project status:** Blocked

## The challenge

This work develops a reusable model for a cell's terminal-voltage trajectory under a supplied current and temperature profile. Calibration data covers warmer conditions than the assessment drive cycles. The task is to maximize a fixed prediction score by choosing the model structure, parameterization, and fitting procedure.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.5991
- **HFM effort:** 100 proposals, 94 scored, 41 accepted improvements, about $8,971 of compute
- **HFM's current best idea:** A signed-current-dependent Panasonic thermal-range ratio selected from characterization pulses, while retaining p3987 as the symmetric control, increases clean-sandbox skill_mean relative to promoted p3987.

## Existing theory and methods

Equivalent-circuit models with RC pairs, Arrhenius temperature dependence of resistance and diffusion, Butler-Volmer kinetics, single-particle models.

## Public data

Kollmeyer's Panasonic 18650PF and LG 18650HG2 datasets on Mendeley Data (drive cycles from 25 C down to -20 C).

## What limits progress

Extrapolating below the calibrated temperature range with fitted thermal slopes.

## Route to a step change

Arrhenius-form parameters with one activation energy per process.

## Experiment and results

No experiment has been run for this project.

- **Why:** The public datasets HFM appears to use (Panasonic 18650PF and LG HG2, Mendeley Data) could not be downloaded from this environment.

## How far this can go

Voltage sensor noise and cell-to-cell variation set a floor of a few millivolts RMS.

## Next steps

Obtain the Mendeley files and run a leave-coldest-out test of Arrhenius against fitted slopes.
