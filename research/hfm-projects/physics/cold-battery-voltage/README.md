# Battery Voltage Model for Cold Operating Conditions

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_cold_cell_voltage` (active on HFM)
- **Project status:** Tested on public data

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

LG 18650HG2 dataset (Kollmeyer et al., Mendeley Data doi:10.17632/cp3473x7xv.3, CC-BY 4.0), used here. Panasonic 18650PF dataset (doi:10.17632/wykht8y7tg.1). Both download from `https://data.mendeley.com/public-api/zip/<id>/download/<version>`.

## What limits progress

Extrapolating below the calibrated temperature range with fitted thermal slopes.

## Route to a step change

Arrhenius-form parameters with one activation energy per process.

## Experiment and results

Code: [`cold_voltage.py`](cold_voltage.py) in this folder. Run instructions are in its header.

- **Setup:** the public LG 18650HG2 dataset (Mendeley Data,
  doi:10.17632/cp3473x7xv.3), one cell tested at 40, 25, 10, 0, −10 and
  −20 °C. The model is calibrated on drive cycles at 40, 25 and 10 °C and
  must predict voltage on 35 drive cycles at 0, −10 and −20 °C from the
  current and measured temperature. The model is an open-circuit voltage
  curve plus an ohmic resistance and two RC pairs. Methods differ only in
  how the three resistances are carried to the cold.
- **Result (voltage RMS error in millivolts):**

  | Temperature | Hold 10 °C values | Linear slope in temperature | Arrhenius form | Oracle (fitted at that temperature) |
  | --- | --- | --- | --- | --- |
  | 0 °C | 73.1 | 64.5 | 57.0 | 51.9 |
  | −10 °C | 160.9 | 134.1 | 104.3 | 78.4 |
  | −20 °C | 306.5 | 262.7 | 193.8 | 114.4 |
  | All | 183.3 | 156.3 | **120.1** | 82.4 |

  | Method | Skill against holding the 10 °C values | Cycles improved |
  | --- | --- | --- |
  | Linear slope | +0.147 | 35/35 |
  | Arrhenius form | **+0.344** | 35/35, and better than the linear slope on 35/35 |
  | Oracle | +0.550 | 35/35 |

- **What this shows:**
  - **The Arrhenius form more than doubles the skill of a fitted linear
    slope,** on every cold drive cycle. Section 7.1 of the overview is
    supported.
  - **It still under-predicts the cold resistance.** Fitted resistances
    at −20 °C are 69, 45 and 126 milliohm; the warm data imply activation
    energies of only 12, 2 and 20 kJ/mol, which is too little growth.
    The 4-second pair barely changes between 10 and 40 °C and then
    quadruples by −20 °C, so its trend cannot be learned from warm data
    at all.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

The oracle shows where the limits are.
- **Limit of temperature extrapolation:** 120 mV (Arrhenius) against 82 mV (oracle). About a third of the remaining error is extrapolation error, concentrated in the fast RC pair.
- **Limit of this model structure:** 82 mV on average and 114 mV at −20 °C even with resistances fitted at that temperature. Constant resistances across state of charge, no capacity loss in the cold and no hysteresis are the likely causes.
- **Measurement floor:** a few millivolts. The structure, not the sensor, is the floor here.

## Next steps

- Make the resistances depend on state of charge and let usable capacity shrink with temperature, to lower the 82 mV oracle error.
- Use a prior activation energy from other cells of the same chemistry for the fast pair, since warm data cannot identify it. The Panasonic 18650PF dataset (doi:10.17632/wykht8y7tg.1) is the natural donor.
- Calibrate on the pulse (HPPC) tests instead of drive cycles, which is closer to HFM's setup.
