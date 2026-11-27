# Wind-Farm Wake Calibration and Power Prediction from Real SCADA Data

- **Field:** physics
- **HFM track:** `discovery_live_rsi_wind_wake_calibration_opus5` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Kelmarsh and Penmanshiel SCADA on Zenodo.
- **Compute:** CPU.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work calibrates a wake model using real SCADA data from the Kelmarsh and Penmanshiel wind farms at 10-minute resolution, then predicts per-turbine power for unseen time periods and unseen wind farms. The data, the FLORIS wake engine, and the evaluation protocol are frozen; only the calibration and prediction method evolves.

- **Metric:** `reward` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.4124
- **HFM effort:** 6 proposals, 6 scored, 0 accepted improvements, about $115 of compute
- **HFM's current best idea:** 在 p252 的「物理骨架 × 分层收缩修正」之上，用每条 record 自己的标定窗回放配对训练每机组的梯度提升残差回归器，并与修正后的物理预测按固定权重平均，能显著压低占 70% 权重的 nMAE 分项(当前 hidden 上 nMAE 分项仅 0.240、eta 分项已 0.571)，从而提高总 reward，且因为回归器逐 record 自训练，在未见场址上同样成立。

## Existing theory and methods

Jensen and Gaussian wake models, FLORIS, wake-parameter calibration, wind-direction bias correction, hybrid physics and machine-learning residual models.

## Public data

Kelmarsh (Zenodo 16807551) and Penmanshiel (Zenodo 16807304) SCADA, CC-BY 4.0, 2017.

## What limits progress

Inflow direction is poorly known, and simple wake models miss terrain and turbine-specific effects.

## Route to a step change

Calibrate direction and wake decay first, then add a learned residual.

## Experiment and results

Code: [`wind_wake.py`](wind_wake.py) in this folder. Run instructions are in its header.

- **Setup:** the same two farms HFM uses, Kelmarsh and Penmanshiel, 2017
  10-minute SCADA from Zenodo. A simple Jensen wake model stands in for
  FLORIS. Each farm gets a 60-day calibration window and is tested on the
  rest of the year. All choices were made on Kelmarsh; Penmanshiel is the
  unseen farm. Error is mean absolute power error as a share of rated
  power.
- **Result:**

  | Method | Kelmarsh, vs no-wake | Penmanshiel (unseen), vs no-wake | Penmanshiel, new wind directions |
  | --- | --- | --- | --- |
  | Wake model, default settings | −27.3% | −1.4% | 0.0810 |
  | Wake model, direction offset and decay calibrated | −27.3% | **−11.9%** | 0.0620 |
  | + per-turbine sector factors | −31.6% | −19.8% | 0.0560 |
  | Boosted trees per turbine (no physics) | −36.0% | −29.3% | 0.0567 |
  | Calibrated wake model + boosted-tree residual | −35.7% | **−29.8%** | **0.0523** |

- **What this shows:**
  - **Calibrating the wind direction matters most for the physics
    model.** At the unseen farm a 9° direction offset turned a 1.4% gain
    into 11.9%. This supports the first half of Section 7.10.
  - **The second half of Section 7.10 is wrong.** It predicted that
    machine-learned residuals would not carry to unseen conditions. In
    fact trees trained on each farm's own calibration window beat the
    physics model on both farms, and physics plus a learned residual was
    best, including in wind directions the calibration window barely
    covered.
  - The Jensen model is cruder than FLORIS, so physics may be
    undersold here.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

The hybrid reaches 0.058 of rated power at the unseen farm (reference 0.083). Ten-minute turbulence makes single-turbine power noisy; published farm-level studies rarely go below about 4 to 5% of rated at this resolution.

## Next steps

Swap in FLORIS's Gaussian model. Add turbulence intensity as an input. Estimate the noise floor from pairs of unwaked turbines.
