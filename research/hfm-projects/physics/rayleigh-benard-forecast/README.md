# 64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well

- **Field:** physics
- **HFM track:** `discovery_live_ai4x_physics_rb` (active on HFM)
- **Project status:** Blocked

## The challenge

The work forecasts two-dimensional Rayleigh–Bénard convection from six consecutive frames of buoyancy, pressure, horizontal velocity, and vertical velocity. Simulations cover 35 Rayleigh number (1e6–1e10) and Prandtl number (0.1–10) combinations on a 512 × 128 grid, periodic in x with walls at y = 0 and 1; predictions span 64 frames.

- **Metric:** `rollout_nrmse` (minimize)
- **HFM baseline:** 0.3693
- **HFM best so far:** 0.3129
- **HFM effort:** 47 proposals, 39 scored, 13 accepted improvements, about $2,102 of compute
- **HFM's current best idea:** Applying p3740's physical projections before feedback only for starts 65 and 130, while leaving onset feedback raw, will lower held-out 64-frame rollout_nrmse relative to the all-regime p3740 recipe.

## Existing theory and methods

Neural operators, PDE foundation models, PDE-Refiner, spectral and conservation losses, ensemble rollouts.

## Public data

The Well (Polymathic AI) rayleigh_benard dataset and Walrus weights are public.

## What limits progress

Chaotic divergence: beyond a few turnover times, point-forecast RMSE rewards blurring.

## Route to a step change

Rollout training with physics losses; ensembles scored with CRPS.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. The Well dataset is hundreds of GB and Walrus is a large model.

## How far this can go

For chaotic regimes, RMSE at frame 64 approaches the climatological spread whatever the model. The metric saturates before the science does.

## Next steps

Measure predictability time per Rayleigh number with twin simulations.
