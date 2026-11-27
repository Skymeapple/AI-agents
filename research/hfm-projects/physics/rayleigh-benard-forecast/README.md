# 64-Frame Rayleigh–Bénard Convection Forecasting with Walrus and The Well

- **Field:** physics
- **HFM track:** `discovery_live_ai4x_physics_rb` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** The Well rayleigh_benard dataset and Walrus weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** A GPU for Walrus. Files are several GB each.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

Code: [`forecast_floor.py`](forecast_floor.py) in this folder. It ran on free GitHub Actions compute (four CPU cores, about two minutes per flow setting); logs are in [`research/compute/results/`](../../../compute/results/) under `rb-forecast-floor-*`.

- **This is a floor, not a forecast model.** The Walrus model needs a GPU. These are five methods with no neural network, to measure how much of the score simple statistics explain.
- **Setup:** The Well's public data at Prandtl 1 and Rayleigh 1e6, 1e8 and 1e10. Weights are fitted on 8 training runs and scored on the 5 test runs, from start frames 0 (onset), 65 and 130. Input is 6 frames, the forecast is the next 64. Error is RMSE divided by the standard deviation of the true frames, averaged over the four fields. This normalisation is ours; HFM's `rollout_nrmse` may differ, so the numbers are not comparable with 0.3693 or 0.3129.

  | Method | Rayleigh 1e6 | 1e8 | 1e10 |
  | --- | --- | --- | --- |
  | Persistence (repeat the last frame) | 0.663 | 0.630 | 0.747 |
  | Mean vertical profile | 0.762 | 0.820 | 0.945 |
  | Damped persistence | 0.580 | 0.616 | 0.877 |
  | Blurred and damped persistence | **0.558** | **0.573** | 1.115 |
  | Linear regression on 60 flow modes | 0.646 | 0.759 | 0.915 |

- **Blurring towards the mean helps only a little.** At 1e6 and 1e8 the best blur-and-shrink forecast is 9% to 16% better than plain persistence. If HFM's metric is close to ours, its 0.37 to 0.31 is well below anything blurring reaches, so the Walrus scores reflect real forecasting skill.
- **Skill is lost within about 8 frames.** From start frame 65 at 1e8, the best floor's error is 0.25 at lead 1, 0.59 at lead 8 and 0.72 at lead 64. Most of the 64-frame score is therefore set by frames where no simple method knows more than the mean.
- **Onset is a different problem.** From frame 0 the velocity error is 1.0 for every method: the flow is still at rest in the input and convection starts inside the forecast window. This matches the HFM agents' choice to treat onset separately.
- **A negative result:** at 1e10 the fitted methods are worse than persistence, and the blurred one fails at onset (1.87). Weights fitted on 8 runs do not transfer to the test runs at the most turbulent setting. A linear forecast on flow modes was never the best method at any setting.

## How far this can go

For chaotic regimes, RMSE at frame 64 approaches the climatological spread whatever the model. The metric saturates before the science does.

## Next steps

Measure predictability time per Rayleigh number with twin simulations.
