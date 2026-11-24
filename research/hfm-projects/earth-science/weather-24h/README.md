# 24-Hour Global Near-Surface Weather Forecasting with ArchesWeather

- **Field:** earth-science
- **HFM track:** `discovery_live_ai4x_weather_wb2` (active on HFM)
- **Project status:** Blocked

## The challenge

This work examines 24-hour global forecasts of 2 m temperature, 10 m wind (u and v), and mean sea-level pressure using ERA5 reanalysis on the WeatherBench2 1.5° grid (121 × 240, poles included) at 6-hour intervals. The starter is ArchesWeather, a published transformer weather model trained on this grid; held-out evaluation is the year 2022.

- **Metric:** `area_weighted_normalized_rmse` (minimize)
- **HFM baseline:** 0.09823
- **HFM best so far:** 0.09749
- **HFM effort:** 36 proposals, 34 scored, 18 accepted improvements, about $3,580 of compute
- **HFM's current best idea:** Adding a constrained geographic slope map only to p3941's broad-innovation feature, while retaining its winning slow/local slopes and causal history, will reduce 24-hour area-weighted normalized RMSE below the promoted p3941 recipe.

## Existing theory and methods

Learned global models (GraphCast, Pangu, ArchesWeather, GenCast), ensemble post-processing (EMOS, quantile mapping), multi-model blending, autoregressive fine-tuning on rollouts.

## Public data

WeatherBench 2 ERA5 at 1.5 degrees (public Google Cloud bucket). ArchesWeather code and weights are public.

## What limits progress

HFM's agents only post-process the frozen model's output, which cannot fix errors in its dynamics. The one fine-tuning attempt used ten of forty public years.

## Route to a step change

Fine-tune on all public years with a surface-weighted, 4-step rollout loss. Blend with a second model family.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. ArchesWeather inference and fine-tuning on ERA5 are not feasible on CPU.

## How far this can go

At 24 h the best published models are within a few percent of each other on surface RMSE. A 1 to 3% gain over ArchesWeather is plausible; more is unlikely without higher resolution or better initial conditions.

## Next steps

Reproduce the post-processing gain on CPU from stored forecasts if a GPU produces them once. Then fine-tune.
