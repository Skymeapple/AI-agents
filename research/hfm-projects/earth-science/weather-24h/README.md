# 24-Hour Global Near-Surface Weather Forecasting with ArchesWeather

- **Field:** earth-science
- **HFM track:** `discovery_live_ai4x_weather_wb2` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** WeatherBench 2 public bucket with ERA5 and stored forecasts of Pangu, IFS HRES, GraphCast, FuXi and others at the challenge grid: Verified reachable from the free compute on 2026-11-25. ArchesWeather weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU on free compute for post-processing and blending of stored forecasts. A GPU to run or fine-tune ArchesWeather itself.
- **Still missing:** A free GPU account to produce ArchesWeather forecasts.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

WeatherBench 2 public bucket (`gs://weatherbench2/datasets/`, read anonymously): ERA5 and stored forecasts of Pangu-Weather, IFS HRES, GraphCast, FuXi, NeuralGCM and others at 240 × 121. ArchesWeather code and weights are on Hugging Face (`gcouairon/ArchesWeather`).

## What limits progress

HFM's agents only post-process the frozen model's output, which cannot fix errors in its dynamics. The one fine-tuning attempt used ten of forty public years.

## Route to a step change

Fine-tune on all public years with a surface-weighted, 4-step rollout loss. Blend with a second model family.

## Experiment and results

Code: [`blend_forecasts.py`](blend_forecasts.py) in this folder. It ran on free GitHub Actions compute in about 90 seconds; the log is in [`research/compute/results/weather-blend-2020/`](../../../compute/results/weather-blend-2020/run.log).

- **Setup:** WeatherBench 2 publishes forecasts from several models on
  the challenge's own grid (1.5°, 240 × 121), so this needs no GPU.
  ArchesWeather's forecasts are not in that archive, so the test uses the
  three models that are and that carry the four surface variables:
  Pangu-Weather, IFS HRES and GraphCast. Forecasts start at 00 UTC every
  second day of 2020, 24-hour lead, scored against ERA5. Alternating
  two-week blocks are used for fitting (91 forecasts) and testing (91).
  The score is latitude-weighted RMSE per variable, divided by the
  variable's standard deviation, averaged over 2 m temperature, 10 m wind
  (u, v) and sea-level pressure.
- **Result (normalised RMSE, lower is better):**

  | Method | 2 m temp. | u wind | v wind | Pressure | Mean | Versus best corrected single model |
  | --- | --- | --- | --- | --- | --- | --- |
  | Pangu-Weather, raw | 0.0268 | 0.1318 | 0.1594 | 0.0403 | 0.0896 | +12.0% |
  | Pangu-Weather, bias-corrected | 0.0265 | 0.1314 | 0.1594 | 0.0399 | 0.0893 | +11.7% |
  | IFS HRES, raw | 0.0412 | 0.1547 | 0.1877 | 0.0454 | 0.1072 | +34.1% |
  | IFS HRES, bias-corrected | 0.0359 | 0.1517 | 0.1855 | 0.0433 | 0.1041 | +30.2% |
  | GraphCast, raw | 0.0240 | 0.1188 | 0.1435 | 0.0356 | 0.0805 | +0.6% |
  | GraphCast, bias-corrected | 0.0234 | 0.1182 | 0.1432 | 0.0351 | 0.0800 | — |
  | Equal blend of the three | 0.0238 | 0.1152 | 0.1400 | 0.0323 | 0.0778 | −2.7% |
  | Weighted blend of the three | 0.0226 | 0.1132 | 0.1374 | 0.0319 | **0.0763** | **−4.6%** |

- **What this shows:**
  - **Bias correction of one machine-learned model gains 0.3–0.6%.**
    That is the same size as the 0.75% HFM's agents have reached by
    post-processing ArchesWeather. Post-processing a good model has
    little left to give.
  - **Blending model families gains 4.6%** over the best corrected
    single model, about eight times as much. Even the weakest model
    (IFS HRES, 30% worse alone) receives 12–27% of the weight, because
    its errors differ from the others'.
  - Section 2.1 of the overview (route 2) is supported. IFS HRES is at a
    disadvantage here because ERA5 is the reference.
  - One year and one split; the blend weights were fitted on half the
    dates. The size of the gain should be confirmed on a second year.

### Follow-up: 2022, HFM's held-out year

Log: [`research/compute/results/weather-blend-2022/`](../../../compute/results/weather-blend-2022/run.log). Same protocol, with every forecast set stored for 2022 on this grid: GraphCast (v2), FGN (ensemble mean), Pangu-Weather, Pangu-Weather started from operational analyses, and IFS HRES.

| Method | 2 m temp. | u wind | v wind | Pressure | Mean | Versus best corrected single model |
| --- | --- | --- | --- | --- | --- | --- |
| GraphCast v2, bias-corrected (best single) | 0.0234 | 0.1155 | 0.1411 | 0.0336 | 0.0784 | — |
| FGN, bias-corrected | 0.0334 | 0.1176 | 0.1441 | 0.0330 | 0.0820 | +4.6% |
| Pangu-Weather, bias-corrected | 0.0267 | 0.1298 | 0.1584 | 0.0389 | 0.0884 | +12.8% |
| Pangu-Weather (operational start), bias-corrected | 0.0332 | 0.1379 | 0.1684 | 0.0416 | 0.0953 | +21.5% |
| IFS HRES, bias-corrected | 0.0372 | 0.1520 | 0.1850 | 0.0417 | 0.1040 | +32.7% |
| Equal blend of the five | 0.0260 | 0.1122 | 0.1370 | 0.0310 | 0.0766 | −2.3% |
| Weighted blend of the five | 0.0226 | 0.1060 | 0.1296 | 0.0291 | **0.0718** | **−8.4%** |

- **The blending gain is confirmed on a second year and is larger: 8.4%.** Most of it comes from combining GraphCast with FGN, which get nearly equal weight for wind and pressure.
- **Bias correction alone gains 0.4–0.5%** for GraphCast and Pangu, again the size of HFM's whole gain so far.
- HFM's ArchesWeather score of 0.0975 uses its own normalisation, so it cannot be placed in this table. The ArchesWeather paper itself reports GraphCast ahead of ArchesWeather at 24 hours on wind.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Post-processing one model:** under 1%, here and in HFM's own record.
- **Blending model families:** 4.6% with three sets in 2020 and 8.4% with five in 2022. Returns diminish as added models become correlated.
- **Fine-tuning the model itself:** untested. Published 24-hour surface scores of the best models differ by a few percent, so a few percent is the plausible range.
- **Floor:** ERA5 is itself an analysis with error, and 24-hour forecasts are already close to it for pressure (0.03 of the variable's spread).

## Next steps

- Produce ArchesWeather forecasts for 2020 on a free GPU and add them to the blend. This is the direct test of whether HFM's 0.0975 can be beaten by blending.
- Let blend weights vary by region and season.
- Add FuXi (its stored dates did not match and it was skipped) and ensemble-mean forecasts.
