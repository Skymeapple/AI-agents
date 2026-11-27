# References: 24-Hour Global Near-Surface Weather Forecasting with ArchesWeather

Checked 2026-11-25. Each entry was confirmed in this session by a web search result, by the arXiv record, by the project's own README on GitHub, or by listing the public data bucket. Nothing is cited from memory.

## Theory and established methods

- **Gneiting, Raftery, Westveld, Goldman (2005).** Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. Monthly Weather Review 133:1098-1118. https://sites.stat.washington.edu/raftery/Research/PDF/gneiting2005.pdf
  - Defines EMOS: a bias-corrected weighted mean of forecast members. This is the standard form for the blending weights HFM's agents are tuning.
- **Vannitsem, Bremnes, Demaeyer et al. (2020).** Statistical Postprocessing for Weather Forecasts: Review, Challenges and Avenues in a Big Data World. BAMS. https://arxiv.org/abs/2004.06582
  - Survey of post-processing methods; use it to pick methods HFM has not tried.
- **Rasp, Lerch (2018).** Neural networks for post-processing ensemble weather forecasts. Monthly Weather Review 146:3885-3900. https://arxiv.org/abs/1805.09091
  - Neural-network alternative to EMOS, with code at https://github.com/slerch/ppnn. A template for a learned correction of a frozen model's output.
- **Bi, Xie, Zhang et al. (2022/2023).** Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast. arXiv; Nature 619:533-538. https://arxiv.org/abs/2211.02556
  - The model ArchesWeather modifies, and the second model family whose stored 2022 forecasts are public (see Datasets).
- **Lam, Sanchez-Gonzalez, Willson et al. (2022/2023).** GraphCast: Learning skillful medium-range global weather forecasting. arXiv; Science. https://arxiv.org/abs/2212.12794
  - Lowest 24 h surface RMSE among the models in the ArchesWeather comparison table. Read it for rollout training, the method named in this project's route to a step change (training details not checked against the text in this session).
- **Keisler (2022).** Forecasting Global Weather with Graph Neural Networks. https://arxiv.org/abs/2202.07575
  - Early graph model; its 2020 forecasts are stored in WeatherBench 2.

## Recent advances and breakthroughs (2023-2026)

- **Couairon, Lessig, Charantonis et al. (2024).** ArchesWeather: An efficient AI weather forecasting model at 1.5° resolution. ICML 2024 workshop. https://arxiv.org/abs/2405.14527
  - The starter model. Table 1 gives 24 h RMSE for it and for competitors (see Published results to beat).
- **Couairon, Singh, Charantonis et al. (2024).** ArchesWeather & ArchesWeatherGen: a deterministic and generative model for efficient ML weather forecasting. arXiv; the geoarches README also links a Science Advances version. https://arxiv.org/abs/2412.12971
  - Trained at 1.5° for about 9 V100-days; ArchesWeatherGen samples an ensemble on top of the deterministic model, so an ensemble mean is available from the same code.
- **Price, Sanchez-Gonzalez, Alet et al. (2023/2024).** GenCast: Diffusion-based ensemble forecasting for medium-range weather. arXiv; Nature (DOI 10.1038/s41586-024-08252-9). https://arxiv.org/abs/2312.15796
  - Diffusion ensemble at 0.25°. Stored GenCast forecasts, including an ensemble mean, exist in WeatherBench 2 for 2020 only.
- **Kochkov, Yuval, Langmore et al. (2023/2024).** Neural General Circulation Models for Weather and Climate. Nature 632:1060-1066. https://arxiv.org/abs/2311.07222
  - Hybrid physics/ML model, a different error structure from pure transformers; code at https://github.com/neuralgcm/neuralgcm.
- **Chen, Zhong, Zhang et al. (2023).** FuXi: A cascade machine learning forecasting system for 15-day global weather forecast. npj Climate and Atmospheric Science. https://arxiv.org/abs/2306.12873
  - Another model family; code and weights at https://github.com/tpys/FuXi.
- **Bodnar, Bruinsma, Lucic et al. (2024/2025).** A Foundation Model for the Earth System (Aurora). Nature 641:1180-1187. https://arxiv.org/abs/2405.13063
  - Stored 2022 Aurora forecasts on the 240x121 grid are in the WeatherBench 2 bucket. Code: https://github.com/microsoft/aurora.
- **Lang, Alexe, Chantry et al. (2024).** AIFS - ECMWF's data-driven forecasting system. https://arxiv.org/abs/2406.01465
  - ECMWF's operational ML model; ECMWF states the weights are open on Hugging Face (https://www.ecmwf.int/node/29246).
- **Alet, Price, El-Kadi et al. (2025).** Skillful joint probabilistic weather forecasting from marginals (FGN). https://arxiv.org/abs/2506.10772
  - Stored 2022 FGN forecasts, including an ensemble mean on the 240x121 grid, are in the WeatherBench 2 bucket.
- **Nguyen, Shah, Bansal et al. (2023).** Scaling transformer neural networks for skillful and reliable medium-range weather forecasting (Stormer). NeurIPS 2024. https://arxiv.org/abs/2312.03876
  - Trains one model to forecast over varying time intervals (abstract). ArchesWeather's Table 1 lists it as a 1.4° competitor with better 24 h wind scores.
- **Chakraborty, Maulik, Harrington et al. (2025).** MoWE: A Mixture of Weather Experts. https://arxiv.org/abs/2509.09052
  - A gating network blends stored forecasts from several ML models per grid point; abstract reports up to 10% lower RMSE than the best single model at 2 days, and better than a plain average.
- **Trotta, Johnson, de Burgh-Day et al. (2025).** Statistical post-processing yields accurate probabilistic forecasts from Artificial Intelligence weather models. AIES. https://arxiv.org/abs/2504.12672
  - Standard operational post-processing (IMPROVER) helps AIFS as much as it helps physics models; blending adds skill even when the added model is not the best one.
- **Sop Nkuiate, Evans, Ferro et al. (2026).** Value of Adding AI Weather Model Ensembles to a Multi-model Probabilistic Blend. https://arxiv.org/abs/2610.42130
  - Blend gain depends on member skill and on how different the members' errors are; argues for blending models from different families.
- **Agrawal, Carver, Gazen et al. (2023).** A Machine Learning Outlook: Post-processing of Global Medium-range Forecasts. https://arxiv.org/abs/2303.16301
  - Neural post-processing on the global grid; abstract reports up to 12% lower RMSE (850 hPa temperature, day 7) and discusses the limits of RMSE as the only metric.

## Datasets and benchmarks

- **Rasp, Hoyer, Merose et al. (2023).** WeatherBench 2: A benchmark for the next generation of data-driven global weather models. https://arxiv.org/abs/2308.15560
  - Defines the grid, variables and evaluation used by this task. Evaluation code: https://github.com/google-research/weatherbench2 (successor: https://github.com/google-research/weatherbenchX).
- **WeatherBench 2 public bucket, stored forecasts (listed in this session with `gsutil ls gs://weatherbench2/datasets/`).** Data guide: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
  - Forecast Zarr stores on the task grid (240x121, poles included), each with 2 m temperature, 10 m u and v, and mean sea-level pressure at 6 h lead steps:
    - `gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr` (3,652 start times, 40 lead steps). Covers the 2022 test year and four earlier years for fitting blend weights.
    - `gs://weatherbench2/datasets/aurora/2022-240x121_equiangular_with_poles_conservative.zarr` (728 start times).
    - `gs://weatherbench2/datasets/fgn/2022-240x121_equiangular_with_poles_conservative_mean.zarr` (1,460 start times, ensemble mean).
    - `gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr` (ECMWF HRES).
    - 2020 only: `graphcast/2020`, `graphcast_hres_init/2020`, `neuralgcm_deterministic`, `neuralgcm_ens`, `fuxi`, `gencast`, `keisler`, `sphericalcnn`. GraphCast also has 2018.
  - Truth: `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`.
  - No ArchesWeather forecasts are stored there. Check each store's start times and whether it was started from ERA5 or from operational analyses before blending.

## Open-source code and tools

- **geoarches (INRIA).** https://github.com/INRIA/geoarches ; docs https://geoarches.readthedocs.io/en/latest/archesweather/
  - Training, inference and evaluation code and checkpoints for ArchesWeather and ArchesWeatherGen. BSD 3-Clause.
- **GraphCast / GenCast / WeatherNext code (Google DeepMind).** https://github.com/google-deepmind/graphcast
  - Reference implementation of rollout fine-tuning.
- **ai-models (ECMWF).** https://github.com/ecmwf-lab/ai-models
  - Runs Pangu, FourCastNet, GraphCast, FuXi and Aurora through plugins. Now archived. A search result reports CPU inference is possible but takes hours for a 10-day forecast.
- **earth2studio (NVIDIA).** https://github.com/NVIDIA/earth2studio
  - One interface to run many pretrained weather models.
- **anemoi-core (ECMWF).** https://github.com/ecmwf/anemoi-core
  - Training framework behind AIFS.
- **IMPROVER (Met Office).** https://github.com/metoppv/improver
  - Operational post-processing and blending library used in the Trotta et al. and Sop Nkuiate et al. studies.

## Breakthroughs in related fields that could transfer

- **Brenowitz, Cohen, Pathak et al. (2024).** A Practical Probabilistic Benchmark for AI Weather Models. GRL 52, e2024GL113656. https://arxiv.org/abs/2401.15305
  - Lagged ensembles: build an ensemble from a library of deterministic forecasts started at different times. Needs only stored forecasts, so it is a CPU experiment.

## Published results to beat

- Source: Couairon et al. 2024, Table 1 (https://arxiv.org/abs/2405.14527), RMSE at 24 h lead time, read from the arXiv HTML in this session. Columns are T2m (K), surface pressure (Pa), U10m, V10m (m/s):
  - ArchesWeather-M: 0.550, 60.9, 0.834, 0.877
  - ArchesWeather-L x2 ensemble: 0.525, 55.956, 0.776, 0.815
  - Pangu-Weather (0.25°): 0.570, 55.14, 0.728, 0.759
  - GraphCast (0.25°): 0.511, 48.72, 0.655, 0.683
  - Stormer (1.4°): 0.570, 53.77, 0.726, 0.760
  - IFS HRES: 0.513, 60.16, 0.833, 0.872
  - Reading: GraphCast is 15 to 20% better than ArchesWeather-M on 10 m wind and pressure at 24 h. The gap between model families is larger than the "few percent" assumed in the README, so blending has room.
- HFM track score to beat: 0.09749 (area-weighted normalised RMSE; this repository's README).

## Cheapest credible way to run experiments

- CPU only: read the stored Pangu forecasts for 2018-2022 and ERA5 truth from the bucket with xarray and gcsfs, select the four surface variables at the 24 h lead (one 240x121 field per variable per start time). Fit post-processing and blend weights on 2018-2021 and test on 2022. Add Aurora and FGN-mean for 2022 as extra blend members.
- ArchesWeather's own 2022 forecasts still need one GPU run (the paper states inference is cheap; training took about 9 V100-days). After that, every blending experiment is CPU work.

## Unverified leads

- Bremnes et al., neural-network calibration of Pangu-Weather site forecasts: named in a search summary, but no title or URL was confirmed.
- FengWu and FuXi-2.1 stored forecasts: not found in the bucket listing for 2022.
- Whether the stored Aurora and FGN forecasts start from ERA5 or from operational analyses was not confirmed; this affects a fair comparison with ERA5 truth.
