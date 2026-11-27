# References: PM2.5 Sensor Calibration Transfer

Every entry below was confirmed on 2026-11-25 by a web search result or by fetching the arXiv or Zenodo record. Items not confirmed are under "Unverified leads".

## Theory and established methods

- **Barkjohn, Gantt, Clements (2021).** Development and Application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor. Atmospheric Measurement Techniques 14, 4617-4637. https://amt.copernicus.org/articles/14/4617/2021/
  - The standard linear-plus-humidity correction; built from about 12,000 24-hour pairs in 16 states. More complex multiplicative models did not help on independent data, which is a warning for this project.
- **Di Antonio et al. (2018).** Developing a Relative Humidity Correction for Low-Cost Sensors Measuring Ambient Particulate Matter. Sensors 18, 2790. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6164928/
  - Argues the humidity correction must act on the particle size distribution, not on mass; this is the README's "infer hygroscopicity from size bins" route.
- **Crilley et al. (2020).** κ-Köhler humidity correction of low-cost optical particle counter (OPC-N2) data tested in four cities on three continents (descriptive title; exact title not captured). Atmospheric Measurement Techniques 13, 1181. https://amt.copernicus.org/articles/13/1181/2020/
  - Field test of the κ-Köhler correction across sites; correction is needed mainly above 60% relative humidity.
- **Wallace, Zhao, Klepeis (2022).** Calibration of PurpleAir PA-I and PA-II Monitors Using Daily Mean PM2.5 Concentrations Measured in California, Washington, and Oregon from 2017 to 2021. Sensors 22, 4741. Cited in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11679817/
  - Source of the ALT-CF3 calibration, which works from the particle-count bins instead of the manufacturer's mass output.

## Recent advances and breakthroughs (2023-2026)

- **Barkjohn et al. (2022, corrected 2024).** Correction and Accuracy of PurpleAir PM2.5 Measurements for Extreme Wildfire Smoke. Sensors 22, 9669. https://www.mdpi.com/1424-8220/22/24/9669 (correction: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11679817/)
  - Extends the US-wide correction to high concentrations; the sensor is linear to about 200 µg/m³. The 2024 correction withdraws one comparison because another study had mislabeled its data.
- **Poggi, Portier, Thulliez (2025).** Geographically Weighted Regression for Air Quality Low-Cost Sensor Calibration. arXiv:2510.05646. https://arxiv.org/abs/2510.05646
  - Spatially varying calibration tested on the SensEURCity Antwerp subset (9 reference stations, 34 sensors); a direct competitor for the hierarchical cross-site idea on the same data.
- **Yin, Gersey, Zhang (2025).** In-field Calibration of Low-Cost Sensors through XGBoost & Aggregate Sensor Data. arXiv:2506.15840. https://arxiv.org/abs/2506.15840
  - Uses neighbouring sensors as inputs to a boosted-tree calibration; a baseline for network-level transfer.
- **AMT paper on the Denver "Love My Air" network (2022).** Transferability metrics for low-cost sensor calibration models (descriptive title; exact title not captured). Atmospheric Measurement Techniques 15, 6309. https://amt.copernicus.org/articles/15/6309/2022/
  - Defines metrics for how a calibration performs away from its co-location site and discusses leave-one-site-out validation; use these metrics alongside the HFM skill.

## Datasets and benchmarks

- **Van Poppel, Schneider, Haugen et al. (2023).** SensEURCity: A multi-city air quality dataset collected for 2020/2021 using open low-cost sensor systems. Scientific Data 10, 322. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10220078/
  - Data descriptor: 85 AirSensEUR systems in Antwerp, Oslo and Zagreb with two co-location campaigns per city.
- **Yatkin, Gerboles, Van Poppel, Schneider et al. (2025).** SensEURCity dataset, version 3.0.0. Zenodo. https://doi.org/10.5281/zenodo.17858205
  - The current data release (one 5.7 GB zip, CC-BY 4.0). Version 3 fixes temperature and humidity column labels and adds a PM2.5 flag column, so use v3, not v2 (https://zenodo.org/records/7669644).
- **US EPA (2022).** Dataset: Correction and Accuracy of PurpleAir PM2.5 Measurements for Extreme Wildfire Smoke. https://catalog.data.gov/dataset/dataset-correction-and-accuracy-of-purpleair-pm2-5-measurements-for-extreme-wildfire-smoke
  - Public PurpleAir and reference pairs under smoke; a small, quick-to-load second aerosol type for testing transfer.

## Published results to beat

- Barkjohn et al. (2021): the US-wide correction cuts 24-hour RMSE from 8 to 3 µg/m³ at a mean reference concentration of 9 µg/m³ (source: AMT abstract, URL above).
- Crilley et al. (2020): κ-Köhler-corrected OPC-N2 PM2.5 within 33% of reference at all four sites (source: AMT 13, 1181).
- Di Antonio et al. (2018): size-distribution-based correction reduces overestimation from a factor of about 5 to 1.05 (source: PMC6164928).

## Unverified leads

- UNVERIFIED: Petters and Kreidenweis (2007), the κ-Köhler single-parameter hygroscopicity paper.
- UNVERIFIED: Abu-Hani et al. (2024, AMT) on transferability of global machine-learning calibration models for NO2 and NO sensors.
- UNVERIFIED: Malings et al. on fine particle mass monitoring with low-cost sensors and humidity/temperature corrections.
- UNVERIFIED: US EPA performance testing protocols and target values for PM2.5 air sensors (2021 report).
- UNVERIFIED: PurpleAir co-location sets on Zenodo and the EPA AirSensor data named in the README; no record was checked in this session.
