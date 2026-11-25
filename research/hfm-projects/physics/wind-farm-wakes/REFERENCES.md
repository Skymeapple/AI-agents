# References: Wind-Farm Wake Calibration and Power Prediction from Real SCADA Data

Status: the shared web-search budget ran out before this project was searched. The entries below were confirmed on 2026-11-25 by fetching the Zenodo and arXiv records. Everything else is under "Unverified leads" and must be checked before use.

## Datasets and benchmarks

- **Plumley, Takeuchi (2025).** Kelmarsh wind farm data. Zenodo, CC-BY 4.0. https://doi.org/10.5281/zenodo.16807551
  - Six Senvion MM92 turbines, 10-minute SCADA and events from 2016 to end 2024, plus turbine coordinates, substation (PMU) and grid-meter data. The README uses only 2017; eight more years are available for calibration and for testing drift.
- **Plumley, Takeuchi (2025).** Penmanshiel wind farm data. Zenodo, CC-BY 4.0. https://doi.org/10.5281/zenodo.16807304
  - Fourteen Senvion MM82 turbines (there is no WT03), 2016 to end 2024, same layout of files; the unseen-farm test site.
- **Plumley (2022).** Kelmarsh wind farm data, first release. Zenodo. https://doi.org/10.5281/zenodo.5841834
  - The earlier record that most published studies of this farm cite; use it to match their data period.
- **Plumley (2022).** Penmanshiel Wind Farm Data, first release. Zenodo. https://doi.org/10.5281/zenodo.5946808
  - Earlier record for the second farm.

## Breakthroughs in related fields that could transfer

- **Rybchuk, Hassanaly, Hamilton, Doubrawa et al. (2023).** Ensemble flow reconstruction in the atmospheric boundary layer from spatially limited measurements through latent diffusion models. Physics of Fluids 35, 126604. https://arxiv.org/abs/2303.00836
  - Recovers an ensemble of inflow fields from limited measurements; relevant because poorly known inflow direction is this project's main limit, and an inflow ensemble could replace a single direction offset.

## Published results to beat

- No published per-turbine power error for Kelmarsh or Penmanshiel was confirmed. The numbers to beat are this folder's: −29.8% error against no-wake at the unseen farm for the calibrated wake model plus boosted-tree residual (README), and HFM's reward of 0.4124.

## Unverified leads

- UNVERIFIED: FLORIS (NREL), the wake engine frozen in the HFM protocol, and its documentation of the Gauss-curl-hybrid, TurbOPark and empirical Gaussian models.
- UNVERIFIED: FLASC (NREL), a SCADA analysis package with northing (yaw/direction bias) calibration and energy-ratio tools built for FLORIS model tuning; if it works as recalled, it covers the README's "calibrate direction first" step.
- UNVERIFIED: Bastankhah and Porté-Agel (2014), the Gaussian wake model, and Jensen (1983) / Katic et al. (1986), the top-hat model used in this folder's stand-in.
- UNVERIFIED: Niayifar and Porté-Agel (2016), wake growth rate as a function of turbulence intensity, a physical route to a wake decay that varies with conditions.
- UNVERIFIED: PyWake (DTU) as an independent wake engine, and OpenOA (NREL) for SCADA-based operational analysis.
- UNVERIFIED: `wind-up` (RES Group), an open tool that uses Kelmarsh data in its examples to measure turbine uplift.
- UNVERIFIED: SMARTEOLE wake-steering field campaign data (open SCADA with documented wake cases) as a third farm for testing transfer.
- UNVERIFIED: 2023-2026 papers on Bayesian or machine-learning calibration of engineering wake models from SCADA, and graph neural network farm-flow surrogates.
- UNVERIFIED: ERA5 or other reanalysis wind direction as an independent inflow reference for the direction-offset calibration.
