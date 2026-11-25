# References: Estimating Wave-Energy Resources from Short Measurement Campaigns

Status: the shared web-search budget ran out before this project was searched, and no source for it could be confirmed in this session. There are no confirmed entries. The list below is a set of leads written from memory; each one must be checked (title, authors, URL) before it is cited or relied on.

## Unverified leads

- UNVERIFIED: NOAA NDBC historical standard meteorological data and its measurement descriptions (the data used in this folder), plus NDBC spectral wave density files, which would allow a sea/swell split per buoy.
- UNVERIFIED: ERA5 reanalysis (Hersbach et al., 2020, QJRMS) wave parameters on the Copernicus Climate Data Store, including separate wind-sea and swell height and period; this is the external swell input the README calls for.
- UNVERIFIED: NOAA WAVEWATCH III production hindcasts and the WAVEWATCH III model code.
- UNVERIFIED: US DOE/NREL/PNNL high-resolution wave hindcast for US coastal waters, served through the Marine Energy Atlas and an open cloud data release.
- UNVERIFIED: IEC TS 62600-101, the technical specification for wave energy resource assessment and characterisation, which sets the metrics and minimum record lengths.
- UNVERIFIED: MHKiT (Marine and Hydrokinetic Toolkit, NREL/Sandia/PNNL), Python code for wave resource statistics and NDBC data access.
- UNVERIFIED: measure-correlate-predict literature from wind energy (for example Carta, Velázquez and Cabrera's review) for the short-campaign-to-long-term method family.
- UNVERIFIED: Hasselmann et al. (1973) JONSWAP report and the Sverdrup-Munk-Bretschneider growth curves for the fetch-limited wind-sea part.
- UNVERIFIED: 2023-2026 machine-learning wave models that map wind fields over a basin to buoy wave height, which would bring in remote swell generation without running a spectral model.
- UNVERIFIED: studies of inter-annual variability of wave power at US buoys, which bound how well any 60-day campaign can predict an annual mean.

## Second pass (confirmed by web search, 25 November 2026)

- **Perez-Molina, Carta (2025). Use of Machine-Learning Techniques to Estimate Long-Term Wave Power at a Target Site Where Short-Term Data Are Available. Journal of Marine Science and Engineering 13(6), 1194.** https://www.mdpi.com/2077-1312/13/6/1194
  - The same task as this project with ERA5 as the long-term reference: models trained on a short buoy record plus concurrent ERA5, then run on ERA5 alone. Multiple linear regression was best (MAE 2.56 against 4.38 kW/m for raw ERA5). Supports the finding here that an outside wave source, not a better model, moves the result.
- **Study of ERA5 against buoys in the Central Tyrrhenian Sea (Energies, Dec 2024).** https://doaj.org/article/3b2442aa4bb44a629da8d1d9624a5ecc
  - Reports ERA5 underestimating wave power by up to 42% against buoy data, so a hindcast must be corrected with the short campaign, not used raw.
- **Uncertainty in wave hindcasts in the North Atlantic Ocean (arXiv 2203.11715).** https://arxiv.org/pdf/2203.11715
  - Quantifies disagreement between hindcast products, which bounds what a hindcast-based reference can deliver.

