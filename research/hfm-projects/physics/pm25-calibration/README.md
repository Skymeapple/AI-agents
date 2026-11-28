# PM2.5 Sensor Calibration Transfer

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_pm_calibration_transfer_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU (on free compute; the dataset is 5.7 GB).
- **Data access:** SensEURCity on Zenodo (5.7 GB): Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU on free compute.
- **Still missing:** Nothing.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

An air agency calibrates low-cost particle sensors to report hourly PM2.5 mass (µg/m³), monthly mean for standard comparison, and concentration tallies for advisories and exposure studies.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.6945
- **HFM effort:** 31 proposals, 29 scored, 10 accepted improvements, about $871 of compute
- **HFM's current best idea:** 保持 p1531 的湿度公式、健康选择和训练目标不变，仅以当前实例的环境温度中位数调节有界增益，可提高原始全部57个月的留一站平均官方 skill。

## Existing theory and methods

Barkjohn (US EPA) correction, kappa-Koehler hygroscopic growth, hierarchical calibration across sites.

## Public data

SensEURCity (Zenodo 17858205, CC-BY 4.0), one 5.7 GB zip: one-minute data from 85 sensor boxes in three cities with co-located reference PM2.5.

## What limits progress

The sensor's response to particles is non-linear and varies with aerosol type, which a linear or simple humidity correction does not capture.

## Route to a step change

Infer hygroscopicity per site from particle-size bins; share calibration across sites hierarchically.

## Experiment and results

Code: [`pm25_transfer.py`](pm25_transfer.py) in this folder. It ran on free GitHub Actions compute; the log is in [`research/compute/results/pm25-transfer/`](../../../compute/results/pm25-transfer/run.log).

- **Setup:** SensEURCity, 85 low-cost sensor boxes in Antwerp, Oslo and Zagreb with co-located reference PM2.5: 309,000 sensor-hours at 36 reference sites. Each box has two optical sensors (Plantower PMS5003 and Alphasense OPC-N3) plus humidity and temperature. Two transfer tests: fit on two cities and test on the third; and fit at a city's main co-location station and test at its other reference stations.
- **Result (RMSE in µg/m³; skill is against the linear calibration):**

  | Test | Sensor | Raw | Linear | Linear + humidity | Hygroscopic growth | Boosted trees |
  | --- | --- | --- | --- | --- | --- | --- |
  | Unseen city: Antwerp | PMS5003 | 16.3 | 8.9 | 9.0 | 9.4 | **6.5** |
  | Unseen city: Oslo | PMS5003 | 10.0 | 6.3 | 6.2 | 6.3 | **5.5** |
  | Unseen city: Zagreb | PMS5003 | 12.9 | **6.2** | 6.3 | 6.2 | 7.2 |
  | Unseen city: Antwerp | OPC-N3 | 34.4 | 11.1 | 11.1 | 10.7 | **8.9** |
  | Unseen city: Oslo | OPC-N3 | 38.3 | 9.9 | 9.9 | 9.3 | **8.1** |
  | Unseen city: Zagreb | OPC-N3 | 55.8 | 9.7 | 9.8 | 10.9 | **9.7** |
  | Other sites: Antwerp (17) | PMS5003 | 21.2 | 10.1 | 10.0 | 10.1 | **3.6** |
  | Other sites: Oslo (7) | PMS5003 | 12.6 | 4.7 | **4.6** | 4.7 | 5.4 |
  | Other sites: Zagreb (9) | PMS5003 | 22.3 | 10.1 | 10.2 | 10.1 | **8.9** |
  | Other sites: Antwerp (17) | OPC-N3 | 47.3 | 11.0 | 11.0 | 9.5 | **7.0** |
  | Other sites: Oslo (7) | OPC-N3 | 55.2 | 9.5 | 9.4 | 9.4 | **6.5** |
  | Other sites: Zagreb (9) | OPC-N3 | 112.0 | 14.5 | 14.2 | 14.0 | **8.2** |

  | Method | Mean skill, unseen city | Better in | Mean skill, other sites | Better in |
  | --- | --- | --- | --- | --- |
  | Linear + humidity | −0.001 | 1/6 | +0.009 | 5/6 |
  | Hygroscopic growth form | −0.013 | 2/6 | +0.031 | 4/6 |
  | Boosted trees on reading, humidity, temperature | **+0.103** | 5/6 | **+0.288** | 5/6 |

- **What this shows:**
  - **Any calibration is far better than the raw reading,** which is off by a factor of two to ten in error.
  - **The physics-based humidity form proposed in the overview (section 7.7) is not supported.** Neither it nor a plain humidity term improves on a linear calibration when moved to a new city, and both help only slightly across sites.
  - **A flexible learned calibration on the same three inputs transfers well:** 10% better in an unseen city and 29% better at unseen sites, in 5 of 6 cases each. The relationship is non-linear in ways a simple growth law does not capture.
  - **It is not uniformly better:** it loses to linear in Zagreb for one sensor and at Oslo's other sites for one sensor.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Reached here:** 6–9 µg/m³ RMSE in an unseen city and 4–9 at unseen sites, hourly.
- **Floor:** reference monitors themselves agree to roughly 10%, and hourly PM2.5 in these cities is often 5–20 µg/m³, so a few µg/m³ is close to what hourly comparison can show.
- **Open:** the two cases where the learned calibration loses suggest a city-specific aerosol effect that three inputs do not carry.

## Next steps

- Add the particle-size bins both sensors report as inputs; they carry aerosol-type information that humidity does not.
- Fit a hierarchical model with a per-city or per-site offset, to handle the cases where the pooled model loses.
- Score on HFM's other targets: monthly means and counts of advisory-level hours.
