# Estimating Wave-Energy Resources from Short Measurement Campaigns

- **Field:** physics
- **HFM track:** `discovery_live_live_physics_wave_resource` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** NOAA NDBC buoy archive.
- **Compute:** CPU.
- **Still missing:** Nothing. A swell hindcast (ERA5 waves) would need a free Copernicus account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reusable method for estimating a site's wave-energy resource beyond the period covered by its wave measurements. A short campaign provides paired wave and wind observations, while a longer assessment period provides wind observations only. The method selects a model, representation, and estimation procedure to maximize a fixed evaluation score.

- **Metric:** `skill_mean` (maximize)
- **HFM baseline:** 0
- **HFM best so far:** 0.5602
- **HFM effort:** 100 proposals, 95 scored, 19 accepted improvements, about $8,217 of compute
- **HFM's current best idea:** Penalizing the p94 gate when the target-weighted p94 and p97 predictions disagree strongly in log space increases audited skill_mean over the unchanged p3600 gate on the frozen practice evaluation.

## Existing theory and methods

Measure-correlate-predict from wind energy, fetch-limited wave growth laws (SMB, JONSWAP), sea and swell partitioning, hindcast models (WAVEWATCH III, ERA5 waves).

## Public data

NOAA NDBC historical standard meteorological files for 16 buoys, 2021-2023 (public domain).

## What limits progress

Swell comes from distant storms and local wind carries no information about it.

## Route to a step change

Add a swell input from outside the site, such as a public hindcast.

## Experiment and results

Code: [`wave_resource.py`](wave_resource.py) in this folder. Run instructions are in its header.

- **Setup:** 16 NOAA buoys (Pacific, Atlantic, Gulf of Mexico, Hawaii),
  2021–2023. A 60-day campaign has wind and waves, and the next 365 days
  have wind only. 200 campaign instances. The reference assumes the
  campaign's mean wave power holds for the year. No settings were tuned.
- **Result:**

  | Method | Error in annual mean power | Skill, annual mean | Skill, hourly | Stations improved |
  | --- | --- | --- | --- | --- |
  | Reference | 38.5% | 0 | 0 | — |
  | Log-regression on wind | 29.3% | **+0.216** | +0.128 | 14/16 |
  | Boosted trees on wind | 29.6% | +0.221 | +0.094 | 14/16 |
  | Additive swell + wind-sea model | 30.9% | +0.182 | +0.124 | 14/16 |

- **What this shows:** using the wind record cuts the error in the annual
  resource by about a fifth, at 14 of 16 sites. The physically structured
  swell-plus-wind-sea model from Section 7.2 does **no better** than a
  plain regression. With local wind as the only long-term input, swell
  can only enter as a constant, so the structure adds nothing. A real
  step change would need a swell input from outside the site, such as a
  public wave hindcast. That is a change of data source, not of model.

### Follow-up: add information from outside the site

The first experiment concluded that local wind cannot see swell, and that the limit would only move with a swell input from elsewhere. That was tested using another buoy in the same sea area as the outside source. Its wave record over the assessment year stands in for a long-term reference record or a public hindcast. The partner buoy is chosen inside the campaign (highest correlation), so nothing from the future is used. 186 campaign instances at 16 buoys.

| Method | Error in annual mean power | Skill, annual mean | Skill, hourly | Stations better than wind only |
| --- | --- | --- | --- | --- |
| Campaign mean (reference) | 38.7% | 0 | 0 | — |
| Regression on local wind | 29.1% | +0.226 | +0.132 | — |
| Regression on local wind and a neighbouring buoy's waves | **19.3%** | **+0.479** | **+0.330** | 12/16 |

- **The wind-only floor of about 30% is broken.** Outside wave information cuts the error in the annual resource by a third and more than doubles both skills.
- **This confirms the diagnosis.** Three model families stopped at the same error on wind alone; one extra data source moved it by ten points.
- **A hindcast should do at least as well** where no neighbouring buoy exists, since it covers every site. That is the next test and needs a free Copernicus account for ERA5 wave data.
- HFM's task supplies wind only for the assessment period. Under that rule the earlier result stands; this shows what the rule costs.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Using local wind alone, three different model families all stop at about 29 to 31% error in annual mean power (reference 38.5%). That agreement suggests about 30% is the floor for wind-only methods. With a neighbouring buoy's wave record added, the error falls to 19%.

## Next steps

Replace the neighbouring buoy with ERA5 or WAVEWATCH III wave fields, which exist for every site, and check the gain holds (19% against 29% here with a buoy).
