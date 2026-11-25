# Galaxy Photometric Redshift Prediction Across Disjoint Sky Regions

- **Field:** astronomy
- **HFM track:** `discovery_live_ai4x_astro_photoz` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** SDSS catalogue used by the experiment. Legacy Survey cutout service and AION-1 weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the catalogue experiment. A GPU for the image model.
- **Still missing:** A free GPU account for the image-based version.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Photometric redshift estimation predicts a galaxy's spectroscopic redshift from g, r, z images and catalogue photometry, using SDSS, BOSS, and eBOSS labels with Legacy Survey DR9 north cutouts. Held-out galaxies occupy sky regions absent from public galaxies, so evaluation tests transfer across disjoint sky areas rather than interpolation among neighboring galaxies.

- **Metric:** `normalized_rmse` (minimize)
- **HFM baseline:** 0.0364
- **HFM best so far:** 0.02434
- **HFM effort:** 36 proposals, 34 scored, 19 accepted improvements, about $2,428 of compute
- **HFM's current best idea:** Combining p3884's bounded readouts with p3885's equal mixture of full-data depth-6 and depth-8 trees at a fixed total tree weight of 10% reduces normalized RMSE on unseen sky regions relative to p3884.

## Existing theory and methods

Template fitting and empirical machine learning are the two standard families. Empirical methods (boosted trees, CNNs on images, mixture-density networks) lead where training spectra exist. Known issues are spatial covariate shift (depth, seeing, Galactic extinction) and catastrophic outliers.

## Public data

SDSS DR17 SpecPhoto catalogue through the public SkyServer SQL service (used here). Legacy Survey DR9 image cutouts and the AION-1 model are public but need a GPU.

## What limits progress

Region-to-region changes in survey depth and dust extinction shift the colour-redshift relation.

## Route to a step change

Give the model each galaxy's extinction and local depth. Fine-tune the image encoder with region-grouped validation.

## Experiment and results

Code: [`photoz_regions.py`](photoz_regions.py) in this folder. Run instructions are in its header.

- **Setup:** 198,350 galaxies with measured redshifts from the public SDSS
  DR17 catalogue, in 42 sky regions. Five-fold cross-validation with whole
  regions held out, so every galaxy is predicted by a model that never saw
  its region. Catalogue brightness only, no images, so errors are larger
  than HFM's.
- **Result:**

  | Method | NRMSE | Change | Core scatter (NMAD) | Outliers | Folds improved |
  | --- | --- | --- | --- | --- | --- |
  | Boosted-tree regression (base) | 0.04280 | — | 0.02342 | 1.22% | — |
  | Same, different random seed | 0.04285 | +0.10% | 0.02344 | 1.22% | 2/5 |
  | + Galactic extinction as an input | 0.04225 | **−1.29%** | 0.02251 | 1.21% | **5/5** |
  | Redshift distribution, mean | 0.04380 | +2.33% | 0.02226 | 1.37% | 0/5 |
  | Redshift distribution, median | 0.04506 | +5.27% | 0.02236 | 1.32% | 0/5 |
  | Blend of regression and distribution | 0.04238 | −0.99% | 0.02209 | 1.24% | 4/5 |

- **What this shows:**
  - **Extinction as an input is a real gain:** −1.29%, in every fold,
    against seed noise of 0.10%. It is small, but about 13 times the noise.
  - **Predicting a distribution does not lower NRMSE.** It tightens the
    core scatter by about 5% and makes the outliers worse, so on an RMSE
    metric it loses. Route 1 in Section 3.1 is not supported for HFM's
    metric. It would help only if the metric rewarded core scatter.

### Follow-up: every extra input the catalogue offers

Code: [`photoz_features.py`](photoz_features.py). The first experiment found that one added input, extinction, helped in every held-out region. This one adds the other things an image carries that five magnitudes do not, all taken from the public SDSS catalogue. Same protocol: 198,350 galaxies, 42 sky regions, whole regions held out.

| Inputs | NRMSE | Change | Core scatter (NMAD) | Outliers | Folds improved |
| --- | --- | --- | --- | --- | --- |
| Magnitudes and colours (base) | 0.04286 | — | 0.02337 | 1.22% | — |
| + extinction | 0.04224 | −1.4% | 0.02247 | 1.22% | 4/5 |
| + size and shape (radii, concentration, profile, axis ratio) | 0.04207 | −1.8% | 0.02309 | 1.12% | 5/5 |
| + per-band point-source and fibre magnitudes | 0.04195 | −2.1% | 0.02275 | 1.17% | 5/5 |
| + magnitude errors | 0.04253 | −0.8% | 0.02349 | 1.19% | 4/5 |
| All of the above | **0.04037** | **−5.8%** | 0.02148 | 1.05% | 5/5 |

- **The gains add up.** Each group of inputs is worth 1–2% alone and 5.8% together, in all five folds. That is more than four times the gain from extinction alone.
- **Both parts of the error improve:** the core scatter by 8% and the outlier rate from 1.22% to 1.05%.
- **This is the same pattern as elsewhere in this programme:** telling the model something it was not being told beats changing how it predicts (the redshift-distribution idea made things worse).
- HFM's version feeds images to a foundation model, which should capture size and shape already. Extinction, survey depth and measurement errors are not in the pixels, so those remain the candidates there.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

On catalogue data, seed noise is 0.1% of NRMSE and the extinction gain is 1.3%. Photometric noise sets a floor near the core scatter (NMAD about 0.022 here); the remaining RMSE is dominated by about 1.2% outliers.

## Next steps

Repeat with Legacy Survey images on a GPU, adding extinction, depth and error inputs that the pixels do not carry. Use the published DR9 photometric redshifts as one more input or blend member. Attack the outlier tail directly, since it dominates RMSE.
