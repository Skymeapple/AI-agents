# Patient-level lung cancer histology classification from H&E slides with Phikon-v2

- **Field:** medicine
- **HFM track:** `discovery_live_ai4x_pathology_lung` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU stand-in on public patches has run. Whole-slide, patient-level work with Phikon-v2 waits for a GPU.
- **Data access:** TCGA-LUAD/LUSC diagnostic slides through the GDC API and Phikon-v2 weights: Verified reachable from the free compute on 2026-11-25. NLST pathology images need a data-access request to the US National Cancer Institute.
- **Compute:** A GPU and tens to hundreds of GB of disk for slides; tile sampling keeps it within free limits.
- **Still missing:** A free GPU account, and an NLST data request for the held-out cohort (CPTAC can stand in until then).
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Patient-level classification distinguishes primary non-small-cell lung cancer as adenocarcinoma (ICD-O 8140/3) or squamous cell carcinoma (8070/3) from H&E whole-slide images using Phikon-v2. Training uses TCGA-LUAD/LUSC patients, development uses CPTAC-LUAD/LSCC, and held-out evaluation uses NLST patients.

- **Metric:** `patient_auroc` (maximize)
- **HFM baseline:** 0.9102
- **HFM best so far:** 0.9784
- **HFM effort:** 71 proposals, 69 scored, 15 accepted improvements, about $4,715 of compute
- **HFM's current best idea:** Shrinking p3977's inverse-density slide weights toward equal weighting according to their own effective sample size improves patient AUROC by preventing concentrated morphology weights from overrepresenting one slide.

## Existing theory and methods

Foundation-model tile embeddings with attention-based multiple-instance learning, stain normalisation and augmentation.

## Public data

TCGA-LUAD/LUSC and CPTAC slides are public (GDC, TCIA). NLST pathology needs a data request. Phikon-v2 is public.

## What limits progress

AUROC is near its ceiling; what is left is cohort shift and ambiguous cases.

## Route to a step change

Feature-level stain augmentation, a second foundation model as an extra view, calibrated abstention.

## Experiment and results

A CPU stand-in on public patches, aimed at a question that matters before any model work: how much of a high accuracy is leakage between near-identical images? Script: [`lc25000_floor.py`](lc25000_floor.py); log in `research/compute/results/lc25000-floor/`. It is patch-level, not patient-level, and is not comparable to HFM's score.

- **Data:** LC25000 lung patches (Borkowski et al. 2019; public copy on Hugging Face): 5,000 adenocarcinoma, 5,000 squamous cell carcinoma, 5,000 benign. The authors made each class by rotating and flipping 250 original images.
- **Method:** ImageNet ResNet-18 features averaged over the eight rotations and flips of each patch, then logistic regression, five-fold cross-validation. Folds are drawn either at random or by keeping near-duplicate groups together; groups link each patch to those of its ten nearest neighbours above a similarity threshold.
- **Duplicates are common:** for over a quarter of the patches the nearest other patch has a rotation-invariant feature identical to it (similarity 1.000), as expected for augmented copies.

| Grouping threshold | Groups | Adenocarcinoma vs squamous, random folds | Same, grouped folds |
|---|---|---|---|
| none | — | 0.995 | — |
| 0.99 | 10,850 | 0.995 | 0.991 |
| 0.97 | 5,816 | 0.995 | 0.980 |
| 0.95 | 3,484 | 0.994 | 0.965 |
| 0.90 | 1,178 | 0.995 | **0.936** |

(below 0.90 one group swallows over a fifth of the patches and grouped folds stop being meaningful)

- **The published near-perfect accuracy is mostly leakage.** Random folds give 99.5% for adenocarcinoma against squamous with nothing more than off-the-shelf features. Keeping near-duplicates together lowers it steadily; at the threshold whose group count (1,178) is closest to the 750 originals, it is 93.6%, and the error rate is twelve times higher (0.5% → 6.4%). The grouping is imperfect in both directions, so the true held-out figure is uncertain, but the direction and size of the leak are clear.
- **For HFM's track:** its data are patient-level whole slides, where the corresponding leak is patches or slides from one patient on both sides of the split. HFM's 0.978 is only meaningful if folds are drawn by patient (and ideally by hospital); this is worth confirming before trying to raise it.

## How far this can go

HFM is at 0.978. Pathologist agreement on adenocarcinoma versus squamous in poorly differentiated tumours is imperfect, so the true ceiling is likely 0.98 to 0.99.

## Next steps

Report sensitivity at fixed specificity and accuracy after abstention; AUROC can no longer separate methods.
