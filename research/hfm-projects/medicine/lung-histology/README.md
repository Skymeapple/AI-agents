# Patient-level lung cancer histology classification from H&E slides with Phikon-v2

- **Field:** medicine
- **HFM track:** `discovery_live_ai4x_pathology_lung` (active on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
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

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. Whole-slide images (TCGA, CPTAC, NLST) are hundreds of GB.

## How far this can go

HFM is at 0.978. Pathologist agreement on adenocarcinoma versus squamous in poorly differentiated tumours is imperfect, so the true ceiling is likely 0.98 to 0.99.

## Next steps

Report sensitivity at fixed specificity and accuracy after abstention; AUROC can no longer separate methods.
