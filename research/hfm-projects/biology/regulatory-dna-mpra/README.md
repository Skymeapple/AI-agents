# Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA

- **Field:** biology
- **HFM track:** `discovery_live_ai4x_genomics_mpra` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** lentiMPRA tables on Zenodo used by the experiment. Evo 2 7B weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the CNN experiment (full-data run on free compute). A GPU for Evo 2.
- **Still missing:** A free GPU account for Evo 2 features; the 7B model may exceed a 16 GB GPU, so the smaller Evo 2 checkpoints are the realistic start.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Prediction of human regulatory DNA activity uses Evo 2 7B with lentiviral massively parallel reporter assays (lentiMPRA) in HepG2, K562, and WTC11 cells. Each 230 bp construct contains a 200 bp insert plus fixed adapters; activity is the mean log2 RNA/DNA ratio across replicates.

- **Metric:** `assay_macro_spearman` (maximize)
- **HFM baseline:** 0.4465
- **HFM best so far:** 0.7076
- **HFM effort:** 65 proposals, 61 scored, 32 accepted improvements, about $5,220 of compute
- **HFM's current best idea:** Widening the mean-driven channel gates from 16 to 32 hidden units only in four-block ranking members with seeds 17 and 29 improves held-out human assay_macro_spearman over the unchanged audit-passed p3900 ensemble.

## Existing theory and methods

Sequence-to-activity CNNs (MPRA-DragoNN, MPRALegNet), multi-task heads across cell types, reverse-complement equivariance and test-time averaging, and fine-tuning of DNA language models (Enformer, Borzoi, Evo 2).

## Public data

Agarwal et al. lentiMPRA tables, Zenodo 10558183 (CC-BY 4.0), with the authors' ten folds. Evo 2 weights are public but the 7B model needs a GPU.

## What limits progress

Each cell line is learned mostly on its own, and Evo 2 enters only as frozen features.

## Route to a step change

One multi-task model with a head per cell line, reverse-complement averaging, then parameter-efficient fine-tuning of Evo 2.

## Experiment and results

Code: [`mpra_multitask.py`](mpra_multitask.py) in this folder. Run instructions are in its header.

- **Setup:** the public lentiMPRA tables of Agarwal et al. (Zenodo
  10558183), the same assay and cell lines as HFM's task. A small CNN
  trained from scratch on CPU, at most 40,000 training sequences per cell
  line, the authors' fold 10 as the test set, two seeds. Sequences in any
  cell line's test fold are removed from all training sets, because the
  libraries overlap and their folds differ.
- **Result (Spearman on the test fold):**

  | Model | HepG2 | K562 | WTC11 | Macro |
  | --- | --- | --- | --- | --- |
  | One model per cell line | 0.549 | 0.554 | 0.504 | 0.535 |
  | + reverse-complement averaging | 0.566 | 0.571 | 0.519 | 0.552 |
  | One multi-task model | 0.595 | 0.589 | 0.549 | 0.578 |
  | Multi-task + reverse-complement averaging | 0.610 | 0.604 | 0.563 | **0.593** |

- **What this shows:** both ideas from Section 4.1 hold. Multi-task
  training adds 0.042 and reverse-complement averaging adds 0.015. Each
  gain appears in every cell line and both seeds (seed-to-seed spread is
  about 0.005 to 0.010). The largest multi-task gain is on WTC11, the
  smallest dataset, as predicted. Absolute scores are far below HFM's
  0.708, because this model is small and uses no Evo 2 features.

### Follow-up: all training data, three seeds

Run on free GitHub Actions compute; log in [`research/compute/results/mpra-full-data/`](../../../compute/results/mpra-full-data/run.log). All training sequences per cell line (not 40,000), four epochs, three seeds.

| Model | HepG2 | K562 | WTC11 | Macro |
| --- | --- | --- | --- | --- |
| One model per cell line | 0.631 | 0.658 | 0.508 | 0.599 |
| + reverse-complement averaging | 0.653 | 0.676 | 0.528 | 0.619 |
| One multi-task model | 0.662 | 0.667 | 0.589 | 0.640 |
| Multi-task + reverse-complement averaging | 0.682 | 0.686 | 0.604 | **0.657** |

- **Both gains hold with full data:** multi-task +0.039 macro, reverse-complement averaging +0.018 to +0.020.
- **Seeds agree to within 0.002,** so the gains are about 20 times the seed noise.
- **WTC11 gains most (+0.076),** as it has the least data of its own.
- A small CNN trained from scratch on CPU reaches 0.657 on the authors' test fold. HFM's baseline and best are 0.447 and 0.708 on its own split, which is not the same split.

### Follow-up: ensemble of five seeds

Five seeds of each model, predictions averaged ([log](../../../compute/results/mpra-full-data/run.log)).

| Model | HepG2 | K562 | WTC11 | Macro |
| --- | --- | --- | --- | --- |
| Separate models, five-seed ensemble | 0.672 | 0.690 | 0.557 | 0.640 |
| Multi-task model, five-seed ensemble | 0.703 | 0.701 | 0.613 | **0.672** |

- **Ensembling adds 0.015** to the multi-task model (0.657 → 0.672) and 0.021 to the separate models.
- **The multi-task advantage survives ensembling:** +0.033 macro, +0.056 on WTC11.
- **Step by step from the start of this project:** separate small models 0.535 → reverse-strand averaging 0.552 → one multi-task model 0.593 → all training data 0.657 → five-seed ensemble 0.672. Each ingredient added on top of the others.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

Replicate-to-replicate reproducibility of the assay bounds Spearman (the source paper reports replicate correlations around 0.76 to 0.98 depending on cell line). HFM's 0.708 is already near the low end of that range for WTC11-like noise.

## Next steps

Run the multi-task head on Evo 2 embeddings on a GPU. Add shift augmentation. Measure the replicate ceiling per cell line from the non-averaged tables.
