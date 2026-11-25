# Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas

- **Field:** biology
- **HFM track:** `discovery_live_ai4x_singlecell_hlca` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** Runs today on CPU.
- **Data access:** Human Lung Cell Atlas on CELLxGENE and UCE weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** A machine with 16 GB RAM can process the atlas in chunks; UCE embedding needs a GPU. The atlas file already carries scANVI embeddings, which gives a CPU-only route.
- **Still missing:** A free GPU account for UCE embeddings. The CPU route on stored embeddings runs now.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work examines automatic cell-type annotation for single-cell RNA-seq from previously unseen studies, using UCE with the Human Lung Cell Atlas full release: about 2.3 million cells from 35 studies annotated to Cell Ontology terms. Entire studies are split across training, development, and held-out evaluation, with no donor appearing in more than one.

- **Metric:** `study_macro_f1` (maximize)
- **HFM baseline:** 0.7812
- **HFM best so far:** 0.8536
- **HFM effort:** 50 proposals, 45 scored, 10 accepted improvements, about $5,980 of compute
- **HFM's current best idea:** 在p3944的伪标签置信度和研究与伪类别容量约束不变时，将每个研究与伪类别组的512行限额改为供体分层轮转选择，能比p3944的全局cell-ID哈希截断提高study_macro_f1。

## Existing theory and methods

Reference mapping (scArches, scANVI), ontology-aware classifiers (CellTypist, OnClass), foundation-model embeddings (UCE, scGPT, Geneformer). Independent benchmarks find frozen foundation-model embeddings often lose to simple count-based baselines across studies.

## Public data

Human Lung Cell Atlas core and full files on CELLxGENE (CC-BY 4.0); labels and the scANVI embedding can be read remotely without downloading the 5.9 GB file. Full-atlas embedding: Zenodo 7599104. UCE weights: Hugging Face `minwoosun/uce-100m`.

## What limits progress

A frozen general embedding with a shallow head. Label conventions differ between studies, and rare types dominate macro-F1.

## Route to a step change

Decode over the Cell Ontology hierarchy with back-off to the parent term. Add a count-based second view (highly variable genes). Train study-invariant representations (group DRO).

## Experiment and results

Code: [`cross_study_annotation.py`](cross_study_annotation.py) in this folder. It ran on free GitHub Actions compute; the log is in [`research/compute/results/hlca-cross-study/`](../../../compute/results/hlca-cross-study/run.log).

- **Setup:** the atlas core (584,944 cells, 11 studies, 61 finest-level
  cell types). Only the labels and the 30-dimension scANVI embedding
  stored in the public file are read, over HTTP, so nothing large is
  downloaded and no GPU is needed. Leave one study out: train on the
  others, label every cell of the held-out study at the finest level.
  Score is macro-F1 over the types present in the held-out study,
  averaged over the 11 studies.
- **Caveat:** the scANVI embedding was trained with labels from all
  studies, so absolute scores are optimistic. HFM uses UCE embeddings and
  the full 35-study atlas. The comparison between methods is still fair.
- **Result:**

  | Method | Study macro-F1 | Versus flat | Studies better | Rare-type F1 |
  | --- | --- | --- | --- | --- |
  | Flat logistic regression | **0.715** | — | — | 0.647 |
  | Class-balanced | 0.708 | −0.006 | 5/11 | 0.635 |
  | Hierarchy-aware (adds ancestor scores) | 0.714 | −0.001 | 5/11 | 0.644 |
  | Nearest neighbours | 0.709 | −0.006 | 5/11 | 0.653 |

- **What this shows:**
  - **None of the classifier changes helps.** Class balancing,
    hierarchy-aware decoding and nearest neighbours all land within 0.006
    of plain logistic regression and win on only 5 of 11 studies.
  - **The hierarchy idea from the overview (section 4.2) is not
    supported** on this embedding.
  - **The variation is between studies, not between methods.** Ten
    studies score 0.68–0.78 and one (Seibold 2020, nasal samples) scores
    0.49 with every method. What limits the score is how a study differs
    from the rest, which a classifier head cannot fix.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Classifier head:** exhausted on this embedding; four designs agree within 0.006.
- **Representation:** the remaining lever. The spread between studies (0.49 to 0.78) is the headroom, and it sits in the embedding and in label conventions.
- **Label ceiling:** studies annotate the same cells differently at the finest level, which bounds macro-F1 below 1. Measuring that disagreement is the way to estimate the true ceiling.

## Next steps

- Repeat on the full 35-study atlas using the public full-atlas embedding (Zenodo 7599104, 2.4 GB), which matches HFM's setting.
- Compare embeddings instead of classifier heads: UCE (needs a GPU), a count-based model trained without held-out-study labels, and their combination.
- Look at where Seibold 2020 fails: tissue (nasal), assay, or label convention.
