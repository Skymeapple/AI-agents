# Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas

- **Field:** biology
- **HFM track:** `discovery_live_ai4x_singlecell_hlca` (active on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Data verified; pipeline or simulator still to build.
- **Data access:** Human Lung Cell Atlas on CELLxGENE and UCE weights: Verified reachable from the free compute on 2026-11-25.
- **Compute:** A machine with 16 GB RAM can process the atlas in chunks; UCE embedding needs a GPU. The atlas file already carries scANVI embeddings, which gives a CPU-only route.
- **Still missing:** The loading and cross-study split pipeline. No external blocker for the CPU route.
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

Human Lung Cell Atlas full release on CELLxGENE (CC-BY 4.0). UCE weights are public.

## What limits progress

A frozen general embedding with a shallow head. Label conventions differ between studies, and rare types dominate macro-F1.

## Route to a step change

Decode over the Cell Ontology hierarchy with back-off to the parent term. Add a count-based second view (highly variable genes). Train study-invariant representations (group DRO).

## Experiment and results

No experiment has been run for this project.

- **Why:** The full Human Lung Cell Atlas (2.3 million cells) and the UCE model need a GPU and tens of GB of memory.

## How far this can go

Cross-study label disagreement bounds macro-F1. Inter-annotator agreement at the finest level of the atlas is well below 1.

## Next steps

Run on the atlas core (584k cells) with a subsample on a machine with 64 GB RAM. Measure label disagreement between studies to estimate the ceiling.
