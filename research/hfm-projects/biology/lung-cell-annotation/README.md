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

### Follow-up: using the held-out study's own unlabelled cells

HFM's agents use pseudo-labels on the unseen study, so four such methods were tested on the same 11 studies.

| Method | Study macro-F1 | Versus flat | Studies better |
| --- | --- | --- | --- |
| Flat logistic regression | 0.715 | — | — |
| Self-training (add confident held-out cells, refit) | 0.718 | +0.003 | 8/11 |
| Shift the study's embedding to the training mean | 0.614 | −0.100 | 0/11 |
| Smooth predictions over neighbouring cells | 0.586 | −0.129 | 0/11 |

- **Self-training gives a small, fairly consistent gain** (+0.003, 8 of 11 studies), the same order as HFM's agents report per step.
- **Smoothing and re-centring are harmful.** Rare types are overwritten by their common neighbours, and a study's mean differs for real biological reasons.
- Eight methods have now been tried on this embedding and none moves the score by more than 0.006. The representation is the limit.

### Follow-up: the full atlas, HFM's setting

The same experiment on the full release: 2.3 million cells, 35 studies, Cell Ontology labels, read remotely ([log](../../../compute/results/hlca-full-atlas/run.log)). Each study with at least 2,000 cells is held out in turn.

| Method | Study macro-F1 | Versus flat | Studies better |
| --- | --- | --- | --- |
| Flat logistic regression | 0.683 | — | — |
| Nearest neighbours (15) | **0.717** | **+0.034** | 30/35 |
| Class-balanced logistic regression | 0.648 | −0.035 | 9/35 |
| Self-training | 0.682 | −0.000 | 21/35 |
| Shift the study's embedding to the training mean | 0.609 | −0.074 | 1/35 |
| Smooth predictions over neighbouring cells | 0.555 | −0.128 | 0/35 |

- **On the full atlas the classifier head does matter.** Nearest neighbours beat the linear head by 0.034 on 30 of 35 held-out studies. On the 11-study core the two were level. With 35 studies there are more cell states near each new cell, and a local method uses them where a linear boundary cannot.
- **This is the first change that moves this project,** and it matches an idea on HFM's own list ("replace the linear head") that its record shows was not taken far.
- The other methods behave as on the core: class balancing, re-centring and smoothing hurt, self-training is neutral.
- A follow-up comparing more non-linear heads is queued.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

### Follow-up: non-linear classifier heads on the full atlas

Log: [`research/compute/results/hlca-full-heads/`](../../../compute/results/hlca-full-heads/run.log). Same protocol as the full-atlas run (35 studies, each held out in turn), with 12,000 training cells per study instead of 6,000.

| Classifier on the stored embedding | Study macro-F1 | Versus linear | Studies better | Rare-type F1 |
| --- | --- | --- | --- | --- |
| Linear (flat) | 0.684 | — | — | 0.607 |
| Class-balanced linear | 0.650 | −0.034 | 9/35 | 0.562 |
| Nearest neighbours, k=15 | 0.733 | +0.050 | 34/35 | 0.677 |
| Linear + neighbours averaged | 0.737 | +0.054 | 35/35 | 0.676 |
| Small neural network (two hidden layers) | 0.768 | +0.085 | 35/35 | 0.696 |
| **Network + neighbours averaged** | **0.786** | **+0.102** | **35/35** | **0.724** |

- **The classifier head matters far more on the full atlas than on the core.** A small network averaged with nearest neighbours lifts macro-F1 by 0.10, in every one of the 35 held-out studies, and rare types gain most (+0.12).
- **Different heads add:** the network and the neighbours each beat the linear head, and their average beats both.
- **Class balancing hurts** on the full atlas (−0.034), the opposite of the usual advice for a macro-F1 score.
- **Caveat, unchanged:** the stored embedding was trained with labels from every study, so absolute scores are optimistic and are not comparable with HFM's UCE-based 0.71 range. The comparison between heads on the same embedding is fair. Whether the gain carries to UCE embeddings needs a GPU to compute them.

## How far this can go

- **Classifier head:** exhausted on this embedding; four designs agree within 0.006.
- **Representation:** the remaining lever. The spread between studies (0.49 to 0.78) is the headroom, and it sits in the embedding and in label conventions.
- **Label ceiling:** studies annotate the same cells differently at the finest level, which bounds macro-F1 below 1. Measuring that disagreement is the way to estimate the true ceiling.

## Next steps

- Repeat on the full 35-study atlas using the public full-atlas embedding (Zenodo 7599104, 2.4 GB), which matches HFM's setting.
- Compare embeddings instead of classifier heads: UCE (needs a GPU), a count-based model trained without held-out-study labels, and their combination.
- Look at where Seibold 2020 fails: tissue (nasal), assay, or label convention.
