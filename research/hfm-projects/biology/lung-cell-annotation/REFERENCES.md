# References: Cross-Study Cell-Type Annotation with UCE on the Human Lung Cell Atlas

Checked 2026-11-25. Each entry was confirmed in this session by the arXiv record, the Zenodo record, or the project's own README on GitHub. Nothing is cited from memory. Where a README did not list individual authors, the releasing group is given in their place.

## Theory and established methods

- **Sikkema et al. (2023).** An integrated cell atlas of the lung in health and disease. Nature Medicine. https://www.nature.com/articles/s41591-023-02327-2
  - The atlas used by this task. Landing page with all download links: https://github.com/LungCellAtlas/HLCA.
- **Lotfollahi et al. (2021).** scArches (single-cell architecture surgery), a package for reference-based analysis of single-cell data. Nature Biotechnology, DOI 10.1038/s41587-021-01001-7. https://github.com/theislab/scarches
  - The method the atlas authors use to map new studies onto the reference and transfer labels; the supervised baseline UCE must beat.
- **scvi-tools (scverse).** scVI and scANVI implementations. Nature Biotechnology, DOI 10.1038/s41587-021-01206-w. https://github.com/scverse/scvi-tools
  - scANVI is the semi-supervised model behind the atlas embedding.
- **CellTypist (Teichmann lab).** Logistic-regression cell-type classifiers. Science, DOI 10.1126/science.abl5197. https://github.com/Teichlab/celltypist
  - A count-based linear baseline. The HLCA README states a pretrained "Human_Lung_Atlas" model is available at https://www.celltypist.org/models.
- **OnClass.** Ontology-based cell-type classification. https://github.com/wangshenguiuc/OnClass ; docs https://onclass.readthedocs.io/en/latest/
  - Ontology-aware classifier named in this project's README; the method class behind the "back off to the parent term" route.
- **scHPL (Michielsen et al., 2021).** Hierarchical progressive learning of cell identities in single-cell data. Nature Communications. https://www.nature.com/articles/s41467-021-23196-8 ; code https://github.com/lcmmichielsen/scHPL
  - Finds relationships between cell populations across datasets and builds a hierarchical classification tree (README); addresses study-to-study label conventions.
- **scib (Theis lab, 2021).** Benchmarking atlas-level data integration in single-cell genomics. Nature Methods, DOI 10.1038/s41592-021-01336-8. https://github.com/theislab/scib
  - Standard metrics for whether an embedding removes study effects while keeping cell types apart.

## Recent advances and breakthroughs (2023-2026)

- **UCE (Stanford SNAP group, 2023/2026).** Universal Cell Embeddings. Preprint https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1 ; the README cites a Nature version, https://www.nature.com/articles/s41586-026-10689-z. Code https://github.com/snap-stanford/UCE
  - The model named in the task. Weights are on Figshare (link in the README).
- **scGPT (Bo Wang lab, 2023).** scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI. https://www.biorxiv.org/content/10.1101/2023.04.30.538439 ; code https://github.com/bowang-lab/scGPT
  - Alternative foundation embedding; several pretrained checkpoints are released.
- **Geneformer (2023).** Nature, https://www.nature.com/articles/s41586-023-06139-9 ; weights https://huggingface.co/ctheodoris/Geneformer
  - The third foundation model commonly included in benchmarks.
- **Microsoft Research (2023).** Assessing the limits of zero-shot foundation models in single-cell biology. https://www.biorxiv.org/content/10.1101/2023.10.16.561085 ; code https://github.com/microsoft/zero-shot-scfoundation
  - Evaluates Geneformer and scGPT without fine-tuning. Read it before relying on a frozen embedding; it is the kind of benchmark this project's README cites for adding a count-based second view.
- **clinicalml (2023).** A Deep Dive into Single-Cell RNA Sequencing Foundation Models. bioRxiv, DOI 10.1101/2023.10.19.563100. https://github.com/clinicalml/sc-foundation-eval
  - Evaluation code and analysis for scBERT and scGPT.
- **Yuan, Zhan, Zhang et al. (2024).** Cell-ontology guided transcriptome foundation model (scCello). NeurIPS 2024 Spotlight. https://arxiv.org/abs/2408.12373
  - Foundation model guided by the Cell Ontology during pretraining. Code https://github.com/DeepGraphLearning/scCello, weights https://huggingface.co/collections/katarinayuan/sccello-67a01b6841f3658ba443c58a. A drop-in replacement for UCE that already encodes the hierarchy.
- **Pearce, Simmonds, Mahmoudabadi et al. (2025).** TranscriptFormer. https://www.biorxiv.org/content/10.1101/2025.04.25.650731v2 ; code https://github.com/czi-ai/transcriptformer
  - Newer cell model released under the czi-ai organisation; another candidate embedding.
- **Zhao, Zhang, Wu et al. (2024).** LangCell: Language-Cell Pre-training for Cell Identity Understanding. ICML 2024. https://arxiv.org/abs/2405.06708
  - Joint pretraining of cells and text aimed at cell identity; a possible route to labelling types that are missing from the training studies.
- **scTab (Theis lab).** De novo cell type prediction model trained across a large collection of curated datasets. https://github.com/theislab/scTab
  - Supervised cell-type prediction trained across many curated datasets; a non-foundation baseline design.

## Datasets and benchmarks

- **HLCA on CELLxGENE.** Core and full atlas with raw and normalised counts, a batch-corrected embedding and UMAP. https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293
  - The task data (CC-BY 4.0 per this project's README).
- **Sikkema (2023).** Human Lung Cell Atlas reference model, embedding and model extensions. Zenodo, DOI 10.5281/zenodo.7599104, CC-BY 4.0. https://zenodo.org/records/7599104
  - Contains `HLCA_full_v1.1_emb.h5ad` (2.4 GB): the precomputed scANVI embedding of the full atlas. Also `HLCA_reference_model.zip` (11 MB) and scArches surgery adapters for every extension dataset. zenodo.org is reachable from this environment.
- **Cell Ontology.** https://github.com/obophenotype/cell-ontology ; paper https://doi.org/10.1038/s41597-022-01886-2
  - The label hierarchy needed for parent-term back-off and for ontology-aware scoring.
- **CZ CELLxGENE Census.** https://github.com/chanzuckerberg/cellxgene-census
  - Python and R API for querying the CELLxGENE corpus without downloading whole files. Whether the atlas is included in the current Census release was not checked.

## Open-source code and tools

- **HLCA mapping tutorial (scArches).** https://github.com/theislab/scarches/blob/hlca_tutorial_improvements/notebooks/hlca_map_classify.ipynb
  - The authors' own label-transfer workflow; reproduce it first as the reference baseline.
- **popV (Yosef lab).** Consensus cell-type transfer by voting across several methods. https://github.com/YosefLab/popV ; pretrained models https://huggingface.co/popV
  - Ready-made ensemble of classifiers; agreement between its members is also a per-cell confidence score.
- **Helical.** https://github.com/helicalAI/helical
  - One package that runs UCE, scGPT, Geneformer and TranscriptFormer; shortens the work of adding a second foundation embedding.

## Breakthroughs in related fields that could transfer

- **Sagawa, Koh, Hashimoto et al. (2019).** Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. https://arxiv.org/abs/1911.08731
  - Group DRO, named in this project's route: train against the worst study. The paper shows it needs strong regularisation to work.

## Published results to beat

- No published study-level macro-F1 on a held-out-study split of the HLCA was confirmed in this session. The numbers to beat are HFM's: baseline 0.7812, best 0.8536.

## Cheapest credible way to run experiments

- CPU only, no foundation model: download `HLCA_full_v1.1_emb.h5ad` (2.4 GB) from Zenodo 7599104 and train classifiers on the scANVI embedding with whole studies held out. Check first what cell metadata the file carries (study, donor, labels); its contents were not inspected here.
- Caveat on that route: scANVI is semi-supervised, so the reference model has very likely seen cell-type labels from the atlas core (not checked against the paper here). Held-out core studies would then not be unseen by the embedding. Use extension studies as the test set, or treat scores as an upper bound.
- For raw counts, work on a per-study subsample and train a CellTypist-style logistic regression on highly variable genes; this fits in ordinary memory.
- UCE embeddings need a GPU once; after that all heads train on CPU.

## Unverified leads

- Precomputed UCE, scGPT or Geneformer embeddings hosted for the atlas (for example through CELLxGENE Census): the Census FAQ fetched here says the Census does not carry embeddings, which may be out of date. Not confirmed.
- scTab paper (believed to be in Nature Communications, 2024): title and URL not confirmed; only the repository was.
- Author lists for UCE, scGPT, Geneformer, CellTypist and the two benchmark preprints were not confirmed; they are cited by group.
- Inter-annotator or cross-study label agreement figures for the atlas: none confirmed.
