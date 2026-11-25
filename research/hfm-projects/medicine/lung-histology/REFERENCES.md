# References: Patient-level lung cancer histology classification from H&E slides with Phikon-v2

Checked 2026-11-25. Each entry was confirmed in this session by the arXiv record (and, where a number is quoted, the arXiv abstract or HTML text) or by the project's own README on GitHub. Nothing is cited from memory. Where a README did not list individual authors, the releasing group is given in their place.

## Theory and established methods

- **Ilse, Tomczak, Welling (2018).** Attention-based Deep Multiple Instance Learning. ICML 2018. https://arxiv.org/abs/1802.04712
  - ABMIL: the attention pooling over tile embeddings that this task's pipeline uses. Code: https://github.com/AMLab-Amsterdam/AttentionDeepMIL.
- **Lu, Williamson, Chen et al. (2020).** Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images (CLAM). https://arxiv.org/abs/2004.09666
  - CLAM, the attention-based weakly supervised pipeline named in the task brief; the usual baseline next to ABMIL.
- **Shao, Bian, Chen et al. (2021).** TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification. NeurIPS 2021. https://arxiv.org/abs/2106.00908
  - Models relations between tiles with self-attention. Code: https://github.com/szc19990412/TransMIL.
- **Li, Li, Eliceiri (2020).** Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning (DSMIL). CVPR 2021. https://arxiv.org/abs/2011.08939
  - Dual-stream aggregator. Code: https://github.com/binli123/dsmil-wsi.
- **Tellez, Litjens, Bandi et al. (2019).** Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis. https://arxiv.org/abs/1902.06543
  - The reference comparison of stain augmentation against stain normalisation across laboratories.
- **Shen, Luo, Shen et al. (2022).** RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization. DOI 10.1007/978-3-031-16434-7_21. https://arxiv.org/abs/2206.12694
  - Combines stain augmentation and normalisation in one step.

## Recent advances and breakthroughs (2023-2026)

- **Filiot, Jacob, Mac Kain et al. (2024).** Phikon-v2, A large and public feature extractor for biomarker prediction. https://arxiv.org/abs/2409.09173
  - The encoder named in the task: DINOv2 ViT-L trained on 460 million tiles from over 100 public cohorts. The pretraining cohort table in the paper lists TCGA-LUAD, TCGA-LUSC, CPTAC-LSCC and "NLST-pathology" (1,225 slides). The encoder has therefore seen unlabelled images from all three cohorts of this task, including the held-out one.
- **Owkin (2023).** Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling (Phikon). medRxiv, https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2 ; code https://github.com/owkin/HistoSSLscaling
  - The first Phikon (ViT-B, iBOT, 40 million TCGA tiles). The repository releases precomputed tile features for 11 TCGA cohorts.
- **Chen, Ding, Lu et al. (2023/2024).** A General-Purpose Self-Supervised Model for Computational Pathology (UNI). Nature Medicine, https://www.nature.com/articles/s41591-024-02857-3 ; preprint https://arxiv.org/abs/2308.15474
  - Evaluates NSCLC subtyping trained on TCGA and tested on CPTAC, the same train/development split as this task. Code: https://github.com/mahmoodlab/UNI.
- **Vorontsov, Bozkurt, Casson et al. (2023).** Virchow: A Million-Slide Digital Pathology Foundation Model. Nature Medicine, DOI 10.1038/s41591-024-03141-0. https://arxiv.org/abs/2309.07778
  - Encoder trained on about a million slides. A candidate second view; check its training data for overlap with the three cohorts. Weights: https://huggingface.co/paige-ai/Virchow.
- **Zimmermann, Vorontsov, Viret et al. (2024).** Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology. https://arxiv.org/abs/2408.00738
  - Trained on 3.1 million slides from diverse institutions and stains. Weights: https://huggingface.co/paige-ai/Virchow2.
- **Lu, Chen, Williamson et al. (2023).** Towards a Visual-Language Foundation Model for Computational Pathology (CONCH). https://arxiv.org/abs/2307.12914
  - Image-text model; classifies lung subtypes without any training slides. Code: https://github.com/mahmoodlab/CONCH.
- **Ding, Wagner, Song et al. (2024).** Multimodal Whole Slide Foundation Model for Pathology (TITAN). https://arxiv.org/abs/2411.19666
  - Gives one embedding per slide, which removes the aggregator. Code: https://github.com/mahmoodlab/TITAN.
- **Shaikovski, Casson, Severson et al. (2024).** PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology. https://arxiv.org/abs/2405.10254
  - Slide-level foundation model; an alternative to training an aggregator.
- **Neidlinger, El Nahhas, Muti et al. (2024).** Benchmarking foundation models as feature extractors for weakly-supervised computational pathology. https://arxiv.org/abs/2408.15823
  - 19 encoders on 13 cohorts including lung. Abstract: models trained on different cohorts learn complementary features, and a CONCH plus Virchow2 ensemble beat single models in 55% of tasks. Direct support for the "second foundation model as an extra view" route.
- **Wölflein, Ferber, Meneghetti et al. (2023).** Benchmarking Pathology Feature Extractors for Whole Slide Image Classification. https://arxiv.org/abs/2311.11772
  - Abstract: skipping stain normalisation and image augmentation does not degrade slide-level performance, and lower magnification is sufficient. Argues against spending compute on image-level stain work with foundation encoders.
- **de Jong, Marcus, Teuwen (2025).** Current Pathology Foundation Models are unrobust to Medical Center Differences. https://arxiv.org/abs/2501.18055
  - Embeddings encode the medical centre more strongly than the tissue; errors come from same-centre confounders. This is the cohort shift (TCGA to NLST) that limits this task.
- **Kömen, de Jong, Hense et al. (2025).** Towards Robust Foundation Models for Digital Pathology (PathoROB). https://arxiv.org/abs/2507.17845
  - Benchmarks 20 encoders for robustness to non-biological features and tests post-hoc robustification, which may be applicable to stored features.
- **Shao, Chen, Song et al. (2025).** Do Multiple Instance Learning Models Transfer? ICML 2025. https://arxiv.org/abs/2506.09022
  - Pretrained aggregators beat aggregators trained from scratch. Pancancer-pretrained ABMIL weights are linked from the TRIDENT README (for example https://huggingface.co/MahmoodLab/abmil.base.uni_v2.pc108-24k).
- **Campanella, Chen, Verma et al. (2024).** A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models. https://arxiv.org/abs/2407.06508
  - Compares public encoders on hospital data from two centres, including lung adenocarcinoma tasks.

## Datasets and benchmarks

- **UNI2-h pre-extracted features (Mahmood lab, 2025).** https://huggingface.co/datasets/MahmoodLab/UNI2-h-features (linked from https://github.com/mahmoodlab/UNI)
  - Precomputed embeddings for TCGA and CPTAC (and PANDA). If the lung cohorts are included (not checked), this covers the task's training and development cohorts without touching a slide.
- **Patho-Bench (Zhang, Jaume, Vaidya et al., 2025).** Accelerating Data Processing and Benchmarking of AI Models for Pathology. https://arxiv.org/abs/2502.06750 ; splits https://huggingface.co/datasets/MahmoodLab/Patho-Bench ; code https://github.com/mahmoodlab/Patho-Bench
  - Public tasks with clean labels and fixed train-test splits; use its lung splits to compare with published numbers.
- **NCI Imaging Data Commons via idc-index.** https://github.com/ImagingDataCommons/idc-index
  - Public, no-login download of cancer imaging including digital pathology slides; the README example queries `tcga_luad`.
- **Prov-GigaPath tile embeddings.** https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings (linked from https://github.com/prov-gigapath/prov-gigapath; paper DOI 10.1038/s41586-024-07441-w)
  - Another released embedding set; which cohorts it covers was not checked.
- **TCGA slides.** https://portal.gdc.cancer.gov/ (address as given in the Owkin README)
  - Source of the training slides and diagnoses.

## Open-source code and tools

- **TRIDENT (Mahmood lab).** https://github.com/mahmoodlab/TRIDENT
  - Tissue segmentation, tiling and feature extraction with many encoders (UNI, CONCH, TITAN and others) in one command.
- **STAMP (Kather lab).** https://github.com/KatherLab/STAMP ; protocol https://www.nature.com/articles/s41596-024-01047-2
  - End-to-end weakly supervised pipeline from slide to patient-level prediction.
- **eva (kaiko.ai).** https://github.com/kaiko-ai/eva
  - Evaluation framework for pathology encoders.
- **Paige ML SDK.** https://github.com/Paige-AI/paige-ml-sdk
  - Training of supervised models on tile-level embeddings, released with Virchow.

## Breakthroughs in related fields that could transfer

- **Shao, Dai, Wang et al. (2023).** AugDiff: Diffusion based Feature Augmentation for Multiple Instance Learning in Whole Slide Image. https://arxiv.org/abs/2303.06371
  - Augments stored tile features instead of images; the feature-level augmentation idea in this project's route, at CPU cost.

## Published results to beat

- TransMIL (Shao et al. 2021, abstract): AUC 96.03% on the TCGA-NSCLC subtyping dataset.
- CONCH (Lu et al. 2023, arXiv HTML): zero-shot NSCLC subtyping accuracy 90.0% on TCGA, with no training slides.
- Phikon-v2 (Filiot et al. 2024, abstract): ensembling five models per split gave +1.75 AUC on average across tasks and encoders, compared with a single retraining.
- Shao et al. 2025 (arXiv HTML, results table): NSCLC morphological subtyping scores of 90.4 to 95.9 for aggregators trained from scratch and up to 96.6 with pancancer pretraining. The metric and test cohort for that row were not confirmed from the extracted text.
- HFM track best on NLST: 0.9784 patient AUROC.

## Cheapest credible way to run experiments

- CPU only: download UNI2-h tile features for TCGA-LUAD/LUSC and CPTAC-LUAD/LSCC, train ABMIL on TCGA and test on CPTAC. Aggregators on stored features train on CPU. This gives a full train/development loop with a second encoder and no slide handling.
- Robustness work (feature-level augmentation, post-hoc robustification, calibrated abstention) also runs on stored features.
- Phikon-v2 features and anything on NLST still need a GPU pass over the slides and the NLST data request.

## Unverified leads

- Coudray et al. 2018 (Nature Medicine), the first deep-learning LUAD versus LUSC study with a widely quoted AUC near 0.97: not confirmed in this session.
- Whether NLST pathology slides are downloadable from the Imaging Data Commons without a data request: not confirmed.
- Whether Owkin's released TCGA features include LUAD and LUSC, and whether any Phikon-v2 feature set is published: not confirmed.
- H-optimus (https://github.com/bioptimus/releases exists but its README gives no details) and a CLAM code repository URL: not confirmed.
- Pathologist agreement figures for adenocarcinoma versus squamous in poorly differentiated tumours: none confirmed.
