# References: Predicting human regulatory DNA activity with Evo 2 7B and lentiMPRA

Checked 2026-11-25. Each entry was confirmed in this session by the arXiv record, the Zenodo record, or the project's own README on GitHub. Nothing is cited from memory. Where a README did not list individual authors, the releasing organisation is given in their place.

## Theory and established methods

- **Agarwal, Inoue, Schubach, Penzar et al. (2023).** Massively parallel characterization of transcriptional regulatory elements. bioRxiv, DOI 10.1101/2023.03.05.531189. https://zenodo.org/records/10558183
  - Source of this task's data: lentiMPRA of over 680,000 sequences in HepG2, K562 and WTC11, plus a 60,000-sequence library tested in all three. The Zenodo record (CC-BY 4.0) holds the data dumps, fold tables and model code.
- **Penzar, Nogina et al. (2023).** LegNet: a best-in-class deep learning model for short DNA regulatory regions. Bioinformatics, DOI 10.1093/bioinformatics/btad457. https://github.com/autosome-ru/LegNet
  - The convolutional architecture behind MPRALegNet; built for reporter-assay sequences of this length.
- **MPRA-LegNet (autosome-ru, 2023).** Code for the lentiMPRA model of Agarwal et al. https://github.com/autosome-ru/human_legnet
  - The published supervised baseline on this exact data, with the authors' 10-fold scheme. README states training needs 7 GB of GPU memory.
- **Avsec, Agarwal, Visentin et al. (2021).** Effective gene expression prediction from sequence by integrating long-range interactions (Enformer). https://github.com/google-deepmind/deepmind-research/tree/master/enformer ; preprint https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649
  - Supervised sequence-to-function model; a source of cell-type-specific features for HepG2 and K562 that a DNA language model does not have.
- **Kundaje lab (2019).** MPRA-DragoNN: Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays. PLoS ONE, DOI 10.1371/journal.pone.0218073. https://github.com/kundajelab/MPRA-DragoNN
  - Early CNN for MPRA prediction named in this project's README.
- **Hu, Shen, Wallis et al. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
  - The parameter-efficient fine-tuning method proposed for Evo 2 in this project; library at https://github.com/huggingface/peft.

## Recent advances and breakthroughs (2023-2026)

- **Arc Institute (2026).** Genome modeling and design across all domains of life with Evo 2. Nature, DOI 10.1038/s41586-026-10176-5. https://github.com/ArcInstitute/evo2
  - The model named in the task. README: the 7B models run in bfloat16 on any supported GPU without FP8 hardware; a 1B model also exists; intermediate-layer embeddings work better than final-layer ones.
- **Calico (2023/2025).** Borzoi: Predicting RNA-seq from DNA Sequence. Nature Genetics, DOI 10.1038/s41588-024-02053-6; preprint https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1. https://github.com/calico/borzoi
  - Supervised model trained on ENCODE and GTEx tracks; a second feature view alongside Evo 2.
- **borzoi-pytorch and Flashzoi (2025).** PyTorch Borzoi and a faster variant (Bioinformatics, DOI 10.1093/bioinformatics/btaf467). https://github.com/johahi/borzoi-pytorch
  - Makes Borzoi usable in the same PyTorch pipeline as the Evo 2 head.
- **Patel, Singhal, Wang et al. (2024).** DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2412.05430
  - Finds DNA language models "do not offer compelling gains over alternative baseline models for most tasks" on regulatory DNA. Supports adding a supervised model next to Evo 2. Code: https://github.com/kundajelab/DART-Eval.
- **Marin, Teufel, Horlacher et al. (2023).** BEND: Benchmarking DNA Language Models on biologically meaningful tasks. ICLR 2024. https://arxiv.org/abs/2311.12570
  - Second independent benchmark of DNA language model embeddings; code at https://github.com/frederikkemarin/BEND.
- **Schiff, Kao, Gokaslan et al. (2024).** Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling. ICML 2024. https://arxiv.org/abs/2403.03234
  - Builds reverse-complement equivariance into the model. This folder's experiment found reverse-complement averaging adds 0.015 Spearman; equivariance gives the same symmetry during training.
- **Benegas, Ye, Albors et al. (2024).** Genomic Language Models: Opportunities and Challenges. https://arxiv.org/abs/2407.11435
  - Review; use it to judge when a language model should help on short regulatory sequences.
- **Zhou, Ji, Li et al. (2023).** DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. ICLR 2024. https://arxiv.org/abs/2306.15006
  - Small DNA language model that can be fine-tuned end to end on modest hardware.
- **Nguyen, Poli, Faizi et al. (2023).** HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. NeurIPS 2023. https://arxiv.org/abs/2306.15794
  - Single-nucleotide predecessor in the Evo line; small checkpoints are a cheap stand-in for testing a fine-tuning recipe.
- **Google DeepMind (2025).** AlphaGenome. Nature, https://www.nature.com/articles/s41586-025-10014-0 ; weights https://github.com/google-deepmind/alphagenome_research
  - Newer supervised sequence-to-function model with downloadable weights (non-commercial terms). A candidate second view; whether it helps on lentiMPRA was not confirmed.

## Datasets and benchmarks

- **Zenodo record 10558183 (2024).** Files: `final_dump.zip` (3.5 GB), `final_joint_dump.zip` (4.0 GB), `Table S6 - folds and performance.xlsx`, `Table S9 - folds and performance.xlsx`, `Table S3 - large-scale lib design.xlsx`, `Table S7 - joint lib design.xlsx`, `human_legnet-main.zip`, `sequence_cnn_models-master.zip`. https://zenodo.org/records/10558183
  - Tables S6 and S9 hold the fold assignments and the authors' per-fold model performance: the published numbers to compare against. The joint library (same sequences in all three cell types) is the cleanest set for a multi-task head. zenodo.org is reachable from this environment.
- **de Boer lab (2023).** Random Promoter DREAM Challenge 2022. Preprint https://www.biorxiv.org/content/10.1101/2023.04.26.538471v2 ; data https://zenodo.org/records/10633252 ; code https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022
  - Community benchmark for short-sequence expression models; the README says the winning models also beat prior models on human datasets. A source of training tricks.
- **CODA / Malinois (2023).** Computational Optimization of DNA Activity. Preprint https://www.biorxiv.org/content/10.1101/2023.08.08.552077v1 ; code https://github.com/sjgosai/boda2
  - A second large human MPRA set (public file at https://storage.googleapis.com/tewhey-public-data/CODA_resources/Table_S2__MPRA_dataset.txt) for pretraining a reporter-assay head before fine-tuning on lentiMPRA.

## Open-source code and tools

- **gReLU (Genentech, 2025).** Nature Methods, DOI 10.1038/s41592-025-02868-z. https://github.com/Genentech/gReLU
  - Library to train and fine-tune sequence models, with a model zoo at https://huggingface.co/collections/Genentech/grelu-model-zoo.
- **enformer-pytorch.** https://github.com/lucidrains/enformer-pytorch
  - Enformer in PyTorch with fine-tuning helpers.
- **Nucleotide Transformer (InstaDeep).** https://github.com/instadeepai/nucleotide-transformer
  - Family of DNA language models of several sizes for an encoder-size ablation.
- **MPRAsnakeflow.** https://github.com/kircherlab/MPRAsnakeflow (named as the successor in https://github.com/shendurelab/MPRAflow)
  - The count-processing pipeline; needed to recompute per-replicate activities for a replicate ceiling.

## Breakthroughs in related fields that could transfer

- **Stark, Jing, Wang et al. (2024).** Dirichlet Flow Matching with Applications to DNA Sequence Design. ICML 2024. https://arxiv.org/abs/2402.05841
  - Generative design of regulatory sequences scored by activity predictors; designed sequences can serve as hard augmentation examples.

## Published results to beat

- No published lentiMPRA performance number was confirmed in a text source in this session. The authors' per-fold results are in `Table S6` and `Table S9` of Zenodo 10558183; read them before claiming a gain over MPRALegNet.
- HFM track best: 0.7076 macro Spearman. This folder's CPU baseline: 0.593.

## Cheapest credible way to run experiments

- CPU only: continue with the small multi-task CNN in this folder. Next CPU steps are the joint library (`final_joint_dump.zip`) for multi-task training, shift augmentation, and the replicate ceiling from the per-replicate tables.
- GPU once: embed all sequences with Evo 2 (1B or 7B; the README supports saving intermediate-layer embeddings), store them, and train every head on CPU.
- Low-GPU fine-tuning test: try the LoRA recipe on a small model (DNABERT-2 or a small HyenaDNA) before spending on Evo 2 7B.

## Unverified leads

- Replicate-to-replicate correlations per cell line (this project's README quotes 0.76 to 0.98): not confirmed from a source in this session; compute from the Zenodo dumps.
- Journal version of Agarwal et al. (believed to be in Nature): venue and year not confirmed.
- Published fine-tuning of Enformer, Borzoi or Evo 2 on this lentiMPRA set with reported Spearman: none confirmed.
- MPRALegNet headline Pearson per cell line: not confirmed; see Table S6.
