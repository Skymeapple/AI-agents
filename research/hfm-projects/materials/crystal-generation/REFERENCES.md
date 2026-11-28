# References: Crystal Generation for Unseen Chemical Systems with MatterGen and MatterSim

Checked 2026-11-25. Each entry was confirmed in this session by the arXiv record (numbers come from the arXiv abstracts) or by the project's own README on GitHub. Nothing is cited from memory. Where a README did not list individual authors, the releasing group is given in their place.

## Theory and established methods

- **Xie, Fu, Ganea et al. (2021).** Crystal Diffusion Variational Autoencoder for Periodic Material Generation (CDVAE). ICLR 2022. https://arxiv.org/abs/2110.06197
  - The usual baseline generator (Gruver et al. and FlowMM compare against it). Code and data: https://github.com/txie-93/cdvae.
- **Jiao, Huang, Lin et al. (2023).** Crystal Structure Prediction by Joint Equivariant Diffusion (DiffCSP). NeurIPS 2023. https://arxiv.org/abs/2309.04475
  - Generates a structure for a given composition. This is the right tool once the composition has been chosen by another method. Code: https://github.com/jiaor17/DiffCSP.
- **Chen, Ong (2022).** A Universal Graph Deep Learning Interatomic Potential for the Periodic Table (M3GNet). Nature Computational Science. https://arxiv.org/abs/2202.02450
  - Early universal potential; a candidate for a first-pass energy screen.
- **Deng, Zhong, Jun et al. (2023).** CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeling. Nature Machine Intelligence. https://arxiv.org/abs/2302.14231
  - Second independent potential for cross-checking MatterSim. Code: https://github.com/CederGroupHub/chgnet.
- **Batatia, Benner, Chiang et al. (2023).** A foundation model for atomistic materials chemistry (MACE-MP). https://arxiv.org/abs/2401.00096
  - Third independent potential; models at https://github.com/ACEsuit/mace-foundations.
- **pymatgen (Materials Project).** https://github.com/materialsproject/pymatgen ; paper DOI 10.1016/j.commatsci.2012.10.028
  - Convex hulls, structure matching and substitution tools: the non-neural parts of the pipeline.

## Recent advances and breakthroughs (2023-2026)

- **Zeni, Pinsler, Zügner et al. (2023/2025).** MatterGen: a generative model for inorganic materials design. Nature, https://www.nature.com/articles/s41586-025-08628-5 ; preprint https://arxiv.org/abs/2312.03687
  - The generator named in the task. The repository ships checkpoints fine-tuned for conditioning on chemical system (`chemical_system`) and on chemical system plus energy above hull (`chemical_system_energy_above_hull`), with a guidance-strength setting.
- **Yang, Hu, Zhou et al. (2024).** MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures. https://arxiv.org/abs/2405.04967
  - The frozen potential that defines "stable" in this task. Two checkpoints are released, 1M (default, faster) and 5M (more accurate).
- **Szymanski, Bartel (2025).** Establishing baselines for generative discovery of inorganic crystals. https://arxiv.org/abs/2501.02144
  - Abstract: data-driven ion exchange of known compounds is better than diffusion, VAE and language-model generators at producing novel materials that are stable, and a post-generation screen with universal potentials improves every method. This is the over-generate-then-select route of this project, already tested. Code: https://github.com/Bartel-Group/matgen_baselines.
- **Miller, Chen, Sriram et al. (2024).** FlowMM: Generating Materials with Riemannian Flow Matching. ICML 2024. https://arxiv.org/abs/2406.04713
  - Abstract: about 3x more efficient in integration steps at finding stable materials than earlier open methods. Code: https://github.com/facebookresearch/flowmm.
- **Sriram, Miller, Chen et al. (2024).** FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions. NeurIPS 2024. https://arxiv.org/abs/2410.23405
  - A language model proposes, flow matching refines. A two-stage design that maps onto "propose compositions, then generate structures".
- **Gruver, Sriram, Madotto et al. (2024).** Fine-Tuned Language Models Generate Stable Inorganic Materials as Text. ICLR 2024. https://arxiv.org/abs/2402.04379
  - Text-based generation, including text-conditional generation and infilling of partial structures. Code: https://github.com/facebookresearch/crystal-llm.
- **Jiao, Huang, Liu et al. (2024).** Space Group Constrained Crystal Generation (DiffCSP++). ICLR 2024. https://arxiv.org/abs/2402.03992
  - Generation under a space-group constraint.
- **Levy, Panigrahi, Kaba et al. (2025).** SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models. ICLR 2025. https://arxiv.org/abs/2502.03638
  - Diffusion model that preserves crystal symmetry.
- **Kazeev, Nong, Romanov et al. (2025).** Wyckoff Transformer: Generation of Symmetric Crystals. ICML 2025. https://arxiv.org/abs/2503.02407
  - Transformer that generates symmetric crystals. Code: https://github.com/SymmetryAdvantage/WyckoffTransformer.
- **Cao, Luo, Lv et al. (2024).** Space Group Informed Transformer for Crystalline Materials Generation (CrystalFormer). Science Bulletin 2025. https://arxiv.org/abs/2403.15734
  - Autoregressive generator conditioned on space group.
- **Hoellmer, Egg, Martirossyan et al. (2025).** Open Materials Generation with Stochastic Interpolants (OMatG). ICML 2025. https://arxiv.org/abs/2502.02582
  - Benchmarks both structure prediction for given compositions and de novo generation of stable, novel, unique structures.
- **Joshi, Fu, Liao et al. (2025).** All-atom Diffusion Transformers: Unified generative modelling of molecules and materials. ICML 2025. https://arxiv.org/abs/2503.03965
  - One generative model for crystals and molecules together.
- **Barroso-Luque, Shuaibi, Fu et al. (2024).** Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models. https://arxiv.org/abs/2410.12771
  - Large open DFT dataset and potentials; models and data through https://github.com/facebookresearch/fairchem (UMA: https://arxiv.org/abs/2506.23971).
- **Neumann, Gin, Rhodes et al. (2024).** Orb: A Fast, Scalable Neural Network Potential. https://arxiv.org/abs/2410.22570
  - Fast potential suited to screening many candidates. Code: https://github.com/orbital-materials/orb-models.

## Datasets and benchmarks

- **Riebesell, Goodall, Benner et al. (2023/2025).** Matbench Discovery: A framework to evaluate machine learning crystal stability predictions. Nature Machine Intelligence, DOI 10.1038/s42256-025-01055-1. https://arxiv.org/abs/2308.14920
  - Ranks potentials as stability pre-filters; leaderboard at https://matbench-discovery.materialsproject.org/. Abstract warns that accurate regressors still give high false-positive rates near 0 eV/atom above the hull, which is exactly where this task's "stable" label is decided.
- **Betala, Gleason, Ramlaoui et al. (2025).** LeMat-GenBench: A Unified Evaluation Framework for Crystal Generative Models. https://arxiv.org/abs/2512.04562
  - Common scoring of validity, novelty, uniqueness and stability across generators. Abstract: higher stability comes with lower novelty and diversity on average. Code https://github.com/LeMaterial/lemat-genbench; reference hull data https://huggingface.co/datasets/LeMaterial/LeMat-Bulk-MLIP-Hull.
- **Google DeepMind (2023).** GNoME: Materials Discovery. Nature, https://www.nature.com/articles/s41586-023-06735-9 ; data https://github.com/google-deepmind/materials_discovery
  - Public set of 381,000 predicted-stable materials. Needed for an honest novelty check, and a source of parent structures for substitution.
- **Schmidt, Hoffmann, Wang et al. (2022).** Large-scale machine-learning-assisted exploration of the whole materials space. https://arxiv.org/abs/2210.00579
  - Large machine-learning-assisted search for stable compounds; likely background for the Alexandria reference data named in this project's README (link not confirmed).

## Open-source code and tools

- **MatterGen (Microsoft).** https://github.com/microsoft/mattergen ; weights https://huggingface.co/microsoft/mattergen
  - `mattergen-evaluate` relaxes with MatterSim and reports stability, uniqueness and novelty against a reference set downloaded on first use. It also accepts externally computed energies (`--energies_path`, `--relax=False`), so structures from any source can be scored the same way. The README recommends the TRI2024 correction scheme and reference set for work beyond literature comparison.
- **MatterSim (Microsoft).** https://github.com/microsoft/mattersim
  - The README's example code selects CUDA if present and otherwise CPU, so relaxation runs on CPU, slowly.
- **matgen_baselines (Bartel group).** https://github.com/Bartel-Group/matgen_baselines
  - Random enumeration of charge-balanced prototypes and ion exchange, with stability and property filters. No generator training needed.
- **CrystaLLM.** https://github.com/lantunes/CrystaLLM ; paper https://www.nature.com/articles/s41467-024-54639-7 (preprint https://arxiv.org/abs/2307.04340, Antunes, Butler, Grau-Crespo)
  - Transformer language model of the CIF file format.
- **Chemeleon.** https://github.com/hspark1212/chemeleon ; paper DOI 10.1038/s41467-025-59636-y
  - Text- or composition-guided diffusion for crystal structures.

## Breakthroughs in related fields that could transfer

- **Deep ensembles of potentials as an error check.** Use CHGNet, MACE-MP, Orb and UMA (entries above) next to MatterSim.
  - Structures that MatterSim calls stable but other potentials do not are likely MatterSim errors. This tests this project's stated concern that the score may reward potential errors, without DFT.

## Published results to beat

- MatterGen (Zeni et al., abstract): structures are more than twice as likely to be novel and stable as those from prior generators, and more than 15 times closer to the local energy minimum.
- Gruver et al. 2024 (abstract): fine-tuned LLaMA-2 70B gives 49% metastable structures against 28% for CDVAE.
- FlowLLM (Sriram et al., abstract): over three times the stable-generation rate of earlier methods and about 50% more stable, unique, novel crystals.
- Matbench Discovery (abstract): universal potentials reach F1 of 0.57 to 0.82 for stability classification; OMat24 models report F1 above 0.9 and 20 meV/atom formation-energy accuracy (Barroso-Luque et al., abstract).
- These rates are for unconditional generation scored with each paper's own protocol. None is for unseen two- and three-element systems with a 64-structure budget. The number to beat is HFM's 0.4395.

## Cheapest credible way to run experiments

- CPU only: generate candidates for each target chemical system by ion exchange and prototype enumeration (matgen_baselines, pymatgen), remove duplicates with pymatgen's structure matcher, and rank by MatterSim-1M single-point energy on CPU. Relax only the short list. Score with `mattergen-evaluate`.
- This needs no MatterGen sampling. MatterGen samples from one GPU run can be added to the same candidate pool later.
- Small subset: pick 5 to 10 chemical systems and compare "64 raw samples" against "64 selected from a larger pool".

## Unverified leads

- A published S.U.N. rate for the open MatterGen checkpoints under MatterSim scoring: not confirmed; the README says the released checkpoints were retrained and may deviate from the paper.
- Alexandria database download location and its link to Schmidt et al. 2022: not confirmed in this session.
- Exact numbers from Szymanski and Bartel (stable-and-novel rates for ion exchange against each generator): the abstract gives the ranking only; read the paper body.
- CPU speed of MatterSim relaxation per structure: not measured or sourced.
