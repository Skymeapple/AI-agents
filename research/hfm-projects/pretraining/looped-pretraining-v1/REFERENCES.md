# References: Looped Transformer Pretraining with Recurrent Depth Architectures

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (2018).** Universal Transformers. ICLR 2019. https://arxiv.org/abs/1807.03819
  - The original depth-recurrent transformer: one block applied repeatedly over all positions, with an optional adaptive computation time (halting) mechanism.
- **Graves (2016).** Adaptive Computation Time for Recurrent Neural Networks. arXiv. https://arxiv.org/abs/1603.08983
  - Baseline for learned exits; its sensitivity to the ponder-cost weight is a known source of failed runs.
- **Banino, Balaguer, Blundell (2021).** PonderNet: Learning to Ponder. ICML 2021 AutoML workshop. https://arxiv.org/abs/2107.05407
  - Learns the number of computation steps end to end and reports large gains over earlier adaptive-computation methods; a baseline for learned exits.
- **Lan, Chen, Goodman, Gimpel, Sharma, Soricut (2019).** ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR 2020. https://arxiv.org/abs/1909.11942
  - Standard reference for cross-layer weight sharing; code at https://github.com/google-research/ALBERT.
- **Bai, Kolter, Koltun (2019).** Deep Equilibrium Models. arXiv. https://arxiv.org/abs/1909.01377
  - Treats a weight-tied network as a fixed-point problem; the starting point for later fixed-point and attractor language models.

## Recent advances and breakthroughs (2023-2026)
- **Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein (2025).** Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. NeurIPS 2025. https://arxiv.org/abs/2502.05171
  - Large-scale reference for a model that runs at several loop counts, which bears on the README's 1/2/4-loop schedule.
- **Zhu et al. (2025).** Scaling Latent Reasoning via Looped Language Models (Ouro). arXiv. https://arxiv.org/abs/2510.25741
  - Looped LMs at 1.4B and 2.6B with learned exit gates, pretrained on 7.7T tokens; project page https://ouro-llm.github.io.
- **Marchenko, Bezrukov, Kashurin, Fedorova, Bocharov, et al. (2026).** Closing the Loop: Practical Training Recipes for Looped Language Models. arXiv. https://arxiv.org/abs/2610.00673
  - Reliability-first recipe for this exact setting: warmup plus stronger exit-gate regularisation gave stable from-scratch looped training without a multi-stage schedule.
- **Prairie, Novack, Berg-Kirkpatrick, Fu (2026).** Parcae: Scaling Laws For Stable Looped Language Models. arXiv. https://arxiv.org/abs/2604.12946
  - Diagnoses residual explosion and loss spikes in looped training; relevant because 18 of 20 proposals here produced no score.
- **Yang et al. (2026).** Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models. ICML 2026. https://arxiv.org/abs/2605.26733
  - Reports that looped models peak at one loop depth and then collapse, and regularises the Jacobian spectral radius with random recurrence depth during training.
- **Bae, Fisch, Harutyunyan, Ji, Kim, Schuster (2024).** Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. ICLR 2025. https://arxiv.org/abs/2410.20672
  - Template for the partial-sharing and adapter variants named in the challenge.
- **Tan, Shen, Chen, Courville, Gan (2023).** Sparse Universal Transformer. EMNLP 2023. https://arxiv.org/abs/2310.07096
  - Template for the MoE-plus-halting variant.
- **Csordás, Irie, Schmidhuber, Potts, Manning (2024).** MoEUT: Mixture-of-Experts Universal Transformers. NeurIPS 2024. https://arxiv.org/abs/2405.16039
  - Uses MoE in feed-forward and attention to restore the parameter count that layer sharing removes; reports slightly better language modelling than a standard transformer at lower compute; code at https://github.com/robertcsordas/moeut.
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Template for learned per-token exits with KV caching that still trains at fixed FLOPs.
- **Jeddi, Ciccone, Taati (2026).** LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. ICLR 2026. https://arxiv.org/abs/2602.11451
  - Trains across variable loop counts with a consistency term so short loops stay useful; an alternative to a hand-set 1/2/4-loop curriculum.
- **Schwethelm, Rueckert, Kaissis (2026).** How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. NeurIPS 2026 (spotlight). https://arxiv.org/abs/2604.21106
  - Shows truncated backpropagation through loops lowers loss while leaving the loop poorly trained; check for this before trusting a curriculum that mostly trains the one-loop path.
- **Liang, Zhang, Zhang, Wang, Lin (2026).** UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute. arXiv. https://arxiv.org/abs/2610.41720
  - A staged shallow-to-looped schedule that keeps total training FLOPs equal to a dense baseline.
- **Pappone, Crisostomi, Rodolà (2025).** Two-Scale Latent Dynamics for Recurrent-Depth Transformers. arXiv. https://arxiv.org/abs/2509.23314
  - Measures loop-step geometry during training and proposes an exit rule based on the second-order difference of step size, compared against the KL exit of Geiping et al.

## Datasets and benchmarks
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - The hidden metric is bits-per-byte on DCLM text; this is the source paper and public pool.
- **Penedo et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.

## Open-source code and tools
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Code: https://github.com/raymin0223/mixture_of_recursions.
- **Csordás, Irie, Schmidhuber, Potts, Manning (2024).** MoEUT: Mixture-of-Experts Universal Transformers. NeurIPS 2024. https://arxiv.org/abs/2405.16039
  - Code: https://github.com/robertcsordas/moeut.
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - A known-good dense pipeline to use as the floor before adding loops.

## Breakthroughs in related fields that could transfer
- **Wang, Li, Sun, Chen, Liu, et al. (2025).** Hierarchical Reasoning Model. arXiv. https://arxiv.org/abs/2506.21734
  - Two nested recurrent modules trained with deep supervision; the reference for nested-loop designs.
- **Jolicoeur-Martineau (2025).** Less is More: Recursive Reasoning with Tiny Networks. arXiv. https://arxiv.org/abs/2510.04871
  - Replaces HRM with one small recursive network and argues that deep supervision, not the hierarchy, drives the gain.
- **Fein-Ashley, Rashidinejad (2026).** Solve the Loop: Attractor Models for Language and Reasoning. arXiv. https://arxiv.org/abs/2605.12466
  - Solves for a fixed point with implicit differentiation so training memory is constant in depth; claims a 770M model beats a 1.3B transformer trained on twice the tokens (authors' claim, not independently checked here).
- **Zhao, Magka, Jiang, Li, et al. (2025).** The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements. NeurIPS 2025 Datasets and Benchmarks. https://arxiv.org/abs/2506.22419
  - Evidence that agent campaigns fail mostly at implementation; supports the README's 'reliability before novelty'.

## Published results to beat
- HFM best so far on this track: -1.553 negative_hidden_dclm_bpb from 2 scored proposals out of 20 (project README). The sibling track looped-pretraining-v2 reached -1.152 on the same metric (its README), which is the practical number to beat.
- Closing the Loop: 1.4B looped model beats a parameter-matched dense model trained on the same data and tokens on all 12 benchmarks, with 310B tokens instead of Ouro's 7.7T (https://arxiv.org/abs/2610.00673).
- Parcae: up to 6.3% lower validation perplexity than earlier looped models (https://arxiv.org/abs/2604.12946).
- LoopCoder-v2: two loops helped, three or more regressed (https://arxiv.org/abs/2606.18023); do not assume more loops is better.

## Unverified leads
- UNVERIFIED: Training code for the Geiping et al. recurrent-depth model (believed to be a public GitHub repository). Not confirmed.
- UNVERIFIED: Lee et al., "Deeply-Supervised Nets" (2015), the original deep-supervision paper. Not searched.
- UNVERIFIED: Hugging Face dataset cards for FineMath (HuggingFaceTB/finemath) and DCLM-Baseline (mlfoundations/dclm-baseline-1.0). Existence of the datasets is confirmed through the SmolLM2 and DCLM papers, but the exact repository names and sizes were not.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
