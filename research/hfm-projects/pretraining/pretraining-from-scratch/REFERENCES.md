# References: Pretraining architecture development from scratch

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Hoffmann, Borgeaud, Mensch, et al. (2022).** Training Compute-Optimal Large Language Models. arXiv. https://arxiv.org/abs/2203.15556
  - Baseline scaling law for splitting a fixed compute budget between model size and tokens.
- **Porian, Wortsman, Jitsev, Schmidt, Carmon (2024).** Resolving Discrepancies in Compute-Optimal Scaling of Language Models. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2406.19146
  - Small-scale scaling fits are distorted by warmup length and untuned optimisers; fix these before ranking ideas on short runs.
- **Geiping, Goldstein (2022).** Cramming: Training a Language Model on a Single GPU in One Day. ICML 2023. https://arxiv.org/abs/2212.14034
  - Closest published analogue of a small fixed budget (one GPU, one day); reports that performance still follows the scaling laws seen at large compute.
- **Henry, Dachapally, Pawar, Chen (2020).** Query-Key Normalization for Transformers. arXiv. https://arxiv.org/abs/2010.04245
  - Source of QK-norm (normalising queries and keys before the attention dot product), a low-cost change often used in small fast recipes.
- **Bernstein, Newhouse (2024).** Old Optimizer, New Norm: An Anthology. arXiv. https://arxiv.org/abs/2409.20325
  - Steepest-descent-under-a-norm view that motivates Muon-style updates for matrix parameters.

## Recent advances and breakthroughs (2023-2026)
- **Choshen, Zhang, Andreas (2024).** A Hitchhiker's Guide to Scaling Law Estimation. ICML 2025. https://arxiv.org/abs/2410.11840
  - Core reference for the README's short-run ranking: use intermediate checkpoints, expect usable final-loss estimates from about one third of a run, and budget for seed variance.
- **Brandfonbrener, Anand, Vyas, Malach, Kakade (2024).** Loss-to-Loss Prediction: Scaling Laws for All Datasets. arXiv. https://arxiv.org/abs/2411.12925
  - Shifted power laws relate loss on one dataset to loss on another and extrapolate to 20x the fitted FLOP budget.
- **Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi (2024).** Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.18392
  - A constant-rate trunk with short cooldown branches gives comparable checkpoints at several token counts from one run; this fits a 168-hour session with pause and resume.
- **Dey, Zhang, Noci, Li, Bordelon, et al. (2025).** Don't be lazy: CompleteP enables compute-efficient deep transformers. NeurIPS 2025. https://arxiv.org/abs/2505.01618
  - Lets learning rates tuned on a shallow proxy transfer to the deep model, which removes one cause of unstable deep candidates.
- **Liu, Su, Yao, Jiang, Lai, et al. (2025).** Muon is Scalable for LLM Training. arXiv. https://arxiv.org/abs/2502.16982
  - Adds weight decay and per-parameter update scaling to Muon and reports about 2x compute efficiency over AdamW in scaling-law experiments; used to train a 3B/16B MoE (https://huggingface.co/moonshotai/Moonlight-16B-A3B).
- **Essential AI: Shah, Polloreno, Stratos, et al. (2025).** Practical Efficiency of Muon for Pretraining. arXiv. https://arxiv.org/abs/2505.02222
  - Finds Muon keeps data efficiency at batch sizes far beyond AdamW's critical batch size, and combines it with muP for hyperparameter transfer.
- **Khona, Vavre, Wang, Fu, Wu, et al. (2026).** SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales. arXiv. https://arxiv.org/abs/2607.20548
  - Compares SOAP, Muon and AdamW with update-RMS matching so learning rates transfer fairly between optimisers.
- **Dewulf, Pai, Yang, Zhang, Keigwin (2026).** Aurora: A Leverage-Aware Spectral Optimizer. arXiv. https://arxiv.org/abs/2606.27715
  - Fixes non-uniform row norms in Muon updates for tall MLP matrices; reports best results among spectral optimisers on the modded-nanoGPT optimiser track.
- **Loshchilov, Hsieh, Sun, Ginsburg (2024).** nGPT: Normalized Transformer with Representation Learning on the Hypersphere. arXiv. https://arxiv.org/abs/2410.01131
  - Claims 4x to 20x fewer training steps to the same accuracy by normalising all vectors; a candidate recipe change when steps are the scarce resource.
- **Du, Luo, Qiu, Huang, Shen, Cheng, Guo, Fu (2024).** Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.15319
  - An alternative to restarting from scratch at each stage: grow the stable 45M-token candidate in depth.

## Datasets and benchmarks
- **Penedo et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Defines DCLM-Pool (240T tokens), DCLM-Baseline and 53 evaluations; project page https://www.datacomp.ai/dclm/.
- **Ben Allal, Lozhkov, Bakouch, Martín Blázquez, Penedo, et al. (2025).** SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. arXiv. https://arxiv.org/abs/2502.02737
  - Introduces FineMath and Stack-Edu and documents a multi-stage web/code/math mixture for a small model.
- **Hu, Mueller, Ross, Williams, Linzen, et al. (2024).** Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora. arXiv. https://arxiv.org/abs/2412.05149
  - A fixed 10M/100M-word benchmark for fast, comparable architecture tests.
- **Eldan, Li (2023).** TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv. https://arxiv.org/abs/2305.07759
  - Small synthetic story corpus built to study very small language models; useful for smoke tests of code, not for ranking ideas for web text.

## Open-source code and tools
- **Jordan et al. (2024-2026).** modded-nanogpt (NanoGPT speedrun). GitHub. https://github.com/KellerJordan/modded-nanogpt
  - Tested recipe to adopt as the low-risk baseline the README asks for.
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - End-to-end pipeline scaled by a single depth setting, close to the 'depth' branch naming in this track.
- **Karpathy (2026).** autoresearch. GitHub. https://github.com/karpathy/autoresearch
  - Single-GPU agent loop: edit the training script, train for a fixed 5 minutes, keep or revert by validation bits-per-byte.

## Breakthroughs in related fields that could transfer
- **Zhao, Magka, Jiang, Li, et al. (2025).** The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements. NeurIPS 2025 Datasets and Benchmarks. https://arxiv.org/abs/2506.22419
  - Finds that agents struggle to re-implement known speedrun improvements even with detailed hints; code in the facebookresearch/llm-speedrunner repository.
- **Ning, Li, Zeng, Kang, Xiong (2026).** Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes. arXiv. https://arxiv.org/abs/2605.05724
  - Shows that an unattended agent loop improves a public recipe only when failures are fed back as structured lineage; this track had 3 proposals and 2 scores.
- **Qu, Lu (2026).** Bilevel Autoresearch: Meta-Autoresearching Itself. arXiv. https://arxiv.org/abs/2603.23420
  - On the autoresearch pretraining benchmark an outer loop that rewrites the search procedure got -0.045 val_bpb against -0.009 for the plain loop; parameter-only tuning gave no reliable gain.
- **Acker, Becker, Nedelkoski, Scheinert, Kao, et al. (2025).** What happens when nanochat meets DiLoCo? arXiv (technical report). https://arxiv.org/abs/2511.13761
  - Uses nanochat as a controlled baseline for multi-worker training with infrequent synchronisation.

## Published results to beat
- HFM best so far on this track: -0.9624 negative_public_validation_bpb from a 45M-token checkpoint (project README). The sibling track pretraining-architecture-pool reports -0.7048 on a metric of the same name (its README); that gap is mostly training length.
- nanochat: depth-20 model in about 4 hours on 8 H100, about $100 (secondary sources found via search; primary: https://github.com/karpathy/nanochat/discussions/1).
- modded-nanogpt: 3.28 FineWeb validation loss in 2.93 minutes on 8 H100 (record table snapshot dated 01/26/25) and about 80 s on 8 H200 in May 2026 (https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html); the original llm.c baseline took 45 minutes (https://github.com/KellerJordan/modded-nanogpt).
- G_stack: 300B-token loss reached at 194B tokens (https://arxiv.org/abs/2405.15319).

## Unverified leads
- UNVERIFIED: Keller Jordan's blog post "Muon: An optimizer for hidden layers in neural networks" (the original Muon description). Not confirmed by a search in this session; Muon is confirmed only through the arXiv papers above.
- UNVERIFIED: A data-limited "slowrun" variant of the nanoGPT speedrun (fixed token budget instead of fixed time). Could not be searched before the search budget ran out.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: Whether a nanochat or modded-nanogpt run fits a free Colab/Kaggle T4. No search result confirmed this; the confirmed low-cost options are autoresearch (one GPU, 5-minute runs) and Cramming (one GPU, one day).
- UNVERIFIED: Hugging Face dataset cards for FineMath (HuggingFaceTB/finemath) and DCLM-Baseline (mlfoundations/dclm-baseline-1.0). Existence of the datasets is confirmed through the SmolLM2 and DCLM papers, but the exact repository names and sizes were not.
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
