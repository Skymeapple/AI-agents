# References: Agentic Pretraining Under Fixed Inference-Cost Constraint

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017).** Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv. https://arxiv.org/abs/1701.06538
  - MoE raises capacity at about constant compute per token, the basic lever when inference cost is capped.
- **Fedus, Zoph, Shazeer (2021).** Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv. https://arxiv.org/abs/2101.03961
  - Top-1 routing MoE recipe that most later MoE language models start from.
- **Hoffmann, Borgeaud, Mensch, et al. (2022).** Training Compute-Optimal Large Language Models. arXiv. https://arxiv.org/abs/2203.15556
  - With inference cost capped, model size is nearly fixed, so the remaining lever is tokens seen: this is why throughput is the README's first priority.
- **Porian, Wortsman, Jitsev, Schmidt, Carmon (2024).** Resolving Discrepancies in Compute-Optimal Scaling of Language Models. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2406.19146
  - Gives scaling rules for learning rate and batch size and notes that AdamW beta2 must be retuned at small batch; relevant when moving micro-batch from 4 to 8-16 at constant global batch.

## Recent advances and breakthroughs (2023-2026)
- **Essential AI: Shah, Polloreno, Stratos, et al. (2025).** Practical Efficiency of Muon for Pretraining. arXiv. https://arxiv.org/abs/2505.02222
  - Muon holds data efficiency at batch sizes well beyond AdamW's critical batch size, so idle GPU memory can go to larger batches with less loss per token.
- **Liu, Su, Yao, Jiang, Lai, et al. (2025).** Muon is Scalable for LLM Training. arXiv. https://arxiv.org/abs/2502.16982
  - Adds weight decay and per-parameter update scaling to Muon and reports about 2x compute efficiency over AdamW in scaling-law experiments; used to train a 3B/16B MoE (https://huggingface.co/moonshotai/Moonlight-16B-A3B).
- **Wan, Han, Zhang, Zhang, Jin (2026).** Holistic Scaling Laws for Optimal Mixture-of-Experts Architecture Optimization. NeurIPS 2026. https://arxiv.org/abs/2603.21862
  - Separates FLOPs per token, active parameters and total parameters, the three quantities a cost-adjusted score can charge for.
- **Krajewski, Ludziejewski, Adamczewski, Pióro, Krutul, et al. (2024).** Scaling Laws for Fine-Grained Mixture of Experts. ICLR 2024. https://arxiv.org/abs/2402.07871
  - Adds expert granularity to the scaling law and reports that setting expert size equal to the FFN size is suboptimal at almost any compute budget.
- **Ludziejewski, Pióro, Krajewski, Stefaniak, Krutul, et al. (2025).** Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient. arXiv. https://arxiv.org/abs/2502.05172
  - Joint dense and MoE scaling laws from over 280 runs for choosing expert count under fixed memory and compute.
- **Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi (2024).** Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.18392
  - Constant learning rate plus cooldown matches cosine and lets one long run be reused for several budgets; code at https://github.com/epfml/schedules-and-scaling.
- **Sanyal, Neerkaje, Kaddour, Kumar, Sanghavi (2023).** Early Weight Averaging meets High Learning Rates for LLM Pre-training. NeurIPS 2023 WANT workshop. https://arxiv.org/abs/2306.03241
  - Checkpoint averaging is free at inference and at training, so it improves the score with no cost-term change.
- **Ousherovitch, Wang (2026).** Terminal Shrinkage Averaging Reveals a Schedule-Estimator Interaction in LLM Pretraining. arXiv. https://arxiv.org/abs/2609.25482
  - Interpolates the final iterate with an average of recent checkpoints and shows the best terminal schedule changes when you do; tested on NanoChat up to depth 22.
- **Lin, Guo, Zhu, Ye, Eshraghian (2026).** Allocating Recurrent Compute in Looped Language Models. arXiv. https://arxiv.org/abs/2608.18230
  - If any depth reuse is tried, loop the mixer only: the FFN dominates forward cost.
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Reports higher throughput and lower perplexity at equal training FLOPs through per-token depth; the cost model of the scorer decides whether this counts.

## Datasets and benchmarks
- **Penedo, Kydlíček, Ben Allal, Lozhkov, et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Defines DCLM-Pool (240T tokens), DCLM-Baseline and 53 evaluations; project page https://www.datacomp.ai/dclm/.

## Open-source code and tools
- **Jordan et al. (2024-2026).** modded-nanogpt (NanoGPT speedrun). GitHub. https://github.com/KellerJordan/modded-nanogpt
  - Source of throughput techniques at this scale (the search result lists an FP8 output head, sliding-window attention and others); each record is timed on fixed hardware, so it is a throughput benchmark as much as a recipe.
- **djdumpling (2026).** modded nanoGPT WR. Blog post. https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html
  - Write-up of a 2026 record with run statistics (80.40 +/- 0.11 s at 3.2794 validation loss on 8 H200, n=10), showing the level of evidence the speedrun expects.
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - Compact full pipeline (tokeniser, pretraining, evaluation); secondary sources put the depth-20 run at about 4 hours on 8 H100 (about $100). Walkthrough: https://github.com/karpathy/nanochat/discussions/1.
- **Karpathy (2026).** autoresearch. GitHub. https://github.com/karpathy/autoresearch
  - Uses a fixed wall-clock budget, so it rewards throughput gains the same way this track does; a cheap place to test micro-batch and compile changes.

## Breakthroughs in related fields that could transfer
- **Elhoushi, Shrivastava, Liskovich, Hosmer, Wasti, et al. (2024).** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024. https://arxiv.org/abs/2404.16710
  - Early exit trained with layer dropout lowers average inference cost without extra modules.
- **Wu, Chen, Luo, Yan, Yu, et al. (2025).** Parallel Loop Transformer for Efficient Test-Time Computation Scaling. arXiv. https://arxiv.org/abs/2510.24824
  - Shows a way to add depth through looping without adding latency; depends on how the scorer defines cost.
- **Geiping, Goldstein (2022).** Cramming: Training a Language Model on a Single GPU in One Day. ICML 2023. https://arxiv.org/abs/2212.14034
  - Shows which recipe changes matter under one GPU-day and that scaling-law behaviour still holds there; code at https://github.com/JonasGeiping/cramming.

## Published results to beat
- HFM best so far on this track: -0.667 neg_cost_adjusted_bpb, 9 of 10 proposals scored (project README).
- Current run p392 uses 40.4 GB of 141 GB memory at 120k tokens/s; the README estimates a 20% throughput gain is worth about 0.0085 bpb (3 sigma). This is the internal number to verify first.
- Muon: about 2x compute efficiency over AdamW (https://arxiv.org/abs/2502.16982).
- Speedrun reference for run-to-run noise: 3.2794 +/- 0.0015 validation loss over 10 runs (https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html); a claimed 3-sigma gain needs repeated seeds.

## Unverified leads
- UNVERIFIED: Keller Jordan's blog post "Muon: An optimizer for hidden layers in neural networks" (the original Muon description). Not confirmed by a search in this session; Muon is confirmed only through the arXiv papers above.
- UNVERIFIED: Fused cross-entropy and wider torch.compile coverage as throughput levers (named in the README). No published measurement was searched for.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: Whether a nanochat or modded-nanogpt run fits a free Colab/Kaggle T4. No search result confirmed this; the confirmed low-cost options are autoresearch (one GPU, 5-minute runs) and Cramming (one GPU, one day).
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
