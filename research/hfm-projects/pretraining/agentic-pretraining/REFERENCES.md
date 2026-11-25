# References: Agentic Pretraining with a Frozen Corpus and Tokenizer

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017).** Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv. https://arxiv.org/abs/1701.06538
  - Origin of sparsely gated MoE layers: more parameters at roughly constant compute per token.
- **Fedus, Zoph, Shazeer (2021).** Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv. https://arxiv.org/abs/2101.03961
  - Top-1 routing MoE recipe that most later MoE language models start from.
- **Su, Lu, Pan, Murtadha, Wen, Liu (2021).** RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv. https://arxiv.org/abs/2104.09864
  - Defines RoPE, the position encoding used in the long-context axis of this track.
- **Hoffmann, Borgeaud, Mensch, et al. (2022).** Training Compute-Optimal Large Language Models. arXiv. https://arxiv.org/abs/2203.15556
  - Sets the model-size versus tokens split for a fixed budget; with 4 GPUs for 24 hours the token count is fixed by throughput, so this decides the model size.
- **Henry, Dachapally, Pawar, Chen (2020).** Query-Key Normalization for Transformers. arXiv. https://arxiv.org/abs/2010.04245
  - Source of QK-norm, a low-cost attention stabiliser used in fast small-scale recipes.

## Recent advances and breakthroughs (2023-2026)
- **Krajewski, Ludziejewski, Adamczewski, Pióro, Krutul, et al. (2024).** Scaling Laws for Fine-Grained Mixture of Experts. ICLR 2024. https://arxiv.org/abs/2402.07871
  - The MoE axis in this track gave a gain; this paper says expert size equal to FFN size is suboptimal at almost any budget, so granularity is an untested knob.
- **Ludziejewski, Pióro, Krajewski, Stefaniak, Krutul, et al. (2025).** Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient. arXiv. https://arxiv.org/abs/2502.05172
  - Joint dense and MoE scaling laws from over 280 runs for choosing expert count under fixed memory and compute.
- **Wan, Han, Zhang, Zhang, Jin (2026).** Holistic Scaling Laws for Optimal Mixture-of-Experts Architecture Optimization. NeurIPS 2026. https://arxiv.org/abs/2603.21862
  - Treats FLOPs per token, active parameters and total parameters as three separate constraints, which matches a track that charges for forward cost.
- **Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi (2024).** Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.18392
  - Direct evidence for the warmup-stable-decay axis: constant rate plus cooldown matches cosine, and the cooldown length can be chosen late in the run.
- **Ousherovitch, Wang (2026).** Terminal Shrinkage Averaging Reveals a Schedule-Estimator Interaction in LLM Pretraining. arXiv. https://arxiv.org/abs/2609.25482
  - Checkpoint averaging interacts with the terminal schedule, so the long-anneal axis and averaging should be tuned together, not added independently.
- **Liu, Su, Yao, Jiang, Lai, et al. (2025).** Muon is Scalable for LLM Training. arXiv. https://arxiv.org/abs/2502.16982
  - Adds weight decay and per-parameter update scaling to Muon and reports about 2x compute efficiency over AdamW in scaling-law experiments; used to train a 3B/16B MoE (https://huggingface.co/moonshotai/Moonlight-16B-A3B).
- **Essential AI: Shah, Polloreno, Stratos, et al. (2025).** Practical Efficiency of Muon for Pretraining. arXiv. https://arxiv.org/abs/2505.02222
  - Finds Muon keeps data efficiency at batch sizes far beyond AdamW's critical batch size, and combines it with muP for hyperparameter transfer.
- **Peng, Quesnelle, Fan, Shippole (2023).** YaRN: Efficient Context Window Extension of Large Language Models. arXiv. https://arxiv.org/abs/2309.00071
  - The 8k-sequence axis changed the RoPE base; this paper explains why RoPE frequency choice matters when sequence length changes.

## Datasets and benchmarks
- **Penedo, Kydlíček, Ben Allal, Lozhkov, et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Defines DCLM-Pool (240T tokens), DCLM-Baseline and 53 evaluations; project page https://www.datacomp.ai/dclm/.

## Open-source code and tools
- **Jordan et al. (2024-2026).** modded-nanogpt (NanoGPT speedrun). GitHub. https://github.com/KellerJordan/modded-nanogpt
  - A tested stack of recipe changes to copy as the fixed floor before testing architecture axes.
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - Reference pipeline with a bits-per-byte validation metric, close to this track's score.
- **Karpathy (2026).** autoresearch. GitHub. https://github.com/karpathy/autoresearch
  - Cheapest credible agent loop for this kind of task: one GPU and 5-minute runs scored by validation bits-per-byte.

## Breakthroughs in related fields that could transfer
- **Zhao, Magka, Jiang, Li, et al. (2025).** The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements. NeurIPS 2025 Datasets and Benchmarks. https://arxiv.org/abs/2506.22419
  - Agents failed to reproduce known recipe improvements even with hints; expect implementation errors, not idea quality, to dominate a 17-proposal campaign.
- **Ning, Li, Zeng, Kang, Xiong (2026).** Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes. arXiv. https://arxiv.org/abs/2605.05724
  - 1,197 agent trials with no human edits; feeding crashes, budget overruns and gate misses back as lineage is what produced later working recipe edits (Parameter Golf bpb -0.81%, NanoChat-D12 CORE +38.7%).
- **Qu, Lu (2026).** Bilevel Autoresearch: Meta-Autoresearching Itself. arXiv. https://arxiv.org/abs/2603.23420
  - On the autoresearch pretraining benchmark an outer loop that rewrites the search procedure got -0.045 val_bpb against -0.009 for the plain loop; parameter-only tuning gave no reliable gain.

## Published results to beat
- HFM best so far on this track: -0.6941 neg_macro_bpb (project README). No public baseline score exists, so this is the only directly comparable number.
- The three axes in the README (MoE, 8k context with RoPE base 1M, long WSD anneal) are claimed to sum to about 0.031 bpb if additive (project README); additivity is untested.
- autoresearch single-GPU reference: val_bpb about 0.998 at baseline, 1.000 to 0.975 over 83 agent experiments (third-party DataCamp guide, via search on https://github.com/karpathy/autoresearch). Different data and tokenizer, so use only as an order-of-magnitude check on agent-driven gains.
- Muon versus AdamW: about 2x compute efficiency (https://arxiv.org/abs/2502.16982).

## Unverified leads
- UNVERIFIED: Keller Jordan's blog post "Muon: An optimizer for hidden layers in neural networks" (the original Muon description). Not confirmed by a search in this session; Muon is confirmed only through the arXiv papers above.
- UNVERIFIED: A data-limited "slowrun" variant of the nanoGPT speedrun (fixed token budget instead of fixed time). Could not be searched before the search budget ran out.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: Whether a nanochat or modded-nanogpt run fits a free Colab/Kaggle T4. No search result confirmed this; the confirmed low-cost options are autoresearch (one GPU, 5-minute runs) and Cramming (one GPU, one day).
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
