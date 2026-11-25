# References: Pretraining architecture development on frozen public training pool

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Chen, Goodfellow, Shlens (2015).** Net2Net: Accelerating Learning via Knowledge Transfer. ICLR 2016. https://arxiv.org/abs/1511.05641
  - Defines function-preserving widening and deepening (Net2WiderNet, Net2DeeperNet).
- **Gesmundo, Maile (2023).** Composable Function-preserving Expansions for Transformer Architectures. arXiv. https://arxiv.org/abs/2308.06103
  - Gives the exact initialisation constraints for adding layers and widening the FFN without changing the function, which is what the depth-26, FFN-3040 continuation needs.
- **Wang, Panda, Hennigen, Greengard, et al. (2023).** Learning to Grow Pretrained Models for Efficient Transformer Training. ICLR 2023. https://arxiv.org/abs/2303.00980
  - Learned growth operator (LiGO); reports 44.7% and 22.5% FLOP savings for BERT-Base and GPT2-Medium; code at https://github.com/VITA-Group/LiGO.
- **Hoffmann, Borgeaud, Mensch, et al. (2022).** Training Compute-Optimal Large Language Models. arXiv. https://arxiv.org/abs/2203.15556
  - Baseline scaling law for splitting a fixed compute budget between model size and tokens.
- **Xie, Pham, Dong, Du, Liu, et al. (2023).** DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. NeurIPS 2023. https://arxiv.org/abs/2305.10429
  - Baseline method for setting domain weights with a small proxy model.

## Recent advances and breakthroughs (2023-2026)
- **Du, Luo, Qiu, Huang, Shen, Cheng, Guo, Fu (2024).** Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.15319
  - Strongest published result for growth during pretraining; note that it copies layers (not function-preserving) and still wins.
- **Ren, Wang, Tan, Kong, Fang, et al. (2026).** Rethinking Normalization Placement for LLMs: Post-Norm under Curriculum Depth Growing. arXiv. https://arxiv.org/abs/2608.13156
  - Normalisation placement matters under depth growth even though it does not under joint training.
- **Kapl, Angelis, Maile, von Oswald, Bauer (2026).** From Growing to Looping: A Unified View of Iterative Computation in LLMs. arXiv. https://arxiv.org/abs/2602.16490
  - Depth-grown models gained most from math-heavy cooldown mixtures, which bears on the FineMath share in the sampling ratio.
- **Liang, Zhang, Zhang, Wang, Lin (2026).** UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute. arXiv. https://arxiv.org/abs/2610.41720
  - Starts shallow, grows depth by copying layers, and spends the saved compute on looping a shared middle block, matching a dense baseline in parameters, tokens and training FLOPs.
- **Ye, Liu, Sun, Zhan, Zhou, Qiu (2024).** Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance. ICLR 2025. https://arxiv.org/abs/2403.16952
  - Directly applicable to choosing 0.60/0.20/0.20 against 0.50/0.25/0.25: fit the mixing law on short runs instead of comparing two full continuations.
- **Shukor, Bethune, Busbridge, Grangier, Fini, et al. (2025).** Scaling Laws for Optimal Data Mixtures. arXiv. https://arxiv.org/abs/2507.09404
  - Predicts loss as a function of model size, tokens and domain weights from a few small runs and extrapolates to larger scales and unseen weights.
- **Held, Paranjape, Koura, Lewis, Zhang, et al. (2025).** Optimizing Pretraining Data Mixtures with LLM-Estimated Utility. arXiv. https://arxiv.org/abs/2501.11747
  - Finds simple token-count heuristics beat manual and learned mixes, then improves on them with utility estimates from reduced-scale ablations.
- **Li, Zhang, Xu, Cai, Zhang (2026).** Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting. arXiv. https://arxiv.org/abs/2610.32141
  - A second control besides sampling ratios: per-domain loss weights at fixed exposure, tested on knowledge, mathematics and code.
- **Choshen, Zhang, Andreas (2024).** A Hitchhiker's Guide to Scaling Law Estimation. ICML 2025. https://arxiv.org/abs/2410.11840
  - The README's route to a step change is scaling-law extrapolation from short runs; this is the how-to, including the warning about seed variance.
- **Brandfonbrener, Anand, Vyas, Malach, Kakade (2024).** Loss-to-Loss Prediction: Scaling Laws for All Datasets. arXiv. https://arxiv.org/abs/2411.12925
  - Shifted power laws relate loss on one dataset to loss on another and extrapolate to 20x the fitted FLOP budget.
- **Hägele, Bakouch, Kosson, Ben Allal, Von Werra, Jaggi (2024).** Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2405.18392
  - Constant-plus-cooldown schedules let a single run be branched at several lengths, which makes short-run ranking cheaper.

## Datasets and benchmarks
- **Penedo, Kydlíček, Ben Allal, Lozhkov, et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- **Ben Allal, Lozhkov, Bakouch, Martín Blázquez, Penedo, et al. (2025).** SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. arXiv. https://arxiv.org/abs/2502.02737
  - Source paper for FineMath and for a code set (Stack-Edu), with a documented web/code/math multi-stage mixture.
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Defines DCLM-Pool (240T tokens), DCLM-Baseline and 53 evaluations; project page https://www.datacomp.ai/dclm/.

## Open-source code and tools
- LiGO code: https://github.com/VITA-Group/LiGO ; G_stack code and models: https://llm-stacking.github.io/ ; schedule code: https://github.com/epfml/schedules-and-scaling (all named in the entries above).
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - Compact full pipeline (tokeniser, pretraining, evaluation); secondary sources put the depth-20 run at about 4 hours on 8 H100 (about $100). Walkthrough: https://github.com/karpathy/nanochat/discussions/1.
- **Jordan et al. (2024-2026).** modded-nanogpt (NanoGPT speedrun). GitHub. https://github.com/KellerJordan/modded-nanogpt
  - Community record of recipe changes (Muon, QK-norm, ReLU-squared MLPs, value embeddings, sliding-window attention, YaRN) for reaching 3.28 validation loss on FineWeb with 8 H100s.

## Breakthroughs in related fields that could transfer
- **Liew, Kato, Takase (2025).** Scaling Laws for Upcycling Mixture-of-Experts Language Models. ICML 2025. https://arxiv.org/abs/2502.03009
  - Growth by upcycling a dense checkpoint into an MoE has its own scaling law, including a term that limits its benefit at large budgets.
- **Nakamura, Akiba, Fujii, Oda, Yokota, et al. (2025).** Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization. ICLR 2025. https://arxiv.org/abs/2502.19261
  - Dense-to-MoE conversion with partial re-initialisation of expert weights to speed up specialisation.
- **Ning, Li, Zeng, Kang, Xiong (2026).** Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes. arXiv. https://arxiv.org/abs/2605.05724
  - 1,197 agent trials with no human edits; feeding crashes, budget overruns and gate misses back as lineage is what produced later working recipe edits (Parameter Golf bpb -0.81%, NanoChat-D12 CORE +38.7%).

## Published results to beat
- HFM best so far on this track: -0.7048 negative_public_validation_bpb, 20 of 26 proposals scored, none accepted (project README).
- G_stack: same loss at 194B tokens as a conventionally trained 7B model at 300B tokens (https://arxiv.org/abs/2405.15319).
- LiGO: 44.7% (BERT-Base) and 22.5% (GPT2-Medium) FLOP savings (https://arxiv.org/abs/2303.00980).
- DoReMi: baseline accuracy with 2.6x fewer steps (https://arxiv.org/abs/2305.10429). Data Mixing Laws: default mix needs 48% more steps to match the optimised mix (https://arxiv.org/abs/2403.16952).
- Loss-to-loss prediction holds at 20x the FLOP budget used for fitting (https://arxiv.org/abs/2411.12925).

## Unverified leads
- UNVERIFIED: Hugging Face dataset cards for FineMath (HuggingFaceTB/finemath) and DCLM-Baseline (mlfoundations/dclm-baseline-1.0). Existence of the datasets is confirmed through the SmolLM2 and DCLM papers, but the exact repository names and sizes were not.
- UNVERIFIED: A public GitHub-code pretraining set (for example the codeparrot/github-code dataset or The Stack). Not confirmed; Stack-Edu is confirmed only via the SmolLM2 paper.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
