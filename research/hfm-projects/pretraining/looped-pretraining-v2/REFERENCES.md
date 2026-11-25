# References: Looped pretraining with two concurrent four-GPU ideas per batch

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (2018).** Universal Transformers. ICLR 2019. https://arxiv.org/abs/1807.03819
  - The original depth-recurrent transformer: one block applied repeatedly, with a per-step (timestep) signal and optional ACT halting.
- **Lan, Chen, Goodman, Gimpel, Sharma, Soricut (2019).** ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR 2020. https://arxiv.org/abs/1909.11942
  - Standard reference for cross-layer weight sharing; code at https://github.com/google-research/ALBERT.
- **Xu, Sato (2024).** On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding. ICML 2025. https://arxiv.org/abs/2410.01405
  - Published basis for loop-index embeddings.
- **Saunshi, Dikkala, Li, Kumar, Reddi (2025).** Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025. https://arxiv.org/abs/2502.17416
  - Looping may cost perplexity even when it helps reasoning; the metric here is DCLM bits-per-byte, so a dense control at equal training FLOPs is needed.

## Recent advances and breakthroughs (2023-2026)
- **Schwethelm, Rueckert, Kaissis (2026).** How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. NeurIPS 2026 (spotlight). https://arxiv.org/abs/2604.21106
  - At matched training compute a looped model loses to a dense one (exponent 0.46); use the exponent, not raw loss, to tell loop improvements from recipe improvements.
- **Ekbote, Pagnoni, Sachan, Bhosale, Lewis, et al. (2026).** How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference. arXiv. https://arxiv.org/abs/2610.40325
  - Under a training-FLOP limit, looping a middle block gave better returns than looping the whole stack.
- **Marchenko, Bezrukov, Kashurin, Fedorova, Bocharov, et al. (2026).** Closing the Loop: Practical Training Recipes for Looped Language Models. arXiv. https://arxiv.org/abs/2610.00673
  - A looped recipe with a parameter-matched dense control and a minimal dense-to-looped conversion (one input-mixing scalar plus a smoothed exit loss), which fits continuing from a published checkpoint.
- **Liang, Zhang, Zhang, Wang, Lin (2026).** UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute. arXiv. https://arxiv.org/abs/2610.41720
  - Starts shallow, grows depth by copying layers, and spends the saved compute on looping a shared middle block, matching a dense baseline in parameters, tokens and training FLOPs.
- **Prairie, Novack, Berg-Kirkpatrick, Fu (2026).** Parcae: Scaling Laws For Stable Looped Language Models. arXiv. https://arxiv.org/abs/2604.12946
  - Traces loss spikes in looped training to the spectral norm of the input-injection parameters and constrains it; reports up to 6.3% lower validation perplexity than earlier looped models.
- **Monea, Ramji, El-Kurdi, Lastras, Artzi, et al. (2026).** The Surprising Effectiveness of Shared Memory in Looped Transformers. arXiv. https://arxiv.org/abs/2610.02383
  - Reports FineWeb-Edu validation perplexity for 150M to 1B looped models, the same training data as this track.
- **Liang, Zhang, Du, Wu, Ma, et al. (2026).** Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder. arXiv. https://arxiv.org/abs/2610.28489
  - FineWeb-Edu results from 60M to 1B under matched effective depth and tokens.
- **Kapl, Angelis, Maile, von Oswald, Bauer (2026).** From Growing to Looping: A Unified View of Iterative Computation in LLMs. arXiv. https://arxiv.org/abs/2602.16490
  - Links depth growth and looping mechanistically, and reports that math-heavy cooldown data mixtures give the largest reasoning gains for depth-grown models.
- **Brandfonbrener, Anand, Vyas, Malach, Kakade (2024).** Loss-to-Loss Prediction: Scaling Laws for All Datasets. arXiv. https://arxiv.org/abs/2411.12925
  - Train on FineWeb-Edu, score on hidden DCLM: this paper gives the shifted power law to predict one loss from the other, so a public FineWeb-Edu validation loss can stand in for the hidden score.
- **Ye, Liu, Sun, Zhan, Zhou, Qiu (2024).** Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance. ICLR 2025. https://arxiv.org/abs/2403.16952
  - The current best idea changes document sampling to q = 0.25p + 0.75 FineWeb-Edu; this paper fits loss as a function of mixture weights from a few short runs.
- **Shukor, Bethune, Busbridge, Grangier, Fini, et al. (2025).** Scaling Laws for Optimal Data Mixtures. arXiv. https://arxiv.org/abs/2507.09404
  - Predicts loss as a function of model size, tokens and domain weights from a few small runs and extrapolates to larger scales and unseen weights.

## Datasets and benchmarks
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Source of the hidden evaluation distribution.
- **Penedo, Kydlíček, Ben Allal, Lozhkov, et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source of the main training data.

## Open-source code and tools
- **Jordan et al. (2024-2026).** modded-nanogpt (NanoGPT speedrun). GitHub. https://github.com/KellerJordan/modded-nanogpt
  - Candidate known-good recipe to freeze as the floor.
- **Karpathy (2025).** nanochat. GitHub. https://github.com/karpathy/nanochat
  - Compact full pipeline (tokeniser, pretraining, evaluation); secondary sources put the depth-20 run at about 4 hours on 8 H100 (about $100). Walkthrough: https://github.com/karpathy/nanochat/discussions/1.
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Code for recursive transformers: https://github.com/raymin0223/mixture_of_recursions.
- **Karpathy (2026).** autoresearch. GitHub. https://github.com/karpathy/autoresearch
  - Lets each of the two concurrent ideas be pre-screened on one GPU in 5-minute runs before spending a 4-GPU slot.

## Breakthroughs in related fields that could transfer
- **Jolicoeur-Martineau (2025).** Less is More: Recursive Reasoning with Tiny Networks. arXiv. https://arxiv.org/abs/2510.04871
  - Evidence that deep supervision carries most of the gain in recursive models.
- **Choshen, Zhang, Andreas (2024).** A Hitchhiker's Guide to Scaling Law Estimation. ICML 2025. https://arxiv.org/abs/2410.11840
  - With many runs failing, fit curves to intermediate checkpoints: final loss is predictable from about one third of a run.
- **Ning, Li, Zeng, Kang, Xiong (2026).** Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes. arXiv. https://arxiv.org/abs/2605.05724
  - 1,197 agent trials with no human edits; feeding crashes, budget overruns and gate misses back as lineage is what produced later working recipe edits (Parameter Golf bpb -0.81%, NanoChat-D12 CORE +38.7%).

## Published results to beat
- HFM best so far on this track: -1.152 negative_hidden_dclm_bpb, 24 of 30 proposals scored, none accepted as an improvement (project README).
- LPT hybrid: FineWeb-Edu validation perplexity 1.12 to 1.82 below a same-size standard transformer with five recursions (https://arxiv.org/abs/2610.02383).
- Looped 410M model at 4 recurrences matches a 580M dense model but costs the training compute of a 1B dense model (https://arxiv.org/abs/2604.21106); a looped entry must beat this trade to win at fixed compute.
- Data Mixing Laws: optimised mixture equals the default mixture trained 48% longer (https://arxiv.org/abs/2403.16952).

## Unverified leads
- UNVERIFIED: Hugging Face dataset cards for FineMath (HuggingFaceTB/finemath) and DCLM-Baseline (mlfoundations/dclm-baseline-1.0). Existence of the datasets is confirmed through the SmolLM2 and DCLM papers, but the exact repository names and sizes were not.
- UNVERIFIED: Lee et al., "Deeply-Supervised Nets" (2015), the original deep-supervision paper. Not searched.
- UNVERIFIED: No public bits-per-byte figure for nanochat depth-12/20/24 was found in search results; check the repository's report card before using it as an external reference.
- UNVERIFIED: A data-limited "slowrun" variant of the nanoGPT speedrun (fixed token budget instead of fixed time). Could not be searched before the search budget ran out.
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
