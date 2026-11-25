# References: Looped-Transformer Pretraining Under a Fixed Compute Budget

Every entry below was confirmed on 2026-11-25 by a web search result or by its arXiv record. Numbers are taken from the cited source's abstract or from the search result named in the bullet; none were re-measured.

## Theory and established methods
- **Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (2018).** Universal Transformers. ICLR 2019. https://arxiv.org/abs/1807.03819
  - The reference design family for this track: shared blocks applied repeatedly, with optional halting.
- **Lan, Chen, Goodman, Gimpel, Sharma, Soricut (2019).** ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR 2020. https://arxiv.org/abs/1909.11942
  - Standard reference for cross-layer weight sharing; code at https://github.com/google-research/ALBERT.
- **Xu, Sato (2024).** On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding. ICML 2025. https://arxiv.org/abs/2410.01405
  - Closest published support for the README's idea: per-loop scaling parameters conditioned on a loop-index encoding, with theory for why plain looping saturates.
- **Saunshi, Dikkala, Li, Kumar, Reddi (2025).** Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025. https://arxiv.org/abs/2502.17416
  - Warns that looping can help reasoning while hurting perplexity; relevant because the score is bits-per-byte and the math shard is the one that stalls.
- **Yang, Lee, Nowak, Papailiopoulos (2023).** Looped Transformers are Better at Learning Learning Algorithms. ICLR 2024. https://arxiv.org/abs/2311.12424
  - Shows a looped block can match a 12-layer model on in-context regression with under 10% of the parameters; code at https://github.com/Leiay/looped_transformer.

## Recent advances and breakthroughs (2023-2026)
- **Schwethelm, Rueckert, Kaissis (2026).** How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models. NeurIPS 2026 (spotlight). https://arxiv.org/abs/2604.21106
  - Most useful diagnostic for this track: a block looped r times counts as about r^0.46 blocks of unique parameters, so the 8x3 reference pays training cost for depth it only partly uses; hyperconnections raised the exponent to 0.65.
- **Ekbote, Pagnoni, Sachan, Bhosale, Lewis, et al. (2026).** How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference. arXiv. https://arxiv.org/abs/2610.40325
  - Tests exactly the design space of this track (what is looped, how many loops, what is passed between loops); loop-specific low-rank adapters are the tested alternative to per-loop norms.
- **Bae, Fisch, Harutyunyan, Ji, Kim, Schuster (2024).** Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. ICLR 2025. https://arxiv.org/abs/2410.20672
  - Per-loop LoRA as a way to let shared blocks specialise by depth within the 9.0e7 unique-parameter cap.
- **Prairie, Novack, Berg-Kirkpatrick, Fu (2026).** Parcae: Scaling Laws For Stable Looped Language Models. arXiv. https://arxiv.org/abs/2604.12946
  - Explains loss spikes in looped training through the injection parameters; relevant to what is fed back at each loop.
- **Lin, Guo, Zhu, Ye, Eshraghian (2026).** Allocating Recurrent Compute in Looped Language Models. arXiv. https://arxiv.org/abs/2608.18230
  - Looping attention only and running the FFN once cuts forward cost, which this track penalises at 0.02 per doubling.
- **Monea, Ramji, El-Kurdi, Lastras, Artzi, et al. (2026).** The Surprising Effectiveness of Shared Memory in Looped Transformers. arXiv. https://arxiv.org/abs/2610.02383
  - A feedback design (shared first-loop memory) that improved perplexity on FineWeb-Edu instead of costing quality.
- **Liang, Zhang, Du, Wu, Ma, et al. (2026).** Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder. arXiv. https://arxiv.org/abs/2610.28489
  - Separates a single-pass context encoder from a weight-tied refinement decoder; lower FineWeb-Edu validation perplexity than full looping at 60M to 1B under matched effective depth and tokens.
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Adaptive per-token depth lowers average forward cost; check whether the scorer measures average or worst-case cost before using it.
- **Chen, Goyal, Krishnamoorthi (2026).** Scaling Laws for Looped Mixture of Experts. arXiv. https://arxiv.org/abs/2609.40316
  - The 3.6e8 total-parameter cap is 4x the unique non-embedding cap, which leaves room for sparse experts inside the loop; this paper models that combination.
- **Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein, et al. (2026).** A Mechanistic Analysis of Looped Reasoning Language Models. NeurIPS 2026. https://arxiv.org/abs/2604.11791
  - Finds that each layer in the cycle converges to its own fixed point and that loops repeat feed-forward-like stages; gives design guidance on what makes this stable.
- **Kapl, Angelis, Maile, von Oswald, Bauer (2026).** From Growing to Looping: A Unified View of Iterative Computation in LLMs. arXiv. https://arxiv.org/abs/2602.16490
  - Reports that math-heavy cooldown mixtures drive reasoning gains, a data-side explanation for the stalled math shard (the recipe is frozen here, so this is context, not an action).

## Datasets and benchmarks
- **Penedo et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.17557
  - Source paper for FineWeb (15T tokens) and FineWeb-Edu (1.3T tokens); dataset at https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu.
- **Li, Fang, Smyrnis, Ivgi, Jordan, et al. (2024).** DataComp-LM: In search of the next generation of training sets for language models. arXiv. https://arxiv.org/abs/2406.11794
  - Defines DCLM-Pool (240T tokens), DCLM-Baseline and 53 evaluations; project page https://www.datacomp.ai/dclm/.

## Open-source code and tools
- **Csordás, Irie, Schmidhuber, Potts, Manning (2024).** MoEUT: Mixture-of-Experts Universal Transformers. NeurIPS 2024. https://arxiv.org/abs/2405.16039
  - Working code for a shared-layer language model with MoE: https://github.com/robertcsordas/moeut.
- **Bae, Kim, Bayat, Kim, Ha, et al. (2025).** Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. NeurIPS 2025. https://arxiv.org/abs/2507.10524
  - Working code for recursive transformers with routers and KV sharing: https://github.com/raymin0223/mixture_of_recursions.
- **Xu, Sato (2024).** On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding. ICML 2025. https://arxiv.org/abs/2410.01405
  - Code for timestep-modulated looping: https://github.com/kevin671/tmlt.

## Breakthroughs in related fields that could transfer
- **Jolicoeur-Martineau (2025).** Less is More: Recursive Reasoning with Tiny Networks. arXiv. https://arxiv.org/abs/2510.04871
  - Deep supervision at every recursion drove the gains in small recursive reasoners; a per-loop loss is a loop-aware training pass that adds no inference cost.
- **Bai, Kolter, Koltun (2019).** Deep Equilibrium Models. arXiv. https://arxiv.org/abs/1909.01377
  - Treats a weight-tied network as a fixed-point problem; the starting point for later fixed-point and attractor language models.

## Published results to beat
- HFM best so far on this track: -0.7786 neg_cost_adjusted_bpb (project README). The looped reference (8 blocks x 3 loops) has no published score.
- Reference model math-shard bpb stalls at 0.993 to 0.994 between 1 h and 20 h of training (project README); any per-loop specialisation idea should first move this number.
- Recurrence-equivalence exponent 0.46 baseline, 0.65 with hyperconnections, 0.38 with truncated backpropagation (https://arxiv.org/abs/2604.21106).
- LPT hybrid, five recursions: FineWeb-Edu validation perplexity 1.12 to 1.82 lower than a same-size standard transformer (https://arxiv.org/abs/2610.02383).
- MixerLoop: 45.9% fewer recurrent-backbone projection FLOPs while keeping 41.5% of the full-loop gain at 110M parameters (https://arxiv.org/abs/2608.18230).

## Unverified leads
- UNVERIFIED: No paper was found that tests loop-index embeddings together with a separate RMSNorm per loop, the exact combination in the README; the nearest confirmed work is Xu and Sato (timestep encoding) and Ekbote et al. (loop-specific adapters).
- UNVERIFIED: Training code for the Geiping et al. recurrent-depth model (believed to be a public GitHub repository). Not confirmed.
- UNVERIFIED: Lee et al., "Deeply-Supervised Nets" (2015), the original deep-supervision paper. Not searched.
- UNVERIFIED: Hugging Face dataset cards for FineMath (HuggingFaceTB/finemath) and DCLM-Baseline (mlfoundations/dclm-baseline-1.0). Existence of the datasets is confirmed through the SmolLM2 and DCLM papers, but the exact repository names and sizes were not.
- UNVERIFIED: The web search budget for this session ran out before these could be checked.
