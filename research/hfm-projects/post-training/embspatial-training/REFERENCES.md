# References: Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Hu et al. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685 [A]
- **Liu et al. (2024).** DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024. https://arxiv.org/abs/2402.09353 [A]
  - The adapter type in HFM's best runs; code at https://github.com/NVlabs/DoRA
- **Wortsman et al. (2022).** Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. ICML 2022. https://arxiv.org/abs/2203.05482 [A]
  - Evidence for averaging the weights of several fine-tuned models; the basis for HFM's 0.5/0.5 adapter-averaging idea.
- **Yadav et al. (2023).** TIES-Merging: Resolving Interference When Merging Models. NeurIPS 2023. https://arxiv.org/abs/2306.01708 [A]
  - Handles sign conflicts that plain averaging ignores; the next thing to try if two complementary adapters average to no gain. Code: https://github.com/prateeky2806/ties-merging

## Recent advances and breakthroughs (2023-2026)
- **Du et al. (2024).** EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models. ACL 2024. https://arxiv.org/abs/2406.05756 [A]
  - The benchmark paper. It also releases EmbSpatial-SFT, an instruction-tuning set built only from the Matterport3D training split, with two tasks: spatial relation identification and object localisation as an auxiliary task.
- **Chen et al. (2024).** SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. arXiv:2401.12168. https://arxiv.org/abs/2401.12168 [A]
  - Automatic generation of spatial VQA from images lifted to metric 3D; the template for synthesising relation pairs at scale.
- **Cheng et al. (2024).** SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models. NeurIPS 2024. https://arxiv.org/abs/2406.01584 [A]
  - Region-aware spatial reasoning with a released dataset and benchmark; a second data-construction recipe.
- **Song et al. (2024).** RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics. CVPR 2025. https://arxiv.org/abs/2411.16537 [A]
  - 1M images, 5k 3D scans and 3M annotated spatial relations from egocentric indoor scans; the largest matching source for the README's "synthesise pairs from 3D scene datasets" route. Its related-work table lists EmbSpatial-Bench.
- **Ray et al. (2024).** SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models. COLM 2025. https://arxiv.org/abs/2412.07755 [A]
  - 175K simulator-generated QA pairs over 20K scenes; reports that synthetic training improves static spatial relations on real-image benchmarks.
- **Li et al. (2025).** SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models. arXiv:2510.08531. https://arxiv.org/abs/2510.08531 [A]
  - Three-stage curriculum starting with object localisation, then spatial understanding, then reasoning, on 26,610 samples; 23.4% average gain over the base model. Supports localisation-first curricula for small data.
- **Cai et al. (2024).** SpatialBot: Precise Spatial Understanding with Vision Language Models. arXiv:2406.13642. https://arxiv.org/abs/2406.13642 [A]
  - Trained on the SpatialQA dataset and evaluated on SpatialBench; model, code and data at https://github.com/BAAI-DCAI/SpatialBot

## Datasets and benchmarks
- **EmbSpatial-Bench**: 3,640 four-choice questions, 2,181 images, 6 relations, from Matterport3D, AI2-THOR and ScanNet (paper Table 1) [A].
- **Kamath, Hessel, Chang (2023).** What's "up" with vision-language models? Investigating their struggle with spatial reasoning. EMNLP 2023. https://arxiv.org/abs/2310.19785 [A]
  - Controlled left/right/above/under sets; a diagnostic for which relations fail. Data: https://github.com/amitakamath/whatsup_vlms
- **Fu et al. (2024).** BLINK: Multimodal Large Language Models Can See but Not Perceive. ECCV 2024. https://arxiv.org/abs/2404.12390 [A]
  - Visual perception tasks on which multimodal models score near chance; a held-out check that training has not overfitted to EmbSpatial phrasing.
- **Ma et al. (2024).** 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark. ICCV 2025. https://arxiv.org/abs/2412.07825 [A]
- **Yang et al. (2024).** Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces. arXiv:2412.14171. https://arxiv.org/abs/2412.14171 [A]
  - Finds that chain-of-thought and self-consistency do not help spatial tasks, while explicit cognitive maps do; argues against prompt-only fixes.

## Open-source code and tools
- **Wang et al. (2024).** Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191. https://arxiv.org/abs/2409.12191 [A]
  - Base model paper (2B, 8B, 72B); code at https://github.com/QwenLM/Qwen2-VL

## Breakthroughs in related fields that could transfer
- **Yuan et al. (2024).** RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics. arXiv:2406.10721. https://arxiv.org/abs/2406.10721 [A]
  - Trains a VLM to output points for spatial affordances and beats GPT-4o by 21.8% on that task; point outputs as an auxiliary target parallel EmbSpatial-SFT's localisation task.

## Published results to beat
- From the EmbSpatial-Bench paper [A]: best zero-shot model 49.11% (Qwen-VL-Max, generation protocol) and 43.85% (likelihood protocol); human 90.33%.
- Same paper, Table 3: MiniGPT-v2 fine-tuned on EmbSpatial-SFT reaches 78.10% overall (likelihood protocol), up from 43.85%; without the object-localisation task 77.34%; tuning only the connector without LoRA 43.76%.
- HFM's 0.785 with Qwen2-VL-2B is level with that 78.10%, which suggests further gains need data beyond the EmbSpatial-SFT recipe, not more adapter tuning.

## Free ways to run
- Qwen2-VL-2B is an open 2B model (paper above) and LoRA or DoRA training of a model this size should fit a single free-tier GPU (an estimate, not a sourced figure).

## Unverified leads
- UNVERIFIED: the public location of EmbSpatial-Bench and EmbSpatial-SFT (believed to be on Hugging Face under the FlagEval or first author's account) was not confirmed.
- UNVERIFIED: a published Qwen2-VL-2B zero-shot score on EmbSpatial-Bench was not found; the README's "publish the base-model score" step is still needed.
