# References: MapQA Map Visual Question Answering

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Chang et al. (2022).** MapQA: A Dataset for Question Answering on Choropleth Maps. NeurIPS 2022 Table Representation Workshop. https://arxiv.org/abs/2211.08545 [A]
  - The task paper: about 800K QA pairs over about 60K maps in three subsets (MapQA-U, -R, -S), scored by Jaccard index. Its V-MODEQA model first extracts the underlying region-to-value table, then answers from the table.
- **Liu et al. (2022).** DePlot: One-shot visual language reasoning by plot-to-table translation. Findings of ACL 2023. https://arxiv.org/abs/2212.10505 [A]
  - Image-to-table then reason over the table; the general form of V-MODEQA, and a ready pre-task (map to state-value table) for a LoRA curriculum.
- **Lee et al. (2022).** Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. ICML 2023. https://arxiv.org/abs/2210.03347 [A]
  - Variable-resolution input that keeps aspect ratio; the argument for native-resolution processing of maps with small legends.
- **Liu et al. (2022).** MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering. ACL 2023. https://arxiv.org/abs/2212.09662 [A]
  - Chart derendering as a pre-training task; nearly 20% over prior work on PlotQA and ChartQA.

## Recent advances and breakthroughs (2023-2026)
- **Wang et al. (2024).** Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191. https://arxiv.org/abs/2409.12191 [A]
  - Base model; dynamic resolution means the pixel budget per image is a recipe choice. Code: https://github.com/QwenLM/Qwen2-VL
- **Laurençon et al. (2024).** What matters when building vision-language models? (Idefics2). arXiv:2405.02246. https://arxiv.org/abs/2405.02246 [A]
  - Releases The Cauldron training collection, which lists MapQA with 37,417 images and 483,416 QA pairs in a ready format.
- **Li et al. (2024).** LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326. https://arxiv.org/abs/2408.03326 [A]
  - Lists MapQA (via MathV360K) among its training sources, so general VLMs may have seen MapQA-style items; relevant when judging base-model scores.
- **Zhang et al. (2024).** TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning. arXiv:2404.16635. https://arxiv.org/abs/2404.16635 [A]
  - A 3B model beats 13B chart models by merging similar vision tokens (affordable high resolution) and by learning to emit programs for numeric answers. Both ideas apply to a 2B model on maps. Code: https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart
- **Masry et al. (2024).** ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild. arXiv:2407.04172. https://arxiv.org/abs/2407.04172 [A]
  - Visual instruction-tuning data for charts with released code, checkpoints and dataset; a recipe to adapt for legend-reading tasks. Code: https://github.com/vis-nlp/ChartGemma
- **Masry et al. (2023).** UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning. arXiv:2305.14761. https://arxiv.org/abs/2305.14761 [A]
  - Low-level pre-tasks (extract visual elements and data) before high-level QA; the staged curriculum the README proposes.
- **Hu et al. (2024).** mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding. arXiv:2403.12895. https://arxiv.org/abs/2403.12895 [A]
  - Structure-parsing tasks across documents, tables and charts as a first training stage.

## Datasets and benchmarks
- **MapQA** (above): 62,367 images and 795,525 questions in the paper's statistics table; question types are surface, retrieval and relational [A].
- **Masry et al. (2022).** ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. Findings of ACL 2022. https://arxiv.org/abs/2203.10244 [A]
  - Closest larger benchmark; auxiliary training data and a transfer check.
- **Dihan et al. (2024).** MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models. ICML 2025. https://arxiv.org/abs/2501.00316 [A]
  - A different map benchmark (places, routes, distances); useful only as an out-of-domain check. https://mapeval.github.io/

## Open-source code and tools
- **Qwen2-VL code.** https://github.com/QwenLM/Qwen2-VL [A]
- **Liu et al. (2024).** DoRA: Weight-Decomposed Low-Rank Adaptation. https://arxiv.org/abs/2402.09353 , code https://github.com/NVlabs/DoRA [A]
  - Reports gains over LoRA on visual instruction tuning.

## Breakthroughs in related fields that could transfer
- **Chen et al. (2022).** Program of Thoughts Prompting. TMLR 2023. https://arxiv.org/abs/2211.12588 [A]
  - Relational MapQA questions (compare, count, list regions) are set operations over an extracted table; emitting the table then computing avoids arithmetic errors in a 2B model.
- **Wortsman et al. (2022).** Model soups. ICML 2022. https://arxiv.org/abs/2203.05482 [A]
  - Averaging adapters from several runs as a free last step.

## Published results to beat
From the MapQA paper, Table 3, Jaccard index (%) [A]:
- V-MODEQA with real OCR (Tesseract): 74.5 on MapQA-U, 85.8 on MapQA-R, 83.3 on MapQA-S. With oracle OCR: 98.4, 89.9, 87.1.
- PReFIL with Tesseract: 73.8, 77.6, 69.2. Human: 97.8, 92.4, 89.8.
- By question type the gap is in retrieval and relational questions (for example V-MODEQA with Tesseract on MapQA-U: surface 100.0, retrieval 60.7, relational 81.0), and OCR of legend and region text accounts for much of it.
- HFM's 0.8283 mean Jaccard on discrete-legend questions is near the 2022 real-OCR numbers and well below human, so headroom of several points is plausible; the HFM dev set is a 1,000-item subset and not the same split.

## Free ways to run
- Qwen2-VL-2B is open-weight and small; MapQA is available inside The Cauldron collection described in the Idefics2 paper [A].

## Unverified leads
- UNVERIFIED: a published Qwen2-VL-2B (zero-shot or fine-tuned) score on MapQA was not found.
- UNVERIFIED: the Hugging Face path of The Cauldron (believed to be HuggingFaceM4/the_cauldron) was not confirmed; huggingface.co was blocked.
