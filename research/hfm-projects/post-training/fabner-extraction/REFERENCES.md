# References: FabNER Material Manufacturing Entity Extraction

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Wang et al. (2023).** InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction. arXiv:2304.08085. https://arxiv.org/abs/2304.08085 [A]
  - Text-to-text NER with instructions across 32 datasets including FabNER; source of the BERT-base FabNER baseline.
- **Wang et al. (2023).** GPT-NER: Named Entity Recognition via Large Language Models. arXiv:2304.10428. https://arxiv.org/abs/2304.10428 [A]
  - NER recast as text generation for LLMs; the abstract reports its advantage is largest when labelled data is scarce.
- **Zaratiana et al. (2023).** GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. arXiv:2311.08526. https://arxiv.org/abs/2311.08526 [A]
  - Small encoder that matches entity-type prompts to spans in parallel; the natural encoder baseline for the README's next step.

## Recent advances and breakthroughs (2023-2026)
- **Zhou et al. (2023).** UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition. ICLR 2024. https://arxiv.org/abs/2308.03279 [A]
  - Distils an LLM into open-NER models (7B); its supervised table is the standard FabNER comparison point.
- **Ding et al. (2024).** Rethinking Negative Instances for Generative Named Entity Recognition (GNER). Findings of ACL 2024. https://arxiv.org/abs/2402.16602 [A]
  - Keeps negative (non-entity) instances in training, which supply context and sharpen label boundaries, and maps free-form output back to entities with a longest-common-subsequence matcher. Highest FabNER number found in this session (see below).
- **Yang et al. (2024).** Beyond Boundaries: Learning a Universal Entity Taxonomy across Datasets and Languages for Open NER (B2NER). COLING 2025. https://arxiv.org/abs/2406.11192 [A]
  - Builds a consistent entity taxonomy across datasets to remove conflicting label definitions; code, data and models at https://github.com/UmeanNever/B2NER
- **Li et al. (2024).** KnowCoder: Coding Structured Knowledge into LLMs for Universal Information Extraction. arXiv:2403.07969. https://arxiv.org/abs/2403.07969 [A]
  - Code-style schema representation for extraction; reports a FabNER result above UniNER.
- **Sainz et al. (2023).** GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction. ICLR 2024. https://arxiv.org/abs/2310.03668 [A]
  - Fine-tunes the model to follow annotation guidelines given in the prompt. FabNER is one of its zero-shot test sets and scores only about 25 F1, which shows FabNER labels are hard to infer from names alone.
- **Zamai et al. (2024).** Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER (SLIMER). arXiv:2407.01272. https://arxiv.org/abs/2407.01272 [A]
  - Per-type definitions and guidelines in the prompt give faster, more robust learning; cheap to add to the FabNER prompt, whose categories overlap.
- **Bogdanov et al. (2024).** NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data. arXiv:2402.15343. https://arxiv.org/abs/2402.15343 [A]
  - A pre-trained NER encoder that is data-efficient when fine-tuned; second encoder baseline.

## Datasets and benchmarks
- **Kumar, Starly (2022).** "FabNER": information extraction from manufacturing process science domain literature using named entity recognition. Journal of Intelligent Manufacturing 33(8):2393-2407. (Citation as given in the reference lists of UniNER, GNER, B2NER and IEPile [A]; publisher page not fetched.)
- **Gui et al. (2024).** IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus. ACL 2024. https://arxiv.org/abs/2402.14710 [A]
  - 33 IE datasets, FabNER among them, in one instruction format; a source of auxiliary NER data for multi-task warm-up.

## Open-source code and tools
- **B2NER** (code, data and models). https://github.com/UmeanNever/B2NER [A]
- **Willard, Louf (2023).** Efficient Guided Generation for Large Language Models (Outlines). arXiv:2307.09702. https://arxiv.org/abs/2307.09702 [A]
  - Finite-state constrained decoding for JSON and regular expressions; implements the README's constrained-JSON route.
- **Dong et al. (2024).** XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. MLSys 2025. https://arxiv.org/abs/2411.15100 [A]
  - Grammar-constrained decoding with near-zero overhead.

## Breakthroughs in related fields that could transfer
- **Park et al. (2024).** Grammar-Aligned Decoding. NeurIPS 2024. https://arxiv.org/abs/2405.21047 [A]
  - Shows that naive constrained decoding distorts the model's distribution and can lower quality; a caution to measure F1 with and without constraints.
- **Kalajdzievski (2023).** A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv:2312.03732. https://arxiv.org/abs/2312.03732 [A]
  - Relevant to the r32 adapters used in HFM's runs.

## Published results to beat
All are entity-level F1 on FabNER in supervised multi-dataset training, as printed in the papers' tables [A]:
- BERT-base 64.20; InstructUIE (11B) 78.38; UniNER-7B 81.87 (UniversalNER table; the same three figures are repeated in B2NER and GNER).
- InstructUIE's own table: 76.20 for its model against 64.20 for BERT-base.
- B2NER 78.82. KnowCoder 82.9 against a listed prior best of 82.3.
- GNER lists 83.20 and 85.39 beside the 78.38 and 81.87 baselines (read here as its T5 and LLaMA variants; check the column headers).
- HFM's 80.68 on a 512-example validation subset sits between InstructUIE and UniNER. The encoder baseline printed in these papers is 64.20, not "high 80s"; the best generative figure found is 85.39, so the confirmed headroom is about 5 points.

## Free ways to run
- GLiNER and NuNER are small encoders that train on a free GPU or CPU [A]. LoRA on Qwen2.5-7B on a small free GPU would need 4-bit loading (QLoRA: https://arxiv.org/abs/2305.14314 [A]).

## Unverified leads
- UNVERIFIED: the original FabNER paper is recalled as reporting a BiLSTM-CRF F1 near 88; this was not seen in this session and would explain the README's "high 80s" claim if true.
- UNVERIFIED: the dataset is believed to be on Hugging Face as DFKI-SLT/fabner; huggingface.co was blocked.
