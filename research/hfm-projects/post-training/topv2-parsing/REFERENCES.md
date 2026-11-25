# References: TOPv2 compositional semantic parsing

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Gupta et al. (2018).** Semantic Parsing for Task Oriented Dialog using Hierarchical Representations. EMNLP 2018. https://arxiv.org/abs/1810.07942 [A]
  - Defines the nested intent and slot tree format (TOP).
- **Chen et al. (2020).** Low-Resource Domain Adaptation for Compositional Task-Oriented Semantic Parsing. EMNLP 2020. https://arxiv.org/abs/2010.03546 [A]
  - Introduces TOPv2 (8 domains) with BART-based seq2seq baselines and low-resource splits.
- **Rongali et al. (2020).** Don't Parse, Generate! A Sequence to Sequence Architecture for Task-Oriented Semantic Parsing. WWW 2020. https://arxiv.org/abs/2001.11458 [A]
  - Direct seq2seq generation of the parse with no restriction on the schema; 3.3% to 7.7% relative exact-match gains on ATIS, SNIPS and TOP. The baseline formulation a LoRA-tuned decoder follows.
- **Scholak, Schucher, Bahdanau (2021).** PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models. EMNLP 2021. https://arxiv.org/abs/2109.05093 [A]
  - Rejects inadmissible tokens at each step with an incremental parser; turned fine-tuned T5 models into state of the art on text-to-SQL without retraining.
- **Shin et al. (2021).** Constrained Language Models Yield Few-Shot Semantic Parsers. EMNLP 2021. https://arxiv.org/abs/2104.08768 [A]
  - Grammar-constrained decoding into a canonical form for semantic parsing.

## Recent advances and breakthroughs (2023-2026)
- **Fernández-González (2022).** Shift-Reduce Task-Oriented Semantic Parsing with Stack-Transformers. Cognitive Computation. https://arxiv.org/abs/2210.11984 [A]
  - Transition-based (shift-reduce) parsers for TOP trees, with the in-order system best; the highest TOPv2 exact match found in this session (below).
- **Shrivastava et al. (2021).** Span Pointer Networks for Non-Autoregressive Task-Oriented Semantic Parsing. arXiv:2104.07275. https://arxiv.org/abs/2104.07275 [A]
  - Predicts slot values as span endpoints instead of copying text token by token; shrinks the output and removes text-copy errors.
- **Cole et al. (2021).** Graph-Based Decoding for Task Oriented Semantic Parsing. EMNLP 2021 Findings. https://arxiv.org/abs/2109.04587 [A]
  - Graph-based decoding instead of sequence decoding; competitive on the standard setting and more data-efficient.
- **Rongali et al. (2023).** Low-Resource Compositional Semantic Parsing with Concept Pretraining. EACL 2023. https://arxiv.org/abs/2301.09809 [A]
  - Few-shot and zero-shot TOPv2 results with concept-aware pre-training.
- **Geng et al. (2023).** Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. EMNLP 2023. https://arxiv.org/abs/2305.13971 [A]
  - Grammar constraints let off-the-shelf models beat unconstrained decoding and sometimes fine-tuned models on structured tasks; the route the README proposes. Code: https://github.com/epfl-dlab/GCD
- **Park et al. (2024).** Grammar-Aligned Decoding. NeurIPS 2024. https://arxiv.org/abs/2405.21047 [A]
  - Shows that plain grammar masking distorts the model's distribution and can pick low-likelihood parses; gives a corrected sampler. Measure exact match with and without constraints.
- **Roy et al. (2022).** BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing. NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2206.10668 [A]
  - Constrained decoding for fine-tuned and prompted language models on parsing datasets, including MTOP in the same bracketed intent and slot format; reports that constrained fine-tuned models match or beat task-specific parsers.

## Datasets and benchmarks
- **TOPv2** dataset link as printed in the shift-reduce paper: https://fb.me/TOPv2Dataset [A]
- **TOP** dataset (44k queries): fb.me/semanticparsingdialog (from the TOP abstract) [A]

## Open-source code and tools
- **Willard, Louf (2023).** Efficient Guided Generation for Large Language Models (Outlines). arXiv:2307.09702. https://arxiv.org/abs/2307.09702 [A]
  - Regular-expression and grammar-guided generation with little overhead.
- **Dong et al. (2024).** XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models. MLSys 2025. https://arxiv.org/abs/2411.15100 [A]
  - Context-free grammar decoding, reported up to 100x faster than earlier engines; suits bracket-balanced TOP trees.
- **GCD code.** https://github.com/epfl-dlab/GCD [A]

## Breakthroughs in related fields that could transfer
- **Poesia et al. (2022).** Synchromesh: Reliable code generation from pre-trained language models. arXiv:2201.11227. https://arxiv.org/abs/2201.11227 [A]
  - Combines constrained decoding with retrieval of structurally similar examples; the two gave complementary gains.
- **Wang et al. (2023).** Grammar Prompting for Domain-Specific Language Generation with Large Language Models. arXiv:2305.19234. https://arxiv.org/abs/2305.19234 [A]
  - The model first predicts a small grammar for the input, then the output under that grammar; for TOPv2 this means predicting the domain's intent and slot inventory first.
- **Mekala, Wolfe, Roy (2022).** ZEROTOP: Zero-Shot Task-Oriented Semantic Parsing using Large Language Models. arXiv:2212.10815. https://arxiv.org/abs/2212.10815 [A]
  - Decomposes parsing into QA sub-problems; an alternative formulation for domains with little data.

## Published results to beat
- TOPv2 exact match, full data: 88.11 (in-order shift-reduce), 87.98 (top-down), 86.27 (bottom-up), from Fernández-González's Table 5, where the column is marked "TOPv2*", so check which domain subset it covers [A].
- 87 exact match on TOPv2 for the non-autoregressive span pointer network [A].
- HFM's 83.0 comes from 12,288 training rows; the figures above use the full training set, so the gap of about 5 points mixes data size and method.

## Free ways to run
- Outlines and XGrammar are open-source decoding libraries [A]; the grammar for TOPv2 can be built from the label inventory at no model cost.

## Unverified leads
- UNVERIFIED: RINE (recursive insertion-based encoder) is recalled as reporting about 87 exact match on TOPv2; the paper was not found by its guessed arXiv ID.
- UNVERIFIED: no published fine-tuning result for a 7B decoder-only model on TOPv2 was found.
