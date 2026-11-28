# References: Building an Agent for Preserving Knowledge from Research Papers

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Adams et al. (2023).** From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting. NewSum workshop 2023. https://arxiv.org/abs/2309.04269 [S][A]
  - Repeatedly adds missing salient entities without growing the summary; a ready procedure for packing more facts into a fixed 3,000-character note, with a measured informativeness versus readability trade-off.
- **Deutsch, Roth (2021).** Incorporating Question Answering-Based Signals into Abstractive Summarization via Salient Span Selection. EACL 2023. https://arxiv.org/abs/2111.07935 [S][A]
  - Generates wh-questions answered by noun phrases in the source and keeps spans whose questions the summary should answer; the direct precedent for "question anticipation" as a content selector.
- **Chen et al. (2023).** Dense X Retrieval: What Retrieval Granularity Should We Use? arXiv:2312.06648. https://arxiv.org/abs/2312.06648 [A]
  - Self-contained propositions beat passages as units for downstream QA under a fixed budget; supports writing notes as atomic fact lines.
- **Liu et al. (2023).** Lost in the Middle: How Language Models Use Long Contexts. TACL. https://arxiv.org/abs/2307.03172 [A]
  - Readers use the start and end of a context best; informs the order of facts inside the note.

## Recent advances and breakthroughs (2023-2026)
- **Zhang et al. (2024).** Chain of Agents: Large Language Models Collaborating on Long-Context Tasks. NeurIPS 2024. https://arxiv.org/abs/2406.02818 [S][A]
  - Worker calls read consecutive chunks and pass a running message to a manager; up to 10% over RAG and full-context baselines. Fits the 8,000-character per-call limit exactly.
- **Sarthi et al. (2024).** RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024. https://arxiv.org/abs/2401.18059 [S][A]
  - Cluster-and-summarise tree; gives published QASPER numbers for retrieval plus a strong reader (see below).
- **Lee et al. (2024).** A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts (ReadAgent). arXiv:2402.09727. https://arxiv.org/abs/2402.09727 [A]
  - Compresses pages into gists and re-reads originals on demand; the gist-only condition is the same setting as this task, and the paper measures it.
- **Pan et al. (2024).** LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. Findings of ACL 2024. https://arxiv.org/abs/2403.12968 [S][A]
  - A small token-classifier that drops tokens at 2x to 5x compression; a cheap extractive pre-filter before the frozen model writes the note.
- **Jiang et al. (2023).** LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. ACL 2024. https://arxiv.org/abs/2310.06839 [S][A]
  - Question-aware coarse-to-fine compression; pair it with anticipated questions when the real ones are hidden.
- **Zhang et al. (2025).** An Empirical Study on Prompt Compression for Large Language Models. ICLR 2025 workshop. https://arxiv.org/abs/2505.00019 [S][A]
  - Compares compression methods across 13 datasets; compression hurts more on long contexts, and moderate compression sometimes helps. Toolkit: https://github.com/3DAgentWorld/Toolkit-for-Prompt-Compression
- **Zhang, Wan, Cattan et al. (2024).** QAPyramid: Fine-grained Evaluation of Content Selection for Text Summarization. COLM 2025. https://arxiv.org/abs/2412.07096 [S][A]
  - Scores a summary by the QA pairs it can answer; a local proxy metric for answer_f1 that needs no hidden reader prompt.

## Datasets and benchmarks
- **Dasigi et al. (2021).** A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (QASPER). NAACL 2021. https://arxiv.org/abs/2105.03011 [S][A]
  - 5,049 questions over 1,585 NLP papers, written by readers who saw only title and abstract; the public equivalent named in the README. Dataset card: https://huggingface.co/datasets/allenai/qasper [S]
- **Baumgärtner, Briscoe, Gurevych (2025).** PeerQA: A Scientific Question Answering Dataset from Peer Reviews. NAACL 2025. https://arxiv.org/abs/2502.13668 [S][A]
  - 579 reviewer questions over 208 papers (average 12k tokens); reports that top retrieved passages beat feeding the full text. Code and data: https://github.com/UKPLab/peerqa
- **Chen et al. (2026).** RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension. ACL 2026. https://arxiv.org/abs/2601.14289 [S][A]
  - 61.3K QA pairs over 4,150 papers with a conciseness-adjusted score (best model 68.2%, 37.46% after the conciseness adjustment); a larger held-out set for testing note quality.
- **Shaham et al. (2022, 2023).** SCROLLS and ZeroSCROLLS. EMNLP 2022 / Findings of EMNLP 2023. https://arxiv.org/abs/2201.03533 and https://arxiv.org/abs/2305.14196 [A]
  - Long-document benchmark suites with a unified format and leaderboards for zero-shot comparison.

## Open-source code and tools
- **LLMLingua-2 code.** https://aka.ms/LLMLingua-2 [A]
- **Chain of Density summaries.** https://huggingface.co/datasets/griffin/chain_of_density [A]
- **Khattab et al. (2023).** DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. https://arxiv.org/abs/2310.03714 , code https://github.com/stanfordnlp/dspy [A]
  - Optimises multi-call prompt pipelines against a metric; the note-writer prompts can be tuned against a QASPER answer-F1 proxy.

## Breakthroughs in related fields that could transfer
- **Agrawal et al. (2025).** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. ICLR 2026. https://arxiv.org/abs/2507.19457 [A]
  - Evolves prompts from reflective feedback with up to 35x fewer rollouts than RL; suited to a harness where each evaluation costs money. Code: https://github.com/gepa-ai/gepa
- **Packer et al. (2023).** MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560. https://arxiv.org/abs/2310.08560 [A]
  - Tiered memory with explicit write decisions; a design reference for what to keep under a hard size limit.

## Published results to beat
- QASPER answer F1 with the paper available: 55.7 for RAPTOR + GPT-4, 53.0 for DPR + GPT-4, 50.2 for BM25 + GPT-4, and 53.9 for CoLT5-XL (all from the RAPTOR paper) [S]. The original paper put models at least 27 F1 below humans [A]. These bound what a note-only reader can reach; HFM's 0.4794 is on its own question set.
- The QASPER dataset card lists the Longformer baseline at 39.37 F1 in one place and 33.63 token F1 in another [S]; do not mix the two.

## Free ways to run
- QASPER, PeerQA and RPC-Bench are public. LLMLingua-2 is a small encoder model that runs on CPU or a free GPU [A].

## Unverified leads
- UNVERIFIED: the full-paper answer-F1 ceiling for a current frozen reader on QASPER (the README's next step) was not found in any source.
- UNVERIFIED: SciDQA and SPIQA were named in a search summary as related scientific QA benchmarks; titles and URLs were not confirmed.
- UNVERIFIED: free-tier API limits were not checked in this session.
