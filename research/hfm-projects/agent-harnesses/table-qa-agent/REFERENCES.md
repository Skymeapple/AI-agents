# References: Agent for Questions over Real Tables

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Chen et al. (2022).** Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. TMLR 2023. https://arxiv.org/abs/2211.12588 [A]
  - The model writes code and an interpreter computes; about 12% average gain over chain-of-thought, and more when combined with self-consistency.
- **Wang et al. (2022).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. https://arxiv.org/abs/2203.11171 [A]
  - Sample several answers and vote; the basis for voting over executed program outputs.
- **Cheng et al. (2022).** Binding Language Models in Symbolic Languages (Binder). ICLR 2023. https://arxiv.org/abs/2210.02875 [A]
  - SQL or Python with model calls embedded for the parts code cannot express; useful for questions needing semantic judgement on cell values.
- **Ye et al. (2023).** Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning (DATER). SIGIR 2023. https://arxiv.org/abs/2301.13808 [A]
  - Decomposes both the evidence (table) and the question before answering.
- **Wang et al. (2024).** Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding. ICLR 2024. https://arxiv.org/abs/2401.04398 [S][A]
  - Stepwise table operations with the intermediate table as state.

## Recent advances and breakthroughs (2023-2026)
- **Xiong et al. (2025).** TableReasoner: Advancing Table Reasoning Framework with Large Language Models. arXiv:2507.08046. https://arxiv.org/abs/2507.08046 [S][A]
  - First place in both subtasks of SemEval-2025 Task 8 (DataBench). Builds a compact table schema (global features plus example values), does schema linking to keep only question-relevant columns, then loops think, program, reflect. Removing schema generation and linking costs 5.37 points on the test set.
- **Osés Grijalba et al. (2025).** SemEval-2025 Task 8: Question Answering over Tabular Data. SemEval 2025. https://aclanthology.org/2025.semeval-1.324/ [S]
  - Task overview with the full ranking for DataBench and DataBench Lite; the authoritative table of scores to beat.
- **Samsung R&D Institute Poland (2025).** SemEval 2025 Task 8: LLM Ensemble Methods for QA Over Tabular Data. Blog post. https://research.samsung.com/blog/SemEval-2025-Task-8-LLM-Ensemble-Methods-for-QA-Over-Tabular-Data [S]
  - Ensemble of code-generating models; 86.21% and 2nd of 100 teams. The closest published description of multi-program voting on this benchmark.
- **Evkarpidi, Tutubalina (2025).** Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA. https://arxiv.org/abs/2506.09657 [S][A]
  - Text-to-SQL and text-to-code modules, self-correction and retrieval with open models; 80% accuracy.
- **Site, Erdemir, Eryiğit (2025).** ITUNLP at SemEval-2025 Task 8: A Zero-Shot Approach using LLM-Driven Code Generation. https://arxiv.org/abs/2508.00762 [S][A]
  - Zero-shot pandas generation with retry on execution error, open models only.
- **Tyagi, Gupta, Bouri (2025).** Agentic LLMs for Question Answering over Tabular Data. https://arxiv.org/abs/2509.09234 [S]
  - SQL route with verification and refinement reaches about 70.5%; shows SQL-only pipelines trail pandas pipelines on this benchmark.
- **Zhang et al. (2023).** ReAcTable: Enhancing ReAct for Table Question Answering. arXiv:2310.00815. https://arxiv.org/abs/2310.00815 [A]
  - A ReAct-style loop for table QA with no fine-tuning; 68.0% on WikiTQ.

## Datasets and benchmarks
- **Osés Grijalba et al. (2024).** Question Answering over Tabular Data with DataBench: A Large-Scale Empirical Evaluation of LLMs. LREC-COLING 2024. https://aclanthology.org/2024.lrec-main.1179 [S]
  - The benchmark: 65 real datasets, 1,308 questions, typed answers (boolean, category, number, lists).
- **DataBench on Hugging Face** (qa and semeval configurations). https://huggingface.co/datasets/cardiffnlp/databench [S]
- **Wu et al. (2024).** TableBench: A Comprehensive and Complex Benchmark for Table Question Answering. arXiv:2408.09174. https://arxiv.org/abs/2408.09174 [A]
  - A second benchmark to check that harness changes are not DataBench-specific.

## Open-source code and tools
- **Binder.** https://github.com/HKUNLP/Binder [A]
- **Program-of-Thoughts.** https://github.com/wenhuchen/Program-of-Thoughts [A]
- **DSPy** (pipeline and prompt optimisation against a metric). https://github.com/stanfordnlp/dspy [A]

## Breakthroughs in related fields that could transfer
- **Agrawal et al. (2025).** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. ICLR 2026. https://arxiv.org/abs/2507.19457 [A]
  - Reflective prompt evolution with few rollouts; can tune the profiling and ranking prompts that the HFM track edits by hand.

## Published results to beat
- SemEval-2025 Task 8 (DataBench test): TableReasoner placed first in both subtasks (exact score not seen in this session) [A]; Samsung R&D Poland 86.21%, 2nd of 100 teams [S]; an NTUA team reports 85.63% and 87.93% on the two subtasks, first among proprietary-model entries (search summary of https://preview.aclanthology.org/vol-to-evt/2025.semeval-1.188.pdf) [S]; ITUNLP and OseiBrefo-Liang report about 84.67% to 86.02% [S]. HFM's 0.877 is at the top of this range.
- TableReasoner with Qwen2.5-7B beats the top entry of the small-model (under 8B) ranking, per its paper [A].

## Free ways to run
- DataBench is on Hugging Face and the tools are pandas on CPU. TableReasoner reports strong accuracy with Qwen2.5-7B, which fits a free GPU when quantised [A].

## Unverified leads
- UNVERIFIED: the first-place accuracy on DataBench was not seen in this session; read it from the task overview paper.
- UNVERIFIED: no published audit of DataBench label noise was found; the README's "low 0.9s" ceiling has no source yet.
- UNVERIFIED: free-tier API limits were not checked in this session.
