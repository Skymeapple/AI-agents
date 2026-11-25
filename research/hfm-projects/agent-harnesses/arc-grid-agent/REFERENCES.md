# References: Agent for ARC Grid Transformation Tasks

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Chollet (2019).** On the Measure of Intelligence. arXiv:1911.01547. https://arxiv.org/abs/1911.01547 [A]
  - Defines ARC and the priors it assumes; the reference for what a "rule" is allowed to depend on.
- **Wang et al. (2023).** Hypothesis Search: Inductive Reasoning with Language Models. ICLR 2024. https://arxiv.org/abs/2309.05660 [A]
  - Natural-language hypotheses first, then Python programs checked on the training pairs; the template for a frozen-model program-and-repair harness (30% vs 17% direct prompting on a 100-task subset, per the abstract).
- **Hodel (2024).** Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation (RE-ARC). arXiv:2404.07353. https://arxiv.org/abs/2404.07353 [A]
  - Generators for many extra examples per ARC training task; useful for building few-shot libraries and for testing candidate programs.

## Recent advances and breakthroughs (2023-2026)
- **Chollet, Knoop, Kamradt, Landers (2024).** ARC Prize 2024: Technical Report. arXiv:2412.04604. https://arxiv.org/abs/2412.04604 [S]
  - Survey of what worked in 2024: LLM-guided program synthesis and test-time training took the private-set score from 33% to 55.5%.
- **Chollet, Knoop, Kamradt et al. (2026).** ARC Prize 2025: Technical Report. arXiv:2601.10904. https://arxiv.org/abs/2601.10904 [S][A]
  - Surveys the 2025 methods and names refinement loops (generate, check, revise) as the main driver; the closest published description of the harness pattern this project uses.
- **Akyürek et al. (2024).** The Surprising Effectiveness of Test-Time Training for Abstract Reasoning. arXiv:2411.07279. https://arxiv.org/abs/2411.07279 [S]
  - Per-task LoRA training on leave-one-out augmented examples: 53.0% on the public validation set with an 8B model, 61.9% in an ensemble with program synthesis. Not usable with a frozen API model, but usable with a local one.
- **Li et al. (2024).** Combining Induction and Transduction for Abstract Reasoning. ICLR 2025. https://arxiv.org/abs/2411.02272 [S]
  - Program induction and direct output prediction solve different task types; ensembling them is the argument for running both routes in one harness.
- **Franzen, Disselhoff, Hartmann (2025).** Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective. ICML 2025. https://arxiv.org/abs/2505.07859 [S]
  - The same model generates candidates and scores them across augmentations (product of experts): 71.6% on the public evaluation set at about 2 cents per task. The scoring step is a direct upgrade to plain majority voting over the eight symmetries.
- **Pourcel, Colas, Oudeyer (2025).** Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR). ICML 2025. https://arxiv.org/abs/2507.14172 [S]
  - Evolutionary sample-and-refine search whose failed attempts are relabelled as training data; 52% on the public test set with open models.
- **Moghe, Chin (2026).** Cost-Effective Agent Harnesses for Abstract Reasoning and Generalization on ARC-AGI-1. arXiv:2607.06764. https://arxiv.org/abs/2607.06764 [S]
  - Same setting as this project (frozen general model, no fine-tuning). Splits pattern discovery from program writing; an orchestrator with a "think" tool reaches 67.25% pass@2 at $0.62 per task with DeepSeek V3.2.
- **Berman (2024).** How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute. Blog post (archived copy). https://gwern.net/doc/www/params.com/ffadd930691cc27e734738d2c8d6068b05122a61.html [S]
  - Evolves Python transform functions over generations, feeding back grid diffs; the basis of the high-scoring public refinement harnesses.
- **Ho et al. (2025).** ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory. arXiv:2509.04439. https://arxiv.org/abs/2509.04439 [A]
  - Stores reusable abstract concepts across tasks; 7.5% relative gain over a no-memory baseline. Addresses the README's point that remaining tasks need abstractions the model lacks.
- **Jolicoeur-Martineau (2025).** Less is More: Recursive Reasoning with Tiny Networks (TRM). arXiv:2510.04871. https://arxiv.org/abs/2510.04871 [A]
  - A 7M-parameter network reaches 45% on ARC-AGI-1 and 8% on ARC-AGI-2; small enough to run locally as a second voter beside the frozen model.
- **Liao, Gu (2025).** ARC-AGI Without Pretraining (CompressARC). arXiv:2512.06104. https://arxiv.org/abs/2512.06104 [A]
  - Solves puzzles by compression at test time with no training set; another cheap local voter.

## Datasets and benchmarks
- **ARC Prize Foundation.** ARC-AGI-2 repository (1,000 public training tasks, 120 public evaluation tasks, JSON). https://github.com/arcprize/ARC-AGI-2 [S]
  - The harder public set; use it to see which task classes still fail after ARC-AGI-1 saturates.
- **Chollet et al. (2025).** ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. arXiv:2505.11831. https://arxiv.org/abs/2505.11831 [S]
  - Describes the benchmark and the human baseline.
- **Moskvichev, Odouard, Mitchell (2023).** The ConceptARC Benchmark. arXiv:2305.07141. https://arxiv.org/abs/2305.07141 [A]
  - Tasks grouped by concept; suited to the README's next step of locating which task classes remain.
- **LeGris et al. (2024).** H-ARC: A Robust Estimate of Human Performance on the ARC Benchmark. arXiv:2409.01374. https://arxiv.org/abs/2409.01374 [A]
  - Human submissions and action traces for every task; a difficulty ranking for budget allocation.

## Open-source code and tools
- **marc** (test-time training code for Akyürek et al.). https://github.com/ekinakyurek/marc [S]
- **SOAR** (evolutionary program synthesis). https://github.com/flowersteam/SOAR [S]
- **cost-effective-arc-harnesses** (Moghe and Chin's agent harnesses, reported by a search summary of the paper). https://github.com/kabirmoghe/cost-effective-arc-harnesses [S]
- **arc_memo** (concept memory). https://github.com/matt-seb-ho/arc_memo [A]
- **Product of Experts project page** (links model, training and inference code). https://da-fr.github.io/Product-of-Experts-ARC-Paper [S]

## Breakthroughs in related fields that could transfer
- **Butt et al. (2024).** CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay. ICML 2024. https://arxiv.org/abs/2402.04858 [A]
  - Hindsight relabelling: a wrong program is still a correct program for the output it produced. A source of extra in-context examples at no model cost.
- **Chen et al. (2025).** Enigmata: Scaling Logical Reasoning in LLMs with Synthetic Verifiable Puzzles. arXiv:2505.19914. https://arxiv.org/abs/2505.19914 [A]
  - RL on generated puzzles lifts a 32B open model to 32.8% on ARC-AGI-1, showing that puzzle training transfers to grids.

## Published results to beat
- ARC-AGI-1 public evaluation set: 71.6% (Franzen et al., open model, about 2 cents per task) [S]; 67.25% pass@2 at $0.62 per task with a frozen DeepSeek V3.2 (Moghe and Chin) [S]; 53.0% single model and 61.9% ensemble with test-time training (Akyürek et al.) [S]. The HFM figure of 0.786 is on HFM's own set, so these are not directly comparable.
- ARC-AGI-2: top verified refinement system 54% at $30 per task (Poetiq on Gemini 3 Pro) and top verified single model 37.6% at $2.20 per task (Opus 4.5 Thinking), both from the ARC Prize 2025 results analysis, https://arcprize.org/blog/arc-prize-2025-results-analysis [S]; Kaggle 2025 winner NVARC 24.0% under contest compute limits, https://arcprize.org/competitions/2025 [S].
- A search summary of a review paper attributes 79.6% on ARC-AGI-1 at about $8.42 per task to Berman's natural-language evolutionary harness with Grok-4; treat as secondary until checked against the ARC Prize leaderboard [S].

## Free ways to run
- Local voters need no API: TRM (7M parameters) and CompressARC run on a free GPU; the test-time-training recipe was shown on an 8B model [S][A].
- Moghe and Chin's numbers come from DeepSeek V3.2 at $0.25 to $0.62 per task, which is low cost but not free [S].

## Unverified leads
- UNVERIFIED: the original ARC-AGI-1 task repository is believed to be github.com/fchollet/ARC-AGI; not returned by search in this session.
- UNVERIFIED: Greenblatt's 2024 write-up of sampling thousands of Python programs per task with GPT-4o (the ARC Prize 2024 report credits him with 42% on ARC-AGI-Pub, but the write-up URL was not confirmed).
- UNVERIFIED: a social-media post claims 91% on the ARC-AGI-2 public set at $3.30 per task; no ARC Prize verification was found.
- UNVERIFIED: free-tier API limits (for example Gemini or Groq free tiers) were not checked in this session.
