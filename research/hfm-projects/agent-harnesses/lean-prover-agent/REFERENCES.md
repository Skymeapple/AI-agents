# References: Agent for Formal Theorem Proving in Lean

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Yang et al. (2023).** LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. NeurIPS 2023. https://arxiv.org/abs/2306.15626 [S][A]
  - Premise retrieval from Mathlib plus best-first tactic search (ReProver); the standard baseline for the "tactic search with lemma retrieval" route.
- **Jiang et al. (2022).** Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs. arXiv:2210.12283. https://arxiv.org/abs/2210.12283 [A]
  - Informal proof, then a formal sketch with gaps, then automation closes the gaps (20.9% to 39.3% in the original setting). The pattern behind most current frozen-model harnesses.
- **Wang et al. (2023).** LEGO-Prover: Neural Theorem Proving with Growing Libraries. arXiv:2310.00656. https://arxiv.org/abs/2310.00656 [A]
  - A lemma library that grows across theorems (over 20,000 skills) and is retrieved for later proofs; the ablation credits it with 47.1% to 50.4% on miniF2F-test. Direct evidence for the README's lemma-cache idea.
- **Thakur et al. (2023).** An In-Context Learning Agent for Formal Theorem-Proving (COPRA). arXiv:2310.04353. https://arxiv.org/abs/2310.04353 [A]
  - A frozen general model doing stateful tactic search with backtracking and failure memory; beats few-shot GPT-4 on miniF2F.
- **Xin et al. (2025).** BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving. arXiv:2502.03438. https://arxiv.org/abs/2502.03438 [A]
  - Plain best-first search scaled up reaches 72.95% on miniF2F-test; shows what tuning the search alone buys.

## Recent advances and breakthroughs (2023-2026)
- **Zhou et al. (2025).** Solving Formal Math Problems by Decomposition and Iterative Reflection (Delta Prover). arXiv:2507.15225. https://arxiv.org/abs/2507.15225 [A]
  - Frozen Gemini 2.5 Pro, no fine-tuning: reflective decomposition into sub-problems plus iterative repair reaches 95.9% on miniF2F-test, with better test-time scaling than best-of-N sampling. The most relevant result for this project.
- **Ospanov, Farnia, Yousefzadeh (2025).** APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning. arXiv:2505.05758. https://arxiv.org/abs/2505.05758 [A]
  - Compiler-guided repair: agents analyse each failed proof, fix syntax errors, locate the mistaken parts and repair only those. Lifts o3-mini and o4-mini from 3-7% to over 40% on miniF2F and cuts sample budgets from 25,600 to a few hundred.
- **Varambally et al. (2025).** Hilbert: Recursively Building Formal Proofs with Informal Reasoning. arXiv:2509.22819. https://arxiv.org/abs/2509.22819 [A]
  - Recursive decomposition combining an informal reasoner, a prover model, a verifier and a retriever; 462/660 (70.0%) on PutnamBench.
- **Requena et al. (2026).** A Minimal Agent for Automated Theorem Proving (AxProverBase). ICML 2026. https://arxiv.org/abs/2602.24273 [S][A]
  - Proposer, compiler check, reviewer and a memory; finds that iterating on one proof beats many single-shot generations in sample efficiency and cost. Bears on the HFM question of fresh samples versus repair requests.
- **Kung et al. (2026).** LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks. arXiv:2606.03303. https://arxiv.org/abs/2606.03303 [S][A]
  - General-purpose models go from below 10% one-shot to 70% on Lean-IMO-Bench with an agentic framework, above a specialised system at 48%.
- **Lin et al. (2025).** Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction. ICLR 2026. https://arxiv.org/abs/2508.03613 [S][A]
  - Open 8B and 32B provers; 32B gets 88.1% pass@32 on miniF2F and 90.4% with two rounds of compiler-feedback self-correction (about 2 points from self-correction).
- **Wang et al. (2025).** Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning. arXiv:2504.11354. https://arxiv.org/abs/2504.11354 [S][A]
  - 80.7% at pass@8192 and 68.85% at pass@32 on miniF2F; distilled 1.5B and 7B versions are open. A later release post reports 92.2%: https://huggingface.co/blog/AI-MO/kimina-prover [S]
- **Ren et al. (2025).** DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition. arXiv:2504.21801. https://arxiv.org/abs/2504.21801 [S][A]
  - A general model writes a sketch with sorry placeholders and a 7B prover closes each one; 88.9% on miniF2F-test for the 671B model.
- **Chen et al. (2025).** Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving. arXiv:2507.23726. https://arxiv.org/abs/2507.23726 [S][A]
  - Lemma-style proving (state lemmas first, reuse them) with compiler feedback; reports 99.6% on miniF2F. Its successor Seed-Prover 1.5 (https://arxiv.org/abs/2512.17260 [A]) keeps verified lemmas in memory and reports 88% of PutnamBench.

## Datasets and benchmarks
- **Zheng, Han, Polu (2021).** MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics. ICLR 2022. https://arxiv.org/abs/2109.00110 [S][A]
  - 488 statements (244 test). A Lean 4 port with fixes is at https://reservoir.lean-lang.org/@google-deepmind/minif2f [S]
- **Ospanov, Farnia, Yousefzadeh (2025).** miniF2F-Lean Revisited: Reviewing Limitations and Charting a Path Forward. arXiv:2511.03108. https://arxiv.org/abs/2511.03108 [S][A]
  - Finds misaligned formal and informal statements and releases miniF2F-v2 (https://github.com/roozbeh-yz/miniF2F_v2); some "unprovable" failures are benchmark errors.
- **Tsoukalas et al. (2024).** PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition. NeurIPS 2024. https://arxiv.org/abs/2407.11214 [A]
  - The next benchmark once miniF2F saturates; https://github.com/trishullab/PutnamBench

## Open-source code and tools
- **Aniva et al. (2024).** Pantograph: A Machine-to-Machine Interaction Interface for Lean 4. arXiv:2410.16429. https://arxiv.org/abs/2410.16429 [S][A]
  - Tactic-state API for search (https://github.com/lenianiva/PyPantograph), usable where LeanDojo lags behind current Lean versions.
- **Santos et al. (2025).** Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification. arXiv:2504.21230. https://arxiv.org/abs/2504.21230 [A]
  - Parallel proof checking with caching, 1.5 to 2 times faster than earlier tools; verification throughput is the limit once sampling is cheap.
- **Goedel-Prover-V2** models, code and data. https://github.com/Goedel-LM/Goedel-Prover-V2 [S][A]
- **DeepSeek-Prover-V2** (7B and 671B weights). https://github.com/deepseek-ai/DeepSeek-Prover-V2 [S]
- **Delta Prover project page.** https://github.com/ByteDance-Seed/lean4-agent [A]
- **Hilbert.** https://github.com/Rose-STL-Lab/ml-hilbert [A]
- **COPRA.** https://github.com/trishullab/copra [A]

## Breakthroughs in related fields that could transfer
- **Li et al. (2026).** Self-Modifying Lean Proof Agents with Verifier-Grounded Benchmark Coevolution. arXiv:2607.17352. https://arxiv.org/abs/2607.17352 [S][A]
  - An agent that rewrites its own harness: 12.7% for the seed to 45.1% held-out miniF2F solve rate. Same loop as HFM harness evolution, with published numbers.
- **Achim et al. (2025).** Aristotle: IMO-level Automated Theorem Proving. arXiv:2510.01346. https://arxiv.org/abs/2510.01346 [A]
  - Combines proof search with an informal system that proposes and formalises lemmas; supports lemma generation as the main lever.

## Published results to beat
- miniF2F-test, frozen general model with a harness: 95.9% (Delta Prover, Gemini 2.5 Pro) [A]; over 40% for o3-mini and o4-mini with APOLLO repair [A]. The HFM figure of 0.566 is far below the first of these.
- miniF2F-test, specialised provers: 99.6% Seed-Prover [S]; 92.2% Kimina-Prover release post [S]; 90.4% Goedel-Prover-V2 32B with self-correction at pass@32 [S]; 88.9% DeepSeek-Prover-V2 671B [S]; 84.9% for APOLLO with a sub-8B model and under one hundred samples [A]; 72.95% BFS-Prover [A].
- Mistral reports that Leanstral 1.5 "saturates" miniF2F and solves 587/672 PutnamBench problems (vendor claim): https://mistral.ai/news/leanstral-1-5/ [S]

## Free ways to run
- Lean 4 and Mathlib run locally on CPU.
- Open provers small enough for a free GPU: Kimina-Prover distilled 1.5B and 7B (paper) and 1.7B and 8B (release post) [S][A]; DeepSeek-Prover-V2-7B [S]; Goedel-Prover-V2 8B [S]; BFS-Prover-V1-7B at https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B [A].
- Leanstral 1.5 is described by its vendor as Apache-2.0 with 6B active parameters [S].

## Unverified leads
- UNVERIFIED: Ax-Prover (arXiv:2510.12787) is listed on arXiv as withdrawn; do not cite without checking its status.
- UNVERIFIED: a NeurIPS 2026 listing "Do Theorem Provers Need Theorem-Proving Agents? mini-Lean-agent" (https://neurips.cc/virtual/2026/168472) claims a bash-only agent is a strong baseline; the model names and numbers in the search summary could not be checked.
- UNVERIFIED: the leanprover-community/repl project and the lean-lsp-mcp server were named in search summaries but their URLs were not returned.
- UNVERIFIED: free-tier API limits were not checked in this session.
