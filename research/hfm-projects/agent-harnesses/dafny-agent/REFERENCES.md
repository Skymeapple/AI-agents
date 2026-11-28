# References: Agent for Verified Programming with Dafny

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Wu, Barrett, Narodytska (2023).** Lemur: Integrating Large Language Models in Automated Program Verification. ICLR 2024. https://arxiv.org/abs/2310.04870 [A]
  - A sound calculus for "LLM proposes invariants, solver checks them"; the formal basis for proposing and verifying invariants separately and then composing.
- **Kamath et al. (2023).** Finding Inductive Loop Invariants using Large Language Models. arXiv:2311.07948. https://arxiv.org/abs/2311.07948 [A]
  - Prompt design and symbolic checking for loop invariants, the obligation type the README says remains.
- **First et al. (2023).** Baldur: Whole-Proof Generation and Repair with Large Language Models. arXiv:2303.04910. https://arxiv.org/abs/2303.04910 [A]
  - The original generate-then-repair-with-error-message loop for formal proofs.
- **Sun et al. (2023).** Clover: Closed-Loop Verifiable Code Generation. arXiv:2310.17807. https://arxiv.org/abs/2310.17807 [S][A]
  - Checks consistency between code, docstring and Dafny annotations; relevant when the task supplies both a description and a specification.
- **Misu, Lopes, Ma, Noble (2024).** Towards AI-Assisted Synthesis of Verified Dafny Methods. FSE 2024. https://arxiv.org/abs/2402.00247 [S]
  - Same task shape as this project (method from specification and description). Decomposition plus retrieved examples raised success from 19% to 58% on MBPP problems.

## Recent advances and breakthroughs (2023-2026)
- **Breen, Letson, Requena Pozo et al. (2026).** AxDafny: Agentic Verified Code Generation in Dafny. arXiv:2606.32007. https://arxiv.org/abs/2606.32007 [S][A]
  - Agentic verifier-feedback loop without a hand-built hint library: 92.7% on DafnyBench; the paper reports that feedback gains are concentrated in early iterations, which supports the HFM finding about stopping stalled repair sequences.
- **Banerjee, Bouissou, Zetzsche (2026).** DafnyPro: LLM-Assisted Automated Verification for Dafny Programs. Dafny workshop at POPL 2026. https://arxiv.org/abs/2601.05385 [S][A]
  - Three inference-time parts: a diff-checker that blocks edits to program logic, a pruner that removes unneeded invariants, and retrieval of problem-independent proof hints. 86% on DafnyBench with Claude Sonnet 3.5; hint retrieval alone is worth about 10 points.
- **Bursuc et al. (2025).** A benchmark for vericoding: formally verified program synthesis. arXiv:2509.22908. https://arxiv.org/abs/2509.22908 [S][A]
  - 3,029 Dafny specifications to implement; off-the-shelf models reach 82% in Dafny (44% Verus, 27% Lean). This is the closest public match to the HFM task and its 0.829 score.
- **Zhao et al. (2026).** AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms. ICML 2026. https://arxiv.org/abs/2602.09464 [S][A]
  - 77 classical algorithms with identical specifications in Dafny, Verus and Lean; best Dafny result 40.3%. A harder test for lemma-heavy proofs.
- **Mugnier et al. (2024).** Laurel: Unblocking Automated Verification with Large Language Models. OOPSLA 2025. https://arxiv.org/abs/2405.16792 [S][A]
  - Uses the verifier's error location to place an assertion placeholder and retrieves assertions from similar lemmas; generates 56.6% of required helper assertions on DafnyGym.
- **Poesia, Loughridge, Amin (2024).** dafny-annotator: AI-Assisted Verification of Dafny Programs. arXiv:2411.15143. https://arxiv.org/abs/2411.15143 [A]
  - Greedy search over where to insert annotations, plus synthetic training data; a fine-tuned 8B Llama reaches 50.6% on DafnyBench.
- **Brandfonbrener et al. (2024).** VerMCTS: Synthesizing Multi-Step Programs using a Verifier, a Large Language Model, and Tree Search. arXiv:2402.08147. https://arxiv.org/abs/2402.08147 [A]
  - Tree search where the verifier scores partial programs; more than 30 points absolute gain in pass@5000 tokens over repeated sampling.
- **Xu et al. (2025).** Local Success Does Not Compose: Benchmarking LLMs for Compositional Formal Verification (DafnyCOMP). arXiv:2509.23061. https://arxiv.org/abs/2509.23061 [S][A]
  - Shows that verified pieces often fail to compose (about 7% success per a search summary); a warning for the "verify lemmas separately, then compose" route.

## Datasets and benchmarks
- **Loughridge et al. (2024).** DafnyBench: A Benchmark for Formal Software Verification. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.08467 [S][A]
  - 782 programs with hints removed; baseline 68% for the best model and prompt.
- **MBPP-DFY-153 / dafny-synthesis.** https://github.com/mondego/dafny-synthesis [S]
  - 153 MBPP problems with verified Dafny solutions (50 human-written); small enough for fast harness iteration.
- **Shefer et al. (2025).** Can LLMs Enable Verification in Mainstream Programming? arXiv:2503.14183. https://arxiv.org/abs/2503.14183 [A]
  - HumanEval translated to Dafny, Nagini and Verus, with a study of which inputs matter.

## Open-source code and tools
- **vericoding-benchmark** (specifications and results). https://github.com/Beneficial-AI-Foundation/vericoding-benchmark [A]
- **llm-verified-with-monte-carlo-tree-search** (VerMCTS). https://github.com/namin/llm-verified-with-monte-carlo-tree-search [A]
- **algoveri** (aligned Dafny, Verus, Lean tasks). https://github.com/haoyuzhao123/algoveri [A]

## Breakthroughs in related fields that could transfer
- **Yang et al. (2024).** AutoVerus: Automated Proof Generation for Rust Code. OOPSLA 2025. https://arxiv.org/abs/2409.13082 [S][A]
  - Three agent phases (draft proof, refine with generic tips, debug by error type); over 90% of 150 tasks, most within 3 model calls. The error-type-specific repair agents map onto Dafny obligation classes.
- **Aggarwal, Parno, Welleck (2024).** AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement. arXiv:2412.06176. https://arxiv.org/abs/2412.06176 [S][A]
  - Tree search over refinements with verifier feedback, and a growing pool of verified examples used as few-shot exemplars with no fine-tuning.
- **Ospanov, Farnia, Yousefzadeh (2025).** APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning. arXiv:2505.05758. https://arxiv.org/abs/2505.05758 [A]
  - Agents analyse a failed proof, fix syntax errors, locate the mistaken parts and repair only those with compiler feedback; a Lean analogue of handling proof obligations one by one instead of regenerating the whole method.

## Published results to beat
- DafnyBench (hint filling): 68% original baseline (Loughridge et al.) [A]; 86% DafnyPro with Claude Sonnet 3.5 [A]; 89% Claude Opus 4.1 alone and 96% for the union of models (Bursuc et al.) [S]; 92.7% AxDafny with GPT-5.5 [A].
- Dafny vericoding (implement from specification, closest to this task): 82.2%, best single model Claude Opus 4.1 (Bursuc et al., Table 3 as summarised by search) [S]. The HFM score of 0.829 is at this level.
- Small open models: fine-tuned Qwen 7B and 14B reach 68% and 70% on DafnyBench (DafnyPro) [A]; fine-tuned Llama 8B reaches 50.6% (dafny-annotator) [A].

## Free ways to run
- Dafny runs locally on CPU. DafnyPro reports that fine-tuned 7B and 14B Qwen models keep 68% to 70% on DafnyBench, so a quantised 7B model on a free GPU is a workable frozen model for harness experiments [A].

## Unverified leads
- UNVERIFIED: an ICML 2026 listing titled "DafnyProver: Agentic Verified Code Generation in Dafny" appeared in search and seems to be the conference version of AxDafny; the link between the two was not confirmed.
- UNVERIFIED: whether the DafnyPro fine-tuned Qwen weights are publicly released.
- UNVERIFIED: free-tier API limits were not checked in this session.
