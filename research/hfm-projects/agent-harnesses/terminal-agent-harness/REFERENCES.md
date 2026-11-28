# References: Agent Harness Evolution Under a Frozen API Model

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the search budget ran out part-way). Numbers are quoted from the named source only.

## Theory and established methods
- **Yao et al. (2022).** ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. https://arxiv.org/abs/2210.03629 [A]
  - The interleaved reason-and-act loop that terminal agents build on.
- **Shinn et al. (2023).** Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. https://arxiv.org/abs/2303.11366 [A]
  - Retry with a written reflection on the failed attempt; the base form of verifier-first loops.
- **Yang et al. (2024).** SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793. https://arxiv.org/abs/2405.15793 [A]
  - Shows that the tool interface (commands, output truncation, guardrails) changes results for a fixed model; the core argument for harness work.
- **Wang et al. (2024).** Executable Code Actions Elicit Better LLM Agents (CodeAct). ICML 2024. https://arxiv.org/abs/2402.01030 [A]
  - Code as the action format instead of JSON tool calls.

## Recent advances and breakthroughs (2023-2026)
- **Merrill et al. (2026).** Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. arXiv:2601.11868. https://arxiv.org/abs/2601.11868 [S][A]
  - Defines Terminal-Bench 2.0 (89 tasks, each with its own container, reference solution and tests), reports frontier agents below 65% and small models around 15%, and gives a failure taxonomy. Read the error analysis before changing the harness.
- **Hu, Lu, Clune (2024).** Automated Design of Agentic Systems. arXiv:2408.08435. https://arxiv.org/abs/2408.08435 [A]
  - A meta-agent writes new agent designs in code and keeps an archive; the method class this HFM track belongs to.
- **Zhang et al. (2025).** Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954. https://arxiv.org/abs/2505.22954 [A]
  - A coding agent edits its own harness under a frozen model: SWE-bench 20.0% to 50.0%, Polyglot 14.2% to 30.7%. The discovered changes (better edit tools, context-window management, peer review) are a checklist for this project.
- **Zhang et al. (2024).** AFlow: Automating Agentic Workflow Generation. arXiv:2410.10762. https://arxiv.org/abs/2410.10762 [A]
  - Tree search over code-represented workflows; an alternative search procedure to greedy proposal acceptance.
- **Agrawal et al. (2025).** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. ICLR 2026. https://arxiv.org/abs/2507.19457 [A]
  - Reflective prompt evolution with up to 35x fewer rollouts than RL; relevant when each scored proposal costs about $20.
- **Paul (2026).** A Control System, a Dataset, and a Recipe for Making Frozen LLM Agents Learn a Domain. arXiv:2607.25415. https://arxiv.org/abs/2607.25415 [S][A]
  - A harness-level control loop for frozen models with released trajectory and reward logs.

## Datasets and benchmarks
- **Terminal-Bench 2.0** dataset, harness and leaderboard. https://tbench.ai [A]
- **Mialon et al. (2023).** GAIA: a benchmark for General AI Assistants. arXiv:2311.12983. https://arxiv.org/abs/2311.12983 [A]
  - One of the four sealed members; needs web browsing and file handling, so tool additions that help the terminal tasks must not break these.
- **Wei et al. (2024).** Measuring short-form factuality in large language models (SimpleQA). arXiv:2411.04368. https://arxiv.org/abs/2411.04368 [A]
  - Sealed member; rewards abstaining when unsure, so a verification step before answering matters.
- **Phan et al. (2025).** Humanity's Last Exam. arXiv:2501.14249. https://arxiv.org/abs/2501.14249 [A]
  - Source of the hle_search member; 2,500 expert questions.
- **Jimenez et al. (2023).** SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. https://arxiv.org/abs/2310.06770 [A]
  - The neighbouring benchmark where most harness ablations are published.

## Open-source code and tools
- **Wang et al. (2024).** OpenHands: An Open Platform for AI Software Developers as Generalist Agents. ICLR 2025. https://arxiv.org/abs/2407.16741 [A]
  - Open agent platform with bash, editor and browser tools; a reference for context condensation and tool design.
- **GEPA code.** https://github.com/gepa-ai/gepa [A]
- **Endless Terminals** (procedural terminal tasks with tests, usable as a practice set). https://github.com/kanishkg/endless-terminals [A]

## Breakthroughs in related fields that could transfer
- **Requena et al. (2026).** A Minimal Agent for Automated Theorem Proving. ICML 2026. https://arxiv.org/abs/2602.24273 [A]
  - With a verifier in the loop, iterating on one attempt beat many independent attempts on cost; the same trade-off as retries under a per-task timeout.
- **Kimi Team (2025).** Kimi K2: Open Agentic Intelligence. arXiv:2507.20534. https://arxiv.org/abs/2507.20534 [A]
  - Technical report for the K2 model family that the default frozen model belongs to; describes its agentic data synthesis pipeline and RL stage, which helps in guessing what tool-use patterns the frozen model was trained on.

## Published results to beat
- Terminal-Bench 2.0: below 65% for all agents at paper time (Merrill et al.) [A]. Third-party snapshots of the live leaderboard list Codex on GPT-5.5 at 82.0% (https://www.codesota.com/benchmark/terminal-bench-2) and GPT-5.3 Codex at 0.773, self-reported (https://llm-stats.com/benchmarks/terminal-bench-2) [S]. Check https://tbench.ai for current verified entries.
- The same sources stress that scores depend on harness as much as model [S]; HFM's 0.5362 is a macro-average over four benchmarks and is not comparable to a Terminal-Bench-only score.

## Free ways to run
- Terminal-Bench tasks run in local Docker containers. Endless Terminals reports results for Qwen2.5-7B and Qwen3-8B class open models on Terminal-Bench 2.0 (2% to 7%), so small open models can exercise the harness for free but will not give useful signal on pass rate [A].

## Unverified leads
- UNVERIFIED: the Terminus-2 reference agent and the Harbor evaluation framework are named in papers read here, but their repositories were not confirmed.
- UNVERIFIED: published work on wall-clock-aware ("anytime") agents for Terminal-Bench was not found; the HFM timeout finding appears to have no public counterpart yet.
- UNVERIFIED: run-to-run variance figures for Terminal-Bench 2.0 (the paper shows 95% confidence intervals in Figure 1) should be read from the paper appendix.
- UNVERIFIED: free-tier API limits for Kimi models were not checked in this session.
