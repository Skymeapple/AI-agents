# References: Agentic RL Training Recipe Exploration with verl and agent-harness

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Schulman et al. (2017).** Proximal Policy Optimization Algorithms. arXiv:1707.06347. https://arxiv.org/abs/1707.06347 [A]
  - The clipped surrogate objective that GRPO and its variants inherit.
- **Shao et al. (2024).** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. https://arxiv.org/abs/2402.03300 [A]
  - The paper that introduces GRPO (group-relative advantages, no value network).
- **DeepSeek-AI (2025).** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. https://arxiv.org/abs/2501.12948 [A]
  - The reference large-scale use of GRPO with rule-based rewards.
- **Sheng et al. (2024).** HybridFlow: A Flexible and Efficient RLHF Framework. arXiv:2409.19256. https://arxiv.org/abs/2409.19256 [A]
  - The system paper behind verl; explains the single-controller plus multi-controller design, which matters when debugging why proposals fail to score.

## Recent advances and breakthroughs (2023-2026)
- **Gandhi, Garg, Goodman et al. (2026).** Endless Terminals: Scaling RL Environments for Terminal Agents. arXiv:2601.16443. https://arxiv.org/abs/2601.16443 [A]
  - The published version of HFM's best idea: a pipeline that generates 3,255 containerised terminal tasks with completion tests and a solvability filter, then trains with plain PPO and binary episode rewards. Held-out dev: Qwen2.5-7B 10.7% to 53.3%. Transfer to Terminal-Bench 2.0 is small: Qwen2.5-7B 2.2% to 3.4%, Qwen3-8B-openthinker-sft 1.1% to 6.7%.
- **Yu et al. (2025).** DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476. https://arxiv.org/abs/2503.14476 [A]
  - Four GRPO techniques (the name covers decoupled clipping and dynamic sampling), built on verl with code released; 50 points on AIME 2024 from Qwen2.5-32B. The default recipe to start from on this stack.
- **Liu et al. (2025).** Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. https://arxiv.org/abs/2503.20783 [A]
  - Finds a GRPO bias that inflates response length, especially for wrong outputs, and proposes Dr. GRPO. Long wrong trajectories are costly in sandboxed rollouts.
- **Cao et al. (2025).** SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent. arXiv:2511.16108. https://arxiv.org/abs/2511.16108 [A]
  - Asynchronous rollout dispatcher (1.55x speedup) that plugs into verl; pure RL takes Qwen3-32B from 24.4% to 39.4% on SWE-Bench Verified and transfers to Terminal-Bench.
- **Luo et al. (2025).** Agent Lightning: Train ANY AI Agents with Reinforcement Learning. arXiv:2508.03680. https://arxiv.org/abs/2508.03680 [A]
  - Decouples agent execution from training by turning traced model calls into transitions with credit assignment; the same design as this project's gateway that converts every model call into training data.
- **Pan et al. (2024).** Training Software Engineering Agents and Verifiers with SWE-Gym. ICML 2025. https://arxiv.org/abs/2412.21139 [A]
  - 2,438 executable task instances; fine-tuning on agent trajectories gives up to 19 points absolute, and a trained verifier adds more at inference.
- **Jain et al. (2025).** R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. arXiv:2504.07164. https://arxiv.org/abs/2504.07164 [A]
  - 8.7K tasks built from commits with generated tests and back-translated task text; 34.4% pass@1 on SWE-Bench Verified with a 32B model.
- **Yang et al. (2025).** SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798. https://arxiv.org/abs/2504.21798 [A]
  - Synthesises tasks that break existing tests in any Python repository: 50k instances from 128 repositories, 40.2% on SWE-bench Verified with a 32B model.
- **Wei et al. (2025).** SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution. NeurIPS 2025. https://arxiv.org/abs/2502.18449 [A]
  - RL with a cheap similarity reward instead of executed tests; an option when sandbox throughput limits training.

## Datasets and benchmarks
- **Merrill et al. (2026).** Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. arXiv:2601.11868. https://arxiv.org/abs/2601.11868 [S][A]
  - The held-out target; small models score around 15% and frontier agents below 65% at paper time.
- **Endless Terminals task set** (3,255 tasks). https://github.com/kanishkg/endless-terminals [A]
- **SWE-Gym, R2E-Gym, SWE-smith** (links in the papers above) [A]
  - Executable training environments with tests, in case generic terminal tasks do not transfer.

## Open-source code and tools
- **verl** (described in HybridFlow and used by DAPO and SkyRL-Agent) [A]; the repository URL was not returned in this session, see Unverified leads.
- **DAPO project page** (training code on verl, curated dataset): linked from https://arxiv.org/abs/2503.14476 [A]

## Breakthroughs in related fields that could transfer
- **Kimi Team (2025).** Kimi K2: Open Agentic Intelligence. arXiv:2507.20534. https://arxiv.org/abs/2507.20534 [A]
  - Describes large-scale agentic data synthesis followed by a joint RL stage with real and synthetic environments; the recipe family that the kimi-k3 target model comes from.
- **Yan et al. (2025).** Learning to Reason under Off-Policy Guidance (LUFFY). arXiv:2504.14945. https://arxiv.org/abs/2504.14945 [A]
  - Mixes demonstrations from a stronger model into GRPO rollouts; trains weak models where on-policy RL fails outright, which is the situation when held-out pass rates start near zero.
- **Yuan et al. (2023).** Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825. https://arxiv.org/abs/2308.01825 [A]
  - Rejection-sampling fine-tuning: train on the model's own verified successes. A stable fallback when the GRPO stack is unreliable.

## Published results to beat
- Terminal-Bench 2.0 after RL on small open models: 3.4% (Qwen2.5-7B) and 6.7% (Qwen3-8B-openthinker-sft), Endless Terminals abstract [A]. HFM's terminal_bench_2 starting point of 0.2069 for kimi-k3 is already far above these, so gains from the same recipe on a stronger model are not established.
- Open-weight agents trained with RL or trajectory fine-tuning on SWE-Bench Verified: 40.2% (SWE-smith), 39.4% (SkyRL-Agent), 34.4% (R2E-Gym), 32.0% with verifier (SWE-Gym) [A].

## Free ways to run
- Endless Terminals reports training Llama-3.2-3B and Qwen2.5-7B with a minimal loop [A]; these sizes fit a single free-tier GPU for smoke tests of the stack, though full RL runs need more.

## Unverified leads
- UNVERIFIED: verl repository at github.com/volcengine/verl (known name; the GitHub API was blocked and search did not return it).
- UNVERIFIED: e2b sandbox free-tier limits and the deepagents repository were not checked.
- UNVERIFIED: no public results for a "kimi-k3" model were found in this session.
