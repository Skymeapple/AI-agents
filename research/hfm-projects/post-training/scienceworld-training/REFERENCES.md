# References: ScienceWorld task training on Qwen2.5-7B-Instruct

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Ross, Gordon, Bagnell (2010).** A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686 [A]
  - Train on states the learner itself visits, labelled by the expert; the fix for compounding errors in behaviour cloning and the README's proposed route.
- **Wang et al. (2022).** ScienceWorld: Is your Agent Smarter than a 5th Grader? EMNLP 2022. https://arxiv.org/abs/2203.07540 [A]
  - The environment paper; shows a small interactively trained agent beating an 11B model trained on static expert demonstrations, which is a warning about pure behaviour cloning.
- **Wang et al. (2022).** Behavior Cloned Transformers are Neurosymbolic Reasoners. EACL 2023. https://arxiv.org/abs/2210.07382 [A]
  - Injecting valid symbolic actions into the action space of a behaviour-cloned agent raised text-game scores by 22% on average; supports choosing among the simulator's valid actions.
- **Côté et al. (2018).** TextWorld: A Learning Environment for Text-based Games. arXiv:1806.11532. https://arxiv.org/abs/1806.11532 [A]
  - Background on text-game RL formalism and generated game variations.

## Recent advances and breakthroughs (2023-2026)
- **Song et al. (2024).** Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents (ETO). ACL 2024. https://arxiv.org/abs/2403.02502 [A]
  - After behaviour cloning, let the agent explore, pair each failed trajectory with the expert one, and train with a contrastive (DPO-style) loss. On ScienceWorld it gives 22% over SFT on the unseen test set. Uses 1,483 training trajectories; SFT at batch 64 and learning rate 1e-5.
- **Xiong et al. (2024).** Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement (IPR). EMNLP 2024. https://arxiv.org/abs/2406.11176 [A]
  - Step-level rewards by Monte Carlo rollouts from each expert step; where the agent's action is worse than the expert's, the pair becomes training data. A DAgger-like relabelling that needs only the simulator.
- **Qiao et al. (2024).** Agent Planning with World Knowledge Model (WKM). NeurIPS 2024. https://arxiv.org/abs/2405.14205 [A]
  - A second LoRA on the same backbone supplies task knowledge before acting and state knowledge during acting; tested on ScienceWorld with 7B to 8B open models. Code: https://github.com/zjunlp/WKM
- **Lin et al. (2025).** QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search. arXiv:2502.02584. https://arxiv.org/abs/2502.02584 [A]
  - Learns stepwise Q-values from exploration trees and uses them to pick actions at inference; keeps performance with about half the annotated data.
- **Lin et al. (2023).** SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks. NeurIPS 2023. https://arxiv.org/abs/2305.17390 [A]
  - A small model fine-tuned on oracle trajectories handles routine steps and a large model plans; the fine-tuned half is this project's setting.
- **Majumder et al. (2023).** CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization. arXiv:2310.10134. https://arxiv.org/abs/2310.10134 [A]
  - A frozen model with a continually updated memory beats Reflexion by 23 points on ScienceWorld; its memory text is a candidate for prompt content or for distillation into training data.
- **Xi et al. (2024).** AgentGym: Evolving Large Language Model-based Agents across Diverse Environments. arXiv:2406.04151. https://arxiv.org/abs/2406.04151 [A]
  - Platform with trajectories and a self-evolution method across environments including SciWorld. Code and data: https://github.com/WooooDyy/AgentGym
- **Fu et al. (2025).** AgentRefine: Enhancing Agent Generalization through Refinement Tuning. ICLR 2025. https://arxiv.org/abs/2501.01702 [A]
  - Finds that agent-tuned models memorise observation-action pairs and repeat the same mistake; proposes refinement tuning, so the model learns to correct its actions from environment feedback.

## Datasets and benchmarks
- **ScienceWorld** (paper above) [A]; split sizes used by ETO: 1,483 train, 194 test-seen, 241 test-unseen trajectories, 14.4 turns on average [A].
- **Song et al. (2024).** AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories. Findings of EMNLP 2024. https://arxiv.org/abs/2410.07706 [A]
  - Large trajectory collection for multi-task warm-up.
- **Jansen et al. (2024).** DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents. NeurIPS 2024. https://arxiv.org/abs/2406.06769 [A]
  - Successor environment from the same group; a transfer test. https://www.github.com/allenai/discoveryworld

## Open-source code and tools
- **AgentGym.** https://github.com/WooooDyy/AgentGym [A]
- **WKM.** https://github.com/zjunlp/WKM [A]
- **Agent-FLAN** (Chen et al. 2024, https://arxiv.org/abs/2403.12881). https://github.com/InternLM/Agent-FLAN [A]
  - Data design for agent tuning, including negative samples against hallucinated actions.

## Breakthroughs in related fields that could transfer
- **Yuan et al. (2023).** Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825. https://arxiv.org/abs/2308.01825 [A]
  - Rejection-sampling fine-tuning; ETO's table includes an RFT row for agents (below).
- **Shinn et al. (2023).** Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. https://arxiv.org/abs/2303.11366 [A]
  - Inference-time retry with reflections; a no-training baseline for the same adapter.

## Published results to beat
Average reward on ScienceWorld, seen / unseen test sets. These use the ETO evaluation subset and a 0 to 100 scale; they are not HFM's official DEV score.
- ETO paper, Llama-2-7B-Chat [A]: no tuning 3.8 / 3.1; SFT 67.4 / 53.0; best-of-N 70.2 / 57.6; RFT 71.6 / 54.3; PPO 59.4 / 51.7; ETO 73.8 / 65.0. GPT-4 64.8 / 64.4; GPT-3.5-Turbo 16.5 / 13.0.
- WKM paper, Llama-3-8B with LoRA [A]: ReAct 24.76 / 22.66; ETO 57.90 / 52.33; WKM 60.12 / 54.75. Mistral-7B: ETO 58.17 / 51.85; WKM 62.12 / 53.62.
- HFM's 35.8 is well below the SFT rows above (53 to 67), which were trained on all 1,483 trajectories with no 140-minute limit; HFM used 320 action samples.

## Free ways to run
- ScienceWorld is a CPU simulator. The papers above train 7B to 8B models with LoRA [A]; on a small free GPU this would need 4-bit loading (QLoRA: https://arxiv.org/abs/2305.14314 [A]).

## Unverified leads
- UNVERIFIED: the ScienceWorld repository (believed to be github.com/allenai/ScienceWorld) and its built-in gold-path generator were not confirmed in this session.
- UNVERIFIED: a published Qwen2.5-7B-Instruct base score on ScienceWorld was not found.
- UNVERIFIED: the NAT and KnowAgent baselines listed in the WKM table were not looked up.
