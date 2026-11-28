# References: SynLogic logical reasoning training

> How each entry was confirmed in this session: **[S]** = returned by a web search; **[A]** = abstract or full-text page fetched directly from arxiv.org (the web-search budget ran out before this project was reached, so entries here are confirmed by direct arXiv fetch). Numbers are quoted from the named source only.

## Theory and established methods
- **Shao et al. (2024).** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. https://arxiv.org/abs/2402.03300 [A]
  - Introduces GRPO.
- **Yuan et al. (2023).** Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825. https://arxiv.org/abs/2308.01825 [A]
  - Rejection-sampling fine-tuning: more distinct correct reasoning paths give larger gains, and weaker models gain most (LLaMA-7B on GSM8K 35.9% SFT to 49.3%). The method for building the SFT warm-start set with the task checkers.
- **Zelikman et al. (2022).** STaR: Bootstrapping Reasoning With Reasoning. arXiv:2203.14465. https://arxiv.org/abs/2203.14465 [A]
  - Keep only rationales that led to correct answers, fine-tune, repeat; the iterative form of the same idea.

## Recent advances and breakthroughs (2023-2026)
- **Liu et al. (2025).** SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond. arXiv:2505.19641. https://arxiv.org/abs/2505.19641 [A]
  - The task paper: 35 tasks with generators and verifiers. Their 7B recipe is zero-RL with DAPO from Qwen2.5-7B-Base on about 16k SynLogic-Easy samples (prompt batch 128, 16 rollouts per prompt, clip-high 0.28, rollouts up to 16,384 tokens). Eight tasks were dropped from the Easy version because 7B models did not learn them.
- **Yu et al. (2025).** DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476. https://arxiv.org/abs/2503.14476 [A]
  - The GRPO variant the SynLogic authors used; its decoupled clipping and dynamic sampling are candidates to add to a 64-update GRPO phase.
- **Liu et al. (2025).** Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. https://arxiv.org/abs/2503.20783 [A]
  - Dr. GRPO removes a length bias in GRPO that inflates wrong answers; with a 120-minute budget, shorter rollouts mean more updates.
- **Xie et al. (2025).** Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning. arXiv:2502.14768. https://arxiv.org/abs/2502.14768 [A]
  - A 7B model trained with GRPO on only 5K logic puzzles, using a system prompt that separates thinking and answer and a strict format reward that penalises shortcuts; the closest small-scale recipe to this task.
- **Yan et al. (2025).** Learning to Reason under Off-Policy Guidance (LUFFY). arXiv:2504.14945. https://arxiv.org/abs/2504.14945 [A]
  - Mixes off-policy correct traces into GRPO; reported to train weak models where on-policy RL fails. An alternative to a separate SFT stage.
- **Chen et al. (2025).** Enigmata: Scaling Logical Reasoning in LLMs with Synthetic Verifiable Puzzles. arXiv:2505.19914. https://arxiv.org/abs/2505.19914 [A]
  - 36 puzzle tasks, each with a generator of controllable difficulty and a rule-based verifier; a second source of verified data and of difficulty curricula.
- **Stojanovski et al. (2025).** Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards. NeurIPS 2025. https://arxiv.org/abs/2505.24760 [A]
  - Procedural generators with adjustable difficulty; usable for extra warm-start data in task families that overlap SynLogic.

## Datasets and benchmarks
- **SynLogic dataset and synthesis pipeline.** https://github.com/MiniMax-AI/SynLogic [A]
  - Run the generators and checkers locally to make unlimited verified training items.
- **Ma et al. (2024).** KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks. arXiv:2410.06526. https://arxiv.org/abs/2410.06526 [A]
  - Out-of-distribution logic benchmark used in the SynLogic paper.
- **Kazemi et al. (2025).** BIG-Bench Extra Hard. arXiv:2502.19187. https://arxiv.org/abs/2502.19187 [A]
  - Second transfer benchmark used in the SynLogic paper; https://github.com/google-deepmind/bbeh
- **Lin et al. (2025).** ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning. ICML 2025. https://arxiv.org/abs/2502.01100 [A]
  - Shows accuracy falling with puzzle complexity; a guide for choosing difficulty levels in a curriculum.

## Open-source code and tools
- **SynLogic repository** (above) and the **DAPO** training code on verl, linked from https://arxiv.org/abs/2503.14476 [A]
- **Reasoning Gym code**, linked from https://arxiv.org/abs/2505.24760 [A]

## Breakthroughs in related fields that could transfer
- **Kalajdzievski (2023).** A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA). arXiv:2312.03732. https://arxiv.org/abs/2312.03732 [A]
  - Proposes a rank-stabilised scaling factor so that larger ranks trade compute for better fine-tuning. Directly relevant to HFM's current idea of moving from r16 to r32 at constant alpha/r.
- **Liu et al. (2024).** DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024. https://arxiv.org/abs/2402.09353 [A]
  - A drop-in LoRA variant reported to learn better at the same rank with no inference overhead.

## Published results to beat
- SynLogic paper table (zero-shot, avg@8, the paper's own validation split): Qwen2.5-7B-Base 2.8, Qwen2.5-7B-Instruct 9.0, SynLogic-7B (their RL-trained model) 44.4 on SynLogic-Val; KOR-Bench 11.6 / 38.6 / 48.1 for the same three models [A]. HFM's 0.3778 on its frozen Easy-135 set is below the paper's 44.4, which came from full-weight RL with a much larger budget; the two validation sets may differ.

## Free ways to run
- The SynLogic generators and checkers need only CPU. 4-bit quantised LoRA training (QLoRA: https://arxiv.org/abs/2305.14314 [A]) is the usual way to fit a 7B model on a small free GPU, though a 120-minute GRPO run there will see far fewer rollouts than the paper's.

## Unverified leads
- UNVERIFIED: whether SynLogic-Val in the paper for the 7B model is the same split as HFM's "Easy-135" (the paper says 10 validation samples per task for the Easy split).
- UNVERIFIED: GRPO trainers in Hugging Face TRL and Unsloth exist as free tooling; not confirmed in this session.
