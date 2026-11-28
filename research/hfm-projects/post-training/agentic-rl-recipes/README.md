# Agentic RL Training Recipe Exploration with verl and agent-harness

- **Field:** post-training
- **HFM track:** `discovery_live_exp_agentic_rl_k3` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. The real stack (verl, terminal sandboxes, a large model) waits for GPUs.
- **Data access:** verl and Terminal-Bench on GitHub: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work explores training recipes for agentic reinforcement learning using a working stack that combines the verl training framework with the agent-harness environment, a fork of deepagents. The stack trains a policy with verl, solves real terminal tasks in e2b sandboxes via agent-harness, and uses a gateway to convert every model call into training data. The focus is on debugging and refining the training setup for the kimi-k3 model.

- **Metric:** `heldout_pass_rate` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.367
- **HFM effort:** 4 proposals, 2 scored, 0 accepted improvements, about $120 of compute
- **HFM's current best idea:** 用程序化合成的 docker 化终端任务(每题自带可执行验证测试)替代通用 agentic 数据混合做 GRPO 训练,能把 terminal_bench_2 从 .2069 显著拉高,单项 ≥+13 点即可独立把四成员宏平均推过门槛。

## Existing theory and methods

GRPO, procedurally generated tasks with executable tests.

## Public data

verl and Terminal-Bench are public.

## What limits progress

Training-stack reliability; only 2 of 4 proposals scored.

## Route to a step change

Procedural terminal tasks with tests. Note that this is already published (Endless Terminals, arXiv 2601.16443): it lifted a held-out set from 10.7% to 53.3% for a 7B model but Terminal-Bench 2.0 only from 2.2% to 3.4%, so test transfer early.

## Experiment and results

A toy-scale study on CPU of the transfer question behind HFM's best idea: GRPO on procedurally generated tasks that carry executable tests. Script: [`grpo_tiny.py`](grpo_tiny.py); logs in `research/compute/results/grpo-tiny-*`. It cannot say what happens at HFM's scale and is not comparable to HFM's score.

- **Task:** a "shell" of eight string commands (reverse, sort, drop repeats, first half, drop first letter, rotate, delete a's, repeat). The model sees two input/output examples and writes a pipeline; the reward is 1 only if the pipeline also passes a third, hidden case. Only behaviour is checked.
- **Model:** a 3-layer transformer (width 128), first trained by supervised learning on 1–2-command pipelines, then 1,500 updates of 16 prompts × 8 samples.
- **Recipes:** GRPO on a narrow generator (2–3 commands from 4 of the 8); GRPO on a diverse generator (1–3 commands from all 8); the same with DAPO-style dynamic sampling (keep only prompts whose samples disagree; this draws three times as many prompts); rejection-sampling fine-tuning on the model's own passing samples (diverse tasks); no second stage.
- **Scores:** greedy pass rate on 400 fixed tasks each from the narrow generator, the diverse generator, and a "benchmark" of 3–4-command pipelines, longer than anything trained on. Changes are paired against each seed's own model before the second stage, three seeds.

| Recipe | Narrow tasks | Diverse tasks | Benchmark (longer) |
|---|---|---|---|
| GRPO, narrow generator | **+0.157** (0.32 → 0.48) | **−0.302** (0.42 → 0.12) | **−0.069** (0.12 → 0.05) |
| GRPO, diverse generator | +0.040 | −0.030 | +0.004 |
| GRPO, diverse, dynamic sampling | +0.105 | −0.001 | +0.004 |
| Rejection-sampling fine-tuning, diverse | +0.105 | −0.020 | +0.002 |

(mean change over three seeds; seed ranges are in the logs: for example narrow GRPO's benchmark change runs from −0.092 to −0.048)

- **Training on a narrow task family raises its own score and wrecks everything else.** Narrow GRPO adds 16 points in its own distribution, loses 30 points on the broader tasks the model could already do, and halves the benchmark score. This is the Endless Terminals pattern in sharper form: a held-out set drawn from the training generator is not evidence of transfer, and can hide a loss.
- **No recipe improves the longer benchmark tasks.** Every recipe trained on diverse tasks moves the benchmark by under one point. Reinforcement learning sharpens pipelines the model can already sample; it does not teach longer compositions that it rarely produces.
- **Most GRPO groups carry no signal.** Only 21–29% of diverse-task groups contain both a pass and a fail. Dynamic sampling raises that to 47–73% and turns the small loss on diverse tasks into no loss, at three times the generation cost. Rejection-sampling fine-tuning does as well as dynamic GRPO here at a third of that cost.
- **For HFM's idea:** synthetic terminal tasks with tests are worth having, but a recipe should be judged on a benchmark drawn from a different generator from day one, and the task generator should cover the target distribution (here: pipeline length), because reward on easier tasks did not carry over.

## How far this can go

Not estimable.

## Next steps

Stabilise the stack first.
