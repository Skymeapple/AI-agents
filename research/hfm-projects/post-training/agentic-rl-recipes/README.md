# Agentic RL Training Recipe Exploration with verl and agent-harness

- **Field:** post-training
- **HFM track:** `discovery_live_exp_agentic_rl_k3` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
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

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable.

## Next steps

Stabilise the stack first.
