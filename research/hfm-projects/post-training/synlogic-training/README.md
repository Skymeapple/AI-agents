# SynLogic logical reasoning training

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_synlogic_auto` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** SynLogic and Qwen2.5-7B: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops a reproducible SFT warm-start to GRPO training recipe for the Qwen2.5-7B base model on SynLogic. The experiment uses one GPU, at most 120 minutes, and trains only a LoRA adapter from the supplied base model.

- **Metric:** `synlogic_easy_accuracy` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.3778
- **HFM effort:** 56 proposals, 49 scored, 0 accepted improvements, about $1,305 of compute
- **HFM's current best idea:** 保持p2162的8023目标、502次SFT更新及64次有效GRPO更新，将七投影LoRA从r16/alpha32改为r32/alpha64以保持alpha/r不变，会使最终适配器的冻结Easy-135聚合准确率高于p2162。

## Existing theory and methods

Supervised fine-tuning, GRPO, rejection-sampling fine-tuning, curriculum by difficulty.

## Public data

SynLogic and Qwen2.5-7B are public.

## What limits progress

Hyper-parameter search under a 120-minute, one-GPU budget; no published baseline.

## Route to a step change

Generate verified training data with the task's own checkers before RL.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish the base-model score.
