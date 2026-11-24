# MapQA Map Visual Question Answering

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_mapqa_auto` (paused on HFM)
- **Project status:** Blocked

## The challenge

Post-train Qwen/Qwen2-VL-2B on the MapQA task using a chosen recipe, curriculum, hyperparameters, and efficiency tradeoffs within the original task rules. Produce a LoRA adapter from this exact base checkpoint. Read frozen evaluation and task context files first.

- **Metric:** `mean_jaccard_discrete` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.8283
- **HFM effort:** 24 proposals, 24 scored, 0 accepted improvements, about $564 of compute
- **HFM's current best idea:** 在p1839独立训练的B语言LoRA上保持字母序答案和每问答平均损失，以2e-5学习率单遍续训2048条父系未用的离散训练问答，可在固定dev1000上取得高于未更新B的平均Jaccard。

## Existing theory and methods

Vision-language LoRA, native-resolution inputs.

## Public data

MapQA and Qwen2-VL-2B are public.

## What limits progress

Recipe tuning.

## Route to a step change

Higher-resolution inputs and legend-reading pre-tasks.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish the base-model score.
