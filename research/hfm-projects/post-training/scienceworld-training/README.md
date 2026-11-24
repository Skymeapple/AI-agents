# ScienceWorld task training on Qwen2.5-7B-Instruct

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_scienceworld_auto` (paused on HFM)
- **Project status:** Blocked

## The challenge

Post-train the Qwen2.5-7B-Instruct model to improve ScienceWorld performance using a single GPU and a LoRA adapter. The total experiment budget is 140 minutes, with 110 minutes for baseline measurement.

- **Metric:** `sciworld_score` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 35.8
- **HFM effort:** 21 proposals, 18 scored, 0 accepted improvements, about $511 of compute
- **HFM's current best idea:** 从同一p1358适配器在相同320条官方TRAIN动作样本上续训20步时，仅对命令正文token计算交叉熵能够比监督完整Action答案提高官方DEV平均交互任务得分。

## Existing theory and methods

Behaviour cloning on expert trajectories, DAgger, RL fine-tuning.

## Public data

ScienceWorld is public.

## What limits progress

A short budget and few trajectories.

## Route to a step change

Programmatic trajectory generation with DAgger-style relabelling.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Best is 35.8; published agents exceed 60 with larger budgets.

## Next steps

Publish the base-model score.
