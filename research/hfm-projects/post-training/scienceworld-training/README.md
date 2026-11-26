# ScienceWorld task training on Qwen2.5-7B-Instruct

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_scienceworld_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU stand-in study has run. Fine-tuning waits for a GPU.
- **Data access:** ScienceWorld on GitHub and Qwen2.5-7B: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

Code: [`scienceworld_small_model.py`](scienceworld_small_model.py) in this folder. It ran on free GitHub Actions compute; logs and per-task records are in `research/compute/results/scienceworld-*`.

- **What was tested:** HFM's fine-tuning needs a GPU. This stand-in measures three agents with no training, on one test variation of each of the 30 tasks, at most 30 steps, with the simulator's "easy" simplifications:
  - random: a uniformly random action from the simulator's valid-action list (mean of 5 runs);
  - free: a 3B open model (Qwen2.5-3B, 4-bit, CPU) writes a command, sent as written;
  - snap: the same model, with its command replaced by the most similar entry of the valid-action list.
- **Result, mean score of 100:**

  | Agent | Mean score | Tasks scoring above 0 | Commands found in the valid list |
  | --- | --- | --- | --- |
  | Gold action sequence (check) | 100.0 | 30 | 13–96% by task |
  | Random valid action | 5.9 | 23 | 100% |
  | Small model, free commands | 12.9 | 22 | 37% |
  | Small model, snapped to the valid list | 16.5 | 15 | 100% |
  | HFM's fine-tuned 7B model (its own evaluation) | 35.8 | | |

- **What this shows:**
  - **An untrained 3B model scores 13–17, two to three times random.** Most of it comes from five tasks (conductivity, power components, lifespan), where the model scores 43–77.
  - **Snapping to the valid list is not a reliable gain.** The 3.6-point difference comes from one task (100 against 8). Snap is better on 3 tasks, worse on 7 and tied on 20.
  - **The valid-action list is incomplete.** Only 13–96% of the gold sequence's own commands appear in it, so forcing commands into the list can block the correct one. This is unlike TOPv2, where the grammar was exact and constraint helped by 19 points.
  - **Tasks that need a long procedure score zero for every agent** (boil, freeze, melt, change of state): 30 steps of an untrained model do not reach the first milestone.
- **Limits:** one variation per task, one run of each model agent, 30 steps, one prompt. HFM's variations, step limit and simplifications are not published, so 35.8 is shown for scale only.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Untrained small model:** 13–17. **Gold sequences:** 100, and they are available for every training variation, so imitation data is unlimited.
- **Open question:** how much of the gap a model closes by imitating gold sequences before any reinforcement learning.

## Next steps

- With a GPU: behaviour cloning on gold sequences of the training variations, then RL fine-tuning, within HFM's 140-minute budget.
- On CPU: repeat with more steps and several variations per task to put intervals on the numbers; test a constraint built from the simulator's command templates instead of its valid-action list.
