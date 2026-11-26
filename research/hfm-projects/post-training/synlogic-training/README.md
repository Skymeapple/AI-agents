# SynLogic logical reasoning training

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_synlogic_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU stand-in study has run. Fine-tuning waits for a GPU.
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

Code: [`synlogic_small_model.py`](synlogic_small_model.py) in this folder. It ran on free GitHub Actions compute (six 4-core runners, about an hour each); logs and per-puzzle outputs are in `research/compute/results/synlogic-direct-*` and `synlogic-code-*`.

- **What was tested:** HFM's fine-tuning needs a GPU. This stand-in uses a 3B open model (Qwen2.5-3B, 4-bit, CPU) with no training on the public easy validation set: 270 puzzles, 10 for each of 27 task types. One route answers directly with the dataset's own prompt. The other writes a Python program and its printed output is the answer. Scoring uses SynLogic's own verifiers.
- **Result:**

  | Route | Correct of 270 | Notes |
  | --- | --- | --- |
  | Direct answer | 13 (4.8%) | 114 replies were cut off at the 600-token limit |
  | Write a program | 4 (1.5%) | 30 programs ran and printed an answer; 125 crashed; 113 printed nothing |
  | Program if it ran, else direct | 15 (5.6%) | |
  | Either route correct | 17 (6.3%) | |
  | HFM's fine-tuned 7B model (its own easy set of 135) | 37.8% | For scale; different model and subset |

- **What this shows:**
  - **Both routes are near the floor.** 20 of 27 task types score zero on both. The correct answers are almost all in three arithmetic types (custom operators, arithmetic targets, fill-in expressions).
  - **The code route fails at writing the program, not at the search.** Only 11% of programs ran and printed anything. Several that ran printed the answer wrapped in formatting the verifier rejects.
  - **So a 3B model without training cannot stand in for this track.** The question the test was built for (which task types are search problems) is not answered by it.
- **Limits:** one small model, one prompt per route, one program attempt with no repair loop, a 600-token reply limit, 10 puzzles per type.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Untrained small model:** about 5%. HFM's trained 7B model: 37.8%.
- **Unknown:** how many task types fall to hand-written solvers checked by the benchmark's verifiers. That decides whether verified training data can be generated without a model.

## Next steps

- Write classical solvers for the grid and arithmetic task types and check them with the verifiers on CPU. Solved types give unlimited verified training targets.
- With a GPU: supervised fine-tuning on solver-verified answers before GRPO, within HFM's 120-minute budget.
