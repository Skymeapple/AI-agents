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

### Follow-up: classical solvers checked by the benchmark's verifiers

Code: [`synlogic_solvers.py`](synlogic_solvers.py); log in [`research/compute/results/synlogic-solvers/`](../../../compute/results/synlogic-solvers/run.log). Thirteen hand-written search programs, one per task type, read each puzzle's inputs from its structured record (never the stored solution) and format an answer as the task asks. SynLogic's own verifiers judge them.

| Task type | Puzzles | Verifier accepts |
| --- | --- | --- |
| Sudoku, campsite, skyscraper, star placement, survo, calcudoko, numbrix, cryptarithm, fill-in expression, bracket completion, bracket error, arithmetic target | 120 | 120 |
| Futoshiki | 10 | 4 |
| **Total** | **130** | **124** |

- **Plain search solves 124 of 130 puzzles in these 13 types, which is 45.9% of the 270-puzzle easy validation set,** in under four seconds per puzzle on a CPU. The other 14 types were not attempted.
- **Six easy futoshiki puzzles have no solution as stated.** The solver and a separate brute-force count over every grid both find zero grids that satisfy the given numbers and inequalities. No model can score on them.
- **The arithmetic-target verifier accepts wrong answers.** Its source compares the computed result with itself. A deliberately wrong expression (the plain sum of the numbers) was accepted in 10 of 10 puzzles. A model trained by reinforcement on this reward can score on that task type without solving it.
- **Two verifiers reject an answer wrapped in `<answer>` tags** (bracket completion and bracket error), although the dataset's prompt asks for those tags. They pass only if the scoring pipeline strips the tags first.
- **What this means for HFM's track:**
  - Verified training answers for at least 12 task types can be produced without a model, in any quantity, from the benchmark's own generators and these solvers. That supports supervised fine-tuning before GRPO.
  - Part of the reward is noise or exploitable: 6 unsolvable puzzles in 270, and one type that accepts anything well-formed. Scores near 38% should be read with that in mind.
- **Limits:** the solvers read structured records, not the natural-language question (futoshiki's given numbers are the exception). This shows the puzzles are search problems; it does not show a model can learn to solve them. HFM's 37.8% is a model answering in text on its own 135-puzzle set, so the two numbers are not comparable.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Untrained small model:** about 5%. HFM's trained 7B model: 37.8%.
- **Search on 13 task types:** 45.9% of the easy validation set, verified. The ceiling for any method is below 100% because some puzzles are unsolvable as stated.

## Next steps

- Solvers for the remaining 14 types where the rules are mechanical (custom operators, word sorting, truth-tellers, schedules).
- With a GPU: supervised fine-tuning on solver-verified answers before GRPO, within HFM's 120-minute budget.
- Report the unsolvable futoshiki puzzles and the arithmetic-target verifier to the dataset's maintainers (needs the user's own account).
