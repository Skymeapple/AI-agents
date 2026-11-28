# MapQA Map Visual Question Answering

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_mapqa_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU study has run. Fine-tuning waits for a GPU.
- **Data access:** MapQA repository and Qwen2-VL-2B: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

Code: [`mapqa_blind.py`](mapqa_blind.py) in this folder; it runs in a few minutes on a CPU. Input: the question and answer text of a public copy (nimapourjafar/mm_mapqa on Hugging Face, first file), fetched without images into `research/compute/results/mapqa-records/`. A second public copy (elliot-mllm/mapqa_cleaned) is access-restricted.

- **What was tested:** HFM's training needs a GPU. On a CPU we can measure the floor: what a model that never sees the map scores, from question wording and answer habits alone. 5,346 maps, 65,071 question-answer pairs, whole maps held out in each of five folds.
- **Result, exact-match accuracy in percent:**

  | Method (no map) | All | Yes/no (30%) | States (38%) | Numbers and ranges (32%) |
  | --- | --- | --- | --- | --- |
  | Always the most common answer | 16.6 | 55.7 | 0.0 | 0.0 |
  | Most common answer for the question type | 19.4 | 55.2 | 4.5 | 3.8 |
  | Most common answer for the exact question | 20.4 | 55.4 | 6.8 | 3.8 |
  | Learned from the question's words | 20.2 | 54.9 | 6.8 | 3.8 |
  | HFM's best trained model (its own evaluation) | 82.8 | | | |

- **What this shows:**
  - **MapQA cannot be answered from language.** A blind model gets about 20%, almost all of it from guessing yes/no questions, which are close to balanced (44% yes).
  - **State and value questions are at 4–7% blind.** The answer depends on the map's colours and legend, which differ on every map.
  - **So HFM's 0.83 reflects reading the map.** This is the opposite of EmbSpatial-Bench, where a blind model reached 59.5%.
  - **The largest single question type is "What is the value of a state?" (17% of pairs).** It needs the full chain: find the state, read its colour, match it to the legend.
- **Limits:** one of seven files of one public copy; HFM's question mix and scoring may differ. This is a floor, not a method.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Blind floor:** about 20%. The remaining 60 points to HFM's 82.8 are map reading.
- **Not measured:** how much a classical pipeline (state outlines, colour sampling, legend matching) would score. That needs the images and is feasible on the free runner.

## Next steps

- Build the classical pipeline on CPU: sample each state's colour at a fixed interior point, match it to the legend swatches, answer by lookup. It would show which question types are perception and which are reasoning.
- With a GPU: fine-tune with auxiliary targets that make the model output the state-to-value table before answering.
