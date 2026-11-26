# Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding

- **Field:** post-training
- **HFM track:** `discovery_live_embspatial` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU study has run. Fine-tuning waits for a GPU.
- **Data access:** EmbSpatial-Bench and Qwen2-VL-2B: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work explores training recipes to improve Qwen2-VL-2B performance on EmbSpatial-Bench, which evaluates embodied egocentric spatial understanding through four-choice questions covering six spatial relation categories, under a frozen evaluation protocol.

- **Metric:** `embspatial_accuracy` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 0.785
- **HFM effort:** 80 proposals, 37 scored, 0 accepted improvements, about $245 of compute
- **HFM's current best idea:** 将 p117 与 p131 两个总体均为 0.785 且逐关系互补的 DoRA 适配器按预先固定的 0.5/0.5 参数平均，会使 embspatial_accuracy 高于 0.785。

## Existing theory and methods

LoRA and DoRA fine-tuning, adapter averaging, spatial-relation data synthesis.

## Public data

EmbSpatial-Bench and Qwen2-VL-2B are public.

## What limits progress

Nearest/farthest questions, which need depth. The four left/right/above/under relations are 2D layout and answerable from object boxes at 98–100%.

## Route to a step change

Supply depth for the nearest/farthest questions; concentrate training there.

## Experiment and results

Code: [`embspatial_no_image.py`](embspatial_no_image.py) in this folder. It runs in under a minute on a CPU. Input: the benchmark's text fields and object boxes, fetched without images by free compute into `research/compute/results/embspatial-records/`.

- **What was tested:** HFM's training needs a GPU. The public benchmark records (3,640 questions) carry the answer options and the names and bounding boxes of the objects each question asks about. Two questions can be answered from those alone: how much is answerable with no image at all, and how far a perfect 2D object detector plus plain geometry would get.
- **Result, accuracy in percent:**

  | Method | All | Left | Right | Above | Under | Nearest | Farthest |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Chance | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 |
  | Blind rule: pick any directional option | 38.7 | 44.9 | 45.4 | 45.7 | 46.1 | 25.0 | 25.0 |
  | Blind, learned from options and object names | 59.5 | 59.9 | 54.4 | 79.9 | 78.2 | 38.6 | 46.6 |
  | Boxes, rules only (lower in the image = nearer) | 88.5 | 98.7 | 98.2 | 99.8 | 99.5 | 69.6 | 64.8 |
  | Boxes, rules for pairs, learned for nearest/farthest | 89.1 | 98.7 | 98.2 | 99.8 | 99.5 | 69.4 | 68.7 |
  | HFM's best trained model (its own evaluation) | 78.5 | | | | | | |

  Learned rows are cross-validated with whole scenes held out. The best box method's 95% interval is 88.2 to 90.1.

- **What this shows:**
  - **The distractors give answers away.** In the 2,434 two-object questions, options such as "touching", "inside", "blocking", "connected" and "outside" are offered 3,855 times and are never correct. Discarding them lifts a blind guess from 25% to about 45% on those questions.
  - **Object names carry the vertical relations.** A model that sees no image reaches about 79% on above/under from names alone (pictures are above rugs).
  - **Four of the six relations are 2D layout.** Comparing box centres answers left, right, above and under at 98–100%. Only 23 of 2,434 are wrong.
  - **Nearest/farthest is the real task.** The best box rule reaches 65–70%, and a learned combination of box features does not improve on it meaningfully. These 1,206 questions need depth, which boxes do not contain.
  - **The arithmetic for HFM's 78.5%:** a model perfect on the four 2D relations and at chance on nearest/farthest would score about 75%. HFM's score is consistent with a model that has learned the 2D relations and little depth. Its per-relation scores are not published, so this is an inference.
- **Limits:** the boxes are the benchmark's own annotations, which a model under HFM's rules does not receive. The box result is a ceiling for "detector plus rules", not a submission. The blind learned model is fitted on benchmark questions from other scenes, which a real recipe may not be allowed to do.

### Follow-up: does a depth estimate fix nearest/farthest?

Code: [`embspatial_depth.py`](embspatial_depth.py); log in [`research/compute/results/embspatial-depth/`](../../../compute/results/embspatial-depth/run.log). A small off-the-shelf monocular depth model (Depth Anything V2 Small, not trained on this benchmark) ran on the 1,206 nearest/farthest images on CPU, 0.8 seconds each. The nearest object is the one with the highest estimated nearness inside its box.

| Rule for nearest/farthest | All | ai2thor | mp3d | scannet |
| --- | --- | --- | --- | --- |
| Box bottom edge, no depth model | 67.2 | 65.7 | 64.3 | 71.8 |
| Estimated depth, median inside the box | 74.8 | 60.6 | 88.4 | 76.0 |

- **Depth helps, by 7.6 points overall,** and takes the whole-benchmark figure for "boxes plus rules" from about 89% to 91.0%.
- **The gain is uneven.** On real scanned homes (mp3d) it reaches 88%. On the simulated scenes (ai2thor) it is worse than the box rule, 60.6 against 65.7.
- **So depth is part of the missing input, not all of it.** A quarter of these questions are still wrong. Likely causes are boxes that include background behind the object, and a depth model weak on rendered images; neither was tested.
- **Limit:** the choice of "median inside the box" was made on these same questions; the other variants tested range from 62.9 to 74.5.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **Detector plus 2D rules:** about 89% if detection were perfect, limited by nearest/farthest at about 69%.
- **With depth:** the remaining 11 points are almost all nearest/farthest. A monocular depth estimate at each box is the missing input.

## Next steps

- With a GPU: fine-tune with the training effort concentrated on nearest/farthest, and add depth supervision (a depth map channel or depth-ordered object lists as auxiliary targets).
- Mask the object inside its box before reading depth, and choose per source between the box rule and depth.
- Ask HFM for per-relation scores to confirm where its 78.5% comes from.
