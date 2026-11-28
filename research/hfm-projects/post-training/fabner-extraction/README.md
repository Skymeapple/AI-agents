# FabNER Material Manufacturing Entity Extraction

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_fabner_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU study runs today. Fine-tuning waits for a GPU.
- **Data access:** FabNER and Qwen2.5-7B: Verified reachable from the free compute on 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Post-train Qwen/Qwen2.5-7B-Instruct on FabNER entity extraction using a LoRA adapter from the same base checkpoint. The recipe, curriculum, hyperparameters, and efficiency tradeoffs are chosen within the original task rules.

- **Metric:** `fabner_entity_micro_f1` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 80.68
- **HFM effort:** 27 proposals, 27 scored, 0 accepted improvements, about $642 of compute
- **HFM's current best idea:** 从p1104的step-400重建p1457的r32初始适配器，固定抽样和300步续训，仅将实体索引及类型值token的普通CE改为均值归一化的动态置信度加权CE，会比p1457提高固定官方验证512例上的精确实体micro-F1。

## Existing theory and methods

Sequence labelling, LoRA fine-tuning, constrained decoding.

## Public data

FabNER (DFKI-SLT/fabner on Hugging Face; Kumar and Starly 2021), public.

## What limits progress

Unseen phrases (14% of test entities) and inconsistent labels (a quarter of repeated-phrase appearances).

## Route to a step change

A trained model that is given the dictionary's answer and decides the unseen phrases. An untrained small model adding entities to the dictionary lowers F1 from 83.2 to 75.7.

## Experiment and results

Code: [`fabner_ceiling.py`](fabner_ceiling.py) in this folder. It ran on free GitHub Actions compute in under three minutes; the log is in [`research/compute/results/fabner-ceiling/`](../../../compute/results/fabner-ceiling/run.log).

- **What was tested:** HFM's fine-tuning needs a GPU. Two things can be measured without one on the public dataset (9,435 training and 2,064 test sentences, 12 entity types): how far classical methods get, and where the errors are.
- **Result, entity-level micro F1 on the public test split:**

  | Method | F1 | Precision | Recall |
  | --- | --- | --- | --- |
  | Dictionary of training entities, longest match, most common type | 83.4 | 81.3 | 85.5 |
  | Conditional random field (classical, CPU, about a minute) | 82.8 | 84.7 | 81.0 |
  | HFM's best fine-tuned language model (HFM's own evaluation) | 80.7 | | |
  | Best published generative model found (GNER) | 85.4 | | |

- **What this shows:**
  - **A plain dictionary of training phrases scores 83.4 on the public split,** above HFM's 80.7 and above one of the two published language-model figures (UniNER 81.9). HFM's evaluator may differ in split or scoring, so this is not a claim to have beaten HFM. It does show that most of this benchmark is recall of seen phrases.
  - **86% of test entities are phrases already labelled in training.** The classical model finds 94–98% of those.
  - **The remaining 14% are phrases never seen as an entity, and the classical model finds 7% of them.** This is where almost all the headroom is, and it is the part a language model is suited to.
  - **Labels are inconsistent:** among phrases that appear five or more times, a quarter of appearances are labelled differently from the phrase's usual label, mostly left unlabelled or absorbed into a longer span. That caps what any model can score.
- **Reading:** fine-tuning a language model end to end spends its capacity relearning a dictionary. A hybrid, with dictionary or CRF predictions supplied to the language model as candidates, targets the 14% directly.

### Follow-up: does a small language model find the unseen phrases?

Code: [`fabner_hybrid.py`](fabner_hybrid.py); log in [`research/compute/results/fabner-hybrid/`](../../../compute/results/fabner-hybrid/run.log). A 3B open model (Qwen2.5-3B, 4-bit, CPU, no fine-tuning) was shown six retrieved training sentences with their entities and asked to list the entities of each of 300 test sentences (1,264 entities).

| System | F1 | Precision | Recall |
| --- | --- | --- | --- |
| Dictionary | 83.2 | 82.0 | 84.4 |
| Small model with 6 retrieved examples | 33.6 | 37.4 | 30.5 |
| Dictionary plus the model's non-overlapping entities | 75.7 | 67.8 | 85.8 |

- **The hybrid is worse than the dictionary by 7.5 points.** The model added 299 entities with phrases unseen in training; 17 were correct (6%).
- **The model does find some unseen entities:** 19.7% of the 188, against 0% for the dictionary and 7% for the CRF. After removing those that overlap dictionary matches, 9.0% remain.
- **Reading:** an untrained small model cannot tell which unseen phrases the annotators would have marked. Recall on the unseen part is available, but precision is not, and at 6% precision every correct addition costs sixteen wrong ones. The hybrid route needs a model trained on this annotation style, or a learned filter on the proposals. That is a GPU fine-tune, or the classical stacking test below.
- **Limits:** 300 sentences, one prompt, one small model. A fine-tuned 7B model is a different object.

### Follow-up: a classical model that is shown the dictionary's answer

Code: [`fabner_stack.py`](fabner_stack.py); log in [`research/compute/results/fabner-stack/`](../../../compute/results/fabner-stack/run.log). The CRF receives the dictionary's tag for each word as a feature. For training sentences the dictionary is built from the other nine tenths of the training set, so the CRF learns how far to trust it. Full test split, 2,064 sentences, 9,633 entities.

| System | F1 | Precision | Recall | Recall, seen phrases | Recall, unseen phrases |
| --- | --- | --- | --- | --- | --- |
| Dictionary | 83.4 | 81.3 | 85.5 | 99.1% | 0.0% |
| CRF | 82.8 | 84.7 | 81.0 | 92.8% | 7.2% |
| Union of the two | 83.3 | 81.0 | 85.7 | | |
| CRF given the dictionary's tags | **86.2** | 87.5 | 84.9 | 97.9% | 3.2% |

- **Stacking adds 2.8 points over the dictionary** (95% interval +2.5 to +3.2 over 1,000 resamples of test sentences; setting chosen on validation, where it scores 90.1).
- **The gain is precision, not unseen phrases.** The stacked model keeps the dictionary's recall on seen phrases and learns when a dictionary match is not an entity in context (precision 81.3 → 87.5). On unseen phrases it finds 3.2%, fewer than the plain CRF.
- **A simple union does nothing.** The two must be combined by a trained model.
- **Scale:** 86.2 is above the best published generative figure found (GNER, 85.4) and HFM's 80.7. Splits and scoring may differ, so this is not a claim to have beaten either. It runs in minutes on a CPU.
- **What remains:** the 1,325 unseen-phrase entities. Nothing tested here finds more than 20% of them, and the one method that did (the untrained small model) had 6% precision.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **On the public split:** 86.2 from a CRF given the dictionary's tags; 83.4 from the dictionary alone. Perfect recall of the unseen phrases would add up to about 12 points, less whatever label inconsistency removes.
- **Published best:** 85.4.

## Next steps

- The untrained hybrid failed (above). Next: a fine-tuned model that sees the dictionary's tags, on a GPU; the CPU version with a CRF reached 86.2 (above).
- Score HFM-style generative output with the same scorer to check comparability.
- Quantify the label-noise ceiling by re-annotating a sample.
