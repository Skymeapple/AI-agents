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

Constrained JSON decoding and hard-example mining.

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

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **On the public split:** 83.4 from a dictionary. Perfect recall of the unseen phrases would add up to about 12 points, less whatever label inconsistency removes.
- **Published best:** 85.4.

## Next steps

- Build the hybrid: classical candidates in the prompt, language model decides unseen phrases and boundaries (needs a GPU or a language-model key).
- Score HFM-style generative output with the same scorer to check comparability.
- Quantify the label-noise ceiling by re-annotating a sample.
