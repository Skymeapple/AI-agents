# FabNER Material Manufacturing Entity Extraction

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_fabner_auto` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
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

FabNER and Qwen2.5-7B are public.

## What limits progress

Recipe tuning on a fixed dataset.

## Route to a step change

Constrained JSON decoding and hard-example mining.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Best is 80.7 F1. The highest published FabNER figure found in the literature pass is 85.4 (GNER), with UniNER-7B at 81.9, so confirmed headroom is about 5 points. An earlier note here said encoder models reach the high 80s; that was not supported by the tables found (BERT-base is reported at 64.2). See `REFERENCES.md`.

## Next steps

Try keeping non-entity instances in training with a longest-common-subsequence matcher (the GNER recipe behind the best published figure).
