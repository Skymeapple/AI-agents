# TOPv2 compositional semantic parsing

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_topv2_fixed_auto` (paused on HFM)
- **Project status:** Blocked

## The challenge

Post-train Qwen2.5-7B-Instruct on TOPv2 compositional semantic parsing. Choose recipe, curriculum, hyperparameters, and efficiency tradeoffs within original task rules. Produce a LoRA adapter from this exact base checkpoint. Read frozen tests, task context, and templates first.

- **Metric:** `exact_match` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 83
- **HFM effort:** 17 proposals, 14 scored, 0 accepted improvements, about $252 of compute
- **HFM's current best idea:** 在保持1353的12288条训练行、清洁dev、单遍数据顺序和按样本进度计算的学习率曲线的条件下，将有效batch从16减至8，会比1353获得更高的耦合全树解析准确率。

## Existing theory and methods

Seq2seq parsing, grammar-constrained decoding.

## Public data

TOPv2 is public.

## What limits progress

Recipe tuning.

## Route to a step change

Grammar-constrained decoding and grammar-sampled synthetic trees.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Best is 83.0 exact match; full-data published results are in the high 80s.

## Next steps

Add constrained decoding.
