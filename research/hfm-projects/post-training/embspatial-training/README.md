# Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial Understanding

- **Field:** post-training
- **HFM track:** `discovery_live_embspatial` (paused on HFM)
- **Project status:** Blocked

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

Small model and small recipe changes.

## Route to a step change

Synthesise spatial-relation training pairs from 3D scene datasets.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Best is 0.785; larger models reach higher, so the ceiling for a 2B model is unclear.

## Next steps

Publish the base-model score.
