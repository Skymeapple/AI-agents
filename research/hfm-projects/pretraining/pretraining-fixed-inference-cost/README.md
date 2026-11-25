# Agentic Pretraining Under Fixed Inference-Cost Constraint

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain_v2` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work examines the design of a model architecture that is pretrained from scratch using a frozen corpus pool and tokenizer, with a budget of 4 GPUs for 24 hours. The architecture is evaluated against a reference model on a frozen held-out suite, with the requirement that it achieves superior performance at a lower or equal inference cost.

- **Metric:** `neg_cost_adjusted_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.667
- **HFM effort:** 10 proposals, 9 scored, 0 accepted improvements, about $640 of compute
- **HFM's current best idea:** p392 训练峰值显存仅 40.4GB/141GB、吞吐 120k tok/s;把 micro_batch 4→8~16(grad_accum 等比下调保持全局 262k tok/step)、加大 ce_chunk 或启用 fused CE、扩大 torch.compile 覆盖范围,可把训练吞吐提升 ≥20%,同墙钟多训 ≥1.4B token,在模型与推理代价路径零改动下把 raw macro bpb 再压低 ≥3σ(≈0.0085)。

## Existing theory and methods

Throughput engineering, small dense models, MoE.

## Public data

Public pretraining corpora.

## What limits progress

Throughput: the best idea is using idle GPU memory.

## Route to a step change

Maximise tokens seen per hour first.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish a reference score.
