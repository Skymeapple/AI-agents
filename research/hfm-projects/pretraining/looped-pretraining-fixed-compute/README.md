# Looped-Transformer Pretraining Under a Fixed Compute Budget

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work explores the design of parameter-shared (looped or recurrent-depth) transformer architectures under a frozen corpus, tokenizer, and training recipe, with a compute budget of 4×H200 GPUs for 24 hours. The investigation focuses on what parameters are shared across loops, the number of loops, what information is fed back at each loop, adaptive depth, and loop-aware forward passes during training. A model is pretrained from scratch and evaluated against a looped reference architecture (8 unique blocks repeated 3 times, effective depth 24) on held-out bits-per-byte adjusted for inference cost, with a penalty of 0.02 per doubling of forward cost relative to a dense reference. The evaluation considers configurations that satisfy the looped family constraints: at most 9.0e7 unique non-embedding parameters, at most 3.6e8 total parameters, at least 2 loops, and a causal, loop-sensitive design.

- **Metric:** `neg_cost_adjusted_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.7786
- **HFM effort:** 4 proposals, 3 scored, 0 accepted improvements, about $196 of compute
- **HFM's current best idea:** 给循环块加入圈序号 embedding 与每圈独立 RMSNorm(参数量级 ~1e5,前向开销≈0)能让共享块按深度特化,缓解参考模型 math bpb 从 1h 到 20h 完全停滞(0.993→0.994)的容量/角色冲突,从而在不改变成本项的情况下把 math 分片显著拉低,整体 cost-adjusted bpb 优于参考 ≥3σ。

## Existing theory and methods

Looped transformers, per-loop normalisation.

## Public data

Public pretraining corpora.

## What limits progress

Shared blocks cannot specialise by depth; math loss stalls.

## Route to a step change

Loop-index embeddings and a separate norm per loop.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

A CPU-scale toy replication is possible but would be weak evidence.
