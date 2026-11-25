# Agentic Pretraining with a Frozen Corpus and Tokenizer

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work designs and pretrains a model architecture from scratch using a frozen corpus pool and tokenizer, with a compute budget of 4 GPUs over 24 hours. The evaluation compares the resulting model against a reference architecture on a frozen held-out suite.

- **Metric:** `neg_macro_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.6941
- **HFM effort:** 17 proposals, 15 scored, 0 accepted improvements, about $1,170 of compute
- **HFM's current best idea:** MoE(p368 轴)、8k 序列+RoPE 1M(p369 轴)、WSD 长退火(p371 轴)三者独立增益之和 ~0.031 bpb;把三轴一次性叠到 main 上,若接近可加则 macro bpb 一步降至 ~0.692 以下,若显著次可加,与本轮两个 pair 实验对照即可定位交互来源。

## Existing theory and methods

Mixture-of-experts, long context with RoPE, warmup-stable-decay schedules.

## Public data

Public pretraining corpora.

## What limits progress

A 24-hour, 4-GPU budget.

## Route to a step change

A fixed efficient recipe as the floor.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish a reference score.
