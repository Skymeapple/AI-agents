# Looped-Transformer Pretraining Under a Fixed Compute Budget

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
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

TinyStories (public, Hugging Face) for the toy test. HFM's corpus and evaluator are private.

## What limits progress

A GPU for anything near HFM's scale. Seed noise is as large as the gains agents accept.

## Route to a step change

Loop-index embeddings and a separate norm per loop.

## Experiment and results

Code: [`looped_tiny.py`](looped_tiny.py) in this folder. It ran on free GitHub Actions compute (four CPU cores, about an hour per model); logs are in [`research/compute/results/`](../../../compute/results/) under `looped-tiny-*`.

- **This is a toy-scale test.** HFM trains on 4 to 8 GPUs for days. These are byte-level models of 2 to 4 million parameters trained on 25 million bytes of the public TinyStories corpus. It can show whether an idea helps when everything else is equal. It cannot say what happens at HFM's scale.
- **Setup:** four models of the same width see the same bytes in the same order for 6,000 steps, two seeds each. Score is bits per byte on held-out text, lower is better.

  | Model | Parameters | Layers of compute | Bits per byte (two seeds) | Mean |
  | --- | --- | --- | --- | --- |
  | Standard, 4 layers | 1.89M | 4 | 1.138, 1.145 | 1.141 |
  | Looped: 1 layer, 2 shared layers applied 3 times, 1 layer | 1.89M | 8 | 1.098, 1.096 | 1.097 |
  | Looped + loop index (HFM agents' idea) | 1.90M | 8 | 1.096, 1.097 | 1.096 |
  | Standard, 8 layers | 3.67M | 8 | 1.069, 1.072 | **1.071** |

- **What this shows:**
  - **Looping buys real quality per parameter.** With the same parameters as the 4-layer model, the looped model closes 63% of the gap to the 8-layer model (1.141 → 1.097, against 1.071).
  - **At equal compute, separate layers still win.** The 8-layer model is 0.026 bits per byte better than the looped model for the same training time. Under a fixed compute budget, which is this track's rule, looping is a loss unless parameters are the binding limit.
  - **The loop index did nothing measurable:** 0.001 better, which is inside the seed-to-seed spread (0.001 to 0.008). This is the idea HFM's agents proposed for letting a shared block specialise by depth.
- **Reading:** at this scale the looped design trades compute for parameters at a poor rate, and small conditioning tricks do not change that. The seed spread also shows why HFM's accepted steps of around 0.1% need repeated runs to be trusted.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

- **At toy scale:** a looped model recovers about two-thirds of the benefit of doubling depth, at no parameter cost but full compute cost.
- **At HFM's scale:** unknown. Published work (see `REFERENCES.md`) reports looped models matching standard ones mainly on algorithmic tasks.

## Next steps

- Give the looped model the compute it saves in memory: a wider looped model at equal training time.
- Vary the number of loops at test time, which a standard model cannot do.
- Repeat on a GPU at 100 times the scale (needs a Kaggle token).
