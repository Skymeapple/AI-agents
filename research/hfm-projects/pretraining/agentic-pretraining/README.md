# Agentic Pretraining with a Frozen Corpus and Tokenizer

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
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

A toy-scale study on CPU of the question behind HFM's best proposal: if three changes each help alone, does stacking them give the sum? HFM's three are mixture-of-experts, 8k sequences with rotary positions, and a long warmup-stable-decay schedule. Here the three switches are cheap analogues: rotary position encoding instead of a learned position table (rope), warmup-stable-decay instead of one-cycle cosine at the same peak rate (wsd), and 256-byte sequences in batches of 16 instead of 128 in batches of 32 (long; same bytes per step). All eight combinations were trained. Script: [`factorial_tiny.py`](factorial_tiny.py); logs in `research/compute/results/factorial-tiny-*`. It cannot say what happens at HFM's scale and is not comparable to HFM's score.

- **Setup:** 4-layer byte-level model (1.9M parameters, width 192), 6,000 steps, 25M bytes seen, TinyStories. Held-out text is scored in 256-byte windows; a model trained on 128-byte sequences reads each window as two halves, so part of the long-sequence gain is longer context at scoring time, which is how such a model would be used.

| Recipe | 6,000 steps | Seeds |
|---|---|---|
| base | 1.147 | 1.151, 1.149, 1.142 |
| rope | 1.115 | 1.115, 1.111, 1.119 |
| wsd | 1.116 | 1.120, 1.117, 1.112 |
| long | 1.131 | 1.108, 1.169, 1.117 |
| rope + wsd | 1.095 | 1.094, 1.092, 1.100 |
| wsd + long | 1.087 | 1.051, 1.123 |
| rope + long | 1.072 | 1.067, 1.069, 1.080 |
| rope + wsd + long | **1.044** | 1.044, 1.045 |

(bits per byte on held-out TinyStories; the last two recipes have two seeds because the third hit the job's time limit)

- **Each change helps alone, by similar amounts.** Averaged over the other switches: rope −0.039, wsd −0.031, long −0.035 bits per byte.
- **The three stacked beat the sum of their separate gains.** Adding the three single-switch gains to the base predicts 1.068; the observed result is 1.044, 0.023 better. The pairs differ: rope with long is 0.027 better than additive, wsd with long 0.013 better, and rope with wsd 0.011 worse. The two that interact positively both involve the longer sequences; rotary positions make longer sequences usable, and the long schedule gives them time to pay off.
- **Long sequences with a learned position table are unstable.** The long and wsd + long recipes have the widest seed spread (up to 0.07); every recipe with rope has a spread of 0.013 or less.
- **For HFM's question:** at this scale the answer is "more than additive" when one change enables another and "less than additive" when two changes do the same job. HFM's planned pair experiments are the right way to find out which case applies; the toy result says not to assume either.

## How far this can go

Not estimable without a baseline. The toy result suggests that stacking changes which enable one another can beat the sum of their separate gains, while changes that overlap give less.

## Next steps

Publish a reference score. With a GPU, run the same 2×2×2 design at 100M+ parameters on HFM's three changes, with at least two seeds per cell.
