# Looped pretraining with two concurrent four-GPU ideas per batch

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain_research_v2` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work uses English for all agent-authored plans, experiment notes, progress reports, code comments, and delivery summaries. Supplied dataset text remains in its original language.

- **Metric:** `negative_hidden_dclm_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -1.152
- **HFM effort:** 30 proposals, 24 scored, 0 accepted improvements, about $1,276 of compute
- **HFM's current best idea:** Starting from proposal 2120's independently published checkpoint, a four-billion-token full-model continuation with stronger FineWeb-Edu document probabilities q = 0.25p + 0.75 I_fineweb will lower hidden DCLM BPB relative to that parent.

## Existing theory and methods

Universal transformers, weight sharing, deep supervision, loop-index embeddings.

## Public data

FineWeb-Edu and DCLM are public.

## What limits progress

Many runs fail to deliver a scorable checkpoint.

## Route to a step change

Fix a known-good recipe as the floor and compare architectures on top of it.

## Experiment and results

A toy-scale study on CPU of the idea behind HFM's best proposal: shifting document sampling towards FineWeb-Edu to lower bits per byte on held-out DCLM text. Script: [`../pretraining-architecture-pool/mix_tiny.py`](../pretraining-architecture-pool/mix_tiny.py) with corpora from [`corpora.py`](../pretraining-architecture-pool/corpora.py); logs in `research/compute/results/mix-tiny-looped-web-*`. It cannot say what happens at HFM's scale and is not comparable to HFM's score.

- **Setup:** the looped byte-level model of [`../looped-pretraining-fixed-compute/`](../looped-pretraining-fixed-compute/README.md) (1.9M parameters, 1 + 2×3 + 1 layers), 6,000 steps, 25M bytes seen, two seeds. Training text: 30 MB from the first FineWeb-Edu sample shard and 30 MB from the first DCLM shard. Held-out text: the next 4 MB of documents from each file. Each training sequence comes from one corpus, drawn with the stated probability.

| FineWeb-Edu share in training | Held-out DCLM | Held-out FineWeb-Edu |
|---|---|---|
| 0 (DCLM only) | **2.090** | 2.084 |
| 0.25 | 2.095 | 2.026 |
| 0.5 | 2.114 | 2.001 |
| 0.75 | 2.153 | 1.986 |
| 1 (FineWeb-Edu only) | 2.282 | 1.982 |

(bits per byte, mean of two seeds; seeds differ by 0.001 to 0.007)

- **Held-out DCLM gets steadily worse as FineWeb-Edu replaces DCLM.** The best DCLM score comes from training on DCLM alone. A 25% FineWeb-Edu share costs 0.005, which is near seed noise; a 75% share costs 0.063.
- **The trade is asymmetric at small shares.** A 25% FineWeb-Edu share gains 0.058 on FineWeb-Edu for 0.005 on DCLM. A model meant to do well on both should include some of each.
- **This runs against HFM's current best idea** (sampling q = 0.25p + 0.75 on FineWeb-Edu). At this scale, the target-domain data is the best data for the target domain. A larger model might gain more from FineWeb-Edu's cleaner text, so the toy result is a reason to test the idea cheaply before spending four billion tokens on it, not a refutation.

## How far this can go

Not estimable without a baseline. The toy result suggests that the proposed shift towards FineWeb-Edu is more likely to raise than lower held-out DCLM bits per byte.

## Next steps

Publish a reference score. With a GPU, repeat the share sweep at 100M+ parameters before the four-billion-token continuation.
