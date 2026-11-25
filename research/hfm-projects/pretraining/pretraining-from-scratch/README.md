# Pretraining architecture development from scratch

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain_research` (paused on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops and trains new architectures from scratch using a frozen public training pool and tokenizer. The research is conducted on eight H20 GPUs in a single container over 168 hours. The session involves proposing, selecting, and iterating on multiple research ideas, with flexibility to revise, pause, or discard experiments.

- **Metric:** `negative_public_validation_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.9624
- **HFM effort:** 3 proposals, 2 scored, 0 accepted improvements, about $18 of compute
- **HFM's current best idea:** Because the p468 depth896 branch was the only complete stable candidate, extending that recipe to substantially more tokens with eight-GPU throughput and screening only low-risk variants will yield a lower four-domain macro BPB than its short 45M-token checkpoint.

## Existing theory and methods

Staged architecture search.

## Public data

Public pretraining corpora.

## What limits progress

Only 3 proposals ran.

## Route to a step change

Short-run scaling-law ranking.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable.

## Next steps

Publish a reference score.
