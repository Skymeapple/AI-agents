# Looped pretraining with two concurrent four-GPU ideas per batch

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain_research_v2` (active on HFM)
- **Project status:** Blocked

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

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish a reference score.
