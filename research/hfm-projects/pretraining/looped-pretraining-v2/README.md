# Looped pretraining with two concurrent four-GPU ideas per batch

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain_research_v2` (active on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
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

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

- **Toy-scale evidence:** a CPU test in [`../looped-pretraining-fixed-compute/`](../looped-pretraining-fixed-compute/README.md) found that looping closes 63% of the gap between a 4-layer and an 8-layer model at equal parameters, loses at equal compute, and that a loop-index embedding adds nothing beyond seed noise. It says nothing about HFM's scale.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish a reference score.
