# Looped Transformer Pretraining with Recurrent Depth Architectures

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain_research_v1` (end on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work explores language models that reuse computation and refine internal state through looped or recurrent-depth architectures. The research campaign uses eight H20 GPUs over 168 hours to design and train a model from random initialization, selecting one final checkpoint. The investigation covers architectures including partial sharing, adapters, mixture of experts, hybrids, nested loops, learned exits, and dynamic computation.

- **Metric:** `negative_hidden_dclm_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -1.553
- **HFM effort:** 20 proposals, 2 scored, 0 accepted improvements, about $130 of compute
- **HFM's current best idea:** Continuing the p481 hybrid checkpoint with four-loop execution exposed immediately and a subsequent 1/2/4-loop balanced schedule will improve held-out causal language-model likelihood, because p481 stopped after 233 steps while its 6000-step curriculum was still training only the one-loop path.

## Existing theory and methods

Recurrent-depth models, learned exits.

## Public data

Public pretraining corpora.

## What limits progress

2 of 20 proposals produced a score.

## Route to a step change

Reliability before novelty.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container. The track has ended.

- **Toy-scale evidence:** a CPU test in [`../looped-pretraining-fixed-compute/`](../looped-pretraining-fixed-compute/README.md) found that looping closes 63% of the gap between a 4-layer and an 8-layer model at equal parameters, loses at equal compute, and that a loop-index embedding adds nothing beyond seed noise. It says nothing about HFM's scale.

## How far this can go

Not estimable.

## Next steps

None; the track is closed.
