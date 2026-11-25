# Pretraining architecture development on frozen public training pool

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_pretrain_proposals_20261110` (active on HFM)
- **Project status:** Blocked

## Readiness

- **State:** Ready to run once a free GPU account is connected.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops and trains pretraining architectures on a frozen public training pool, using prior research to improve generalization and four-domain public-validation BPB. Each proposal delivers its agent-selected final checkpoint for independent scoring.

- **Metric:** `negative_public_validation_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.7048
- **HFM effort:** 26 proposals, 20 scored, 0 accepted improvements, about $785 of compute
- **HFM's current best idea:** Training a function-preserving depth-26, FFN-3040 continuation of p1645 with 0.60/0.20/0.20 FineWeb-Edu/GitHub-code/FineMath document sampling will improve web and overall four-domain generalization relative to the same capacity expansion trained with balanced 0.50/0.25/0.25 sampling.

## Existing theory and methods

Function-preserving growth (Net2Net), data-mix scheduling.

## Public data

FineWeb-Edu, GitHub code and FineMath are public.

## What limits progress

Engineering reliability and throughput.

## Route to a step change

Scaling-law extrapolation from short runs to rank ideas.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs a GPU. This analysis ran on a 4-CPU container.

## How far this can go

Not estimable without a baseline.

## Next steps

Publish a reference score.
